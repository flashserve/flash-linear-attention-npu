# Chunk KDA 大融合算子设计文档

## 1. 目标

为 KDA 提供一套与 fla-org 公式语义一致的 NPU 正反向算子：

- 对外暴露稳定的 `torch.ops.npu.npu_chunk_kda_fwd/bwd` ABI。
- 尽可能复用 GDN 已有的 hidden-state 算子。
- 对 KDA 特有的 key-wise decay、块内 `Aqk/Akk`、`qg/kg/w/u/v_new` 中间量保持独立语义，不污染普通 GDN `g` 路径。
- 支持 `chunk_size=64/128`、`K=128`、`V=128/256`、GVA、varlen、`initial_state/output_final_state`。

## 2. 三方语义对齐

KDA forward 主链路：

```text
gk -> chunk 内累计 key-wise gate
q,k,v,beta,gk -> Aqk/Akk/w/u/qg/kg
kg,w,u,gk -> h/v_new/final_state
qg,Aqk,h,v_new -> o
```

KDA backward 主链路：

```text
recompute forward intermediates
do,Aqk,v_new -> dv_local/dAqk
qg,kg,w,do,dv_local,gk -> dh/dv2
q,k,v,v_new,gk,beta,Akk,h,do,dh,dv2 -> dq/dk/dv/dbeta/dgk
```

其中 `gk` 是 `[B, HV, T, K]` 的 key-wise 累计 gate，状态衰减使用 `exp2(gk_last)`，不是 GDN scalar `g` 的自然指数路径。

## 3. 复用边界

| 模块 | 复用结论 | 说明 |
|---|---|---|
| `chunk_gated_delta_rule_fwd_h` | 复用 | 已增强 `gk` 分支；`K=128,V=128` 可作为 KDA forward 状态段 native 路径 |
| `chunk_gated_delta_rule_bwd_dhu` | 复用 | 已增强 `gk` 分支；`dv2 = kg @ dh + dv`，`dh` 按 K 维 `exp2(gk_last)` 衰减 |
| `recompute_wu_fwd` | 不直接复用 ABI | KDA 需要同时产生 `qg/kg/Aqk/Akk`，不应改造 GDN ABI |
| `chunk_fwd_o` | 不复用 | GDN 内部自行构造 local A；KDA 使用 `Aqk` 和 `qg` |
| `chunk_bwd_dqkwg` | 不复用 | KDA 输出 `dbeta/dgk` 语义与 GDN 不同 |

## 4. 接口设计

Forward：

```text
npu_chunk_kda_fwd(
    q, k, v, gk, beta,
    scale: float,
    chunk_size: int,
    *,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    return_intermediate=False,
    safe_gate=False,
    transpose_state_layout=False
) -> (o, final_state, Aqk, Akk, w, u, qg, kg, v_new, h)
```

Backward：

```text
npu_chunk_kda_bwd(
    q, k, v, gk, beta, Aqk, Akk, d_o,
    scale: float,
    chunk_size: int,
    *,
    initial_state=None,
    dht=None,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    transpose_state_layout=False
) -> (dq, dk, dv, dbeta, dgk, dh0)
```

当前实现约束：

- 输入为 head-first 布局：`q/k=[B,H,T,K]`，`v=[B,HV,T,V]`，`gk=[B,HV,T,K]`，`beta=[B,HV,T]`。
- `HV % H == 0`，用于 GVA。
- `safe_gate=True`、`transpose_state_layout=True` 暂不支持，显式报错。
- varlen 要求 `B=1`，`cu_seqlens/chunk_indices` 与三方 chunk 语义一致。

## 5. 大融合实现策略

首版落地为 L2 大融合入口：

1. 用 PyTorch/NPU tensor 算子完成 KDA 块内 `Aqk/Akk/w/u/qg/kg` 计算。
2. 对固定长、`K=128,V=128` 场景，优先调用 native `npu_chunk_gated_delta_rule_fwd_h(gk=...)` 完成 `h/v_new` 状态递推。
3. 若 native `fwd_h(gk)` 不可用、形状不满足约束，或运行时报错，则自动回退 composite 状态递推。
4. `V=256` 明确走 composite fallback，避免复用现有 `fwd_h` 的 V256 挂起路径。
5. `final_state` 在 KDA L2 内按 `h_last * exp2(gk_last) + kg^T @ v_new` 重建，避免依赖 `fwd_h` 在无初始态边界下的 final_state 输出。
6. backward 当前以相同 forward 语义重算并由 autograd 返回完整梯度，保证功能闭环；`bwd_dhu(gk)` native kernel 已单独补齐并通过黄金测试，为后续进一步拆分 backward L0 打基础。

## 6. `bwd_dhu(gk)` Native 设计

三方 KDA 调用方式为：

```text
chunk_gated_delta_rule_bwd_dhu(q=qg, k=kg, w=w, do=do, dv=dv_local, gk=g)
```

关键语义：

- `q` 已经是 `qg`，不再乘 scalar gate。
- `k` 已经是 `kg`，`dv2` 不再乘 `exp(g_last - g)`。
- 非末 chunk：

```text
dv2 = kg @ dh + dv
dh  = dh * exp2(gk_last)[:, None] + qg^T @ do * scale - w^T @ dv2
```

实现要点：

- tiling 增加 `hasGk`，`g/gk` 至少传一个；`gk` 支持 fp16/bf16/fp32。
- cube 侧复用原有 `kg @ dh`、`qg^T @ do`、`w^T @ dv2`。
- vector 侧在 UB 内加载当前 K half 的 `gk_last`，转换为 `exp2` 后逐 K 行衰减 `dh`。
- 原 GDN `g` 路径保持不变。

## 7. V256 策略

已验证现有 `fwd_h` 在 `chunk_size=128,V=256` 下不可作为稳定复用路径。因此 KDA 大融合首版采用：

- `V=128`：尝试 native `fwd_h(gk)`。
- `V=256`：走 KDA composite fallback，保持功能和精度覆盖。

后续若继续推进 monolithic Ascend C KDA kernel，状态段应按 V tile 设计：

- `BV=64/128` 切分 `[K,V]` state，避免整块 state 驻留单个 UB buffer。
- cube 侧处理 `w @ h`、`kg^T @ v_new`，vector 侧在当前 V tile 内完成 state decay 和累加。
- `final_state` 按 V tile 拼接写回。

## 8. 已落地文件

- `torch_custom/fla_npu/fla_npu/kda.py`
- `torch_custom/fla_npu/test/test_npu_chunk_kda.py`
- `torch_custom/fla_npu/test/test_npu_chunk_gated_delta_rule_bwd_dhu_gk.py`
- `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/**`
- `fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/**`

## 9. 验证结论

- KDA V128 forward 已确认命中 native `fwd_h(gk)`，`final_state` 对 reference 最大绝对误差 `2.22e-6`。
- KDA V256 forward 通过 reference 对齐。
- KDA backward medium 通过 autograd golden 对齐。
- `bwd_dhu(gk)` native kernel 通过 CPU golden 精度对齐。

因此当前版本完成 KDA 正反向功能闭环，并完成 `fwd_h(gk)`/`bwd_dhu(gk)` 两个可复用 native 状态算子的 KDA 语义补齐。
