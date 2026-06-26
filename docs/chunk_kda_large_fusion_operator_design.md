# Chunk KDA 大融合算子设计文档

## 1. 目标

本文档设计一套面向 KDA 的 NPU 算子方案，目标是在尽可能复用现有 GDN 算子的前提下，对 KDA 特有的 key-wise decay、块内矩阵构造、WY 表示重算、输出和反向梯度做重新融合设计。

设计目标：

- 功能语义对齐 fla-org KDA chunk 实现。
- 保留现有 GDN 算子 ABI 和精度行为，不把 KDA 语义强塞进普通 GDN 路径。
- 可复用的 GDN 部分在 ACLNN L2 层通过 L0 接口拼接。
- 无法复用的 KDA 特有部分按 NPU 特性重新设计，优先让有数据依赖的块内计算驻留 UB/L1 完成。
- cube 核和 vector 核尽量并行，沿用本仓 `recompute_wu_fwd`、`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o` 中已有的 AIC/AIV 分工、workspace、cross-core flag 和 ping-pong 设计。
- 首期覆盖 `chunk_size in {64, 128}`，重点支持典型模型场景 `K=128`、`V=128/256`、bf16/fp16、定长和 varlen。

## 2. 三方实现功能对标

### 2.1 KDA 前向调用图

fla-org KDA 前向主链路如下：

```text
g -> chunk_local_cumsum / kda_gate_chunk_cumsum
  -> chunk_kda_fwd_intra(q, k, v, gk, beta)
       outputs: w, u, qg, kg, Aqk, Akk
  -> chunk_gated_delta_rule_fwd_h(k=kg, w=w, u=u, gk=g)
       outputs: h, v_new, final_state
  -> chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=Aqk, h=h)
       output: o
```

KDA 与 GDN 的关键差异：

- gate 是 key-wise gate，形状含 K 维，记为 `gk`。
- 块内矩阵分为 `Aqk` 和 `Akk`：
  - `Aqk` 用于最终输出的 local 部分。
  - `Akk` 经过 triangular solve 后用于 WY 表示。
- `recompute_w_u_fwd` 不只输出 `w/u`，还需要输出 `qg/kg`：
  - `qg = q * exp2(gk)`，用于 hidden-state backward。
  - `kg = k * exp2(gk_last - gk)`，用于 hidden-state forward/backward。
- 输出不使用普通 `chunk_fwd_o(q,k,v,h,g)`，而使用 `chunk_gla_fwd_o_gk(q,v,gk,Aqk,h)`。

### 2.2 KDA 反向调用图

三方 KDA 反向核心链路：

```text
必要时重算 gk
必要时重算 w, u, qg, kg, Aqk, Akk
必要时重算 h, v_new

chunk_kda_bwd_dAv(q, k, v_new, do, Aqk)
  -> dAqk, dv_local

chunk_gated_delta_rule_bwd_dhu(q=qg, k=kg, w=w, do, dv_local, gk=g)
  -> dh, dh0, dv

chunk_kda_bwd_wy_dqkg_fused(q,k,v,v_new,gk,beta,Akk,h,do,dh,dv)
  -> dq, dk, dv, dbeta, dg, dAkk

chunk_kda_bwd_intra(q,k,gk,beta,dAqk,dAkk,dq,dk,dbeta,dg)
  -> final dq, dk, dbeta, dg
```

### 2.3 与现有 GDN 算子的可复用关系

| 模块 | GDN 现状 | KDA 是否复用 | 结论 |
|---|---|---:|---|
| `chunk_gated_delta_rule_fwd_h` | 已有 GDN hidden-state 前向，`gk` 接口预留但未实现 | 是 | 必须增强为 `g`/`gk` 双路径 |
| `chunk_gated_delta_rule_bwd_dhu` | 接口已有 `gkOptional`，kernel 未完整消费 | 是 | 必须补齐 `gk` 分支 |
| `recompute_wu_fwd` | GDN 版输出 `w/u`，内部已有 vector 预处理 + cube `A @ X` | 部分复用设计 | 新增 KDA 版，不建议改 GDN ABI |
| `chunk_fwd_o` | 计算 GDN 输出，内部自己构造 local qk 矩阵 | 否 | KDA 需要 `Aqk`，应新增 `chunk_kda_fwd_o_gk` |
| `chunk_bwd_dv_local` | GDN local dv | 否 | KDA 需要同时生成 `dAqk` |
| `chunk_bwd_dqkwg` | GDN dq/dk/dw/dg | 否 | KDA 反向公式和输出不同 |
| `prepare_wy_repr_bwd_*` | GDN WY backward | 否 | KDA 需要 key-wise `dg` 和 `dAkk`，应新增融合反向 |

## 3. 数据布局和符号

为兼容本仓现有 NPU 自定义算子，首期采用 head-first 布局：

| 符号 | 形状 | 说明 |
|---|---|---|
| `q` | `[B, H, T, K]` | query |
| `k` | `[B, H, T, K]` | key |
| `v` | `[B, HV, T, V]` | value，支持 GVA，`HV % H == 0` |
| `gk` | `[B, HV, T, K]` | chunk 内累计后的 key-wise gate，base-2 log space |
| `beta` | `[B, HV, T]` | beta |
| `Aqk` | `[B, HV, T, BT]` | q-k local attention 矩阵 |
| `Akk` | `[B, HV, T, BT]` | k-k WY 矩阵，solve 后保存 |
| `w` | `[B, HV, T, K]` | WY 表示中的 W |
| `u` | `[B, HV, T, V]` | WY 表示中的 U |
| `qg` | `[B, HV, T, K]` | `q * exp2(gk)`，可选保存 |
| `kg` | `[B, HV, T, K]` | `k * exp2(gk_last - gk)` |
| `h` | `[B, HV, NT, K, V]` | chunk state |
| `v_new` | `[B, HV, T, V]` | delta update 后 value |
| `o` | `[B, HV, T, V]` | 输出 |

`BT` 表示 `chunk_size`。首期支持 `BT=64/128`。`BC=16` 为 KDA intra 子块大小，用于控制 triangular solve 的 UB 依赖链。

## 4. 接口设计

### 4.1 PyTorch 公开接口

建议新增低层接口，而不是复用 GDN 名称：

```yaml
- func: npu_chunk_kda_fwd(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor gk,
    Tensor beta,
    float scale,
    int chunk_size,
    *,
    Tensor? initial_state=None,
    bool? output_final_state=False,
    int[]? cu_seqlens=None,
    int[]? chunk_indices=None,
    bool? return_intermediate=False,
    bool? safe_gate=False,
    bool? transpose_state_layout=False
  ) -> (Tensor o, Tensor final_state, Tensor Aqk, Tensor Akk, Tensor? w, Tensor? u, Tensor? qg, Tensor? kg, Tensor? v_new, Tensor? h)
```

说明：

- `npu_chunk_kda_fwd/bwd` 定义为低层 NPU core ABI，`gk` 传入 chunk 内累计后的 key-wise gate。这样可以直接对齐 `chunk_kda_fwd_intra`、`chunk_gated_delta_rule_fwd_h(gk=...)` 和 `chunk_gla_fwd_o_gk` 的真实消费语义，避免在每个 L0 中重复做 prefix-sum。
- 三方 `chunk_kda_fwd` 的外层入口接收 raw gate `g`，随后调用 `chunk_local_cumsum` 或 `kda_gate_chunk_cumsum` 得到累计 gate。NPU 对外兼容层应保留同名 `g` 入口，并在 ACLNN L2 外或 L2 起始阶段生成 `gk` 后再进入 core ABI；若反向返回 raw `dg`，则需要补充 cumsum backward/decumsum 逻辑。
- 首期交付的 core ABI 不把 raw gate cumsum 融进主 kernel；后续可新增 `gate_mode` 或 `use_gate_in_kernel` 参数，把 raw gate、`A_log/dt_bias/lower_bound` 合入同一 L2 组合，但不改变 core L0 的累计 `gk` 约定。
- `return_intermediate=False` 时，只保证返回 `o/final_state/Aqk/Akk`；其余中间结果可为空。训练反向若需要重算，则使用 `Aqk/Akk` 和输入重算。
- `transpose_state_layout` 作为 ABI 保留。首期若底层 GDN `fwd_h` 尚未支持 V-first 布局，应在 L2 参数检查中拒绝，不能静默忽略。

反向接口：

```yaml
- func: npu_chunk_kda_bwd(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor gk,
    Tensor beta,
    Tensor Aqk,
    Tensor Akk,
    Tensor do,
    float scale,
    int chunk_size,
    *,
    Tensor? initial_state=None,
    Tensor? dht=None,
    int[]? cu_seqlens=None,
    int[]? chunk_indices=None,
    bool? safe_gate=False,
    bool? transpose_state_layout=False
  ) -> (Tensor dq, Tensor dk, Tensor dv, Tensor dbeta, Tensor dgk, Tensor dh0)
```

### 4.2 ACLNN L2 组合接口

ACLNN L2 层负责组合 L0 原子算子：

```text
aclnnChunkKdaFwd
  1. l0op::ChunkKdaFwdIntraRecompute
       outputs: Aqk, Akk, w, u, qg?, kg
  2. l0op::ChunkGatedDeltaRuleFwdH
       inputs: k=kg, w, u, gOptional=nullptr, gkOptional=gk
       outputs: h, v_new, final_state
  3. l0op::ChunkKdaFwdOGk
       inputs: q, v_new, gk, Aqk, h
       outputs: o
```

```text
aclnnChunkKdaBwd
  1. l0op::ChunkKdaFwdIntraRecompute    # recompute w/u/qg/kg if not saved
  2. l0op::ChunkGatedDeltaRuleFwdH      # recompute h/v_new
  3. l0op::ChunkKdaBwdDAv
  4. l0op::ChunkGatedDeltaRuleBwdDhu
  5. l0op::ChunkKdaBwdWyDqkgFused
  6. l0op::ChunkKdaBwdIntra
```

设计理由：

- `fwd_h/bwd_dhu` 是跨 chunk hidden state 递推，现有 GDN kernel 的 cube/vec 协作结构最接近 KDA 需要，优先复用。
- KDA 的 intra 和 output 与 GDN 公式不同，单独做 L0，避免污染 GDN 算子。
- L2 组合能先交付功能和精度，后续再把 `ChunkKdaFwdIntraRecompute + ChunkGatedDeltaRuleFwdH + ChunkKdaFwdOGk` 做成更大的 monolithic kernel，不改变 Python 侧 ABI。

### 4.3 L0 原子算子

首期新增以下 L0：

| L0 算子 | 作用 | 是否训练必需 |
|---|---|---:|
| `ChunkKdaFwdIntraRecompute` | 生成 `Aqk/Akk/w/u/kg`，可选 `qg` | 是 |
| `ChunkKdaFwdOGk` | 计算 `o = scale * qg @ h + Aqk @ v_new` | 是 |
| `ChunkKdaBwdDAv` | 计算 `dAqk` 和 local `dv` | 是 |
| `ChunkKdaBwdWyDqkgFused` | 计算 WY/inter 部分的 `dq/dk/dv/dbeta/dgk/dAkk` | 是 |
| `ChunkKdaBwdIntra` | 处理 `dAqk/dAkk` 的 intra 反向 | 是 |

增强以下 GDN L0：

| L0 算子 | 增强点 |
|---|---|
| `ChunkGatedDeltaRuleFwdH` | 支持 `gOptional == nullptr && gkOptional != nullptr`，实现 key-wise state decay |
| `ChunkGatedDeltaRuleBwdDhu` | 完整消费 `gkOptional`，实现 key-wise backward state decay |

## 5. 前向融合设计

### 5.1 `ChunkKdaFwdIntraRecompute`

该算子融合三方实现中的：

- `chunk_kda_fwd_intra` 的 `Aqk/Akk` 构造。
- `Akk` 的 triangular solve。
- `kda/wy_fast.py::recompute_w_u_fwd`。

核心公式：

```text
g_rel(i,j,k) = gk[i,k] - gk[j,k]
Aqk[i,j] = sum_k q[i,k] * k[j,k] * exp2(g_rel(i,j,k)) * scale, j <= i
Akk[i,j] = sum_k k[i,k] * k[j,k] * exp2(g_rel(i,j,k)) * beta[i], j < i
Akk = solve_tril(I + Akk)
u = Akk @ (v * beta)
w = Akk @ (k * beta * exp2(gk))
qg = q * exp2(gk)                         # 可选
kg = k * exp2(gk_last - gk)
```

### 5.2 UB 驻留策略

`chunk_size=128` 时，`Aqk/Akk` 全量矩阵为 `128 x 128`，fp32 同时驻留两个矩阵需要约 128 KiB，UB 压力高。设计采用 `BC=16` 子块：

- diagonal block 的 `Akkd[BC,BC]` 驻留 UB，完成局部 triangular solve 后再写入 `Akk`。
- inter block 按 `(i_block, j_block)` 流水生成，`Aqk` 和 `Akk` 只写最终结果。
- `gk` 每次读取 `[BC, BK]`，`BK` 取 `32/64`，按 UB 剩余容量选择。
- 对 `safe_gate=False` 常规路径，使用 `g_anchor = gk[block_left_or_last]` 做数值平移，避免 `exp2` 溢出。
- 对 `safe_gate=True` 路径，按三方实现保留更稳妥的 gather/anchor 策略，首期可降级为较小 `BK`。

### 5.3 cube/vector 分工

vector 侧：

- 读取 `q/k/gk/beta/v`。
- cast 到 fp32。
- 计算 `exp2(gk - anchor)`、`exp2(anchor - gk)`。
- 生成局部 `qg_tile/kg_tile/k_beta_g_tile/v_beta_tile`。
- 对 `qg/kg` 需要持久化的输出写 GM。

cube 侧：

- 消费 vector 侧准备好的 tile。
- 计算 `Aqk`、`Akk` 子块矩阵乘。
- 计算 `u = Akk @ (v*beta)`。
- 计算 `w = Akk @ (k*beta*exp2(gk))`。

同步方式：

- 沿用本仓 `recompute_wu_fwd` 的 `CrossCoreFlagWithReverse` 模式。
- vector 完成当前 tile 写入 workspace 后 set flag。
- cube wait flag 后读取 workspace 做 MMAD。
- double buffer：`qg/kg/kbg/vb` workspace 均按 ping-pong 划分，避免读写冲突。

### 5.4 `ChunkGatedDeltaRuleFwdH` 复用增强

KDA 调用：

```text
ChunkGatedDeltaRuleFwdH(k=kg, w=w, u=u, gOptional=nullptr, gkOptional=gk)
```

需要修改点：

- L2 参数检查允许 `gOptional == nullptr`。
- L0 OpDef 增加或启用 `gk` 输入。
- tiling 增加 `hasGk`、`gkDataType`、`gkWorkspaceOffset`。
- vector epilogue 中按 K 维计算：

```text
h *= exp2(gk_last)[:, None]
v_new = u - w @ h_before_decay
h += kg @ v_new
```

注意：这里传入的 `k` 已经是 `kg`，因此 state update 的 `k @ v_new` 不需要再次乘 `exp2(gk_last - gk)`。

### 5.5 `ChunkKdaFwdOGk`

计算：

```text
o_inter = scale * (q * exp2(gk)) @ h
o_local = Aqk @ v_new
o = o_inter + o_local
```

设计：

- 不复用 `chunk_fwd_o`，因为 `chunk_fwd_o` 会自行根据 `q/k/g` 构造 local A，而 KDA 已经有 `Aqk`。
- `Aqk @ v_new` 适合 cube。
- `qg @ h` 适合 cube，`qg` 可由 `ChunkKdaFwdIntraRecompute` 持久化，或在本算子 vector 侧重算。
- 首期为减少 GM 写，若 `return_intermediate=False` 可不保存 `qg`，由 `ChunkKdaFwdOGk` 内部 vector 侧重算 `qg`。
- `V=256` 时按 `BV=64/128` 分块，两段 `V` tile 累加输出，避免一次加载过大 `h[K,V]`。

## 6. 反向融合设计

### 6.1 `ChunkKdaBwdDAv`

输入 `q/k/v_new/do/Aqk`，输出：

```text
dAqk = do @ v_new^T * scale
dv_local = Aqk^T @ do
```

`dAqk` shape 与 `Aqk` 相同。`dv_local` 作为 `bwd_dhu` 的输入。

### 6.2 `ChunkGatedDeltaRuleBwdDhu` 复用增强

KDA 调用：

```text
ChunkGatedDeltaRuleBwdDhu(q=qg, k=kg, w=w, do=do, dv=dv_local, gOptional=nullptr, gkOptional=gk)
```

增强点：

- kernel 中真实使用 `gkOptional`：

```text
dh *= exp2(gk_last)[:, None]
dq path 使用 qg，不再内部乘 scalar g
dv path 使用 kg 和 dh
```

- 当前接口已预留 `gkOptional`，但需要补 kernel 计算和 ATK case。

### 6.3 `ChunkKdaBwdWyDqkgFused`

融合三方 KDA 中 WY/inter 反向核心：

输入：

```text
q, k, v, v_new, gk, beta, Akk, h, do, dh, dv
```

输出：

```text
dq_part, dk_part, dv2, dbeta_part, dgk_part, dAkk
```

关键计算：

- `do @ h` 产生 inter 的 `dq`。
- `v_new @ dh` 产生 inter 的 `dk`。
- `dv @ h` 产生 `dw`。
- `Akk @ dw` 回传到 `dk/dbeta/dgk/dAkk`。
- `gk` 梯度是 K 维张量 `[B, HV, T, K]`，不能沿 K 维规约成 scalar。

NPU 设计：

- `V` 方向按 `BV=64/128` 分块累加。
- `K` 方向按 `BK=32/64` 分块。
- 对 `h/dh` 读取采用当前 `chunk_bwd_dqkwg` 的 GM 布局和 tile copy 经验。
- `dAkk` 的 triangular solve 反传保持 UB 内完成，降低 GM 往返。

### 6.4 `ChunkKdaBwdIntra`

输入：

```text
q, k, gk, beta, dAqk, dAkk, dq_part, dk_part, dbeta_part, dgk_part
```

输出最终：

```text
dq, dk, dbeta, dgk
```

设计：

- 对每个 `BC x BC` 子块计算 `dAqk/dAkk` 对 `q/k/gk/beta` 的贡献。
- `dgk` 使用 fp32 累加，输出 fp32 或按 Python wrapper 需要 cast。
- 对 `HV != H` 的 GVA，`dq/dk` 需要从 HV 维规约回 H 维；L2 层可在输出前调用规约 L0 或在 kernel 内按 group 累加。首期建议在 kernel 内以 HV 为并行维输出临时 `dq_hv/dk_hv`，L2 层用已有 reduce/sum L0，避免原子写热点。

## 7. Tiling 和 workspace 设计

### 7.1 支持范围

| 维度 | 首期支持 | 扩展策略 |
|---|---:|---|
| `chunk_size` | 64, 128 | 32 可后续补；128 为重点 |
| `K` | 64, 128 | 256 需拆 `BK` 并评估 UB |
| `V` | 64, 128, 256 | `BV=64/128` 分块 |
| dtype | fp16, bf16；gate/beta 可 fp16/bf16/fp32 | fp32 输入后续评估 |
| varlen | 支持 `cu_seqlens + chunk_indices` | 复用现有 chunk offset 生成 |
| initial/final state | 支持 | state_v_first 首期可拒绝或复用已有实现 |

### 7.2 Workspace

前向首期 workspace：

```text
vb_workspace      [B, HV, T, V]    临时或 ping-pong，v * beta
kbg_workspace     [B, HV, T, K]    临时或 ping-pong，k * beta * exp2(gk)
kg_workspace      [B, HV, T, K]    输出或临时，k * exp2(gk_last - gk)
qg_workspace      [B, HV, T, K]    可选输出，q * exp2(gk)
fwd_h_workspace   复用 GDN fwd_h
```

为了减少 HBM 压力：

- `vb/kbg` 优先作为算子内部 workspace，不作为公开输出。
- `kg` 必须输出给 `fwd_h/bwd_dhu`，除非未来做 monolithic fwd_h fusion。
- `qg` 在训练反向需要；若 forward 不保存，则 backward 重算。

### 7.3 `chunk_size=128, V=256`

`V=256` 不能假设一次性完整驻留：

- `u = Akk @ vb` 按 `BV=128` 两次 cube 计算，或 `BV=64` 四次。
- `fwd_h` 的 `w @ h` 和 `kg @ v_new` 复用现有按 V tile 的策略，不能把 `[K,V]` 全量放 UB。
- `ChunkKdaFwdOGk` 中 `Aqk @ v_new` 按 `BV` tile。
- `qg @ h` 同样按 `BV` tile，`qg` 按 `BK` tile。

验收必须包含：

- `BT=128, K=128, V=256, bf16`
- `BT=128, K=128, V=256, fp16`
- varlen 末块不足 128 的 case

## 8. 功能一致性和扩展性

### 8.1 与三方一致

首期需要对齐：

- `chunk_kda_fwd` 的默认路径：三方外层 raw gate `g` 通过 cumsum 变为累计 `gk`；NPU core ABI 接收累计 `gk`，兼容封装负责 raw/cumsum 模式转换。
- `safe_gate=False` 主路径。
- `disable_recompute=False` 时 forward 可不保存全部中间量，backward 重算。
- `cu_seqlens/chunk_indices` varlen。
- `initial_state/output_final_state`。
- GVA：`HV % H == 0`。

### 8.2 暂不纳入首期

- `use_gate_in_kernel=True` 的 raw gate 融合。
- `lower_bound` gate 下界。
- CP 上下文通信相关逻辑。
- `state_v_first=True`，除非现有 GDN fwd_h/bwd_dhu 同步补齐并验证。
- `safe_gate=True` 的完整高精度路径。可先保留 ABI，未支持时明确报错。

### 8.3 ABI 扩展

建议公开大算子接口保留：

- `return_intermediate`
- `safe_gate`
- `transpose_state_layout`
- `recompute_mode`

这些参数首期可拒绝部分组合，但不要在后续增加接口时改变参数顺序。

## 9. 实施计划

### 阶段 0：composite 功能闭环

1. 在 `fla_npu` 包内新增 `torch.ops.npu.npu_chunk_kda_fwd` 和 `torch.ops.npu.npu_chunk_kda_bwd`。
2. forward 使用 PyTorch/NPU 张量算子实现完整 KDA 公式，输出 `o/final_state/Aqk/Akk/w/u/qg/kg/v_new/h`。
3. backward 基于同一 forward 语义重算并使用 autograd 返回 `dq/dk/dv/dbeta/dgk/dh0`。
4. 该阶段用于固定公开 ABI、打通正反向功能和精度验证，不作为最终性能路径。
5. 后续 Ascend C 大融合 L0/kernel 在不改变 Python ABI 的前提下替换 composite core。

### 阶段 1：复用链路打通

1. 增强 `ChunkGatedDeltaRuleFwdH` 支持 `gk`。
2. 增强 `ChunkGatedDeltaRuleBwdDhu` 支持 `gk`。
3. 新增 `ChunkKdaFwdIntraRecompute`。
4. 新增 `ChunkKdaFwdOGk`。
5. ACLNN L2 新增 `aclnnChunkKdaFwd`，串接 3 个 L0。
6. PyTorch wrapper 新增 `npu_chunk_kda_fwd`。

### 阶段 2：训练反向

1. 新增 `ChunkKdaBwdDAv`。
2. 新增 `ChunkKdaBwdWyDqkgFused`。
3. 新增 `ChunkKdaBwdIntra`。
4. ACLNN L2 新增 `aclnnChunkKdaBwd`。
5. PyTorch wrapper 新增 `npu_chunk_kda_bwd`。

### 阶段 3：性能融合

1. 将 `ChunkKdaFwdOGk` 内部重算 `qg`，减少 forward 保存。
2. 评估 `ChunkKdaFwdIntraRecompute` 与 `ChunkKdaFwdOGk` 的 workspace 复用。
3. 评估将 `fwd_h` 的 `h/v_new` 与 output kernel 做 chunk 级流水，减少 `h/v_new` GM 写回。

## 10. 测试设计

### 10.1 ATK 单算子

每个新增 L0 需要 ATK：

- `aclnn_chunk_kda_fwd_intra_recompute`
- `aclnn_chunk_kda_fwd_o_gk`
- `aclnn_chunk_kda_bwd_dav`
- `aclnn_chunk_kda_bwd_wy_dqkg_fused`
- `aclnn_chunk_kda_bwd_intra`
- 增强后的 `aclnn_chunk_gated_delta_rule_fwd_h` gk case
- 增强后的 `aclnn_chunk_gated_delta_rule_bwd_dhu` gk case

ATK golden 使用 PyTorch 参考实现，公式来自 fla-org KDA，不依赖外部 GPU kernel。

### 10.2 端到端典型模型场景

| 场景 | B | H | HV | T | K | V | chunk_size | dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| small smoke | 1 | 4 | 4 | 2048 | 128 | 128 | 64 | fp16 |
| bf16 common | 1 | 8 | 8 | 4096 | 128 | 128 | 64 | bf16 |
| long context | 1 | 16 | 16 | 8192 | 128 | 128 | 128 | bf16 |
| value 256 | 1 | 16 | 16 | 4096 | 128 | 256 | 128 | bf16 |
| GVA | 1 | 8 | 16 | 4096 | 128 | 128 | 128 | bf16 |
| varlen | 1 | 8 | 8 | sum(lens) | 128 | 128 | 64/128 | bf16 |

### 10.3 精度阈值

建议阈值：

- fp16：`rtol <= 3e-2, atol <= 3e-2`
- bf16：`rtol <= 5e-2, atol <= 5e-2`
- 梯度 fp32 中间输出：按 `max_abs/max_rel/cosine` 同时记录。

输出项：

- forward：`o, final_state, Aqk, Akk`
- backward：`dq, dk, dv, dbeta, dgk, dh0`

### 10.4 内存异常检查

对新增/增强 kernel：

- race 类问题使用 `mssanitizer --tool=racecheck`。
- 越界类问题使用 `memcheck`。
- 未初始化读取使用 `initcheck`。
- 同步问题使用 `synccheck`。

必须确认实际命中 sanitizer 版本对象，不能用未加载 sanitizer 的日志作为无问题结论。

## 11. 风险和约束

- `fwd_h/bwd_dhu` 现有接口对 `g` 的假设较强，增强 `gk` 时必须确保 `g` 路径数值完全不变。
- `BT=128,V=256` 对 UB/L1/workspace 压力更高，必须按 V/K 分块，不能复用只适合 `V=128` 的固定 buffer。
- `dgk` 是 `[B,HV,T,K]`，比 GDN scalar `dg` 大很多，反向写带宽和 workspace 都要重新估算。
- KDA 反向中 `dAqk/dAkk` 是两个不同矩阵，不能合并成 GDN 的单一 `A/dA` 语义。
- 如果 forward 不保存 `qg/kg/h/v_new`，backward 必须重算，L2 executor 需要管理中间 tensor 生命周期。

## 12. 结论

KDA 不能通过“给所有 GDN 算子加 `gk`”实现。正确方案是：

1. 复用并增强 hidden-state 相关的 `fwd_h/bwd_dhu`。
2. 新增 KDA 专属 intra、output 和 backward 融合 L0。
3. 用 ACLNN L2 大算子把可复用 GDN L0 与 KDA L0 拼接。
4. 首期保证 `chunk_size=64/128`、`V=128/256`、bf16/fp16、定长/varlen 的功能和精度。
## 13. 实现校准与复用边界

当前已落地一套 KDA 正反向 composite operator，用于先打通公开 ABI、NPU 功能执行和精度闭环。该路径已覆盖 `chunk_size=128,V=256` 的 forward 精度，以及中等规模 backward 梯度对齐。

`ChunkGatedDeltaRuleFwdH` 已增强 `gk` 输入下发和 K 维 state decay，已通过 `chunk_size=64,K=128,V=128` 的 NPU 精度验证。实现中需要注意：

- KDA 传入 `fwd_h` 的 `k` 应为已预缩放的 `kg = k * exp2(gk_last - gk)`。
- 因此 `gk` 分支的 V1 不应再对 `v_new` 做标量 gate decay。
- V2 只对历史 state 做 `exp2(gk_last)[:, None]` 的 K 维衰减，再加上 `kg^T @ v_new`。
- 原 GDN `g` 分支仍保持自然指数语义，不能被 KDA 的 `exp2` 改动污染。

验证中发现现有 `ChunkGatedDeltaRuleFwdH` 在 `chunk_size=128,V=256` 下，即使走原 `g` 分支也会在运行阶段挂起。这说明 V256 不是单纯给现有 `fwd_h` 增加 `gk` 就能复用的形状。KDA 大融合首期若要求 `V=256`，状态更新部分需要单独按 V tile 重新实现：

- V 维按 `BV=64/128` 切分，避免将完整 `[K,V]` state 固定驻留单个 UB buffer。
- cube 的 `w @ h` 与 `kg^T @ v_new` 使用 L1/UB ping-pong，vector 侧只处理当前 V tile 的 state decay 和累加。
- `final_state` 写回也按 V tile 拼接，不能复用当前 V128 假设的固定 buffer。

因此，后续 Ascend C 大融合实现的推荐路线是：保留当前 composite 作为功能 fallback；对 `V<=128` 可继续评估复用增强后的 `fwd_h(gk)`；对 `V=256` 新增 KDA 专用状态更新 kernel 或在 KDA monolithic kernel 内融合该递推。
