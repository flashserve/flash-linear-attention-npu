# ChunkGatedDeltaRuleFwd

## 功能

`ChunkGatedDeltaRuleFwd` 实现 Gated Delta Rule 的分块前向计算。仅当 `useExp2=false`、
`useQkL2norm=false`、`useGateInKernel=false`、不启用 beta sigmoid、`allowNegEigval=false`、不请求分块状态 h、
`stateVFirst=false` 且 layout 为 `BNSD/NTD` 时使用原 Phase6 kernel，A 可按需输出；任意条件不满足时，
A5 依次调度 `ChunkGatedDeltaRuleFwdPrepare`、`ChunkFwdH` 和 `ChunkFwdO`，三个阶段统一使用
调用方传入的 `useExp2`。
新路径不支持的参数组合由 `ChunkGatedDeltaRuleFwdPrepare` 报错。当前实现支持定长和变长序列、GVA、可选初始状态
以及可选最终状态输出。

用于精度对比的公开算子链依次由以下算子组成：

1. `ChunkLocalCumsum`（`chunk_local_cumsum`）；
2. `ChunkScaledDotKkt`（`chunk_scaled_dot_kkt`）；
3. `SolveTri`（`solve_tri`）；
4. `RecomputeWUFwd`（`recompute_w_u_fwd`）；
5. `ChunkFwdH`（`chunk_fwd_h`）；
6. `ChunkFwdO`（`chunk_fwd_o`）。

融合 kernel 内部实现上述等价计算阶段，不调用或链接这些公开算子的 ACLNN 实现。
公开算子链只作为 ATK 精度标杆使用。

## 输入

令 `B` 为物理 batch size，`Hk` 为 q/k 头数，`Hv` 为 v 头数，`T` 为 token 数，
`K=128`，`V` 为 value 维度，`N` 为逻辑序列数。

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q` | 必选 | 由 `layout` 决定；FP16/BF16 | Query |
| `k` | 必选 | 与 q 同 shape/dtype | Key |
| `v` | 必选 | 由 `layout` 决定；与 q 同 dtype | Value |
| `g` | 必选 | `[B,T,Hv]`；FP32 或与 q 同 dtype | 门控值，固定为 sequence-major |
| `beta` | 必选 | 与 g 同 shape；FP32 或与 q 同 dtype | Delta 系数 |
| `aLogOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `dtBiasOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `initialStateOptional` | 可选 | `stateVFirst=false` 时为 `[N,Hv,K,V]`，否则为 `[N,Hv,V,K]`；FP32 或与 q 同 dtype | 初始状态 |
| `cuSeqlensOptional` | 可选 | `[N+1]`；INT64 | 变长序列累计长度，需与 `chunkIndicesOptional` 同时提供 |
| `chunkIndicesOptional` | 可选 | `[2*Nc]`；INT64 | canonical sequence-major chunk 索引 |

四种 QKV 布局均使用四维输入：`BNSD/NTD=[B,H,T,D]`，
`BSND/TND=[B,T,H,D]`。`Hv` 必须能被 `Hk` 整除。变长模式使用物理 `B=1`，`cuSeqlensOptional` 必须从 0 开始、
以 `T` 结束且单调不降。

## 输出

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `oOut` | 必选 | `[B,T,Hv,V]`（BSND）；与 q 同 dtype | 前向输出固定使用 sequence-major 布局 |
| `finalStateOutOptional` | 可选 | 末两维由 `stateVFirst` 决定；与初始状态同 dtype，无初始状态时为 FP32 | 是否为空直接决定是否计算并输出最终状态 |
| `gCumsumOutOptional` | 可选 | 与 g 同 shape；FP32 | chunk 内门控累加结果；为空时使用内部临时张量 |
| `aOutOptional` | 可选 | `[B,Hv,T,chunkSize]`；与 q 同 dtype | 系数矩阵；为空时使用内部临时张量 |
| `qHatOutOptional`、`kHatOutOptional` | A5 新路径可选 | 与 q/k 相同 | L2Norm 结果 |
| `qRstdOutOptional`、`kRstdOutOptional` | A5 新路径可选 | 固定 `[B,Hk,T]`；FP32 | L2Norm rstd |
| `betaEffOutOptional` | A5 新路径可选 | 与 beta 同 shape；FP32 | 非空时启用并输出 beta sigmoid |
| `hOutOptional` | A5 新路径可选 | `stateVFirst=false` 时末两维为 `[K,V]`，否则为 `[V,K]`；与 q 同 dtype | 分块状态 |

Python 稳定入口固定返回
`(o, final_state, g_cumsum, A, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd)` 十元组。
启用 `use_qk_l2norm_in_kernel` 时，q_hat/k_hat 为归一化结果，shape/layout/dtype 与输入一致，
q_rstd/k_rstd 为 FP32 `[B,Hk,T]`，不随 layout 改变；关闭时 q_hat/k_hat 分别为原始 q/k
对象的别名（不分配、不复制），q_rstd/k_rstd 为 None。新增四项不受 disable_recompute 控制。
旧六项解包调用需要迁移。rstd 的头数为 Hk，不扩展到 Hv；变长模式使用物理 batch/token 维度。
`disable_recompute=True` 时导出 g_cumsum/A，否则这两项为 None。
`output_final_state`、`use_beta_sigmoid_in_kernel` 和 `return_intermediate_states`
分别控制 final_state、beta_eff 和 h 是否为 None。
h 的 shape 为 dense `[B,NT,Hv,K,V]`，带 cu_seqlens 的 packed 为 `[1,total_NT,Hv,K,V]`；
`state_v_first=True` 时末两维为 `[V,K]`。A5 prepare 路径导出遵循此契约，旧融合路径仍不导出 h。

`g/beta` 固定以 BSN 输入，在 ACLNN 内转为 BNS；任一输入为 FP32 时，
另一个先提升为 FP32。两个输入均为主 dtype 时保留该 dtype，后续分支支持范围不变。
Python 接受 `a_log=None, dt_bias=None` 预留参数；当前仅支持
`use_gate_in_kernel=False`，非空 a_log/dt_bias 或启用 gate 均报错。

## 属性

| 名称 | 当前支持范围 | 说明 |
| --- | --- | --- |
| `layout` | 原 Phase6 路径支持 `BNSD/NTD`；A5 新路径支持 `BNSD/BSND/NTD/TND` | q/k/v 的输入布局；BSND/TND 输入在拼接路径内转为 head-first，o 固定输出 BSND |
| `scale` | 通常为 `K**-0.5` | Query 缩放因子 |
| `chunkSize` | `64`、`128` | 分块大小 |
| `useExp2` | A5 新路径支持 `true/false` | 只控制门控累计和后续状态、输出阶段使用 `exp2` 或 `exp`，三个小算子均按新路径规格选择优化实现 |
| `useQkL2norm` | A5 新路径支持 `true/false` | true 时由 prepare 生成归一化 Q/K；false 时 H/O 直接使用输入 Q/K |
| `allowNegEigval` | A5 新路径支持 | 为 true 时必须提供 `betaEffOutOptional` |
| `stateVFirst` | A5 新路径支持 `true/false` | 控制初始状态、分块状态和最终状态的末两维采用 `[V,K]` 或 `[K,V]` |

当前未实现的扩展组合会返回参数错误，不会静默忽略。

## 支持范围

- A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`）。
- 原 Phase6 路径支持 FP16、BF16，`K=128`、`V=128/256`、`chunkSize=64/128`。
- A5 新路径支持 `useExp2=true/false`、`useQkL2norm=true/false`、BF16、`K=V=128`、`chunkSize=64`、`Hv/Hk in {1,2,3,4}`。
- 支持 MHA、GVA、定长和变长序列。
- A2/A3 使用 `arch22` 私有实现，A5 使用 `arch35` 私有实现；两套架构代码隔离维护。

## 验证

ATK 用例和执行说明位于
[`tests/atk/chunk_gated_delta_rule_fwd`](../../../../../../tests/atk/chunk_gated_delta_rule_fwd/README.md)。
精度双标杆分别使用融合算子、上述公开算子链和 FP64 recurrence，并在相同冻结输入上比较结果。

ACLNN ABI 合同可通过以下命令检查：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```

## 归一化结果导出与反向复用

归一化沿用 prepare 的公式：`rstd = rsqrt(sum(x**2, dim=-1) + 1e-6)`，
`hat = cast(x * rstd, input_dtype)`。关闭归一化时 hats 与原始输入共享对象及存储，修改任一对象会影响另一个。
反向传入返回的 hats/rstd 和与前向一致的归一化开关，返回的 dq/dk 对应归一化前的原始输入。

ACLNN 参数数量和顺序不变；关闭归一化时四个可选输出 descriptor 仍为空。
开启时从 prepare 的内部结果导出 hats，rstd 直接按 `[B,Hk,T]` ViewCopy，不再随 sequence-major 布局转置。
BSND/TND 的旧 `[B,T,Hk]` rstd 输出 descriptor 不再接受。
内部 prepare → H → O DAG、workspace、UB/L1 布局及同步均不变；公开输出由调用方持有。

### 直接串联示例

以下代码使用调用方准备的 BSND Q/K/V、BSN g/beta 及上游梯度 d_o，K=V=128、chunk_size=64。
`use_qk_l2norm` 可取 True 或 False；关闭时返回的 q_hat/k_hat 就是原始输入。

```python
from fla_npu.ops.ascendc import (
    npu_chunk_gated_delta_rule_fwd,
    npu_chunk_gated_delta_rule_bwd,
)

use_qk_l2norm = True
scale = q.shape[-1] ** -0.5
(o, final_state, g_cumsum, A, beta_eff, h,
 q_hat, k_hat, q_rstd, k_rstd) = npu_chunk_gated_delta_rule_fwd(
    q, k, v, g, beta,
    layout="BSND", scale=scale, chunk_size=64, use_exp2=True,
    use_qk_l2norm_in_kernel=use_qk_l2norm,
    disable_recompute=True, output_final_state=True,
)
torch.npu.synchronize()
print("FWD_DONE", flush=True)

(dq, dk, dv, d_beta, d_g, dh0, d_a_log, d_dt_bias) = npu_chunk_gated_delta_rule_bwd(
    q_hat, k_hat, v, g_cumsum, beta, A, d_o, scale,
    layout="BSND", chunk_size=64, use_exp2=True,
    use_qk_l2norm_in_kernel=use_qk_l2norm,
    q_rstd=q_rstd, k_rstd=k_rstd,
)
torch.npu.synchronize()
print("BWD_DONE", flush=True)
```
