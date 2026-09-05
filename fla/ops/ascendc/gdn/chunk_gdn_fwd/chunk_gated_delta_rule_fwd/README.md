# ChunkGatedDeltaRuleFwd

## 功能

`ChunkGatedDeltaRuleFwd` 实现 Gated Delta Rule 的分块前向计算，将门控累加、系数生成、
状态更新和输出计算融合到一个 Phase6 kernel 中。当前实现支持定长和变长序列、GVA、
可选初始状态以及可选最终状态输出。

用于精度对比的公开算子链依次由以下算子组成：

1. `ChunkLocalCumsum`（`chunk_local_cumsum`）；
2. `ChunkScaledDotKkt`（`chunk_scaled_dot_kkt`）；
3. `SolveTri`（`solve_tri`）；
4. `RecomputeWUFwd`（`recompute_w_u_fwd`）；
5. `ChunkGatedDeltaRuleFwdH`（`chunk_gated_delta_rule_fwd_h`）；
6. `ChunkFwdO`（`chunk_fwd_o`）。

融合 kernel 内部实现上述等价计算阶段，不调用或链接这些公开算子的 ACLNN 实现。
公开算子链只作为 ATK 精度标杆使用。

## 输入

令 `B` 为物理 batch size，`Hk` 为 q/k 头数，`Hv` 为 v 头数，`T` 为 token 数，
`K=128`，`V` 为 value 维度，`N` 为逻辑序列数。

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q` | 必选 | `[B,Hk,T,K]`；FP16/BF16 | Query |
| `k` | 必选 | `[B,Hk,T,K]`；与 q 同 dtype | Key |
| `v` | 必选 | `[B,Hv,T,V]`；与 q 同 dtype | Value |
| `g` | 必选 | `[B,T,Hv]`；FP32 | 门控值 |
| `beta` | 必选 | `[B,T,Hv]`；FP32 或与 q 同 dtype | Delta 系数 |
| `aLogOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `dtBiasOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `initialStateOptional` | 可选 | `[N,Hv,K,V]`；FP32 或与 q 同 dtype | 初始状态 |
| `cuSeqlensOptional` | 可选 | `[N+1]`；INT64 | 变长序列累计长度，需与 `chunkIndicesOptional` 同时提供 |
| `chunkIndicesOptional` | 可选 | `[2*Nc]`；INT64 | canonical sequence-major chunk 索引 |

`Hv` 必须能被 `Hk` 整除。变长模式使用物理 `B=1`，`cuSeqlensOptional` 必须从 0 开始、
以 `T` 结束且单调不降。

## 输出

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `oOut` | 必选 | `[B,Hv,T,V]`；与 q 同 dtype | 前向输出 |
| `finalStateOutOptional` | 可选 | `[N,Hv,K,V]`；与初始状态同 dtype，无初始状态时为 FP32 | 是否为空直接决定是否计算并输出最终状态 |
| `gCumsumOutOptional` | 可选 | `[B,T,Hv]`；FP32 | chunk 内门控累加结果；为空时使用内部临时张量 |
| `aOutOptional` | 可选 | `[B,Hv,T,chunkSize]`；与 q 同 dtype | 系数矩阵；为空时使用内部临时张量 |
| `qHatOutOptional`、`kHatOutOptional` | 当前未支持 | - | L2Norm 扩展输出预留，必须为空 |
| `qRstdOutOptional`、`kRstdOutOptional` | 当前未支持 | - | L2Norm 扩展输出预留，必须为空 |
| `betaEffOutOptional` | 当前未支持 | - | beta sigmoid 扩展输出预留，必须为空 |
| `hOutOptional` | 当前未支持 | - | 中间状态扩展输出预留，必须为空 |

## 属性

| 名称 | 当前支持范围 | 说明 |
| --- | --- | --- |
| `layout` | `BNSD` | q/k/v/o 的布局 |
| `scale` | 通常为 `K**-0.5` | Query 缩放因子 |
| `chunkSize` | `64`、`128` | 分块大小 |
| `useExp2` | `false` | 扩展接口预留 |
| `allowNegEigval` | `false` | 扩展接口预留 |
| `stateVFirst` | `false` | 扩展接口预留 |

当前未实现的扩展组合会返回参数错误，不会静默忽略。

## 支持范围

- A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`）。
- q/k/v 支持 FP16、BF16；`K=128`，`V=128/256`。
- `chunkSize=64/128`。
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
