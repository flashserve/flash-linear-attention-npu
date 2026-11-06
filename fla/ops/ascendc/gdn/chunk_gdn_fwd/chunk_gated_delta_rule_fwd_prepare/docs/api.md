# ChunkGatedDeltaRuleFwdPrepare API

本算子在 A5（`ascend950`）和 A6 开发分支（`ascend960dt`）上注册。两者使用相同的 Python 入口和 ACLNN 签名。

## Python 接口

从 `fla_npu.ops.ascendc` 导入以下函数。

```python
def chunk_gated_delta_rule_fwd_prepare(
    q, k, v, g, beta, chunk_size=64, *,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, use_exp2=False,
    a_log=None, dt_bias=None, cu_seqlens=None, chunk_indices=None, output_a=True,
): ...
```

返回顺序为 `(q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, A)`。关闭核内归一化时，前四项返回输入 `q`、`k` 和两个 `None`。关闭 beta sigmoid 时，`beta_out` 返回 `beta` 的 FP32 值。`output_a=False` 时，第九项是占位张量，其内容未定义。ACLNN 参数校验失败会返回错误状态，Python 调用方会收到异常。

## ACLNN 接口

两段式：

1. `aclnnChunkGatedDeltaRuleFwdPrepareGetWorkspaceSize` 获取 workspace 与 executor
2. `aclnnChunkGatedDeltaRuleFwdPrepare` 在指定 stream 上执行

`use_qk_l2norm_in_kernel` / `use_gate_in_kernel` / `use_beta_sigmoid_in_kernel` **不是** ACLNN 公开 bool，由 optional 张量是否为 `nullptr` 推断：

| 推断出的 flag | True 条件 |
|---|---|
| `useQkL2norm` | `qHat / kHat / qRstd / kRstd` 四个都非空 |
| `useGateInKernel` | `aLogOptional != nullptr` |
| `useBetaSigmoid` | `betaEffOptional != nullptr` |

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepareGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, bool allowNegEigval, bool useExp2, bool outputA,
    aclTensor *gOut, aclTensor *wOut, aclTensor *uOut, aclTensor *aOut,
    aclTensor *qHatOptional, aclTensor *kHatOptional,
    aclTensor *qRstdOptional, aclTensor *kRstdOptional,
    aclTensor *betaEffOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 参数说明

### 输入参数（Inputs）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|---|
| `q` | 输入 | 必选 | Query | BNSD | `BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持（ACLNN 会 contiguous） |
| `k` | 输入 | 必选 | Key | 与 `q` 同形 | `BFLOAT16` | `ND` | `[B, HK, T, K]` | 支持 |
| `v` | 输入 | 必选 | Value，不做 L2Norm | | `BFLOAT16` | `ND` | `[B, HV, T, V]` | 支持 |
| `g` | 输入 | 必选 | Gate。`use_gate=False` 时为已处理好的 log-gate；`True` 时为 raw dt logits | | `FLOAT` / `BFLOAT16` | `ND` | `[B, HV, T]` | 支持 |
| `beta` | 输入 | 必选 | Beta。`use_beta_sigmoid=True` 时为 raw logits | | `FLOAT` / `BFLOAT16` | `ND` | `[B, HV, T]` | 支持 |
| `aLogOptional` | 输入 | 可选 | fused gate 的 `A_log` | `use_gate_in_kernel=True` 时必填 | `FLOAT` / `BFLOAT16` | `ND` | `[HV]` | 支持 |
| `dtBiasOptional` | 输入 | 可选 | fused gate 的 `dt_bias` | 仅与 `a_log` 同时出现；省略视为 0 | `FLOAT` / `BFLOAT16` | `ND` | `[HV]` | 支持 |
| `cuSeqlensOptional` | 输入 | 可选 | 变长累计长度 | 变长时与 `chunkIndices` 成对，`B=1` | `INT64` | `ND` | `[N+1]` | - |
| `chunkIndicesOptional` | 输入 | 可选 | 变长 chunk 索引 | 扁平 `[seqIdx, chunkIdx, ...]`，长度 `2 * numChunks` | `INT64` | `ND` | 1 维 | - |

### 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `chunkSize` | 输入 | 可选属性，接口侧必传 | 分块大小 | 当前仅 64 | `int64_t` | `64` |
| `allowNegEigval` | 输入 | 可选属性，接口侧必传 | 负特征值缩放 | True 时要求写入 `betaEff`（sigmoid 打开） | `bool` | `true` / `false` |
| `useExp2` | 输入 | 可选属性，接口侧必传 | 指数底 | True：cumsum × `RCP_LN2`，后续 `exp2`；False：cumsum × 1，后续自然 `exp` | `bool` | `true` / `false` |
| `outputA` | 输入 | 可选属性，接口侧必传 | 是否把 A 写回 GM | False 时 A 只留在 L1 | `bool` | `true` / `false` |

IR attr 里还有 `use_qk_l2norm` / `use_gate_in_kernel` / `use_beta_sigmoid`，由 L0 根据 optional 张量填入，不出现在 ACLNN C 接口参数表中。

### 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `gOut` | 输出 | chunk-local cumsum 后的 G | `FLOAT` | `ND` | `[B, HV, T]` | 支持 |
| `wOut` | 输出 | WY 的 w | 同 k | `ND` | `[B, HV, T, K]` | 支持 |
| `uOut` | 输出 | WY 的 u | 同 v | `ND` | `[B, HV, T, V]` | 支持 |
| `aOut` | 输出 | `(I+L)^{-1}` | 同 k | `ND` | `[B, HV, T, chunkSize]` | 支持 |
| `qHatOptional` | 输出 | L2Norm 后的 q；flag 关闭时不要传 | 同 q | `ND` | `[B, HK, T, K]` | 支持 |
| `kHatOptional` | 输出 | L2Norm 后的 k；flag 关闭时不要传 | 同 k | `ND` | `[B, HK, T, K]` | 支持 |
| `qRstdOptional` | 输出 | q 的 rstd | `FLOAT` | `ND` | `[B, HK, T]` | 支持 |
| `kRstdOptional` | 输出 | k 的 rstd | `FLOAT` | `ND` | `[B, HK, T]` | 支持 |
| `betaEffOptional` | 输出 | sigmoid 后的 β；flag 关闭时不要传（Python 侧回填 `beta.float()`） | `FLOAT` | `ND` | `[B, HV, T]` | 支持 |

### 形状与约束

- `q`、`k` 必须为 `[B, HK, T, K]` 且完全同形。
- `v` 为 `[B, HV, T, V]`，与 `q` 的 `B`、`T` 一致。
- `g`、`beta`、`gOut`、`betaEff` 为 `[B, HV, T]`。
- `wOut` 为 `[B, HV, T, K]`；`aOut` 为 `[B, HV, T, chunkSize]`。
- **GVA**：`HV % HK == 0` 且 `HV/HK ∈ {1,2,3,4}`。任务按 HV 计数，K 头 `hk = hv / (HV/HK)` 复用。
- `K = 128`，`V ∈ {128, 256}`，`chunkSize = 64`。
- `q/k/v` 当前仅 `BFLOAT16`。按注册的 dtype 组合，`g`、`beta` 和已提供的 `a_log`、`dt_bias` 统一使用 FP32 或 BF16。
- 变长：`cuSeqlens` 与 `chunkIndices` 成对，且 `B = 1`。
- `allow_neg_eigval=True` 要求 `use_beta_sigmoid=True`。
- `dt_bias` 不能在没有 `a_log` 时单独出现。
- `use_qk_l2norm=False` 时调用方应先对 **q 和 k** 做 L2Norm，以获得归一化的 WY 结果。

---

## 调用约束与执行语义

### 可选参数

- `cuSeqlensOptional` 与 `chunkIndicesOptional` 必须同时提供或同时省略。
- `aLogOptional` 非空 ⇒ fused gate；`dtBiasOptional` 仅在 fused gate 时合法。
- hats / rstd 四个输出必须同时有或同时无。
- `outputA=False` 时仍需传入 `aOut` 占位（ACLNN 要求非空），kernel 可不写 GM。

### `use_exp2`

| `useExp2` | cumsum | 后续 KKT / `k * β * exp*(g)` |
|---|---|---|
| `true` | `g' = RCP_LN2 * chunk_cumsum(g_raw)` | `exp2(g')`（实现为 `exp(ln2 * g')`） |
| `false` | `g' = chunk_cumsum(g_raw)` | 自然 `exp(g')` |

fused gate 的 `softplus` / `exp(A_log)` 始终在自然指数域，与 `use_exp2` 无关；`use_exp2` 只作用在 cumsum 之后。

### 变长模式（VarLen）

提供 `cuSeqlensOptional` 时：

```text
B = 1
T = cu_seqlens[-1]
numChunks = len(chunk_indices) / 2
```

Python 未传 `chunk_indices` 时，按 `cu_seqlens` 以 sequence-major 自动生成。
