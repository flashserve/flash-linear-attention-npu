# ChunkGatedDeltaRuleFwdPrepare 算子说明

`ChunkGatedDeltaRuleFwdPrepare` 是分块门控 delta 规则（Chunk Gated Delta Rule）前向准备阶段的融合算子。它把 L2Norm(q/k)、chunk-local gate cumsum、beta sigmoid、KKT + 严格下三角求解 `A=(I+L)^{-1}`、以及 RecomputeWU 收进一个 MIX AIC:AIV 1:2 kernel。

---

## 1. 算子功能

对每个 `(batch, value-head, chunk)` 计算：

- **q_hat / k_hat / q_rstd / k_rstd**：可选的核内 L2Norm 输出（`use_qk_l2norm_in_kernel=True`）
- **g_cumsum**：chunk 内前缀和后的门控（`use_exp2=True` 时再乘 `RCP_LN2`，后续用 `exp2`）
- **beta_eff**：可选 sigmoid（`allow_neg_eigval=True` 时为 `2 * sigmoid`）
- **A**：`(I+L)^{-1}`，`L` 为带 gate 的严格下三角
- **w / u**：`w = A @ (k̂ * β * exp*(g'))`，`u = A @ (v * β)`

`use_gate_in_kernel=True` 时，核内先做 fused gate：

```text
g_tok[t] = -exp(A_log[hv]) * softplus(g[t] + dt_bias[hv])
```

再对 `g_tok` 做 chunk-local cumsum。此时 `g` 必须是 raw dt logits，且必须提供 `a_log`（`dt_bias` 可省略，视为 0）。

---

## 2. 接口定义

### 2.1 ACLNN 接口

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

## 3. 参数说明

### 3.1 输入参数（Inputs）

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

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `chunkSize` | 输入 | 可选属性，接口侧必传 | 分块大小 | 当前仅 64 | `int64_t` | `64` |
| `allowNegEigval` | 输入 | 可选属性，接口侧必传 | 负特征值缩放 | True 时要求写入 `betaEff`（sigmoid 打开） | `bool` | `true` / `false` |
| `useExp2` | 输入 | 可选属性，接口侧必传 | 指数底 | True：cumsum × `RCP_LN2`，后续 `exp2`；False：cumsum × 1，后续自然 `exp` | `bool` | `true` / `false` |
| `outputA` | 输入 | 可选属性，接口侧必传 | 是否把 A 写回 GM | False 时 A 只留在 L1 | `bool` | `true` / `false` |

IR attr 里还有 `use_qk_l2norm` / `use_gate_in_kernel` / `use_beta_sigmoid`，由 L0 根据 optional 张量填入，不出现在 ACLNN C 接口参数表中。

### 3.3 输出参数（Outputs）

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

### 3.4 形状与约束

- `q`、`k` 必须为 `[B, HK, T, K]` 且完全同形。
- `v` 为 `[B, HV, T, V]`，与 `q` 的 `B`、`T` 一致。
- `g`、`beta`、`gOut`、`betaEff` 为 `[B, HV, T]`。
- `wOut` 为 `[B, HV, T, K]`；`aOut` 为 `[B, HV, T, chunkSize]`。
- **GVA**：`HV % HK == 0` 且 `HV/HK ∈ {1,2,3,4}`。任务按 HV 计数，K 头 `hk = hv / (HV/HK)` 复用。
- `K = 128`，`V ∈ {128, 256}`，`chunkSize = 64`。
- `q/k/v` 当前仅 `BFLOAT16`。
- 变长：`cuSeqlens` 与 `chunkIndices` 成对，且 `B = 1`。
- `allow_neg_eigval=True` 要求 `use_beta_sigmoid=True`。
- `dt_bias` 不能在没有 `a_log` 时单独出现。
- `use_qk_l2norm=False` 时调用方应先对 **q 和 k** 做 L2Norm，否则 WY 会 inf/nan。

---

## 4. 调用约束与执行语义

### 4.1 可选参数

- `cuSeqlensOptional` 与 `chunkIndicesOptional` 必须同时提供或同时省略。
- `aLogOptional` 非空 ⇒ fused gate；`dtBiasOptional` 仅在 fused gate 时合法。
- hats / rstd 四个输出必须同时有或同时无。
- `outputA=False` 时仍需传入 `aOut` 占位（ACLNN 要求非空），kernel 可不写 GM。

### 4.2 `use_exp2`

| `useExp2` | cumsum | 后续 KKT / `k * β * exp*(g)` |
|---|---|---|
| `true`（FLA 默认） | `g' = RCP_LN2 * chunk_cumsum(g_raw)` | `exp2(g')`（实现为 `exp(ln2 * g')`） |
| `false` | `g' = chunk_cumsum(g_raw)` | 自然 `exp(g')` |

fused gate 的 `softplus` / `exp(A_log)` 始终在自然指数域，与 `use_exp2` 无关；`use_exp2` 只作用在 cumsum 之后。

### 4.3 变长模式（VarLen）

提供 `cuSeqlensOptional` 时：

```text
B = 1
T = cu_seqlens[-1]
numChunks = len(chunk_indices) / 2
```

Python 未传 `chunk_indices` 时，按 `cu_seqlens` 以 sequence-major 自动生成。

---

## 5. Torch 测试调用示例

### 5.1 固定长度（默认 flag：L2Norm + sigmoid + exp2，无 fused gate）

```python
import torch
from fla_npu.ops import ascendc as ascendc_ops

def test_chunk_gated_delta_rule_fwd_prepare_fix():
    B, HK, HV, T, K, V = 1, 4, 8, 256, 128, 128
    q = torch.randn(B, HK, T, K, dtype=torch.bfloat16).npu()
    k = torch.randn(B, HK, T, K, dtype=torch.bfloat16).npu()
    v = torch.randn(B, HV, T, V, dtype=torch.bfloat16).npu()
    g = torch.log(torch.rand(B, HV, T).clamp_min(1e-4)).npu()
    beta = torch.randn(B, HV, T, dtype=torch.float32).npu()

    q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, A = (
        ascendc_ops.chunk_gated_delta_rule_fwd_prepare(
            q, k, v, g, beta,
            chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            use_exp2=True,
        )
    )
    assert w.shape == (B, HV, T, K)
    assert A.shape == (B, HV, T, 64)
    print("Fix-length Execution Successful!")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_prepare_fix()
```

### 5.2 fused gate + `use_exp2=False`

```python
import torch
from fla_npu.ops import ascendc as ascendc_ops

def test_prepare_fused_gate_natural_exp():
    B, HK, HV, T, K, V = 1, 4, 8, 192, 128, 128
    q = torch.randn(B, HK, T, K, dtype=torch.bfloat16).npu()
    k = torch.randn(B, HK, T, K, dtype=torch.bfloat16).npu()
    v = torch.randn(B, HV, T, V, dtype=torch.bfloat16).npu()
    g = torch.randn(B, HV, T, dtype=torch.float32).npu()          # raw dt logits
    beta = torch.randn(B, HV, T, dtype=torch.float32).npu()
    a_log = (torch.randn(HV) * 0.1).npu()
    dt_bias = (torch.randn(HV) * 0.1).npu()

    q_hat, k_hat, q_rstd, k_rstd, beta_out, g_cumsum, w, u, A = (
        ascendc_ops.chunk_gated_delta_rule_fwd_prepare(
            q, k, v, g, beta,
            chunk_size=64,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=False,
            use_exp2=False,
            a_log=a_log,
            dt_bias=dt_bias,
        )
    )
    print("g_cumsum", tuple(g_cumsum.shape), "w", tuple(w.shape))

if __name__ == "__main__":
    test_prepare_fused_gate_natural_exp()
```

### 5.3 变长模式

```python
import torch
from fla_npu.ops import ascendc as ascendc_ops

def test_prepare_varlen():
    seqlens = (96, 64)
    B, HK, HV, T, K, V = 1, 4, 8, sum(seqlens), 128, 128
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int64)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int64).cumsum(0)
    q = torch.randn(B, HK, T, K, dtype=torch.bfloat16).npu()
    k = torch.randn(B, HK, T, K, dtype=torch.bfloat16).npu()
    v = torch.randn(B, HV, T, V, dtype=torch.bfloat16).npu()
    g = torch.log(torch.rand(B, HV, T).clamp_min(1e-4)).npu()
    beta = torch.randn(B, HV, T, dtype=torch.float32).npu()

    outs = ascendc_ops.chunk_gated_delta_rule_fwd_prepare(
        q, k, v, g, beta,
        chunk_size=64,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        use_exp2=True,
        cu_seqlens=cu.npu(),
    )
    print("varlen w", tuple(outs[6].shape))

if __name__ == "__main__":
    test_prepare_varlen()
```

---

## 6. 目录结构

```text
chunk_gated_delta_rule_fwd_prepare/
├── README.md
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_chunk_gated_delta_rule_fwd_prepare.cpp
│   │   ├── aclnn_chunk_gated_delta_rule_fwd_prepare.h
│   │   ├── chunk_gated_delta_rule_fwd_prepare.cpp
│   │   └── chunk_gated_delta_rule_fwd_prepare.h
│   ├── chunk_gated_delta_rule_fwd_prepare_tiling.cpp
│   ├── chunk_gated_delta_rule_fwd_prepare_tiling.h
│   ├── chunk_gated_delta_rule_fwd_prepare_def.cpp
│   ├── chunk_gated_delta_rule_fwd_prepare_infershape.cpp
│   └── CMakeLists.txt
├── op_kernel/
│   ├── chunk_gated_delta_rule_fwd_prepare.cpp
│   └── arch35/
│       ├── chunk_gated_delta_rule_fwd_prepare.h
│       ├── chunk_gated_delta_rule_fwd_prepare_common.h
│       ├── chunk_gated_delta_rule_fwd_prepare_vf.h
│       ├── chunk_gated_delta_rule_fwd_prepare_matmul.h
│       └── chunk_gated_delta_rule_fwd_prepare_datacopy.h
└── test/
    ├── test_chunk_gated_delta_rule_fwd_prepare.py
    ├── test_chunk_gated_delta_rule_fwd_prepare_pipe_acc.py
    ├── test_chunk_gated_delta_rule_fwd_prepare_perf.py
    ├── test_chunk_gated_delta_rule_fwd_prepare_l2norm_false.py
    └── cpu_golden.py
```

Python 入口：`fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_prepare`。

编译（算子 IR 名，不是旧名 `chunk_gdn_fwd_prepare`）：

```bash
bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu \
  --ops=chunk_gated_delta_rule_fwd_prepare -j8
```
