# ChunkGatedDeltaRuleFwd32 算子说明

`ChunkGatedDeltaRuleFwd32` 是 Gated Delta Rule 的融合前向算子。一次 launch 完成原来拆开的 prepare、状态递推和输出计算，直接写出注意力输出 `o` 和每条序列的 `final_state`。块长固定为 32，只注册 `ascend950`。

Q、K、V 使用 head-major 布局 `[B, H, T, D]`，与同目录算子里的 `BNSD` / `NTD` 是同一种物理形状。`BSND` / `TND` 不在本算子范围内。

---

## 1. 算子功能

对每个 value head，状态 \(R\) 是 \(128 \times 128\)。没有 `initial_state` 时从 0 开始。每个 token 上：

1. 对 \(q\)、\(k\) 做 L2 归一化，对 `beta` 做 sigmoid。\(v\) 不做 L2。
2. 用门控 \(g\) 衰减上一步状态，再按 delta rule 写回。
3. 用更新后的状态和归一化后的 \(q\) 得到 \(o\)，并乘 `scale`。

片上按 32 个 token 一块计算。一块内部做 \(K K^{\mathsf T}\) / \(Q K^{\mathsf T}\)、\((I+L)\) 的 \(32 \times 32\) 求逆，以及状态更新。最后一块不足 32 行时，在 UB 里补零到 32，只把有效行写入 `o`。`g_cumsum`、\(A\)、\(w\)、\(u\)、\(v_{\mathrm{new}}\) 和块间状态只活在片上，不写出 GM。

指数是自然指数 `exp`，不是 `exp2`。L2 和 beta sigmoid 固定打开，没有单独的开关。

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**
   调用 `aclnnChunkGatedDeltaRuleFwd32GetWorkspaceSize`，获取 workspace 大小并创建执行器。

2. **执行算子计算**
   调用 `aclnnChunkGatedDeltaRuleFwd32`，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口（见 `op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_32.h`）：

```cpp
/* function: aclnnChunkGatedDeltaRuleFwd32GetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * v : required
 * g : required
 * beta : required
 * initialStateOptional : optional
 * cuSeqlensOptional : optional
 * chunkIndicesOptional : optional
 * scale : required
 * o : required
 * finalState : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwd32GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    const aclTensor *o,
    const aclTensor *finalState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnChunkGatedDeltaRuleFwd32
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwd32(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

Python 入口是 `npu_chunk_gated_delta_rule_fwd_32`，返回 `(o, final_state)`。`scale=None` 时取 `1/sqrt(K)`。只给 `cu_seqlens` 时，Python 按块长 32 生成 canonical `chunk_indices`。

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|
| `q` | 输入 | 必选 | Query | `BFLOAT16` | `ND` | BNSD / NTD `[B, Hk, T, K]` | 支持 |
| `k` | 输入 | 必选 | Key | `BFLOAT16` | `ND` | BNSD / NTD `[B, Hk, T, K]` | 支持 |
| `v` | 输入 | 必选 | Value | `BFLOAT16` | `ND` | BNSD / NTD `[B, Hv, T, V]` | 支持 |
| `g` | 输入 | 必选 | 原始门控，kernel 内做 cumsum 和 `exp` | `FLOAT` | `ND` | BNSD / NTD `[B, Hv, T]` | 支持 |
| `beta` | 输入 | 必选 | Delta 系数，kernel 内做 sigmoid | `BFLOAT16` | `ND` | BNSD / NTD `[B, Hv, T]` | 支持 |
| `initial_state` | 输入 | 可选 | 每条序列的初始状态 | `BFLOAT16` | `ND` | BNSD `[B, Hv, K, V]`；NTD `[N, Hv, K, V]` | 支持 |
| `cu_seqlens` | 输入 | 可选 | NTD 的累计长度。给出后进入 NTD | `INT64` | `ND` | `[N+1]` | - |
| `chunk_indices` | 输入 | 可选 | NTD 的 `(seqId, localChunkId)` 扁平列表 | `INT64` | `ND` | `[2·Nc]` | - |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|
| `scale` | 输入 | 必选 | Query 缩放系数 | `float` | 缺省 `0.0883883476`，即 \(1/\sqrt{128}\) |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `o` | 输出 | 前向注意力输出 | `BFLOAT16` | `ND` | BNSD / NTD `[B, Hv, T, V]` | 支持 |
| `final_state` | 输出 | 每条序列最后一个 token 之后的状态 | `BFLOAT16` | `ND` | BNSD `[B, Hv, K, V]`；NTD `[N, Hv, K, V]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器 | `aclOpExecutor*` | - | - | - |

`final_state` 的末两维是 `[K, V]`。本算子不支持 `stateVFirst`。

### 3.4 形状与约束

BNSD 和 NTD 的轴顺序相同，都是 `[B, H, T, D]`。本算子没有 `layout` 属性，靠 `cu_seqlens` 区分这两种布局。`BSND` / `TND`（`[B, T, H, D]`）不支持。

- **BNSD**：不传 `cu_seqlens`。`B` 是等长序列条数，`N = B`。`q/k` 为 `[B, Hk, T, K]`，`v/g/beta/o` 的 head 维是 `Hv`，`initial_state` 和 `final_state` 为 `[B, Hv, K, V]`。
- **NTD**：传 `cu_seqlens`。物理 `B = 1`，多条序列按顺序排在 `T` 里。`N = len(cu_seqlens) - 1`。`q/k/v/g/beta/o` 的 batch 维仍是 1，`T` 是拼接后的总长度；`initial_state` 和 `final_state` 为 `[N, Hv, K, V]`。
- `q`、`k` 必须同形。`q` 和 `v` 的 `B`、`T` 一致。
- `K = 128`，`V = 128`。
- `Hv % Hk == 0`。GVA 时 key head 为 `hv / (Hv / Hk)`，多个 value head 重读同一份 q、k。
- `T > 0`。`T` 不必是 32、64 或 128 的倍数。
- 按 value head 分核：`blockDim = min(B·Hv, AIC 核数)`。
- 只支持 `ascend950`。tiling key 为 0。host 只申请 lib workspace。

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `cu_seqlens` 和 `chunk_indices` 必须同时给出，或同时不给。
- 只给 `chunk_indices` 会报错。Python 侧只给 `cu_seqlens` 时，会按块长 32 补出 canonical `chunk_indices`。
- `initial_state` 可空。给出时形状与 `final_state` 相同。

### 4.2 形状约束（强约束）

必须满足以下条件：

- `q, k`: BNSD / NTD `[B, Hk, T, K]`
- `v, o`: BNSD / NTD `[B, Hv, T, V]`
- `g`: BNSD / NTD `[B, Hv, T]`，`FLOAT`
- `beta`: BNSD / NTD `[B, Hv, T]`，`BFLOAT16`
- `initial_state`, `final_state`: BNSD `[B, Hv, K, V]`；NTD `[N, Hv, K, V]`
- `HV % HK == 0`
- `K = V = 128`
- `T > 0`

### 4.3 BNSD

不传 `cu_seqlens`。一个 `(batch, value head)` 是一个任务。一个核上的下一个任务在写出本任务 `final_state` 之前发出自己的 Prepare 读，这段 GM 读和状态写出重叠。

`B > 1` 时每行等长，状态槽就是 batch 下标。相邻 batch 行通常落在不同核上，核内重叠的是该核自己的下一个任务，不是同一 head 的相邻 batch。

### 4.4 NTD

传入 `cu_seqlens` 后即是 NTD：

- 物理 `B = 1`，多条序列按顺序排在 `T` 里
- `cu_seqlens[0] = 0`，`cu_seqlens[N] = T`，单调不降。相邻值相等表示空序列
- `chunk_indices` 必须是按序列、按块排列的 `(seqId, localChunkId)`，长度 `2·Nc`，块长 32。空序列不占 chunk，但仍占一个状态槽
- 每条序列从自己的起始 token 重新分块，起始 token 不必 32 对齐
- 一条序列结束时写出该序列的 `final_state`，下一条从自己的 `initial_state` 或 0 重新开始。状态不跨序列传递
- 下一条序列的 Prepare 读在本条 `final_state` 写出之前发出，两段在同一核上重叠
- 空序列不跑 cube。有 `initial_state` 时把对应槽抄到 `final_state`，否则该槽写 0

同一 head 上的多条序列属于同一个任务，按 `cu_seqlens` 的顺序在这个核上串行处理。

### 4.5 数值语义

- `scale` 在 ACL 属性里必填，缺省 \(1/\sqrt{128}\)。Python 传 `None` 时同样取 \(1/\sqrt{K}\)。
- `o` 在写出前乘 `scale`。
- 比较精度时，把 FP64 标杆先 cast 成 BF16 再转回 FP32。`o` 和 `final_state` 的通过标准是 BF16 ulp 比值 `max < 5`、`mean < 1.5`、`rms < 1.5`。

---

## 5. Torch 测试调用示例

### 5.1 定长模式调用示例

```python
import math

import torch
import torch_npu
from fla_npu.ops import ascendc as ascendc_ops

def test_chunk_gated_delta_rule_fwd_32_fixed():
    B, HK, HV, T, K, V = 1, 2, 4, 96, 128, 128
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.bfloat16

    q = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    k = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    v = torch.randn(B, HV, T, V, device=device, dtype=dtype)
    g = torch.randn(B, HV, T, device=device, dtype=torch.float32) * 0.02
    beta = torch.randn(B, HV, T, device=device, dtype=dtype)
    initial_state = torch.randn(B, HV, K, V, device=device, dtype=dtype) * 0.01

    o, final_state = ascendc_ops.npu_chunk_gated_delta_rule_fwd_32(
        q, k, v, g, beta,
        initial_state=initial_state,
        scale=scale,
    )

    print("o shape:", tuple(o.shape))
    print("final_state shape:", tuple(final_state.shape))
    assert o.shape == (B, HV, T, V)
    assert final_state.shape == (B, HV, K, V)
    print("Execution Successful!")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_32_fixed()
```

`B > 1` 时 `final_state` 的第 0 维仍是 `B`。不需要 `initial_state` 时传 `None`。

### 5.2 变长模式调用示例

变长模式下 `B = 1`。`T` 是若干序列拼起来的长度，`cu_seqlens` 长度是 `N + 1`。`chunk_indices` 可以不传，由 `cu_seqlens` 按块长 32 生成。

```python
import math

import torch
import torch_npu
from fla_npu.ops import ascendc as ascendc_ops

def test_chunk_gated_delta_rule_fwd_32_varlen():
    B, HK, HV, K, V = 1, 2, 4, 128, 128
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.bfloat16

    seqlens = [64, 0, 33]                       # 中间一条是空序列
    T = sum(seqlens)
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)  # [0, 64, 64, 97]
    nseq = len(seqlens)

    q = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    k = torch.randn(B, HK, T, K, device=device, dtype=dtype)
    v = torch.randn(B, HV, T, V, device=device, dtype=dtype)
    g = torch.randn(B, HV, T, device=device, dtype=torch.float32) * 0.02
    beta = torch.randn(B, HV, T, device=device, dtype=dtype)
    initial_state = torch.randn(nseq, HV, K, V, device=device, dtype=dtype) * 0.01

    o, final_state = ascendc_ops.npu_chunk_gated_delta_rule_fwd_32(
        q, k, v, g, beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
    )

    print("o shape:", tuple(o.shape))
    print("final_state shape:", tuple(final_state.shape))
    assert o.shape == (B, HV, T, V)
    assert final_state.shape == (nseq, HV, K, V)
    print("Execution Successful!")

if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_32_varlen()
```

> 精度脚本见同目录 `test_precision.py`、`test_unaligned_t.py`、`test_multibatch.py`、`test_ntd.py`、`test_gva.py`、`test_cu_seqlens.py`。标杆是 `cpu_golden_fused_infer.py` 里的逐 token FP64 递推。片上公式见 `chunk_gated_delta_rule_fwd_fused_infer_ascendc-design.md`。

---

## 6. 目录结构

```text
chunk_gated_delta_rule_fwd_32/
├── CMakeLists.txt
├── README.md
├── chunk_gated_delta_rule_fwd_fused_infer_ascendc-design.md
├── cpu_golden_fused_infer.py
├── test_precision.py
├── test_unaligned_t.py
├── test_multibatch.py
├── test_ntd.py
├── test_gva.py
├── test_cu_seqlens.py
├── op_host/
│   ├── CMakeLists.txt
│   ├── chunk_gated_delta_rule_fwd32_def.cpp
│   ├── chunk_gated_delta_rule_fwd_32_infershape.cpp
│   ├── chunk_gated_delta_rule_fwd_32_tiling.cpp
│   ├── chunk_gated_delta_rule_fwd_32_tiling.h
│   └── op_api/
│       ├── aclnn_chunk_gated_delta_rule_fwd_32.cpp
│       ├── aclnn_chunk_gated_delta_rule_fwd_32.h
│       └── chunk_gated_delta_rule_fwd_32.cpp
└── op_kernel/
    ├── chunk_gated_delta_rule_fwd_32.cpp
    └── arch35/
        ├── chunk_gated_delta_rule_fwd_32.h
        ├── chunk_gated_delta_rule_fwd_32_common.h
        ├── chunk_gated_delta_rule_fwd_32_matmul.h
        ├── chunk_gated_delta_rule_fwd_32_tiling_data.h
        └── chunk_gated_delta_rule_fwd_32_vf.h
```
