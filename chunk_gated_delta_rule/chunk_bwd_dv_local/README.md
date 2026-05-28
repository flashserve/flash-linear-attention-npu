# ChunkBwdDvLocal 算子说明
`ChunkBwdDvLocal` 是 Gated Delta Rule（门控 Delta 规则）线性注意力机制反向传播过程中的自定义算子。该算子根据查询（Q）、键（K）、门控（g）和输出梯度（dO）计算 Value 的本地梯度 $dV_{\text{local}}$。

---

## 1. 算子功能

在分块序列模型反向传播中，计算以下张量的梯度：

- **dV**：Value 的本地梯度

计算公式为：

$$dV_{\text{local}} = \text{mask}\!\left(\exp(g_{\text{col}} - g_{\text{row}})\right) \odot (K Q^T) \cdot dO$$

分解为三个阶段：

| 阶段 | 执行单元 | 计算 | 说明 |
|------|----------|------|------|
| Phase 1 | Cube (AIC) | $W_s = K \times Q^T$ | 矩阵乘，生成 chunk 内 attention score 矩阵 |
| Phase 1.5 | Vector (AIV) | $W_{s\_gated} = \text{mask}(\exp(g)) \odot W_s$ | gating：exp、下三角 mask、逐元素乘 |
| Phase 2 | Cube (AIC) | $dV = W_{s\_gated} \times dO$ | 矩阵乘，生成最终梯度输出 |

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnChunkBwdDvLocalGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnChunkBwdDvLocal` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：
```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnChunkBwdDvLocalGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *dO,
    const aclTensor *g,
    const aclTensor *gGammaOptional,
    const aclTensor *aOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnChunkBwdDvLocal(
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
| `q` | 输入 | 必选 | Query 输入张量 | 参与反向计算 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `k` | 输入 | 必选 | Key 输入张量 | 参与反向计算 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `dO` | 输入 | 必选 | 前向输出的梯度张量 | 参与反向计算 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `g` | 输入 | 必选 | Gate 输入张量（门控衰减系数） | 参与 gating 计算 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `gGammaOptional` | 输入 | 可选 | 预留门控参数输入 | 当前版本不支持，须传 `None` | `FLOAT` | `ND` | - | - |
| `aOptional` | 输入 | 可选 | 预留参数输入 | 当前版本不支持，须传 `None` | `FLOAT16`、`BFLOAT16` | `ND` | - | - |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列的累计长度信息 | 变长模式输入，形如 `[0, T1, T1+T2, ...]`，形状为 `[N+1]`（N 为 batch 内序列段数，等于 B） | `INT64` | `ND` | 1 维 | - |
| `chunkIndicesOptional` | 输入 | 可选 | 分块索引信息 | 变长模式输入，形状为 `[num_chunks, 2]`（num_chunks 为所有序列的总分块数，每行为 `[batch_idx, chunk_idx]`） | `INT64` | `ND` | 2 维 | - |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `scale` | 输入 | 必选 | 缩放系数 | 推荐按 `1 / sqrt(K)` 设置 | `double` | 建议按 `1 / sqrt(K)` 设置 |
| `chunkSize` | 输入 | 必选 | 分块大小 | 仅支持 `64` 或 `128` | `int64_t` | 仅支持 `64` / `128` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `out`（即 `dV`） | 输出 | Value 的本地梯度输出张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

### 3.4 形状与约束

- `q`、`k` 的形状必须为 `[B, H, T, K]`。
- `dO` 的形状必须为 `[B, H, T, V]`。
- `g` 的形状必须为 `[B, H, T]`。
- `K` 须为 `128`。
- `V` 须为 `128` 或 `256`。
- `chunkSize` 仅支持 `64` 或 `128`。
- 当启用变长模式时，`cuSeqlensOptional` 和 `chunkIndicesOptional` 必须同时提供；变长模式仅支持 `B = 1`。
- `gGammaOptional` 和 `aOptional` 须传 `None`（当前版本不支持）。

### 3.5 补充说明

- `q`、`k`、`dO`、`g` 支持非连续 Tensor 输入。
- 输出 `dV` 支持非连续视图作为输出目标。

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `cuSeqlensOptional` 和 `chunkIndicesOptional`：
  - 同时出现时启用变长模式（varlen）
  - 变长模式仅支持 `B = 1`

- `gGammaOptional` 和 `aOptional`：
  - 接口层存在，但当前实现未启用
  - **必须传入空指针，否则执行失败**

---

### 4.2 形状约束（强约束）

必须满足以下条件：

- `q, k`: `[B, H, T, K]`
- `dO`: `[B, H, T, V]`
- `g`: `[B, H, T]`

额外限制：

- `K = 128`
- `V = 128` 或 `V = 256`
- `chunkSize ∈ {64, 128}`

---

### 4.3 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时：

- `chunkIndicesOptional` 必须同时提供
- 当前实现仅支持：

```text
B = 1
```

---

### 4.4 数值语义

- `scale`：
  - 必须显式传入
  - 推荐设置为：

```text
1 / sqrt(K)
```

---

## 5. Torch 测试调用示例

### 5.1 定长场景（Padding-mode）

```python
import torch
import torch_npu
import math

def test_chunk_bwd_dv_local_fixed():
    B, H, T, K, V = 2, 4, 128, 128, 128
    chunk_size = 64
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.bfloat16

    q   = torch.randn(B, H, T, K, device=device, dtype=dtype)
    k   = torch.randn(B, H, T, K, device=device, dtype=dtype)
    d_o = torch.randn(B, H, T, V, device=device, dtype=dtype)
    g   = torch.randn(B, H, T,    device=device, dtype=torch.float32)

    # 定长模式：cu_seqlens=None, chunk_indices=None
    dv = torch_npu.npu_chunk_bwd_dv_local(
        q, k, d_o, g,
        g_gamma=None,
        A=None,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        chunk_size=chunk_size
    )

    assert dv.shape == (B, H, T, V)
    print("dv shape:", dv.shape)  # (2, 4, 128, 128)

if __name__ == "__main__":
    test_chunk_bwd_dv_local_fixed()
```

### 5.2 变长场景（VarLen-mode）

变长模式需要额外准备 `cu_seqlens` 和 `chunk_indices`（当前仅支持 `B = 1`）。

```python
import torch
import torch_npu
import math

def prepare_chunk_indices(cu_seqlens, chunk_size):
    """根据累积序列长度生成分块索引，返回形如 [num_chunks, 2] 的张量。"""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    result = []
    for batch_idx, sl in enumerate(seq_lens.tolist()):
        num_chunks = (sl + chunk_size - 1) // chunk_size
        for ci in range(num_chunks):
            result.append([batch_idx, ci])
    return torch.tensor(result, dtype=torch.long)

def test_chunk_bwd_dv_local_varlen():
    H, K, V = 4, 128, 128
    chunk_size = 64
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.bfloat16

    # 变长：B=1，包含两段序列，长度分别为 192 和 64，总长 256
    cu_seqlens = torch.tensor([0, 192, 256], dtype=torch.long)
    T = int(cu_seqlens[-1])

    q   = torch.randn(1, H, T, K, device=device, dtype=dtype)
    k   = torch.randn(1, H, T, K, device=device, dtype=dtype)
    d_o = torch.randn(1, H, T, V, device=device, dtype=dtype)
    g   = torch.randn(1, H, T,    device=device, dtype=torch.float32)

    # 每行 [batch_idx, chunk_idx]，展平后传入
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    dv = torch_npu.npu_chunk_bwd_dv_local(
        q, k, d_o, g,
        g_gamma=None,
        A=None,
        cu_seqlens=cu_seqlens.tolist(),
        chunk_indices=chunk_indices.flatten().tolist(),
        scale=scale,
        chunk_size=chunk_size
    )

    assert dv.shape == (1, H, T, V)
    print("dv shape:", dv.shape)  # (1, 4, 256, 128)

if __name__ == "__main__":
    test_chunk_bwd_dv_local_varlen()
```

---

## 6. 目录结构

```text
chunk_bwd_dv_local/
├── examples/
│   └── test_aclnn_chunk_bwd_dv_local_variable.cpp
├── op_api/
│   ├── aclnn_chunk_bwd_dv_local.cpp
│   ├── aclnn_chunk_bwd_dv_local.h
│   ├── chunk_bwd_dv_local.cpp
│   └── chunk_bwd_dv_local.h
├── op_host/
│   ├── chunk_bwd_dv_local_def.cpp
│   ├── chunk_bwd_dv_local_tiling.cpp
│   ├── chunk_bwd_dv_local_tiling.h
│   └── CMakeLists.txt
└── op_kernel/
    ├── chunk_bwd_dv_local_common.h
    ├── chunk_bwd_dv_local_cube.h
    ├── chunk_bwd_dv_local_vector.h
    └── chunk_bwd_dv_local.cpp
```
