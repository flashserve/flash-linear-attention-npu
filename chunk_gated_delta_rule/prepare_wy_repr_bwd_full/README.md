# PrepareWyReprBwdFull 算子说明
`PrepareWyReprBwdFull` 是一个用于分块门控 delta 规则（Chunk Gated Delta Rule）中 WY 表示（WY Representation）反向传播过程的自定义算子。该算子根据前向保存的中间矩阵、权重及上游梯度，计算并输出针对 Key（k）、Value（v）、Beta（β）和 Gate（g）的梯度。

---

## 1. 算子功能

在分块序列模型的 WY 表示反向传播阶段，计算以下张量的梯度：

- **dk**：Key 的梯度
- **dv**：Value 的梯度
- **dbeta**：Beta（衰减/缩放参数）的梯度
- **dg**：Gate（门控）的梯度

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnPrepareWyReprBwdFullGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnPrepareWyReprBwdFull` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：
```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnPrepareWyReprBwdFullGetWorkspaceSize(
    const aclTensor *k, const aclTensor *v, const aclTensor *beta,
    const aclTensor *a, const aclTensor *dA, const aclTensor *dw, const aclTensor *du,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnPrepareWyReprBwdFull(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream
);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|---|
| `k` | 输入 | 必选 | Key 输入张量 | 参与反向计算 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `v` | 输入 | 必选 | Value 输入张量 | 参与反向计算 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `beta` | 输入 | 必选 | Beta 缩放参数张量 | 三维 Beta 参数 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `a` | 输入 | 必选 | 前向保存的块内因果注意力矩阵 A | 每个 chunk 内的下三角因果矩阵，最后一维为 `chunkSize` | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, chunkSize]` | 支持 |
| `dA` | 输入 | 必选 | 矩阵 A 的上游梯度 | 与 `a` 形状相同 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, chunkSize]` | 支持 |
| `dw` | 输入 | 必选 | Weight w 的上游梯度 | 形状与 `k` 相同 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `du` | 输入 | 必选 | 更新量 u 的上游梯度 | 形状与 `v` 相同 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `g` | 输入 | 必选 | Gate 输入张量 | 三维 Gate 参数 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列的累计长度信息 | 变长模式输入，形状为 `[N+1]`，首元素须为 `0` | `INT64` | `ND` | 1 维 | - |
| `chunkIndicesOptional` | 输入 | 可选 | 分块索引信息 | 变长模式输入，扁平化存储 `[seqIdx0, chunkIdx0, seqIdx1, chunkIdx1, ...]`，长度为 `2 * numChunks` | `INT64` | `ND` | 1 维 | - |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `chunkSize` | 输入 | 必选属性 | 分块大小 | 默认值为 `64`，仅支持 `64` 或 `128` | `int64_t` | 仅支持 `64` / `128` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `dkOut` | 输出 | Key 梯度输出张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `dvOut` | 输出 | Value 梯度输出张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `dbetaOut` | 输出 | Beta 梯度输出张量 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `dgOut` | 输出 | Gate 梯度输出张量 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

### 3.4 形状与约束

- `k`、`dw` 的形状必须为 `[B, H, T, K]`。
- `v`、`du` 的形状必须为 `[B, H, T, V]`。
- `beta`、`g` 的形状必须为 `[B, H, T]`。
- `a`、`dA` 的形状必须为 `[B, H, T, chunkSize]`。
- `v` 与 `k` 的前三维（B、H、T）必须一致。
- `v` 与 `du` 的全部四维必须一致。
- `k` 与 `dw` 的全部四维必须一致。
- `a` 与 `dA` 的全部四维必须一致。
- `beta`、`g`、`k` 的前三维（B、H、T）必须一致。
- `chunkSize` 当前仅支持 `64` 或 `128`。
- 当启用变长模式时，`cuSeqlensOptional` 和 `chunkIndicesOptional` 须同时提供，且仅支持 `B = 1`。

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `cuSeqlensOptional` 和 `chunkIndicesOptional`：
  - 必须同时提供或同时省略
  - 同时出现时启用变长模式（varlen）
  - `cuSeqlensOptional` 的首元素（`cuSeqlens[0]`）必须为 `0`，且各序列长度严格为正
  - `chunkIndicesOptional` 为扁平化 1D 数组，内容格式为 `[seqIdx0, chunkIdx0, seqIdx1, chunkIdx1, ...]`，长度为 `2 * numChunks`
  - `chunkIndicesOptional` 的内容须与 `cuSeqlensOptional` 严格对应
  - 变长模式仅支持 `B = 1`

---

### 4.2 形状约束（强约束）

必须满足以下条件：

- `k, dw`: `[B, H, T, K]`
- `v, du`: `[B, H, T, V]`
- `beta, g`: `[B, H, T]`
- `a, dA`: `[B, H, T, chunkSize]`

额外限制：

- `chunkSize ∈ {64, 128}`

---

### 4.3 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时：

- `chunkIndicesOptional` 必须同时提供
- `numChunks` 由 `chunkIndices` 的长度推导（`len(chunkIndices) / 2`）
- 要求 `cuSeqlens[0] == 0`，且各段序列长度严格为正
- 当前实现仅支持：

```text
B = 1
```

---

### 4.4 数值语义

算子在每个 chunk 内执行以下反向传播计算（以伪代码表示）：

**dv**（Value 的梯度）：
```text
dv_chunk = A_chunk^T @ du_chunk * beta_chunk[:, None]
```

**dk**（Key 的梯度）：
```text
b_dk  = dA_chunk @ (k_chunk * beta_chunk[:, None])
      + dA_chunk^T @ k_chunk * beta_chunk[:, None]
      + A_chunk^T @ dw_chunk * (beta_chunk * exp(g_chunk))[:, None]
```

**dbeta**（Beta 的梯度）：
```text
dbeta_chunk = sum(dA_chunk^T @ k_chunk * k_chunk, dim=1)
            + sum(A_chunk^T @ dw_chunk * k_chunk * exp(g_chunk)[:, None], dim=1)
            + sum(A_chunk^T @ du_chunk * v_chunk, dim=1)
```

**dg**（Gate 的梯度）：
```text
b_kkt = k_chunk @ k_chunk^T * beta_chunk[:, None]
dg_chunk = sum(A_chunk^T @ dw_chunk * k_chunk * beta_chunk[:, None] * exp(g_chunk)[:, None], dim=1)
         + sum(dA_chunk^T * b_kkt, dim=1) - sum(dA_chunk^T * b_kkt, dim=0)
```

---

## 5. Torch 测试调用示例

### 5.1 固定长度模式

```python
import torch
import torch_npu

def test_prepare_wy_repr_bwd_full_fix():
    # 参数设置
    B, H, T, K, V = 1, 8, 512, 128, 128
    chunk_size = 64
    device = "npu:0"
    ktype = torch.float16
    btype = torch.float16

    # 构造输入
    k    = torch.rand(B, H, T, K,          dtype=ktype).to(device)
    v    = torch.rand(B, H, T, V,          dtype=ktype).to(device)
    beta = torch.rand(B, H, T,             dtype=btype).to(device)
    A    = torch.rand(B, H, T, chunk_size, dtype=ktype).to(device)
    dA   = torch.rand(B, H, T, chunk_size, dtype=ktype).to(device)
    dw   = torch.rand(B, H, T, K,          dtype=ktype).to(device)
    du   = torch.rand(B, H, T, V,          dtype=ktype).to(device)
    g    = torch.rand(B, H, T,             dtype=btype).to(device)

    # 调用算子（固定长度模式）
    dk, dv, dbeta, dg = torch_npu.npu_prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, du, g,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=chunk_size
    )

    print("dk    shape:", dk.shape)
    print("dv    shape:", dv.shape)
    print("dbeta shape:", dbeta.shape)
    print("dg    shape:", dg.shape)
    assert dk.shape    == (B, H, T, K)
    assert dv.shape    == (B, H, T, V)
    assert dbeta.shape == (B, H, T)
    assert dg.shape    == (B, H, T)
    print("Fix-length Execution Successful!")

if __name__ == "__main__":
    test_prepare_wy_repr_bwd_full_fix()
```

### 5.2 变长模式

```python
import torch
import torch_npu
import random

def prepare_cu_seqlens(total_length: int, num_seqs: int, seed: int = 42):
    """生成随机 cu_seqlens，首元素为 0，末元素为 total_length"""
    random.seed(seed)
    if num_seqs < 2 or num_seqs > total_length + 1:
        raise ValueError("num_seqs must satisfy 2 <= num_seqs <= total_length + 1")
    if num_seqs == 2:
        return [0, total_length]
    middle = sorted(random.sample(range(1, total_length), num_seqs - 2))
    return [0] + middle + [total_length]

def prepare_chunk_indices(cu_seqlens, chunk_size: int):
    """根据 cu_seqlens 生成扁平化 chunk_indices"""
    chunk_indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices

def test_prepare_wy_repr_bwd_full_varlen():
    # 参数设置（变长模式要求 B = 1）
    B, H, T, K, V = 1, 8, 512, 128, 128
    chunk_size = 64
    device = "npu:0"
    ktype = torch.float16
    btype = torch.float16

    # 生成变长序列信息（将 T 个 token 随机切分为 4 段）
    cu_seqlens    = prepare_cu_seqlens(total_length=T, num_seqs=5)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    # 构造输入
    k    = torch.rand(B, H, T, K,          dtype=ktype).to(device)
    v    = torch.rand(B, H, T, V,          dtype=ktype).to(device)
    beta = torch.rand(B, H, T,             dtype=btype).to(device)
    A    = torch.rand(B, H, T, chunk_size, dtype=ktype).to(device)
    dA   = torch.rand(B, H, T, chunk_size, dtype=ktype).to(device)
    dw   = torch.rand(B, H, T, K,          dtype=ktype).to(device)
    du   = torch.rand(B, H, T, V,          dtype=ktype).to(device)
    g    = torch.rand(B, H, T,             dtype=btype).to(device)

    # 调用算子（变长模式）
    dk, dv, dbeta, dg = torch_npu.npu_prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, du, g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size
    )

    print("dk    shape:", dk.shape)
    print("dv    shape:", dv.shape)
    print("dbeta shape:", dbeta.shape)
    print("dg    shape:", dg.shape)
    assert dk.shape    == (B, H, T, K)
    assert dv.shape    == (B, H, T, V)
    assert dbeta.shape == (B, H, T)
    assert dg.shape    == (B, H, T)
    print("Variable-length Execution Successful!")

if __name__ == "__main__":
    test_prepare_wy_repr_bwd_full_varlen()
```

---

## 6. 目录结构

```text
prepare_wy_repr_bwd_full/
├── examples/
│   └── test_aclnn_prepare_wy_repr_bwd_full.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_prepare_wy_repr_bwd_full.cpp
│   │   ├── aclnn_prepare_wy_repr_bwd_full.h
│   │   ├── prepare_wy_repr_bwd_full.cpp
│   │   └── prepare_wy_repr_bwd_full.h
│   ├── op_tiling/
│   │   ├── prepare_wy_repr_bwd_full_tiling.cpp
│   │   └── prepare_wy_repr_bwd_full_tiling.h
│   ├── prepare_wy_repr_bwd_full_def.cpp
│   └── CMakeLists.txt
├── op_kernel/
│   ├── prepare_wy_repr_bwd_full_common.h
│   ├── prepare_wy_repr_bwd_full_cube.h
│   ├── prepare_wy_repr_bwd_full_vector.h
│   └── prepare_wy_repr_bwd_full.cpp
└── test/
    ├── test.py
    └── ATK/
        ├── generator_prepare_wy_repr_bwd_full.py
        ├── executor_prepare_wy_repr_bwd_full.py
        ├── all_prepare_wy_repr_bwd_full.json
        └── aclnn_prepare_wy_repr_bwd_full.yaml
```
