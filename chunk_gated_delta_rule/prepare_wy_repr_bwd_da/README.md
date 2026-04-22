# PrepareWyReprBwdDa 算子说明

`PrepareWyReprBwdDa` 是用于线性注意力（Linear Attention）或相关变体（如 Delta Rule）中 WY 表示（WY Representation）反向传播过程的自定义算子。该算子根据前向输入的 Key、Value、Beta、A 矩阵以及反向传入的 dw、du 梯度和 Gate，计算并输出针对 A 矩阵的梯度 `dA`。

---

## 1. 算子功能

在分块序列模型的反向传播过程中，计算以下张量的梯度：

- **dA**：矩阵 A 的梯度。A 矩阵通常是在分块线性注意力中用于构建线性算子核的中间表示。

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnPrepareWyReprBwdDaGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnPrepareWyReprBwdDa` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnPrepareWyReprBwdDaGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dw,
    const aclTensor *du,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *dAOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnPrepareWyReprBwdDa(
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
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `k` | 输入 | 必选 | Key 输入张量 | 参与反向计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `v` | 输入 | 必选 | Value 输入张量 | 参与反向计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `beta` | 输入 | 必选 | Beta 权重张量 | 参与反向计算；通常为 FP32 | `FLOAT16`、`BFLOAT16`、 `FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `a` | 输入 | 必选 | 前向输出 A 矩阵 | 参与反向计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, BT]` | 支持 |
| `dw` | 输入 | 必选 | Weight 梯度输入 | 对应 w 分支的梯度输入 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `du` | 输入 | 必选 | U 分支梯度输入 | 对应 u 分支的梯度输入 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `g` | 输入 | 必选 | Gate 门控张量 | 参与反向计算；通常为 FP32 | `FLOAT16`、`BFLOAT16`、 `FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列累计长度 | 变长模式输入，形状为 `[N+1]` | `INT64` | `ND` | 1 维 | - |
| `chunkIndicesOptional` | 输入 | 可选 | 分块索引信息 | 扁平化的一维数组 `[num_chunks * 2]` | `INT64` | `ND` | 1 维 | - |

*注：`BT` 维度通常等于 `chunkSize`。*

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `chunkSize` | 输入 | 接口侧必传 | 分块大小 | 对应 A 矩阵的最后一个维度 | `int64_t` | 常用 `64` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `dAOut` | 输出 | A 矩阵梯度输出张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, BT]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器 | `aclOpExecutor*` | - | - | - |

---

## 4. 调用约束与执行语义

### 4.1 形状与模式约束
- `k` 与 `dw` 的形状必须完全一致：`[B, H, T, K]`。
- `v` 与 `du` 的形状必须完全一致：`[B, H, T, V]`。
- `a` 与 `dAOut` 的形状必须完全一致：`[B, H, T, BT]`。
- `beta` 与 `g` 的形状一致：`[B, H, T]`。
- 当 `cuSeqlensOptional` 存在时，开启变长模式，此时 `B` 必须为 `1`。

### 4.2 连续性
- 算子内部会调用 `Contiguous` 接口。若输入 Tensor 已经连续，则可避免额外的内存拷贝开销。

---

## 5. Torch 测试调用示例

该算子可通过 PyTorch 接口直接调用，底层的两阶段接口（workspace + executor）已被封装，无需手动处理。
### 5.1 定长场景
```python
import torch
import torch_npu

device = "npu:0"

# 基本参数
B, H, T, K, V = 1, 4, 2048, 128, 128
chunk_size = 64
BT = chunk_size

# 构造输入
k = torch.randn(B, H, T, K, device=device, dtype=torch.float16)
v = torch.randn(B, H, T, V, device=device, dtype=torch.float16)
beta = torch.randn(B, H, T, device=device, dtype=torch.float32)
a = torch.randn(B, H, T, BT, device=device, dtype=torch.float16)
dw = torch.randn(B, H, T, K, device=device, dtype=torch.float16)
du = torch.randn(B, H, T, V, device=device, dtype=torch.float16)
# 构造满足约束的 g：负数且沿 T 单调递减
base = torch.rand(B, H, T, device=device, dtype=torch.float32) * 0.1 + 0.01
g = -torch.cumsum(base, dim=-1)

# 调用算子
dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
    k, v, beta, a, dw, du, g,
    chunk_size=chunk_size,
    cu_seqlens=None,
    chunk_indices=None
)

print(dA.shape) # Expected: [1, 4, 2048, 64]
```

### 5.2 变长场景
```python
import torch
import torch_npu

device = "npu:0"

# B 必须为 1
B, H, K, V = 1, 4, 128, 128
chunk_size = 64
BT = chunk_size

# 变长序列: [64, 128] -> 总长 192
cu_seqlens = torch.tensor([0, 64, 192], device=device, dtype=torch.int64)
total_len = cu_seqlens[-1].item()

# 构造 chunk_indices (flattened: [seq_id, chunk_id, ...])
# seq 0 (len 64): 1 chunk -> [0, 0]
# seq 1 (len 128): 2 chunks -> [1, 0, 1, 1]
chunk_indices = torch.tensor([0, 0, 1, 0, 1, 1], device=device, dtype=torch.int64)

k = torch.randn(B, H, total_len, K, device=device, dtype=torch.float16)
v = torch.randn(B, H, total_len, V, device=device, dtype=torch.float16)
beta = torch.randn(B, H, total_len, device=device, dtype=torch.float32)
a = torch.randn(B, H, total_len, BT, device=device, dtype=torch.float16)
dw = torch.randn(B, H, total_len, K, device=device, dtype=torch.float16)
du = torch.randn(B, H, total_len, V, device=device, dtype=torch.float16)
# 构造满足约束的 g：负数且沿 T 单调递减
base = torch.rand(B, H, T, device=device, dtype=torch.float32) * 0.1 + 0.01
g = -torch.cumsum(base, dim=-1)

dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
    k, v, beta, a, dw, du, g,
    chunk_size=chunk_size,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices
)

print(dA.shape)
```

---

## 6. 目录结构

```text
prepare_wy_repr_bwd_da/
├── examples/
│   └── test_aclnn_prepare_wy_repr_bwd_da.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_prepare_wy_repr_bwd_da.cpp
│   │   ├── aclnn_prepare_wy_repr_bwd_da.h
│   │   ├── prepare_wy_repr_bwd_da.cpp
│   │   └── prepare_wy_repr_bwd_da.h
│   ├── op_tiling/
│   │   ├── prepare_wy_repr_bwd_da_tiling.cpp
│   │   └── prepare_wy_repr_bwd_da_tiling.h
│   ├── CMakeLists.txt
│   └── prepare_wy_repr_bwd_da_def.cpp
├── op_kernel/
│   ├── prepare_wy_repr_bwd_da_common.h
│   ├── prepare_wy_repr_bwd_da_cube.h
│   ├── prepare_wy_repr_bwd_da_vector.h
│   └── prepare_wy_repr_bwd_da.cpp
├── test/
│   ├── test_da_all.py
│   └── test_da.py
├── CMakeLists.txt
├── README.md
└── run.sh
```