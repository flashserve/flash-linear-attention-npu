# ChunkBwdDqkwg 算子说明

`ChunkBwdDqkwg` 是一个用于分块门控 delta 规则（Chunk Gated Delta Rule）反向传播过程中的自定义算子。该算子根据前向激活值、权重及梯度输入，计算并输出针对 Query（q）、Key（k）、Weight（w）和 Gate（g）的梯度。

---

## 1. 算子功能

在分块序列模型中，计算以下张量的梯度：

- **dq**：Query 的梯度  
- **dk**：Key 的梯度  
- **dw**：Weight（衰减/权重参数）的梯度  
- **dg**：Gate（门控）的梯度  

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnChunkBwdDqkwgGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnChunkBwdDqkwg` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnChunkBwdDqkwgGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *h,
    const aclTensor *dox,
    const aclTensor *dh,
    const aclTensor *dv,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const aclTensor *w,
    const aclTensor *gGamma,
    float scale,
    int64_t chunkSize,
    bool use_exp2,
    bool transpose_state_layout,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnChunkBwdDqkwg(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream
);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

| 参数名                 | 输入/输出 | 必选/可选                    | 描述                    | 使用说明                                                                 | 数据类型                       | 数据格式 | 维度（Shape）             | 非连续 Tensor |
| ---------------------- | --------- | ---------------------------- | ----------------------- | ------------------------------------------------------------------------ | ------------------------------ | -------- | ------------------------- | ------------- |
| `q`                    | 输入      | 必选                         | Query 输入张量          | 参与反向计算；接口执行前会先转为连续内存                                 | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, K]`            | 支持          |
| `k`                    | 输入      | 必选                         | Key 输入张量            | 参与反向计算；接口执行前会先转为连续内存                                 | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, K]`            | 支持          |
| `v`                    | 输入      | 必选                         | Value 输入张量          | 参与反向计算；接口执行前会先转为连续内存                                 | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, V]`            | 支持          |
| `g`                    | 输入      | 必选                         | Gate 输入张量           | 要求沿序列维 `T` 为非正且单调递减（`g[..., t] <= g[..., t-1]`）；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND`     | `[B, H, T]`               | 支持          |
| `h`                    | 输入      | 必选                         | 前向保存的隐藏状态张量  | 用于反向传播；接口执行前会先转为连续内存                                 | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, numChunks, K, V]` | 支持          |
| `dox`                  | 输入      | 必选                         | 前向输出 `o` 的梯度张量 | 即输出梯度；接口执行前会先转为连续内存                                   | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, V]`            | 支持          |
| `dh`                   | 输入      | 必选                         | 隐藏状态梯度张量        | 与 `h` 对应；接口执行前会先转为连续内存                                  | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, numChunks, K, V]` | 支持          |
| `dv`                   | 输入      | 必选                         | Value 分支梯度张量      | 参与 delta rule 反向计算（非输出）；接口执行前会先转为连续内存            | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, V]`            | 支持          |
| `cuSeqlensOptional`    | 输入      | 可选                         | 变长序列的累计长度信息  | 变长模式输入，形状为 `[N+1]`                                             | `INT64`                        | `ND`     | 1 维                      | -             |
| `chunkIndicesOptional` | 输入      | 可选                         | 分块索引信息            | 逻辑上表示为 `[num_chunks, 2]`，实际需按一维数组 `[num_chunks * 2]` 传入（flatten） | `INT64`                        | `ND`     | 1 维                      | -             |
| `w`                    | 输入      | 接口为必传；当前实现要求传空 | 预留权重输入            | 当前实现未启用，必须传 `nullptr`                                         | `FLOAT16`、`BFLOAT16`          | `ND`     | 未启用                    | -             |
| `gGamma`               | 输入      | 接口为必传；当前实现要求传空 | 预留门控参数输入        | 当前实现未启用，必须传 `nullptr`                                         | `FLOAT16`、`BFLOAT16`          | `ND`     | 未启用                    | -             |

### 3.2 属性参数（Attributes）

| 参数名                   | 输入/输出 | 必选/可选            | 描述                   | 使用说明                                          | 数据类型  | 取值约束      |
| ------------------------ | --------- | -------------------- | ---------------------- | ------------------------------------------------- | --------- | ------------- |
| `scale`                  | 输入      | 可选属性，接口侧必传 | 缩放系数               | 建议设置为 `1 / sqrt(K)`                          | `float`   | 建议大于 0    |
| `chunkSize`              | 输入      | 可选属性，接口侧必传 | 分块大小               | 当前仅支持 `64` 或 `128`                          | `int64_t` | `64` / `128` |
| `use_exp2`               | 输入      | 可选                 | 是否使用 exp2 近似实现 | `true` 使用 exp2，`false` 使用标准 exp            | `bool`    | `true` / `false` `必须是false` |
| `transpose_state_layout` | 输入      | 可选                 | 是否转置内部状态布局   | 影响 `h/dh` 对应的内部 state layout 与 kernel 访存 | `bool`    | `true` / `false` `必须是false` |

### 3.3 输出参数（Outputs）

| 参数名          | 输入/输出 | 描述                         | 数据类型                       | 数据格式 | 维度（Shape）  | 非连续 Tensor |
| --------------- | --------- | ---------------------------- | ------------------------------ | -------- | -------------- | ------------- |
| `dqOut`         | 输出      | Query 梯度输出张量           | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, K]` | 支持          |
| `dkOut`         | 输出      | Key 梯度输出张量             | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, K]` | 支持          |
| `dwOut`         | 输出      | 权重梯度输出张量             | `FLOAT16`、`BFLOAT16`          | `ND`     | `[B, H, T, K]` | 支持          |
| `dgOut`         | 输出      | Gate 梯度输出张量            | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND`     | `[B, H, T]`    | 支持          |
| `workspaceSize` | 输出      | Device 侧所需 workspace 大小 | `uint64_t`                     | -        | 标量           | -             |
| `executor`      | 输出      | 算子执行器                   | `aclOpExecutor*`               | -        | -              | -             |      |

### 3.4 形状与约束

- `q`、`k` 的形状必须为 `[B, H, T, K]`。  
- `v`、`dox`、`dv` 的形状必须为 `[B, H, T, V]`。  
- `g` 的形状必须为 `[B, H, T]`。  
- `h`、`dh` 的形状必须为 `[B, H, numChunks, K, V]`。  
- 当前实现要求 `K = 128`。  
- 当前实现要求 `V = 128` 或 `256`。  
- `chunkSize` 当前仅支持 `64` 或 `128`。  
- 当启用变长模式时，`cuSeqlensOptional` 和 `chunkIndicesOptional` 用于描述变长分块；同时当前实现仅支持 `B = 1`。  
- 当前实现要求 `w` 和 `gGamma` 传空指针，否则 tiling 阶段会直接报错。  

---

## 4. 调用约束与执行语义

### 4.1 可选参数约束

- `cuSeqlensOptional` 和 `chunkIndicesOptional`：
  - 同时出现时启用变长模式（varlen）
  - 变长模式仅支持 `B = 1`

- `w` 和 `gGamma`：
  - 接口层存在，但当前实现未启用
  - **必须传入空指针，否则执行失败**

---

### 4.2 形状约束（强约束）

必须满足以下条件：

- `q, k`: `[B, H, T, K]`
- `v, dox, dv`: `[B, H, T, V]`
- `g`: `[B, H, T]`
- `h, dh`: `[B, H, numChunks, K, V]`

额外限制：

- `K = 128`
- `V ∈ {128, 256}`
- `chunkSize ∈ {64, 128}`

---

### 4.3 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时：

- `chunkIndicesOptional` 必须同时提供
- 当前实现仅支持：B = 1


**chunkIndicesOptional 示例：**

```text
cuSeqlensOptional = [0, 4, 10]
chunkSize = 4

序列划分：
- 第 0 个序列长度 = 4 → 需要 1 个 chunk
- 第 1 个序列长度 = 6 → 需要 2 个 chunk

chunkIndicesOptional（二维表示）：
[
  [0, 0],   # 第 0 个序列，第 0 个 chunk
  [1, 0],   # 第 1 个序列，第 0 个 chunk
  [1, 1]    # 第 1 个序列，第 1 个 chunk
]

扁平化表示（实际接口传入）：
[0, 0, 1, 0, 1, 1]
```

---

### 4.4 数值语义

- `scale`：
  - 必须显式传入
  - 推荐设置为：$1 / \sqrt{K}$


- 当前算子实现配置为：

```text
USE_G = True
USE_DW = True
USE_G_GAMMA = False
```

即：

- 启用 Gate
- 计算 `dw`
- 不使用 `gGamma`

---

## 5. Torch 测试调用示例

该算子可通过 PyTorch 接口直接调用，底层的两阶段接口（workspace + executor）已被封装，无需手动处理。
### 5.1 定长场景
```python
import torch
import torch_npu

# 设备
device = "npu:0"

# 基本参数
B, H, T, K, V = 1, 8, 1024, 128, 128
chunk_size = 64
scale = 1.0 / (K ** 0.5)

# 构造输入（注意 shape）
q = torch.randn(B, H, T, K, device=device, dtype=torch.float16)
k = torch.randn(B, H, T, K, device=device, dtype=torch.float16)
v = torch.randn(B, H, T, V, device=device, dtype=torch.float16)

# 构造满足约束的 g：负数且沿 T 单调递减
base = torch.rand(B, H, T, device=device, dtype=torch.float32) * 0.1 + 0.01
g = -torch.cumsum(base, dim=-1)

dox = torch.randn(B, H, T, V, device=device, dtype=torch.float16)
dv = torch.randn(B, H, T, V, device=device, dtype=torch.float16)

# num_chunks = T // chunk_size（简单场景）
num_chunks = T // chunk_size
h = torch.randn(B, H, num_chunks, K, V, device=device, dtype=torch.float16)
dh = torch.randn(B, H, num_chunks, K, V, device=device, dtype=torch.float16)

# 调用算子
dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
    q, k, v, g, h, dox, dh, dv,
    chunk_size,
    cu_seqlens=None,
    chunk_indices=None,
    scale=scale,
    w=None,
    g_gamma=None,
    transpose_state_layout=None
)

print(dq.shape, dk.shape, dw.shape, dg.shape)
```
### 5.2 变长场景

```python
import torch
import torch_npu

device = "npu:0"

B, H, K, V = 1, 8, 128, 128
chunk_size = 64
scale = 1.0 / (K ** 0.5)

# 3 条变长序列
seq_lens = [70, 130, 60]
cu_seqlens = torch.tensor([0, 70, 200, 260], device=device, dtype=torch.int64)
total_len = cu_seqlens[-1].item()

# 构造 chunk_indices（flatten 形式）
chunk_indices_list = []
for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
    for pos in range(s, e, chunk_size):
        chunk_indices_list.extend([pos, min(pos + chunk_size, e)])

chunk_indices = torch.tensor(chunk_indices_list, device=device, dtype=torch.int64)
num_chunks = len(chunk_indices_list) // 2

q = torch.randn(B, H, total_len, K, device=device, dtype=torch.float16)
k = torch.randn(B, H, total_len, K, device=device, dtype=torch.float16)
v = torch.randn(B, H, total_len, V, device=device, dtype=torch.float16)

# g: 非正且单调递减
base = torch.rand(B, H, total_len, device=device, dtype=torch.float32) * 0.1 + 0.01
g = -torch.cumsum(base, dim=-1)

dox = torch.randn(B, H, total_len, V, device=device, dtype=torch.float16)
dv = torch.randn(B, H, total_len, V, device=device, dtype=torch.float16)

h = torch.randn(B, H, num_chunks, K, V, device=device, dtype=torch.float16)
dh = torch.randn(B, H, num_chunks, K, V, device=device, dtype=torch.float16)

dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
    q, k, v, g, h, dox, dh, dv,
    chunk_size,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    scale=scale,
    w=None,
    g_gamma=None,
    use_exp2=False,
    transpose_state_layout=False
)

print(dq.shape, dk.shape, dw.shape, dg.shape)
```
### 说明

- 输入 shape：
  - `q/k`: `[B, H, T, K]`
  - `v/dox/dv`: `[B, H, T, V]`
  - `g`: `[B, H, T]` `g需要为负数且单调递减`
  - `h/dh`: `[B, H, num_chunks, K, V]`
- `chunk_size` 当前仅支持 `64` 或 `128`
- `scale` 通常为 `1 / sqrt(K)`
- `w`、`g_gamma` 当前版本需传 `None`

---

## 6. 目录结构

```text
chunk_bwd_dqkwg/
├── examples/
│   └── test_chunk_bwd_dqkwg.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_chunk_bwd_dqkwg.cpp
│   │   └── aclnn_chunk_bwd_dqkwg.h
│   ├── op_tiling/
│   │   ├── chunk_bwd_dqkwg_tiling.cpp
│   │   └── chunk_bwd_dqkwg_tiling.h
│   ├── chunk_bwd_dqkwg_def.cpp
│   └── CMakeLists.txt
└── op_kernel/
    ├── chunk_bwd_dqkwg_common.h
    ├── chunk_bwd_dqkwg_cube.h
    ├── chunk_bwd_dqkwg_vector.h
    └── chunk_bwd_dqkwg.cpp
```