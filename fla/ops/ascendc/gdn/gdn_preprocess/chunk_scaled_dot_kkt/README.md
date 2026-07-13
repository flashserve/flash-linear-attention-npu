# ChunkScaledDotKkt 算子说明

`ChunkScaledDotKkt` 用于 Gated Delta Rule 线性注意力中的 chunk-wise WY 表示构建。当前实现覆盖 `gk is None` 的定长序列场景，输入和输出均采用 head-first 排布。

## 1. 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 2. 算子功能

对每个 batch、head 和 chunk，计算带 gate 衰减和 beta 缩放的严格下三角 `K @ K^T`：

$$
A_{b,h,t,c} =
\begin{cases}
\beta_{b,h,t} \cdot \exp(\mathrm{clip}(g_{b,h,t} - g_{b,h,s}, -50, 50)) \cdot
\sum_d k_{b,h,t,d} k_{b,h,s,d}, & c < r \\
0, & c \ge r
\end{cases}
$$

其中 `BT=chunk_size`，`r=t mod BT`，`s=t-r+c`。输出只保留每个 chunk 内的严格下三角列，最后一维长度固定为 `BT`。

## 3. 接口定义

### 3.1 ACLNN 接口

```cpp
aclnnStatus aclnnChunkScaledDotKktGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    int64_t chunkSize,
    aclTensor *A,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkScaledDotKkt(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### 3.2 PyTorch 接口

```python
torch.ops.npu.npu_chunk_scaled_dot_kkt(
    k: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
) -> Tensor
```

## 4. 输入参数

| 参数 | 数据类型 | Shape | 是否必须 | 描述 |
| -- | -- | -- | -- | -- |
| k | FLOAT16/BF16 | `[B,H,T,K]` | 是 | key 张量，head-first 排布 |
| g | FLOAT | `[B,H,T]` | 是 | chunk 内 cumulative gate，head-first 排布 |
| beta | FLOAT | `[B,H,T]` | 是 | 每个 token/head 的缩放系数，head-first 排布 |
| chunk_size | INT | - | 否 | chunk 大小，默认 64 |

## 5. 输出参数

| 输出 | 数据类型 | Shape | 描述 |
| -- | -- | -- | -- |
| A | FLOAT | `[B,H,T,chunk_size]` | WY 表示中的严格下三角块 |

## 6. 输入约束

1. `k` 必须为 4D，shape 为 `[B,H,T,K]`。
2. `g` 和 `beta` 必须为 3D，shape 为 `[B,H,T]`。
3. `chunk_size` 仅支持 `16`、`32`、`64`、`128`。
4. 当前版本仅支持定长 dense 序列，不支持 `cu_seqlens`、`chunk_indices` 或 `gk` 分支。
5. 输入建议传入 contiguous Tensor；PyTorch wrapper 会在调用 ACLNN 前执行 contiguous。

## 7. 目录结构

```text
chunk_scaled_dot_kkt/
├── docs/
│   └── aclnnChunkScaledDotKkt.md
├── op_host/
│   ├── chunk_scaled_dot_kkt_def.cpp
│   ├── chunk_scaled_dot_kkt_infershape.cpp
│   ├── chunk_scaled_dot_kkt_tiling.cpp
│   ├── chunk_scaled_dot_kkt_tiling.h
│   └── CMakeLists.txt
├── op_kernel/
│   ├── chunk_scaled_dot_kkt.cpp
│   ├── chunk_scaled_dot_kkt.h
│   └── chunk_scaled_dot_kkt_tiling_key.h
├── test/
│   └── test.py
├── CMakeLists.txt
└── README.md
```

## 8. 验证说明

算子目录测试入口：

```bash
python3 fla/ops/ascendc/gdn/gdn_preprocess/chunk_scaled_dot_kkt/test/test.py
```

torch 插件侧测试入口：

```bash
cd torch_custom/fla_npu/test
python3 test_npu_chunk_scaled_dot_kkt.py
```
