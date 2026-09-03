# aclnnChunkScaledDotKkt

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 接口功能：计算 Gated Delta Rule 中 `gk is None` 分支的 chunk-wise scaled dot product，输出严格下三角 WY 表示。
- 计算公式：

  $$
  A_{b,h,t,c} =
  \beta_{b,h,t}
  \exp(\mathrm{clip}(g_{b,h,t} - g_{b,h,s}, -50, 50))
  \sum_d k_{b,h,t,d} k_{b,h,s,d}
  $$

  其中 `BT=chunkSize`，`r=t mod BT`，`s=t-r+c`。当 `c >= r` 时输出为 0。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnChunkScaledDotKktGetWorkspaceSize` 接口获取计算所需 workspace 大小以及执行器，再调用 `aclnnChunkScaledDotKkt` 接口执行计算。

```cpp
aclnnStatus aclnnChunkScaledDotKktGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    int64_t chunkSize,
    aclTensor *A,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnChunkScaledDotKkt(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

### aclnnChunkScaledDotKktGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 |
| -- | -- | -- |
| k | 输入 | key 张量，Device 侧 aclTensor，shape 为 `[B,H,T,K]`，数据类型为 FLOAT16 或 BF16 |
| g | 输入 | cumulative gate 张量，Device 侧 aclTensor，shape 为 `[B,H,T]`，数据类型为 FLOAT |
| beta | 输入 | beta 缩放张量，Device 侧 aclTensor，shape 为 `[B,H,T]`，数据类型为 FLOAT |
| chunkSize | 输入 | chunk 大小，仅支持 `16`、`32`、`64`、`128` |
| A | 输出 | 输出张量，Device 侧 aclTensor，shape 为 `[B,H,T,chunkSize]`，数据类型为 FLOAT |
| workspaceSize | 输出 | 返回执行该算子所需的 workspace 大小 |
| executor | 输出 | 返回算子执行器 |

### aclnnChunkScaledDotKkt

| 参数名 | 输入/输出 | 描述 |
| -- | -- | -- |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 执行流 |

## 输入约束

1. `k` 支持 FLOAT16 和 BF16。
2. `g` 和 `beta` 仅支持 FLOAT。
3. `k` shape 为 `[B,H,T,K]`，`g`/`beta` shape 为 `[B,H,T]`。
4. `chunkSize` 仅支持 `16`、`32`、`64`、`128`。
5. 当前版本仅支持定长 dense 序列，不支持变长序列参数，也不支持 `gk` 分支。
6. `aclnnChunkScaledDotKkt` 默认确定性实现。

## 输出说明

| 输出 | 数据类型 | Shape | 描述 |
| -- | -- | -- | -- |
| A | FLOAT | `[B,H,T,chunkSize]` | 每个 chunk 内的严格下三角 scaled dot product |

## 返回值

| 返回值 | 描述 |
| -- | -- |
| ACLNN_SUCCESS | 成功 |
| 其他 | 失败 |

## PyTorch 接口

```python
torch.ops.npu.npu_chunk_scaled_dot_kkt(
    k: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
) -> Tensor
```

### 参数说明

| 参数 | 类型 | 描述 |
| -- | -- | -- |
| k | Tensor | shape 为 `[B,H,T,K]`，dtype 为 `torch.float16` 或 `torch.bfloat16` |
| g | Tensor | shape 为 `[B,H,T]`，dtype 为 `torch.float32` |
| beta | Tensor | shape 为 `[B,H,T]`，dtype 为 `torch.float32` |
| chunk_size | int | chunk 大小，默认 64 |

### 返回值

| 返回值 | 类型 | 描述 |
| -- | -- | -- |
| A | Tensor | shape 为 `[B,H,T,chunk_size]`，dtype 为 `torch.float32` |

### 使用示例

```python
import torch
import torch_npu
import fla_npu

B, H, T, K, BT = 2, 4, 128, 64, 64
k = torch.randn(B, H, T, K, dtype=torch.float16, device="npu")
g = torch.randn(B, H, T, dtype=torch.float32, device="npu")
beta = torch.rand(B, H, T, dtype=torch.float32, device="npu")

A = torch.ops.npu.npu_chunk_scaled_dot_kkt(k, g, beta, chunk_size=BT)
assert A.shape == (B, H, T, BT)
```
