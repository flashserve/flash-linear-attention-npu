# aclnnSolveTri

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|      √     |
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|      √     |

## 功能说明

- 接口功能：计算下三角矩阵求逆 $(I + A)^{-1}$，其中 $A$ 是严格下三角矩阵。

- 计算公式：

  SolveTri 算子用于高效计算块对角下三角矩阵的逆矩阵。对于输入矩阵 $A$（严格下三角），输出为 $(I + A)^{-1}$。

  该算子常用于 Gated Delta Rule 线性注意力机制中的 chunk-wise 矩阵求逆操作。

  $$
  Y = (I + A)^{-1}
  $$

  其中 $A$ 是严格下三角矩阵（对角线为0），$I$ 是单位矩阵。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnSolveTriGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnSolveTri` 接口执行计算。

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnSolveTriGetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *cuSeqlens,
    const aclIntArray *chunkIndices,
    const char *layout,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子计算
aclnnStatus aclnnSolveTri(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

### aclnnSolveTriGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| x | 输入 | 公式中的输入矩阵 A，Device 侧的 aclTensor，数据类型支持 FLOAT16、BFLOAT16 |
| cuSeqlens | 输入 | TND/NTD 变长序列的累积长度；`aclIntArray`，INT64，定长模式为 null |
| chunkIndices | 输入 | TND/NTD 的展平 chunk 索引对；`aclIntArray`，INT64，定长模式为 null |
| layout | 输入 | `bsnd`、`bnsd`、`bhtd`、`tnd` 或 `ntd`，null 默认 `bsnd`；`bhtd` 为 `bnsd` 别名 |
| xOut | 输出 | Device 侧连续 aclTensor，shape 和数据类型与 x 一致 |
| workspaceSize | 输出 | 返回执行该算子所需的 workspace 大小 |
| executor | 输出 | 返回算子执行器 |

### aclnnSolveTri

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 执行流 |

## 输入约束

1. 输入 x 支持 FLOAT16 和 BFLOAT16，最后一维 `chunkSize` 支持 16、32、64 和 128。
2. `bsnd` 使用 `[B, T, H, chunkSize]`；`bnsd` / `bhtd` 使用 `[B, H, T, chunkSize]`；`tnd` 使用 `[total_T, H, chunkSize]`；`ntd` 使用 `[H, total_T, chunkSize]`。B、H、T 和 total_T 均须为正。
3. `tnd` / `ntd` 必须同时提供 `cuSeqlens` 和 `chunkIndices`；定长布局不得提供。`cuSeqlens` 从 0 开始、以 total_T 结束且非递减，相邻相等表示空序列。当前不支持全部为空的序列集合。
4. `chunkIndices` 按 `(seq_id, chunk_id)` 对展平，序列编号保留 `cuSeqlens` 的原始下标，chunk 编号从 0 开始。每个非空 chunk 恰好出现一次，顺序不限；末尾不足一个 chunk 时只处理有效区域。
5. 详细契约参见 [api.md](api.md)。A2/A3 的 chunk 128 路径使用 FP32 Vector 中间计算，公开输入和输出仍为 FLOAT16/BFLOAT16。

## 输出说明

| 输出 | 数据类型 | 描述 |
|------|----------|------|
| xOut | FLOAT16/BFLOAT16 | 输出与输入同 shape，为 $(I + A)^{-1}$ 的计算结果 |

## 返回值

| 返回值 | 描述 |
|--------|------|
| ACLNN_SUCCESS | 成功 |
| 其他 | 失败 |

## PyTorch 接口

```python
torch.ops.npu.npu_solve_tri(
    x: Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    layout: str = "bsnd"
) -> Tensor
```

### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| x | Tensor | 输入下三角矩阵，shape 由上述四种布局确定 |
| cu_seqlens | Optional[List[int]] | 变长模式下的累积序列长度，如 `[0, 100, 200]` 表示两个序列长度分别为 100 和 100 |
| chunk_indices | Optional[List[int]] | 变长模式下的 chunk 索引，格式为 `[seq_id_0, chunk_id_0, seq_id_1, chunk_id_1, ...]` |
| layout | str | 数据布局，可选 "bsnd"、"bnsd"、"tnd"、"ntd"，默认 "bsnd" |

### 返回值

| 返回值 | 类型 | 描述 |
|--------|------|------|
| output | Tensor | 输出矩阵，shape 与输入一致 |

### 使用示例

```python
import torch
import torch_npu
import fla_npu
from fla_npu.ops import ascendc

# 基本用法 (BSND layout)
B, T, H, chunkSize = 2, 128, 4, 64
x = torch.zeros(B, T, H, chunkSize, dtype=torch.float16, device="npu")
for start in range(0, T, chunkSize):
    block = torch.tril(
        torch.randn(B, H, chunkSize, chunkSize, dtype=x.dtype, device=x.device),
        diagonal=-1,
    )
    x[:, start:start + chunkSize, :, :] = block.permute(0, 2, 1, 3)
y = ascendc.npu_solve_tri(x, layout="bsnd")


# 变长序列模式 (TND layout)
seq_lens = [100, 128, 80]  # 3 个序列
total_T = sum(seq_lens)
cu_seqlens = [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist())
# 计算 chunk_indices
chunk_indices = []
for seq_idx, seq_len in enumerate(seq_lens):
    num_chunks = (seq_len + chunkSize - 1) // chunkSize
    for chunk_id in range(num_chunks):
        chunk_indices.extend([seq_idx, chunk_id])

x_tnd = torch.zeros(total_T, H, chunkSize, dtype=torch.float16, device="npu")
for seq_idx, seq_len in enumerate(seq_lens):
    for offset in range(0, seq_len, chunkSize):
        valid = min(chunkSize, seq_len - offset)
        start = cu_seqlens[seq_idx] + offset
        block = torch.tril(
            torch.randn(H, valid, valid, dtype=x_tnd.dtype, device=x_tnd.device),
            diagonal=-1,
        )
        x_tnd[start:start + valid, :, :valid] = block.permute(1, 0, 2)
y_tnd = ascendc.npu_solve_tri(
    x_tnd,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    layout="tnd"
)
```
