# aclnnChunkLocalCumsum

## 产品支持情况

| 产品 | 是否支持 |
|:----------------------------|:-----------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 接口功能：对输入张量 `g` 按 `chunk_size` 在时间维上执行 chunk 内局部累加，输出 shape 与输入一致。

- 计算公式：

  对于输入 `g`，输出 `out` 的每个元素由当前 token 所在 chunk 内的局部累加得到。

  正向模式 `reverse=false`：

  $$
  out[b,h,t,...] = scale \times \sum_{k=chunk\_start}^{t} g[b,h,k,...]
  $$

  反向模式 `reverse=true`：

  $$
  out[b,h,t,...] = scale \times \sum_{k=t}^{chunk\_end-1} g[b,h,k,...]
  $$

  其中 `chunk_start = floor(t / chunk_size) * chunk_size`，`chunk_end = min(chunk_start + chunk_size, T)`。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnChunkLocalCumsumGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnChunkLocalCumsum` 接口执行计算。

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnChunkLocalCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOutOptional,
    int64_t chunkSize,
    bool reverse,
    double scale,
    bool headFirst,
    char *outputDtypeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子计算
aclnnStatus aclnnChunkLocalCumsum(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

### aclnnChunkLocalCumsumGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| g | 输入 | Device 侧 aclTensor，数据类型仅支持 FLOAT32，shape 为 `[B, H, T, *]` |
| cuSeqlensOptional | 输入 | 变长序列模式下的累积序列长度，可选参数。数据类型为 INT64 |
| chunkIndicesOutOptional | 输入 | 变长序列模式下 block 到 `(seq_id, block_id)` 的映射，可选参数。数据类型为 INT64 |
| chunkSize | 输入 | chunk 长度，必须为 2 的幂 |
| reverse | 输入 | 是否执行反向 chunk 内累加 |
| scale | 输入 | 输出缩放系数 |
| headFirst | 输入 | 数据布局开关。当前仅支持 `true` |
| outputDtypeOptional | 输入 | 输出 dtype 字符串。当前仅支持 `float32`、`torch.float`、`torch.float32` |
| out | 输出 | Device 侧 aclTensor，数据类型仅支持 FLOAT32，shape 与 `g` 一致 |
| workspaceSize | 输出 | 返回执行该算子所需的 workspace 大小 |
| executor | 输出 | 返回算子执行器 |

### aclnnChunkLocalCumsum

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小 |
| executor | 输入 | 算子执行器 |
| stream | 输入 | 执行流 |

## 输入约束

1. **数据类型**：输入 `g` 和输出 `out` 仅支持 FLOAT32。
2. **输入维度**：`g` 必须为 rank >= 3 的 tensor，shape 为 `[B, H, T, *]`，且各维为正数。
3. **chunkSize**：必须为 2 的幂。
4. **数据布局**：当前仅支持 `[B, H, T, *]`，`headFirst=false` 会返回失败。
5. **输出 dtype**：`outputDtypeOptional` 仅支持 `float32`、`torch.float`、`torch.float32`。
6. **变长模式**：当 `cuSeqlensOptional` 非空时，`B` 必须为 1，`chunkIndicesOutOptional` 必须非空且元素个数为偶数。

## 输出说明

| 输出 | 数据类型 | 描述 |
|------|----------|------|
| out | FLOAT32 | 输出与输入 `g` 同 shape，为 chunk-local cumsum 结果 |

## 返回值

| 返回值 | 描述 |
|--------|------|
| ACLNN_SUCCESS | 成功 |
| 其他 | 失败 |

## 使用示例

固定长度模式下，`cuSeqlensOptional` 和 `chunkIndicesOutOptional` 可使用空 INT64 tensor 作为占位输入：

```cpp
#include "aclnnop/aclnn_chunk_local_cumsum.h"

uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

auto ret = aclnnChunkLocalCumsumGetWorkspaceSize(
    gTensor,
    emptyCuSeqlensTensor,
    emptyChunkIndicesTensor,
    64,
    true,
    1.0,
    false,
    const_cast<char *>("float32"),
    outTensor,
    &workspaceSize,
    &executor);

if (ret == ACLNN_SUCCESS) {
    ret = aclnnChunkLocalCumsum(workspace, workspaceSize, executor, stream);
}
```

PyTorch 测试接口：

```python
out = torch.ops.npu.npu_chunk_local_cumsum(
    g,
    chunk_size,
    cu_seqlens=cu_seqlens,
    chunk_indices_out=chunk_indices_out,
    reverse=False,
    scale=1.0,
    head_first=True,
    output_dtype="float32",
)
```

仓内测试入口：

```bash
bash torch_custom/fla_npu/test/test.sh --device 0 --op chunk_local_cumsum
```

## 实现说明

- Host tiling 根据 `g` 的 `[B, H, T, *]`、`chunkSize`、`cuSeqlensOptional` 和 `chunkIndicesOutOptional` 计算任务数与 `blockDim`。
- 固定长度模式按 `(batch * head, chunk, tail tile)` 划分任务，避免多个 AIV core 写同一段尾部连续维度。
- 变长模式使用 `chunkIndicesOutOptional` 中的 `(seq_id, block_id)` 定位当前序列 chunk，并通过 `cuSeqlensOptional` 获取序列边界。
- Kernel 仅使用 AIV，输出写回 `out`。
