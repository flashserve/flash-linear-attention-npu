# ChunkLocalCumsum 接口对齐说明

## 对齐目标

对齐 Triton 函数:

```python
chunk_local_cumsum(
    g,
    chunk_size,
    reverse=False,
    scale=None,
    cu_seqlens=None,
    chunk_indices_out=None,
    head_first=False,
    output_dtype=torch.float,
    **kwargs,
)
```

参考调用:

```python
g = chunk_local_cumsum(
    g,
    chunk_size=chunk_size,
    cu_seqlens=cu_seqlens,
    chunk_indices_out=chunk_indices,
    head_first=False,
)
```

## AscendC 当前 ACLNN 入参

重新生成后的 ACLNN C 接口为:

```cpp
aclnnChunkLocalCumsumGetWorkspaceSize(
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
```

## 参数对应关系

| Triton 参数 | AscendC/ACLNN 参数 | 对齐状态 | 说明 |
| --- | --- | --- | --- |
| `g` | `g` | 一致 | `float32 [B,T,H]` |
| `chunk_size` | `chunkSize` | 一致 | required attr |
| `reverse` | `reverse` | 一致 | 默认 `false` |
| `scale` | `scale` | 语义一致 | Triton `None` 等价于 AscendC 默认 `1.0` |
| `cu_seqlens` | `cuSeqlensOptional` | 一致 | fixed 模式传空 tensor |
| `chunk_indices_out` | `chunkIndicesOutOptional` | 语义一致 | AscendC 接收 `chunk_indices_out[str(block_t)]` 解析后的 tensor |
| `head_first` | `headFirst` | 一致 | 当前 scalar 路径按 `[B,T,H]` 处理 |
| `output_dtype` | `outputDtypeOptional` | 受限一致 | 当前仅支持 `float32`、`torch.float`、`torch.float32` |
| `**kwargs` | 无 | 不适用 | Triton 预留参数，当前实现未使用 |

## 重要说明

- Python dict 不能作为 AscendC kernel 输入，因此 `chunk_indices_out` 在 ACLNN 层是已经根据 `block_t` 取出的 tensor，不是 dict 本身。
- 已删除旧的 AscendC 专用 `has_scale` 属性，避免接口多出 Triton 没有的参数。
- `output_dtype` 已加入 schema，但当前 kernel 只实现 float32 输出；传其他 dtype 会在 tiling 阶段报错。

## 验证结果

- 编译命令: `bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16`
- 精度命令: `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/examples/run_precision_test.sh`
- 结果: 30/30 case 通过
- 最大 `max_abs`: 0.000001907
- 所有 case `bad_count=0`
