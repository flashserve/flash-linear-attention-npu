# aclnnChunkCumsumKkt

## 功能

`aclnnChunkCumsumKkt` 是 A2 GDN Phase 3 的独立局部融合入口。它在每个序列、每个 chunk 内对
raw FP32 gate 做 forward cumulative sum，并直接用该结果计算未求解的严格下三角 KKT 矩阵。

该入口不包含 `solve_tri`，不修改 Phase 1/2 GDN core，也不实现原生 GVA。它保留为共享 helper
的独立正确性/局部微基准入口；Phase 3 最终 core 不调用它，而调用累积融合
`ChunkCumsumKktSolveTri`。

## 接口

```cpp
aclnnStatus aclnnChunkCumsumKktGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *gCumsumOut,
    const aclTensor *aOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkCumsumKkt(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 输入输出

| 名称 | 类型 | Shape | 说明 |
| --- | --- | --- | --- |
| `k` | FP16/BF16 | `[B,H,T,128]` | head-first key |
| `g` | FP32 | `[B,H,T]` | raw gate，不是 cumulative gate |
| `beta` | FP32 | `[B,H,T]` | KKT row scale |
| `cuSeqlensOptional` | INT64 host array | `[numSeq+1]` | varlen 时提供，dense 时为空 |
| `chunkIndicesOptional` | INT64 host array | flatten `[seq,localChunk]` | 与 `cuSeqlensOptional` 同时提供 |
| `chunkSize` | INT64 | scalar | 仅支持 `64/128` |
| `gCumsumOut` | FP32 | `[B,H,T]` | chunk-local forward cumulative sum |
| `aOut` | FP32 | `[B,H,T,chunkSize]` | 未求解 KKT；对角、上三角与尾部 padding 为零 |

varlen 模式要求物理 `B=1`，`cuSeqlens` 从 `0` 开始并以 `T` 结束，`chunkIndices` 使用 canonical
sequence-major 顺序。首版要求 `k/g/beta` 的物理 head 数完全相同。

## Python

推荐稳定入口：

```python
from fla_npu.ops.ascendc import chunk_cumsum_kkt

g_cumsum, A_raw = chunk_cumsum_kkt(
    k,
    raw_g,
    beta,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
)
```
