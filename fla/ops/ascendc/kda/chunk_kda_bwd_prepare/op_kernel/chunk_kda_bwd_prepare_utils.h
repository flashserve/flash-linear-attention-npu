#ifndef CHUNK_KDA_BWD_PREPARE_UTILS_H
#define CHUNK_KDA_BWD_PREPARE_UTILS_H
#include "kernel_operator.h"
#include "chunk_kda_bwd_prepare_struct.h"
namespace KDA {
// Chunk metadata shared by Cube and Vector; head is decoded separately.
struct ChunkInfo {
    int64_t batchIndex = 0; // Dense: batch index. Varlen: packed storage uses batch 0.
    int64_t seq = 0;
    int64_t localChunk = 0;
    int64_t stateIndex = 0;
    int64_t tokenStart = 0;
    int64_t validRows = 0;
    bool valid = false;
};

__aicore__ inline int64_t KdaMin(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void ResolveChunk(
    int64_t task, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdPrepareTilingData &tiling, ChunkInfo &info)
{
    info.valid = false;
    if (task < 0 || task >= tiling.chunkTaskNum) {
        return;
    }
    if (tiling.isVariable == 0) {
        info.batchIndex = task / tiling.denseChunkNum;
        info.seq = info.batchIndex;
        info.localChunk = task - info.batchIndex * tiling.denseChunkNum;
        info.stateIndex = info.localChunk;
        info.tokenStart = info.localChunk * tiling.chunkSize;
        info.validRows = KdaMin(tiling.chunkSize, tiling.T - info.tokenStart);
        info.valid = info.batchIndex >= 0 && info.batchIndex < tiling.B && info.validRows > 0;
        return;
    }
    if (cuSeqlens == nullptr || chunkIndices == nullptr) {
        return;
    }
    AscendC::GlobalTensor<int64_t> cu;
    AscendC::GlobalTensor<int64_t> indices;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    info.seq = indices.GetValue(2 * task);
    info.localChunk = indices.GetValue(2 * task + 1);
    if (info.seq < 0 || info.seq >= tiling.seqNum || info.localChunk < 0) {
        return;
    }
    const int64_t seqBegin = cu.GetValue(info.seq);
    const int64_t seqEnd = cu.GetValue(info.seq + 1);
    info.batchIndex = 0;
    info.stateIndex = task;
    info.tokenStart = seqBegin + info.localChunk * tiling.chunkSize;
    info.validRows = KdaMin(tiling.chunkSize, seqEnd - info.tokenStart);
    info.valid = seqBegin >= 0 && seqEnd >= seqBegin && seqEnd <= tiling.T && info.validRows > 0;
}

__aicore__ inline int64_t TokenOutputOffset(
    const ChunkKdaBwdPrepareTilingData &tiling, const ChunkInfo &chunk,
    int64_t head, int64_t width)
{
    if (tiling.isVariable != 0) {
        return (head * tiling.T + chunk.tokenStart) * width;
    }
    return ((chunk.batchIndex * tiling.NV + head) * tiling.T + chunk.tokenStart) * width;
}

__aicore__ inline int64_t StateOffset(
    const ChunkKdaBwdPrepareTilingData &tiling, const ChunkInfo &chunk, int64_t head)
{
    if (tiling.isVariable != 0) {
        return (chunk.stateIndex * tiling.NV + head) * tiling.K * tiling.V;
    }
    return ((chunk.batchIndex * tiling.denseChunkNum + chunk.stateIndex) * tiling.NV + head) *
           tiling.K * tiling.V;
}


template <uint32_t LAYOUT>
__aicore__ inline int64_t TokenInputOffset(
    const ChunkKdaBwdPrepareTilingData &tiling, const ChunkInfo &chunk,
    int64_t head, int64_t width)
{
    if constexpr (LAYOUT == 1 || LAYOUT == 3) {
        return ((chunk.batchIndex * tiling.T + chunk.tokenStart) * tiling.NV + head) * width;
    }
    return TokenOutputOffset(tiling, chunk, head, width);
}
template <uint32_t LAYOUT>
__aicore__ inline int64_t TokenInputRowStride(const ChunkKdaBwdPrepareTilingData &tiling, int64_t width)
{
    if constexpr (LAYOUT == 1 || LAYOUT == 3) return tiling.NV * width;
    return width;
}
}
#endif
