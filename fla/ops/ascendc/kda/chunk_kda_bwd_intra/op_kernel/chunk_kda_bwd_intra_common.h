#ifndef CHUNK_KDA_BWD_INTRA_COMMON_H
#define CHUNK_KDA_BWD_INTRA_COMMON_H

#include "kernel_operator.h"

namespace KDA {

constexpr uint32_t kRowBlock = 16;
constexpr uint32_t kChunkSize = 64;
constexpr uint32_t kHeadsPerWindow = 2;
constexpr uint32_t kWorkspaceSlots = 4;
constexpr uint32_t kVecToCubeReadyFlag = 2;
constexpr uint32_t kCubeToVecReadyFlag = 4;
constexpr float kLn2 = 0.69314718055994530942f;

__aicore__ inline uint32_t WorkspaceSlot(uint64_t windowIdx, uint32_t headInWindow)
{
    return static_cast<uint32_t>(((windowIdx & 1U) << 1U) + headInWindow);
}

struct ChunkTask {
    uint32_t batchIdx;
    uint32_t chunkIdx;
    uint32_t begin;
    uint32_t end;
};

template <bool VARLEN_TND>
__aicore__ inline ChunkTask GetChunkTask(
    const ChunkKdaBwdIntraTilingData &tiling,
    AscendC::GlobalTensor<int64_t> &chunkMetadata, uint32_t taskIdx)
{
    ChunkTask task{};
    if constexpr (VARLEN_TND) {
        const uint64_t base = static_cast<uint64_t>(taskIdx) * 4;
        // The hot path only needs global begin/end.  seq/localChunk remain in
        // packed metadata for validation/debugging but are not loaded here.
        task.batchIdx = 0;
        task.begin = static_cast<uint32_t>(chunkMetadata.GetValue(base + 1));
        task.end = static_cast<uint32_t>(chunkMetadata.GetValue(base + 2));
        task.chunkIdx = taskIdx;
    } else {
        task.batchIdx = taskIdx / static_cast<uint32_t>(tiling.chunkNumPerBatch);
        task.chunkIdx = taskIdx % static_cast<uint32_t>(tiling.chunkNumPerBatch);
        task.begin = task.chunkIdx * static_cast<uint32_t>(tiling.chunkSize);
        task.end = task.begin + static_cast<uint32_t>(tiling.chunkSize);
        const uint32_t t = static_cast<uint32_t>(tiling.seqlen);
        if (task.end > t) {
            task.end = t;
        }
    }
    return task;
}

template <bool VARLEN_TND>
__aicore__ inline uint64_t TensorOffset(
    const ChunkKdaBwdIntraTilingData &tiling, uint32_t batchIdx, uint32_t headIdx,
    uint32_t tokenIdx, uint32_t col = 0)
{
    if constexpr (VARLEN_TND) {
        return ((static_cast<uint64_t>(tokenIdx) * tiling.headNum + headIdx) *
                tiling.headDim) + col;
    } else {
        return (((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
                 tiling.seqlen + tokenIdx) * tiling.headDim) + col;
    }
}

template <bool VARLEN_TND>
__aicore__ inline uint64_t MatrixOffset(
    const ChunkKdaBwdIntraTilingData &tiling, uint32_t batchIdx, uint32_t headIdx,
    uint32_t tokenIdx, uint32_t col = 0)
{
    if constexpr (VARLEN_TND) {
        return ((static_cast<uint64_t>(tokenIdx) * tiling.headNum + headIdx) *
                tiling.chunkSize) + col;
    } else {
        return (((static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
                 tiling.seqlen + tokenIdx) * tiling.chunkSize) + col;
    }
}

template <bool VARLEN_TND>
__aicore__ inline uint64_t ScalarOffset(
    const ChunkKdaBwdIntraTilingData &tiling, uint32_t batchIdx, uint32_t headIdx, uint32_t tokenIdx)
{
    if constexpr (VARLEN_TND) {
        return static_cast<uint64_t>(tokenIdx) * tiling.headNum + headIdx;
    } else {
        return (static_cast<uint64_t>(batchIdx) * tiling.headNum + headIdx) *
               tiling.seqlen + tokenIdx;
    }
}

} // namespace KDA

#endif // CHUNK_KDA_BWD_INTRA_COMMON_H
