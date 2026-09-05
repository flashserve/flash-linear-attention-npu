/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH35_COMMON_H
#define CHUNK_KDA_BWD_PREPARE_ARCH35_COMMON_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/cross_core_sync.hpp"
#include "../chunk_kda_bwd_prepare_struct.h"
#include "kernel_operator.h"

namespace KDA {

// The A5 direct L0C-to-UB path requires a tightly coupled 1C2V block.  Follow
// the mature DHU convention: each AIV publishes local FREE ids 0/1, while the
// AIC observes AIV1 through the hardware sub-block flag stride.
constexpr uint8_t KDA_PREPARE_CROSS_CORE_MODE = 0x4;
constexpr uint64_t KDA_PREPARE_FREE_FLAG_BASE = 0;
constexpr uint64_t KDA_PREPARE_READY_FLAG_BASE = 6;
constexpr uint64_t KDA_PREPARE_Q_FREE_FLAG = 2;
constexpr uint64_t KDA_PREPARE_D_FREE_FLAG = 3;
constexpr uint64_t KDA_PREPARE_Q_READY_FLAG = 8;
constexpr uint64_t KDA_PREPARE_D_READY_FLAG = 9;
constexpr uint64_t KDA_PREPARE_SUBBLOCK_FLAG_STRIDE = 16;
constexpr uint32_t KDA_PREPARE_RAW_SLOT_COUNT = 2;
constexpr uint32_t KDA_PREPARE_CHUNK = 64;
constexpr uint32_t KDA_PREPARE_DIM = 128;
constexpr uint32_t KDA_PREPARE_RAW_BF16_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_CHUNK * sizeof(bfloat16_t);
constexpr uint32_t KDA_PREPARE_FP32_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_CHUNK * sizeof(float);
constexpr uint32_t KDA_PREPARE_A_UB_BYTES =
    2 * KDA_PREPARE_RAW_BF16_BYTES + 3 * KDA_PREPARE_FP32_BYTES;
constexpr uint32_t KDA_PREPARE_Q_FP32_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_DIM * sizeof(float);
constexpr uint32_t KDA_PREPARE_D_BF16_BYTES =
    KDA_PREPARE_CHUNK * KDA_PREPARE_DIM * sizeof(bfloat16_t);
constexpr uint32_t KDA_PREPARE_Q_UB_OFFSET = KDA_PREPARE_A_UB_BYTES;
constexpr uint32_t KDA_PREPARE_D_UB_OFFSET =
    KDA_PREPARE_Q_UB_OFFSET + KDA_PREPARE_Q_FP32_BYTES;
constexpr uint32_t KDA_PREPARE_TOTAL_UB_BYTES =
    KDA_PREPARE_D_UB_OFFSET + KDA_PREPARE_D_BF16_BYTES;
constexpr uint32_t KDA_PREPARE_A5_UB_BUDGET_BYTES = 192 * 1024;
static_assert(KDA_PREPARE_TOTAL_UB_BYTES <= KDA_PREPARE_A5_UB_BUDGET_BYTES,
              "KernelA AIV buffers exceed the A5 192 KiB UB budget.");

struct ChunkInfo {
    int64_t b = 0;
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
        info.b = task / tiling.denseChunkNum;
        info.seq = info.b;
        info.localChunk = task - info.b * tiling.denseChunkNum;
        info.stateIndex = info.localChunk;
        info.tokenStart = info.localChunk * tiling.chunkSize;
        info.validRows = KdaMin(tiling.chunkSize, tiling.T - info.tokenStart);
        info.valid = info.b >= 0 && info.b < tiling.B && info.validRows > 0;
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
    info.b = 0;
    info.stateIndex = task;
    info.tokenStart = seqBegin + info.localChunk * tiling.chunkSize;
    info.validRows = KdaMin(tiling.chunkSize, seqEnd - info.tokenStart);
    info.valid = seqBegin >= 0 && seqEnd >= seqBegin && seqEnd <= tiling.T && info.validRows > 0;
}

__aicore__ inline int64_t TokenOffset(
    const ChunkKdaBwdPrepareTilingData &tiling, const ChunkInfo &chunk,
    int64_t head, int64_t width)
{
    if (tiling.isVariable != 0) {
        return (head * tiling.T + chunk.tokenStart) * width;
    }
    return ((chunk.b * tiling.NV + head) * tiling.T + chunk.tokenStart) * width;
}

__aicore__ inline int64_t StateOffset(
    const ChunkKdaBwdPrepareTilingData &tiling, const ChunkInfo &chunk, int64_t head)
{
    if (tiling.isVariable != 0) {
        return (head * tiling.totalChunkNum + chunk.stateIndex) * tiling.K * tiling.V;
    }
    return ((chunk.b * tiling.NV + head) * tiling.denseChunkNum + chunk.stateIndex) *
           tiling.K * tiling.V;
}

} // namespace KDA

#endif
