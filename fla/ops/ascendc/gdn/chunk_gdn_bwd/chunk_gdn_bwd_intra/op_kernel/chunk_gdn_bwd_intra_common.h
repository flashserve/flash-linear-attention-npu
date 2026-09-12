/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_COMMON_H
#define CHUNK_GDN_BWD_INTRA_COMMON_H

#include "kernel_operator.h"
#include "chunk_gdn_bwd_intra_struct.h"

namespace GDN {

constexpr uint32_t INTRA_CG_MAX = 4;
// AIV1 使用同一组逻辑 flag 加 16 的编号，AIV0 使用基础编号。
constexpr uint32_t INTRA_SUBBLOCK_FLAG_OFFSET = 16;
// Score 和 Stage 1 record 都采用 ready -> wait -> free -> wait 的双向握手；
// 一次 Wait 消耗当前信号后，反向通知复用同一个 flag ID。
constexpr uint32_t INTRA_SCORE_FREE_FLAG = 0;
constexpr uint32_t INTRA_SCORE_READY_FLAG = 0;
constexpr uint32_t INTRA_WORKSPACE_READY_FLAG = 8;
constexpr uint32_t INTRA_SCORE_SLOT_BYTES = 8 * 1024;
constexpr uint32_t INTRA_SCORE_UB_BASE = 0;
constexpr uint32_t INTRA_MATRIX_HALF_BYTES = 4 * 1024;
constexpr uint32_t INTRA_VDO_INPUT_HALF_BYTES = 8 * 1024;

class ChunkGdnBwdIntraWorkMapper {
public:
    __aicore__ inline void Init(
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGdnBwdIntraTilingData *__restrict tiling)
    {
        tiling_ = tiling;
        if (tiling_->isVarlen != 0) {
            cuSeqlens_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
            chunkIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        }
    }

    // 常规调度：先遍历一个 chunk 内的 HV 切片，再进入下一个 chunk。
    __aicore__ inline void ResolveChunkMajor(
        int64_t workId, ChunkGdnBwdIntraWorkMeta &meta) const
    {
        const int64_t chunkId = workId / tiling_->hvSliceCount;
        const int64_t hvSlice = workId - chunkId * tiling_->hvSliceCount;
        ResolveChunkMetadata(chunkId, hvSlice, meta);
    }

private:
    // 同时覆盖定长和变长输入，统一产出本次 work 的 token/head 边界。
    __aicore__ inline void ResolveChunkMetadata(
        int64_t chunkId, int64_t hvSlice, ChunkGdnBwdIntraWorkMeta &meta) const
    {
        meta.hvBegin = hvSlice * tiling_->cg;
        const int64_t remainHeads = tiling_->valueHeads - meta.hvBegin;
        meta.validHeads = remainHeads < tiling_->cg ? remainHeads : tiling_->cg;
        if (tiling_->isVarlen == 0) {
            meta.batch = chunkId / tiling_->chunksPerBatch;
            const int64_t localChunk = chunkId - meta.batch * tiling_->chunksPerBatch;
            meta.tokenStart = localChunk * tiling_->chunkSize;
            const int64_t remain = tiling_->seqlen - meta.tokenStart;
            meta.validTokens = remain < tiling_->chunkSize ? remain : tiling_->chunkSize;
            return;
        }
        const int64_t sequence = chunkIndices_.GetValue(chunkId * 2);
        const int64_t localChunk = chunkIndices_.GetValue(chunkId * 2 + 1);
        const int64_t begin = cuSeqlens_.GetValue(sequence);
        const int64_t end = cuSeqlens_.GetValue(sequence + 1);
        meta.batch = 0;
        meta.tokenStart = begin + localChunk * tiling_->chunkSize;
        const int64_t remain = end - meta.tokenStart;
        meta.validTokens = remain < tiling_->chunkSize ? remain : tiling_->chunkSize;
    }

    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<int64_t> cuSeqlens_;
    AscendC::GlobalTensor<int64_t> chunkIndices_;
};

template <uint32_t G>
__aicore__ inline bool ChunkGdnBwdIntraScoreLeader(int64_t hvBegin, int64_t r)
{
    // 当前 r 是该 q/k head 在 HV 切片内的首次出现时，才需要计算一份 Score。
    if (r == 0) {
        return true;
    }
    return (hvBegin + r) / G != (hvBegin + r - 1) / G;
}

template <uint32_t G>
__aicore__ inline int64_t ChunkGdnBwdIntraLeaderR(int64_t hvBegin, int64_t r)
{
    // 回溯到共享同一个 q/k head 的首个 r，用它定位复用的 Score 和 k 槽。
    const int64_t hk = (hvBegin + r) / G;
    for (int64_t candidate = r; candidate > 0; --candidate) {
        if ((hvBegin + candidate - 1) / G != hk) {
            return candidate;
        }
    }
    return 0;
}

} // namespace GDN

#endif // CHUNK_GDN_BWD_INTRA_COMMON_H
