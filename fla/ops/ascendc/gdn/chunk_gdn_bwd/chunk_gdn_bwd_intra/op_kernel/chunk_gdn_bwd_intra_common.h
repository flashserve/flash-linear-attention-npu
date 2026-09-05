/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_COMMON_H
#define CHUNK_GDN_BWD_INTRA_COMMON_H

#include "kernel_operator.h"
#include "chunk_gdn_bwd_intra_struct.h"

namespace GDN {

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

    __aicore__ inline void Resolve(int64_t workId, ChunkGdnBwdIntraWorkMeta &meta) const
    {
        const int64_t chunkId = workId / tiling_->hvSliceCount;
        const int64_t hvSlice = workId - chunkId * tiling_->hvSliceCount;
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

private:
    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    AscendC::GlobalTensor<int64_t> cuSeqlens_;
    AscendC::GlobalTensor<int64_t> chunkIndices_;
};

__aicore__ inline bool ChunkGdnBwdIntraScoreLeader(
    int64_t hvBegin, int64_t r, int64_t headRatio)
{
    if (r == 0) {
        return true;
    }
    return (hvBegin + r) / headRatio != (hvBegin + r - 1) / headRatio;
}

__aicore__ inline int64_t ChunkGdnBwdIntraLeaderR(
    int64_t hvBegin, int64_t r, int64_t headRatio)
{
    const int64_t hk = (hvBegin + r) / headRatio;
    for (int64_t candidate = r; candidate > 0; --candidate) {
        if ((hvBegin + candidate - 1) / headRatio != hk) {
            return candidate;
        }
    }
    return 0;
}

} // namespace GDN

#endif // CHUNK_GDN_BWD_INTRA_COMMON_H
