#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_COMMON_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_COMMON_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"
#include "chunk_gated_delta_rule_bwd_finalize_struct.h"

namespace GDN {

constexpr int64_t CHUNK_SIZE_64 = 64;
constexpr int64_t K_SIZE_128 = 128;
constexpr int64_t V_SIZE_128 = 128;
constexpr int64_t AIV_COUNT_2 = 2;
constexpr int64_t BANK_COUNT_2 = 2;
constexpr int64_t UB_ALIGN_BYTES = 32;
constexpr int64_t WORKSPACE_HEADS_PER_GROUP_4 = 4;
constexpr int64_t WORKSPACE_WINDOW_COUNT_2 = 2;
constexpr int64_t WORKSPACE_BUFFER_COUNT_8 =
    WORKSPACE_HEADS_PER_GROUP_4 * WORKSPACE_WINDOW_COUNT_2;
constexpr int64_t WORKSPACE_REGION_COUNT_7 = 7;
constexpr int64_t WORKSPACE_VECTOR_ELEMS = CHUNK_SIZE_64 * K_SIZE_128;

__aicore__ inline int64_t GetWorkspaceHeadOffset(
    int64_t coreIdx, int64_t groupRound, int64_t headOffset)
{
    const int64_t windowStart =
        (groupRound & 1) * WORKSPACE_HEADS_PER_GROUP_4;
    return (coreIdx * WORKSPACE_BUFFER_COUNT_8 + windowStart + headOffset) *
        WORKSPACE_VECTOR_ELEMS;
}

// 所有核间同步按方向合并为两条有序链，每个方向只使用一个 ready ID。
constexpr uint64_t VEC_TO_CUBE_READY_FLAG = 1;
constexpr uint64_t CUBE_TO_VEC_READY_FLAG = 3;

struct ChunkInfo {
    // 定长模式使用 bIdx 和 batch 内 tokenStart；变长模式的 tokenStart 已经是
    // packed T 上的全局偏移，因此 bIdx 固定为 0。
    int64_t bIdx = 0;
    int64_t tokenStart = 0;
    int64_t chunkLen = 0;
    // h/dh 的第三维索引。定长模式是 batch 内 chunk 编号；变长模式
    // 是 chunk_indices 当前任务对应的两个扁平元素所表示的全局输出 chunk 编号。
    int64_t stateChunkIdx = 0;
    bool valid = false;
};

__aicore__ inline int64_t Min(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline void GetChunkInfo(
    int64_t chunkTaskIdx, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkGatedDeltaRuleBwdFinalizeTilingData &tiling, ChunkInfo &info)
{
    info.valid = false;
    info.stateChunkIdx = 0;
    if (chunkTaskIdx < 0 || chunkTaskIdx >= tiling.totalChunkNum) {
        return;
    }
    if (tiling.isVariable == 0) {
        // 定长任务顺序为 [batch, chunk]。尾 chunk 保存真实 chunkLen，
        // Vector 指令只处理有效 token，不从 GM 读取 padding 区。
        const int64_t bIdx = chunkTaskIdx / tiling.chunkNumForT;
        const int64_t chunkIdx = chunkTaskIdx % tiling.chunkNumForT;
        const int64_t tokenStart = chunkIdx * tiling.chunkSize;
        info.bIdx = bIdx;
        info.tokenStart = tokenStart;
        info.chunkLen = Min(tiling.chunkSize, tiling.T - tokenStart);
        info.stateChunkIdx = chunkIdx;
        info.valid = bIdx < tiling.B && info.chunkLen > 0;
        return;
    }

    if (cuSeqlens == nullptr || chunkIndices == nullptr) {
        return;
    }
    AscendC::GlobalTensor<int64_t> cu;
    AscendC::GlobalTensor<int64_t> chunks;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
    chunks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    // 每个变长任务读取一组 [序列编号, 序列内 chunk 编号]，再换算成所有 stage
    // 共用的 packed token offset。
    const int64_t seqIdx = chunks.GetValue(chunkTaskIdx * 2);
    const int64_t chunkIdx = chunks.GetValue(chunkTaskIdx * 2 + 1);
    if (seqIdx < 0 || seqIdx >= tiling.seqNum || chunkIdx < 0) {
        return;
    }
    const int64_t seqStart = cu.GetValue(seqIdx);
    const int64_t seqEnd = cu.GetValue(seqIdx + 1);
    const int64_t tokenStart = seqStart + chunkIdx * tiling.chunkSize;
    info.bIdx = 0;
    info.tokenStart = tokenStart;
    info.chunkLen = Min(tiling.chunkSize, seqEnd - tokenStart);
    info.stateChunkIdx = chunkTaskIdx;
    info.valid = seqStart >= 0 && seqEnd <= tiling.T && tokenStart >= seqStart && info.chunkLen > 0;
}

} // namespace GDN

#endif
