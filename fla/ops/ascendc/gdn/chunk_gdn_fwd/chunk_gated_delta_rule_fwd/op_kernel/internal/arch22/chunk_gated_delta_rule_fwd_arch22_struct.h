/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GATED_DELTA_RULE_FWD_ARCH22_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_FWD_ARCH22_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace GDN {

constexpr uint64_t FP32_SOLVE_MERGE_BATCH_SIZE = 16;
// 前半段parent切批粒度独立于Solve内部16任务缓冲协议。
constexpr uint64_t FRONT_PARENT_BATCH_SIZE = 64;
constexpr uint64_t FP32_SOLVE_RESULT_BUFFER_COUNT = 2;
constexpr uint64_t FP32_SOLVE_SMALL_TEMP_SLOT_COUNT = 16;
constexpr uint64_t FP32_SOLVE_LARGE_TEMP_SLOT_COUNT = 2;
// 各层固定使用同一物理核组 arena，避免不同矩阵宽度改变跨组 scratch 边界。
constexpr uint64_t FP32_SOLVE_TEMP_ELEMENTS = FP32_SOLVE_SMALL_TEMP_SLOT_COUNT * 32 * 32;
constexpr uint64_t FP32_SOLVE_RESULT_SLOTS =
    FP32_SOLVE_RESULT_BUFFER_COUNT * FP32_SOLVE_MERGE_BATCH_SIZE;
constexpr uint64_t FP32_SOLVE_ARENA64_ELEMENTS =
    FP32_SOLVE_TEMP_ELEMENTS + FP32_SOLVE_RESULT_SLOTS * 32 * 32;
constexpr uint64_t FP32_SOLVE_ARENA128_ELEMENTS =
    FP32_SOLVE_TEMP_ELEMENTS + FP32_SOLVE_RESULT_SLOTS * 64 * 64;
static_assert(FP32_SOLVE_TEMP_ELEMENTS >= FP32_SOLVE_LARGE_TEMP_SLOT_COUNT * 64 * 64);

struct Arch22ChunkGatedDeltaRuleFwdAbcTiling {
    uint64_t B;
    uint64_t Hk;
    uint64_t Hv;
    uint64_t hvPerHk;
    uint64_t T;
    uint64_t K;
    uint64_t BT;
    uint64_t NT;
    uint64_t taskNum;
    uint64_t usedAicNum;
    uint64_t usedAivNum;
    uint64_t btAlign;
    uint64_t isVarlen;
    uint64_t scoreWorkspaceBytes;
    uint64_t aWorkspaceBytes;
    uint64_t solveWorkspacePerCoreBytes;
    int64_t totalTiles;
    int64_t matrixSize;
    int64_t numHeads;
    int64_t seqLen;
    int64_t batchSize;
    int64_t isLower;
    int64_t hasCuSeqlens;
    int64_t tilesPerCore;
    int64_t chunkSize;
    int64_t numChunks;
    int64_t lastChunkValidSize;
    int64_t totalChunks;
    int64_t layoutMode;
    int64_t dtypeMode;
    int64_t totalTokens;
    uint64_t qkvLayout;  // 0: [B,H,T,D], 1: [B,T,H,D]; intermediates remain head-major.
    uint64_t oLayout;    // 0: [B,H,T,V], 1: [B,T,H,V].
    TCubeTiling cubeTilingData;
};

struct Arch22ChunkGatedDeltaRuleFwdTrailer {
    Arch22ChunkGatedDeltaRuleFwdAbcTiling abc;
    uint64_t scoreWorkspaceOffset;
    uint64_t aWorkspaceOffset;
    uint64_t solveWorkspaceOffset;
    uint64_t gCumsumBhtOffset;
    // 仅 DAV_2201 的分层 FP32 Solve 使用；其余架构保持零值且不访问。
    uint64_t solveFp32InputOffset;
    uint64_t solveD16Offset;
    uint64_t solveD32Offset;
    uint64_t solveD64Offset;
    uint64_t solveSequenceCount;
    uint64_t outputGCumsum;
    // HO 空闲流水通知协议的 GM ready 区，仅 arch22 私有 kernel 读取。host 只在
    // “可能命中”预留判定通过时写入；hoPipelineAvailable=0 时三者全 0 且不分配
    // ready。hoReadyBankCount 为 chunk 数上界（协议侧参数为 uint32，host 已核查
    // 不超界）；hoReadyWorkspaceOffset 与上方中间区字段同以 userWorkspace 为
    // 基址，区域大小 AlignUp(hoReadyBankCount*2*C*32B, 512)。
    uint64_t hoPipelineAvailable;
    uint64_t hoReadyWorkspaceOffset;
    uint64_t hoReadyBankCount;
    // A2小批WU：每物理组两个独立的Vb/KbgExp段，容量由实际任务数推导。
    uint64_t frontWuWorkspaceOffset;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_FWD_ARCH22_STRUCT_H
