/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_FWD_H_POLICY_H
#define CHUNK_FWD_H_POLICY_H

#include <cstdint>

#include "chunk_fwd_h_tiling_key.h"
#include "kernel_operator.h"

namespace GDN {

constexpr uint32_t FWD_H_CHUNK = 64;
constexpr uint32_t FWD_H_K = 128;
constexpr uint32_t FWD_H_V = 128;
constexpr uint32_t FWD_H_AIC_HEAD_SLOTS = 4;
constexpr uint32_t FWD_H_AIV_COUNT = 2;
constexpr uint32_t FWD_H_AIV_HEAD_SLOTS = 2;
constexpr uint32_t FWD_H_FLAG_SUBBLOCK_STRIDE = 16;

constexpr uint32_t FWD_H_TOKEN_MATRIX_BF16_BYTES = FWD_H_CHUNK * FWD_H_V * sizeof(bfloat16_t);
constexpr uint32_t FWD_H_TOKEN_MATRIX_FP32_BYTES = FWD_H_CHUNK * FWD_H_V * sizeof(float);
constexpr uint32_t FWD_H_STATE_BF16_BYTES = FWD_H_K * FWD_H_V * sizeof(bfloat16_t);
constexpr uint32_t FWD_H_STATE_FP32_BYTES = FWD_H_K * FWD_H_V * sizeof(float);

// AIC L1 固定布局。四个 W、四个 H/right、四个 kg 槽都按 roundHead 绑定，round 内不复用。
constexpr uint32_t FWD_H_L1_W_BASE = 0;
constexpr uint32_t FWD_H_L1_W_SLOT_BYTES = 16 * 1024;
constexpr uint32_t FWD_H_L1_H_RIGHT_BASE = 128 * 1024;
constexpr uint32_t FWD_H_L1_H_RIGHT_SLOT_BYTES = 32 * 1024;
constexpr uint32_t FWD_H_L1_KG_BASE = 256 * 1024;
constexpr uint32_t FWD_H_L1_KG_SLOT_BYTES = 16 * 1024;
constexpr uint32_t FWD_H_L1_USED_BYTES = 320 * 1024;

// 每个 AIV 的两个 local slot 固定为 64 KiB。P 使用低 32 KiB，Stage1 可在高 32 KiB
// 生成 BF16 右操作数；D 使用完整 64 KiB。
constexpr uint32_t FWD_H_UB_LOCAL_SLOT_BYTES = 64 * 1024;
constexpr uint32_t FWD_H_UB_LOCAL0_BASE = 0;
constexpr uint32_t FWD_H_UB_LOCAL1_BASE = 64 * 1024;
constexpr uint32_t FWD_H_UB_BF16_STATE_BASE = 128 * 1024;
constexpr uint32_t FWD_H_UB_FP32_STATE_BASE = 128 * 1024;
constexpr uint32_t FWD_H_UB_BF16_WORK_BASE = 192 * 1024;
constexpr uint32_t FWD_H_UB_GATE_BASE = 224 * 1024;
constexpr uint32_t FWD_H_UB_SHARE_BASE = 226 * 1024;

// ready/free 始终成对，且同一 ID 在复用前必须已被上一代 wait 消费。
// A5 的 mode=0x4 是 split AIC/AIV 子块同步，flagId 可通过 16-ID 步长区分两个 AIV。
// A2/A3 的 mode=0x2 只保留低 4 位 flagId，因此每类信号用 aiv 选择 0/1 两个 ID；
// localSlot 按各 AIV 固定的 slot 顺序形成计数事件，不能让两个 AIV 共用同一个 ID。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
constexpr uint32_t FWD_H_AIV_FLAG_STRIDE = 16;
#else
constexpr uint32_t FWD_H_AIV_FLAG_STRIDE = 0;
#endif
constexpr uint32_t FWD_H_P_FREE_FLAG = 0;
constexpr uint32_t FWD_H_P_READY_FLAG = 2;
constexpr uint32_t FWD_H_D_FREE_FLAG = 4;
constexpr uint32_t FWD_H_D_READY_FLAG = 6;
constexpr uint32_t FWD_H_RIGHT_FREE_FLAG = 8;
constexpr uint32_t FWD_H_RIGHT_READY_FLAG = 10;
constexpr uint32_t FWD_H_H_READY_FLAG = 12;
constexpr uint32_t FWD_H_ROUND_DONE_FLAG = 14;

struct FwdHSequenceSpan {
    uint32_t sequence = 0;
    uint32_t physicalBatch = 0;
    uint32_t tokenBegin = 0;
    uint32_t length = 0;
    uint32_t chunkPrefix = 0;
    uint32_t chunkCount = 0;
    uint32_t totalChunks = 0;
};

struct FwdHChunkSpan {
    uint32_t chunk = 0;
    uint32_t globalChunk = 0;
    uint32_t tokenBegin = 0;
    uint32_t validTokens = 0;
    bool first = false;
    bool last = false;
};

struct FwdHHeadBinding {
    uint32_t roundHead = 0;
    uint32_t hv = 0;
    uint32_t kh = 0;
    uint32_t kgSlot = 0;
    uint32_t aiv = 0;
    uint32_t localSlot = 0;
};

struct FwdHKgBinding {
    uint32_t kh = 0;
    uint32_t slot = 0;
    uint32_t firstConsumer = 0;
    uint32_t lastConsumer = 0;
};

struct FwdHHeadRoundPlan {
    uint32_t round = 0;
    uint32_t activeHeadCount = 0;
    uint32_t requiredKhCount = 0;
    FwdHHeadBinding heads[FWD_H_AIC_HEAD_SLOTS]{};
    FwdHKgBinding kg[FWD_H_AIC_HEAD_SLOTS]{};
};

struct FwdHWorkUnit {
    FwdHSequenceSpan sequence{};
    FwdHHeadRoundPlan headRound{};
};

struct FwdHWorkspace {
    __gm__ uint8_t *base = nullptr;
    int64_t pOffset = 0;
    int64_t rightOffset = 0;
    int64_t stateOffset = 0;
    int64_t dOffset = 0;
};

// Kernel 入口一次性从 GM tiling 读取这些标量，后续调度和各 stage 只访问寄存器/栈上的副本。
struct FwdHRuntimeTiling {
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t kNumHead = 0;
    int64_t vNumHead = 0;
    int64_t kHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t chunkSize = 0;
    bool useInitialState = false;
    bool storeFinalState = false;
    int64_t dataType = 0;
    int64_t gDataType = 0;
    int64_t stateDataType = 0;
    int64_t isVariedLen = 0;
    int64_t shapeBatch = 0;
    int64_t tokenBatch = 0;
    bool useG = false;
    bool useGk = false;
    bool useExp2 = false;
    bool stateVFirst = false;
    int64_t vWorkspaceOffset = 0;
    int64_t vUpdateWorkspaceOffset = 0;
    int64_t kDecayWorkspaceOffset = 0;
    int64_t hWorkspaceOffset = 0;
    int64_t numSeqWorkspaceOffset = 0;
    int64_t numChunksWorkspaceOffset = 0;
};

struct FwdHKernelArgs {
    GM_ADDR k = nullptr;
    GM_ADDR w = nullptr;
    GM_ADDR u = nullptr;
    GM_ADDR g = nullptr;
    GM_ADDR gk = nullptr;
    GM_ADDR initialState = nullptr;
    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;
    GM_ADDR h = nullptr;
    GM_ADDR vNew = nullptr;
    GM_ADDR finalState = nullptr;
    GM_ADDR workspace = nullptr;
    FwdHRuntimeTiling tiling{};
};

// AIC 侧选择配对 AIV。A2/A3 的低 4 位 flag 按 aiv 区分，A5 的每个 subblock
// 拥有独立 16-ID 空间，因此再按 subblock stride 区分两个 AIV。
__aicore__ inline uint32_t FwdHAicPeerFlag(uint32_t base, uint32_t localSlot, uint32_t aiv)
{
    (void)localSlot;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return base + localSlot + aiv * FWD_H_AIV_FLAG_STRIDE;
#else
    return base + aiv;
#endif
}

// AIV 侧已经处于具体 subblock。A2/A3 按 subblock 选择 flag，localSlot
// 通过同一 ID 的 ready/free 计数顺序复用；A5 在本 subblock 内再按 localSlot 区分。
__aicore__ inline uint32_t FwdHAivLocalFlag(uint32_t base, uint32_t localSlot)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return base + localSlot;
#else
    (void)localSlot;
    return base + AscendC::GetSubBlockIdx();
#endif
}

__aicore__ inline uint32_t FwdHLocalSlotBase(uint32_t localSlot)
{
    return localSlot == 0 ? FWD_H_UB_LOCAL0_BASE : FWD_H_UB_LOCAL1_BASE;
}

__aicore__ inline uint32_t FwdHAivHeadCount(uint32_t activeHeadCount, uint32_t aiv)
{
    return activeHeadCount > aiv ? (activeHeadCount - aiv + 1) / 2 : 0;
}

} // namespace GDN

#endif // CHUNK_FWD_H_POLICY_H
