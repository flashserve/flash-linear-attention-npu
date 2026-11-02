// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "kernel_operator.h"
#include "../recurrent_kda_tiling_data.h"
#include "recurrent_kda_regbase.h"

namespace RecurrentKda {
using namespace AscendC;
constexpr uint16_t RKDA_DIM_128 = 128;
constexpr uint32_t RKDA_ROW_BF16_BYTES_256 = RKDA_DIM_128 * sizeof(bfloat16_t);
constexpr uint32_t RKDA_ROW_FP32_BYTES_512 = RKDA_DIM_128 * sizeof(float);
constexpr uint32_t RKDA_GATE_DOUBLE_BYTES_1024 = 2 * RKDA_ROW_FP32_BYTES_512;
constexpr uint32_t RKDA_PARAM_SLOT_BYTES_544 = RKDA_ROW_FP32_BYTES_512 + 32;
constexpr uint32_t RKDA_PARAM_DOUBLE_BYTES_1088 = 2 * RKDA_PARAM_SLOT_BYTES_544;
constexpr uint32_t RKDA_GUARD_BYTES_2048 = 4 * RKDA_ROW_FP32_BYTES_512;
constexpr uint64_t RKDA_INVALID_SLOT = static_cast<uint64_t>(-1);
constexpr uint32_t RKDA_NO_BANK = 3;
enum RKDABankUse : uint32_t {
    RKDA_BANK_FREE = 0, RKDA_BANK_RECURRENCE = 1, RKDA_BANK_PREFETCHED = 2, RKDA_BANK_MTE3_SOURCE = 4
};
struct RKDATask {
    uint64_t id, request, head;
    int64_t begin, end;
};
struct RKDAInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

// Pure per-core data. Functions and event operations live outside this context.
template <typename inType, typename outType, typename stateType, bool stateVFirst>
struct RKDAVectorContext {
    const RecurrentKdaTilingData *td_;
    RKDAInitParams args_;
    TBuf<TPosition::VECCALC> storage_;
    LocalTensor<uint8_t> ub_;
    GlobalTensor<inType> q_, k_, v_;
    GlobalTensor<stateType> stateIn_, stateOut_;
    GlobalTensor<outType> out_;
    GlobalTensor<float> aLog_, bias_;
    uint32_t x_, betaSlot_, qOffset_, kOffset_, vOffset_, gateOffset_, betaOffset_, initialBank_, alternateBank_;
    uint32_t paramOffset_, outOffset_, eventOffset_;
    uint32_t bankUse_[3] = {RKDA_BANK_FREE, RKDA_BANK_FREE, RKDA_BANK_FREE};
    uint32_t prefetchedBank_ = RKDA_NO_BANK, pendingStateBank_ = RKDA_NO_BANK;
#ifdef RKDA_EXECUTION_TRACE
    GlobalTensor<uint64_t> trace_;
    uint64_t traceCount_ = 0;
#endif
};
} // namespace RecurrentKda
