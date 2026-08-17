/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "../chunk_kda_fwd_varlen.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif
#endif
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "../chunk_kda_fwd_plan.h"

using namespace AscendC;

namespace KdaPostWu {
namespace {
using KdaInt64 = tla::Int<64>;
using KdaInt128 = tla::Int<128>;
constexpr float LN2 = 0.69314718055994530942f;
constexpr float KDA_EXP2_CLAMP = 80.0f;
constexpr float KDA_EXP_INPUT_MAX = KDA_EXP2_CLAMP * LN2;
constexpr float KDA_EXP_INPUT_MIN = -KDA_EXP2_CLAMP * LN2;
constexpr float KDA_FP16_MAX = 65504.0f;
constexpr uint32_t EXP2_UB_ELEMENTS = 256;
constexpr uint32_t EXP2_UB_BYTES = EXP2_UB_ELEMENTS * (sizeof(float) + sizeof(uint16_t));
constexpr uint32_t EXP2_EVENT_ID = 0;
constexpr uint32_t KDA_SOLVE_BT = 64;
constexpr uint32_t KDA_SOLVE_MATRIX_ELEMENTS = KDA_SOLVE_BT * KDA_SOLVE_BT;
constexpr uint32_t KDA_SOLVE_SCRATCH_X = 0;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y0 = 1;
constexpr uint32_t KDA_SOLVE_SCRATCH_TMP = 2;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y1 = 3;
constexpr uint32_t KDA_SOLVE_SCRATCH_IDENTITY = 4;
constexpr uint32_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint32_t KDA_SOLVE_DIAG_BT = 16;
constexpr uint32_t KDA_SOLVE_DIAG_BLOCKS = KDA_SOLVE_BT / KDA_SOLVE_DIAG_BT;
constexpr uint32_t KDA_SOLVE_DIAG_MCH_ITERS = 3;
constexpr uint32_t KDA_SCORE_REF_BC = 16;
constexpr uint32_t KDA_VEC_ARENA_ELEMENTS = 32768;
constexpr uint32_t KDA_BITS_PER_MASK_BYTE = 8;
constexpr uint32_t KDA_SELECT_COL_BLOCKS = 2;
constexpr uint32_t KDA_SELECT_COL_MASK_BYTES = KDA_SOLVE_MATRIX_ELEMENTS / KDA_BITS_PER_MASK_BYTE;
constexpr uint32_t KDA_SELECT_MASK_BYTES = KDA_SELECT_COL_BLOCKS * KDA_SELECT_COL_MASK_BYTES;
constexpr uint32_t KDA_SELECT_AQK_MASK_BYTE_OFFSET = 120 * 1024;
constexpr uint32_t KDA_SELECT_AKK_MASK_BYTE_OFFSET = KDA_SELECT_AQK_MASK_BYTE_OFFSET + KDA_SELECT_MASK_BYTES;
constexpr uint32_t KDA_SELECT_ZERO_BYTE_OFFSET = KDA_SELECT_AKK_MASK_BYTE_OFFSET + KDA_SELECT_MASK_BYTES;
constexpr uint32_t KDA_SELECT_ZERO_FLOAT_OFFSET = KDA_SELECT_ZERO_BYTE_OFFSET / sizeof(float);
constexpr uint8_t KDA_SCORE_DONE_FLAG0 = 2;
constexpr uint8_t KDA_SCORE_DONE_FLAG1 = 3;
constexpr uint8_t KDA_SCORE_READY_FLAG0 = 4;
constexpr uint8_t KDA_SCORE_READY_FLAG1 = 5;
constexpr uint32_t KDA_SCORE_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_SYNC_REVERSE_DEPTH = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint32_t KDA_SCORE_SCRATCH_QG = 0;
constexpr uint32_t KDA_SCORE_SCRATCH_W = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_KG = 2;
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_GATE_TILE_ROWS = 32;
constexpr uint32_t KDA_TYPICAL_GATE_TILE_ROWS = 16;
constexpr uint32_t KDA_TYPICAL_GATE_PIPELINE_ROWS = 32;
constexpr uint16_t KDA_TYPICAL_GATE_PIPELINE_STAGES = 3;
constexpr uint32_t KDA_TYPICAL_GATE_RESIDENT_ROWS = 32;
constexpr uint32_t KDA_FALLBACK_K_RESIDENT_ROWS = 32;
constexpr uint32_t KDA_FALLBACK_K_RESIDENT_BYTE_OFFSET = 112 * 1024;
constexpr uint32_t KDA_POST_EVENT = 3;
constexpr uint32_t KDA_POST_EVENT_NEXT = 4;
constexpr uint32_t KDA_POST_EVENT_FIX = 5;
constexpr uint32_t KDA_POST_PIPELINE_L1_SLOT_BYTES = 24 * 1024;
constexpr uint32_t KDA_POST_PIPELINE_L1_A_BYTES = 64 * 64 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L1_B_BYTES = 64 * 128 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L1_U_SLOT_BYTES = 64 * 128 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L0_A_SLOT_BYTES = 64 * 64 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L0_B_SLOT_BYTES = 64 * 256 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L0_C_SLOT_BYTES = 64 * 256 * sizeof(float);
constexpr uint16_t KDA_POST_PIPELINE_STAGE_COUNT = 2;
constexpr uint16_t KDA_POST_FUSED_BATCH_TASKS = 4;
// 主输入和 U 输入都使用 MTE2_MTE1/MTE1_MTE2，必须占用互不重叠的
// 事件区间。主双槽保留 3/4，U 双槽使用 0/1，并避开保留的 6/7。
constexpr uint16_t KDA_POST_PIPELINE_U_EVENT = 0;
static_assert(KDA_POST_EVENT + KDA_POST_PIPELINE_STAGE_COUNT - 1 <= 5,
              "PostWU main pipeline event IDs must stay within 0..5");
static_assert(KDA_POST_PIPELINE_U_EVENT + KDA_POST_PIPELINE_STAGE_COUNT - 1 <= 5,
              "PostWU U pipeline event IDs must stay within 0..5");
static_assert(
    KDA_POST_PIPELINE_U_EVENT + KDA_POST_PIPELINE_STAGE_COUNT <= KDA_POST_EVENT ||
        KDA_POST_EVENT + KDA_POST_PIPELINE_STAGE_COUNT <= KDA_POST_PIPELINE_U_EVENT,
    "PostWU main and U pipelines must use disjoint event IDs");
constexpr bool KDA_ENABLE_POST_AIC_PIPELINE = true;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename InputT>
__simd_callee__ inline void LoadPostKdaGateRegbasePair(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    __ubuf__ InputT *src,
    AscendC::MicroAPI::MaskReg &inputMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<InputT, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zeroReg, oneReg, src);
    } else {
        RegTensor<InputT> inputReg;
        LoadIn<InputT, false>(inputReg, src);
        CastHalf2Float<InputT>(zeroReg, oneReg, inputReg, inputMask);
    }
}

template <typename OutputT>
__simd_callee__ inline void StorePostKdaGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<OutputT, half>()) {
        Mins(zeroReg, zeroReg, KDA_FP16_MAX, floatMask);
        Mins(oneReg, oneReg, KDA_FP16_MAX, floatMask);
        Maxs(zeroReg, zeroReg, -KDA_FP16_MAX, floatMask);
        Maxs(oneReg, oneReg, -KDA_FP16_MAX, floatMask);
    }
    RegTensor<OutputT> outputReg;
    CastFloat2Half<OutputT>(outputReg, zeroReg, oneReg, floatMask);
    StoreAlign(dst, outputReg, inputMask);
}

template <typename T, typename GK_T>
static __simd_vf__ inline void ComputePostKdaKgRegbase(
    __ubuf__ T *kAndKg, __ubuf__ GK_T *gate, __ubuf__ float *ref,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<T>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> refZeroReg;
            RegTensor<float> refOneReg;
            RegTensor<float> expZeroReg;
            RegTensor<float> expOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;

            LoadPostKdaGateRegbasePair<GK_T>(
                gateZeroReg, gateOneReg, gate + offset, inputMask);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                refZeroReg, refOneReg, ref + col);
            SubFloatTwoReg(expZeroReg, expOneReg, refZeroReg, refOneReg,
                           gateZeroReg, gateOneReg, floatMask);
            Muls(expZeroReg, expZeroReg, LN2, floatMask);
            Muls(expOneReg, expOneReg, LN2, floatMask);
            MinsFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg,
                            KDA_EXP_INPUT_MAX, floatMask);
            Maxs(expZeroReg, expZeroReg, KDA_EXP_INPUT_MIN, floatMask);
            Maxs(expOneReg, expOneReg, KDA_EXP_INPUT_MIN, floatMask);
            ExpFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg, floatMask);

            LoadPostKdaGateRegbasePair<T>(
                inputZeroReg, inputOneReg, kAndKg + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            StorePostKdaGateRegbasePair<T>(
                kAndKg + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
        }
    }
}
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif
// Cube 与 Fixpipe 通过两个 L0C 槽并行，事件只负责槽位所有权交接。
using KdaDispatchPolicy = Common::MmadPingpong<KdaArchTag, false, false, 2>;
// K/V > 128 的 256 列 tile 会占满 A2 L0C，必须使用单槽；窄 tile
// 保留原有策略，避免改变 K/V <= 128 的热路径。
using KdaWideDispatchPolicy = Common::MmadPingpong<KdaArchTag, false, false, 1>;

__aicore__ inline uint32_t FloatToBits(float value)
{
    union Bits {
        __aicore__ Bits() {}
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline float BitsToFloat(uint32_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint32_t u;
        float f;
    } bits;
    bits.u = value;
    return bits.f;
}

__aicore__ inline uint16_t Bf16ToBits(bfloat16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        bfloat16_t f;
        uint16_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline bfloat16_t BitsToBf16(uint16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint16_t u;
        bfloat16_t f;
    } bits;
    bits.u = value;
    return bits.f;
}

template <typename T>
__aicore__ inline T FloatToType(float value)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        uint32_t bits = FloatToBits(value);
        uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
        return BitsToBf16(static_cast<uint16_t>((bits + bias) >> 16));
    }
    return static_cast<T>(value);
}

template <typename T, typename GK_T = float, typename BETA_T = float>
class ChunkKdaFwdPostWuKernel {
public:
    using OUT_T = T;
    using AKK_T = T;
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
                                GM_ADDR preparedQG, GM_ADDR preparedAqk,
                                GM_ADDR propagatedVNew, GM_ADDR propagatedH, GM_ADDR o, GM_ADDR finalState, GM_ADDR aqk,
                                GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
                                GM_ADDR workspace, const TilingData &tiling, TPipe *pipe,
                                bool initVecBuffers = true)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ T *)q);
        k_.SetGlobalBuffer((__gm__ T *)k);
        v_.SetGlobalBuffer((__gm__ T *)v);
        gk_.SetGlobalBuffer((__gm__ GK_T *)gk);
        beta_.SetGlobalBuffer((__gm__ BETA_T *)beta);
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer((__gm__ float *)initialState);
        }
        cuSeqlensAddr_ = reinterpret_cast<__gm__ int64_t *>(cuSeqlens);
        compactPlanAddr_ = compactPlan;
        if (preparedQG != nullptr) {
            preparedQG_.SetGlobalBuffer((__gm__ T *)preparedQG);
        }
        if (preparedAqk != nullptr) {
            preparedAqk_.SetGlobalBuffer((__gm__ T *)preparedAqk);
        }
        if (propagatedVNew != nullptr) {
            propagatedVNew_.SetGlobalBuffer((__gm__ T *)propagatedVNew);
        }
        if (propagatedH != nullptr) {
            propagatedH_.SetGlobalBuffer((__gm__ T *)propagatedH);
        }
        chunkIndicesAddr_ = reinterpret_cast<__gm__ int64_t *>(chunkIndices);
        o_.SetGlobalBuffer((__gm__ OUT_T *)o);
        finalState_.SetGlobalBuffer((__gm__ float *)finalState);
        aqk_.SetGlobalBuffer((__gm__ float *)aqk);
        akk_.SetGlobalBuffer((__gm__ AKK_T *)akk);
        w_.SetGlobalBuffer((__gm__ T *)w);
        u_.SetGlobalBuffer((__gm__ OUT_T *)u);
        qg_.SetGlobalBuffer((__gm__ T *)qg);
        kg_.SetGlobalBuffer((__gm__ T *)kg);
        vNew_.SetGlobalBuffer((__gm__ T *)vNew);
        h_.SetGlobalBuffer((__gm__ float *)h);
        solveWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);

        B_ = tiling.batch;
        N_ = tiling.seqNum;
        H_ = tiling.qHeadNum;
        HV_ = tiling.vHeadNum;
        T_ = tiling.seqlen;
        K_ = tiling.kHeadDim;
        V_ = tiling.vHeadDim;
        BT_ = tiling.chunkSize;
        NT_ = tiling.totalChunks;
        scale_ = tiling.scale;
        hasInitial_ = tiling.hasInitialState;
        isVarLen_ = tiling.isVarLen;
        inputSequenceMajor_ = tiling.inputSequenceMajor;
        if ASCEND_IS_AIV {
            uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            solveCoreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        } else {
            solveCoreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }
        if (pipe_ != nullptr && initVecBuffers) {
            pipe_->InitBuffer(exp2Buf_, EXP2_UB_BYTES);
            pipe_->InitBuffer(vecBuf_, KDA_VEC_ARENA_ELEMENTS * sizeof(float));
            const uint64_t gateWritebackRows =
                ScoreVectorMaxRows(5 * sizeof(float) + 2 * sizeof(T) + sizeof(GK_T));
            pipe_->InitBuffer(gateWritebackBuf_,
                              static_cast<uint32_t>(gateWritebackRows * K_ *
                                                    (3 * sizeof(T) + sizeof(GK_T))));
            AllocVectorEvents();
        }
    }
    __aicore__ inline void ProcessAiv()
    {
        ProcessPostAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAic()
    {
        ProcessPostAic();
    }

    __aicore__ inline void ProcessTailSeedCopyAiv()
    {
        ProcessVarlenTailSeedCopyAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessTailAic()
    {
        ProcessVarlenTailAic();
    }

    __aicore__ inline void ProcessTailAiv()
    {
        ProcessVarlenTailAiv();
        ReleaseVectorEvents();
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessPreparedFullHeadBatchArch35(
        const uint64_t *batchB, const uint64_t *batchHv,
        const uint64_t *batchStart, uint16_t taskCount)
    {
        ProcessPreparedFullHeadBatchItemsArch35(
            batchB, batchHv, batchStart, taskCount);
    }

    __attribute__((noinline)) __aicore__ void ProcessPreparedTailSingleArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        // 该单槽 helper 只服务 curT < BT_ 的尾块；完整 64 行块统一进入
        // batch helper，由同一组事件驱动两个物理槽位轮转。
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        Catlass::Arch::Resource<KdaArchTag> resource;
        InitializePostWuPipelineSlot(0);
        PrefetchPostWuPipelineArch35(resource, 0, b, hv, start, curT, false);
        PrefetchPostWuPipelineU(resource, 0, b, hv, start, curT, false);
        ComputePrefetchedPostWuPipelineArch35(resource, 0, b, hv, start, curT);
        FinalizePostWuPipelineEvents(1);
    }

#endif

private:
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessPreparedFullHeadBatchItemsArch35(
        const uint64_t *batchB, const uint64_t *batchHv,
        const uint64_t *batchStart, uint16_t taskCount)
    {
        if (taskCount == 0) {
            return;
        }
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        Catlass::Arch::Resource<KdaArchTag> resource;
        // 每个item对应一个实际HV；headCnt只决定item数量，所有窗口共用
        // 同一套两槽L0C流水和事件协议。
        const uint16_t itemCount = taskCount;
        uint16_t slot = 0;
        uint16_t usedSlotCount = 1;
        uint64_t b = batchB[0];
        uint64_t hv = batchHv[0];
        uint64_t start = batchStart[0];
        InitializePostWuPipelineSlot(slot);
        PrefetchPostWuPipelineArch35(resource, slot, b, hv, start, BT_, false);
        PrefetchPostWuPipelineU(resource, slot, b, hv, start, BT_, false);

        for (uint16_t item = 0; item < itemCount; ++item) {
            const uint16_t nextItem = item + 1;
            if (nextItem < itemCount) {
                const uint16_t nextSlot = slot ^ 1;
                const bool reuseSlot =
                    nextItem >= KDA_POST_PIPELINE_STAGE_COUNT;
                if (!reuseSlot) {
                    InitializePostWuPipelineSlot(nextSlot);
                    ++usedSlotCount;
                }
                PrefetchPostWuPipelineArch35(
                    resource, nextSlot, batchB[nextItem], batchHv[nextItem],
                    batchStart[nextItem], BT_, reuseSlot);
                PrefetchPostWuPipelineU(
                    resource, nextSlot, batchB[nextItem], batchHv[nextItem],
                    batchStart[nextItem], BT_, reuseSlot);
            }

            ComputePrefetchedPostWuPipelineArch35(
                resource, slot, b, hv, start, BT_);
            if (nextItem < itemCount) {
                b = batchB[nextItem];
                hv = batchHv[nextItem];
                start = batchStart[nextItem];
                slot ^= 1;
            }
        }
        FinalizePostWuPipelineEvents(usedSlotCount);
    }
#endif

    __aicore__ inline void AllocVectorEvents()
    {
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte2ToMte3Event_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2Event_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        vectorEventsAllocated_ = true;
    }

    __aicore__ inline void ReleaseVectorEvents()
    {
        if (!vectorEventsAllocated_) {
            return;
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Event_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        vectorEventsAllocated_ = false;
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * H_ + h) * K_ + d;
        }
        return ((b * H_ + h) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t KVOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d, uint64_t dim) const
    {
        return ((b * HV_ + hv) * T_ + t) * dim + d;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * HV_ + hv) * T_ + t;
    }

    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t j) const
    {
        return ((b * HV_ + hv) * T_ + t) * BT_ + j;
    }

    __aicore__ inline uint64_t HOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t d, uint64_t r) const
    {
        return (((b * HV_ + hv) * NT_ + chunkIdx) * K_ + d) * V_ + r;
    }

    __aicore__ inline uint64_t WScratchOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t t, uint64_t d) const
    {
        return (((b * HV_ + hv) * NT_ + chunkIdx) * BT_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t SolveScratchOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                  uint64_t slot) const
    {
        (void)b;
        (void)hv;
        (void)chunkIdx;
        uint64_t matrixElements = BT_ * BT_;
        return solveCoreIdx_ * KDA_SOLVE_SCRATCH_SLOTS * matrixElements + slot * matrixElements;
    }

    __aicore__ inline uint64_t ScoreScratchOffset(uint64_t slot, uint64_t plane, uint64_t t = 0,
                                                  uint64_t d = 0) const
    {
        return (((solveCoreIdx_ * KDA_SCORE_QUEUE_DEPTH + slot) * KDA_SCORE_SCRATCH_PLANES + plane) * BT_ + t) *
                   K_ +
               d;
    }



    __aicore__ inline uint64_t ScoreRefBlockSize() const
    {
        return KDA_SCORE_REF_BC;
    }

    __aicore__ inline uint64_t ScoreRowBlockCount(uint64_t curT, uint64_t rowBegin) const
    {
        uint64_t blockSize = ScoreRefBlockSize();
        uint64_t rowCount = curT - rowBegin;
        if (rowCount > blockSize) {
            rowCount = blockSize;
        }
        return rowCount;
    }

    __aicore__ inline uint64_t ScoreRefToken(uint64_t start, uint64_t curT, uint64_t rowBegin,
                                             uint64_t rowCount) const
    {
        uint64_t ref = rowBegin + rowCount / 2;
        if (ref >= curT) {
            ref = curT - 1;
        }
        return start + ref;
    }

    __aicore__ inline void RunExp2(LocalTensor<float> &tensor, uint32_t count)
    {
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        ClampExpInput(tensor, count);
        Exp(tensor, tensor, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
    }

    __aicore__ inline void ClampExpInput(LocalTensor<float> &tensor, uint32_t count)
    {
        Mins(tensor, tensor, KDA_EXP_INPUT_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, KDA_EXP_INPUT_MIN, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ClampFp32ToOutputType(LocalTensor<float> &tensor, uint32_t count)
    {
        if constexpr (IsSameType<T, half>::value) {
            Mins(tensor, tensor, KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
            Maxs(tensor, tensor, -KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset,
                                        uint64_t count)
    {
        uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowsIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src,
                                      uint64_t offset, uint64_t rows, uint64_t cols,
                                      uint64_t rowStride)
    {
        if (rows == 0 || cols == 0) {
            return;
        }
        if (rowStride == cols) {
            CopyVectorIn(dst, src, offset, rows * cols);
            return;
        }
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(CopyT)),
            static_cast<uint32_t>((rowStride - cols) * sizeof(CopyT)),
            0,
            0};
        DataCopyPadExtParams<CopyT> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src,
                                         uint64_t count)
    {
        uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst[offset], src, static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPad(dst[offset], src, params);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset)
    {
        CopyVectorIn(dst, src, offset, K_);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src)
    {
        CopyVectorOut(dst, offset, src, K_);
    }

    __aicore__ inline LocalTensor<float> VecScratch(uint64_t slot)
    {
        return vecBuf_.Get<float>()[slot * EXP2_UB_ELEMENTS];
    }

    template <typename CopyT>
    __aicore__ inline void LoadAsFloatRow(GlobalTensor<CopyT> &src, uint64_t srcOffset, LocalTensor<float> &dst,
                                          uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            CopyVectorIn(dst, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Adds(dst, dst, 0.0f, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        } else {
            constexpr uint32_t typedOffset = EXP2_UB_ELEMENTS * sizeof(float) / sizeof(CopyT);
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>()[typedOffset];
            CopyVectorIn(rowLocal, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(dst, rowLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename CopyT>
    __aicore__ inline void LoadAsFloatVector(GlobalTensor<CopyT> &src, uint64_t srcOffset,
                                              LocalTensor<float> &dst, LocalTensor<CopyT> &typedScratch,
                                              uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            CopyVectorIn(dst, src, srcOffset, count);
        } else {
            CopyVectorIn(typedScratch, src, srcOffset, count);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if constexpr (!IsSameType<CopyT, float>::value) {
            Cast(dst, typedScratch, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename CopyT>
    __aicore__ inline void StoreFloatRow(GlobalTensor<CopyT> &dst, uint64_t dstOffset, LocalTensor<float> &src,
                                         uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(dst, dstOffset, src, count);
        } else {
            constexpr uint32_t typedOffset = EXP2_UB_ELEMENTS * sizeof(float) / sizeof(CopyT);
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>()[typedOffset];
            Cast(rowLocal, src, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(dst, dstOffset, rowLocal, count);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }





    __aicore__ inline LocalTensor<float> Exp2NegG(uint64_t b, uint64_t hv, uint64_t t)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        LoadAsFloatRow(gk_, KVOffset(b, hv, t, 0, K_), exp2Local, K_);
        Muls(exp2Local, exp2Local, -LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }


    __aicore__ inline uint64_t ScoreVectorMaxRows(uint64_t bytesPerElem) const
    {
        constexpr uint64_t arenaBytes = static_cast<uint64_t>(KDA_VEC_ARENA_ELEMENTS) * sizeof(float);
        uint64_t maxRows = (arenaBytes / bytesPerElem) / K_;
        if (K_ >= 128 && maxRows > 32) {
            maxRows = 32;
        }
        return maxRows;
    }
    __aicore__ inline bool UsePostWuCube(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline bool UsePostWuCubeArch35(uint64_t curT) const
    {
        return curT == 64 && BT_ == 64 && K_ == 128 && V_ == 128;
    }

    __aicore__ inline bool UseFullPostWuPipelineArch35(uint64_t curT) const
    {
        return curT == 64 && BT_ == 64 && K_ == 128 && V_ == 128;
    }

    __aicore__ inline void ComputePostWuCubeFusedArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;

        constexpr uint32_t capacityM = 64;
        constexpr uint32_t n = 128;
        constexpr uint32_t capacityK = 64;
        const uint32_t m = static_cast<uint32_t>(curT);
        const uint32_t k = static_cast<uint32_t>(curT);
        SetLoadDataPaddingValue<T>(static_cast<T>(0));

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(capacityM, capacityK);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(capacityK, n);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(capacityM, n);
        auto tensorA = tla::MakeTensor(
            preparedAqk_[AOffset(b, hv, start, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW = tla::MakeTensor(
            preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB, Catlass::Arch::PositionGM{});
        auto tensorU = tla::MakeTensor(
            propagatedVNew_[KVOffset(b, hv, start, 0, V_)], layoutB, Catlass::Arch::PositionGM{});
        auto tensorWOut = tla::MakeTensor(
            w_[KVOffset(b, hv, start, 0, K_)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorUOut = tla::MakeTensor(
            u_[KVOffset(b, hv, start, 0, V_)], layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto blockU = GetTile(tensorU, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto blockWOut = GetTile(tensorWOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto blockUOut = GetTile(tensorUOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        Catlass::Arch::Resource<KdaArchTag> resource;
        constexpr uint32_t aBytes = capacityM * capacityK * sizeof(ElementA);
        constexpr uint32_t bBytes = capacityK * n * sizeof(ElementB);
        LocalTensor<ElementA> l1A = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l1B0 = resource.l1Buf.template GetBufferByByte<ElementB>(aBytes);
        LocalTensor<ElementB> l1B1 = resource.l1Buf.template GetBufferByByte<ElementB>(aBytes + bBytes);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<float> l0C = resource.l0CBuf.template GetBufferByByte<float>(0);

        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockA)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockW)>;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockWOut)>;
        using TileMmad =
            Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(capacityM, capacityK);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(capacityK, n);
        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(capacityM, capacityK);
        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(capacityK, n);
        auto layoutL0C = tla::MakeLayoutL0C(capacityM, n);
        auto tensorL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(l1B0, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(l1B1, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B0 = GetTile(tensorL1B0, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL1B1 = GetTile(tensorL1B1, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        TileMmad tileMmad;

        copyGmToL1A(tensorL1A, blockA);
        copyGmToL1B(tensorL1B0, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT);
        copyL1ToL0A(tileL0A, tileL1A);
        copyL1ToL0B(tileL0B, tileL1B0);
        SetFlag<HardEvent::MTE1_M>(KDA_POST_EVENT);
        copyGmToL1B(tensorL1B1, blockU);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT_NEXT);
        WaitFlag<HardEvent::MTE1_M>(KDA_POST_EVENT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        SetFlag<HardEvent::M_MTE1>(KDA_POST_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT_NEXT);
        WaitFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        WaitFlag<HardEvent::M_MTE1>(KDA_POST_EVENT);
        copyL0CToDst(blockWOut, tileL0C);
        SetFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);

        copyL1ToL0B(tileL0B, tileL1B1);
        SetFlag<HardEvent::MTE1_M>(KDA_POST_EVENT_NEXT);
        WaitFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);
        WaitFlag<HardEvent::MTE1_M>(KDA_POST_EVENT_NEXT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        WaitFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        copyL0CToDst(blockUOut, tileL0C);
        SetFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);
        WaitFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);
    }

    __aicore__ inline void FinalizePostWuPipelineEvents(uint16_t usedSlotCount)
    {
        for (uint16_t slot = 0; slot < usedSlotCount; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(KDA_POST_EVENT + slot);
            WaitFlag<HardEvent::MTE1_MTE2>(KDA_POST_PIPELINE_U_EVENT + slot);
            WaitFlag<HardEvent::M_MTE1>(KDA_POST_EVENT + slot);
            WaitFlag<HardEvent::FIX_M>(KDA_POST_EVENT + slot);
        }
    }

    __aicore__ inline void InitializePostWuPipelineSlot(uint16_t slot)
    {
        SetFlag<HardEvent::M_MTE1>(KDA_POST_EVENT + slot);
        SetFlag<HardEvent::FIX_M>(KDA_POST_EVENT + slot);
    }

    __aicore__ inline void PrefetchPostWuPipelineArch35(
        Catlass::Arch::Resource<KdaArchTag> &resource, uint16_t slot,
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT, bool reuseSlot)
    {
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, T, LayoutTagA, T, LayoutTagB, T, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        constexpr uint32_t capacityM = 64;
        constexpr uint32_t n = 128;
        constexpr uint32_t capacityK = 64;
        const uint32_t m = static_cast<uint32_t>(curT);
        const uint32_t k = static_cast<uint32_t>(curT);
        auto layoutA = tla::MakeLayout<T, LayoutTagA>(capacityM, capacityK);
        auto layoutB = tla::MakeLayout<T, LayoutTagB>(capacityK, n);
        auto tensorA = tla::MakeTensor(
            preparedAqk_[AOffset(b, hv, start, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW = tla::MakeTensor(
            preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockA)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockW)>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;

        uint32_t slotBase = static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_SLOT_BYTES;
        LocalTensor<T> l1A = resource.l1Buf.template GetBufferByByte<T>(slotBase);
        LocalTensor<T> l1W = resource.l1Buf.template GetBufferByByte<T>(
            slotBase + KDA_POST_PIPELINE_L1_A_BYTES);
        auto layoutL1A = tla::MakeLayout<T, LayoutTagL1A>(capacityM, capacityK);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(capacityK, n);
        auto tensorL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1W = tla::MakeTensor(l1W, layoutL1B, Catlass::Arch::PositionL1{});

        uint16_t pipelineEvent = KDA_POST_EVENT + slot;
        if (reuseSlot) {
            WaitFlag<HardEvent::MTE1_MTE2>(pipelineEvent);
        }
        copyGmToL1A(tensorL1A, blockA);
        copyGmToL1B(tensorL1W, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(pipelineEvent);
    }

    __aicore__ inline void PrefetchPostWuPipelineU(
        Catlass::Arch::Resource<KdaArchTag> &resource, uint16_t slot,
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT, bool reuseStage)
    {
        using LayoutTagB = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, T, Catlass::layout::RowMajor, T, LayoutTagB,
            T, Catlass::layout::RowMajor>;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        constexpr uint32_t capacityK = 64;
        constexpr uint32_t n = 128;
        const uint32_t k = static_cast<uint32_t>(curT);
        auto layoutB = tla::MakeLayout<T, LayoutTagB>(capacityK, n);
        auto tensorU = tla::MakeTensor(
            propagatedVNew_[KVOffset(b, hv, start, 0, V_)],
            layoutB, Catlass::Arch::PositionGM{});
        auto blockU = GetTile(tensorU, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockU)>;
        CopyGmToL1B copyGmToL1B;

        uint32_t uOffset = KDA_POST_PIPELINE_STAGE_COUNT * KDA_POST_PIPELINE_L1_SLOT_BYTES +
                           static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_U_SLOT_BYTES;
        LocalTensor<T> l1U = resource.l1Buf.template GetBufferByByte<T>(uOffset);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(capacityK, n);
        auto tensorL1U = tla::MakeTensor(l1U, layoutL1B, Catlass::Arch::PositionL1{});

        uint16_t pipelineEvent = KDA_POST_PIPELINE_U_EVENT + slot;
        if (reuseStage) {
            WaitFlag<HardEvent::MTE1_MTE2>(pipelineEvent);
        }
        copyGmToL1B(tensorL1U, blockU);
        SetFlag<HardEvent::MTE2_MTE1>(pipelineEvent);
    }

    __aicore__ inline void ComputePrefetchedPostWuPipelineArch35(
        Catlass::Arch::Resource<KdaArchTag> &resource, uint16_t slot,
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, T, LayoutTagA, T, LayoutTagB, T, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, T, LayoutTagL1A>;

        constexpr uint32_t capacityM = 64;
        constexpr uint32_t n = 128;
        constexpr uint32_t packedN = 256;
        constexpr uint32_t capacityK = 64;
        const uint32_t m = static_cast<uint32_t>(curT);
        const uint32_t k = static_cast<uint32_t>(curT);
        auto layoutC = tla::MakeLayout<T, LayoutTagC>(capacityM, n);
        auto tensorWOut = tla::MakeTensor(
            w_[KVOffset(b, hv, start, 0, K_)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorUOut = tla::MakeTensor(
            u_[KVOffset(b, hv, start, 0, V_)], layoutC, Catlass::Arch::PositionGM{});
        auto blockWOut = GetTile(tensorWOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto blockUOut = GetTile(tensorUOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockWOut)>;

        uint32_t l1Base = static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_SLOT_BYTES;
        LocalTensor<T> l1A = resource.l1Buf.template GetBufferByByte<T>(l1Base);
        LocalTensor<T> l1W = resource.l1Buf.template GetBufferByByte<T>(
            l1Base + KDA_POST_PIPELINE_L1_A_BYTES);
        uint32_t uOffset = KDA_POST_PIPELINE_STAGE_COUNT * KDA_POST_PIPELINE_L1_SLOT_BYTES +
                           static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_U_SLOT_BYTES;
        LocalTensor<T> l1U = resource.l1Buf.template GetBufferByByte<T>(uOffset);
        LocalTensor<T> l0A = resource.l0ABuf.template GetBufferByByte<T>(
            static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L0_A_SLOT_BYTES);
        LocalTensor<T> l0B = resource.l0BBuf.template GetBufferByByte<T>(
            static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L0_B_SLOT_BYTES);
        LocalTensor<float> l0C = resource.l0CBuf.template GetBufferByByte<float>(
            static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L0_C_SLOT_BYTES);

        auto layoutL1A = tla::MakeLayout<T, LayoutTagL1A>(capacityM, capacityK);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(capacityK, n);
        auto layoutL0A = tla::MakeLayout<T, LayoutTagL0A>(capacityM, capacityK);
        auto layoutL0B = tla::MakeLayout<T, LayoutTagL0B>(capacityK, packedN);
        auto layoutL0C = tla::MakeLayoutL0C(capacityM, packedN);
        auto tensorL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1W = tla::MakeTensor(l1W, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL1U = tla::MakeTensor(l1U, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1W = GetTile(tensorL1W, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL1U = GetTile(tensorL1U, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0BW = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0BU = GetTile(tensorL0B, tla::MakeCoord(0, n), tla::MakeShape(k, n));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, packedN));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, packedN));
        auto tileL0CW = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto tileL0CU = GetTile(tensorL0C, tla::MakeCoord(0, n), tla::MakeShape(m, n));

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        TileMmad tileMmad;

        uint16_t pipelineEvent = KDA_POST_EVENT + slot;
        uint16_t uPipelineEvent = KDA_POST_PIPELINE_U_EVENT + slot;
        WaitFlag<HardEvent::MTE2_MTE1>(pipelineEvent);
        WaitFlag<HardEvent::MTE2_MTE1>(uPipelineEvent);
        WaitFlag<HardEvent::M_MTE1>(pipelineEvent);
        copyL1ToL0A(tileL0A, tileL1A);
        copyL1ToL0B(tileL0BW, tileL1W);
        copyL1ToL0B(tileL0BU, tileL1U);
        SetFlag<HardEvent::MTE1_M>(pipelineEvent);
        SetFlag<HardEvent::MTE1_MTE2>(pipelineEvent);
        SetFlag<HardEvent::MTE1_MTE2>(uPipelineEvent);
        WaitFlag<HardEvent::MTE1_M>(pipelineEvent);
        WaitFlag<HardEvent::FIX_M>(pipelineEvent);
        tileMmad(tileL0C, tileL0A, tileL0B, m, packedN, k, true, 0);
        SetFlag<HardEvent::M_MTE1>(pipelineEvent);
        SetFlag<HardEvent::M_FIX>(pipelineEvent);
        WaitFlag<HardEvent::M_FIX>(pipelineEvent);
        copyL0CToDst(blockWOut, tileL0CW);
        copyL0CToDst(blockUOut, tileL0CU);
        SetFlag<HardEvent::FIX_M>(pipelineEvent);
    }
#endif

    __aicore__ inline void ComputePostWuCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (UsePostWuCubeArch35(curT)) {
            ComputePostWuCubeFusedArch35(b, hv, start, curT);
            return;
        }
#endif
        using ElementA = AKK_T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        using WTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, T, LayoutTagC>;
#else
        using WTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, float, LayoutTagC>;
#endif
        using UTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, OUT_T, LayoutTagC>;
        using PostL1TileShape128 = tla::Shape<KdaInt128, KdaInt128, tla::_256>;
        using PostL0TileShape128 = tla::Shape<KdaInt128, KdaInt128, KdaInt128>;
        using PostL1TileShape256 = tla::Shape<KdaInt128, tla::_256, tla::_256>;
        using PostL0TileShape256 = tla::Shape<KdaInt128, tla::_256, KdaInt64>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using WBlockMmad128 = Common::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                   PostL0TileShape128,
                                                   ElementA, ElementB, T, void, WTileCopy>;
        using WBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy, PostL1TileShape256,
                                                   PostL0TileShape256,
                                                   ElementA, ElementB, T, void, WTileCopy>;
#else
        using WBlockMmad128 = Common::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                   PostL0TileShape128,
                                                   ElementA, ElementB, float, void, WTileCopy>;
        using WBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy, PostL1TileShape256,
                                                   PostL0TileShape256,
                                                   ElementA, ElementB, float, void, WTileCopy>;
#endif
        using UBlockMmad128 = Common::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                                  PostL0TileShape128,
                                                                  ElementA, ElementB, OUT_T, void, UTileCopy>;
        using UBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy, PostL1TileShape256,
                                                                  PostL0TileShape256,
                                                                  ElementA, ElementB, OUT_T, void, UTileCopy>;
        LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(BT_, BT_);
        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto tensorA = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start, 0)], layoutA,
                                       Catlass::Arch::PositionGM{});

        {
            LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(BT_, K_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            LayoutTagC tagC = LayoutTagC::template MakeLayout<T>(BT_, K_);
#else
            LayoutTagC tagC = LayoutTagC::template MakeLayout<float>(BT_, K_);
#endif
            auto layoutB = tla::MakeLayoutFromTag(tagB);
            auto layoutC = tla::MakeLayoutFromTag(tagC);
            Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(K_),
                                     static_cast<uint32_t>(curT)};
            auto tensorB = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB,
                                           Catlass::Arch::PositionGM{});
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            auto tensorC = tla::MakeTensor(w_[KVOffset(b, hv, start, 0, K_)], layoutC,
                                           Catlass::Arch::PositionGM{});
#else
            auto tensorC = tla::MakeTensor(h_[WScratchOffset(b, hv, chunkIdx, 0, 0)], layoutC,
                                            Catlass::Arch::PositionGM{});
#endif
            auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
            auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
            auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
            Catlass::Arch::Resource<KdaArchTag> wResource;
            if (K_ <= 128) {
                WBlockMmad128 wBlockMmad(wResource);
                wBlockMmad(blockA, blockB, blockC, shape);
            } else {
                WBlockMmad256 wBlockMmad(wResource);
                wBlockMmad(blockA, blockB, blockC, shape);
            }
            // 离开当前作用域时由 BlockMmad 排空 L0C credit，确保 W 已写回 GM。
        }

        {
            LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(BT_, V_);
            LayoutTagC tagC = LayoutTagC::template MakeLayout<OUT_T>(BT_, V_);
            auto layoutB = tla::MakeLayoutFromTag(tagB);
            auto layoutC = tla::MakeLayoutFromTag(tagC);
            Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(V_),
                                     static_cast<uint32_t>(curT)};
            auto tensorB = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, 0, V_)], layoutB,
                                           Catlass::Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(u_[KVOffset(b, hv, start, 0, V_)], layoutC,
                                           Catlass::Arch::PositionGM{});
            auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
            auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
            auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
            Catlass::Arch::Resource<KdaArchTag> uResource;
            if (V_ <= 128) {
                UBlockMmad128 uBlockMmad(uResource);
                uBlockMmad(blockA, blockB, blockC, shape);
            } else {
                UBlockMmad256 uBlockMmad(uResource);
                uBlockMmad(blockA, blockB, blockC, shape);
            }
            // UBlockMmad 已在分支作用域结束时排空 L0C credit，确保 U 已写回 GM。
        }

    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline bool UseTypicalPostWuGate(uint64_t curT) const
    {
        return curT == 64 && BT_ == 64 && K_ == 128 && V_ == 128;
    }

    __aicore__ inline LocalTensor<T> FallbackKResidentArch35()
    {
        return vecBuf_.Get<T>()[
            KDA_FALLBACK_K_RESIDENT_BYTE_OFFSET / sizeof(T)];
    }

    __aicore__ inline bool FallbackKResidentContainsArch35(
        uint64_t b, uint64_t h, uint64_t token, uint64_t rows) const
    {
        return fallbackKResidentEnabled_ && b == fallbackKResidentB_ &&
               h == fallbackKResidentH_ &&
               token >= fallbackKResidentTokenBegin_ &&
               token + rows <= fallbackKResidentTokenEnd_;
    }

    __aicore__ inline void BeginFallbackKResidentGroupArch35(
        uint64_t b, uint64_t h, uint64_t start, uint64_t curT,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        fallbackKResidentEnabled_ = false;
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if (fallbackKResidentHasVectorReader_) {
                // 新qHead覆盖resident前，先闭合上一组V读到MTE2写的WAR依赖。
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
                fallbackKResidentHasVectorReader_ = false;
            }
            if (BT_ != 64 || K_ != 128 || V_ != 128 || curT > BT_ ||
                subBlockNum != 2 || subBlockIdx >= subBlockNum) {
                return;
            }
            static_assert(
                4 * KDA_FALLBACK_K_RESIDENT_ROWS * 128 * sizeof(float) <=
                    KDA_FALLBACK_K_RESIDENT_BYTE_OFFSET,
                "arch35 fallback fp32 planes overlap K resident");
            static_assert(
                20480 * sizeof(float) +
                        KDA_FALLBACK_K_RESIDENT_ROWS * 128 * sizeof(T) +
                        KDA_FALLBACK_K_RESIDENT_ROWS * 128 * sizeof(GK_T) <=
                    KDA_FALLBACK_K_RESIDENT_BYTE_OFFSET,
                "arch35 fallback typed scratch overlaps K resident");
            static_assert(
                KDA_TYPICAL_GATE_PIPELINE_STAGES *
                            (KDA_TYPICAL_GATE_PIPELINE_ROWS * 128 *
                                 (sizeof(T) + sizeof(float)) +
                             128 * sizeof(float)) +
                        KDA_TYPICAL_GATE_RESIDENT_ROWS * 128 * sizeof(T) <=
                    KDA_FALLBACK_K_RESIDENT_BYTE_OFFSET,
                "arch35 typical pipeline overlaps fallback K resident");
            static_assert(
                KDA_FALLBACK_K_RESIDENT_BYTE_OFFSET +
                        KDA_FALLBACK_K_RESIDENT_ROWS * 128 * sizeof(T) <=
                    KDA_SELECT_AQK_MASK_BYTE_OFFSET,
                "arch35 fallback K resident overlaps reserved mask arena");
            static_assert(
                KDA_FALLBACK_K_RESIDENT_BYTE_OFFSET +
                        KDA_FALLBACK_K_RESIDENT_ROWS * 128 * sizeof(T) <=
                    KDA_VEC_ARENA_ELEMENTS * sizeof(float),
                "arch35 fallback K resident exceeds vector arena");

            const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
            const uint64_t rowEnd =
                (curT * (subBlockIdx + 1)) / subBlockNum;
            if (rowBegin >= rowEnd ||
                rowEnd - rowBegin > KDA_FALLBACK_K_RESIDENT_ROWS) {
                return;
            }
            fallbackKResidentB_ = b;
            fallbackKResidentH_ = h;
            fallbackKResidentTokenBegin_ = start + rowBegin;
            fallbackKResidentTokenEnd_ = start + rowEnd;
            LocalTensor<T> residentK = FallbackKResidentArch35();
            CopyRowsIn(
                residentK, k_,
                QOffset(b, h, fallbackKResidentTokenBegin_, 0),
                rowEnd - rowBegin, K_,
                inputSequenceMajor_ ? H_ * K_ : K_);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            fallbackKResidentEnabled_ = true;
        }
    }

    __aicore__ inline bool StageFallbackKFromResidentArch35(
        LocalTensor<T> dst, uint64_t b, uint64_t h,
        uint64_t token, uint64_t rows)
    {
        if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if (FallbackKResidentContainsArch35(b, h, token, rows)) {
                const uint64_t residentOffset =
                    (token - fallbackKResidentTokenBegin_) * K_;
                Adds(dst, FallbackKResidentArch35()[residentOffset], 0.0f,
                     static_cast<uint32_t>(rows * K_));
                PipeBarrier<PIPE_V>();
                fallbackKResidentHasVectorReader_ = true;
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint64_t TypicalGateStageElems() const
    {
        return static_cast<uint64_t>(KDA_TYPICAL_GATE_TILE_ROWS) * 128;
    }

    __aicore__ inline uint64_t TypicalGateStageBytes() const
    {
        return TypicalGateStageElems() * (sizeof(T) + sizeof(GK_T));
    }

    __aicore__ inline LocalTensor<T> TypicalGateK(uint64_t slot)
    {
        return gateWritebackBuf_.Get<T>()[slot * TypicalGateStageBytes() / sizeof(T)];
    }

    __aicore__ inline LocalTensor<GK_T> TypicalGateG(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGateStageBytes() + TypicalGateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<GK_T>()[byteOffset / sizeof(GK_T)];
    }

    __aicore__ inline void PrefetchTypicalKg(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                             uint64_t token, uint64_t rows)
    {
        uint64_t elems = rows * K_;
        LocalTensor<T> kStage = TypicalGateK(slot);
        LocalTensor<GK_T> gateStage = TypicalGateG(slot);
        if (!FallbackKResidentContainsArch35(b, h, token, rows)) {
            CopyRowsIn(kStage, k_, QOffset(b, h, token, 0), rows, K_,
                       inputSequenceMajor_ ? H_ * K_ : K_);
        }
        DataCopy(gateStage, gk_[KVOffset(b, hv, token, 0, K_)], static_cast<uint32_t>(elems));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void ComputeTypicalKg(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                            uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return;
        }

        LocalTensor<float> gateLast = exp2Buf_.Get<float>();
        LoadAsFloatRow(gk_, KVOffset(b, hv, start + curT - 1, 0, K_), gateLast, K_);

        uint64_t slot = 0;
        uint64_t firstRows = rowEnd - rowBegin;
        if (firstRows > KDA_TYPICAL_GATE_TILE_ROWS) {
            firstRows = KDA_TYPICAL_GATE_TILE_ROWS;
        }
        PrefetchTypicalKg(slot, b, h, hv, start + rowBegin, firstRows);

        bool outputPending = false;
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += KDA_TYPICAL_GATE_TILE_ROWS) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > KDA_TYPICAL_GATE_TILE_ROWS) {
                tileRows = KDA_TYPICAL_GATE_TILE_ROWS;
            }
            uint64_t elems = tileRows * K_;
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            StageFallbackKFromResidentArch35(
                TypicalGateK(slot), b, h, start + tileRow, tileRows);

            if (outputPending) {
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            }
            uint64_t nextTileRow = tileRow + KDA_TYPICAL_GATE_TILE_ROWS;
            if (nextTileRow < rowEnd) {
                uint64_t nextRows = rowEnd - nextTileRow;
                if (nextRows > KDA_TYPICAL_GATE_TILE_ROWS) {
                    nextRows = KDA_TYPICAL_GATE_TILE_ROWS;
                }
                PrefetchTypicalKg(slot ^ 1, b, h, hv, start + nextTileRow, nextRows);
            }

            LocalTensor<T> kAndKg = TypicalGateK(slot);
            LocalTensor<GK_T> gateStage = TypicalGateG(slot);
            ComputePostKdaKgRegbase<T, GK_T>(
                (__ubuf__ T *)reinterpret_cast<uint64_t>(kAndKg.GetPhyAddr()),
                (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateStage.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(gateLast.GetPhyAddr()),
                static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_));

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(kg_[KVOffset(b, hv, start + tileRow, 0, K_)], kAndKg,
                     static_cast<uint32_t>(elems));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            outputPending = true;
            slot ^= 1;
        }
        if (outputPending) {
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
    }

    __aicore__ inline uint64_t TypicalGatePipelineStageElems() const
    {
        return static_cast<uint64_t>(KDA_TYPICAL_GATE_PIPELINE_ROWS) * 128;
    }

    __aicore__ inline uint64_t TypicalGatePipelineStageBytes() const
    {
        return TypicalGatePipelineStageElems() * (sizeof(T) + sizeof(float)) +
               128 * sizeof(float);
    }

    __aicore__ inline LocalTensor<T> TypicalGatePipelineK(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGatePipelineStageBytes();
        return vecBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<float> TypicalGatePipelineG(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGatePipelineStageBytes() +
                              TypicalGatePipelineStageElems() * sizeof(T);
        return vecBuf_.Get<float>()[byteOffset / sizeof(float)];
    }

    __aicore__ inline LocalTensor<float> TypicalGatePipelineRef(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGatePipelineStageBytes() +
                              TypicalGatePipelineStageElems() * (sizeof(T) + sizeof(float));
        return vecBuf_.Get<float>()[byteOffset / sizeof(float)];
    }

    __aicore__ inline LocalTensor<T> TypicalGatePipelineResidentK()
    {
        const uint64_t byteOffset =
            KDA_TYPICAL_GATE_PIPELINE_STAGES * TypicalGatePipelineStageBytes();
        return vecBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline bool CanPipelineTypicalKg(
        uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum) const
    {
        if constexpr (!IsSameType<GK_T, float>::value) {
            return false;
        }
        if (!UseTypicalPostWuGate(curT) || subBlockNum == 0) {
            return false;
        }
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        return rowBegin < rowEnd && rowEnd - rowBegin <= KDA_TYPICAL_GATE_PIPELINE_ROWS;
    }

    __aicore__ inline void PrefetchTypicalKgPipeline(
        uint64_t slot, uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
        uint64_t curT, uint64_t rowBegin, uint64_t rowEnd,
        bool useResidentK, bool reloadResidentK)
    {
        if constexpr (IsSameType<GK_T, float>::value) {
            uint64_t elems = (rowEnd - rowBegin) * K_;
            LocalTensor<T> kStage = TypicalGatePipelineK(slot);
            LocalTensor<float> gateStage = TypicalGatePipelineG(slot);
            LocalTensor<float> refStage = TypicalGatePipelineRef(slot);
            if (useResidentK) {
                if (reloadResidentK) {
                    LocalTensor<T> residentK = TypicalGatePipelineResidentK();
                    CopyRowsIn(
                        residentK, k_,
                        QOffset(b, h, start + rowBegin, 0),
                        rowEnd - rowBegin, K_,
                        inputSequenceMajor_ ? H_ * K_ : K_);
                }
            } else {
                CopyRowsIn(kStage, k_,
                           QOffset(b, h, start + rowBegin, 0),
                           rowEnd - rowBegin, K_,
                           inputSequenceMajor_ ? H_ * K_ : K_);
            }
            DataCopy(gateStage, gk_[KVOffset(b, hv, start + rowBegin, 0, K_)],
                     static_cast<uint32_t>(elems));
            DataCopy(refStage, gk_[KVOffset(b, hv, start + curT - 1, 0, K_)],
                     static_cast<uint32_t>(K_));
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
    }

    __aicore__ inline void StageTypicalKgResidentK(
        uint64_t slot, uint64_t rows)
    {
        const uint32_t elems = static_cast<uint32_t>(rows * K_);
        Adds(TypicalGatePipelineK(slot), TypicalGatePipelineResidentK(),
             0.0f, elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeTypicalKgPipelineRegs(
        uint64_t slot, uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t rows = rowEnd - rowBegin;
        LocalTensor<T> kAndKg = TypicalGatePipelineK(slot);
        LocalTensor<float> gateStage = TypicalGatePipelineG(slot);
        LocalTensor<float> refStage = TypicalGatePipelineRef(slot);
        ComputePostKdaKgRegbase<T, float>(
            (__ubuf__ T *)reinterpret_cast<uint64_t>(kAndKg.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gateStage.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(refStage.GetPhyAddr()),
            static_cast<uint16_t>(rows), static_cast<uint16_t>(K_));
    }

    __aicore__ inline void StoreTypicalKgPipeline(
        uint64_t slot, uint64_t b, uint64_t hv, uint64_t start,
        uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t elems = (rowEnd - rowBegin) * K_;
        LocalTensor<T> kAndKg = TypicalGatePipelineK(slot);
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(kg_[KVOffset(b, hv, start + rowBegin, 0, K_)], kAndKg,
                 static_cast<uint32_t>(elems));
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
    }
#endif

    __aicore__ inline void CopyScratchWAndFinalizeKg(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                                     uint64_t start, uint64_t curT, uint64_t subBlockIdx,
                                                     uint64_t subBlockNum)
    {
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        constexpr uint64_t kgFp32Planes = 4;
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return;
        }
        uint64_t maxRows = (typedOffsetFloats / kgFp32Planes) / K_;
        if (maxRows > 32) {
            maxRows = 32;
        }
        if (maxRows == 0) {
            return;
        }

        uint64_t last = start + curT - 1;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> gateLast = exp2Buf_.Get<float>();
        LocalTensor<T> typedLocal = vecBuf_.Get<T>()[typedOffset];
        LoadAsFloatRow(gk_, KVOffset(b, hv, last, 0, K_), gateLast, K_);

        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxRows) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxRows) {
                tileRows = maxRows;
            }
            uint64_t elemCount = tileRows * K_;
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            uint64_t scratchBase = WScratchOffset(b, hv, chunkIdx, tileRow, 0);
#else
            (void)chunkIdx;
#endif
            uint64_t token = start + tileRow;

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            DataCopy(arena, h_[scratchBase], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(typedLocal, arena, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(w_[KVOffset(b, hv, token, 0, K_)], typedLocal, static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
#endif

            LocalTensor<float> kLocal = arena;
            LocalTensor<float> gLocal = arena[elemCount];
            LocalTensor<float> expLocal = arena[2 * elemCount];
            LocalTensor<float> outLocal = arena[3 * elemCount];
            const uint64_t gateOffsetBytes = (typedOffset + elemCount) * sizeof(T);
            LocalTensor<GK_T> gateTyped = vecBuf_.Get<GK_T>()[
                (gateOffsetBytes + sizeof(GK_T) - 1) / sizeof(GK_T)];
            bool stagedResidentK = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            stagedResidentK = StageFallbackKFromResidentArch35(
                typedLocal, b, h, token, tileRows);
#endif
            if (!stagedResidentK) {
                CopyRowsIn(typedLocal, k_, QOffset(b, h, token, 0),
                           tileRows, K_,
                           inputSequenceMajor_ ? H_ * K_ : K_);
            }
            LoadAsFloatVector(gk_, KVOffset(b, hv, token, 0, K_), gLocal, gateTyped, elemCount);
            Cast(kLocal, typedLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();

            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expLocal[row * K_], gateLast, gLocal[row * K_], static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expLocal, expLocal, LN2, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            ClampExpInput(expLocal, static_cast<uint32_t>(elemCount));
            Exp(expLocal, expLocal, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            Mul(outLocal, kLocal, expLocal, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outLocal, static_cast<uint32_t>(elemCount));
            Cast(typedLocal, outLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(kg_, KVOffset(b, hv, token, 0, K_), typedLocal, elemCount);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
        if (rowEnd == curT) {
            bool stagedResidentK = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            stagedResidentK = StageFallbackKFromResidentArch35(
                typedLocal, b, h, last, 1);
#endif
            if (stagedResidentK) {
                SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            } else {
                CopyVectorIn(typedLocal, k_, QOffset(b, h, last, 0), K_);
                SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
                WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
            }
            CopyVectorOut(kg_, KVOffset(b, hv, last, 0, K_), typedLocal, K_);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void ComputeTailWuRow(GlobalTensor<SrcTensor> &src, GlobalTensor<DstTensor> &dst,
                                            uint64_t akkBase, uint64_t srcBase, uint64_t dstBase, uint64_t curT,
                                            uint64_t dim)
    {
        LocalTensor<float> acc = vecBuf_.Get<float>();
        LocalTensor<float> value = vecBuf_.Get<float>()[512];
        LocalTensor<SrcTensor> typed = vecBuf_.Get<SrcTensor>()[4096];
        LocalTensor<T> coefficientTyped = exp2Buf_.Get<T>();
        LocalTensor<float> coefficients = exp2Buf_.Get<float>()[128];
        CopyVectorIn(coefficientTyped, preparedAqk_, akkBase, curT);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        Cast(coefficients, coefficientTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(curT));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        for (uint64_t j = 0; j < curT; ++j) {
            LoadAsFloatVector(src, srcBase + j * dim, value, typed, dim);
            float coefficient = coefficients.GetValue(j);
            SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            Muls(value, value, coefficient, static_cast<uint32_t>(dim));
            PipeBarrier<PIPE_V>();
            if (j == 0) {
                Adds(acc, value, 0.0f, static_cast<uint32_t>(dim));
            } else {
                Add(acc, acc, value, static_cast<uint32_t>(dim));
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        }
        SetFlag<HardEvent::S_MTE2>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_MTE2>(EXP2_EVENT_ID);
        ClampFp32ToOutputType(acc, static_cast<uint32_t>(dim));
        StoreFloatRow(dst, dstBase, acc, dim);
    }

    __aicore__ inline void ComputeTailWuVector(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                               uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        for (uint64_t row = rowBegin; row < rowEnd; ++row) {
            ComputeTailWuRow(
                preparedQG_, w_, AOffset(b, hv, start + row, 0), KVOffset(b, hv, start, 0, K_),
                KVOffset(b, hv, start + row, 0, K_), curT, K_);
            ComputeTailWuRow(
                propagatedVNew_, u_, AOffset(b, hv, start + row, 0), KVOffset(b, hv, start, 0, V_),
                KVOffset(b, hv, start + row, 0, V_), curT, V_);
        }
    }

    __aicore__ inline void CopyTailSeedRows(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        const uint64_t rows = rowEnd - rowBegin;
        if (rows == 0) {
            return;
        }
        LocalTensor<T> typed = vecBuf_.Get<T>();
        const uint64_t wOffset = KVOffset(b, hv, start + rowBegin, 0, K_);
        CopyVectorIn(typed, preparedQG_, wOffset, rows * K_);
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        CopyVectorOut(w_, wOffset, typed, rows * K_);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);

        const uint64_t vOffset = KVOffset(b, hv, start + rowBegin, 0, V_);
        CopyVectorIn(typed, propagatedVNew_, vOffset, rows * V_);
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        CopyVectorOut(u_, vOffset, typed, rows * V_);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
    }

#endif
    struct OwnedChunkDesc {
        uint64_t seq = 0;
        uint64_t b = 0;
        uint64_t chunkIdx = 0;
        uint64_t start = 0;
        uint64_t end = 0;
    };

    struct FullChunkIterator {
        uint64_t sequence = 0;
        uint64_t localChunk = 0;
        uint64_t sequenceStart = 0;
        uint64_t fullChunkCount = 0;
        bool sequenceLoaded = false;
    };

    struct GroupedFullTaskIterator {
        FullChunkIterator chunks{};
        OwnedChunkDesc chunk{};
        uint64_t loadedChunkOrdinal = 0;
        bool chunkLoaded = false;
    };

    struct GroupedTailTaskIterator {
        OwnedChunkDesc chunk{};
        uint64_t loadedChunkOrdinal = 0;
        bool chunkLoaded = false;
    };

    __aicore__ inline bool LoadOwnedFullChunk(
        const KdaForward::CompactSequencePlanView &plan,
        FullChunkIterator &iterator, OwnedChunkDesc &desc)
    {
        while (iterator.sequence < plan.SequenceCount()) {
            if (!iterator.sequenceLoaded) {
                uint64_t sequenceEnd = T_;
                iterator.sequenceStart = 0;
                if (isVarLen_) {
                    iterator.sequenceStart = static_cast<uint64_t>(
                        cuSeqlensAddr_[iterator.sequence]);
                    sequenceEnd = static_cast<uint64_t>(
                        cuSeqlensAddr_[iterator.sequence + 1]);
                }
                iterator.fullChunkCount =
                    (sequenceEnd - iterator.sequenceStart) / BT_;
                iterator.sequenceLoaded = true;
            }
            if (iterator.localChunk < iterator.fullChunkCount) {
                desc.seq = iterator.sequence;
                desc.b = isVarLen_ ? 0 : iterator.sequence;
                desc.chunkIdx = isVarLen_
                    ? plan.SequenceChunkOffset(
                          static_cast<uint32_t>(iterator.sequence)) +
                          iterator.localChunk
                    : iterator.localChunk;
                desc.start = iterator.sequenceStart + iterator.localChunk * BT_;
                desc.end = desc.start + BT_;
                ++iterator.localChunk;
                if (iterator.localChunk == iterator.fullChunkCount) {
                    ++iterator.sequence;
                    iterator.localChunk = 0;
                    iterator.sequenceLoaded = false;
                }
                return true;
            }
            ++iterator.sequence;
            iterator.localChunk = 0;
            iterator.sequenceLoaded = false;
        }
        return false;
    }

    __aicore__ inline bool LoadOwnedTailChunk(
        const KdaForward::CompactSequencePlanView &plan, uint64_t tailOrdinal,
        OwnedChunkDesc &desc)
    {
        const uint64_t sequence = plan.TailedSequenceId(
            static_cast<uint32_t>(tailOrdinal));
        if (sequence >= plan.SequenceCount()) {
            return false;
        }
        uint64_t sequenceStart = 0;
        uint64_t sequenceEnd = T_;
        if (isVarLen_) {
            sequenceStart = static_cast<uint64_t>(cuSeqlensAddr_[sequence]);
            sequenceEnd = static_cast<uint64_t>(cuSeqlensAddr_[sequence + 1]);
        }
        const uint64_t fullChunks = (sequenceEnd - sequenceStart) / BT_;
        desc.seq = sequence;
        desc.b = isVarLen_ ? 0 : sequence;
        desc.chunkIdx = isVarLen_
            ? plan.SequenceChunkOffset(static_cast<uint32_t>(sequence)) +
                  fullChunks
            : fullChunks;
        desc.start = sequenceStart + fullChunks * BT_;
        desc.end = sequenceEnd;
        return desc.start < desc.end;
    }

    __aicore__ inline bool LoadGroupedFullTask(
        const KdaForward::CompactSequencePlanView &plan, uint64_t task,
        GroupedFullTaskIterator &iterator, OwnedChunkDesc &chunk,
        uint64_t &headBegin, uint64_t &headEnd)
    {
        uint32_t chunkOrdinal = 0;
        uint32_t begin = 0;
        uint32_t end = 0;
        if (!plan.DecodeChunkHeadGroupTask(
                static_cast<uint32_t>(task), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_), chunkOrdinal, begin, end)) {
            return false;
        }
        if (!iterator.chunkLoaded ||
            iterator.loadedChunkOrdinal != chunkOrdinal) {
            if (!LoadOwnedFullChunk(plan, iterator.chunks, iterator.chunk)) {
                return false;
            }
            iterator.loadedChunkOrdinal = chunkOrdinal;
            iterator.chunkLoaded = true;
        }
        chunk = iterator.chunk;
        headBegin = begin;
        headEnd = end;
        return true;
    }

    __aicore__ inline bool LoadGroupedTailTask(
        const KdaForward::CompactSequencePlanView &plan, uint64_t task,
        GroupedTailTaskIterator &iterator, OwnedChunkDesc &chunk,
        uint64_t &headBegin, uint64_t &headEnd)
    {
        uint32_t chunkOrdinal = 0;
        uint32_t begin = 0;
        uint32_t end = 0;
        if (!plan.DecodeChunkHeadGroupTask(
                static_cast<uint32_t>(task), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_), chunkOrdinal, begin, end)) {
            return false;
        }
        if (!iterator.chunkLoaded ||
            iterator.loadedChunkOrdinal != chunkOrdinal) {
            if (!LoadOwnedTailChunk(plan, chunkOrdinal, iterator.chunk)) {
                return false;
            }
            iterator.loadedChunkOrdinal = chunkOrdinal;
            iterator.chunkLoaded = true;
        }
        chunk = iterator.chunk;
        headBegin = begin;
        headEnd = end;
        return true;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessTypicalFullPostAivHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t headBase,
        uint64_t subBlockIdx, uint64_t subBlockNum, uint32_t headCnt)
    {
        const uint64_t rowBegin = (BT_ * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (BT_ * (subBlockIdx + 1)) / subBlockNum;
        const bool useResidentK =
            sizeof(T) == sizeof(uint16_t) && BT_ == 64 && K_ == 128 &&
            V_ == 128 && subBlockNum == 2 &&
            rowEnd - rowBegin <= KDA_TYPICAL_GATE_RESIDENT_ROWS;
        static_assert(
            sizeof(T) != sizeof(uint16_t) ||
                KDA_TYPICAL_GATE_PIPELINE_STAGES *
                        (KDA_TYPICAL_GATE_PIPELINE_ROWS * 128 *
                             (sizeof(T) + sizeof(float)) +
                         128 * sizeof(float)) +
                        KDA_TYPICAL_GATE_RESIDENT_ROWS * 128 * sizeof(T) <=
                    KDA_VEC_ARENA_ELEMENTS * sizeof(float),
            "arch35 PostWU K resident exceeds vector arena");
        uint16_t slot = 0;
        bool outputPending = false;
        uint64_t hv = headBase;
        uint64_t h = hv / (HV_ / H_);
        PrefetchTypicalKgPipeline(
            slot, chunk.b, h, hv, chunk.start, BT_, rowBegin, rowEnd,
            useResidentK, useResidentK);
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            if (useResidentK) {
                StageTypicalKgResidentK(slot, rowEnd - rowBegin);
            }
            if (lane + 1 < headCnt) {
                const uint64_t nextHv = headBase + lane + 1;
                const uint64_t nextH = nextHv / (HV_ / H_);
                const uint16_t nextSlot =
                    (slot + 1) % KDA_TYPICAL_GATE_PIPELINE_STAGES;
                const bool reloadResidentK = useResidentK && nextH != h;
                if (reloadResidentK) {
                    // 当前V已把resident复制到本轮stage；重载下一个qHead前
                    // 闭合V读到MTE2写的WAR依赖。
                    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
                }
                PrefetchTypicalKgPipeline(
                    nextSlot, chunk.b, nextH, nextHv, chunk.start, BT_,
                    rowBegin, rowEnd, useResidentK, reloadResidentK);
            }
            ComputeTypicalKgPipelineRegs(slot, rowBegin, rowEnd);
            if (outputPending) {
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            }
            hv = headBase + lane;
            StoreTypicalKgPipeline(
                slot, chunk.b, hv, chunk.start, rowBegin, rowEnd);
            outputPending = true;
            slot = (slot + 1) % KDA_TYPICAL_GATE_PIPELINE_STAGES;
            if (lane + 1 < headCnt) {
                h = (headBase + lane + 1) / (HV_ / H_);
            }
        }
        if (outputPending) {
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
    }

    __aicore__ inline void ProcessTypicalFullPostAicHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t headBase, uint32_t headCnt)
    {
        static_assert(sizeof(T) == sizeof(uint16_t),
                      "arch35 PostWU pipeline is specialized for fp16/bf16 inputs");
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        Catlass::Arch::Resource<KdaArchTag> resource;
        uint16_t slot = 0;
        uint16_t usedSlotCount = 1;
        InitializePostWuPipelineSlot(slot);
        PrefetchPostWuPipelineArch35(
            resource, slot, chunk.b, headBase, chunk.start, BT_, false);
        PrefetchPostWuPipelineU(
            resource, slot, chunk.b, headBase, chunk.start, BT_, false);
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            if (lane + 1 < headCnt) {
                const uint16_t nextSlot = slot ^ 1;
                const bool reuseSlot =
                    lane + 1 >= KDA_POST_PIPELINE_STAGE_COUNT;
                if (!reuseSlot) {
                    InitializePostWuPipelineSlot(nextSlot);
                    ++usedSlotCount;
                }
                const uint64_t nextHv = headBase + lane + 1;
                PrefetchPostWuPipelineArch35(
                    resource, nextSlot, chunk.b, nextHv, chunk.start,
                    BT_, reuseSlot);
                PrefetchPostWuPipelineU(
                    resource, nextSlot, chunk.b, nextHv, chunk.start,
                    BT_, reuseSlot);
            }
            ComputePrefetchedPostWuPipelineArch35(
                resource, slot, chunk.b, headBase + lane, chunk.start, BT_);
            slot ^= 1;
        }
        FinalizePostWuPipelineEvents(usedSlotCount);
    }
#endif

    template <bool IS_AIC, bool IS_TAIL>
    __attribute__((noinline)) __aicore__ void ProcessCompactPostHead(
        const OwnedChunkDesc &chunk, uint64_t hv,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if constexpr (IS_AIC) {
            if constexpr (!IS_TAIL && IsSameType<AKK_T, T>::value) {
                if (UsePostWuCube(BT_)) {
                    ComputePostWuCube(
                        chunk.b, hv, chunk.chunkIdx, chunk.start, BT_);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
                    Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(
                        syncDoneFlag_);
#endif
                }
            }
            return;
        }

        const uint64_t h = hv / (HV_ / H_);
        if constexpr (IS_TAIL) {
            const uint64_t curT = chunk.end - chunk.start;
            if (curT == 0 || !UsePostWuCube(curT)) {
                return;
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            ComputeTailWuVector(
                chunk.b, hv, chunk.chunkIdx, chunk.start, curT,
                subBlockIdx, subBlockNum);
#else
            return;
#endif
            CopyScratchWAndFinalizeKg(
                chunk.b, h, hv, chunk.chunkIdx, chunk.start, curT,
                subBlockIdx, subBlockNum);
        } else {
            if (!UsePostWuCube(BT_)) {
                return;
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (UseTypicalPostWuGate(BT_)) {
                ComputeTypicalKg(
                    chunk.b, h, hv, chunk.start, BT_,
                    subBlockIdx, subBlockNum);
            } else {
                CopyScratchWAndFinalizeKg(
                    chunk.b, h, hv, chunk.chunkIdx, chunk.start, BT_,
                    subBlockIdx, subBlockNum);
            }
#else
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(
                syncDoneFlag_);
            CopyScratchWAndFinalizeKg(
                chunk.b, h, hv, chunk.chunkIdx, chunk.start, BT_,
                subBlockIdx, subBlockNum);
#endif
        }
    }

    template <bool IS_AIC, bool IS_TAIL>
    __aicore__ inline void ProcessCompactPostHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t headBase,
        uint64_t subBlockIdx, uint64_t subBlockNum, uint32_t headCnt)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!IS_TAIL) {
            if constexpr (IS_AIC) {
                if (KDA_ENABLE_POST_AIC_PIPELINE &&
                    UseFullPostWuPipelineArch35(BT_)) {
                    ProcessTypicalFullPostAicHeadWindow(
                        chunk, headBase, headCnt);
                    return;
                }
            } else if (CanPipelineTypicalKg(
                           BT_, subBlockIdx, subBlockNum)) {
                ProcessTypicalFullPostAivHeadWindow(
                    chunk, headBase, subBlockIdx, subBlockNum, headCnt);
                return;
            }
        }
#endif
        uint64_t residentQHead = 0;
        bool residentQHeadValid = false;
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!IS_AIC) {
                const uint64_t hv = headBase + lane;
                const uint64_t qHead = hv / (HV_ / H_);
                if (!residentQHeadValid || qHead != residentQHead) {
                    // 每个runtime窗口从自身首项重新建resident；因此ratio8
                    // 跨两个四head窗口会按约定重读同一个qHead。
                    BeginFallbackKResidentGroupArch35(
                        chunk.b, qHead, chunk.start,
                        IS_TAIL ? chunk.end - chunk.start : BT_,
                        subBlockIdx, subBlockNum);
                    residentQHead = qHead;
                    residentQHeadValid = true;
                }
            }
#endif
            ProcessCompactPostHead<IS_AIC, IS_TAIL>(
                chunk, headBase + lane, subBlockIdx, subBlockNum);
        }
    }

    template <bool IS_AIC, bool IS_TAIL>
    __aicore__ inline void ProcessCompactPostHeadRange(
        const OwnedChunkDesc &chunk, uint64_t headBegin, uint64_t headEnd,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                headCnt = static_cast<uint32_t>(headEnd - head);
            }
            // runtime headCnt 只限定循环次数；每处理一个真实 head，AIC
            // 与 AIV 仍在同一条 flag 流上各推进一次信号计数。
            ProcessCompactPostHeadWindow<IS_AIC, IS_TAIL>(
                chunk, head, subBlockIdx, subBlockNum, headCnt);
            head += headCnt;
        }
    }

    template <bool IS_AIC>
    __aicore__ inline void ProcessG1PostStage(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        FullChunkIterator fullIterator{};
        fullIterator.sequence = cursor.fullStartSequence;
        fullIterator.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t ordinal = cursor.fullBegin;
             ordinal < cursor.fullEnd; ++ordinal) {
            OwnedChunkDesc chunk{};
            if (LoadOwnedFullChunk(plan, fullIterator, chunk)) {
                ProcessCompactPostHeadRange<IS_AIC, false>(
                    chunk, 0, HV_, subBlockIdx, subBlockNum);
            }
        }
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            OwnedChunkDesc chunk{};
            if (LoadOwnedTailChunk(plan, ordinal, chunk)) {
                ProcessCompactPostHeadRange<IS_AIC, true>(
                    chunk, 0, HV_, subBlockIdx, subBlockNum);
            }
        }
    }

    template <bool IS_AIC>
    __aicore__ inline void ProcessGroupedPostStage(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        GroupedFullTaskIterator fullIterator{};
        fullIterator.chunks.sequence = cursor.fullStartSequence;
        fullIterator.chunks.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t task = cursor.fullBegin;
             task < cursor.fullEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedFullTask(
                    plan, task, fullIterator, chunk, headBegin, headEnd)) {
                ProcessCompactPostHeadRange<IS_AIC, false>(
                    chunk, headBegin, headEnd, subBlockIdx, subBlockNum);
            }
        }
        GroupedTailTaskIterator tailIterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedTailTask(
                    plan, task, tailIterator, chunk, headBegin, headEnd)) {
                ProcessCompactPostHeadRange<IS_AIC, true>(
                    chunk, headBegin, headEnd, subBlockIdx, subBlockNum);
            }
        }
    }

    template <bool IS_AIC>
    __aicore__ inline void ProcessCompactPostStage()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        uint64_t subBlockIdx = 0;
        uint64_t subBlockNum = 1;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        if constexpr (!IS_AIC) {
            subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            if (subBlockNum == 0) {
                return;
            }
            subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
            coreIdx /= subBlockNum;
        }
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        KdaForward::ChunkCoreCursor cursor{};
        if (!plan.LoadChunkCoreCursor(
                static_cast<uint32_t>(coreIdx), cursor)) {
            return;
        }
        if (plan.HeadGroupCount() == 1) {
            ProcessG1PostStage<IS_AIC>(
                plan, cursor, subBlockIdx, subBlockNum);
        } else {
            ProcessGroupedPostStage<IS_AIC>(
                plan, cursor, subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessPostAiv()
    {
        ProcessCompactPostStage<false>();
    }

    __aicore__ inline void ProcessPostAic()
    {
        ProcessCompactPostStage<true>();
    }

    __aicore__ inline void ProcessChunkPostAicTyped(
        uint64_t b, uint64_t hv, uint64_t chunkIdx,
        uint64_t start, uint64_t end)
    {
        const uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        ComputePostWuCube(b, hv, chunkIdx, start, curT);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(
            syncDoneFlag_);
#endif
    }

    template <bool COPY_SEED, bool CUBE_AIC>
    __aicore__ inline void ProcessTailAuxHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t headBase,
        uint64_t subBlockIdx, uint64_t subBlockNum, uint32_t headCnt)
    {
        static_assert(!(COPY_SEED && CUBE_AIC),
                      "tail auxiliary stage must have one compute role");
        const uint64_t curT = chunk.end - chunk.start;
        uint64_t residentQHead = 0;
        bool residentQHeadValid = false;
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            const uint64_t hv = headBase + lane;
            if constexpr (COPY_SEED) {
                CopyTailSeedRows(
                    chunk.b, hv, chunk.start, curT,
                    subBlockIdx, subBlockNum);
            } else if constexpr (CUBE_AIC) {
                ProcessChunkPostAicTyped(
                    chunk.b, hv, chunk.chunkIdx, chunk.start, chunk.end);
            } else {
                const uint64_t h = hv / (HV_ / H_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if (!residentQHeadValid || h != residentQHead) {
                    BeginFallbackKResidentGroupArch35(
                        chunk.b, h, chunk.start, curT,
                        subBlockIdx, subBlockNum);
                    residentQHead = h;
                    residentQHeadValid = true;
                }
#endif
                CopyScratchWAndFinalizeKg(
                    chunk.b, h, hv, chunk.chunkIdx, chunk.start, curT,
                    subBlockIdx, subBlockNum);
            }
        }
    }

    template <bool COPY_SEED, bool CUBE_AIC>
    __aicore__ inline void ProcessTailAuxHeadRange(
        const OwnedChunkDesc &chunk, uint64_t headBegin, uint64_t headEnd,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                headCnt = static_cast<uint32_t>(headEnd - head);
            }
            ProcessTailAuxHeadWindow<COPY_SEED, CUBE_AIC>(
                chunk, head, subBlockIdx, subBlockNum, headCnt);
            head += headCnt;
        }
    }

    template <bool COPY_SEED, bool CUBE_AIC>
    __aicore__ inline void ProcessTailAuxStageFromPlan()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        if (!isVarLen_ || BT_ != 64 || K_ != 128 || V_ != 128) {
            return;
        }
        uint64_t subBlockIdx = 0;
        uint64_t subBlockNum = 1;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        if constexpr (!CUBE_AIC) {
            subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            if (subBlockNum == 0) {
                return;
            }
            subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
            coreIdx /= subBlockNum;
        }
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        KdaForward::ChunkCoreCursor cursor{};
        if (!plan.LoadChunkCoreCursor(
                static_cast<uint32_t>(coreIdx), cursor)) {
            return;
        }
        if (plan.HeadGroupCount() == 1) {
            for (uint64_t ordinal = cursor.tailBegin;
                 ordinal < cursor.tailEnd; ++ordinal) {
                OwnedChunkDesc chunk{};
                if (LoadOwnedTailChunk(plan, ordinal, chunk)) {
                    ProcessTailAuxHeadRange<COPY_SEED, CUBE_AIC>(
                        chunk, 0, HV_, subBlockIdx, subBlockNum);
                }
            }
            return;
        }
        GroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            OwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                ProcessTailAuxHeadRange<COPY_SEED, CUBE_AIC>(
                    chunk, headBegin, headEnd, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessVarlenTailSeedCopyAiv()
    {
        ProcessTailAuxStageFromPlan<true, false>();
    }

    __aicore__ inline void ProcessVarlenTailAic()
    {
        ProcessTailAuxStageFromPlan<false, true>();
    }

    __aicore__ inline void ProcessVarlenTailAiv()
    {
        ProcessTailAuxStageFromPlan<false, false>();
    }


private:
    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<T> v_;
    GlobalTensor<GK_T> gk_;
    GlobalTensor<BETA_T> beta_;
    GlobalTensor<float> initialState_;
    GlobalTensor<OUT_T> o_;
    GlobalTensor<float> finalState_;
    GlobalTensor<float> aqk_;
    GlobalTensor<AKK_T> akk_;
    GlobalTensor<T> w_;
    GlobalTensor<OUT_T> u_;
    GlobalTensor<T> qg_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<float> h_;
    GlobalTensor<T> preparedQG_;
    GlobalTensor<T> preparedAqk_;
    GlobalTensor<T> propagatedVNew_;
    GlobalTensor<T> propagatedH_;
    GlobalTensor<float> solveWorkspace_;
    GlobalTensor<T> scoreWorkspace_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> exp2Buf_;
    TBuf<TPosition::VECCALC> vecBuf_;
    TBuf<TPosition::VECCALC> gateWritebackBuf_;
    TEventID mte2ToVEvent_ = 0;
    TEventID vToMte2Event_ = 0;
    TEventID vToMte3Event_ = 0;
    TEventID mte3ToVEvent_ = 0;
    TEventID mte2ToMte3Event_ = 0;
    TEventID mte3ToMte2Event_ = 0;
    bool vectorEventsAllocated_ = false;
    bool fallbackKResidentEnabled_ = false;
    bool fallbackKResidentHasVectorReader_ = false;
    uint64_t fallbackKResidentB_ = 0;
    uint64_t fallbackKResidentH_ = 0;
    uint64_t fallbackKResidentTokenBegin_ = 0;
    uint64_t fallbackKResidentTokenEnd_ = 0;
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreReadyFlag_{KDA_SCORE_READY_FLAG0,
                                                                                  KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreDoneFlag_{KDA_SCORE_DONE_FLAG0,
                                                                                 KDA_SCORE_DONE_FLAG1};
    // solve 开始前 score 生产已经完全排空，因此 solve 握手可以安全复用
    // 现有 score flag，不再额外占用硬件 flag ID。
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> syncReadyFlag_{KDA_SCORE_READY_FLAG0,
                                                                                  KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> syncDoneFlag_{KDA_SCORE_DONE_FLAG0,
                                                                                 KDA_SCORE_DONE_FLAG1};
    uint64_t B_ = 0;
    uint64_t N_ = 0;
    uint64_t H_ = 0;
    uint64_t HV_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t BT_ = 0;
    uint64_t NT_ = 0;
    float scale_ = 1.0f;
    bool hasInitial_ = false;
    bool isVarLen_ = false;
    bool isAivOnly_ = false;
    bool inputSequenceMajor_ = false;
    uint64_t solveCoreIdx_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
    GM_ADDR compactPlanAddr_ = nullptr;
};
} // namespace

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaPostWu(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
    GM_ADDR wSeed, GM_ADDR akk, GM_ADDR uSeed,
    GM_ADDR w, GM_ADDR u, GM_ADDR kg, GM_ADDR vNew, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    GM_ADDR postScratch = userWorkspace + tiling.postWuScratchOffset;
    if ASCEND_IS_AIC {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace, akk, w, u,
                userWorkspace, kg, vNew, postScratch, postScratch, tiling, &pipe, false);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace, akk, w, u,
                userWorkspace, kg, vNew, postScratch, postScratch, tiling, &pipe);
        op.ProcessAiv();
    }
}

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaPostWuTailSeedCopy(
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan, GM_ADDR wSeed,
    GM_ADDR vNewSeed, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    if ASCEND_IS_AIV {
        GM_ADDR scratchW = userWorkspace + tiling.outputScratchOffset;
        GM_ADDR scratchV = scratchW + tiling.seqlen * tiling.vHeadNum *
            tiling.kHeadDim * sizeof(T);
        GM_ADDR postScratch = userWorkspace + tiling.postWuScratchOffset;
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(wSeed, wSeed, vNewSeed, userWorkspace, userWorkspace,
                nullptr, cuSeqlens, chunkIndices, compactPlan, wSeed, userWorkspace,
                vNewSeed, nullptr, userWorkspace, userWorkspace, userWorkspace,
                userWorkspace, scratchW, scratchV, userWorkspace, userWorkspace,
                vNewSeed, postScratch, postScratch, tiling, &pipe);
        op.ProcessTailSeedCopyAiv();
    }
}

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaPostWuTail(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
    GM_ADDR akk, GM_ADDR vNew, GM_ADDR w, GM_ADDR u,
    GM_ADDR kg, GM_ADDR userWorkspace, const TilingData &tiling, TPipe &pipe)
{
    GM_ADDR scratchW = userWorkspace + tiling.outputScratchOffset;
    GM_ADDR scratchV = scratchW + tiling.seqlen * tiling.vHeadNum *
        tiling.kHeadDim * sizeof(T);
    GM_ADDR postScratch = userWorkspace + tiling.postWuScratchOffset;
    // Tail Cube 读取不可变快照并写入公开 W/U 张量，既保留快速 MMAD
    // 路径，也避免旧实现中的输入输出别名。
    if ASCEND_IS_AIC {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                scratchW, akk, scratchV, nullptr, userWorkspace, userWorkspace,
                userWorkspace, akk, w, u, userWorkspace, kg, vNew,
                postScratch, postScratch, tiling, &pipe, false);
        op.ProcessTailAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                scratchW, akk, scratchV, nullptr, userWorkspace, userWorkspace,
                userWorkspace, akk, w, u, userWorkspace, kg, vNew,
                postScratch, postScratch, tiling, &pipe);
        op.ProcessTailAiv();
    }
}

} // namespace KdaPostWu
