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
#include "catlass/gemm_coord.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "../chunk_kda_fwd_plan.h"
#include "../chunk_kda_fwd_varlen.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace AscendC;

namespace KdaFinalize {
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
constexpr uint32_t KDA_SCORE_QUEUE_DEPTH = 2;
// Finalize 每完成一个 head 就发布一个 descriptor，并用 mode2 completion
// 在复用前回收槽位；这里是 descriptor 级双缓冲，不是 dHU 的整组
// 4-head task-window 双 bank workspace。
constexpr uint32_t KDA_OUTPUT_SLOT_DEPTH = 2;
// 当前手写 A5 输出流水只覆盖 C=64、K=V=128；每个 FP32 L0C 槽占 32 KiB。
constexpr uint32_t KDA_OUTPUT_L0C_SLOT_DEPTH = 2;
constexpr uint32_t KDA_OUTPUT_L0C_SLOT_BYTES = 64 * 128 * sizeof(float);
constexpr uint8_t KDA_OUTPUT_DONE_FLAG = 2;
constexpr uint8_t KDA_OUTPUT_COMPLETION_FLAG = 4;
constexpr uint32_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint32_t KDA_SCORE_SCRATCH_QG = 0;
constexpr uint32_t KDA_SCORE_SCRATCH_W = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_KG = 2;
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_GATE_TILE_ROWS = 32;
constexpr uint32_t KDA_CUBE_MIN_REDUCTION = 16;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif
// Cube 与 Fixpipe 通过两个 L0C 槽并行，事件只负责槽位所有权交接。
using KdaDispatchPolicy = Common::MmadPingpong<KdaArchTag, false, false, 2>;
using KdaL1TileShape = tla::Shape<KdaInt64, KdaInt128, KdaInt128>;
using KdaL0TileShape = KdaL1TileShape;

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
class ChunkKdaFwdFinalizeKernel {
public:
    using OUT_T = float;
    using AKK_T = float;
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
        usedCoreNum_ = tiling.outputUsedCoreNum;
        if ASCEND_IS_AIV {
            uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            solveCoreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        } else {
            solveCoreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }
        outputTileElements_ = BT_ * V_;
        const uint64_t coreScratchOffset =
            2 * KDA_OUTPUT_SLOT_DEPTH * solveCoreIdx_ * outputTileElements_;
        o_.SetGlobalBuffer(
            (__gm__ OUT_T *)workspace + coreScratchOffset);
        u_.SetGlobalBuffer(
            (__gm__ OUT_T *)workspace + coreScratchOffset +
            outputTileElements_);
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
        ProcessOutAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAic()
    {
        ProcessOutAic();
    }

private:
    __aicore__ inline void AllocVectorEvents()
    {
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte2ToMte3Event_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2Event_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        vToSEvent_ = pipe_->AllocEventID<HardEvent::V_S>();
        sToVEvent_ = pipe_->AllocEventID<HardEvent::S_V>();
        sToMte2Event_ = pipe_->AllocEventID<HardEvent::S_MTE2>();
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
        pipe_->ReleaseEventID<HardEvent::V_S>(vToSEvent_);
        pipe_->ReleaseEventID<HardEvent::S_V>(sToVEvent_);
        pipe_->ReleaseEventID<HardEvent::S_MTE2>(sToMte2Event_);
        vectorEventsAllocated_ = false;
    }

    struct OutputL0CPipelineState {
        TEventID mToFixEvents[KDA_OUTPUT_L0C_SLOT_DEPTH]{};
        TEventID fixToMEvents[KDA_OUTPUT_L0C_SLOT_DEPTH]{};
        uint32_t nextSlot = 0;
    };

    __aicore__ inline void InitOutputL0CPipelineState(
        OutputL0CPipelineState &state)
    {
        state.nextSlot = 0;
        for (uint32_t slot = 0; slot < KDA_OUTPUT_L0C_SLOT_DEPTH; ++slot) {
            state.mToFixEvents[slot] = pipe_->AllocEventID<HardEvent::M_FIX>();
            state.fixToMEvents[slot] = pipe_->AllocEventID<HardEvent::FIX_M>();
            // 每个物理 L0C 槽只在上层调度入口投放一次初始槽位令牌。
            SetFlag<HardEvent::FIX_M>(state.fixToMEvents[slot]);
        }
    }

    __aicore__ inline void DrainOutputL0CPipelineState(
        OutputL0CPipelineState &state)
    {
        for (uint32_t slot = 0; slot < KDA_OUTPUT_L0C_SLOT_DEPTH; ++slot) {
            // 等 Fixpipe 归还槽位后再释放事件，避免异步写回仍引用旧事件。
            WaitFlag<HardEvent::FIX_M>(state.fixToMEvents[slot]);
        }
        for (uint32_t slot = 0; slot < KDA_OUTPUT_L0C_SLOT_DEPTH; ++slot) {
            pipe_->ReleaseEventID<HardEvent::M_FIX>(state.mToFixEvents[slot]);
            pipe_->ReleaseEventID<HardEvent::FIX_M>(state.fixToMEvents[slot]);
        }
        state.nextSlot = 0;
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        return ((b * H_ + h) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t KVOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d, uint64_t dim) const
    {
        return ((b * HV_ + hv) * T_ + t) * dim + d;
    }

    __aicore__ inline uint64_t OutputOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d) const
    {
        return ((b * T_ + t) * HV_ + hv) * V_ + d;
    }

    __aicore__ inline uint64_t OutputScratchOffset(
        uint64_t row, uint64_t d) const
    {
        return activeOutputSlot_ * 2 * outputTileElements_ + row * V_ + d;
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
    __aicore__ inline void CopyRowsOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src,
                                       uint64_t rows, uint64_t cols, uint64_t dstStride)
    {
        if (cols == dstStride) {
            CopyVectorOut(dst, offset, src, rows * cols);
            return;
        }
        constexpr uint64_t blockBytes = 32;
        const uint64_t rowBytes = cols * sizeof(CopyT);
        const uint64_t gapBytes = (dstStride - cols) * sizeof(CopyT);
        DataCopyParams params{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            1,
#else
            static_cast<uint16_t>(rows),
#endif
            static_cast<uint16_t>(rowBytes / blockBytes),
            0,
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            0
#else
            static_cast<uint16_t>(gapBytes / blockBytes)
#endif
        };
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        const uint64_t dstRowBytes = dstStride * sizeof(CopyT);
        LoopModeParams loopParams{
            static_cast<uint32_t>(rows), 1, rowBytes, dstRowBytes, 0, 0};
        // 循环搬运寄存器属于核内状态，每次DMA调用后必须复位，不能泄漏到下一次搬运。
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopy(dst[offset], src, params);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
#else
        DataCopy(dst[offset], src, params);
#endif
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

    __aicore__ inline void ComputeTailLocalRows(LocalTensor<float> &dst, uint64_t b, uint64_t hv,
                                                uint64_t start, uint64_t curT, uint64_t rowBegin,
                                                uint64_t rows)
    {
        LocalTensor<float> vRow = exp2Buf_.Get<float>();
        LocalTensor<T> coefficientTyped = gateWritebackBuf_.Get<T>();
        LocalTensor<float> coefficients = gateWritebackBuf_.Get<float>()[BT_];
        for (uint64_t localRow = 0; localRow < rows; ++localRow) {
            LocalTensor<float> dstRow = dst[localRow * V_];
            CopyVectorIn(
                coefficientTyped, preparedAqk_,
                AOffset(b, hv, start + rowBegin + localRow, 0), curT);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(
                coefficients, coefficientTyped, RoundMode::CAST_NONE,
                static_cast<uint32_t>(curT));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(vToSEvent_);
            WaitFlag<HardEvent::V_S>(vToSEvent_);
            Duplicate(dstRow, 0.0f, static_cast<uint32_t>(V_));
            PipeBarrier<PIPE_V>();
            for (uint64_t j = 0; j < curT; ++j) {
                LoadAsFloatRow(
                    propagatedVNew_, KVOffset(b, hv, start + j, 0, V_), vRow, V_);
                float weight = coefficients.GetValue(j);
                SetFlag<HardEvent::S_V>(sToVEvent_);
                WaitFlag<HardEvent::S_V>(sToVEvent_);
                Muls(vRow, vRow, weight, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                Add(dstRow, dstRow, vRow, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
            }
            SetFlag<HardEvent::S_MTE2>(sToMte2Event_);
            WaitFlag<HardEvent::S_MTE2>(sToMte2Event_);
        }
    }

    __aicore__ inline void ComputeTailStateRows(LocalTensor<float> &dst, uint64_t b, uint64_t hv,
                                                uint64_t chunkIdx, uint64_t start, uint64_t rowBegin,
                                                uint64_t rows)
    {
        LocalTensor<float> hRow = exp2Buf_.Get<float>();
        LocalTensor<T> coefficientTyped = gateWritebackBuf_.Get<T>();
        LocalTensor<float> coefficients = gateWritebackBuf_.Get<float>()[BT_];
        for (uint64_t localRow = 0; localRow < rows; ++localRow) {
            LocalTensor<float> dstRow = dst[localRow * V_];
            CopyVectorIn(
                coefficientTyped, preparedQG_,
                KVOffset(b, hv, start + rowBegin + localRow, 0, K_), K_);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(
                coefficients, coefficientTyped, RoundMode::CAST_NONE,
                static_cast<uint32_t>(K_));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(vToSEvent_);
            WaitFlag<HardEvent::V_S>(vToSEvent_);
            Duplicate(dstRow, 0.0f, static_cast<uint32_t>(V_));
            PipeBarrier<PIPE_V>();
            for (uint64_t d = 0; d < K_; ++d) {
                LoadAsFloatRow(
                    propagatedH_, HOffset(b, hv, chunkIdx, d, 0), hRow, V_);
                float weight = coefficients.GetValue(d);
                SetFlag<HardEvent::S_V>(sToVEvent_);
                WaitFlag<HardEvent::S_V>(sToVEvent_);
                Muls(hRow, hRow, weight, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                Add(dstRow, dstRow, hRow, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
            }
            SetFlag<HardEvent::S_MTE2>(sToMte2Event_);
            WaitFlag<HardEvent::S_MTE2>(sToMte2Event_);
        }
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ComputeOutputCubeStagedArch35(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
        uint64_t curT, OutputL0CPipelineState &l0cState)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        constexpr uint16_t kMte2Event = 0;
        constexpr uint16_t kMte1Event = 0;
        constexpr uint16_t kMmadEvent = 0;
        constexpr uint32_t kL1A0Offset = 0;
        constexpr uint32_t kL1B0Offset = 64 * 128 * sizeof(ElementA);
        constexpr uint32_t kL1A1Offset = kL1B0Offset + 128 * 128 * sizeof(ElementB);
        constexpr uint32_t kL1B1Offset = kL1A1Offset + 64 * 64 * sizeof(ElementA);

        Catlass::Arch::Resource<KdaArchTag> resource;
        LocalTensor<ElementA> l1A0 = resource.l1Buf.template GetBufferByByte<ElementA>(kL1A0Offset);
        LocalTensor<ElementB> l1B0 = resource.l1Buf.template GetBufferByByte<ElementB>(kL1B0Offset);
        LocalTensor<ElementA> l1A1 = resource.l1Buf.template GetBufferByByte<ElementA>(kL1A1Offset);
        LocalTensor<ElementB> l1B1 = resource.l1Buf.template GetBufferByByte<ElementB>(kL1B1Offset);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        auto layoutQ = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutH = tla::MakeLayout<ElementB, LayoutTagB>(K_, V_);
        auto layoutO = tla::MakeLayout<ElementC, LayoutTagC>(BT_, V_);
        auto layoutAqk = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutV = tla::MakeLayout<ElementB, LayoutTagB>(BT_, V_);

        for (uint64_t nOffset = 0; nOffset < V_; nOffset += 128) {
            const uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));
            auto tensorH = tla::MakeTensor(propagatedH_[HOffset(b, hv, chunkIdx, 0, nOffset)], layoutH,
                                           Catlass::Arch::PositionGM{});
            auto tensorVNew = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, nOffset, V_)], layoutV,
                                              Catlass::Arch::PositionGM{});

            for (uint64_t mOffset = 0; mOffset < curT; mOffset += 64) {
                const uint32_t curM = static_cast<uint32_t>((curT - mOffset) > 64 ? 64 : (curT - mOffset));
                auto tensorQ = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start + mOffset, 0, K_)], layoutQ,
                                               Catlass::Arch::PositionGM{});
                auto tensorAqk = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start + mOffset, 0)], layoutAqk,
                                                 Catlass::Arch::PositionGM{});
                auto tensorO = tla::MakeTensor(o_[OutputScratchOffset(mOffset, nOffset)], layoutO,
                                               Catlass::Arch::PositionGM{});
                auto tensorLocal = tla::MakeTensor(u_[OutputScratchOffset(mOffset, nOffset)], layoutO,
                                                   Catlass::Arch::PositionGM{});

                auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(curM, K_));
                auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(K_, curN));
                auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(curM, curT));
                auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(curT, curN));
                auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));
                auto blockLocal =
                    GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));

                using CopyGmToL1A0 = typename TileCopy::template CopyGmToL1A<decltype(blockQ)>;
                using CopyGmToL1B0 = typename TileCopy::template CopyGmToL1B<decltype(blockH)>;
                using CopyGmToL1A1 = typename TileCopy::template CopyGmToL1A<decltype(blockAqk)>;
                using CopyGmToL1B1 = typename TileCopy::template CopyGmToL1B<decltype(blockVNew)>;
                using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockO)>;
                CopyGmToL1A0 copyGmToL1A0;
                CopyGmToL1B0 copyGmToL1B0;
                CopyGmToL1A1 copyGmToL1A1;
                CopyGmToL1B1 copyGmToL1B1;
                CopyL0CToDst copyL0CToDst;

                auto layoutL1A0 = tla::MakeLayout<ElementA, LayoutTagL1A>(curM, K_);
                auto layoutL1B0 = tla::MakeLayout<ElementB, LayoutTagL1B>(K_, curN);
                auto layoutL1A1 = tla::MakeLayout<ElementA, LayoutTagL1A>(curM, curT);
                auto layoutL1B1 = tla::MakeLayout<ElementB, LayoutTagL1B>(curT, curN);
                auto layoutL0A0 = tla::MakeLayout<ElementA, LayoutTagL0A>(curM, K_);
                auto layoutL0B0 = tla::MakeLayout<ElementB, LayoutTagL0B>(K_, curN);
                auto layoutL0A1 = tla::MakeLayout<ElementA, LayoutTagL0A>(curM, curT);
                auto layoutL0B1 = tla::MakeLayout<ElementB, LayoutTagL0B>(curT, curN);
                auto layoutL0C = tla::MakeLayoutL0C(curM, curN);

                auto tensorL1A0 = tla::MakeTensor(l1A0, layoutL1A0, Catlass::Arch::PositionL1{});
                auto tensorL1B0 = tla::MakeTensor(l1B0, layoutL1B0, Catlass::Arch::PositionL1{});
                auto tensorL1A1 = tla::MakeTensor(l1A1, layoutL1A1, Catlass::Arch::PositionL1{});
                auto tensorL1B1 = tla::MakeTensor(l1B1, layoutL1B1, Catlass::Arch::PositionL1{});
                auto tensorL0A0 = tla::MakeTensor(l0A, layoutL0A0, Catlass::Arch::PositionL0A{});
                auto tensorL0B0 = tla::MakeTensor(l0B, layoutL0B0, Catlass::Arch::PositionL0B{});
                auto tensorL0A1 = tla::MakeTensor(l0A, layoutL0A1, Catlass::Arch::PositionL0A{});
                auto tensorL0B1 = tla::MakeTensor(l0B, layoutL0B1, Catlass::Arch::PositionL0B{});
                uint32_t localRow = 0;
                uint32_t localColumn = 0;
                auto tileL1A0 = GetTile(tensorL1A0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, K_));
                auto tileL1B0 = GetTile(tensorL1B0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(K_, curN));
                auto tileL1A1 = GetTile(tensorL1A1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, curT));
                auto tileL1B1 = GetTile(tensorL1B1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curT, curN));
                auto tileL0A0 = GetTile(tensorL0A0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, K_));
                auto tileL0B0 = GetTile(tensorL0B0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(K_, curN));
                auto tileL0A1 = GetTile(tensorL0A1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, curT));
                auto tileL0B1 = GetTile(tensorL0B1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curT, curN));
                copyGmToL1A0(tensorL1A0, blockQ);
                copyGmToL1B0(tensorL1B0, blockH);
                copyGmToL1A1(tensorL1A1, blockAqk);
                copyGmToL1B1(tensorL1B1, blockVNew);
                SetFlag<HardEvent::MTE2_MTE1>(kMte2Event);
                WaitFlag<HardEvent::MTE2_MTE1>(kMte2Event);

                copyL1ToL0A(tileL0A0, tileL1A0);
                copyL1ToL0B(tileL0B0, tileL1B0);
                SetFlag<HardEvent::MTE1_M>(kMte1Event);
                WaitFlag<HardEvent::MTE1_M>(kMte1Event);
                // 两个 32 KiB 槽按乘法结果轮转，使本次 Cube 可与上一槽的 Fixpipe 写回重叠。
                const uint32_t qhL0CSlot = l0cState.nextSlot;
                l0cState.nextSlot ^= 1U;
                LocalTensor<ElementC> qhL0C = resource.l0CBuf.template GetBufferByByte<ElementC>(
                    qhL0CSlot * KDA_OUTPUT_L0C_SLOT_BYTES);
                auto tensorQhL0C = tla::MakeTensor(qhL0C, layoutL0C, Catlass::Arch::PositionL0C{});
                auto tileQhL0C = GetTile(tensorQhL0C, tla::MakeCoord(localRow, localColumn),
                                         tla::MakeShape(curM, curN));
                WaitFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[qhL0CSlot]);
                tileMmad(tileQhL0C, tileL0A0, tileL0B0, curM, curN,
                         static_cast<uint32_t>(K_), true, 0);
                SetFlag<HardEvent::M_MTE1>(kMmadEvent);
                WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
                SetFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[qhL0CSlot]);
                WaitFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[qhL0CSlot]);
                copyL0CToDst(blockO, tileQhL0C);
                SetFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[qhL0CSlot]);

                copyL1ToL0A(tileL0A1, tileL1A1);
                copyL1ToL0B(tileL0B1, tileL1B1);
                SetFlag<HardEvent::MTE1_M>(kMte1Event);
                SetFlag<HardEvent::MTE1_MTE2>(kMte2Event);
                WaitFlag<HardEvent::MTE1_M>(kMte1Event);
                WaitFlag<HardEvent::MTE1_MTE2>(kMte2Event);
                const uint32_t aqkVL0CSlot = l0cState.nextSlot;
                l0cState.nextSlot ^= 1U;
                LocalTensor<ElementC> aqkVL0C = resource.l0CBuf.template GetBufferByByte<ElementC>(
                    aqkVL0CSlot * KDA_OUTPUT_L0C_SLOT_BYTES);
                auto tensorAqkVL0C = tla::MakeTensor(aqkVL0C, layoutL0C, Catlass::Arch::PositionL0C{});
                auto tileAqkVL0C = GetTile(tensorAqkVL0C, tla::MakeCoord(localRow, localColumn),
                                           tla::MakeShape(curM, curN));
                WaitFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[aqkVL0CSlot]);
                tileMmad(tileAqkVL0C, tileL0A1, tileL0B1, curM, curN,
                         static_cast<uint32_t>(curT), true, 0);
                SetFlag<HardEvent::M_MTE1>(kMmadEvent);
                WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
                SetFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[aqkVL0CSlot]);
                WaitFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[aqkVL0CSlot]);
                copyL0CToDst(blockLocal, tileAqkVL0C);
                SetFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[aqkVL0CSlot]);
            }
        }
    }

    __aicore__ inline void PrefetchOutputTileArch35(Catlass::Arch::Resource<KdaArchTag> &resource, uint32_t slot,
                                                uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                uint64_t curT, uint64_t nOffset, bool reuseSlot)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        constexpr uint32_t kL1SlotBytes = 96 * 1024;
        constexpr uint32_t kL1A0Offset = 0;
        constexpr uint32_t kL1B0Offset = 64 * 128 * sizeof(ElementA);
        constexpr uint32_t kL1A1Offset = kL1B0Offset + 128 * 128 * sizeof(ElementB);
        constexpr uint32_t kL1B1Offset = kL1A1Offset + 64 * 64 * sizeof(ElementA);
        const uint32_t slotBase = slot * kL1SlotBytes;
        const uint32_t curM = static_cast<uint32_t>(curT);
        const uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));

        auto layoutQ = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutH = tla::MakeLayout<ElementB, LayoutTagB>(K_, V_);
        auto layoutAqk = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutV = tla::MakeLayout<ElementB, LayoutTagB>(BT_, V_);
        auto tensorQ = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutQ,
                                       Catlass::Arch::PositionGM{});
        auto tensorH = tla::MakeTensor(propagatedH_[HOffset(b, hv, chunkIdx, 0, nOffset)], layoutH,
                                       Catlass::Arch::PositionGM{});
        auto tensorAqk = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start, 0)], layoutAqk,
                                         Catlass::Arch::PositionGM{});
        auto tensorVNew = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, nOffset, V_)], layoutV,
                                          Catlass::Arch::PositionGM{});
        auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(curM, K_));
        auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(K_, curN));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(curM, curT));
        auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(curT, curN));

        using CopyGmToL1A0 = typename TileCopy::template CopyGmToL1A<decltype(blockQ)>;
        using CopyGmToL1B0 = typename TileCopy::template CopyGmToL1B<decltype(blockH)>;
        using CopyGmToL1A1 = typename TileCopy::template CopyGmToL1A<decltype(blockAqk)>;
        using CopyGmToL1B1 = typename TileCopy::template CopyGmToL1B<decltype(blockVNew)>;
        CopyGmToL1A0 copyGmToL1A0;
        CopyGmToL1B0 copyGmToL1B0;
        CopyGmToL1A1 copyGmToL1A1;
        CopyGmToL1B1 copyGmToL1B1;

        LocalTensor<ElementA> l1A0 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A0Offset);
        LocalTensor<ElementB> l1B0 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B0Offset);
        LocalTensor<ElementA> l1A1 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A1Offset);
        LocalTensor<ElementB> l1B1 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B1Offset);
        auto tensorL1A0 = tla::MakeTensor(
            l1A0, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, K_), Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(
            l1B0, tla::MakeLayout<ElementB, LayoutTagL1B>(K_, curN), Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(
            l1A1, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, curT), Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(
            l1B1, tla::MakeLayout<ElementB, LayoutTagL1B>(curT, curN), Catlass::Arch::PositionL1{});

        if (reuseSlot) {
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
        }
        copyGmToL1A0(tensorL1A0, blockQ);
        copyGmToL1B0(tensorL1B0, blockH);
        copyGmToL1A1(tensorL1A1, blockAqk);
        copyGmToL1B1(tensorL1B1, blockVNew);
        SetFlag<HardEvent::MTE2_MTE1>(slot);
    }

    __aicore__ inline void ComputePrefetchedOutputTileArch35(Catlass::Arch::Resource<KdaArchTag> &resource,
                                                         uint32_t slot, uint64_t b, uint64_t hv, uint64_t start,
                                                         uint64_t curT, uint64_t nOffset,
                                                         OutputL0CPipelineState &l0cState)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        constexpr uint16_t kMte1Event = 0;
        constexpr uint16_t kMmadEvent = 0;
        constexpr uint32_t kL1SlotBytes = 96 * 1024;
        constexpr uint32_t kL1A0Offset = 0;
        constexpr uint32_t kL1B0Offset = 64 * 128 * sizeof(ElementA);
        constexpr uint32_t kL1A1Offset = kL1B0Offset + 128 * 128 * sizeof(ElementB);
        constexpr uint32_t kL1B1Offset = kL1A1Offset + 64 * 64 * sizeof(ElementA);
        const uint32_t slotBase = slot * kL1SlotBytes;
        const uint32_t curM = static_cast<uint32_t>(curT);
        const uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));

        LocalTensor<ElementA> l1A0 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A0Offset);
        LocalTensor<ElementB> l1B0 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B0Offset);
        LocalTensor<ElementA> l1A1 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A1Offset);
        LocalTensor<ElementB> l1B1 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B1Offset);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);

        auto tensorL1A0 = tla::MakeTensor(
            l1A0, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, K_), Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(
            l1B0, tla::MakeLayout<ElementB, LayoutTagL1B>(K_, curN), Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(
            l1A1, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, curT), Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(
            l1B1, tla::MakeLayout<ElementB, LayoutTagL1B>(curT, curN), Catlass::Arch::PositionL1{});
        auto tensorL0A0 = tla::MakeTensor(
            l0A, tla::MakeLayout<ElementA, LayoutTagL0A>(curM, K_), Catlass::Arch::PositionL0A{});
        auto tensorL0B0 = tla::MakeTensor(
            l0B, tla::MakeLayout<ElementB, LayoutTagL0B>(K_, curN), Catlass::Arch::PositionL0B{});
        auto tensorL0A1 = tla::MakeTensor(
            l0A, tla::MakeLayout<ElementA, LayoutTagL0A>(curM, curT), Catlass::Arch::PositionL0A{});
        auto tensorL0B1 = tla::MakeTensor(
            l0B, tla::MakeLayout<ElementB, LayoutTagL0B>(curT, curN), Catlass::Arch::PositionL0B{});
        auto layoutL0C = tla::MakeLayoutL0C(curM, curN);

        uint32_t localRow = 0;
        uint32_t localColumn = 0;
        auto tileL1A0 = GetTile(tensorL1A0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, K_));
        auto tileL1B0 = GetTile(tensorL1B0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(K_, curN));
        auto tileL1A1 = GetTile(tensorL1A1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, curT));
        auto tileL1B1 = GetTile(tensorL1B1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curT, curN));
        auto tileL0A0 = GetTile(tensorL0A0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, K_));
        auto tileL0B0 = GetTile(tensorL0B0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(K_, curN));
        auto tileL0A1 = GetTile(tensorL0A1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, curT));
        auto tileL0B1 = GetTile(tensorL0B1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curT, curN));

        auto layoutO = tla::MakeLayout<ElementC, LayoutTagC>(BT_, V_);
        auto tensorO = tla::MakeTensor(o_[OutputScratchOffset(0, nOffset)], layoutO,
                                       Catlass::Arch::PositionGM{});
        auto tensorLocal = tla::MakeTensor(u_[OutputScratchOffset(0, nOffset)], layoutO,
                                           Catlass::Arch::PositionGM{});
        auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));
        auto blockLocal = GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockO)>;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        TileMmad tileMmad;

        WaitFlag<HardEvent::MTE2_MTE1>(slot);
        copyL1ToL0A(tileL0A0, tileL1A0);
        copyL1ToL0B(tileL0B0, tileL1B0);
        SetFlag<HardEvent::MTE1_M>(kMte1Event);
        WaitFlag<HardEvent::MTE1_M>(kMte1Event);
        // 两个 32 KiB 槽按乘法结果轮转，使本次 Cube 可与上一槽的 Fixpipe 写回重叠。
        const uint32_t qhL0CSlot = l0cState.nextSlot;
        l0cState.nextSlot ^= 1U;
        LocalTensor<ElementC> qhL0C = resource.l0CBuf.template GetBufferByByte<ElementC>(
            qhL0CSlot * KDA_OUTPUT_L0C_SLOT_BYTES);
        auto tensorQhL0C = tla::MakeTensor(qhL0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileQhL0C = GetTile(tensorQhL0C, tla::MakeCoord(localRow, localColumn),
                                 tla::MakeShape(curM, curN));
        WaitFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[qhL0CSlot]);
        tileMmad(tileQhL0C, tileL0A0, tileL0B0, curM, curN,
                 static_cast<uint32_t>(K_), true, 0);
        SetFlag<HardEvent::M_MTE1>(kMmadEvent);
        WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
        SetFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[qhL0CSlot]);
        WaitFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[qhL0CSlot]);
        copyL0CToDst(blockO, tileQhL0C);
        SetFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[qhL0CSlot]);

        copyL1ToL0A(tileL0A1, tileL1A1);
        copyL1ToL0B(tileL0B1, tileL1B1);
        SetFlag<HardEvent::MTE1_M>(kMte1Event);
        SetFlag<HardEvent::MTE1_MTE2>(slot);
        WaitFlag<HardEvent::MTE1_M>(kMte1Event);
        const uint32_t aqkVL0CSlot = l0cState.nextSlot;
        l0cState.nextSlot ^= 1U;
        LocalTensor<ElementC> aqkVL0C = resource.l0CBuf.template GetBufferByByte<ElementC>(
            aqkVL0CSlot * KDA_OUTPUT_L0C_SLOT_BYTES);
        auto tensorAqkVL0C = tla::MakeTensor(aqkVL0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileAqkVL0C = GetTile(tensorAqkVL0C, tla::MakeCoord(localRow, localColumn),
                                   tla::MakeShape(curM, curN));
        WaitFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[aqkVL0CSlot]);
        tileMmad(tileAqkVL0C, tileL0A1, tileL0B1, curM, curN,
                 static_cast<uint32_t>(curT), true, 0);
        SetFlag<HardEvent::M_MTE1>(kMmadEvent);
        WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
        SetFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[aqkVL0CSlot]);
        WaitFlag<HardEvent::M_FIX>(l0cState.mToFixEvents[aqkVL0CSlot]);
        copyL0CToDst(blockLocal, tileAqkVL0C);
        SetFlag<HardEvent::FIX_M>(l0cState.fixToMEvents[aqkVL0CSlot]);
    }

    __aicore__ inline void DrainOutputInputPipelineEvents(uint64_t tileCount)
    {
        if (tileCount > 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(0);
        }
        if (tileCount > 1) {
            WaitFlag<HardEvent::MTE1_MTE2>(1);
        }
    }

#endif

    __aicore__ inline void ComputeOutputCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT,
                                             OutputL0CPipelineState *l0cState)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (curT < KDA_CUBE_MIN_REDUCTION) {
            return;
        }
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        // 只有完整 64 行块的上层调度会传入双槽状态；尾块和其他维度继续走通用实现。
        if (BT_ == 64 && K_ == 128 && V_ == 128 && curT == BT_ && l0cState != nullptr) {
            ComputeOutputCubeStagedArch35(b, hv, chunkIdx, start, curT, *l0cState);
            return;
        }
#endif
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Common::BlockMmadTla<KdaDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);

        auto layoutQ = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutH = tla::MakeLayout<ElementB, LayoutTagB>(K_, V_);
        auto layoutO = tla::MakeLayout<ElementC, LayoutTagC>(BT_, V_);
        for (uint64_t nOffset = 0; nOffset < V_; nOffset += 128) {
            uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));
            auto tensorH = tla::MakeTensor(propagatedH_[HOffset(b, hv, chunkIdx, 0, nOffset)], layoutH,
                                           Catlass::Arch::PositionGM{});
            for (uint64_t mOffset = 0; mOffset < curT; mOffset += 64) {
                uint32_t curM = static_cast<uint32_t>((curT - mOffset) > 64 ? 64 : (curT - mOffset));
                Catlass::GemmCoord shapeQH{curM, curN, static_cast<uint32_t>(K_)};
                auto tensorQ = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start + mOffset, 0, K_)], layoutQ,
                                               Catlass::Arch::PositionGM{});
                auto tensorO = tla::MakeTensor(o_[OutputScratchOffset(mOffset, nOffset)], layoutO,
                                               Catlass::Arch::PositionGM{});
                auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.k()));
                auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.k(), shapeQH.n()));
                auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.n()));
                blockMmad(blockQ, blockH, blockO, shapeQH);
                // 输出由下一阶段通过 MTE2 读取，只等待 Fixpipe 写回完成。
            }
        }

        if (curT < KDA_CUBE_MIN_REDUCTION) {
            return;
        }

        auto layoutAqk = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutV = tla::MakeLayout<ElementB, LayoutTagB>(BT_, V_);
        for (uint64_t nOffset = 0; nOffset < V_; nOffset += 128) {
            uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));
            auto tensorVNew = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, nOffset, V_)], layoutV,
                                              Catlass::Arch::PositionGM{});
            for (uint64_t mOffset = 0; mOffset < curT; mOffset += 64) {
                uint32_t curM = static_cast<uint32_t>((curT - mOffset) > 64 ? 64 : (curT - mOffset));
                Catlass::GemmCoord shapeAV{curM, curN, static_cast<uint32_t>(curT)};
                auto tensorAqk = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start + mOffset, 0)], layoutAqk,
                                                 Catlass::Arch::PositionGM{});
                auto tensorLocal = tla::MakeTensor(u_[OutputScratchOffset(mOffset, nOffset)], layoutO,
                                                   Catlass::Arch::PositionGM{});
                auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.k()));
                auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.k(), shapeAV.n()));
                auto blockLocal = GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.n()));
                blockMmad(blockAqk, blockVNew, blockLocal, shapeAV);
                // local 项写回完成后才允许 AIV 消费当前输出槽。
            }
        }
    }

    __aicore__ inline void FinalizeOutputRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum || V_ == 0) {
            return;
        }
        const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        const uint64_t gateWritebackRows =
            ScoreVectorMaxRows(5 * sizeof(float) + 2 * sizeof(T) + sizeof(GK_T));
        const uint64_t gateWritebackBytes =
            gateWritebackRows * K_ * (3 * sizeof(T) + sizeof(GK_T));
        uint64_t maxRows = KDA_VEC_ARENA_ELEMENTS / (3 * V_);
        const uint64_t typedMaxRows = gateWritebackBytes / (V_ * sizeof(T));
        if (maxRows > typedMaxRows) {
            maxRows = typedMaxRows;
        }
        if (maxRows == 0) {
            return;
        }

        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxRows) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxRows) {
                tileRows = maxRows;
            }
            const uint64_t elems = tileRows * V_;
            const uint64_t ti = start + tileRow;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> stateLocal = arena;
            LocalTensor<float> localLocal = arena[elems];
            LocalTensor<float> outLocal = arena[2 * elems];
            LocalTensor<T> outTyped = gateWritebackBuf_.Get<T>();

            if (curT < KDA_CUBE_MIN_REDUCTION) {
                ComputeTailStateRows(
                    stateLocal, b, hv, chunkIdx, start, tileRow, tileRows);
                ComputeTailLocalRows(localLocal, b, hv, start, curT, tileRow, tileRows);
            } else {
                CopyVectorIn(stateLocal, o_, OutputScratchOffset(tileRow, 0), elems);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                CopyVectorIn(localLocal, u_, OutputScratchOffset(tileRow, 0), elems);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            }
            Add(outLocal, stateLocal, localLocal, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outLocal, static_cast<uint32_t>(elems));
            Cast(outTyped, outLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyRowsOut(vNew_, OutputOffset(b, hv, ti, 0), outTyped, tileRows, V_, HV_ * V_);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }
    __aicore__ inline bool ResolveFlatChunk(uint64_t task, uint64_t &seq, uint64_t &b, uint64_t &h, uint64_t &hv,
                                            uint64_t &chunkIdx, uint64_t &start, uint64_t &end)
    {
        hv = task % HV_;
        uint64_t flatChunk = task / HV_;
        if (!isVarLen_) {
            seq = flatChunk / NT_;
            b = seq;
            chunkIdx = flatChunk % NT_;
            start = chunkIdx * BT_;
            end = start + BT_;
            if (end > T_) {
                end = T_;
            }
        } else {
            if (!KdaVarlen::ResolveChunkRange(
                    cuSeqlensAddr_, chunkIndicesAddr_, N_, T_, BT_, flatChunk,
                    seq, start, end)) {
                return false;
            }
            b = 0;
            chunkIdx = flatChunk;
        }
        h = hv / (HV_ / H_);
        return start < end;
    }

    struct OwnedChunkDesc {
        uint64_t b = 0;
        uint64_t chunkIdx = 0;
        uint64_t start = 0;
        uint64_t end = 0;
    };

    struct OutputProducerState {
        uint64_t descriptorIndex = 0;
        uint32_t outstandingCount = 0;
    };

    struct OutputConsumerState {
        uint64_t descriptorIndex = 0;
    };

    __aicore__ inline void WaitOutputCompletion()
    {
        Catlass::Arch::CrossCoreWaitFlag(outputCompletionFlag_);
    }

    __aicore__ inline void SetOutputDone()
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(outputDoneFlag_);
    }

    __aicore__ inline void WaitOutputDone()
    {
        Catlass::Arch::CrossCoreWaitFlag(outputDoneFlag_);
    }

    __aicore__ inline void SetOutputCompletion()
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(outputCompletionFlag_);
    }

    __aicore__ inline void AcquireOutputProducerSlot(
        OutputProducerState &state)
    {
        const uint32_t slot = static_cast<uint32_t>(
            state.descriptorIndex % KDA_OUTPUT_SLOT_DEPTH);
        // 两个真实输出槽共享同一条 completion 计数流。未完成描述符达到
        // 队列深度后，先消费最早的一份聚合 credit，再复用对应物理槽。
        if (state.outstandingCount >= KDA_OUTPUT_SLOT_DEPTH) {
            WaitOutputCompletion();
            --state.outstandingCount;
        }
        activeOutputSlot_ = slot;
    }

    __aicore__ inline void PublishOutputProducerSlot(OutputProducerState &state)
    {
        SetOutputDone();
        ++state.outstandingCount;
        ++state.descriptorIndex;
    }

    __aicore__ inline void DrainOutputProducerState(
        OutputProducerState &state)
    {
        while (state.outstandingCount != 0) {
            WaitOutputCompletion();
            --state.outstandingCount;
        }
        state.descriptorIndex = 0;
        activeOutputSlot_ = 0;
    }

    __aicore__ inline void AcquireOutputConsumerSlot(
        OutputConsumerState &state)
    {
        const uint32_t slot = static_cast<uint32_t>(
            state.descriptorIndex % KDA_OUTPUT_SLOT_DEPTH);
        activeOutputSlot_ = slot;
        WaitOutputDone();
    }

    __aicore__ inline void ReleaseOutputConsumerSlot(OutputConsumerState &state)
    {
        // 两个 AIV 对每个真实 head 各回传一次同号 mode2 token；AIC 只在
        // 两份都到达后得到一份 completion credit。
        SetOutputCompletion();
        ++state.descriptorIndex;
    }

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
        desc.b = isVarLen_ ? 0 : sequence;
        desc.chunkIdx = isVarLen_
            ? plan.SequenceChunkOffset(static_cast<uint32_t>(sequence)) + fullChunks
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
                static_cast<uint32_t>(HV_),
                chunkOrdinal, begin, end)) {
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
                static_cast<uint32_t>(HV_),
                chunkOrdinal, begin, end)) {
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

    __aicore__ inline void ProcessOwnedFullHeadWindowAicPipelinedArch35(
        Catlass::Arch::Resource<KdaArchTag> &resource,
        const OwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t &tileIndex, OutputProducerState &producerState,
        OutputL0CPipelineState &l0cState, uint32_t headCnt)
    {
        const uint64_t headEnd = headBegin + headCnt;
        uint64_t currentHv = headBegin;
        uint64_t currentNOffset = 0;
        AcquireOutputProducerSlot(producerState);
        uint32_t inputSlot = static_cast<uint32_t>(tileIndex & 1U);
        PrefetchOutputTileArch35(
            resource, inputSlot, chunk.b, currentHv, chunk.chunkIdx,
            chunk.start, chunk.end - chunk.start,
            currentNOffset, tileIndex >= 2);

        while (true) {
            uint64_t nextHv = currentHv;
            uint64_t nextNOffset = currentNOffset + 128;
            if (nextNOffset >= V_) {
                nextNOffset = 0;
                ++nextHv;
            }
            const bool hasNext = nextHv < headEnd;
            const uint32_t nextInputSlot = inputSlot ^ 1U;
            if (hasNext) {
                PrefetchOutputTileArch35(
                    resource, nextInputSlot, chunk.b, nextHv,
                    chunk.chunkIdx, chunk.start,
                    chunk.end - chunk.start, nextNOffset,
                    tileIndex + 1 >= 2);
            }

            ComputePrefetchedOutputTileArch35(
                resource, inputSlot, chunk.b, currentHv,
                chunk.start, chunk.end - chunk.start, currentNOffset,
                l0cState);
            if (currentNOffset + 128 >= V_) {
                PublishOutputProducerSlot(producerState);
            }
            ++tileIndex;
            if (!hasNext) {
                break;
            }
            currentHv = nextHv;
            currentNOffset = nextNOffset;
            inputSlot = nextInputSlot;
            if (currentNOffset == 0) {
                AcquireOutputProducerSlot(producerState);
            }
        }
    }

    __aicore__ inline void ProcessOwnedFullChunksAicPipelinedArch35(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor)
    {
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        Catlass::Arch::Resource<KdaArchTag> resource;
        OutputProducerState producerState{};
        OutputL0CPipelineState l0cState{};
        InitOutputL0CPipelineState(l0cState);
        uint64_t tileIndex = 0;
        FullChunkIterator iterator{};
        iterator.sequence = cursor.fullStartSequence;
        iterator.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t ordinal = cursor.fullBegin;
             ordinal < cursor.fullEnd; ++ordinal) {
            OwnedChunkDesc chunk{};
            if (!LoadOwnedFullChunk(plan, iterator, chunk)) {
                continue;
            }
            for (uint64_t head = 0; head < HV_;) {
                const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                    static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                    static_cast<uint32_t>(HV_));
                if (headCnt == 0) {
                    break;
                }
                ProcessOwnedFullHeadWindowAicPipelinedArch35(
                    resource, chunk, head, tileIndex, producerState,
                    l0cState, headCnt);
                head += headCnt;
            }
        }
        DrainOutputInputPipelineEvents(tileIndex);
        DrainOutputL0CPipelineState(l0cState);
        DrainOutputProducerState(producerState);
    }

    __attribute__((noinline)) __aicore__ void ProcessChunkOutAiv(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
        uint64_t end, uint64_t subBlockIdx, uint64_t subBlockNum,
        OutputConsumerState &consumerState)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        AcquireOutputConsumerSlot(consumerState);
        FinalizeOutputRows(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
        ReleaseOutputConsumerSlot(consumerState);
    }

    __attribute__((noinline)) __aicore__ void ProcessChunkOutAic(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
        uint64_t end, OutputProducerState &producerState,
        OutputL0CPipelineState *l0cState)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        AcquireOutputProducerSlot(producerState);
        ComputeOutputCube(b, hv, chunkIdx, start, curT, l0cState);
        PublishOutputProducerSlot(producerState);
    }

    __aicore__ inline void ProcessChunkOutAivHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t subBlockIdx, uint64_t subBlockNum,
        OutputConsumerState &consumerState, uint32_t headCnt)
    {
        for (uint32_t headOffset = 0; headOffset < headCnt; ++headOffset) {
            ProcessChunkOutAiv(
                chunk.b, headBegin + headOffset, chunk.chunkIdx,
                chunk.start, chunk.end, subBlockIdx, subBlockNum,
                consumerState);
        }
    }

    __aicore__ inline void ProcessChunkOutAivHeads(
        const OwnedChunkDesc &chunk, uint64_t headBegin, uint64_t headEnd,
        uint64_t subBlockIdx, uint64_t subBlockNum,
        OutputConsumerState &consumerState)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                return;
            }
            // descriptor 状态跨 runtime 窗口连续保留，每个真实 head
            // 在同一条 done/completion flag 流上恰好推进一次。
            ProcessChunkOutAivHeadWindow(
                chunk, head, subBlockIdx, subBlockNum, consumerState, headCnt);
            head += headCnt;
        }
    }

    __aicore__ inline void ProcessChunkOutAicHeadWindow(
        const OwnedChunkDesc &chunk, uint64_t headBegin,
        OutputProducerState &producerState,
        OutputL0CPipelineState *l0cState, uint32_t headCnt)
    {
        for (uint32_t headOffset = 0; headOffset < headCnt; ++headOffset) {
            ProcessChunkOutAic(
                chunk.b, headBegin + headOffset, chunk.chunkIdx,
                chunk.start, chunk.end, producerState, l0cState);
        }
    }

    __aicore__ inline void ProcessChunkOutAicHeads(
        const OwnedChunkDesc &chunk, uint64_t headBegin, uint64_t headEnd,
        OutputProducerState &producerState,
        OutputL0CPipelineState *l0cState)
    {
        for (uint64_t head = headBegin; head < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(head), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - head) {
                return;
            }
            ProcessChunkOutAicHeadWindow(
                chunk, head, producerState, l0cState, headCnt);
            head += headCnt;
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessOwnedChunksAiv(
        uint64_t coreIdx, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        KdaForward::ChunkCoreCursor cursor{};
        if (!plan.LoadChunkCoreCursor(static_cast<uint32_t>(coreIdx), cursor)) {
            return;
        }
        OutputConsumerState consumerState{};
        if constexpr (!IS_TAIL) {
            if (plan.HeadGroupCount() == 1) {
                FullChunkIterator iterator{};
                iterator.sequence = cursor.fullStartSequence;
                iterator.localChunk = cursor.fullStartLocalChunk;
                for (uint64_t ordinal = cursor.fullBegin;
                     ordinal < cursor.fullEnd; ++ordinal) {
                    OwnedChunkDesc chunk{};
                    if (!LoadOwnedFullChunk(plan, iterator, chunk)) {
                        continue;
                    }
                    ProcessChunkOutAivHeads(
                        chunk, 0, HV_, subBlockIdx, subBlockNum,
                        consumerState);
                }
            } else {
                GroupedFullTaskIterator iterator{};
                iterator.chunks.sequence = cursor.fullStartSequence;
                iterator.chunks.localChunk = cursor.fullStartLocalChunk;
                for (uint64_t task = cursor.fullBegin;
                     task < cursor.fullEnd; ++task) {
                    OwnedChunkDesc chunk{};
                    uint64_t headBegin = 0;
                    uint64_t headEnd = 0;
                    if (!LoadGroupedFullTask(
                            plan, task, iterator, chunk,
                            headBegin, headEnd)) {
                        continue;
                    }
                    ProcessChunkOutAivHeads(
                        chunk, headBegin, headEnd,
                        subBlockIdx, subBlockNum, consumerState);
                }
            }
        } else {
            if (plan.HeadGroupCount() == 1) {
                for (uint64_t ordinal = cursor.tailBegin;
                     ordinal < cursor.tailEnd; ++ordinal) {
                    OwnedChunkDesc chunk{};
                    if (!LoadOwnedTailChunk(plan, ordinal, chunk)) {
                        continue;
                    }
                    ProcessChunkOutAivHeads(
                        chunk, 0, HV_, subBlockIdx, subBlockNum,
                        consumerState);
                }
            } else {
                GroupedTailTaskIterator iterator{};
                for (uint64_t task = cursor.tailBegin;
                     task < cursor.tailEnd; ++task) {
                    OwnedChunkDesc chunk{};
                    uint64_t headBegin = 0;
                    uint64_t headEnd = 0;
                    if (!LoadGroupedTailTask(
                            plan, task, iterator, chunk,
                            headBegin, headEnd)) {
                        continue;
                    }
                    ProcessChunkOutAivHeads(
                        chunk, headBegin, headEnd,
                        subBlockIdx, subBlockNum, consumerState);
                }
            }
        }
        consumerState.descriptorIndex = 0;
        activeOutputSlot_ = 0;
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessOwnedChunksAic(uint64_t coreIdx)
    {
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        KdaForward::ChunkCoreCursor cursor{};
        if (!plan.LoadChunkCoreCursor(static_cast<uint32_t>(coreIdx), cursor)) {
            return;
        }
        OutputProducerState producerState{};
        OutputL0CPipelineState l0cState{};
        OutputL0CPipelineState *l0cStatePtr = nullptr;
        if constexpr (!IS_TAIL) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (plan.HeadGroupCount() == 1 &&
                BT_ == 64 && K_ == 128 && V_ == 128) {
                ProcessOwnedFullChunksAicPipelinedArch35(plan, cursor);
                return;
            }
            if (BT_ == 64 && K_ == 128 && V_ == 128) {
                // 分组完整块路径也由本阶段上层调度统一管理两份 L0C 槽位令牌。
                InitOutputL0CPipelineState(l0cState);
                l0cStatePtr = &l0cState;
            }
#endif
            if (plan.HeadGroupCount() == 1) {
                FullChunkIterator iterator{};
                iterator.sequence = cursor.fullStartSequence;
                iterator.localChunk = cursor.fullStartLocalChunk;
                for (uint64_t ordinal = cursor.fullBegin;
                     ordinal < cursor.fullEnd; ++ordinal) {
                    OwnedChunkDesc chunk{};
                    if (!LoadOwnedFullChunk(plan, iterator, chunk)) {
                        continue;
                    }
                    ProcessChunkOutAicHeads(
                        chunk, 0, HV_, producerState, l0cStatePtr);
                }
            } else {
                GroupedFullTaskIterator iterator{};
                iterator.chunks.sequence = cursor.fullStartSequence;
                iterator.chunks.localChunk = cursor.fullStartLocalChunk;
                for (uint64_t task = cursor.fullBegin;
                     task < cursor.fullEnd; ++task) {
                    OwnedChunkDesc chunk{};
                    uint64_t headBegin = 0;
                    uint64_t headEnd = 0;
                    if (!LoadGroupedFullTask(
                            plan, task, iterator, chunk,
                            headBegin, headEnd)) {
                        continue;
                    }
                    ProcessChunkOutAicHeads(
                        chunk, headBegin, headEnd, producerState,
                        l0cStatePtr);
                }
            }
        } else {
            if (plan.HeadGroupCount() == 1) {
                for (uint64_t ordinal = cursor.tailBegin;
                     ordinal < cursor.tailEnd; ++ordinal) {
                    OwnedChunkDesc chunk{};
                    if (!LoadOwnedTailChunk(plan, ordinal, chunk)) {
                        continue;
                    }
                    ProcessChunkOutAicHeads(
                        chunk, 0, HV_, producerState, nullptr);
                }
            } else {
                GroupedTailTaskIterator iterator{};
                for (uint64_t task = cursor.tailBegin;
                     task < cursor.tailEnd; ++task) {
                    OwnedChunkDesc chunk{};
                    uint64_t headBegin = 0;
                    uint64_t headEnd = 0;
                    if (!LoadGroupedTailTask(
                            plan, task, iterator, chunk,
                            headBegin, headEnd)) {
                        continue;
                    }
                    ProcessChunkOutAicHeads(
                        chunk, headBegin, headEnd, producerState, nullptr);
                }
            }
        }
        if (l0cStatePtr != nullptr) {
            DrainOutputL0CPipelineState(l0cState);
        }
        DrainOutputProducerState(producerState);
    }

    __aicore__ inline void ProcessOutAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        // 模板参数只选择完整块阶段或尾块阶段，不表示运行时一定存在尾块。
        // tailBegin == tailEnd 时尾块阶段为空；只有一个尾块时，仅处理归属
        // 当前核的实际 [start, end) 范围。AIV 与 AIC 必须保持相同阶段顺序。
        ProcessOwnedChunksAiv<false>(coreIdx, subBlockIdx, subBlockNum);
        ProcessOwnedChunksAiv<true>(coreIdx, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void ProcessOutAic()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        // <false>/<true> 分别选择完整块和尾块的游标、加载路径。是否执行
        // 由 full/tail 的运行时游标范围决定；单个尾块只迭代其所属任务
        // 范围，分组模式再按 head group 展开。结束前仍会排空输出 credit。
        ProcessOwnedChunksAic<false>(GetBlockIdx());
        ProcessOwnedChunksAic<true>(GetBlockIdx());
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
    TEventID vToSEvent_ = 0;
    TEventID sToVEvent_ = 0;
    TEventID sToMte2Event_ = 0;
    bool vectorEventsAllocated_ = false;
    // 两个物理输出槽按描述符序号轮转，但所有 head 复用同一组 mode2
    // done/completion 信号；队列深度负责保护槽位复用。
    Catlass::Arch::CrossCoreFlag outputDoneFlag_{KDA_OUTPUT_DONE_FLAG};
    Catlass::Arch::CrossCoreFlag outputCompletionFlag_{KDA_OUTPUT_COMPLETION_FLAG};
    uint64_t B_ = 0;
    uint64_t N_ = 0;
    uint64_t H_ = 0;
    uint64_t HV_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t BT_ = 0;
    uint64_t NT_ = 0;
    uint64_t outputTileElements_ = 0;
    uint64_t activeOutputSlot_ = 0;
    float scale_ = 1.0f;
    bool hasInitial_ = false;
    bool isVarLen_ = false;
    bool isAivOnly_ = false;
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
    GM_ADDR compactPlanAddr_ = nullptr;
};
} // namespace

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaOutput(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
    GM_ADDR qgScaled, GM_ADDR aqk,
    GM_ADDR propagatedVNew, GM_ADDR propagatedH, GM_ADDR o, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    GM_ADDR outputScratch = userWorkspace + tiling.outputScratchOffset;
    GM_ADDR stateScratch = outputScratch;
    GM_ADDR localScratch = outputScratch;
    if ASCEND_IS_AIC {
        ChunkKdaFwdFinalizeKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                qgScaled, aqk, propagatedVNew, propagatedH, stateScratch, userWorkspace, aqk, userWorkspace,
                userWorkspace, localScratch, userWorkspace, userWorkspace, o, propagatedH,
                outputScratch, tiling, &pipe, false);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdFinalizeKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                compactPlan,
                qgScaled, aqk, propagatedVNew, propagatedH, stateScratch, userWorkspace, aqk, userWorkspace,
                userWorkspace, localScratch, userWorkspace, userWorkspace, o, propagatedH,
                outputScratch, tiling, &pipe);
        op.ProcessAiv();
    }
}

} // namespace KdaFinalize
