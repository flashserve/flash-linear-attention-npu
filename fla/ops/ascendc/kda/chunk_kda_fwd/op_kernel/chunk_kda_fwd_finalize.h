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
#define CATLASS_ARCH 2201
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
#include "chunk_kda_fwd_plan.h"
#include "chunk_kda_fwd_varlen.h"
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
constexpr uint8_t KDA_OUTPUT_DONE_FLAG = 2;
constexpr uint8_t KDA_OUTPUT_COMPLETION_FLAG = 4;
constexpr uint32_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint32_t KDA_SCORE_SCRATCH_QG = 0;
constexpr uint32_t KDA_SCORE_SCRATCH_W = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_KG = 2;
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_GATE_TILE_ROWS = 32;

using KdaArchTag = Catlass::Arch::AtlasA2;
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
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR preparedQG, GM_ADDR preparedAqk,
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
        o_.SetGlobalBuffer((__gm__ OUT_T *)workspace + coreScratchOffset);
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

    __aicore__ inline void SetCompactPlan(GM_ADDR compactPlan)
    {
        compactPlanAddr_ = compactPlan;
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
            static_cast<uint16_t>(rows),
            static_cast<uint16_t>(rowBytes / blockBytes),
            0,
            static_cast<uint16_t>(gapBytes / blockBytes)
        };
        DataCopy(dst[offset], src, params);
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

    __aicore__ inline void ComputeOutputCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT)
    {
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

    __aicore__ inline void FinalizeOutputRows(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                              uint64_t subBlockIdx, uint64_t subBlockNum)
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

            CopyVectorIn(stateLocal, o_, OutputScratchOffset(tileRow, 0), elems);
            CopyVectorIn(localLocal, u_, OutputScratchOffset(tileRow, 0), elems);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
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

    struct CompactOwnedChunkDesc {
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
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(outputCompletionFlag_.id);
    }

    __aicore__ inline void SetOutputDone()
    {
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(outputDoneFlag_);
    }

    __aicore__ inline void WaitOutputDone()
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(outputDoneFlag_.id);
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

    __aicore__ inline void ResetOutputConsumerState(
        OutputConsumerState &state)
    {
        state.descriptorIndex = 0;
        activeOutputSlot_ = 0;
    }

    struct CompactFullChunkIterator {
        uint64_t sequence = 0;
        uint64_t localChunk = 0;
        uint64_t sequenceStart = 0;
        uint64_t fullChunkCount = 0;
        bool sequenceLoaded = false;
    };

    struct CompactGroupedFullTaskIterator {
        CompactFullChunkIterator chunks{};
        CompactOwnedChunkDesc chunk{};
        uint64_t loadedChunkOrdinal = 0;
        bool chunkLoaded = false;
    };

    struct CompactGroupedTailTaskIterator {
        CompactOwnedChunkDesc chunk{};
        uint64_t loadedChunkOrdinal = 0;
        bool chunkLoaded = false;
    };

    __aicore__ inline bool LoadCompactFullChunk(
        const KdaForward::CompactSequencePlanView &plan,
        CompactFullChunkIterator &iterator, CompactOwnedChunkDesc &chunk)
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
                chunk.b = isVarLen_ ? 0 : iterator.sequence;
                chunk.chunkIdx = isVarLen_
                    ? plan.SequenceChunkOffset(
                          static_cast<uint32_t>(iterator.sequence)) +
                          iterator.localChunk
                    : iterator.localChunk;
                chunk.start = iterator.sequenceStart +
                    iterator.localChunk * BT_;
                chunk.end = chunk.start + BT_;
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

    __aicore__ inline bool LoadCompactTailChunk(
        const KdaForward::CompactSequencePlanView &plan,
        uint64_t tailOrdinal, CompactOwnedChunkDesc &chunk)
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
            sequenceEnd = static_cast<uint64_t>(
                cuSeqlensAddr_[sequence + 1]);
        }
        const uint64_t fullChunkCount =
            (sequenceEnd - sequenceStart) / BT_;
        chunk.b = isVarLen_ ? 0 : sequence;
        chunk.chunkIdx = isVarLen_
            ? plan.SequenceChunkOffset(static_cast<uint32_t>(sequence)) +
                  fullChunkCount
            : fullChunkCount;
        chunk.start = sequenceStart + fullChunkCount * BT_;
        chunk.end = sequenceEnd;
        return chunk.start < chunk.end;
    }

    __aicore__ inline bool LoadCompactGroupedFullTask(
        const KdaForward::CompactSequencePlanView &plan, uint64_t task,
        CompactGroupedFullTaskIterator &iterator,
        CompactOwnedChunkDesc &chunk, uint64_t &headBegin,
        uint64_t &headEnd)
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
            if (!LoadCompactFullChunk(plan, iterator.chunks,
                                      iterator.chunk)) {
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

    __aicore__ inline bool LoadCompactGroupedTailTask(
        const KdaForward::CompactSequencePlanView &plan, uint64_t task,
        CompactGroupedTailTaskIterator &iterator,
        CompactOwnedChunkDesc &chunk, uint64_t &headBegin,
        uint64_t &headEnd)
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
            if (!LoadCompactTailChunk(plan, chunkOrdinal, iterator.chunk)) {
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

    __attribute__((noinline)) __aicore__ void ProcessChunkOutAiv(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start, uint64_t end, uint64_t subBlockIdx,
        uint64_t subBlockNum, OutputConsumerState &consumerState)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        AcquireOutputConsumerSlot(consumerState);
        FinalizeOutputRows(b, hv, start, curT, subBlockIdx, subBlockNum);
        ReleaseOutputConsumerSlot(consumerState);
    }

    __attribute__((noinline)) __aicore__ void ProcessChunkOutAic(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start, uint64_t end,
        OutputProducerState &producerState)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        AcquireOutputProducerSlot(producerState);
        ComputeOutputCube(b, hv, chunkIdx, start, curT);
        PublishOutputProducerSlot(producerState);
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactOutAivHeadWindow(
        const CompactOwnedChunkDesc &chunk, uint64_t headBase,
        uint32_t headCnt,
        uint64_t subBlockIdx, uint64_t subBlockNum,
        OutputConsumerState &consumerState)
    {
        uint64_t chunkEnd = chunk.start + BT_;
        if constexpr (IS_TAIL) {
            chunkEnd = chunk.end;
        }
        // producer/consumer 握手保留在 runtime 循环内，使共享信号量的
        // 计数深度逐个覆盖窗口内的真实 head。
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            ProcessChunkOutAiv(
                chunk.b, headBase + lane, chunk.chunkIdx,
                chunk.start, chunkEnd, subBlockIdx, subBlockNum,
                consumerState);
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactOutAivHeadRange(
        const CompactOwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd, uint64_t subBlockIdx, uint64_t subBlockNum,
        OutputConsumerState &consumerState)
    {
        for (uint64_t headBase = headBegin; headBase < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(headBase), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - headBase) {
                return;
            }
            ProcessCompactOutAivHeadWindow<IS_TAIL>(
                chunk, headBase, headCnt, subBlockIdx, subBlockNum,
                consumerState);
            headBase += headCnt;
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactOutAicHeadWindow(
        const CompactOwnedChunkDesc &chunk, uint64_t headBase,
        uint32_t headCnt,
        OutputProducerState &producerState)
    {
        uint64_t chunkEnd = chunk.start + BT_;
        if constexpr (IS_TAIL) {
            chunkEnd = chunk.end;
        }
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            ProcessChunkOutAic(
                chunk.b, headBase + lane, chunk.chunkIdx,
                chunk.start, chunkEnd, producerState);
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactOutAicHeadRange(
        const CompactOwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd, OutputProducerState &producerState)
    {
        for (uint64_t headBase = headBegin; headBase < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(headBase), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - headBase) {
                return;
            }
            ProcessCompactOutAicHeadWindow<IS_TAIL>(
                chunk, headBase, headCnt, producerState);
            headBase += headCnt;
        }
    }

    __aicore__ inline void ProcessG1FullOutAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum, OutputConsumerState &consumerState)
    {
        CompactFullChunkIterator iterator{};
        iterator.sequence = cursor.fullStartSequence;
        iterator.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t ordinal = cursor.fullBegin;
             ordinal < cursor.fullEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (!LoadCompactFullChunk(plan, iterator, chunk)) {
                continue;
            }
            ProcessCompactOutAivHeadRange<false>(
                chunk, 0, HV_, subBlockIdx, subBlockNum, consumerState);
        }
    }

    __aicore__ inline void ProcessG1TailOutAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum, OutputConsumerState &consumerState)
    {
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (!LoadCompactTailChunk(plan, ordinal, chunk)) {
                continue;
            }
            ProcessCompactOutAivHeadRange<true>(
                chunk, 0, HV_, subBlockIdx, subBlockNum, consumerState);
        }
    }

    __aicore__ inline void ProcessGroupedFullOutAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum, OutputConsumerState &consumerState)
    {
        CompactGroupedFullTaskIterator iterator{};
        iterator.chunks.sequence = cursor.fullStartSequence;
        iterator.chunks.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t task = cursor.fullBegin;
             task < cursor.fullEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadCompactGroupedFullTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                ProcessCompactOutAivHeadRange<false>(
                    chunk, headBegin, headEnd, subBlockIdx, subBlockNum,
                    consumerState);
            }
        }
    }

    __aicore__ inline void ProcessGroupedTailOutAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum, OutputConsumerState &consumerState)
    {
        CompactGroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadCompactGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                ProcessCompactOutAivHeadRange<true>(
                    chunk, headBegin, headEnd, subBlockIdx, subBlockNum,
                    consumerState);
            }
        }
    }

    __aicore__ inline void ProcessG1FullOutAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor,
        OutputProducerState &producerState)
    {
        CompactFullChunkIterator iterator{};
        iterator.sequence = cursor.fullStartSequence;
        iterator.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t ordinal = cursor.fullBegin;
             ordinal < cursor.fullEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (LoadCompactFullChunk(plan, iterator, chunk)) {
                ProcessCompactOutAicHeadRange<false>(
                    chunk, 0, HV_, producerState);
            }
        }
    }

    __aicore__ inline void ProcessG1TailOutAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor,
        OutputProducerState &producerState)
    {
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (LoadCompactTailChunk(plan, ordinal, chunk)) {
                ProcessCompactOutAicHeadRange<true>(
                    chunk, 0, HV_, producerState);
            }
        }
    }

    __aicore__ inline void ProcessGroupedFullOutAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor,
        OutputProducerState &producerState)
    {
        CompactGroupedFullTaskIterator iterator{};
        iterator.chunks.sequence = cursor.fullStartSequence;
        iterator.chunks.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t task = cursor.fullBegin;
             task < cursor.fullEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadCompactGroupedFullTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                ProcessCompactOutAicHeadRange<false>(
                    chunk, headBegin, headEnd, producerState);
            }
        }
    }

    __aicore__ inline void ProcessGroupedTailOutAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor,
        OutputProducerState &producerState)
    {
        CompactGroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadCompactGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                ProcessCompactOutAicHeadRange<true>(
                    chunk, headBegin, headEnd, producerState);
            }
        }
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
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        if (plan.IsValid()) {
            KdaForward::ChunkCoreCursor cursor{};
            if (!plan.LoadChunkCoreCursor(
                    static_cast<uint32_t>(coreIdx), cursor)) {
                return;
            }
            OutputConsumerState fullState{};
            OutputConsumerState tailState{};
            if (plan.HeadGroupCount() == 1) {
                ProcessG1FullOutAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum, fullState);
                ResetOutputConsumerState(fullState);
                ProcessG1TailOutAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum, tailState);
            } else {
                ProcessGroupedFullOutAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum, fullState);
                ResetOutputConsumerState(fullState);
                ProcessGroupedTailOutAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum, tailState);
            }
            ResetOutputConsumerState(tailState);
            return;
        }
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        OutputConsumerState consumerState{};
        for (uint64_t task = coreIdx; task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                (void)seq;
                (void)h;
                (void)chunkIdx;
                ProcessChunkOutAiv(
                    b, hv, chunkIdx, start, end, subBlockIdx, subBlockNum,
                    consumerState);
            }
        }
        ResetOutputConsumerState(consumerState);
    }

    __aicore__ inline void ProcessOutAic()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        if (plan.IsValid()) {
            KdaForward::ChunkCoreCursor cursor{};
            if (!plan.LoadChunkCoreCursor(
                    static_cast<uint32_t>(GetBlockIdx()), cursor)) {
                return;
            }
            OutputProducerState fullState{};
            OutputProducerState tailState{};
            if (plan.HeadGroupCount() == 1) {
                ProcessG1FullOutAicPhase(plan, cursor, fullState);
                DrainOutputProducerState(fullState);
                ProcessG1TailOutAicPhase(plan, cursor, tailState);
            } else {
                ProcessGroupedFullOutAicPhase(plan, cursor, fullState);
                DrainOutputProducerState(fullState);
                ProcessGroupedTailOutAicPhase(plan, cursor, tailState);
            }
            DrainOutputProducerState(tailState);
            return;
        }
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        OutputProducerState producerState{};
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                (void)seq;
                (void)h;
                ProcessChunkOutAic(
                    b, hv, chunkIdx, start, end, producerState);
            }
        }
        DrainOutputProducerState(producerState);
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
    float scale_ = 1.0f;
    bool hasInitial_ = false;
    bool isVarLen_ = false;
    bool isAivOnly_ = false;
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    uint64_t outputTileElements_ = 0;
    uint64_t activeOutputSlot_ = 0;
    GM_ADDR compactPlanAddr_ = nullptr;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
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
                qgScaled, aqk, propagatedVNew, propagatedH, stateScratch, userWorkspace, aqk, userWorkspace,
                userWorkspace, localScratch, userWorkspace, userWorkspace, o, propagatedH,
                outputScratch, tiling, &pipe, false);
        op.SetCompactPlan(compactPlan);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdFinalizeKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                qgScaled, aqk, propagatedVNew, propagatedH, stateScratch, userWorkspace, aqk, userWorkspace,
                userWorkspace, localScratch, userWorkspace, userWorkspace, o, propagatedH,
                outputScratch, tiling, &pipe);
        op.SetCompactPlan(compactPlan);
        op.ProcessAiv();
    }
}

} // namespace KdaFinalize
