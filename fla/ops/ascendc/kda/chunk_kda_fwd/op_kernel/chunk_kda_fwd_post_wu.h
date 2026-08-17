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
#include "catlass/gemm/tile/tile_mmad.hpp"
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
constexpr uint32_t KDA_POST_EVENT = 3;
constexpr uint32_t KDA_POST_EVENT_NEXT = 4;
constexpr uint32_t KDA_POST_EVENT_FIX = 5;

using KdaArchTag = Catlass::Arch::AtlasA2;
// compact 完整块的运行时 head 窗口在整个循环中共用一个实例，
// Cube 与 Fixpipe 才能通过两个 L0C 槽真正并行。
using KdaDispatchPolicy = Common::MmadPingpong<KdaArchTag, false, false, 2>;
// 单项路径没有跨调用的 BlockMmad 生命周期，使用单槽避免假双缓冲。
using KdaSingleDispatchPolicy = Common::MmadPingpong<KdaArchTag, false, false, 1>;
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
        inputSequenceMajor_ = tiling.inputSequenceMajor;
        usedCoreNum_ = tiling.postWuUsedCoreNum;
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

    __aicore__ inline void SetCompactPlan(GM_ADDR compactPlan)
    {
        compactPlanAddr_ = compactPlan;
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

    __aicore__ inline bool UseRawKResident() const
    {
        return BT_ == 64 && K_ == 128 && V_ == 128;
    }

    __aicore__ inline LocalTensor<T> ResidentKTyped()
    {
        return gateWritebackBuf_.Get<T>();
    }

    __aicore__ inline void LoadResidentRawK(
        uint64_t b, uint64_t h, uint64_t start, uint64_t curT,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        // resident K 复用前先闭合上一 qHead 的 V 读取；搬入后分别向 V 和
        // MTE3 发布可见性，后续同一窗口只读该槽位。
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        LocalTensor<T> kResident = ResidentKTyped();
        const uint64_t rowBegin = curT * subBlockIdx / subBlockNum;
        const uint64_t rowEnd = curT * (subBlockIdx + 1) / subBlockNum;
        const uint64_t rows = rowEnd - rowBegin;
        // 两个 AIV 各自只搬固定 owner 行，并保留原始 row offset；合起来
        // 每个 distinct qHead 的每个 raw K 行只访问一次 GM。
        if (rows > 0) {
            LocalTensor<T> kOwned = kResident[rowBegin * K_];
            CopyRowsIn(kOwned, k_, QOffset(b, h, start + rowBegin, 0), rows, K_,
                       inputSequenceMajor_ ? H_ * K_ : K_);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        residentRawKActive_ = true;
    }

    __aicore__ inline bool UsePostWuCube(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

    __aicore__ inline bool UseVarlenTailCubeSnapshot(uint64_t curT) const
    {
        return isVarLen_ && curT > 0 && curT < BT_ && BT_ == 64 &&
               K_ == 128 && V_ == 128;
    }

    template <typename WBlockMmad>
    __aicore__ inline void ComputePostWuW(WBlockMmad &wBlockMmad, uint64_t b, uint64_t hv,
                                         uint64_t chunkIdx, uint64_t start, uint64_t curT)
    {
        using ElementA = AKK_T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(BT_, BT_);
        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto tensorA = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start, 0)], layoutA,
                                       Catlass::Arch::PositionGM{});
        LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(BT_, K_);
        LayoutTagC tagC = LayoutTagC::template MakeLayout<float>(BT_, K_);
        auto layoutB = tla::MakeLayoutFromTag(tagB);
        auto layoutC = tla::MakeLayoutFromTag(tagC);
        Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(K_),
                                 static_cast<uint32_t>(curT)};
        auto tensorB = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB,
                                       Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(h_[WScratchOffset(b, hv, chunkIdx, 0, 0)], layoutC,
                                       Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        wBlockMmad(blockA, blockB, blockC, shape);
    }

    template <typename UBlockMmad>
    __aicore__ inline void ComputePostWuU(UBlockMmad &uBlockMmad, uint64_t b, uint64_t hv,
                                         uint64_t start, uint64_t curT)
    {
        using ElementA = AKK_T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(BT_, BT_);
        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto tensorA = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start, 0)], layoutA,
                                       Catlass::Arch::PositionGM{});
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
        uBlockMmad(blockA, blockB, blockC, shape);
    }

    __attribute__((noinline)) __aicore__ void ComputePostWuCube(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                               uint64_t start, uint64_t curT)
    {
        using ElementA = AKK_T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using WTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, float, LayoutTagC>;
        using UTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, OUT_T, LayoutTagC>;
        using PostL1TileShape128 = tla::Shape<KdaInt128, KdaInt128, tla::_256>;
        using PostL0TileShape128 = tla::Shape<KdaInt128, KdaInt128, KdaInt128>;
        using PostL1TileShape256 = tla::Shape<KdaInt128, tla::_256, tla::_256>;
        using PostL0TileShape256 = tla::Shape<KdaInt128, tla::_256, KdaInt64>;
        using WBlockMmad128 = Common::BlockMmadTla<KdaSingleDispatchPolicy, PostL1TileShape128,
                                                   PostL0TileShape128,
                                                   ElementA, ElementB, float, void, WTileCopy>;
        using WBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy, PostL1TileShape256,
                                                   PostL0TileShape256,
                                                   ElementA, ElementB, float, void, WTileCopy>;
        using UBlockMmad128 = Common::BlockMmadTla<KdaSingleDispatchPolicy, PostL1TileShape128,
                                                   PostL0TileShape128,
                                                   ElementA, ElementB, OUT_T, void, UTileCopy>;
        using UBlockMmad256 = Common::BlockMmadTla<KdaWideDispatchPolicy, PostL1TileShape256,
                                                   PostL0TileShape256,
                                                   ElementA, ElementB, OUT_T, void, UTileCopy>;
        {
            Catlass::Arch::Resource<KdaArchTag> wResource;
            if (K_ <= 128) {
                WBlockMmad128 wBlockMmad(wResource);
                ComputePostWuW(wBlockMmad, b, hv, chunkIdx, start, curT);
            } else {
                WBlockMmad256 wBlockMmad(wResource);
                ComputePostWuW(wBlockMmad, b, hv, chunkIdx, start, curT);
            }
            // 单项路径在本作用域排空唯一 L0C 槽，确保 W 已写回 GM。
        }
        {
            Catlass::Arch::Resource<KdaArchTag> uResource;
            if (V_ <= 128) {
                UBlockMmad128 uBlockMmad(uResource);
                ComputePostWuU(uBlockMmad, b, hv, start, curT);
            } else {
                UBlockMmad256 uBlockMmad(uResource);
                ComputePostWuU(uBlockMmad, b, hv, start, curT);
            }
            // U 路径同样在分支作用域内排空，发布信号前写回已完成。
        }
    }

    __attribute__((noinline)) __aicore__ void ComputeCompactPostWuCubeHeadWindow(
        uint64_t b, uint64_t chunkIdx, uint64_t start,
        uint64_t headBase, uint32_t headCnt)
    {
        using ElementA = AKK_T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using WTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, float, LayoutTagC>;
        using UTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, OUT_T, LayoutTagC>;
        using PostL1TileShape128 = tla::Shape<KdaInt128, KdaInt128, tla::_256>;
        using PostL0TileShape128 = tla::Shape<KdaInt128, KdaInt128, KdaInt128>;
        using WBlockMmad = Common::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                PostL0TileShape128,
                                                ElementA, ElementB, float, void, WTileCopy>;
        using UBlockMmad = Common::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                PostL0TileShape128,
                                                ElementA, ElementB, OUT_T, void, UTileCopy>;

        {
            Catlass::Arch::Resource<KdaArchTag> wResource;
            WBlockMmad wBlockMmad(wResource);
            for (uint32_t lane = 0; lane < headCnt; ++lane) {
                ComputePostWuW(wBlockMmad, b, headBase + lane,
                               chunkIdx, start, BT_);
            }
            // 一个 W BlockMmad 覆盖整个运行时 head 窗口，相邻 head 轮转两个 L0C 槽。
        }
        {
            Catlass::Arch::Resource<KdaArchTag> uResource;
            UBlockMmad uBlockMmad(uResource);
            for (uint32_t lane = 0; lane < headCnt; ++lane) {
                ComputePostWuU(uBlockMmad, b, headBase + lane,
                               start, BT_);
            }
            // W 阶段排空后再用独立实例处理 U，U 内部继续轮转两个 L0C 槽。
        }

        // W/U 均已写回后，按真实 head 数在同一 mode2 信号量上发布。
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
        }
    }

    __attribute__((noinline)) __aicore__ void CopyScratchWAndFinalizeKg(
        uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx, uint64_t start, uint64_t curT,
        uint64_t subBlockIdx, uint64_t subBlockNum, bool copyScratchW)
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
            uint64_t token = start + tileRow;

            if (copyScratchW) {
                uint64_t scratchBase = WScratchOffset(b, hv, chunkIdx, tileRow, 0);
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
            }

            LocalTensor<float> kLocal = arena;
            LocalTensor<float> gLocal = arena[elemCount];
            LocalTensor<float> expLocal = arena[2 * elemCount];
            LocalTensor<float> outLocal = arena[3 * elemCount];
            const uint64_t gateOffsetBytes = (typedOffset + elemCount) * sizeof(T);
            LocalTensor<GK_T> gateTyped = vecBuf_.Get<GK_T>()[
                (gateOffsetBytes + sizeof(GK_T) - 1) / sizeof(GK_T)];
            if (!residentRawKActive_) {
                CopyRowsIn(typedLocal, k_, QOffset(b, h, token, 0), tileRows, K_,
                           inputSequenceMajor_ ? H_ * K_ : K_);
            }
            LoadAsFloatVector(gk_, KVOffset(b, hv, token, 0, K_), gLocal, gateTyped, elemCount);
            if (residentRawKActive_) {
                LocalTensor<T> kResident = ResidentKTyped()[tileRow * K_];
                Cast(kLocal, kResident, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            } else {
                Cast(kLocal, typedLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            }
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
            if (residentRawKActive_) {
                LocalTensor<T> kLast = ResidentKTyped()[(curT - 1) * K_];
                CopyVectorOut(kg_, KVOffset(b, hv, last, 0, K_), kLast, K_);
            } else {
                CopyVectorIn(typedLocal, k_, QOffset(b, h, last, 0), K_);
                SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
                WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
                CopyVectorOut(kg_, KVOffset(b, hv, last, 0, K_), typedLocal, K_);
            }
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

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

    __attribute__((noinline)) __aicore__ void ComputeTailWuVector(uint64_t b, uint64_t hv, uint64_t start,
                                                                 uint64_t curT, uint64_t subBlockIdx,
                                                                 uint64_t subBlockNum)
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

    __attribute__((noinline)) __aicore__ void CopyTailSeedRows(uint64_t b, uint64_t hv, uint64_t start,
                                                              uint64_t curT, uint64_t subBlockIdx,
                                                              uint64_t subBlockNum)
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

    __aicore__ inline void ProcessChunkPostAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                               uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                               uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        if (curT < BT_) {
            if (!UseVarlenTailCubeSnapshot(curT)) {
                ComputeTailWuVector(b, hv, start, curT, subBlockIdx, subBlockNum);
                CopyScratchWAndFinalizeKg(
                    b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum, false);
            }
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
        CopyScratchWAndFinalizeKg(
            b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum, true);
    }

    __aicore__ inline void ProcessChunkPostAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                               uint64_t end)
    {
        if constexpr (IsSameType<AKK_T, T>::value) {
            ProcessChunkPostAicTyped(b, hv, chunkIdx, start, end);
        }
    }

    __aicore__ inline void ProcessChunkPostAicTyped(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                    uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        if (curT < BT_) {
            return;
        }
        ComputePostWuCube(b, hv, chunkIdx, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactChunkPostAiv(
        const CompactOwnedChunkDesc &chunk, uint64_t hv,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t curT = BT_;
        if constexpr (IS_TAIL) {
            curT = chunk.end - chunk.start;
        }
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        const uint64_t h = hv / (HV_ / H_);
        if constexpr (IS_TAIL) {
            if (!UseVarlenTailCubeSnapshot(curT)) {
                ComputeTailWuVector(
                    chunk.b, hv, chunk.start, curT,
                    subBlockIdx, subBlockNum);
                CopyScratchWAndFinalizeKg(
                    chunk.b, h, hv, chunk.chunkIdx, chunk.start, curT,
                    subBlockIdx, subBlockNum, false);
            }
        } else {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(
                syncDoneFlag_);
            CopyScratchWAndFinalizeKg(
                chunk.b, h, hv, chunk.chunkIdx, chunk.start, curT,
                subBlockIdx, subBlockNum, true);
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactChunkPostAic(
        const CompactOwnedChunkDesc &chunk, uint64_t hv)
    {
        if constexpr (!IS_TAIL && IsSameType<AKK_T, T>::value) {
            if (!UsePostWuCube(BT_)) {
                return;
            }
            ComputePostWuCube(
                chunk.b, hv, chunk.chunkIdx, chunk.start, BT_);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(
                syncDoneFlag_);
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactPostAivHeadWindow(
        const CompactOwnedChunkDesc &chunk, uint64_t headBase,
        uint32_t headCnt,
        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        // 每个 runtime head 都推进同一个跨核信号量；无论窗口含一个
        // 还是四个 head，循环进度都由同一信号量的计数深度承载。
        uint64_t curT = BT_;
        if constexpr (IS_TAIL) {
            curT = chunk.end - chunk.start;
            if (UseVarlenTailCubeSnapshot(curT)) {
                // snapshot tail 由后续专用阶段完成；本阶段没有消费者，不能
                // 提前加载一次 raw K 后再由真实消费阶段重复加载。
                return;
            }
        }
        const uint64_t headRatio = HV_ / H_;
        uint64_t residentQHead = H_;
        residentRawKActive_ = false;
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            const uint64_t hv = headBase + lane;
            const uint64_t qHead = hv / headRatio;
            if (UseRawKResident() && qHead != residentQHead) {
                // 窗口内相邻 Hv 共享同一 qHead 时复用 raw K；窗口边界
                // 不保留状态，因此 ratio=8 的下一窗口可以重新搬入。
                LoadResidentRawK(
                    chunk.b, qHead, chunk.start, curT,
                    subBlockIdx, subBlockNum);
                residentQHead = qHead;
            }
            ProcessCompactChunkPostAiv<IS_TAIL>(
                chunk, hv, subBlockIdx, subBlockNum);
        }
        residentRawKActive_ = false;
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactPostAivHeadRange(
        const CompactOwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        for (uint64_t headBase = headBegin; headBase < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(headBase), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - headBase) {
                return;
            }
            ProcessCompactPostAivHeadWindow<IS_TAIL>(
                chunk, headBase, headCnt, subBlockIdx, subBlockNum);
            headBase += headCnt;
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactPostAicHeadWindow(
        const CompactOwnedChunkDesc &chunk, uint64_t headBase,
        uint32_t headCnt)
    {
        if constexpr (!IS_TAIL && IsSameType<AKK_T, T>::value) {
            if (BT_ == 64 && K_ == 128 && V_ == 128 &&
                UsePostWuCube(BT_)) {
                ComputeCompactPostWuCubeHeadWindow(
                    chunk.b, chunk.chunkIdx, chunk.start,
                    headBase, headCnt);
                return;
            }
        }
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            ProcessCompactChunkPostAic<IS_TAIL>(chunk, headBase + lane);
        }
    }

    template <bool IS_TAIL>
    __aicore__ inline void ProcessCompactPostAicHeadRange(
        const CompactOwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd)
    {
        for (uint64_t headBase = headBegin; headBase < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(headBase), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - headBase) {
                return;
            }
            ProcessCompactPostAicHeadWindow<IS_TAIL>(
                chunk, headBase, headCnt);
            headBase += headCnt;
        }
    }

    __aicore__ inline void ProcessG1FullPostAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
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
            ProcessCompactPostAivHeadRange<false>(
                chunk, 0, HV_, subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessG1TailPostAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (!LoadCompactTailChunk(plan, ordinal, chunk)) {
                continue;
            }
            ProcessCompactPostAivHeadRange<true>(
                chunk, 0, HV_, subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessGroupedFullPostAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
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
                ProcessCompactPostAivHeadRange<false>(
                    chunk, headBegin, headEnd, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessGroupedTailPostAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        CompactGroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadCompactGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                ProcessCompactPostAivHeadRange<true>(
                    chunk, headBegin, headEnd, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessG1FullPostAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor)
    {
        CompactFullChunkIterator iterator{};
        iterator.sequence = cursor.fullStartSequence;
        iterator.localChunk = cursor.fullStartLocalChunk;
        for (uint64_t ordinal = cursor.fullBegin;
             ordinal < cursor.fullEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (LoadCompactFullChunk(plan, iterator, chunk)) {
                ProcessCompactPostAicHeadRange<false>(chunk, 0, HV_);
            }
        }
    }

    __aicore__ inline void ProcessG1TailPostAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor)
    {
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (LoadCompactTailChunk(plan, ordinal, chunk)) {
                ProcessCompactPostAicHeadRange<true>(chunk, 0, HV_);
            }
        }
    }

    __aicore__ inline void ProcessGroupedFullPostAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor)
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
                ProcessCompactPostAicHeadRange<false>(
                    chunk, headBegin, headEnd);
            }
        }
    }

    __aicore__ inline void ProcessGroupedTailPostAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor)
    {
        CompactGroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (LoadCompactGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                ProcessCompactPostAicHeadRange<true>(
                    chunk, headBegin, headEnd);
            }
        }
    }

    __aicore__ inline void ProcessPostAiv()
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
            if (plan.HeadGroupCount() == 1) {
                ProcessG1FullPostAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
                ProcessG1TailPostAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
            } else {
                ProcessGroupedFullPostAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
                ProcessGroupedTailPostAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
            }
            return;
        }
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
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
                ProcessChunkPostAiv(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessCompactTailSeedAivHeadWindow(
        const CompactOwnedChunkDesc &chunk, uint64_t headBase,
        uint32_t headCnt, uint64_t curT, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            CopyTailSeedRows(
                chunk.b, headBase + lane, chunk.start, curT,
                subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessCompactTailSeedAivHeadRange(
        const CompactOwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd, uint64_t curT, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        for (uint64_t headBase = headBegin; headBase < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(headBase), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - headBase) {
                return;
            }
            ProcessCompactTailSeedAivHeadWindow(
                chunk, headBase, headCnt, curT, subBlockIdx, subBlockNum);
            headBase += headCnt;
        }
    }

    __aicore__ inline void ProcessG1TailSeedAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (!LoadCompactTailChunk(plan, ordinal, chunk)) {
                continue;
            }
            const uint64_t curT = chunk.end - chunk.start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            ProcessCompactTailSeedAivHeadRange(
                chunk, 0, HV_, curT, subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessGroupedTailSeedAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        CompactGroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (!LoadCompactGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                continue;
            }
            const uint64_t curT = chunk.end - chunk.start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            ProcessCompactTailSeedAivHeadRange(
                chunk, headBegin, headEnd, curT,
                subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessVarlenTailSeedCopyAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        if (!isVarLen_) {
            return;
        }
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        const uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        if (plan.IsValid()) {
            KdaForward::ChunkCoreCursor cursor{};
            if (!plan.LoadChunkCoreCursor(
                    static_cast<uint32_t>(coreIdx), cursor)) {
                return;
            }
            if (plan.HeadGroupCount() == 1) {
                ProcessG1TailSeedAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
            } else {
                ProcessGroupedTailSeedAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
            }
            return;
        }
        const uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        const uint64_t taskNum = NT_ * HV_;
        for (uint64_t task = coreIdx; task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (!ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                continue;
            }
            const uint64_t curT = end - start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            (void)seq;
            (void)h;
            (void)chunkIdx;
            CopyTailSeedRows(b, hv, start, curT, subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessCompactTailSnapshotAicHeadWindow(
        const CompactOwnedChunkDesc &chunk, uint64_t headBase,
        uint32_t headCnt, uint64_t curT)
    {
        // 每个 runtime head 都在同一个 flag 上发布一次 ready；流水深度
        // 由 flag 计数控制，而不是由模板展开控制。
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            ComputePostWuCube(
                chunk.b, headBase + lane, chunk.chunkIdx, chunk.start, curT);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(
                syncDoneFlag_);
        }
    }

    __aicore__ inline void ProcessCompactTailSnapshotAicHeadRange(
        const CompactOwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd, uint64_t curT)
    {
        for (uint64_t headBase = headBegin; headBase < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(headBase), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - headBase) {
                return;
            }
            ProcessCompactTailSnapshotAicHeadWindow(
                chunk, headBase, headCnt, curT);
            headBase += headCnt;
        }
    }

    __aicore__ inline void ProcessG1TailSnapshotAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor)
    {
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (!LoadCompactTailChunk(plan, ordinal, chunk)) {
                continue;
            }
            const uint64_t curT = chunk.end - chunk.start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            ProcessCompactTailSnapshotAicHeadRange(chunk, 0, HV_, curT);
        }
    }

    __aicore__ inline void ProcessGroupedTailSnapshotAicPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor)
    {
        CompactGroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (!LoadCompactGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                continue;
            }
            const uint64_t curT = chunk.end - chunk.start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            ProcessCompactTailSnapshotAicHeadRange(
                chunk, headBegin, headEnd, curT);
        }
    }

    __aicore__ inline void ProcessVarlenTailAic()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        if (!isVarLen_) {
            return;
        }
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        if (plan.IsValid()) {
            KdaForward::ChunkCoreCursor cursor{};
            if (!plan.LoadChunkCoreCursor(
                    static_cast<uint32_t>(GetBlockIdx()), cursor)) {
                return;
            }
            if (plan.HeadGroupCount() == 1) {
                ProcessG1TailSnapshotAicPhase(plan, cursor);
            } else {
                ProcessGroupedTailSnapshotAicPhase(plan, cursor);
            }
            return;
        }
        const uint64_t taskNum = NT_ * HV_;
        const uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (!ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                continue;
            }
            const uint64_t curT = end - start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            (void)seq;
            (void)h;
            ComputePostWuCube(b, hv, chunkIdx, start, curT);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
        }
    }

    __aicore__ inline void ProcessCompactTailSnapshotAivHeadWindow(
        const CompactOwnedChunkDesc &chunk, uint64_t headBase,
        uint32_t headCnt, uint64_t curT, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        const uint64_t headRatio = HV_ / H_;
        uint64_t residentQHead = H_;
        residentRawKActive_ = false;
        for (uint32_t lane = 0; lane < headCnt; ++lane) {
            const uint64_t hv = headBase + lane;
            const uint64_t h = hv / headRatio;
            if (UseRawKResident() && h != residentQHead) {
                LoadResidentRawK(
                    chunk.b, h, chunk.start, curT,
                    subBlockIdx, subBlockNum);
                residentQHead = h;
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(
                syncDoneFlag_);
            CopyScratchWAndFinalizeKg(
                chunk.b, h, hv, chunk.chunkIdx, chunk.start, curT,
                subBlockIdx, subBlockNum, true);
        }
        residentRawKActive_ = false;
    }

    __aicore__ inline void ProcessCompactTailSnapshotAivHeadRange(
        const CompactOwnedChunkDesc &chunk, uint64_t headBegin,
        uint64_t headEnd, uint64_t curT, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        for (uint64_t headBase = headBegin; headBase < headEnd;) {
            const uint32_t headCnt = KdaForward::HeadWindowHeadCount(
                static_cast<uint32_t>(headBase), static_cast<uint32_t>(H_),
                static_cast<uint32_t>(HV_));
            if (headCnt == 0 || headCnt > headEnd - headBase) {
                return;
            }
            ProcessCompactTailSnapshotAivHeadWindow(
                chunk, headBase, headCnt, curT, subBlockIdx, subBlockNum);
            headBase += headCnt;
        }
    }

    __aicore__ inline void ProcessG1TailSnapshotAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        for (uint64_t ordinal = cursor.tailBegin;
             ordinal < cursor.tailEnd; ++ordinal) {
            CompactOwnedChunkDesc chunk{};
            if (!LoadCompactTailChunk(plan, ordinal, chunk)) {
                continue;
            }
            const uint64_t curT = chunk.end - chunk.start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            ProcessCompactTailSnapshotAivHeadRange(
                chunk, 0, HV_, curT, subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessGroupedTailSnapshotAivPhase(
        const KdaForward::CompactSequencePlanView &plan,
        const KdaForward::ChunkCoreCursor &cursor, uint64_t subBlockIdx,
        uint64_t subBlockNum)
    {
        CompactGroupedTailTaskIterator iterator{};
        for (uint64_t task = cursor.tailBegin;
             task < cursor.tailEnd; ++task) {
            CompactOwnedChunkDesc chunk{};
            uint64_t headBegin = 0;
            uint64_t headEnd = 0;
            if (!LoadCompactGroupedTailTask(
                    plan, task, iterator, chunk, headBegin, headEnd)) {
                continue;
            }
            const uint64_t curT = chunk.end - chunk.start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            ProcessCompactTailSnapshotAivHeadRange(
                chunk, headBegin, headEnd, curT,
                subBlockIdx, subBlockNum);
        }
    }

    __aicore__ inline void ProcessVarlenTailAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        if (!isVarLen_) {
            return;
        }
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        const uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        if (plan.IsValid()) {
            KdaForward::ChunkCoreCursor cursor{};
            if (!plan.LoadChunkCoreCursor(
                    static_cast<uint32_t>(coreIdx), cursor)) {
                return;
            }
            if (plan.HeadGroupCount() == 1) {
                ProcessG1TailSnapshotAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
            } else {
                ProcessGroupedTailSnapshotAivPhase(
                    plan, cursor, subBlockIdx, subBlockNum);
            }
            return;
        }
        const uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        const uint64_t taskNum = NT_ * HV_;
        for (uint64_t task = coreIdx; task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (!ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                continue;
            }
            const uint64_t curT = end - start;
            if (!UseVarlenTailCubeSnapshot(curT)) {
                continue;
            }
            (void)seq;
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
            CopyScratchWAndFinalizeKg(
                b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum, true);
        }
    }

    __aicore__ inline void ProcessPostAic()
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
            if (plan.HeadGroupCount() == 1) {
                ProcessG1FullPostAicPhase(plan, cursor);
                ProcessG1TailPostAicPhase(plan, cursor);
            } else {
                ProcessGroupedFullPostAicPhase(plan, cursor);
                ProcessGroupedTailPostAicPhase(plan, cursor);
            }
            return;
        }
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
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
                ProcessChunkPostAic(b, hv, chunkIdx, start, end);
            }
        }
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
    bool residentRawKActive_ = false;
    TEventID mte2ToVEvent_ = 0;
    TEventID vToMte2Event_ = 0;
    TEventID vToMte3Event_ = 0;
    TEventID mte3ToVEvent_ = 0;
    TEventID mte2ToMte3Event_ = 0;
    TEventID mte3ToMte2Event_ = 0;
    bool vectorEventsAllocated_ = false;
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreReadyFlag_{KDA_SCORE_READY_FLAG0,
                                                                                  KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreDoneFlag_{KDA_SCORE_DONE_FLAG0,
                                                                                 KDA_SCORE_DONE_FLAG1};
    // Solve 开始前 score 生产已完全排空，因此 solve 握手可以安全复用现有
    // score flag，不额外消耗硬件 flag ID。
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
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    GM_ADDR compactPlanAddr_ = nullptr;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
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
                wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace, akk, w, u,
                userWorkspace, kg, vNew, postScratch, postScratch, tiling, &pipe, false);
        op.SetCompactPlan(compactPlan);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace, akk, w, u,
                userWorkspace, kg, vNew, postScratch, postScratch, tiling, &pipe);
        op.SetCompactPlan(compactPlan);
        op.ProcessAiv();
    }
}

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaPostWuTailSeedCopy(
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
    GM_ADDR wSeed, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    if ASCEND_IS_AIV {
        const uint64_t uSeedBytes = tiling.seqlen * tiling.vHeadNum *
            tiling.vHeadDim * sizeof(T);
        GM_ADDR scratchW = userWorkspace + tiling.outputScratchOffset + uSeedBytes;
        GM_ADDR postScratch = userWorkspace + tiling.postWuScratchOffset;
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(wSeed, wSeed, userWorkspace, userWorkspace, userWorkspace,
                nullptr, cuSeqlens, chunkIndices, wSeed, userWorkspace,
                userWorkspace, nullptr, userWorkspace, userWorkspace, userWorkspace,
                userWorkspace, scratchW, userWorkspace, userWorkspace, userWorkspace,
                userWorkspace, postScratch, postScratch, tiling, &pipe);
        op.SetCompactPlan(compactPlan);
        op.ProcessTailSeedCopyAiv();
    }
}

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaPostWuTail(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
    GM_ADDR initialState, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR compactPlan,
    GM_ADDR akk, GM_ADDR uSeed, GM_ADDR w, GM_ADDR u,
    GM_ADDR kg, GM_ADDR userWorkspace, const TilingData &tiling, TPipe &pipe)
{
    const uint64_t uSeedBytes = tiling.seqlen * tiling.vHeadNum *
        tiling.vHeadDim * sizeof(T);
    GM_ADDR scratchW = userWorkspace + tiling.outputScratchOffset + uSeedBytes;
    GM_ADDR postScratch = userWorkspace + tiling.postWuScratchOffset;
    // Tail Cube 先读取不可变快照，再写入公开的 W 张量。
    if ASCEND_IS_AIC {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                scratchW, akk, uSeed, nullptr, userWorkspace, userWorkspace,
                userWorkspace, akk, w, u, userWorkspace, kg, userWorkspace,
                postScratch, postScratch, tiling, &pipe, false);
        op.SetCompactPlan(compactPlan);
        op.ProcessTailAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                scratchW, akk, uSeed, nullptr, userWorkspace, userWorkspace,
                userWorkspace, akk, w, u, userWorkspace, kg, userWorkspace,
                postScratch, postScratch, tiling, &pipe);
        op.SetCompactPlan(compactPlan);
        op.ProcessTailAiv();
    }
}

} // namespace KdaPostWu
