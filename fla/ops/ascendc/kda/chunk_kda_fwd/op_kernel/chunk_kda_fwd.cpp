/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"
#if __has_include("../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h")
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../../../gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#else
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp"
#else
#include "../../chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"
#endif
#endif

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

using namespace AscendC;
using KdaInt64 = tla::Int<64>;
using KdaInt128 = tla::Int<128>;

namespace {
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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif
using KdaDispatchPolicy = Catlass::Gemm::MmadPingpong<KdaArchTag, true, false>;
using KdaSolveDispatchPolicy = Catlass::Gemm::MmadPingpong<KdaArchTag, true, false>;
static_assert(!KdaSolveDispatchPolicy::USE_HF32_MODE, "KDA triangular solve must use IEEE FP32 Cube mode");
using KdaL1TileShape = tla::Shape<KdaInt64, KdaInt128, KdaInt128>;
using KdaL0TileShape = KdaL1TileShape;
using KdaSolveL1TileShape = tla::Shape<KdaInt64, KdaInt64, KdaInt64>;
using KdaSolveL0TileShape = KdaSolveL1TileShape;

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

enum class KdaPhase : uint8_t {
    PREPARE,
    POST_WU,
    OUTPUT,
};

template <KdaPhase PHASE, bool SAFE_GATE, typename T, typename OUT_T = T, typename AKK_T = float,
          typename GK_T = float, typename BETA_T = float>
class ChunkKdaFwdKernel {
public:
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
        (void)cuSeqlens;
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
        hasChunkIndices_ = chunkIndicesAddr_ != nullptr;
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
        // safe_gate is a numerical implementation mode, not a different KDA formula. Both template
        // instances use the reference-centered score path and FP32 block solve; keeping the flag in
        // the type prevents a runtime branch from entering the chunk hot path.
        hasInitial_ = tiling.hasInitialState;
        isVarLen_ = tiling.isVarLen;
        if constexpr (PHASE == KdaPhase::PREPARE) {
            usedCoreNum_ = tiling.prepareUsedCoreNum;
        } else if constexpr (PHASE == KdaPhase::POST_WU) {
            usedCoreNum_ = tiling.postWuUsedCoreNum;
        } else {
            usedCoreNum_ = tiling.outputUsedCoreNum;
        }
        if constexpr (PHASE == KdaPhase::PREPARE) {
            const uint64_t solveBytes = usedCoreNum_ * KDA_SOLVE_SCRATCH_SLOTS * BT_ * BT_ * sizeof(float);
            const uint64_t alignedSolveBytes =
                (solveBytes + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
            scoreWorkspace_.SetGlobalBuffer((__gm__ T *)(workspace + alignedSolveBytes));
        }
        if constexpr (PHASE == KdaPhase::OUTPUT) {
            const uint64_t outputElements = B_ * HV_ * T_ * V_;
            o_.SetGlobalBuffer((__gm__ OUT_T *)workspace);
            u_.SetGlobalBuffer((__gm__ OUT_T *)workspace + outputElements);
        }
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

    __aicore__ inline void ProcessAivOnly()
    {
        if constexpr (PHASE == KdaPhase::PREPARE) {
            isAivOnly_ = true;
            ProcessPreAiv();
            ReleaseVectorEvents();
            return;
        }
        if constexpr (PHASE == KdaPhase::OUTPUT) {
            isAivOnly_ = true;
            ProcessOutAiv();
            ReleaseVectorEvents();
            return;
        }
        if constexpr (PHASE == KdaPhase::POST_WU) {
            isAivOnly_ = true;
            ProcessPostAiv();
            ReleaseVectorEvents();
            return;
        }
        return;
    }

    __aicore__ inline void ProcessAiv()
    {
        if constexpr (PHASE == KdaPhase::PREPARE) {
            ProcessPreAiv();
            ReleaseVectorEvents();
            return;
        }
        if constexpr (PHASE == KdaPhase::OUTPUT) {
            ProcessOutAiv();
            ReleaseVectorEvents();
            return;
        }
        if constexpr (PHASE == KdaPhase::POST_WU) {
            ProcessPostAiv();
            ReleaseVectorEvents();
            return;
        }
        return;
    }

    __aicore__ inline void ProcessAic()
    {
        if constexpr (PHASE == KdaPhase::PREPARE) {
            ProcessPreAic();
            return;
        }
        if constexpr (PHASE == KdaPhase::OUTPUT) {
            ProcessOutAic();
            return;
        }
        if constexpr (PHASE == KdaPhase::POST_WU) {
            ProcessPostAic();
            return;
        }
        return;
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
        if constexpr (IsSameType<T, half>::value) {
            return 2;
        }
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

    __aicore__ inline void PrepareScoreFactorsBulk(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                    uint64_t subBlockIdx, uint64_t subBlockNum,
                                                    uint64_t refToken, uint64_t scoreRowBegin,
                                                    uint64_t scoreRowCount, uint64_t validColEnd,
                                                    uint64_t scoreSlot)
    {
        LocalTensor<float> refFp32 = exp2Buf_.Get<float>();
        LoadAsFloatRow(gk_, KVOffset(b, hv, refToken, 0, K_), refFp32, K_);

        uint64_t qwBegin = scoreRowBegin + (scoreRowCount * subBlockIdx) / subBlockNum;
        uint64_t qwEnd = scoreRowBegin + (scoreRowCount * (subBlockIdx + 1)) / subBlockNum;
        uint64_t qwMaxRows = ScoreVectorMaxRows(5 * sizeof(float) + 2 * sizeof(T) + sizeof(GK_T));
        for (uint64_t tileRow = qwBegin; tileRow < qwEnd; tileRow += qwMaxRows) {
            uint64_t tileRows = qwEnd - tileRow;
            if (tileRows > qwMaxRows) {
                tileRows = qwMaxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];
            LocalTensor<T> typedBase = gateWritebackBuf_.Get<T>();
            LocalTensor<T> qTyped = typedBase;
            LocalTensor<T> kTyped = typedBase[elems];
            LocalTensor<GK_T> gateTyped =
                gateWritebackBuf_.Get<GK_T>()[3 * elems * sizeof(T) / sizeof(GK_T)];

            uint64_t token = start + tileRow;
            CopyVectorIn(qTyped, q_, QOffset(b, h, token, 0), elems);
            CopyVectorIn(kTyped, k_, QOffset(b, h, token, 0), elems);
            LoadAsFloatVector(gk_, KVOffset(b, hv, token, 0, K_), gFp32, gateTyped, elems);
            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expFp32[row * K_], gFp32[row * K_], refFp32, static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            Cast(qTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            Cast(kTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                          qTyped, elems);
            CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                          kTyped, elems);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }

        uint64_t kgBegin = (validColEnd * subBlockIdx) / subBlockNum;
        uint64_t kgEnd = (validColEnd * (subBlockIdx + 1)) / subBlockNum;
        uint64_t kgMaxRows = ScoreVectorMaxRows(4 * sizeof(float) + sizeof(T) + sizeof(GK_T));
        for (uint64_t tileRow = kgBegin; tileRow < kgEnd; tileRow += kgMaxRows) {
            uint64_t tileRows = kgEnd - tileRow;
            if (tileRows > kgMaxRows) {
                tileRows = kgMaxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> kFp32 = arena;
            LocalTensor<float> gFp32 = arena[elems];
            LocalTensor<float> expFp32 = arena[2 * elems];
            LocalTensor<float> outFp32 = arena[3 * elems];
            LocalTensor<T> kTyped = gateWritebackBuf_.Get<T>();
            LocalTensor<GK_T> gateTyped =
                gateWritebackBuf_.Get<GK_T>()[3 * elems * sizeof(T) / sizeof(GK_T)];

            uint64_t token = start + tileRow;
            CopyVectorIn(kTyped, k_, QOffset(b, h, token, 0), elems);
            LoadAsFloatVector(gk_, KVOffset(b, hv, token, 0, K_), gFp32, gateTyped, elems);
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expFp32[row * K_], refFp32, gFp32[row * K_], static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            Cast(kTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                          kTyped, elems);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }

    __aicore__ inline void PrepareGateProductsBulk(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                   uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum,
                                                   bool useRef, uint64_t refToken, uint64_t validColEnd,
                                                   bool writeScoreScratch, uint64_t scoreSlot)
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum || K_ == 0) {
            return;
        }
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return;
        }

        constexpr uint64_t arenaBytes = static_cast<uint64_t>(KDA_VEC_ARENA_ELEMENTS) * sizeof(float);
        constexpr uint64_t bytesPerElem = 5 * sizeof(float) + 3 * sizeof(T) + sizeof(GK_T);
        uint64_t maxElems = arenaBytes / bytesPerElem;
        uint64_t maxRows = maxElems / K_;
        // Keep the multi-row SIMD tile below the 192 KiB per-core UB budget.
        // K=128 uses five FP32 work planes plus three typed planes; 32 rows
        // leaves headroom for alignment and the surrounding pipeline buffers.
        if (K_ >= 128 && maxRows > 32) {
            maxRows = 32;
        }
        if (maxRows == 0) {
            return;
        }
        LocalTensor<float> refFp32 = exp2Buf_.Get<float>();
        if (useRef) {
            LoadAsFloatRow(gk_, KVOffset(b, hv, refToken, 0, K_), refFp32, K_);
        }

        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxRows) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxRows) {
                tileRows = maxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];

            LocalTensor<T> typedBase = gateWritebackBuf_.Get<T>();
            LocalTensor<T> qTyped = typedBase;
            LocalTensor<T> kTyped = typedBase[elems];
            LocalTensor<T> kgTyped = typedBase[2 * elems];
            LocalTensor<GK_T> gateTyped =
                gateWritebackBuf_.Get<GK_T>()[3 * elems * sizeof(T) / sizeof(GK_T)];

            uint64_t token = start + tileRow;
            CopyVectorIn(qTyped, q_, QOffset(b, h, token, 0), elems);
            CopyVectorIn(kTyped, k_, QOffset(b, h, token, 0), elems);
            LoadAsFloatVector(gk_, KVOffset(b, hv, token, 0, K_), gFp32, gateTyped, elems);

            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            if (useRef) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    Sub(expFp32[row * K_], gFp32[row * K_], refFp32, static_cast<uint32_t>(K_));
                }
            } else {
                Adds(expFp32, gFp32, 0.0f, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            Cast(qTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            Cast(kTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            if (useRef) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    Sub(expFp32[row * K_], refFp32, gFp32[row * K_], static_cast<uint32_t>(K_));
                }
            } else {
                Muls(expFp32, gFp32, -1.0f, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (useRef && tileRow + tileRows > validColEnd) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    if (tileRow + row >= validColEnd) {
                        Duplicate(outFp32[row * K_], 0.0f, static_cast<uint32_t>(K_));
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            Cast(kgTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (writeScoreScratch) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                              qTyped, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                              kTyped, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                              kgTyped, elems);
            } else {
                CopyVectorOut(qg_, KVOffset(b, hv, token, 0, K_), qTyped, elems);
                CopyVectorOut(w_, KVOffset(b, hv, token, 0, K_), kTyped, elems);
                CopyVectorOut(kg_, KVOffset(b, hv, token, 0, K_), kgTyped, elems);
            }
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
        return;
    }

    __aicore__ inline void PrepareGateProducts(uint64_t b, uint64_t h, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum, bool useRef = false,
                                               uint64_t refToken = 0, uint64_t validColEnd = 0,
                                               bool writeScoreScratch = false, uint64_t scoreSlot = 0,
                                               uint64_t scoreRowBegin = 0, uint64_t scoreRowCount = 0)
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum) {
            return;
        }
        if (validColEnd == 0 || validColEnd > curT) {
            validColEnd = curT;
        }
        if (writeScoreScratch) {
            PrepareScoreFactorsBulk(b, h, hv, start, subBlockIdx, subBlockNum, refToken, scoreRowBegin,
                                    scoreRowCount, validColEnd, scoreSlot);
            return;
        }
        PrepareGateProductsBulk(b, h, hv, start, curT, subBlockIdx, subBlockNum, useRef, refToken,
                                validColEnd, writeScoreScratch, scoreSlot);
    }

    __aicore__ inline void ComputeRawAqkAkkCube(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        ComputeRawAqkAkkCubeBlock(b, hv, start, curT, 0, curT);
    }

    __aicore__ inline void ComputeRawAqkAkkCubeBlock(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                                     uint64_t rowBegin, uint64_t rowCount,
                                                     bool readScoreScratch = false, uint64_t scoreSlot = 0,
                                                     uint64_t colCount = 0)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K_, BT_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(BT_, BT_);
        if (colCount == 0 || colCount > curT) {
            colCount = curT;
        }
        Catlass::GemmCoord shape{static_cast<uint32_t>(rowCount), static_cast<uint32_t>(colCount),
                                 static_cast<uint32_t>(K_)};

        auto tensorQPos = readScoreScratch ?
                              tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG)],
                                              layoutA, Catlass::Arch::PositionGM{}) :
                              tla::MakeTensor(qg_[KVOffset(b, hv, start, 0, K_)], layoutA,
                                              Catlass::Arch::PositionGM{});
        auto tensorKPos = readScoreScratch ?
                              tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W)],
                                              layoutA, Catlass::Arch::PositionGM{}) :
                              tla::MakeTensor(w_[KVOffset(b, hv, start, 0, K_)], layoutA,
                                              Catlass::Arch::PositionGM{});
        auto tensorKNeg = readScoreScratch ?
                              tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG)],
                                              layoutB, Catlass::Arch::PositionGM{}) :
                              tla::MakeTensor(kg_[KVOffset(b, hv, start, 0, K_)], layoutB,
                                              Catlass::Arch::PositionGM{});
        auto tensorAqk = tla::MakeTensor(aqk_[AOffset(b, hv, start, 0)], layoutC,
                                         Catlass::Arch::PositionGM{});
        auto tensorAkk = tla::MakeTensor(akk_[AOffset(b, hv, start, 0)], layoutC,
                                         Catlass::Arch::PositionGM{});

        auto blockQPos = GetTile(tensorQPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKPos = GetTile(tensorKPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKNeg = GetTile(tensorKNeg, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.n()));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.n()));

        blockMmad(blockQPos, blockKNeg, blockAqk, shape);
        PipeBarrier<PIPE_ALL>();
        blockMmad(blockKPos, blockKNeg, blockAkk, shape);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool UseAkkCubeSolve(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

    __aicore__ inline bool UsePostWuCube(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

    __aicore__ inline void CopyLocalFloat(LocalTensor<float> dst, LocalTensor<float> src, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        Adds(dst, src, 0.0f, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FillLocalFloat(LocalTensor<float> dst, float value, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        Duplicate(dst, value, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildPrefixMask(LocalTensor<float> dst, uint64_t prefix, uint64_t count)
    {
        if (prefix > count) {
            prefix = count;
        }
        Duplicate(dst, 0.0f, static_cast<uint32_t>(count));
        if (prefix > 0) {
            Duplicate(dst, 1.0f, static_cast<uint32_t>(prefix));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t BuildCausalMask(uint64_t threshold, uint64_t colBegin) const
    {
        if (threshold <= colBegin) {
            return ~0ULL;
        }
        if (threshold >= colBegin + KDA_SOLVE_BT) {
            return 0ULL;
        }
        return ~0ULL << (threshold - colBegin);
    }

    __aicore__ inline void BuildCausalSelectMasks(LocalTensor<uint8_t> aqkMask, LocalTensor<uint8_t> akkMask,
                                                  uint64_t rowBegin, uint64_t rowCount, uint64_t colBegin)
    {
        __ubuf__ uint64_t *aqkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(aqkMask.GetPhyAddr());
        __ubuf__ uint64_t *akkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(akkMask.GetPhyAddr());
        for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
            uint32_t row = static_cast<uint32_t>(rowBegin + localRow);
            aqkMaskPtr[localRow] = BuildCausalMask(static_cast<uint64_t>(row) + 1, colBegin);
            akkMaskPtr[localRow] = BuildCausalMask(static_cast<uint64_t>(row), colBegin);
        }
    }

    __aicore__ inline void SelectCausalRows(LocalTensor<float> aqkMat, LocalTensor<float> akkMat,
                                            uint64_t rowBegin, uint64_t rowCount)
    {
        LocalTensor<uint8_t> aqkMask = vecBuf_.Get<uint8_t>()[KDA_SELECT_AQK_MASK_BYTE_OFFSET];
        LocalTensor<uint8_t> akkMask = vecBuf_.Get<uint8_t>()[KDA_SELECT_AKK_MASK_BYTE_OFFSET];
        LocalTensor<float> zeroLocal = vecBuf_.Get<float>()[KDA_SELECT_ZERO_FLOAT_OFFSET];
        Duplicate(zeroLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();

        uint64_t colBlockCount = (BT_ + KDA_SOLVE_BT - 1) / KDA_SOLVE_BT;
        for (uint64_t colBlock = 0; colBlock < colBlockCount; ++colBlock) {
            uint64_t maskOffset = colBlock * KDA_SELECT_COL_MASK_BYTES;
            uint64_t colBegin = colBlock * KDA_SOLVE_BT;
            BuildCausalSelectMasks(aqkMask[maskOffset], akkMask[maskOffset], rowBegin, rowCount, colBegin);
        }
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);

        uint8_t rowStride = static_cast<uint8_t>(BT_ * sizeof(float) / 32);
        BinaryRepeatParams repeatParams = {1, 0, 1, rowStride, 0, rowStride};
        for (uint64_t colBlock = 0; colBlock < colBlockCount; ++colBlock) {
            uint64_t maskOffset = colBlock * KDA_SELECT_COL_MASK_BYTES;
            uint64_t colBegin = colBlock * KDA_SOLVE_BT;
            Select(aqkMat[colBegin], aqkMask[maskOffset], zeroLocal, aqkMat[colBegin],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_SOLVE_BT, static_cast<uint8_t>(rowCount), repeatParams);
            Select(akkMat[colBegin], akkMask[maskOffset], zeroLocal, akkMat[colBegin],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_SOLVE_BT, static_cast<uint8_t>(rowCount), repeatParams);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
    }

    __aicore__ inline void PrepareAqkAkkSolveInput64(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];

        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, KDA_SOLVE_BT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        DataCopy(aqkMat, aqk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(akkMat, akk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        SelectCausalRows(aqkMat, akkMat, 0, KDA_SOLVE_BT);

        Muls(xMat, akkMat, -1.0f, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[row * KDA_SOLVE_BT], xMat[row * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(aqk_[AOffset(b, hv, start, 0)], aqkMat, KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(akk_[AOffset(b, hv, start, 0)], akkMat, KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void PrepareAqkAkkSolveInputTail(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                       uint64_t start, uint64_t curT)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        DataCopyParams aqkValidParams{1, static_cast<uint16_t>(elemCount * sizeof(float)), 0, 0};
        DataCopyParams akkValidParams{1, static_cast<uint16_t>(elemCount * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];

        FillLocalFloat(betaLocal, 0.0f, KDA_SOLVE_BT);
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, curT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        DataCopyPad(aqkMat, aqk_[AOffset(b, hv, start, 0)], aqkValidParams, padParams);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
            FillLocalFloat(aqkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
        }
        DataCopyPad(akkMat, akk_[AOffset(b, hv, start, 0)], akkValidParams, padParams);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
            FillLocalFloat(akkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
        }

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        SelectCausalRows(aqkMat, akkMat, 0, KDA_SOLVE_BT);

        Muls(xMat, akkMat, -1.0f, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[row * KDA_SOLVE_BT], xMat[row * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopyPad(aqk_[AOffset(b, hv, start, 0)], aqkMat, aqkValidParams);
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0)], akkMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void GetSolveRowRange(uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum,
                                            uint64_t &rowBegin, uint64_t &rowEnd) const
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum) {
            rowBegin = 0;
            rowEnd = 0;
            return;
        }
        rowBegin = (curT * subBlockIdx) / subBlockNum;
        rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
    }

    __aicore__ inline void PrepareAqkAkkSolveInputRows(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                       uint64_t start, uint64_t curT, uint64_t rowBegin,
                                                       uint64_t rowEnd, bool storeLToAkk, bool storeLToScratch)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        uint64_t elemCount = rowCount * BT_;
        uint64_t validElemCount = validRowCount * BT_;
        DataCopyParams aqkValidParams{1, static_cast<uint16_t>(validElemCount * sizeof(float)), 0, 0};
        DataCopyParams akkValidParams{1, static_cast<uint16_t>(validElemCount * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[elemCount];
        LocalTensor<float> xMat = arena[2 * elemCount];
        LocalTensor<float> betaLocal = arena[3 * elemCount];
        LocalTensor<float> betaBrcb = arena[3 * elemCount + BT_];
        LocalTensor<float> maskLocal = arena[3 * elemCount + BT_ + 512];
        LocalTensor<float> oneHotLocal = arena[3 * elemCount + BT_ + 512 + BT_];

        uint64_t token = start + rowBegin;

        FillLocalFloat(aqkMat, 0.0f, elemCount);
        FillLocalFloat(akkMat, 0.0f, elemCount);
        FillLocalFloat(betaLocal, 0.0f, rowCount);
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        if (validRowCount > 0) {
            LoadAsFloatRow(beta_, BetaOffset(b, hv, token), betaLocal, validRowCount);
            DataCopyPad(aqkMat, aqk_[AOffset(b, hv, token, 0)], aqkValidParams, padParams);
            DataCopyPad(akkMat, akk_[AOffset(b, hv, token, 0)], akkValidParams, padParams);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();

        uint8_t rowStride = static_cast<uint8_t>(BT_ * sizeof(float) / 32);
        for (uint64_t col = 0; col < BT_; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, static_cast<uint8_t>(rowCount),
                {1, 1, 0, rowStride, rowStride, 1});
            PipeBarrier<PIPE_V>();
        }
        if (validRowCount > 0) {
            SelectCausalRows(aqkMat, akkMat, rowBegin, validRowCount);
        }

        Muls(xMat, akkMat, -1.0f, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();
        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            BuildPrefixMask(maskLocal, row + 1, BT_);
            BuildPrefixMask(oneHotLocal, row, BT_);
            Sub(maskLocal, maskLocal, oneHotLocal, static_cast<uint32_t>(BT_));
            PipeBarrier<PIPE_V>();
            Add(xMat[localRow * BT_], xMat[localRow * BT_], maskLocal, static_cast<uint32_t>(BT_));
            PipeBarrier<PIPE_V>();
        }

        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t lBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0) + rowBegin * BT_;
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        if (validRowCount > 0) {
            DataCopyPad(aqk_[AOffset(b, hv, token, 0)], aqkMat, aqkValidParams);
            if (storeLToAkk) {
                DataCopyPad(akk_[AOffset(b, hv, token, 0)], akkMat, akkValidParams);
            }
        }
        DataCopy(solveWorkspace_[xBase], xMat, static_cast<uint32_t>(elemCount));
        if (storeLToScratch) {
            DataCopy(solveWorkspace_[lBase], akkMat, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void CubeGemmSolveSub(GlobalTensor<float> &tensorA, uint64_t baseA, uint64_t rowA, uint64_t colA,
                                            GlobalTensor<float> &tensorB, uint64_t baseB, uint64_t rowB, uint64_t colB,
                                            GlobalTensor<float> &tensorC, uint64_t baseC, uint64_t rowC, uint64_t colC,
                                            uint32_t m, uint32_t n, uint32_t k)
    {
        using ElementA = float;
        using ElementB = float;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaSolveDispatchPolicy, KdaSolveL1TileShape,
                                                              KdaSolveL0TileShape, ElementA, ElementB, ElementC,
                                                              void, TileCopy>;
        Catlass::Arch::Resource<KdaArchTag> resource;
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(BT_, BT_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(BT_, BT_);
        auto tensorLayoutA = tla::MakeTensor(tensorA[baseA], layoutA, Catlass::Arch::PositionGM{});
        auto tensorLayoutB = tla::MakeTensor(tensorB[baseB], layoutB, Catlass::Arch::PositionGM{});
        auto tensorLayoutC = tla::MakeTensor(tensorC[baseC], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto blockA = GetTile(tensorLayoutA, tla::MakeCoord(rowA, colA), tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tensorLayoutB, tla::MakeCoord(rowB, colB), tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tensorLayoutC, tla::MakeCoord(rowC, colC), tla::MakeShape(shape.m(), shape.n()));
        BlockMmad blockMmad(resource);
        blockMmad(blockA, blockB, blockC, shape);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AddSolveTmpToX(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                          bool storeAkk)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        if (storeAkk) {
            DataCopy(akk_[AOffset(b, hv, start, 0)], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXTail(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, bool storeAkk)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(float)), 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        if (storeAkk) {
            DataCopyPad(akk_[AOffset(b, hv, start, 0)], xLocal, validParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, uint64_t rowBegin, uint64_t rowEnd, bool storeAkk)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        uint64_t elemCount = rowCount * BT_;
        uint64_t validElemCount = validRowCount * BT_;
        DataCopyParams validParams{1, static_cast<uint16_t>(validElemCount * sizeof(float)), 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[elemCount];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowBegin * BT_;
        uint64_t token = start + rowBegin;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        DataCopy(tmpLocal, solveWorkspace_[tmpBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(solveWorkspace_[xBase], xLocal, static_cast<uint32_t>(elemCount));
        if (storeAkk && validRowCount > 0) {
            DataCopyPad(akk_[AOffset(b, hv, token, 0)], xLocal, validParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXDiagRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                  uint64_t rowBegin, uint64_t rowEnd, bool storeAkk)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t elemCount = rowCount * BT_;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(float)), 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[elemCount];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowBegin * BT_;
        uint64_t token = start + rowBegin;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        DataCopy(tmpLocal, solveWorkspace_[tmpBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            uint64_t col = (row / KDA_SOLVE_DIAG_BT) * KDA_SOLVE_DIAG_BT;
            uint64_t offset = localRow * BT_ + col;
            Add(xLocal[offset], xLocal[offset], tmpLocal[offset], KDA_SOLVE_DIAG_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(solveWorkspace_[xBase], xLocal, static_cast<uint32_t>(elemCount));
        if (storeAkk) {
            DataCopyPad(akk_[AOffset(b, hv, token, 0)], xLocal, validParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void StoreSolveXRowsToAkk(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                uint64_t curT, uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        uint64_t rowCount = rowEnd - rowBegin;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        if (validRowCount == 0) {
            return;
        }
        uint64_t elemCount = validRowCount * BT_;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(float)), 0, 0};
        LocalTensor<float> xLocal = vecBuf_.Get<float>();
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        DataCopyPad(akk_[AOffset(b, hv, start + rowBegin, 0)], xLocal, validParams);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
    }

    __aicore__ inline void ComputeAkkMergeCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aiBase = AOffset(b, hv, start, 0);
        uint64_t negABase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        for (uint32_t mergeSize = 2 * KDA_SOLVE_DIAG_BT; mergeSize <= BT_; mergeSize *= 2) {
            uint32_t half = mergeSize / 2;
            for (uint32_t block = 0; block < BT_; block += mergeSize) {
                uint32_t lower = block + half;
                CubeGemmSolveSub(akk_, aiBase, lower, lower, solveWorkspace_, negABase, lower, block,
                                 solveWorkspace_, tmpBase, 0, 0, half, half, half);
                CubeGemmSolveSub(solveWorkspace_, tmpBase, 0, 0, akk_, aiBase, block, block,
                                 akk_, aiBase, lower, block, half, half, half);
            }
        }
    }

    __aicore__ inline void ComputeAkkMergeCubeWorkspace(uint64_t b, uint64_t hv, uint64_t chunkIdx)
    {
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        for (uint32_t mergeSize = 2 * KDA_SOLVE_DIAG_BT; mergeSize <= BT_; mergeSize *= 2) {
            uint32_t half = mergeSize / 2;
            for (uint32_t block = 0; block < BT_; block += mergeSize) {
                uint32_t lower = block + half;
                CubeGemmSolveSub(solveWorkspace_, xBase, lower, lower, solveWorkspace_, xBase, lower, block,
                                 solveWorkspace_, tmpBase, 0, 0, half, half, half);
                CubeGemmSolveSub(solveWorkspace_, tmpBase, 0, 0, solveWorkspace_, xBase, block, block,
                                 solveWorkspace_, xBase, lower, block, half, half, half);
            }
        }
    }

    __aicore__ inline void ComputeAkkInverseMchFull(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aBase = AOffset(b, hv, start, 0);
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t yBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t yNextBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y1);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        uint32_t diagBlocks = static_cast<uint32_t>(BT_ / KDA_SOLVE_DIAG_BT);
        for (uint32_t block = 0; block < diagBlocks; ++block) {
            uint32_t off = block * KDA_SOLVE_DIAG_BT;
            CubeGemmSolveSub(akk_, aBase, off, off, akk_, aBase, off, off, solveWorkspace_, yBase, off, off,
                             KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
        }
        for (uint32_t iter = 0; iter < KDA_SOLVE_DIAG_MCH_ITERS; ++iter) {
            for (uint32_t block = 0; block < diagBlocks; ++block) {
                uint32_t off = block * KDA_SOLVE_DIAG_BT;
                CubeGemmSolveSub(solveWorkspace_, xBase, off, off, solveWorkspace_, yBase, off, off,
                                 solveWorkspace_, tmpBase, off, off,
                                 KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                for (uint32_t block = 0; block < diagBlocks; ++block) {
                    uint32_t off = block * KDA_SOLVE_DIAG_BT;
                    CubeGemmSolveSub(solveWorkspace_, yBase, off, off, solveWorkspace_, yBase, off, off,
                                     solveWorkspace_, yNextBase, off, off,
                                     KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
                }
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(syncReadyFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                uint64_t oldYBase = yBase;
                yBase = yNextBase;
                yNextBase = oldYBase;
            }
        }
        ComputeAkkMergeCube(b, hv, chunkIdx, start);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
    }

    __aicore__ inline void ComputeAkkInverseMchTail(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                    uint64_t start, uint64_t curT)
    {
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t lBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t yBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y1);
        uint64_t yNextBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);
        (void)start;
        (void)curT;

        uint32_t diagBlocks = static_cast<uint32_t>(BT_ / KDA_SOLVE_DIAG_BT);
        for (uint32_t block = 0; block < diagBlocks; ++block) {
            uint32_t off = block * KDA_SOLVE_DIAG_BT;
            CubeGemmSolveSub(solveWorkspace_, lBase, off, off, solveWorkspace_, lBase, off, off,
                             solveWorkspace_, yBase, off, off,
                             KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
        }
        for (uint32_t iter = 0; iter < KDA_SOLVE_DIAG_MCH_ITERS; ++iter) {
            for (uint32_t block = 0; block < diagBlocks; ++block) {
                uint32_t off = block * KDA_SOLVE_DIAG_BT;
                CubeGemmSolveSub(solveWorkspace_, xBase, off, off, solveWorkspace_, yBase, off, off,
                                 solveWorkspace_, tmpBase, off, off,
                                 KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                for (uint32_t block = 0; block < diagBlocks; ++block) {
                    uint32_t off = block * KDA_SOLVE_DIAG_BT;
                    CubeGemmSolveSub(solveWorkspace_, yBase, off, off, solveWorkspace_, yBase, off, off,
                                     solveWorkspace_, yNextBase, off, off,
                                     KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
                }
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(syncReadyFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                uint64_t oldYBase = yBase;
                yBase = yNextBase;
                yNextBase = oldYBase;
            }
        }
        ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
    }



    __aicore__ inline void ScaleRowsByBeta(GlobalTensor<T> &src, GlobalTensor<T> &dst, uint64_t b, uint64_t hv,
                                           uint64_t start, uint64_t rowBegin, uint64_t rowCount, uint64_t dim,
                                           LocalTensor<float> &betaBrcb, LocalTensor<float> &matrixLocal)
    {
        constexpr uint64_t vecElemsPerRepeat = 64;
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t elemCount = rowCount * dim;
        uint64_t baseOffset = KVOffset(b, hv, start + rowBegin, 0, dim);

        if constexpr (IsSameType<T, float>::value) {
            DataCopy(matrixLocal, src[baseOffset], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            DataCopy(matrixTyped, src[baseOffset], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(matrixLocal, matrixTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
        }

        uint8_t repeatStride = static_cast<uint8_t>(dim * sizeof(float) / 32);
        for (uint64_t col = 0; col < dim; col += vecElemsPerRepeat) {
            uint64_t mask = dim - col;
            if (mask > vecElemsPerRepeat) {
                mask = vecElemsPerRepeat;
            }
            Mul(matrixLocal[col], matrixLocal[col], betaBrcb, mask, static_cast<uint8_t>(rowCount),
                {1, 1, 0, repeatStride, repeatStride, 1});
            PipeBarrier<PIPE_V>();
        }

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(dst[baseOffset], matrixLocal, static_cast<uint32_t>(elemCount));
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            Cast(matrixTyped, matrixLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(dst[baseOffset], matrixTyped, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void PrepareWuCubeInputs(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t rowsPerSubBlock = (curT + subBlockNum - 1) / subBlockNum;
        uint64_t rowBegin = subBlockIdx * rowsPerSubBlock;
        if (rowBegin >= curT) {
            return;
        }
        uint64_t rowCount = curT - rowBegin;
        if (rowCount > rowsPerSubBlock) {
            rowCount = rowsPerSubBlock;
        }
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> betaLocal = arena;
        LocalTensor<float> betaBrcb = arena[KDA_SOLVE_BT];
        LocalTensor<float> matrixLocal = arena[KDA_SOLVE_BT + 512];
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start + rowBegin), betaLocal, rowCount);
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        ScaleRowsByBeta(w_, w_, b, hv, start, rowBegin, rowCount, K_, betaBrcb, matrixLocal);
        ScaleRowsByBeta(v_, vNew_, b, hv, start, rowBegin, rowCount, V_, betaBrcb, matrixLocal);
    }

    __aicore__ inline void FinalizePrepareIntermediates(uint64_t b, uint64_t hv, uint64_t start,
                                                        uint64_t curT, uint64_t subBlockIdx,
                                                        uint64_t subBlockNum)
    {
        constexpr uint64_t tileRows = 32;
        // Keep tail rows on the same AIV that owns their padded solve rows. Splitting by curT would
        // move short-tail export to AIV1 while AIV0 is still writing the solved matrix.
        const uint64_t rowBegin = (BT_ * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (BT_ * (subBlockIdx + 1)) / subBlockNum;
        if (rowEnd > curT) {
            rowEnd = curT;
        }
        if (rowBegin >= rowEnd) {
            return;
        }
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += tileRows) {
            const uint64_t rows = (rowEnd - tileRow) > tileRows ? tileRows : (rowEnd - tileRow);
            const uint64_t matrixElems = rows * BT_;
            const uint64_t qgElems = rows * K_;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> aqkLocal = arena;
            LocalTensor<float> akkLocal = arena[matrixElems];
            LocalTensor<float> qgLocal = arena[2 * matrixElems];
            const uint64_t typedOffset =
                (2 * matrixElems + qgElems) * sizeof(float) / sizeof(T);
            LocalTensor<T> typedBase = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> aqkTyped = typedBase;
            LocalTensor<T> akkTyped = typedBase[matrixElems];
            LocalTensor<T> qgTyped = typedBase[2 * matrixElems];

            CopyVectorIn(aqkLocal, aqk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            CopyVectorIn(akkLocal, akk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            CopyVectorIn(qgTyped, qg_, KVOffset(b, hv, start + tileRow, 0, K_), qgElems);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

            Muls(aqkLocal, aqkLocal, scale_, static_cast<uint32_t>(matrixElems));
            Cast(qgLocal, qgTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(qgElems));
            PipeBarrier<PIPE_V>();
            Muls(qgLocal, qgLocal, scale_, static_cast<uint32_t>(qgElems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(aqkLocal, static_cast<uint32_t>(matrixElems));
            ClampFp32ToOutputType(akkLocal, static_cast<uint32_t>(matrixElems));
            ClampFp32ToOutputType(qgLocal, static_cast<uint32_t>(qgElems));
            Cast(aqkTyped, aqkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            Cast(akkTyped, akkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            Cast(qgTyped, qgLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(qgElems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(o_, AOffset(b, hv, start + tileRow, 0), aqkTyped, matrixElems);
            CopyVectorOut(u_, AOffset(b, hv, start + tileRow, 0), akkTyped, matrixElems);
            CopyVectorOut(kg_, KVOffset(b, hv, start + tileRow, 0, K_), qgTyped, qgElems);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }

    __aicore__ inline void ComputePostWuCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT)
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
        using WBlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                               PostL0TileShape128,
                                                               ElementA, ElementB, float, void, WTileCopy>;
        using UBlockMmad128 = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                                  PostL0TileShape128,
                                                                  ElementA, ElementB, OUT_T, void, UTileCopy>;
        using UBlockMmad256 = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape256,
                                                                  PostL0TileShape256,
                                                                  ElementA, ElementB, OUT_T, void, UTileCopy>;

        LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(BT_, BT_);
        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto tensorA = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start, 0)], layoutA,
                                       Catlass::Arch::PositionGM{});

        {
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
            Catlass::Arch::Resource<KdaArchTag> wResource;
            WBlockMmad wBlockMmad(wResource);
            wBlockMmad(blockA, blockB, blockC, shape);
            PipeBarrier<PIPE_ALL>();
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
            PipeBarrier<PIPE_ALL>();
        }

    }

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
            uint64_t scratchBase = WScratchOffset(b, hv, chunkIdx, tileRow, 0);
            uint64_t token = start + tileRow;

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

            LocalTensor<float> kLocal = arena;
            LocalTensor<float> gLocal = arena[elemCount];
            LocalTensor<float> expLocal = arena[2 * elemCount];
            LocalTensor<float> outLocal = arena[3 * elemCount];
            const uint64_t gateOffsetBytes = (typedOffset + elemCount) * sizeof(T);
            LocalTensor<GK_T> gateTyped = vecBuf_.Get<GK_T>()[
                (gateOffsetBytes + sizeof(GK_T) - 1) / sizeof(GK_T)];
            CopyVectorIn(typedLocal, k_, QOffset(b, h, token, 0), elemCount);
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
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
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
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
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
                auto tensorO = tla::MakeTensor(o_[KVOffset(b, hv, start + mOffset, nOffset, V_)], layoutO,
                                               Catlass::Arch::PositionGM{});
                auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.k()));
                auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.k(), shapeQH.n()));
                auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.n()));
                blockMmad(blockQ, blockH, blockO, shapeQH);
                PipeBarrier<PIPE_ALL>();
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
                auto tensorLocal = tla::MakeTensor(u_[KVOffset(b, hv, start + mOffset, nOffset, V_)], layoutO,
                                                   Catlass::Arch::PositionGM{});
                auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.k()));
                auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.k(), shapeAV.n()));
                auto blockLocal = GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.n()));
                blockMmad(blockAqk, blockVNew, blockLocal, shapeAV);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void FinalizeOutputRows(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                              uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        LocalTensor<float> stateLocal = VecScratch(0);
        LocalTensor<float> localLocal = VecScratch(1);
        LocalTensor<float> outLocal = VecScratch(2);
        for (uint64_t i = subBlockIdx; i < curT; i += subBlockNum) {
            uint64_t ti = start + i;
            LoadAsFloatRow(o_, KVOffset(b, hv, ti, 0, V_), stateLocal, V_);
            LoadAsFloatRow(u_, KVOffset(b, hv, ti, 0, V_), localLocal, V_);
            Add(outLocal, stateLocal, localLocal, static_cast<uint32_t>(V_));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outLocal, static_cast<uint32_t>(V_));
            StoreFloatRow(vNew_, KVOffset(b, hv, ti, 0, V_), outLocal, V_);
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
            if (hasChunkIndices_) {
                const uint64_t metadataOffset = flatChunk * 4;
                seq = static_cast<uint64_t>(chunkIndicesAddr_[metadataOffset]);
                start = static_cast<uint64_t>(chunkIndicesAddr_[metadataOffset + 1]);
                end = static_cast<uint64_t>(chunkIndicesAddr_[metadataOffset + 2]);
                b = 0;
                chunkIdx = flatChunk;
                if (seq >= N_ || start >= end || end > T_) {
                    return false;
                }
                h = hv / (HV_ / H_);
                return start < end;
            }
            return false;
        }
        h = hv / (HV_ / H_);
        return start < end;
    }

    __aicore__ inline void ProcessChunkPreAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                              uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                              uint64_t subBlockNum)
    {
        if constexpr (IsSameType<AKK_T, float>::value) {
            ProcessChunkPreAivFp32(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void RunAicAfterBothAivReady(uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            (void)subBlockIdx;
            (void)subBlockNum;
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(syncReadyFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
        }
    }

    __aicore__ inline void ProcessChunkPreAivFp32(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                                  uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                                  uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        }

        if (K_ < 16) {
            return;
        }
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(BT_, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
        uint64_t scoreBlockSize = ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount =
            (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = ScoreRowBlockCount(curT, rowBegin);
                uint64_t refToken = ScoreRefToken(start, curT, rowBegin, rowCount);
                PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum, true, refToken,
                                    rowBegin + rowCount, true, block % KDA_SCORE_QUEUE_DEPTH,
                                    rowBegin, rowCount);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
            if (block > 0) {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
            }
        }
        if (pipelineBlockCount > 0) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
        }
        PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum);
        if (useAkkCubeSolve) {
            bool fullChunk = curT == BT_;
            PrepareAqkAkkSolveInputRows(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd,
                                        fullChunk, !fullChunk);
        }
        if (useAkkCubeSolve) {
            bool fullChunk = curT == BT_;
            uint32_t solveIters = KDA_SOLVE_DIAG_MCH_ITERS;
            RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
            for (uint32_t iter = 0; iter < solveIters; ++iter) {
                AddSolveTmpToXDiagRows(b, hv, chunkIdx, start, solveRowBegin, solveRowEnd,
                                       fullChunk && iter + 1 == solveIters);
                RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
            }
            if (!fullChunk) {
                StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
            }
        }
        // Host validation guarantees every accepted shape has enough workspace for this cube path.
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        FinalizePrepareIntermediates(b, hv, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void ProcessChunkPreAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t end)
    {
        if constexpr (IsSameType<AKK_T, float>::value) {
            ProcessChunkPreAicFp32(b, hv, chunkIdx, start, end);
        }
    }

    __aicore__ inline void ProcessChunkPreAicFp32(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                  uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || K_ < 16) {
            return;
        }
        uint64_t scoreBlockSize = ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount =
            (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = ScoreRowBlockCount(curT, rowBegin);
                ComputeRawAqkAkkCubeBlock(b, hv, start, curT, rowBegin, rowCount, true,
                                          block % KDA_SCORE_QUEUE_DEPTH, rowBegin + rowCount);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        }
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        if (useAkkCubeSolve) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(syncReadyFlag_);
            if (curT == BT_) {
                ComputeAkkInverseMchFull(b, hv, chunkIdx, start);
            } else {
                ComputeAkkInverseMchTail(b, hv, chunkIdx, start, curT);
            }
        }
        (void)usePostWuCube;
        (void)chunkIdx;
    }

    __aicore__ inline void ProcessChunkPostAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                               uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                               uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
        CopyScratchWAndFinalizeKg(b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
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
        ComputePostWuCube(b, hv, chunkIdx, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
    }

    __aicore__ inline void ProcessChunkOutAiv(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t end, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
        FinalizeOutputRows(b, hv, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void ProcessChunkOutAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        ComputeOutputCube(b, hv, chunkIdx, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
    }

    __aicore__ inline void ProcessPreAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            isAivOnly_ = true;
        }
        uint64_t subBlockNum = isAivOnly_ ? 1 : static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        uint64_t subBlockIdx = isAivOnly_ ? 0 : static_cast<uint64_t>(GetSubBlockIdx());
        uint64_t coreNum = isAivOnly_ ? static_cast<uint64_t>(GetBlockNum()) : usedCoreNum_;
        uint64_t coreIdx = isAivOnly_ ? static_cast<uint64_t>(GetBlockIdx()) :
                                        static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
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
                ProcessChunkPreAiv(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessPreAic()
    {
        if constexpr (IsSameType<T, float>::value) {
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
                ProcessChunkPreAic(b, hv, chunkIdx, start, end);
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
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
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

    __aicore__ inline void ProcessPostAic()
    {
        if constexpr (IsSameType<T, float>::value) {
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
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
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
                (void)h;
                (void)chunkIdx;
                ProcessChunkOutAiv(b, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessOutAic()
    {
        if constexpr (IsSameType<T, float>::value) {
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
                ProcessChunkOutAic(b, hv, chunkIdx, start, end);
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
    // Score production is fully drained before solve starts, so the solve handshake can safely reuse
    // the A2/A5-proven score flags without consuming additional hardware flag IDs.
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
    bool hasChunkIndices_ = false;
    bool isAivOnly_ = false;
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
};
} // namespace

template <bool SAFE_GATE, typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaFused(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR o, GM_ADDR finalState,
    GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg,
    GM_ADDR vNew, GM_ADDR h, GM_ADDR userWorkspace, const TilingData& tiling, TPipe& pipe)
{
    GM_ADDR qgScaled = userWorkspace + tiling.prepareQgScaledOffset;
    GM_ADDR wSeed = userWorkspace + tiling.prepareWSeedOffset;
    GM_ADDR uSeed = userWorkspace + tiling.prepareUSeedOffset;
    GM_ADDR aqkFp32 = userWorkspace + tiling.prepareAqkFp32Offset;
    GM_ADDR akkFp32 = userWorkspace + tiling.prepareAkkFp32Offset;
    GM_ADDR prepareScratch = userWorkspace + tiling.prepareScratchOffset;

    {
        if ASCEND_IS_AIC {
            ChunkKdaFwdKernel<KdaPhase::PREPARE, SAFE_GATE, T, T, float, GK_T, BETA_T> op;
            op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                    nullptr, nullptr, nullptr, nullptr, aqk, finalState, aqkFp32, akkFp32,
                    wSeed, akk, qg, qgScaled, uSeed, h, prepareScratch, tiling, &pipe, false);
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            ChunkKdaFwdKernel<KdaPhase::PREPARE, SAFE_GATE, T, T, float, GK_T, BETA_T> op;
            op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                    nullptr, nullptr, nullptr, nullptr, aqk, finalState, aqkFp32, akkFp32,
                    wSeed, akk, qg, qgScaled, uSeed, h, prepareScratch, tiling, &pipe);
            op.ProcessAiv();
        }
    }
    pipe.Reset();
    AscendC::SyncAll<false>();

    GM_ADDR postScratch = userWorkspace + tiling.postWuScratchOffset;
    {
        if ASCEND_IS_AIC {
            ChunkKdaFwdKernel<KdaPhase::POST_WU, false, T, T, T, GK_T, BETA_T> op;
            op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                    wSeed, akk, uSeed, nullptr, o, finalState, aqk, akk, w, u,
                    qg, kg, vNew, postScratch, postScratch, tiling, &pipe, false);
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            ChunkKdaFwdKernel<KdaPhase::POST_WU, false, T, T, T, GK_T, BETA_T> op;
            op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                    wSeed, akk, uSeed, nullptr, o, finalState, aqk, akk, w, u,
                    qg, kg, vNew, postScratch, postScratch, tiling, &pipe);
            op.ProcessAiv();
        }
    }
    pipe.Reset();
    AscendC::SyncAll<false>();

    ChunkGatedDeltaRuleFwdHTilingData fwdHTiling{};
    fwdHTiling.batch = tiling.fwdHBatch;
    fwdHTiling.seqlen = tiling.fwdHSeqlen;
    fwdHTiling.kNumHead = tiling.fwdHKNumHead;
    fwdHTiling.vNumHead = tiling.fwdHVNumHead;
    fwdHTiling.kHeadDim = tiling.fwdHKHeadDim;
    fwdHTiling.vHeadDim = tiling.fwdHVHeadDim;
    fwdHTiling.chunkSize = tiling.fwdHChunkSize;
    fwdHTiling.useInitialState = tiling.fwdHUseInitialState;
    fwdHTiling.storeFinalState = tiling.fwdHStoreFinalState;
    fwdHTiling.isVariedLen = tiling.fwdHIsVariedLen;
    fwdHTiling.shapeBatch = tiling.fwdHShapeBatch;
    fwdHTiling.tokenBatch = tiling.fwdHTokenBatch;
    fwdHTiling.vWorkspaceOffset = tiling.fwdHVWorkspaceOffset;
    fwdHTiling.vUpdateWorkspaceOffset = tiling.fwdHVUpdateWorkspaceOffset;
    fwdHTiling.kDecayWorkspaceOffset = tiling.fwdHKDecayWorkspaceOffset;
    fwdHTiling.hWorkspaceOffset = tiling.fwdHHWorkspaceOffset;
    fwdHTiling.numSeqWorkspaceOffset = tiling.fwdHNumSeqWorkspaceOffset;
    fwdHTiling.numChunksWorkspaceOffset = tiling.fwdHNumChunksWorkspaceOffset;
    GM_ADDR fwdHWorkspace = userWorkspace + tiling.fwdHWorkspaceBaseOffset;
    if (tiling.vHeadDim > 128) {
        using FwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
            T, GK_T, float, float, Catlass::Gemm::Kernel::GDNFwdHTileShapes256, true, false>;
        FwdHKernel stateOp;
        stateOp.InitFromData(kg, w, u, gk, gk, initialState, cuSeqlens, chunkIndices,
                             h, vNew, finalState, fwdHTiling, fwdHWorkspace);
        stateOp.Process();
    } else {
        using FwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
            T, GK_T, float, float, Catlass::Gemm::Kernel::GDNFwdHTileShapes128, true, false>;
        FwdHKernel stateOp;
        stateOp.InitFromData(kg, w, u, gk, gk, initialState, cuSeqlens, chunkIndices,
                             h, vNew, finalState, fwdHTiling, fwdHWorkspace);
        stateOp.Process();
    }
    pipe.Reset();
    AscendC::SyncAll<false>();

    GM_ADDR outputScratch = userWorkspace + tiling.outputScratchOffset;
    {
        if ASCEND_IS_AIC {
            ChunkKdaFwdKernel<KdaPhase::OUTPUT, false, T, float, float, GK_T, BETA_T> op;
            op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                    qgScaled, aqk, vNew, h, o, finalState, aqk, akk, w, u,
                    qg, kg, o, h, outputScratch, tiling, &pipe, false);
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            ChunkKdaFwdKernel<KdaPhase::OUTPUT, false, T, float, float, GK_T, BETA_T> op;
            op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                    qgScaled, aqk, vNew, h, o, finalState, aqk, akk, w, u,
                    qg, kg, o, h, outputScratch, tiling, &pipe);
            op.ProcessAiv();
        }
    }
}

#ifndef KDA_FAST_KERNEL_LAUNCH
extern "C" __global__ __aicore__ void chunk_kda_fwd(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
                                                      GM_ADDR initial_state, GM_ADDR cu_seqlens,
                                                      GM_ADDR chunk_indices, GM_ADDR o, GM_ADDR final_state,
                                                      GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u,
                                                      GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        TPipe pipe;
        if (tilingData.safeGate) {
            RunChunkKdaFused<true, DTYPE_Q, DTYPE_GK, DTYPE_BETA>(
                q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, o, final_state,
                aqk, akk, w, u, qg, kg, v_new, h, userWS, tilingData, pipe);
        } else {
            RunChunkKdaFused<false, DTYPE_Q, DTYPE_GK, DTYPE_BETA>(
                q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, o, final_state,
                aqk, akk, w, u, qg, kg, v_new, h, userWS, tilingData, pipe);
        }
    }
}
#endif
