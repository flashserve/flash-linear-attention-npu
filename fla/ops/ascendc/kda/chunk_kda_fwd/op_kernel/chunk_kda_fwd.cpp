/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include <type_traits>
using _128 = tla::Int<128>;
#else
#define CATLASS_ARCH 2201
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#endif

#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

using namespace AscendC;

namespace {
constexpr float LN2 = 0.69314718055994530942f;
constexpr float KDA_EXP_MAX = 80.0f;
constexpr uint32_t EXP2_UB_ELEMENTS = 256;
constexpr uint32_t EXP2_EVENT_ID = 0;
constexpr uint32_t KDA_MTE2_V_EVENT_ID = 1;
constexpr uint32_t KDA_SCALAR_MTE2_V_EVENT_ID = 2;
constexpr uint32_t KDA_SCALAR_V_S_EVENT_ID = 3;
constexpr uint32_t KDA_SCALAR_V_MTE3_EVENT_ID = 4;
constexpr uint32_t KDA_SCALAR_MTE3_V_EVENT_ID = 5;
constexpr uint32_t KDA_MTE2_MTE3_EVENT_ID = 6;
constexpr uint32_t KDA_MTE3_MTE2_EVENT_ID = 7;
constexpr uint32_t KDA_VEC_BUFFER_NUM = 2;
constexpr uint32_t KDA_SOLVE_BT = 64;
constexpr uint32_t KDA_SOLVE_MATRIX_ELEMENTS = KDA_SOLVE_BT * KDA_SOLVE_BT;
constexpr uint32_t KDA_SOLVE_SCRATCH_X = 0;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y0 = 1;
constexpr uint32_t KDA_SOLVE_SCRATCH_TMP = 2;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y1 = 3;
constexpr uint32_t KDA_SOLVE_SCRATCH_SLOTS = 4;
constexpr uint32_t KDA_SOLVE_MCH_ITERS = 2;
constexpr uint32_t KDA_SOLVE_MXR_DIAG_ITERS = 1;
constexpr uint32_t KDA_SOLVE_MXR_BLOCK = 16;
constexpr uint32_t KDA_SOLVE_FRAC = 16;
constexpr uint32_t KDA_SOLVE_FRAC_ELEMENTS = KDA_SOLVE_FRAC * KDA_SOLVE_FRAC;
constexpr uint32_t KDA_SOLVE_L1_SLOT_A = 0;
constexpr uint32_t KDA_SOLVE_L1_SLOT_B = 1;
constexpr uint32_t KDA_SOLVE_L1_SLOT_TMP = 2;
constexpr uint32_t KDA_SOLVE_L1_SLOT_COUNT = 3;
constexpr uint32_t KDA_SOLVE_EVT_COPY = 0;
constexpr uint32_t KDA_SOLVE_EVT_MMAD = 1;
constexpr uint32_t KDA_VEC_ARENA_ELEMENTS = 32768;
constexpr uint32_t KDA_MASK_BITS_PER_BYTE = 8;
constexpr uint32_t KDA_SELECT_ZERO_FLOATS = 8;
constexpr uint32_t KDA_SOLVE_MASK_OFFSET =
    3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512;
constexpr uint32_t KDA_SOLVE_ONEHOT_OFFSET = KDA_SOLVE_MASK_OFFSET + KDA_SOLVE_BT;
constexpr uint32_t KDA_SELECT_ZERO_OFFSET = KDA_SOLVE_ONEHOT_OFFSET + KDA_SOLVE_BT;
constexpr uint32_t KDA_SELECT_MASK_BYTE_OFFSET =
    (KDA_SELECT_ZERO_OFFSET + KDA_SELECT_ZERO_FLOATS) * sizeof(float);
constexpr uint32_t KDA_LOCAL_MATRIX_BASE = EXP2_UB_ELEMENTS * 8;
constexpr uint32_t KDA_LOCAL_AQK_OFFSET = KDA_LOCAL_MATRIX_BASE;
constexpr uint32_t KDA_LOCAL_AKK_OFFSET = KDA_LOCAL_AQK_OFFSET + KDA_SOLVE_MATRIX_ELEMENTS;
constexpr uint8_t KDA_SCORE_DONE_FLAG0 = 2;
constexpr uint8_t KDA_SCORE_DONE_FLAG1 = 3;
constexpr uint8_t KDA_SCORE_READY_FLAG0 = 4;
constexpr uint8_t KDA_SCORE_READY_FLAG1 = 5;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif
using KdaDispatchPolicy = Catlass::Gemm::MmadPingpong<KdaArchTag, true, false>;
using KdaL1TileShape = tla::Shape<_128, _128, _128>;
using KdaL0TileShape = KdaL1TileShape;
using KdaFloatL1TileShape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
using KdaFloatL0TileShape = KdaFloatL1TileShape;
using KdaPostL1TileShape = tla::Shape<_128, _128, _256>;
using KdaPostL0TileShape = tla::Shape<_128, _128, _128>;
using KdaFloatPostL1TileShape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<128>>;
using KdaFloatPostL0TileShape = KdaFloatL0TileShape;

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

template <typename T, typename OUT_T = T>
class ChunkKdaFwdKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqkInput, GM_ADDR akkInput,
                                GM_ADDR wInput, GM_ADDR uInput, GM_ADDR qgInput, GM_ADDR kgInput,
                                GM_ADDR vNewInput, GM_ADDR hInput, GM_ADDR o, GM_ADDR finalState,
                                GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg,
                                GM_ADDR vNew, GM_ADDR h, const ChunkKdaFwdTilingData &tiling, TPipe *pipe,
                                bool initVecBuffers = true)
    {
        pipe_ = pipe;
        stage_ = tiling.stage;
        q_.SetGlobalBuffer((__gm__ T *)q);
        k_.SetGlobalBuffer((__gm__ T *)k);
        v_.SetGlobalBuffer((__gm__ T *)v);
        gk_.SetGlobalBuffer((__gm__ float *)gk);
        beta_.SetGlobalBuffer((__gm__ float *)beta);
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer((__gm__ float *)initialState);
        }
        if (cuSeqlens != nullptr) {
            cuSeqlens_.SetGlobalBuffer((__gm__ int64_t *)cuSeqlens);
        }
        if (chunkIndices != nullptr) {
            chunkIndices_.SetGlobalBuffer((__gm__ int64_t *)chunkIndices);
        }
        hasChunkIndices_ = chunkIndices != nullptr;
        o_.SetGlobalBuffer((__gm__ OUT_T *)o);
        finalState_.SetGlobalBuffer((__gm__ float *)finalState);
        GM_ADDR aqkBuffer = (stage_ == 2 && aqkInput != nullptr) ? aqkInput : aqk;
        GM_ADDR akkBuffer = (stage_ == 2 && akkInput != nullptr) ? akkInput : akk;
        GM_ADDR wBuffer = (stage_ == 2 && wInput != nullptr) ? wInput : w;
        GM_ADDR uBuffer = (stage_ == 4 && uInput != nullptr) ? uInput : u;
        GM_ADDR qgBuffer = (stage_ == 2 && qgInput != nullptr) ? qgInput : qg;
        GM_ADDR kgBuffer = (stage_ == 2 && kgInput != nullptr) ? kgInput : kg;
        GM_ADDR vNewBuffer = (stage_ == 2 && vNewInput != nullptr) ? vNewInput : vNew;
        GM_ADDR hBuffer = (stage_ == 2 && hInput != nullptr) ? hInput : h;
        if (stage_ == 4) {
            wBuffer = wInput != nullptr ? wInput : w;
            kgBuffer = kgInput != nullptr ? kgInput : kg;
        }
        aqk_.SetGlobalBuffer((__gm__ T *)aqkBuffer);
        akk_.SetGlobalBuffer((__gm__ T *)akkBuffer);
        w_.SetGlobalBuffer((__gm__ T *)wBuffer);
        u_.SetGlobalBuffer((__gm__ OUT_T *)uBuffer);
        qg_.SetGlobalBuffer((__gm__ T *)qgBuffer);
        kg_.SetGlobalBuffer((__gm__ T *)kgBuffer);
        vNew_.SetGlobalBuffer((__gm__ T *)vNewBuffer);
        h_.SetGlobalBuffer((__gm__ T *)hBuffer);

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
        usedCoreNum_ = tiling.usedCoreNum;

        if (pipe_ != nullptr) {
            pipe_->InitBuffer(scalarInBuf_, 32);
            pipe_->InitBuffer(scalarFp32Buf_, 32);
            pipe_->InitBuffer(scalarOutBuf_, 32);
            pipe_->InitBuffer(scalarI64Buf_, 32);
        }
        if (pipe_ != nullptr && initVecBuffers) {
            pipe_->InitBuffer(exp2Buf_, EXP2_UB_ELEMENTS * sizeof(float));
            pipe_->InitBuffer(vecBuf_, KDA_VEC_ARENA_ELEMENTS * sizeof(float));
            pipe_->InitBuffer(qInQue_, KDA_VEC_BUFFER_NUM, EXP2_UB_ELEMENTS * sizeof(T));
            pipe_->InitBuffer(kInQue_, KDA_VEC_BUFFER_NUM, EXP2_UB_ELEMENTS * sizeof(T));
            pipe_->InitBuffer(gInQue_, KDA_VEC_BUFFER_NUM, EXP2_UB_ELEMENTS * sizeof(float));
            pipe_->InitBuffer(qgOutQue_, KDA_VEC_BUFFER_NUM, EXP2_UB_ELEMENTS * sizeof(T));
            pipe_->InitBuffer(wOutQue_, KDA_VEC_BUFFER_NUM, EXP2_UB_ELEMENTS * sizeof(T));
            pipe_->InitBuffer(kgOutQue_, KDA_VEC_BUFFER_NUM, EXP2_UB_ELEMENTS * sizeof(T));
        }
        if (pipe_ != nullptr && !initVecBuffers) {
            pipe_->InitBuffer(solveL1Buf_,
                              KDA_SOLVE_L1_SLOT_COUNT * KDA_SOLVE_MATRIX_ELEMENTS * sizeof(T));
            solveL1_ = solveL1Buf_.Get<T>();
            pipe_->InitBuffer(solveL0aBuf_, KDA_SOLVE_MATRIX_ELEMENTS * sizeof(T));
            solveL0a_ = solveL0aBuf_.Get<T>();
            pipe_->InitBuffer(solveL0bBuf_, KDA_SOLVE_MATRIX_ELEMENTS * sizeof(T));
            solveL0b_ = solveL0bBuf_.Get<T>();
            pipe_->InitBuffer(solveL0cBuf_, KDA_SOLVE_MATRIX_ELEMENTS * sizeof(float));
            solveL0c_ = solveL0cBuf_.Get<float>();
        }
    }

    __aicore__ inline void ProcessAivOnly()
    {
        if (stage_ == 1) {
            isAivOnly_ = true;
            ProcessPreAiv();
            return;
        }
        if (stage_ == 2) {
            isAivOnly_ = true;
            ProcessOutAiv();
            return;
        }
        if (stage_ == 3) {
            isAivOnly_ = true;
            ProcessPostAiv();
            return;
        }
        if (stage_ == 4) {
            isAivOnly_ = true;
            ProcessStateAiv();
            return;
        }
        isAivOnly_ = true;
        uint64_t taskNum = static_cast<uint64_t>(N_ * HV_);
        uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += blockNum) {
            uint64_t seq = task / HV_;
            uint64_t hv = task % HV_;
            ProcessSeqHeadAiv(seq, hv);
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        if (stage_ == 1) {
            ProcessPreAiv();
            return;
        }
        if (stage_ == 2) {
            ProcessOutAiv();
            return;
        }
        if (stage_ == 3) {
            ProcessPostAiv();
            return;
        }
        if (stage_ == 4) {
            ProcessStateAiv();
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            ProcessAivOnly();
        } else {
            isAivOnly_ = false;
            uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            if (subBlockNum == 0) {
                return;
            }
            uint64_t taskNum = static_cast<uint64_t>(N_ * HV_);
            uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
            uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
            for (uint64_t task = coreIdx; task < taskNum; task += coreNum) {
                uint64_t seq = task / HV_;
                uint64_t hv = task % HV_;
                ProcessSeqHeadAiv(seq, hv);
            }
        }
    }

    __aicore__ inline void ProcessAic()
    {
        if (stage_ == 1) {
            ProcessPreAic();
            return;
        }
        if (stage_ == 2) {
            ProcessOutAic();
            return;
        }
        if (stage_ == 3) {
            ProcessPostAic();
            return;
        }
        if (stage_ == 4) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        } else {
            uint64_t taskNum = static_cast<uint64_t>(N_ * HV_);
            uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
            for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum) {
                uint64_t seq = task / HV_;
                uint64_t hv = task % HV_;
                ProcessSeqHeadAic(seq, hv);
            }
        }
    }

    __aicore__ inline void ProcessOutOnlyAiv()
    {
        ProcessOutAiv();
    }

    __aicore__ inline void ProcessOutOnlyAic()
    {
        ProcessOutAic();
    }

private:
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
        return HOffset(b, hv, chunkIdx, 0, 0) + slot * KDA_SOLVE_MATRIX_ELEMENTS;
    }

    __aicore__ inline uint64_t StateOffset(uint64_t seq, uint64_t hv, uint64_t d, uint64_t r) const
    {
        return ((seq * HV_ + hv) * K_ + d) * V_ + r;
    }

    __aicore__ inline uint64_t ChunkCountBefore(uint64_t seq)
    {
        if (!isVarLen_) {
            return 0;
        }
        uint64_t count = 0;
        for (uint64_t s = 0; s < seq; ++s) {
            uint64_t start = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, s));
            uint64_t end = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, s + 1));
            count += (end - start + BT_ - 1) / BT_;
        }
        return count;
    }

    __aicore__ inline void ExpClamped(LocalTensor<float> &tensor, uint32_t count)
    {
        Mins(tensor, tensor, KDA_EXP_MAX, count);
        PipeBarrier<PIPE_V>();
        Exp(tensor, tensor, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunExp2(LocalTensor<float> &tensor, uint32_t count)
    {
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        ExpClamped(tensor, count);
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
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

    template <typename DstT, typename SrcT>
    __aicore__ inline void CopyTensorRow(GlobalTensor<DstT> &dst, uint64_t dstOffset, GlobalTensor<SrcT> &src,
                                         uint64_t srcOffset, uint64_t count)
    {
        LocalTensor<float> rowFp32 = vecBuf_.Get<float>();
        LoadAsFloatRow(src, srcOffset, rowFp32, count);
        StoreFloatRow(dst, dstOffset, rowFp32, count);
    }

    template <typename CopyT>
    __aicore__ inline void ZeroTensorRow(GlobalTensor<CopyT> &dst, uint64_t dstOffset, uint64_t count)
    {
        LocalTensor<float> rowFp32 = vecBuf_.Get<float>();
        Duplicate(rowFp32, 0.0f, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
        StoreFloatRow(dst, dstOffset, rowFp32, count);
    }

    template <typename CopyT>
    __aicore__ inline float ReadAsFloat(GlobalTensor<CopyT> &tensor, uint64_t offset)
    {
        LocalTensor<CopyT> scalarIn = scalarInBuf_.Get<CopyT>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(CopyT)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalarIn, tensor[offset], params, padParams);
        SetFlag<HardEvent::MTE2_V>(KDA_SCALAR_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_SCALAR_MTE2_V_EVENT_ID);

        LocalTensor<float> scalarFp32 = scalarFp32Buf_.Get<float>();
        if constexpr (IsSameType<CopyT, float>::value) {
            Adds(scalarFp32, scalarIn, 0.0f, 1);
        } else {
            Cast(scalarFp32, scalarIn, RoundMode::CAST_NONE, 1);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(KDA_SCALAR_V_S_EVENT_ID);
        WaitFlag<HardEvent::V_S>(KDA_SCALAR_V_S_EVENT_ID);
        __ubuf__ float *ptr = (__ubuf__ float *)scalarFp32.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline int64_t ReadInt64(GlobalTensor<int64_t> &tensor, uint64_t offset)
    {
        LocalTensor<int64_t> scalarI64 = scalarI64Buf_.Get<int64_t>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(int64_t)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(scalarI64, tensor[offset], params, padParams);
        SetFlag<HardEvent::MTE2_V>(KDA_SCALAR_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_SCALAR_MTE2_V_EVENT_ID);
        SetFlag<HardEvent::V_S>(KDA_SCALAR_V_S_EVENT_ID);
        WaitFlag<HardEvent::V_S>(KDA_SCALAR_V_S_EVENT_ID);
        __ubuf__ int64_t *ptr = (__ubuf__ int64_t *)scalarI64.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline int64_t ReadMetaInt64(GlobalTensor<int64_t> &tensor, uint64_t offset)
    {
        return tensor.GetValue(offset);
    }

    template <typename CopyT>
    __aicore__ inline void WriteFromFloat(GlobalTensor<CopyT> &tensor, uint64_t offset, float value)
    {
        LocalTensor<float> scalarFp32 = scalarFp32Buf_.Get<float>();
        Duplicate(scalarFp32, value, 1);
        PipeBarrier<PIPE_V>();
        DataCopyParams params{1, static_cast<uint16_t>(sizeof(CopyT)), 0, 0};
        if constexpr (IsSameType<CopyT, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(tensor[offset], scalarFp32, params);
        } else {
            LocalTensor<CopyT> scalarOut = scalarOutBuf_.Get<CopyT>();
            Cast(scalarOut, scalarFp32, RoundMode::CAST_RINT, 1);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(tensor[offset], scalarOut, params);
        }
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline float ReadFloat(GlobalTensor<float> &tensor, uint64_t offset)
    {
        return ReadAsFloat(tensor, offset);
    }

    __aicore__ inline void WriteFloat(GlobalTensor<float> &tensor, uint64_t offset, float value)
    {
        WriteFromFloat(tensor, offset, value);
    }

    __aicore__ inline __ubuf__ float *UbPtr(LocalTensor<float> &tensor)
    {
        return (__ubuf__ float *)tensor.GetPhyAddr();
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
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Adds(dst, dst, 0.0f, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            WaitFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        } else {
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>();
            CopyVectorIn(rowLocal, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(dst, rowLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            WaitFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename CopyT>
    __aicore__ inline void StoreFloatRow(GlobalTensor<CopyT> &dst, uint64_t dstOffset, LocalTensor<float> &src,
                                         uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            CopyVectorOut(dst, dstOffset, src, count);
        } else {
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>();
            Cast(rowLocal, src, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            CopyVectorOut(dst, dstOffset, rowLocal, count);
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void LoadExp2GTo(LocalTensor<float> &dst, uint64_t b, uint64_t hv, uint64_t t)
    {
        CopyRowIn(dst, gk_, KVOffset(b, hv, t, 0, K_));
        SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        Muls(dst, dst, LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(dst, static_cast<uint32_t>(K_));
    }

    template <typename CopyT>
    __aicore__ inline void CopyStackFloatRowOut(GlobalTensor<CopyT> &dst, uint64_t dstOffset, float *values,
                                                uint64_t count, uint64_t slot = 0)
    {
        LocalTensor<float> rowFp32 = vecBuf_.Get<float>()[slot * EXP2_UB_ELEMENTS];
        __ubuf__ float *rowPtr = UbPtr(rowFp32);
        for (uint64_t idx = 0; idx < count; ++idx) {
            rowPtr[idx] = values[idx];
        }
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        Adds(rowFp32, rowFp32, 0.0f, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<CopyT, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            CopyVectorOut(dst, dstOffset, rowFp32, count);
        } else {
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>();
            Cast(rowLocal, rowFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            CopyVectorOut(dst, dstOffset, rowLocal, count);
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        SetFlag<HardEvent::MTE3_S>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_S>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    template <typename CopyT>
    __aicore__ inline void LoadStackFloatRow(GlobalTensor<CopyT> &src, uint64_t srcOffset, float *values,
                                             uint64_t count, uint64_t slot = 0)
    {
        LocalTensor<float> rowFp32 = vecBuf_.Get<float>()[slot * EXP2_UB_ELEMENTS];
        if constexpr (IsSameType<CopyT, float>::value) {
            CopyVectorIn(rowFp32, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Adds(rowFp32, rowFp32, 0.0f, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            WaitFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        } else {
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>();
            CopyVectorIn(rowLocal, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(rowFp32, rowLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            WaitFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(KDA_SCALAR_V_S_EVENT_ID);
        WaitFlag<HardEvent::V_S>(KDA_SCALAR_V_S_EVENT_ID);
        __ubuf__ float *rowPtr = UbPtr(rowFp32);
        for (uint64_t idx = 0; idx < count; ++idx) {
            values[idx] = rowPtr[idx];
        }
    }

    __aicore__ inline LocalTensor<float> Exp2G(uint64_t b, uint64_t hv, uint64_t t)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        CopyRowIn(exp2Local, gk_, KVOffset(b, hv, t, 0, K_));
        SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        Muls(exp2Local, exp2Local, LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline LocalTensor<float> Exp2NegG(uint64_t b, uint64_t hv, uint64_t t)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        CopyRowIn(exp2Local, gk_, KVOffset(b, hv, t, 0, K_));
        SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        Muls(exp2Local, exp2Local, -LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline LocalTensor<float> Exp2GDiff(uint64_t b, uint64_t hv, uint64_t lhs, uint64_t rhs)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        LocalTensor<float> rhsLocal = vecBuf_.Get<float>();
        CopyRowIn(exp2Local, gk_, KVOffset(b, hv, lhs, 0, K_));
        CopyRowIn(rhsLocal, gk_, KVOffset(b, hv, rhs, 0, K_));
        SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        Sub(exp2Local, exp2Local, rhsLocal, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        Muls(exp2Local, exp2Local, LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline uint64_t GateProductToken(uint64_t start, uint64_t logicalIdx, uint64_t subBlockIdx,
                                                uint64_t subBlockNum) const
    {
        return start + subBlockIdx + logicalIdx * subBlockNum;
    }

    __aicore__ inline void LoadGateProductRow(uint64_t b, uint64_t h, uint64_t hv, uint64_t ti)
    {
        LocalTensor<T> qLocal = qInQue_.AllocTensor<T>();
        LocalTensor<T> kLocal = kInQue_.AllocTensor<T>();
        LocalTensor<float> gLocal = gInQue_.AllocTensor<float>();
        CopyRowIn(qLocal, q_, QOffset(b, h, ti, 0));
        CopyRowIn(kLocal, k_, QOffset(b, h, ti, 0));
        CopyRowIn(gLocal, gk_, KVOffset(b, hv, ti, 0, K_));
        qInQue_.EnQue(qLocal);
        kInQue_.EnQue(kLocal);
        gInQue_.EnQue(gLocal);
    }

    __aicore__ inline void StoreGateProductRow(uint64_t b, uint64_t hv, uint64_t ti)
    {
        LocalTensor<T> qPosLocal = qgOutQue_.DeQue<T>();
        LocalTensor<T> kPosLocal = wOutQue_.DeQue<T>();
        LocalTensor<T> kNegLocal = kgOutQue_.DeQue<T>();
        SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
        WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
        CopyRowOut(qg_, KVOffset(b, hv, ti, 0, K_), qPosLocal);
        CopyRowOut(w_, KVOffset(b, hv, ti, 0, K_), kPosLocal);
        CopyRowOut(kg_, KVOffset(b, hv, ti, 0, K_), kNegLocal);
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        qgOutQue_.FreeTensor(qPosLocal);
        wOutQue_.FreeTensor(kPosLocal);
        kgOutQue_.FreeTensor(kNegLocal);
    }

    __aicore__ inline void ComputeGateProductRow(LocalTensor<float> &qFp32, LocalTensor<float> &kFp32,
                                                 LocalTensor<float> &gFp32, LocalTensor<float> &expFp32,
                                                 LocalTensor<float> &outFp32)
    {
        LocalTensor<T> qPosLocal = qgOutQue_.AllocTensor<T>();
        LocalTensor<T> kPosLocal = wOutQue_.AllocTensor<T>();
        LocalTensor<T> kNegLocal = kgOutQue_.AllocTensor<T>();

        Muls(expFp32, gFp32, LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        ExpClamped(expFp32, static_cast<uint32_t>(K_));

        Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(qPosLocal, outFp32, static_cast<uint32_t>(K_));
        } else {
            Cast(qPosLocal, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(K_));
        }
        PipeBarrier<PIPE_V>();

        Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(kPosLocal, outFp32, static_cast<uint32_t>(K_));
        } else {
            Cast(kPosLocal, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(K_));
        }
        PipeBarrier<PIPE_V>();

        Muls(expFp32, gFp32, -LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        ExpClamped(expFp32, static_cast<uint32_t>(K_));
        Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(kNegLocal, outFp32, static_cast<uint32_t>(K_));
        } else {
            Cast(kNegLocal, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(K_));
        }

        qgOutQue_.EnQue(qPosLocal);
        wOutQue_.EnQue(kPosLocal);
        kgOutQue_.EnQue(kNegLocal);
    }

    __aicore__ inline bool PrepareGateProductsBulk(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                   uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if constexpr (IsSameType<T, float>::value) {
            return false;
        }
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum || K_ == 0) {
            return false;
        }
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return true;
        }

        constexpr uint64_t arenaBytes = static_cast<uint64_t>(KDA_VEC_ARENA_ELEMENTS) * sizeof(float);
        constexpr uint64_t bytesPerElem = 5 * sizeof(float) + 3 * sizeof(T);
        uint64_t maxElems = arenaBytes / bytesPerElem;
        uint64_t maxRows = maxElems / K_;
        if (maxRows == 0) {
            return false;
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

            uint64_t typedOffset = (5 * elems * sizeof(float) + sizeof(T) - 1) / sizeof(T);
            uint64_t typedCapacity = arenaBytes / sizeof(T);
            if (typedOffset + 3 * elems > typedCapacity) {
                return false;
            }
            LocalTensor<T> typedBase = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> qTyped = typedBase;
            LocalTensor<T> kTyped = typedBase[elems];
            LocalTensor<T> kgTyped = typedBase[2 * elems];

            uint64_t token = start + tileRow;
            CopyVectorIn(qTyped, q_, QOffset(b, h, token, 0), elems);
            CopyVectorIn(kTyped, k_, QOffset(b, h, token, 0), elems);
            CopyVectorIn(gFp32, gk_, KVOffset(b, hv, token, 0, K_), elems);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);

            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Muls(expFp32, gFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ExpClamped(expFp32, static_cast<uint32_t>(elems));

            Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Cast(qTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Cast(kTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Muls(expFp32, gFp32, -LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ExpClamped(expFp32, static_cast<uint32_t>(elems));
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Cast(kgTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            CopyVectorOut(qg_, KVOffset(b, hv, token, 0, K_), qTyped, elems);
            CopyVectorOut(w_, KVOffset(b, hv, token, 0, K_), kTyped, elems);
            CopyVectorOut(kg_, KVOffset(b, hv, token, 0, K_), kgTyped, elems);
            SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
            WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        }
        return true;
    }

    __aicore__ inline void PrepareGateProducts(uint64_t b, uint64_t h, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if (subBlockIdx >= curT || subBlockNum == 0) {
            return;
        }
        if (PrepareGateProductsBulk(b, h, hv, start, curT, subBlockIdx, subBlockNum)) {
            return;
        }

        LocalTensor<float> vecLocal = vecBuf_.Get<float>();
        LocalTensor<float> qFp32 = vecLocal;
        LocalTensor<float> kFp32 = vecLocal[EXP2_UB_ELEMENTS];
        LocalTensor<float> gFp32 = vecLocal[2 * EXP2_UB_ELEMENTS];
        LocalTensor<float> expFp32 = vecLocal[3 * EXP2_UB_ELEMENTS];
        LocalTensor<float> outFp32 = vecLocal[4 * EXP2_UB_ELEMENTS];

        uint64_t rowCount = 0;
        for (uint64_t i = subBlockIdx; i < curT; i += subBlockNum) {
            ++rowCount;
        }

        LoadGateProductRow(b, h, hv, GateProductToken(start, 0, subBlockIdx, subBlockNum));
        for (uint64_t logicalIdx = 0; logicalIdx < rowCount; ++logicalIdx) {
            uint64_t ti = GateProductToken(start, logicalIdx, subBlockIdx, subBlockNum);
            LocalTensor<T> qLocal = qInQue_.DeQue<T>();
            LocalTensor<T> kLocal = kInQue_.DeQue<T>();
            LocalTensor<float> gLocal = gInQue_.DeQue<float>();

            if (logicalIdx + 1 < rowCount) {
                LoadGateProductRow(b, h, hv, GateProductToken(start, logicalIdx + 1, subBlockIdx, subBlockNum));
            }

            if constexpr (IsSameType<T, float>::value) {
                DataCopy(qFp32, qLocal, static_cast<uint32_t>(K_));
                DataCopy(kFp32, kLocal, static_cast<uint32_t>(K_));
            } else {
                Cast(qFp32, qLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(K_));
                Cast(kFp32, kLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(K_));
            }
            DataCopy(gFp32, gLocal, static_cast<uint32_t>(K_));
            qInQue_.FreeTensor(qLocal);
            kInQue_.FreeTensor(kLocal);
            gInQue_.FreeTensor(gLocal);
            PipeBarrier<PIPE_V>();

            if (logicalIdx > 0) {
                StoreGateProductRow(b, hv, GateProductToken(start, logicalIdx - 1, subBlockIdx, subBlockNum));
            }
            ComputeGateProductRow(qFp32, kFp32, gFp32, expFp32, outFp32);
        }
        StoreGateProductRow(b, hv, GateProductToken(start, rowCount - 1, subBlockIdx, subBlockNum));
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
    }

    __aicore__ inline void ComputeRawAqkAkkScalar(uint64_t b, uint64_t h, uint64_t hv, uint64_t start, uint64_t curT)
    {
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            float aqkRow[EXP2_UB_ELEMENTS];
            float akkRow[EXP2_UB_ELEMENTS];
            for (uint64_t j = 0; j < BT_; ++j) {
                aqkRow[j] = 0.0f;
                akkRow[j] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float aqkRaw = 0.0f;
                float akkRaw = 0.0f;
                LocalTensor<float> gateLocal = Exp2GDiff(b, hv, ti, tj);
                __ubuf__ float *gatePtr = UbPtr(gateLocal);
                for (uint64_t d = 0; d < K_; ++d) {
                    float qi = ReadAsFloat(q_, QOffset(b, h, ti, d));
                    float ki = ReadAsFloat(k_, QOffset(b, h, ti, d));
                    float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
                    aqkRaw += qi * kj * gatePtr[d];
                    akkRaw += ki * kj * gatePtr[d];
                }
                aqkRow[j] = aqkRaw;
                akkRow[j] = akkRaw;
            }
            CopyStackFloatRowOut(aqk_, AOffset(b, hv, ti, 0), aqkRow, BT_, 0);
            CopyStackFloatRowOut(akk_, AOffset(b, hv, ti, 0), akkRow, BT_, 1);
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
    }

    __aicore__ inline void ComputeRawAqkAkkCube(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LocalL1TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                               std::is_same<ElementB, float>::value,
                                                           KdaFloatL1TileShape, KdaL1TileShape>::type;
        using LocalL0TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                               std::is_same<ElementB, float>::value,
                                                           KdaFloatL0TileShape, KdaL0TileShape>::type;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, LocalL1TileShape, LocalL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K_, BT_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(BT_, BT_);
        Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(curT),
                                 static_cast<uint32_t>(K_)};

        auto tensorQPos = tla::MakeTensor(qg_[KVOffset(b, hv, start, 0, K_)], layoutA,
                                          Catlass::Arch::PositionGM{});
        auto tensorKPos = tla::MakeTensor(w_[KVOffset(b, hv, start, 0, K_)], layoutA,
                                          Catlass::Arch::PositionGM{});
        auto tensorKNeg = tla::MakeTensor(kg_[KVOffset(b, hv, start, 0, K_)], layoutB,
                                          Catlass::Arch::PositionGM{});
        auto tensorAqk = tla::MakeTensor(aqk_[AOffset(b, hv, start, 0)], layoutC,
                                         Catlass::Arch::PositionGM{});
        auto tensorAkk = tla::MakeTensor(akk_[AOffset(b, hv, start, 0)], layoutC,
                                         Catlass::Arch::PositionGM{});

        auto blockQPos = GetTile(tensorQPos, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKPos = GetTile(tensorKPos, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKNeg = GetTile(tensorKNeg, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));

        blockMmad(blockQPos, blockKNeg, blockAqk, shape);
        PipeBarrier<PIPE_ALL>();
        blockMmad(blockKPos, blockKNeg, blockAkk, shape);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool UseAkkCubeSolve(uint64_t curT) const
    {
        return curT > 1 && curT <= KDA_SOLVE_BT && K_ >= 16 && V_ >= 16 && V_ <= 128 && K_ % 16 == 0 && V_ % 16 == 0 &&
               K_ * V_ >= KDA_SOLVE_SCRATCH_SLOTS * KDA_SOLVE_MATRIX_ELEMENTS &&
               K_ * V_ >= curT * (K_ + V_);
    }

    __aicore__ inline bool UsePostWuCube(uint64_t curT) const
    {
        return curT > 1 && curT <= KDA_SOLVE_BT && K_ >= 16 && V_ >= 16 && V_ <= 128 && K_ % 16 == 0 && V_ % 16 == 0 &&
               K_ * V_ >= curT * (K_ + V_);
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

    __aicore__ inline void PrepareSelectZero(LocalTensor<float> zeroLocal)
    {
        Duplicate(zeroLocal, 0.0f, KDA_SELECT_ZERO_FLOATS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildZeroAfterPrefixBitMask(LocalTensor<uint8_t> dst, uint64_t prefix,
                                                       uint64_t count)
    {
        if (prefix > count) {
            prefix = count;
        }
        uint64_t blocks = (count + KDA_MASK_BITS_PER_BYTE - 1) / KDA_MASK_BITS_PER_BYTE;
        for (uint64_t block = 0; block < blocks; ++block) {
            uint8_t maskValue = 0;
            for (uint64_t bit = 0; bit < KDA_MASK_BITS_PER_BYTE; ++bit) {
                uint64_t col = block * KDA_MASK_BITS_PER_BYTE + bit;
                if (col >= prefix || col >= count) {
                    maskValue |= static_cast<uint8_t>(1U << bit);
                }
            }
            dst.SetValue(block, maskValue);
        }
    }

    __aicore__ inline void ApplyPrefixSelectZero(LocalTensor<float> rowLocal, uint64_t prefix,
                                                 LocalTensor<float> zeroLocal)
    {
        LocalTensor<uint8_t> selectMask = vecBuf_.Get<uint8_t>()[KDA_SELECT_MASK_BYTE_OFFSET];
        BuildZeroAfterPrefixBitMask(selectMask, prefix, KDA_SOLVE_BT);
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
        Select(rowLocal, selectMask, zeroLocal, rowLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               KDA_SOLVE_BT, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareAqkAkkSolveInput64(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[KDA_SOLVE_MASK_OFFSET];
        LocalTensor<float> oneHotLocal = arena[KDA_SOLVE_ONEHOT_OFFSET];
        LocalTensor<float> zeroLocal = arena[KDA_SELECT_ZERO_OFFSET];

        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);

        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, KDA_SOLVE_BT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            DataCopy(aqkMat, aqk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(akkMat, akk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedAkk = typedAqk[KDA_SOLVE_MATRIX_ELEMENTS];
            DataCopy(typedAqk, aqk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(typedAkk, akk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(aqkMat, typedAqk, RoundMode::CAST_NONE, KDA_SOLVE_MATRIX_ELEMENTS);
            Cast(akkMat, typedAkk, RoundMode::CAST_NONE, KDA_SOLVE_MATRIX_ELEMENTS);
            PipeBarrier<PIPE_V>();
        }

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        PrepareSelectZero(zeroLocal);
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            ApplyPrefixSelectZero(aqkMat[row * KDA_SOLVE_BT], row + 1, zeroLocal);
            ApplyPrefixSelectZero(akkMat[row * KDA_SOLVE_BT], row, zeroLocal);
        }

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

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(aqk_[AOffset(b, hv, start, 0)], aqkMat, KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(akk_[AOffset(b, hv, start, 0)], akkMat, KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                     KDA_SOLVE_MATRIX_ELEMENTS);
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedAkk = typedAqk[KDA_SOLVE_MATRIX_ELEMENTS];
            LocalTensor<T> typedX = typedAkk[KDA_SOLVE_MATRIX_ELEMENTS];
            Cast(typedAqk, aqkMat, RoundMode::CAST_RINT, KDA_SOLVE_MATRIX_ELEMENTS);
            Cast(typedAkk, akkMat, RoundMode::CAST_RINT, KDA_SOLVE_MATRIX_ELEMENTS);
            Cast(typedX, xMat, RoundMode::CAST_RINT, KDA_SOLVE_MATRIX_ELEMENTS);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(aqk_[AOffset(b, hv, start, 0)], typedAqk, KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(akk_[AOffset(b, hv, start, 0)], typedAkk, KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], typedX,
                     KDA_SOLVE_MATRIX_ELEMENTS);
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void PrepareAqkAkkSolveInputTail(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                       uint64_t start, uint64_t curT)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(T)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[KDA_SOLVE_MASK_OFFSET];
        LocalTensor<float> oneHotLocal = arena[KDA_SOLVE_ONEHOT_OFFSET];
        LocalTensor<float> zeroLocal = arena[KDA_SELECT_ZERO_OFFSET];

        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);

        FillLocalFloat(betaLocal, 0.0f, KDA_SOLVE_BT);
        SetFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::V_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, curT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(aqkMat, aqk_[AOffset(b, hv, start, 0)], validParams, padParams);
            DataCopyPad(akkMat, akk_[AOffset(b, hv, start, 0)], validParams, padParams);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
                FillLocalFloat(aqkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
                FillLocalFloat(akkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
            }
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedAkk = typedAqk[KDA_SOLVE_MATRIX_ELEMENTS];
            DataCopyPad(typedAqk, aqk_[AOffset(b, hv, start, 0)], validParams, padParams);
            DataCopyPad(typedAkk, akk_[AOffset(b, hv, start, 0)], validParams, padParams);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(aqkMat, typedAqk, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            Cast(akkMat, typedAkk, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
                FillLocalFloat(aqkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
                FillLocalFloat(akkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
            }
        }

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        PrepareSelectZero(zeroLocal);
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            ApplyPrefixSelectZero(aqkMat[row * KDA_SOLVE_BT], row + 1, zeroLocal);
            ApplyPrefixSelectZero(akkMat[row * KDA_SOLVE_BT], row, zeroLocal);
        }

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

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(aqk_[AOffset(b, hv, start, 0)], aqkMat, validParams);
            DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                     KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0)], akkMat,
                     KDA_SOLVE_MATRIX_ELEMENTS);
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedAkk = typedAqk[KDA_SOLVE_MATRIX_ELEMENTS];
            LocalTensor<T> typedX = typedAkk[KDA_SOLVE_MATRIX_ELEMENTS];
            Cast(typedAqk, aqkMat, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            Cast(typedAkk, akkMat, RoundMode::CAST_RINT, KDA_SOLVE_MATRIX_ELEMENTS);
            Cast(typedX, xMat, RoundMode::CAST_RINT, KDA_SOLVE_MATRIX_ELEMENTS);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(aqk_[AOffset(b, hv, start, 0)], typedAqk, validParams);
            DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], typedX,
                     KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0)], typedAkk,
                     KDA_SOLVE_MATRIX_ELEMENTS);
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
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
                                                       uint64_t start, uint64_t rowBegin, uint64_t rowEnd,
                                                       bool storeLToAkk, bool storeLToScratch)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t elemCount = rowCount * KDA_SOLVE_BT;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(T)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[elemCount];
        LocalTensor<float> xMat = arena[2 * elemCount];
        LocalTensor<float> betaLocal = arena[3 * elemCount];
        LocalTensor<float> betaBrcb = arena[3 * elemCount + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[3 * elemCount + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[3 * elemCount + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];
        LocalTensor<float> zeroLocal = arena[3 * elemCount + KDA_SOLVE_BT + 512 + 2 * KDA_SOLVE_BT];

        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t token = start + rowBegin;

        LoadAsFloatRow(beta_, BetaOffset(b, hv, token), betaLocal, rowCount);
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(aqkMat, aqk_[AOffset(b, hv, token, 0)], validParams, padParams);
            DataCopyPad(akkMat, akk_[AOffset(b, hv, token, 0)], validParams, padParams);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedAkk = typedAqk[elemCount];
            DataCopyPad(typedAqk, aqk_[AOffset(b, hv, token, 0)], validParams, padParams);
            DataCopyPad(typedAkk, akk_[AOffset(b, hv, token, 0)], validParams, padParams);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(aqkMat, typedAqk, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            Cast(akkMat, typedAkk, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
        }

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, static_cast<uint8_t>(rowCount), {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        PrepareSelectZero(zeroLocal);
        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            ApplyPrefixSelectZero(aqkMat[localRow * KDA_SOLVE_BT], row + 1, zeroLocal);
            ApplyPrefixSelectZero(akkMat[localRow * KDA_SOLVE_BT], row, zeroLocal);
        }

        Muls(xMat, akkMat, -1.0f, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();
        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[localRow * KDA_SOLVE_BT], xMat[localRow * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * KDA_SOLVE_BT;
        uint64_t lBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0) + rowBegin * KDA_SOLVE_BT;
        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(aqk_[AOffset(b, hv, token, 0)], aqkMat, validParams);
            if (storeLToAkk) {
                DataCopyPad(akk_[AOffset(b, hv, token, 0)], akkMat, validParams);
            }
            DataCopy(h_[xBase], xMat, static_cast<uint32_t>(elemCount));
            if (storeLToScratch) {
                DataCopy(h_[lBase], akkMat, static_cast<uint32_t>(elemCount));
            }
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedAkk = typedAqk[elemCount];
            LocalTensor<T> typedX = typedAkk[elemCount];
            Cast(typedAqk, aqkMat, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            Cast(typedAkk, akkMat, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            Cast(typedX, xMat, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(aqk_[AOffset(b, hv, token, 0)], typedAqk, validParams);
            if (storeLToAkk) {
                DataCopyPad(akk_[AOffset(b, hv, token, 0)], typedAkk, validParams);
            }
            DataCopy(h_[xBase], typedX, static_cast<uint32_t>(elemCount));
            if (storeLToScratch) {
                DataCopy(h_[lBase], typedAkk, static_cast<uint32_t>(elemCount));
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void PrepareAqkAkkMxrInputRows(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                     uint64_t start, uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t elemCount = rowCount * KDA_SOLVE_BT;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(T)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[elemCount];
        LocalTensor<float> negLMat = arena[2 * elemCount];
        LocalTensor<float> xMat = arena[3 * elemCount];
        LocalTensor<float> betaLocal = arena[4 * elemCount];
        LocalTensor<float> betaBrcb = arena[4 * elemCount + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[4 * elemCount + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[4 * elemCount + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];
        LocalTensor<float> zeroLocal = arena[4 * elemCount + KDA_SOLVE_BT + 512 + 2 * KDA_SOLVE_BT];

        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t token = start + rowBegin;

        LoadAsFloatRow(beta_, BetaOffset(b, hv, token), betaLocal, rowCount);
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(aqkMat, aqk_[AOffset(b, hv, token, 0)], validParams, padParams);
            DataCopyPad(akkMat, akk_[AOffset(b, hv, token, 0)], validParams, padParams);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedAkk = typedAqk[elemCount];
            DataCopyPad(typedAqk, aqk_[AOffset(b, hv, token, 0)], validParams, padParams);
            DataCopyPad(typedAkk, akk_[AOffset(b, hv, token, 0)], validParams, padParams);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(aqkMat, typedAqk, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            Cast(akkMat, typedAkk, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
        }

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, static_cast<uint8_t>(rowCount), {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        PrepareSelectZero(zeroLocal);
        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            ApplyPrefixSelectZero(aqkMat[localRow * KDA_SOLVE_BT], row + 1, zeroLocal);
            ApplyPrefixSelectZero(akkMat[localRow * KDA_SOLVE_BT], row, zeroLocal);
        }

        Muls(negLMat, akkMat, -1.0f, static_cast<uint32_t>(elemCount));
        Muls(xMat, akkMat, -1.0f, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();
        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            uint64_t blockStart = (row / KDA_SOLVE_MXR_BLOCK) * KDA_SOLVE_MXR_BLOCK;
            BuildPrefixMask(maskLocal, row, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, blockStart, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Mul(xMat[localRow * KDA_SOLVE_BT], xMat[localRow * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();

            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[localRow * KDA_SOLVE_BT], xMat[localRow * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        uint64_t negLBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * KDA_SOLVE_BT;
        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(aqk_[AOffset(b, hv, token, 0)], aqkMat, validParams);
            DataCopyPad(akk_[AOffset(b, hv, token, 0)], xMat, validParams);
            DataCopy(h_[negLBase], negLMat, static_cast<uint32_t>(elemCount));
        } else {
            LocalTensor<T> typedAqk = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedX = typedAqk[elemCount];
            LocalTensor<T> typedNegL = typedX[elemCount];
            Cast(typedAqk, aqkMat, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            Cast(typedX, xMat, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            Cast(typedNegL, negLMat, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopyPad(aqk_[AOffset(b, hv, token, 0)], typedAqk, validParams);
            DataCopyPad(akk_[AOffset(b, hv, token, 0)], typedX, validParams);
            DataCopy(h_[negLBase], typedNegL, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void CubeGemmSolveSub(GlobalTensor<T> &tensorA, uint64_t baseA, uint64_t rowA, uint64_t colA,
                                            GlobalTensor<T> &tensorB, uint64_t baseB, uint64_t rowB, uint64_t colB,
                                            GlobalTensor<T> &tensorC, uint64_t baseC, uint64_t rowC, uint64_t colC,
                                            uint32_t m, uint32_t n, uint32_t k)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LocalPostL1TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                                   std::is_same<ElementB, float>::value,
                                                               KdaFloatPostL1TileShape, KdaPostL1TileShape>::type;
        using LocalPostL0TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                                   std::is_same<ElementB, float>::value,
                                                               KdaFloatPostL0TileShape, KdaPostL0TileShape>::type;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, LocalPostL1TileShape,
                                                              LocalPostL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(KDA_SOLVE_BT, KDA_SOLVE_BT);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(KDA_SOLVE_BT, KDA_SOLVE_BT);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(KDA_SOLVE_BT, KDA_SOLVE_BT);
        auto tensorLayoutA = tla::MakeTensor(tensorA[baseA], layoutA, Catlass::Arch::PositionGM{});
        auto tensorLayoutB = tla::MakeTensor(tensorB[baseB], layoutB, Catlass::Arch::PositionGM{});
        auto tensorLayoutC = tla::MakeTensor(tensorC[baseC], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto blockA = GetTile(tensorLayoutA, tla::MakeCoord(rowA, colA), tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tensorLayoutB, tla::MakeCoord(rowB, colB), tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tensorLayoutC, tla::MakeCoord(rowC, colC), tla::MakeShape(shape.m(), shape.n()));
        blockMmad(blockA, blockB, blockC, shape);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AddSolveTmpToX(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                          bool storeAkk)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        if constexpr (IsSameType<T, float>::value) {
            DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedTmp = typedX[KDA_SOLVE_MATRIX_ELEMENTS];
            DataCopy(typedX, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(typedTmp, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(xLocal, typedX, RoundMode::CAST_NONE, KDA_SOLVE_MATRIX_ELEMENTS);
            Cast(tmpLocal, typedTmp, RoundMode::CAST_NONE, KDA_SOLVE_MATRIX_ELEMENTS);
            PipeBarrier<PIPE_V>();
        }

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
            if (storeAkk) {
                DataCopy(akk_[AOffset(b, hv, start, 0)], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
            }
        } else {
            LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
            Cast(typedX, xLocal, RoundMode::CAST_RINT, KDA_SOLVE_MATRIX_ELEMENTS);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(h_[xBase], typedX, KDA_SOLVE_MATRIX_ELEMENTS);
            if (storeAkk) {
                DataCopy(akk_[AOffset(b, hv, start, 0)], typedX, KDA_SOLVE_MATRIX_ELEMENTS);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void AddSolveTmpToXTail(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, bool storeAkk)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(T)), 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        if constexpr (IsSameType<T, float>::value) {
            DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedTmp = typedX[KDA_SOLVE_MATRIX_ELEMENTS];
            DataCopy(typedX, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
            DataCopy(typedTmp, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(xLocal, typedX, RoundMode::CAST_NONE, KDA_SOLVE_MATRIX_ELEMENTS);
            Cast(tmpLocal, typedTmp, RoundMode::CAST_NONE, KDA_SOLVE_MATRIX_ELEMENTS);
            PipeBarrier<PIPE_V>();
        }

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
            if (storeAkk) {
                DataCopyPad(akk_[AOffset(b, hv, start, 0)], xLocal, validParams);
            }
        } else {
            LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
            Cast(typedX, xLocal, RoundMode::CAST_RINT, KDA_SOLVE_MATRIX_ELEMENTS);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(h_[xBase], typedX, KDA_SOLVE_MATRIX_ELEMENTS);
            if (storeAkk) {
                DataCopyPad(akk_[AOffset(b, hv, start, 0)], typedX, validParams);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void AddSolveTmpToXRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t rowBegin, uint64_t rowEnd, bool storeAkk)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t elemCount = rowCount * KDA_SOLVE_BT;
        DataCopyParams validParams{1, static_cast<uint16_t>(elemCount * sizeof(T)), 0, 0};
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[elemCount];
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * KDA_SOLVE_BT;
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowBegin * KDA_SOLVE_BT;
        uint64_t token = start + rowBegin;

        if constexpr (IsSameType<T, float>::value) {
            DataCopy(xLocal, h_[xBase], static_cast<uint32_t>(elemCount));
            DataCopy(tmpLocal, h_[tmpBase], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> typedTmp = typedX[elemCount];
            DataCopy(typedX, h_[xBase], static_cast<uint32_t>(elemCount));
            DataCopy(typedTmp, h_[tmpBase], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(xLocal, typedX, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            Cast(tmpLocal, typedTmp, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
        }

        Add(xLocal, xLocal, tmpLocal, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(h_[xBase], xLocal, static_cast<uint32_t>(elemCount));
            if (storeAkk) {
                DataCopyPad(akk_[AOffset(b, hv, token, 0)], xLocal, validParams);
            }
        } else {
            LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
            Cast(typedX, xLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(h_[xBase], typedX, static_cast<uint32_t>(elemCount));
            if (storeAkk) {
                DataCopyPad(akk_[AOffset(b, hv, token, 0)], typedX, validParams);
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void AddMxrDiagTmpToAkkRows(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                  uint64_t start, uint64_t rowBegin, uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MXR_BLOCK * KDA_SOLVE_MXR_BLOCK];
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        DataCopyPadParams padParams{false, 0, 0, 0};

        for (uint64_t block = 0; block < KDA_SOLVE_BT; block += KDA_SOLVE_MXR_BLOCK) {
            uint64_t blockRowBegin = rowBegin > block ? rowBegin : block;
            uint64_t blockRowEnd = rowEnd < block + KDA_SOLVE_MXR_BLOCK ? rowEnd : block + KDA_SOLVE_MXR_BLOCK;
            if (blockRowBegin >= blockRowEnd) {
                continue;
            }
            uint64_t rowCount = blockRowEnd - blockRowBegin;
            uint64_t elemCount = rowCount * KDA_SOLVE_MXR_BLOCK;
            DataCopyParams blockParams{static_cast<uint16_t>(rowCount),
                                       static_cast<uint16_t>(KDA_SOLVE_MXR_BLOCK * sizeof(T)),
                                       static_cast<uint16_t>((KDA_SOLVE_BT - KDA_SOLVE_MXR_BLOCK) * sizeof(T)),
                                       0};
            DataCopyParams outParams{static_cast<uint16_t>(rowCount),
                                     static_cast<uint16_t>(KDA_SOLVE_MXR_BLOCK * sizeof(T)),
                                     0,
                                     static_cast<uint16_t>((KDA_SOLVE_BT - KDA_SOLVE_MXR_BLOCK) * sizeof(T))};
            uint64_t rowOffset = blockRowBegin * KDA_SOLVE_BT + block;
            uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowOffset;
            uint64_t akkBase = AOffset(b, hv, start + blockRowBegin, block);

            if constexpr (IsSameType<T, float>::value) {
                DataCopyPad(xLocal, akk_[akkBase], blockParams, padParams);
                DataCopyPad(tmpLocal, h_[tmpBase], blockParams, padParams);
                SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
                WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            } else {
                LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
                LocalTensor<T> typedTmp = typedX[KDA_SOLVE_MXR_BLOCK * KDA_SOLVE_MXR_BLOCK];
                DataCopyPad(typedX, akk_[akkBase], blockParams, padParams);
                DataCopyPad(typedTmp, h_[tmpBase], blockParams, padParams);
                SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
                WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
                Cast(xLocal, typedX, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
                Cast(tmpLocal, typedTmp, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
                PipeBarrier<PIPE_V>();
            }

            Add(xLocal, xLocal, tmpLocal, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();

            if constexpr (IsSameType<T, float>::value) {
                SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
                WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
                DataCopyPad(akk_[akkBase], xLocal, outParams);
            } else {
                LocalTensor<T> typedX = vecBuf_.Get<T>()[typedOffset];
                Cast(typedX, xLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
                WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
                DataCopyPad(akk_[akkBase], typedX, outParams);
            }
            SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
            SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
            WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        }
    }

    __aicore__ inline void ComputeAkkInverseMxr64(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aBase = AOffset(b, hv, start, 0);
        uint64_t negLBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t yBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t yNextBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y1);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        for (uint32_t block = 0; block < KDA_SOLVE_BT; block += KDA_SOLVE_MXR_BLOCK) {
            CubeGemmSolveSub(h_, negLBase, block, block, h_, negLBase, block, block,
                             h_, yBase, block, block, KDA_SOLVE_MXR_BLOCK, KDA_SOLVE_MXR_BLOCK,
                             KDA_SOLVE_MXR_BLOCK);
        }

        for (uint32_t iter = 0; iter < KDA_SOLVE_MXR_DIAG_ITERS; ++iter) {
            for (uint32_t block = 0; block < KDA_SOLVE_BT; block += KDA_SOLVE_MXR_BLOCK) {
                CubeGemmSolveSub(akk_, aBase, block, block, h_, yBase, block, block,
                                 h_, tmpBase, block, block, KDA_SOLVE_MXR_BLOCK, KDA_SOLVE_MXR_BLOCK,
                                 KDA_SOLVE_MXR_BLOCK);
            }
            if (iter + 1 < KDA_SOLVE_MXR_DIAG_ITERS) {
                for (uint32_t block = 0; block < KDA_SOLVE_BT; block += KDA_SOLVE_MXR_BLOCK) {
                    CubeGemmSolveSub(h_, yBase, block, block, h_, yBase, block, block,
                                     h_, yNextBase, block, block, KDA_SOLVE_MXR_BLOCK,
                                     KDA_SOLVE_MXR_BLOCK, KDA_SOLVE_MXR_BLOCK);
                }
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (iter + 1 < KDA_SOLVE_MXR_DIAG_ITERS) {
                uint64_t oldYBase = yBase;
                yBase = yNextBase;
                yNextBase = oldYBase;
            }
        }
        ComputeAkkMerge64Cube(b, hv, chunkIdx, start);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
    }

    __aicore__ inline void ComputeAkkInverseMch64(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aBase = AOffset(b, hv, start, 0);
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t yBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t yNextBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y1);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        CubeGemmSolveSub(akk_, aBase, 0, 0, akk_, aBase, 0, 0, h_, yBase, 0, 0, KDA_SOLVE_BT,
                         KDA_SOLVE_BT, KDA_SOLVE_BT);
        for (uint32_t iter = 0; iter < KDA_SOLVE_MCH_ITERS; ++iter) {
            CubeGemmSolveSub(h_, xBase, 0, 0, h_, yBase, 0, 0, h_, tmpBase, 0, 0, KDA_SOLVE_BT,
                             KDA_SOLVE_BT, KDA_SOLVE_BT);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
            if (iter + 1 < KDA_SOLVE_MCH_ITERS) {
                CubeGemmSolveSub(h_, yBase, 0, 0, h_, yBase, 0, 0, h_, yNextBase, 0, 0, KDA_SOLVE_BT,
                                 KDA_SOLVE_BT, KDA_SOLVE_BT);
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (iter + 1 < KDA_SOLVE_MCH_ITERS) {
                uint64_t oldYBase = yBase;
                yBase = yNextBase;
                yNextBase = oldYBase;
            }
        }
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

        CubeGemmSolveSub(h_, lBase, 0, 0, h_, lBase, 0, 0, h_, yBase, 0, 0, KDA_SOLVE_BT,
                         KDA_SOLVE_BT, KDA_SOLVE_BT);
        for (uint32_t iter = 0; iter < KDA_SOLVE_MCH_ITERS; ++iter) {
            CubeGemmSolveSub(h_, xBase, 0, 0, h_, yBase, 0, 0, h_, tmpBase, 0, 0, KDA_SOLVE_BT,
                             KDA_SOLVE_BT, KDA_SOLVE_BT);
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
            if (iter + 1 < KDA_SOLVE_MCH_ITERS) {
                CubeGemmSolveSub(h_, yBase, 0, 0, h_, yBase, 0, 0, h_, yNextBase, 0, 0,
                                 KDA_SOLVE_BT, KDA_SOLVE_BT, KDA_SOLVE_BT);
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (iter + 1 < KDA_SOLVE_MCH_ITERS) {
                uint64_t oldYBase = yBase;
                yBase = yNextBase;
                yNextBase = oldYBase;
            }
        }
    }

    __aicore__ inline void CopySolveSubMatrixToL1(GlobalTensor<T> &src, uint64_t base, uint64_t row,
                                                  uint64_t col, uint32_t blockSize, uint32_t slot)
    {
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = blockSize;
        params.dValue = blockSize;
        params.srcDValue = KDA_SOLVE_BT;
        params.srcNdMatrixStride = 0;
        params.dstNzNStride = 1;
        params.dstNzC0Stride = blockSize;
        params.dstNzMatrixStride = 0;
        DataCopy(solveL1_[slot * KDA_SOLVE_MATRIX_ELEMENTS], src[base + row * KDA_SOLVE_BT + col], params);
    }

    __aicore__ inline void WaitSolveCopyToL1()
    {
        SetFlag<HardEvent::MTE2_MTE1>(KDA_SOLVE_EVT_COPY);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_SOLVE_EVT_COPY);
    }

    __aicore__ inline void LoadSolveL1ToL0A(uint32_t slot, uint32_t blockSize)
    {
        uint32_t fracNum = blockSize / KDA_SOLVE_FRAC;
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint16_t>(fracNum);
        params.srcStride = static_cast<uint16_t>(fracNum);
        params.dstGap = 0;
        params.ifTranspose = false;
        for (uint32_t idx = 0; idx < fracNum; ++idx) {
            uint32_t srcOffset = slot * KDA_SOLVE_MATRIX_ELEMENTS + idx * KDA_SOLVE_FRAC_ELEMENTS;
            uint32_t dstOffset = idx * fracNum * KDA_SOLVE_FRAC_ELEMENTS;
            LoadData(solveL0a_[dstOffset], solveL1_[srcOffset], params);
        }
    }

    __aicore__ inline void LoadSolveL1ToL0B(uint32_t slot, uint32_t blockSize)
    {
        uint32_t fracNum = blockSize / KDA_SOLVE_FRAC;
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = static_cast<uint16_t>(fracNum);
        params.srcStride = static_cast<uint16_t>(fracNum);
        params.dstGap = 0;
        params.ifTranspose = true;
        for (uint32_t idx = 0; idx < fracNum; ++idx) {
            uint32_t srcOffset = slot * KDA_SOLVE_MATRIX_ELEMENTS + idx * KDA_SOLVE_FRAC_ELEMENTS;
            uint32_t dstOffset = idx * fracNum * KDA_SOLVE_FRAC_ELEMENTS;
            LoadData(solveL0b_[dstOffset], solveL1_[srcOffset], params);
        }
    }

    __aicore__ inline void MmadSolveBlock(uint32_t blockSize, bool initC)
    {
        MmadParams params;
        params.m = blockSize;
        params.n = blockSize;
        params.k = blockSize;
        params.cmatrixInitVal = initC;
        params.cmatrixSource = false;
        params.unitFlag = 0;
        Mmad(solveL0c_, solveL0a_, solveL0b_, params);
    }

    __aicore__ inline void StoreSolveL0CToL1Tmp(uint64_t tmpBase, uint32_t blockSize)
    {
        auto params = AscendC::FixpipeParamsV220(blockSize, blockSize, blockSize, blockSize, false);
        if constexpr (IsSameType<T, half>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(h_[tmpBase], solveL0c_, params);
        SetFlag<HardEvent::FIX_MTE2>(KDA_SOLVE_EVT_COPY);
        WaitFlag<HardEvent::FIX_MTE2>(KDA_SOLVE_EVT_COPY);

        Nd2NzParams nd2nz;
        nd2nz.ndNum = 1;
        nd2nz.nValue = blockSize;
        nd2nz.dValue = blockSize;
        nd2nz.srcDValue = blockSize;
        nd2nz.srcNdMatrixStride = 0;
        nd2nz.dstNzNStride = 1;
        nd2nz.dstNzC0Stride = blockSize;
        nd2nz.dstNzMatrixStride = 0;
        DataCopy(solveL1_[KDA_SOLVE_L1_SLOT_TMP * KDA_SOLVE_MATRIX_ELEMENTS], h_[tmpBase], nd2nz);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_SOLVE_EVT_COPY);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_SOLVE_EVT_COPY);
    }

    __aicore__ inline void StoreSolveL0CToAkk(uint64_t base, uint64_t row, uint64_t col, uint32_t blockSize)
    {
        auto params = AscendC::FixpipeParamsV220(blockSize, blockSize, blockSize, KDA_SOLVE_BT, false);
        if constexpr (IsSameType<T, half>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<T, float, AscendC::CFG_ROW_MAJOR>(
            akk_[base + row * KDA_SOLVE_BT + col], solveL0c_, params);
        SetFlag<HardEvent::FIX_MTE2>(KDA_SOLVE_EVT_COPY);
        WaitFlag<HardEvent::FIX_MTE2>(KDA_SOLVE_EVT_COPY);
    }

    __aicore__ inline void MxrMergePairLocal(uint64_t aiBase, uint64_t negLBase, uint64_t tmpBase,
                                             uint32_t lowerStart, uint32_t upperStart, uint32_t blockSize)
    {
        CopySolveSubMatrixToL1(akk_, aiBase, lowerStart, lowerStart, blockSize, KDA_SOLVE_L1_SLOT_A);
        CopySolveSubMatrixToL1(h_, negLBase, lowerStart, upperStart, blockSize, KDA_SOLVE_L1_SLOT_B);
        WaitSolveCopyToL1();
        LoadSolveL1ToL0A(KDA_SOLVE_L1_SLOT_A, blockSize);
        LoadSolveL1ToL0B(KDA_SOLVE_L1_SLOT_B, blockSize);
        SetFlag<HardEvent::MTE1_M>(KDA_SOLVE_EVT_MMAD);
        WaitFlag<HardEvent::MTE1_M>(KDA_SOLVE_EVT_MMAD);
        MmadSolveBlock(blockSize, true);
        SetFlag<HardEvent::M_FIX>(KDA_SOLVE_EVT_MMAD);
        WaitFlag<HardEvent::M_FIX>(KDA_SOLVE_EVT_MMAD);
        StoreSolveL0CToL1Tmp(tmpBase, blockSize);

        CopySolveSubMatrixToL1(akk_, aiBase, upperStart, upperStart, blockSize, KDA_SOLVE_L1_SLOT_B);
        WaitSolveCopyToL1();
        LoadSolveL1ToL0A(KDA_SOLVE_L1_SLOT_TMP, blockSize);
        LoadSolveL1ToL0B(KDA_SOLVE_L1_SLOT_B, blockSize);
        SetFlag<HardEvent::MTE1_M>(KDA_SOLVE_EVT_MMAD);
        WaitFlag<HardEvent::MTE1_M>(KDA_SOLVE_EVT_MMAD);
        MmadSolveBlock(blockSize, true);
        SetFlag<HardEvent::M_FIX>(KDA_SOLVE_EVT_MMAD);
        WaitFlag<HardEvent::M_FIX>(KDA_SOLVE_EVT_MMAD);
        StoreSolveL0CToAkk(aiBase, lowerStart, upperStart, blockSize);
    }

    __aicore__ inline void ComputeAkkMerge64Cube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aiBase = AOffset(b, hv, start, 0);
        uint64_t negLBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);

        MxrMergePairLocal(aiBase, negLBase, tmpBase, 16, 0, 16);
        MxrMergePairLocal(aiBase, negLBase, tmpBase, 48, 32, 16);
        MxrMergePairLocal(aiBase, negLBase, tmpBase, 32, 0, 32);
    }

    __aicore__ inline void FinalizeAqkAkk(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            float aqkRow[EXP2_UB_ELEMENTS];
            float akkRow[EXP2_UB_ELEMENTS];
            for (uint64_t j = 0; j < BT_; ++j) {
                float aqkValue = 0.0f;
                float akkValue = 0.0f;
                if (j < curT && j <= i) {
                    aqkValue = ReadAsFloat(aqk_, AOffset(b, hv, ti, j)) * (stage_ == 1 ? 1.0f : scale_);
                    if (j < i) {
                        akkValue = ReadAsFloat(akk_, AOffset(b, hv, ti, j)) *
                                   ReadFloat(beta_, BetaOffset(b, hv, ti));
                    }
                }
                aqkRow[j] = aqkValue;
                akkRow[j] = akkValue;
            }

            for (uint64_t j = 0; j < i; ++j) {
                float sum = 0.0f;
                for (uint64_t m = j; m < i; ++m) {
                    float lim = akkRow[m];
                    float ymj = ReadAsFloat(akk_, AOffset(b, hv, start + m, j));
                    sum += lim * ymj;
                }
                akkRow[j] = -sum;
            }
            akkRow[i] = 1.0f;
            CopyStackFloatRowOut(aqk_, AOffset(b, hv, ti, 0), aqkRow, BT_, 0);
            CopyStackFloatRowOut(akk_, AOffset(b, hv, ti, 0), akkRow, BT_, 1);
        }
    }

    __aicore__ inline void ComputePostKgWUVec(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                              uint64_t start, uint64_t curT)
    {
        uint64_t last = start + curT - 1;
        uint64_t wScratchBase = HOffset(b, hv, chunkIdx, 0, 0);
        LocalTensor<float> rowLocal = VecScratch(0);
        LocalTensor<float> accLocal = VecScratch(1);
        LocalTensor<float> tmpLocal = VecScratch(2);
        LocalTensor<float> gateLast = VecScratch(4);
        LoadExp2GTo(gateLast, b, hv, last);

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            LoadAsFloatRow(kg_, KVOffset(b, hv, ti, 0, K_), rowLocal, K_);
            Mul(tmpLocal, rowLocal, gateLast, static_cast<uint32_t>(K_));
            PipeBarrier<PIPE_V>();
            StoreFloatRow(kg_, KVOffset(b, hv, ti, 0, K_), tmpLocal, K_);
        }

        LocalTensor<float> gateLocal = VecScratch(5);

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;

            Duplicate(accLocal, 0.0f, static_cast<uint32_t>(K_));
            PipeBarrier<PIPE_V>();
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float coeff = ReadAsFloat(akk_, AOffset(b, hv, ti, j)) * ReadFloat(beta_, BetaOffset(b, hv, tj));
                LoadAsFloatRow(k_, QOffset(b, h, tj, 0), rowLocal, K_);
                LoadExp2GTo(gateLocal, b, hv, tj);
                Mul(rowLocal, rowLocal, gateLocal, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Muls(tmpLocal, rowLocal, coeff, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                Add(accLocal, accLocal, tmpLocal, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
            }
            StoreFloatRow(h_, wScratchBase + i * K_, accLocal, K_);

            Duplicate(accLocal, 0.0f, static_cast<uint32_t>(V_));
            PipeBarrier<PIPE_V>();
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float coeff = ReadAsFloat(akk_, AOffset(b, hv, ti, j)) * ReadFloat(beta_, BetaOffset(b, hv, tj));
                LoadAsFloatRow(v_, KVOffset(b, hv, tj, 0, V_), rowLocal, V_);
                Muls(tmpLocal, rowLocal, coeff, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                Add(accLocal, accLocal, tmpLocal, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
            }
            StoreFloatRow(u_, KVOffset(b, hv, ti, 0, V_), accLocal, V_);
        }
        for (uint64_t i = 0; i < curT; ++i) {
            CopyTensorRow(w_, KVOffset(b, hv, start + i, 0, K_), h_, wScratchBase + i * K_, K_);
        }
    }

    __aicore__ inline void ComputePostKgWUScalar(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                 uint64_t curT)
    {
        uint64_t last = start + curT - 1;
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            LocalTensor<float> expLastMinusG = Exp2GDiff(b, hv, last, ti);
            __ubuf__ float *expLastMinusGPtr = UbPtr(expLastMinusG);
            float kgRow[EXP2_UB_ELEMENTS];
            for (uint64_t d = 0; d < K_; ++d) {
                float kv = ReadAsFloat(k_, QOffset(b, h, ti, d));
                kgRow[d] = kv * expLastMinusGPtr[d];
            }
            CopyStackFloatRowOut(kg_, KVOffset(b, hv, ti, 0, K_), kgRow, K_, 2);

            float wSum[EXP2_UB_ELEMENTS];
            for (uint64_t d = 0; d < K_; ++d) {
                wSum[d] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                LocalTensor<float> expGj = Exp2G(b, hv, tj);
                __ubuf__ float *expGjPtr = UbPtr(expGj);
                float betaJ = ReadFloat(beta_, BetaOffset(b, hv, tj));
                float a = ReadAsFloat(akk_, AOffset(b, hv, ti, j));
                for (uint64_t d = 0; d < K_; ++d) {
                    float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
                    wSum[d] += a * kj * betaJ * expGjPtr[d];
                }
            }
            CopyStackFloatRowOut(w_, KVOffset(b, hv, ti, 0, K_), wSum, K_, 3);

            float uRow[EXP2_UB_ELEMENTS];
            for (uint64_t r = 0; r < V_; ++r) {
                uRow[r] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float vRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(v_, KVOffset(b, hv, tj, 0, V_), vRow, V_, 0);
                float a = ReadAsFloat(akk_, AOffset(b, hv, ti, j));
                float betaJ = ReadFloat(beta_, BetaOffset(b, hv, tj));
                for (uint64_t r = 0; r < V_; ++r) {
                    uRow[r] += a * vRow[r] * betaJ;
                }
            }
            CopyStackFloatRowOut(u_, KVOffset(b, hv, ti, 0, V_), uRow, V_, 4);
        }
    }

    __aicore__ inline void ComputeAqkAkkPostLocal(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                  uint64_t curT)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena[KDA_LOCAL_AQK_OFFSET];
        LocalTensor<float> akkInvMat = arena[KDA_LOCAL_AKK_OFFSET];
        __ubuf__ float *aqkPtr = UbPtr(aqkMat);
        __ubuf__ float *akkPtr = UbPtr(akkInvMat);

        for (uint64_t idx = 0; idx < KDA_SOLVE_MATRIX_ELEMENTS; ++idx) {
            aqkPtr[idx] = 0.0f;
            akkPtr[idx] = 0.0f;
        }

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            float lRow[EXP2_UB_ELEMENTS];
            for (uint64_t j = 0; j < BT_; ++j) {
                lRow[j] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float aqkRaw = 0.0f;
                float akkRaw = 0.0f;
                LocalTensor<float> gateLocal = Exp2GDiff(b, hv, ti, tj);
                __ubuf__ float *gatePtr = UbPtr(gateLocal);
                for (uint64_t d = 0; d < K_; ++d) {
                    float qi = ReadAsFloat(q_, QOffset(b, h, ti, d));
                    float ki = ReadAsFloat(k_, QOffset(b, h, ti, d));
                    float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
                    aqkRaw += qi * kj * gatePtr[d];
                    akkRaw += ki * kj * gatePtr[d];
                }
                if (j <= i) {
                    aqkPtr[i * BT_ + j] = aqkRaw;
                    if (j < i) {
                        lRow[j] = akkRaw * ReadFloat(beta_, BetaOffset(b, hv, ti));
                    }
                }
            }

            for (uint64_t j = 0; j < i; ++j) {
                float sum = 0.0f;
                for (uint64_t m = j; m < i; ++m) {
                    sum += lRow[m] * akkPtr[m * BT_ + j];
                }
                akkPtr[i * BT_ + j] = -sum;
            }
            akkPtr[i * BT_ + i] = 1.0f;
        }

        float row[EXP2_UB_ELEMENTS];
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            for (uint64_t j = 0; j < BT_; ++j) {
                row[j] = aqkPtr[i * BT_ + j];
            }
            CopyStackFloatRowOut(aqk_, AOffset(b, hv, ti, 0), row, BT_, 0);
            for (uint64_t j = 0; j < BT_; ++j) {
                row[j] = akkPtr[i * BT_ + j];
            }
            CopyStackFloatRowOut(akk_, AOffset(b, hv, ti, 0), row, BT_, 1);
        }

        uint64_t last = start + curT - 1;
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            LocalTensor<float> expLastMinusG = Exp2GDiff(b, hv, last, ti);
            __ubuf__ float *expLastMinusGPtr = UbPtr(expLastMinusG);
            float kgRow[EXP2_UB_ELEMENTS];
            for (uint64_t d = 0; d < K_; ++d) {
                float kv = ReadAsFloat(k_, QOffset(b, h, ti, d));
                kgRow[d] = kv * expLastMinusGPtr[d];
            }
            CopyStackFloatRowOut(kg_, KVOffset(b, hv, ti, 0, K_), kgRow, K_, 2);

            float wSum[EXP2_UB_ELEMENTS];
            for (uint64_t d = 0; d < K_; ++d) {
                wSum[d] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                LocalTensor<float> expGj = Exp2G(b, hv, tj);
                __ubuf__ float *expGjPtr = UbPtr(expGj);
                float betaJ = ReadFloat(beta_, BetaOffset(b, hv, tj));
                float a = akkPtr[i * BT_ + j];
                for (uint64_t d = 0; d < K_; ++d) {
                    float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
                    wSum[d] += a * kj * betaJ * expGjPtr[d];
                }
            }
            CopyStackFloatRowOut(w_, KVOffset(b, hv, ti, 0, K_), wSum, K_, 3);

            float uRow[EXP2_UB_ELEMENTS];
            for (uint64_t r = 0; r < V_; ++r) {
                uRow[r] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float vRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(v_, KVOffset(b, hv, tj, 0, V_), vRow, V_, 0);
                float a = akkPtr[i * BT_ + j];
                float betaJ = ReadFloat(beta_, BetaOffset(b, hv, tj));
                for (uint64_t r = 0; r < V_; ++r) {
                    uRow[r] += a * vRow[r] * betaJ;
                }
            }
            CopyStackFloatRowOut(u_, KVOffset(b, hv, ti, 0, V_), uRow, V_, 4);
        }
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
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            DataCopy(matrixTyped, src[baseOffset], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
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
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(dst[baseOffset], matrixLocal, static_cast<uint32_t>(elemCount));
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            Cast(matrixTyped, matrixLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(dst[baseOffset], matrixTyped, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
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

    __aicore__ inline void ComputePostWuCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LocalPostL1TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                                   std::is_same<ElementB, float>::value,
                                                               KdaFloatPostL1TileShape, KdaPostL1TileShape>::type;
        using LocalPostL0TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                                   std::is_same<ElementB, float>::value,
                                                               KdaFloatPostL0TileShape, KdaPostL0TileShape>::type;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, LocalPostL1TileShape,
                                                              LocalPostL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(BT_, BT_);
        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto tensorA = tla::MakeTensor(akk_[AOffset(b, hv, start, 0)], layoutA, Catlass::Arch::PositionGM{});

        {
            LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(BT_, K_);
            LayoutTagC tagC = LayoutTagC::template MakeLayout<ElementC>(BT_, K_);
            auto layoutB = tla::MakeLayoutFromTag(tagB);
            auto layoutC = tla::MakeLayoutFromTag(tagC);
            Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(K_),
                                     static_cast<uint32_t>(curT)};
            auto tensorB = tla::MakeTensor(w_[KVOffset(b, hv, start, 0, K_)], layoutB,
                                           Catlass::Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(h_[WScratchOffset(b, hv, chunkIdx, 0, 0)], layoutC,
                                            Catlass::Arch::PositionGM{});
            auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
            auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
            auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
            blockMmad(blockA, blockB, blockC, shape);
            PipeBarrier<PIPE_ALL>();
        }

        {
            LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(BT_, V_);
            LayoutTagC tagC = LayoutTagC::template MakeLayout<ElementC>(BT_, V_);
            auto layoutB = tla::MakeLayoutFromTag(tagB);
            auto layoutC = tla::MakeLayoutFromTag(tagC);
            Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(V_),
                                     static_cast<uint32_t>(curT)};
            auto tensorB = tla::MakeTensor(vNew_[KVOffset(b, hv, start, 0, V_)], layoutB,
                                           Catlass::Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(u_[KVOffset(b, hv, start, 0, V_)], layoutC,
                                           Catlass::Arch::PositionGM{});
            auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
            auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
            auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
            blockMmad(blockA, blockB, blockC, shape);
            PipeBarrier<PIPE_ALL>();
        }

    }

    __aicore__ inline void CopyScratchWAndFinalizeKg(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                                     uint64_t start, uint64_t curT)
    {
        (void)h;
        constexpr uint64_t vecElemsPerRepeat = 64;
        constexpr uint64_t matrixOffset = 1024;
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t last = start + curT - 1;
        uint64_t elemCount = curT * K_;
        uint64_t repeatStride = K_ * sizeof(float) / 32;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> gateLast = arena;
        LocalTensor<float> matrixLocal = arena[matrixOffset];
        uint64_t scratchBase = WScratchOffset(b, hv, chunkIdx, 0, 0);

        if constexpr (IsSameType<T, float>::value) {
            DataCopy(matrixLocal, h_[scratchBase], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_MTE3>(KDA_MTE2_MTE3_EVENT_ID);
            WaitFlag<HardEvent::MTE2_MTE3>(KDA_MTE2_MTE3_EVENT_ID);
            DataCopy(w_[KVOffset(b, hv, start, 0, K_)], matrixLocal, static_cast<uint32_t>(elemCount));
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            DataCopy(matrixTyped, h_[scratchBase], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_MTE3>(KDA_MTE2_MTE3_EVENT_ID);
            WaitFlag<HardEvent::MTE2_MTE3>(KDA_MTE2_MTE3_EVENT_ID);
            DataCopy(w_[KVOffset(b, hv, start, 0, K_)], matrixTyped, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);

        LoadExp2GTo(gateLast, b, hv, last);

        if constexpr (IsSameType<T, float>::value) {
            DataCopy(matrixLocal, kg_[KVOffset(b, hv, start, 0, K_)], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            DataCopy(matrixTyped, kg_[KVOffset(b, hv, start, 0, K_)], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            WaitFlag<HardEvent::MTE2_V>(KDA_MTE2_V_EVENT_ID);
            Cast(matrixLocal, matrixTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
        }

        for (uint64_t col = 0; col < K_; col += vecElemsPerRepeat) {
            uint64_t mask = K_ - col;
            if (mask > vecElemsPerRepeat) {
                mask = vecElemsPerRepeat;
            }
            Mul(matrixLocal[col], matrixLocal[col], gateLast[col], mask, static_cast<uint8_t>(curT),
                {1, 1, 1, static_cast<uint8_t>(repeatStride), static_cast<uint8_t>(repeatStride), 0});
            PipeBarrier<PIPE_V>();
        }

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(kg_[KVOffset(b, hv, start, 0, K_)], matrixLocal, static_cast<uint32_t>(elemCount));
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            Cast(matrixTyped, matrixLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            WaitFlag<HardEvent::V_MTE3>(KDA_SCALAR_V_MTE3_EVENT_ID);
            DataCopy(kg_[KVOffset(b, hv, start, 0, K_)], matrixTyped, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(KDA_MTE3_MTE2_EVENT_ID);
        SetFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
        WaitFlag<HardEvent::MTE3_V>(KDA_SCALAR_MTE3_V_EVENT_ID);
    }

    __aicore__ inline void InitState(uint64_t seq, uint64_t hv)
    {
        for (uint64_t d = 0; d < K_; ++d) {
            uint64_t stateOffset = StateOffset(seq, hv, d, 0);
            if (hasInitial_) {
                CopyTensorRow(finalState_, stateOffset, initialState_, stateOffset, V_);
            } else {
                ZeroTensorRow(finalState_, stateOffset, V_);
            }
        }
    }

    __aicore__ inline void StoreCurrentState(uint64_t b, uint64_t hv, uint64_t seq, uint64_t chunkIdx)
    {
        for (uint64_t d = 0; d < K_; ++d) {
            CopyTensorRow(h_, HOffset(b, hv, chunkIdx, d, 0), finalState_, StateOffset(seq, hv, d, 0), V_);
        }
    }

    __aicore__ inline void ProcessChunkAiv(uint64_t b, uint64_t seq, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                           uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }

        if constexpr (IsSameType<T, float>::value) {
            PrepareGateProducts(b, h, hv, start, curT, 0, 1);
            ComputeRawAqkAkkScalar(b, h, hv, start, curT);
        } else {
            uint64_t subBlockIdx = isAivOnly_ ? 0 : static_cast<uint64_t>(GetSubBlockIdx());
            if (K_ < 16 || curT == 1) {
                if (!isAivOnly_ && subBlockIdx != 0) {
                    return;
                }
                PrepareGateProducts(b, h, hv, start, curT, 0, 1);
                ComputeRawAqkAkkScalar(b, h, hv, start, curT);
            } else {
                if (subBlockIdx == 0) {
                    PrepareGateProducts(b, h, hv, start, curT, 0, 1);
                    PipeBarrier<PIPE_ALL>();
                }
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
                if (subBlockIdx != 0) {
                    return;
                }
            }
        }
        FinalizeAqkAkk(b, hv, start, curT);

        uint64_t last = end - 1;
        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            LocalTensor<float> expLastMinusG = Exp2GDiff(b, hv, last, ti);
            __ubuf__ float *expLastMinusGPtr = UbPtr(expLastMinusG);
            float kgRow[EXP2_UB_ELEMENTS];
            for (uint64_t d = 0; d < K_; ++d) {
                float kv = ReadAsFloat(k_, QOffset(b, h, ti, d));
                kgRow[d] = kv * expLastMinusGPtr[d];
            }
            CopyStackFloatRowOut(kg_, KVOffset(b, hv, ti, 0, K_), kgRow, K_, 2);

            float wSum[EXP2_UB_ELEMENTS];
            for (uint64_t d = 0; d < K_; ++d) {
                wSum[d] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                LocalTensor<float> expGj = Exp2G(b, hv, tj);
                __ubuf__ float *expGjPtr = UbPtr(expGj);
                float betaJ = ReadFloat(beta_, BetaOffset(b, hv, tj));
                float a = ReadAsFloat(akk_, AOffset(b, hv, ti, j));
                for (uint64_t d = 0; d < K_; ++d) {
                    float kj = ReadAsFloat(k_, QOffset(b, h, tj, d));
                    wSum[d] += a * kj * betaJ * expGjPtr[d];
                }
            }
            CopyStackFloatRowOut(w_, KVOffset(b, hv, ti, 0, K_), wSum, K_, 3);

            float uRow[EXP2_UB_ELEMENTS];
            for (uint64_t r = 0; r < V_; ++r) {
                uRow[r] = 0.0f;
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float vRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(v_, KVOffset(b, hv, tj, 0, V_), vRow, V_, 0);
                float a = ReadAsFloat(akk_, AOffset(b, hv, ti, j));
                float betaJ = ReadFloat(beta_, BetaOffset(b, hv, tj));
                for (uint64_t r = 0; r < V_; ++r) {
                    uRow[r] += a * vRow[r] * betaJ;
                }
            }
            CopyStackFloatRowOut(u_, KVOffset(b, hv, ti, 0, V_), uRow, V_, 4);

            float vNewRow[EXP2_UB_ELEMENTS];
            for (uint64_t r = 0; r < V_; ++r) {
                vNewRow[r] = uRow[r];
            }
            for (uint64_t d = 0; d < K_; ++d) {
                float hRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(finalState_, StateOffset(seq, hv, d, 0), hRow, V_, 1);
                float wi = wSum[d];
                for (uint64_t r = 0; r < V_; ++r) {
                    vNewRow[r] -= wi * hRow[r];
                }
            }
            CopyStackFloatRowOut(vNew_, KVOffset(b, hv, ti, 0, V_), vNewRow, V_, 2);
        }

        StoreCurrentState(b, hv, seq, chunkIdx);

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            float oRow[EXP2_UB_ELEMENTS];
            for (uint64_t r = 0; r < V_; ++r) {
                oRow[r] = 0.0f;
            }
            float qgRow[EXP2_UB_ELEMENTS];
            LoadStackFloatRow(qg_, KVOffset(b, hv, ti, 0, K_), qgRow, K_, 0);
            for (uint64_t d = 0; d < K_; ++d) {
                float hRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(finalState_, StateOffset(seq, hv, d, 0), hRow, V_, 1);
                float qgValue = qgRow[d] * scale_;
                for (uint64_t r = 0; r < V_; ++r) {
                    oRow[r] += qgValue * hRow[r];
                }
            }
            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float vNewRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(vNew_, KVOffset(b, hv, tj, 0, V_), vNewRow, V_, 2);
                float a = ReadAsFloat(aqk_, AOffset(b, hv, ti, j));
                for (uint64_t r = 0; r < V_; ++r) {
                    oRow[r] += a * vNewRow[r];
                }
            }
            CopyStackFloatRowOut(o_, KVOffset(b, hv, ti, 0, V_), oRow, V_, 3);
        }

        LocalTensor<float> decayGate = Exp2G(b, hv, last);
        __ubuf__ float *decayGatePtr = UbPtr(decayGate);
        float decayValues[EXP2_UB_ELEMENTS];
        for (uint64_t d = 0; d < K_; ++d) {
            decayValues[d] = decayGatePtr[d];
        }
        for (uint64_t d = 0; d < K_; ++d) {
            float decay = decayValues[d];
            float stateRow[EXP2_UB_ELEMENTS];
            LoadStackFloatRow(finalState_, StateOffset(seq, hv, d, 0), stateRow, V_, 0);
            for (uint64_t r = 0; r < V_; ++r) {
                stateRow[r] *= decay;
            }
            for (uint64_t i = 0; i < curT; ++i) {
                uint64_t ti = start + i;
                float kgRow[EXP2_UB_ELEMENTS];
                float vNewRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(kg_, KVOffset(b, hv, ti, 0, K_), kgRow, K_, 1);
                LoadStackFloatRow(vNew_, KVOffset(b, hv, ti, 0, V_), vNewRow, V_, 2);
                float kgValue = kgRow[d];
                for (uint64_t r = 0; r < V_; ++r) {
                    stateRow[r] += kgValue * vNewRow[r];
                }
            }
            CopyStackFloatRowOut(finalState_, StateOffset(seq, hv, d, 0), stateRow, V_, 4);
        }
    }

    __aicore__ inline void ProcessChunkAic(uint64_t b, uint64_t hv, uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || K_ < 16 || curT == 1) {
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
        ComputeRawAqkAkkCube(b, hv, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
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
        using LocalL1TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                               std::is_same<ElementB, float>::value,
                                                           KdaFloatL1TileShape, KdaL1TileShape>::type;
        using LocalL0TileShape = typename std::conditional<std::is_same<ElementA, float>::value &&
                                                               std::is_same<ElementB, float>::value,
                                                           KdaFloatL0TileShape, KdaL0TileShape>::type;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, LocalL1TileShape, LocalL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);

        auto layoutQ = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutH = tla::MakeLayout<ElementB, LayoutTagB>(K_, V_);
        auto layoutO = tla::MakeLayout<ElementC, LayoutTagC>(BT_, V_);
        Catlass::GemmCoord shapeQH{static_cast<uint32_t>(curT), static_cast<uint32_t>(V_),
                                   static_cast<uint32_t>(K_)};
        auto tensorQ = tla::MakeTensor(qg_[KVOffset(b, hv, start, 0, K_)], layoutQ,
                                       Catlass::Arch::PositionGM{});
        auto tensorH = tla::MakeTensor(h_[HOffset(b, hv, chunkIdx, 0, 0)], layoutH,
                                       Catlass::Arch::PositionGM{});
        auto tensorO = tla::MakeTensor(o_[KVOffset(b, hv, start, 0, V_)], layoutO,
                                       Catlass::Arch::PositionGM{});
        auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.k()));
        auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.k(), shapeQH.n()));
        auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.n()));
        blockMmad(blockQ, blockH, blockO, shapeQH);
        PipeBarrier<PIPE_ALL>();

        auto layoutAqk = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutV = tla::MakeLayout<ElementB, LayoutTagB>(BT_, V_);
        Catlass::GemmCoord shapeAV{static_cast<uint32_t>(curT), static_cast<uint32_t>(V_),
                                   static_cast<uint32_t>(curT)};
        auto tensorAqk = tla::MakeTensor(aqk_[AOffset(b, hv, start, 0)], layoutAqk,
                                         Catlass::Arch::PositionGM{});
        auto tensorVNew = tla::MakeTensor(vNew_[KVOffset(b, hv, start, 0, V_)], layoutV,
                                          Catlass::Arch::PositionGM{});
        auto tensorLocal = tla::MakeTensor(u_[KVOffset(b, hv, start, 0, V_)], layoutO,
                                           Catlass::Arch::PositionGM{});
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.k()));
        auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.k(), shapeAV.n()));
        auto blockLocal = GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.n()));
        blockMmad(blockAqk, blockVNew, blockLocal, shapeAV);
        PipeBarrier<PIPE_ALL>();
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
            StoreFloatRow(o_, KVOffset(b, hv, ti, 0, V_), outLocal, V_);
        }
    }

    __aicore__ inline void ComputeOutputScalarRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                   uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        for (uint64_t i = subBlockIdx; i < curT; i += subBlockNum) {
            uint64_t ti = start + i;
            float oRow[EXP2_UB_ELEMENTS];
            for (uint64_t r = 0; r < V_; ++r) {
                oRow[r] = 0.0f;
            }

            float qgRow[EXP2_UB_ELEMENTS];
            LoadStackFloatRow(qg_, KVOffset(b, hv, ti, 0, K_), qgRow, K_, 0);
            for (uint64_t d = 0; d < K_; ++d) {
                float hRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(h_, HOffset(b, hv, chunkIdx, d, 0), hRow, V_, 1);
                float qgValue = qgRow[d];
                for (uint64_t r = 0; r < V_; ++r) {
                    oRow[r] += qgValue * hRow[r];
                }
            }

            for (uint64_t j = 0; j < curT; ++j) {
                uint64_t tj = start + j;
                float vNewRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(vNew_, KVOffset(b, hv, tj, 0, V_), vNewRow, V_, 2);
                float a = ReadAsFloat(aqk_, AOffset(b, hv, ti, j));
                for (uint64_t r = 0; r < V_; ++r) {
                    oRow[r] += a * vNewRow[r];
                }
            }
            CopyStackFloatRowOut(o_, KVOffset(b, hv, ti, 0, V_), oRow, V_, 3);
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
                seq = static_cast<uint64_t>(ReadMetaInt64(chunkIndices_, flatChunk * 2));
                uint64_t localChunk = static_cast<uint64_t>(ReadMetaInt64(chunkIndices_, flatChunk * 2 + 1));
                uint64_t seqStart = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, seq));
                uint64_t seqEnd = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, seq + 1));
                b = 0;
                chunkIdx = flatChunk;
                start = seqStart + localChunk * BT_;
                end = start + BT_;
                if (end > seqEnd) {
                    end = seqEnd;
                }
                h = hv / (HV_ / H_);
                return start < end;
            } else {
                uint64_t remain = flatChunk;
                for (uint64_t s = 0; s < N_; ++s) {
                    uint64_t seqStart = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, s));
                    uint64_t seqEnd = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, s + 1));
                    uint64_t chunks = (seqEnd - seqStart + BT_ - 1) / BT_;
                    if (remain < chunks) {
                        seq = s;
                        b = 0;
                        chunkIdx = flatChunk;
                        start = seqStart + remain * BT_;
                        end = start + BT_;
                        if (end > seqEnd) {
                            end = seqEnd;
                        }
                        h = hv / (HV_ / H_);
                        return start < end;
                    }
                    remain -= chunks;
                }
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
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if (K_ < 16 || curT == 1) {
            if (subBlockIdx != 0) {
                return;
            }
            PrepareGateProducts(b, h, hv, start, curT, 0, 1);
            ComputeRawAqkAkkScalar(b, h, hv, start, curT);
            FinalizeAqkAkk(b, hv, start, curT);
            ComputePostKgWUVec(b, h, hv, chunkIdx, start, curT);
            return;
        }

        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(curT, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
        PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
        if (useAkkCubeSolve) {
            if (curT == KDA_SOLVE_BT) {
                PrepareAqkAkkMxrInputRows(b, hv, chunkIdx, start, solveRowBegin, solveRowEnd);
            } else if (subBlockIdx == 0) {
                PrepareAqkAkkSolveInputTail(b, hv, chunkIdx, start, curT);
            }
        } else {
            if (subBlockIdx == 0) {
                FinalizeAqkAkk(b, hv, start, curT);
                if (!usePostWuCube) {
                    ComputePostKgWUVec(b, h, hv, chunkIdx, start, curT);
                }
            }
        }
        if (useAkkCubeSolve) {
            if (curT == KDA_SOLVE_BT) {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
                for (uint32_t iter = 0; iter < KDA_SOLVE_MXR_DIAG_ITERS; ++iter) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
                    AddMxrDiagTmpToAkkRows(b, hv, chunkIdx, start, solveRowBegin, solveRowEnd);
                    Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
                }
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
            } else {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
                for (uint32_t iter = 0; iter < KDA_SOLVE_MCH_ITERS; ++iter) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
                    if (subBlockIdx == 0) {
                        AddSolveTmpToXTail(b, hv, chunkIdx, start, curT, iter + 1 == KDA_SOLVE_MCH_ITERS);
                    }
                    Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
                }
            }
        }
        if (!usePostWuCube) {
            return;
        }
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void ProcessChunkPreAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || K_ < 16 || curT == 1) {
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
        ComputeRawAqkAkkCube(b, hv, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        if (useAkkCubeSolve) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (curT == KDA_SOLVE_BT) {
                ComputeAkkInverseMxr64(b, hv, chunkIdx, start);
            } else {
                ComputeAkkInverseMchTail(b, hv, chunkIdx, start, curT);
            }
        }
        if (!useAkkCubeSolve && !usePostWuCube) {
            return;
        }
        (void)chunkIdx;
    }

    __aicore__ inline void ProcessChunkPostAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                               uint64_t start, uint64_t end, uint64_t subBlockIdx)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
        if (subBlockIdx == 0) {
            CopyScratchWAndFinalizeKg(b, h, hv, chunkIdx, start, curT);
        }
    }

    __aicore__ inline void ProcessChunkPostAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                               uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        ComputePostWuCube(b, hv, chunkIdx, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
    }

    __aicore__ inline void ProcessChunkOutAiv(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t end, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if (curT == 1) {
            if (subBlockIdx == 0) {
                ComputeOutputScalarRows(b, hv, chunkIdx, start, curT, 0, 1);
            }
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
        FinalizeOutputRows(b, hv, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void ProcessChunkOutAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || curT == 1) {
            return;
        }
        ComputeOutputCube(b, hv, chunkIdx, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
    }

    __aicore__ inline void ProcessChunkStateAiv(uint64_t b, uint64_t seq, uint64_t hv, uint64_t chunkIdx,
                                                uint64_t start, uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        StoreCurrentState(b, hv, seq, chunkIdx);

        for (uint64_t i = 0; i < curT; ++i) {
            uint64_t ti = start + i;
            float vNewRow[EXP2_UB_ELEMENTS];
            LoadStackFloatRow(u_, KVOffset(b, hv, ti, 0, V_), vNewRow, V_, 0);
            for (uint64_t d = 0; d < K_; ++d) {
                float hRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(finalState_, StateOffset(seq, hv, d, 0), hRow, V_, 1);
                float wi = ReadAsFloat(w_, KVOffset(b, hv, ti, d, K_));
                for (uint64_t r = 0; r < V_; ++r) {
                    vNewRow[r] -= wi * hRow[r];
                }
            }
            CopyStackFloatRowOut(vNew_, KVOffset(b, hv, ti, 0, V_), vNewRow, V_, 2);
        }

        uint64_t last = end - 1;
        LocalTensor<float> decayGate = Exp2G(b, hv, last);
        __ubuf__ float *decayGatePtr = UbPtr(decayGate);
        float decayValues[EXP2_UB_ELEMENTS];
        for (uint64_t d = 0; d < K_; ++d) {
            decayValues[d] = decayGatePtr[d];
        }

        for (uint64_t d = 0; d < K_; ++d) {
            float stateRow[EXP2_UB_ELEMENTS];
            LoadStackFloatRow(finalState_, StateOffset(seq, hv, d, 0), stateRow, V_, 0);
            float decay = decayValues[d];
            for (uint64_t r = 0; r < V_; ++r) {
                stateRow[r] *= decay;
            }
            for (uint64_t i = 0; i < curT; ++i) {
                uint64_t ti = start + i;
                float kgRow[EXP2_UB_ELEMENTS];
                float vNewRow[EXP2_UB_ELEMENTS];
                LoadStackFloatRow(kg_, KVOffset(b, hv, ti, 0, K_), kgRow, K_, 1);
                LoadStackFloatRow(vNew_, KVOffset(b, hv, ti, 0, V_), vNewRow, V_, 2);
                float kgValue = kgRow[d];
                for (uint64_t r = 0; r < V_; ++r) {
                    stateRow[r] += kgValue * vNewRow[r];
                }
            }
            CopyStackFloatRowOut(finalState_, StateOffset(seq, hv, d, 0), stateRow, V_, 4);
        }
    }

    __aicore__ inline void ProcessStateSeqHeadAiv(uint64_t seq, uint64_t hv)
    {
        uint64_t b = 0;
        uint64_t seqStart = 0;
        uint64_t seqEnd = 0;
        uint64_t chunkBase = 0;
        ResolveSeq(seq, b, seqStart, seqEnd, chunkBase);
        InitState(seq, hv);
        uint64_t localChunk = 0;
        for (uint64_t start = seqStart; start < seqEnd; start += BT_) {
            uint64_t end = start + BT_;
            if (end > seqEnd) {
                end = seqEnd;
            }
            ProcessChunkStateAiv(b, seq, hv, chunkBase + localChunk, start, end);
            ++localChunk;
        }
    }

    __aicore__ inline void ProcessStateAiv()
    {
        uint64_t taskNum = static_cast<uint64_t>(N_ * HV_);
        uint64_t blockNum = static_cast<uint64_t>(GetBlockNum());
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += blockNum) {
            uint64_t seq = task / HV_;
            uint64_t hv = task % HV_;
            ProcessStateSeqHeadAiv(seq, hv);
        }
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
                ProcessChunkPostAiv(b, h, hv, chunkIdx, start, end, subBlockIdx);
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
                ProcessChunkOutAiv(b, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessOutAic()
    {
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

    __aicore__ inline void ResolveSeq(uint64_t seq, uint64_t &b, uint64_t &seqStart, uint64_t &seqEnd,
                                      uint64_t &chunkBase)
    {
        b = isVarLen_ ? 0 : seq;
        seqStart = 0;
        seqEnd = T_;
        if (isVarLen_) {
            seqStart = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, seq));
            seqEnd = static_cast<uint64_t>(ReadMetaInt64(cuSeqlens_, seq + 1));
        }
        chunkBase = isVarLen_ ? ChunkCountBefore(seq) : 0;
    }

    __aicore__ inline void ProcessSeqHeadAiv(uint64_t seq, uint64_t hv)
    {
        uint64_t b = 0;
        uint64_t seqStart = 0;
        uint64_t seqEnd = 0;
        uint64_t chunkBase = 0;
        ResolveSeq(seq, b, seqStart, seqEnd, chunkBase);
        uint64_t h = hv / (HV_ / H_);
        InitState(seq, hv);
        uint64_t localChunk = 0;
        for (uint64_t start = seqStart; start < seqEnd; start += BT_) {
            uint64_t end = start + BT_;
            if (end > seqEnd) {
                end = seqEnd;
            }
            ProcessChunkAiv(b, seq, h, hv, chunkBase + localChunk, start, end);
            ++localChunk;
        }
    }

    __aicore__ inline void ProcessSeqHeadAic(uint64_t seq, uint64_t hv)
    {
        uint64_t b = 0;
        uint64_t seqStart = 0;
        uint64_t seqEnd = 0;
        uint64_t chunkBase = 0;
        ResolveSeq(seq, b, seqStart, seqEnd, chunkBase);
        uint64_t localChunk = 0;
        for (uint64_t start = seqStart; start < seqEnd; start += BT_) {
            uint64_t end = start + BT_;
            if (end > seqEnd) {
                end = seqEnd;
            }
            ProcessChunkAic(b, hv, start, end);
            ++localChunk;
        }
    }

private:
    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<T> v_;
    GlobalTensor<float> gk_;
    GlobalTensor<float> beta_;
    GlobalTensor<float> initialState_;
    GlobalTensor<int64_t> cuSeqlens_;
    GlobalTensor<int64_t> chunkIndices_;
    GlobalTensor<OUT_T> o_;
    GlobalTensor<float> finalState_;
    GlobalTensor<T> aqk_;
    GlobalTensor<T> akk_;
    GlobalTensor<T> w_;
    GlobalTensor<OUT_T> u_;
    GlobalTensor<T> qg_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<T> h_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> scalarInBuf_;
    TBuf<TPosition::VECCALC> scalarFp32Buf_;
    TBuf<TPosition::VECCALC> scalarOutBuf_;
    TBuf<TPosition::VECCALC> scalarI64Buf_;
    TBuf<TPosition::VECCALC> exp2Buf_;
    TBuf<TPosition::VECCALC> vecBuf_;
    TBuf<TPosition::A1> solveL1Buf_;
    LocalTensor<T> solveL1_;
    TBuf<TPosition::A2> solveL0aBuf_;
    LocalTensor<T> solveL0a_;
    TBuf<TPosition::B2> solveL0bBuf_;
    LocalTensor<T> solveL0b_;
    TBuf<TPosition::CO1> solveL0cBuf_;
    LocalTensor<float> solveL0c_;
    TQue<TPosition::VECIN, KDA_VEC_BUFFER_NUM> qInQue_;
    TQue<TPosition::VECIN, KDA_VEC_BUFFER_NUM> kInQue_;
    TQue<TPosition::VECIN, KDA_VEC_BUFFER_NUM> gInQue_;
    TQue<TPosition::VECOUT, KDA_VEC_BUFFER_NUM> qgOutQue_;
    TQue<TPosition::VECOUT, KDA_VEC_BUFFER_NUM> wOutQue_;
    TQue<TPosition::VECOUT, KDA_VEC_BUFFER_NUM> kgOutQue_;
    Catlass::Arch::CrossCoreFlagWithReverse<> scoreReadyFlag_{KDA_SCORE_READY_FLAG0, KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<> scoreDoneFlag_{KDA_SCORE_DONE_FLAG0, KDA_SCORE_DONE_FLAG1};

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
    int64_t stage_ = 0;
};
} // namespace

extern "C" __global__ __aicore__ void chunk_kda_fwd(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta,
                                                      GM_ADDR initial_state, GM_ADDR cu_seqlens,
                                                      GM_ADDR chunk_indices, GM_ADDR aqk_in, GM_ADDR akk_in,
                                                      GM_ADDR w_in, GM_ADDR u_in, GM_ADDR qg_in, GM_ADDR kg_in,
                                                      GM_ADDR v_new_in, GM_ADDR h_in, GM_ADDR o,
                                                      GM_ADDR final_state, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w,
                                                      GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    (void)userWS;
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        ChunkKdaFwdKernel<float> op;
        op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                tilingData, &pipe);
        op.ProcessAivOnly();
    } else if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
        if (tilingData.dataType == 2) {
            if ASCEND_IS_AIC {
                ChunkKdaFwdKernel<float> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe, false);
                op.ProcessAic();
            }
            if ASCEND_IS_AIV {
                ChunkKdaFwdKernel<float> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe);
                op.ProcessAiv();
            }
        } else if (tilingData.dataType == 1) {
            if ASCEND_IS_AIC {
                ChunkKdaFwdKernel<bfloat16_t> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe, false);
                op.ProcessAic();
            }
            if ASCEND_IS_AIV {
                ChunkKdaFwdKernel<bfloat16_t> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe);
                op.ProcessAiv();
            }
        } else {
            if ASCEND_IS_AIC {
                ChunkKdaFwdKernel<half> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe, false);
                op.ProcessAic();
            }
            if ASCEND_IS_AIV {
                ChunkKdaFwdKernel<half> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe);
                op.ProcessAiv();
            }
        }
    } else if (TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
        if (tilingData.dataType == 1) {
            if ASCEND_IS_AIC {
                ChunkKdaFwdKernel<bfloat16_t, float> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe, false);
                op.ProcessOutOnlyAic();
            }
            if ASCEND_IS_AIV {
                ChunkKdaFwdKernel<bfloat16_t, float> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe);
                op.ProcessOutOnlyAiv();
            }
        } else {
            if ASCEND_IS_AIC {
                ChunkKdaFwdKernel<half, float> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe, false);
                op.ProcessOutOnlyAic();
            }
            if ASCEND_IS_AIV {
                ChunkKdaFwdKernel<half, float> op;
                op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                        qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                        tilingData, &pipe);
                op.ProcessOutOnlyAiv();
            }
        }
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        if (tilingData.dataType == 2) {
            ChunkKdaFwdKernel<float> op;
            op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                    qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                    tilingData, &pipe);
            op.ProcessAivOnly();
        } else if (tilingData.dataType == 1) {
            ChunkKdaFwdKernel<bfloat16_t> op;
            op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                    qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                    tilingData, &pipe);
            op.ProcessAivOnly();
        } else {
            ChunkKdaFwdKernel<half> op;
            op.Init(q, k, v, gk, beta, initial_state, cu_seqlens, chunk_indices, aqk_in, akk_in, w_in, u_in,
                    qg_in, kg_in, v_new_in, h_in, o, final_state, aqk, akk, w, u, qg, kg, v_new, h,
                    tilingData, &pipe);
            op.ProcessAivOnly();
        }
    }
}
