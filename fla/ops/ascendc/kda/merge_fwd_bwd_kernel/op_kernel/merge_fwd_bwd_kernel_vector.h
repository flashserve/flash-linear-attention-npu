/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 *
 * 910b / 910_93 Mix Vector (dav-2201 / AtlasA2).
 *
 * Schedule: serial per-head (AIV0 even / AIV1 odd). Mode-2 FFTS aggregates both
 * AIVs, so the idle subblock DummyHandshake-matches Ready/Free/C1. Cube A/B are
 * FP32 via workspace H|M slots (StoreHFp32 / StoreMFp32). Ascend950 pair-stage
 * overlap lives in arch35/.
 */
#ifndef MERGE_FWD_BWD_KERNEL_VECTOR_H
#define MERGE_FWD_BWD_KERNEL_VECTOR_H

#include "merge_fwd_bwd_kernel_common.h"

#include <type_traits>

using namespace AscendC;

namespace MergeFwBwd {

template <typename T>
class MergeFwdBwdKernelVectorProcess {
public:
    __aicore__ inline MergeFwdBwdKernelVectorProcess(
        GM_ADDR agHm, GM_ADDR h, GM_ADDR workspace, const MergeFwdBwdKernelTilingData *tiling)
        : agHm_(agHm), h_(h), workspace_(workspace), tiling_(tiling)
    {
    }

    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe_ = pipe;
        hv_ = tiling_->Hv;
        s_ = tiling_->S;
        rank_ = tiling_->rank;
        n_ = tiling_->N;
        forward_ = tiling_->forward != 0;
        usedCoreNum_ = tiling_->usedAic > 0 ? static_cast<uint32_t>(tiling_->usedAic) : 1U;

        agHmTensor_.SetGlobalBuffer((__gm__ T *)agHm_);
        hTensor_.SetGlobalBuffer((__gm__ T *)h_);
        agHmTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        hTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        pipe_->InitBuffer(fp32Buf_, kKvElems * sizeof(float));
        pipe_->InitBuffer(hmmBuf_, kKvElems * sizeof(float));
        pipe_->InitBuffer(bf16Buf_, kKvElems * sizeof(bfloat16_t));
        mte2ToV_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte2ToMte3_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        const uint32_t subIdx = GetSubBlockIdx();
        uint64_t hBegin = 0;
        uint64_t hEnd = 0;
        CoreTaskRange(coreIdx, usedCoreNum_, static_cast<uint64_t>(hv_), hBegin, hEnd);
        for (uint64_t h = hBegin; h < hEnd; ++h) {
            if ((h & 1U) != subIdx) {
                DummyHandshake();
                continue;
            }
            ProcessHead(coreIdx, subIdx, static_cast<int64_t>(h));
        }
        PipeBarrier<PIPE_ALL>();
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3_);
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

private:
    static constexpr uint32_t kTileElems = kTileM * kVDim;

    __aicore__ inline void DummyHandshake()
    {
        if (n_ <= 1) {
            return;
        }
        AivSetChunkReady<PIPE_MTE3>(0);
        for (int64_t i = 1; i < n_; ++i) {
            AivWaitChunkFree<PIPE_MTE2>(0);
            AivWaitChunkC1<PIPE_MTE2>(0);
            if (i + 1 < n_) {
                AivSetChunkReady<PIPE_MTE3>(0);
            }
        }
    }

    __aicore__ inline void CopySingleHead(int64_t h)
    {
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        const int64_t src0 = SrcRank(rank_, n_, 0, forward_, s_);
        if constexpr (std::is_same<T, float>::value) {
            CopyStridedToUb(hFp32, src0, h, 0, kVDim);
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_);
            DataCopy(hTensor_[static_cast<uint64_t>(h) * kKvElems], hFp32, kKvElems);
        } else {
            LoadHeFp32(src0, h);
            StoreUserH(h);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline void ProcessRankStep(uint32_t coreIdx, uint32_t slot, int64_t h, int64_t i)
    {
        LocalTensor<float> hTop = fp32Buf_.Get<float>(kTileElems);
        LocalTensor<float> hBot = fp32Buf_.Get<float>()[kTileElems];
        LocalTensor<float> cTile = hmmBuf_.Get<float>(kTileElems);
        LocalTensor<float> heTop = hmmBuf_.Get<float>()[kTileElems];

        AivWaitChunkFree<PIPE_MTE2>(static_cast<uint16_t>(slot));
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        LoadHeFp32(SrcRank(rank_, n_, i, forward_, s_), h);
        Adds(heTop, hTop, 0.0f, kTileElems);
        PipeBarrier<PIPE_V>();
        LoadCFromWorkspace(cTile, h, 0);
        Add(hTop, cTile, heTop, kTileElems);
        PipeBarrier<PIPE_V>();

        AivWaitChunkC1<PIPE_MTE2>(static_cast<uint16_t>(slot));
        LoadCFromWorkspace(cTile, h, 1);
        Add(hBot, cTile, hBot, kTileElems);
        PipeBarrier<PIPE_V>();

        if (i + 1 < n_) {
            StoreStageInputs(coreIdx, slot, h, SrcRank(rank_, n_, i + 1, forward_, s_));
            AivSetChunkReady<PIPE_MTE3>(static_cast<uint16_t>(slot));
        } else {
            StoreUserH(h);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        }
    }

    __aicore__ inline void ProcessHead(uint32_t coreIdx, uint32_t slot, int64_t h)
    {
        const int64_t src0 = SrcRank(rank_, n_, 0, forward_, s_);
        if (n_ <= 1) {
            CopySingleHead(h);
            return;
        }
        LoadHeFp32(src0, h);
        StoreStageInputs(coreIdx, slot, h, SrcRank(rank_, n_, 1, forward_, s_));
        AivSetChunkReady<PIPE_MTE3>(static_cast<uint16_t>(slot));
        for (int64_t i = 1; i < n_; ++i) {
            ProcessRankStep(coreIdx, slot, h, i);
        }
    }

    __aicore__ inline void CopyStridedToUb(LocalTensor<T> dst, int64_t srcRank, int64_t h, uint32_t col, uint32_t cols)
    {
        const uint64_t base = AgHmHeadOffset(srcRank, hv_, h) + col;
        DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(kKDim);
        params.blockLen = static_cast<uint32_t>(cols * sizeof(T));
        params.srcStride = static_cast<uint32_t>((kRowStride - cols) * sizeof(T));
        params.dstStride = 0;
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, agHmTensor_[base], params, pad);
    }

    __aicore__ inline void LoadHeFp32(int64_t srcRank, int64_t h)
    {
        LocalTensor<float> dst = fp32Buf_.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            CopyStridedToUb(dst, srcRank, h, 0, kVDim);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
        } else {
            LocalTensor<T> src = bf16Buf_.Get<T>();
            CopyStridedToUb(src, srcRank, h, 0, kVDim);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
            Cast(dst, src, RoundMode::CAST_NONE, kKvElems);
        }
    }

    __aicore__ inline void StoreUserH(int64_t h)
    {
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3_);
            DataCopy(hTensor_[static_cast<uint64_t>(h) * kKvElems], hFp32, kKvElems);
        } else {
            LocalTensor<T> dst = bf16Buf_.Get<T>();
            Cast(dst, hFp32, RoundMode::CAST_RINT, kKvElems);
            SetFlag<HardEvent::V_MTE3>(vToMte3_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3_);
            DataCopy(hTensor_[static_cast<uint64_t>(h) * kKvElems], dst, kKvElems);
        }
    }

    __aicore__ inline void LoadCFromWorkspace(LocalTensor<float> dst, int64_t h, int64_t tile)
    {
        const uint64_t elemOff = CScratchElemOff(static_cast<uint64_t>(h), tile, usedCoreNum_);
        GlobalTensor<float> gmC;
        gmC.SetGlobalBuffer((__gm__ float *)workspace_ + elemOff);
        gmC.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopy(dst, gmC, kTileElems);
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void StoreMFp32(int64_t srcRank, int64_t h, uint32_t coreIdx, uint32_t slot)
    {
        LocalTensor<float> dst = hmmBuf_.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            CopyStridedToUb(dst, srcRank, h, kVDim, kKDim);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
        } else {
            LocalTensor<T> src = bf16Buf_.Get<T>();
            CopyStridedToUb(src, srcRank, h, kVDim, kKDim);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
            Cast(dst, src, RoundMode::CAST_NONE, kKkElems);
        }
        GlobalTensor<float> gmM;
        gmM.SetGlobalBuffer((__gm__ float *)(workspace_ + SlotBase(coreIdx, slot) + kHFp32Bytes));
        gmM.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(gmM, dst, kKkElems);
    }

    __aicore__ inline void StoreHFp32(uint32_t coreIdx, uint32_t slot)
    {
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        GlobalTensor<float> gmH;
        gmH.SetGlobalBuffer((__gm__ float *)(workspace_ + SlotBase(coreIdx, slot)));
        gmH.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(gmH, hFp32, kKvElems);
    }

    __aicore__ inline void StoreStageInputs(uint32_t coreIdx, uint32_t slot, int64_t h, int64_t mSrc)
    {
        StoreHFp32(coreIdx, slot);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        StoreMFp32(mSrc, h, coreIdx, slot);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    TPipe *pipe_ = nullptr;
    GM_ADDR agHm_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    const MergeFwdBwdKernelTilingData *tiling_ = nullptr;
    GlobalTensor<T> agHmTensor_;
    GlobalTensor<T> hTensor_;
    TBuf<TPosition::VECCALC> fp32Buf_;
    TBuf<TPosition::VECCALC> hmmBuf_;
    TBuf<TPosition::VECCALC> bf16Buf_;
    int32_t mte2ToV_ = 0;
    int32_t vToMte3_ = 0;
    int32_t mte2ToMte3_ = 0;
    int32_t mte3ToMte2_ = 0;
    int64_t hv_ = 1;
    int64_t s_ = 1;
    int64_t rank_ = 0;
    int64_t n_ = 1;
    bool forward_ = true;
    uint32_t usedCoreNum_ = 1;
};

} // namespace MergeFwBwd

#endif
