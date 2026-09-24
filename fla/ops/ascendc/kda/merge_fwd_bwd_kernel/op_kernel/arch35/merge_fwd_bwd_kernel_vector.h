/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 *
 * Fused Mix Vector: pack FP32 H/M into scratch, add He, write user h.
 * M ping-pong overlaps pack of M_{i+1} with Cube GEMM i. Mode-4 per-head
 * CV/VC replaces chip-wide SyncAll: add(h) overlaps GEMM(h+1), and GEMM
 * step i+1(h) starts when add(h) of step i finishes.
 */
#ifndef MERGE_FWD_BWD_KERNEL_ARCH35_VECTOR_H
#define MERGE_FWD_BWD_KERNEL_ARCH35_VECTOR_H

#include "merge_fwd_bwd_kernel_common.h"

#include <type_traits>

using namespace AscendC;

namespace MergeFwBwd {

template <typename T>
class MergeFwdBwdKernelVectorProcess {
public:
    __aicore__ inline MergeFwdBwdKernelVectorProcess(
        GM_ADDR agHm, GM_ADDR scratch, GM_ADDR h, GM_ADDR workspace, const MergeFwdBwdKernelTilingData *tiling)
        : agHm_(agHm), scratch_(scratch), h_(h), tiling_(tiling)
    {
        (void)workspace;
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
        scratchTensor_.SetGlobalBuffer((__gm__ float *)scratch_);
        agHmTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        hTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        scratchTensor_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

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
        if (n_ <= 1) {
            for (uint64_t h = hBegin; h < hEnd; ++h) {
                if ((h & 1U) != subIdx) {
                    continue;
                }
                LoadHeFp32(SrcRank(rank_, n_, 0, forward_, s_), static_cast<int64_t>(h));
                StoreUserH(static_cast<int64_t>(h));
            }
            PipeBarrier<PIPE_ALL>();
            ReleaseEvents();
            return;
        }
        // First C Fixpipe needs a free slot; Cube Wait matches this Set.
        bool ownsHead = false;
        for (uint64_t h = hBegin; h < hEnd; ++h) {
            if ((h & 1U) == subIdx) {
                ownsHead = true;
                break;
            }
        }
        if (ownsHead) {
            VecSetCUbFree(subIdx);
        }
        for (uint64_t h = hBegin; h < hEnd; ++h) {
            if ((h & 1U) != subIdx) {
                continue;
            }
            LoadHeFp32(SrcRank(rank_, n_, 0, forward_, s_), static_cast<int64_t>(h));
            StoreScratchH(static_cast<int64_t>(h), true);
            StoreScratchM(SrcRank(rank_, n_, 1, forward_, s_), static_cast<int64_t>(h), 0U);
            VecSetVc(subIdx);
        }
        for (int64_t i = 1; i < n_; ++i) {
            // Per-head: consume GEMM, add, pack M_{i+1}, then VC so Cube step
            // i+1(h) can start as soon as this head is ready (not after all M).
            for (uint64_t h = hBegin; h < hEnd; ++h) {
                if ((h & 1U) != subIdx) {
                    continue;
                }
                LoadHeToHmm(SrcRank(rank_, n_, i, forward_, s_), static_cast<int64_t>(h));
                VecWaitCv0(subIdx);
                {
                    LocalTensor<float> acc = fp32Buf_.Get<float>();
                    LocalTensor<float> he = hmmBuf_.Get<float>();
                    Add(acc, acc, he, kCTileElems);
                    PipeBarrier<PIPE_V>();
                }
                VecWaitCv1(subIdx);
                {
                    LocalTensor<float> acc = fp32Buf_.Get<float>()[kCTileElems];
                    LocalTensor<float> he = hmmBuf_.Get<float>()[kCTileElems];
                    Add(acc, acc, he, kCTileElems);
                    PipeBarrier<PIPE_V>();
                }
                if (i + 1 >= n_) {
                    StoreUserH(static_cast<int64_t>(h));
                } else {
                    StoreScratchH(static_cast<int64_t>(h), true);
                    StoreScratchM(
                        SrcRank(rank_, n_, i + 1, forward_, s_), static_cast<int64_t>(h),
                        static_cast<uint32_t>(i & 1));
                    VecSetVc(subIdx);
                }
                // fp32Buf drained to GM — Cube may Fixpipe next C into this AIV UB.
                VecSetCUbFree(subIdx);
            }
        }
        PipeBarrier<PIPE_ALL>();
        ReleaseEvents();
    }

private:
    __aicore__ inline void ReleaseEvents()
    {
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3_);
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    __aicore__ inline void CopyStridedToUb(
        LocalTensor<T> dst, int64_t srcRank, int64_t h, uint32_t col, uint32_t cols)
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

    __aicore__ inline void IssueHeToHmm(int64_t srcRank, int64_t h)
    {
        if constexpr (std::is_same<T, float>::value) {
            CopyStridedToUb(hmmBuf_.Get<float>(), srcRank, h, 0, kVDim);
        } else {
            CopyStridedToUb(bf16Buf_.Get<T>(), srcRank, h, 0, kVDim);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void WaitHeToHmm()
    {
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
        if constexpr (!std::is_same<T, float>::value) {
            Cast(hmmBuf_.Get<float>(), bf16Buf_.Get<T>(), RoundMode::CAST_NONE, kKvElems);
        }
    }

    __aicore__ inline void LoadHeToHmm(int64_t srcRank, int64_t h)
    {
        IssueHeToHmm(srcRank, h);
        WaitHeToHmm();
    }

    __aicore__ inline void LoadScratchHTile(int64_t h, int64_t tile)
    {
        LocalTensor<float> dst = fp32Buf_.Get<float>()[static_cast<uint32_t>(tile) * kCTileElems];
        DataCopy(
            dst,
            scratchTensor_[ScratchHElemOff(static_cast<uint64_t>(h)) +
                           static_cast<uint64_t>(tile) * static_cast<uint64_t>(kCTileElems)],
            kCTileElems);
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void LoadScratchH(int64_t h)
    {
        LocalTensor<float> dst = fp32Buf_.Get<float>();
        DataCopy(dst, scratchTensor_[ScratchHElemOff(static_cast<uint64_t>(h))], kKvElems);
        SetFlag<HardEvent::MTE2_V>(mte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
    }

    __aicore__ inline void StoreScratchH(int64_t h, bool drainMte3)
    {
        LocalTensor<float> src = fp32Buf_.Get<float>();
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(scratchTensor_[ScratchHElemOff(static_cast<uint64_t>(h))], src, kKvElems);
        if (drainMte3) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        }
    }

    __aicore__ inline void StoreScratchHTile(int64_t h, int64_t tile, bool drainMte3)
    {
        LocalTensor<float> src = fp32Buf_.Get<float>()[static_cast<uint32_t>(tile) * kCTileElems];
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(
            scratchTensor_[ScratchHElemOff(static_cast<uint64_t>(h)) +
                           static_cast<uint64_t>(tile) * static_cast<uint64_t>(kCTileElems)],
            src, kCTileElems);
        if (drainMte3) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        }
    }

    __aicore__ inline void StoreScratchM(int64_t srcRank, int64_t h, uint32_t mBank)
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
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopy(
            scratchTensor_[ScratchMElemOff(static_cast<uint64_t>(hv_), static_cast<uint64_t>(h), mBank)],
            dst, kKkElems);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
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
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
    }

    TPipe *pipe_ = nullptr;
    GM_ADDR agHm_ = nullptr;
    GM_ADDR scratch_ = nullptr;
    GM_ADDR h_ = nullptr;
    const MergeFwdBwdKernelTilingData *tiling_ = nullptr;
    GlobalTensor<T> agHmTensor_;
    GlobalTensor<T> hTensor_;
    GlobalTensor<float> scratchTensor_;
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
