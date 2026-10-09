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
        stateVFirst_ = tiling_->stateVFirst != 0;
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
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        vToMte2_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
#endif
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
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
            for (uint64_t h = hBegin; h < hEnd; ++h) {
                if ((h & 1U) != subIdx) {
                    continue;
                }
                LoadHeToHmm(SrcRank(rank_, n_, i, forward_, s_), static_cast<int64_t>(h));
                PrepareGemmTile(subIdx, static_cast<int64_t>(h), 0);
                {
                    LocalTensor<float> acc = fp32Buf_.Get<float>();
                    LocalTensor<float> he = hmmBuf_.Get<float>();
                    Add(acc, acc, he, kCTileElems);
                    PipeBarrier<PIPE_V>();
                }
                PrepareGemmTile(subIdx, static_cast<int64_t>(h), 1);
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
                VecSetCUbFree(subIdx);
            }
        }
#else
        // 910b/910_93 mode 0x2 is one AIC plus both AIVs on the same flag.
        // The non-owning subblock still has to set and wait; only the owner moves data.
        for (uint64_t h = hBegin; h < hEnd; ++h) {
            if ((h & 1U) == subIdx) {
                LoadHeFp32(SrcRank(rank_, n_, 0, forward_, s_), static_cast<int64_t>(h));
                StoreScratchH(static_cast<int64_t>(h), true);
                StoreScratchM(SrcRank(rank_, n_, 1, forward_, s_), static_cast<int64_t>(h), 0U);
            }
            VecSetVc(subIdx);
        }
        for (int64_t i = 1; i < n_; ++i) {
            for (uint64_t h = hBegin; h < hEnd; ++h) {
                const bool owner = (h & 1U) == subIdx;
                if (owner) {
                    LoadHeToHmm(SrcRank(rank_, n_, i, forward_, s_), static_cast<int64_t>(h));
                }
                WaitGemmTile(subIdx, 0);
                if (owner) {
                    LoadGemmTile(static_cast<int64_t>(h), 0);
                    LocalTensor<float> acc = fp32Buf_.Get<float>();
                    LocalTensor<float> he = hmmBuf_.Get<float>();
                    Add(acc, acc, he, kCTileElems);
                    PipeBarrier<PIPE_V>();
                }
                WaitGemmTile(subIdx, 1);
                if (owner) {
                    LoadGemmTile(static_cast<int64_t>(h), 1);
                    LocalTensor<float> acc = fp32Buf_.Get<float>()[kCTileElems];
                    LocalTensor<float> he = hmmBuf_.Get<float>()[kCTileElems];
                    Add(acc, acc, he, kCTileElems);
                    PipeBarrier<PIPE_V>();
                    if (i + 1 >= n_) {
                        StoreUserH(static_cast<int64_t>(h));
                    } else {
                        StoreScratchH(static_cast<int64_t>(h), true);
                        StoreScratchM(
                            SrcRank(rank_, n_, i + 1, forward_, s_), static_cast<int64_t>(h),
                            static_cast<uint32_t>(i & 1));
                    }
                }
                if (i + 1 < n_) {
                    VecSetVc(subIdx);
                }
            }
        }
#endif
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
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(vToMte2_);
#endif
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

    // A5: Cube already Fixpiped this tile into fp32Buf.
    // 910b/910_93 Fixpipes C to GM scratch. Use the same PIPE_V CV wait as A5,
    // then reload the tile. V_MTE2 orders that copy after the CV wait.
    __aicore__ inline void WaitGemmTile(uint32_t subIdx, int64_t tile)
    {
        if (tile == 0) {
            VecWaitCv0(subIdx);
        } else {
            VecWaitCv1(subIdx);
        }
    }

    __aicore__ inline void LoadGemmTile(int64_t h, int64_t tile)
    {
        SetFlag<HardEvent::V_MTE2>(vToMte2_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2_);
        LoadScratchHTile(h, tile);
    }

    __aicore__ inline void PrepareGemmTile(uint32_t subIdx, int64_t h, int64_t tile)
    {
        WaitGemmTile(subIdx, tile);
        (void)h;
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

    // [K, V] row-major in src -> [V, K] row-major in dst. Same addressing as
    // recurrent_kda's K-first to V-first state transpose, specialized to 128x128.
    __aicore__ inline void TransposeKvToVk(LocalTensor<float> dst, LocalTensor<float> src)
    {
        constexpr uint32_t kBlock = 16;
        constexpr uint32_t kDataBlockBytes = 32;
        constexpr uint32_t kElemsPerBlock = kDataBlockBytes / sizeof(float);
        uint64_t dstList[kBlock];
        uint64_t srcList[kBlock];
        const uint64_t dstAddr = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
        const uint64_t srcAddr = reinterpret_cast<uint64_t>(src.GetPhyAddr());
        constexpr uint32_t kRowCount = kKDim;
        constexpr uint32_t kColCount = kVDim;
        constexpr uint32_t kSrcStride = kVDim;
        constexpr uint32_t kDstStride = kKDim;
        const uint16_t repeatTimes = static_cast<uint16_t>(kColCount / kElemsPerBlock);
        TransDataTo5HDParams transposeParams{
            false, false, static_cast<uint8_t>(repeatTimes),
            static_cast<uint16_t>(repeatTimes > 1 ? kDstStride : 0),
            static_cast<uint16_t>(repeatTimes > 1 ? 1 : 0)};
        for (uint32_t rowBlock = 0; rowBlock < kRowCount; rowBlock += kBlock) {
            for (uint32_t i = 0; i < kBlock; ++i) {
                srcList[i] = srcAddr + (rowBlock + i) * kSrcStride * sizeof(float);
                dstList[i] = dstAddr +
                    (rowBlock + (i / 2U) * kDstStride + (i % 2U) * kElemsPerBlock) * sizeof(float);
            }
            TransDataTo5HD<float>(dstList, srcList, transposeParams);
        }
    }

    __aicore__ inline void StoreUserTensor(int64_t h, LocalTensor<float> hFp32)
    {
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

    __aicore__ inline void StoreUserH(int64_t h)
    {
        LocalTensor<float> hFp32 = fp32Buf_.Get<float>();
        // Scratch and the GEMM stay [K, V]. state_v_first writes the user buffer as [V, K].
        if (stateVFirst_) {
            LocalTensor<float> transposed = hmmBuf_.Get<float>();
            PipeBarrier<PIPE_V>();
            TransposeKvToVk(transposed, hFp32);
            PipeBarrier<PIPE_V>();
            StoreUserTensor(h, transposed);
            return;
        }
        StoreUserTensor(h, hFp32);
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
    int32_t vToMte2_ = 0;
    int64_t hv_ = 1;
    int64_t s_ = 1;
    int64_t rank_ = 0;
    int64_t n_ = 1;
    bool forward_ = true;
    bool stateVFirst_ = false;
    uint32_t usedCoreNum_ = 1;
};

} // namespace MergeFwBwd

#endif
