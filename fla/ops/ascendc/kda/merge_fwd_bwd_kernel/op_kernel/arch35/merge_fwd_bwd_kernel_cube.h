/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 *
 * Ascend950 Mix Cube (dav-3510 / arch35).
 */
#ifndef MERGE_FWD_BWD_KERNEL_ARCH35_CUBE_H
#define MERGE_FWD_BWD_KERNEL_ARCH35_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <type_traits>

#include "merge_fwd_bwd_kernel_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;

namespace MergeFwBwd {

template <typename DstT>
class MergeFwdBwdKernelCubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using CubeT = float;
    using AccT = float;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, CubeT, LayoutTagA, CubeT, LayoutTagB, AccT, LayoutTagC>;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, CubeT, LayoutTagL1A>;

    // Two L1 ping-pong regions (t&1) need independent MTE1<->MTE2 events,
    // otherwise C0_t0 must release L1 before Load_t1 and C1 cannot reuse.
    static constexpr int32_t kEventL1Ping = 0;
    static constexpr int32_t kEventL1Pong = 1;
    static constexpr int32_t kEventL0A = 0;
    static constexpr int32_t kEventL0B = 1;
    static constexpr int32_t kEventL0C0 = 0;
    static constexpr int32_t kEventL0C1 = 1;
    static constexpr int32_t kEventMte1M = 0;

    __aicore__ inline int32_t L1Event(uint32_t t) const
    {
        return (t & 1U) != 0U ? kEventL1Pong : kEventL1Ping;
    }
    static constexpr uint32_t kL0CTileBytes = 128 * 1024;
    // H NZ can exceed 32KiB; keep A0/A1 past a 128KiB H region so C0's A
    // tile is not clobbered (C1 was correct while C0 was not).
    static constexpr uint32_t kL1HOffset = 0;
    static constexpr uint32_t kL1A0Offset = 128 * 1024;
    static constexpr uint32_t kL1A1Offset = 192 * 1024;
    static constexpr uint32_t kL1SlotStride = 256 * 1024;

    __aicore__ inline MergeFwdBwdKernelCubeProcess(GM_ADDR agHm, GM_ADDR h, GM_ADDR workspace)
        : agHm_(agHm), h_(h), workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const MergeFwdBwdKernelTilingData *tiling)
    {
        hv_ = static_cast<uint64_t>(tiling->Hv);
        s_ = tiling->S;
        rank_ = tiling->rank;
        n_ = tiling->N;
        forward_ = tiling->forward != 0;
        usedCoreNum_ = tiling->usedAic > 0 ? static_cast<uint32_t>(tiling->usedAic) : 1U;
        (void)agHm_;
        (void)h_;
        (void)s_;
        (void)rank_;
        (void)forward_;
    }

    __aicore__ inline void Process()
    {
        if (n_ <= 1) {
            return;
        }
        Arch::Resource<ArchTag> resource;
        AscendC::LocalTensor<CubeT> l0A = resource.l0ABuf.template GetBufferByByte<CubeT>(0);
        AscendC::LocalTensor<CubeT> l0B = resource.l0BBuf.template GetBufferByByte<CubeT>(0);
        AscendC::LocalTensor<AccT> l0C0 = resource.l0CBuf.template GetBufferByByte<AccT>(0);
        AscendC::LocalTensor<AccT> l0C1 = resource.l0CBuf.template GetBufferByByte<AccT>(kL0CTileBytes);

        auto layoutL1A = tla::MakeLayout<CubeT, LayoutTagL1A>(kTileM, kKDim);
        auto layoutL1B = tla::MakeLayout<CubeT, LayoutTagL1B>(kKDim, kVDim);
        LayoutTagA tagMTile = LayoutTagA::template MakeLayout<CubeT>(kTileM, kKDim);
        LayoutTagB tagH = LayoutTagB::template MakeLayout<CubeT>(kKDim, kVDim);
        auto layoutMTile = MakeLayoutFromTag(tagMTile);
        auto layoutH = MakeLayoutFromTag(tagH);
        auto layoutL0C = tla::MakeLayoutL0C(kTileM, kVDim);
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1Ping);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1Pong);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        coreIdx_ = coreIdx;
        uint64_t hBegin = 0;
        uint64_t hEnd = 0;
        CoreTaskRange(coreIdx, usedCoreNum_, hv_, hBegin, hEnd);

        const int64_t gemmSteps = n_ - 1;
        auto tensorL0C0 = tla::MakeTensor(l0C0, layoutL0C, Arch::PositionL0C{});
        auto tensorL0C1 = tla::MakeTensor(l0C1, layoutL0C, Arch::PositionL0C{});
        // Mode 2 AIV-pair rank-step. Load H/M once per head; C1 reuses L1.
        // Dual L1 events so t0 and t1 stay resident across SetFree.
        for (uint64_t roundBegin = hBegin; roundBegin < hEnd; roundBegin += kMaxTaskPerAic) {
            const uint32_t nRound = RoundTaskCount(roundBegin, hEnd);
            for (uint32_t t = 0; t < nRound; ++t) {
                const uint64_t h = roundBegin + t;
                const int32_t l1Ev = L1Event(t);
                const uint32_t l1Base = (t & 1U) * kL1SlotStride;
                AscendC::LocalTensor<CubeT> hL1 =
                    resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1HOffset);
                AscendC::LocalTensor<CubeT> a0L1 =
                    resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1A0Offset);
                AscendC::LocalTensor<CubeT> a1L1 =
                    resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1A1Offset);
                auto tensorL1B = tla::MakeTensor(hL1, layoutL1B, Arch::PositionL1{});
                auto tensorL1A0 = tla::MakeTensor(a0L1, layoutL1A, Arch::PositionL1{});
                auto tensorL1A1 = tla::MakeTensor(a1L1, layoutL1A, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Ev);
                GemmKSplitFixpipe(
                    copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A0, tensorL1B,
                    tensorL0C0, h, kCWarmupTileL0C0, false, kEventL0C0, l1Ev);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
                GemmKSplitFixpipe(
                    copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A1, tensorL1B,
                    tensorL0C1, h, kCWarmupTile, false, kEventL0C1, l1Ev);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
                DelayPreloadA(copyL1ToL0A, l0A, tensorL1A1);
                DelayPreloadA(copyL1ToL0A, l0A, tensorL1A1);
                DelayPreloadA(copyL1ToL0A, l0A, tensorL1A1);
                DelayPreloadA(copyL1ToL0A, l0A, tensorL1A1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Ev);
            }
            for (int64_t step = 0; step < gemmSteps; ++step) {
                const uint32_t numPhases = (nRound + 1U) / 2U;
                for (uint32_t p = 0; p < numPhases; ++p) {
                    const uint32_t tBegin = p * 2U;
                    const uint32_t nPh = (tBegin + 2U <= nRound) ? 2U : (nRound - tBegin);
                    AicWaitChunkReady<PIPE_MTE2>(0);
                    for (uint32_t dt = 0; dt < nPh; ++dt) {
                        const uint32_t t = tBegin + dt;
                        const uint64_t h = roundBegin + t;
                        const int32_t l1Ev = L1Event(t);
                        const uint32_t l1Base = (t & 1U) * kL1SlotStride;
                        AscendC::LocalTensor<CubeT> hL1 =
                            resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1HOffset);
                        AscendC::LocalTensor<CubeT> a0L1 =
                            resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1A0Offset);
                        AscendC::LocalTensor<CubeT> a1L1 =
                            resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1A1Offset);
                        auto tensorL1B = tla::MakeTensor(hL1, layoutL1B, Arch::PositionL1{});
                        auto tensorL1A0 = tla::MakeTensor(a0L1, layoutL1A, Arch::PositionL1{});
                        auto tensorL1A1 = tla::MakeTensor(a1L1, layoutL1A, Arch::PositionL1{});
                        LoadStageToL1(
                            tensorL1B, tensorL1A0, tensorL1A1, layoutH, layoutMTile, t, l1Ev);
                        MmadKSplit(
                            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A0,
                            tensorL1B, tensorL0C0, false, kEventL0C0, l1Ev);
                        DelayPreloadA(copyL1ToL0A, l0A, tensorL1A1);
                        DrainCopyC(tensorL0C0, h, 0, kEventL0C0);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
                    }
                    AicSetChunkFree<PIPE_FIX>(0);
                    for (uint32_t dt = 0; dt < nPh; ++dt) {
                        const uint32_t t = tBegin + dt;
                        const uint64_t h = roundBegin + t;
                        const int32_t l1Ev = L1Event(t);
                        const uint32_t l1Base = (t & 1U) * kL1SlotStride;
                        AscendC::LocalTensor<CubeT> hL1 =
                            resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1HOffset);
                        AscendC::LocalTensor<CubeT> a1L1 =
                            resource.l1Buf.template GetBufferByByte<CubeT>(l1Base + kL1A1Offset);
                        auto tensorL1B = tla::MakeTensor(hL1, layoutL1B, Arch::PositionL1{});
                        auto tensorL1A1 = tla::MakeTensor(a1L1, layoutL1A, Arch::PositionL1{});
                        GemmKSplitFixpipe(
                            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A1,
                            tensorL1B, tensorL0C1, h, 1, true, kEventL0C1, l1Ev);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
                    }
                    AicSetChunkC1Done<PIPE_FIX>(0);
                }
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1Ping);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1Pong);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
    }

private:
    __aicore__ inline GM_ADDR SlotPtr(uint32_t slot) const
    {
        return workspace_ + SlotBase(coreIdx_, slot);
    }

    template <typename TensorL1B, typename LayoutH>
    __aicore__ inline void LoadHToL1(TensorL1B const &tensorL1B, LayoutH const &layoutH, uint32_t slot)
    {
        AscendC::GlobalTensor<CubeT> gmH;
        gmH.SetGlobalBuffer((__gm__ CubeT *)SlotPtr(slot));
        gmH.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto tensorHGm = tla::MakeTensor(gmH, layoutH, Arch::PositionGM{});
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(tensorHGm)>;
        CopyGmToL1B copyGmToL1B;
        copyGmToL1B(tensorL1B, tensorHGm);
    }

    template <typename TensorL1B, typename TensorL1A, typename LayoutH, typename LayoutM>
    __aicore__ inline void LoadStageToL1(
        TensorL1B const &tensorL1B, TensorL1A const &tensorL1A0, TensorL1A const &tensorL1A1,
        LayoutH const &layoutH, LayoutM const &layoutM, uint32_t slot, int32_t l1Event)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        LoadHToL1(tensorL1B, layoutH, slot);
        LoadATileToL1(tensorL1A0, layoutM, slot, 0);
        LoadATileToL1(tensorL1A1, layoutM, slot, kTileM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
    }

    template <typename TensorL1A>
    __aicore__ inline void DelayPreloadA(
        CopyL1ToL0A &copyL1ToL0A, AscendC::LocalTensor<CubeT> &l0A, TensorL1A const &tensorL1A)
    {
        auto layoutL0A = tla::MakeLayout<CubeT, LayoutTagL0A>(kTileM, kTileM);
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Arch::PositionL0A{});
        auto tileA = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(kTileM, kTileM));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
        copyL1ToL0A(tensorL0A, tileA);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
    }

    template <typename TensorL1A, typename TensorL1B, typename TensorL0C>
    __aicore__ inline void MmadKSplit(
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
        AscendC::LocalTensor<CubeT> &l0A, AscendC::LocalTensor<CubeT> &l0B,
        TensorL1A const &tensorL1A, TensorL1B const &tensorL1B, TensorL0C const &tensorL0C,
        bool releaseL1, int32_t l0cEvent, int32_t l1Event)
    {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
        for (uint32_t k0 = 0; k0 < kKDim; k0 += kTileM) {
            const bool lastK = (k0 + kTileM >= kKDim);
            auto layoutL0A = tla::MakeLayout<CubeT, LayoutTagL0A>(kTileM, kTileM);
            auto layoutL0B = tla::MakeLayout<CubeT, LayoutTagL0B>(kTileM, kVDim);
            auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Arch::PositionL0A{});
            auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Arch::PositionL0B{});
            auto tileA = GetTile(tensorL1A, tla::MakeCoord(0, k0), tla::MakeShape(kTileM, kTileM));
            auto tileB = GetTile(tensorL1B, tla::MakeCoord(k0, 0), tla::MakeShape(kTileM, kVDim));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
            copyL1ToL0A(tensorL0A, tileA);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
            copyL1ToL0B(tensorL0B, tileB);
            if (lastK && releaseL1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEventMte1M);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEventMte1M);
            const uint8_t unitFlag = lastK ? static_cast<uint8_t>(0b11) : static_cast<uint8_t>(0b10);
            tileMmad(tensorL0C, tensorL0A, tensorL0B, kTileM, kVDim, kTileM, k0 == 0, unitFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cEvent);
    }

    template <typename TensorL0C>
    __aicore__ inline void DrainCopyC(
        TensorL0C const &tensorL0C, uint64_t h, int64_t tile, int32_t l0cEvent)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cEvent);
        CopyCToWorkspace(h, tensorL0C, tile);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
        AscendC::PipeBarrier<PIPE_FIX>();
    }

    template <typename TensorL1A, typename TensorL1B, typename TensorL0C>
    __aicore__ inline void GemmKSplitFixpipe(
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
        AscendC::LocalTensor<CubeT> &l0A, AscendC::LocalTensor<CubeT> &l0B,
        TensorL1A const &tensorL1A, TensorL1B const &tensorL1B, TensorL0C const &tensorL0C,
        uint64_t h, int64_t slotIndex, bool releaseL1, int32_t l0cEvent, int32_t l1Event)
    {
        MmadKSplit(
            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A, tensorL1B, tensorL0C,
            releaseL1, l0cEvent, l1Event);
        DrainCopyC(tensorL0C, h, slotIndex, l0cEvent);
    }

    template <typename TensorL1A, typename LayoutM>
    __aicore__ inline void LoadATileToL1(
        TensorL1A const &tensorL1A, LayoutM const &layoutM, uint32_t slot, uint32_t m0)
    {
        AscendC::GlobalTensor<CubeT> gmM;
        gmM.SetGlobalBuffer(
            (__gm__ CubeT *)(SlotPtr(slot) + kHFp32Bytes) + static_cast<uint64_t>(m0) * kKDim);
        gmM.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        auto tensorMGm = tla::MakeTensor(gmM, layoutM, Arch::PositionGM{});
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(tensorMGm)>;
        CopyGmToL1A copyGmToL1A;
        copyGmToL1A(tensorL1A, tensorMGm);
    }

    template <typename TensorL0C>
    __aicore__ inline void CopyCToWorkspace(
        uint64_t h, TensorL0C const &tensorL0C, int64_t tile)
    {
        const uint64_t elemOff = CScratchElemOff(h, tile, usedCoreNum_);
        AscendC::GlobalTensor<float> gmC;
        gmC.SetGlobalBuffer((__gm__ float *)workspace_ + elemOff);
        gmC.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        CopyL0CTile(gmC, tensorL0C, kVDim);
    }

    template <typename OutT, typename TensorL0C>
    __aicore__ inline void CopyL0CTile(
        AscendC::GlobalTensor<OutT> const &gmC, TensorL0C const &tensorL0C, uint32_t dstStride)
    {
        AscendC::DataCopyCO12DstParams params;
        params.nSize = kVDim;
        params.mSize = kTileM;
        params.dstStride = dstStride;
        params.srcStride = tla::get<1, 1>(tensorL0C.stride()) / tla::get<0, 0>(tensorL0C.stride());
        params.quantPre = Gemm::Tile::CopyL0CToDstQuantMode<
            ArchTag, AccT, OutT, Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
        params.nz2ndEn = true;
        params.reluPre = false;
        params.unitFlag = 0b11;
        auto srcOffset = tensorL0C.layout()(tensorL0C.coord());
        AscendC::DataCopy(gmC[0], tensorL0C.data()[srcOffset], params);
    }

    GM_ADDR agHm_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    uint64_t hv_ = 1;
    int64_t s_ = 1;
    int64_t rank_ = 0;
    int64_t n_ = 1;
    bool forward_ = true;
    uint32_t usedCoreNum_ = 1;
    uint32_t coreIdx_ = 0;
};

} // namespace MergeFwBwd

#endif
