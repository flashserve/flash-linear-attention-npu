/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 *
 * Ascend950 Mix Cube (dav-3510 / arch35), host-split.
 *
 * One rank-step GEMM per launch: C = M @ H on the shared FP32 scratch tensor.
 * No CrossCore. Vector packed M/H in the previous launch on this stream.
 */
#ifndef MERGE_FWD_BWD_KERNEL_ARCH35_CUBE_H
#define MERGE_FWD_BWD_KERNEL_ARCH35_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

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

    static constexpr int32_t kEventL1_0 = 0;
    static constexpr int32_t kEventL1_1 = 1;
    // M1 tile ready (same L1 stage); overlaps C0 MMAD.
    static constexpr int32_t kEventL1M1_0 = 2;
    static constexpr int32_t kEventL1M1_1 = 3;
    static constexpr int32_t kEventL0A0 = 0;
    static constexpr int32_t kEventL0A1 = 1;
    static constexpr int32_t kEventL0B0 = 2;
    static constexpr int32_t kEventL0B1 = 3;
    static constexpr int32_t kEventL0C0 = 0;
    static constexpr int32_t kEventL0C1 = 1;
    static constexpr int32_t kEventMte1M0 = 0;
    static constexpr int32_t kEventMte1M1 = 1;
    static constexpr uint32_t kL0CTileBytes = 128 * 1024;
    static constexpr uint32_t kL0ATileElems = kTileM * kTileM;
    static constexpr uint32_t kL0BTileElems = kTileM * kVDim;
    static constexpr uint32_t kL1StageBytes = 256 * 1024;
    static constexpr uint32_t kL1HOffset = 0;
    static constexpr uint32_t kL1A0Offset = 128 * 1024;
    static constexpr uint32_t kL1A1Offset = 192 * 1024;
    // L0C NZ → owning AIV UB ND. Must match Vector TPipe fp32Buf_ at UB 0.
    static constexpr AscendC::FixpipeConfig kCfgRowMajorUb{
        AscendC::CO2Layout::ROW_MAJOR, true};

    __aicore__ inline MergeFwdBwdKernelCubeProcess(GM_ADDR agHm, GM_ADDR scratch, GM_ADDR h, GM_ADDR workspace)
        : agHm_(agHm), scratch_(scratch), h_(h), workspace_(workspace)
    {
        (void)agHm_;
        (void)h_;
    }

    __aicore__ inline void Init(const MergeFwdBwdKernelTilingData *tiling)
    {
        hv_ = static_cast<uint64_t>(tiling->Hv);
        n_ = tiling->N;
        usedCoreNum_ = tiling->usedAic > 0 ? static_cast<uint32_t>(tiling->usedAic) : 1U;
        (void)workspace_;
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
        ubC_ = resource.ubBuf.template GetBufferByByte<float>(0);

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

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0A1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEventL0B1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        uint64_t hBegin = 0;
        uint64_t hEnd = 0;
        CoreTaskRange(coreIdx, usedCoreNum_, hv_, hBegin, hEnd);

        auto tensorL0C0 = tla::MakeTensor(l0C0, layoutL0C, Arch::PositionL0C{});
        auto tensorL0C1 = tla::MakeTensor(l0C1, layoutL0C, Arch::PositionL0C{});
        AscendC::LocalTensor<CubeT> hL10 = resource.l1Buf.template GetBufferByByte<CubeT>(kL1HOffset);
        AscendC::LocalTensor<CubeT> a0L10 = resource.l1Buf.template GetBufferByByte<CubeT>(kL1A0Offset);
        AscendC::LocalTensor<CubeT> a1L10 = resource.l1Buf.template GetBufferByByte<CubeT>(kL1A1Offset);
        AscendC::LocalTensor<CubeT> hL11 =
            resource.l1Buf.template GetBufferByByte<CubeT>(kL1HOffset + kL1StageBytes);
        AscendC::LocalTensor<CubeT> a0L11 =
            resource.l1Buf.template GetBufferByByte<CubeT>(kL1A0Offset + kL1StageBytes);
        AscendC::LocalTensor<CubeT> a1L11 =
            resource.l1Buf.template GetBufferByByte<CubeT>(kL1A1Offset + kL1StageBytes);
        auto tensorL1B0 = tla::MakeTensor(hL10, layoutL1B, Arch::PositionL1{});
        auto tensorL1A00 = tla::MakeTensor(a0L10, layoutL1A, Arch::PositionL1{});
        auto tensorL1A10 = tla::MakeTensor(a1L10, layoutL1A, Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(hL11, layoutL1B, Arch::PositionL1{});
        auto tensorL1A01 = tla::MakeTensor(a0L11, layoutL1A, Arch::PositionL1{});
        auto tensorL1A11 = tla::MakeTensor(a1L11, layoutL1A, Arch::PositionL1{});

        if (hBegin < hEnd) {
            // Split GM→L1: H+M0 first (enough for C0), M1 overlaps C0; next H+M0
            // prefetched into the other L1 stage during C0.
            CubeWaitVc(hBegin);
            LoadHM0ToL1(
                tensorL1B0, tensorL1A00, layoutH, layoutMTile, hBegin, 0U, kEventL1_0);
            uint32_t curStage = 0U;
            for (int64_t step = 1; step < n_; ++step) {
                const uint32_t mBank = static_cast<uint32_t>((step - 1) & 1);
                const uint32_t nextMBank = static_cast<uint32_t>(step & 1);
                for (uint64_t h = hBegin; h < hEnd; ++h) {
                    const int32_t curEv = curStage ? kEventL1_1 : kEventL1_0;
                    const int32_t curM1Ev = curStage ? kEventL1M1_1 : kEventL1M1_0;
                    const uint32_t nxtStage = curStage ^ 1U;
                    const int32_t nxtEv = nxtStage ? kEventL1_1 : kEventL1_0;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(curEv);

                    // M1 load overlaps C0; next H+M0 overlaps C1 (after WaitVc).
                    if (curStage == 0U) {
                        IssueM1ToL1(tensorL1A10, layoutMTile, h, mBank, curM1Ev);
                        MmadKSplit(
                            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A00, tensorL1B0,
                            tensorL0C0, false, kEventL0C0, curEv);
                    } else {
                        IssueM1ToL1(tensorL1A11, layoutMTile, h, mBank, curM1Ev);
                        MmadKSplit(
                            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A01, tensorL1B1,
                            tensorL0C0, false, kEventL0C0, curEv);
                    }
                    // Drain/SetCv0 can run while trailing M1 MTE2 completes.
                    DrainCopyC(tensorL0C0, h, 0, kEventL0C0);
                    CubeSetCv0(h);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(curM1Ev);

                    uint64_t nextH = 0;
                    uint32_t nextBank = mBank;
                    bool doPrefetch = false;
                    if (h + 1U < hEnd) {
                        nextH = h + 1U;
                        nextBank = mBank;
                        doPrefetch = true;
                    } else if (step + 1 < n_) {
                        nextH = hBegin;
                        nextBank = nextMBank;
                        doPrefetch = true;
                    }
                    if (doPrefetch) {
                        CubeWaitVc(nextH);
                        if (nxtStage == 0U) {
                            LoadHM0ToL1(
                                tensorL1B0, tensorL1A00, layoutH, layoutMTile, nextH, nextBank,
                                nxtEv);
                        } else {
                            LoadHM0ToL1(
                                tensorL1B1, tensorL1A01, layoutH, layoutMTile, nextH, nextBank,
                                nxtEv);
                        }
                    }

                    if (curStage == 0U) {
                        MmadKSplit(
                            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A10, tensorL1B0,
                            tensorL0C1, true, kEventL0C1, curEv);
                        DrainCopyC(tensorL0C1, h, 1, kEventL0C1);
                    } else {
                        MmadKSplit(
                            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A11, tensorL1B1,
                            tensorL0C1, true, kEventL0C1, curEv);
                        DrainCopyC(tensorL0C1, h, 1, kEventL0C1);
                    }
                    CubeSetCv1(h);
                    curStage = nxtStage;
                }
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kEventL1_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0A1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEventL0B1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEventL0C1);
    }
private:
    template <typename TensorL1B, typename LayoutH>
    __aicore__ inline void LoadHToL1(TensorL1B const &tensorL1B, LayoutH const &layoutH, uint64_t h)
    {
        AscendC::GlobalTensor<CubeT> gmH;
        gmH.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gmH.SetGlobalBuffer((__gm__ CubeT *)scratch_ + ScratchHElemOff(h));
        auto tensorHGm = tla::MakeTensor(gmH, layoutH, Arch::PositionGM{});
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(tensorHGm)>;
        CopyGmToL1B copyGmToL1B;
        copyGmToL1B(tensorL1B, tensorHGm);
    }

    template <typename TensorL1A, typename LayoutM>
    __aicore__ inline void LoadATileToL1(
        TensorL1A const &tensorL1A, LayoutM const &layoutM, uint64_t h, uint32_t m0, uint32_t mBank)
    {
        AscendC::GlobalTensor<CubeT> gmM;
        gmM.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gmM.SetGlobalBuffer(
            (__gm__ CubeT *)scratch_ + ScratchMElemOff(hv_, h, mBank) +
            static_cast<uint64_t>(m0) * kKDim);
        auto tensorMGm = tla::MakeTensor(gmM, layoutM, Arch::PositionGM{});
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(tensorMGm)>;
        CopyGmToL1A copyGmToL1A;
        copyGmToL1A(tensorL1A, tensorMGm);
    }

    template <typename TensorL1B, typename TensorL1A, typename LayoutH, typename LayoutM>
    __aicore__ inline void LoadHM0ToL1(
        TensorL1B const &tensorL1B, TensorL1A const &tensorL1A0, LayoutH const &layoutH,
        LayoutM const &layoutM, uint64_t h, uint32_t mBank, int32_t l1Event)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        LoadHToL1(tensorL1B, layoutH, h);
        LoadATileToL1(tensorL1A0, layoutM, h, 0, mBank);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
    }

    template <typename TensorL1A, typename LayoutM>
    __aicore__ inline void IssueM1ToL1(
        TensorL1A const &tensorL1A1, LayoutM const &layoutM, uint64_t h, uint32_t mBank,
        int32_t m1Event)
    {
        LoadATileToL1(tensorL1A1, layoutM, h, kTileM, mBank);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(m1Event);
    }

    template <typename TensorL1B, typename TensorL1A, typename LayoutH, typename LayoutM>
    __aicore__ inline void LoadStageToL1(
        TensorL1B const &tensorL1B, TensorL1A const &tensorL1A0, TensorL1A const &tensorL1A1,
        LayoutH const &layoutH, LayoutM const &layoutM, uint64_t h, uint32_t mBank, int32_t l1Event)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        LoadHToL1(tensorL1B, layoutH, h);
        LoadATileToL1(tensorL1A0, layoutM, h, 0, mBank);
        LoadATileToL1(tensorL1A1, layoutM, h, kTileM, mBank);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
    }

    template <typename TensorL1A, typename TensorL1B>
    __aicore__ inline void LoadABToL0(
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B,
        AscendC::LocalTensor<CubeT> &l0A, AscendC::LocalTensor<CubeT> &l0B,
        TensorL1A const &tensorL1A, TensorL1B const &tensorL1B, uint32_t k0, uint32_t slot)
    {
        auto layoutL0A = tla::MakeLayout<CubeT, LayoutTagL0A>(kTileM, kTileM);
        auto layoutL0B = tla::MakeLayout<CubeT, LayoutTagL0B>(kTileM, kVDim);
        auto tensorL0A = tla::MakeTensor(
            l0A[slot * kL0ATileElems], layoutL0A, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B[slot * kL0BTileElems], layoutL0B, Arch::PositionL0B{});
        auto tileA = GetTile(tensorL1A, tla::MakeCoord(0, k0), tla::MakeShape(kTileM, kTileM));
        auto tileB = GetTile(tensorL1B, tla::MakeCoord(k0, 0), tla::MakeShape(kTileM, kVDim));
        const int32_t evA = slot ? kEventL0A1 : kEventL0A0;
        const int32_t evB = slot ? kEventL0B1 : kEventL0B0;
        const int32_t evM = slot ? kEventMte1M1 : kEventMte1M0;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(evA);
        copyL1ToL0A(tensorL0A, tileA);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(evB);
        copyL1ToL0B(tensorL0B, tileB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(evM);
    }

    template <typename TensorL1A, typename TensorL1B, typename TensorL0C>
    __aicore__ inline void MmadKSplit(
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
        AscendC::LocalTensor<CubeT> &l0A, AscendC::LocalTensor<CubeT> &l0B,
        TensorL1A const &tensorL1A, TensorL1B const &tensorL1B, TensorL0C const &tensorL0C,
        bool releaseL1, int32_t l0cEvent, int32_t l1Event)
    {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
        LoadABToL0(copyL1ToL0A, copyL1ToL0B, l0A, l0B, tensorL1A, tensorL1B, 0, 0U);
        for (uint32_t k0 = 0; k0 < kKDim; k0 += kTileM) {
            const uint32_t slot = (k0 / kTileM) & 1U;
            const bool lastK = (k0 + kTileM >= kKDim);
            if (!lastK) {
                LoadABToL0(
                    copyL1ToL0A, copyL1ToL0B, l0A, l0B, tensorL1A, tensorL1B, k0 + kTileM, slot ^ 1U);
            }
            const int32_t evM = slot ? kEventMte1M1 : kEventMte1M0;
            const int32_t evA = slot ? kEventL0A1 : kEventL0A0;
            const int32_t evB = slot ? kEventL0B1 : kEventL0B0;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(evM);
            if (lastK && releaseL1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
            }
            auto layoutL0A = tla::MakeLayout<CubeT, LayoutTagL0A>(kTileM, kTileM);
            auto layoutL0B = tla::MakeLayout<CubeT, LayoutTagL0B>(kTileM, kVDim);
            auto tensorL0A = tla::MakeTensor(
                l0A[slot * kL0ATileElems], layoutL0A, Arch::PositionL0A{});
            auto tensorL0B = tla::MakeTensor(
                l0B[slot * kL0BTileElems], layoutL0B, Arch::PositionL0B{});
            const uint8_t unitFlag = lastK ? static_cast<uint8_t>(0b11) : static_cast<uint8_t>(0b10);
            tileMmad(tensorL0C, tensorL0A, tensorL0B, kTileM, kVDim, kTileM, k0 == 0, unitFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(evA);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(evB);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cEvent);
    }

    template <typename TensorL0C>
    __aicore__ inline void DrainCopyC(
        TensorL0C const &tensorL0C, uint64_t h, int64_t tile, int32_t l0cEvent)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cEvent);
        // Whole fp32Buf is one C slot per AIV; free only after both Adds.
        // Wait once before C0; C1 reuses the same in-flight slot for this head.
        if (tile == 0) {
            CubeWaitCUbFree(h);
        }
        CopyCToVecUb(h, tensorL0C, tile);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
    }

    template <typename TensorL1A, typename TensorL1B, typename TensorL0C>
    __aicore__ inline void GemmKSplitFixpipe(
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
        AscendC::LocalTensor<CubeT> &l0A, AscendC::LocalTensor<CubeT> &l0B,
        TensorL1A const &tensorL1A, TensorL1B const &tensorL1B, TensorL0C const &tensorL0C,
        uint64_t h, int64_t tile, bool releaseL1, int32_t l0cEvent)
    {
        MmadKSplit(
            copyL1ToL0A, copyL1ToL0B, tileMmad, l0A, l0B, tensorL1A, tensorL1B, tensorL0C,
            releaseL1, l0cEvent, kEventL1_0);
        DrainCopyC(tensorL0C, h, tile, l0cEvent);
    }

    template <typename TensorL0C>
    __aicore__ inline void CopyCToVecUb(uint64_t h, TensorL0C const &tensorL0C, int64_t tile)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = kVDim;
        params.mSize = kTileM;
        params.srcStride = tla::get<1, 1>(tensorL0C.stride()) / tla::get<0, 0>(tensorL0C.stride());
        params.dstStride = kVDim;
        params.quantPre = Gemm::Tile::CopyL0CToDstQuantMode<
            ArchTag, AccT, float, Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
        params.reluEn = false;
        params.unitFlag = 0b11;
        params.dualDstCtl = 0;
        params.subBlockId = static_cast<uint8_t>(h & 1U);
        auto srcOffset = tensorL0C.layout()(tensorL0C.coord());
        AscendC::Fixpipe<float, float, kCfgRowMajorUb>(
            ubC_[static_cast<uint32_t>(tile) * kCTileElems],
            tensorL0C.data()[srcOffset],
            params);
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
    GM_ADDR scratch_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    AscendC::LocalTensor<float> ubC_;
    uint64_t hv_ = 1;
    int64_t n_ = 1;
    uint32_t usedCoreNum_ = 1;
};

} // namespace MergeFwBwd

#endif
