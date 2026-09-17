/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_CUBE_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "../chunk_kda_bwd_recompute_struct.h"
#include "../chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace Catlass;
using namespace tla;

namespace KDA {

template <typename QkType>
class ChunkKdaBwdRecomputeCubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    using TileCopy =
        Gemm::Tile::PackedTileCopyTla<ArchTag, QkType, LayoutTagA, QkType, LayoutTagB, QkType, LayoutTagC>;
    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using TileMmad = Gemm::Tile::TileMmadTla<ArchTag, QkType, LayoutTagL1A>;
    using ElementAcc = typename TileCopy::ElementAccumulator;

    __aicore__ inline ChunkKdaBwdRecomputeCubeProcess(
        GM_ADDR a, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace)
        : a_(a), cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), w_(w), u_(u), workspace_(workspace)
    {
        (void)workspace_;
    }

    __aicore__ inline void Init(const ChunkKdaBwdRecomputeTilingData *tiling)
    {
        B_ = static_cast<uint64_t>(tiling->B);
        Hv_ = static_cast<uint64_t>(tiling->Hv);
        T_ = static_cast<uint64_t>(tiling->T);
        K_ = static_cast<uint64_t>(tiling->K);
        V_ = static_cast<uint64_t>(tiling->V);
        chunkNum_ = static_cast<uint64_t>(tiling->chunkNum);
        chunkSize_ = static_cast<uint64_t>(tiling->chunkSize);
        isVariable_ = tiling->isVariable;
        usedCoreNum_ = tiling->vecRow > 0 ? static_cast<uint32_t>(tiling->vecRow)
                                          : static_cast<uint32_t>(1);
    }

    __aicore__ inline void Process()
    {
        Arch::Resource<ArchTag> resource;
        AscendC::LocalTensor<QkType> aL1 = resource.l1Buf.template GetBufferByByte<QkType>(
            KdaBwdRecomputeArch35::kL1AOffset);
        AscendC::LocalTensor<QkType> l0A = resource.l0ABuf.template GetBufferByByte<QkType>(0);
        AscendC::LocalTensor<QkType> l0B = resource.l0BBuf.template GetBufferByByte<QkType>(0);
        AscendC::LocalTensor<ElementAcc> l0C0 = resource.l0CBuf.template GetBufferByByte<ElementAcc>(0);
        AscendC::LocalTensor<ElementAcc> l0C1 = resource.l0CBuf.template GetBufferByByte<ElementAcc>(
            KdaBwdRecomputeArch35::kL0CTileBytes);

        const uint32_t bt = KdaBwdRecomputeArch35::kBt;
        auto layoutL1A = tla::MakeLayout<QkType, LayoutTagL1A>(bt, bt);
        auto layoutL1B = tla::MakeLayout<QkType, LayoutTagL1B>(bt, static_cast<uint32_t>(K_));

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C1);
        // Catlass CopyL0CToGm writes LOOP3_PARA before every FIX_L0C_TO_DST.
        // Program it once; local CopyL0CToGmNd does not repeat the SPR.
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = usedCoreNum_ == 0 ? AscendC::GetBlockNum() : usedCoreNum_;
        LayoutTagA tagA = LayoutTagA::template MakeLayout<QkType>(chunkSize_, chunkSize_);
        LayoutTagC tagU = LayoutTagC::template MakeLayout<QkType>(chunkSize_, V_);
        LayoutTagC tagW = LayoutTagC::template MakeLayout<QkType>(chunkSize_, K_);
        auto layoutA = MakeLayoutFromTag(tagA);
        auto layoutU = MakeLayoutFromTag(tagU);
        auto layoutW = MakeLayoutFromTag(tagW);

        const uint64_t hv = (Hv_ == 0) ? 1 : Hv_;
        const uint64_t totalTasks = chunkNum_ * hv;
        uint64_t taskBegin = 0;
        uint64_t taskEnd = 0;
        KdaBwdRecomputeArch35::CoreTaskRange(
            coreIdx, coreNum, totalTasks, taskBegin, taskEnd);
        bool skipLoadA = false;
        if (taskBegin < taskEnd) {
            const uint64_t firstLoop = taskBegin / hv;
            const uint64_t lastLoop = (taskEnd - 1U) / hv;
            for (uint64_t loopIdx = firstLoop; loopIdx <= lastLoop; ++loopIdx) {
                uint32_t bos = 0;
                uint32_t eos = 0;
                KdaBwdRecomputeGetChunkOffset(
                    cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_,
                    static_cast<uint32_t>(loopIdx), bos, eos, isVariable_);
                const uint32_t curChunkSize = eos - bos;
                uint32_t mActual = curChunkSize;
                if (mActual == 1) {
                    mActual = 16;
                }
                uint64_t hStart = 0;
                uint64_t hEnd = 0;
                KdaBwdRecomputeArch35::HeadsOnChunk(
                    taskBegin, taskEnd, hv, loopIdx, firstLoop, lastLoop, hStart, hEnd);
                for (uint64_t hBase = hStart; hBase < hEnd; hBase += KdaBwdRecomputeArch35::kHeadRotate) {
                    const uint64_t groupEnd =
                        (hBase + KdaBwdRecomputeArch35::kHeadRotate < hEnd) ?
                            (hBase + KdaBwdRecomputeArch35::kHeadRotate) : hEnd;
                    for (uint64_t h = hBase; h < groupEnd; ++h) {
                    const uint32_t slot = static_cast<uint32_t>(h & 1U);

                    AscendC::GlobalTensor<QkType> gmA;
                    AscendC::GlobalTensor<QkType> gmU;
                    AscendC::GlobalTensor<QkType> gmW;
                    gmA.SetGlobalBuffer((__gm__ QkType *)a_ + (h * T_ + bos) * chunkSize_);
                    gmU.SetGlobalBuffer((__gm__ QkType *)u_ + (h * T_ + bos) * V_);
                    gmW.SetGlobalBuffer((__gm__ QkType *)w_ + (h * T_ + bos) * K_);
                    // A is 8KiB ND→NZ; keep L2 NORMAL. DISABLE made AIC MTE2 the wall
                    // (~2.5 GB/s). w/u stay streaming writes.
                    KdaBwdRecomputeArch35::BypassL2(gmU);
                    KdaBwdRecomputeArch35::BypassL2(gmW);

                    auto tensorAGm = tla::MakeTensor(gmA, layoutA, Arch::PositionGM{});
                    auto tensorUGm = tla::MakeTensor(gmU, layoutU, Arch::PositionGM{});
                    auto tensorWGm = tla::MakeTensor(gmW, layoutW, Arch::PositionGM{});
                    auto blockA = GetTile(
                        tensorAGm, tla::MakeCoord(0, 0), tla::MakeShape(curChunkSize, curChunkSize));
                    auto blockU = GetTile(
                        tensorUGm, tla::MakeCoord(0, 0),
                        tla::MakeShape(curChunkSize, static_cast<uint32_t>(V_)));
                    auto blockW = GetTile(
                        tensorWGm, tla::MakeCoord(0, 0),
                        tla::MakeShape(curChunkSize, static_cast<uint32_t>(K_)));

                    using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockA)>;
                    CopyGmToL1A copyGmToL1A;

                    auto tensorL1A = tla::MakeTensor(aL1, layoutL1A, Arch::PositionL1{});
                    if (!skipLoadA) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
                        copyGmToL1A(tensorL1A, blockA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KdaBwdRecomputeArch35::kEventA);
                    }
                    KdaBwdRecomputeArch35::AicWaitChunkReady<PIPE_MTE1>(slot);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KdaBwdRecomputeArch35::kEventA);

                    AscendC::LocalTensor<QkType> vbL1 = resource.l1Buf.template GetBufferByByte<QkType>(
                        KdaBwdRecomputeArch35::VbSlotOffset(slot));
                    AscendC::LocalTensor<QkType> kbgL1 = resource.l1Buf.template GetBufferByByte<QkType>(
                        KdaBwdRecomputeArch35::KbgSlotOffset(slot));
                    auto tensorL1Vb = tla::MakeTensor(vbL1, layoutL1B, Arch::PositionL1{});
                    auto tensorL1Kbg = tla::MakeTensor(kbgL1, layoutL1B, Arch::PositionL1{});

                    RunMmadFromL1(
                        copyL1ToL0A, copyL1ToL0B, tileMmad,
                        tensorL1A, tensorL1Vb, blockU, l0A, l0B, l0C0,
                        mActual, static_cast<uint32_t>(V_), curChunkSize, slot,
                        KdaBwdRecomputeArch35::kEventL0C0,
                        true, false, true, false, false);
                    DrainFixpipe(
                        blockU, l0C0, mActual, static_cast<uint32_t>(V_),
                        KdaBwdRecomputeArch35::kEventL0C0);
                    uint64_t nxtLoop = 0;
                    uint64_t nxtH = 0;
                    if (KdaBwdRecomputeArch35::NextChunkHead(
                            loopIdx, h, hEnd, lastLoop, nxtLoop, nxtH)) {
                        uint32_t nxtBos = bos;
                        uint32_t nxtEos = eos;
                        if (nxtLoop != loopIdx) {
                            KdaBwdRecomputeGetChunkOffset(
                                cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_,
                                static_cast<uint32_t>(nxtLoop), nxtBos, nxtEos, isVariable_);
                        }
                        const uint32_t nxtChunk = nxtEos - nxtBos;
                        AscendC::GlobalTensor<QkType> gmANxt;
                        gmANxt.SetGlobalBuffer((__gm__ QkType *)a_ + (nxtH * T_ + nxtBos) * chunkSize_);
                        auto tensorAGmNxt = tla::MakeTensor(gmANxt, layoutA, Arch::PositionGM{});
                        auto blockANxt = GetTile(
                            tensorAGmNxt, tla::MakeCoord(0, 0),
                            tla::MakeShape(nxtChunk, nxtChunk));
                        using CopyGmToL1ANxt = typename TileCopy::template CopyGmToL1A<decltype(blockANxt)>;
                        CopyGmToL1ANxt copyGmToL1ANxt;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
                        copyGmToL1ANxt(tensorL1A, blockANxt);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KdaBwdRecomputeArch35::kEventA);
                        skipLoadA = true;
                    } else {
                        skipLoadA = false;
                    }
                    RunMmadFromL1(
                        copyL1ToL0A, copyL1ToL0B, tileMmad,
                        tensorL1A, tensorL1Kbg, blockW, l0A, l0B, l0C1,
                        mActual, static_cast<uint32_t>(K_), curChunkSize, slot,
                        KdaBwdRecomputeArch35::kEventL0C1,
                        false, true, false, true, true);
                    }
                }
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C1);
    }

private:
    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyL0CToGmNd(
        TensorC const &tensorC, TensorL0C const &tensorL0C, uint8_t unitFlag)
    {
        AscendC::DataCopyCO12DstParams params;
        params.nSize = tla::get<1>(tensorC.originShape());
        params.mSize = tla::get<0>(tensorC.originShape());
        params.dstStride = tla::get<0>(tensorC.stride());
        params.srcStride = tla::get<1, 1>(tensorL0C.stride()) / tla::get<0, 0>(tensorL0C.stride());
        params.quantPre = Gemm::Tile::CopyL0CToDstQuantMode<
            ArchTag, ElementAcc, QkType, Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
        params.nz2ndEn = true;
        params.reluPre = false;
        params.unitFlag = unitFlag;
        auto dstOffset = tensorC.layout()(tensorC.coord());
        auto srcOffset = tensorL0C.layout()(tensorL0C.coord());
        AscendC::DataCopy(tensorC.data()[dstOffset], tensorL0C.data()[srcOffset], params);
    }

    template <typename TensorL1A, typename TensorL1B, typename TensorC>
    __aicore__ inline void RunMmadFromL1(
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
        TensorL1A &tensorL1A, TensorL1B &tensorL1B, TensorC &tensorC,
        AscendC::LocalTensor<QkType> &l0A, AscendC::LocalTensor<QkType> &l0B,
        AscendC::LocalTensor<ElementAcc> &l0C, uint32_t m, uint32_t n, uint32_t k,
        uint16_t slot, int32_t l0cEvent, bool copyL0A, bool releaseL0A, bool releaseL1A,
        bool releaseL1B, bool doFixpipe)
    {
        auto layoutL0A = tla::MakeLayout<QkType, LayoutTagL0A>(m, k);
        auto layoutL0B = tla::MakeLayout<QkType, LayoutTagL0B>(k, n);
        auto layoutL0C = tla::MakeLayoutL0C(m, n);
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Arch::PositionL0C{});
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));

        if (copyL0A) {
            auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
            copyL1ToL0A(tensorL0A, tileL1A);
            if (releaseL1A) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        copyL1ToL0B(tensorL0B, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(KdaBwdRecomputeArch35::kEventMte1M);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(KdaBwdRecomputeArch35::kEventMte1M);
        if (releaseL1B) {
            KdaBwdRecomputeArch35::AicSetChunkFree<PIPE_MTE1>(slot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
        tileMmad(tensorL0C, tensorL0A, tensorL0B, m, n, k, true, 0b11);
        if (releaseL0A) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cEvent);
        if (doFixpipe) {
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cEvent);
            CopyL0CToGmNd(tensorC, tensorL0C, 0b11);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
        }
    }

    template <typename TensorC>
    __aicore__ inline void DrainFixpipe(
        TensorC &tensorC,
        AscendC::LocalTensor<ElementAcc> &l0C, uint32_t m, uint32_t n, int32_t l0cEvent)
    {
        auto layoutL0C = tla::MakeLayoutL0C(m, n);
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cEvent);
        CopyL0CToGmNd(tensorC, tensorL0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEvent);
    }

    GM_ADDR a_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR w_;
    GM_ADDR u_;
    GM_ADDR workspace_;
    uint64_t B_ = 0;
    uint64_t Hv_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 128;
    uint64_t V_ = 128;
    uint64_t chunkNum_ = 0;
    uint64_t chunkSize_ = 64;
    uint32_t usedCoreNum_ = 1;
    int64_t isVariable_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_CUBE_H
