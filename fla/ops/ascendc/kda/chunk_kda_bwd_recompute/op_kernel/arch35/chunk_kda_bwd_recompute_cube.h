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
    }

    __aicore__ inline void Process()
    {
        Arch::Resource<ArchTag> resource;
        AscendC::LocalTensor<QkType> aL1 = resource.l1Buf.template GetBufferByByte<QkType>(
            KdaBwdRecomputeArch35::kL1AOffset);
        AscendC::LocalTensor<QkType> l0A = resource.l0ABuf.template GetBufferByByte<QkType>(0);
        AscendC::LocalTensor<QkType> l0B = resource.l0BBuf.template GetBufferByByte<QkType>(0);
        AscendC::LocalTensor<ElementAcc> l0C = resource.l0CBuf.template GetBufferByByte<ElementAcc>(0);

        const uint32_t bt = KdaBwdRecomputeArch35::kBt;
        auto layoutL1A = tla::MakeLayout<QkType, LayoutTagL1A>(bt, bt);
        auto layoutL1B = tla::MakeLayout<QkType, LayoutTagL1B>(bt, static_cast<uint32_t>(K_));

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C);

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        if (coreIdx < chunkNum_) {
            KdaBwdRecomputeArch35::AicSetChunkFree<PIPE_FIX>();
            KdaBwdRecomputeArch35::AicSetChunkFree<PIPE_FIX>();
        }
        LayoutTagA tagA = LayoutTagA::template MakeLayout<QkType>(chunkSize_, chunkSize_);
        LayoutTagC tagU = LayoutTagC::template MakeLayout<QkType>(chunkSize_, V_);
        LayoutTagC tagW = LayoutTagC::template MakeLayout<QkType>(chunkSize_, K_);
        auto layoutA = MakeLayoutFromTag(tagA);
        auto layoutU = MakeLayoutFromTag(tagU);
        auto layoutW = MakeLayoutFromTag(tagW);

        for (uint32_t loopIdx = coreIdx; loopIdx < chunkNum_; loopIdx += AscendC::GetBlockNum()) {
            uint32_t bos = 0;
            uint32_t eos = 0;
            KdaBwdRecomputeGetChunkOffset(
                cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_, loopIdx, bos, eos, isVariable_);
            const uint32_t curChunkSize = eos - bos;
            uint32_t mActual = curChunkSize;
            if (mActual == 1) {
                mActual = 16;
            }

            for (uint64_t h = 0; h < Hv_; ++h) {
                const uint32_t slot =
                    (loopIdx * static_cast<uint32_t>(Hv_) + static_cast<uint32_t>(h)) & 1U;
                KdaBwdRecomputeArch35::AicWaitChunkReady<PIPE_MTE1>();

                AscendC::GlobalTensor<QkType> gmA;
                AscendC::GlobalTensor<QkType> gmU;
                AscendC::GlobalTensor<QkType> gmW;
                gmA.SetGlobalBuffer((__gm__ QkType *)a_ + (h * T_ + bos) * chunkSize_);
                gmU.SetGlobalBuffer((__gm__ QkType *)u_ + (h * T_ + bos) * V_);
                gmW.SetGlobalBuffer((__gm__ QkType *)w_ + (h * T_ + bos) * K_);

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
                using CopyL0CToDstU = typename TileCopy::template CopyL0CToDst<decltype(blockU)>;
                using CopyL0CToDstW = typename TileCopy::template CopyL0CToDst<decltype(blockW)>;
                CopyGmToL1A copyGmToL1A;
                CopyL0CToDstU copyL0CToU;
                CopyL0CToDstW copyL0CToW;

                auto tensorL1A = tla::MakeTensor(aL1, layoutL1A, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
                copyGmToL1A(tensorL1A, blockA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(KdaBwdRecomputeArch35::kEventA);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(KdaBwdRecomputeArch35::kEventA);

                AscendC::LocalTensor<QkType> vbL1 = resource.l1Buf.template GetBufferByByte<QkType>(
                    KdaBwdRecomputeArch35::VbSlotOffset(slot));
                AscendC::LocalTensor<QkType> kbgL1 = resource.l1Buf.template GetBufferByByte<QkType>(
                    KdaBwdRecomputeArch35::KbgSlotOffset(slot));
                auto tensorL1Vb = tla::MakeTensor(vbL1, layoutL1B, Arch::PositionL1{});
                auto tensorL1Kbg = tla::MakeTensor(kbgL1, layoutL1B, Arch::PositionL1{});

                RunMmadFromL1(
                    copyL1ToL0A, copyL1ToL0B, tileMmad, copyL0CToU,
                    tensorL1A, tensorL1Vb, blockU, l0A, l0B, l0C,
                    mActual, static_cast<uint32_t>(V_), curChunkSize);
                RunMmadFromL1(
                    copyL1ToL0A, copyL1ToL0B, tileMmad, copyL0CToW,
                    tensorL1A, tensorL1Kbg, blockW, l0A, l0B, l0C,
                    mActual, static_cast<uint32_t>(K_), curChunkSize);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
                KdaBwdRecomputeArch35::AicSetChunkFree<PIPE_FIX>();
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(KdaBwdRecomputeArch35::kEventA);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C);
    }

private:
    template <typename CopyL0CToDst, typename TensorL1A, typename TensorL1B, typename TensorC>
    __aicore__ inline void RunMmadFromL1(
        CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
        CopyL0CToDst &copyL0CToDst, TensorL1A &tensorL1A, TensorL1B &tensorL1B, TensorC &tensorC,
        AscendC::LocalTensor<QkType> &l0A, AscendC::LocalTensor<QkType> &l0B,
        AscendC::LocalTensor<ElementAcc> &l0C, uint32_t m, uint32_t n, uint32_t k)
    {
        auto layoutL0A = tla::MakeLayout<QkType, LayoutTagL0A>(m, k);
        auto layoutL0B = tla::MakeLayout<QkType, LayoutTagL0B>(k, n);
        auto layoutL0C = tla::MakeLayoutL0C(m, n);
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        copyL1ToL0A(tensorL0A, tileL1A);
        copyL1ToL0B(tensorL0B, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(KdaBwdRecomputeArch35::kEventMte1M);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(KdaBwdRecomputeArch35::kEventMte1M);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C);
        tileMmad(tensorL0C, tensorL0A, tensorL0B, m, n, k, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0A);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(KdaBwdRecomputeArch35::kEventL0B);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(KdaBwdRecomputeArch35::kEventL0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(KdaBwdRecomputeArch35::kEventL0C);
        copyL0CToDst(tensorC, tensorL0C, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(KdaBwdRecomputeArch35::kEventL0C);
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
    int64_t isVariable_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_CUBE_H
