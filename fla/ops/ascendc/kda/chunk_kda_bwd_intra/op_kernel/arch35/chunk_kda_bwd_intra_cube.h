/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_ARCH35_CUBE_H
#define CHUNK_KDA_BWD_INTRA_ARCH35_CUBE_H

#define CATLASS_ARCH 3510
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "../chunk_kda_bwd_intra_struct.h"
#include "../chunk_kda_bwd_intra_common.h"

namespace KDA {

template <uint32_t K_DIM, uint32_t CHUNK_SIZE, bool SAFE_GATE, bool VARLEN_TND>
class ChunkKdaBwdIntraCubeProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using Element = float;
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;

    using LowerCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, Element, RowMajor, Element, RowMajor, Element, RowMajor>;
    using UpperCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, Element, ColumnMajor, Element, RowMajor, Element, RowMajor>;
    using LowerL1ALayout = typename LowerCopy::LayoutTagL1A;
    using LowerL1BLayout = typename LowerCopy::LayoutTagL1B;
    using LowerL0ALayout = typename LowerCopy::LayoutTagL0A;
    using LowerL0BLayout = typename LowerCopy::LayoutTagL0B;
    using UpperL1ALayout = typename UpperCopy::LayoutTagL1A;
    using UpperL1BLayout = typename UpperCopy::LayoutTagL1B;
    using UpperL0ALayout = typename UpperCopy::LayoutTagL0A;
    using UpperL0BLayout = typename UpperCopy::LayoutTagL0B;
    using LowerTileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, LowerL1ALayout>;
    using UpperTileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, Element, UpperL1ALayout>;

    template <typename Tensor>
    using CopyLowerGmToL1A = typename LowerCopy::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyLowerGmToL1B = typename LowerCopy::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyLowerL0CToGm = typename LowerCopy::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyUpperGmToL1A = typename UpperCopy::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyUpperGmToL1B = typename UpperCopy::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyUpperL0CToGm = typename UpperCopy::template CopyL0CToDst<Tensor>;

    __aicore__ ChunkKdaBwdIntraCubeProcess(GM_ADDR chunkMetadata, GM_ADDR workspace)
        : chunkMetadata_(chunkMetadata), workspace_(workspace)
    {
        static_assert(SAFE_GATE, "The unsafe gate kernel is intentionally not instantiated in v1.");
        static_assert(K_DIM == 64 || K_DIM == 128 || K_DIM == 256, "Unsupported K.");
        static_assert(!VARLEN_TND || K_DIM == 128, "Varlen P0 only supports K=128.");
        static_assert(CHUNK_SIZE == kChunkSize, "Only chunk_size=64 is supported.");
    }

    __aicore__ inline void Init(const ChunkKdaBwdIntraTilingData &tiling)
    {
        tiling_ = tiling;
        if constexpr (VARLEN_TND) {
            chunkMetadataGm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ int64_t *>(chunkMetadata_));
        }
    }

    __aicore__ inline void Process()
    {
        AscendC::SetHF32Mode(false);
        if constexpr (K_DIM == 128) {
            ProcessPipelinedK128();
        } else {
            ProcessBlockMmad();
        }
    }

private:
    static constexpr uint32_t kProcessRowBlock =
        ProcessRowBlock<K_DIM, VARLEN_TND>::value;

    __aicore__ inline void ProcessBlockMmad()
    {
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false>;
        static constexpr uint32_t kL0ReductionTile = K_DIM == 256 ? 32 : 64;
        using LowerL1Shape = tla::Shape<tla::Int<32>, tla::Int<K_DIM>, tla::Int<CHUNK_SIZE>>;
        using LowerL0Shape =
            tla::Shape<tla::Int<32>, tla::Int<K_DIM>, tla::Int<kL0ReductionTile>>;
        using UpperL1Shape = tla::Shape<tla::Int<16>, tla::Int<K_DIM>, tla::Int<2 * CHUNK_SIZE>>;
        using UpperL0Shape =
            tla::Shape<tla::Int<16>, tla::Int<K_DIM>, tla::Int<kL0ReductionTile>>;

        using LowerMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, LowerL1Shape, LowerL0Shape, Element, Element, Element, void, LowerCopy>;
        using UpperMmad = Catlass::Gemm::Block::BlockMmadTla<
            DispatchPolicy, UpperL1Shape, UpperL0Shape, Element, Element, Element, void, UpperCopy>;

        Catlass::Arch::Resource<ArchTag> resource;
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kHeadsPerWindow - 1) / kHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
        uint32_t taskIdx = coreIdx / headWindowCount;
        uint32_t headWindowIdx = coreIdx % headWindowCount;
        const uint32_t taskStride = coreNum / headWindowCount;
        const uint32_t headWindowStride = coreNum % headWindowCount;
        uint64_t windowIdx = 0;

        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum) {
            const uint32_t headBase = headWindowIdx * kHeadsPerWindow;
            const uint32_t headCount = headBase + 1 < headNum ? 2 : 1;
            const ChunkTask task =
                GetChunkTask<VARLEN_TND>(tiling_, chunkMetadataGm_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            const uint32_t rowBlockCount =
                (validLen + kProcessRowBlock - 1) / kProcessRowBlock;
            for (uint32_t rowBlock = 0; rowBlock < rowBlockCount; ++rowBlock) {
                const uint32_t rowStart = rowBlock * kProcessRowBlock;
                const uint32_t validRows =
                    rowStart + kProcessRowBlock <= validLen ?
                        kProcessRowBlock : validLen - rowStart;
                const uint32_t prefix = rowStart + validRows;
                const uint32_t future = validLen - rowStart;
                for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeReadyFlag_);
                    const uint32_t slot = WorkspaceSlot(windowIdx, headInWindow);
                    const uint64_t slotBase =
                        static_cast<uint64_t>(coreIdx) * tiling_.workspaceCoreSize +
                        static_cast<uint64_t>(slot) * tiling_.workspaceSlotSize;
                    RunLower<LowerMmad, RowMajor>(resource, slotBase, prefix);
                    RunUpper<UpperMmad, RowMajor, ColumnMajor>(resource, slotBase, future);
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecReadyFlag_);
                }
                ++windowIdx;
            }
            taskIdx += taskStride;
            headWindowIdx += headWindowStride;
            if (headWindowIdx >= headWindowCount) {
                headWindowIdx -= headWindowCount;
                ++taskIdx;
            }
        }
    }

    __aicore__ inline void InitPipelineFlags()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kLowerEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kUpperEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kLowerEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kUpperEvent);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kLowerEvent);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kUpperEvent);
    }

    __aicore__ inline void DrainPipelineFlags()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kLowerEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kUpperEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kLowerEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kUpperEvent);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kLowerEvent);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kUpperEvent);
    }

    __aicore__ inline void ProcessPipelinedK128()
    {
        static_assert(K_DIM == 128, "The explicit A5 lower/upper pipeline is specialized for K=128.");
        static_assert(kL1UsedBytes <= kL1CapacityBytes,
                      "chunk_kda_bwd_intra A5 pipeline exceeds the 512KB L1 capacity.");
        static_assert(kL0AUsedBytes <= kL0ABCapacityBytes,
                      "chunk_kda_bwd_intra A5 pipeline exceeds the L0A capacity.");
        static_assert(kL0BUsedBytes <= kL0ABCapacityBytes,
                      "chunk_kda_bwd_intra A5 pipeline exceeds the L0B capacity.");
        static_assert(kL0CUsedBytes <= kL0CCapacityBytes,
                      "chunk_kda_bwd_intra A5 pipeline exceeds the 256KB L0C capacity.");

        Catlass::Arch::Resource<ArchTag> resource;
        InitPipelineFlags();
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t headNum = static_cast<uint32_t>(tiling_.headNum);
        const uint32_t headWindowCount =
            (headNum + kHeadsPerWindow - 1) / kHeadsPerWindow;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
        uint32_t taskIdx = coreIdx / headWindowCount;
        uint32_t headWindowIdx = coreIdx % headWindowCount;
        const uint32_t taskStride = coreNum / headWindowCount;
        const uint32_t headWindowStride = coreNum % headWindowCount;
        uint64_t windowIdx = 0;

        for (uint64_t taskGroupIdx = coreIdx; taskGroupIdx < taskGroupCount;
             taskGroupIdx += coreNum) {
            const uint32_t headBase = headWindowIdx * kHeadsPerWindow;
            const uint32_t headCount = headBase + 1 < headNum ? 2 : 1;
            const ChunkTask task =
                GetChunkTask<VARLEN_TND>(tiling_, chunkMetadataGm_, taskIdx);
            const uint32_t validLen = task.end - task.begin;
            const uint32_t rowBlockCount =
                (validLen + kProcessRowBlock - 1) / kProcessRowBlock;
            for (uint32_t rowBlock = 0; rowBlock < rowBlockCount; ++rowBlock) {
                const uint32_t rowStart = rowBlock * kProcessRowBlock;
                const uint32_t validRows =
                    rowStart + kProcessRowBlock <= validLen ?
                        kProcessRowBlock : validLen - rowStart;
                const uint32_t prefix = rowStart + validRows;
                const uint32_t future = validLen - rowStart;
                for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
                    Catlass::Arch::CrossCoreWaitFlag(vecToCubeReadyFlag_);
                    const uint32_t slot = WorkspaceSlot(windowIdx, headInWindow);
                    const uint64_t slotBase =
                        static_cast<uint64_t>(coreIdx) * tiling_.workspaceCoreSize +
                        static_cast<uint64_t>(slot) * tiling_.workspaceSlotSize;
                    RunLowerUpperPipeline(resource, slotBase, prefix, future);
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecReadyFlag_);
                }
                ++windowIdx;
            }
            taskIdx += taskStride;
            headWindowIdx += headWindowStride;
            if (headWindowIdx >= headWindowCount) {
                headWindowIdx -= headWindowCount;
                ++taskIdx;
            }
        }
        DrainPipelineFlags();
    }

    __aicore__ inline void RunLowerUpperPipeline(
        Catlass::Arch::Resource<ArchTag> &resource, uint64_t slotBase,
        uint32_t prefix, uint32_t future)
    {
        AscendC::GlobalTensor<Element> lowerA;
        AscendC::GlobalTensor<Element> lowerB;
        AscendC::GlobalTensor<Element> lowerC;
        AscendC::GlobalTensor<Element> upperA;
        AscendC::GlobalTensor<Element> upperB;
        AscendC::GlobalTensor<Element> upperC;
        lowerA.SetGlobalBuffer((__gm__ Element *)(workspace_ + slotBase + tiling_.aLowerOffset));
        lowerB.SetGlobalBuffer((__gm__ Element *)(workspace_ + slotBase + tiling_.bLowerOffset));
        lowerC.SetGlobalBuffer((__gm__ Element *)(
            workspace_ + slotBase + tiling_.resultRegionOffset + tiling_.resultDqOffset));
        upperA.SetGlobalBuffer((__gm__ Element *)(workspace_ + slotBase + tiling_.aUpperOffset));
        upperB.SetGlobalBuffer((__gm__ Element *)(workspace_ + slotBase + tiling_.bUpperOffset));
        upperC.SetGlobalBuffer((__gm__ Element *)(
            workspace_ + slotBase + tiling_.resultRegionOffset + tiling_.resultDkUpperOffset));

        const uint32_t lowerRows = 2 * kProcessRowBlock;
        const uint32_t upperRows = kProcessRowBlock;
        const uint32_t upperReduction = 2 * future;
        auto lowerLayoutA = tla::MakeLayout<Element, RowMajor>(lowerRows, prefix);
        auto lowerLayoutB = tla::MakeLayout<Element, RowMajor>(prefix, K_DIM);
        auto lowerLayoutC = tla::MakeLayout<Element, RowMajor>(lowerRows, K_DIM);
        auto upperLayoutA = tla::MakeLayout<Element, ColumnMajor>(upperRows, upperReduction);
        auto upperLayoutB = tla::MakeLayout<Element, RowMajor>(upperReduction, K_DIM);
        auto upperLayoutC = tla::MakeLayout<Element, RowMajor>(upperRows, K_DIM);
        auto lowerTensorA = tla::MakeTensor(lowerA, lowerLayoutA, Catlass::Arch::PositionGM{});
        auto lowerTensorB = tla::MakeTensor(lowerB, lowerLayoutB, Catlass::Arch::PositionGM{});
        auto lowerTensorC = tla::MakeTensor(lowerC, lowerLayoutC, Catlass::Arch::PositionGM{});
        auto upperTensorA = tla::MakeTensor(upperA, upperLayoutA, Catlass::Arch::PositionGM{});
        auto upperTensorB = tla::MakeTensor(upperB, upperLayoutB, Catlass::Arch::PositionGM{});
        auto upperTensorC = tla::MakeTensor(upperC, upperLayoutC, Catlass::Arch::PositionGM{});

        AscendC::LocalTensor<Element> lowerL1A =
            resource.l1Buf.template GetBufferByByte<Element>(kLowerL1AOffset);
        AscendC::LocalTensor<Element> lowerL1B =
            resource.l1Buf.template GetBufferByByte<Element>(kLowerL1BOffset);
        AscendC::LocalTensor<Element> upperL1A =
            resource.l1Buf.template GetBufferByByte<Element>(kUpperL1AOffset);
        AscendC::LocalTensor<Element> upperL1B =
            resource.l1Buf.template GetBufferByByte<Element>(kUpperL1BOffset);
        auto lowerL1BaseA = tla::MakeTensor(
            lowerL1A, kLowerL1ALayoutSpec, Catlass::Arch::PositionL1{});
        auto lowerL1BaseB = tla::MakeTensor(
            lowerL1B, kLowerL1BLayoutSpec, Catlass::Arch::PositionL1{});
        auto upperL1BaseA = tla::MakeTensor(
            upperL1A, kUpperL1ALayoutSpec, Catlass::Arch::PositionL1{});
        auto upperL1BaseB = tla::MakeTensor(
            upperL1B, kUpperL1BLayoutSpec, Catlass::Arch::PositionL1{});
        auto lowerL1TensorA = tla::GetTile(
            lowerL1BaseA, tla::MakeCoord(0U, 0U), tla::MakeShape(lowerRows, prefix));
        auto lowerL1TensorB = tla::GetTile(
            lowerL1BaseB, tla::MakeCoord(0U, 0U), tla::MakeShape(prefix, K_DIM));
        auto upperL1TensorA = tla::GetTile(
            upperL1BaseA, tla::MakeCoord(0U, 0U), tla::MakeShape(upperRows, upperReduction));
        auto upperL1TensorB = tla::GetTile(
            upperL1BaseB, tla::MakeCoord(0U, 0U), tla::MakeShape(upperReduction, K_DIM));

        CopyLowerGmToL1A<decltype(lowerTensorA)> copyLowerGmToL1A;
        CopyLowerGmToL1B<decltype(lowerTensorB)> copyLowerGmToL1B;
        CopyUpperGmToL1A<decltype(upperTensorA)> copyUpperGmToL1A;
        CopyUpperGmToL1B<decltype(upperTensorB)> copyUpperGmToL1B;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kLowerEvent);
        copyLowerGmToL1A(lowerL1TensorA, lowerTensorA);
        copyLowerGmToL1B(lowerL1TensorB, lowerTensorB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kLowerEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kUpperEvent);
        copyUpperGmToL1A(upperL1TensorA, upperTensorA);
        copyUpperGmToL1B(upperL1TensorB, upperTensorB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kUpperEvent);

        AscendC::LocalTensor<Element> lowerL0A =
            resource.l0ABuf.template GetBufferByByte<Element>(kLowerL0AOffset);
        AscendC::LocalTensor<Element> lowerL0B =
            resource.l0BBuf.template GetBufferByByte<Element>(kLowerL0BOffset);
        AscendC::LocalTensor<Element> upperL0A =
            resource.l0ABuf.template GetBufferByByte<Element>(kUpperL0AOffset);
        AscendC::LocalTensor<Element> upperL0B =
            resource.l0BBuf.template GetBufferByByte<Element>(kUpperL0BOffset);
        auto lowerL0TensorA = tla::MakeTensor(
            lowerL0A, tla::MakeLayout<Element, LowerL0ALayout>(lowerRows, prefix),
            Catlass::Arch::PositionL0A{});
        auto lowerL0TensorB = tla::MakeTensor(
            lowerL0B, tla::MakeLayout<Element, LowerL0BLayout>(prefix, K_DIM),
            Catlass::Arch::PositionL0B{});
        auto upperL0TensorA = tla::MakeTensor(
            upperL0A, tla::MakeLayout<Element, UpperL0ALayout>(upperRows, upperReduction),
            Catlass::Arch::PositionL0A{});
        auto upperL0TensorB = tla::MakeTensor(
            upperL0B, tla::MakeLayout<Element, UpperL0BLayout>(upperReduction, K_DIM),
            Catlass::Arch::PositionL0B{});
        typename LowerCopy::CopyL1ToL0A copyLowerL1ToL0A;
        typename LowerCopy::CopyL1ToL0B copyLowerL1ToL0B;
        typename UpperCopy::CopyL1ToL0A copyUpperL1ToL0A;
        typename UpperCopy::CopyL1ToL0B copyUpperL1ToL0B;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kLowerEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kLowerEvent);
        copyLowerL1ToL0A(lowerL0TensorA, lowerL1TensorA);
        copyLowerL1ToL0B(lowerL0TensorB, lowerL1TensorB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kLowerEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kLowerEvent);

        using Accumulator = typename LowerCopy::ElementAccumulator;
        AscendC::LocalTensor<Accumulator> lowerL0C =
            resource.l0CBuf.template GetBufferByByte<Accumulator>(kLowerL0COffset);
        AscendC::LocalTensor<Accumulator> upperL0C =
            resource.l0CBuf.template GetBufferByByte<Accumulator>(kUpperL0COffset);
        auto lowerL0CTensor = tla::MakeTensor(
            lowerL0C, tla::MakeLayoutL0C(lowerRows, K_DIM), Catlass::Arch::PositionL0C{});
        auto upperL0CTensor = tla::MakeTensor(
            upperL0C, tla::MakeLayoutL0C(upperRows, K_DIM), Catlass::Arch::PositionL0C{});
        LowerTileMmad lowerMmad;
        UpperTileMmad upperMmad;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kLowerEvent);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kLowerEvent);
        lowerMmad(lowerL0CTensor, lowerL0TensorA, lowerL0TensorB,
                  lowerRows, K_DIM, prefix, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kLowerEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kLowerEvent);

        // Lower and Upper reuse the same L0A/L0B storage. Once Lower MMAD
        // releases it, Upper MTE1 can refill L0 while Lower FIX writes GM.
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kUpperEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kLowerEvent);
        copyUpperL1ToL0A(upperL0TensorA, upperL1TensorA);
        copyUpperL1ToL0B(upperL0TensorB, upperL1TensorB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kUpperEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kUpperEvent);

        CopyLowerL0CToGm<decltype(lowerTensorC)> copyLowerL0CToGm;
        CopyUpperL0CToGm<decltype(upperTensorC)> copyUpperL0CToGm;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kLowerEvent);
        copyLowerL0CToGm(lowerTensorC, lowerL0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kLowerEvent);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kUpperEvent);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kUpperEvent);
        upperMmad(upperL0CTensor, upperL0TensorA, upperL0TensorB,
                  upperRows, K_DIM, upperReduction, true, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kLowerEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kUpperEvent);

        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kUpperEvent);
        copyUpperL0CToGm(upperTensorC, upperL0CTensor, 0b11);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kUpperEvent);
    }

    template <typename LowerMmad, typename RowMajor>
    __aicore__ inline void RunLower(
        Catlass::Arch::Resource<ArchTag> &resource, uint64_t slotBase, uint32_t prefix)
    {
        AscendC::GlobalTensor<float> a;
        AscendC::GlobalTensor<float> b;
        AscendC::GlobalTensor<float> c;
        a.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.aLowerOffset));
        b.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.bLowerOffset));
        c.SetGlobalBuffer((__gm__ float *)(
            workspace_ + slotBase + tiling_.resultRegionOffset + tiling_.resultDqOffset));

        const uint32_t m = 2 * kProcessRowBlock;
        auto layoutA = tla::MakeLayout<float, RowMajor>(m, prefix);
        auto layoutB = tla::MakeLayout<float, RowMajor>(prefix, K_DIM);
        auto layoutC = tla::MakeLayout<float, RowMajor>(m, K_DIM);
        auto tensorA = tla::MakeTensor(a, layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(b, layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(c, layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, K_DIM, prefix};
        LowerMmad mma(resource);
        mma(tensorA, tensorB, tensorC, shape);
    }

    template <typename UpperMmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunUpper(
        Catlass::Arch::Resource<ArchTag> &resource, uint64_t slotBase, uint32_t future)
    {
        AscendC::GlobalTensor<float> a;
        AscendC::GlobalTensor<float> b;
        AscendC::GlobalTensor<float> c;
        a.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.aUpperOffset));
        b.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.bUpperOffset));
        c.SetGlobalBuffer((__gm__ float *)(
            workspace_ + slotBase + tiling_.resultRegionOffset + tiling_.resultDkUpperOffset));

        const uint32_t reduction = 2 * future;
        auto layoutA =
            tla::MakeLayout<float, ColumnMajor>(kProcessRowBlock, reduction);
        auto layoutB = tla::MakeLayout<float, RowMajor>(reduction, K_DIM);
        auto layoutC =
            tla::MakeLayout<float, RowMajor>(kProcessRowBlock, K_DIM);
        auto tensorA = tla::MakeTensor(a, layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(b, layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(c, layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{kProcessRowBlock, K_DIM, reduction};
        UpperMmad mma(resource);
        mma(tensorA, tensorB, tensorC, shape);
    }

    static constexpr uint32_t kLowerRows = 2 * kProcessRowBlock;
    static constexpr uint32_t kUpperRows = kProcessRowBlock;
    static constexpr uint32_t kLowerReduction = CHUNK_SIZE;
    static constexpr uint32_t kUpperReduction = 2 * CHUNK_SIZE;
    static constexpr uint32_t kLowerL1ABytes = kLowerRows * kLowerReduction * sizeof(Element);
    static constexpr uint32_t kLowerL1BBytes = kLowerReduction * K_DIM * sizeof(Element);
    static constexpr uint32_t kUpperL1ABytes = kUpperRows * kUpperReduction * sizeof(Element);
    static constexpr uint32_t kUpperL1BBytes = kUpperReduction * K_DIM * sizeof(Element);
    static constexpr uint32_t kLowerL1AOffset = 0;
    static constexpr uint32_t kLowerL1BOffset = kLowerL1AOffset + kLowerL1ABytes;
    static constexpr uint32_t kUpperL1AOffset = kLowerL1BOffset + kLowerL1BBytes;
    static constexpr uint32_t kUpperL1BOffset = kUpperL1AOffset + kUpperL1ABytes;
    static constexpr uint32_t kL1UsedBytes = kUpperL1BOffset + kUpperL1BBytes;
    static constexpr uint32_t kL1CapacityBytes = 512 * 1024;
    static constexpr auto kLowerL1ALayoutSpec = tla::MakeLayout<Element, LowerL1ALayout>(
        tla::Int<kLowerRows>{}, tla::Int<kLowerReduction>{});
    static constexpr auto kLowerL1BLayoutSpec = tla::MakeLayout<Element, LowerL1BLayout>(
        tla::Int<kLowerReduction>{}, tla::Int<K_DIM>{});
    static constexpr auto kUpperL1ALayoutSpec = tla::MakeLayout<Element, UpperL1ALayout>(
        tla::Int<kUpperRows>{}, tla::Int<kUpperReduction>{});
    static constexpr auto kUpperL1BLayoutSpec = tla::MakeLayout<Element, UpperL1BLayout>(
        tla::Int<kUpperReduction>{}, tla::Int<K_DIM>{});

    static constexpr uint32_t kLowerL0ABytes = kLowerL1ABytes;
    static constexpr uint32_t kLowerL0BBytes = kLowerL1BBytes;
    static constexpr uint32_t kUpperL0ABytes = kUpperL1ABytes;
    static constexpr uint32_t kUpperL0BBytes = kUpperL1BBytes;
    static constexpr uint32_t kLowerL0AOffset = 0;
    static constexpr uint32_t kUpperL0AOffset = kLowerL0AOffset;
    static constexpr uint32_t kLowerL0BOffset = 0;
    static constexpr uint32_t kUpperL0BOffset = kLowerL0BOffset;
    static constexpr uint32_t kL0AUsedBytes =
        kLowerL0ABytes > kUpperL0ABytes ? kLowerL0ABytes : kUpperL0ABytes;
    static constexpr uint32_t kL0BUsedBytes =
        kLowerL0BBytes > kUpperL0BBytes ? kLowerL0BBytes : kUpperL0BBytes;
    static constexpr uint32_t kL0ABCapacityBytes = 64 * 1024;

    static constexpr uint32_t kLowerL0CBytes = kLowerRows * K_DIM * sizeof(Element);
    static constexpr uint32_t kUpperL0CBytes = kUpperRows * K_DIM * sizeof(Element);
    static constexpr uint32_t kLowerL0COffset = 0;
    static constexpr uint32_t kUpperL0COffset = kLowerL0COffset + kLowerL0CBytes;
    static constexpr uint32_t kL0CUsedBytes = kUpperL0COffset + kUpperL0CBytes;
    static constexpr uint32_t kL0CCapacityBytes = 256 * 1024;

    static constexpr int32_t kLowerEvent = 0;
    static constexpr int32_t kUpperEvent = 1;

    GM_ADDR chunkMetadata_;
    GM_ADDR workspace_;
    ChunkKdaBwdIntraTilingData tiling_{};
    AscendC::GlobalTensor<int64_t> chunkMetadataGm_;
    Catlass::Arch::CrossCoreFlag vecToCubeReadyFlag_{kVecToCubeReadyFlag};
    Catlass::Arch::CrossCoreFlag cubeToVecReadyFlag_{kCubeToVecReadyFlag};
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_INTRA_ARCH35_CUBE_H
