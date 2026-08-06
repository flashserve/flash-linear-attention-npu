/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_CUBE_H
#define CHUNK_KDA_BWD_INTRA_CUBE_H

#define CATLASS_ARCH 2201
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_kda_bwd_intra_struct.h"
#include "chunk_kda_bwd_intra_common.h"

namespace KDA {

template <uint32_t K_DIM, uint32_t CHUNK_SIZE, bool SAFE_GATE, bool VARLEN_TND>
class ChunkKdaBwdIntraCubeProcess {
public:
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
        using ArchTag = Catlass::Arch::AtlasA2;
        using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true, false>;
        using Element = float;
        // Keep L0B ping-pong within capacity for the FP32 K=256 branch.
        static constexpr uint32_t kL0ReductionTile = K_DIM == 256 ? 32 : 64;
        using LowerL1Shape = tla::Shape<tla::Int<32>, tla::Int<K_DIM>, tla::Int<CHUNK_SIZE>>;
        using LowerL0Shape =
            tla::Shape<tla::Int<32>, tla::Int<K_DIM>, tla::Int<kL0ReductionTile>>;
        using UpperL1Shape = tla::Shape<tla::Int<16>, tla::Int<K_DIM>, tla::Int<2 * CHUNK_SIZE>>;
        using UpperL0Shape =
            tla::Shape<tla::Int<16>, tla::Int<K_DIM>, tla::Int<kL0ReductionTile>>;

        using RowMajor = Catlass::layout::RowMajor;
        using ColumnMajor = Catlass::layout::ColumnMajor;
        using LowerCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, RowMajor, Element, RowMajor, Element, RowMajor>;
        using UpperCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, Element, ColumnMajor, Element, RowMajor, Element, RowMajor>;
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
            const uint32_t rowBlockCount = (validLen + kRowBlock - 1) / kRowBlock;
            for (uint32_t rowBlock = 0; rowBlock < rowBlockCount; ++rowBlock) {
                const uint32_t rowStart = rowBlock * kRowBlock;
                const uint32_t validRows =
                    rowStart + kRowBlock <= validLen ? kRowBlock : validLen - rowStart;
                const uint32_t prefix = rowStart + validRows;
                const uint32_t future = validLen - rowStart;
                for (uint32_t headInWindow = 0; headInWindow < headCount; ++headInWindow) {
                    AscendC::CrossCoreWaitFlag(kVecToCubeReadyFlag);
                    const uint32_t slot = WorkspaceSlot(windowIdx, headInWindow);
                    const uint64_t slotBase =
                        static_cast<uint64_t>(coreIdx) * tiling_.workspaceCoreSize +
                        static_cast<uint64_t>(slot) * tiling_.workspaceSlotSize;
                    RunLower<LowerMmad, RowMajor>(resource, slotBase, prefix);
                    RunUpper<UpperMmad, RowMajor, ColumnMajor>(resource, slotBase, future);
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kCubeToVecReadyFlag);
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

private:
    template <typename LowerMmad, typename RowMajor>
    __aicore__ inline void RunLower(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        uint64_t slotBase, uint32_t prefix)
    {
        AscendC::GlobalTensor<float> a;
        AscendC::GlobalTensor<float> b;
        AscendC::GlobalTensor<float> c;
        a.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.aLowerOffset));
        b.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.bLowerOffset));
        c.SetGlobalBuffer((__gm__ float *)(
            workspace_ + slotBase + tiling_.resultRegionOffset + tiling_.resultDqOffset));

        const uint32_t m = 2 * kRowBlock;
        auto layoutA = tla::MakeLayout<float, RowMajor>(m, prefix);
        auto layoutB = tla::MakeLayout<float, RowMajor>(prefix, K_DIM);
        auto layoutC = tla::MakeLayout<float, RowMajor>(m, K_DIM);
        auto tensorA = tla::MakeTensor(a, layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(b, layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(c, layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, K_DIM, prefix};
        LowerMmad mmad(resource);
        mmad(tensorA, tensorB, tensorC, shape);
    }

    template <typename UpperMmad, typename RowMajor, typename ColumnMajor>
    __aicore__ inline void RunUpper(
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource,
        uint64_t slotBase, uint32_t future)
    {
        AscendC::GlobalTensor<float> a;
        AscendC::GlobalTensor<float> b;
        AscendC::GlobalTensor<float> c;
        a.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.aUpperOffset));
        b.SetGlobalBuffer((__gm__ float *)(workspace_ + slotBase + tiling_.bUpperOffset));
        c.SetGlobalBuffer((__gm__ float *)(
            workspace_ + slotBase + tiling_.resultRegionOffset + tiling_.resultDkUpperOffset));

        const uint32_t reduction = 2 * future;
        // A is physically [2*future, validRows] RowMajor.  A ColumnMajor
        // [validRows, 2*future] view performs the transpose in Cube, so AIV
        // never scatters a transposed DA6_T-like matrix.
        auto layoutA = tla::MakeLayout<float, ColumnMajor>(kRowBlock, reduction);
        auto layoutB = tla::MakeLayout<float, RowMajor>(reduction, K_DIM);
        auto layoutC = tla::MakeLayout<float, RowMajor>(kRowBlock, K_DIM);
        auto tensorA = tla::MakeTensor(a, layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(b, layoutB, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(c, layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{kRowBlock, K_DIM, reduction};
        UpperMmad mmad(resource);
        mmad(tensorA, tensorB, tensorC, shape);
    }

    GM_ADDR chunkMetadata_;
    GM_ADDR workspace_;
    ChunkKdaBwdIntraTilingData tiling_{};
    AscendC::GlobalTensor<int64_t> chunkMetadataGm_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_INTRA_CUBE_H
