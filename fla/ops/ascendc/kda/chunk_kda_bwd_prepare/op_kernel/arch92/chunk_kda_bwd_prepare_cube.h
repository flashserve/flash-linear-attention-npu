/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH92_CUBE_H
#define CHUNK_KDA_BWD_PREPARE_ARCH92_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 9201
#endif

#include <type_traits>
#include "chunk_kda_bwd_prepare_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace KDA::Arch92 {

template <bool STATE_V_FIRST>
class ChunkKdaBwdPrepareCube {
public:
    __aicore__ inline void Init(
        GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
        GM_ADDR cu, GM_ADDR indices, const ChunkKdaBwdPrepareTilingData *tiling)
    {
        tiling_ = tiling;
        cu_ = cu;
        indices_ = indices;
        aqk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aqk));
        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(vNew));
        dO_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dO));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
    }

    __aicore__ inline void Process()
    {
        // As in A5, keep one Resource for the complete AIC task loop. Its
        // internal TPipe is destroyed before our fixed event IDs are seeded.
        Catlass::Arch::Resource<ArchTag> resource;
        InitEvents();
        WorkIterator iterator(tiling_, cu_, indices_);
        WorkInfo pending[kSlots];
        // Only submit actual tasks. There is no all-inputs-ready barrier.
        for (uint32_t s = 0; s < kSlots; ++s) {
            if (iterator.Next(pending[s])) Load(resource, pending[s]);
        }
        uint32_t owner = 0;
        while (pending[owner].chunk.valid) {
            Compute(resource, pending[owner]);
            // D has already handed the input owner back to MTE2. Its MMAD,
            // Fixpipe and the AIV GM writes need not finish before this load.
            if (iterator.Next(pending[owner])) Load(resource, pending[owner]);
            owner = (owner + 1) % kSlots;
        }
        DrainEvents();
    }

private:
    // DA retains dO in L0A; DQ reuses and releases it; DV releases the L1 owner.
    enum class Formula { DA, DQ, DV };

    using ArchTag = Catlass::Arch::Ascend960DT;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using TileCopyA = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, bfloat16_t, LayoutRM, bfloat16_t, LayoutCM, float, LayoutRM>;
    using TileCopyQRow = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, bfloat16_t, LayoutRM, bfloat16_t, LayoutRM, float, LayoutRM>;
    using TileCopyQCol = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, bfloat16_t, LayoutRM, bfloat16_t, LayoutCM, float, LayoutRM>;
    using TileCopyQ = std::conditional_t<STATE_V_FIRST, TileCopyQRow, TileCopyQCol>;
    using QLayoutB = std::conditional_t<STATE_V_FIRST, LayoutRM, LayoutCM>;
    using TileCopyD = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, bfloat16_t, LayoutCM, bfloat16_t, LayoutRM, bfloat16_t, LayoutRM>;

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> Owner(
        Catlass::Arch::Resource<ArchTag> &resource, uint32_t slot, uint32_t offset)
    {
        return resource.l1Buf.template GetBufferByByte<bfloat16_t>(slot * kOwnerBytes + offset);
    }

    template <typename LayoutGM, typename LayoutL1>
    __aicore__ inline void CopyInput(
        AscendC::LocalTensor<bfloat16_t> dst, AscendC::GlobalTensor<bfloat16_t> src,
        uint32_t rows, uint32_t cols, uint32_t gmRows, uint32_t gmCols,
        uint32_t l1Rows, uint32_t l1Cols)
    {
        if (rows != l1Rows || cols != l1Cols) {
            // Preserve the complete physical padding: D reduces along a tail
            // token axis, so an old owner's padding must never participate.
            AscendC::InitConstValueParams<uint16_t> zero(
                1, static_cast<uint16_t>(l1Rows * l1Cols / 16), 0, 0);
            AscendC::Fill(dst.template ReinterpretCast<uint16_t>(), zero);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        auto tensorGM = tla::MakeTensor(
            src, tla::MakeLayout<bfloat16_t, LayoutGM>(gmRows, gmCols), Catlass::Arch::PositionGM{});
        auto blockGM = tla::GetTile(tensorGM, tla::MakeCoord(0, 0), tla::MakeShape(rows, cols));
        // Build the layout from the physical owner shape, then take the valid
        // tile. Using rows here would compact the NZ pitch on partial chunks.
        auto tensorL1 = tla::MakeTensor(
            dst, tla::MakeLayout<bfloat16_t, LayoutL1>(l1Rows, l1Cols), Catlass::Arch::PositionL1{});
        using CopyGmToL1 = Catlass::Gemm::Tile::TileCopyTla<
            ArchTag, decltype(blockGM), decltype(tensorL1)>;
        CopyGmToL1{}(tensorL1, blockGM);
    }

    __aicore__ inline void Load(Catlass::Arch::Resource<ArchTag> &resource, const WorkInfo &work)
    {
        const auto &chunk = work.chunk;
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        const uint32_t s = work.slot;
        const int64_t tokenOffset = TokenOutputOffset(*tiling_, chunk, work.head, kDim);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ownerFree_[s]);
        CopyInput<LayoutRM, typename TileCopyA::LayoutTagL1A>(
            Owner(resource, s, kDoOffset), dO_[tokenOffset], rows, kDim, kRows, kDim, kRows, kDim);
        // Column-major [128,M] describes vNew[M,128].T without a GM transpose.
        CopyInput<LayoutCM, typename TileCopyA::LayoutTagL1B>(
            Owner(resource, s, kVNewOffset), vNew_[tokenOffset], kDim, rows, kDim, kRows, kDim, kRows);
        CopyInput<QLayoutB, typename TileCopyQ::LayoutTagL1B>(
            Owner(resource, s, kHOffset), h_[StateOffset(*tiling_, chunk, work.head)],
            kDim, kDim, kDim, kDim, kDim, kDim);
        // Keep the entire valid square, including the upper triangle. Its
        // column-major view is aqk_valid.T, used only by D.
        CopyInput<LayoutCM, typename TileCopyD::LayoutTagL1A>(
            Owner(resource, s, kAqkOffset), aqk_[TokenOutputOffset(*tiling_, chunk, work.head, kRows)],
            rows, rows, kRows, kRows, kRows, kRows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(inputReady_[s]);
    }

    template <typename TileCopy, typename OutT, Formula FORMULA>
    __aicore__ inline void ComputeAndCopyToUb(
        Catlass::Arch::Resource<ArchTag> &resource, const WorkInfo &work,
        uint32_t slot, uint32_t aSlot,
        uint32_t offset, uint32_t cStride, uint16_t ready)
    {
        using LayoutL1A = typename TileCopy::LayoutTagL1A;
        using LayoutL1B = typename TileCopy::LayoutTagL1B;
        using LayoutL0A = typename TileCopy::LayoutTagL0A;
        using LayoutL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, bfloat16_t, LayoutL1A>;
        const uint32_t rows = static_cast<uint32_t>(work.chunk.validRows);
        const uint32_t m = Align16(rows);
        const uint32_t n = FORMULA == Formula::DA ? m : kDim;
        const uint32_t k = FORMULA == Formula::DV ? m : kDim;
        constexpr uint32_t l1ACols = FORMULA == Formula::DV ? kRows : kDim;
        constexpr uint32_t l1BRows = FORMULA == Formula::DV ? kRows : kDim;
        constexpr uint32_t l1BCols = FORMULA == Formula::DA ? kRows : kDim;
        constexpr uint32_t aOffset = FORMULA == Formula::DV ? kAqkOffset : kDoOffset;
        constexpr uint32_t bOffset = FORMULA == Formula::DA ? kVNewOffset :
                                    (FORMULA == Formula::DQ ? kHOffset : kDoOffset);
        auto a = resource.l0ABuf.template GetBufferByByte<bfloat16_t>(aSlot * kL0SlotBytes);
        auto b = resource.l0BBuf.template GetBufferByByte<bfloat16_t>(slot * kL0SlotBytes);
        auto c = resource.l0CBuf.template GetBufferByByte<float>(slot * kL0SlotBytes);
        auto tensorL1A = tla::MakeTensor(
            Owner(resource, work.slot, aOffset), tla::MakeLayout<bfloat16_t, LayoutL1A>(kRows, l1ACols),
            Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(
            Owner(resource, work.slot, bOffset), tla::MakeLayout<bfloat16_t, LayoutL1B>(l1BRows, l1BCols),
            Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(
            a, tla::MakeLayout<bfloat16_t, LayoutL0A>(m, k), Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            b, tla::MakeLayout<bfloat16_t, LayoutL0B>(k, n), Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            c, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        auto tileL1A = tla::GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = tla::GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = tla::GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = tla::GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        if constexpr (FORMULA != Formula::DQ) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(aFree_[aSlot]);
            CopyL1ToL0A{}(tileL0A, tileL1A);
        }
        // Q retains A's dO. Waiting for aFree here would consume a credit
        // which A deliberately holds until Q has finished reading it.
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bFree_[slot]);
        CopyL1ToL0B{}(tileL0B, tileL1B);
        if constexpr (FORMULA == Formula::DV) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ownerFree_[work.slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cFree_[slot]);
        TileMmad{}(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        if constexpr (FORMULA != Formula::DA) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(aFree_[aSlot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bFree_[slot]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
        if constexpr (FORMULA == Formula::DA) {
            AscendC::CrossCoreWaitFlag<kCrossMode, PIPE_FIX>(kUbFree + work.slot);
        }
        // The AIV arena starts at byte zero. Only the result groups are
        // written here; A_out and the shared mask belong to the vector kernel.
        auto dst = resource.ubBuf.template GetBufferByByte<OutT>(work.slot * kGroupBytes + offset);
        auto tensorC = tla::MakeTensor(
            dst, tla::MakeLayout<OutT, LayoutRM>(kRows, cStride), Catlass::Arch::PositionUB{});
        auto blockC = tla::GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(rows, n));
        using CopyL0CToUB = typename TileCopy::template CopyL0CToDst<decltype(blockC)>;
        // NO_SPLIT, target subBlockId=0, unitFlag=0: a single AIV owns all rows.
        CopyL0CToUB{}(blockC, tileL0C, false, 0);
        AscendC::CrossCoreSetFlag<kCrossMode, PIPE_FIX>(ready + work.slot);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cFree_[slot]);
    }

    __aicore__ inline void Compute(Catlass::Arch::Resource<ArchTag> &resource, const WorkInfo &work)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(inputReady_[work.slot]);
        const uint32_t aSlot = static_cast<uint32_t>(matmulStep_++ % kL0Slots);
        ComputeAndCopyToUb<TileCopyA, float, Formula::DA>(
            resource, work, aSlot, aSlot, kRawOffset, kRows, kAReady);
        const uint32_t qSlot = static_cast<uint32_t>(matmulStep_++ % kL0Slots);
        ComputeAndCopyToUb<TileCopyQ, float, Formula::DQ>(
            resource, work, qSlot, aSlot, kQOffset, kDim, kQReady);
        const uint32_t dSlot = static_cast<uint32_t>(matmulStep_++ % kL0Slots);
        ComputeAndCopyToUb<TileCopyD, bfloat16_t, Formula::DV>(
            resource, work, dSlot, dSlot, kDOffset, kDim, kDReady);
    }

    __aicore__ inline void InitEvents()
    {
        // IDs are local to each HardEvent direction. Resource has already
        // drained the synchronization inserted by its internal TPipe.
        for (uint32_t s = 0; s < kSlots; ++s) {
            ownerFree_[s] = static_cast<AscendC::TEventID>(s);
            inputReady_[s] = static_cast<AscendC::TEventID>(s);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ownerFree_[s]);
        }
        for (uint32_t s = 0; s < kL0Slots; ++s) {
            aFree_[s] = static_cast<AscendC::TEventID>(s);
            bFree_[s] = static_cast<AscendC::TEventID>(kL0Slots + s);
            operandReady_[s] = static_cast<AscendC::TEventID>(s);
            resultReady_[s] = static_cast<AscendC::TEventID>(s);
            cFree_[s] = static_cast<AscendC::TEventID>(s);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(aFree_[s]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bFree_[s]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cFree_[s]);
        }
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t s = 0; s < kSlots; ++s) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ownerFree_[s]);
        }
        for (uint32_t s = 0; s < kL0Slots; ++s) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(aFree_[s]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bFree_[s]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cFree_[s]);
        }
        // All ready events are consumed by their corresponding operations.
        for (uint16_t s = 0; s < kSlots; ++s) {
            AscendC::CrossCoreWaitFlag<kCrossMode, PIPE_FIX>(kUbFree + s);
        }
    }

    const ChunkKdaBwdPrepareTilingData *tiling_ = nullptr;
    GM_ADDR cu_ = nullptr;
    GM_ADDR indices_ = nullptr;
    uint64_t matmulStep_ = 0;
    AscendC::GlobalTensor<bfloat16_t> aqk_, vNew_, dO_, h_;
    AscendC::TEventID ownerFree_[kSlots], inputReady_[kSlots];
    AscendC::TEventID aFree_[kL0Slots], bFree_[kL0Slots];
    AscendC::TEventID operandReady_[kL0Slots], resultReady_[kL0Slots], cFree_[kL0Slots];
};

} // namespace KDA::Arch92
#endif
