/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH35_CUBE_H
#define CHUNK_KDA_BWD_PREPARE_ARCH35_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <type_traits>

#include "chunk_kda_bwd_prepare_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace KDA {

template <bool STATE_V_FIRST>
class ChunkKdaBwdPrepareCube {
public:
    __aicore__ inline void Init(
        GM_ADDR aqk, GM_ADDR vNew, GM_ADDR dO, GM_ADDR h,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
        const ChunkKdaBwdPrepareTilingData *tiling)
    {
        aqk_ = aqk;
        vNew_ = vNew;
        dO_ = dO;
        h_ = h;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        dAqk_ = dAqk;
        dv_ = dv;
        dqRaw_ = dqRaw;
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        AscendC::SetMMLayoutTransform(true);
        // Keep one Resource for the complete AIC lifetime.  Resource owns an
        // internal TPipe whose construction/destruction emits synchronization
        // instructions; constructing one in every stage both bloats scalar
        // issue and reinitializes event state between pipeline stages.
        Catlass::Arch::Resource<ArchTag> resource;
        // Resource's internal TPipe initializes and drains M_MTE1 ids 0..2 in
        // its constructor.  Initialize our fixed events only afterwards so
        // the two lifetimes never publish the same flag generation.
        InitEvents();
        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        uint64_t headGeneration = 0;
        uint64_t formulaGeneration = 0;
        for (int64_t workTask = blockIdx; workTask < tiling_->workTaskNum;
             workTask += blockNum) {
            // Head-major task order keeps adjacent chunks of the same four-head
            // window contiguous in the global task space.
            const int64_t headWindow = workTask / tiling_->chunkTaskNum;
            const int64_t chunkTask = workTask - headWindow * tiling_->chunkTaskNum;
            const int64_t headBegin = headWindow * HEADS_PER_WORK_TASK;
            const int64_t headEnd = KdaMin(headBegin + HEADS_PER_WORK_TASK, tiling_->NV);
            ChunkInfo chunk;
            ResolveChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            // Queue the complete four-head window into four independent L1
            // owners before consuming it.  Pair copies keep the lower MTE2
            // instruction count, while the second pair can progress in MTE2
            // during the first pair's MTE1/MMAD/FixPipe work.
            for (int64_t preloadHead = headBegin; preloadHead < headEnd;) {
                const uint32_t preloadOwner =
                    static_cast<uint32_t>(preloadHead - headBegin);
                const uint32_t remaining = static_cast<uint32_t>(headEnd - preloadHead);
                const uint32_t preloadCount =
                    preloadOwner + 1U < OWNER_COUNT && remaining >= 2U ? 2U : 1U;
                const int64_t preloadAqkOffset =
                    TokenOffset(*tiling_, chunk, preloadHead, tiling_->chunkSize);
                const int64_t preloadTokenOffset =
                    TokenOffset(*tiling_, chunk, preloadHead, tiling_->V);
                const int64_t preloadStateOffset =
                    StateOffset(*tiling_, chunk, preloadHead);
                const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
                const uint32_t tokenHeadStride = preloadCount == 2U
                    ? static_cast<uint32_t>(
                          TokenOffset(*tiling_, chunk, preloadHead + 1, tiling_->V) -
                          preloadTokenOffset)
                    : 0U;
                const uint32_t stateHeadStride = preloadCount == 2U
                    ? static_cast<uint32_t>(
                          StateOffset(*tiling_, chunk, preloadHead + 1) -
                          preloadStateOffset)
                    : 0U;
                const uint32_t aqkHeadStride = preloadCount == 2U
                    ? static_cast<uint32_t>(
                          TokenOffset(*tiling_, chunk, preloadHead + 1, tiling_->chunkSize) -
                          preloadAqkOffset)
                    : 0U;

                LoadAStage(
                    resource, preloadOwner, preloadTokenOffset, rows,
                    preloadCount, tokenHeadStride);
                LoadQStage(
                    resource, preloadOwner, preloadStateOffset,
                    preloadCount, stateHeadStride);
                LoadDStage(
                    resource, preloadOwner, preloadAqkOffset, rows,
                    preloadCount, aqkHeadStride);
                preloadHead += preloadCount;
            }

            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner =
                    static_cast<uint32_t>(head - headBegin);
                const bool pairLeader = (owner & 1U) == 0U;
                const uint32_t aivIdx = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t aivSlot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
                const uint32_t aSlot = FormulaSlot(formulaGeneration++);
                RunResident<TileCopyA, TileCopyAToUB, bfloat16_t,
                            false, true, false>(
                    resource, pairLeader ? STAGE_A : STAGE_NO_WAIT,
                    owner, DO_OFFSET, VNEW_OFFSET, rows, rows, KDA_PREPARE_DIM,
                    KDA_PREPARE_CHUNK, aSlot, aSlot, aivIdx,
                    aivSlot * KDA_PREPARE_RAW_BF16_BYTES,
                    KDA_PREPARE_FREE_FLAG_BASE + aivSlot,
                    KDA_PREPARE_READY_FLAG_BASE + aivSlot);

                // A and Q share dO as their left operand. Keep A's dO tile in
                // L0A and let Q consume it directly; only Q's H tile enters
                // the next L0B slot. Q releases the resident L0A tile after
                // its MMAD has consumed it.
                const uint32_t qSlot = FormulaSlot(formulaGeneration++);
                RunResident<TileCopyQ, TileCopyQToUB, float,
                            false, false, true>(
                    resource, pairLeader ? STAGE_Q : STAGE_NO_WAIT,
                    owner, DO_OFFSET, H_OFFSET, rows, KDA_PREPARE_DIM, KDA_PREPARE_DIM,
                    KDA_PREPARE_DIM, qSlot, aSlot, aivIdx,
                    KDA_PREPARE_Q_UB_OFFSET,
                    KDA_PREPARE_Q_FREE_FLAG, KDA_PREPARE_Q_READY_FLAG);

                const uint32_t dSlot = FormulaSlot(formulaGeneration++);
                RunResident<TileCopyD, TileCopyDToUB, bfloat16_t,
                            true, true, true>(
                    resource, pairLeader ? STAGE_D : STAGE_NO_WAIT,
                    owner, AQK_OFFSET, DO_OFFSET, rows, KDA_PREPARE_DIM, rows,
                    KDA_PREPARE_DIM, dSlot, dSlot, aivIdx,
                    KDA_PREPARE_D_UB_OFFSET,
                    KDA_PREPARE_D_FREE_FLAG, KDA_PREPARE_D_READY_FLAG);
            }
        }
        DrainEvents();
        AscendC::SetMMLayoutTransform(false);
    }

private:
    using ArchTag = Catlass::Arch::Ascend950;
    using DT = bfloat16_t;
    using Acc = float;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using TileCopyA = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRM, DT, LayoutCM, float, LayoutRM>;
    using TileCopyAToUB = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRM, DT, LayoutCM, DT, LayoutRM>;
    using TileCopyQRow = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRM, DT, LayoutRM, float, LayoutRM>;
    using TileCopyQCol = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRM, DT, LayoutCM, float, LayoutRM>;
    using TileCopyQ = std::conditional_t<STATE_V_FIRST, TileCopyQRow, TileCopyQCol>;
    using TileCopyQToUBRow = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRM, DT, LayoutRM, float, LayoutRM>;
    using TileCopyQToUBCol = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRM, DT, LayoutCM, float, LayoutRM>;
    using TileCopyQToUB =
        std::conditional_t<STATE_V_FIRST, TileCopyQToUBRow, TileCopyQToUBCol>;
    using QLayoutB = std::conditional_t<STATE_V_FIRST, LayoutRM, LayoutCM>;
    using TileCopyD = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutCM, DT, LayoutRM, bfloat16_t, LayoutRM>;
    using TileCopyDToUB = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutCM, DT, LayoutRM, bfloat16_t, LayoutRM>;

    static constexpr uint32_t OWNER_COUNT = 4;
    static constexpr uint32_t L0_SLOT_COUNT = 2;
    static constexpr int64_t HEADS_PER_WORK_TASK = 4;
    // MTE2_MTE1 supports event ids 0..7. Two adjacent heads share one
    // readiness event per stage because each batched copy completes both
    // owners before publishing READY: 2 pairs * 3 stages = 6 events.
    static constexpr uint32_t STAGE_COUNT = 3;
    static constexpr uint32_t STAGE_A = 0;
    static constexpr uint32_t STAGE_Q = 1;
    static constexpr uint32_t STAGE_D = 2;
    static constexpr uint32_t STAGE_NO_WAIT = STAGE_COUNT;
    static constexpr uint32_t OWNER_BYTES = 72 * 1024;
    static constexpr uint32_t DO_OFFSET = 0;
    static constexpr uint32_t VNEW_OFFSET = 16 * 1024;
    static constexpr uint32_t H_OFFSET = 32 * 1024;
    static constexpr uint32_t AQK_OFFSET = 64 * 1024;
    static constexpr uint32_t L0_TILE_BYTES = 32 * 1024;
    static constexpr uint32_t L0C_TILE_BYTES = 32 * 1024;
    static constexpr uint32_t OWNER_ELEMENTS = OWNER_BYTES / sizeof(DT);

    __aicore__ inline uint32_t FormulaSlot(uint64_t generation) const
    {
        return static_cast<uint32_t>(generation & (L0_SLOT_COUNT - 1U));
    }

    __aicore__ inline uint32_t OwnerBase(uint32_t owner) const
    {
        return owner * OWNER_BYTES;
    }

    __aicore__ inline AscendC::TEventID StageReady(uint32_t owner, uint32_t stage) const
    {
        return static_cast<AscendC::TEventID>((owner >> 1U) * STAGE_COUNT + stage);
    }

    __aicore__ inline void LoadAStage(
        Catlass::Arch::Resource<ArchTag> &resource,
        uint32_t owner, int64_t tokenOffset, uint32_t rows,
        uint32_t headCount, uint32_t tokenHeadStride)
    {
        auto l1DO = resource.l1Buf.template GetBufferByByte<DT>(OwnerBase(owner) + DO_OFFSET);
        auto l1V = resource.l1Buf.template GetBufferByByte<DT>(OwnerBase(owner) + VNEW_OFFSET);
        AscendC::GlobalTensor<DT> gmDO;
        AscendC::GlobalTensor<DT> gmV;
        gmDO.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO_) + tokenOffset);
        gmV.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vNew_) + tokenOffset);
        auto tensorDO = tla::MakeTensor(
            gmDO, tla::MakeLayout<DT, LayoutRM>(rows, KDA_PREPARE_DIM), Catlass::Arch::PositionGM{});
        auto tensorV = tla::MakeTensor(
            gmV, tla::MakeLayout<DT, LayoutCM>(KDA_PREPARE_DIM, rows), Catlass::Arch::PositionGM{});
        auto blockDO = tla::GetTile(
            tensorDO, tla::MakeCoord(0, 0), tla::MakeShape(rows, KDA_PREPARE_DIM));
        auto blockV = tla::GetTile(
            tensorV, tla::MakeCoord(0, 0), tla::MakeShape(KDA_PREPARE_DIM, rows));
        auto tensorL1DO = tla::MakeTensor(
            l1DO, tla::MakeLayout<DT, typename TileCopyA::LayoutTagL1A>(rows, KDA_PREPARE_DIM),
            Catlass::Arch::PositionL1{});
        auto tensorL1V = tla::MakeTensor(
            l1V, tla::MakeLayout<DT, typename TileCopyA::LayoutTagL1B>(KDA_PREPARE_DIM, rows),
            Catlass::Arch::PositionL1{});
        using CopyDO = typename TileCopyA::template CopyGmToL1A<decltype(blockDO)>;
        using CopyV = typename TileCopyA::template CopyGmToL1B<decltype(blockV)>;
        CopyDO copyDO;
        CopyV copyV;
        for (uint32_t i = 0; i < headCount; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ownerCredit_[owner + i]);
        }
        copyDO(tensorL1DO, blockDO, headCount, tokenHeadStride, OWNER_ELEMENTS);
        copyV(tensorL1V, blockV, headCount, tokenHeadStride, OWNER_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            StageReady(owner, STAGE_A));
    }

    __aicore__ inline void LoadQStage(
        Catlass::Arch::Resource<ArchTag> &resource,
        uint32_t owner, int64_t stateOffset,
        uint32_t headCount, uint32_t stateHeadStride)
    {
        auto l1H = resource.l1Buf.template GetBufferByByte<DT>(OwnerBase(owner) + H_OFFSET);
        AscendC::GlobalTensor<DT> gmH;
        gmH.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(h_) + stateOffset);
        auto tensorH = tla::MakeTensor(
            gmH, tla::MakeLayout<DT, QLayoutB>(KDA_PREPARE_DIM, KDA_PREPARE_DIM),
            Catlass::Arch::PositionGM{});
        auto blockH = tla::GetTile(
            tensorH, tla::MakeCoord(0, 0),
            tla::MakeShape(KDA_PREPARE_DIM, KDA_PREPARE_DIM));
        auto tensorL1H = tla::MakeTensor(
            l1H, tla::MakeLayout<DT, typename TileCopyQ::LayoutTagL1B>(
                      KDA_PREPARE_DIM, KDA_PREPARE_DIM),
            Catlass::Arch::PositionL1{});
        using CopyH = typename TileCopyQ::template CopyGmToL1B<decltype(blockH)>;
        CopyH copyH;
        copyH(tensorL1H, blockH, headCount, stateHeadStride, OWNER_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            StageReady(owner, STAGE_Q));
    }

    __aicore__ inline void LoadDStage(
        Catlass::Arch::Resource<ArchTag> &resource,
        uint32_t owner, int64_t aqkOffset, uint32_t rows,
        uint32_t headCount, uint32_t aqkHeadStride)
    {
        auto l1Aqk = resource.l1Buf.template GetBufferByByte<DT>(OwnerBase(owner) + AQK_OFFSET);
        AscendC::GlobalTensor<DT> gmAqk;
        gmAqk.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(aqk_) + aqkOffset);
        auto tensorAqk = tla::MakeTensor(
            gmAqk, tla::MakeLayout<DT, LayoutCM>(KDA_PREPARE_CHUNK, KDA_PREPARE_CHUNK),
            Catlass::Arch::PositionGM{});
        auto blockAqk = tla::GetTile(
            tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(rows, rows));
        auto tensorL1Aqk = tla::MakeTensor(
            l1Aqk, tla::MakeLayout<DT, typename TileCopyD::LayoutTagL1A>(rows, rows),
            Catlass::Arch::PositionL1{});
        using CopyAqk = typename TileCopyD::template CopyGmToL1A<decltype(blockAqk)>;
        CopyAqk copyAqk;
        copyAqk(tensorL1Aqk, blockAqk, headCount, aqkHeadStride, OWNER_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
            StageReady(owner, STAGE_D));
    }

    template <typename TileCopy, typename DirectTileCopy, typename OutT,
              bool RELEASE_OWNER, bool COPY_L0A, bool RELEASE_L0A>
    __aicore__ inline void RunResident(
        Catlass::Arch::Resource<ArchTag> &resource, uint32_t stage,
        uint32_t owner, uint32_t l1AOffset, uint32_t l1BOffset,
        uint32_t m, uint32_t n, uint32_t k,
        uint32_t cStride, uint32_t slot, uint32_t l0ASlot,
        uint32_t aivIdx, uint32_t aivUbOffset,
        uint64_t freeFlag, uint64_t readyFlag)
    {
        using LayoutC = Catlass::layout::RowMajor;
        using LayoutL1A = typename TileCopy::LayoutTagL1A;
        using LayoutL1B = typename TileCopy::LayoutTagL1B;
        using LayoutL0A = typename TileCopy::LayoutTagL0A;
        using LayoutL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, DT, LayoutL1A>;
        auto l1A = resource.l1Buf.template GetBufferByByte<DT>(OwnerBase(owner) + l1AOffset);
        auto l1B = resource.l1Buf.template GetBufferByByte<DT>(OwnerBase(owner) + l1BOffset);
        auto l0A = resource.l0ABuf.template GetBufferByByte<DT>(l0ASlot * L0_TILE_BYTES);
        auto l0B = resource.l0BBuf.template GetBufferByByte<DT>(slot * L0_TILE_BYTES);
        auto l0C = resource.l0CBuf.template GetBufferByByte<Acc>(slot * L0C_TILE_BYTES);
        auto tensorL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<DT, LayoutL1A>(m, k), Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<DT, LayoutL1B>(k, n), Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<DT, LayoutL0A>(m, k), Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<DT, LayoutL0B>(k, n), Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        auto tileL1A = tla::GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = tla::GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = tla::GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = tla::GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        if (stage < STAGE_COUNT) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(StageReady(owner, stage));
        }
        if constexpr (COPY_L0A) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AFree_[l0ASlot]);
            copyL1ToL0A(tileL0A, tileL1A);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BFree_[slot]);
        copyL1ToL0B(tileL0B, tileL1B);
        if constexpr (RELEASE_OWNER) {
            // D is the final consumer of this owner's complete A/Q/D batch.
            // Release the L1 owner immediately after D reaches L0 so the next
            // same-parity head can overlap its MTE2 with this head's MMAD/FIX.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ownerCredit_[owner]);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Ready_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Ready_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cFree_[slot]);
        // L0C ownership is already protected by the explicit M_FIX/FIX_M
        // events below. Keep MMAD/FixPipe unit-flag synchronization disabled
        // so the two mechanisms do not serialize the same dependency twice.
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        if constexpr (RELEASE_L0A) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AFree_[l0ASlot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BFree_[slot]);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cReady_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cReady_[slot]);

        const uint64_t flagOffset =
            static_cast<uint64_t>(aivIdx) * KDA_PREPARE_SUBBLOCK_FLAG_STRIDE;
        AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_FIX>(
            freeFlag + flagOffset);
        auto aivUb = resource.ubBuf.template GetBufferByByte<OutT>(aivUbOffset);
        auto tensorC = tla::MakeTensor(
            aivUb, tla::MakeLayout<OutT, LayoutC>(m, cStride), Catlass::Arch::PositionUB{});
        auto blockC = tla::GetTile(
            tensorC, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        using CopyL0CToUB =
            typename DirectTileCopy::template CopyL0CToDst<decltype(blockC)>;
        // Direct owner-AIV handoff: L0C is reusable as soon as FixPipe has
        // filled UB; the target AIV publishes FREE only after its last use.
        CopyL0CToUB{}(blockC, tileL0C, static_cast<uint8_t>(aivIdx), 0);
        AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_FIX>(
            readyFlag + flagOffset);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cFree_[slot]);
    }

    __aicore__ inline void InitEvents()
    {
        for (uint32_t owner = 0; owner < OWNER_COUNT; ++owner) {
            // Match the mature DHU A5 Cube path: AIC has no TPipe-owned
            // buffers, so use direction-local fixed event IDs and avoid the
            // AIC TPipe destructor reserving/releasing M_MTE1 IDs 0..2.
            ownerCredit_[owner] = static_cast<AscendC::TEventID>(owner);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ownerCredit_[owner]);
        }
        for (uint32_t slot = 0; slot < L0_SLOT_COUNT; ++slot) {
            l0Ready_[slot] = static_cast<AscendC::TEventID>(slot);
            l0AFree_[slot] = static_cast<AscendC::TEventID>(slot);
            l0BFree_[slot] = static_cast<AscendC::TEventID>(L0_SLOT_COUNT + slot);
            l0cReady_[slot] = static_cast<AscendC::TEventID>(slot);
            l0cFree_[slot] = static_cast<AscendC::TEventID>(slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cFree_[slot]);
        }
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t owner = 0; owner < OWNER_COUNT; ++owner) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ownerCredit_[owner]);
        }
        for (uint32_t slot = 0; slot < L0_SLOT_COUNT; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cFree_[slot]);
        }
        for (uint32_t aivIdx = 0; aivIdx < 2; ++aivIdx) {
            for (uint32_t slot = 0; slot < KDA_PREPARE_RAW_SLOT_COUNT; ++slot) {
                const uint64_t flagOffset =
                    static_cast<uint64_t>(aivIdx) * KDA_PREPARE_SUBBLOCK_FLAG_STRIDE;
                AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_FIX>(
                    KDA_PREPARE_FREE_FLAG_BASE + flagOffset + slot);
            }
            const uint64_t flagOffset =
                static_cast<uint64_t>(aivIdx) * KDA_PREPARE_SUBBLOCK_FLAG_STRIDE;
            AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_FIX>(
                KDA_PREPARE_Q_FREE_FLAG + flagOffset);
            AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_FIX>(
                KDA_PREPARE_D_FREE_FLAG + flagOffset);
        }
    }

    GM_ADDR aqk_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR dO_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR dAqk_ = nullptr;
    GM_ADDR dv_ = nullptr;
    GM_ADDR dqRaw_ = nullptr;
    const ChunkKdaBwdPrepareTilingData *tiling_ = nullptr;
    AscendC::TEventID ownerCredit_[OWNER_COUNT];
    AscendC::TEventID l0Ready_[L0_SLOT_COUNT];
    AscendC::TEventID l0AFree_[L0_SLOT_COUNT];
    AscendC::TEventID l0BFree_[L0_SLOT_COUNT];
    AscendC::TEventID l0cReady_[L0_SLOT_COUNT];
    AscendC::TEventID l0cFree_[L0_SLOT_COUNT];
};

} // namespace KDA

#endif
