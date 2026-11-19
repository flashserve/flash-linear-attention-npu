/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH92_CUBE_H
#define CHUNK_GDN_BWD_INTRA_ARCH92_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 9201
#endif

#include <type_traits>
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "chunk_gdn_bwd_intra_common.h"
#include "../chunk_gdn_bwd_intra_common.h"

namespace GDN::Arch92 {

template <typename MainT, uint32_t G>
class ChunkGdnBwdIntraCube {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR dvLocal,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGdnBwdIntraTilingData *tiling)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(k));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(u));
        dvLocalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dvLocal));

        qGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        kGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        wGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        uGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dvLocalGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        mapper_.Init(cuSeqlens, chunkIndices, tiling);
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        AscendC::TPipe pipe;
        InitEvents(pipe);
        const int64_t stride = tiling_->blockDim;
        for (int64_t work = AscendC::GetBlockIdx(); work < tiling_->workCount;) {
            ChunkGdnBwdIntraWorkMeta meta{};
            mapper_.ResolveChunkMajor(work, meta);
            RunStage0(meta);
            if (meta.validTokens == TILE_ROWS) {
                RunStage2<true>(meta);
            } else {
                RunStage2<false>(meta);
            }
            if (stride >= tiling_->workCount - work) {
                break;
            }
            work += stride;
        }
        DrainEvents(pipe);
    }

private:
    using HardEvent = AscendC::HardEvent;
    using ArchTag = Catlass::Arch::Ascend960DT;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using ScoreTile = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, MainT, LayoutRM, MainT, LayoutCM, float, LayoutRM>;
    using OutputTile = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, MainT, LayoutRM, MainT, LayoutRM, MainT, LayoutRM>;
    template <class Tensor>
    using CopyGmToL1A = typename ScoreTile::template CopyGmToL1A<Tensor>;
    template <class Tensor>
    using CopyGmToL1B = typename ScoreTile::template CopyGmToL1B<Tensor>;
    template <class Tensor>
    using CopyL0CToUb = typename ScoreTile::template CopyL0CToDst<Tensor>;
    template <class Tensor>
    using CopyL0CToGm = typename OutputTile::template CopyL0CToDst<Tensor>;

    struct SlotEvents {
        AscendC::TEventID qkReady;
        AscendC::TEventID qkFree;
        AscendC::TEventID operandsReady;
        AscendC::TEventID operandsFree;
        AscendC::TEventID resultReady;
        AscendC::TEventID resultFree;
    };

    __aicore__ inline void InitEvents(AscendC::TPipe &pipe)
    {
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            auto &events = events_[slot];
            events.qkReady = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
            events.qkFree = pipe.AllocEventID<HardEvent::MTE1_MTE2>();
            events.operandsReady = pipe.AllocEventID<HardEvent::MTE1_M>();
            events.operandsFree = pipe.AllocEventID<HardEvent::M_MTE1>();
            events.resultReady = pipe.AllocEventID<HardEvent::M_FIX>();
            events.resultFree = pipe.AllocEventID<HardEvent::FIX_M>();
            AscendC::SetFlag<HardEvent::M_MTE1>(events.operandsFree);
            AscendC::SetFlag<HardEvent::FIX_M>(events.resultFree);
        }
    }

    __aicore__ inline void DrainEvents(AscendC::TPipe &pipe)
    {
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            const auto &events = events_[slot];
            if (scoreSlotUsed_[slot]) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(SCORE_FLAG_BASE + slot);
                AscendC::WaitFlag<HardEvent::MTE1_MTE2>(events.qkFree);
            }
            AscendC::WaitFlag<HardEvent::M_MTE1>(events.operandsFree);
            AscendC::WaitFlag<HardEvent::FIX_M>(events.resultFree);
            pipe.ReleaseEventID<HardEvent::MTE2_MTE1>(events.qkReady);
            pipe.ReleaseEventID<HardEvent::MTE1_MTE2>(events.qkFree);
            pipe.ReleaseEventID<HardEvent::MTE1_M>(events.operandsReady);
            pipe.ReleaseEventID<HardEvent::M_MTE1>(events.operandsFree);
            pipe.ReleaseEventID<HardEvent::M_FIX>(events.resultReady);
            pipe.ReleaseEventID<HardEvent::FIX_M>(events.resultFree);
        }
    }

    __aicore__ inline void RunStage0(const ChunkGdnBwdIntraWorkMeta &meta)
    {
        for (uint32_t r = 0; r < meta.validHeads; ++r) {
            if (ChunkGdnBwdIntraScoreLeader<G>(meta.hvBegin, r, tiling_->groupSize)) {
                PrefetchQk(meta, r);
            }
        }

        const int64_t group = ChunkGdnBwdIntraGroup<G>(tiling_->groupSize);
        const int64_t firstHk = meta.hvBegin / group;
        for (uint32_t r = 0; r < meta.validHeads; ++r) {
            if (ChunkGdnBwdIntraScoreLeader<G>(meta.hvBegin, r, tiling_->groupSize)) {
                const uint32_t leaderIndex = static_cast<uint32_t>((meta.hvBegin + r) / group - firstHk);
                ComputeScore(r, leaderIndex % SLOT_COUNT);
                scoreSlotUsed_[r] = true;
            }
        }
    }

    __aicore__ inline void PrefetchQk(const ChunkGdnBwdIntraWorkMeta &meta, uint32_t slot)
    {
        if (scoreSlotUsed_[slot]) {
            AscendC::WaitFlag<HardEvent::MTE1_MTE2>(events_[slot].qkFree);
        }
        const int64_t hk = (meta.hvBegin + slot) / ChunkGdnBwdIntraGroup<G>(tiling_->groupSize);
        const uint64_t offset = ((static_cast<uint64_t>(meta.batch) * tiling_->qkHeads + hk) *
                                 tiling_->seqlen + meta.tokenStart) * HEAD_DIM;
        auto l1K = L1(L1_K_BASE + slot * HEAD_BYTES);
        auto l1Q = L1(L1_Q_BASE + slot * HEAD_BYTES);
        if (meta.validTokens < TILE_ROWS) {
            const AscendC::InitConstValueParams<MainT> clear(
                1, HEAD_BYTES / BLOCK_BYTES, 0, static_cast<MainT>(0));
            AscendC::InitConstValue(l1K, clear);
            AscendC::InitConstValue(l1Q, clear);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }

        auto gmK = tla::MakeTensor(
            kGm_[offset], tla::MakeLayout<MainT, LayoutRM>(meta.validTokens, HEAD_DIM),
            Catlass::Arch::PositionGM{});
        auto gmQ = tla::MakeTensor(
            qGm_[offset], tla::MakeLayout<MainT, LayoutCM>(HEAD_DIM, meta.validTokens),
            Catlass::Arch::PositionGM{});
        auto tensorL1K = tla::MakeTensor(
            l1K, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL1A>(TILE_ROWS, HEAD_DIM),
            Catlass::Arch::PositionL1{});
        auto tensorL1Q = tla::MakeTensor(
            l1Q, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL1B>(HEAD_DIM, TILE_ROWS),
            Catlass::Arch::PositionL1{});
        auto l1KTile = tla::GetTile(
            tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(meta.validTokens, HEAD_DIM));
        auto l1QTile = tla::GetTile(
            tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(HEAD_DIM, meta.validTokens));
        CopyGmToL1A<decltype(gmK)>{}(l1KTile, gmK);
        CopyGmToL1B<decltype(gmQ)>{}(l1QTile, gmQ);
        AscendC::SetFlag<HardEvent::MTE2_MTE1>(events_[slot].qkReady);
    }

    __aicore__ inline void ComputeScore(uint32_t leader, uint32_t l0Slot)
    {
        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(events_[leader].qkReady);
        // Q uses a column-major logical view of the row-major GM data for K @ Q^T.
        Matmul<true>(l0Slot, L1(L1_K_BASE + leader * HEAD_BYTES),
                     L1(L1_Q_BASE + leader * HEAD_BYTES), TILE_ROWS);

        const auto &events = events_[l0Slot];
        AscendC::WaitFlag<HardEvent::M_FIX>(events.resultReady);
        const uint16_t scoreFlag = SCORE_FLAG_BASE + leader;
        if (scoreSlotUsed_[leader]) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(scoreFlag);
        }
        auto score = resource_.ubBuf.template GetBufferByByte<float>(
            SCORE_UB_BASE + leader * SCORE_SLOT_BYTES);
        auto scoreTensor = tla::MakeTensor(
            score, tla::MakeLayout<float, LayoutRM>(TILE_ROWS, TILE_ROWS), Catlass::Arch::PositionUB{});
        auto scoreL0C = tla::MakeTensor(
            L0C(l0Slot), tla::MakeLayoutL0C(TILE_ROWS, TILE_ROWS), Catlass::Arch::PositionL0C{});
        CopyL0CToUb<decltype(scoreTensor)>{}(scoreTensor, scoreL0C, uint8_t{0}, uint8_t{0});
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(scoreFlag);
        AscendC::SetFlag<HardEvent::FIX_M>(events.resultFree);
    }

    template <bool FULL_TOKENS>
    __aicore__ inline void RunStage2(const ChunkGdnBwdIntraWorkMeta &meta)
    {
        const uint64_t headStride = static_cast<uint64_t>(tiling_->seqlen) * HEAD_DIM;
        const uint64_t outputBase =
            (static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + meta.hvBegin) * headStride +
            static_cast<uint64_t>(meta.tokenStart) * HEAD_DIM;
        const uint32_t rows = FULL_TOKENS ? TILE_ROWS : static_cast<uint32_t>(meta.validTokens);
        uint32_t issued = 0;
        for (uint32_t r = 0; r < meta.validHeads; ++r) {
            const uint32_t slot = r % SLOT_COUNT;
            const uint16_t recordFlag = RECORD_FLAG_BASE + slot;
            const uint32_t record = L1_RECORD_BASE + slot * RECORD_SLOT_BYTES;
            const uint32_t leader = static_cast<uint32_t>(
                ChunkGdnBwdIntraLeaderR<G>(meta.hvBegin, r, tiling_->groupSize));
            const uint64_t outputOffset = outputBase + r * headStride;
            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(recordFlag);

            const uint32_t dvSlot = issued++ % SLOT_COUNT;
            Matmul<false>(dvSlot, L1(record + RECORD_D_OFFSET), L1(L1_DO_BASE + slot * HEAD_BYTES), rows);
            StoreOutput(dvSlot, dvLocalGm_[outputOffset], rows);

            const uint32_t wSlot = issued++ % SLOT_COUNT;
            Matmul<false>(wSlot, L1(record + RECORD_A_BG_OFFSET), L1(L1_K_BASE + leader * HEAD_BYTES), rows);
            if (r + 1 == meta.validHeads ||
                ChunkGdnBwdIntraScoreLeader<G>(meta.hvBegin, r + 1, tiling_->groupSize)) {
                AscendC::SetFlag<HardEvent::MTE1_MTE2>(events_[leader].qkFree);
            }
            StoreOutput(wSlot, wGm_[outputOffset], rows);

            const uint32_t uSlot = issued++ % SLOT_COUNT;
            Matmul<false>(uSlot, L1(record + RECORD_A_BETA_OFFSET), L1(L1_V_BASE + slot * HEAD_BYTES), rows);
            // The U load is the last MTE1 consumer of this shared record slot.
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(recordFlag);
            StoreOutput(uSlot, uGm_[outputOffset], rows);
        }
    }

    template <bool SCORE>
    __aicore__ inline void Matmul(uint32_t slot, AscendC::LocalTensor<MainT> left,
                                  AscendC::LocalTensor<MainT> right, uint32_t rows)
    {
        using Tile = std::conditional_t<SCORE, ScoreTile, OutputTile>;
        using CopyL1ToL0A = typename Tile::CopyL1ToL0A;
        using CopyL1ToL0B = typename Tile::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, MainT, typename Tile::LayoutTagL1A>;
        constexpr uint32_t l1K = SCORE ? HEAD_DIM : TILE_ROWS;
        constexpr uint32_t n = SCORE ? TILE_ROWS : HEAD_DIM;
        const uint32_t k = SCORE ? HEAD_DIM : rows;

        // L1 retains the fixed NZ stride. The C2 L0 tiles use the actual tail shape.
        auto tensorL1A = tla::MakeTensor(
            left, tla::MakeLayout<MainT, typename Tile::LayoutTagL1A>(TILE_ROWS, l1K),
            Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(
            right, tla::MakeLayout<MainT, typename Tile::LayoutTagL1B>(l1K, n),
            Catlass::Arch::PositionL1{});
        auto tileL1A = tla::GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(rows, k));
        auto tileL1B = tla::GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tensorL0A = tla::MakeTensor(
            resource_.l0ABuf.template GetBufferByByte<MainT>(slot * L0_AB_SLOT_BYTES),
            tla::MakeLayout<MainT, typename Tile::LayoutTagL0A>(rows, k), Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            resource_.l0BBuf.template GetBufferByByte<MainT>(slot * L0_AB_SLOT_BYTES),
            tla::MakeLayout<MainT, typename Tile::LayoutTagL0B>(k, n), Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            L0C(slot), tla::MakeLayoutL0C(rows, n), Catlass::Arch::PositionL0C{});

        const auto &events = events_[slot];
        AscendC::WaitFlag<HardEvent::M_MTE1>(events.operandsFree);
        CopyL1ToL0A{}(tensorL0A, tileL1A);
        CopyL1ToL0B{}(tensorL0B, tileL1B);
        AscendC::SetFlag<HardEvent::MTE1_M>(events.operandsReady);

        AscendC::WaitFlag<HardEvent::MTE1_M>(events.operandsReady);
        AscendC::WaitFlag<HardEvent::FIX_M>(events.resultFree);
        TileMmad{}(tensorL0C, tensorL0A, tensorL0B, true, uint8_t{0});
        AscendC::SetFlag<HardEvent::M_MTE1>(events.operandsFree);
        AscendC::SetFlag<HardEvent::M_FIX>(events.resultReady);
    }

    __aicore__ inline void StoreOutput(uint32_t slot, AscendC::GlobalTensor<MainT> out,
                                       uint32_t validRows)
    {
        auto tensorL0C = tla::MakeTensor(
            L0C(slot), tla::MakeLayoutL0C(validRows, HEAD_DIM), Catlass::Arch::PositionL0C{});
        auto output = tla::MakeTensor(
            out, tla::MakeLayout<MainT, LayoutRM>(validRows, HEAD_DIM), Catlass::Arch::PositionGM{});
        const auto &events = events_[slot];
        AscendC::WaitFlag<HardEvent::M_FIX>(events.resultReady);
        CopyL0CToGm<decltype(output)>{}(output, tensorL0C, uint8_t{0});
        AscendC::SetFlag<HardEvent::FIX_M>(events.resultFree);
    }

    __aicore__ inline AscendC::LocalTensor<MainT> L1(uint32_t byteOffset)
    {
        return resource_.l1Buf.template GetBufferByByte<MainT>(byteOffset);
    }

    __aicore__ inline AscendC::LocalTensor<float> L0C(uint32_t slot)
    {
        return resource_.l0CBuf.template GetBufferByByte<float>(slot * L0_C_SLOT_BYTES);
    }

    Catlass::Arch::Resource<ArchTag> resource_;
    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;
    AscendC::GlobalTensor<MainT> qGm_;
    AscendC::GlobalTensor<MainT> kGm_;
    AscendC::GlobalTensor<MainT> wGm_;
    AscendC::GlobalTensor<MainT> uGm_;
    AscendC::GlobalTensor<MainT> dvLocalGm_;
    SlotEvents events_[SLOT_COUNT];
    bool scoreSlotUsed_[SLOT_COUNT] = {};
};

} // namespace GDN::Arch92

#endif // CHUNK_GDN_BWD_INTRA_ARCH92_CUBE_H
