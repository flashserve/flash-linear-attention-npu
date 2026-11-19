/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH92_CUBE_H
#define CHUNK_GDN_BWD_INTRA_ARCH92_CUBE_H

#include <type_traits>

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
            RunStage2(meta);
            if (stride >= tiling_->workCount - work) {
                break;
            }
            work += stride;
        }
        DrainEvents(pipe);
    }

private:
    using HardEvent = AscendC::HardEvent;

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

        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = static_cast<uint16_t>(meta.validTokens);
        copy.dValue = HEAD_DIM;
        copy.srcDValue = HEAD_DIM;
        copy.dstNzC0Stride = TILE_ROWS;
        copy.dstNzNStride = 1;
        AscendC::DataCopy(l1K, kGm_[offset], copy);
        AscendC::DataCopy(l1Q, qGm_[offset], copy);
        AscendC::SetFlag<HardEvent::MTE2_MTE1>(events_[slot].qkReady);
    }

    __aicore__ inline void ComputeScore(uint32_t leader, uint32_t l0Slot)
    {
        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(events_[leader].qkReady);
        // Both operands are stored as [64,128] NZ. The non-transposed L0B load
        // presents Q as the right operand in K @ Q^T.
        Matmul<true>(l0Slot, L1(L1_K_BASE + leader * HEAD_BYTES),
                     L1(L1_Q_BASE + leader * HEAD_BYTES));

        const auto &events = events_[l0Slot];
        AscendC::WaitFlag<HardEvent::M_FIX>(events.resultReady);
        const uint16_t scoreFlag = SCORE_FLAG_BASE + leader;
        if (scoreSlotUsed_[leader]) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(scoreFlag);
        }
        AscendC::LocalTensor<float> score(AscendC::TPosition::VECIN,
            SCORE_UB_BASE + leader * SCORE_SLOT_BYTES, SCORE_SLOT_BYTES / sizeof(float));
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> params{};
        params.nSize = TILE_ROWS;
        params.mSize = TILE_ROWS;
        params.srcStride = TILE_ROWS;
        params.dstStride = TILE_ROWS;
        params.quantPre = QuantMode_t::NoQuant;
        params.dualDstCtl = 0;
        params.subBlockId = 0;
        params.params.ndNum = 1;
        AscendC::Fixpipe<float, float, SCORE_FIXPIPE_CONFIG>(score, L0C(l0Slot), params);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(scoreFlag);
        AscendC::SetFlag<HardEvent::FIX_M>(events.resultFree);
    }

    __aicore__ inline void RunStage2(const ChunkGdnBwdIntraWorkMeta &meta)
    {
        const uint64_t headStride = static_cast<uint64_t>(tiling_->seqlen) * HEAD_DIM;
        const uint64_t outputBase =
            (static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + meta.hvBegin) * headStride +
            static_cast<uint64_t>(meta.tokenStart) * HEAD_DIM;
        const uint16_t rows = static_cast<uint16_t>(meta.validTokens);
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
            Matmul<false>(dvSlot, L1(record + RECORD_D_OFFSET), L1(L1_DO_BASE + slot * HEAD_BYTES));
            StoreOutput(dvSlot, dvLocalGm_[outputOffset], rows);

            const uint32_t wSlot = issued++ % SLOT_COUNT;
            Matmul<false>(wSlot, L1(record + RECORD_A_BG_OFFSET), L1(L1_K_BASE + leader * HEAD_BYTES));
            if (r + 1 == meta.validHeads ||
                ChunkGdnBwdIntraScoreLeader<G>(meta.hvBegin, r + 1, tiling_->groupSize)) {
                AscendC::SetFlag<HardEvent::MTE1_MTE2>(events_[leader].qkFree);
            }
            StoreOutput(wSlot, wGm_[outputOffset], rows);

            const uint32_t uSlot = issued++ % SLOT_COUNT;
            Matmul<false>(uSlot, L1(record + RECORD_A_BETA_OFFSET), L1(L1_V_BASE + slot * HEAD_BYTES));
            // The U load is the last MTE1 consumer of this shared record slot.
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(recordFlag);
            StoreOutput(uSlot, uGm_[outputOffset], rows);
        }
    }

    template <bool SCORE>
    __aicore__ inline void Matmul(uint32_t slot, AscendC::LocalTensor<MainT> left,
                                  AscendC::LocalTensor<MainT> right)
    {
        constexpr uint16_t rowFractals = TILE_ROWS / C0_ELEMENTS;
        constexpr uint16_t headFractals = HEAD_DIM / C0_ELEMENTS;
        auto l0A = AscendC::LocalTensor<MainT>(
            AscendC::TPosition::A2, slot * L0_AB_SLOT_BYTES, L0_AB_SLOT_BYTES / sizeof(MainT));
        auto l0B = AscendC::LocalTensor<MainT>(
            AscendC::TPosition::B2, slot * L0_AB_SLOT_BYTES, L0_AB_SLOT_BYTES / sizeof(MainT));
        const auto &events = events_[slot];
        AscendC::WaitFlag<HardEvent::M_MTE1>(events.operandsFree);
        AscendC::LoadData2DParamsV2 load{};
        load.mStep = rowFractals;
        load.kStep = SCORE ? headFractals : rowFractals;
        load.srcStride = rowFractals;
        load.dstStride = rowFractals;
        AscendC::LoadData(l0A, left, load);

        // C2 RHS is [K,N] NZ, so transpose the fractals for L0B. Score RHS
        // already has [N,K] layout in L1.
        load.kStep = headFractals;
        load.dstStride = SCORE ? rowFractals : headFractals;
        load.ifTranspose = !SCORE;
        AscendC::LoadData(l0B, right, load);
        AscendC::SetFlag<HardEvent::MTE1_M>(events.operandsReady);

        AscendC::WaitFlag<HardEvent::MTE1_M>(events.operandsReady);
        AscendC::WaitFlag<HardEvent::FIX_M>(events.resultFree);
        AscendC::MmadParams params{};
        params.m = TILE_ROWS;
        params.n = SCORE ? TILE_ROWS : HEAD_DIM;
        params.k = SCORE ? HEAD_DIM : TILE_ROWS;
        params.cmatrixInitVal = true;
        AscendC::Mmad(L0C(slot), l0A, l0B, params);
        AscendC::SetFlag<HardEvent::M_MTE1>(events.operandsFree);
        AscendC::SetFlag<HardEvent::M_FIX>(events.resultReady);
    }

    __aicore__ inline void StoreOutput(uint32_t slot, AscendC::GlobalTensor<MainT> out,
                                       uint16_t validRows)
    {
        const auto &events = events_[slot];
        AscendC::WaitFlag<HardEvent::M_FIX>(events.resultReady);
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> params{};
        params.nSize = HEAD_DIM;
        params.mSize = validRows;
        params.srcStride = TILE_ROWS;
        params.dstStride = HEAD_DIM;
        params.quantPre = std::is_same_v<MainT, half> ? QuantMode_t::F322F16 : QuantMode_t::F322BF16;
        params.params.ndNum = 1;
        AscendC::Fixpipe<MainT, float, AscendC::CFG_ROW_MAJOR>(out, L0C(slot), params);
        AscendC::SetFlag<HardEvent::FIX_M>(events.resultFree);
    }

    __aicore__ inline AscendC::LocalTensor<MainT> L1(uint32_t byteOffset) const
    {
        return AscendC::LocalTensor<MainT>(
            AscendC::TPosition::A1, byteOffset, (L1_BYTES - byteOffset) / sizeof(MainT));
    }

    __aicore__ inline AscendC::LocalTensor<float> L0C(uint32_t slot) const
    {
        return AscendC::LocalTensor<float>(
            AscendC::TPosition::CO1, slot * L0_C_SLOT_BYTES, L0_C_SLOT_BYTES / sizeof(float));
    }

    static constexpr AscendC::FixpipeConfig SCORE_FIXPIPE_CONFIG = {AscendC::CO2Layout::ROW_MAJOR, true};

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
