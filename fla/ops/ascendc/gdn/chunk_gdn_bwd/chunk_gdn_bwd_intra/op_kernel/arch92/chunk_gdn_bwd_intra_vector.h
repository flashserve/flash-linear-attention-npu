/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH92_VECTOR_H
#define CHUNK_GDN_BWD_INTRA_ARCH92_VECTOR_H

#include <type_traits>

#include "chunk_gdn_bwd_intra_common.h"
#include "../chunk_gdn_bwd_intra_common.h"

namespace GDN::Arch92 {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr CastTrait INTRA_B16_TO_F32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN
};
constexpr CastTrait INTRA_F32_TO_B16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT
};

template <typename T>
__simd_callee__ inline void LoadRowFp32(RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same_v<T, float>) {
        LoadAlign(dst, src);
    } else {
        RegTensor<T> raw;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        Cast<float, T, INTRA_B16_TO_F32>(dst, raw, mask);
    }
}

template <typename T>
__simd_callee__ inline void BroadcastFp32(RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same_v<T, float>) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst, src);
    } else {
        RegTensor<T> raw;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(raw, src);
        Cast<float, T, INTRA_B16_TO_F32>(dst, raw, mask);
    }
}

template <typename MainT>
__simd_callee__ inline void StoreNzRecordRow(__ubuf__ MainT *dst, RegTensor<float> &src,
                                             MaskReg &fp32Mask, MaskReg &nzMask)
{
    RegTensor<MainT> even, packed, odd;
    Cast<MainT, float, INTRA_F32_TO_B16>(even, src, fp32Mask);
    // The cast places values in the even lanes. Pack the 64 values into
    // four C0 blocks before storing them at the full-tile NZ stride.
    DeInterleave(packed, odd, even, even);
    StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(dst, packed, TILE_ROWS, nzMask);
}

template <typename MainT, typename GateT, typename BetaT, bool USE_EXP2>
__simd_vf__ inline void ComputeRecordVF(
    __ubuf__ MainT *record, __ubuf__ MainT *a, __ubuf__ float *score,
    __ubuf__ GateT *g, __ubuf__ BetaT *beta, uint16_t validRows, float scale)
{
    constexpr float ln2 = 0.6931471805599453f;
    auto dOut = record + RECORD_D_OFFSET / sizeof(MainT);
    auto aBgOut = record + RECORD_A_BG_OFFSET / sizeof(MainT);
    auto aBetaOut = record + RECORD_A_BETA_OFFSET / sizeof(MainT);

    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    uint32_t nzElements = TILE_ROWS;
    MaskReg nzMask = UpdateMask<MainT>(nzElements);
    RegTensor<float> gate, betaRow, bg, col, limit, rowIndex, zero;
    MaskReg validCols;
    Arange(col, 0.0f);
    Duplicate(limit, static_cast<float>(validRows), full);
    Duplicate(rowIndex, 0.0f, full);
    Duplicate(zero, 0.0f, full);
    Compare<float, CMPMODE::LT>(validCols, col, limit, full);

    LoadRowFp32(gate, g, full);
    LoadRowFp32(betaRow, beta, full);
    Select(bg, gate, zero, validCols);
    if constexpr (USE_EXP2) {
        Muls(bg, bg, ln2, validCols);
    }
    Exp(bg, bg, validCols);
    Mul(bg, bg, betaRow, validCols);

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t ndOffset = static_cast<uint32_t>(row) * TILE_ROWS;
        const uint32_t nzOffset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        RegTensor<float> aRow, scores, gRow, delta, result;
        LoadRowFp32(aRow, a + ndOffset, full);
        LoadAlign(scores, score + ndOffset);

        Mul(result, aRow, bg, validCols);
        Select(result, result, zero, validCols);
        StoreNzRecordRow(aBgOut + nzOffset, result, full, nzMask);

        Mul(result, aRow, betaRow, validCols);
        Select(result, result, zero, validCols);
        StoreNzRecordRow(aBetaOut + nzOffset, result, full, nzMask);

        MaskReg causal, valid;
        Compare<float, CMPMODE::GE>(causal, col, rowIndex, full);
        And(valid, causal, validCols, full);
        BroadcastFp32(gRow, g + row, full);
        Sub(delta, gate, gRow, valid);
        if constexpr (USE_EXP2) {
            Muls(delta, delta, ln2, valid);
        }
        Exp(delta, delta, valid);
        Mul(result, scores, delta, valid);
        Muls(result, result, scale, valid);
        Select(result, result, zero, valid);
        StoreNzRecordRow(dOut + nzOffset, result, full, nzMask);
        Adds(rowIndex, rowIndex, 1.0f, full);
    }

    RegTensor<MainT> zeroRow;
    Duplicate(zeroRow, static_cast<MainT>(0), nzMask);
    for (uint16_t row = validRows; row < TILE_ROWS; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(dOut + offset, zeroRow, TILE_ROWS, nzMask);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(aBgOut + offset, zeroRow, TILE_ROWS, nzMask);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(aBetaOut + offset, zeroRow, TILE_ROWS, nzMask);
    }
}

template <typename MainT>
__simd_vf__ inline void PackVdoVF(__ubuf__ MainT *nz, __ubuf__ MainT *v, __ubuf__ MainT *dO,
                                  uint16_t validRows)
{
    auto dONz = nz + HEAD_BYTES / sizeof(MainT);
    MaskReg mask = CreateMask<MainT, MaskPattern::ALL>();
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t ndOffset = static_cast<uint32_t>(row) * HEAD_DIM;
        const uint32_t nzOffset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        RegTensor<MainT> input;
        LoadAlign(input, v + ndOffset);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(nz + nzOffset, input, TILE_ROWS, mask);
        LoadAlign(input, dO + ndOffset);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(dONz + nzOffset, input, TILE_ROWS, mask);
    }

    RegTensor<MainT> zeroRow;
    Duplicate(zeroRow, static_cast<MainT>(0), mask);
    for (uint16_t row = validRows; row < TILE_ROWS; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(nz + offset, zeroRow, TILE_ROWS, mask);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(dONz + offset, zeroRow, TILE_ROWS, mask);
    }
}

template <typename MainT, typename GateT, typename BetaT, uint32_t G>
class ChunkGdnBwdIntraVector {
public:
    __aicore__ inline void Init(
        GM_ADDR v, GM_ADDR dO, GM_ADDR a, GM_ADDR g, GM_ADDR beta,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGdnBwdIntraTilingData *tiling)
    {
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(v));
        dOGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dO));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(a));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        vGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        dOGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        mapper_.Init(cuSeqlens, chunkIndices, tiling);
        tiling_ = tiling;
    }

    __aicore__ inline void Process()
    {
        int64_t work = GetBlockIdx();
        const int64_t stride = tiling_->blockDim;
        const int64_t workCount = tiling_->workCount;
        if (work >= workCount) {
            return;
        }

        TPipe pipe;
        InitEvents(pipe);
        ChunkGdnBwdIntraWorkMeta meta{};
        mapper_.ResolveChunkMajor(work, meta);
        for (uint32_t r = 0; r < meta.validHeads; ++r) {
            PrefetchInput(meta, r);
        }
        while (true) {
            const bool hasNext = stride < workCount - work;
            ChunkGdnBwdIntraWorkMeta next{};
            if (hasNext) {
                mapper_.ResolveChunkMajor(work + stride, next);
            }
            ProcessWork(meta, next);
            if (!hasNext) {
                break;
            }
            meta = next;
            work += stride;
        }
        DrainEvents(pipe);
    }

private:
    struct SlotEvents {
        TEventID inputReady;
        TEventID inputFree;
        TEventID outputReady;
        TEventID outputFree;
    };

    __aicore__ inline void InitEvents(TPipe &pipe)
    {
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            auto &events = events_[slot];
            events.inputReady = pipe.AllocEventID<HardEvent::MTE2_V>();
            events.inputFree = pipe.AllocEventID<HardEvent::V_MTE2>();
            events.outputReady = pipe.AllocEventID<HardEvent::V_MTE3>();
            events.outputFree = pipe.AllocEventID<HardEvent::MTE3_V>();

            // A short GM copy leaves part of the g/beta vector untouched.
            // Initialize the full read span once; V1 masks all tail lanes.
            const uint32_t raw = RAW_UB_BASE + slot * RAW_SLOT_BYTES;
            Duplicate(Ub<GateT>(raw, RAW_BETA_OFFSET), static_cast<GateT>(0), TILE_ROWS);
            Duplicate(Ub<BetaT>(raw + RAW_BETA_OFFSET, RAW_BETA_OFFSET), static_cast<BetaT>(0), TILE_ROWS);
            SetFlag<HardEvent::V_MTE2>(events.inputFree);
            SetFlag<HardEvent::MTE3_V>(events.outputFree);
        }
    }

    __aicore__ inline void DrainEvents(TPipe &pipe)
    {
        for (uint32_t slot = 0; slot < SLOT_COUNT; ++slot) {
            const auto &events = events_[slot];
            if (recordSlotUsed_[slot]) {
                CrossCoreWaitFlag<0x4, PIPE_MTE3>(RECORD_FLAG_BASE + slot);
            }
            WaitFlag<HardEvent::V_MTE2>(events.inputFree);
            WaitFlag<HardEvent::MTE3_V>(events.outputFree);
            pipe.ReleaseEventID<HardEvent::MTE2_V>(events.inputReady);
            pipe.ReleaseEventID<HardEvent::V_MTE2>(events.inputFree);
            pipe.ReleaseEventID<HardEvent::V_MTE3>(events.outputReady);
            pipe.ReleaseEventID<HardEvent::MTE3_V>(events.outputFree);
        }
    }

    __aicore__ inline void ProcessWork(const ChunkGdnBwdIntraWorkMeta &meta,
                                       const ChunkGdnBwdIntraWorkMeta &next)
    {
        for (uint32_t r = 0; r < meta.validHeads; ++r) {
            const uint32_t slot = r % SLOT_COUNT;
            const uint32_t leader = static_cast<uint32_t>(
                ChunkGdnBwdIntraLeaderR<G>(meta.hvBegin, r, tiling_->groupSize));
            const bool lastConsumer = r + 1 == meta.validHeads ||
                ChunkGdnBwdIntraScoreLeader<G>(meta.hvBegin, r + 1, tiling_->groupSize);
            if (r == leader) {
                CrossCoreWaitFlag<0x4, PIPE_V>(SCORE_FLAG_BASE + leader);
            }
            RunStage1(slot, leader, static_cast<uint16_t>(meta.validTokens), lastConsumer);

            // MTE2 may refill this slot while MTE3 sends its independent outputs to L1.
            if (r < next.validHeads) {
                PrefetchInput(next, r);
            }
            StoreStage1(slot);
        }

        // The next work may have more heads than the current tail group.
        for (uint32_t r = meta.validHeads; r < next.validHeads; ++r) {
            PrefetchInput(next, r);
        }
    }

    __aicore__ inline void PrefetchInput(const ChunkGdnBwdIntraWorkMeta &meta, uint32_t r)
    {
        const uint32_t slot = r % SLOT_COUNT;
        const uint64_t offset = (static_cast<uint64_t>(meta.batch) * tiling_->valueHeads +
                                 meta.hvBegin + r) * tiling_->seqlen + meta.tokenStart;
        const uint32_t rows = static_cast<uint32_t>(meta.validTokens);
        const uint32_t raw = RAW_UB_BASE + slot * RAW_SLOT_BYTES;
        const auto &events = events_[slot];
        WaitFlag<HardEvent::V_MTE2>(events.inputFree);
        DataCopy(Ub<MainT>(A_UB_BASE + slot * MATRIX_BYTES, MATRIX_BYTES),
                 aGm_[offset * TILE_ROWS], rows * TILE_ROWS);
        DataCopy(Ub<MainT>(V_INPUT_UB_BASE + slot * HEAD_BYTES, HEAD_BYTES),
                 vGm_[offset * HEAD_DIM], rows * HEAD_DIM);
        DataCopy(Ub<MainT>(DO_INPUT_UB_BASE + slot * HEAD_BYTES, HEAD_BYTES),
                 dOGm_[offset * HEAD_DIM], rows * HEAD_DIM);
        CopyAux(Ub<GateT>(raw, RAW_BETA_OFFSET), gGm_[offset], rows);
        CopyAux(Ub<BetaT>(raw + RAW_BETA_OFFSET, RAW_BETA_OFFSET), betaGm_[offset], rows);
        SetFlag<HardEvent::MTE2_V>(events.inputReady);
    }

    template <typename T>
    __aicore__ inline void CopyAux(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t rows) const
    {
        const DataCopyExtParams params{1, rows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, params, pad);
    }

    __aicore__ inline void RunStage1(uint32_t slot, uint32_t leader,
                                     uint16_t validRows, bool lastConsumer)
    {
        const uint32_t raw = RAW_UB_BASE + slot * RAW_SLOT_BYTES;
        auto record = Ub<MainT>(RECORD_UB_BASE + slot * RECORD_SLOT_BYTES, RECORD_SLOT_BYTES);
        auto a = Ub<MainT>(A_UB_BASE + slot * MATRIX_BYTES, MATRIX_BYTES);
        auto score = Ub<float>(SCORE_UB_BASE + leader * SCORE_SLOT_BYTES, SCORE_SLOT_BYTES);
        auto g = Ub<GateT>(raw, RAW_BETA_OFFSET);
        auto beta = Ub<BetaT>(raw + RAW_BETA_OFFSET, RAW_BETA_OFFSET);
        auto v = Ub<MainT>(V_INPUT_UB_BASE + slot * HEAD_BYTES, HEAD_BYTES);
        auto dO = Ub<MainT>(DO_INPUT_UB_BASE + slot * HEAD_BYTES, HEAD_BYTES);
        auto vdo = Ub<MainT>(VDO_NZ_UB_BASE + slot * VDO_NZ_SLOT_BYTES, VDO_NZ_SLOT_BYTES);
        const auto &events = events_[slot];
        WaitFlag<HardEvent::MTE2_V>(events.inputReady);
        WaitFlag<HardEvent::MTE3_V>(events.outputFree);
        auto recordAddr = reinterpret_cast<__ubuf__ MainT *>(record.GetPhyAddr());
        auto aAddr = reinterpret_cast<__ubuf__ MainT *>(a.GetPhyAddr());
        auto scoreAddr = reinterpret_cast<__ubuf__ float *>(score.GetPhyAddr());
        auto gAddr = reinterpret_cast<__ubuf__ GateT *>(g.GetPhyAddr());
        auto betaAddr = reinterpret_cast<__ubuf__ BetaT *>(beta.GetPhyAddr());
        if (tiling_->useExp2 != 0) {
            ComputeRecordVF<MainT, GateT, BetaT, true>(
                recordAddr, aAddr, scoreAddr, gAddr, betaAddr, validRows, tiling_->scale);
        } else {
            ComputeRecordVF<MainT, GateT, BetaT, false>(
                recordAddr, aAddr, scoreAddr, gAddr, betaAddr, validRows, tiling_->scale);
        }
        if (lastConsumer) {
            CrossCoreSetFlag<0x4, PIPE_V>(SCORE_FLAG_BASE + leader);
        }

        PackVdoVF<MainT>(
            reinterpret_cast<__ubuf__ MainT *>(vdo.GetPhyAddr()),
            reinterpret_cast<__ubuf__ MainT *>(v.GetPhyAddr()),
            reinterpret_cast<__ubuf__ MainT *>(dO.GetPhyAddr()), validRows);
        SetFlag<HardEvent::V_MTE2>(events.inputFree);
        SetFlag<HardEvent::V_MTE3>(events.outputReady);
    }

    __aicore__ inline void StoreStage1(uint32_t slot)
    {
        const auto &events = events_[slot];
        const uint16_t flag = RECORD_FLAG_BASE + slot;
        if (recordSlotUsed_[slot]) {
            CrossCoreWaitFlag<0x4, PIPE_MTE3>(flag);
        }
        WaitFlag<HardEvent::V_MTE3>(events.outputReady);

        auto record = Ub<MainT>(RECORD_UB_BASE + slot * RECORD_SLOT_BYTES, RECORD_SLOT_BYTES);
        auto vdo = Ub<MainT>(VDO_NZ_UB_BASE + slot * VDO_NZ_SLOT_BYTES, VDO_NZ_SLOT_BYTES);
        CopyToL1(L1_RECORD_BASE + slot * RECORD_SLOT_BYTES, record, RECORD_SLOT_BYTES);
        CopyToL1(L1_V_BASE + slot * HEAD_BYTES, vdo, HEAD_BYTES);
        CopyToL1(L1_DO_BASE + slot * HEAD_BYTES, vdo[HEAD_BYTES / sizeof(MainT)], HEAD_BYTES);

        SetFlag<HardEvent::MTE3_V>(events.outputFree);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(flag);
        recordSlotUsed_[slot] = true;
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> Ub(uint32_t offset, uint32_t bytes) const
    {
        return LocalTensor<T>(TPosition::VECIN, offset, bytes / sizeof(T));
    }

    __aicore__ inline void CopyToL1(uint32_t offset, LocalTensor<MainT> src, uint32_t bytes) const
    {
        LocalTensor<MainT> dst(TPosition::A1, offset, bytes / sizeof(MainT));
        DataCopy(dst, src, DataCopyParams(1, bytes / BLOCK_BYTES, 0, 0));
    }

    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;
    GlobalTensor<MainT> vGm_;
    GlobalTensor<MainT> dOGm_;
    GlobalTensor<MainT> aGm_;
    GlobalTensor<GateT> gGm_;
    GlobalTensor<BetaT> betaGm_;
    SlotEvents events_[SLOT_COUNT];
    bool recordSlotUsed_[SLOT_COUNT] = {};
};

} // namespace GDN::Arch92

#endif // CHUNK_GDN_BWD_INTRA_ARCH92_VECTOR_H
