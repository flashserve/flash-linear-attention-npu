/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH92_VECTOR_H
#define CHUNK_GDN_BWD_INTRA_ARCH92_VECTOR_H

#include <type_traits>
#include "kernel_utils/vector/regbase.hpp"

#include "chunk_gdn_bwd_intra_common.h"
#include "../chunk_gdn_bwd_intra_common.h"

namespace GDN::Arch92 {

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename T>
__simd_callee__ inline void LoadIntraRow(
    RegTensor<float> &even, RegTensor<float> &odd, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>()) {
        RegTensor<float> row;
        LoadIn<float, false>(row, src);
        // One 64-element row fits exactly in the slot. Split that register
        // into even/odd lanes without reading a second row from UB.
        DeInterleave(even, odd, row, row);
    } else {
        RegTensor<T> row;
        // Masked block load covers just the first four C0 blocks of the register.
        LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY>(row, src, 1, mask);
        CastHalf2Float<T>(even, odd, row, mask);
    }
}

template <typename MainT, typename GateT, typename BetaT, bool USE_EXP2>
__simd_vf__ inline void ComputeRecordVF(
    __ubuf__ MainT *record, __ubuf__ MainT *a, __ubuf__ float *score,
    __ubuf__ GateT *g, __ubuf__ BetaT *beta, uint16_t validRows, float scale)
{
    constexpr float LN2 = 0.6931471805599453f;
    auto dOut = record + RECORD_D_OFFSET / sizeof(MainT);
    auto aBgOut = record + RECORD_A_BG_OFFSET / sizeof(MainT);
    auto aBetaOut = record + RECORD_A_BETA_OFFSET / sizeof(MainT);

    MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFp16 = CreateMask<half, MaskPattern::ALL>();
    uint32_t mainElements = TILE_ROWS;
    uint32_t gateElements = TILE_ROWS;
    uint32_t betaElements = TILE_ROWS;
    MaskReg maskMain = UpdateMask<MainT>(mainElements);
    MaskReg maskGate = UpdateMask<GateT>(gateElements);
    MaskReg maskBeta = UpdateMask<BetaT>(betaElements);

    RegTensor<float> gZero, gOne, betaZero, betaOne, bgZero, bgOne;
    RegTensor<float> colZero, colOne, rowIndex, validLimit, scaleReg, zeros;
    RegTensor<half> colIndex;
    MaskReg validTailZero, validTailOne;

    Arange(colIndex, 0);
    CastHalf2Float<half>(colZero, colOne, colIndex, maskFp16);
    Duplicate(rowIndex, 0.0f, maskFp32);
    Duplicate(validLimit, static_cast<float>(validRows), maskFp32);
    Duplicate(scaleReg, scale, maskFp32);
    Duplicate(zeros, 0.0f, maskFp32);
    CompareTwoReg<float, CMPMODE::LT>(
        validTailZero, validTailOne, colZero, colOne, validLimit, validLimit, maskFp32);

    LoadIntraRow(gZero, gOne, g, maskGate);
    LoadIntraRow(betaZero, betaOne, beta, maskBeta);
    Select(bgZero, gZero, zeros, validTailZero);
    Select(bgOne, gOne, zeros, validTailOne);
    if constexpr (USE_EXP2) {
        Muls(bgZero, bgZero, LN2, validTailZero);
        Muls(bgOne, bgOne, LN2, validTailOne);
    }
    Exp(bgZero, bgZero, validTailZero);
    Exp(bgOne, bgOne, validTailOne);
    Mul(bgZero, bgZero, betaZero, validTailZero);
    Mul(bgOne, bgOne, betaOne, validTailOne);

    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t ndOffset = static_cast<uint32_t>(row) * TILE_ROWS;
        const uint32_t nzOffset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        RegTensor<MainT> aBgReg, aBetaReg, dReg;
        RegTensor<GateT> gRawReg;
        RegTensor<float> aZero, aOne, scoreZero, scoreOne, rowG;
        RegTensor<float> deltaZero, deltaOne, resultZero, resultOne;
        MaskReg causalZero, causalOne, validZero, validOne;

        LoadIntraRow(aZero, aOne, a + ndOffset, maskMain);
        LoadIntraRow(scoreZero, scoreOne, score + ndOffset, maskFp32);

        Mul(resultZero, aZero, bgZero, validTailZero);
        Mul(resultOne, aOne, bgOne, validTailOne);
        Select(resultZero, resultZero, zeros, validTailZero);
        Select(resultOne, resultOne, zeros, validTailOne);
        CastFloat2Half<MainT>(aBgReg, resultZero, resultOne, maskFp32);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(aBgOut + nzOffset, aBgReg, TILE_ROWS, maskMain);

        Mul(resultZero, aZero, betaZero, validTailZero);
        Mul(resultOne, aOne, betaOne, validTailOne);
        Select(resultZero, resultZero, zeros, validTailZero);
        Select(resultOne, resultOne, zeros, validTailOne);
        CastFloat2Half<MainT>(aBetaReg, resultZero, resultOne, maskFp32);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(aBetaOut + nzOffset, aBetaReg, TILE_ROWS, maskMain);

        LoadIn<GateT, true>(gRawReg, g + row);
        HalfOrFloat2Float<GateT>(rowG, gRawReg, maskFp16, maskFp32);
        CompareTwoReg<float, CMPMODE::GE>(
            causalZero, causalOne, colZero, colOne, rowIndex, rowIndex, maskFp32);
        And(validZero, causalZero, validTailZero, maskFp32);
        And(validOne, causalOne, validTailOne, maskFp32);
        Sub(deltaZero, gZero, rowG, validZero);
        Sub(deltaOne, gOne, rowG, validOne);
        if constexpr (USE_EXP2) {
            Muls(deltaZero, deltaZero, LN2, validZero);
            Muls(deltaOne, deltaOne, LN2, validOne);
        }
        Exp(deltaZero, deltaZero, validZero);
        Exp(deltaOne, deltaOne, validOne);
        Mul(resultZero, scoreZero, deltaZero, validZero);
        Mul(resultOne, scoreOne, deltaOne, validOne);
        Mul(resultZero, resultZero, scaleReg, validZero);
        Mul(resultOne, resultOne, scaleReg, validOne);
        Select(resultZero, resultZero, zeros, validZero);
        Select(resultOne, resultOne, zeros, validOne);
        CastFloat2Half<MainT>(dReg, resultZero, resultOne, maskFp32);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(dOut + nzOffset, dReg, TILE_ROWS, maskMain);
        Adds(rowIndex, rowIndex, 1.0f, maskFp32);
    }

    RegTensor<MainT> zeroRow;
    Duplicate(zeroRow, static_cast<MainT>(0), maskMain);
    for (uint16_t row = validRows; row < TILE_ROWS; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(dOut + offset, zeroRow, TILE_ROWS, maskMain);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(aBgOut + offset, zeroRow, TILE_ROWS, maskMain);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(aBetaOut + offset, zeroRow, TILE_ROWS, maskMain);
    }
}

template <typename MainT>
__simd_vf__ inline void PackMatrixVF(__ubuf__ MainT *nz, __ubuf__ MainT *nd, uint16_t validRows)
{
    uint32_t elements = HEAD_DIM;
    MaskReg mask = UpdateMask<MainT>(elements);
    RegTensor<MainT> rowData;
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t ndOffset = static_cast<uint32_t>(row) * HEAD_DIM;
        const uint32_t nzOffset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        LoadIn<MainT, false>(rowData, nd + ndOffset);
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(nz + nzOffset, rowData, TILE_ROWS, mask);
    }

    Duplicate(rowData, static_cast<MainT>(0), mask);
    for (uint16_t row = validRows; row < TILE_ROWS; ++row) {
        const uint32_t nzOffset = static_cast<uint32_t>(row) * C0_ELEMENTS;
        StoreAlign<MainT, DataCopyMode::DATA_BLOCK_COPY>(nz + nzOffset, rowData, TILE_ROWS, mask);
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

        auto vNz = reinterpret_cast<__ubuf__ MainT *>(vdo.GetPhyAddr());
        auto dONz = vNz + HEAD_BYTES / sizeof(MainT);
        PackMatrixVF<MainT>(vNz, reinterpret_cast<__ubuf__ MainT *>(v.GetPhyAddr()), validRows);
        PackMatrixVF<MainT>(dONz, reinterpret_cast<__ubuf__ MainT *>(dO.GetPhyAddr()), validRows);
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
