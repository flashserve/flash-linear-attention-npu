/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_STAGE0_H
#define CHUNK_GDN_BWD_INTRA_STAGE0_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include <type_traits>
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "kernel_utils/vector/regbase.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "chunk_gdn_bwd_intra_common.h"

namespace GDN {

constexpr uint32_t INTRA_CG_MAX = 4;
constexpr uint32_t INTRA_SUBBLOCK_FLAG_OFFSET = 16;
constexpr uint32_t INTRA_SCORE_FREE_FLAG = 0;
constexpr uint32_t INTRA_SCORE_READY_FLAG = 0;
constexpr uint32_t INTRA_WORKSPACE_FREE_FLAG = 4;
constexpr uint32_t INTRA_WORKSPACE_READY_FLAG = 4;
constexpr uint32_t INTRA_SCORE_SLOT_BYTES = 8 * 1024;
constexpr uint32_t INTRA_SCORE_UB_BASE = 0;
constexpr uint32_t INTRA_STAGE0_DEBUG_UB_BASE = 32 * 1024;
constexpr uint32_t INTRA_A_UB_BASE = 32 * 1024;
constexpr uint32_t INTRA_A_BG_UB_BASE = 40 * 1024;
constexpr uint32_t INTRA_A_BETA_UB_BASE = 48 * 1024;
constexpr uint32_t INTRA_RAW_UB_BASE = 56 * 1024;
constexpr uint32_t INTRA_D_UB_BASE = 72 * 1024;
constexpr uint32_t INTRA_MATRIX_HALF_BYTES = 4 * 1024;
constexpr uint32_t INTRA_RAW_SLOT_BYTES = 512;
constexpr uint32_t INTRA_RAW_BETA_OFFSET = 256;

template <typename MainT, typename GateT, typename BetaT, bool USE_EXP2>
__simd_vf__ inline void ChunkGdnBwdIntraStage1VF(
    __ubuf__ MainT *aBgOut, __ubuf__ MainT *aBetaOut, __ubuf__ MainT *dOut,
    __ubuf__ MainT *aIn, __ubuf__ float *scoreIn, __ubuf__ GateT *gIn,
    __ubuf__ GateT *gRowIn, __ubuf__ BetaT *betaIn,
    uint16_t rows, uint16_t rowBegin,
    uint16_t validTokens, float scale)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    constexpr uint32_t ROW_ELEMENTS = 64;
    constexpr float LN2 = 0.6931471805599453f;

    RegTensor<GateT> gRawReg;
    RegTensor<BetaT> betaRawReg;
    RegTensor<MainT> aReg, aBgReg, aBetaReg, dReg;
    RegTensor<float> gZero, gOne, betaZero, betaOne, gateZero, gateOne;
    RegTensor<float> aZero, aOne, scoreZero, scoreOne;
    RegTensor<float> deltaZero, deltaOne, resultZero, resultOne;
    RegTensor<float> rowG, rowIndex, validLimit, scaleReg;
    RegTensor<half> colIndexRaw;
    RegTensor<float> colIndexZero, colIndexOne;
    MaskReg validCausalZero, validCausalOne;
    MaskReg validTailZero, validTailOne;
    MaskReg validZero, validOne;
    MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFp16 = CreateMask<half, MaskPattern::ALL>();
    uint32_t maskMainElements = ROW_ELEMENTS;
    uint32_t maskGateElements = ROW_ELEMENTS;
    uint32_t maskBetaElements = ROW_ELEMENTS;
    MaskReg maskMain = UpdateMask<MainT>(maskMainElements);
    MaskReg maskGate = UpdateMask<GateT>(maskGateElements);
    MaskReg maskBeta = UpdateMask<BetaT>(maskBetaElements);

    Duplicate(scaleReg, scale, maskFp32);
    Duplicate(rowIndex, static_cast<float>(rowBegin), maskFp32);
    Arange(colIndexRaw, 0);
    CastHalf2Float<half>(colIndexZero, colIndexOne, colIndexRaw, maskFp16);
    Duplicate(validLimit, static_cast<float>(validTokens), maskFp32);
    CompareTwoReg<float, CMPMODE::LT>(
        validTailZero, validTailOne, colIndexZero, colIndexOne,
        validLimit, validLimit, maskFp32);

    if constexpr (std::is_same<GateT, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(gZero, gOne, gIn);
    } else {
        LoadIn<GateT, false>(gRawReg, gIn);
        CastHalf2Float<GateT>(gZero, gOne, gRawReg, maskGate);
    }
    if constexpr (std::is_same<BetaT, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(betaZero, betaOne, betaIn);
    } else {
        LoadIn<BetaT, false>(betaRawReg, betaIn);
        CastHalf2Float<BetaT>(betaZero, betaOne, betaRawReg, maskBeta);
    }

    Adds(gateZero, gZero, 0.0f, maskFp32);
    Adds(gateOne, gOne, 0.0f, maskFp32);
    if constexpr (USE_EXP2) {
        Muls(gateZero, gateZero, LN2, maskFp32);
        Muls(gateOne, gateOne, LN2, maskFp32);
    }
    ExpFloatTwoReg(gateZero, gateOne, gateZero, gateOne, maskFp32);
    MulFloatTwoReg(gateZero, gateOne, gateZero, gateOne,
                   betaZero, betaOne, maskFp32);

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        LoadIn<MainT, false>(aReg, aIn + rowOffset);
        CastHalf2Float<MainT>(aZero, aOne, aReg, maskMain);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            scoreZero, scoreOne, scoreIn + rowOffset);

        Mul(resultZero, aZero, gateZero, validTailZero);
        Mul(resultOne, aOne, gateOne, validTailOne);
        CastFloat2Half<MainT>(aBgReg, resultZero, resultOne, maskFp32);
        StoreAlign(aBgOut + rowOffset, aBgReg, maskMain);

        Mul(resultZero, aZero, betaZero, validTailZero);
        Mul(resultOne, aOne, betaOne, validTailOne);
        CastFloat2Half<MainT>(aBetaReg, resultZero, resultOne, maskFp32);
        StoreAlign(aBetaOut + rowOffset, aBetaReg, maskMain);

        LoadIn<GateT, true>(gRawReg, gRowIn + row);
        HalfOrFloat2Float<GateT>(rowG, gRawReg, maskFp16, maskFp32);
        CompareTwoReg<float, CMPMODE::GE>(
            validCausalZero, validCausalOne, colIndexZero, colIndexOne,
            rowIndex, rowIndex, maskFp32);
        And(validZero, validCausalZero, validTailZero, maskFp32);
        And(validOne, validCausalOne, validTailOne, maskFp32);
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
        CastFloat2Half<MainT>(dReg, resultZero, resultOne, maskFp32);
        StoreAlign(dOut + rowOffset, dReg, maskMain);
        Adds(rowIndex, rowIndex, 1.0f, maskFp32);
    }
}

template <typename MainT>
class ChunkGdnBwdIntraStage0Cube {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using ScoreTile = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, MainT, LayoutRM, MainT, LayoutCM, float, LayoutRM>;
    using CopyL1ToL0A = typename ScoreTile::CopyL1ToL0A;
    using CopyL1ToL0B = typename ScoreTile::CopyL1ToL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, MainT, typename ScoreTile::LayoutTagL1A>;
    using OutputTile = Common::Tile::PackedTileCopyTla<
        ArchTag, MainT, LayoutRM, MainT, LayoutRM, MainT, LayoutRM>;
    using OutputCopyL1ToL0A = typename OutputTile::CopyL1ToL0A;
    using OutputCopyL1ToL0B = typename OutputTile::CopyL1ToL0B;
    template <class Tensor>
    using CopyGmToL1A = typename ScoreTile::template CopyGmToL1A<Tensor>;
    template <class Tensor>
    using CopyGmToL1B = typename ScoreTile::template CopyGmToL1B<Tensor>;
    template <class Tensor>
    using CopyL0CToUb = typename ScoreTile::template CopyL0CToDst<Tensor>;
    template <class Tensor>
    using OutputCopyGmToL1A = typename OutputTile::template CopyGmToL1A<Tensor>;
    template <class Tensor>
    using OutputCopyGmToL1B = typename OutputTile::template CopyGmToL1B<Tensor>;
    template <class Tensor>
    using OutputCopyL0CToGm = typename OutputTile::template CopyL0CToDst<Tensor>;

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dO,
        GM_ADDR w, GM_ADDR u, GM_ADDR dvLocal, GM_ADDR workspace,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGdnBwdIntraTilingData *__restrict tiling)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(v));
        dO_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dO));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(w));
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(u));
        dvLocal_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dvLocal));
        if (tiling->stage != 0) {
            workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(workspace));
        }
        mapper_.Init(cuSeqlens, chunkIndices, tiling);
        tiling_ = tiling;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
    }

    __aicore__ inline void Process()
    {
        AscendC::SetHF32Mode(false);
        AscendC::SetMMLayoutTransform(true);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
        }
        bool scoreSlotUsed[INTRA_CG_MAX] = {false, false, false, false};
        for (int64_t work = coreIdx_; work < tiling_->workCount;
             work += static_cast<int64_t>(AscendC::GetBlockNum())) {
            ChunkGdnBwdIntraWorkMeta meta{};
            mapper_.Resolve(work, meta);
            // Issue every unique q/k prefetch before starting the first Score MMAD.
            for (int64_t r = 0; r < meta.validHeads; ++r) {
                if (ChunkGdnBwdIntraScoreLeader(meta.hvBegin, r, tiling_->headRatio)) {
                    const uint32_t slot = static_cast<uint32_t>(r);
                    PrefetchLeader(meta, slot, scoreSlotUsed[slot]);
                    scoreSlotUsed[slot] = true;
                }
            }
            for (int64_t r = 0; r < meta.validHeads; ++r) {
                if (ChunkGdnBwdIntraScoreLeader(meta.hvBegin, r, tiling_->headRatio)) {
                    ComputeLeader(meta, static_cast<uint32_t>(r));
                }
            }
            if (tiling_->stage == 1) {
                DrainStage1Workspace(meta);
            } else if (tiling_->stage == 2) {
                RunStage2(meta);
            }
        }
        // Drain the final Vector consumer before the mixed kernel exits.
        for (uint32_t slot = 0; slot < INTRA_CG_MAX; ++slot) {
            if (scoreSlotUsed[slot]) {
                WaitScoreFree(slot);
            }
        }
        if (tiling_->stage == 2) {
            // The final direct-to-GM Fixpipe must finish before kernel exit.
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        }
        AscendC::SetMMLayoutTransform(false);
    }

private:
    static constexpr uint32_t L1_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t L1_K_BASE = 0;
    static constexpr uint32_t L1_Q_BASE = 64 * 1024;
    static constexpr uint32_t L1_A_BG_BASE = 64 * 1024;
    static constexpr uint32_t L1_A_BETA_BASE = 96 * 1024;
    static constexpr uint32_t L1_D_BASE = 128 * 1024;
    static constexpr uint32_t L1_V_BASE = 160 * 1024;
    static constexpr uint32_t L1_DO_BASE = 224 * 1024;
    static constexpr uint32_t L1_MATRIX_SLOT_BYTES = 8 * 1024;
    static constexpr uint32_t L0_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t L0C_SCORE_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t L0C_OUTPUT_SLOT_BYTES = 32 * 1024;
    static constexpr uint32_t VDO_READY_EVENT = 4;

    __aicore__ inline AscendC::LocalTensor<uint8_t> L0ABuffer() const
    {
        return resource_.l0ABuf.template GetBufferByByte<uint8_t>(0);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> L0BBuffer() const
    {
        return resource_.l0BBuf.template GetBufferByByte<uint8_t>(0);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> L0CBuffer() const
    {
        return resource_.l0CBuf.template GetBufferByByte<uint8_t>(0);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> UbBuffer() const
    {
        return resource_.ubBuf.template GetBufferByByte<uint8_t>(0);
    }

    __aicore__ inline void ClearL1(AscendC::LocalTensor<MainT> tensor) const
    {
        AscendC::InitConstValueParams<MainT> params(
            1, static_cast<uint16_t>(L1_SLOT_BYTES / 32), 0, static_cast<MainT>(0));
        AscendC::InitConstValue(tensor, params);
    }

    __aicore__ inline void WaitScoreFree(uint32_t slot) const
    {
        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(INTRA_SCORE_FREE_FLAG + slot);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
            INTRA_SCORE_FREE_FLAG + slot + INTRA_SUBBLOCK_FLAG_OFFSET);
    }

    __aicore__ inline void DrainStage1Workspace(
        const ChunkGdnBwdIntraWorkMeta &meta) const
    {
        for (uint32_t r = 0; r < static_cast<uint32_t>(meta.validHeads); ++r) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                INTRA_WORKSPACE_READY_FLAG + r);
            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
                INTRA_WORKSPACE_READY_FLAG + r + INTRA_SUBBLOCK_FLAG_OFFSET);
            // Stage 1 debug consumes no data; the acknowledgement closes ring reuse.
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE2>(
                INTRA_WORKSPACE_FREE_FLAG + r);
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE2>(
                INTRA_WORKSPACE_FREE_FLAG + r + INTRA_SUBBLOCK_FLAG_OFFSET);
        }
    }

    __aicore__ inline void PrefetchLeader(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t r, bool scoreSlotUsed)
    {
        if (scoreSlotUsed) {
            WaitScoreFree(r);
        }
        const int64_t hk = (meta.hvBegin + r) / tiling_->headRatio;
        const uint64_t gmOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->qkHeads + hk) *
                 tiling_->seqlen + meta.tokenStart) * tiling_->keyDim;
        auto l1K = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_K_BASE + r * L1_SLOT_BYTES);
        auto l1Q = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_Q_BASE + r * L1_SLOT_BYTES);
        if (meta.validTokens < tiling_->chunkSize) {
            ClearL1(l1K);
            ClearL1(l1Q);
        }

        auto gmK = tla::MakeTensor(
            k_[gmOffset], tla::MakeLayout<MainT, LayoutRM>(meta.validTokens, 128),
            Catlass::Arch::PositionGM{});
        auto gmQ = tla::MakeTensor(
            q_[gmOffset], tla::MakeLayout<MainT, LayoutCM>(128, meta.validTokens),
            Catlass::Arch::PositionGM{});
        auto tensorL1K = tla::MakeTensor(
            l1K, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL1A>(64, 128),
            Catlass::Arch::PositionL1{});
        auto tensorL1Q = tla::MakeTensor(
            l1Q, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL1B>(128, 64),
            Catlass::Arch::PositionL1{});
        auto l1KBlock = tla::GetTile(
            tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(meta.validTokens, 128));
        auto l1QBlock = tla::GetTile(
            tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(128, meta.validTokens));
        CopyGmToL1A<decltype(gmK)> copyK;
        CopyGmToL1B<decltype(gmQ)> copyQ;
        copyK(l1KBlock, gmK);
        copyQ(l1QBlock, gmQ);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(r);
    }

    __aicore__ inline void ComputeLeader(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t r)
    {
        const uint32_t leaderIndex = r / static_cast<uint32_t>(tiling_->headRatio);
        const uint32_t l0Slot = leaderIndex & 1U;
        const uint32_t l0AEvent = 2 * l0Slot;
        const uint32_t l0BEvent = 2 * l0Slot + 1;
        auto l1K = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_K_BASE + r * L1_SLOT_BYTES);
        auto l1Q = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_Q_BASE + r * L1_SLOT_BYTES);
        auto l0A = L0ABuffer()[l0Slot * L0_SLOT_BYTES].template ReinterpretCast<MainT>();
        auto l0B = L0BBuffer()[l0Slot * L0_SLOT_BYTES].template ReinterpretCast<MainT>();
        auto l0C = L0CBuffer()[l0Slot * L0C_SCORE_SLOT_BYTES].template ReinterpretCast<float>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(r);
        auto tensorL1K = tla::MakeTensor(
            l1K, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL1A>(64, 128),
            Catlass::Arch::PositionL1{});
        auto tensorL1Q = tla::MakeTensor(
            l1Q, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL1B>(128, 64),
            Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL0A>(64, 128),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<MainT, typename ScoreTile::LayoutTagL0B>(128, 64),
            Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(64, 64), Catlass::Arch::PositionL0C{});
        auto tileL1K = tla::GetTile(
            tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(64, 128));
        auto tileL1Q = tla::GetTile(
            tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(128, 64));
        auto tileL0A = tla::GetTile(
            tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(64, 128));
        auto tileL0B = tla::GetTile(
            tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(128, 64));
        auto tileL0C = tla::GetTile(
            tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(64, 64));
        auto tileL0CTop = tla::GetTile(
            tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(32, 64));
        auto tileL0CBottom = tla::GetTile(
            tensorL0C, tla::MakeCoord(32, 0), tla::MakeShape(32, 64));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
        CopyL1ToL0A{}(tileL0A, tileL1K);
        CopyL1ToL0B{}(tileL0B, tileL1Q);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0Slot);

        TileMmad{}(tileL0C, tileL0A, tileL0B, 64U, 64U, 128U,
                   true, static_cast<uint8_t>(0));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0Slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0Slot);

        auto score = UbBuffer()[INTRA_SCORE_UB_BASE + r * INTRA_SCORE_SLOT_BYTES]
                         .template ReinterpretCast<float>();
        auto scoreTensor = tla::MakeTensor(
            score, tla::MakeLayout<float, LayoutRM>(32, 64),
            Catlass::Arch::PositionUB{});
        // Publish the upper and lower Score halves to their owning AIV separately.
        CopyL0CToUb<decltype(scoreTensor)>{}(
            scoreTensor, tileL0CTop, static_cast<uint8_t>(0), static_cast<uint8_t>(0));
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(INTRA_SCORE_READY_FLAG + r);

        CopyL0CToUb<decltype(scoreTensor)>{}(
            scoreTensor, tileL0CBottom, static_cast<uint8_t>(1), static_cast<uint8_t>(0));
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
            INTRA_SCORE_READY_FLAG + r + INTRA_SUBBLOCK_FLAG_OFFSET);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0Slot);
    }

    __aicore__ inline void PrefetchStage2Vectors(
        const ChunkGdnBwdIntraWorkMeta &meta)
    {
        const uint64_t hvBase = static_cast<uint64_t>(meta.hvBegin);
        const uint64_t gmOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hvBase) *
                 tiling_->seqlen + meta.tokenStart) * tiling_->valueDim;
        auto l1V = resource_.l1Buf.template GetBufferByByte<MainT>(L1_V_BASE);
        auto l1DO = resource_.l1Buf.template GetBufferByByte<MainT>(L1_DO_BASE);
        if (meta.validTokens < tiling_->chunkSize) {
            for (int64_t r = 0; r < meta.validHeads; ++r) {
                ClearL1(l1V[r * (L1_SLOT_BYTES / sizeof(MainT))]);
                ClearL1(l1DO[r * (L1_SLOT_BYTES / sizeof(MainT))]);
            }
        }
        auto gmV = tla::MakeTensor(
            v_[gmOffset], tla::MakeLayout<MainT, LayoutRM>(meta.validTokens, 128),
            Catlass::Arch::PositionGM{});
        auto gmDO = tla::MakeTensor(
            dO_[gmOffset], tla::MakeLayout<MainT, LayoutRM>(meta.validTokens, 128),
            Catlass::Arch::PositionGM{});
        auto tensorL1V = tla::MakeTensor(
            l1V, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1B>(64, 128),
            Catlass::Arch::PositionL1{});
        auto tensorL1DO = tla::MakeTensor(
            l1DO, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1B>(64, 128),
            Catlass::Arch::PositionL1{});
        const uint32_t srcStride = static_cast<uint32_t>(
            tiling_->seqlen * tiling_->valueDim);
        const uint32_t dstStride = L1_SLOT_BYTES / sizeof(MainT);
        OutputCopyGmToL1B<decltype(gmV)>{}(
            tensorL1V, gmV, static_cast<uint32_t>(meta.validHeads), srcStride, dstStride);
        OutputCopyGmToL1B<decltype(gmDO)>{}(
            tensorL1DO, gmDO, static_cast<uint32_t>(meta.validHeads), srcStride, dstStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(VDO_READY_EVENT);
    }

    __aicore__ inline void LoadStage2Workspace(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t r)
    {
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(INTRA_WORKSPACE_READY_FLAG + r);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
            INTRA_WORKSPACE_READY_FLAG + r + INTRA_SUBBLOCK_FLAG_OFFSET);
        const uint64_t rho = static_cast<uint64_t>(coreIdx_) * tiling_->cg + r;
        const uint64_t matrixStride = tiling_->matrixStrideBytes / sizeof(MainT);
        const uint64_t aBgOffset = tiling_->aBgWorkspaceOffset / sizeof(MainT) +
                                   rho * matrixStride;
        const uint64_t aBetaOffset = tiling_->aBetaWorkspaceOffset / sizeof(MainT) +
                                     rho * matrixStride;
        const uint64_t dOffset = tiling_->dWorkspaceOffset / sizeof(MainT) +
                                 rho * matrixStride;
        auto l1ABg = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_A_BG_BASE + r * L1_MATRIX_SLOT_BYTES);
        auto l1ABeta = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_A_BETA_BASE + r * L1_MATRIX_SLOT_BYTES);
        auto l1D = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_D_BASE + r * L1_MATRIX_SLOT_BYTES);
        auto gmABg = tla::MakeTensor(
            workspace_[aBgOffset], tla::MakeLayout<MainT, LayoutRM>(64, 64),
            Catlass::Arch::PositionGM{});
        auto gmABeta = tla::MakeTensor(
            workspace_[aBetaOffset], tla::MakeLayout<MainT, LayoutRM>(64, 64),
            Catlass::Arch::PositionGM{});
        auto gmD = tla::MakeTensor(
            workspace_[dOffset], tla::MakeLayout<MainT, LayoutRM>(64, 64),
            Catlass::Arch::PositionGM{});
        auto tensorL1ABg = tla::MakeTensor(
            l1ABg, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1A>(64, 64),
            Catlass::Arch::PositionL1{});
        auto tensorL1ABeta = tla::MakeTensor(
            l1ABeta, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1A>(64, 64),
            Catlass::Arch::PositionL1{});
        auto tensorL1D = tla::MakeTensor(
            l1D, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1A>(64, 64),
            Catlass::Arch::PositionL1{});
        OutputCopyGmToL1A<decltype(gmABg)>{}(tensorL1ABg, gmABg);
        OutputCopyGmToL1A<decltype(gmABeta)>{}(tensorL1ABeta, gmABeta);
        OutputCopyGmToL1A<decltype(gmD)>{}(tensorL1D, gmD);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(r);
        // Both AIV halves may reuse this ring record after all three copies finish.
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE2>(INTRA_WORKSPACE_FREE_FLAG + r);
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE2>(
            INTRA_WORKSPACE_FREE_FLAG + r + INTRA_SUBBLOCK_FLAG_OFFSET);
    }

    template <bool FULL_TOKENS>
    __aicore__ inline void RunStage2Mmad(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t l0Slot,
        AscendC::LocalTensor<MainT> l1Left, AscendC::LocalTensor<MainT> l1Right,
        AscendC::GlobalTensor<MainT> &output, uint64_t outputOffset)
    {
        const uint32_t l0AEvent = 2 * l0Slot;
        const uint32_t l0BEvent = 2 * l0Slot + 1;
        const uint32_t mActual = FULL_TOKENS ? 64U : static_cast<uint32_t>(meta.validTokens);
        const uint32_t kActual = FULL_TOKENS ? 64U : static_cast<uint32_t>(meta.validTokens);
        auto l0A = L0ABuffer()[l0Slot * L0_SLOT_BYTES].template ReinterpretCast<MainT>();
        auto l0B = L0BBuffer()[l0Slot * L0_SLOT_BYTES].template ReinterpretCast<MainT>();
        auto l0C = L0CBuffer()[l0Slot * L0C_OUTPUT_SLOT_BYTES].template ReinterpretCast<float>();
        auto tensorL1A = tla::MakeTensor(
            l1Left, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1A>(64, 64),
            Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(
            l1Right, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1B>(64, 128),
            Catlass::Arch::PositionL1{});
        // L0 layout describes the actual tail tile; L1 keeps its fixed physical layout.
        auto tensorL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL0A>(
                     mActual, kActual),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL0B>(
                     kActual, 128),
            Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(mActual, 128), Catlass::Arch::PositionL0C{});
        auto outputTensor = tla::MakeTensor(
            output[outputOffset],
            tla::MakeLayout<MainT, LayoutRM>(meta.validTokens, 128),
            Catlass::Arch::PositionGM{});
        auto tileL1A = tla::GetTile(
            tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kActual));
        auto tileL1B = tla::GetTile(
            tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(kActual, 128));

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
        OutputCopyL1ToL0A{}(tensorL0A, tileL1A);
        OutputCopyL1ToL0B{}(tensorL0B, tileL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Slot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0Slot);
        TileMmad{}(tensorL0C, tensorL0A, tensorL0B,
                   mActual, 128U, kActual, true, static_cast<uint8_t>(0));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0Slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0Slot);
        OutputCopyL0CToGm<decltype(outputTensor)>{}(outputTensor, tensorL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0Slot);
    }

    template <bool FULL_TOKENS>
    __aicore__ inline void RunStage2Impl(const ChunkGdnBwdIntraWorkMeta &meta)
    {
        PrefetchStage2Vectors(meta);
        if (meta.validHeads > 0) {
            LoadStage2Workspace(meta, 0);
        }
        bool vectorsReady = false;
        for (uint32_t r = 0; r < static_cast<uint32_t>(meta.validHeads); ++r) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(r);
            if (!vectorsReady) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(VDO_READY_EVENT);
                vectorsReady = true;
            }
            // Keep MTE2 one head ahead of the current MTE1/Cube/Fixpipe work.
            if (r + 1 < static_cast<uint32_t>(meta.validHeads)) {
                LoadStage2Workspace(meta, r + 1);
            }
            const uint32_t leader = static_cast<uint32_t>(ChunkGdnBwdIntraLeaderR(
                meta.hvBegin, r, tiling_->headRatio));
            auto l1ABg = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_A_BG_BASE + r * L1_MATRIX_SLOT_BYTES);
            auto l1ABeta = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_A_BETA_BASE + r * L1_MATRIX_SLOT_BYTES);
            auto l1D = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_D_BASE + r * L1_MATRIX_SLOT_BYTES);
            auto l1K = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_K_BASE + leader * L1_SLOT_BYTES);
            auto l1V = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_V_BASE + r * L1_SLOT_BYTES);
            auto l1DO = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_DO_BASE + r * L1_SLOT_BYTES);
            const uint64_t hv = static_cast<uint64_t>(meta.hvBegin + r);
            const uint64_t outputOffset =
                ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                     tiling_->seqlen + meta.tokenStart) * tiling_->valueDim;
            // Continue ping/pong across head boundaries as well as within one head.
            const uint32_t firstL0Slot = r & 1U;
            RunStage2Mmad<FULL_TOKENS>(
                meta, firstL0Slot, l1D, l1DO, dvLocal_, outputOffset);
            RunStage2Mmad<FULL_TOKENS>(
                meta, firstL0Slot ^ 1U, l1ABg, l1K, w_, outputOffset);
            RunStage2Mmad<FULL_TOKENS>(
                meta, firstL0Slot, l1ABeta, l1V, u_, outputOffset);
        }
    }

    __aicore__ inline void RunStage2(const ChunkGdnBwdIntraWorkMeta &meta)
    {
        if (meta.validTokens == tiling_->chunkSize) {
            RunStage2Impl<true>(meta);
        } else {
            RunStage2Impl<false>(meta);
        }
    }

    Catlass::Arch::Resource<ArchTag> resource_;
    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;
    AscendC::GlobalTensor<MainT> q_;
    AscendC::GlobalTensor<MainT> k_;
    AscendC::GlobalTensor<MainT> v_;
    AscendC::GlobalTensor<MainT> dO_;
    AscendC::GlobalTensor<MainT> w_;
    AscendC::GlobalTensor<MainT> u_;
    AscendC::GlobalTensor<MainT> dvLocal_;
    AscendC::GlobalTensor<MainT> workspace_;
    int64_t coreIdx_ = 0;
};

template <typename MainT, typename GateT, typename BetaT>
class ChunkGdnBwdIntraStage0Vector {
public:
    using ArchTag = Catlass::Arch::Ascend950;

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR g, GM_ADDR beta, GM_ADDR w, GM_ADDR u,
        GM_ADDR dvLocal, GM_ADDR workspace, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGdnBwdIntraTilingData *__restrict tiling)
    {
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(a));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(w));
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(u));
        dvLocal_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dvLocal));
        if (tiling->stage != 0) {
            workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(workspace));
        }
        mapper_.Init(cuSeqlens, chunkIndices, tiling);
        tiling_ = tiling;
        const int64_t subblocks = static_cast<int64_t>(AscendC::GetSubBlockNum());
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx()) / subblocks;
        part_ = static_cast<int64_t>(AscendC::GetSubBlockIdx());
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
    }

    __aicore__ inline void Process()
    {
        InitializeRawSlots();
        bool workspaceUsed[INTRA_CG_MAX] = {false, false, false, false};
        for (int64_t work = coreIdx_; work < tiling_->workCount;
             work += static_cast<int64_t>(AscendC::GetBlockNum())) {
            ChunkGdnBwdIntraWorkMeta meta{};
            mapper_.Resolve(work, meta);
            if (tiling_->stage != 0 && meta.validHeads > 0) {
                PrefetchStage1Input(meta, 0);
            }
            for (int64_t r = 0; r < meta.validHeads; ++r) {
                // Queue the next head before VF consumes the current ping/pong slot.
                if (tiling_->stage != 0 && r + 1 < meta.validHeads) {
                    PrefetchStage1Input(meta, r + 1);
                }
                const int64_t leader = ChunkGdnBwdIntraLeaderR(
                    meta.hvBegin, r, tiling_->headRatio);
                if (r == leader) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                        INTRA_SCORE_READY_FLAG + static_cast<uint32_t>(leader));
                }
                if (tiling_->stage == 0) {
                    WriteDebugScore(meta, r, leader);
                } else {
                    RunStage1(meta, r, leader, workspaceUsed[r]);
                    workspaceUsed[r] = true;
                }
                const bool lastConsumer = r + 1 == meta.validHeads ||
                    (meta.hvBegin + r) / tiling_->headRatio !=
                        (meta.hvBegin + r + 1) / tiling_->headRatio;
                if (lastConsumer) {
                    if (tiling_->stage == 0) {
                        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(
                            INTRA_SCORE_FREE_FLAG + static_cast<uint32_t>(leader));
                    } else {
                        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                            INTRA_SCORE_FREE_FLAG + static_cast<uint32_t>(leader));
                    }
                }
            }
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
    }

private:
    __aicore__ inline AscendC::LocalTensor<uint8_t> UbBuffer() const
    {
        return resource_.ubBuf.template GetBufferByByte<uint8_t>(0);
    }

    __aicore__ inline void InitializeRawSlots()
    {
        constexpr uint32_t ROW_ELEMENTS = 64;
        for (uint32_t r = 0; r < INTRA_CG_MAX; ++r) {
            auto gLocal = UbBuffer()[INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES]
                              .template ReinterpretCast<GateT>();
            auto betaLocal = UbBuffer()[INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES +
                                        INTRA_RAW_BETA_OFFSET]
                                 .template ReinterpretCast<BetaT>();
            AscendC::Duplicate(gLocal, static_cast<GateT>(0), ROW_ELEMENTS);
            AscendC::Duplicate(betaLocal, static_cast<BetaT>(0), ROW_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    }

    __aicore__ inline void PrefetchStage1Input(
        const ChunkGdnBwdIntraWorkMeta &meta, int64_t r)
    {
        constexpr int64_t PART_ROWS = 32;
        constexpr int64_t ROW_ELEMENTS = 64;
        const int64_t rowBegin = part_ * PART_ROWS;
        int64_t rows = meta.validTokens - rowBegin;
        rows = rows < 0 ? 0 : (rows > PART_ROWS ? PART_ROWS : rows);
        if (rows == 0) {
            return;
        }

        const uint32_t slot = static_cast<uint32_t>(r & 1);
        auto aLocal = UbBuffer()[INTRA_A_UB_BASE + slot * INTRA_MATRIX_HALF_BYTES]
                          .template ReinterpretCast<MainT>();
        auto gLocal = UbBuffer()[INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES]
                          .template ReinterpretCast<GateT>();
        auto betaLocal = UbBuffer()[INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES +
                                    INTRA_RAW_BETA_OFFSET]
                             .template ReinterpretCast<BetaT>();
        const int64_t hv = meta.hvBegin + r;
        const uint64_t matrixOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                 tiling_->seqlen + meta.tokenStart + rowBegin) * tiling_->chunkSize;
        const uint64_t vectorOffset =
            (static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                tiling_->seqlen + meta.tokenStart;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
        AscendC::DataCopy(aLocal, a_[matrixOffset],
                          static_cast<uint32_t>(rows * ROW_ELEMENTS));
        AscendC::DataCopyExtParams vectorCopy{
            1, static_cast<uint32_t>(meta.validTokens * sizeof(GateT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<GateT> gatePad{false, 0, 0, 0};
        AscendC::DataCopyPad(gLocal, g_[vectorOffset], vectorCopy, gatePad);
        vectorCopy.blockLen = static_cast<uint32_t>(meta.validTokens * sizeof(BetaT));
        AscendC::DataCopyPadExtParams<BetaT> betaPad{false, 0, 0, 0};
        AscendC::DataCopyPad(betaLocal, beta_[vectorOffset], vectorCopy, betaPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
    }

    __aicore__ inline void WriteDebugScore(
        const ChunkGdnBwdIntraWorkMeta &meta, int64_t r, int64_t leader)
    {
        constexpr int64_t PART_ROWS = 32;
        const int64_t rowBegin = part_ * PART_ROWS;
        int64_t rows = meta.validTokens - rowBegin;
        if (rows <= 0) {
            return;
        }
        if (rows > PART_ROWS) {
            rows = PART_ROWS;
        }
        auto score = UbBuffer()[INTRA_SCORE_UB_BASE +
                                static_cast<uint32_t>(leader) * INTRA_SCORE_SLOT_BYTES]
                         .template ReinterpretCast<float>();
        auto output = UbBuffer()[INTRA_STAGE0_DEBUG_UB_BASE].template ReinterpretCast<MainT>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::Cast(output, score, AscendC::RoundMode::CAST_RINT,
                      static_cast<uint32_t>(rows * 64));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        const int64_t hv = meta.hvBegin + r;
        const uint64_t gmOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                 tiling_->seqlen + meta.tokenStart + rowBegin) * tiling_->keyDim;
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(64 * sizeof(MainT)),
            0, static_cast<uint32_t>(64 * sizeof(MainT)), 0};
        AscendC::DataCopyPad(w_[gmOffset], output, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    template <bool USE_EXP2>
    __aicore__ inline void InvokeStage1VF(
        AscendC::LocalTensor<MainT> aBgLocal,
        AscendC::LocalTensor<MainT> aBetaLocal,
        AscendC::LocalTensor<MainT> dLocal,
        AscendC::LocalTensor<MainT> aLocal,
        AscendC::LocalTensor<float> score,
        AscendC::LocalTensor<GateT> gLocal,
        AscendC::LocalTensor<BetaT> betaLocal,
        uint16_t rows, uint16_t rowBegin, uint16_t validTokens)
    {
        ChunkGdnBwdIntraStage1VF<MainT, GateT, BetaT, USE_EXP2>(
            reinterpret_cast<__ubuf__ MainT *>(aBgLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ MainT *>(aBetaLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ MainT *>(dLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ MainT *>(aLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(score.GetPhyAddr()),
            reinterpret_cast<__ubuf__ GateT *>(gLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ GateT *>(gLocal.GetPhyAddr()) + rowBegin,
            reinterpret_cast<__ubuf__ BetaT *>(betaLocal.GetPhyAddr()),
            rows, rowBegin, validTokens, tiling_->scale);
    }

    __aicore__ inline void RunStage1(
        const ChunkGdnBwdIntraWorkMeta &meta, int64_t r, int64_t leader,
        bool workspaceUsed)
    {
        constexpr int64_t PART_ROWS = 32;
        constexpr int64_t ROW_ELEMENTS = 64;
        const uint32_t slot = static_cast<uint32_t>(r & 1);
        const int64_t rowBegin = part_ * PART_ROWS;
        int64_t rows = meta.validTokens - rowBegin;
        rows = rows < 0 ? 0 : (rows > PART_ROWS ? PART_ROWS : rows);

        if (rows > 0) {
            auto aLocal = UbBuffer()[INTRA_A_UB_BASE + slot * INTRA_MATRIX_HALF_BYTES]
                              .template ReinterpretCast<MainT>();
            auto aBgLocal = UbBuffer()[INTRA_A_BG_UB_BASE + slot * INTRA_MATRIX_HALF_BYTES]
                                .template ReinterpretCast<MainT>();
            auto aBetaLocal = UbBuffer()[INTRA_A_BETA_UB_BASE + slot * INTRA_MATRIX_HALF_BYTES]
                                  .template ReinterpretCast<MainT>();
            auto dLocal = UbBuffer()[INTRA_D_UB_BASE + slot * INTRA_MATRIX_HALF_BYTES]
                              .template ReinterpretCast<MainT>();
            auto gLocal = UbBuffer()[INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES]
                              .template ReinterpretCast<GateT>();
            auto betaLocal = UbBuffer()[INTRA_RAW_UB_BASE + r * INTRA_RAW_SLOT_BYTES +
                                        INTRA_RAW_BETA_OFFSET]
                                 .template ReinterpretCast<BetaT>();
            auto score = UbBuffer()[INTRA_SCORE_UB_BASE +
                                    static_cast<uint32_t>(leader) * INTRA_SCORE_SLOT_BYTES]
                             .template ReinterpretCast<float>();

            const int64_t hv = meta.hvBegin + r;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);

            if (tiling_->useExp2 != 0) {
                InvokeStage1VF<true>(
                    aBgLocal, aBetaLocal, dLocal, aLocal, score, gLocal, betaLocal,
                    static_cast<uint16_t>(rows), static_cast<uint16_t>(rowBegin),
                    static_cast<uint16_t>(meta.validTokens));
            } else {
                InvokeStage1VF<false>(
                    aBgLocal, aBetaLocal, dLocal, aLocal, score, gLocal, betaLocal,
                    static_cast<uint16_t>(rows), static_cast<uint16_t>(rowBegin),
                    static_cast<uint16_t>(meta.validTokens));
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);

            if (workspaceUsed) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(
                    INTRA_WORKSPACE_FREE_FLAG + static_cast<uint32_t>(r) +
                    static_cast<uint32_t>(part_) * INTRA_SUBBLOCK_FLAG_OFFSET);
            }
            const uint64_t rho = static_cast<uint64_t>(coreIdx_) * tiling_->cg + r;
            const uint64_t rowOffset = static_cast<uint64_t>(part_) * PART_ROWS * ROW_ELEMENTS;
            const uint64_t matrixStride = tiling_->matrixStrideBytes / sizeof(MainT);
            const uint64_t aBgOffset = tiling_->aBgWorkspaceOffset / sizeof(MainT) +
                                       rho * matrixStride + rowOffset;
            const uint64_t aBetaOffset = tiling_->aBetaWorkspaceOffset / sizeof(MainT) +
                                         rho * matrixStride + rowOffset;
            const uint64_t dOffset = tiling_->dWorkspaceOffset / sizeof(MainT) +
                                     rho * matrixStride + rowOffset;
            const uint32_t halfElements = static_cast<uint32_t>(rows * ROW_ELEMENTS);
            AscendC::DataCopy(workspace_[dOffset], dLocal, halfElements);
            AscendC::DataCopy(workspace_[aBgOffset], aBgLocal, halfElements);
            AscendC::DataCopy(workspace_[aBetaOffset], aBetaLocal, halfElements);
            if (tiling_->stage == 1) {
                WriteDebugMatrix(w_, meta, hv, rowBegin, rows, aBgLocal);
                WriteDebugMatrix(u_, meta, hv, rowBegin, rows, aBetaLocal);
                WriteDebugMatrix(dvLocal_, meta, hv, rowBegin, rows, dLocal);
            }
        } else if (workspaceUsed) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE3>(
                INTRA_WORKSPACE_FREE_FLAG + static_cast<uint32_t>(r) +
                static_cast<uint32_t>(part_) * INTRA_SUBBLOCK_FLAG_OFFSET);
        }

        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
            INTRA_WORKSPACE_READY_FLAG + static_cast<uint32_t>(r) +
            static_cast<uint32_t>(part_) * INTRA_SUBBLOCK_FLAG_OFFSET);
        if (rows > 0) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
        }
    }

    __aicore__ inline void WriteDebugMatrix(
        AscendC::GlobalTensor<MainT> &dst,
        const ChunkGdnBwdIntraWorkMeta &meta, int64_t hv,
        int64_t rowBegin, int64_t rows, AscendC::LocalTensor<MainT> src)
    {
        const uint64_t gmOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads + hv) *
                 tiling_->seqlen + meta.tokenStart + rowBegin) * tiling_->valueDim;
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(64 * sizeof(MainT)),
            0, static_cast<uint32_t>(64 * sizeof(MainT)), 0};
        AscendC::DataCopyPad(dst[gmOffset], src, params);
    }

    Catlass::Arch::Resource<ArchTag> resource_;
    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;
    AscendC::GlobalTensor<MainT> a_;
    AscendC::GlobalTensor<GateT> g_;
    AscendC::GlobalTensor<BetaT> beta_;
    AscendC::GlobalTensor<MainT> w_;
    AscendC::GlobalTensor<MainT> u_;
    AscendC::GlobalTensor<MainT> dvLocal_;
    AscendC::GlobalTensor<MainT> workspace_;
    int64_t coreIdx_ = 0;
    int64_t part_ = 0;
};

} // namespace GDN

#endif // CHUNK_GDN_BWD_INTRA_STAGE0_H
