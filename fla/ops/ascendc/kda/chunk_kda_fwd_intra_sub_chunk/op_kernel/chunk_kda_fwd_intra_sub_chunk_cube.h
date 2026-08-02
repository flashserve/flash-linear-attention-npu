/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaFwdIntraSubChunk — Cube (AIC) side.
 *
 * Tile-level dual score GEMM (NOT BlockMmad):
 *   Aqk_raw = Qg @ Kg.T ;  Akk_raw = W(=Kgq) @ Kg.T
 *   - Kg → L1B resident across both MMADs
 *   - L1A ping-pong (PR190 scratch[2]): l1A[0]=Qg, l1A[1]=W
 *       MTE2(W) ‖ MMAD1(Qg@Kg); Wait W ready before Fixpipe(Aqk)
 *   - Do NOT use single-L1A overwrite (USE_SCORE_MMAD1_LOAD_W) — multi-HV aqk flake
 *   - C1: Akk Fix ‖ next-tile MTE2 (USE_SCORE_FIX_MTE2_DBUF); drain before SetCubeDone
 *   - C2: window dual-head L1 resident (USE_SCORE_WIN_L1_RESIDENT)
 *
 * NO MCH — (I+L)^{-1} is Vector Forward Substitution.
 * Plan: SCORE_TILE_DBUF_PLAN.md P1 / Cube Fix-MTE2 plan C1–C2.
 */

#ifndef CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H
#define CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H

#include "chunk_kda_fwd_intra_sub_chunk_common.h"

namespace kda_isub {

// P1: dual L1A slots. Default on for Vec2Win.
#ifndef USE_SCORE_L1A_DBUF
#define USE_SCORE_L1A_DBUF 1
#endif
// Legacy single-L1A W‖MMAD — forced off when L1A_DBUF is on.
#ifndef USE_SCORE_MMAD1_LOAD_W
#define USE_SCORE_MMAD1_LOAD_W 0
#endif
#if USE_SCORE_L1A_DBUF
#undef USE_SCORE_MMAD1_LOAD_W
#define USE_SCORE_MMAD1_LOAD_W 0
#endif

// C1: delay Wait(FIX_MTE2) after Akk so next tile MTE2 can overlap Fixpipe.
#ifndef USE_SCORE_FIX_MTE2_DBUF
#define USE_SCORE_FIX_MTE2_DBUF 1
#endif

// C2: WaitS0 bulk-load both heads into L1; Score skips per-head GM→L1.
// Default off: precision fail (aqk_err≈14); also weakens P1/C1 (see CUBE_OPTIMAL_PIPELINE path A).
#ifndef USE_SCORE_WIN_L1_RESIDENT
#define USE_SCORE_WIN_L1_RESIDENT 0
#endif
#if USE_SCORE_WIN_L1_RESIDENT && !USE_SCORE_L1A_DBUF
#error "USE_SCORE_WIN_L1_RESIDENT requires USE_SCORE_L1A_DBUF"
#endif

template <typename T>
class KdaSubChunkCube : public KdaSubChunkBase<T> {
    using Base = KdaSubChunkBase<T>;
    using Base::bc_;
    using Base::kDim_;
    using Base::hv_;
    using Base::nc_;
    using Base::totalTasks_;
    using Base::usedCoreNum_;
    using Base::coreIdx_;
    using Base::scoreWs_;
    using Base::cmatWs_;

public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR aqk, GM_ADDR akkd, GM_ADDR userWS,
                                const ChunkKdaFwdIntraSubChunkTilingData &tiling, TPipe *pipe)
    {
        (void)pipe;
        this->InitCommon(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tiling);
        coreIdx_ = static_cast<uint64_t>(GetBlockIdx());
#if USE_SCORE_FIX_MTE2_DBUF
        akkFixPending_ = false;
#endif
    }

    __aicore__ inline void Process()
    {
        if (!this->ValidShapes()) {
            return;
        }
        Catlass::Arch::Resource<KdaArchTag> resource;
        uint64_t tasksOnCore = 0;
        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            const uint64_t nHvWin = this->NumHvWindows();
            const uint64_t W = nc_ * nHvWin;
            if (W == 0) {
                continue;
            }
            ++tasksOnCore;
            for (uint64_t w = 0; w < W; ++w) {
                const uint64_t slot0 = this->SlotOfWindow(w, 0);
                const uint64_t slot1 = this->SlotOfWindow(w, 1);
                const uint64_t hvBase = (w % nHvWin) * 2ULL;
                const bool twoHeads = (hvBase + 1ULL < hv_);
                Catlass::Arch::CrossCoreWaitFlag(s0Ready_);
#if USE_SCORE_WIN_L1_RESIDENT
                PrefetchWindowToL1(slot0, twoHeads ? slot1 : slot0, twoHeads, resource);
                ComputeScoreTileFromL1(/*headIdx=*/0, slot0, resource);
                if (twoHeads) {
                    ComputeScoreTileFromL1(/*headIdx=*/1, slot1, resource);
                }
#else
                ComputeScoreTile(slot0, resource);
                if (twoHeads) {
                    ComputeScoreTile(slot1, resource);
                }
#endif
#if USE_SCORE_FIX_MTE2_DBUF
                DrainAkkFix();
#endif
                PipeBarrier<PIPE_FIX>();
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeDone_);
            }
        }
        if (tasksOnCore > 0) {
            for (uint32_t s = 0; s < NUM_GM_SLOTS; ++s) {
                Catlass::Arch::CrossCoreWaitFlag(slotFree_[s]);
            }
        }
    }

private:
    using ElementA = T;
    using ElementB = T;
    using ElementC = float;
    using LayoutTagA = Catlass::layout::RowMajor;
    using LayoutTagB = Catlass::layout::ColumnMajor;
    using LayoutTagC = Catlass::layout::RowMajor;
    using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
                                                            ElementC, LayoutTagC>;

    static constexpr uint16_t SCORE_EVT = 3;
    static constexpr uint16_t SCORE_EVT_W = 4;
#if USE_SCORE_FIX_MTE2_DBUF
    static constexpr uint16_t SCORE_EVT_FIX = 5; // Akk Fix ‖ next MTE2 (clear of 3/4)
#endif

#if USE_SCORE_WIN_L1_RESIDENT
    // Per-head: Qg | W | Kg, stride = logical tile bytes (same as P1 l1ABytes).
    // Must NOT use 8KiB — BC*K*bf16 = 16KiB; smaller stride overlaps Qg/W/Kg.
    static constexpr uint32_t kHeadL1Slots = 3;
#endif

#if USE_SCORE_FIX_MTE2_DBUF
    __aicore__ inline void DrainAkkFix()
    {
        if (akkFixPending_) {
            WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT_FIX);
            akkFixPending_ = false;
        }
    }
#endif

    __aicore__ inline void ComputeScoreTile(uint64_t slot, Catlass::Arch::Resource<KdaArchTag> &resource)
    {
        const uint32_t m = static_cast<uint32_t>(bc_);
        const uint32_t n = static_cast<uint32_t>(bc_);
        const uint32_t k = static_cast<uint32_t>(kDim_);
        Catlass::GemmCoord shape{m, n, k};

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(bc_, kDim_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(kDim_, bc_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(bc_, bc_);

        auto tensorQg =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_QG, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_W, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorKg =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_KG, 0, 0)], layoutB, Catlass::Arch::PositionGM{});
        auto tensorAqk =
            tla::MakeTensor(cmatWs_[this->CmatOff(slot, PLANE_AQK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorAkk =
            tla::MakeTensor(cmatWs_[this->CmatOff(slot, PLANE_AKK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});

        auto blockQg = GetTile(tensorQg, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKg = GetTile(tensorKg, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));

        const uint32_t l1ABytes = m * k * sizeof(ElementA);
#if USE_SCORE_L1A_DBUF
        LocalTensor<ElementA> l1A0 = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementA> l1A1 = resource.l1Buf.template GetBufferByByte<ElementA>(l1ABytes);
        LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(2 * l1ABytes);
#else
        LocalTensor<ElementA> l1A = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(l1ABytes);
#endif
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementC> l0C = resource.l0CBuf.template GetBufferByByte<ElementC>(0);

        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockQg)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockKg)>;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(blockAqk)>;
#else
        using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<decltype(blockAqk)>;
#endif
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(m, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);
        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto layoutL0C = tla::MakeLayoutL0C(m, n);

#if USE_SCORE_L1A_DBUF
        auto tL1A0 = tla::MakeTensor(l1A0, layoutL1A, Catlass::Arch::PositionL1{});
        auto tL1A1 = tla::MakeTensor(l1A1, layoutL1A, Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        auto tL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A0 = GetTile(tL1A0, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1A1 = GetTile(tL1A1, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = GetTile(tL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
#else
        auto tL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        auto tL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A = GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = GetTile(tL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
#endif

        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;
        TileMmad tileMmad;

#if USE_SCORE_L1A_DBUF
        // --- P1 + C1: Kg MTE2 may overlap previous Akk Fix; Wait W before Fix(Aqk) ---
#if USE_SCORE_FIX_MTE2_DBUF
        if (akkFixPending_) {
            copyGmToL1B(tL1B, blockKg);
            SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
            WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT_FIX);
            akkFixPending_ = false;
            WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        } else {
            copyGmToL1B(tL1B, blockKg);
            SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
            WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        }
#else
        copyGmToL1B(tL1B, blockKg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
#endif
        copyGmToL1A(tL1A0, blockQg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);

        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A0);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        copyGmToL1A(tL1A1, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT_W);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        SetFlag<HardEvent::M_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT_W); // ★ before Fix(Aqk) — precision redline
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_MTE1>(SCORE_EVT);
        copyL0CToGm(blockAqk, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        // C1: MTE1(MMAD2) ‖ Fix(Aqk); Wait FIX before MMAD2 touches L0C.
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A1);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        copyL0CToGm(blockAkk, tL0C);
#if USE_SCORE_FIX_MTE2_DBUF
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT_FIX);
        akkFixPending_ = true;
#else
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
#endif
#else
        // --- Serial single-L1A fallback ---
#if USE_SCORE_FIX_MTE2_DBUF
        if (akkFixPending_) {
            copyGmToL1B(tL1B, blockKg);
            SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
            WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT_FIX);
            akkFixPending_ = false;
            WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        } else {
            copyGmToL1B(tL1B, blockKg);
            SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
            WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        }
#else
        copyGmToL1B(tL1B, blockKg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
#endif
        copyGmToL1A(tL1A, blockQg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        SetFlag<HardEvent::M_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::M_MTE1>(SCORE_EVT);
        copyL0CToGm(blockAqk, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);

        copyGmToL1A(tL1A, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        copyL0CToGm(blockAkk, tL0C);
#if USE_SCORE_FIX_MTE2_DBUF
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT_FIX);
        akkFixPending_ = true;
#else
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
#endif
#endif
    }

#if USE_SCORE_WIN_L1_RESIDENT
    __aicore__ inline uint32_t HeadL1BaseBytes(uint32_t headIdx, uint32_t tileBytes) const
    {
        return headIdx * kHeadL1Slots * tileBytes;
    }

    __aicore__ inline void PrefetchOneHeadToL1(uint32_t headIdx, uint64_t slot,
                                               Catlass::Arch::Resource<KdaArchTag> &resource)
    {
        const uint32_t m = static_cast<uint32_t>(bc_);
        const uint32_t n = static_cast<uint32_t>(bc_);
        const uint32_t k = static_cast<uint32_t>(kDim_);
        const uint32_t tileBytes = m * k * sizeof(ElementA);
        const uint32_t base = HeadL1BaseBytes(headIdx, tileBytes);

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(bc_, kDim_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(kDim_, bc_);
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        LocalTensor<ElementA> l1A0 = resource.l1Buf.template GetBufferByByte<ElementA>(base);
        LocalTensor<ElementA> l1A1 =
            resource.l1Buf.template GetBufferByByte<ElementA>(base + tileBytes);
        LocalTensor<ElementB> l1B =
            resource.l1Buf.template GetBufferByByte<ElementB>(base + 2 * tileBytes);
        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(m, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);
        auto tL1A0 = tla::MakeTensor(l1A0, layoutL1A, Catlass::Arch::PositionL1{});
        auto tL1A1 = tla::MakeTensor(l1A1, layoutL1A, Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});

        auto tensorQg =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_QG, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_W, 0, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorKg =
            tla::MakeTensor(scoreWs_[this->ScoreOff(slot, PLANE_KG, 0, 0)], layoutB, Catlass::Arch::PositionGM{});
        auto blockQg = GetTile(tensorQg, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockKg = GetTile(tensorKg, tla::MakeCoord(0, 0), tla::MakeShape(k, n));

        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockQg)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockKg)>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;

        copyGmToL1B(tL1B, blockKg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        copyGmToL1A(tL1A0, blockQg);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        copyGmToL1A(tL1A1, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::MTE2_MTE1>(SCORE_EVT);
    }

    __aicore__ inline void PrefetchWindowToL1(uint64_t slot0, uint64_t slot1, bool twoHeads,
                                              Catlass::Arch::Resource<KdaArchTag> &resource)
    {
        // 2 heads × 3 × tileBytes (16KiB @ BC=64,K=128) ≪ 512KiB.
        // DrainAkkFix runs after both heads, so Prefetch never sees akkFixPending.
        PrefetchOneHeadToL1(0, slot0, resource);
        if (twoHeads) {
            PrefetchOneHeadToL1(1, slot1, resource);
        }
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeScoreTileFromL1(uint32_t headIdx, uint64_t cmatSlot,
                                                  Catlass::Arch::Resource<KdaArchTag> &resource)
    {
        const uint32_t m = static_cast<uint32_t>(bc_);
        const uint32_t n = static_cast<uint32_t>(bc_);
        const uint32_t k = static_cast<uint32_t>(kDim_);
        const uint32_t tileBytes = m * k * sizeof(ElementA);
        const uint32_t base = HeadL1BaseBytes(headIdx, tileBytes);

        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(bc_, bc_);
        auto tensorAqk =
            tla::MakeTensor(cmatWs_[this->CmatOff(cmatSlot, PLANE_AQK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorAkk =
            tla::MakeTensor(cmatWs_[this->CmatOff(cmatSlot, PLANE_AKK, 0, 0)], layoutC, Catlass::Arch::PositionGM{});
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        LocalTensor<ElementA> l1A0 = resource.l1Buf.template GetBufferByByte<ElementA>(base);
        LocalTensor<ElementA> l1A1 =
            resource.l1Buf.template GetBufferByByte<ElementA>(base + tileBytes);
        LocalTensor<ElementB> l1B =
            resource.l1Buf.template GetBufferByByte<ElementB>(base + 2 * tileBytes);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementC> l0C = resource.l0CBuf.template GetBufferByByte<ElementC>(0);

        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToGm = typename TileCopy::template CopyL0CToDst<decltype(blockAqk)>;
#else
        using CopyL0CToGm = typename TileCopy::template CopyL0CToGm<decltype(blockAqk)>;
#endif
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(m, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);
        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto layoutL0C = tla::MakeLayoutL0C(m, n);

        auto tL1A0 = tla::MakeTensor(l1A0, layoutL1A, Catlass::Arch::PositionL1{});
        auto tL1A1 = tla::MakeTensor(l1A1, layoutL1A, Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        auto tL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A0 = GetTile(tL1A0, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1A1 = GetTile(tL1A1, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = GetTile(tL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;
        TileMmad tileMmad;

#if USE_SCORE_FIX_MTE2_DBUF
        if (akkFixPending_) {
            WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT_FIX);
            akkFixPending_ = false;
        }
#endif
        // Prefetch completed Kg/Qg/W (W ready before Fix — P1 redline).
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A0);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        SetFlag<HardEvent::M_MTE1>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_MTE1>(SCORE_EVT);
        copyL0CToGm(blockAqk, tL0C);
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A1);
        SetFlag<HardEvent::MTE1_M>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::MTE1_M>(SCORE_EVT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(SCORE_EVT);
        WaitFlag<HardEvent::M_FIX>(SCORE_EVT);
        copyL0CToGm(blockAkk, tL0C);
#if USE_SCORE_FIX_MTE2_DBUF
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT_FIX);
        akkFixPending_ = true;
#else
        SetFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
        WaitFlag<HardEvent::FIX_MTE2>(SCORE_EVT);
#endif
    }
#endif // USE_SCORE_WIN_L1_RESIDENT

    Catlass::Arch::CrossCoreFlag s0Ready_{FLAG_S0_READY};
    Catlass::Arch::CrossCoreFlag cubeDone_{FLAG_CUBE_DONE};
    Catlass::Arch::CrossCoreFlag slotFree_[NUM_GM_SLOTS] = {FLAG_SLOT_FREE0, FLAG_SLOT_FREE1, FLAG_SLOT_FREE2,
                                                             FLAG_SLOT_FREE3};
#if USE_SCORE_FIX_MTE2_DBUF
    bool akkFixPending_ = false;
#endif
};

} // namespace kda_isub

#endif // CHUNK_KDA_FWD_INTRA_SUB_CHUNK_CUBE_H
