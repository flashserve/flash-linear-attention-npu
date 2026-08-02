/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * ChunkKdaFwdIntraSubChunk — Vector (AIV) side.
 *   PrepareSub : gate (fp32) → Qg / W / Kg (cast to qk dtype) → scoreWs
 *   PostSub    : tril(Aqk)*scale ; L = strict_tril(Akk)*beta ;
 *                Akkd = (I+L)^{-1} via Forward Substitution ; store diag block + Akkd
 *
 * Vec 2-win dual-issue (VEC_2WIN_PIPE.md): AIV0/AIV1 by head; S0 prefill=2;
 * SetS0ReadyJoined (Barrier before Set); SetFree Process bookend only; raw 0x2.
 */

#ifndef CHUNK_KDA_FWD_INTRA_SUB_CHUNK_VECTOR_H
#define CHUNK_KDA_FWD_INTRA_SUB_CHUNK_VECTOR_H

#include "chunk_kda_fwd_intra_sub_chunk_common.h"

// Shipped Vector defaults (verified board path, Dur med 2.075ms @ model case):
//   - MTE2 merge: S0 mid‖q/k/g ; Post cmat‖beta (single Wait each)
//   - FwdSub: row-broadcast Mul (no brcd) + Add-fold; one PipeBarrier before fold
// Rejected experiments removed: partial Mul/fold, resident mask/+I, Post‖S0 MTE defer.

namespace kda_isub {

template <typename T>
class KdaSubChunkVector : public KdaSubChunkBase<T> {
    using Base = KdaSubChunkBase<T>;
    using Base::bc_;
    using Base::bt_;
    using Base::kDim_;
    using Base::hv_;
    using Base::nc_;
    using Base::group_;
    using Base::totalTasks_;
    using Base::usedCoreNum_;
    using Base::coreIdx_;
    using Base::scale_;
    using Base::hasVarlen_;
    using Base::q_;
    using Base::k_;
    using Base::g_;
    using Base::beta_;
    using Base::aqk_;
    using Base::akkd_;
    using Base::scoreWs_;
    using Base::cmatWs_;

public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR g, GM_ADDR beta, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                GM_ADDR aqk, GM_ADDR akkd, GM_ADDR userWS,
                                const ChunkKdaFwdIntraSubChunkTilingData &tiling, TPipe *pipe)
    {
        this->InitCommon(q, k, g, beta, cuSeqlens, chunkIndices, aqk, akkd, userWS, tiling);
        pipe_ = pipe;
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        coreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        subBlockIdx_ = static_cast<uint64_t>(GetSubBlockIdx());

        pipe_->InitBuffer(vecBuf_, MAX_BC * MAX_K * 6 * sizeof(float));
        pipe_->InitBuffer(midBuf_, MAX_K * sizeof(float));
        pipe_->InitBuffer(betaBuf_, MAX_BC * sizeof(float));
        pipe_->InitBuffer(aqkBuf_, MAX_BC * MAX_BC * sizeof(float));
        pipe_->InitBuffer(akkBuf_, MAX_BC * MAX_BC * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, MAX_BC * sizeof(float));
        pipe_->InitBuffer(inBuf_, MAX_K * sizeof(T));
        pipe_->InitBuffer(aqkTBuf_, MAX_BC * MAX_BC * sizeof(T));
        pipe_->InitBuffer(zeroBuf_, MAX_BC * MAX_K * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (!this->ValidShapes()) {
            return;
        }
#ifndef KDA_ISUB_PREFILL_WINDOWS
#define KDA_ISUB_PREFILL_WINDOWS 2
#endif
        constexpr uint64_t prefillCap = static_cast<uint64_t>(KDA_ISUB_PREFILL_WINDOWS);
        // SetFree×4 once per Process (wy_full). Per-task re-seed races multi-task cores.
        bool seeded = false;
        for (uint64_t task = coreIdx_; task < totalTasks_; task += usedCoreNum_) {
            uint64_t iB = 0, iChunk = 0;
            this->DecodeChunkTask(task, iB, iChunk);
            uint64_t bos = 0, localT = 0, localChunk = 0, bIdx = 0;
            this->ResolveChunkScalar(iChunk, iB, bos, localT, localChunk, bIdx);

            const uint64_t nHvWin = this->NumHvWindows();
            const uint64_t W = nc_ * nHvWin;
            if (W == 0) {
                continue;
            }

            if (!seeded) {
                Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                PipeBarrier<PIPE_MTE3>();
                for (uint32_t s = 0; s < NUM_GM_SLOTS; ++s) {
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(slotFree_[s]);
                }
                seeded = true;
            }

            const uint64_t prefill = (W < prefillCap) ? W : prefillCap;
            // Real-fill prefill windows; slot reuse via WaitCube(w) before S0(w+prefill).
            // Bank (w%2) free only after GEMM(w) done — do not S0(w+2) earlier.
            for (uint64_t w = 0; w < prefill; ++w) {
                RunS0Window(w, nHvWin, bIdx, bos, localT, localChunk);
                SetS0ReadyJoined();
            }
            for (uint64_t w = 0; w < W; ++w) {
                Catlass::Arch::CrossCoreWaitFlag(cubeDone_);
                // cmat GM visibility: AIC SetCubeDone is after DrainAkkFix + PIPE_FIX.
                RunPostWindow(w, nHvWin, bIdx, bos, localT, localChunk);
                if (w + prefill < W) {
                    // Safe: WaitCube(w) ⇒ score/cmat bank (w%2) no longer owned by Cube(w).
                    RunS0Window(w + prefill, nHvWin, bIdx, bos, localT, localChunk);
                    SetS0ReadyJoined();
                }
            }
        }
    }

private:
    // Barrier before Set so a fast AIV cannot double-SetS0 before the peer's first Set
    // (0x2 prefill skew → hang / later handshake corruption). VEC_2WIN_PIPE.md §5.1.
    __aicore__ inline void SetS0ReadyJoined()
    {
        Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
        PipeBarrier<PIPE_MTE3>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(s0Ready_);
    }

    // Dual-AIV by head: AIV0↔hvBase, AIV1↔hvBase+1. Empty head returns; caller still Sets.
    __aicore__ inline void RunS0Window(uint64_t w, uint64_t nHvWin, uint64_t bIdx, uint64_t bos, uint64_t localT,
                                       uint64_t localChunk)
    {
        const uint64_t iSub = w / nHvWin;
        const uint64_t hvBase = (w % nHvWin) * 2ULL;
        const uint64_t iHv = hvBase + subBlockIdx_;
        if (iHv >= hv_) {
            return;
        }
        const uint64_t iH = iHv / group_;
        PrepareSub(bIdx, iH, iHv, bos, localT, localChunk, iSub, this->SlotOfWindow(w, subBlockIdx_));
    }

    __aicore__ inline void RunPostWindow(uint64_t w, uint64_t nHvWin, uint64_t bIdx, uint64_t bos, uint64_t localT,
                                         uint64_t localChunk)
    {
        const uint64_t iSub = w / nHvWin;
        const uint64_t hvBase = (w % nHvWin) * 2ULL;
        const uint64_t iHv = hvBase + subBlockIdx_;
        if (iHv >= hv_) {
            return;
        }
        PostSub(bIdx, iHv, bos, localT, localChunk, iSub, this->SlotOfWindow(w, subBlockIdx_));
    }

    // -------------------- stage_0: Prepare Qg/W/Kg → scoreWs --------------------
    __aicore__ inline void ZeroScorePlanes(uint64_t slot, uint64_t rowBegin, uint64_t rowEnd)
    {
        if (rowBegin >= rowEnd) {
            return;
        }
        const uint32_t elems = static_cast<uint32_t>((rowEnd - rowBegin) * kDim_);
        LocalTensor<T> z = zeroBuf_.Get<T>();
        Duplicate(z, static_cast<T>(0), elems);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        this->CopyVectorOut(scoreWs_, this->ScoreOff(slot, PLANE_QG, rowBegin, 0), z, elems);
        this->CopyVectorOut(scoreWs_, this->ScoreOff(slot, PLANE_W, rowBegin, 0), z, elems);
        this->CopyVectorOut(scoreWs_, this->ScoreOff(slot, PLANE_KG, rowBegin, 0), z, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    __aicore__ inline void LoadMidRow(uint64_t off)
    {
        LocalTensor<float> mid = midBuf_.Get<float>();
        LocalTensor<T> in = inBuf_.Get<T>();
        this->CopyVectorIn(in, g_, off, kDim_);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Cast(mid, in, RoundMode::CAST_NONE, static_cast<uint32_t>(kDim_));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ClampExp(LocalTensor<float> &tensor, uint32_t count)
    {
        Mins(tensor, tensor, EXP_INPUT_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, EXP_INPUT_MIN, count);
    }
    __aicore__ inline void ClampFp16(LocalTensor<float> &tensor, uint32_t count)
    {
        if constexpr (IsSameType<T, half>::value) {
            Mins(tensor, tensor, FP16_MAX, count);
            PipeBarrier<PIPE_V>();
            Maxs(tensor, tensor, -FP16_MAX, count);
        }
    }

    __aicore__ inline void PrepareSub(uint64_t bIdx, uint64_t iH, uint64_t iHv, uint64_t bos, uint64_t localT,
                                      uint64_t localChunk, uint64_t iSub, uint64_t slot)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        const bool empty = (iTi >= localT);
        const uint64_t valid = empty ? 0 : ((iTi + bc_ <= localT) ? bc_ : (localT - iTi));

        if (valid < bc_) {
            ZeroScorePlanes(slot, valid, bc_);
        }
        if (valid == 0) {
            return;
        }

        const uint64_t midRel = (bc_ / 2 < localT - iTi) ? (bc_ / 2) : (localT - iTi - 1);
        const uint64_t elems = valid * kDim_;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> qFp = arena;
        LocalTensor<float> kFp = arena[elems];
        LocalTensor<float> gFp = arena[2 * elems];
        LocalTensor<float> expP = arena[3 * elems];
        LocalTensor<float> expN = arena[4 * elems];
        LocalTensor<float> out = arena[5 * elems];
        const uint64_t typedOff = (6 * elems * sizeof(float) + sizeof(T) - 1) / sizeof(T);
        LocalTensor<T> typed = vecBuf_.Get<T>()[typedOff];
        LocalTensor<T> qT = typed;
        LocalTensor<T> kT = typed[elems];
        LocalTensor<T> gT = typed[2 * elems];
        LocalTensor<T> outQg = typed[3 * elems];
        LocalTensor<T> outW = typed[4 * elems];
        LocalTensor<T> outKg = typed[5 * elems];

        LocalTensor<float> mid = midBuf_.Get<float>();
        LocalTensor<T> midIn = inBuf_.Get<T>();
        const uint64_t tok0 = bos + iTi;
        // mid ‖ q/k/g — one Wait(MTE2_V).
        this->CopyVectorIn(midIn, g_, this->HvRowOff(bIdx, iHv, bos + iTi + midRel), kDim_);
        this->CopyVectorIn(qT, q_, this->QkRowOff(bIdx, iH, tok0), elems);
        this->CopyVectorIn(kT, k_, this->QkRowOff(bIdx, iH, tok0), elems);
        this->CopyVectorIn(gT, g_, this->HvRowOff(bIdx, iHv, tok0), elems);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Cast(mid, midIn, RoundMode::CAST_NONE, static_cast<uint32_t>(kDim_));
        Cast(qFp, qT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
        Cast(kFp, kT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
        Cast(gFp, gT, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();

        for (uint64_t row = 0; row < valid; ++row) {
            Sub(expP[row * kDim_], gFp[row * kDim_], mid, static_cast<uint32_t>(kDim_));
        }
        PipeBarrier<PIPE_V>();
        Muls(expP, expP, LN2, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        Muls(expN, expP, -1.0f, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        ClampExp(expP, static_cast<uint32_t>(elems));
        ClampExp(expN, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        Exp(expP, expP, static_cast<uint32_t>(elems));
        Exp(expN, expN, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();

        // Shared `out`: barrier only at chain boundaries (Cast before next Mul).
        Mul(out, qFp, expP, static_cast<uint32_t>(elems)); // Qg = q * gq
        PipeBarrier<PIPE_V>();
        ClampFp16(out, static_cast<uint32_t>(elems));
        Cast(outQg, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();

        Mul(out, kFp, expP, static_cast<uint32_t>(elems)); // W = Kgq = k * gq
        PipeBarrier<PIPE_V>();
        ClampFp16(out, static_cast<uint32_t>(elems));
        Cast(outW, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();

        Mul(out, kFp, expN, static_cast<uint32_t>(elems)); // Kg = k * gk
        PipeBarrier<PIPE_V>();
        ClampFp16(out, static_cast<uint32_t>(elems));
        Cast(outKg, out, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        this->CopyVectorOut(scoreWs_, this->ScoreOff(slot, PLANE_QG, 0, 0), outQg, elems);
        this->CopyVectorOut(scoreWs_, this->ScoreOff(slot, PLANE_W, 0, 0), outW, elems);
        this->CopyVectorOut(scoreWs_, this->ScoreOff(slot, PLANE_KG, 0, 0), outKg, elems);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    // -------------------- stage_1 Vector: tril + FwdSub + store --------------------
    __aicore__ inline void LoadBetaRows(uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t valid,
                                        LocalTensor<float> beta)
    {
        Duplicate(beta, 0.0f, static_cast<uint32_t>(bc_));
        PipeBarrier<PIPE_V>();
        if (valid == 0) {
            return;
        }
        LocalTensor<T> betaIn = inBuf_.Get<T>();
        this->CopyVectorIn(betaIn, beta_, this->BetaOff(bIdx, iHv, tok0), valid);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        Cast(beta, betaIn, RoundMode::CAST_NONE, static_cast<uint32_t>(valid));
        PipeBarrier<PIPE_V>();
    }

    // Issue-only beta GM→UB (V-C2); caller does unified Wait(MTE2_V) + Cast.
    __aicore__ inline void IssueBetaRows(uint64_t bIdx, uint64_t iHv, uint64_t tok0, uint64_t valid)
    {
        if (valid == 0) {
            return;
        }
        LocalTensor<T> betaIn = inBuf_.Get<T>();
        this->CopyVectorIn(betaIn, beta_, this->BetaOff(bIdx, iHv, tok0), valid);
    }

    __aicore__ inline uint64_t BuildCausalMask(uint64_t threshold, uint64_t colBegin) const
    {
        if (threshold <= colBegin) {
            return ~0ULL;
        }
        if (threshold >= colBegin + bc_) {
            return 0ULL;
        }
        return ~0ULL << (threshold - colBegin);
    }

    __aicore__ inline void BuildCausalSelectMasks(LocalTensor<uint8_t> aqkMask, LocalTensor<uint8_t> akkMask,
                                                  uint64_t valid)
    {
        __ubuf__ uint64_t *aqkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(aqkMask.GetPhyAddr());
        __ubuf__ uint64_t *akkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(akkMask.GetPhyAddr());
        for (uint32_t row = 0; row < static_cast<uint32_t>(bc_); ++row) {
            const uint64_t aqkThr = (row < valid) ? (row + 1) : 0;      // tril incl diag
            const uint64_t akkThr = (row < valid && row > 0) ? row : 0; // strict tril
            aqkMaskPtr[row] = BuildCausalMask(aqkThr, 0);
            akkMaskPtr[row] = BuildCausalMask(akkThr, 0);
        }
    }

    // aqk = tril(aqk_raw)*scale ; akk = -(strict_tril(akk_raw)*beta) = -L
    __aicore__ inline void ApplyTrilScaleBeta(LocalTensor<float> aqk, LocalTensor<float> akk, LocalTensor<float> beta,
                                              uint64_t valid)
    {
        const uint32_t live = static_cast<uint32_t>(bc_ * bc_);
        LocalTensor<float> betaBrcb = vecBuf_.Get<float>();
        const uint8_t rowBlk = static_cast<uint8_t>((bc_ * sizeof(float)) / 32);

        Muls(aqk, aqk, scale_, live);
        const uint8_t brcbRepeat = static_cast<uint8_t>((bc_ + 7) / 8);
        Brcb(betaBrcb, beta, brcbRepeat, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint64_t col = 0; col < bc_; col += 8) {
            Mul(akk[static_cast<uint32_t>(col)], akk[static_cast<uint32_t>(col)], betaBrcb, 8,
                static_cast<uint8_t>(bc_), {1, 1, 1, rowBlk, rowBlk, 1});
        }
        PipeBarrier<PIPE_V>();

        // Select tril masks (carved from vecBuf after betaBrcb).
        constexpr uint32_t kBetaBrcbFloats = MAX_BC * 8;
        constexpr uint32_t kMaskBytes = MAX_BC * static_cast<uint32_t>(sizeof(uint64_t));
        LocalTensor<uint8_t> aqkMask = vecBuf_.Get<uint8_t>()[kBetaBrcbFloats * sizeof(float)];
        LocalTensor<uint8_t> akkMask = vecBuf_.Get<uint8_t>()[kBetaBrcbFloats * sizeof(float) + kMaskBytes];
        LocalTensor<float> zeroLocal =
            vecBuf_.Get<float>()[(kBetaBrcbFloats * sizeof(float) + 2 * kMaskBytes) / sizeof(float)];
        Duplicate(zeroLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        BuildCausalSelectMasks(aqkMask, akkMask, valid);
        SetFlag<HardEvent::S_V>(EVT_S_V);
        WaitFlag<HardEvent::S_V>(EVT_S_V);

        BinaryRepeatParams repeatParams = {1, 0, 1, rowBlk, 0, rowBlk};
        Select(aqk, aqkMask, zeroLocal, aqk, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(bc_),
               static_cast<uint8_t>(bc_), repeatParams);
        Select(akk, akkMask, zeroLocal, akk, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(bc_),
               static_cast<uint8_t>(bc_), repeatParams);
        PipeBarrier<PIPE_V>();
        Muls(akk, akk, -1.0f, live); // akk = -L
        PipeBarrier<PIPE_V>();
    }

    // Forward Substitution on akk (= -L = Ai). Triton:
    //   a += ReduceSum(a[:, None] * Ai, axis=0);  then Ai[i]=a;  Akkd=Ai+I
    // Shipped: Brcb(a) → row-broadcast Mul (no brcd UB2UB) → Add-fold col-reduce.
    // Sync: issue both col-tile Muls, then one PipeBarrier before fold (RAW on prod).
    __aicore__ inline void ForwardSub(LocalTensor<float> akk, LocalTensor<float> tmp, uint64_t valid)
    {
        if (valid < 2) {
            for (uint64_t i = 0; i < valid; ++i) {
                const uint32_t diag = static_cast<uint32_t>(i * bc_ + i);
                akk.SetValue(diag, akk.GetValue(diag) + 1.0f);
            }
            return;
        }

        constexpr uint32_t kBrcbStride = 8;
        const uint32_t bc = static_cast<uint32_t>(bc_);
        const uint32_t elems = bc * bc;
        // Layout: prod[BC*BC] | aBrcb[MAX_BC*8] | acc[BC]
        LocalTensor<float> prod = vecBuf_.Get<float>();
        LocalTensor<float> aBrcb = prod[elems];
        LocalTensor<float> acc = aBrcb[static_cast<uint32_t>(MAX_BC * kBrcbStride)];

        for (uint64_t i = 2; i < valid; ++i) {
            const uint32_t rowOff = static_cast<uint32_t>(i * bc);
            DataCopy(tmp, akk[rowOff], bc);
            PipeBarrier<PIPE_V>();

            const uint8_t brcbRepeat = static_cast<uint8_t>((bc + kBrcbStride - 1) / kBrcbStride);
            Brcb(aBrcb, tmp, brcbRepeat, {1, 8});
            PipeBarrier<PIPE_V>();
            // prod[p,c] = akk[p,c] * a[p]. Col tiles write disjoint prod columns →
            // no mid-tile fence; one barrier before Add-fold (RAW on prod).
            {
                const uint8_t rowBlk = static_cast<uint8_t>((bc * sizeof(float)) / 32);
                for (uint32_t col = 0; col < bc; col += kBrcbStride) {
                    Mul(prod[col], akk[col], aBrcb, static_cast<uint64_t>(kBrcbStride),
                        static_cast<uint8_t>(bc), {1, 1, 0, rowBlk, rowBlk, 1});
                }
                PipeBarrier<PIPE_V>();
            }

            // ReduceSum axis=0 via Add-fold (BC=16 = 2^n).
            uint32_t remain = bc;
            while (remain > 1) {
                const uint32_t calcCnt = (remain / 2) * bc;
                remain = (remain + 1) / 2;
                const uint32_t offset = remain * bc;
                Add(prod, prod, prod[offset], calcCnt);
                PipeBarrier<PIPE_V>();
            }
            DataCopy(acc, prod, bc);
            PipeBarrier<PIPE_V>();
            Add(tmp, tmp, acc, bc);
            PipeBarrier<PIPE_V>();
            DataCopy(akk[rowOff], tmp, bc);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_S>(EVT_V_S);
        WaitFlag<HardEvent::V_S>(EVT_V_S);
        for (uint64_t i = 0; i < valid; ++i) {
            const uint32_t diag = static_cast<uint32_t>(i * bc + i);
            akk.SetValue(diag, akk.GetValue(diag) + 1.0f);
        }
        SetFlag<HardEvent::S_V>(EVT_S_V);
        WaitFlag<HardEvent::S_V>(EVT_S_V);
    }

    __aicore__ inline void PostSub(uint64_t bIdx, uint64_t iHv, uint64_t bos, uint64_t localT, uint64_t localChunk,
                                   uint64_t iSub, uint64_t slot)
    {
        const uint64_t iTi = localChunk * bt_ + iSub * bc_;
        if (iTi >= localT) {
            return;
        }
        const uint64_t valid = (iTi + bc_ <= localT) ? bc_ : (localT - iTi);

        LocalTensor<float> aqk = aqkBuf_.Get<float>();
        LocalTensor<float> akk = akkBuf_.Get<float>();
        LocalTensor<float> beta = betaBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();

        const uint32_t elems = static_cast<uint32_t>(bc_ * bc_);
        Duplicate(beta, 0.0f, static_cast<uint32_t>(bc_));
        PipeBarrier<PIPE_V>();
        // cmat Aqk/Akk ‖ beta — one Wait(MTE2_V).
        DataCopy(aqk, cmatWs_[this->CmatOff(slot, PLANE_AQK, 0, 0)], elems);
        DataCopy(akk, cmatWs_[this->CmatOff(slot, PLANE_AKK, 0, 0)], elems);
        IssueBetaRows(bIdx, iHv, bos + iTi, valid);
        SetFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EVT_MTE2_V);
        if (valid > 0) {
            LocalTensor<T> betaIn = inBuf_.Get<T>();
            Cast(beta, betaIn, RoundMode::CAST_NONE, static_cast<uint32_t>(valid));
            PipeBarrier<PIPE_V>();
        }
#ifdef KDA_ISUB_DEBUG_DUMP
        // Dump gate: first chunk / iSub0 / hv=3 (known model-sample miss).
        if (localChunk == 0 && iSub == 0 && iHv == 3) {
            AscendC::printf("[isub] post raw core=%u sub=%u slot=%u hv=%u\n",
                            static_cast<uint32_t>(coreIdx_), static_cast<uint32_t>(subBlockIdx_),
                            static_cast<uint32_t>(slot), static_cast<uint32_t>(iHv));
            AscendC::DumpTensor(aqk, __LINE__, elems); // aqk_raw from Cube
            AscendC::DumpTensor(akk, __LINE__, elems); // akk_raw from Cube
        }
#endif
        ApplyTrilScaleBeta(aqk, akk, beta, valid);
#ifdef KDA_ISUB_DEBUG_DUMP
        if (localChunk == 0 && iSub == 0 && iHv == 3) {
            AscendC::DumpTensor(aqk, __LINE__, elems); // after tril*scale
        }
#endif
        ForwardSub(akk, tmp, valid);

        LocalTensor<T> aqkT = aqkTBuf_.Get<T>();
        const uint32_t liveVals = static_cast<uint32_t>(valid * bc_);
        Cast(aqkT, aqk, RoundMode::CAST_RINT, liveVals);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVT_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVT_V_MTE3);

        const uint64_t tok0 = bos + iTi;
        // Aqk diagonal block: rows [tok0, tok0+valid), cols [iSub*bc, iSub*bc+bc).
        const uint64_t aqkBase = this->AqkOff(bIdx, iHv, tok0, iSub * bc_);
        DataCopyExtParams aqkParams;
        aqkParams.blockCount = static_cast<uint16_t>(valid);
        aqkParams.blockLen = static_cast<uint32_t>(bc_ * sizeof(T));
        aqkParams.srcStride = 0;
        aqkParams.dstStride = static_cast<uint32_t>((bt_ - bc_) * sizeof(T));
        aqkParams.rsv = 0;
        DataCopyPad(aqk_[aqkBase], aqkT, aqkParams);
        // Akkd: rows [tok0, tok0+valid), 16 cols contiguous in [T,BC].
        this->CopyVectorOut(akkd_, this->AkkdOff(bIdx, iHv, tok0), akk, liveVals);
        SetFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        SetFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
        WaitFlag<HardEvent::MTE3_MTE2>(EVT_MTE3_MTE2);
        WaitFlag<HardEvent::MTE3_V>(EVT_MTE3_V);
    }

    TPipe *pipe_ = nullptr;
    TBuf<> vecBuf_, midBuf_, betaBuf_, aqkBuf_, akkBuf_, tmpBuf_, inBuf_, aqkTBuf_, zeroBuf_;
    uint64_t subBlockIdx_ = 0;
    Catlass::Arch::CrossCoreFlag s0Ready_{FLAG_S0_READY};
    Catlass::Arch::CrossCoreFlag cubeDone_{FLAG_CUBE_DONE};
    Catlass::Arch::CrossCoreFlag slotFree_[NUM_GM_SLOTS] = {FLAG_SLOT_FREE0, FLAG_SLOT_FREE1, FLAG_SLOT_FREE2,
                                                             FLAG_SLOT_FREE3};
};

} // namespace kda_isub

#endif // CHUNK_KDA_FWD_INTRA_SUB_CHUNK_VECTOR_H
