/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH22_VECTOR_H
#define CHUNK_GDN_BWD_INTRA_ARCH22_VECTOR_H
#include "chunk_gdn_bwd_intra_common.h"
#include <type_traits>

namespace GDN::Arch22 {
template <typename MainT, typename GateT, typename BetaT, uint32_t G>
class ChunkGdnBwdIntraVector {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
        GM_ADDR workspace, GM_ADDR cu, GM_ADDR indices,
        const ChunkGdnBwdIntraTilingData *tiling)
    {
        t_ = tiling;
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(k));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(v));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        ws_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(workspace));
        mapper_.Init(cu, indices, tiling);
        pipe_.InitBuffer(buffer_, UB_BYTES);
    }
    __aicore__ inline void Process()
    {
        const int64_t part = AscendC::GetSubBlockIdx();
        const int64_t core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        for (int64_t work = core; work < t_->workCount; work += t_->blockDim) {
            ChunkGdnBwdIntraWorkMeta m;
            mapper_.ResolveChunkMajor(work, m);
            if (m.validTokens <= 0 || m.validHeads <= 0) continue;
            bool scoreSeen[4] = {false, false, false, false};
            PrefetchRaw(m, m.hvBegin, part, true);
            for (int64_t r = 0; r < m.validHeads; ++r) {
                const uint16_t slot = static_cast<uint16_t>(r % INPUT_SLOTS);
                if (r >= INPUT_SLOTS) Wait(INPUTS_FREE + slot);
                const int64_t hv = m.hvBegin + r;
                const uint64_t aux = (static_cast<uint64_t>(m.batch) * t_->valueHeads + hv)
                    * t_->seqlen + m.tokenStart;
                auto rawG = Slice<GateT>(UB_GATE_RAW_OFFSET);
                auto rawBeta = Slice<BetaT>(UB_BETA_RAW_OFFSET);
                auto gate = Slice<float>(UB_GATE_FLOAT_OFFSET);
                auto beta = Slice<float>(UB_BETA_FLOAT_OFFSET);
                LoadAux(rawG, g_, aux, m.validTokens);
                LoadAux(rawBeta, beta_, aux, m.validTokens);
                Fence<AscendC::HardEvent::MTE2_V>();
                Convert(gate, rawG, CHUNK_TOKENS);
                Convert(beta, rawBeta, CHUNK_TOKENS);
                AscendC::PipeBarrier<PIPE_V>();
                ScaleInputs(m, hv, part, Record(core, t_->cg, r), gate, beta);
                const int64_t leader = ChunkGdnBwdIntraLeaderR<G>(m.hvBegin, r);
                if (!scoreSeen[leader]) {
                    Wait(static_cast<uint16_t>(leader));
                    scoreSeen[leader] = true;
                }
                GateScore(m, part, Record(core, t_->cg, leader),
                    Record(core, t_->cg, r), gate);
                Notify<PIPE_MTE3>(INPUTS_READY + slot);
            }
            const int64_t firstPending = m.validHeads > INPUT_SLOTS ?
                m.validHeads - INPUT_SLOTS : 0;
            for (int64_t r = firstPending; r < m.validHeads; ++r) {
                Wait(INPUTS_FREE + static_cast<uint16_t>(r % INPUT_SLOTS));
            }
        }
    }
private:
    __aicore__ inline void PrefetchRaw(const ChunkGdnBwdIntraWorkMeta &m,
        int64_t hv, int64_t part, bool loadK)
    {
        const uint32_t begin = part * VECTOR_PART_TOKENS;
        const uint32_t rows = m.validTokens > begin ? m.validTokens - begin : 0;
        const uint32_t count = (rows < VECTOR_PART_TOKENS ? rows : VECTOR_PART_TOKENS) * HEAD_DIM;
        if (count == 0) return;
        const uint64_t vv = ((static_cast<uint64_t>(m.batch) * t_->valueHeads + hv)
            * t_->seqlen + m.tokenStart + begin) * HEAD_DIM;
        if (loadK) {
            const uint64_t kk = ((static_cast<uint64_t>(m.batch) * t_->qkHeads + hv / G)
                * t_->seqlen + m.tokenStart + begin) * HEAD_DIM;
            AscendC::DataCopy(Slice<MainT>(UB_K_RAW_OFFSET), k_[kk], count);
        }
        AscendC::DataCopy(Slice<MainT>(UB_V_RAW_OFFSET), v_[vv], count);
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> Slice(uint32_t bytes)
    {
        return buffer_.template Get<uint8_t>()[bytes].template ReinterpretCast<T>();
    }
    template <typename T>
    __aicore__ inline void LoadAux(AscendC::LocalTensor<T> dst,
        AscendC::GlobalTensor<T> &src, uint64_t offset, uint32_t valid)
    {
        AscendC::Duplicate(dst, T(0), CHUNK_TOKENS);
        Fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{true, 0,
            static_cast<uint8_t>((DMA_BLOCK_BYTES / sizeof(T) - valid % (DMA_BLOCK_BYTES / sizeof(T))) % (DMA_BLOCK_BYTES / sizeof(T))), 0};
        AscendC::DataCopyPad(dst, src[offset], copy, pad);
    }
    template <typename T>
    __aicore__ inline void Convert(AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<T> src, uint32_t count)
    {
        if constexpr (std::is_same_v<T, float>) AscendC::DataCopy(dst, src, count);
        else AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    }
    __aicore__ inline void Exponent(AscendC::LocalTensor<float> dst,
        AscendC::LocalTensor<float> src, uint32_t count)
    {
        if (t_->useExp2 != 0) {
            AscendC::Muls(dst, src, 0.6931471805599453f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(dst, dst, count);
        } else AscendC::Exp(dst, src, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ScaleInputs(const ChunkGdnBwdIntraWorkMeta &m,
        int64_t hv, int64_t part, uint64_t rec, AscendC::LocalTensor<float> gate,
        AscendC::LocalTensor<float> beta)
    {
        const uint32_t begin = part * VECTOR_PART_TOKENS;
        const uint32_t rows = m.validTokens > begin ?
            static_cast<uint32_t>(m.validTokens - begin) : 0;
        const uint32_t validRows = rows < VECTOR_PART_TOKENS ? rows : VECTOR_PART_TOKENS;
        auto ki = Slice<MainT>(UB_K_RAW_OFFSET);
        auto vi = Slice<MainT>(UB_V_RAW_OFFSET);
        auto ko = Slice<MainT>(UB_K_OUT_OFFSET);
        auto vo = Slice<MainT>(UB_V_OUT_OFFSET);
        auto kf = Slice<float>(UB_K_FLOAT_OFFSET);
        auto vf = Slice<float>(UB_V_FLOAT_OFFSET);
        auto coeff = Slice<float>(UB_COEFF_OFFSET);
        // Entire physical halves are initialized so tail records cannot retain prior-work data.
        AscendC::Duplicate(ko, MainT(0), PART_KV_ELEMENTS);
        AscendC::Duplicate(vo, MainT(0), PART_KV_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        if (validRows != 0) {
            Fence<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(kf, ki, AscendC::RoundMode::CAST_NONE, validRows * HEAD_DIM);
            AscendC::Cast(vf, vi, AscendC::RoundMode::CAST_NONE, validRows * HEAD_DIM);
            Fence<AscendC::HardEvent::V_MTE2>();
            if (hv + 1 < m.hvBegin + m.validHeads) {
                const bool nextHk = (hv + 1) / G != hv / G;
                PrefetchRaw(m, hv + 1, part, nextHk);
            }
            Exponent(coeff, gate, CHUNK_TOKENS);
            AscendC::Mul(coeff, coeff, beta, CHUNK_TOKENS);
            Fence<AscendC::HardEvent::V_S>();
            // Existing scratch areas hold row coefficients; raw staging remains separate.
            auto coeffRows = Slice<float>(UB_COEFF_ROWS_OFFSET);
            auto betaRows = Slice<float>(UB_BETA_ROWS_OFFSET);
            const uint8_t repeats = static_cast<uint8_t>((validRows + 7) / 8);
            AscendC::Brcb(coeffRows, coeff[begin], repeats, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(betaRows, beta[begin], repeats, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            Fence<AscendC::HardEvent::V_S>();
            AscendC::Mul(kf, kf, coeffRows, FLOAT_VECTOR_LANES, validRows, {1, 1, 0, KV_ROW_BLOCKS, KV_ROW_BLOCKS, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(kf[FLOAT_VECTOR_LANES], kf[FLOAT_VECTOR_LANES], coeffRows, FLOAT_VECTOR_LANES, validRows, {1, 1, 0, KV_ROW_BLOCKS, KV_ROW_BLOCKS, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(vf, vf, betaRows, FLOAT_VECTOR_LANES, validRows, {1, 1, 0, KV_ROW_BLOCKS, KV_ROW_BLOCKS, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(vf[FLOAT_VECTOR_LANES], vf[FLOAT_VECTOR_LANES], betaRows, FLOAT_VECTOR_LANES, validRows, {1, 1, 0, KV_ROW_BLOCKS, KV_ROW_BLOCKS, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ko, kf, AscendC::RoundMode::CAST_RINT, validRows * HEAD_DIM);
            AscendC::Cast(vo, vf, AscendC::RoundMode::CAST_RINT, validRows * HEAD_DIM);
        }
        Fence<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(ws_[rec + K_OFFSET + begin * HEAD_DIM], ko, PART_KV_ELEMENTS);
        AscendC::DataCopy(ws_[rec + V_OFFSET + begin * HEAD_DIM], vo, PART_KV_ELEMENTS);
        // GateScore uses disjoint scratch, so K/V writeback can progress during its compute.
    }
    __aicore__ inline void GateScore(const ChunkGdnBwdIntraWorkMeta &m,
        int64_t part, uint64_t source, uint64_t rec, AscendC::LocalTensor<float> gate)
    {
        auto si = Slice<MainT>(UB_SCORE_RAW_OFFSET);
        auto sf = Slice<float>(UB_SCORE_FLOAT_OFFSET);
        auto delta = Slice<float>(UB_DELTA_OFFSET);
        auto result = Slice<MainT>(UB_SCORE_OUT_OFFSET);
        const uint32_t begin = part * VECTOR_PART_TOKENS;
        const uint32_t rows = m.validTokens > begin ?
            static_cast<uint32_t>(m.validTokens - begin) : 0;
        const uint32_t validRows = rows < VECTOR_PART_TOKENS ? rows : VECTOR_PART_TOKENS;
        AscendC::Duplicate(result, MainT(0), PART_SCORE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        // Only effective Score rows are read; the Cube need not clear its padded GM rows.
        if (validRows != 0) {
            Fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(si, ws_[source + S_OFFSET + begin * CHUNK_TOKENS], validRows * CHUNK_TOKENS);
            Fence<AscendC::HardEvent::MTE2_V>();
            AscendC::Cast(sf, si, AscendC::RoundMode::CAST_NONE, validRows * CHUNK_TOKENS);
            Fence<AscendC::HardEvent::V_S>();
            auto rowGate = Slice<float>(UB_ROW_GATE_OFFSET);
            AscendC::Brcb(rowGate, gate[begin], static_cast<uint8_t>((validRows + 7) / 8), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            Fence<AscendC::HardEvent::V_S>();
            AscendC::Sub(delta, gate, rowGate, CHUNK_TOKENS, validRows, {1, 1, 0, SCORE_ROW_BLOCKS, 0, 1});
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < validRows; ++row) {
                const uint32_t token = begin + row;
                // Neutralize invalid elements before Exp; all Vector addresses stay 32B aligned.
                uint64_t mask[2] = {InvalidColumns(token, m.validTokens), 0};
                if (mask[0] != 0) {
                    AscendC::Duplicate(delta[row * CHUNK_TOKENS], 0.0f, mask, 1, 1, SCORE_ROW_BLOCKS);
                    AscendC::Duplicate(sf[row * CHUNK_TOKENS], 0.0f, mask, 1, 1, SCORE_ROW_BLOCKS);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            const uint32_t count = validRows * CHUNK_TOKENS;
            Exponent(delta, delta, count);
            AscendC::Muls(delta, delta, t_->scale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(delta, delta, sf, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(result, delta, AscendC::RoundMode::CAST_RINT, count);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < validRows; ++row) {
                const uint32_t token = begin + row;
                uint64_t mask[2] = {InvalidColumns(token, m.validTokens), 0};
                if (mask[0] != 0) {
                    AscendC::Duplicate(result[row * CHUNK_TOKENS], MainT(0), mask, 1, 1, SCORE_ROW_BLOCKS);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        Fence<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopy(ws_[rec + D_OFFSET + begin * CHUNK_TOKENS], result, PART_SCORE_ELEMENTS);
        Fence<AscendC::HardEvent::MTE3_V>();
    }
    __aicore__ inline uint64_t InvalidColumns(uint32_t token, uint32_t validTokens)
    {
        const uint64_t lower = (uint64_t(1) << token) - 1;
        const uint64_t tail = validTokens < CHUNK_TOKENS ? (~uint64_t(0) << validTokens) : 0;
        return lower | tail;
    }
    const ChunkGdnBwdIntraTilingData *t_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::GlobalTensor<MainT> k_, v_, ws_;
    AscendC::GlobalTensor<GateT> g_;
    AscendC::GlobalTensor<BetaT> beta_;
};
} // namespace GDN::Arch22
#endif
