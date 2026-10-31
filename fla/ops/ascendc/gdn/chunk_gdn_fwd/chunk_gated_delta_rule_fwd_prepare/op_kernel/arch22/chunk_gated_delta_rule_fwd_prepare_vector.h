// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. prefix/beta 常驻、条带 KKT 到 UB 叶块、固定 16 叶运算与 RHS。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_VECTOR_H
#define GDN_ARCH22_UNIFIED_PREPARE_VECTOR_H

#include "chunk_gated_delta_rule_fwd_prepare_resources.h"
#include "chunk_gated_delta_rule_fwd_prepare_activation.h"

namespace GdnUnifiedPrepare {
template <typename InputT, bool Prepared, int V, int BT, int QkvLayout, bool QkNorm = false>
class Vector {
    static constexpr uint32_t Half = kHeadsPerWindow / 2;
    static constexpr uint32_t KktRows = BT == 64 ? 64 : 32;
    static constexpr uint32_t KktBatchRows = Half * KktRows;
    static_assert(KktBatchRows <= 255, "KKT head batch exceeds instruction repeat range");
    static_assert(KktRows > 0 && KktRows <= 64 && KktRows % 16 == 0,
        "KKT rows require 16-row leaves and a 64-bit mask");
    static constexpr uint32_t Leaves = BT / 16;
    static constexpr uint32_t L = Half * Leaves;
    static constexpr uint32_t E = L * 256;
    static constexpr uint32_t ContextBytes = Half * BT * sizeof(float);
    static constexpr uint32_t LeafBytes = E * sizeof(float);
    static constexpr uint32_t InputBth = 4096;
    static constexpr uint32_t PairTokenIndices = InputBth + 64 * 8 * sizeof(float);
    static constexpr uint32_t PrefixBth = 8192;
    static constexpr uint32_t HeadIndices = 12288;
    static constexpr uint32_t TokenIndices = 14336;
    static constexpr uint32_t Acc = 18432;
    static constexpr uint32_t Factor = 20480;
    static constexpr uint32_t TriangleMask = 21504;
    static constexpr uint32_t SelectZero = TriangleMask + KktRows * sizeof(uint64_t);
    static constexpr uint32_t NormRows = 64;
    static constexpr uint32_t NormIndexElements = 2 * NormRows;
    static constexpr uint32_t NormIndices = Local::kContextEnd - 2 * NormIndexElements * sizeof(uint32_t);
    static_assert(NormIndices == 23552 && Local::kContextEnd == 24576,
        "Norm full/tail indices require the dedicated 1024-byte context gap");
    static_assert(SelectZero + 32 <= NormIndices, "Norm indices overlap live KKT context");
    static_assert(SelectZero + 32 <= Local::kContextEnd, "KKT mask/zero exceed context");
    static_assert(Local::kArena + KktBatchRows * BT * (2 * sizeof(float) + sizeof(InputT)) +
        KktBatchRows * 8 * sizeof(float) <= Local::kUbBytes, "KKT rows exceed UB");
    static constexpr uint32_t RhsKCache = Local::kUbBytes - 64 * 128 * sizeof(float);
    static_assert(Local::kArena + KktBatchRows * BT * (2 * sizeof(float) + sizeof(InputT)) +
        KktBatchRows * 8 * sizeof(float) <= RhsKCache, "KKT overlaps retained K cache");
    static_assert(Local::kLeaf + BT * BT * sizeof(InputT) <= Local::kArena &&
        Local::kArena + BT * BT * sizeof(float) <= 120 * 1024 &&
        120 * 1024 <= RhsKCache, "Final overlaps retained K cache");
    static_assert(Local::kLeaf + 64 * V * (2 * sizeof(InputT) + sizeof(float)) <= RhsKCache,
        "RHS FP32 workspace overlaps retained K cache");
    static_assert(Local::kG + 2 * ContextBytes <= Local::kBeta &&
        Local::kBeta + 2 * ContextBytes <= InputBth && InputBth == 4096 &&
        InputBth + 64 * 8 * sizeof(float) == 6144 &&
        InputBth + 64 * 8 * sizeof(float) <= PrefixBth &&
        Factor + BT * sizeof(float) <= TriangleMask, "RHS broadcast overlaps live context");
    Resources &r_;
    const Args &a_;
    WorkspaceLayout p_;
    GM_ADDR core_;
    uint32_t sub_;
    bool prefixPairSafe_ = false;
    bool rhsKChecked_ = false;
    bool rhsKSafe_ = false;
    bool rhsKReady_ = false;
    bool rhsKChunkSet_ = false;
    Chunk rhsKChunk_{};
    uint64_t rhsKHk_ = 0;

    __aicore__ static inline bool ScaleBytes(uint64_t &bytes, uint64_t factor)
    {
        const uint32_t aLo = static_cast<uint32_t>(bytes);
        const uint32_t aHi = static_cast<uint32_t>(bytes >> 32);
        const uint32_t bLo = static_cast<uint32_t>(factor);
        const uint32_t bHi = static_cast<uint32_t>(factor >> 32);
        if (aHi != 0 && bHi != 0) return false;
        // 两个32位分肢相乘均能放入uint64；高分肢同时非零已在上方拒绝。
        const uint64_t low = static_cast<uint64_t>(aLo) * bLo;
        const uint64_t cross = aHi != 0 ? static_cast<uint64_t>(aHi) * bLo :
            static_cast<uint64_t>(aLo) * bHi;
        const uint64_t high = (low >> 32) + cross;
        if (high > 0xFFFFFFFFULL) return false;
        bytes = (high << 32) | static_cast<uint32_t>(low);
        return true;
    }

    __aicore__ static inline bool Disjoint(GM_ADDR p, uint64_t n, GM_ADDR q, uint64_t m)
    {
        if (n == 0 || m == 0) return true;
        const uint64_t x = reinterpret_cast<uint64_t>(p);
        const uint64_t y = reinterpret_cast<uint64_t>(q);
        return x <= y ? y - x >= n : x - y >= m;
    }

    __aicore__ inline bool PrefixPairOutputSafe() const
    {
        if (a_.outputG == 0 || a_.gBth == nullptr) return true;
        uint64_t rows = a_.batch;
        if (!ScaleBytes(rows, a_.hv) || !ScaleBytes(rows, a_.tokens)) return false;
        uint64_t gate = rows, values = rows, matrix = rows, keys = a_.batch;
        if (!ScaleBytes(gate, sizeof(float)) ||
            !ScaleBytes(values, V) || !ScaleBytes(values, sizeof(InputT)) ||
            !ScaleBytes(matrix, BT) || !ScaleBytes(matrix, sizeof(InputT)) ||
            !ScaleBytes(keys, a_.hk) || !ScaleBytes(keys, a_.tokens) ||
            !ScaleBytes(keys, 128) || !ScaleBytes(keys, sizeof(InputT))) return false;
        // 另一 AIV 的旧 RHS1 / Cube WU1 可能尚未关闭；重叠只回退原 Prefix。
        return Disjoint(a_.gBth, gate, a_.rawG, gate) &&
            Disjoint(a_.gBth, gate, a_.beta, gate) &&
            Disjoint(a_.gBth, gate, a_.k, keys) &&
            Disjoint(a_.gBth, gate, a_.v, values) &&
            Disjoint(a_.gBth, gate, a_.a, matrix);
    }

    __aicore__ static inline bool RhsKRangeSafe(GM_ADDR address, uint64_t bytes)
    { return reinterpret_cast<uint64_t>(address) <= ~uint64_t(0) - bytes; }

    __aicore__ inline bool RhsKInputSafe() const
    {
        if (a_.hk == 0 || a_.hv / a_.hk < 2) return false;
        uint64_t keys = a_.batch, rows = a_.batch;
        if (!ScaleBytes(keys, a_.hk) || !ScaleBytes(keys, a_.tokens) ||
            !ScaleBytes(keys, 128 * sizeof(InputT)) ||
            !ScaleBytes(rows, a_.hv) || !ScaleBytes(rows, a_.tokens)) return false;
        uint64_t matrix = rows;
        if (!ScaleBytes(matrix, BT * sizeof(InputT)) || !RhsKRangeSafe(a_.k, keys) ||
            !RhsKRangeSafe(a_.a, matrix) || !Disjoint(a_.k, keys, a_.a, matrix)) return false;
        if (a_.betaEff == nullptr && (a_.outputG == 0 || a_.gBth == nullptr)) return true;
        uint64_t gate = rows;
        if (!ScaleBytes(gate, sizeof(float))) return false;
        if (a_.betaEff != nullptr && (!RhsKRangeSafe(a_.betaEff, gate) ||
            !Disjoint(a_.k, keys, a_.betaEff, gate))) return false;
        return a_.outputG == 0 || a_.gBth == nullptr ||
            (RhsKRangeSafe(a_.gBth, gate) && Disjoint(a_.k, keys, a_.gBth, gate));
    }

    __aicore__ inline bool SameRhsKChunk(const Chunk &c) const
    {
        return rhsKChunk_.batch == c.batch && rhsKChunk_.index == c.index &&
            rhsKChunk_.row == c.row && rhsKChunk_.valid == c.valid &&
            rhsKChunk_.hkBegin == c.hkBegin && rhsKChunk_.hkEnd == c.hkEnd &&
            rhsKChunk_.headBegin == c.headBegin && rhsKChunk_.headEnd == c.headEnd;
    }

    __aicore__ inline GM_ADDR Head(uint32_t slot, uint32_t localHead) const
    { return core_ + p_.slotsOffset + slot * p_.slotBytes + localHead * p_.headBytes; }

    __aicore__ inline uint32_t Heads(uint32_t active) const
    { return active > sub_ * Half ? static_cast<uint32_t>(Min(Half, active - sub_ * Half)) : 0; }

    __aicore__ inline void LoadHeadMajor(LocalTensor<float> dst, GM_ADDR src,
                                        const Chunk &c, uint64_t begin, uint32_t count)
    {
        auto gm = Gm<float>(src, a_.batch * a_.hv * a_.tokens);
        const uint32_t rounded = (static_cast<uint32_t>(c.valid) + 7) / 8 * 8;
        DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(c.valid * 4),
            static_cast<uint32_t>((a_.tokens - c.valid) * 4), (BT - rounded) / 8, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(rounded - c.valid), 0};
        DataCopyPad(dst, gm[HeadOffset(a_, c, begin)], cp, pad);
    }

    __aicore__ inline void StoreHeadMajor(GM_ADDR dst, LocalTensor<float> src,
                                         const Chunk &c, uint64_t begin, uint32_t count)
    {
        auto gm = Gm<float>(dst, a_.batch * a_.hv * a_.tokens);
        const uint32_t rounded = (static_cast<uint32_t>(c.valid) + 7) / 8 * 8;
        DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(c.valid * 4),
            (BT - rounded) / 8, static_cast<uint32_t>((a_.tokens - c.valid) * 4), 0};
        DataCopyPad(gm[HeadOffset(a_, c, begin)], src, cp);
    }

    __aicore__ inline void LoadRect(LocalTensor<float> dst, GlobalTensor<float> src,
                                    uint32_t rows, uint32_t cols, uint32_t srcStride, uint32_t dstStride)
    {
        if (rows == 0 || cols == 0) return;
        const uint32_t rounded = (cols + 7) / 8 * 8;
        DataCopyExtParams cp{static_cast<uint16_t>(rows), cols * 4, (srcStride - cols) * 4,
            (dstStride - rounded) / 8, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(rounded - cols), 0};
        DataCopyPad(dst, src, cp, pad);
    }

    __aicore__ inline void EffectiveBeta(const Chunk &c, uint64_t begin, uint32_t count,
                                         uint32_t slot, uint32_t scratchSlot)
    {
        if (a_.useBetaSigmoid == 0 && a_.betaEff == nullptr) return;
        auto beta = r_.template UB<float>(Local::kBeta + slot * ContextBytes);
        ActivatePrepareBeta<BT>(r_, a_, beta, count, scratchSlot);
        if (a_.betaEff != nullptr) {
            r_.template Fence<HardEvent::V_MTE3>();
            StoreHeadMajor(a_.betaEff, beta, c, begin, count);
        }
    }

public:
    __aicore__ inline Vector(Resources &r, const Args &a, WorkspaceLayout p)
        : r_(r), a_(a), p_(p), core_(a.scratch + GroupId() * p.coreBytes), sub_(GetSubBlockIdx())
    {
        if constexpr (Prepared) prefixPairSafe_ = PrefixPairOutputSafe();
        auto heads = r_.template UB<uint32_t>(HeadIndices);
        auto tokens = r_.template UB<uint32_t>(TokenIndices);
        for (uint32_t h = 0; h < Half; ++h) {
            for (uint32_t t = 0; t < BT; ++t) heads.SetValue(h * BT + t, (t * 8 + h) * 4);
        }
        if constexpr (!Prepared) {
            for (uint32_t t = 0; t < BT; ++t) {
                for (uint32_t h = 0; h < 8; ++h) tokens.SetValue(t * 8 + h, ((h % Half) * BT + t) * 4);
            }
        }
        auto triangleMask = r_.template UB<uint64_t>(TriangleMask);
        for (uint32_t row = 0; row < KktRows; ++row)
            triangleMask.SetValue(row, (KktRows == 64 ? ~0ULL : 0xFFFFFFFFULL) & (~0ULL << row));
        auto zero = r_.template UB<uint64_t>(SelectZero);
        for (uint32_t word = 0; word < 4; ++word) zero.SetValue(word, uint64_t(0));
        r_.template Fence<HardEvent::S_V>();
        if constexpr (!Prepared && BT == 64 && QkvLayout == 0) {
            static_assert(InputBth + BT * 8 * sizeof(float) <= PairTokenIndices &&
                PairTokenIndices % 32 == 0 &&
                PairTokenIndices + BT * 8 * sizeof(uint32_t) <= PrefixBth &&
                TokenIndices + BT * 8 * sizeof(uint32_t) <= Acc,
                "Public Prefix pair indices overlap live context");
            auto pairTokens = r_.template UB<int32_t>(PairTokenIndices);
            SetMaskNorm();
            Adds(pairTokens, r_.template UB<int32_t>(TokenIndices), int32_t(0),
                uint64_t(64), uint8_t(8), {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            uint64_t mask[2] = {0xCCCCCCCCCCCCCCCCULL, 0};
            Adds(pairTokens, pairTokens, int32_t(2 * BT * sizeof(float)), mask,
                uint8_t(8), {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            ResetMask();
        }
    }

    __aicore__ inline void Norm(const Chunk &c)
    {
        static_assert(QkNorm, "Norm is instantiated only for the feature path");
        static_assert(sizeof(InputT) == 2, "Packed Norm requires FP16/BF16 input vectors");
        // 两槽Final都已发布；6/7只保护Cube的GM/L1，不能在Norm前等待。
        const uint32_t firstRows = (static_cast<uint32_t>(c.valid) + 1) / 2;
        const uint32_t begin = sub_ == 0 ? 0 : firstRows;
        const uint32_t count = sub_ == 0 ? firstRows : static_cast<uint32_t>(c.valid) - firstRows;
        if (count == 0 || c.hkBegin == c.hkEnd) return; // 外层仍发布同一个work的NormReady。
        constexpr uint32_t Input = Local::kLeaf;
        constexpr uint32_t Fp = Input + NormRows * 128 * sizeof(InputT);
        constexpr uint32_t Square = Fp + NormRows * 128 * sizeof(float);
        constexpr uint32_t Pair = Square + NormRows * 128 * sizeof(float);
        constexpr uint32_t Work = Pair + NormRows * 64 * sizeof(float);
        constexpr uint32_t Sum = Work + 4096;
        constexpr uint32_t Broadcast = Sum + NormRows * 8 * sizeof(float);
        constexpr uint32_t One = Broadcast + 64 * sizeof(float);
        constexpr uint32_t Compact = One + 8 * sizeof(float);
        static_assert(Compact + NormRows * sizeof(float) <= RhsKCache,
            "Packed Norm exceeds the dead arena or retained K cache");
        static_assert(NormIndexElements * sizeof(float) <= 4096,
            "Padded rstd exceeds the reused broadcast workspace");
        auto input = r_.template UB<InputT>(Input);
        auto fp = r_.template UB<float>(Fp);
        auto square = r_.template UB<float>(Square);
        auto pair = r_.template UB<float>(Pair);
        auto work = r_.template UB<float>(Work);
        auto sums = r_.template UB<float>(Sum);
        // Pair后的V屏障关闭Square读者；两Mul/Cast后的V屏障关闭Work广播读者。
        auto ones = r_.template UB<float>(Square);
        const uint64_t elements = a_.batch * a_.hk * a_.tokens * 128;
        const uint64_t rstdElements = a_.batch * a_.hk * a_.tokens;
        // Decode/MakeWorkspaceLayout保证owned HK连续且不多于16(BT64)/8(BT128)。
        const uint32_t heads = QkvLayout == 1 ? static_cast<uint32_t>(c.hkEnd - c.hkBegin) : 1;
        const uint32_t fullTokens = static_cast<uint32_t>(Min(count, NormRows / heads));
        const uint32_t tailTokens = count % fullTokens;
        // K/Q紧凑rstd起点均按32B对齐时，无需Gather打包。
        const bool directRstd = QkvLayout == 0 && BT == 64 && count % 8 == 0;
        if (!directRstd) {
            // 仅构造索引，不Scalar读写数据；full/tail两表在kind/批次循环外复用。
            for (uint32_t table = 0; table < 2; ++table) {
                const uint32_t tokens = table == 0 ? fullTokens : tailTokens;
                if (tokens == 0) continue;
                const uint32_t rounded = (tokens + 7) / 8 * 8;
                auto indices = r_.template UB<uint32_t>(NormIndices + table * NormIndexElements * sizeof(uint32_t));
                for (uint32_t h = 0; h < heads; ++h) {
                    for (uint32_t t = 0; t < rounded; ++t) {
                        const uint32_t source = QkvLayout == 1 ? t * heads + h : t;
                        indices.SetValue(h * rounded + t, source * sizeof(float));
                    }
                }
            }
            // 上次Norm Gather已由V_MTE3及Final/Publish关闭；S_V只发布本次新索引。
            r_.template Fence<HardEvent::S_V>();
        }
        if constexpr (QkvLayout == 0 && BT == 64) {
            // 同HK的K/Q各至多32行；两次load后共用原64行数学批，不跨kind归约。
            const uint32_t tokens = count;
            const uint32_t rows = 2 * tokens;
            const uint32_t rounded = (tokens + 7) / 8 * 8;
            const uint32_t kindElements = tokens * 128;
            const uint32_t kindBytes = kindElements * sizeof(InputT);
            auto rawK = Gm<InputT>(a_.normK, elements);
            auto rawQ = Gm<InputT>(a_.normQ, elements);
            DataCopyExtParams load{1, kindBytes, 0, 0, 0};
            DataCopyPadExtParams<InputT> pad{false, 0, 0, 0};
            for (uint64_t hk = c.hkBegin; hk < c.hkEnd; ++hk) {
                r_.template Fence<HardEvent::MTE3_V>();
                r_.template Fence<HardEvent::MTE3_MTE2>();
                r_.template Fence<HardEvent::V_MTE2>();
                const uint64_t offset = InputOffset<QkvLayout>(a_, c, hk, a_.hk, 128, begin);
                DataCopyPad(input, rawK[offset], load, pad);
                DataCopyPad(input[kindElements], rawQ[offset], load, pad);
                r_.template Fence<HardEvent::MTE2_V>();
                Cast(fp, input, RoundMode::CAST_NONE, rows * 128);
                PipeBarrier<PIPE_V>();
                Mul(square, fp, fp, rows * 128);
                PipeBarrier<PIPE_V>();
                Add(pair, square, square[64], 64, static_cast<uint8_t>(rows), {1, 1, 1, 8, 16, 16});
                Duplicate(sums, 0.0f, NormRows * 8);
                PipeBarrier<PIPE_V>();
                SetMaskNorm();
                WholeReduceSum(sums, pair, 64, rows, 1, 1, 8);
                Duplicate(ones, 1.0f, rows);
                PipeBarrier<PIPE_V>();
                Adds(sums, sums, 1e-6f, rows);
                PipeBarrier<PIPE_V>();
                Sqrt(sums, sums, rows);
                PipeBarrier<PIPE_V>();
                Div(sums, ones, sums, rows);
                PipeBarrier<PIPE_V>();
                Brcb(work, sums, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                PipeBarrier<PIPE_V>();
                Mul(fp, fp, work, 64, static_cast<uint8_t>(rows), {1, 1, 0, 16, 16, 1});
                PipeBarrier<PIPE_V>();
                Mul(fp[64], fp[64], work, 64, static_cast<uint8_t>(rows), {1, 1, 0, 16, 16, 1});
                PipeBarrier<PIPE_V>();
                Cast(input, fp, RoundMode::CAST_RINT, rows * 128);
                PipeBarrier<PIPE_V>();
                const uint64_t rsOffset = (c.batch * a_.hk + hk) * a_.tokens + c.row + begin;
                DataCopyExtParams store{1, kindBytes, 0, 0, 0};
                DataCopyExtParams rs{1, static_cast<uint32_t>(tokens * sizeof(float)), 0, 0, 0};
                for (uint32_t kind = 0; kind < 2; ++kind) {
                    if (!directRstd) {
                        // 上一路rstd仍由MTE3读Work；关闭读者后第二Gather才可覆写。
                        if (kind != 0) r_.template Fence<HardEvent::MTE3_V>();
                        Gather(work, sums, r_.template UB<uint32_t>(NormIndices),
                            kind * tokens * sizeof(float), rounded);
                        PipeBarrier<PIPE_V>();
                    }
                    r_.template Fence<HardEvent::V_MTE3>();
                    auto hat = Gm<InputT>(kind == 0 ? a_.k : a_.qHat, elements);
                    auto rstd = Gm<float>(kind == 0 ? a_.kRstd : a_.qRstd, rstdElements);
                    DataCopyPad(hat[offset], input[kind * kindElements], store);
                    if (directRstd) {
                        DataCopyPad(rstd[rsOffset], sums[kind * tokens], rs);
                    } else {
                        DataCopyPad(rstd[rsOffset], work, rs);
                    }
                }
            }
            return;
        }
        for (uint64_t hk = c.hkBegin; hk < c.hkEnd; hk += heads) {
            for (uint32_t kind = 0; kind < 2; ++kind) { // K再Q；Hv兄弟共用该唯一写者。
                auto source = Gm<InputT>(kind == 0 ? a_.normK : a_.normQ, elements);
                auto hat = Gm<InputT>(kind == 0 ? a_.k : a_.qHat, elements);
                auto rstd = Gm<float>(kind == 0 ? a_.kRstd : a_.qRstd, rstdElements);
                for (uint32_t row = begin; row < begin + count; row += fullTokens) {
                    const uint32_t tokens = static_cast<uint32_t>(Min(fullTokens, begin + count - row));
                    const uint32_t rows = heads * tokens;
                    const uint32_t rounded = (tokens + 7) / 8 * 8;
                    // 关闭上批hat/rstd的MTE3读者后，才复用input/work/sums。
                    r_.template Fence<HardEvent::MTE3_V>();
                    r_.template Fence<HardEvent::MTE3_MTE2>();
                    r_.template Fence<HardEvent::V_MTE2>();
                    const uint16_t blocks = static_cast<uint16_t>(QkvLayout == 1 ? tokens : 1);
                    const uint32_t blockBytes = (QkvLayout == 1 ? heads : tokens) * 128 * sizeof(InputT);
                    const uint32_t gap = QkvLayout == 1
                        ? static_cast<uint32_t>((a_.hk - heads) * 128 * sizeof(InputT)) : 0;
                    DataCopyExtParams load{blocks, blockBytes, gap, 0, 0};
                    DataCopyPadExtParams<InputT> pad{false, 0, 0, 0};
                    const uint64_t offset = InputOffset<QkvLayout>(a_, c, hk, a_.hk, 128, row);
                    DataCopyPad(input, source[offset], load, pad);
                    r_.template Fence<HardEvent::MTE2_V>();
                    Cast(fp, input, RoundMode::CAST_NONE, rows * 128);
                    PipeBarrier<PIPE_V>();
                    Mul(square, fp, fp, rows * 128);
                    PipeBarrier<PIPE_V>();
                    Add(pair, square, square[64], 64, static_cast<uint8_t>(rows), {1, 1, 1, 8, 16, 16});
                    Duplicate(sums, 0.0f, NormRows * 8);
                    PipeBarrier<PIPE_V>();
                    // WholeReduceSum只设置向量mask；显式恢复normal模式。
                    SetMaskNorm();
                    WholeReduceSum(sums, pair, 64, rows, 1, 1, 8);
                    Duplicate(ones, 1.0f, rows);
                    PipeBarrier<PIPE_V>();
                    Adds(sums, sums, 1e-6f, rows);
                    PipeBarrier<PIPE_V>();
                    Sqrt(sums, sums, rows);
                    PipeBarrier<PIPE_V>();
                    Div(sums, ones, sums, rows);
                    PipeBarrier<PIPE_V>();
                    Brcb(work, sums, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                    PipeBarrier<PIPE_V>();
                    Mul(fp, fp, work, 64, static_cast<uint8_t>(rows), {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                    Mul(fp[64], fp[64], work, 64, static_cast<uint8_t>(rows), {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                    Cast(input, fp, RoundMode::CAST_RINT, rows * 128);
                    PipeBarrier<PIPE_V>();
                    // Work广播读者已结束。一次Gather得到[h,AlignUp(tokens,8)]，padding仅读初始化Sum。
                    const uint32_t tableOffset = tokens == fullTokens ? 0 : NormIndexElements * sizeof(uint32_t);
                    Gather(work, sums, r_.template UB<uint32_t>(NormIndices + tableOffset), 0, heads * rounded);
                    PipeBarrier<PIPE_V>();
                    r_.template Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams store{blocks, blockBytes, 0, gap, 0};
                    DataCopyPad(hat[offset], input, store);
                    const uint64_t rsOffset = (c.batch * a_.hk + hk) * a_.tokens + c.row + row;
                    // 新多head输出gap先核uint32域；旧逐head路径没有这个限制，不收紧API。
                    if (heads > 1 && a_.tokens - tokens <= 0xFFFFFFFFULL / sizeof(float)) {
                        DataCopyExtParams rs{static_cast<uint16_t>(heads), static_cast<uint32_t>(tokens * sizeof(float)), 0,
                            static_cast<uint32_t>((a_.tokens - tokens) * sizeof(float)), 0};
                        DataCopyPad(rstd[rsOffset], work, rs);
                    } else {
                        DataCopyExtParams rs{1, static_cast<uint32_t>(tokens * sizeof(float)), 0, 0, 0};
                        for (uint32_t h = 0; h < heads; ++h)
                            DataCopyPad(rstd[rsOffset + h * a_.tokens], work[h * rounded], rs);
                    }
                }
            }
        }
    }

    __aicore__ inline bool CanPrefixPair() const { return prefixPairSafe_; }

    __aicore__ inline void PrefixPair(const Chunk &c, uint64_t headBegin,
                                      uint32_t activeHeads, uint32_t firstSlot)
    {
        auto input = r_.template UB<float>(InputBth);
        auto prefix = r_.template UB<float>(PrefixBth);
        auto acc = r_.template UB<float>(Acc);
        r_.template Fence<HardEvent::MTE3_V>();
        r_.template Fence<HardEvent::MTE3_MTE2>();
        r_.template Fence<HardEvent::V_MTE2>();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            Duplicate(r_.template UB<float>(Local::kG + slot * ContextBytes), 0.0f, Half * BT);
            Duplicate(r_.template UB<float>(Local::kBeta + slot * ContextBytes), 0.0f, Half * BT);
        }
        Duplicate(prefix, 0.0f, BT * 8);
        Duplicate(acc, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE2>();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            LoadHeadMajor(r_.template UB<float>(Local::kBeta + slot * ContextBytes), a_.beta,
                c, headBegin + i * kHeadsPerWindow + sub_ * Half, count);
        }
        auto raw = Gm<float>(a_.rawG, a_.batch * a_.tokens * a_.hv);
        DataCopyExtParams load{static_cast<uint16_t>(c.valid), activeHeads * 4,
            static_cast<uint32_t>((a_.hv - activeHeads) * 4), 0, 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(8 - activeHeads), 0};
        DataCopyPad(input, raw[(c.batch * a_.tokens + c.row) * a_.hv + headBegin], load, pad);
        r_.template Fence<HardEvent::MTE2_V>();
        if (a_.useBetaSigmoid != 0 || a_.betaEff != nullptr) {
            for (uint32_t i = 0; i < 2; ++i) {
                const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
                if (count == 0) continue;
                const uint32_t slot = (firstSlot + i) & 1;
                EffectiveBeta(c, headBegin + i * kHeadsPerWindow + sub_ * Half, count, slot, firstSlot);
            }
        }
        ActivatePrepareGate<BT>(r_, a_, input, headBegin, activeHeads,
            static_cast<uint32_t>(c.valid), firstSlot);
        for (uint32_t row = 0; row < c.valid; ++row) {
            if (row == 0) Adds(acc, input[row * 8], 0.0f, activeHeads);
            else Add(acc, acc, input[row * 8], activeHeads);
            PipeBarrier<PIPE_V>();
            Copy(prefix[row * 8], acc, activeHeads, 1, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        ScalePrepareCumsum(a_, prefix, acc, static_cast<uint32_t>(c.valid), activeHeads);
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            const uint32_t base = (i * kHeadsPerWindow + sub_ * Half) * sizeof(float);
            Gather(r_.template UB<float>(Local::kG + slot * ContextBytes), prefix,
                r_.template UB<uint32_t>(HeadIndices), base, Half * BT);
            PipeBarrier<PIPE_V>();
        }
        r_.template Fence<HardEvent::V_MTE3>();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            StoreHeadMajor(a_.gBht, r_.template UB<float>(Local::kG + slot * ContextBytes),
                c, headBegin + i * kHeadsPerWindow + sub_ * Half, count);
        }
        if (sub_ == 0 && a_.outputG != 0 && a_.gBth != nullptr) {
            auto publicG = Gm<float>(a_.gBth, a_.batch * a_.tokens * a_.hv);
            DataCopyExtParams cp{static_cast<uint16_t>(c.valid), activeHeads * 4, 0,
                static_cast<uint32_t>((a_.hv - activeHeads) * 4), 0};
            DataCopyPad(publicG[(c.batch * a_.tokens + c.row) * a_.hv + headBegin], prefix, cp);
        }
        Duplicate(input, c.valid == 1 ? 1.0f : 0.0f, BT * 8);
        PipeBarrier<PIPE_V>();
        if (c.valid > 1) {
            for (uint32_t row = 0; row < c.valid; ++row) Sub(input[row * 8], acc, prefix[row * 8], activeHeads);
            PipeBarrier<PIPE_V>();
            PrepareExp(a_, input, input, static_cast<uint32_t>(c.valid * 8));
            PipeBarrier<PIPE_V>();
        }
        PrepareExp(a_, acc, acc, activeHeads);
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE3>();
        if (sub_ == 0) {
            auto last = Gm<float>(a_.gLastExp, a_.batch * a_.nt * ((a_.hv + 7) / 8 * 8));
            DataCopyExtParams cp{1, activeHeads * 4, 0, 0, 0};
            DataCopyPad(last[(c.batch * a_.nt + c.index) * ((a_.hv + 7) / 8 * 8) + headBegin], acc, cp);
        }
        r_.template Fence<HardEvent::MTE3_V>(); // 公开 BTH 的最后 Prefix 读者退出。
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            auto decay = r_.template UB<float>(PrefixBth + i * ContextBytes);
            const uint32_t base = (i * kHeadsPerWindow + sub_ * Half) * sizeof(float);
            Gather(decay, input, r_.template UB<uint32_t>(HeadIndices), base, Half * BT);
            PipeBarrier<PIPE_V>();
            r_.template Fence<HardEvent::V_MTE3>();
            StoreHeadMajor(a_.gDecay, decay, c, headBegin + i * kHeadsPerWindow + sub_ * Half, count);
        }
    }

    __aicore__ inline void PrefixPairPublic(const Chunk &c, uint64_t headBegin,
                                            uint32_t activeHeads, uint32_t firstSlot)
    {
        static_assert(!Prepared && BT == 64 && QkvLayout == 0, "Public Prefix pair requires BT64 BHT input");
        auto input = r_.template UB<float>(InputBth);
        auto prefix = r_.template UB<float>(PrefixBth);
        auto acc = r_.template UB<float>(Acc);
        r_.template Fence<HardEvent::MTE3_V>();
        r_.template Fence<HardEvent::MTE3_MTE2>();
        r_.template Fence<HardEvent::V_MTE2>();
        // 空第二窗的 context 也先清零；配对 Gather 固定读取四条物理 owner 行。
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t slot = (firstSlot + i) & 1;
            Duplicate(r_.template UB<float>(Local::kG + slot * ContextBytes), 0.0f, Half * BT);
            Duplicate(r_.template UB<float>(Local::kBeta + slot * ContextBytes), 0.0f, Half * BT);
        }
        Duplicate(prefix, 0.0f, BT * 8);
        Duplicate(acc, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE2>();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            const uint64_t begin = headBegin + i * kHeadsPerWindow + sub_ * Half;
            LoadHeadMajor(r_.template UB<float>(Local::kBeta + slot * ContextBytes), a_.beta, c, begin, count);
            LoadHeadMajor(r_.template UB<float>(Local::kG + slot * ContextBytes), a_.rawG, c, begin, count);
        }
        r_.template Fence<HardEvent::MTE2_V>();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            const uint64_t begin = headBegin + i * kHeadsPerWindow + sub_ * Half;
            auto g = r_.template UB<float>(Local::kG + slot * ContextBytes);
            if (a_.useGate != 0) {
                Gather(input, g, r_.template UB<uint32_t>(TokenIndices), 0, BT * 8);
                PipeBarrier<PIPE_V>();
            }
            EffectiveBeta(c, begin, count, slot, slot);
            if (a_.useGate != 0) {
                ActivatePrepareGate<BT>(r_, a_, input, begin, count, static_cast<uint32_t>(c.valid), slot);
                Gather(g, input, r_.template UB<uint32_t>(HeadIndices), 0, Half * BT);
                PipeBarrier<PIPE_V>();
            }
        }
        Gather(input, r_.template UB<float>(Local::kG), r_.template UB<uint32_t>(PairTokenIndices), 0, BT * 8);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < c.valid; ++row) {
            if (row == 0) Adds(acc, input[row * 8], 0.0f, 2 * Half);
            else Add(acc, acc, input[row * 8], 2 * Half);
            PipeBarrier<PIPE_V>();
            Copy(prefix[row * 8], acc, 2 * Half, 1, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        ScalePrepareCumsum(a_, prefix, acc, static_cast<uint32_t>(c.valid), 2 * Half);
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            const uint32_t base = slot * Half * sizeof(float);
            Gather(r_.template UB<float>(Local::kG + slot * ContextBytes), prefix,
                r_.template UB<uint32_t>(HeadIndices), base, Half * BT);
            PipeBarrier<PIPE_V>();
        }
        r_.template Fence<HardEvent::V_MTE3>();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t count = Heads(activeHeads - i * kHeadsPerWindow);
            if (count == 0) continue;
            const uint32_t slot = (firstSlot + i) & 1;
            StoreHeadMajor(a_.gBht, r_.template UB<float>(Local::kG + slot * ContextBytes),
                c, headBegin + i * kHeadsPerWindow + sub_ * Half, count);
        }
    }

    __aicore__ inline void Prefix(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        const uint32_t count = Heads(activeHeads);
        if (count == 0) return;
        const uint64_t begin = headBegin + sub_ * Half;
        auto g = r_.template UB<float>(Local::kG + slot * ContextBytes);
        auto beta = r_.template UB<float>(Local::kBeta + slot * ContextBytes);
        auto input = r_.template UB<float>(InputBth);
        auto prefix = r_.template UB<float>(PrefixBth);
        auto acc = r_.template UB<float>(Acc);
        // 前一 pack 的 UB 换义同时约束下一位 Vector/MTE2 写者，不能只等 MTE3→V。
        r_.template Fence<HardEvent::MTE3_V>();
        r_.template Fence<HardEvent::MTE3_MTE2>();
        r_.template Fence<HardEvent::V_MTE2>();
        Duplicate(g, 0.0f, Half * BT);
        Duplicate(beta, 0.0f, Half * BT);
        Duplicate(prefix, 0.0f, BT * 8);
        Duplicate(acc, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE2>();
        LoadHeadMajor(beta, a_.beta, c, begin, count);
        if constexpr (Prepared) {
            auto raw = Gm<float>(a_.rawG, a_.batch * a_.tokens * a_.hv);
            DataCopyExtParams cp{static_cast<uint16_t>(c.valid), count * 4,
                static_cast<uint32_t>((a_.hv - count) * 4), 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(8 - count), 0};
            DataCopyPad(input, raw[(c.batch * a_.tokens + c.row) * a_.hv + begin], cp, pad);
            r_.template Fence<HardEvent::MTE2_V>();
        } else {
            LoadHeadMajor(g, a_.rawG, c, begin, count);
            r_.template Fence<HardEvent::MTE2_V>();
            Gather(input, g, r_.template UB<uint32_t>(TokenIndices), 0, BT * 8);
            PipeBarrier<PIPE_V>();
        }
        EffectiveBeta(c, begin, count, slot, slot);
        ActivatePrepareGate<BT>(r_, a_, input, begin, count, static_cast<uint32_t>(c.valid), slot);
        for (uint32_t row = 0; row < c.valid; ++row) {
            if (row == 0) Adds(acc, input[row * 8], 0.0f, count);
            else Add(acc, acc, input[row * 8], count);
            PipeBarrier<PIPE_V>();
            Copy(prefix[row * 8], acc, count, 1, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>(); // Copy 对 acc 的末次读取先于下一行 Add。
        }
        ScalePrepareCumsum(a_, prefix, acc, static_cast<uint32_t>(c.valid), count);
        Gather(g, prefix, r_.template UB<uint32_t>(HeadIndices), 0, Half * BT);
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE3>();
        StoreHeadMajor(a_.gBht, g, c, begin, count);
        if (a_.outputG != 0 && a_.gBth != nullptr) {
            auto publicG = Gm<float>(a_.gBth, a_.batch * a_.tokens * a_.hv);
            DataCopyExtParams cp{static_cast<uint16_t>(c.valid), count * 4, 0,
                static_cast<uint32_t>((a_.hv - count) * 4), 0};
            DataCopyPad(publicG[(c.batch * a_.tokens + c.row) * a_.hv + begin], prefix, cp);
        }
        if constexpr (Prepared) {
            // 保留原有效长度 1 的单位衰减，包括非有限 raw g；其他长度严格 final-prefix。
            Duplicate(input, c.valid == 1 ? 1.0f : 0.0f, BT * 8);
            PipeBarrier<PIPE_V>();
            if (c.valid > 1) {
                for (uint32_t row = 0; row < c.valid; ++row) Sub(input[row * 8], acc, prefix[row * 8], count);
                PipeBarrier<PIPE_V>();
                PrepareExp(a_, input, input, static_cast<uint32_t>(c.valid * 8));
                PipeBarrier<PIPE_V>();
            }
            PrepareExp(a_, acc, acc, count);
            PipeBarrier<PIPE_V>();
            r_.template Fence<HardEvent::V_MTE3>();
            auto last = Gm<float>(a_.gLastExp, a_.batch * a_.nt * ((a_.hv + 7) / 8 * 8));
            DataCopyExtParams cp{1, count * 4, 0, 0, 0};
            DataCopyPad(last[(c.batch * a_.nt + c.index) * ((a_.hv + 7) / 8 * 8) + begin], acc, cp);
            // prefixBth 旧公开输出的读者已退出，才将它换义为 head-major decay。
            r_.template Fence<HardEvent::MTE3_V>();
            Gather(prefix, input, r_.template UB<uint32_t>(HeadIndices), 0, Half * BT);
            PipeBarrier<PIPE_V>();
            r_.template Fence<HardEvent::V_MTE3>();
            StoreHeadMajor(a_.gDecay, prefix, c, begin, count);
        }
    }

    __aicore__ inline void KktAndLeaves(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        auto leaves = r_.template UB<float>(Local::kLeaf + slot * LeafBytes);
        auto score = r_.template UB<float>(Local::kArena);
        auto gate = r_.template UB<float>(Local::kArena + KktBatchRows * BT * 4);
        auto out = gate;
        auto typed = r_.template UB<InputT>(Local::kArena + 2 * KktBatchRows * BT * 4);
        auto brcb = r_.template UB<float>(Local::kArena + 2 * KktBatchRows * BT * 4 + KktBatchRows * BT * sizeof(InputT));
        auto triangleMask = r_.template UB<uint64_t>(TriangleMask);
        auto zero = r_.template UB<float>(SelectZero);
        r_.template Fence<HardEvent::MTE3_V>();
        Duplicate(leaves, 0.0f, E);
        PipeBarrier<PIPE_V>();
        auto scores = Gm<float>(core_, p_.scoreBytes / 4);
        Invalidate(scores);
        const uint32_t count = Heads(activeHeads);
        uint64_t hk[Half]{};
        GM_ADDR xAddress[Half]{};
        for (uint32_t h = 0; h < count; ++h) {
            const uint32_t local = sub_ * Half + h;
            hk[h] = (headBegin + local) / (a_.hv / a_.hk) - c.hkBegin;
            xAddress[h] = Head(slot, local) + p_.xOffset;
        }
        auto g = r_.template UB<float>(Local::kG + slot * ContextBytes);
        auto beta = r_.template UB<float>(Local::kBeta + slot * ContextBytes);
        for (uint32_t rowBase = 0; count != 0 && rowBase < c.valid; rowBase += KktRows) {
            const uint32_t rows = static_cast<uint32_t>(Min(KktRows, c.valid - rowBase));
            const uint32_t batchRows = count * rows;
            const uint32_t cols = rowBase + rows;
            r_.template Fence<HardEvent::MTE3_V>();
            r_.template Fence<HardEvent::V_MTE2>();
            for (uint32_t h = 0; h < count; ++h)
                LoadRect(score[h * rows * BT], scores[hk[h] * BT * BT + rowBase * BT], rows, c.valid, BT, BT);
            Duplicate(gate, 0.0f, batchRows * BT);
            PipeBarrier<PIPE_V>();
            const uint8_t rs = BT / 8;
            for (uint32_t h = 0; h < count; ++h) {
                for (uint32_t col = 0; col < cols; col += 64) {
                    Copy(gate[h * rows * BT + col], g[h * BT + col], static_cast<uint16_t>(Min(64, cols - col)),
                        static_cast<uint8_t>(rows), {1, 1, rs, 0});
                }
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < count; ++h) {
                Brcb(brcb[h * rows * 8], g[h * BT + rowBase], static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                PipeBarrier<PIPE_V>(); // 尾部广播padding可覆盖下一head，先关闭WAW。
            }
            for (uint32_t col = 0; col < cols; col += 64) {
                Sub(gate[col], brcb, gate[col], Min(64, cols - col), static_cast<uint8_t>(batchRows),
                    {1, 0, 1, rs, 1, rs});
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < cols; col += 64)
                Maxs(gate[col], gate[col], -50.0f, Min(64, cols - col), static_cast<uint8_t>(batchRows), {1, 1, rs, rs});
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < cols; col += 64)
                Mins(gate[col], gate[col], 50.0f, Min(64, cols - col), static_cast<uint8_t>(batchRows), {1, 1, rs, rs});
            PipeBarrier<PIPE_V>();
            if (a_.useExp2 != 0) {
                for (uint32_t col = 0; col < cols; col += 64)
                    Muls(gate[col], gate[col], kPrepareLn2, Min(64, cols - col),
                        static_cast<uint8_t>(batchRows), {1, 1, rs, rs});
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t col = 0; col < cols; col += 64)
                Exp(gate[col], gate[col], Min(64, cols - col), static_cast<uint8_t>(batchRows), {1, 1, rs, rs});
            PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < count; ++h) {
                Brcb(brcb[h * rows * 8], beta[h * BT + rowBase], static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                PipeBarrier<PIPE_V>(); // 与g广播同样保留尾部WAW关闭点。
            }
            for (uint32_t col = 0; col < cols; col += 64) {
                Mul(gate[col], gate[col], brcb, Min(64, cols - col), static_cast<uint8_t>(batchRows),
                    {1, 1, 0, rs, rs, 1});
            }
            PipeBarrier<PIPE_V>();
            r_.template Fence<HardEvent::MTE2_V>();
            for (uint32_t col = 0; col < cols; col += 64) {
                Mul(out[col], score[col], gate[col], Min(64, cols - col), static_cast<uint8_t>(batchRows),
                    {1, 1, 1, rs, rs, rs});
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < count; ++h) {
                Select(out[h * rows * BT + rowBase], triangleMask, zero, out[h * rows * BT + rowBase],
                    SELMODE::VSEL_TENSOR_TENSOR_MODE, uint64_t(KktRows), static_cast<uint8_t>(rows), {1, 0, 1, rs, 0, rs});
            }
            PipeBarrier<PIPE_V>();
            // 仍逐point保留FP32 -> InputT RINT -> FP32边界。
            Cast(typed, out, RoundMode::CAST_RINT, batchRows * BT);
            PipeBarrier<PIPE_V>();
            Cast(out, typed, RoundMode::CAST_NONE, batchRows * BT);
            PipeBarrier<PIPE_V>();
            for (uint32_t h = 0; h < count; ++h) {
                for (uint32_t leafRow = 0; leafRow < rows; leafRow += 16) {
                    const uint32_t leafRows = static_cast<uint32_t>(Min(16, rows - leafRow));
                    DataCopy(leaves[(h * Leaves + (rowBase + leafRow) / 16) * 256],
                        out[h * rows * BT + leafRow * BT + rowBase + leafRow],
                        {static_cast<uint16_t>(leafRows), 2, static_cast<uint16_t>((BT - 16) / 8), 0});
                }
            }
            PipeBarrier<PIPE_V>();
            r_.template Fence<HardEvent::V_MTE3>();
            for (uint32_t h = 0; h < count; ++h) {
                auto x = Gm<float>(xAddress[h], BT * BT);
                DataCopyExtParams cp{static_cast<uint16_t>(rows), BT * 4, 0, 0, 0};
                DataCopyPad(x[rowBase * BT], out[h * rows * BT], cp);
            }
        }
        r_.template Fence<HardEvent::MTE3_V>(); // prod 与 epilogue arena 换义。
        auto prod = r_.template UB<float>(Local::kArena);
        auto row = r_.template UB<float>(Local::kLeafRow);
        Muls(leaves, leaves, -1.0f, E);
        PipeBarrier<PIPE_V>();
        for (int i = 2; i < 16; ++i) {
            DataCopy(row, leaves[i * 16], {static_cast<uint16_t>(L), 2, 30, 0});
            PipeBarrier<PIPE_V>();
            Brcb(prod, row, L * 2, {2, 16});
            Brcb(prod[8], row, L * 2, {2, 16});
            PipeBarrier<PIPE_V>();
            Mul(prod, leaves, prod, E);
            PipeBarrier<PIPE_V>();
            // 与 solve_tri_pipeline.h 的 16 叶固定 Add 树逐条一致。
            Add(prod, prod, prod[128], 64, L, {1, 1, 1, 32, 32, 32});
            Add(prod[64], prod[64], prod[192], 64, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[64], 64, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[32], 32, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[16], 16, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(leaves[i * 16], leaves[i * 16], prod, 16, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
        }
        for (int group = 0; group < 4; ++group) {
            uint64_t mask[2] = {0x0008000400020001ULL << (4 * group), 0};
            Duplicate(leaves[group * 64], 1.0f, mask, L, 1, 32);
        }
        PipeBarrier<PIPE_V>();
        r_.template Fence<HardEvent::V_MTE3>();
        for (uint32_t h = 0; h < count; ++h) {
            auto d = Gm<float>(Head(slot, sub_ * Half + h) + p_.d0Offset, BT * 16);
            for (uint32_t leaf = 0; leaf < Leaves && leaf * 16 < c.valid; ++leaf) {
                const uint32_t rows = static_cast<uint32_t>(Min(16, c.valid - leaf * 16));
                DataCopyExtParams cp{static_cast<uint16_t>(rows), 16 * 4, 0, 0, 0};
                DataCopyPad(d[leaf * 256], leaves[(h * Leaves + leaf) * 256], cp);
            }
        }
    }

    __aicore__ inline void Publish(uint32_t slot)
    {
        // DCCI/DSB 是 Scalar 指令：先用 MTE3→S 明确关闭所有异步 GM 写，再 clean/DSB。
        // 不把 DDR DSB 的名字当作 MTE3 已完成的证据，也不在每条小搬运后排空。
        r_.template Fence<HardEvent::MTE3_S>();
        Invalidate(Gm<float>(core_ + p_.slotsOffset + slot * p_.slotBytes, p_.slotBytes / 4));
        CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + slot);
    }

    template <int S>
    __aicore__ inline void Assemble(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        constexpr bool Final = 2 * S == BT;
        auto out = r_.template UB<float>(Local::kArena);
        auto typed = r_.template UB<InputT>(Local::kLeaf); // Final 使用全区起点，不附加 slot 偏移。
        auto leaves = r_.template UB<float>(Local::kLeaf + slot * LeafBytes); // 仅 Assemble16 读取。
        const uint32_t count = Heads(activeHeads);
        auto publicA = Gm<InputT>(a_.a, a_.batch * a_.hv * a_.tokens * BT);
        // Cube 的整包 result 已 ready，diag 在本 phase 内不再被生产者修改。
        // ENTIRE_DATA_CACHE 一次覆盖所有 head；空 AIV 不额外清理。
        if (count != 0) Invalidate(Gm<float>(Head(slot, sub_ * Half), p_.headBytes / sizeof(float)));
        for (uint32_t h = 0; h < count; ++h) {
            const uint32_t local = sub_ * Half + h;
            auto result = Gm<float>(Head(slot, local) + p_.resultOffset, BT * BT / 4);
            auto diag = Gm<float>(Head(slot, local) + (S == 32 ? p_.d1Offset : p_.d0Offset), BT * S);
            auto next = Gm<float>(Head(slot, local) + (S == 16 ? p_.d1Offset : p_.d0Offset), BT * 2 * S);
            for (uint32_t row = 0, pair = 0; row < c.valid; row += 2 * S, ++pair) {
                const uint32_t n0 = static_cast<uint32_t>(Min(S, c.valid - row));
                const uint32_t n1 = c.valid > row + S ? static_cast<uint32_t>(Min(S, c.valid - row - S)) : 0;
                r_.template Fence<HardEvent::MTE3_V>();
                Duplicate(out, 0.0f, 4 * S * S);
                PipeBarrier<PIPE_V>();
                r_.template Fence<HardEvent::V_MTE2>();
                if constexpr (S == 16) {
                    // 两个对角叶仍在 UB；绝不从旧 X/leaf GM 重新加载它们。
                    Copy(out, leaves[(h * Leaves + row / 16) * 256], n0, static_cast<uint8_t>(n0), {1, 1, 4, 2});
                    if (n1 > 0) Copy(out[2 * S * S + S], leaves[(h * Leaves + row / 16 + 1) * 256],
                        n1, static_cast<uint8_t>(n1), {1, 1, 4, 2});
                    PipeBarrier<PIPE_V>();
                } else {
                    LoadRect(out, diag[row * S], n0, n0, S, 2 * S);
                    if (n1 > 0) LoadRect(out[2 * S * S + S], diag[(row + S) * S], n1, n1, S, 2 * S);
                }
                if constexpr (BT == 64 && S == 16) {
                    if (c.valid == BT) {
                        // 打包 R 为 32x32，仅消费 (0,0)/(16,16) 两个对角块。
                        LoadRect(out[2 * S * S], result[pair * (2 * S * S + S)], n1, n0, 2 * S, 2 * S);
                    } else if (n1 > 0) {
                        LoadRect(out[2 * S * S], result[pair * S * S], n1, n0, S, 2 * S);
                    }
                } else {
                    if (n1 > 0) LoadRect(out[2 * S * S], result[pair * S * S], n1, n0, S, 2 * S);
                }
                r_.template Fence<HardEvent::MTE2_V>();
                if (n1 > 0) Muls(out[2 * S * S], out[2 * S * S], -1.0f, S,
                    static_cast<uint8_t>(n1), {1, 1, static_cast<uint8_t>(2 * S / 8), static_cast<uint8_t>(2 * S / 8)});
                PipeBarrier<PIPE_V>();
                if constexpr (Final) {
                    Cast(typed, out, RoundMode::CAST_RINT, 4 * S * S);
                    PipeBarrier<PIPE_V>();
                    r_.template Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams cp{static_cast<uint16_t>(n0 + n1), 2 * S * sizeof(InputT), 0, 0, 0};
                    DataCopyPad(publicA[HeadOffset(a_, c, headBegin + local, BT) + row * BT], typed, cp);
                } else {
                    r_.template Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams cp{static_cast<uint16_t>(n0 + n1), 2 * S * 4, 0, 0, 0};
                    DataCopyPad(next[row * 2 * S], out, cp);
                }
            }
        }
    }

    __aicore__ inline void Rhs(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        constexpr uint32_t Rows = 64;
        constexpr uint32_t MatrixBytes = Rows * V * sizeof(InputT);
        auto in = r_.template UB<InputT>(Local::kLeaf);
        auto out = r_.template UB<InputT>(Local::kLeaf + MatrixBytes);
        auto fp = r_.template UB<float>(Local::kLeaf + 2 * MatrixBytes);
        auto factor = r_.template UB<float>(Factor);
        auto k = Gm<InputT>(a_.k, a_.batch * a_.hk * a_.tokens * 128);
        auto v = Gm<InputT>(a_.v, a_.batch * a_.hv * a_.tokens * V);
        // 本 AIV 两槽叶/低层组装的最后读者已结束，整体换义为完整 RHS 工作区。
        // 输出写独立 GM，入口在 Final 等待前关闭最后 MTE3 reader，再复用本区组装 A。
        r_.template Fence<HardEvent::MTE3_V>();
        r_.template Fence<HardEvent::MTE3_MTE2>();
        const uint32_t count = Heads(activeHeads);
        bool useKCache = false;
        if (count != 0 && c.valid != 0 && c.valid <= Rows) {
            if (!rhsKChecked_) {
                rhsKSafe_ = RhsKInputSafe();
                rhsKChecked_ = true;
            }
            useKCache = rhsKSafe_;
        }
        if (useKCache) {
            if (!rhsKChunkSet_ || !SameRhsKChunk(c)) {
                rhsKReady_ = false;
                rhsKChunk_ = c;
                rhsKChunkSet_ = true;
            }
        } else {
            rhsKReady_ = false; // 原fallback的尾部Brcb会覆盖cache首2KiB。
        }
        // V/K两kind统一迁移；不能让V的Brcb污染跨head/slot保留的K。
        auto broadcast = r_.template UB<float>(useKCache ? InputBth :
            Local::kLeaf + 2 * MatrixBytes + Rows * V * 4);
        auto cache = r_.template UB<float>(RhsKCache);
        for (uint32_t h = 0; h < count; ++h) {
            const uint32_t local = sub_ * Half + h;
            const uint64_t global = headBegin + local;
            auto beta = r_.template UB<float>(Local::kBeta + slot * ContextBytes)[h * BT];
            auto g = r_.template UB<float>(Local::kG + slot * ContextBytes)[h * BT];
            PrepareExp(a_, factor, g, static_cast<uint32_t>(c.valid));
            PipeBarrier<PIPE_V>();
            Mul(factor, beta, factor, static_cast<uint32_t>(c.valid));
            PipeBarrier<PIPE_V>();
            auto rhs = Gm<InputT>(Head(slot, local), p_.headBytes / sizeof(InputT));
            // 顺序与原 WU 相同：V*beta->InputT，然后 K*(beta*exp(g))->InputT。
            for (uint32_t kind = 0; kind < 2; ++kind) {
                const uint32_t width = kind == 0 ? V : 128;
                const uint64_t sourceHead = kind == 0 ? global : global / (a_.hv / a_.hk);
                const uint64_t sourceHeads = kind == 0 ? a_.hv : a_.hk;
                const uint64_t sourceStride = QkvLayout == 1 ? sourceHeads * width : width;
                auto source = kind == 0 ? v : k;
                for (uint32_t row = 0; row < c.valid; row += Rows) {
                    const uint32_t rows = static_cast<uint32_t>(Min(Rows, c.valid - row));
                    const bool cachedK = useKCache && kind == 1;
                    const bool kHit = cachedK && rhsKReady_ && rhsKHk_ == sourceHead;
                    r_.template Fence<HardEvent::V_MTE2>();
                    DataCopyExtParams load{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(InputT)),
                        static_cast<uint32_t>((sourceStride - width) * sizeof(InputT)), 0, 0};
                    DataCopyPadExtParams<InputT> pad{false, 0, 0, 0};
                    if (!kHit) DataCopyPad(in, source[InputOffset<QkvLayout>(a_, c, sourceHead, sourceHeads, width, row)], load, pad);
                    r_.template Fence<HardEvent::MTE2_V>();
                    if (cachedK) {
                        if (!kHit) Cast(cache, in, RoundMode::CAST_NONE, rows * width);
                    } else {
                        Cast(fp, in, RoundMode::CAST_NONE, rows * width);
                    }
                    PipeBarrier<PIPE_V>();
                    if (cachedK && !kHit) {
                        rhsKHk_ = sourceHead;
                        rhsKReady_ = true;
                    }
                    Brcb(broadcast, kind == 0 ? beta[row] : factor[row], static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                    PipeBarrier<PIPE_V>();
                    for (uint32_t col = 0; col < width; col += 64) {
                        Mul(fp[col], cachedK ? cache[col] : fp[col], broadcast, 64, static_cast<uint8_t>(rows),
                            {1, 1, 0, static_cast<uint8_t>(width / 8), static_cast<uint8_t>(width / 8), 1});
                    }
                    PipeBarrier<PIPE_V>();
                    r_.template Fence<HardEvent::MTE3_V>();
                    Cast(out, fp, RoundMode::CAST_RINT, rows * width);
                    PipeBarrier<PIPE_V>();
                    r_.template Fence<HardEvent::V_MTE3>();
                    DataCopyExtParams store{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(InputT)), 0, 0, 0};
                    const uint64_t base = kind == 0 ? p_.vbOffset : p_.kbgOffset;
                    DataCopyPad(rhs[base / sizeof(InputT) + row * width], out, store);
                }
            }
        }
    }
};
} // namespace GdnUnifiedPrepare
#endif
