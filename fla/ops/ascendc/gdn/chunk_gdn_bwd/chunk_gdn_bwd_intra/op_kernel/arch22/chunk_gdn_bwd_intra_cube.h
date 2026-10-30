/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_ARCH22_CUBE_H
#define CHUNK_GDN_BWD_INTRA_ARCH22_CUBE_H
#include "chunk_gdn_bwd_intra_common.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN::Arch22 {
template <typename MainT, uint32_t G>
class ChunkGdnBwdIntraCube {
    using Arch = Catlass::Arch::AtlasA2;
    template <class BLayout>
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, MainT,
        Catlass::layout::RowMajor, MainT, BLayout, MainT, Catlass::layout::RowMajor>;
    using RowCopy = Copy<Catlass::layout::RowMajor>;
    using ScoreCopy = Copy<Catlass::layout::ColumnMajor>;
    static_assert(sizeof(MainT) == 2, "Main dtype must be FP16/BF16");
    static_assert(L1_SCORE_BYTES + MAX_HEADS_PER_WORK * L1_HEAD_BYTES <= Arch::L1_SIZE, "L1 budget exceeded");
    static_assert(CHUNK_TOKENS * HEAD_DIM * sizeof(MainT) <= Arch::L0A_SIZE, "L0A budget exceeded");
    static_assert(HEAD_DIM * HEAD_DIM * sizeof(MainT) <= Arch::L0B_SIZE, "L0B budget exceeded");
    static_assert(CHUNK_TOKENS * HEAD_DIM * sizeof(float) <= Arch::L0C_SIZE, "L0C budget exceeded");
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR a, GM_ADDR dO,
        GM_ADDR w, GM_ADDR u, GM_ADDR dv, GM_ADDR workspace, GM_ADDR cu,
        GM_ADDR indices, const ChunkGdnBwdIntraTilingData *tiling)
    {
        t_ = tiling;
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(k));
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(a));
        dO_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dO));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(w));
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(u));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dv));
        ws_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(workspace));
        mapper_.Init(cu, indices, tiling);
    }

    __aicore__ inline void Process()
    {
        const int64_t core = AscendC::GetBlockIdx();
        Catlass::Arch::Resource<Arch> resource;
        for (int64_t work = core; work < t_->workCount; work += t_->blockDim) {
            ChunkGdnBwdIntraWorkMeta m;
            mapper_.ResolveChunkMajor(work, m);
            if (m.validTokens <= 0 || m.validHeads <= 0) continue;
            for (int64_t r = 0; r < m.validHeads; ++r) {
                if (!ChunkGdnBwdIntraScoreLeader<G>(m.hvBegin, r)) continue;
                const int64_t hk = (m.hvBegin + r) / G;
                const uint64_t qk = ((static_cast<uint64_t>(m.batch) * t_->qkHeads + hk)
                    * t_->seqlen + m.tokenStart) * HEAD_DIM;
                auto kGm = tla::MakeTensor(k_[qk], tla::MakeLayout<MainT,
                    Catlass::layout::RowMajor>(CHUNK_TOKENS, HEAD_DIM), Catlass::Arch::PositionGM{});
                auto qGm = tla::MakeTensor(q_[qk], tla::MakeLayout<MainT,
                    Catlass::layout::ColumnMajor>(HEAD_DIM, CHUNK_TOKENS), Catlass::Arch::PositionGM{});
                auto score = tla::MakeTensor(ws_[Record(core, t_->cg, r) + S_OFFSET],
                    tla::MakeLayout<MainT, Catlass::layout::RowMajor>(CHUNK_TOKENS, CHUNK_TOKENS),
                    Catlass::Arch::PositionGM{});
                auto kL1 = LoadA<ScoreCopy>(resource, kGm, r * L1_SCORE_HEAD_BYTES, CHUNK_TOKENS, HEAD_DIM,
                    m.validTokens, HEAD_DIM);
                auto qL1 = LoadB<ScoreCopy>(resource, qGm,
                    t_->cg * L1_SCORE_HEAD_BYTES + r * L1_SCORE_HEAD_BYTES, HEAD_DIM, CHUNK_TOKENS, HEAD_DIM, m.validTokens);
                Multiply<ScoreCopy>(resource, kL1, qL1, score,
                    m.validTokens, m.validTokens, HEAD_DIM);
                Notify<PIPE_FIX>(static_cast<uint16_t>(r));
            }
            for (int64_t r = 0; r < m.validHeads; ++r) {
                // Mode 2 waits for both AIV subblocks, as in the reference kernels.
                if (r == 0) PrefetchOriginal(resource, m, r);
                const uint16_t slot = static_cast<uint16_t>(r % INPUT_SLOTS);
                Wait(INPUTS_READY + slot);
                Fence<AscendC::HardEvent::MTE2_MTE1>();
                const int64_t hv = m.hvBegin + r;
                const uint64_t out = ((static_cast<uint64_t>(m.batch) * t_->valueHeads + hv)
                    * t_->seqlen + m.tokenStart) * HEAD_DIM;
                const uint64_t aOffset = ((static_cast<uint64_t>(m.batch) * t_->valueHeads + hv)
                    * t_->seqlen + m.tokenStart) * CHUNK_TOKENS;
                const uint64_t rec = Record(core, t_->cg, r);
                const uint32_t base = L1_SCORE_BYTES + r * L1_HEAD_BYTES;
                auto aGm = Matrix(a_, aOffset, CHUNK_TOKENS);
                auto kGm = Matrix(ws_, rec + K_OFFSET, HEAD_DIM);
                auto vGm = Matrix(ws_, rec + V_OFFSET, HEAD_DIM);
                auto dGm = Matrix(ws_, rec + D_OFFSET, CHUNK_TOKENS);
                auto oGm = Matrix(dO_, out, HEAD_DIM);
                auto aL1 = tla::MakeTensor(resource.l1Buf.template GetBufferByByte<MainT>(base),
                    tla::MakeLayout<MainT, typename RowCopy::LayoutTagL1A>(uint32_t(CHUNK_TOKENS), uint32_t(CHUNK_TOKENS)),
                    Catlass::Arch::PositionL1{});
                auto kL1 = LoadB<RowCopy>(resource, kGm, base + L1_K_OFFSET, CHUNK_TOKENS, HEAD_DIM,
                    m.validTokens, HEAD_DIM);
                auto vL1 = LoadB<RowCopy>(resource, vGm, base + L1_V_OFFSET, CHUNK_TOKENS, HEAD_DIM,
                    m.validTokens, HEAD_DIM);
                auto dL1 = LoadA<RowCopy>(resource, dGm, base + L1_D_OFFSET, CHUNK_TOKENS, CHUNK_TOKENS,
                    m.validTokens, m.validTokens);
                auto oL1 = tla::MakeTensor(resource.l1Buf.template GetBufferByByte<MainT>(base + L1_DO_OFFSET),
                    tla::MakeLayout<MainT, typename RowCopy::LayoutTagL1B>(uint32_t(CHUNK_TOKENS), uint32_t(HEAD_DIM)),
                    Catlass::Arch::PositionL1{});
                auto wGm = Matrix(w_, out, HEAD_DIM);
                auto uGm = Matrix(u_, out, HEAD_DIM);
                auto dvGm = Matrix(dv_, out, HEAD_DIM);
                MultiplyTriplet(resource, aL1, dL1, kL1, vL1, oL1,
                    wGm, uGm, dvGm, m, r);
                Notify<PIPE_FIX>(INPUTS_FREE + slot);
            }
        }
    }
private:
    __aicore__ inline void PrefetchOriginal(Catlass::Arch::Resource<Arch> &res,
        const ChunkGdnBwdIntraWorkMeta &m, int64_t r)
    {
        const uint64_t hv = m.hvBegin + r;
        const uint64_t token = (static_cast<uint64_t>(m.batch) * t_->valueHeads + hv)
            * t_->seqlen + m.tokenStart;
        const uint32_t base = L1_SCORE_BYTES + r * L1_HEAD_BYTES;
        auto aGm = Matrix(a_, token * CHUNK_TOKENS, CHUNK_TOKENS);
        auto oGm = Matrix(dO_, token * HEAD_DIM, HEAD_DIM);
        LoadA<RowCopy, false>(res, aGm, base, CHUNK_TOKENS, CHUNK_TOKENS, m.validTokens, m.validTokens);
        LoadB<RowCopy, false>(res, oGm, base + L1_DO_OFFSET, CHUNK_TOKENS, HEAD_DIM, m.validTokens, HEAD_DIM);
    }

    static_assert(L0_SLOTS * L0A_SLOT_BYTES <= Arch::L0A_SIZE, "Triplet L0A budget exceeded");
    static_assert(L0_SLOTS * L0B_SLOT_BYTES <= Arch::L0B_SIZE, "Triplet L0B budget exceeded");
    static_assert(L0_SLOTS * L0C_TILE_BYTES <= Arch::L0C_SIZE, "Triplet L0C budget exceeded");

    template <class A, class B>
    __aicore__ inline void PreloadOperands(Catlass::Arch::Resource<Arch> &res,
        A &a, B &b, uint32_t slot, uint32_t cubeM, uint32_t k)
    {
        auto l0a = tla::MakeTensor(res.l0ABuf.template GetBufferByByte<MainT>(slot * L0A_SLOT_BYTES),
            tla::MakeLayout<MainT, typename RowCopy::LayoutTagL0A>(cubeM, k),
            Catlass::Arch::PositionL0A{});
        auto l0b = tla::MakeTensor(res.l0BBuf.template GetBufferByByte<MainT>(slot * L0B_SLOT_BYTES),
            tla::MakeLayout<MainT, typename RowCopy::LayoutTagL0B>(k, HEAD_DIM),
            Catlass::Arch::PositionL0B{});
        auto at = tla::GetTile(a, tla::MakeCoord(0, 0), tla::MakeShape(cubeM, k));
        auto bt = tla::GetTile(b, tla::MakeCoord(0, 0), tla::MakeShape(k, HEAD_DIM));
        typename RowCopy::CopyL1ToL0A copyA;
        typename RowCopy::CopyL1ToL0B copyB;
        copyA(l0a, at);
        copyB(l0b, bt);
    }

    template <class A, class B, class Out>
    __aicore__ inline void MultiplyTriplet(Catlass::Arch::Resource<Arch> &res,
        A &a, A &d, B &b, B &v, B &o, Out &w, Out &u, Out &dv,
        const ChunkGdnBwdIntraWorkMeta &meta, int64_t r)
    {
        const uint32_t m = meta.validTokens;
        const uint32_t cubeM = m == 1 ? 16 : m;
        PreloadOperands(res, a, b, 0, cubeM, m);
        for (uint32_t index = 0; index < 3; ++index) {
            const uint32_t slot = index & 1U;
            auto l0a = tla::MakeTensor(res.l0ABuf.template GetBufferByByte<MainT>(slot * L0A_SLOT_BYTES),
                tla::MakeLayout<MainT, typename RowCopy::LayoutTagL0A>(cubeM, m),
                Catlass::Arch::PositionL0A{});
            auto l0b = tla::MakeTensor(res.l0BBuf.template GetBufferByByte<MainT>(slot * L0B_SLOT_BYTES),
                tla::MakeLayout<MainT, typename RowCopy::LayoutTagL0B>(m, HEAD_DIM),
                Catlass::Arch::PositionL0B{});
            auto l0c = tla::MakeTensor(res.l0CBuf.template GetBufferByByte<float>(slot * L0C_TILE_BYTES),
                tla::MakeLayoutL0C(cubeM, HEAD_DIM), Catlass::Arch::PositionL0C{});
            Fence<AscendC::HardEvent::MTE1_M>();
            Catlass::Gemm::Tile::TileMmadTla<Arch, MainT, typename RowCopy::LayoutTagL1A> mma;
            mma(l0c, l0a, l0b, cubeM, HEAD_DIM, m, true, 0);
            if (index == 0 && r + 1 < meta.validHeads) PrefetchOriginal(res, meta, r + 1);
            // Preload only the alternate slot while the current MMA consumes its operands.
            if (index == 0) PreloadOperands(res, a, v, 1, cubeM, m);
            if (index == 1) PreloadOperands(res, d, o, 0, cubeM, m);
            Fence<AscendC::HardEvent::M_MTE1>();
            Fence<AscendC::HardEvent::M_FIX>();
            auto &out = index == 0 ? w : (index == 1 ? u : dv);
            auto dst = tla::GetTile(out, tla::MakeCoord(0, 0), tla::MakeShape(m, HEAD_DIM));
            auto src = tla::GetTile(l0c, tla::MakeCoord(0, 0), tla::MakeShape(m, HEAD_DIM));
            typename RowCopy::template CopyL0CToGm<decltype(dst)> store;
            store(dst, src);
            Fence<AscendC::HardEvent::FIX_M>();
            Fence<AscendC::HardEvent::MTE1_MTE2>();
        }
    }

    __aicore__ inline auto Matrix(AscendC::GlobalTensor<MainT> &gm,
        uint64_t offset, uint32_t columns)
    {
        return tla::MakeTensor(gm[offset], tla::MakeLayout<MainT,
            Catlass::layout::RowMajor>(CHUNK_TOKENS, columns), Catlass::Arch::PositionGM{});
    }

    template <class C, bool WAIT = true, class Tensor>
    __aicore__ inline auto LoadA(Catlass::Arch::Resource<Arch> &res, Tensor &gm,
        uint32_t bytes, uint32_t rows, uint32_t cols, uint32_t validRows, uint32_t validCols)
    {
        auto local = res.l1Buf.template GetBufferByByte<MainT>(bytes);
        if ((validRows & 15U) != 0 || (validCols & 15U) != 0) {
            AscendC::InitConstValueParams<MainT> zero{1,
                static_cast<uint16_t>(rows * cols * sizeof(MainT) / DMA_BLOCK_BYTES), 0, MainT(0)};
            AscendC::InitConstValue(local, zero);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        auto l1 = tla::MakeTensor(local,
            tla::MakeLayout<MainT, typename C::LayoutTagL1A>(rows, cols),
            Catlass::Arch::PositionL1{});
        auto tile = tla::GetTile(gm, tla::MakeCoord(0, 0), tla::MakeShape(validRows, validCols));
        typename C::template CopyGmToL1A<decltype(tile)> copy;
        copy(l1, tile);
        if constexpr (WAIT) Fence<AscendC::HardEvent::MTE2_MTE1>();
        return l1;
    }
    template <class C, bool WAIT = true, class Tensor>
    __aicore__ inline auto LoadB(Catlass::Arch::Resource<Arch> &res, Tensor &gm,
        uint32_t bytes, uint32_t rows, uint32_t cols, uint32_t validRows, uint32_t validCols)
    {
        auto local = res.l1Buf.template GetBufferByByte<MainT>(bytes);
        if ((validRows & 15U) != 0 || (validCols & 15U) != 0) {
            AscendC::InitConstValueParams<MainT> zero{1,
                static_cast<uint16_t>(rows * cols * sizeof(MainT) / DMA_BLOCK_BYTES), 0, MainT(0)};
            AscendC::InitConstValue(local, zero);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        auto l1 = tla::MakeTensor(local,
            tla::MakeLayout<MainT, typename C::LayoutTagL1B>(rows, cols),
            Catlass::Arch::PositionL1{});
        auto tile = tla::GetTile(gm, tla::MakeCoord(0, 0), tla::MakeShape(validRows, validCols));
        typename C::template CopyGmToL1B<decltype(tile)> copy;
        copy(l1, tile);
        if constexpr (WAIT) Fence<AscendC::HardEvent::MTE2_MTE1>();
        return l1;
    }
    template <class C, class A, class B, class Out>
    __aicore__ inline void Multiply(Catlass::Arch::Resource<Arch> &res,
        A &a, B &b, Out &out, uint32_t m, uint32_t n, uint32_t k)
    {
        // AtlasA2's M=1 GEMV mode is avoided; copy-out retains the original M.
        const uint32_t cubeM = m == 1 ? 16 : m;
        auto la = res.l0ABuf.template GetBufferByByte<MainT>(0);
        auto lb = res.l0BBuf.template GetBufferByByte<MainT>(0);
        auto lc = res.l0CBuf.template GetBufferByByte<float>(0);
        auto l0a = tla::MakeTensor(la, tla::MakeLayout<MainT, typename C::LayoutTagL0A>(cubeM, k),
            Catlass::Arch::PositionL0A{});
        auto l0b = tla::MakeTensor(lb, tla::MakeLayout<MainT, typename C::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        auto l0c = tla::MakeTensor(lc, tla::MakeLayoutL0C(cubeM, n), Catlass::Arch::PositionL0C{});
        auto at = tla::GetTile(a, tla::MakeCoord(0, 0), tla::MakeShape(cubeM, k));
        auto bt = tla::GetTile(b, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        typename C::CopyL1ToL0A copyA;
        typename C::CopyL1ToL0B copyB;
        copyA(l0a, at);
        copyB(l0b, bt);
        Fence<AscendC::HardEvent::MTE1_M>();
        Catlass::Gemm::Tile::TileMmadTla<Arch, MainT, typename C::LayoutTagL1A> mma;
        mma(l0c, l0a, l0b, cubeM, n, k, true, 0);
        Fence<AscendC::HardEvent::M_MTE1>();
        Fence<AscendC::HardEvent::M_FIX>();
        auto dst = tla::GetTile(out, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto src = tla::GetTile(l0c, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        typename C::template CopyL0CToGm<decltype(dst)> store;
        store(dst, src);
        Fence<AscendC::HardEvent::FIX_M>();
        Fence<AscendC::HardEvent::MTE1_MTE2>();
    }
    const ChunkGdnBwdIntraTilingData *t_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;
    AscendC::GlobalTensor<MainT> q_, k_, a_, dO_, w_, u_, dv_, ws_;
};
} // namespace GDN::Arch22
#endif
