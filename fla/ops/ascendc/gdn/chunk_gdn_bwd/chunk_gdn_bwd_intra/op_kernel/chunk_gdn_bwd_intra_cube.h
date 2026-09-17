/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_GDN_BWD_INTRA_CUBE_H
#define CHUNK_GDN_BWD_INTRA_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "chunk_gdn_bwd_intra_common.h"

namespace GDN {

template <typename MainT, uint32_t G>
class ChunkGdnBwdIntraCube {
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
        (void)v;
        (void)dO;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(k));
        // q/k 只搬入一次并在 L1 内复用，绕过 L2 可避免无收益的缓存占用。
        qGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        kGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(u));
        dvLocalGm_.SetGlobalBuffer(reinterpret_cast<__gm__ MainT *>(dvLocal));
        // 三个正式输出不会被本 kernel 再读，写出时无需保留在 L2。
        wGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        uGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dvLocalGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        (void)workspace;
        mapper_.Init(cuSeqlens, chunkIndices, tiling);
        tiling_ = tiling;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
    }

    __aicore__ inline void Process()
    {
        AscendC::SetHF32Mode(false);
        AscendC::SetMMLayoutTransform(true);
        // qkMutex 串联 q/k 的 MTE2 写入、Stage 0 MTE1 读取和 Stage 2 k 末次读取。
        // l0abMutex/l0cMutex 分别保护两份 L0A/B 和 L0C，形成 MTE1 -> MMAD -> Fixpipe 流水。
        for (uint32_t slot = 0; slot < 2; ++slot) {
            l0abMutex_[slot] = AscendC::AllocMutexID();
            l0cMutex_[slot] = AscendC::AllocMutexID();
        }
        for (uint32_t slot = 0; slot < INTRA_CG_MAX; ++slot) {
            qkMutex_[slot] = AscendC::AllocMutexID();
        }
        bool scoreSlotUsed[INTRA_CG_MAX] = {};
        const int64_t blockDim = static_cast<int64_t>(AscendC::GetBlockNum());
        // 每个 AIC 按 blockDim 步长连续处理多个独立 work，复用本核 L1/L0 物理空间。
        for (int64_t work = coreIdx_; work < tiling_->workCount; work += blockDim) {
            ChunkGdnBwdIntraWorkMeta meta{};
            mapper_.ResolveChunkMajor(work, meta);
            ProcessWork(meta, scoreSlotUsed);
        }
        // Drain the final Vector consumer before the mixed kernel exits.
        for (uint32_t slot = 0; slot < INTRA_CG_MAX; ++slot) {
            if (scoreSlotUsed[slot]) {
                WaitScoreFree(slot);
            }
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            // Make the final Fixpipe owner release each L0C slot before exit.
            AscendC::Mutex::Lock<PIPE_M>(l0cMutex_[slot]);
            AscendC::Mutex::Unlock<PIPE_M>(l0cMutex_[slot]);
            AscendC::ReleaseMutexID(l0abMutex_[slot]);
            AscendC::ReleaseMutexID(l0cMutex_[slot]);
        }
        for (uint32_t slot = 0; slot < INTRA_CG_MAX; ++slot) {
            AscendC::ReleaseMutexID(qkMutex_[slot]);
        }
        AscendC::SetMMLayoutTransform(false);
    }

private:
    // 编译期布局常量放在函数前，运行时成员统一放在函数后。
    static constexpr uint32_t L1_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t L1_K_BASE = 0;
    static constexpr uint32_t L1_Q_BASE = 64 * 1024;
    static constexpr uint32_t L1_RECORD_BASE = 128 * 1024;
    static constexpr uint32_t L1_RECORD_SLOT_BYTES = 24 * 1024;
    static constexpr uint32_t L1_D_OFFSET = 0;
    static constexpr uint32_t L1_A_BG_OFFSET = 8 * 1024;
    static constexpr uint32_t L1_A_BETA_OFFSET = 16 * 1024;
    static constexpr uint32_t L1_V_BASE = 224 * 1024;
    static constexpr uint32_t L1_DO_BASE = 256 * 1024;
    static constexpr uint32_t L0_SLOT_BYTES = 16 * 1024;
    static constexpr uint32_t L0C_OUTPUT_SLOT_BYTES = 32 * 1024;

    static_assert(L1_K_BASE + INTRA_CG_MAX * L1_SLOT_BYTES == L1_Q_BASE,
                  "k slots must end where q slots begin.");
    static_assert(L1_Q_BASE + INTRA_CG_MAX * L1_SLOT_BYTES == L1_RECORD_BASE,
                  "q slots must end where Stage 1 records begin.");
    static_assert(L1_RECORD_BASE + INTRA_CG_MAX * L1_RECORD_SLOT_BYTES == L1_V_BASE,
                  "Stage 1 records must end where v ping-pong begins.");
    static_assert(L1_DO_BASE + 2 * L1_SLOT_BYTES == 288 * 1024,
                  "Cube L1 layout must fit in 288 KiB.");
    static_assert(2 * L0_SLOT_BYTES == 32 * 1024,
                  "L0A/L0B ping-pong storage must occupy 32 KiB.");
    static_assert(2 * L0C_OUTPUT_SLOT_BYTES <= 128 * 1024,
                  "The largest L0C ping-pong layout must fit in 128 KiB.");

    __aicore__ inline void ProcessWork(
        const ChunkGdnBwdIntraWorkMeta &meta,
        bool (&scoreSlotUsed)[INTRA_CG_MAX])
    {
        RunStage0(meta, scoreSlotUsed);
        if (meta.validTokens == tiling_->chunkSize) {
            RunStage2<true>(meta);
        } else {
            RunStage2<false>(meta);
        }
    }

    __aicore__ inline void RunStage0(
        const ChunkGdnBwdIntraWorkMeta &meta,
        bool (&scoreSlotUsed)[INTRA_CG_MAX])
    {
        // 先把当前 HV 切片中的所有唯一 q/k head 排入 MTE2，隐藏首个 MMAD 后的搬运时间。
        for (int64_t r = 0; r < meta.validHeads; ++r) {
            const bool isLeader =
                ChunkGdnBwdIntraScoreLeader<G>(meta.hvBegin, r);
            if (isLeader) {
                const uint32_t slot = static_cast<uint32_t>(r);
                PrefetchLeader(meta, slot, scoreSlotUsed[slot]);
                scoreSlotUsed[slot] = true;
            }
        }
        for (int64_t r = 0; r < meta.validHeads; ++r) {
            const bool isLeader =
                ChunkGdnBwdIntraScoreLeader<G>(meta.hvBegin, r);
            if (isLeader) {
                ComputeLeader(meta, static_cast<uint32_t>(r));
            }
        }
    }

    __aicore__ inline void PrefetchLeader(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t r, bool scoreSlotUsed)
    {
        if (scoreSlotUsed) {
            // 同一个物理 Score slot 跨 work 复用，覆盖前等待上一 work 的 Vector 消费结束。
            WaitScoreFree(r);
        }
        const int64_t hk = (meta.hvBegin + r) / G;
        const uint64_t gmOffset =
            ((static_cast<uint64_t>(meta.batch) * tiling_->qkHeads + hk) *
                 tiling_->seqlen + meta.tokenStart) * tiling_->keyDim;
        auto l1K = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_K_BASE + r * L1_SLOT_BYTES);
        auto l1Q = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_Q_BASE + r * L1_SLOT_BYTES);
        auto gmK = tla::MakeTensor(
            kGm_[gmOffset], tla::MakeLayout<MainT, LayoutRM>(meta.validTokens, 128),
            Catlass::Arch::PositionGM{});
        auto gmQ = tla::MakeTensor(
            qGm_[gmOffset], tla::MakeLayout<MainT, LayoutCM>(128, meta.validTokens),
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
        // 清零和搬入都会覆盖 q/k L1；先等待上一 work 的 Stage 2 完成 k 的最后一次读取。
        AscendC::Mutex::Lock<PIPE_MTE2>(qkMutex_[r]);
        if (meta.validTokens < tiling_->chunkSize) {
            ClearL1(l1K);
            ClearL1(l1Q);
        }
        copyK(l1KBlock, gmK);
        copyQ(l1QBlock, gmQ);
        // MTE2 写完 q/k 后交给 MTE1；该 Mutex 将由 ComputeLeader 在 MTE1 流水接手。
        AscendC::Mutex::Unlock<PIPE_MTE2>(qkMutex_[r]);
    }

    __aicore__ inline void ComputeLeader(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t r)
    {
        const uint32_t leaderIndex = r / G;
        const uint32_t l0Slot = leaderIndex & 1U;
        auto l1K = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_K_BASE + r * L1_SLOT_BYTES);
        auto l1Q = resource_.l1Buf.template GetBufferByByte<MainT>(
            L1_Q_BASE + r * L1_SLOT_BYTES);
        auto l0A = resource_.l0ABuf.template GetBufferByByte<MainT>(
            l0Slot * L0_SLOT_BYTES);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<MainT>(
            l0Slot * L0_SLOT_BYTES);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            l0Slot * L0C_OUTPUT_SLOT_BYTES);
        // 等待 MTE2 完成 q/k 写入。MTE1 接手后暂不释放：k 还要供 Stage 2 的 W 复用。
        AscendC::Mutex::Lock<PIPE_MTE1>(qkMutex_[r]);
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
        // 当前 MTE1 等待上一轮 MMAD 释放 L0A/B，装载完成后把该 slot 交给本轮 MMAD。
        AscendC::Mutex::Lock<PIPE_MTE1>(l0abMutex_[l0Slot]);
        CopyL1ToL0A{}(tileL0A, tileL1K);
        CopyL1ToL0B{}(tileL0B, tileL1Q);
        AscendC::Mutex::Unlock<PIPE_MTE1>(l0abMutex_[l0Slot]);
        // MMAD 同时等待 L0A/B 输入 ready 和上一轮 Fixpipe 释放 L0C。
        AscendC::Mutex::Lock<PIPE_M>(l0abMutex_[l0Slot]);
        AscendC::Mutex::Lock<PIPE_M>(l0cMutex_[l0Slot]);
        TileMmad{}(tileL0C, tileL0A, tileL0B, 64U, 64U, 128U,
                   true, static_cast<uint8_t>(0));
        AscendC::Mutex::Unlock<PIPE_M>(l0abMutex_[l0Slot]);
        AscendC::Mutex::Unlock<PIPE_M>(l0cMutex_[l0Slot]);

        // Fixpipe 等待 Score MMAD 完成，并在读完后释放 L0C，允许下一轮 MMAD 覆盖。
        AscendC::Mutex::Lock<PIPE_FIX>(l0cMutex_[l0Slot]);
        const uint32_t scoreOffset =
            INTRA_SCORE_UB_BASE + r * INTRA_SCORE_SLOT_BYTES;
        auto score = resource_.ubBuf.template GetBufferByByte<float>(scoreOffset);
        auto scoreTensor = tla::MakeTensor(
            score, tla::MakeLayout<float, LayoutRM>(32, 64),
            Catlass::Arch::PositionUB{});
        // 前后 32 行分别写入 AIV0/AIV1 的同号 UB slot，写完半片立即发布 ready。
        CopyL0CToUb<decltype(scoreTensor)>{}(
            scoreTensor, tileL0CTop, static_cast<uint8_t>(0), static_cast<uint8_t>(0));
        const uint32_t scoreFlag = INTRA_SCORE_READY_FLAG + r;
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(scoreFlag);

        CopyL0CToUb<decltype(scoreTensor)>{}(
            scoreTensor, tileL0CBottom, static_cast<uint8_t>(1), static_cast<uint8_t>(0));
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
            scoreFlag + INTRA_SUBBLOCK_FLAG_OFFSET);
        AscendC::Mutex::Unlock<PIPE_FIX>(l0cMutex_[l0Slot]);
    }

    __aicore__ inline void ClearL1(AscendC::LocalTensor<MainT> tensor) const
    {
        AscendC::InitConstValueParams<MainT> params(
            1, static_cast<uint16_t>(L1_SLOT_BYTES / 32), 0, static_cast<MainT>(0));
        AscendC::InitConstValue(tensor, params);
    }

    __aicore__ inline void WaitScoreFree(uint32_t slot) const
    {
        const uint32_t flag = INTRA_SCORE_FREE_FLAG + slot;
        // 两个 AIV 各持有 Score 的半行；二者都释放后 AIC 才能覆盖该 UB slot。
        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(flag);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(
            flag + INTRA_SUBBLOCK_FLAG_OFFSET);
    }

    template <bool FULL_TOKENS>
    __aicore__ inline void RunStage2(
        const ChunkGdnBwdIntraWorkMeta &meta)
    {
        const uint32_t validHeads = static_cast<uint32_t>(meta.validHeads);
        const uint64_t headStride =
            static_cast<uint64_t>(tiling_->seqlen) * tiling_->valueDim;
        const uint64_t outputBase =
            ((static_cast<uint64_t>(meta.batch) * tiling_->valueHeads +
              static_cast<uint64_t>(meta.hvBegin)) * tiling_->seqlen +
             static_cast<uint64_t>(meta.tokenStart)) * tiling_->valueDim;
        for (uint32_t r = 0; r < validHeads; ++r) {
            const uint32_t slot = r & 1U;
            const uint32_t workspaceFlag =
                INTRA_WORKSPACE_READY_FLAG + r;
            // 两个 AIV 分别写共享 L1 的前/后 32 行；两份 ready 都到达后 MTE1 才能读取。
            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(workspaceFlag);
            AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE1>(
                workspaceFlag + INTRA_SUBBLOCK_FLAG_OFFSET);
            const uint32_t leader = static_cast<uint32_t>(
                ChunkGdnBwdIntraLeaderR<G>(meta.hvBegin, r));
            auto l1Record = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_RECORD_BASE + r * L1_RECORD_SLOT_BYTES);
            auto l1D = l1Record[L1_D_OFFSET / sizeof(MainT)];
            auto l1ABg = l1Record[L1_A_BG_OFFSET / sizeof(MainT)];
            auto l1ABeta = l1Record[L1_A_BETA_OFFSET / sizeof(MainT)];
            auto l1K = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_K_BASE + leader * L1_SLOT_BYTES);
            auto l1V = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_V_BASE + slot * L1_SLOT_BYTES);
            auto l1DO = resource_.l1Buf.template GetBufferByByte<MainT>(
                L1_DO_BASE + slot * L1_SLOT_BYTES);
            const uint64_t outputOffset = outputBase + r * headStride;
            // Continue ping/pong across head boundaries as well as within one head.
            const uint32_t firstL0Slot = r & 1U;
            RunStage2Mmad<FULL_TOKENS, true>(
                meta, firstL0Slot, l1D, l1DO, dvLocalGm_, outputOffset);
            RunStage2Mmad<FULL_TOKENS, false>(
                meta, firstL0Slot ^ 1U, l1ABg, l1K, wGm_, outputOffset);
            // 同一 hk 的最后一个 W 已把 k 装入 L0A/B，此时才允许下一 work 覆盖 q/k L1 slot。
            const bool lastKConsumer = r + 1U == validHeads ||
                (meta.hvBegin + r) / G != (meta.hvBegin + r + 1U) / G;
            if (lastKConsumer) {
                AscendC::Mutex::Unlock<PIPE_MTE1>(qkMutex_[leader]);
            }
            RunStage2Mmad<FULL_TOKENS, true>(
                meta, firstL0Slot, l1ABeta, l1V, uGm_, outputOffset);
            // 三组右操作数均已由 MTE1 读取，向两个 AIV 归还 record 与 v/d_o ping/pong 槽。
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(workspaceFlag);
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(
                workspaceFlag + INTRA_SUBBLOCK_FLAG_OFFSET);
        }
    }

    template <bool FULL_TOKENS, bool SPLIT_RIGHT>
    __aicore__ inline void RunStage2Mmad(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t l0Slot,
        AscendC::LocalTensor<MainT> l1Left, AscendC::LocalTensor<MainT> l1Right,
        AscendC::GlobalTensor<MainT> &output, uint64_t outputOffset)
    {
        RunStage2MmadToL0<FULL_TOKENS, SPLIT_RIGHT>(
            meta, l0Slot, l1Left, l1Right);
        StoreStage2Single(meta, l0Slot, output, outputOffset);
    }

    template <bool FULL_TOKENS, bool SPLIT_RIGHT>
    __aicore__ inline void RunStage2MmadToL0(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t l0Slot,
        AscendC::LocalTensor<MainT> l1Left, AscendC::LocalTensor<MainT> l1Right)
    {
        const uint32_t mActual = FULL_TOKENS ? 64U : static_cast<uint32_t>(meta.validTokens);
        const uint32_t kActual = FULL_TOKENS ? 64U : static_cast<uint32_t>(meta.validTokens);
        auto l0A = resource_.l0ABuf.template GetBufferByByte<MainT>(
            l0Slot * L0_SLOT_BYTES);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<MainT>(
            l0Slot * L0_SLOT_BYTES);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            l0Slot * L0C_OUTPUT_SLOT_BYTES);
        auto tensorL1ATop = tla::MakeTensor(
            l1Left, tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1A>(32, 64),
            Catlass::Arch::PositionL1{});
        auto tensorL1ABottom = tla::MakeTensor(
            l1Left[INTRA_MATRIX_HALF_BYTES / sizeof(MainT)],
            tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1A>(32, 64),
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
        const uint32_t topRows = mActual < 32U ? mActual : 32U;
        const uint32_t bottomRows = mActual > 32U ? mActual - 32U : 0U;
        auto tileL1ATop = tla::GetTile(
            tensorL1ATop, tla::MakeCoord(0, 0), tla::MakeShape(topRows, kActual));
        auto tileL0ATop = tla::GetTile(
            tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(topRows, kActual));
        // L0A/B 为两份 ping/pong：MTE1 等待对应 slot 的上一轮 MMAD 读取完成。
        AscendC::Mutex::Lock<PIPE_MTE1>(l0abMutex_[l0Slot]);
        OutputCopyL1ToL0A{}(tileL0ATop, tileL1ATop);
        if (bottomRows > 0U) {
            auto tileL1ABottom = tla::GetTile(
                tensorL1ABottom, tla::MakeCoord(0, 0),
                tla::MakeShape(bottomRows, kActual));
            auto tileL0ABottom = tla::GetTile(
                tensorL0A, tla::MakeCoord(32, 0),
                tla::MakeShape(bottomRows, kActual));
            OutputCopyL1ToL0A{}(tileL0ABottom, tileL1ABottom);
        }
        if constexpr (SPLIT_RIGHT) {
            auto tensorL1BTop = tla::MakeTensor(
                l1Right,
                tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1B>(32, 128),
                Catlass::Arch::PositionL1{});
            auto tileL1BTop = tla::GetTile(
                tensorL1BTop, tla::MakeCoord(0, 0), tla::MakeShape(topRows, 128));
            auto tileL0BTop = tla::GetTile(
                tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(topRows, 128));
            OutputCopyL1ToL0B{}(tileL0BTop, tileL1BTop);
            if (bottomRows > 0U) {
                auto tensorL1BBottom = tla::MakeTensor(
                    l1Right[INTRA_VDO_INPUT_HALF_BYTES / sizeof(MainT)],
                    tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1B>(32, 128),
                    Catlass::Arch::PositionL1{});
                auto tileL1BBottom = tla::GetTile(
                    tensorL1BBottom, tla::MakeCoord(0, 0),
                    tla::MakeShape(bottomRows, 128));
                auto tileL0BBottom = tla::GetTile(
                    tensorL0B, tla::MakeCoord(32, 0),
                    tla::MakeShape(bottomRows, 128));
                OutputCopyL1ToL0B{}(tileL0BBottom, tileL1BBottom);
            }
        } else {
            auto tensorL1B = tla::MakeTensor(
                l1Right,
                tla::MakeLayout<MainT, typename OutputTile::LayoutTagL1B>(64, 128),
                Catlass::Arch::PositionL1{});
            auto tileL1B = tla::GetTile(
                tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(kActual, 128));
            OutputCopyL1ToL0B{}(tensorL0B, tileL1B);
        }
        AscendC::Mutex::Unlock<PIPE_MTE1>(l0abMutex_[l0Slot]);
        // L0C 同样为 ping/pong；当前 MMAD 必须等使用该 slot 的上一轮 Fixpipe 读完。
        AscendC::Mutex::Lock<PIPE_M>(l0abMutex_[l0Slot]);
        AscendC::Mutex::Lock<PIPE_M>(l0cMutex_[l0Slot]);
        TileMmad{}(tensorL0C, tensorL0A, tensorL0B,
                   mActual, 128U, kActual, true, static_cast<uint8_t>(0));
        AscendC::Mutex::Unlock<PIPE_M>(l0abMutex_[l0Slot]);
        AscendC::Mutex::Unlock<PIPE_M>(l0cMutex_[l0Slot]);
    }

    __aicore__ inline void StoreStage2Single(
        const ChunkGdnBwdIntraWorkMeta &meta, uint32_t l0Slot,
        AscendC::GlobalTensor<MainT> &output, uint64_t outputOffset)
    {
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            l0Slot * L0C_OUTPUT_SLOT_BYTES);
        auto tensorL0C = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(static_cast<uint32_t>(meta.validTokens), 128),
            Catlass::Arch::PositionL0C{});
        auto outputTensor = tla::MakeTensor(
            output[outputOffset],
            tla::MakeLayout<MainT, LayoutRM>(meta.validTokens, 128),
            Catlass::Arch::PositionGM{});
        // 等待 MMAD 发布 L0C；Fixpipe 转主 dtype 并直写正式输出，完成后归还该 slot。
        AscendC::Mutex::Lock<PIPE_FIX>(l0cMutex_[l0Slot]);
        OutputCopyL0CToGm<decltype(outputTensor)>{}(outputTensor, tensorL0C);
        AscendC::Mutex::Unlock<PIPE_FIX>(l0cMutex_[l0Slot]);
    }

    // 运行时状态放在所有成员函数之后。
    Catlass::Arch::Resource<ArchTag> resource_;
    const ChunkGdnBwdIntraTilingData *tiling_ = nullptr;
    ChunkGdnBwdIntraWorkMapper mapper_;

    // GM tensor 统一使用 Gm 后缀，与函数内的 L1/L0/UB tensor 区分。
    AscendC::GlobalTensor<MainT> qGm_;
    AscendC::GlobalTensor<MainT> kGm_;
    AscendC::GlobalTensor<MainT> wGm_;
    AscendC::GlobalTensor<MainT> uGm_;
    AscendC::GlobalTensor<MainT> dvLocalGm_;

    // 每个组内 head 槽独立保护一对 q/k L1 缓冲。
    AscendC::MutexID qkMutex_[INTRA_CG_MAX];
    AscendC::MutexID l0abMutex_[2];
    AscendC::MutexID l0cMutex_[2];
    int64_t coreIdx_ = 0;
};

} // namespace GDN

#endif // CHUNK_GDN_BWD_INTRA_CUBE_H
