/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
// Stage | Order within each work window | Main output
// 0     | h0, h1 | dk_state/dv_b GM, dW L1/raw UB, zV UB
// 2/4/6 | h0, h1 at each Stage | dKgb/zW, Tza hi/lo, dAkk
// 8/10  | each 32-row band: 8h0,8h1,10h0,10h1 | [dq;left], right
//
// Memory | KiB interval | Lifetime
// L1 | A[0,16), dW[32,96), kE[96,160), Zb[160,192) | Stages 0..6
// L1 | streams[192,416), Tza[416,448) | Stages 0 and 4..6
// L1 | Local owner0[64,192), owner1[192,320) | replace retired inputs, Stage7..10
// L0 | A/B: 2x32, C: 2x128 | MMAD -> Fixpipe -> slot reuse
//
// Handoff | Producer -> consumer | Reuse protection
// raw/UB | FIX ready -> AIV V | AIV free -> FIX; flags are directed owner IDs
// Local | AIV MTE3 ready -> MTE1 | next band starts after both right results
// stream | MTE2 ready -> MTE1 | MTE1_MTE2 credits per stream + per task
// DQ_FREE for BOTH owners must precede the AIV waiting for either RIGHT.
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH92_CUBE_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH92_CUBE_H
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 9201
#endif
#include <type_traits>
#include "chunk_kda_bwd_finalize_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace KDA::Arch92 {
using ArchTag = Catlass::Arch::Ascend960DT;
using LayoutRM = Catlass::layout::RowMajor;
using LayoutCM = Catlass::layout::ColumnMajor;
using CopyTransB =
    Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutRM, bfloat16_t, LayoutCM, float, LayoutRM>;
using CopyTransA =
    Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutCM, bfloat16_t, LayoutRM, float, LayoutRM>;
using CopyRegular =
    Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutRM, bfloat16_t, LayoutRM, float, LayoutRM>;
using CopyStage3 =
    Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, bfloat16_t, LayoutCM, bfloat16_t, LayoutCM, float, LayoutRM>;
using CopyTransBToUb =
    Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, bfloat16_t, LayoutRM, bfloat16_t, LayoutCM, float, LayoutRM>;
using CopyTransAToUbBf16 = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, bfloat16_t, LayoutCM, bfloat16_t,
                                                                      LayoutRM, bfloat16_t, LayoutRM>;
using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, bfloat16_t, typename CopyRegular::LayoutTagL1A>;

enum class ResultPath : uint32_t {
    GM_FP32,
    L1_BF16,
    L1_BF16_WITH_RAW,
    AIV_UB_FP32,
    AIV_UB_BF16,
};

static constexpr uint32_t L1_AKK = 0;
static constexpr uint32_t L1_DW = 32 * 1024;
static constexpr uint32_t L1_KE = 96 * 1024;
static constexpr uint32_t L1_ZB = 160 * 1024;
static constexpr uint32_t L1_STREAM = 192 * 1024;
static constexpr uint32_t STREAM_BYTES = 112 * 1024;
static constexpr uint32_t STREAM_VNEW = 0;
static constexpr uint32_t STREAM_DH = 16 * 1024;
static constexpr uint32_t STREAM_DVSCAN = 48 * 1024;
static constexpr uint32_t STREAM_H = 64 * 1024;
static constexpr uint32_t STREAM_V = 96 * 1024;
static constexpr uint32_t L1_TZA = 416 * 1024;
static constexpr uint32_t STAGE0_READY_COUNT = 4;
static constexpr uint32_t L0_BYTES = 32 * 1024;
static constexpr uint32_t L0C_BYTES = 128 * 1024;
static constexpr AscendC::FixpipeConfig FIX_NZ_L1 = {AscendC::CO2Layout::NZ, false};


struct CubeContext {
    GM_ADDR v_ = nullptr;
    GM_ADDR akk_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dh_ = nullptr;
    GM_ADDR dvScan_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    uint32_t l0Slot_ = 0;
};

__aicore__ inline void CubeInit(CubeContext &ctx, GM_ADDR v, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
                                GM_ADDR dvScan, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
                                const ChunkKdaBwdFinalizeTilingData *tiling);
__aicore__ inline GM_ADDR CubeStreamingInput(CubeContext &ctx, GM_ADDR address);
__aicore__ inline void CubeProcess(CubeContext &ctx);
__aicore__ inline void CubeReleaseTaskL1(CubeContext &ctx);
template <typename TileCopy, typename GmLayout>
__aicore__ inline void CubeLoadGmToL1A(CubeContext &ctx, AscendC::LocalTensor<bfloat16_t> dst, GM_ADDR src, uint32_t m,
                                       uint32_t k, uint32_t validM = 0, uint32_t validK = 0);
template <typename TileCopy, typename GmLayout>
__aicore__ inline void CubeLoadGmToL1B(CubeContext &ctx, AscendC::LocalTensor<bfloat16_t> dst, GM_ADDR src, uint32_t k,
                                       uint32_t n, uint32_t validK = 0, uint32_t validN = 0, uint32_t gmLeadingK = 0);
template <typename TileCopy, ResultPath PATH, bool SIGNAL_L1_READY = false, bool WAIT_AIV_FREE = true,
          bool CONCAT_RIGHT = false, uint32_t SPLIT = 0>
__aicore__ inline void
CubeRunGemm(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource, AscendC::LocalTensor<bfloat16_t> l1A,
            AscendC::LocalTensor<bfloat16_t> l1B, uint32_t m, uint32_t n, uint32_t k, uint32_t slot, GM_ADDR gmDst,
            AscendC::LocalTensor<bfloat16_t> l1Dst, uint32_t owner, uint32_t aivUbOffset, uint64_t freeFlag,
            uint64_t readyFlag, AscendC::LocalTensor<bfloat16_t> l1ASecond = {},
            AscendC::LocalTensor<bfloat16_t> l1BSecond = {});
__aicore__ inline AscendC::TEventID CubeStage0Ready(CubeContext &ctx, uint32_t stream, uint32_t stage);
__aicore__ inline void CubeLoadStage0(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                      const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner);
__aicore__ inline void CubeComputeStage0(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                         const FinalizeChunkInfo &chunk, uint32_t owner, int64_t coreIdx,
                                         uint64_t groupGeneration);
__aicore__ inline void CubeRunStage2(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner, int64_t coreIdx,
                                     uint64_t groupGeneration);
__aicore__ inline void CubeRunStage4(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner);
__aicore__ inline void CubeRunStage6(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner);
template <bool TRANSPOSE_A, bool CONCAT_RIGHT = false, bool WAIT_FREE = true, bool STACK_LEFT = false>
__aicore__ inline void CubeRunLocalGemm(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                        AscendC::LocalTensor<FinalizeLocalType> a,
                                        AscendC::LocalTensor<FinalizeLocalType> b, uint32_t owner, uint32_t ubOffset,
                                        uint64_t freeFlag, uint64_t readyFlag, uint32_t rowBegin, uint32_t validRows,
                                        AscendC::LocalTensor<FinalizeLocalType> aSecond = {},
                                        AscendC::LocalTensor<FinalizeLocalType> bSecond = {});
__aicore__ inline void CubeRunStage8(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner, uint32_t rowBegin);
__aicore__ inline void CubeRunStage10(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                      const FinalizeChunkInfo &chunk, uint32_t owner, uint32_t rowBegin);
__aicore__ inline uint32_t CubeNextSlot(CubeContext &ctx);
__aicore__ inline void CubeInitEvents(CubeContext &ctx);
__aicore__ inline void CubeDrainEvents(CubeContext &ctx);

__aicore__ inline void CubeInit(CubeContext &ctx, GM_ADDR v, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
                                GM_ADDR dvScan, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR workspace,
                                const ChunkKdaBwdFinalizeTilingData *tiling)
{
    ctx.v_ = v;
    ctx.akk_ = CubeStreamingInput(ctx, akk);
    ctx.vNew_ = CubeStreamingInput(ctx, vNew);
    ctx.h_ = h;
    ctx.dh_ = dh;
    ctx.dvScan_ = CubeStreamingInput(ctx, dvScan);
    ctx.cuSeqlens_ = cuSeqlens;
    ctx.chunkIndices_ = chunkIndices;
    ctx.workspace_ = workspace;
    ctx.tiling_ = tiling;
}

__aicore__ inline GM_ADDR CubeStreamingInput(CubeContext &ctx, GM_ADDR address)
{
    AscendC::GlobalTensor<bfloat16_t> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(address));
    tensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    return reinterpret_cast<GM_ADDR>(const_cast<__gm__ bfloat16_t *>(tensor.GetPhyAddr()));
}

__aicore__ inline void CubeProcess(CubeContext &ctx)
{
    AscendC::SetHF32Mode(false);
    Catlass::Arch::Resource<ArchTag> resource;
    CubeInitEvents(ctx);
    CubeReleaseTaskL1(ctx);
    const int64_t coreIdx = AscendC::GetBlockIdx();
    const int64_t coreNum = AscendC::GetBlockNum();
    uint64_t groupGeneration = 0;

    for (int64_t workTask = coreIdx; workTask < ctx.tiling_->workTaskNum; workTask += coreNum, ++groupGeneration) {
        const int64_t headWindow = workTask / ctx.tiling_->chunkTaskNum;
        const int64_t chunkTask = workTask - headWindow * ctx.tiling_->chunkTaskNum;
        const int64_t headBegin = headWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
        const int64_t headEnd = FinalizeMin(headBegin + KDA_FINALIZE_HEADS_PER_WINDOW, ctx.tiling_->NV);
        FinalizeChunkInfo chunk;
        ResolveFinalizeChunk(chunkTask, ctx.cuSeqlens_, ctx.chunkIndices_, *ctx.tiling_, chunk);
        if (!chunk.valid) {
            continue;
        }

        // Prime both 112-KiB streams before consuming them in owner order.
        // The task-level event protects their alias with Local operands.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
        const int64_t preloadEnd = FinalizeMin(headBegin + 2, headEnd);
        for (int64_t head = headBegin; head < preloadEnd; ++head) {
            const uint32_t owner = static_cast<uint32_t>(head - headBegin);
            CubeLoadStage0(ctx, resource, chunk, head, owner);
        }
        for (int64_t head = headBegin; head < headEnd; ++head) {
            const uint32_t owner = static_cast<uint32_t>(head - headBegin);
            CubeComputeStage0(ctx, resource, chunk, owner, coreIdx, groupGeneration);
        }
        // Stage2 starts per head as soon as both dW and kE are ready; it
        // does not wait for a group-wide Vector barrier.
        for (int64_t head = headBegin; head < headEnd; ++head) {
            const uint32_t owner = static_cast<uint32_t>(head - headBegin);
            CubeRunStage2(ctx, resource, chunk, owner, coreIdx, groupGeneration);
        }
        // Stage4 consumes each Zb as soon as its owner AIV publishes the
        // L1 slot.  Akk^T is the same nZ resident copy loaded by Stage0.
        for (int64_t head = headBegin; head < headEnd; ++head) {
            const uint32_t owner = static_cast<uint32_t>(head - headBegin);
            CubeRunStage4(ctx, resource, chunk, owner);
        }
        // Stage6 consumes AkkT/Tza and writes each owner's FP32 dAkk.
        // The first band converts it before dG reuses the same address.
        for (int64_t head = headBegin; head < headEnd; ++head) {
            const uint32_t owner = static_cast<uint32_t>(head - headBegin);
            CubeRunStage6(ctx, resource, chunk, owner);
        }

        // Each row band has its own stable gate anchor. The existing
        // residual GEMMs operate on 32 output rows, with no extra FLOPs
        // for full chunks. Each band reuses the same owner L1 slot.
        for (uint32_t rowBegin = 0; rowBegin < chunk.validRows; rowBegin += KDA_FINALIZE_INTRA_ROWS) {
            for (int64_t head = headBegin; head < headEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                CubeRunStage8(ctx, resource, chunk, owner, rowBegin);
            }
            for (int64_t head = headBegin; head < headEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                CubeRunStage10(ctx, resource, chunk, owner, rowBegin);
            }
        }
        // Next task's streams alias this task's LocalOperand owners.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
        CubeReleaseTaskL1(ctx);
    }
    CubeDrainEvents(ctx);
}

__aicore__ inline void CubeReleaseTaskL1(CubeContext &ctx)
{
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(KDA_FINALIZE_TASK_L1_FREE);
}

template <typename TileCopy, typename GmLayout>
__aicore__ inline void CubeLoadGmToL1A(CubeContext &ctx, AscendC::LocalTensor<bfloat16_t> dst, GM_ADDR src, uint32_t m,
                                       uint32_t k, uint32_t validM, uint32_t validK)
{
    validM = validM == 0 ? m : validM;
    validK = validK == 0 ? k : validK;
    if (validM != m || validK != k) {
        AscendC::Fill(
            dst.template ReinterpretCast<uint16_t>(),
            AscendC::InitConstValueParams<uint16_t>(1, static_cast<uint16_t>(m * k * sizeof(bfloat16_t) / 32), 0, 0));
        // Clearing and ND-to-NZ DMA write overlapping L1 addresses.
        // Finish the clear before loading valid rows, or its late writes
        // can zero the last column blocks of a short tile.
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    AscendC::GlobalTensor<bfloat16_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));
    auto tensorGm = tla::MakeTensor(gm, tla::MakeLayout<bfloat16_t, GmLayout>(m, k), Catlass::Arch::PositionGM{});
    auto block = tla::GetTile(tensorGm, tla::MakeCoord(0, 0), tla::MakeShape(validM, validK));
    auto tensorL1 = tla::MakeTensor(dst, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1A>(m, k),
                                    Catlass::Arch::PositionL1{});
    typename TileCopy::template CopyGmToL1A<decltype(block)>{}(tensorL1, block);
}

template <typename TileCopy, typename GmLayout>
__aicore__ inline void CubeLoadGmToL1B(CubeContext &ctx, AscendC::LocalTensor<bfloat16_t> dst, GM_ADDR src, uint32_t k,
                                       uint32_t n, uint32_t validK, uint32_t validN, uint32_t gmLeadingK)
{
    validK = validK == 0 ? k : validK;
    validN = validN == 0 ? n : validN;
    if (validK != k || validN != n) {
        AscendC::Fill(
            dst.template ReinterpretCast<uint16_t>(),
            AscendC::InitConstValueParams<uint16_t>(1, static_cast<uint16_t>(k * n * sizeof(bfloat16_t) / 32), 0, 0));
        // Same L1 write-after-write dependency as LoadGmToL1A.
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    AscendC::GlobalTensor<bfloat16_t> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));
    auto tensorGm = tla::MakeTensor(gm, tla::MakeLayout<bfloat16_t, GmLayout>(gmLeadingK ? gmLeadingK : k, n),
                                    Catlass::Arch::PositionGM{});
    auto block = tla::GetTile(tensorGm, tla::MakeCoord(0, 0), tla::MakeShape(validK, validN));
    auto tensorL1 = tla::MakeTensor(dst, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1B>(k, n),
                                    Catlass::Arch::PositionL1{});
    typename TileCopy::template CopyGmToL1B<decltype(block)>{}(tensorL1, block);
}

template <typename TileCopy, ResultPath PATH, bool SIGNAL_L1_READY, bool WAIT_AIV_FREE, bool CONCAT_RIGHT,
          uint32_t SPLIT>
__aicore__ inline void
CubeRunGemm(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource, AscendC::LocalTensor<bfloat16_t> l1A,
            AscendC::LocalTensor<bfloat16_t> l1B, uint32_t m, uint32_t n, uint32_t k, uint32_t slot, GM_ADDR gmDst,
            AscendC::LocalTensor<bfloat16_t> l1Dst, uint32_t owner, uint32_t aivUbOffset, uint64_t freeFlag,
            uint64_t readyFlag, AscendC::LocalTensor<bfloat16_t> l1ASecond, AscendC::LocalTensor<bfloat16_t> l1BSecond)
{
    auto l0A = resource.l0ABuf.template GetBufferByByte<bfloat16_t>(slot * L0_BYTES);
    auto l0B = resource.l0BBuf.template GetBufferByByte<bfloat16_t>(slot * L0_BYTES);
    auto l0C = resource.l0CBuf.template GetBufferByByte<float>(slot * L0C_BYTES);
    auto tensorL1A = tla::MakeTensor(l1A, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1A>(m, k),
                                     Catlass::Arch::PositionL1{});
    auto tensorL1B = tla::MakeTensor(l1B, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1B>(k, n),
                                     Catlass::Arch::PositionL1{});
    auto tensorL0A = tla::MakeTensor(l0A, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL0A>(m, k),
                                     Catlass::Arch::PositionL0A{});
    auto tensorL0B = tla::MakeTensor(l0B, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL0B>(k, n),
                                     Catlass::Arch::PositionL0B{});
    auto tensorL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
    auto tileA = tla::GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
    auto tileB = tla::GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
    auto tileC = tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
    if constexpr (CONCAT_RIGHT) {
        // Assemble the two reduction segments in L0, retaining the four
        // original L1 operands and initializing L0C in a single MMAD.
        auto aFirst = tla::MakeTensor(l1A, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1A>(m, k / 2),
                                      Catlass::Arch::PositionL1{});
        auto aSecond =
            tla::MakeTensor(l1ASecond, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1A>(m, k / 2),
                            Catlass::Arch::PositionL1{});
        auto bFirst = tla::MakeTensor(l1B, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1B>(k / 2, n),
                                      Catlass::Arch::PositionL1{});
        auto bSecond =
            tla::MakeTensor(l1BSecond, tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1B>(k / 2, n),
                            Catlass::Arch::PositionL1{});
        auto a0 = tla::GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k / 2));
        auto a1 = tla::GetTile(tensorL0A, tla::MakeCoord(0, k / 2), tla::MakeShape(m, k / 2));
        auto b0 = tla::GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k / 2, n));
        auto b1 = tla::GetTile(tensorL0B, tla::MakeCoord(k / 2, 0), tla::MakeShape(k / 2, n));
        auto sa0 = tla::GetTile(aFirst, tla::MakeCoord(0, 0), tla::MakeShape(m, k / 2));
        auto sa1 = tla::GetTile(aSecond, tla::MakeCoord(0, 0), tla::MakeShape(m, k / 2));
        auto sb0 = tla::GetTile(bFirst, tla::MakeCoord(0, 0), tla::MakeShape(k / 2, n));
        auto sb1 = tla::GetTile(bSecond, tla::MakeCoord(0, 0), tla::MakeShape(k / 2, n));
        typename TileCopy::CopyL1ToL0A{}(a0, sa0);
        typename TileCopy::CopyL1ToL0A{}(a1, sa1);
        typename TileCopy::CopyL1ToL0B{}(b0, sb0);
        typename TileCopy::CopyL1ToL0B{}(b1, sb1);
    } else {
        typename TileCopy::CopyL1ToL0A{}(tensorL0A, tileA);
        typename TileCopy::CopyL1ToL0B{}(tensorL0B, tileB);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
    Catlass::Gemm::Tile::TileMmadTla<ArchTag, bfloat16_t, typename TileCopy::LayoutTagL1A>{}(
        tileC, tensorL0A, tensorL0B, m, n, k, true, 0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
    if constexpr (SPLIT != 0) {
        for (uint32_t correction = 0; correction < (SPLIT == 3 ? 2U : 1U); ++correction) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            const bool lowA = SPLIT == 1 || (SPLIT == 3 && correction == 1);
            auto residualA = tla::MakeTensor(lowA ? l1ASecond : l1A,
                                             tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1A>(m, k),
                                             Catlass::Arch::PositionL1{});
            auto residualB = tla::MakeTensor(lowA ? l1B : l1BSecond,
                                             tla::MakeLayout<bfloat16_t, typename TileCopy::LayoutTagL1B>(k, n),
                                             Catlass::Arch::PositionL1{});
            auto extraA = tla::GetTile(residualA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
            auto extraB = tla::GetTile(residualB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
            typename TileCopy::CopyL1ToL0A{}(tensorL0A, extraA);
            typename TileCopy::CopyL1ToL0B{}(tensorL0B, extraB);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
            Catlass::Gemm::Tile::TileMmadTla<ArchTag, bfloat16_t, typename TileCopy::LayoutTagL1A>{}(
                tileC, tensorL0A, tensorL0B, m, n, k, false, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(slot);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(slot);

    if constexpr (PATH == ResultPath::GM_FP32) {
        AscendC::GlobalTensor<float> dst;
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmDst));
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = n;
        params.mSize = m;
        params.srcStride = (m + 15U) / 16U * 16U;
        params.dstStride = n;
        params.quantPre = QuantMode_t::NoQuant;
        params.unitFlag = 0;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dst, l0C, params);
    } else if constexpr (PATH == ResultPath::L1_BF16 || PATH == ResultPath::L1_BF16_WITH_RAW) {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> params;
        params.nSize = n;
        params.mSize = m;
        params.srcStride = (m + 15U) / 16U * 16U;
        params.dstStride = KDA_FINALIZE_CHUNK * 16U;
        params.quantPre = QuantMode_t::F322BF16;
        params.unitFlag = 0;
        AscendC::Fixpipe<bfloat16_t, float, FIX_NZ_L1>(l1Dst, l0C, params);
        if constexpr (PATH == ResultPath::L1_BF16_WITH_RAW) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(freeFlag + owner);
            auto dst = resource.ubBuf.template GetBufferByByte<float>(aivUbOffset);
            auto tensorUb = tla::MakeTensor(dst, tla::MakeLayout<float, LayoutRM>(m, n), Catlass::Arch::PositionUB{});
            auto tileUb = tla::GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
            typename CopyTransBToUb::template CopyL0CToDst<decltype(tileUb)>{}(tileUb, tensorL0C, false, 0);
            AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(readyFlag + owner);
        }
        if constexpr (SIGNAL_L1_READY) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<AscendC::TEventID>(readyFlag));
        }
    } else if constexpr (PATH == ResultPath::AIV_UB_FP32) {
        if constexpr (WAIT_AIV_FREE) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(freeFlag + owner);
        }
        auto remoteUb = resource.ubBuf.template GetBufferByByte<float>(aivUbOffset);
        auto tensorUb = tla::MakeTensor(remoteUb, tla::MakeLayout<float, LayoutRM>(m, n), Catlass::Arch::PositionUB{});
        auto blockUb = tla::GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        typename CopyTransBToUb::template CopyL0CToDst<decltype(blockUb)>{}(blockUb, tensorL0C, false, 0);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(readyFlag + owner);
    } else {
        if constexpr (WAIT_AIV_FREE) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(freeFlag + owner);
        }
        auto remoteUb = resource.ubBuf.template GetBufferByByte<bfloat16_t>(aivUbOffset);
        auto tensorUb =
            tla::MakeTensor(remoteUb, tla::MakeLayout<bfloat16_t, LayoutRM>(m, n), Catlass::Arch::PositionUB{});
        auto blockUb = tla::GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        typename CopyTransAToUbBf16::template CopyL0CToDst<decltype(blockUb)>{}(blockUb, tensorL0C, false, 0);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(readyFlag + owner);
    }
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
}

__aicore__ inline AscendC::TEventID CubeStage0Ready(CubeContext &ctx, uint32_t stream, uint32_t stage)
{
    return static_cast<AscendC::TEventID>(stream * STAGE0_READY_COUNT + stage);
}

__aicore__ inline void CubeLoadStage0(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                      const FinalizeChunkInfo &chunk, int64_t head, uint32_t owner)
{
    constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
    const uint32_t stream = owner & 1U;
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(stream);
    const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
    auto akkL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto vNewL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_VNEW);
    auto dhL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_DH);
    auto dvScanL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_DVSCAN);
    auto hL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_H);
    auto vL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_V);

    const int64_t token = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_DIM);
    const int64_t akkToken = FinalizeTokenOffset(*ctx.tiling_, chunk, head, KDA_FINALIZE_CHUNK);
    const int64_t hState = FinalizeHOffset(*ctx.tiling_, chunk, head);
    const int64_t state = FinalizeDhOffset(*ctx.tiling_, chunk, head);
    CubeLoadGmToL1A<CopyTransB, LayoutRM>(ctx, vNewL1, ctx.vNew_ + token * sizeof(bfloat16_t), rows, KDA_FINALIZE_DIM,
                                          chunk.validRows);
    CubeLoadGmToL1B<CopyTransB, LayoutCM>(ctx, dhL1, ctx.dh_ + state * sizeof(bfloat16_t), KDA_FINALIZE_DIM,
                                          KDA_FINALIZE_DIM);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 0));

    CubeLoadGmToL1A<CopyTransA, LayoutCM>(ctx, akkL1, ctx.akk_ + akkToken * sizeof(bfloat16_t), rows, rows,
                                          chunk.validRows, chunk.validRows);
    CubeLoadGmToL1A<CopyRegular, LayoutRM>(ctx, dvScanL1, ctx.dvScan_ + token * sizeof(bfloat16_t), rows,
                                           KDA_FINALIZE_DIM, chunk.validRows);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 1));

    CubeLoadGmToL1B<CopyTransB, LayoutCM>(ctx, hL1, ctx.h_ + hState * sizeof(bfloat16_t), KDA_FINALIZE_DIM,
                                          KDA_FINALIZE_DIM);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 2));
    CubeLoadGmToL1B<CopyTransB, LayoutCM>(
        ctx, vL1,
        ctx.v_ + FinalizeOutputOffset(*ctx.tiling_, chunk, head, 0, KDA_FINALIZE_DIM, ctx.tiling_->vTokenMajor != 0U) *
                     sizeof(bfloat16_t),
        KDA_FINALIZE_DIM, rows, KDA_FINALIZE_DIM, chunk.validRows,
        ctx.tiling_->vTokenMajor ? ctx.tiling_->NV * KDA_FINALIZE_DIM : KDA_FINALIZE_DIM);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 3));
}

__aicore__ inline void CubeComputeStage0(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                         const FinalizeChunkInfo &chunk, uint32_t owner, int64_t coreIdx,
                                         uint64_t groupGeneration)
{
    constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
    const uint32_t stream = owner & 1U;
    const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
    auto akkL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto dwL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_DW + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
    auto vNewL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_VNEW);
    auto dhL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_DH);
    auto dvScanL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_DVSCAN);
    auto hL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_H);
    auto vL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(streamBase + STREAM_V);

    const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
    GM_ADDR wsBase = ctx.workspace_ + ws;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 0));
    CubeRunGemm<CopyTransB, ResultPath::GM_FP32>(ctx, resource, vNewL1, dhL1, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM,
                                                 CubeNextSlot(ctx), wsBase + KDA_FINALIZE_WS_DK_STATE_RAW, {}, 0, 0, 0,
                                                 0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 1));
    CubeRunGemm<CopyTransA, ResultPath::GM_FP32>(ctx, resource, akkL1, dvScanL1, rows, KDA_FINALIZE_DIM, rows,
                                                 CubeNextSlot(ctx), wsBase + KDA_FINALIZE_WS_DVB, {}, 0, 0, 0, 0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 2));
    CubeRunGemm<CopyTransB, ResultPath::L1_BF16_WITH_RAW>(
        ctx, resource, dvScanL1, hL1, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM, CubeNextSlot(ctx), nullptr, dwL1, owner,
        kDwRaw, KDA_FINALIZE_LOCAL_READY_BASE, KDA_FINALIZE_ZB_READY_BASE);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(CubeStage0Ready(ctx, stream, 3));
    CubeRunGemm<CopyTransB, ResultPath::AIV_UB_FP32>(
        ctx, resource, dvScanL1, vL1, rows, rows, KDA_FINALIZE_DIM, CubeNextSlot(ctx), nullptr, {}, owner,
        kZv + owner * KDA_FINALIZE_MATRIX_FP32_BYTES, KDA_FINALIZE_ZV_FREE_BASE, KDA_FINALIZE_ZV_READY_BASE);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(stream);
}

__aicore__ inline void CubeRunStage2(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner, int64_t coreIdx,
                                     uint64_t groupGeneration)
{
    constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
    auto akkL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto dwL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_DW + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
    auto kEL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_KE + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
    const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
    GM_ADDR wsBase = ctx.workspace_ + ws;
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(KDA_FINALIZE_KE_READY_BASE + owner);
    auto dwLow =
        resource.l1Buf.template GetBufferByByte<bfloat16_t>(64 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
    auto keLow =
        resource.l1Buf.template GetBufferByByte<bfloat16_t>(128 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
    // Akk retains the ColumnMajor L1 layout loaded for Stage0 DVb.
    // Preserve that interpretation here to compute Akk^T @ dW.
    CubeRunGemm<CopyTransA, ResultPath::GM_FP32, false, true, false, 2>(
        ctx, resource, akkL1, dwL1, rows, KDA_FINALIZE_DIM, rows, CubeNextSlot(ctx), wsBase + KDA_FINALIZE_WS_DKGB_RAW,
        {}, 0, 0, 0, 0, {}, dwLow);
    // The combined ready above protects both dW and kE residual planes.
    CubeRunGemm<CopyTransB, ResultPath::AIV_UB_FP32, false, true, false, 3>(
        ctx, resource, dwL1, kEL1, rows, rows, KDA_FINALIZE_DIM, CubeNextSlot(ctx), nullptr, {}, owner,
        kZw + owner * KDA_FINALIZE_MATRIX_FP32_BYTES, KDA_FINALIZE_ZW_FREE_BASE, KDA_FINALIZE_ZW_READY_BASE, dwLow,
        keLow);
}

__aicore__ inline void CubeRunStage4(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner)
{
    constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(KDA_FINALIZE_ZB_READY_BASE + owner);
    auto zBL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_ZB + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto akkL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto tzaL1 =
        resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_TZA + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto zBLow =
        resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_ZB + (owner * 2 + 1) * KDA_FINALIZE_MATRIX_BF16_BYTES);
    CubeRunGemm<CopyTransB, ResultPath::L1_BF16_WITH_RAW, false, true, false, 1>(
        ctx, resource, zBL1, akkL1, rows, rows, rows, CubeNextSlot(ctx), nullptr, tzaL1, owner, 0,
        KDA_FINALIZE_ZV_FREE_BASE, KDA_FINALIZE_ZW_READY_BASE, zBLow, {});
    // PIPE_FIX is deliberately used for the return credit, matching the
    // mature fwd_h L1 producer/consumer protocol.  It is conservative:
    // Zb is certainly no longer read when the Tza FixPipe completes.
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(KDA_FINALIZE_ZB_FREE_BASE + owner);
}

__aicore__ inline void CubeRunStage6(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner)
{
    constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
    auto akkL1 = resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto tzaL1 =
        resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_TZA + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
    auto tzaLow =
        resource.l1Buf.template GetBufferByByte<bfloat16_t>(L1_TZA + (owner * 2 + 1) * KDA_FINALIZE_MATRIX_BF16_BYTES);
    // dAkk raw occupies the owner's not-yet-live dG frame. Both heads
    // retain it until their first-band local operands have been built.
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(KDA_FINALIZE_KE_READY_BASE + owner);
    CubeRunGemm<CopyTransA, ResultPath::AIV_UB_FP32, false, false, false, 2>(
        ctx, resource, akkL1, tzaL1, rows, rows, rows, CubeNextSlot(ctx), nullptr, {}, owner, HeadBase(owner) + kHeadDg,
        KDA_FINALIZE_DAKK_FREE_BASE, KDA_FINALIZE_DAKK_READY_BASE, {}, tzaLow);
}

template <bool TRANSPOSE_A, bool CONCAT_RIGHT, bool WAIT_FREE, bool STACK_LEFT>
__aicore__ inline void
CubeRunLocalGemm(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                 AscendC::LocalTensor<FinalizeLocalType> a, AscendC::LocalTensor<FinalizeLocalType> b, uint32_t owner,
                 uint32_t ubOffset, uint64_t freeFlag, uint64_t readyFlag, uint32_t rowBegin, uint32_t validRows,
                 AscendC::LocalTensor<FinalizeLocalType> aSecond, AscendC::LocalTensor<FinalizeLocalType> bSecond)
{
    using ALayout = std::conditional_t<TRANSPOSE_A, LayoutCM, LayoutRM>;
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, FinalizeLocalType, ALayout, FinalizeLocalType,
                                                        LayoutRM, float, LayoutRM>;
    const uint32_t slot = CubeNextSlot(ctx);
    auto l0a = resource.l0ABuf.template GetBufferByByte<FinalizeLocalType>(slot * L0_BYTES);
    auto l0c = resource.l0CBuf.template GetBufferByByte<float>(slot * L0C_BYTES);
    // Causal tiles only consume columns up to this band for A, or
    // rows starting at this band for A transpose. Round K for L0Copy.
    const uint32_t kBegin = TRANSPOSE_A ? rowBegin : 0U;
    const uint32_t kEnd = TRANSPOSE_A ? validRows : FinalizeMin(validRows, rowBegin + KDA_FINALIZE_INTRA_ROWS);
    const uint32_t reduction = (kEnd - kBegin + 7U) / 8U * 8U;
    const uint32_t tileK = CONCAT_RIGHT ? 2U * reduction : reduction;
    const bool wideB = tileK * KDA_FINALIZE_DIM * sizeof(FinalizeLocalType) > L0_BYTES;
    auto l0b = resource.l0BBuf.template GetBufferByByte<FinalizeLocalType>(wideB ? 0U : slot * L0_BYTES);
    constexpr uint32_t outputRows = STACK_LEFT ? 2U * KDA_FINALIZE_INTRA_ROWS : KDA_FINALIZE_INTRA_ROWS;
    auto ta = tla::MakeTensor(l0a, tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL0A>(outputRows, tileK),
                              Catlass::Arch::PositionL0A{});
    auto tb = tla::MakeTensor(l0b, tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL0B>(tileK, 128),
                              Catlass::Arch::PositionL0B{});
    auto tc = tla::MakeTensor(l0c, tla::MakeLayoutL0C(outputRows, 128), Catlass::Arch::PositionL0C{});
    auto tileC = tla::GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(outputRows, 128));
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
    // A full transposed band uses both L0B slots. Hold both B credits
    // until its concatenated FP32 product has consumed the operands.
    if (wideB) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * (slot ^ 1U) + 1);
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
    for (uint32_t part = 0; part < (STACK_LEFT || CONCAT_RIGHT ? 2U : 1U); ++part) {
        const bool second = part != 0;
        auto sourceA = second ? aSecond : a;
        auto sourceB = CONCAT_RIGHT && second ? bSecond : b;
        const uint32_t dstK = CONCAT_RIGHT ? part * reduction : 0U;
        auto sa = tla::MakeTensor(sourceA, tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL1A>(64, 64),
                                  Catlass::Arch::PositionL1{});
        auto sb = tla::MakeTensor(sourceB, tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL1B>(64, 128),
                                  Catlass::Arch::PositionL1{});
        auto tileA =
            tla::GetTile(sa, tla::MakeCoord(rowBegin, kBegin), tla::MakeShape(KDA_FINALIZE_INTRA_ROWS, reduction));
        auto tileB = tla::GetTile(sb, tla::MakeCoord(kBegin, 0), tla::MakeShape(reduction, 128));
        auto dstA = tla::GetTile(ta, tla::MakeCoord(STACK_LEFT ? part * KDA_FINALIZE_INTRA_ROWS : 0U, dstK),
                                 tla::MakeShape(KDA_FINALIZE_INTRA_ROWS, reduction));
        auto dstB = tla::GetTile(tb, tla::MakeCoord(dstK, 0), tla::MakeShape(reduction, 128));
        typename Copy::CopyL1ToL0A{}(dstA, tileA);
        if (!STACK_LEFT || part == 0) {
            typename Copy::CopyL1ToL0B{}(dstB, tileB);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
    Catlass::Gemm::Tile::TileMmadTla<ArchTag, FinalizeLocalType, typename Copy::LayoutTagL1A>{}(
        tileC, ta, tb, outputRows, 128, tileK, true, 0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
    if (wideB) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * (slot ^ 1U) + 1);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(slot);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(slot);
    if constexpr (WAIT_FREE) {
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(freeFlag + owner);
    }
    auto dst = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    auto tu = tla::MakeTensor(dst, tla::MakeLayout<float, LayoutRM>(outputRows, 128), Catlass::Arch::PositionUB{});
    auto tileU = tla::GetTile(tu, tla::MakeCoord(0, 0), tla::MakeShape(outputRows, 128));
    typename CopyTransBToUb::template CopyL0CToDst<decltype(tileU)>{}(tileU, tc, false, 0);
    AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(readyFlag + owner);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
}

__aicore__ inline void CubeRunStage8(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                     const FinalizeChunkInfo &chunk, uint32_t owner, uint32_t rowBegin)
{
    AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(KDA_FINALIZE_LOCAL_READY_BASE + owner);
    auto local = resource.l1Buf.template GetBufferByByte<FinalizeLocalType>(KDA_FINALIZE_LOCAL_BASE +
                                                                            owner * KDA_FINALIZE_LOCAL_BYTES);
    auto dAqkL1 = local[KDA_FINALIZE_LOCAL_DAQK / sizeof(FinalizeLocalType)];
    auto kNegL1 = local[KDA_FINALIZE_LOCAL_K_NEG / sizeof(FinalizeLocalType)];
    // Stack dq and left vertically: both multiply the same kNeg. Each
    // band produces [dq; left] in a single 64-row FP32 UB result.
    CubeRunLocalGemm<false, false, true, true>(ctx, resource, dAqkL1, kNegL1, owner, ResultBase(owner),
                                               KDA_FINALIZE_DQ_LOCAL_FREE_BASE, KDA_FINALIZE_DQ_LOCAL_READY_BASE,
                                               rowBegin, chunk.validRows,
                                               local[KDA_FINALIZE_LOCAL_DAKK / sizeof(FinalizeLocalType)]);
}

__aicore__ inline void CubeRunStage10(CubeContext &ctx, Catlass::Arch::Resource<ArchTag> &resource,
                                      const FinalizeChunkInfo &chunk, uint32_t owner, uint32_t rowBegin)
{
    auto local = resource.l1Buf.template GetBufferByByte<FinalizeLocalType>(KDA_FINALIZE_LOCAL_BASE +
                                                                            owner * KDA_FINALIZE_LOCAL_BYTES);
    auto daq = local[KDA_FINALIZE_LOCAL_DAQK / sizeof(FinalizeLocalType)];
    auto dak = local[KDA_FINALIZE_LOCAL_DAKK / sizeof(FinalizeLocalType)];
    auto qp = local[KDA_FINALIZE_LOCAL_Q_POS / sizeof(FinalizeLocalType)];
    auto bp = local[KDA_FINALIZE_LOCAL_BK_POS / sizeof(FinalizeLocalType)];
    CubeRunLocalGemm<true, true>(ctx, resource, daq, qp, owner, RightBase(owner), KDA_FINALIZE_KE_READY_BASE,
                                 KDA_FINALIZE_ZW_READY_BASE, rowBegin, chunk.validRows, dak, bp);
}

__aicore__ inline uint32_t CubeNextSlot(CubeContext &ctx)
{
    const uint32_t current = ctx.l0Slot_;
    ctx.l0Slot_ ^= 1U;
    return current;
}

__aicore__ inline void CubeInitEvents(CubeContext &ctx)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
    }
}

__aicore__ inline void CubeDrainEvents(CubeContext &ctx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
    }
}

} // namespace KDA::Arch92
#endif
