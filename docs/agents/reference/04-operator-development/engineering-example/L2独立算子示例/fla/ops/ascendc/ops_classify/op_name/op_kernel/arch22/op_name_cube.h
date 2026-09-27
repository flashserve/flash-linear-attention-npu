/**
 * 示例文件：.../op_kernel/arch22/op_name_cube.h（A2/A3 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *          （计算层的 K 分块累加步骤另参考 chunk_gated_delta_rule_bwd_dhu_cube.h 的 RunResidentMmad）
 *
 * 与 arch35 版本**同骨架、同命名**：traits 类型与 layout 层 → CubeChunkStateGemm 计算层 →
 * 数据结构体（只放数据）→ 行为层函数（Init/Process/Stage/收尾）。差异只在下表：
 *
 * ── 平台差异表（与 arch35 逐条对照）──────────────────────────────────────────
 *   项               arch22（A2/A3）                        arch35（A5）
 *   CATLASS_ARCH     2201                                   3510
 *   ArchTag          Catlass::Arch::AtlasA2                 Catlass::Arch::Ascend950
 *   L1/L0 偏移       同名常量（见 arch22/op_name_struct.h）   同名常量（见 arch35/op_name_struct.h）
 *   L0C 容量上限     128 KiB × 2（A2/A3 L0C 较小）           256 KiB × 2
 *   Fixpipe 目标     等价语义                                可用 FixpipeConfig 指定 NZ/L1 目标
 *   事件协议         MTE1_MTE2 / MTE2_MTE1 / M_MTE1(A/B 分开) / MTE1_M / FIX_M / M_FIX：同 arch35
 *   计算层           同一个 CubeChunkStateGemm（K 分块 + unit flag 累加 + L0C 搬出）
 *
 * ── Stage 表 / L1+L0 布局表 / 同步协议表 ─────────────────────────────────────
 *   与 arch35 一致：S1a CopyIn（含 MTE2_MTE1 发布）→ S1b Matmul（K 分块 MMAD）→ S1c Fixpipe（释放
 *   L0C 并广播给 AIV）；L1 只由 AIC 写入，没有 UB→L1 通路；事件 id 与物理槽一一对应。
 */

#ifndef OP_NAME_CUBE_ARCH22_H
#define OP_NAME_CUBE_ARCH22_H

// Catlass 按架构选择编译分支：A2/A3 = 2201，A5 = 3510。
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

// ══ ② 类型与 layout 层：与 arch35 同名同序，只换 ArchTag ══════════════════════

template <typename DT>
struct OpNameCubePrimitives {
    using ArchTag = Catlass::Arch::AtlasA2;
    using LayoutRowMajor = Catlass::layout::RowMajor;
    using LayoutColumnMajor = Catlass::layout::ColumnMajor;

    using TileCopyState = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyStateToUb = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using ElementAccumulator = typename TileCopyState::ElementAccumulator;
    using CopyStateGmToL1A = typename TileCopyState::template CopyGmToL1A;
    using CopyStateGmToL1B = typename TileCopyState::template CopyGmToL1B;
    using CopyStateL1ToL0A = typename TileCopyState::CopyL1ToL0A;
    using CopyStateL1ToL0B = typename TileCopyState::CopyL1ToL0B;
    using CopyStateL0CToDst = typename TileCopyStateToUb::template CopyL0CToDst;
    using TileMmadState = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, DT, typename TileCopyState::LayoutTagL1A>;

    static constexpr AscendC::FixpipeConfig FIXPIPE_NZ_CONFIG = {AscendC::CO2Layout::NZ, false};

    static constexpr auto GM_NORM_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<DIM_128>{});
    static constexpr auto L1_NORM_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyState::LayoutTagL1A>(
            tla::Int<CHUNK_SIZE_64>{}, tla::Int<DIM_128>{});
    static constexpr auto L1_STATE_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyState::LayoutTagL1B>(
            tla::Int<DIM_128>{}, tla::Int<CHUNK_SIZE_64>{});
};

// L0 tile 形状：与 arch35 保持同一个 K 分块粒度，便于两平台对照与 baseline 比较。
constexpr uint32_t CUBE_K_TILE = 64;

static_assert(L1_STATE_OFFSET + L1_SLOT_COUNT_2 * L1_TILE_BYTES <= L1_TOTAL_BYTES,
              "L1 state 常驻槽必须落在 L1_TOTAL_BYTES 内。");
static_assert(L0A_BYTES * L0_BUFFER_COUNT_2 <= 64 * 1024, "L0A ping/pong 超出 L0A 容量。");
static_assert(L0B_BYTES * L0_BUFFER_COUNT_2 <= 64 * 1024, "L0B ping/pong 超出 L0B 容量。");
static_assert(L0C_BYTES * L0_BUFFER_COUNT_2 <= 128 * 1024, "L0C ping/pong 超出 A2/A3 的 L0C 容量。");

// ══ ③ Cube 计算层：与 arch35 同名同序（K 分块 + MMAD 累加 + 搬出一次调用）═════

template <typename Ctx, typename CopyL1ToL0A, typename CopyL1ToL0B, typename TileMmad,
          typename CopyL0CToDst, typename TensorL1A, typename TensorL1B, typename TensorDst>
__aicore__ inline void CubeChunkStateGemm(
    Ctx &ctx, CopyL1ToL0A &copyL1ToL0A, CopyL1ToL0B &copyL1ToL0B, TileMmad &tileMmad,
    CopyL0CToDst &copyL0CToDst, TensorL1A &tensorL1A, TensorL1B &tensorL1B, TensorDst &tensorDst,
    uint32_t m, uint32_t n, uint32_t k, int32_t l1Event, bool waitL1Ready, bool releaseL1AfterUse)
{
    using DT = typename Ctx::DT;
    using ElementAccumulator = typename Ctx::ElementAccumulator;
    using LayoutTagL0A = typename Ctx::Prims::TileCopyState::LayoutTagL0A;
    using LayoutTagL0B = typename Ctx::Prims::TileCopyState::LayoutTagL0B;
    const uint32_t mActual = (m == 1U) ? 16U : m;
    const uint32_t l0cSlot = ctx.l0cSlot;
    const int32_t l0cFreeEvent = static_cast<int32_t>(l0cSlot);   // FIX_M：L0C 可写
    const int32_t l0cDoneEvent = static_cast<int32_t>(l0cSlot);   // M_FIX：L0C 已写完

    auto tensorL0C = tla::MakeTensor(
        ctx.l0C[l0cSlot], tla::MakeLayoutL0C(mActual, n), Catlass::Arch::PositionL0C{});
    auto tensorTileL0C =
        tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, n));

    for (uint32_t kOffset = 0; kOffset < k; kOffset += CUBE_K_TILE) {
        const uint32_t curK = (kOffset + CUBE_K_TILE > k) ? (k - kOffset) : CUBE_K_TILE;
        const bool firstK = (kOffset == 0U);
        const bool lastK = (kOffset + curK >= k);
        const uint32_t l0Slot = ctx.l0Slot;
        const int32_t l0AEvent = static_cast<int32_t>(2U * l0Slot);
        const int32_t l0BEvent = static_cast<int32_t>(2U * l0Slot + 1U);
        const int32_t l0ReadyEvent = l0AEvent;

        auto tensorL0A = tla::MakeTensor(
            ctx.l0A[l0Slot], tla::MakeLayout<DT, LayoutTagL0A>(mActual, curK),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            ctx.l0B[l0Slot], tla::MakeLayout<DT, LayoutTagL0B>(curK, n),
            Catlass::Arch::PositionL0B{});

        if (waitL1Ready) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
            waitL1Ready = false;
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
        auto tileL1A = tla::GetTile(tensorL1A, tla::MakeCoord(0, kOffset),
                                    tla::MakeShape(mActual, curK));
        copyL1ToL0A(tensorL0A, tileL1A);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
        auto tileL1B = tla::GetTile(tensorL1B, tla::MakeCoord(kOffset, 0),
                                    tla::MakeShape(curK, n));
        copyL1ToL0B(tensorL0B, tileL1B);
        if (lastK && releaseL1AfterUse) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
        ctx.l0Slot ^= 1U;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
        if (firstK) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cFreeEvent);
        }
        const uint8_t mmadUnitFlag = lastK ? 0b11 : 0b10;
        tileMmad(tensorTileL0C, tensorL0A, tensorL0B, firstK, mmadUnitFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
        if (lastK) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cDoneEvent);
        }
    }
    ctx.l0cSlot ^= 1U;
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cDoneEvent);
    copyL0CToDst(tensorDst, tensorL0C, 0b11, ElementAccumulator{});
}

// ══ ④ 数据结构体：只放数据，不放函数 ═════════════════════════════════════════

template <typename DType, uint32_t NORM_MODE, uint32_t OUTPUT_MODE>
struct OpNameCubeContext {
    using DT = DType;
    using Prims = OpNameCubePrimitives<DT>;
    using ElementAccumulator = typename Prims::ElementAccumulator;
    static constexpr uint32_t kOutputMode = OUTPUT_MODE;
    static constexpr uint32_t kNormMode = NORM_MODE;

    Catlass::Arch::Resource<typename Prims::ArchTag> resource;
    AscendC::GlobalTensor<DT> normWorkspaceGm;
    AscendC::GlobalTensor<DT> stateWorkspaceGm;

    AscendC::LocalTensor<DT> normL1[L1_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> stateL1[L1_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> l0A[L0_BUFFER_COUNT_2];
    AscendC::LocalTensor<DT> l0B[L0_BUFFER_COUNT_2];
    AscendC::LocalTensor<ElementAccumulator> l0C[L0_BUFFER_COUNT_2];

    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;
    const OpNameTilingData *tiling = nullptr;
    int64_t coreIdx = 0;
    int64_t coreNum = 1;
    int64_t taskRound = 0;
    uint32_t streamSlot = 0;
    uint32_t l0Slot = 0;
    uint32_t l0cSlot = 0;
};

// ══ ⑤ 行为层：文件作用域 inline 函数 ═════════════════════════════════════════

template <typename Ctx>
__aicore__ inline void InitOpNameCube(
    Ctx &ctx, GM_ADDR x, GM_ADDR g, GM_ADDR normWorkspace, GM_ADDR stateWorkspace,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, const OpNameTilingData *tiling)
{
    ctx.normWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename Ctx::DT *>(normWorkspace));
    ctx.stateWorkspaceGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename Ctx::DT *>(stateWorkspace));
    ctx.cuSeqlens = cuSeqlens;
    ctx.chunkIndices = chunkIndices;
    ctx.tiling = tiling;
    (void)x;
    (void)g;

    ctx.coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    ctx.coreNum = static_cast<int64_t>(AscendC::GetBlockNum());
    if (ctx.coreNum <= 0) {
        ctx.coreNum = 1;
    }

    for (int64_t slot = 0; slot < L1_SLOT_COUNT_2; ++slot) {
        ctx.normL1[slot] = ctx.resource.l1Buf.template GetBufferByByte<typename Ctx::DT>(
            L1_NORM_OFFSET + slot * L1_TILE_BYTES);
        ctx.stateL1[slot] = ctx.resource.l1Buf.template GetBufferByByte<typename Ctx::DT>(
            L1_STATE_OFFSET + slot * L1_TILE_BYTES);
    }
    for (int64_t slot = 0; slot < L0_BUFFER_COUNT_2; ++slot) {
        ctx.l0A[slot] =
            ctx.resource.l0ABuf.template GetBufferByByte<typename Ctx::DT>(slot * L0A_BYTES);
        ctx.l0B[slot] =
            ctx.resource.l0BBuf.template GetBufferByByte<typename Ctx::DT>(slot * L0B_BYTES);
        ctx.l0C[slot] = ctx.resource.l0CBuf.template GetBufferByByte<typename Ctx::ElementAccumulator>(
            slot * L0C_BYTES);
    }
}

// S1a：等 AIV ready → 等 L1 槽可写 → GM→L1 → 发布 L1 可读（MTE2_MTE1）
template <typename Ctx>
__aicore__ inline void Stage1CopyInNorm(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    using Prims = typename Ctx::Prims;
    const int32_t l1Event = static_cast<int32_t>(slot);
    AscendC::CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
    auto normGm = tla::MakeTensor(
        ctx.normWorkspaceGm[GetWorkspaceChunkOffset(ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
        Prims::GM_NORM_LAYOUT, Catlass::Arch::PositionGM{});
    auto normTile =
        tla::GetTile(normGm, tla::MakeCoord(0, 0), tla::MakeShape(chunk.chunkLen, DIM_128));
    auto normL1 = tla::MakeTensor(ctx.normL1[slot], Prims::L1_NORM_LAYOUT,
                                  Catlass::Arch::PositionL1{});
    typename Prims::template CopyStateGmToL1A<decltype(normTile)>{}(normL1, normTile);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
}

// S1b：切 tile 后调用 CubeChunkStateGemm（K 分块 + MMAD + 搬出）
template <typename Ctx>
__aicore__ inline void Stage1Matmul(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    using Prims = typename Ctx::Prims;
    const int32_t l1Event = static_cast<int32_t>(slot);
    auto normL1 = tla::MakeTensor(ctx.normL1[slot], Prims::L1_NORM_LAYOUT,
                                  Catlass::Arch::PositionL1{});
    auto stateL1 = tla::MakeTensor(ctx.stateL1[slot], Prims::L1_STATE_LAYOUT,
                                   Catlass::Arch::PositionL1{});
    auto stateGm = tla::MakeTensor(
        ctx.stateWorkspaceGm[GetWorkspaceChunkOffset(ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
        Prims::GM_NORM_LAYOUT, Catlass::Arch::PositionGM{});
    auto stateTile =
        tla::GetTile(stateGm, tla::MakeCoord(0, 0), tla::MakeShape(chunk.chunkLen, DIM_128));
    typename Prims::CopyStateL1ToL0A copyL1ToL0A;
    typename Prims::CopyStateL1ToL0B copyL1ToL0B;
    typename Prims::TileMmadState tileMmad;
    typename Prims::CopyStateL0CToDst copyL0CToDst;
    CubeChunkStateGemm<Ctx>(ctx, copyL1ToL0A, copyL1ToL0B, tileMmad, copyL0CToDst, normL1, stateL1,
                            stateTile, static_cast<uint32_t>(chunk.chunkLen), DIM_128, DIM_128,
                            l1Event, /*waitL1Ready=*/true, /*releaseL1AfterUse=*/true);
}

// S1c：save 档额外搬出 → 广播给 AIV（PIPE_FIX）；L0C 已由计算层释放（M_FIX→FIX_M 闭环）
template <typename Ctx>
__aicore__ inline void Stage1FixpipeOut(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    (void)chunk;
    (void)slot;
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        // ... save 档额外搬出；none 档整体被裁掉 ...
    }
    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_TO_VEC_READY_FLAG);
}

// 收尾：与 arch35 完全一致的顺序
template <typename Ctx>
__aicore__ inline void CloseAndReleaseEvents(Ctx &ctx)
{
    for (int32_t slot = 0; slot < static_cast<int32_t>(L1_SLOT_COUNT_2); ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    }
    for (int32_t slot = 0; slot < static_cast<int32_t>(L0_BUFFER_COUNT_2); ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
    }
    (void)ctx;
}

// 任务主循环：事件预置 → 取 ChunkInfo → 按 Stage 顺序调函数 → 收尾
template <typename Ctx>
__aicore__ inline void ProcessOpNameCube(Ctx &ctx)
{
    for (int32_t slot = 0; slot < static_cast<int32_t>(L1_SLOT_COUNT_2); ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    }
    for (int32_t slot = 0; slot < static_cast<int32_t>(L0_BUFFER_COUNT_2); ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
    }

    ChunkInfo chunk;
    for (int64_t taskIdx = ctx.coreIdx; taskIdx < ctx.tiling->taskNum; taskIdx += ctx.coreNum) {
        GetChunkInfo(taskIdx, ctx.cuSeqlens, ctx.chunkIndices, *ctx.tiling, chunk);
        if (!chunk.valid) {
            continue;
        }
        ctx.taskRound = (taskIdx - ctx.coreIdx) / ctx.coreNum;
        const uint32_t slot = ctx.streamSlot;
        Stage1CopyInNorm(ctx, chunk, slot);
        Stage1Matmul(ctx, chunk, slot);
        Stage1FixpipeOut(ctx, chunk, slot);
        ctx.streamSlot ^= 1U;
    }
    CloseAndReleaseEvents(ctx);
}

} // namespace OpsClassify

#endif // OP_NAME_CUBE_ARCH22_H
