/**
 * 示例文件：.../op_kernel/arch35/op_name_cube.h（A5 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *          （计算层的 K 分块累加步骤另参考 chunk_gated_delta_rule_bwd_dhu_cube.h 的 RunResidentMmad）
 *
 * 与 vec 版本同骨架（结构体只放数据、函数全写在结构体外），AIC 的计算原语不同：
 *   vec  ：`__simd_vf__` + MicroAPI，融合点是"一条 VF 完成读入→计算→写回"；
 *   cube ：Catlass `TileCopy` + `TileMmad` + Fixpipe，融合点是 **CubeChunkStateGemm**：
 *          一次调用内完成 "K 方向分块搬 L1→L0 + MMAD 累加（unit flag 表达首/末块）+ L0C 搬出"，
 *          调用方只提供已切好的 L1/L0 槽、copy/mad 对象与 m/n/k。
 * Cube 不需要入口传 TPipe：`Catlass::Arch::Resource<ArchTag>` 自带 L1/L0 buffer。
 *
 * 文件顺序（复制到新算子时保持）：
 *   ① 本注释块：Stage 表 / L1+L0 布局表 / 同步协议表
 *   ② 类型与 layout 层：`OpNameCubePrimitives<DT>`（TileCopy/TileMmad 组合、layout、Fixpipe 配置、空间断言）
 *   ③ Cube 计算层：CubeChunkStateGemm(...)（K 分块 + MMAD + 搬出，一次调用）
 *   ④ 数据结构体：OpNameCubeContext（GlobalTensor / Catlass Resource / L1 常驻槽 / L0 ping-pong / 只读状态）
 *   ⑤ 行为层：InitOpNameCube → ProcessOpNameCube（含事件预置）→ Stage1CopyInNorm →
 *             Stage1Matmul → Stage1FixpipeOut → CloseAndReleaseEvents
 *
 * ── Stage 表 ─────────────────────────────────────────────────────────────────
 *   S1a CopyIn    AIC  CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG) → TileCopy GM(norm)→L1
 *                      → SetFlag(MTE2_MTE1) 告诉计算层 L1 可读
 *   S1b Matmul    AIC  CubeChunkStateGemm：K 分块 CopyL1ToL0A/B → TileMmad → L0C
 *                      写入前 WaitFlag(M_MTE1) 保护 L0、首块前 WaitFlag(FIX_M) 保护 L0C
 *   S1c Fixpipe   AIC  Fixpipe 把 L0C 写 stateWorkspace → SetFlag(FIX_M) 释放 L0C
 *                      → CrossCoreSetFlag(CUBE_TO_VEC_READY_FLAG) 通知 AIV 做 S2
 *
 * ── L1 / L0 布局表（常量在 arch35/op_name_struct.h，逐行对应）────────────────
 *   偏移常量          大小                  内容              生命周期
 *   L1_NORM_OFFSET    L1_TILE_BYTES         AIV 产出的 norm    S1a GM→L1 → S1b L1→L0A；末块后 MTE1_MTE2 释放
 *   L1_STATE_OFFSET   2 * L1_TILE_BYTES      state 分块常驻      S1b 读 → S1c Fixpipe 写回
 *   L0A / L0B         L0A_BYTES/L0B_BYTES   左右矩阵           CopyL1ToL0A/B 写 → TileMmad 读（ping/pong）
 *   L0C               L0C_BYTES             累加结果           TileMmad 写 → CopyL0CToDst 读（ping/pong）
 *   L1 只由 AIC 写入：AIV 结果先落 GM，AIC 在对应 Stage 内搬入，算子中没有 UB→L1 通路。
 *
 * ── 同步协议（事件 id 与物理槽一一对应，不用 AllocEventID，样板同款固定表）─────
 *   MTE1_MTE2：id = L1 槽号，L1 槽"可写"（首轮在 Process 预置，MTE1 读完末块后重新发布）；
 *   MTE2_MTE1：id = L1 槽号，L1 槽"可读"（S1a 搬完后发布，计算层首块前等待）；
 *   M_MTE1   ：id = 2*L0 槽号 / 2*L0 槽号+1（A/B 分开），L0 槽"可写"（Cube 读完重新发布）；
 *   MTE1_M   ：id = 2*L0 槽号 / 2*L0 槽号+1，L0 槽"可读"；
 *   FIX_M / M_FIX：id = L0C 槽号，"L0C 可写 / L0C 已写完"，首位与末位各发布一次。
 *   核间：等 AIV 的 ready（VEC_TO_CUBE_READY_FLAG），Fixpipe 完成后再广播（PIPE_FIX）。
 */

#ifndef OP_NAME_CUBE_ARCH35_H
#define OP_NAME_CUBE_ARCH35_H

// Catlass 按架构选择编译分支：A5 = 3510，A2/A3 = 2201（样板算子同款）。
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
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

// ══ ② 类型与 layout 层：只依赖 dtype，用 traits 结构体参数化 ═══════════════════
// 这些别名/layout 随 DT 变化，所以用 traits 承载（无函数）；Stage 与计算层只引用它，
// 不重复写 PackedTileCopyTla 的模板参数表；换 operand 布局时只改这里一处。

template <typename DT>
struct OpNameCubePrimitives {
    using ArchTag = Catlass::Arch::Ascend950;
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

    // Fixpipe 配置：L0C 按 NZ 搬出（样板 FIXPIPE_NZ_L1_CONFIG 同款）。
    static constexpr AscendC::FixpipeConfig FIXPIPE_NZ_CONFIG = {AscendC::CO2Layout::NZ, false};

    // GM 与 L1/L0 的形状+布局一次写清，Stage 里只 MakeTensor/GetTile，不再重复推导。
    static constexpr auto GM_NORM_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<DIM_128>{});
    static constexpr auto L1_NORM_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyState::LayoutTagL1A>(
            tla::Int<CHUNK_SIZE_64>{}, tla::Int<DIM_128>{});
    static constexpr auto L1_STATE_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyState::LayoutTagL1B>(
            tla::Int<DIM_128>{}, tla::Int<CHUNK_SIZE_64>{});
};

// L0 tile 形状：Cube 上 K 方向按这个粒度分块累加；改它要同步 L0A/L0B 容量断言。
constexpr uint32_t CUBE_K_TILE = 64;

// 空间上限用 static_assert 钉住：偏移算错时编译期报错，而不是运行时踩 buffer。
static_assert(L1_STATE_OFFSET + L1_SLOT_COUNT_2 * L1_TILE_BYTES <= L1_TOTAL_BYTES,
              "L1 state 常驻槽必须落在 L1_TOTAL_BYTES 内。");
static_assert(L0A_BYTES * L0_BUFFER_COUNT_2 <= 64 * 1024, "L0A ping/pong 超出 L0A 容量。");
static_assert(L0B_BYTES * L0_BUFFER_COUNT_2 <= 64 * 1024, "L0B ping/pong 超出 L0B 容量。");
static_assert(L0C_BYTES * L0_BUFFER_COUNT_2 <= 256 * 1024, "L0C ping/pong 超出 L0C 容量。");

// ══ ③ Cube 计算层：一次调用完成"K 分块搬入 + MMAD 累加 + 搬出" ════════════════
// 对照样板 dhu 的 RunResidentMmad：K 分块循环、unit flag 表达"首块不清零 / 末块收尾"、
// L0 与 L0C 的 ping/pong 取反、L1 槽的"可读等待/用后释放"都在这里一次做完；
// 调用方只负责切好 L1/L0 槽与 m/n/k，不重复写事件对。

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
    // M=1 的尾块在 Cube 上按 16 行最小单元计算，与样板一致：只影响内部 tile 形状，
    // 不影响对外 chunkLen 语义。
    const uint32_t mActual = (m == 1U) ? 16U : m;
    const uint32_t l0cSlot = ctx.l0cSlot;
    const int32_t l0cFreeEvent = static_cast<int32_t>(l0cSlot);          // FIX_M：可写
    const int32_t l0cDoneEvent = static_cast<int32_t>(l0cSlot);          // M_FIX：已写完

    auto tensorL0C = tla::MakeTensor(
        ctx.l0C[l0cSlot], tla::MakeLayoutL0C(mActual, n), Catlass::Arch::PositionL0C{});
    auto tensorTileL0C =
        tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, n));

    for (uint32_t kOffset = 0; kOffset < k; kOffset += CUBE_K_TILE) {
        const uint32_t curK = (kOffset + CUBE_K_TILE > k) ? (k - kOffset) : CUBE_K_TILE;
        const bool firstK = (kOffset == 0U);
        const bool lastK = (kOffset + curK >= k);
        const uint32_t l0Slot = ctx.l0Slot;
        const int32_t l0AEvent = static_cast<int32_t>(2U * l0Slot);      // M_MTE1：A 可写
        const int32_t l0BEvent = static_cast<int32_t>(2U * l0Slot + 1U);// M_MTE1：B 可写
        const int32_t l0ReadyEvent = l0AEvent;                          // MTE1_M：A/B 同步写

        auto tensorL0A = tla::MakeTensor(
            ctx.l0A[l0Slot], tla::MakeLayout<DT, LayoutTagL0A>(mActual, curK),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            ctx.l0B[l0Slot], tla::MakeLayout<DT, LayoutTagL0B>(curK, n),
            Catlass::Arch::PositionL0B{});

        if (waitL1Ready) {
            // L1 由 S1a 的 MTE2 生产：等"可读"事件后才允许 MTE1 读。
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
            // 末块读完后 L1 槽才可复用，避免提前释放被别的 HEAD 覆盖。
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
        ctx.l0Slot ^= 1U;   // L0A/L0B 使用一次取反一次

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ReadyEvent);
        if (firstK) {
            // 首块写 L0C 前等上一轮 Fixpipe 读完（L0C ping/pong）。
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cFreeEvent);
        }
        // unit flag：首块是覆盖，后续块是累加；末块置终止位（0b11），其余 0b10。
        const uint8_t mmadUnitFlag = lastK ? 0b11 : 0b10;
        tileMmad(tensorTileL0C, tensorL0A, tensorL0B, firstK, mmadUnitFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);
        if (lastK) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cDoneEvent);
        }
    }
    ctx.l0cSlot ^= 1U;      // L0C 使用一次取反一次
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cDoneEvent);
    copyL0CToDst(tensorDst, tensorL0C, 0b11, ElementAccumulator{});
}

// ══ ④ 数据结构体：只放数据（含类型别名），不放函数 ═══════════════════════════

template <typename DType, uint32_t NORM_MODE, uint32_t OUTPUT_MODE>
struct OpNameCubeContext {
    using DT = DType;
    using Prims = OpNameCubePrimitives<DT>;
    using ElementAccumulator = typename Prims::ElementAccumulator;
    static constexpr uint32_t kOutputMode = OUTPUT_MODE;
    static constexpr uint32_t kNormMode = NORM_MODE;

    // Catlass Resource 自带 L1/L0 buffer，AIC 不需要入口传 TPipe。
    Catlass::Arch::Resource<typename Prims::ArchTag> resource;

    AscendC::GlobalTensor<DT> normWorkspaceGm;
    AscendC::GlobalTensor<DT> stateWorkspaceGm;

    // L1 常驻槽：按 slot 切片，哪一段给谁读写在文件头布局表里。
    AscendC::LocalTensor<DT> normL1[L1_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> stateL1[L1_SLOT_COUNT_2];
    // L0 ping/pong：A/B 与 L0C 各自独立取反，事件 id 跟着槽号走。
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
    // ① GM 接线：AIC 只读 workspace 与 tiling（x/g 形参保留是为了与 AIV 的 Init 对称）。
    ctx.normWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename Ctx::DT *>(normWorkspace));
    ctx.stateWorkspaceGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename Ctx::DT *>(stateWorkspace));
    ctx.cuSeqlens = cuSeqlens;
    ctx.chunkIndices = chunkIndices;
    ctx.tiling = tiling;
    (void)x;
    (void)g;

    // ② 只读状态
    ctx.coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    ctx.coreNum = static_cast<int64_t>(AscendC::GetBlockNum());
    if (ctx.coreNum <= 0) {
        ctx.coreNum = 1;
    }

    // ③ L1/L0 槽切片：与文件头布局表逐行对应；L1 只由 AIC 写，没有 UB→L1 通路。
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

// S1a 输入：normWorkspace（GM，AIV 生产）；输出：normL1 常驻槽
//     同步：等 AIV 的 ready；等 L1 槽可写（MTE1_MTE2），搬完发布可读（MTE2_MTE1）
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

// S1b 输入：normL1、stateL1 常驻槽；输出：L0C → stateWorkspace
//     实现：调用 CubeChunkStateGemm（K 分块 + MMAD + 搬出）；本函数只切 tile 与传事件参数
template <typename Ctx>
__aicore__ inline void Stage1Matmul(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    using Prims = typename Ctx::Prims;
    const int32_t l1Event = static_cast<int32_t>(slot);
    auto normL1 = tla::MakeTensor(ctx.normL1[slot], Prims::L1_NORM_LAYOUT,
                                  Catlass::Arch::PositionL1{});
    auto stateL1 = tla::MakeTensor(ctx.stateL1[slot], Prims::L1_STATE_LAYOUT,
                                   Catlass::Arch::PositionL1{});
    // 结果直接落到 stateWorkspace（save 档再由 AIV 的 Stage3 导出公开 state）。
    auto stateGm = tla::MakeTensor(
        ctx.stateWorkspaceGm[GetWorkspaceChunkOffset(ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
        Prims::GM_NORM_LAYOUT, Catlass::Arch::PositionGM{});
    auto stateTile =
        tla::GetTile(stateGm, tla::MakeCoord(0, 0), tla::MakeShape(chunk.chunkLen, DIM_128));
    // 每次 GEMM 用各自的 copy/mad 对象（样板同款：对象无状态，可局部构造）。
    typename Prims::CopyStateL1ToL0A copyL1ToL0A;
    typename Prims::CopyStateL1ToL0B copyL1ToL0B;
    typename Prims::TileMmadState tileMmad;
    typename Prims::CopyStateL0CToDst copyL0CToDst;
    CubeChunkStateGemm<Ctx>(ctx, copyL1ToL0A, copyL1ToL0B, tileMmad, copyL0CToDst, normL1, stateL1,
                            stateTile, static_cast<uint32_t>(chunk.chunkLen), DIM_128, DIM_128,
                            l1Event, /*waitL1Ready=*/true, /*releaseL1AfterUse=*/true);
}

// S1c 输入：stateWorkspace（已由 Fixpipe 写入）；输出：save 档公开 state
//     同步：释放 L0C（FIX_M）→ 广播给 AIV（PIPE_FIX）
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

// 收尾：等回在途事件（与 Process 开头的"初始可写"发布一一对应）。
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
    // 事件预置：首轮不存在旧消费者，先发布"可写"；L1 的可读事件由 Stage1a 自己发布。
    for (int32_t slot = 0; slot < static_cast<int32_t>(L1_SLOT_COUNT_2); ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    }
    for (int32_t slot = 0; slot < static_cast<int32_t>(L0_BUFFER_COUNT_2); ++slot) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);       // L0A ping/pong
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);   // L0B ping/pong
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);            // L0C ping/pong
    }

    ChunkInfo chunk;
    for (int64_t taskIdx = ctx.coreIdx; taskIdx < ctx.tiling->taskNum; taskIdx += ctx.coreNum) {
        GetChunkInfo(taskIdx, ctx.cuSeqlens, ctx.chunkIndices, *ctx.tiling, chunk);
        // host 侧已保证 taskIdx 范围与 metadata 合法性；这里是防御性跳过（不可达），
        // 不是拦截：不返回错误码、不打错误日志。
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

#endif // OP_NAME_CUBE_ARCH35_H
