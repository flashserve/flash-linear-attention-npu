/**
 * 示例文件：.../op_kernel/arch22/op_name_cube.h（A2/A3 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *
 * 与 arch35 版本同骨架：**结构体只放数据，函数全部写在结构体外**（文件作用域 inline）；
 * 同名函数、同名事件数组、同名收尾函数。AIC 不做 VF 融合，Cube 通路的差异只在下表。
 *
 * ── 平台差异表（与 arch35 逐条对照）──────────────────────────────────────────
 *   项             arch22（A2/A3）                    arch35（A5）
 *   L1/L0 偏移     同名常量（见 struct.h）            同名常量（见 struct.h）
 *   Fixpipe 配置   等价语义                            可用 FixpipeConfig 指定 NZ/L1 目标
 *   事件集合       同 arch35（MTE1_MTE2 / M_MTE1 / FIX_M）  同 arch22
 *   同步协议       common.h 具名 flag                  同 arch22
 *   融合点         L0 双缓冲 + Fixpipe 直写            同上，另可按 arch35 规格放大 tile
 *
 * ── Stage 表 / L1+L0 布局表 / 同步协议表 ─────────────────────────────────────
 *   与 arch35 一致：S1a CopyIn → S1b Matmul → S1c Fixpipe，Fixpipe 后广播给 AIV。
 */

#ifndef OP_NAME_CUBE_ARCH22_H
#define OP_NAME_CUBE_ARCH22_H

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

// ══ ② 数据层：只放数据，不放函数 ═════════════════════════════════════════════

template <typename DT, uint32_t NORM_MODE, uint32_t OUTPUT_MODE>
struct OpNameCubeContext {
    using DType = DT;
    static constexpr uint32_t kOutputMode = OUTPUT_MODE;
    static constexpr uint32_t kNormMode = NORM_MODE;

    AscendC::GlobalTensor<float> normWorkspaceGm;
    AscendC::GlobalTensor<DT> stateWorkspaceGm;
    AscendC::TPipe *pipe = nullptr;

    AscendC::TBuf<AscendC::TPosition::A1> l1Buf;
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf;
    AscendC::TBuf<AscendC::TPosition::B2> l0bBuf;
    AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf;
    AscendC::LocalTensor<DT> normL1;
    AscendC::LocalTensor<DT> stateL1;

    int32_t mte1ToMte2[L1_SLOT_COUNT_2] = {0, 0};
    int32_t cubeMte1[L1_SLOT_COUNT_2] = {0, 0};
    int32_t fixpipeToCube[L1_SLOT_COUNT_2] = {0, 0};

    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;
    const OpNameTilingData *tiling = nullptr;
    int64_t coreIdx = 0;
    int64_t coreNum = 1;
    int64_t taskRound = 0;
    uint32_t streamSlot = 0;
};

// ══ ③ 行为层：文件作用域 inline 函数 ═════════════════════════════════════════

template <typename Ctx>
__aicore__ inline void InitOpNameCube(
    Ctx &ctx, GM_ADDR x, GM_ADDR g, GM_ADDR normWorkspace, GM_ADDR stateWorkspace,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, const OpNameTilingData *tiling,
    AscendC::TPipe *pipe)
{
    // ① GM 接线：AIC 只读 workspace 与 tiling
    ctx.normWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normWorkspace));
    ctx.stateWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(stateWorkspace));
    ctx.pipe = pipe;
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

    // ③ L1/L0 资源划分：与 arch35 同一张布局表（常量同名）
    ctx.pipe->InitBuffer(ctx.l1Buf, L1_TOTAL_BYTES);
    ctx.normL1 = ctx.l1Buf.template GetWithOffset<typename Ctx::DType>(L1_TILE_ELEMS, L1_NORM_OFFSET);
    ctx.stateL1 = ctx.l1Buf.template GetWithOffset<typename Ctx::DType>(
        2 * L1_TILE_ELEMS, L1_STATE_OFFSET);
    ctx.pipe->InitBuffer(ctx.l0aBuf, L0A_BYTES);
    ctx.pipe->InitBuffer(ctx.l0bBuf, L0B_BYTES);
    ctx.pipe->InitBuffer(ctx.l0cBuf, L0C_BYTES);

    // ④ 核内事件：每 slot 一组，首轮先开放
    for (uint32_t slot = 0; slot < L1_SLOT_COUNT_2; ++slot) {
        ctx.mte1ToMte2[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
        ctx.cubeMte1[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::M_MTE1>();
        ctx.fixpipeToCube[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::FIX_M>();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.cubeMte1[slot]);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
    }
}

// S1a：MTE2 搬 norm 到 L1
template <typename Ctx>
__aicore__ inline void Stage1CopyIn(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
    AscendC::DataCopy(ctx.normL1,
                      ctx.normWorkspaceGm[GetWorkspaceChunkOffset(
                          ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
                      chunk.chunkLen * DIM_128);
}

// S1b：L1→L0 → Cube → L0C
template <typename Ctx>
__aicore__ inline void Stage1Matmul(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    (void)chunk;
    // ... 与 arch35 相同的三段（MTE1 读 L1 / Cube 计算 / L0 与 L0C 复用事件）...
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.cubeMte1[slot]);
}

// S1c：Fixpipe 落 stateWorkspace；save 档才多落公开 state
template <typename Ctx>
__aicore__ inline void Stage1FixpipeOut(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    (void)chunk;
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        // ... 与 arch35 相同的 stateWorkspace 写回语义 ...
    }
}

// 收尾：与 arch35 完全一致的顺序
template <typename Ctx>
__aicore__ inline void CloseAndReleaseEvents(Ctx &ctx)
{
    for (uint32_t slot = 0; slot < L1_SLOT_COUNT_2; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ctx.cubeMte1[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
        ctx.pipe->template ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
        ctx.pipe->template ReleaseEventID<AscendC::HardEvent::M_MTE1>(ctx.cubeMte1[slot]);
        ctx.pipe->template ReleaseEventID<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
    }
}

// 任务主循环
template <typename Ctx>
__aicore__ inline void ProcessOpNameCube(Ctx &ctx)
{
    ChunkInfo chunk;
    for (int64_t taskIdx = ctx.coreIdx; taskIdx < ctx.tiling->taskNum; taskIdx += ctx.coreNum) {
        GetChunkInfo(taskIdx, ctx.cuSeqlens, ctx.chunkIndices, *ctx.tiling, chunk);
        if (!chunk.valid) {
            continue;
        }
        ctx.taskRound = (taskIdx - ctx.coreIdx) / ctx.coreNum;
        const uint32_t slot = ctx.streamSlot;
        AscendC::CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG);
        Stage1CopyIn(ctx, chunk, slot);
        Stage1Matmul(ctx, chunk, slot);
        Stage1FixpipeOut(ctx, chunk, slot);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_TO_VEC_READY_FLAG);
        ctx.streamSlot ^= 1U;
    }
    CloseAndReleaseEvents(ctx);
}

} // namespace OpsClassify

#endif // OP_NAME_CUBE_ARCH22_H
