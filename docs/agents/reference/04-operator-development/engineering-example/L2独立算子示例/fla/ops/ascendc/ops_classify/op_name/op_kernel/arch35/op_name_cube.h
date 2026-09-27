/**
 * 示例文件：.../op_kernel/arch35/op_name_cube.h（A5 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *
 * 与 vec 版本同骨架：**结构体只放数据，函数全部写在结构体外**（文件作用域 inline）。
 * AIC 不做 VF 融合：Cube 通路是 MTE2 → MTE1 → Cube → Fixpipe，融合点体现在
 * "L0 双缓冲 + Fixpipe 直写 L1/GM"，而不是 RegTensor 级别的指令融合。
 *
 * 文件顺序：
 *   ① 本注释块：Stage 表 / L1+L0 布局表 / 同步协议表
 *   ② 数据层：OpNameCubeContext（GM / L1 / L0 / 事件 / 只读状态，无成员函数）
 *   ③ 行为层：InitOpNameCube → Stage1CopyIn → Stage1Matmul → Stage1FixpipeOut
 *             → CloseAndReleaseEvents → ProcessOpNameCube
 *
 * ── Stage 表 ─────────────────────────────────────────────────────────────────
 *   S1a CopyIn    AIC  CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG) → MTE2 搬 norm 到 L1
 *   S1b Matmul    AIC  MTE1 把 L1 数据搬 L0A/L0B → Cube 计算 → L0C
 *   S1c Fixpipe   AIC  Fixpipe 把 L0C 写 stateWorkspace
 *                      → CrossCoreSetFlag(CUBE_TO_VEC_READY_FLAG) 通知 AIV 做 S2
 *
 * ── L1 / L0 布局表（常量在 arch35/op_name_struct.h，逐行对应）────────────────
 *   偏移常量          大小                内容              生命周期
 *   L1_NORM_OFFSET    L1_TILE_BYTES       AIV 产出的 norm    S1a 搬入 → S1b MTE1 读
 *   L1_STATE_OFFSET   2 * L1_TILE_BYTES   矩阵分块常驻        S1b 读 → S1c Fixpipe 写回
 *   L0A / L0B         L0A_BYTES / L0B_BYTES  左右矩阵        MTE1 写 → Cube 读（按 slot 双缓冲）
 *   L0C               L0C_BYTES           累加结果           Cube 写 → Fixpipe 读
 *
 * ── 同步协议 ─────────────────────────────────────────────────────────────────
 *   核间：等 AIV 的 ready，Fixpipe 完成后再广播（PIPE_FIX），保证 AIV 看到的是已落 L1/GM 的数据；
 *   核内：三类事件（MTE1_MTE2 L1 复用、FIX_M L0C 复用、M_MTE1 L0 复用）按 slot 成对，
 *         Init 预置首轮，Process 末尾闭环后 Release。
 */

#ifndef OP_NAME_CUBE_ARCH35_H
#define OP_NAME_CUBE_ARCH35_H

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

    int32_t mte1ToMte2[L1_SLOT_COUNT_2] = {0, 0};   // L1 复用：MTE1 读完 → 放行下一轮 MTE2
    int32_t cubeMte1[L1_SLOT_COUNT_2] = {0, 0};     // L0 复用：Cube 读完 → 放行下一轮 MTE1
    int32_t fixpipeToCube[L1_SLOT_COUNT_2] = {0, 0};// L0C 复用：Fixpipe 读完 → 放行下一轮 Cube

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
    // ① GM 接线：AIC 只读 workspace 与 tiling，不直接读用户输入（x/g 形参保留是为了
    //    与 AIV 的 Init 签名对称，便于入口按角色传同一组地址）。
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

    // ③ L1/L0 资源划分：与文件头布局表逐行对应
    ctx.pipe->InitBuffer(ctx.l1Buf, L1_TOTAL_BYTES);
    ctx.normL1 = ctx.l1Buf.template GetWithOffset<typename Ctx::DType>(L1_TILE_ELEMS, L1_NORM_OFFSET);
    ctx.stateL1 = ctx.l1Buf.template GetWithOffset<typename Ctx::DType>(
        2 * L1_TILE_ELEMS, L1_STATE_OFFSET);
    ctx.pipe->InitBuffer(ctx.l0aBuf, L0A_BYTES);
    ctx.pipe->InitBuffer(ctx.l0bBuf, L0B_BYTES);
    ctx.pipe->InitBuffer(ctx.l0cBuf, L0C_BYTES);

    // ④ 核内事件：每 slot 一组，首轮先开放（首轮没有上一轮消费者）
    for (uint32_t slot = 0; slot < L1_SLOT_COUNT_2; ++slot) {
        ctx.mte1ToMte2[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
        ctx.cubeMte1[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::M_MTE1>();
        ctx.fixpipeToCube[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::FIX_M>();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.cubeMte1[slot]);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
    }
}

// S1a：MTE2 把 AIV 产出的 norm 搬进 L1。地址只在本函数出现，Stage 内不重算偏移。
template <typename Ctx>
__aicore__ inline void Stage1CopyIn(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
    AscendC::DataCopy(ctx.normL1,
                      ctx.normWorkspaceGm[GetWorkspaceChunkOffset(
                          ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
                      chunk.chunkLen * DIM_128);
}

// S1b：MTE1 搬 L1→L0，Cube 计算写 L0C。只用样板同名的三类事件。
template <typename Ctx>
__aicore__ inline void Stage1Matmul(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    (void)chunk;
    // ... MTE1：L1 -> L0A/L0B（按 slot 双缓冲）...
    // MTE1 读完 L1 立即释放给下一轮 MTE2，这是本 Stage 唯一的 free 方向。
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ctx.mte1ToMte2[slot]);
    // 复用 L0C 前先等上一轮 Fixpipe 读完。
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
    // ... Cube：L0A x L0B -> L0C；累加顺序按 chunk 顺序，不得并行重排 ...
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ctx.cubeMte1[slot]);
}

// S1c：Fixpipe 把 L0C 落到 stateWorkspace；save 档才多落公开 state，none 档整体裁掉。
template <typename Ctx>
__aicore__ inline void Stage1FixpipeOut(Ctx &ctx, const ChunkInfo &chunk, uint32_t slot)
{
    (void)chunk;
    // ... Fixpipe：L0C -> L1/GM，按 arch35 的 FixpipeConfig ...
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ctx.fixpipeToCube[slot]);
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        // ... 额外落 stateWorkspace，供 AIV 的 Stage3 导出 ...
    }
}

// 收尾：等回每个 slot 的在途事件再 Release，顺序与 Init 的 SetFlag 一一对应。
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

// 任务主循环：AIC 与 AIV 用同一份 GetChunkInfo，保证两角色算出的 tokenStart/chunkLen 一致。
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
        // 本次广播既是 S2 的生产信号，也是下一轮 AIV ready 的背压来源。
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_TO_VEC_READY_FLAG);
        ctx.streamSlot ^= 1U;
    }
    CloseAndReleaseEvents(ctx);
}

} // namespace OpsClassify

#endif // OP_NAME_CUBE_ARCH35_H
