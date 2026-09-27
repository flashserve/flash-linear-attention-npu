/**
 * 示例文件：.../op_kernel/arch22/op_name_vec.h（A2/A3 的 AIV 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_vector.h
 *
 * 与 arch35 版本**同骨架、同命名**：结构体只放数据，函数全部写在结构体外；
 * 同名函数、同名事件数组、同名收尾函数。平台差异只允许出现在下表列出的位置。
 *
 * ── 平台差异表（与 arch35 逐条对照）──────────────────────────────────────────
 *   项               arch22（A2/A3）                          arch35（A5）
 *   向量计算          普通向量指令（Muls/Add/ReduceSum/Rsqrt），  __simd_vf__ + MicroAPI
 *                     入参是 LocalTensor，按 repeat 展开         （RegTensor/MaskReg），
 *                                                               入参是 __ubuf__ 裸指针，一条 VF 融合
 *   融合能力          无 VF 通路：中间量要落回 UB 再下一次读     融合：中间量留在寄存器不回 UB
 *   tile 行数         VEC_TILE_ELEMS = TILE_T_ARCH22 × DIM_128   TILE_T_ARCH35 更大
 *   UB 布局           同名常量，偏移数值不同（见 struct.h）        同名常量（见 struct.h）
 *   事件集合          每 slot 一组，同 arch35                    同 arch22
 *   同步协议          common.h 的具名 flag                       同 arch22（不允许各写一套）
 *
 * ── Stage 表 / UB 布局表 / 同步协议表 ────────────────────────────────────────
 *   Stage 顺序与 arch35 一致：S0 CopyInNorm → S1 Matrix(AIC) → S2 ScanWrite → S3 SaveExport。
 *   UB 布局常量在 arch22/op_name_struct.h（UB_X_OFFSET / UB_G_OFFSET / UB_NORM_OFFSET /
 *   UB_SCAN_OFFSET / UB_Y_OFFSET / UB_STATE_OFFSET / UB_XNORM_OFFSET），与 arch35 同名。
 */

#ifndef OP_NAME_VEC_ARCH22_H
#define OP_NAME_VEC_ARCH22_H

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

// ══ ② 计算层：普通向量指令实现，同样写在结构体外 ═══════════════════════════════
// A2/A3 没有 VF 通路，这里按 repeat 展开；尾块用 mask 收尾，padding 不参与计算。
// 函数的入参、顺序、语义与 arch35 版本保持一致，便于两平台逐行对照。

// S0：norm = rsqrt(sum_d(x^2) / D + epsilon)
template <typename DT>
__aicore__ inline void Stage0NormVf(
    const AscendC::LocalTensor<DT> &x, const AscendC::LocalTensor<float> &xWork,
    uint16_t validRows, AscendC::LocalTensor<float> &norm, AscendC::LocalTensor<float> &rowSum,
    float epsilon)
{
    // ... Cast(xWork, x) → Mul(xWork, xWork, xWork) → 按行 ReduceSum →
    //     Muls/Adds(1/D, epsilon) → Rsqrt(norm) → 行广播回整行；逐步依赖用 PipeBarrier<PIPE_V>() ...
    (void)x;
    (void)xWork;
    (void)validRows;
    (void)norm;
    (void)rowSum;
    (void)epsilon;
}

// S2：y = x * scan(g * scale) * norm
template <typename DT, typename GT>
__aicore__ inline void Stage2ScanVf(
    const AscendC::LocalTensor<DT> &x, const AscendC::LocalTensor<GT> &g,
    const AscendC::LocalTensor<float> &norm, AscendC::LocalTensor<DT> &y,
    AscendC::LocalTensor<float> &scan, uint16_t validRows, float scale)
{
    // ... Muls(scan, g, scale) → chunk 内前缀和 → Mul(y, x, scan) → Mul(y, y, norm) → Cast 回 DT ...
    (void)x;
    (void)g;
    (void)norm;
    (void)y;
    (void)scan;
    (void)validRows;
    (void)scale;
}

// S3：save 档整理 state 行；none 档调用点被 if constexpr 裁掉。
template <typename DT>
__aicore__ inline void Stage3ExportVf(
    const AscendC::LocalTensor<DT> &stateRow, AscendC::LocalTensor<DT> &stateOut,
    uint16_t validRows)
{
    // ... 需要重量化/转置时在这里做；纯搬出时直接在调用方 DataCopyPad ...
    (void)stateRow;
    (void)stateOut;
    (void)validRows;
}

// ══ ③ 数据层：只放数据，不放函数 ═════════════════════════════════════════════

template <typename DT, typename GT, uint32_t NORM_MODE, bool USE_STATE, uint32_t OUTPUT_MODE>
struct OpNameVectorContext {
    using DTypeX = DT;
    using DTypeG = GT;
    static constexpr bool kUseState = USE_STATE;
    static constexpr uint32_t kOutputMode = OUTPUT_MODE;

    // GM：输入 / 输出 / workspace
    AscendC::GlobalTensor<DT> xGm;
    AscendC::GlobalTensor<GT> gGm;
    AscendC::GlobalTensor<float> aLogGm;
    AscendC::GlobalTensor<DT> initialStateGm;
    AscendC::GlobalTensor<DT> yGm;
    AscendC::GlobalTensor<DT> stateGm;
    AscendC::GlobalTensor<float> xNormGm;
    AscendC::GlobalTensor<float> normWorkspaceGm;
    AscendC::GlobalTensor<DT> stateWorkspaceGm;

    // UB：按 arch22/op_name_struct.h 的布局常量切片
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
    AscendC::LocalTensor<DT> x[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<GT> g[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> norm[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> scan[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> y[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> state[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> xWork;      // 无 VF 通路时，cast/中间量必须落回 UB 复用
    AscendC::LocalTensor<float> xNormOut;

    int32_t mte2ToV[UB_SLOT_COUNT_2] = {0, 0};
    int32_t vToMte3[UB_SLOT_COUNT_2] = {0, 0};
    int32_t mte3ToMte2[UB_SLOT_COUNT_2] = {0, 0};

    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;
    const OpNameTilingData *tiling = nullptr;
    AscendC::TPipe *pipe = nullptr;
    int64_t coreIdx = 0;
    int64_t coreNum = 1;
    int64_t subBlockIdx = 0;
    int64_t taskRound = 0;
    uint32_t streamSlot = 0;
    const uint32_t exportSlot = 1;
};

// ══ ④ 行为层：文件作用域 inline 函数，第一参数是上面的数据 ═══════════════════

template <typename Ctx>
__aicore__ inline void InitOpNameVector(
    Ctx &ctx, GM_ADDR x, GM_ADDR g, GM_ADDR aLog, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR y, GM_ADDR state, GM_ADDR xNorm,
    GM_ADDR normWorkspace, GM_ADDR stateWorkspace,
    const OpNameTilingData *tiling, AscendC::TPipe *pipe)
{
    // ① GM 接线
    ctx.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename Ctx::DTypeX *>(x));
    ctx.gGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename Ctx::DTypeG *>(g));
    ctx.yGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename Ctx::DTypeX *>(y));
    ctx.normWorkspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normWorkspace));
    ctx.stateWorkspaceGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ typename Ctx::DTypeX *>(stateWorkspace));
    if constexpr (Ctx::kUseState) {
        ctx.stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ typename Ctx::DTypeX *>(state));
        ctx.initialStateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ typename Ctx::DTypeX *>(initialState));
    }
    if (aLog != nullptr) {
        ctx.aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
    }
    ctx.xNormGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xNorm));

    // ② 只读状态：核号 / 子核号 clamp，行为与 arch35 一致
    ctx.tiling = tiling;
    ctx.pipe = pipe;
    ctx.cuSeqlens = cuSeqlens;
    ctx.chunkIndices = chunkIndices;
    ctx.coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx()) / AIV_COUNT_2;
    ctx.coreNum = static_cast<int64_t>(AscendC::GetBlockNum());
    ctx.subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
    if (ctx.coreNum <= 0) {
        ctx.coreNum = 1;
    }
    if (ctx.subBlockIdx < 0 || ctx.subBlockIdx >= AIV_COUNT_2) {
        ctx.subBlockIdx = 0;
    }

    // ③ UB 划分：与 arch35 同一张表，数值按 arch22 常量；xWork 是本平台多出来的一块
    //    （没有 VF 时 cast/中间量要落 UB），因此单独说明它的生命周期。
    ctx.pipe->InitBuffer(ctx.ubBuf, UB_TOTAL_BYTES);
    ctx.x[0] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(VEC_TILE_ELEMS, UB_X_OFFSET);
    ctx.x[1] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(
        VEC_TILE_ELEMS, UB_X_OFFSET + VEC_X_BYTES);
    ctx.g[0] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeG>(
        CHUNK_SIZE_64, UB_G_OFFSET);
    ctx.g[1] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeG>(
        CHUNK_SIZE_64, UB_G_OFFSET + VEC_G_BYTES);
    ctx.norm[0] = ctx.ubBuf.template GetWithOffset<float>(CHUNK_SIZE_64, UB_NORM_OFFSET);
    ctx.norm[1] = ctx.ubBuf.template GetWithOffset<float>(
        CHUNK_SIZE_64, UB_NORM_OFFSET + VEC_F32_BYTES);
    ctx.scan[0] = ctx.ubBuf.template GetWithOffset<float>(CHUNK_SIZE_64, UB_SCAN_OFFSET);
    ctx.scan[1] = ctx.ubBuf.template GetWithOffset<float>(
        CHUNK_SIZE_64, UB_SCAN_OFFSET + VEC_F32_BYTES);
    ctx.y[0] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(VEC_TILE_ELEMS, UB_Y_OFFSET);
    ctx.y[1] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(
        VEC_TILE_ELEMS, UB_Y_OFFSET + VEC_X_BYTES);
    // xWork 与 scan 共段：S0 用它做 cast 中间量，S0 结束即释放，S2 才改成 scan。
    ctx.xWork = ctx.scan[0];
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        ctx.state[0] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(
            VEC_TILE_ELEMS, UB_STATE_OFFSET);
        ctx.state[1] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(
            VEC_TILE_ELEMS, UB_STATE_OFFSET + VEC_X_BYTES);
        ctx.xNormOut = ctx.ubBuf.template GetWithOffset<float>(CHUNK_SIZE_64, UB_XNORM_OFFSET);
    }

    // ④ 核内事件：每 slot 一组，首轮先开放 free 方向（与 arch35 一致）
    for (uint32_t slot = 0; slot < UB_SLOT_COUNT_2; ++slot) {
        ctx.mte2ToV[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::MTE2_V>();
        ctx.vToMte3[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::V_MTE3>();
        ctx.mte3ToMte2[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    }
}

// S0 输入：x、g（GM）；输出：normWorkspace（GM）
//    同步：MTE2→V→MTE3 核内事件 + CrossCoreSetFlag 通知 AIC
template <typename Ctx>
__aicore__ inline void Stage0CopyInNorm(Ctx &ctx, const ChunkInfo &chunk)
{
    const uint32_t slot = ctx.streamSlot;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    AscendC::DataCopy(ctx.x[slot], ctx.xGm[chunk.tokenStart * DIM_128], chunk.chunkLen * DIM_128);
    AscendC::DataCopy(ctx.g[slot], ctx.gGm[chunk.tokenStart], chunk.chunkLen);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
    Stage0NormVf<typename Ctx::DTypeX>(ctx.x[slot], ctx.xWork,
                                       static_cast<uint16_t>(chunk.chunkLen),
                                       ctx.norm[slot], ctx.scan[slot], ctx.tiling->epsilon);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::DataCopyPad(ctx.normWorkspaceGm[GetWorkspaceChunkOffset(
                             ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
                         ctx.norm[slot],
                         {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0});
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        AscendC::DataCopy(ctx.xNormOut, ctx.norm[slot], chunk.chunkLen);
    }
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(VEC_TO_CUBE_READY_FLAG);
}

// S2 输入：x、g（GM）、stateWorkspace（GM，AIC 生产）；输出：y（GM）
template <typename Ctx>
__aicore__ inline void Stage2ScanWrite(Ctx &ctx, const ChunkInfo &chunk)
{
    const uint32_t slot = ctx.streamSlot;
    AscendC::CrossCoreWaitFlag(CUBE_TO_VEC_READY_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    AscendC::DataCopy(ctx.x[slot], ctx.xGm[chunk.tokenStart * DIM_128], chunk.chunkLen * DIM_128);
    AscendC::DataCopy(ctx.g[slot], ctx.gGm[chunk.tokenStart], chunk.chunkLen);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
    Stage2ScanVf<typename Ctx::DTypeX, typename Ctx::DTypeG>(
        ctx.x[slot], ctx.g[slot], ctx.norm[slot], ctx.y[slot], ctx.scan[slot],
        static_cast<uint16_t>(chunk.chunkLen), ctx.tiling->scale);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::DataCopyPad(ctx.yGm[chunk.tokenStart * DIM_128], ctx.y[slot],
                         {static_cast<uint16_t>(chunk.chunkLen),
                          static_cast<uint32_t>(chunk.chunkLen * sizeof(typename Ctx::DTypeX)), 0,
                          static_cast<uint32_t>((DIM_128 - chunk.chunkLen) *
                                                sizeof(typename Ctx::DTypeX)), 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    ctx.streamSlot ^= 1U;
}

// S3 输入：stateWorkspace（GM，AIC 生产）；输出：state、x_norm（GM）
template <typename Ctx>
__aicore__ inline void Stage3SaveExport(Ctx &ctx, const ChunkInfo &chunk)
{
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        const uint32_t slot = ctx.exportSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
        AscendC::DataCopy(ctx.state[slot],
                          ctx.stateWorkspaceGm[GetWorkspaceChunkOffset(
                              ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
                          chunk.chunkLen * DIM_128);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
        Stage3ExportVf<typename Ctx::DTypeX>(ctx.state[slot], ctx.state[slot],
                                            static_cast<uint16_t>(chunk.chunkLen));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
        AscendC::DataCopyPad(ctx.stateGm[chunk.chunkIdx * DIM_128], ctx.state[slot],
                             {static_cast<uint16_t>(chunk.chunkLen),
                              static_cast<uint32_t>(chunk.chunkLen *
                                                    sizeof(typename Ctx::DTypeX)), 0,
                              static_cast<uint32_t>((DIM_128 - chunk.chunkLen) *
                                                    sizeof(typename Ctx::DTypeX)), 0});
        AscendC::DataCopyPad(ctx.xNormGm[chunk.tokenStart * DIM_128], ctx.xNormOut,
                             {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    }
}

// 收尾：与 arch35 顺序一致——先等回每个 slot，再 Release
template <typename Ctx>
__aicore__ inline void CloseAndReleaseEvents(Ctx &ctx)
{
    for (uint32_t slot = 0; slot < UB_SLOT_COUNT_2; ++slot) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
        ctx.pipe->template ReleaseEventID<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
        ctx.pipe->template ReleaseEventID<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
        ctx.pipe->template ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    }
}

// 任务主循环：取 ChunkInfo → 按 Stage 顺序调函数 → 收尾
template <typename Ctx>
__aicore__ inline void ProcessOpNameVector(Ctx &ctx)
{
    ChunkInfo chunk;
    for (int64_t taskIdx = ctx.coreIdx; taskIdx < ctx.tiling->taskNum; taskIdx += ctx.coreNum) {
        GetChunkInfo(taskIdx, ctx.cuSeqlens, ctx.chunkIndices, *ctx.tiling, chunk);
        if (!chunk.valid) {
            continue;
        }
        ctx.taskRound = (taskIdx - ctx.coreIdx) / ctx.coreNum;
        Stage0CopyInNorm(ctx, chunk);
        Stage2ScanWrite(ctx, chunk);
        Stage3SaveExport(ctx, chunk);   // none 档内部被 if constexpr 整体裁掉
    }
    CloseAndReleaseEvents(ctx);
}

} // namespace OpsClassify

#endif // OP_NAME_VEC_ARCH22_H
