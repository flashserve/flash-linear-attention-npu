/**
 * 示例文件：.../op_kernel/arch35/op_name_vec.h（A5 的 AIV 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_vector.h
 *
 * 本文件按样板算子的写法组织，两条硬约束：
 *   **① 函数不放进类里**：`struct` 只放数据，所有行为都写成文件作用域的 `inline` 函数，
 *      第一参数是数据本身（数据与行为分离，读代码时先看数据形状，再看函数对它的操作）。
 *   **② arch35 的向量计算必须是 VF 融合函数**：计算层用 `__simd_vf__ inline` +
 *      `AscendC::MicroAPI`（`RegTensor` / `MaskReg` / `LoadAlign` / `StoreAlign` / ...），
 *      一条 VF 内完成"读入 → 计算 → 写回"，不做逐 repeat 的标量拼装；
 *      arch22 没有 VF 通路，用普通向量指令实现同名函数（见 arch22 版本的同名文件）。
 *
 * 文件顺序（复制到新算子时保持）：
 *   ① 本注释块：Stage 表 / UB 布局表 / 同步协议表
 *   ② VF 计算层：`StageNVf(...)`，`__simd_vf__ inline`，只碰 UB，不搬 GM、不发同步事件
 *   ③ 数据层：`OpNameVectorContext`（GM 张量 / UB 张量 / 事件 id / 只读状态，无成员函数）
 *   ④ 行为层：`InitOpNameVector` / `ProcessOpNameVector` / `StageN...` / `CloseAndReleaseEvents`
 *
 * ── Stage 表（谁生产、谁消费、怎么同步）────────────────────────────────────────
 *   S0 CopyInNorm  AIV  MTE2 搬 x/g → VF 融合求 norm → MTE3 写 normWorkspace
 *                      → CrossCoreSetFlag(VEC_TO_CUBE_READY_FLAG) 通知 AIC
 *   S1 Matrix      AIC  CrossCoreWaitFlag → MTE2/MTE1 → Cube → Fixpipe 写 stateWorkspace
 *                      → CrossCoreSetFlag(CUBE_TO_VEC_READY_FLAG) 通知 AIV
 *   S2 ScanWrite   AIV  CrossCoreWaitFlag → MTE2 搬 x/g/stateWorkspace → VF 融合扫描 → MTE3 写 y
 *   S3 SaveExport  AIV  save 档：MTE2 搬 stateWorkspace → VF 整理 → MTE3 写 state/x_norm
 *
 * ── UB 布局表（常量在 arch35/op_name_struct.h，逐行对应）──────────────────────
 *   偏移常量          大小              内容                生命周期
 *   UB_X_OFFSET       2 * VEC_X_BYTES   x slot0 / slot1     S0 MTE2 写 → S2 VF 读；S2 末尾释放
 *   UB_G_OFFSET       2 * VEC_G_BYTES   g slot0 / slot1     S0 MTE2 写 → S2 VF 读
 *   UB_NORM_OFFSET    2 * VEC_F32_BYTES norm slot0 / slot1  S0 VF 写 → S0 MTE3 读 → S2 复用为 scan
 *   UB_SCAN_OFFSET    2 * VEC_F32_BYTES scan slot0 / slot1  S2 VF 写 → S2 VF 读（carry 跨 chunk）
 *   UB_Y_OFFSET       2 * VEC_X_BYTES   y slot0 / slot1     S2 VF 写 → S2 MTE3 读
 *   UB_STATE_OFFSET   2 * VEC_X_BYTES   state slot0/slot1   save 档：AIC 写 stateWorkspace → 本核搬入再搬出
 *   UB_XNORM_OFFSET   VEC_F32_BYTES     x_norm 暂存         save 档：S0 留 norm → S3 搬出
 *
 * ── 同步协议（flag 定义在 op_kernel/op_name_common.h）─────────────────────────
 *   核间：VEC_TO_CUBE_READY_FLAG / CUBE_TO_VEC_READY_FLAG 双向握手，互为背压；
 *         不依赖核启动顺序，AIC/AIV 按同一 HEAD 顺序消费。
 *   核内：每 slot 一组 EventID（MTE2→V 入、V→MTE3 出、MTE3→MTE2 复用）；
 *         `InitOpNameVector` 末尾预置首轮，`CloseAndReleaseEvents` 闭环后 Release。
 *
 * 说明：本文件是结构骨架，VF 函数体只给代表性指令；实际指令、模板参数、mask 语义
 * 必须按目标 CANN 版本的 Ascend C API 文档与头文件核对后再落地。
 */

#ifndef OP_NAME_VEC_ARCH35_H
#define OP_NAME_VEC_ARCH35_H

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

// VF 融合计算用 MicroAPI 名字空间；只在本文件内 using，不影响平台无关头。
using namespace AscendC::MicroAPI;

// ══ ② VF 计算层：__simd_vf__ inline，函数写在结构体外，只碰 UB ════════════════
// 约定：入参用 `__ubuf__` 裸指针 + 有效行数，UB 地址由调用方按布局常量算好；
//       每个函数一条 VF 完成一段 Stage，不在函数里搬 GM、不发同步事件。
// Cast 的舍入/打包策略集中成 CastTrait 常量（样板算子同款写法），不在函数里散落参数；
// 具体取值必须按目标 CANN 版本头文件核对。
constexpr CastTrait BF16_TO_FP32_NORM = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_NONE,
};
constexpr CastTrait FP32_TO_BF16_NORM = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT,
};

// S0：norm[i] = rsqrt(sum_d(x[i, d]^2) / D + epsilon)
// 融合点：一行 x 只读一次（LoadAlign），cast/平方/ReduceSum/加 eps/开方/写出都在同一 VF 内完成，
//         避免"读→写回 UB→再读"的多次往返。
template <typename DT>
__simd_vf__ inline void Stage0NormVf(
    __ubuf__ const DT *x, uint16_t validRows,
    __ubuf__ float *norm, __ubuf__ float *rowSum, float epsilon)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);
    RegTensor<DT> xReg;
    RegTensor<float> xFp32Reg;
    RegTensor<float> squareReg;
    RegTensor<float> sumReg;
    RegTensor<float> normReg;

    for (uint16_t row = 0; row < validRows; ++row) {
        LoadAlign(xReg, x + row * DIM_128);
        Cast(xFp32Reg, xReg, BF16_TO_FP32_NORM, validMask);
        Mul(squareReg, xFp32Reg, xFp32Reg, validMask);
        ReduceSum(sumReg, squareReg, validMask);
        Muls(sumReg, sumReg, 1.0f / static_cast<float>(DIM_128), floatMask);
        Adds(sumReg, sumReg, epsilon, floatMask);
        Rsqrt(normReg, sumReg, floatMask);
        // 同一行结果既作为行标量和（给 S2/导出用），也广播回整行参与后续乘。
        StoreAlign(rowSum + row, normReg, floatMask);
        StoreAlign(norm + row * DIM_128, normReg, floatMask);
    }
    (void)validCount;
}

// S2：y = x * scan(g * scale) * norm；scan 在 chunk 内累加，跨 chunk 用 carry 续算。
// 融合点：一行 g 的缩放与累加、以及 x → y 的两次乘法在同一 VF 内串起来，
//         中间结果留在 RegTensor，不回 UB；carry 是 chunk 级状态，由外层用 UB 里的末值维护。
template <typename DT, typename GT>
__simd_vf__ inline void Stage2ScanVf(
    __ubuf__ const DT *x, __ubuf__ const GT *g, __ubuf__ const float *norm,
    __ubuf__ DT *y, __ubuf__ float *scan, uint16_t validRows, float scale)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);
    RegTensor<GT> gReg;
    RegTensor<float> gScaledReg;
    RegTensor<float> scanReg;
    RegTensor<DT> xReg;
    RegTensor<float> xFp32Reg;
    RegTensor<float> normReg;
    RegTensor<float> yReg;
    RegTensor<DT> yOutReg;

    Duplicate(scanReg, 0.0f, floatMask);
    for (uint16_t row = 0; row < validRows; ++row) {
        LoadAlign(gReg, g + row);
        Cast(gScaledReg, gReg, BF16_TO_FP32_NORM, validMask);
        Muls(gScaledReg, gScaledReg, scale, validMask);
        Add(scanReg, scanReg, gScaledReg, validMask);          // chunk 内前缀和
        StoreAlign(scan + row, scanReg, floatMask);
        LoadAlign(xReg, x + row * DIM_128);
        Cast(xFp32Reg, xReg, BF16_TO_FP32_NORM, validMask);
        LoadAlign(normReg, norm + row * DIM_128);
        Mul(yReg, xFp32Reg, scanReg, validMask);
        Mul(yReg, yReg, normReg, validMask);
        Cast(yOutReg, yReg, FP32_TO_BF16_NORM, validMask);
        StoreAlign(y + row * DIM_128, yOutReg, validMask);
    }
    (void)validCount;
}

// S3：save 档把 state 行做 dtype/布局整理；none 档调用点被 if constexpr 裁掉，函数不会被实例化。
template <typename DT>
__simd_vf__ inline void Stage3ExportVf(
    __ubuf__ const DT *stateRow, __ubuf__ DT *stateOut, uint16_t validRows)
{
    MaskReg validMask = UpdateMask<DT>(static_cast<uint32_t>(validRows) * DIM_128);
    RegTensor<DT> inReg;
    LoadAlign(inReg, stateRow);
    // 纯搬出不需要额外计算时，可以省略这一步直接在调用方 DataCopyPad；
    // 需要重量化/转置（state 布局与输出不同）时在这里加 Cast / Transpose 等 VF 指令。
    StoreAlign(stateOut, inReg, validMask);
}

// ══ ③ 数据层：只放数据，不放函数 ═════════════════════════════════════════════

template <typename DT, typename GT, uint32_t NORM_MODE, bool USE_STATE, uint32_t OUTPUT_MODE>
struct OpNameVectorContext {
    using DTypeX = DT;
    using DTypeG = GT;
    static constexpr bool kUseState = USE_STATE;
    static constexpr uint32_t kOutputMode = OUTPUT_MODE;

    // GM：输入 / 输出 / workspace 各一组，顺序与入口形参一致。
    AscendC::GlobalTensor<DT> xGm;
    AscendC::GlobalTensor<GT> gGm;
    AscendC::GlobalTensor<float> aLogGm;
    AscendC::GlobalTensor<DT> initialStateGm;
    AscendC::GlobalTensor<DT> yGm;
    AscendC::GlobalTensor<DT> stateGm;
    AscendC::GlobalTensor<float> xNormGm;
    AscendC::GlobalTensor<float> normWorkspaceGm;
    AscendC::GlobalTensor<DT> stateWorkspaceGm;

    // UB：按 arch35/op_name_struct.h 的布局常量切片，每个 slot 一个张量。
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
    AscendC::LocalTensor<DT> x[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<GT> g[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> norm[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> scan[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> y[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> state[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> xNormOut;

    // 核内事件：每 slot 一组，Init 预置首轮，收尾统一 Release。
    int32_t mte2ToV[UB_SLOT_COUNT_2] = {0, 0};
    int32_t vToMte3[UB_SLOT_COUNT_2] = {0, 0};
    int32_t mte3ToMte2[UB_SLOT_COUNT_2] = {0, 0};

    // 只读状态与自增游标。
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

// ══ ④ 行为层：全部是文件作用域 inline 函数，第一参数是上面的数据 ═══════════════

// 接线 + 资源划分 + 事件预置。函数在结构体外，`ctx` 只被读写，不承载逻辑。
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
    // 可选入参的"缺席"由 host 归一（L2 对缺席的可选张量传零元素 descriptor 占位，见规范 §5.1 第 3 条），
    // 因此 kernel 不做 nullptr 判断：本函数只接线，指针合法性由 host 侧保证。
    ctx.aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
    ctx.xNormGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xNorm));

    // ② 只读状态：核号 / 子核号两个 AIV 都要 clamp
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

    // ③ UB 划分：与文件头布局表逐行对应；norm 与 scan 共段，注释写明复用时机。
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
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        // save 档的导出暂存在尾部：none 档不申请，UB 总量不被撑大。
        ctx.state[0] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(
            VEC_TILE_ELEMS, UB_STATE_OFFSET);
        ctx.state[1] = ctx.ubBuf.template GetWithOffset<typename Ctx::DTypeX>(
            VEC_TILE_ELEMS, UB_STATE_OFFSET + VEC_X_BYTES);
        ctx.xNormOut = ctx.ubBuf.template GetWithOffset<float>(CHUNK_SIZE_64, UB_XNORM_OFFSET);
    }

    // ④ 核内事件：每 slot 一组，首轮先开放 free 方向（首轮没有上一轮消费者）。
    for (uint32_t slot = 0; slot < UB_SLOT_COUNT_2; ++slot) {
        ctx.mte2ToV[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::MTE2_V>();
        ctx.vToMte3[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::V_MTE3>();
        ctx.mte3ToMte2[slot] = ctx.pipe->template AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    }
}

// S0 输入：x、g（GM）；输出：normWorkspace（GM）
//    复用：slot 在本核承包的 HV 之间轮转，每轮只搬当前 owner，不预搬下一个
//    同步：MTE2→V→MTE3 三段核内事件 + 结束后 CrossCoreSetFlag 通知 AIC
template <typename Ctx>
__aicore__ inline void Stage0CopyInNorm(Ctx &ctx, const ChunkInfo &chunk)
{
    const uint32_t slot = ctx.streamSlot;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    AscendC::DataCopy(ctx.x[slot], ctx.xGm[chunk.tokenStart * DIM_128], chunk.chunkLen * DIM_128);
    AscendC::DataCopy(ctx.g[slot], ctx.gGm[chunk.tokenStart], chunk.chunkLen);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.mte2ToV[slot]);
    // VF 融合调用：入参是 UB 裸指针 + 有效行数，函数内不再碰 GM。
    Stage0NormVf<typename Ctx::DTypeX>(
        reinterpret_cast<__ubuf__ const typename Ctx::DTypeX *>(ctx.x[slot].GetPhyAddr()),
        static_cast<uint16_t>(chunk.chunkLen),
        reinterpret_cast<__ubuf__ float *>(ctx.norm[slot].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ctx.scan[slot].GetPhyAddr()),
        ctx.tiling->epsilon);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::DataCopyPad(ctx.normWorkspaceGm[GetWorkspaceChunkOffset(
                             ctx.coreIdx, ctx.taskRound, chunk.chunkIdx)],
                         ctx.norm[slot],
                         {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0});
    if constexpr (Ctx::kOutputMode == OP_NAME_TPL_OUTPUT_SAVE) {
        // x_norm 是反向中间量：S0 已经算出 norm，先留专用暂存给 S3，
        // 避免 S2 把 norm/scan 共享区改成 scan 之后再算一遍。
        AscendC::DataCopy(ctx.xNormOut, ctx.norm[slot], chunk.chunkLen);
    }
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(VEC_TO_CUBE_READY_FLAG);
}

// S2 输入：x、g（GM）、stateWorkspace（GM，AIC 生产）；输出：y（GM）
//    复用：norm 段已被 S0 的 MTE3 读完，本 Stage 把同一段改成 scan
//    同步：先 CrossCoreWaitFlag 等 AIC，再做核内三段事件
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
        reinterpret_cast<__ubuf__ const typename Ctx::DTypeX *>(ctx.x[slot].GetPhyAddr()),
        reinterpret_cast<__ubuf__ const typename Ctx::DTypeG *>(ctx.g[slot].GetPhyAddr()),
        reinterpret_cast<__ubuf__ const float *>(ctx.norm[slot].GetPhyAddr()),
        reinterpret_cast<__ubuf__ typename Ctx::DTypeX *>(ctx.y[slot].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ctx.scan[slot].GetPhyAddr()),
        static_cast<uint16_t>(chunk.chunkLen), ctx.tiling->scale);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.vToMte3[slot]);
    AscendC::DataCopyPad(ctx.yGm[chunk.tokenStart * DIM_128], ctx.y[slot],
                         {static_cast<uint16_t>(chunk.chunkLen),
                          static_cast<uint32_t>(chunk.chunkLen * sizeof(typename Ctx::DTypeX)), 0,
                          static_cast<uint32_t>((DIM_128 - chunk.chunkLen) *
                                                sizeof(typename Ctx::DTypeX)), 0});
    // 本 slot 的 MTE3 已发射完，开放给下一轮 MTE2（本轮唯一 free 方向）。
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.mte3ToMte2[slot]);
    ctx.streamSlot ^= 1U;
}

// S3 输入：stateWorkspace（GM，AIC 生产）；输出：state、x_norm（GM）
//    说明：只在 save 档实例化；用独立导出 slot，不与 S2 的 streamSlot 抢 buffer
//    同步：MTE2 搬入 → VF 整理 → MTE3 搬出，三段事件成对，末尾恢复 free
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
        Stage3ExportVf<typename Ctx::DTypeX>(
            reinterpret_cast<__ubuf__ const typename Ctx::DTypeX *>(ctx.state[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ typename Ctx::DTypeX *>(ctx.state[slot].GetPhyAddr()),
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

// 收尾：把每个 slot 的事件等回闭环再释放，顺序与 Init 的预置一一对应。
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

// 任务主循环：只做三件事——取 ChunkInfo、按 Stage 顺序调函数、退出前闭环事件。
template <typename Ctx>
__aicore__ inline void ProcessOpNameVector(Ctx &ctx)
{
    ChunkInfo chunk;
    for (int64_t taskIdx = ctx.coreIdx; taskIdx < ctx.tiling->taskNum; taskIdx += ctx.coreNum) {
        GetChunkInfo(taskIdx, ctx.cuSeqlens, ctx.chunkIndices, *ctx.tiling, chunk);
        // host 侧已保证 taskIdx 范围与 metadata 合法性；这里是防御性跳过（不可达），
        // 不是拦截：不返回错误码、不打错误日志。
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

#endif // OP_NAME_VEC_ARCH35_H
