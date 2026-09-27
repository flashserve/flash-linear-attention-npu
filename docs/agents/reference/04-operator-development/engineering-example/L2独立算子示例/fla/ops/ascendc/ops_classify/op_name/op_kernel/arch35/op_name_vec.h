/**
 * 示例文件：.../op_kernel/arch35/op_name_vec.h（A5 的 AIV 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_vector.h
 *
 * 只靠"薄入口 + 一个类"不足以让读者理解 kernel：真正的可读性来自**固定的四层结构**。
 * 本文件按参考算子的顺序排列，复制到新算子时保持同样的四层，不要打散：
 *
 *   ① 本注释块：Stage 表 / UB 布局表 / 同步协议表——读者先建立全局模型，再往下看代码
 *   ② StageNVf(...)：按 Stage 顺序排列的计算函数（只算 UB，不搬 GM、不发同步事件）
 *   ③ class OpNameVector：public Init（接线 + 资源划分）→ public Process（任务主循环 + 阶段编排）
 *   ④ private：阶段函数（搬入/计算/写回）→ 常量（UB 布局）→ 成员（buffer / 事件 / 只读状态）
 *
 * ── Stage 表（谁生产、谁消费、怎么同步）────────────────────────────────────────
 *   S0 CopyInNorm  AIV  MTE2 搬 x/g → VF 求 norm → MTE3 写 normWorkspace
 *                      → CrossCoreSetFlag(VEC_TO_CUBE_READY_FLAG) 通知 AIC
 *   S1 Matrix      AIC  CrossCoreWaitFlag → MTE2/MTE1 → Cube → Fixpipe 写 stateWorkspace
 *                      → CrossCoreSetFlag(CUBE_TO_VEC_READY_FLAG) 通知 AIV
 *   S2 ScanWrite   AIV  CrossCoreWaitFlag → MTE2 搬 x/g/stateWorkspace → VF 扫描 → MTE3 写 y
 *   S3 SaveExport  AIV  save 档额外 MTE3 写 state/x_norm；none 档被 `if constexpr` 整体裁掉
 *
 * ── UB 布局表（字节偏移，与 private 常量块逐行对应）───────────────────────────
 *   偏移        大小            内容                  生命周期（谁写 → 谁读 → 何时可复用）
 *   0           2 * X_BYTES     x slot0 / slot1       S0 MTE2 写 → S2 VF 读；S2 末尾释放
 *   32 KiB      2 * G_BYTES     g slot0 / slot1       S0 MTE2 写 → S2 VF 读
 *   48 KiB      2 * NORM_BYTES  norm slot0 / slot1    S0 VF 写 → S0 MTE3 读 → S2 复用为 scan
 *   56 KiB      2 * SCAN_BYTES  scan slot0 / slot1    S2 VF 写 → S2 VF 读（V pipe 内，用 PipeBarrier）
 *   64 KiB      2 * Y_BYTES     y 输出 slot0 / slot1  S2 VF 写 → S2 MTE3 读
 *   96 KiB      2 * STATE_BYTES state 导出暂存 slot0/1 save 档专用：AIC 写 stateWorkspace → 本核搬入再搬出
 *   128 KiB     XNORM_BYTES     x_norm 导出暂存       save 档专用：S0 把 norm 留在专用暂存 → S3 搬出
 *   A5 的 UB 更大，tile 取 TILE_T_ARCH35；A2/A3 用 arch22 的同名常量。
 *
 * ── 同步协议（flagId 定义在 op_kernel/op_name_common.h）───────────────────────
 *   核间：VEC_TO_CUBE_READY_FLAG / CUBE_TO_VEC_READY_FLAG 双向握手，互为背压；
 *         VF 只处理一个 owner HV，AIC/AIV 按同一 HEAD 顺序消费，不依赖核启动顺序。
 *   核内：每个 slot 一组 EventID（MTE2→V 入、V→MTE3 出、MTE3→MTE2 复用），
 *         首轮在 Init 末尾预置，收尾在 Process 末尾闭环后 Release，不留悬空事件。
 */

#ifndef OP_NAME_VEC_ARCH35_H
#define OP_NAME_VEC_ARCH35_H

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

// ══ ② 计算层：只碰 UB ═════════════════════════════════════════════════════════
// 入参统一为 `__ubuf__` 裸指针 + 有效长度，理由与 finalize 相同：
//   - 读代码时只关心数据流，不用在计算里反推 buffer 偏移；
//   - UB 地址由类内常量集中给出，VF 函数不重复推导；
//   - 尾 chunk 只传 validLen，padding 区永不参与计算。

// S0：norm[i] = rsqrt(sum_d(x[i, d]^2) / D + epsilon)
template <typename DT>
__aicore__ inline void Stage0NormVf(
    __ubuf__ const DT *x, uint16_t validLen,
    __ubuf__ float *norm, __ubuf__ float *rowSum, float epsilon)
{
    // 先用 float 累加再回写，避免 BF16 累加丢精度；
    // 同一 UB 上 V pipe 内先写后读时用 PipeBarrier<PIPE_V>() 定序，
    // 跨 pipe 的 RAW/WAR 交给外层事件，不用 barrier 掩盖。
    // ... Mul / ReduceSum / Adds(epsilon) / Rsqrt / StoreAlign(norm) ...
    (void)x;
    (void)validLen;
    (void)norm;
    (void)rowSum;
    (void)epsilon;
}

// S2：chunk 内扫描 + 归一化回写 y。
// 数据流：y = x * scan(g * scale) * norm；scan 在 chunk 内累加，跨 chunk 用 carry 续算。
template <typename DT, typename GT>
__aicore__ inline void Stage2ScanVf(
    __ubuf__ const DT *x, __ubuf__ const GT *g, __ubuf__ const float *norm,
    __ubuf__ DT *y, __ubuf__ float *scan, uint16_t validLen, float scale)
{
    // ... LoadAlign(g) / Muls(scale) / CumSum 或逐步 Add / Mul(norm) / Mul(x) / StoreAlign(y) ...
    (void)x;
    (void)g;
    (void)norm;
    (void)y;
    (void)scan;
    (void)validLen;
    (void)scale;
}

// S3：save 档把 state 行从 UB 逐行搬出；none 档调用点被 if constexpr 裁掉，本函数不会被实例化。
template <typename DT>
__aicore__ inline void Stage3ExportVf(
    __ubuf__ const DT *stateRow, __ubuf__ DT *stateOut, uint16_t validLen)
{
    // ... 仅做 dtype 转换/重排，不在这里搬 GM ...
    (void)stateRow;
    (void)stateOut;
    (void)validLen;
}

// ══ ③ 角色类：Init 只接线，Process 只编排 ═════════════════════════════════════

template <typename DT, typename GT, uint32_t NORM_MODE, bool USE_STATE, uint32_t OUTPUT_MODE>
class OpNameVector {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR g, GM_ADDR aLog, GM_ADDR initialState,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR y, GM_ADDR state, GM_ADDR xNorm,
        GM_ADDR normWorkspace, GM_ADDR stateWorkspace,
        const OpNameTilingData *tiling, AscendC::TPipe *pipe)
    {
        // ① GM 接线：一张输入 / 一张输出 / 一张 workspace，逐行对齐入口的形参顺序。
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(x));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(g));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(y));
        normWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normWorkspace));
        stateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(stateWorkspace));
        if constexpr (USE_STATE) {
            // 只有带初始状态时才绑定 state，避免 none 档留下未使用地址。
            stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(state));
            initialStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(initialState));
        }
        if (aLog != nullptr) {
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
        }
        xNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xNorm));

        // ② 只读状态：核号 / 子核号，两个 AIV 都要 clamp（与 finalize 一致），
        //    避免异常核号进入后面的 offset 计算。
        tiling_ = tiling;
        pipe_ = pipe;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx()) / AIV_COUNT_2;
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        subBlockIdx_ = static_cast<int64_t>(AscendC::GetSubBlockIdx());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
        if (subBlockIdx_ < 0 || subBlockIdx_ >= AIV_COUNT_2) {
            subBlockIdx_ = 0;
        }

        // ③ UB 划分：与文件头布局表逐行对应，一行一个 slot，并写清复用关系。
        pipe_->InitBuffer(ubBuf_, UB_TOTAL_BYTES);
        x_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, X_OFFSET);
        x_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, X_OFFSET + X_BYTES);
        g_[0] = ubBuf_.GetWithOffset<GT>(CHUNK_SIZE_64, G_OFFSET);
        g_[1] = ubBuf_.GetWithOffset<GT>(CHUNK_SIZE_64, G_OFFSET + G_BYTES);
        // norm 与 scan 共用同一段空间：S0 的 norm 被 MTE3 读完即释放，S2 才改成 scan。
        norm_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, NORM_OFFSET);
        norm_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, NORM_OFFSET + NORM_BYTES);
        scan_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, SCAN_OFFSET);
        scan_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, SCAN_OFFSET + SCAN_BYTES);
        y_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, Y_OFFSET);
        y_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, Y_OFFSET + Y_BYTES);
        if constexpr (OUTPUT_MODE == OP_NAME_TPL_OUTPUT_SAVE) {
            // save 档的两块导出暂存放在尾部：none 档不申请，UB 总量也不会被撑大。
            state_[0] = ubBuf_.GetWithOffset<DT>(CHUNK_SIZE_64 * DIM_128, STATE_OFFSET);
            state_[1] = ubBuf_.GetWithOffset<DT>(CHUNK_SIZE_64 * DIM_128, STATE_OFFSET + STATE_BYTES);
            xNormOut_ = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, XNORM_OFFSET);
        }

        // ④ 核内事件：每个 slot 一组。首轮不存在上一轮的 MTE3 消费者，
        //    必须先在 Init 里 SetFlag 开放 slot，否则第一次 Wait 会等一个不会到来的事件。
        for (uint32_t slot = 0; slot < UB_SLOT_COUNT_2; ++slot) {
            mte2ToV_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToMte2_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
    }

    __aicore__ inline void Process()
    {
        // 任务主循环：只做三件事——取 ChunkInfo、按 Stage 顺序调阶段函数、退出前闭环事件。
        // 定长/变长、batch 内偏移、packed offset 的差异全部封在 GetChunkInfo 里。
        ChunkInfo chunk;
        for (int64_t taskIdx = coreIdx_; taskIdx < tiling_->taskNum; taskIdx += coreNum_) {
            GetChunkInfo(taskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            // workspace 轮次：同一个核第几轮处理任务，workspace 双窗口按它切换。
            taskRound_ = (taskIdx - coreIdx_) / coreNum_;
            Stage0CopyInNorm(chunk);
            Stage2ScanWrite(chunk);
            if constexpr (OUTPUT_MODE == OP_NAME_TPL_OUTPUT_SAVE) {
                // 档位是编译期模板参数：none 档的这段搬出会被整体裁掉，不留运行期判断。
                Stage3SaveExport(chunk);
            }
        }
        CloseAndReleaseEvents();
    }

private:
    // ── ④a 阶段函数：一个 Stage 一个函数，函数头固定写"输入 / 输出 / 复用 / 同步" ──

    // S0 输入：x、g（GM）；输出：normWorkspace（GM）
    //    复用：slot 在本 AIV 承包的 HV 之间轮转，每轮只搬当前 owner，不预搬下一个
    //    同步：MTE2→V→MTE3 三段核内事件 + 结束后 CrossCoreSetFlag 通知 AIC
    __aicore__ inline void Stage0CopyInNorm(const ChunkInfo &chunk)
    {
        const uint32_t slot = streamSlot_;
        const int64_t tokenOffset = chunk.tokenStart;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::DataCopy(x_[slot], xGm_[tokenOffset * DIM_128], chunk.chunkLen * DIM_128);
        AscendC::DataCopy(g_[slot], gGm_[tokenOffset], chunk.chunkLen);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        Stage0NormVf<DT>(
            reinterpret_cast<__ubuf__ const DT *>(x_[slot].GetPhyAddr()),
            static_cast<uint16_t>(chunk.chunkLen),
            reinterpret_cast<__ubuf__ float *>(norm_[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(scan_[slot].GetPhyAddr()),
            tiling_->epsilon);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopyPad(normWorkspaceGm_[GetWorkspaceChunkOffset(
                                 coreIdx_, taskRound_, chunk.chunkIdx)],
                             norm_[slot],
                             {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0});
        if constexpr (OUTPUT_MODE == OP_NAME_TPL_OUTPUT_SAVE) {
            // x_norm 是反向中间量：S0 已经算出 norm，先留在专用暂存里由 S3 搬出，
            // 避免 S2 把 norm/scan 共享区改写成 scan 之后还要重算一遍。
            AscendC::DataCopy(xNormOut_, norm_[slot], chunk.chunkLen);
        }
        // 一个 HEAD 计算完立刻通知 AIC，不等整组完成：AIC 侧按同一 HEAD 顺序消费。
        AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(VEC_TO_CUBE_READY_FLAG);
    }

    // S2 输入：x、g（GM）、stateWorkspace（GM，由 AIC 生产）；输出：y（GM）
    //    复用：norm 段已被 S0 的 MTE3 读完，本 Stage 把同一段改成 scan（V 内先写后读用 barrier）
    //    同步：先 CrossCoreWaitFlag 等 AIC，再做核内三段事件
    __aicore__ inline void Stage2ScanWrite(const ChunkInfo &chunk)
    {
        const uint32_t slot = streamSlot_;
        AscendC::CrossCoreWaitFlag(CUBE_TO_VEC_READY_FLAG);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::DataCopy(x_[slot], xGm_[chunk.tokenStart * DIM_128], chunk.chunkLen * DIM_128);
        AscendC::DataCopy(g_[slot], gGm_[chunk.tokenStart], chunk.chunkLen);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        Stage2ScanVf<DT, GT>(
            reinterpret_cast<__ubuf__ const DT *>(x_[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ const GT *>(g_[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ const float *>(norm_[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ DT *>(y_[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(scan_[slot].GetPhyAddr()),
            static_cast<uint16_t>(chunk.chunkLen), tiling_->scale);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopyPad(yGm_[chunk.tokenStart * DIM_128], y_[slot],
                             {static_cast<uint16_t>(chunk.chunkLen),
                              static_cast<uint32_t>(chunk.chunkLen * sizeof(DT)), 0,
                              static_cast<uint32_t>((DIM_128 - chunk.chunkLen) * sizeof(DT)), 0});
        // 本 slot 的 MTE3 已发射完，开放给下一轮的 MTE2（也是本轮唯一的 free 方向）。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        streamSlot_ ^= 1U;
    }

    // S3 输入：stateWorkspace（GM，AIC 生产）；输出：state、x_norm（GM）
    //    说明：只在 save 档实例化；用独立的导出 slot，不与 S2 的 streamSlot_ 抢 buffer。
    //    同步：MTE2 搬入 → V 整理 → MTE3 搬出，三段事件成对，末尾重新开放 free。
    __aicore__ inline void Stage3SaveExport(const ChunkInfo &chunk)
    {
        const uint32_t slot = exportSlot_;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::DataCopy(state_[slot],
                          stateWorkspaceGm_[GetWorkspaceChunkOffset(
                              coreIdx_, taskRound_, chunk.chunkIdx)],
                          chunk.chunkLen * DIM_128);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
        Stage3ExportVf<DT>(
            reinterpret_cast<__ubuf__ const DT *>(state_[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ DT *>(state_[slot].GetPhyAddr()),
            static_cast<uint16_t>(chunk.chunkLen));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
        AscendC::DataCopyPad(stateGm_[chunk.chunkIdx * DIM_128], state_[slot],
                             {static_cast<uint16_t>(chunk.chunkLen),
                              static_cast<uint32_t>(chunk.chunkLen * sizeof(DT)), 0,
                              static_cast<uint32_t>((DIM_128 - chunk.chunkLen) * sizeof(DT)), 0});
        AscendC::DataCopyPad(xNormGm_[chunk.tokenStart * DIM_128], xNormOut_,
                             {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
    }

    // 收尾：把每个 slot 的事件等回闭环再释放，避免 kernel 退出时留下未消费的 flag。
    // 这一段的顺序与 Init 的 SetFlag 一一对应，不做"看起来对称"的额外 Wait。
    __aicore__ inline void CloseAndReleaseEvents()
    {
        for (uint32_t slot = 0; slot < UB_SLOT_COUNT_2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
    }

    // ── ④b 常量：UB 布局，与文件头布局表逐行对应；改一处必须同步改表和 arch22 ──
    static constexpr uint32_t VECTOR_ELEMS = TILE_T_ARCH35 * DIM_128;
    static constexpr uint32_t X_BYTES = VECTOR_ELEMS * sizeof(DT);
    static constexpr uint32_t G_BYTES = CHUNK_SIZE_64 * sizeof(GT);
    static constexpr uint32_t NORM_BYTES = CHUNK_SIZE_64 * sizeof(float);
    static constexpr uint32_t SCAN_BYTES = CHUNK_SIZE_64 * sizeof(float);
    static constexpr uint32_t Y_BYTES = VECTOR_ELEMS * sizeof(DT);
    static constexpr uint32_t X_OFFSET = 0;
    static constexpr uint32_t G_OFFSET = 32 * 1024;
    static constexpr uint32_t NORM_OFFSET = 48 * 1024;
    static constexpr uint32_t SCAN_OFFSET = 56 * 1024;
    static constexpr uint32_t Y_OFFSET = 64 * 1024;
    static constexpr uint32_t STATE_BYTES = CHUNK_SIZE_64 * DIM_128 * sizeof(DT);
    static constexpr uint32_t XNORM_BYTES = CHUNK_SIZE_64 * sizeof(float);
    static constexpr uint32_t STATE_OFFSET = 96 * 1024;
    static constexpr uint32_t XNORM_OFFSET = 128 * 1024;

    // ── ④c 成员：GM / UB / 事件 / 只读状态。分组排列，新增成员进对应分组 ──
    AscendC::GlobalTensor<DT> xGm_;
    AscendC::GlobalTensor<GT> gGm_;
    AscendC::GlobalTensor<float> aLogGm_;
    AscendC::GlobalTensor<DT> initialStateGm_;
    AscendC::GlobalTensor<DT> yGm_;
    AscendC::GlobalTensor<DT> stateGm_;
    AscendC::GlobalTensor<float> xNormGm_;
    AscendC::GlobalTensor<float> normWorkspaceGm_;
    AscendC::GlobalTensor<DT> stateWorkspaceGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<DT> x_[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<GT> g_[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> norm_[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> scan_[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> y_[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<DT> state_[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> xNormOut_;

    int32_t mte2ToV_[UB_SLOT_COUNT_2] = {0, 0};
    int32_t vToMte3_[UB_SLOT_COUNT_2] = {0, 0};
    int32_t mte3ToMte2_[UB_SLOT_COUNT_2] = {0, 0};

    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const OpNameTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    int64_t subBlockIdx_ = 0;
    int64_t taskRound_ = 0;
    uint32_t streamSlot_ = 0;
    uint32_t exportSlot_ = 1;
};

} // namespace OpsClassify

#endif // OP_NAME_VEC_ARCH35_H
