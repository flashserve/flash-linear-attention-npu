/**
 * 示例文件：.../op_kernel/arch22/op_name_vec.h（A2/A3 的 AIV 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_vector.h
 *
 * 与 arch35 版本**同骨架、同命名**：四层结构（Stage 函数 → 类 → 常量 → 成员）、
 * 同名 Stage、同名 EventID 数组、同名收尾函数。平台差异只允许出现在下表列出的位置，
 * 这样两个平台可以逐行对照检视，不用重新建立心智模型。
 *
 * ── 平台差异表（与 arch35 逐条对照）──────────────────────────────────────────
 *   项               arch22（A2/A3）                  arch35（A5）
 *   tile 行数        TILE_T_ARCH22                    TILE_T_ARCH35（更大）
 *   UB 总量          UB_TOTAL_BYTES（arch22 常量）    UB_TOTAL_BYTES（arch35 常量）
 *   向量计算         逐 repeat 的向量指令 + MaskReg   可合并为 VF/RegBase 通路
 *   尾块处理         UpdateMask 后按 repeat 收尾      同 arch35（语义一致，粒度不同）
 *   事件数量         每 slot 一组，同 arch35          同 arch22
 *   同步协议         common.h 的具名 flag             同 arch22（不允许各写一套）
 *
 * ── Stage 表 / UB 布局表 / 同步协议表 ────────────────────────────────────────
 *   三张表与 arch35 版本同名同序，只有偏移数值不同（arch35 文件头表里给了逐行数值，这里是
 *   arch22 的等价表）：X 0 / G 16 KiB / NORM 24 KiB / SCAN 28 KiB / Y 32 KiB / STATE 48 KiB /
 *   XNORM 80 KiB，UB 总量见 arch22/op_name_struct.h 的 UB_TOTAL_BYTES。
 *   Stage 顺序一致：S0 CopyInNorm(AIV) → S1 Matrix(AIC) → S2 ScanWrite(AIV) → S3 SaveExport。
 *   两平台共用 op_kernel/op_name_common.h 的 flag 常量与 GetChunkInfo/GetWorkspaceChunkOffset。
 */

#ifndef OP_NAME_VEC_ARCH22_H
#define OP_NAME_VEC_ARCH22_H

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

// ══ ② 计算层：只碰 UB ═════════════════════════════════════════════════════════

// S0：norm[i] = rsqrt(sum_d(x[i, d]^2) / D + epsilon)
// A2/A3 没有 A5 的 VF 通路，按 repeat 展开；尾块由 UpdateMask 收尾，padding 不参与计算。
template <typename DT>
__aicore__ inline void Stage0NormVf(
    __ubuf__ const DT *x, uint16_t validLen,
    __ubuf__ float *norm, __ubuf__ float *rowSum, float epsilon)
{
    // ... Mul / ReduceSum / Adds(epsilon) / Rsqrt / StoreAlign(norm)，
    //     逐步 Add 之间若依赖同一 UB 的写后读，用 PipeBarrier<PIPE_V>() 定序 ...
    (void)x;
    (void)validLen;
    (void)norm;
    (void)rowSum;
    (void)epsilon;
}

// S2：y = x * scan(g * scale) * norm；scan 在 chunk 内累加，跨 chunk 用 carry 续算。
template <typename DT, typename GT>
__aicore__ inline void Stage2ScanVf(
    __ubuf__ const DT *x, __ubuf__ const GT *g, __ubuf__ const float *norm,
    __ubuf__ DT *y, __ubuf__ float *scan, uint16_t validLen, float scale)
{
    // ... Muls(scale) / 前缀和 / Mul(norm) / Mul(x) / StoreAlign(y) ...
    (void)x;
    (void)g;
    (void)norm;
    (void)y;
    (void)scan;
    (void)validLen;
    (void)scale;
}

// S3：save 档把 state 行搬到导出暂存；none 档调用点被 if constexpr 裁掉。
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
        // ① GM 接线
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(x));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(g));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(y));
        normWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normWorkspace));
        stateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(stateWorkspace));
        if constexpr (USE_STATE) {
            // 只有带初始状态时才绑定 state/initialState。
            stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(state));
            initialStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(initialState));
        }
        if (aLog != nullptr) {
            aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
        }
        xNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xNorm));

        // ② 只读状态：核号 / 子核号都要 clamp，行为与 arch35 一致。
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

        // ③ UB 划分：与 arch35 同一张布局表，只是 tile 与总量按 arch22 常量取。
        pipe_->InitBuffer(ubBuf_, UB_TOTAL_BYTES);
        x_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, X_OFFSET);
        x_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, X_OFFSET + X_BYTES);
        g_[0] = ubBuf_.GetWithOffset<GT>(CHUNK_SIZE_64, G_OFFSET);
        g_[1] = ubBuf_.GetWithOffset<GT>(CHUNK_SIZE_64, G_OFFSET + G_BYTES);
        // norm 与 scan 共用同一段：S0 的 norm 被 MTE3 读完即释放，S2 才改成 scan。
        norm_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, NORM_OFFSET);
        norm_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, NORM_OFFSET + NORM_BYTES);
        scan_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, SCAN_OFFSET);
        scan_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, SCAN_OFFSET + SCAN_BYTES);
        y_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, Y_OFFSET);
        y_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, Y_OFFSET + Y_BYTES);
        if constexpr (OUTPUT_MODE == OP_NAME_TPL_OUTPUT_SAVE) {
            state_[0] = ubBuf_.GetWithOffset<DT>(CHUNK_SIZE_64 * DIM_128, STATE_OFFSET);
            state_[1] = ubBuf_.GetWithOffset<DT>(CHUNK_SIZE_64 * DIM_128, STATE_OFFSET + STATE_BYTES);
            xNormOut_ = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, XNORM_OFFSET);
        }

        // ④ 核内事件：每 slot 一组，首轮先开放 free 方向（与 arch35 完全一致）。
        for (uint32_t slot = 0; slot < UB_SLOT_COUNT_2; ++slot) {
            mte2ToV_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToMte2_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
    }

    __aicore__ inline void Process()
    {
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
                Stage3SaveExport(chunk);
            }
        }
        CloseAndReleaseEvents();
    }

private:
    // ── ④a 阶段函数：函数头固定写"输入 / 输出 / 复用 / 同步"，与 arch35 逐字对应 ──

    // S0 输入：x、g（GM）；输出：normWorkspace（GM）
    //    复用：slot 在本核承包的 HV 之间轮转
    //    同步：MTE2→V→MTE3 核内事件 + CrossCoreSetFlag 通知 AIC
    __aicore__ inline void Stage0CopyInNorm(const ChunkInfo &chunk)
    {
        const uint32_t slot = streamSlot_;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        AscendC::DataCopy(x_[slot], xGm_[chunk.tokenStart * DIM_128], chunk.chunkLen * DIM_128);
        AscendC::DataCopy(g_[slot], gGm_[chunk.tokenStart], chunk.chunkLen);
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
        AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(VEC_TO_CUBE_READY_FLAG);
    }

    // S2 输入：x、g（GM）、stateWorkspace（GM，AIC 生产）；输出：y（GM）
    //    复用：norm 段改成 scan（V 内先写后读用 PipeBarrier<PIPE_V>()）
    //    同步：先 CrossCoreWaitFlag 等 AIC，再做核内三段事件，末尾恢复正常 free
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

    // 收尾：与 arch35 顺序一致——先等回每个 slot，再 Release，不额外加 Wait。
    __aicore__ inline void CloseAndReleaseEvents()
    {
        for (uint32_t slot = 0; slot < UB_SLOT_COUNT_2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[slot]);
        }
    }

    // ── ④b 常量：UB 布局，与 arch35 同名同序，数值按 arch22 的 tile 取 ──
    static constexpr uint32_t VECTOR_ELEMS = TILE_T_ARCH22 * DIM_128;
    static constexpr uint32_t X_BYTES = VECTOR_ELEMS * sizeof(DT);
    static constexpr uint32_t G_BYTES = CHUNK_SIZE_64 * sizeof(GT);
    static constexpr uint32_t NORM_BYTES = CHUNK_SIZE_64 * sizeof(float);
    static constexpr uint32_t SCAN_BYTES = CHUNK_SIZE_64 * sizeof(float);
    static constexpr uint32_t Y_BYTES = VECTOR_ELEMS * sizeof(DT);
    static constexpr uint32_t STATE_BYTES = CHUNK_SIZE_64 * DIM_128 * sizeof(DT);
    static constexpr uint32_t XNORM_BYTES = CHUNK_SIZE_64 * sizeof(float);
    static constexpr uint32_t X_OFFSET = 0;
    static constexpr uint32_t G_OFFSET = 16 * 1024;
    static constexpr uint32_t NORM_OFFSET = 24 * 1024;
    static constexpr uint32_t SCAN_OFFSET = 28 * 1024;
    static constexpr uint32_t Y_OFFSET = 32 * 1024;
    static constexpr uint32_t STATE_OFFSET = 48 * 1024;
    static constexpr uint32_t XNORM_OFFSET = 80 * 1024;

    // ── ④c 成员：GM / UB / 事件 / 只读状态，分组与 arch35 一致 ──
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

#endif // OP_NAME_VEC_ARCH22_H
