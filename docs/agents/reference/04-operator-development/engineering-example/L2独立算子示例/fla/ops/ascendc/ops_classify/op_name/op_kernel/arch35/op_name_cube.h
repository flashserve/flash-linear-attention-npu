/**
 * 示例文件：.../op_kernel/arch35/op_name_cube.h（A5 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *
 * 与 vec 版本相同的四层结构，只是角色换成 AIC：
 *   ① 本注释块：Stage 表 / L1+L0 布局表 / 同步协议表
 *   ② 常量：L1/L0 偏移、元素与字节数、Fixpipe 与 layout 常量
 *   ③ class OpNameCube：Init（接线 + 事件预置）→ Process（任务主循环 + 阶段编排 + 收尾）
 *   ④ private：阶段函数（搬入 → Cube → Fixpipe），一个 Stage 一个函数
 *
 * ── Stage 表 ─────────────────────────────────────────────────────────────────
 *   S1a CopyIn    AIC  CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG) → MTE2 搬 norm/state 到 L1
 *   S1b Matmul    AIC  MTE1 把 L1 数据搬 L0A/L0B → Cube 计算 → L0C
 *   S1c Fixpipe   AIC  Fixpipe 把 L0C 写 stateWorkspace
 *                      → CrossCoreSetFlag(CUBE_TO_VEC_READY_FLAG) 通知 AIV 做 S2
 *
 * ── L1 / L0 布局表（与 private 常量块逐行对应）────────────────────────────────
 *   层级  偏移        大小            内容                     生命周期
 *   L1    L1_NORM_OFFSET   1 * L1_SLOT_BYTES   AIV 产出的 norm    S1a 搬入 → S1b MTE1 读
 *   L1    L1_STATE_OFFSET  2 * L1_SLOT_BYTES   矩阵分块常驻        S1b 读 → S1c Fixpipe 写回
 *   L0A   -                L0A_BYTES          左矩阵             MTE1 写 → Cube 读（按 slot 双缓冲）
 *   L0B   -                L0B_BYTES          右矩阵             MTE1 写 → Cube 读（按 slot 双缓冲）
 *   L0C   -                L0C_BYTES          累加结果           Cube 写 → Fixpipe 读
 *
 * ── 同步协议 ─────────────────────────────────────────────────────────────────
 *   核间：等待 AIV 的 ready，再在 Fixpipe 完成后广播给两个 AIV；广播用 PIPE_FIX，
 *         保证 AIV 看到的是已经落 L1/GM 的数据，而不是仍在 Fixpipe 里的中间态。
 *   核内：三类事件（MTE1→MTE2 复用、Cube→MTE1 复用、Fixpipe→Cube 复用）按 slot 成对，
 *         首轮在 Init 预置，收尾在 Process 末尾闭环后 Release，不靠核启动顺序。
 */

#ifndef OP_NAME_CUBE_ARCH35_H
#define OP_NAME_CUBE_ARCH35_H

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

template <typename DT, uint32_t NORM_MODE, uint32_t OUTPUT_MODE>
class OpNameCube {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR g, GM_ADDR normWorkspace, GM_ADDR stateWorkspace,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, const OpNameTilingData *tiling,
        AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        // ① GM 接线：AIC 只读 workspace 与 tiling，不直接读用户输入。
        normWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normWorkspace));
        stateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(stateWorkspace));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;

        // ② 只读状态：AIC 与 AIV 用同一套核号语义，便于两平台对照。
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
        chunkTaskNum_ = static_cast<int64_t>(tiling_->taskNum);

        // ③ L1/L0 资源划分：与文件头布局表逐行对应，L1 偏移用 arch35 常量。
        pipe_->InitBuffer(l1Buf_, L1_TOTAL_BYTES);
        normL1_ = l1Buf_.GetWithOffset<DT>(L1_SLOT_ELEMS, L1_NORM_OFFSET);
        stateL1_ = l1Buf_.GetWithOffset<DT>(2 * L1_SLOT_ELEMS, L1_STATE_OFFSET);
        pipe_->InitBuffer(l0aBuf_, L0A_BYTES);
        pipe_->InitBuffer(l0bBuf_, L0B_BYTES);
        pipe_->InitBuffer(l0cBuf_, L0C_BYTES);

        // ④ 核内事件：每个 slot 一组。首轮不存在上一轮的消费者，
        //    必须先在 Init 里 SetFlag 开放，否则第一次 Wait 会挂住。
        for (uint32_t slot = 0; slot < L1_SLOT_COUNT_2; ++slot) {
            mte1ToMte2_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            cubeMte1_[slot] = pipe_->AllocEventID<AscendC::HardEvent::M_MTE1>();
            fixpipeToCube_[slot] = pipe_->AllocEventID<AscendC::HardEvent::FIX_M>();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(mte1ToMte2_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(cubeMte1_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixpipeToCube_[slot]);
        }
    }

    __aicore__ inline void Process()
    {
        // 任务主循环：AIC 与 AIV 用同一份 GetChunkInfo，保证两个角色算出的
        // tokenStart/chunkLen 完全一致；只有 Stage 划分不同。
        ChunkInfo chunk;
        for (int64_t taskIdx = coreIdx_; taskIdx < chunkTaskNum_; taskIdx += coreNum_) {
            GetChunkInfo(taskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            // workspace 轮次：与 AIV 用同一算式，两个角色才能命中同一段 workspace。
            taskRound_ = (taskIdx - coreIdx_) / coreNum_;
            Stage1MatrixChunk(chunk);
        }
        CloseAndReleaseEvents();
    }

private:
    // ── ④ 阶段函数：一个 Stage 一个函数，函数头写"输入 / 输出 / 复用 / 同步" ──

    // S1 输入：normWorkspace（GM，AIV 生产）；输出：stateWorkspace（GM）
    //    复用：L1 两块常驻，L0 按 slot 双缓冲轮转
    //    同步：先等 AIV ready，Fixpipe 后再广播给 AIV（PIPE_FIX）
    __aicore__ inline void Stage1MatrixChunk(const ChunkInfo &chunk)
    {
        const uint32_t slot = streamSlot_;
        AscendC::CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG);
        Stage1CopyIn(chunk, slot);
        Stage1Matmul(chunk, slot);
        Stage1FixpipeOut(chunk, slot);
        // 全程一条 ready 链：本次广播既是 S2 的生产信号，也是下一轮 AIV ready 的背压来源。
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_TO_VEC_READY_FLAG);
        streamSlot_ ^= 1U;
    }

    // S1a：MTE2 把 AIV 产出的 norm 搬进 L1。地址只出现在这里，Stage 内不重算偏移。
    __aicore__ inline void Stage1CopyIn(const ChunkInfo &chunk, uint32_t slot)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(mte1ToMte2_[slot]);
        AscendC::DataCopy(normL1_, normWorkspaceGm_[GetWorkspaceChunkOffset(
                                       coreIdx_, taskRound_, chunk.chunkIdx)],
                          chunk.chunkLen * DIM_128);
    }

    // S1b：MTE1 搬 L1→L0，Cube 计算后写 L0C。
    //    只使用样板算子同名的三类事件：MTE1_MTE2（L1 复用）、FIX_M（L0C 复用）、M_MTE1（L0 复用）。
    __aicore__ inline void Stage1Matmul(const ChunkInfo &chunk, uint32_t slot)
    {
        // ... MTE1：L1 -> L0A/L0B（按 slot 双缓冲）...
        // MTE1 读完 L1 就释放给下一轮 MTE2，这是本 Stage 唯一的 free 方向。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(mte1ToMte2_[slot]);
        // 复用 L0C 之前先等上一轮 Fixpipe 读完。
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixpipeToCube_[slot]);
        // ... Cube：L0A x L0B -> L0C；累加顺序按 chunk 顺序，不得并行重排 ...
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(cubeMte1_[slot]);
    }

    // S1c：Fixpipe 把 L0C 落到 stateWorkspace；save 档才写公开 state，none 档整体裁掉。
    __aicore__ inline void Stage1FixpipeOut(const ChunkInfo &chunk, uint32_t slot)
    {
        // ... Fixpipe：L0C -> L1/GM，按 arch35 的 FixpipeConfig ...
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixpipeToCube_[slot]);
        if constexpr (OUTPUT_MODE == OP_NAME_TPL_OUTPUT_SAVE) {
            // ... 额外落 stateWorkspace_[...]，供 AIV 的 Stage3 导出 ...
        }
    }

    // 收尾：等回每个 slot 的在途事件再 Release，顺序与 Init 的 SetFlag 一一对应。
    __aicore__ inline void CloseAndReleaseEvents()
    {
        for (uint32_t slot = 0; slot < L1_SLOT_COUNT_2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(mte1ToMte2_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(cubeMte1_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(fixpipeToCube_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(mte1ToMte2_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::M_MTE1>(cubeMte1_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::FIX_M>(fixpipeToCube_[slot]);
        }
    }

    // ── 常量：L1/L0 布局，与文件头布局表逐行对应；改一处要同步 host 侧 tiling 常量 ──
    static constexpr uint32_t L1_SLOT_ELEMS = CHUNK_SIZE_64 * DIM_128;
    static constexpr uint32_t L1_SLOT_BYTES = L1_SLOT_ELEMS * sizeof(DT);
    static constexpr uint32_t L1_NORM_OFFSET = 0;
    static constexpr uint32_t L1_STATE_OFFSET = 64 * 1024;
    static constexpr uint32_t L1_TOTAL_BYTES = L1_STATE_OFFSET + 2 * L1_SLOT_BYTES;
    static constexpr uint32_t L0A_BYTES = 32 * 1024;
    static constexpr uint32_t L0B_BYTES = 32 * 1024;
    static constexpr uint32_t L0C_BYTES = 128 * 1024;

    // ── 成员：GM / L1 / L0 / 事件 / 只读状态 ──
    AscendC::GlobalTensor<float> normWorkspaceGm_;
    AscendC::GlobalTensor<DT> stateWorkspaceGm_;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::A1> l1Buf_;
    AscendC::TBuf<AscendC::TPosition::A2> l0aBuf_;
    AscendC::TBuf<AscendC::TPosition::B2> l0bBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> l0cBuf_;
    AscendC::LocalTensor<DT> normL1_;
    AscendC::LocalTensor<DT> stateL1_;

    int32_t mte1ToMte2_[L1_SLOT_COUNT_2] = {0, 0};
    int32_t cubeMte1_[L1_SLOT_COUNT_2] = {0, 0};
    int32_t fixpipeToCube_[L1_SLOT_COUNT_2] = {0, 0};

    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const OpNameTilingData *tiling_ = nullptr;
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    int64_t chunkTaskNum_ = 0;
    int64_t taskRound_ = 0;
    uint32_t streamSlot_ = 0;
};

} // namespace OpsClassify

#endif // OP_NAME_CUBE_ARCH35_H
