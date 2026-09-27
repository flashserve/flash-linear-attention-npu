/**
 * 示例文件：.../op_kernel/arch22/op_name_cube.h（A2/A3 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *
 * 与 arch35 版本同骨架：Stage 表 → 常量 → 类（Init/Process）→ 阶段函数 → 收尾函数。
 * 平台差异只允许出现在下面这张表里，其余行必须与 arch35 逐行对应。
 *
 * ── 平台差异表（与 arch35 逐条对照）──────────────────────────────────────────
 *   项             arch22（A2/A3）                arch35（A5）
 *   L1 偏移        L1_*_OFFSET（arch22 常量）     L1_*_OFFSET（arch35 常量）
 *   L0C 大小       L0C_BYTES                     同 arch22（Cube 规格一致）
 *   Fixpipe 配置   与 arch35 同语义              可按 arch35 的 FixpipeConfig 指定 NZ/L1
 *   事件集合       同 arch35                       同 arch22
 *   同步协议       common.h 具名 flag              同 arch22
 *
 * ── Stage 表 / L1+L0 布局表 / 同步协议表 ─────────────────────────────────────
 *   与 arch35 版本一致：S1a CopyIn → S1b Matmul → S1c Fixpipe，Fixpipe 后广播给 AIV。
 */

#ifndef OP_NAME_CUBE_ARCH22_H
#define OP_NAME_CUBE_ARCH22_H

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
        // ① GM 接线：AIC 只读 workspace 与 tiling。
        normWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(normWorkspace));
        stateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(stateWorkspace));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;

        // ② 只读状态
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
        chunkTaskNum_ = static_cast<int64_t>(tiling_->taskNum);

        // ③ L1/L0 资源划分，与 arch35 同一张布局表（数值按 arch22 常量）。
        pipe_->InitBuffer(l1Buf_, L1_TOTAL_BYTES);
        normL1_ = l1Buf_.GetWithOffset<DT>(L1_SLOT_ELEMS, L1_NORM_OFFSET);
        stateL1_ = l1Buf_.GetWithOffset<DT>(2 * L1_SLOT_ELEMS, L1_STATE_OFFSET);
        pipe_->InitBuffer(l0aBuf_, L0A_BYTES);
        pipe_->InitBuffer(l0bBuf_, L0B_BYTES);
        pipe_->InitBuffer(l0cBuf_, L0C_BYTES);

        // ④ 核内事件：每 slot 一组，首轮先预置（与 arch35 相同）。
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
    // S1 输入：normWorkspace（GM，AIV 生产）；输出：stateWorkspace（GM）
    __aicore__ inline void Stage1MatrixChunk(const ChunkInfo &chunk)
    {
        const uint32_t slot = streamSlot_;
        AscendC::CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG);
        Stage1CopyIn(chunk, slot);
        Stage1Matmul(chunk, slot);
        Stage1FixpipeOut(chunk, slot);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_TO_VEC_READY_FLAG);
        streamSlot_ ^= 1U;
    }

    // S1a：MTE2 搬 norm 到 L1
    __aicore__ inline void Stage1CopyIn(const ChunkInfo &chunk, uint32_t slot)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(mte1ToMte2_[slot]);
        AscendC::DataCopy(normL1_, normWorkspaceGm_[GetWorkspaceChunkOffset(
                                       coreIdx_, taskRound_, chunk.chunkIdx)],
                          chunk.chunkLen * DIM_128);
    }

    // S1b：L1→L0 → Cube → L0C
    __aicore__ inline void Stage1Matmul(const ChunkInfo &chunk, uint32_t slot)
    {
        (void)chunk;
        // ... 与 arch35 相同的三段（MTE2→MTE1 / MTE1→M / M→MTE1 复用）...
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(cubeMte1_[slot]);
    }

    // S1c：Fixpipe 落 stateWorkspace；save 档才额外落公开 state
    __aicore__ inline void Stage1FixpipeOut(const ChunkInfo &chunk, uint32_t slot)
    {
        (void)chunk;
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(fixpipeToCube_[slot]);
        if constexpr (OUTPUT_MODE == OP_NAME_TPL_OUTPUT_SAVE) {
            // ... 与 arch35 相同的 stateWorkspace 写回语义 ...
        }
    }

    // 收尾：与 arch35 完全一致的顺序
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

    // ── 常量：L1/L0 布局，同名同序，数值按 arch22 ──
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

#endif // OP_NAME_CUBE_ARCH22_H
