/**
 * 示例文件：.../op_kernel/arch22/op_name_cube.h（A2/A3 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *
 * 注意事项（可读性结构）：
 *   1. 一个角色一个类，接口固定：`Init(...)` 只保存地址/指针并派生只读状态（coreIdx/coreNum/任务数），
 *      不做搬运与计算；`Process()` 是主循环骨架，按 Stage 顺序调用阶段函数。
 *   2. 每个 Stage 一个函数，函数名带 Stage 号与作用域（参考 `ProcessStage1Head`）；不要写成一个
 *      几百行的 Process。
 *   3. 资源划分集中在 `Process()` 开头：L1/L0 buffer 的偏移与份数只在这里出现，Stage 内只引用取好的
 *      `LocalTensor`；份数必须与 archXX/<算子>_struct.h 的常量一致。
 *   4. 注释解释"为什么"：跨 Stage 依赖、workspace 复用、事件复用与背压、物理布局重解释；
 *      不要写"这里搬入 x"这类复述代码的注释。
 *   5. 同步使用 common.h 的具名 flag，生产/消费两侧成对出现；跨 pipe 依赖用事件，
 *      只有同一 pipe 内前后依赖才用 `PipeBarrier`。
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
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, const OpNameTilingData *tiling)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(x));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(g));
        normWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(normWorkspace));
        stateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(stateWorkspace));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;

        // 只读派生量集中在这里；Stage 内不再重复取 blockIdx/blockNum 或重算任务数。
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        chunkTaskNum_ = static_cast<int64_t>(tiling_->chunksPerSequence);
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
    }

    __aicore__ inline void Process()
    {
        // 资源划分只写在 Process 开头：偏移与份数在这里一次确定，Stage 内只做消费。
        // 份数与 arch22/<算子>_struct.h 的 UB_SLOT_COUNT_2 / L1_SLOT_COUNT_2 一致。
        ChunkInfo chunk;
        for (int64_t taskIdx = coreIdx_; taskIdx < chunkTaskNum_; taskIdx += coreNum_) {
            // 任务 -> 逻辑位置只通过 GetChunkInfo 换算：定长/变长分支不在这里重复出现。
            GetChunkInfo(taskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            // Stage 顺序与设计文档一致：S0 消费 AIV 写入的 norm/scan 中间量 -> S2 完成矩阵部分。
            ProcessStage0Chunk(chunk);
            ProcessStage2Chunk(chunk);
        }
    }

private:
    // S0：读取本 chunk 的中间量并完成矩阵乘。等待 AIV 的 ready 后再搬入，
    // 消费完成后立即 set free，供 AIV 复用同一 workspace 窗口（反向同步，避免覆盖）。
    __aicore__ inline void ProcessStage0Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        AscendC::CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG);
        // ... MTE2 搬入 -> MTE1 进 L0A/L0B -> Cube -> Fixpipe 写 L0C ...
        // 每个跨 pipe 依赖成对 set/wait；复用 buffer 前必须等上一轮消费者释放。
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(CUBE_TO_VEC_READY_FLAG);
    }

    // S2：把本 chunk 的矩阵结果写回 GM/workspace。save 档才写出 state，否则只写 workspace，
    // 由 L2 决定是否导出——档位在编译期已裁掉不做的搬出（if constexpr）。
    __aicore__ inline void ProcessStage2Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        if constexpr (OUTPUT_MODE == TPL_OUTPUT_SAVE) {
            // ... 把 state 写入 stateWorkspaceGm_，供 AIV 导出 ...
        }
    }

    AscendC::GlobalTensor<DT> xGm_;
    AscendC::GlobalTensor<DT> gGm_;
    AscendC::GlobalTensor<DT> normWorkspaceGm_;
    AscendC::GlobalTensor<DT> stateWorkspaceGm_;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const OpNameTilingData *tiling_ = nullptr;
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    int64_t chunkTaskNum_ = 0;
};

} // namespace OpsClassify

#endif // OP_NAME_CUBE_ARCH22_H
