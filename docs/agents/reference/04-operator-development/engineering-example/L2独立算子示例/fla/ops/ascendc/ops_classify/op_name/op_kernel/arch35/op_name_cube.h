/**
 * 示例文件：.../op_kernel/arch35/op_name_cube.h（A5 的 AIC 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h
 *
 * 注意事项（可读性结构）：
 *   1. 类名、`Init(...)`/`Process()` 接口、Stage 函数命名与 arch22 版本保持一致：
 *      平台差异只体现在资源常量、搬运粒度与流水，不改变代码骨架，便于两平台对照检视。
 *   2. 与 arch22 的差异必须能一句话说清（tile 更大 / L1 份数不同 / 新指令）；
 *      说不清差异的代码应上移到根目录共用，不要在两份实现里复制。
 *   3. 资源划分同样集中在 `Process()` 开头，份数与 arch35/<算子>_struct.h 的常量、
 *      host 侧 arch35 tiling 常量三者一致。
 *   4. 注释解释"为什么"：A5 上更依赖哪种流水、为什么某个 wait 可以省、哪块 buffer 必须常驻。
 *   5. 公开原型、TilingKey、输出契约与 arch22 完全一致，不因平台不同改变对外行为。
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
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, const OpNameTilingData *tiling)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(x));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(g));
        normWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(normWorkspace));
        stateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(stateWorkspace));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        chunkTaskNum_ = static_cast<int64_t>(tiling_->chunksPerSequence);
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
    }

    __aicore__ inline void Process()
    {
        // A5：tile 取 TILE_T_ARCH35（= arch22 的两倍），其余资源划分规则与 arch22 相同。
        ChunkInfo chunk;
        for (int64_t taskIdx = coreIdx_; taskIdx < chunkTaskNum_; taskIdx += coreNum_) {
            GetChunkInfo(taskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            ProcessStage0Chunk(chunk);
            ProcessStage2Chunk(chunk);
        }
    }

private:
    // S0：与 arch22 相同的 Stage 语义，差别在搬运粒度与事件流水。
    __aicore__ inline void ProcessStage0Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        AscendC::CrossCoreWaitFlag(VEC_TO_CUBE_READY_FLAG);
        // ... A5 流水：MTE2 -> MTE1 -> Cube -> Fixpipe，事件成对且与 arch22 命名一致 ...
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(CUBE_TO_VEC_READY_FLAG);
    }

    // S2：写回规则与 arch22 一致；save 档的额外搬出由编译期档位裁剪。
    __aicore__ inline void ProcessStage2Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        if constexpr (OUTPUT_MODE == TPL_OUTPUT_SAVE) {
            // ... 写 stateWorkspaceGm_，交给 AIV 导出 ...
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

#endif // OP_NAME_CUBE_ARCH35_H
