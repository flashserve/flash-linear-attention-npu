/**
 * 示例文件：.../op_kernel/arch35/op_name_vec.h（A5 的 AIV 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_vector.h
 *
 * 注意事项（可读性结构）：
 *   1. 与 arch22 版本保持同样的类名、`Init(...)`/`Process()` 接口与阶段函数命名；
 *      A5 的差异（更大的 UB、RegBase/VF 通路、指令粒度）写在实现里，不改骨架。
 *   2. UB 布局与 slot 轮转仍在 `Init(...)` 里一次说明清楚；份数取 arch35 常量。
 *   3. 使用 A5 专用通路时注释要写清收益与代价（为什么更快、牺牲了哪块 buffer 或哪个并行度），
 *      并给 profiling 证据；没有证据不要声称 pipelining/ping-pong 已实现。
 *   4. 首轮、尾块、空任务路径的同步计数与 arch22 一致，不允许"某平台跳过 free/ready"。
 *   5. 公开契约与 arch22 完全一致；平台宏只在本目录内使用，不污染根目录共用头。
 */

#ifndef OP_NAME_VEC_ARCH35_H
#define OP_NAME_VEC_ARCH35_H

#include "../op_name_common.h"
#include "../op_name_struct.h"

namespace OpsClassify {

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
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(x));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(g));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(y));
        normWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(normWorkspace));
        if constexpr (USE_STATE) {
            stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(state));
        }
        xNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(xNorm));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        subBlockIdx_ = static_cast<int64_t>(AscendC::GetSubBlockIdx());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }

        // A5 的 UB 更大：tile 取 TILE_T_ARCH35，slot 数仍为 UB_SLOT_COUNT_2。
        pipe_->InitBuffer(ubBuf_, UB_TOTAL_BYTES);
        x_[0] = ubBuf_.GetWithOffset<DT>(TILE_T_ARCH35 * DIM_128, X_OFFSET);
        x_[1] = ubBuf_.GetWithOffset<DT>(TILE_T_ARCH35 * DIM_128, X_OFFSET + X_BYTES);
        norm_[0] = ubBuf_.GetWithOffset<float>(TILE_T_ARCH35, NORM_OFFSET);
    }

    __aicore__ inline void Process()
    {
        ChunkInfo chunk;
        for (int64_t taskIdx = coreIdx_; taskIdx < static_cast<int64_t>(tiling_->chunksPerSequence);
             taskIdx += coreNum_) {
            GetChunkInfo(taskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            ProcessStage0Chunk(chunk);
            WriteBackStage2Chunk(chunk);
        }
    }

private:
    // S0：Stage 语义与 arch22 相同；A5 上可以合并更多的向量计算（RegBase/VF），
    // 但 ready/free 的配对关系不变。
    __aicore__ inline void ProcessStage0Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(VEC_TO_CUBE_READY_FLAG);
    }

    __aicore__ inline void WriteBackStage2Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        if constexpr (OUTPUT_MODE == TPL_OUTPUT_SAVE) {
            // ... 与 arch22 相同的导出语义 ...
        }
    }

    AscendC::GlobalTensor<DT> xGm_;
    AscendC::GlobalTensor<GT> gGm_;
    AscendC::GlobalTensor<DT> yGm_;
    AscendC::GlobalTensor<DT> stateGm_;
    AscendC::GlobalTensor<float> xNormGm_;
    AscendC::GlobalTensor<DT> normWorkspaceGm_;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const OpNameTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<DT> x_[UB_SLOT_COUNT_2];
    AscendC::LocalTensor<float> norm_[UB_SLOT_COUNT_2];
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    int64_t subBlockIdx_ = 0;
};

} // namespace OpsClassify

#endif // OP_NAME_VEC_ARCH35_H
