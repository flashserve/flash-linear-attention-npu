/**
 * 示例文件：.../op_kernel/arch22/op_name_vec.h（A2/A3 的 AIV 实现）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_vector.h
 *
 * 注意事项（可读性结构）：
 *   1. `Init(...)` 只做三件事：把 GM 地址 SetGlobalBuffer、保存 tiling 指针、按 slot 申请 UB
 *      （`pipe->InitBuffer`），并派生 coreIdx/subBlockIdx 等只读状态；初始化顺序与物理布局注释写在这里。
 *   2. UB 布局与 slot 轮转集中说明一次（每份多少字节、哪些 Stage 复用哪一份），
 *      Stage 内不再重算偏移。
 *   3. `Process()` 是任务主循环骨架：取 ChunkInfo -> 按 Stage 调用阶段函数；每个阶段一个函数，
 *      名称体现数据流（例如 `ProcessStage0Chunk`、`WriteBackStage2Chunk`）。
 *   4. 注释解释"为什么"：为什么某个寄存器/UB 区必须常驻、为什么这里不能合并 store/load、
 *      为什么某处只需要 V pipe 内的 `PipeBarrier`。
 *   5. 与 AIC 的交接只用 common.h 的具名 flag 与 workspace 协议，不依赖核启动顺序。
 */

#ifndef OP_NAME_VEC_ARCH22_H
#define OP_NAME_VEC_ARCH22_H

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
            // 只有带初始状态时才绑定 initialState/state，避免 read_state 为空时留下未使用地址。
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

        // UB 布局只在这里说明一次：每份 UB_SLOT_COUNT_2 的 x tile 大小固定，
        // S0 用 slot0 搬入、S2 复用 slot1 写回；两份都由 ping/pong 轮转，等价于 2 份物理存储。
        pipe_->InitBuffer(ubBuf_, UB_TOTAL_BYTES);
        x_[0] = ubBuf_.GetWithOffset<DT>(TILE_T_ARCH22 * DIM_128, X_OFFSET);
        x_[1] = ubBuf_.GetWithOffset<DT>(TILE_T_ARCH22 * DIM_128, X_OFFSET + X_BYTES);
        norm_[0] = ubBuf_.GetWithOffset<float>(TILE_T_ARCH22, NORM_OFFSET);
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
    // S0：计算归一化系数并写入 workspace，随后通知 AIC 该轮可消费（ready 方向）。
    __aicore__ inline void ProcessStage0Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        // MTE2 搬入 -> VEC 计算 norm -> MTE3 写 workspace；
        // VEC 写 UB 后由 MTE3 读，需要 V->MTE3 事件；同一 UB 内二次读才用 PipeBarrier<PIPE_V>()。
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(VEC_TO_CUBE_READY_FLAG);
    }

    // S2：写回 y；save 档额外导出 state 与 x_norm。档位是编译期模板参数，
    // 因此不做的搬出会被 `if constexpr` 整体裁掉，而不是运行期再判断一次。
    __aicore__ inline void WriteBackStage2Chunk(const ChunkInfo &chunk)
    {
        (void)chunk;
        if constexpr (OUTPUT_MODE == TPL_OUTPUT_SAVE) {
            // ... 从 stateWorkspaceGm_ 搬出 state、写 x_norm ...
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

#endif // OP_NAME_VEC_ARCH22_H
