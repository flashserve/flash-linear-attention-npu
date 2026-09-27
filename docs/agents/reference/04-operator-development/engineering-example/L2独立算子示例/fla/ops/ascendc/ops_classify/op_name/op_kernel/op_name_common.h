/**
 * 示例文件：.../op_kernel/op_name_common.h（平台无关，放在 op_kernel 根目录一份）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_common.h
 *
 * 注意事项（可读性结构）：
 *   1. 常量集中在这里并语义化命名：尺寸常量带数值后缀（`CHUNK_SIZE_64`、`DIM_128`），
 *      workspace 尺寸用"由什么组成"表达（`WORKSPACE_CHUNK_COUNT = 每核 chunk 上限 × 窗口数`）；
 *      实现里不要再出现裸数字。
 *   2. 跨核/跨 pipe 的同步协议写在文件顶部：flag 数量、方向、复用规则与背压来源一次说清；
 *      各 Stage 只引用这里的具名常量，不写魔法 flagId。
 *   3. 把"任务 -> 逻辑位置"的换算封成一个函数（`GetChunkInfo`）：定长/变长分支只在这里出现，
 *      各 Stage 只消费 `ChunkInfo`，不要各自再算一遍 offset。
 *   4. 本文件平台无关（arch22 与 arch35 共用），因此只放常量与纯函数：不申请 buffer、不写同步、不碰计算；
 *      平台差异分别落在 `archXX/<算子>_struct.h`（资源常量）与 `archXX/<算子>_{cube,vec}.h`（实现）。
 */

#ifndef OP_NAME_COMMON_H
#define OP_NAME_COMMON_H

#include "kernel_operator.h"

#include "op_name_struct.h"

namespace OpsClassify {

// 尺寸常量（`CHUNK_SIZE_64`、`DIM_128`、`UB_ALIGN_BYTES`）与平台资源常量定义在
// `archXX/<算子>_struct.h`：那里是本平台唯一的事实来源，common.h 只消费不重复定义。

// ---- workspace 尺寸：用组成关系表达，不要写 32768 这类推导结果 ----
constexpr int64_t WORKSPACE_CHUNKS_PER_CORE_16 = 16;
constexpr int64_t WORKSPACE_WINDOW_COUNT_2 = 2;
constexpr int64_t WORKSPACE_CHUNK_COUNT =
    WORKSPACE_CHUNKS_PER_CORE_16 * WORKSPACE_WINDOW_COUNT_2;
constexpr int64_t WORKSPACE_VECTOR_ELEMS = CHUNK_SIZE_64 * DIM_128;

// ---- 核间同步协议 ----
// 两个方向各合并成一条有序链、每个方向只用一个业务 ready id：
//   - VEC_TO_CUBE_READY_FLAG：AIV 完成某轮 Stage 后通知 AIC 搬入该轮结果；
//   - CUBE_TO_VEC_READY_FLAG：AIC 完成该轮矩阵后通知 AIV 继续下一轮。
// 两个方向互为背压：生产者必须先等到消费者释放上一轮，未消费计数才不会超过硬件上限。
// 需要新增同步方向时在这里登记具名常量，并同步更新生产者/消费者两处。
constexpr uint64_t VEC_TO_CUBE_READY_FLAG = 1;
constexpr uint64_t CUBE_TO_VEC_READY_FLAG = 3;

__aicore__ inline int64_t Min(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

// 任务对应的逻辑位置。定长模式的 tokenStart 是 batch 内偏移；变长模式已经是 packed T 上的全局偏移，
// 因此 bIdx 固定为 0（与 finalize 的 ChunkInfo 语义一致）。
struct ChunkInfo {
    int64_t bIdx = 0;
    int64_t tokenStart = 0;
    int64_t chunkLen = 0;
    int64_t chunkIdx = 0;
    bool valid = false;
};

// 把"任务序号 -> ChunkInfo"的换算封在这里：定长/变长两支只在本函数内出现，
// 各 Stage 只消费 ChunkInfo，不再自己解析 cu_seqlens/chunk_indices。
__aicore__ inline void GetChunkInfo(
    int64_t chunkTaskIdx, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const OpNameTilingData &tiling, ChunkInfo &info)
{
    info.valid = false;
    if (chunkTaskIdx < 0 || chunkTaskIdx >= tiling.chunksPerSequence) {
        return;
    }
    if (tiling.isVarLen) {
        if (cuSeqlens == nullptr || chunkIndices == nullptr) {
            return;
        }
        // 变长任务读取一组 [序列编号, 序列内 chunk 编号]，再换算成 packed token offset。
        AscendC::GlobalTensor<int64_t> cu;
        AscendC::GlobalTensor<int64_t> chunks;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        chunks.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
        const int64_t seqIdx = chunks.GetValue(chunkTaskIdx * 2);
        const int64_t chunkIdx = chunks.GetValue(chunkTaskIdx * 2 + 1);
        const int64_t seqStart = cu.GetValue(seqIdx);
        const int64_t seqEnd = cu.GetValue(seqIdx + 1);
        info.chunkIdx = chunkTaskIdx;
        info.tokenStart = seqStart + chunkIdx * tiling.chunkSize;
        info.chunkLen = Min(tiling.chunkSize, seqEnd - info.tokenStart);
        info.valid = seqStart >= 0 && seqEnd <= static_cast<int64_t>(tiling.seqLen) &&
            info.tokenStart >= seqStart && info.chunkLen > 0;
        return;
    }
    // 定长任务顺序为 [batch, chunk]；尾 chunk 保存真实 chunkLen，
    // 后续 Stage 只处理有效 token，不读 padding 区。
    info.bIdx = chunkTaskIdx / tiling.chunksPerSequence;
    info.chunkIdx = chunkTaskIdx % tiling.chunksPerSequence;
    info.tokenStart = info.chunkIdx * tiling.chunkSize;
    info.chunkLen = Min(tiling.chunkSize, static_cast<int64_t>(tiling.seqLen) - info.tokenStart);
    info.valid = info.chunkLen > 0;
}

// workspace 偏移也封成函数：调用点只表达"第几核、第几轮、第几个 chunk"，
// 不出现 `coreIdx * N + ...` 这类需要读者反推的算式。
__aicore__ inline int64_t GetWorkspaceChunkOffset(
    int64_t coreIdx, int64_t chunkRound, int64_t chunkOffset)
{
    const int64_t windowStart = (chunkRound & 1) * WORKSPACE_CHUNKS_PER_CORE_16;
    return (coreIdx * WORKSPACE_CHUNK_COUNT + windowStart + chunkOffset) * WORKSPACE_VECTOR_ELEMS;
}

} // namespace OpsClassify

#endif // OP_NAME_COMMON_H
