/**
 * 示例文件：.../op_kernel/arch35/op_name_struct.h（A5）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_struct.h
 *
 * 注意事项：
 *   1. 与 arch22 版本保持同一份 TilingData 字段顺序与类型（同一公开原型、同一 tiling 契约），
 *      差异只体现在平台资源常量上。
 *   2. 平台常量必须与 op_host/op_tiling/arch35 的 tiling 常量、以及本目录 cube/vec 的
 *      实际 buffer 申请一致；改一处要同步另外两处。
 *   3. TilingData/TPL 的归属规则见 arch22 版本第 2 条，两边保持一致。
 */

#ifndef OP_NAME_STRUCT_ARCH35_H
#define OP_NAME_STRUCT_ARCH35_H

#include <cstdint>

#include "../op_name_tiling_key.h"

namespace OpsClassify {

// 尺寸常量与 arch22 版本保持一致；平台差异只体现在下一条的资源常量上。
constexpr int64_t CHUNK_SIZE_64 = 64;
constexpr int64_t DIM_128 = 128;
constexpr int64_t UB_ALIGN_BYTES = 32;

// 平台资源：A5 的 UB/L1 更大，tile 可以翻倍；份数仍为 2 份 ping/pong。
constexpr int64_t AIV_COUNT_2 = 2;
constexpr int64_t TILE_T_ARCH35 = 64;
constexpr int64_t UB_SLOT_COUNT_2 = 2;
constexpr int64_t L1_SLOT_COUNT_2 = 2;

// UB 总量：只在这里给上限，逐段偏移在 arch35/op_name_vec.h 的常量块里（与文件头布局表对应）。
// 改动任一段偏移后，必须回头核对本值是否仍覆盖最后一段的尾部。
constexpr int64_t UB_TOTAL_BYTES = 132 * 1024;   // = XNORM_OFFSET 128 KiB + 256 B 导出暂存

struct OpNameTilingData {
    // 字段顺序与 arch22 版本完全一致，仅注释里标注的平台常量不同。
    // 逻辑 shape 与 chunk 配置：tiling 侧固定校验 D=128、chunkSize∈{64,128}。
    uint32_t batch;
    uint32_t seqLen;
    uint32_t headNum;
    uint32_t dim;
    uint32_t chunkSize;
    uint32_t chunksPerSequence;
    // 任务数与分核：taskNum = chunksPerSequence × batch（varlen 时取 chunk_indices 行数），
    // 两个角色都用它做 `for (taskIdx = coreIdx; taskIdx < taskNum; taskIdx += coreNum)`。
    uint32_t taskNum;
    uint32_t usedCoreNum;
    uint32_t headsPerCore;
    uint32_t outputMode;
    float scale;
    float epsilon;
    bool isVarLen;
    bool hasInitialState;
};

} // namespace OpsClassify

#endif // OP_NAME_STRUCT_ARCH35_H
