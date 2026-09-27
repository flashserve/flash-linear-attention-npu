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

// ── UB 布局常量（字节）：与 arch35/op_name_vec.h 的文件头布局表逐行对应 ──
// 函数不再放进类里，所以布局常量就落在本文件（平台唯一事实来源）；vec 只引用不重算。
// 尺寸按 BF16/FP16（2 字节）与 FP32（4 字节）给出；换 dtype 时同步核对整张表。
constexpr int64_t VEC_TILE_ELEMS = TILE_T_ARCH35 * DIM_128;
constexpr int64_t VEC_X_BYTES = VEC_TILE_ELEMS * 2;
constexpr int64_t VEC_G_BYTES = CHUNK_SIZE_64 * 2;
constexpr int64_t VEC_F32_BYTES = CHUNK_SIZE_64 * 4;
constexpr int64_t UB_X_OFFSET = 0;                    // x slot0 / slot1
constexpr int64_t UB_G_OFFSET = 32 * 1024;            // g slot0 / slot1
constexpr int64_t UB_NORM_OFFSET = 48 * 1024;         // norm slot0 / slot1（S2 复用为 scan）
constexpr int64_t UB_SCAN_OFFSET = 56 * 1024;         // scan slot0 / slot1
constexpr int64_t UB_Y_OFFSET = 64 * 1024;            // y 输出 slot0 / slot1
constexpr int64_t UB_STATE_OFFSET = 96 * 1024;        // state 导出 slot0 / slot1（仅 save 档）
constexpr int64_t UB_XNORM_OFFSET = 128 * 1024;       // x_norm 导出暂存（仅 save 档）
constexpr int64_t UB_TOTAL_BYTES = UB_XNORM_OFFSET + VEC_F32_BYTES;

// ── L1 / L0 布局常量（字节）：与 arch35/op_name_cube.h 的文件头布局表逐行对应 ──
constexpr int64_t L1_TILE_ELEMS = CHUNK_SIZE_64 * DIM_128;
constexpr int64_t L1_TILE_BYTES = L1_TILE_ELEMS * 2;
constexpr int64_t L1_NORM_OFFSET = 0;
constexpr int64_t L1_STATE_OFFSET = 64 * 1024;
constexpr int64_t L1_TOTAL_BYTES = L1_STATE_OFFSET + 2 * L1_TILE_BYTES;
constexpr int64_t L0A_BYTES = 32 * 1024;
constexpr int64_t L0B_BYTES = 32 * 1024;
constexpr int64_t L0C_BYTES = 128 * 1024;

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
