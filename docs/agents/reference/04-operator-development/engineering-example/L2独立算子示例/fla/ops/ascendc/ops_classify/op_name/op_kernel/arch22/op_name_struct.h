/**
 * 示例文件：.../op_kernel/arch22/op_name_struct.h（A2/A3）
 *
 * 结构参考：finalize/op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_struct.h
 *
 * 注意事项：
 *   1. TilingData 按语义分组并逐组注释（逻辑 shape / 分核 / 档位），字段顺序与类型必须与
 *      op_host/op_name_tiling.h 的 TILING_DATA_FIELD_DEF 逐项一致；即使当前只支持固定值
 *      （如 D=128），也作为运行时 tiling 数据传入，不要用编译期常量替代。
 *   2. 模板参数与实例枚举本示例统一放 `../op_name_tiling_key.h`（本仓新规范）；参考算子放在本文件，
 *      两者等价，但同一算子只能有一份，避免 key 与结构体漂移。
 *   3. 平台相关资源常量写在本文件：语义化命名并带数值后缀（参考 `WORKSPACE_BUFFER_COUNT_8`），
 *      便于与实现、host 侧 arch tiling 常量交叉核对。
 */

#ifndef OP_NAME_STRUCT_ARCH22_H
#define OP_NAME_STRUCT_ARCH22_H

#include <cstdint>

#include "../op_name_tiling_key.h"

namespace OpsClassify {

// 尺寸常量：带数值后缀，便于和 shape/规格交叉核对（平台无关，但按"本平台唯一事实来源"放在这里）。
constexpr int64_t CHUNK_SIZE_64 = 64;
constexpr int64_t DIM_128 = 128;
constexpr int64_t UB_ALIGN_BYTES = 32;

// 平台资源：A2/A3 的 UB 较小，tile 取保守值；份数与 op_host/op_tiling/arch22 的常量一致。
constexpr int64_t AIV_COUNT_2 = 2;
constexpr int64_t TILE_T_ARCH22 = 32;
constexpr int64_t UB_SLOT_COUNT_2 = 2;
constexpr int64_t L1_SLOT_COUNT_2 = 2;

// UB 总量：只在这里给上限，逐段偏移在 arch22/op_name_vec.h 的常量块里（与文件头布局表对应）；
// tile 比 arch35 小，导出暂存也随之下移。改动任一段偏移后必须回头核对本值。
constexpr int64_t UB_TOTAL_BYTES = 84 * 1024;    // = XNORM_OFFSET 80 KiB + 256 B 导出暂存

struct OpNameTilingData {
    // 逻辑 shape 与 chunk 配置。tiling 侧固定校验 D=128、chunkSize∈{64,128}。
    uint32_t batch;
    uint32_t seqLen;
    uint32_t headNum;
    uint32_t dim;
    uint32_t chunkSize;
    uint32_t chunksPerSequence;
    // 任务数与分核：taskNum = chunksPerSequence × batch（varlen 时取 chunk_indices 行数）；
    // AIC 与 AIV 用同一个 taskNum 与同一套 stride，保证两角色看到同样的任务集合。
    uint32_t taskNum;
    // 分核：任务按 head 分核，每核 chunk 顺序执行（同一序列不被拆到多个核）。
    uint32_t usedCoreNum;
    uint32_t headsPerCore;
    // 档位与模式：outputMode 由 L2 的输出指针组合推导；isVarLen/hasInitialState 在运行时选分支。
    uint32_t outputMode;
    float scale;
    float epsilon;
    bool isVarLen;
    bool hasInitialState;
};

} // namespace OpsClassify

#endif // OP_NAME_STRUCT_ARCH22_H
