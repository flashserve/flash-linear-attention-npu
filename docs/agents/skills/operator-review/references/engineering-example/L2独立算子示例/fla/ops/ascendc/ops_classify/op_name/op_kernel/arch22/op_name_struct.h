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
constexpr int64_t L0_BUFFER_COUNT_2 = 2;   // L0A/L0B/L0C 各自 ping/pong，使用一次取反一次

// ── UB 布局常量（字节）：与 arch22/op_name_vec.h 的文件头布局表逐行对应 ──
// 函数不放进类里，布局常量落在本文件（平台唯一事实来源）；名字与 arch35 完全同名，
// 只差 tile 数值，便于两平台逐行对照。尺寸按 BF16/FP16（2 字节）与 FP32（4 字节）给出。
constexpr int64_t VEC_TILE_ELEMS = TILE_T_ARCH22 * DIM_128;
constexpr int64_t VEC_X_BYTES = VEC_TILE_ELEMS * 2;
constexpr int64_t VEC_G_BYTES = CHUNK_SIZE_64 * 2;
constexpr int64_t VEC_F32_BYTES = CHUNK_SIZE_64 * 4;
constexpr int64_t UB_X_OFFSET = 0;                    // x slot0 / slot1
constexpr int64_t UB_G_OFFSET = 16 * 1024;            // g slot0 / slot1
constexpr int64_t UB_NORM_OFFSET = 24 * 1024;         // norm slot0 / slot1（S2 复用为 scan）
constexpr int64_t UB_SCAN_OFFSET = 28 * 1024;         // scan slot0 / slot1
constexpr int64_t UB_Y_OFFSET = 32 * 1024;            // y 输出 slot0 / slot1
constexpr int64_t UB_STATE_OFFSET = 48 * 1024;        // state 导出 slot0 / slot1（仅 save 档）
constexpr int64_t UB_XNORM_OFFSET = 80 * 1024;        // x_norm 导出暂存（仅 save 档）
constexpr int64_t UB_TOTAL_BYTES = UB_XNORM_OFFSET + VEC_F32_BYTES;

// ── L1 / L0 布局常量：与 arch22/op_name_cube.h 的文件头布局表逐行对应，数值与 arch35 一致 ──
constexpr int64_t L1_TILE_ELEMS = CHUNK_SIZE_64 * DIM_128;
constexpr int64_t L1_TILE_BYTES = L1_TILE_ELEMS * 2;
constexpr int64_t L1_NORM_OFFSET = 0;
constexpr int64_t L1_STATE_OFFSET = 64 * 1024;
constexpr int64_t L1_TOTAL_BYTES = L1_STATE_OFFSET + 2 * L1_TILE_BYTES;
constexpr int64_t L0A_BYTES = 32 * 1024;
constexpr int64_t L0B_BYTES = 32 * 1024;
constexpr int64_t L0C_BYTES = 128 * 1024;

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
