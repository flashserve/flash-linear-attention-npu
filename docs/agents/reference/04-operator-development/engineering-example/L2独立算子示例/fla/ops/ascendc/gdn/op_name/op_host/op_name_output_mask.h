/**
 * 示例文件：fla/ops/ascendc/gdn/op_name/op_host/op_name_output_mask.h
 *
 * 注意事项：
 *   1. 只要有"多档可选输出"就必须有本文件：档位与掩码是 L2、tiling、kernel 三方共用的唯一事实，
 *      不允许在 L2 里散写位运算、在 tiling 里另算一套。
 *   2. 每个输出槽位一个 bit，必选与可选互斥，且两组合并必须覆盖全部槽位——用 static_assert 卡住。
 *   3. 档位用语义命名（none/save/...），不要用位组合当档位名。
 *   4. 位宽要能容纳全部槽位；槽位超过 32 个时改用 uint64_t 并同步 L2 与 kernel。
 *   5. 档位到掩码的映射由 L2 依据"哪些输出指针非空"推导（见 op_api/aclnn_op_name.cpp）。
 */

#ifndef OP_NAME_OUTPUT_MASK_H
#define OP_NAME_OUTPUT_MASK_H

#include <cstdint>

namespace optiling {

// 输出槽位：顺序必须与 *_def.cpp 的 Output 声明顺序一致，kernel 形参顺序也照此。
enum OpNameOutputSlot : uint32_t {
    OP_NAME_OUTPUT_Y = 0,
    OP_NAME_OUTPUT_STATE,
    OP_NAME_OUTPUT_X_NORM,
    OP_NAME_OUTPUT_COUNT,
};

enum OpNameOutputMask : uint32_t {
    OP_NAME_MASK_Y = 1U << OP_NAME_OUTPUT_Y,
    OP_NAME_MASK_STATE = 1U << OP_NAME_OUTPUT_STATE,
    OP_NAME_MASK_X_NORM = 1U << OP_NAME_OUTPUT_X_NORM,
};

// 必选：任何档位都要写出。
constexpr uint32_t OP_NAME_REQUIRED_OUTPUT_MASK = OP_NAME_MASK_Y;
// 可选：只在 save 档导出（反向重计算需要）。
constexpr uint32_t OP_NAME_OPTIONAL_OUTPUT_MASK =
    OP_NAME_MASK_STATE | OP_NAME_MASK_X_NORM;

// 档位（与 def 的 output_mode 属性取值一致，L2 负责换算）。
enum OpNameOutputMode : uint32_t {
    OP_NAME_OUTPUT_MODE_NONE = 0,
    OP_NAME_OUTPUT_MODE_SAVE = 1,
};

constexpr uint32_t OP_NAME_NONE_OUTPUT_MASK = OP_NAME_REQUIRED_OUTPUT_MASK;
constexpr uint32_t OP_NAME_SAVE_OUTPUT_MASK =
    OP_NAME_REQUIRED_OUTPUT_MASK | OP_NAME_OPTIONAL_OUTPUT_MASK;

static_assert(OP_NAME_OUTPUT_COUNT < sizeof(uint32_t) * 8U,
              "output_mask 必须能容纳全部输出槽位。");
static_assert((OP_NAME_REQUIRED_OUTPUT_MASK & OP_NAME_OPTIONAL_OUTPUT_MASK) == 0,
              "必选输出与可选输出的掩码不能重叠。");
static_assert((OP_NAME_REQUIRED_OUTPUT_MASK | OP_NAME_OPTIONAL_OUTPUT_MASK) ==
                  ((1U << OP_NAME_OUTPUT_COUNT) - 1U),
              "output_mask 必须覆盖全部输出槽位。");

} // namespace optiling

#endif // OP_NAME_OUTPUT_MASK_H
