/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_host/arch22/example_scan_tiling_impl.h
 *
 * 注意事项：
 *   1. arch22 与 arch35 都可以有平台专用实现；本目录放 A2/A3（ascend910b / ascend910_93）的常量与
 *      tile 选择，arch35 目录放 A5（ascend950）。
 *   2. 只放"host 侧平台差异"：tile 形状、每核任务数、UB/L1 驻留份数、workspace 预留。
 *      两侧相同的校验、任务描述结构不要复制到这里。
 *   3. 常量命名带平台后缀（_ARCH22_），避免与 arch35 的同名宏冲突；宿主按 SoC 只 include 一份。
 *   4. 这里的份数必须与 op_kernel/arch22 下实际申请的资源一致；改一处必须改另一处，
 *      并用 host UT 或运行时记录给出实际选择证据。
 *   5. 平台判定统一用 platform.GetCurNpuArch()（A5 = NpuArch::DAV_3510），不要用 SoC 字符串比较。
 */

#ifndef EXAMPLE_SCAN_TILING_IMPL_ARCH22_H
#define EXAMPLE_SCAN_TILING_IMPL_ARCH22_H

#include <cstdint>

namespace optiling {

// A2/A3：UB 较小，tile 取保守值。
constexpr uint32_t EXAMPLE_SCAN_ARCH22_TILE_T = 32;
constexpr uint32_t EXAMPLE_SCAN_ARCH22_TILE_D = 128;
constexpr uint32_t EXAMPLE_SCAN_ARCH22_UB_SLOT_COUNT = 2;
constexpr uint32_t EXAMPLE_SCAN_ARCH22_L1_SLOT_COUNT = 2;
constexpr uint64_t EXAMPLE_SCAN_ARCH22_STATE_SLOT_BYTES = 0;  // 按实际 head 数计算

inline constexpr bool IsArch22Supported(uint32_t socArch) { return socArch != 0; }

} // namespace optiling

#endif // EXAMPLE_SCAN_TILING_IMPL_ARCH22_H
