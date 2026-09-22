/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_host/arch35/example_scan_tiling_impl.h
 *
 * 注意事项：
 *   1. arch35 = A5（ascend950）。平台差异只体现在常量与 tile 选择，公开原型、TilingKey、输出契约
 *      与 arch22 完全一致。
 *   2. 本目录的实现要与 op_kernel/arch35 的同名文件配套：host 说能放 N 份，kernel 就必须只申请 N 份。
 *   3. 更大的 tile 必须同时核对 UB/L1 容量与对齐 padding；只按元素数乘 dtype 大小会低估占用。
 *   4. 只有 A5 需要的编译期开关（例如 SSBUF 通路）写在 op_host/CMakeLists.txt 的平台分支与 kernel 里，
 *      不要用运行期宏在这里分叉。
 */

#ifndef EXAMPLE_SCAN_TILING_IMPL_ARCH35_H
#define EXAMPLE_SCAN_TILING_IMPL_ARCH35_H

#include <cstdint>

namespace optiling {

// A5：UB/L1 更大，tile 可以翻倍；份数仍为 2 份 ping/pong。
constexpr uint32_t EXAMPLE_SCAN_ARCH35_TILE_T = 64;
constexpr uint32_t EXAMPLE_SCAN_ARCH35_TILE_D = 128;
constexpr uint32_t EXAMPLE_SCAN_ARCH35_UB_SLOT_COUNT = 2;
constexpr uint32_t EXAMPLE_SCAN_ARCH35_L1_SLOT_COUNT = 2;
constexpr uint64_t EXAMPLE_SCAN_ARCH35_STATE_SLOT_BYTES = 0;  // 按实际 head 数计算

} // namespace optiling

#endif // EXAMPLE_SCAN_TILING_IMPL_ARCH35_H
