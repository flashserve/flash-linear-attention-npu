/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_kernel/example_scan_struct.h
 *
 * 注意事项：
 *   1. 本文件只放设备侧需要的 TilingData 结构、args 结构与常量；模板参数与实例枚举在
 *      <算子>_tiling_key.h（历史算子把两者合并在 _struct.h，新算子不再这么做）。
 *   2. TilingData 的字段顺序/类型/数量必须与 op_host/<算子>_tiling.h 的 BEGIN_TILING_DATA_DEF
 *      完全一致：设备侧用 GET_TILING_DATA_WITH_STRUCT 按同一布局解析，改字段就是改 ABI。
 *   3. 常量集中在这里（tile 上限、对齐、slot 数），不要在实现头里散布魔法数字；
 *      平台相关常量按 arch 目录分开，不要在这里用 #if 分叉。
 *   4. args 结构只承载地址与标量，不放计算中间状态。
 *   5. `nodiscard`/对齐等编译期属性要与 host 侧认知一致，否则设备侧读取会静默错位。
 */

#ifndef EXAMPLE_SCAN_STRUCT_H
#define EXAMPLE_SCAN_STRUCT_H

#include <cstdint>

// 必须与 op_host/example_scan_tiling.h 的 TilingData 字段一一对应（顺序、类型、数量）。
struct ExampleScanTilingData {
    uint32_t batch;
    uint32_t seqLen;
    uint32_t headNum;
    uint32_t dim;
    uint32_t chunkSize;
    uint32_t chunksPerSequence;
    uint32_t usedCoreNum;
    uint32_t headsPerCore;
    uint32_t outputMode;
    float scale;
    float epsilon;
    bool isVarLen;
    bool hasInitialState;
};

struct ExampleScanArgs {
    GM_ADDR x = nullptr;
    GM_ADDR g = nullptr;
    GM_ADDR aLog = nullptr;
    GM_ADDR initialState = nullptr;
    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;
    GM_ADDR y = nullptr;
    GM_ADDR state = nullptr;
    GM_ADDR xNorm = nullptr;
    GM_ADDR userWorkspace = nullptr;
    ExampleScanTilingData tiling{};
};

// 设备侧常量：与 host 侧 tiling 的假设保持一致。
constexpr uint32_t EXAMPLE_SCAN_DIM = 128;
constexpr uint32_t EXAMPLE_SCAN_DEFAULT_CHUNK = 64;

#endif // EXAMPLE_SCAN_STRUCT_H
