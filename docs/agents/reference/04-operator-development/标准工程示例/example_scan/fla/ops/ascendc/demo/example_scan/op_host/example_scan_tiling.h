/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_host/example_scan_tiling.h
 *
 * 注意事项：
 *   1. TilingData 的字段顺序、类型、数量必须与设备侧读取完全一致；设备侧用
 *      GET_TILING_DATA_WITH_STRUCT 按同一结构体解析，改字段就是改 ABI。
 *   2. REGISTER_TILING_DATA_CLASS 的 op 名（PascalCase）必须与 *_def.cpp 的 OP_ADD 一致。
 *   3. 输入/属性索引枚举放在这里，tiling 与 host UT 共用，禁止在 .cpp 里写裸下标。
 *   4. 本头文件只放数据结构与枚举，不放校验逻辑；校验放 *_tiling.cpp，计算放 *_tiling_processor.h。
 *   5. 平台相关常量不要写在这里，放 op_host/arch22|arch35/<算子>_tiling_impl.h。
 */

#ifndef EXAMPLE_SCAN_TILING_H
#define EXAMPLE_SCAN_TILING_H

#include <cstddef>

#include "example_scan_output_mask.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ExampleScanTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, seqNum);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, headNum);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, chunkSize);
TILING_DATA_FIELD_DEF(uint32_t, chunksPerSequence);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, headsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, outputMode);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(bool, isVarLen);
TILING_DATA_FIELD_DEF(bool, hasInitialState);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ExampleScan, ExampleScanTilingData)

enum ExampleScanInputIndex : size_t {
    EXAMPLE_SCAN_INPUT_X = 0,
    EXAMPLE_SCAN_INPUT_G,
    EXAMPLE_SCAN_INPUT_A_LOG,
    EXAMPLE_SCAN_INPUT_INITIAL_STATE,
    EXAMPLE_SCAN_INPUT_CU_SEQLENS,
    EXAMPLE_SCAN_INPUT_CHUNK_INDICES,
};

enum ExampleScanAttrIndex : size_t {
    EXAMPLE_SCAN_ATTR_LAYOUT = 0,
    EXAMPLE_SCAN_ATTR_SCALE,
    EXAMPLE_SCAN_ATTR_CHUNK_SIZE,
    EXAMPLE_SCAN_ATTR_EPSILON,
    EXAMPLE_SCAN_ATTR_OUTPUT_MODE,
};

struct ExampleScanCompileInfo {};

} // namespace optiling

#endif // EXAMPLE_SCAN_TILING_H
