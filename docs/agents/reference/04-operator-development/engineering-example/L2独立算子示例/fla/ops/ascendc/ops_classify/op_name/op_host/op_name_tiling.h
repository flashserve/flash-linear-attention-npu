/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_host/op_name_tiling.h
 *
 * 注意事项：
 *   1. TilingData 的字段顺序、类型、数量必须与设备侧读取完全一致；设备侧用
 *      GET_TILING_DATA_WITH_STRUCT 按同一结构体解析，改字段就是改 ABI。
 *   2. REGISTER_TILING_DATA_CLASS 的 op 名（PascalCase）必须与 *_def.cpp 的 OP_ADD 一致。
 *   3. 输入/属性索引枚举放在这里，tiling 与 host UT 共用，禁止在 .cpp 里写裸下标。
 *   4. 本头文件只放数据结构与枚举，不放校验逻辑；校验放 *_tiling.cpp，计算放 *_tiling_processor.h。
 *   5. 平台相关常量不要写在这里，放 op_host/arch22|arch35/<算子>_tiling_impl.h。
 */

#ifndef OP_NAME_TILING_H
#define OP_NAME_TILING_H

#include <cstddef>

#include "op_name_output_mask.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(OpNameTilingData)
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

REGISTER_TILING_DATA_CLASS(OpName, OpNameTilingData)

enum OpNameInputIndex : size_t {
    OP_NAME_INPUT_X = 0,
    OP_NAME_INPUT_G,
    OP_NAME_INPUT_A_LOG,
    OP_NAME_INPUT_INITIAL_STATE,
    OP_NAME_INPUT_CU_SEQLENS,
    OP_NAME_INPUT_CHUNK_INDICES,
};

enum OpNameAttrIndex : size_t {
    OP_NAME_ATTR_LAYOUT = 0,
    OP_NAME_ATTR_SCALE,
    OP_NAME_ATTR_CHUNK_SIZE,
    OP_NAME_ATTR_EPSILON,
    OP_NAME_ATTR_OUTPUT_MODE,
};

struct OpNameCompileInfo {};

} // namespace optiling

#endif // OP_NAME_TILING_H
