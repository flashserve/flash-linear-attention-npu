/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_host/op_api/op_name.h
 *
 * 注意事项：
 *   1. 本文件是 L0 内部 exec 的接口（namespace l0op），不是公开接口；公开接口是 aclnn_<算子>.h。
 *   2. 返回值用固定长度的 std::array<const aclTensor *, N>：N 必须等于 def 的输出个数，
 *      顺序也必须一致（L2 按槽位取结果）。不要用 vector、不要按档位改变返回长度。
 *   3. 可选输出在这里也以"槽位"存在：L2 传 nullptr 时由 L0 换成一个零元素 descriptor，
 *      保证 launcher 形参不发生压缩错位（见 op_name.cpp 的注意事项）。
 *   4. outputMode 由 L2 传入，是编译期档位的选择依据；L0 不自己判断"调用方要不要"。
 *   5. 依赖的其它算子 L0（组合形态）通过 include 其 op_host/op_api/<依赖算子>.h 直接调用。
 */

#ifndef OP_API_INC_LEVEL0_OP_NAME_H
#define OP_API_INC_LEVEL0_OP_NAME_H

#include <array>
#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

using OpNameOutputs = std::array<const aclTensor *, 3>;

OpNameOutputs OpName(
    const aclTensor *x,
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *stateOut,
    const aclTensor *xNormOut,
    int64_t outputMode,
    aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_NAME_H
