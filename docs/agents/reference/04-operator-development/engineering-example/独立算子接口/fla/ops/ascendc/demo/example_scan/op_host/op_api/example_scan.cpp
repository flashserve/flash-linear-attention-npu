/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_host/op_api/example_scan.cpp
 *
 * 注意事项：
 *   1. 本层只做四件事：把 aclIntArray 转成张量、把缺席的 REQUIRED 输出换成零元素 descriptor、
 *      调 ADD_TO_LAUNCHER_LIST_AICORE 下发、按 def 的顺序返回结果数组。
 *   2. **不要把 nullptr 直接交给 launcher**：def 的输出是 REQUIRED，部分 CANN 版本会压缩空参数，
 *      使后续 kernel 形参错位；缺席时用 executor->AllocTensor(MakeShape({0}), dtype, ND) 占位。
 *   3. 零元素 descriptor 只占位、不承载数据；是否真正搬出由编译期 outputMode 决定，两者必须一致。
 *   4. OP_INPUT/OP_OUTPUT/OP_ATTR 的顺序必须与 def 的声明顺序、kernel 形参顺序完全一致；
 *      任何一处顺序错位都会静默产生错误结果。
 *   5. 失败路径返回空数组 {}，由 L2 检查并转成 ACLNN_ERR_INNER_*；这里不返回部分结果。
 *   6. 本层不做布局能力判定、不做 dense 拷贝：非连续输入如实交给算子（见适配层设计 §3.6）。
 */

#include "example_scan.h"

#include <initializer_list>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ExampleScan);

namespace {

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

// 缺席的可选输出也要占住槽位：零元素 descriptor，不做数据搬出。
const aclTensor *OutputOrEmptyDescriptor(const aclTensor *output, DataType dtype,
                                        aclOpExecutor *executor)
{
    if (output != nullptr) {
        return output;
    }
    return executor->AllocTensor(MakeShape({0}), dtype, Format::FORMAT_ND);
}

const aclTensor *ConvertIntArrayToTensor(const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

ExampleScanOutputs ExampleScan(
    const aclTensor *x, const aclTensor *g, const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout, double scale,
    int64_t chunkSize, double epsilon, const aclTensor *yOut,
    const aclTensor *stateOut, const aclTensor *xNormOut, int64_t outputMode,
    aclOpExecutor *executor)
{
    const aclTensor *cuSeqlens = ConvertIntArrayToTensor(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices = ConvertIntArrayToTensor(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "转换 cu_seqlens/chunk_indices 到 Tensor 失败。");
        return {};
    }

    const aclTensor *stateForKernel =
        OutputOrEmptyDescriptor(stateOut, x->GetDataType(), executor);
    const aclTensor *xNormForKernel =
        OutputOrEmptyDescriptor(xNormOut, DataType::DT_FLOAT, executor);
    if (stateForKernel == nullptr || xNormForKernel == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "创建空输出 descriptor 失败。");
        return {};
    }

    const float scaleAttr = static_cast<float>(scale);
    const float epsilonAttr = static_cast<float>(epsilon);
    const auto status = ADD_TO_LAUNCHER_LIST_AICORE(
        ExampleScan,
        OP_INPUT(x, g, aLogOptional, initialStateOptional, cuSeqlens, chunkIndices),
        OP_OUTPUT(yOut, stateForKernel, xNormForKernel),
        OP_ATTR(layout, scaleAttr, chunkSize, epsilonAttr, outputMode));
    if (status != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "添加 ExampleScan AI Core 任务失败。");
        return {};
    }
    return {yOut, stateOut, xNormOut};
}

} // namespace l0op
