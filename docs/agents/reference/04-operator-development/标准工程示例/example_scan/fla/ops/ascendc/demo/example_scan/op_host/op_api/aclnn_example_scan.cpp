/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_host/op_api/aclnn_example_scan.cpp
 *
 * 注意事项：
 *   1. L2 只做：非空校验 -> 参数/档位校验 -> 输入连续化 -> 调 L0 -> 返回 workspaceSize。
 *      计算、分核、同步都不在这里。
 *   2. 档位由"哪些输出指针非空"推导（ResolveOutputMode），非法组合返回
 *      ACLNN_ERR_PARAM_INVALID 并打印实际组合，不能静默取默认档位。
 *   3. 报错文本要带实际值（布局、chunkSize、输出组合），这是用户唯一能看到的上下文。
 *   4. 每次调用都新建 UniqueExecutor 并把所有权交给调用方：GetWorkspaceSize 只登记任务，
 *      Launch 阶段才真正下发；不要在这里同步执行。
 *   5. 不做 dense 拷贝之外的布局改写：非连续输入交给 l0op::Contiguous，其余按 view 交给算子。
 *   6. 不在本层判断 disable_recompute 之类策略；只认输出指针。
 *   7. Launch 阶段失败返回 ACLNN_ERR_INNER，并给出算子名。
 */

#include "aclnn_example_scan.h"

#include "example_scan.h"

#include <cstring>

#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct ExampleScanParams {
    const aclTensor *x = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    double epsilon = 1e-6;
    const aclTensor *yOut = nullptr;
    const aclTensor *stateOut = nullptr;
    const aclTensor *xNormOut = nullptr;
};

// 档位只由输出指针组合推导；与 example_scan_output_mask.h 的档位定义一一对应。
int64_t ResolveOutputMode(const ExampleScanParams &params)
{
    const bool hasState = params.stateOut != nullptr;
    const bool hasXNorm = params.xNormOut != nullptr;
    if (!hasState && !hasXNorm) {
        return optiling::EXAMPLE_SCAN_OUTPUT_MODE_NONE;
    }
    if (hasState && hasXNorm) {
        return optiling::EXAMPLE_SCAN_OUTPUT_MODE_SAVE;
    }
    return -1;  // 只给一个可选输出：非法组合
}

aclnnStatus CheckParams(const ExampleScanParams &params)
{
    // 必选输入/输出：缺一个就返回 PARAM_NULLPTR，并点名是哪个。
    CHECK_COND(params.x != nullptr && params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "x 与 g 不能为 nullptr。");
    CHECK_COND(params.yOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "yOut 不能为 nullptr。");
    CHECK_COND(params.layout != nullptr, ACLNN_ERR_PARAM_NULLPTR, "layout 不能为 nullptr。");

    const int64_t outputMode = ResolveOutputMode(params);
    CHECK_COND(outputMode >= 0, ACLNN_ERR_PARAM_INVALID,
               "stateOut/xNormOut 必须同时给出或同时为空，当前 stateOut=%s xNormOut=%s。",
               params.stateOut ? "非空" : "空", params.xNormOut ? "非空" : "空");

    CHECK_COND(params.scale > 0.0, ACLNN_ERR_PARAM_INVALID,
               "scale 必须为正数，当前 scale=%f。", params.scale);
    CHECK_COND(params.epsilon > 0.0, ACLNN_ERR_PARAM_INVALID,
               "epsilon 必须为正数，当前 epsilon=%f。", params.epsilon);
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize 只支持 64/128，当前 chunkSize=%ld。",
               static_cast<long>(params.chunkSize));
    // D、cu_seqlens 单调性、chunk_indices 成对等约束：与 README「已知限制」逐条对应。
    // ...
    return ACLNN_SUCCESS;
}

aclnnStatus MakeInputsContiguous(ExampleScanParams &params, aclOpExecutor *executor)
{
    const aclTensor **inputs[] = {&params.x, &params.g, &params.aLogOptional,
                                  &params.initialStateOptional};
    for (const aclTensor **input : inputs) {
        if (*input == nullptr || IsContiguous(*input)) {
            continue;
        }
        *input = l0op::Contiguous(*input, executor);
        CHECK_RET(*input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnExampleScanGetWorkspaceSize(
    const aclTensor *x, const aclTensor *g, const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout, double scale,
    int64_t chunkSize, double epsilon, const aclTensor *yOut,
    const aclTensor *stateOut, const aclTensor *xNormOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnExampleScan,
                   DFX_IN(x, g, aLogOptional, initialStateOptional, layout, scale, chunkSize,
                          epsilon),
                   DFX_OUT(yOut, stateOut, xNormOut));
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize 不能为 nullptr。");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor 不能为 nullptr。");

    ExampleScanParams params{x, g, aLogOptional, initialStateOptional, cuSeqlensOptional,
                             chunkIndicesOptional, layout, scale, chunkSize, epsilon,
                             yOut, stateOut, xNormOut};
    const int64_t outputMode = ResolveOutputMode(params);
    aclnnStatus status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    status = MakeInputsContiguous(params, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const auto result = l0op::ExampleScan(
        params.x, params.g, params.aLogOptional, params.initialStateOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.layout, params.scale,
        params.chunkSize, params.epsilon, params.yOut, params.stateOut, params.xNormOut,
        outputMode, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnExampleScan(void *workspace, uint64_t workspaceSize,
                            aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnExampleScan);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ExampleScan AI Core 启动失败。");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
