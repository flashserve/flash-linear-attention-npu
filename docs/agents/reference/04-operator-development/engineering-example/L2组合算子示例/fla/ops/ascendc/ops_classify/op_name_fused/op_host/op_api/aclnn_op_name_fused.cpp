/**
 * 示例文件：.../op_name_fused/op_host/op_api/aclnn_op_name_fused.cpp
 *
 * 注意事项：
 *   1. 组合入口的 L2 只做四件事：校验、按顺序调 L0、拼接公开输出、把 workspace 交给调用方。
 *      计算、分核、同步全部在被组合算子里，这里一行都不能重复实现。
 *   2. 依赖算子的 L0 头按相对路径 include（示例：`../../../op_name/op_host/op_api/op_name.h`）；
 *      这些头只声明 L0，不暴露 kernel，不会破坏分层。
 *   3. 所有子调用共用同一个 executor：一次 GetWorkspaceSize 登记全部任务，一次 launch 执行；
 *      不要中途 ReleaseTo 或另起 executor，否则 workspace 会算成两份且任务跨 launch 失去依赖保证。
 *   4. 中间张量用 executor 内部张量（`executor->AllocTensor(...)` 或子 L0 返回的 result），
 *      不对外暴露；公开输出如果布局不同，做一次明确 `l0op::ViewCopy` 并在设计文档里记代价。
 *   5. 档位由公开输出指针组合推导；非法组合（只给 state 或只给 xNorm）返回
 *      ACLNN_ERR_PARAM_INVALID 并打印实际组合。
 *   6. 支持范围不满足时返回 ACLNN_ERR_PARAM_INVALID 并提示回落入口，不要在本层静默改成单算子调用。
 *   7. 依赖算子缺失导致的失败要在报错里点名缺失的算子，便于定位"只编了本算子"。
 */

#include "aclnn_op_name_fused.h"

#include "../../../op_name/op_host/op_api/op_name.h"
#include "../../../op_name_tail/op_host/op_api/op_name_tail.h"

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

constexpr int64_t OP_NAME_FUSED_SUPPORTED_DIM = 128;
constexpr int64_t OP_NAME_FUSED_SUPPORTED_CHUNK = 64;

struct OpNameFusedParams {
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
    const aclTensor *tailOut = nullptr;
};

// 档位：none（只 y）/ save（y + state + x_norm）/ save_tail（再加 tail）。
enum class OpNameFusedMode : int64_t { kInvalid = -1, kNone = 0, kSave = 1, kSaveTail = 2 };

OpNameFusedMode ResolveMode(const OpNameFusedParams &params)
{
    const bool hasState = params.stateOut != nullptr;
    const bool hasXNorm = params.xNormOut != nullptr;
    const bool hasTail = params.tailOut != nullptr;
    if (hasState != hasXNorm) {
        return OpNameFusedMode::kInvalid;
    }
    if (!hasState) {
        return hasTail ? OpNameFusedMode::kInvalid : OpNameFusedMode::kNone;
    }
    return hasTail ? OpNameFusedMode::kSaveTail : OpNameFusedMode::kSave;
}

aclnnStatus CheckParams(const OpNameFusedParams &params)
{
    CHECK_COND(params.x != nullptr && params.g != nullptr && params.yOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "x/g/yOut 不能为 nullptr。");
    const OpNameFusedMode mode = ResolveMode(params);
    CHECK_COND(mode != OpNameFusedMode::kInvalid, ACLNN_ERR_PARAM_INVALID,
               "stateOut/xNormOut 必须同时给出或同时为空；tailOut 只能在 save 档给出。"
               "当前 stateOut=%s xNormOut=%s tailOut=%s。",
               params.stateOut ? "非空" : "空", params.xNormOut ? "非空" : "空",
               params.tailOut ? "非空" : "空");
    CHECK_COND(params.chunkSize == OP_NAME_FUSED_SUPPORTED_CHUNK, ACLNN_ERR_PARAM_INVALID,
               "本入口只支持 chunk_size=%ld，当前 chunk_size=%ld；其它取值请改用 op_name。",
               static_cast<long>(OP_NAME_FUSED_SUPPORTED_CHUNK),
               static_cast<long>(params.chunkSize));
    // D、dtype、连续性与变长元数据校验：与算子 README「已知限制」逐条对应。
    // ...
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnOpNameFusedGetWorkspaceSize(
    const aclTensor *x, const aclTensor *g, const aclTensor *aLogOptional,
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, const char *layout, double scale,
    int64_t chunkSize, double epsilon, const aclTensor *yOut, const aclTensor *stateOut,
    const aclTensor *xNormOut, const aclTensor *tailOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnOpNameFused,
                   DFX_IN(x, g, aLogOptional, initialStateOptional, layout, scale, chunkSize,
                          epsilon),
                   DFX_OUT(yOut, stateOut, xNormOut, tailOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize/executor 不能为 nullptr。");

    OpNameFusedParams params{x,          g,          aLogOptional,
                                  initialStateOptional, cuSeqlensOptional,
                                  chunkIndicesOptional, layout,     scale,
                                  chunkSize,  epsilon,    yOut,
                                  stateOut,   xNormOut,   tailOut};
    const OpNameFusedMode mode = ResolveMode(params);
    aclnnStatus status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    // 组合第一步：主实现。它产出 y（以及 save 档的 state/x_norm）。
    const auto scanOut = l0op::OpName(
        params.x, params.g, params.aLogOptional, params.initialStateOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.layout, params.scale,
        params.chunkSize, params.epsilon, params.yOut, params.stateOut, params.xNormOut,
        mode == OpNameFusedMode::kNone ? 0 : 1, executorPtr);
    CHECK_RET(scanOut[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 组合第二步：尾部归一化。只在 save+tail 档调用，中间结果用 executor 内部张量。
    if (mode == OpNameFusedMode::kSaveTail) {
        const auto tailOut = l0op::OpNameTail(scanOut[0], params.tailOut, executorPtr);
        CHECK_RET(tailOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnOpNameFused(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnOpNameFused);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "OpNameFused AI Core 启动失败。");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
