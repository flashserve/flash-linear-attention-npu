/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
// Load structured reporters before opdev's legacy logging macros. The opdev
// headers must retain their logging convention for CHECK_RET and DFX helpers.
#include "op_common/log/log.h"
#include "aclnn_chunk_local_cumsum.h"
#include <new>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkLocalCumsum";

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkLocalCumsumParams {
    const aclTensor *g = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOutOptional = nullptr;
    int64_t chunkSize = 0;
    bool reverse = false;
    double scale = 1.0;
    bool headFirst = true;
    const char *outputDtypeOptional = "float32";
    const aclTensor *out = nullptr;
};

static aclnnStatus CheckNotNull(ChunkLocalCumsumParams params)
{
    if (params.g == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.out == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "out");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.outputDtypeOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "output_dtype");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkLocalCumsumParams params)
{
    {
        const aclnnStatus checkRet = CheckNotNull(params);
        if (checkRet != ACLNN_SUCCESS) {
            return checkRet;
        }
    }
    CHECK_COND(params.chunkSize > 0, ACLNN_ERR_PARAM_INVALID, "chunkSize must be positive.");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkLocalCumsumParams &params, aclOpExecutor *executorPtr)
{
    if (DataContiguous(params.g, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(g) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkLocalCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOutOptional,
    int64_t chunkSize,
    bool reverse,
    double scale,
    bool headFirst,
    char *outputDtypeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    const char *outputDtype = outputDtypeOptional == nullptr ? "float32" : outputDtypeOptional;
    ChunkLocalCumsumParams params{
        g, cuSeqlensOptional, chunkIndicesOutOptional, chunkSize, reverse, scale, headFirst, outputDtype, out};

    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnChunkLocalCumsum,
                   DFX_IN(g, cuSeqlensOptional, chunkIndicesOutOptional, chunkSize, reverse, scale, headFirst,
                          outputDtypeOptional),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_chunk_local_cumsum",
            "npu_chunk_local_cumsum is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused npu_chunk_gated_delta_rule_fwd instead.");

    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    if (ParamsDataContiguous(params, executorPtr) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "contiguous",
            "Contiguous(conversion) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto result = l0op::ChunkLocalCumsum(params.g, params.cuSeqlensOptional, params.chunkIndicesOutOptional,
                                        params.chunkSize, params.reverse, params.scale, params.headFirst,
                                        params.outputDtypeOptional, params.out, executorPtr);
    if (result == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "l0op",
            "l0op(ChunkLocalCumsum) returned nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(result, params.out, executorPtr);
    if (viewCopyResult == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "view_copy",
            "ViewCopy(out) failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkLocalCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkLocalCumsum);
    const aclnnStatus runRet = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (runRet != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "executor",
            "CommonOpExecutorRun failed, ret = " +
                std::to_string(static_cast<int64_t>(static_cast<int>(runRet))));
        return runRet;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
