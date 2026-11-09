/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
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
#include "../../../../../common/host_utils/log.h"

using namespace op;

static constexpr const char *CHUNK_LOCAL_CUMSUM_ENTITY = "aclnnChunkLocalCumsum";

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
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_LOCAL_CUMSUM_ENTITY, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.out == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_LOCAL_CUMSUM_ENTITY, "out");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.outputDtypeOptional == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_LOCAL_CUMSUM_ENTITY, "outputDtype");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkLocalCumsumParams params)
{
    auto status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (params.chunkSize <= 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_LOCAL_CUMSUM_ENTITY, "chunkSize", std::to_string(params.chunkSize), "positive");
        return ACLNN_ERR_PARAM_INVALID;
    }
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
    auto dfxRetStatus1 = DataContiguous(params.g, executorPtr);
    if (dfxRetStatus1 != ACLNN_SUCCESS) {
        return dfxRetStatus1;
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

    auto dfxRetStatus2 = ParamsDataContiguous(params, executorPtr);
    if (dfxRetStatus2 != ACLNN_SUCCESS) {
        return dfxRetStatus2;
    }

    auto result = l0op::ChunkLocalCumsum(params.g, params.cuSeqlensOptional, params.chunkIndicesOutOptional,
                                        params.chunkSize, params.reverse, params.scale, params.headFirst,
                                        params.outputDtypeOptional, params.out, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, params.out, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkLocalCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkLocalCumsum);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ChunkLocalCumsum launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
