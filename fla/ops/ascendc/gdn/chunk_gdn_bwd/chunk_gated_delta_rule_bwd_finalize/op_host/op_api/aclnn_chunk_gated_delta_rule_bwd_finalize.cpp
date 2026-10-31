#ifndef GDN_FINALIZE_ENABLE_ARCH22
#define GDN_FINALIZE_ENABLE_ARCH22 0
#endif

#include "aclnn_chunk_gated_delta_rule_bwd_finalize.h"
#include "chunk_gated_delta_rule_bwd_finalize.h"

#include <string>

#include "aclnn_kernels/contiguous.h"
#if GDN_FINALIZE_ENABLE_ARCH22
#include "external/aclnn_kernels/aclnn_platform.h"
#endif
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "op_common/log/log.h"

#if GDN_FINALIZE_ENABLE_ARCH22
#include <cstring>

#include "../arch22/aclnn_chunk_gated_delta_rule_bwd_finalize_arch22.h"
#endif

using namespace op;

namespace {

constexpr const char *OP_NAME = "ChunkGatedDeltaRuleBwdFinalize";

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE(OP_NAME, "Failed to create a contiguous input.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalizeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *dO, const aclTensor *du,
    const aclTensor *g, const aclTensor *beta, const aclTensor *h,
    const aclTensor *dh, const aclTensor *a,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool useQkL2NormInKernel,
    bool useBetaSigmoidInKernel, bool useGateInKernel, bool stateVFirst,
    bool useExp2,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dbetaOut, const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleBwdFinalize,
        DFX_IN(q, k, v, vNew, dO, du, g, beta, h, dh, a,
               qRstdOptional, kRstdOptional, betaRawOptional,
               cuSeqlensOptional, chunkIndicesOptional),
        DFX_OUT(dqOut, dkOut, dvOut, dbetaOut, dgOut));

    const aclTensor *requiredInputs[] = {q, k, v, vNew, dO, du, g, beta, h, dh, a};
    const char *requiredInputNames[] = {"q", "k", "v", "vNew", "dO", "du", "g", "beta", "h", "dh", "a"};
    for (size_t index = 0; index < sizeof(requiredInputs) / sizeof(requiredInputs[0]); ++index) {
        if (requiredInputs[index] == nullptr) {
            const char *param_name = requiredInputNames[index];
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, param_name);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (dqOut == nullptr || dkOut == nullptr || dvOut == nullptr || dbetaOut == nullptr ||
        dgOut == nullptr || workspaceSize == nullptr || executor == nullptr) {
        const char *param_name = dqOut == nullptr ? "dqOut" : dkOut == nullptr ? "dkOut" :
            dvOut == nullptr ? "dvOut" : dbetaOut == nullptr ? "dbetaOut" : dgOut == nullptr ? "dgOut" :
            workspaceSize == nullptr ? "workspaceSize" : "executor";
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, param_name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
#if GDN_FINALIZE_ENABLE_ARCH22
    // A2/A3 使用私有 Host 路径；A5 保留原有校验条件、连续化和 executor 流程。
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        return gdn_finalize_arch22::GetWorkspaceSize(
            q, k, v, vNew, dO, du, g, beta, h, dh, a,
            qRstdOptional, kRstdOptional, betaRawOptional,
            cuSeqlensOptional, chunkIndicesOptional,
            scale, chunkSize, useQkL2NormInKernel,
            useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2,
            dqOut, dkOut, dvOut, dbetaOut, dgOut, workspaceSize, executor);
    }
    const char *socName = aclrtGetSocName();
    if (socName == nullptr || std::strstr(socName, "Ascend950") == nullptr) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "soc", socName == nullptr ? "nullptr" : socName,
                                 "Ascend910B, Ascend910C or Ascend950");
        return ACLNN_ERR_PARAM_INVALID;
    }
#endif

    if (chunkSize != 64) {
        const std::string actual = std::to_string(chunkSize);
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "chunkSize", actual.c_str(), "64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useGateInKernel) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "useGateInKernel", "true", "false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((cuSeqlensOptional == nullptr) != (chunkIndicesOptional == nullptr)) {
        const char *actual = cuSeqlensOptional == nullptr ? "cuSeqlens absent; chunkIndices present" :
            "cuSeqlens present; chunkIndices absent";
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "cuSeqlensOptional/chunkIndicesOptional", actual,
                                 "both present or both absent");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE(OP_NAME, "Failed to create an executor.");
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto *executorPtr = uniqueExecutor.get();

    const aclTensor **contiguousInputs[] = {&q, &k, &v, &vNew, &dO, &du, &g, &beta, &h, &dh, &a,
                                            &qRstdOptional, &kRstdOptional, &betaRawOptional};
    for (auto **tensor : contiguousInputs) {
        if (MakeContiguous(*tensor, executorPtr) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const auto result = l0op::ChunkGatedDeltaRuleBwdFinalize(
        q, k, v, vNew, dO, du, g, beta, h, dh, a,
        qRstdOptional, kRstdOptional, betaRawOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useQkL2NormInKernel,
        useBetaSigmoidInKernel, useGateInKernel, stateVFirst, useExp2,
        dqOut, dkOut, dvOut, dbetaOut, dgOut,
        executorPtr);
    for (const aclTensor *tensor : result) {
        if (tensor == nullptr) {
            OP_LOGE(OP_NAME, "Failed to create a Finalize output.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    const aclTensor *outputs[] = {dqOut, dkOut, dvOut, dbetaOut, dgOut};
    for (size_t index = 0; index < result.size(); ++index) {
        const aclTensor *copied = l0op::ViewCopy(result[index], outputs[index], executorPtr);
        if (copied == nullptr) {
            OP_LOGE(OP_NAME, "Failed to copy output[%zu] to the output view.", index);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalize(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwdFinalize);
    if (CommonOpExecutorRun(workspace, workspaceSize, executor, stream) != ACLNN_SUCCESS) {
        OP_LOGE(OP_NAME, "ChunkGatedDeltaRuleBwdFinalize launch failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
