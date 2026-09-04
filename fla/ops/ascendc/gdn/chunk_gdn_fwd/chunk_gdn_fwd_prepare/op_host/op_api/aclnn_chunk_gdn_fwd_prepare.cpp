/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "aclnn_chunk_gdn_fwd_prepare.h"
#include "chunk_gdn_fwd_prepare.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr int64_t K_SIZE = 128;
constexpr int64_t CHUNK_SIZE = 64;
constexpr int64_t DIM_4 = 4;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_1 = 1;

struct ChunkGdnFwdPrepareParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    bool allowNegEigval = false;
    bool useExp2 = false;
    const aclTensor *gOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *aOut = nullptr;
    const aclTensor *qHatOptional = nullptr;
    const aclTensor *kHatOptional = nullptr;
    const aclTensor *qRstdOptional = nullptr;
    const aclTensor *kRstdOptional = nullptr;
    const aclTensor *betaEffOptional = nullptr;
};

static bool IsHalfDtype(DataType dtype)
{
    return dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16;
}

static bool IsGateDtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT || dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16;
}

static aclnnStatus CheckNotNull(const ChunkGdnFwdPrepareParams &params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(params.gOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gOut must not be nullptr.");
    CHECK_COND(params.wOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "wOut must not be nullptr.");
    CHECK_COND(params.uOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "uOut must not be nullptr.");
    CHECK_COND(params.aOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "aOut must not be nullptr.");
    CHECK_COND(params.qHatOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "qHat is required: this version always runs Q/K L2Norm.");
    CHECK_COND(params.kHatOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "kHat is required: this version always runs Q/K L2Norm.");
    CHECK_COND(params.qRstdOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "qRstd is required: this version always runs Q/K L2Norm.");
    CHECK_COND(params.kRstdOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "kRstd is required: this version always runs Q/K L2Norm.");
    CHECK_COND(params.chunkSize == CHUNK_SIZE, ACLNN_ERR_PARAM_INVALID,
               "chunkSize currently only supports 64.");
    CHECK_COND(params.useExp2, ACLNN_ERR_PARAM_INVALID,
               "useExp2 currently only supports true.");
    CHECK_COND(params.aLogOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "use_gate_in_kernel currently only supports false (aLog must be nullptr).");
    CHECK_COND(params.dtBiasOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "dtBiasOptional requires aLogOptional (use_gate_in_kernel), which is unsupported.");
    CHECK_COND(!(params.allowNegEigval && params.betaEffOptional == nullptr), ACLNN_ERR_PARAM_INVALID,
               "allowNegEigval requires betaEff (use_beta_sigmoid_in_kernel).");
    if (params.cuSeqlensOptional != nullptr) {
        CHECK_COND(params.chunkIndicesOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "varlen requires cuSeqlensOptional and chunkIndicesOptional together.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const ChunkGdnFwdPrepareParams &params)
{
    CHECK_COND(IsHalfDtype(params.q->GetDataType()) && params.q->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q currently only supports bfloat16.");
    CHECK_COND(params.k->GetDataType() == params.q->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "k dtype must match q.");
    CHECK_COND(params.v->GetDataType() == params.q->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "v dtype must match q.");
    CHECK_COND(IsGateDtype(params.g->GetDataType()), ACLNN_ERR_PARAM_INVALID, "g must be fp32/bf16/fp16.");
    CHECK_COND(IsGateDtype(params.beta->GetDataType()), ACLNN_ERR_PARAM_INVALID, "beta must be fp32/bf16/fp16.");
    CHECK_COND(params.gOut->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "gOut must be fp32.");
    CHECK_COND(params.wOut->GetDataType() == params.k->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "wOut dtype must match k.");
    CHECK_COND(params.uOut->GetDataType() == params.v->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "uOut dtype must match v.");
    CHECK_COND(params.aOut->GetDataType() == params.k->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "aOut dtype must match k.");
    if (params.aLogOptional != nullptr) {
        CHECK_COND(IsGateDtype(params.aLogOptional->GetDataType()), ACLNN_ERR_PARAM_INVALID,
                   "aLogOptional must be fp32/bf16/fp16.");
    }
    if (params.dtBiasOptional != nullptr) {
        CHECK_COND(IsGateDtype(params.dtBiasOptional->GetDataType()), ACLNN_ERR_PARAM_INVALID,
                   "dtBiasOptional must be fp32/bf16/fp16.");
    }
    CHECK_COND(params.qHatOptional->GetDataType() == params.q->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "qHat dtype must match q.");
    CHECK_COND(params.kHatOptional->GetDataType() == params.k->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "kHat dtype must match k.");
    CHECK_COND(params.qRstdOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "qRstd must be fp32.");
    CHECK_COND(params.kRstdOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "kRstd must be fp32.");
    if (params.betaEffOptional != nullptr) {
        CHECK_COND(params.betaEffOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "betaEffOptional must be fp32.");
    }
    return ACLNN_SUCCESS;
}

static bool ShapeEqual(const op::Shape &lhs, const op::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckShape(const ChunkGdnFwdPrepareParams &params)
{
    const op::Shape qShape = params.q->GetViewShape();
    const op::Shape kShape = params.k->GetViewShape();
    const op::Shape vShape = params.v->GetViewShape();
    const op::Shape gShape = params.g->GetViewShape();
    const op::Shape betaShape = params.beta->GetViewShape();
    CHECK_COND(qShape.GetDimNum() == DIM_4, ACLNN_ERR_PARAM_INVALID, "q must be [B, Hk, T, K].");
    CHECK_COND(ShapeEqual(kShape, qShape), ACLNN_ERR_PARAM_INVALID, "k shape must match q.");
    CHECK_COND(vShape.GetDimNum() == DIM_4, ACLNN_ERR_PARAM_INVALID, "v must be [B, Hv, T, V].");
    CHECK_COND(gShape.GetDimNum() == DIM_3, ACLNN_ERR_PARAM_INVALID, "g must be [B, Hv, T].");
    CHECK_COND(ShapeEqual(betaShape, gShape), ACLNN_ERR_PARAM_INVALID, "beta shape must match g.");

    const int64_t B = qShape.GetDim(0);
    const int64_t HK = qShape.GetDim(1);
    const int64_t T = qShape.GetDim(2);
    const int64_t K = qShape.GetDim(3);
    const int64_t HV = vShape.GetDim(1);
    const int64_t V = vShape.GetDim(3);
    CHECK_COND(vShape.GetDim(0) == B && vShape.GetDim(2) == T, ACLNN_ERR_PARAM_INVALID,
               "v batch/seq must match q.");
    CHECK_COND(gShape.GetDim(0) == B && gShape.GetDim(1) == HV && gShape.GetDim(2) == T, ACLNN_ERR_PARAM_INVALID,
               "g must be [B, Hv, T].");
    CHECK_COND(K == K_SIZE, ACLNN_ERR_PARAM_INVALID, "K currently only supports 128.");
    CHECK_COND(V == 128 || V == 256, ACLNN_ERR_PARAM_INVALID, "V must be 128 or 256.");
    CHECK_COND(HK > 0 && HV % HK == 0, ACLNN_ERR_PARAM_INVALID, "Hv must be divisible by Hk.");
    const int64_t ratio = HV / HK;
    CHECK_COND(ratio >= 1 && ratio <= 4, ACLNN_ERR_PARAM_INVALID, "Hv/Hk must be in {1,2,3,4}.");

    if (params.aLogOptional != nullptr) {
        const op::Shape aLogShape = params.aLogOptional->GetViewShape();
        CHECK_COND(aLogShape.GetDimNum() == DIM_1 && aLogShape.GetDim(0) == HV, ACLNN_ERR_PARAM_INVALID,
                   "aLogOptional must be [Hv].");
    }
    if (params.dtBiasOptional != nullptr) {
        const op::Shape dtShape = params.dtBiasOptional->GetViewShape();
        CHECK_COND(dtShape.GetDimNum() == DIM_1 && dtShape.GetDim(0) == HV, ACLNN_ERR_PARAM_INVALID,
                   "dtBiasOptional must be [Hv].");
    }
    if (params.cuSeqlensOptional != nullptr) {
        CHECK_COND(B == 1, ACLNN_ERR_PARAM_INVALID, "varlen requires B=1.");
    }
    CHECK_COND(ShapeEqual(params.qHatOptional->GetViewShape(), qShape), ACLNN_ERR_PARAM_INVALID,
               "qHat shape must match q.");
    CHECK_COND(ShapeEqual(params.kHatOptional->GetViewShape(), kShape), ACLNN_ERR_PARAM_INVALID,
               "kHat shape must match k.");
    {
        const op::Shape qRstd = params.qRstdOptional->GetViewShape();
        CHECK_COND(qRstd.GetDimNum() == DIM_3 && qRstd.GetDim(0) == B && qRstd.GetDim(1) == HK &&
                       qRstd.GetDim(2) == T,
                   ACLNN_ERR_PARAM_INVALID, "qRstd must be [B, Hk, T].");
        const op::Shape kRstd = params.kRstdOptional->GetViewShape();
        CHECK_COND(kRstd.GetDimNum() == DIM_3 && kRstd.GetDim(0) == B && kRstd.GetDim(1) == HK &&
                       kRstd.GetDim(2) == T,
                   ACLNN_ERR_PARAM_INVALID, "kRstd must be [B, Hk, T].");
    }
    if (params.betaEffOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.betaEffOptional->GetViewShape(), gShape), ACLNN_ERR_PARAM_INVALID,
                   "betaEffOptional shape must match g.");
    }
    CHECK_COND(ShapeEqual(params.gOut->GetViewShape(), gShape), ACLNN_ERR_PARAM_INVALID, "gOut shape must match g.");
    CHECK_COND(ShapeEqual(params.uOut->GetViewShape(), vShape), ACLNN_ERR_PARAM_INVALID, "uOut shape must match v.");
    const op::Shape wShape = params.wOut->GetViewShape();
    CHECK_COND(wShape.GetDimNum() == DIM_4 && wShape.GetDim(0) == B && wShape.GetDim(1) == HV &&
                   wShape.GetDim(2) == T && wShape.GetDim(3) == K,
               ACLNN_ERR_PARAM_INVALID, "wOut must be [B, Hv, T, K].");
    const op::Shape aShape = params.aOut->GetViewShape();
    CHECK_COND(aShape.GetDimNum() == DIM_4 && aShape.GetDim(0) == B && aShape.GetDim(1) == HV &&
                   aShape.GetDim(2) == T && aShape.GetDim(3) == params.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "aOut must be [B, Hv, T, chunkSize].");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus OptionalDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    return DataContiguous(tensor, executor);
}

static aclnnStatus ParamsDataContiguous(ChunkGdnFwdPrepareParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous g failed.");
    CHECK_COND(DataContiguous(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous beta failed.");
    CHECK_COND(OptionalDataContiguous(params.aLogOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous aLog failed.");
    CHECK_COND(OptionalDataContiguous(params.dtBiasOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dtBias failed.");
    return ACLNN_SUCCESS;
}

static aclnnStatus ViewCopyIfPresent(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    if (dst == nullptr || src == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto copied = l0op::ViewCopy(src, dst, executor);
    CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGdnFwdPrepareParams &params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnChunkGdnFwdPrepareGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    bool allowNegEigval,
    bool useExp2,
    aclTensor *gOut,
    aclTensor *wOut,
    aclTensor *uOut,
    aclTensor *aOut,
    aclTensor *qHatOptional,
    aclTensor *kHatOptional,
    aclTensor *qRstdOptional,
    aclTensor *kRstdOptional,
    aclTensor *betaEffOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkGdnFwdPrepareParams params{
        q, k, v, g, beta, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkIndicesOptional,
        chunkSize, allowNegEigval, useExp2, gOut, wOut, uOut, aOut,
        qHatOptional, kHatOptional, qRstdOptional, kRstdOptional, betaEffOptional};

    L2_DFX_PHASE_1(aclnnChunkGdnFwdPrepare,
                   DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional, cuSeqlensOptional,
                          chunkIndicesOptional, chunkSize, allowNegEigval, useExp2),
                   DFX_OUT(gOut, wOut, uOut, aOut, qHatOptional, kHatOptional, qRstdOptional,
                           kRstdOptional, betaEffOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");

    auto result = l0op::ChunkGdnFwdPrepare(
        params.q, params.k, params.v, params.g, params.beta, params.aLogOptional, params.dtBiasOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize, params.allowNegEigval,
        params.useExp2, true,
        params.aLogOptional != nullptr, params.betaEffOptional != nullptr, params.gOut, params.wOut,
        params.uOut, params.aOut, params.qHatOptional, params.kHatOptional, params.qRstdOptional,
        params.kRstdOptional, params.betaEffOptional, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[3] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(ViewCopyIfPresent(result[0], params.gOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[1], params.wOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[2], params.uOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[3], params.aOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[4], params.qHatOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[5], params.kHatOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[6], params.qRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[7], params.kRstdOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopyIfPresent(result[8], params.betaEffOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGdnFwdPrepare(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGdnFwdPrepare);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkGdnFwdPrepare launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
