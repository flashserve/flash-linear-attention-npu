#include "aclnn_recurrent_gated_delta_rule_v2.h"
#include "recurrent_gated_delta_rule_v2.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include <cmath>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
namespace {
constexpr int Q = 0, KEY = 1, VALUE = 2, BETA = 3, STATE = 4, SEQ = 5, IDX = 6;
constexpr int G = 7, GK = 8, ACCEPTED = 9, ALOG = 10, DTBIAS = 11;

struct Params {
    const aclTensor *query;
    const aclTensor *key;
    const aclTensor *value;
    const aclTensor *beta;
    aclTensor *state;
    const aclTensor *seq;
    const aclTensor *indices;
    const aclTensor *g;
    const aclTensor *accepted;
    const aclTensor *aLog;
    const aclTensor *dtBias;
    float scale;
    bool qkNorm;
    bool gateInKernel;
    bool betaSigmoid;
    bool allowNeg;
    const aclTensor *out;
};

static bool IsRank(const aclTensor *tensor, int rank)
{
    return tensor != nullptr && tensor->GetViewShape().GetDimNum() == rank;
}

static bool CheckShape(const Params &p)
{
    const auto qs = p.query->GetViewShape();
    const auto ks = p.key->GetViewShape();
    const auto vs = p.value->GetViewShape();
    const auto bs = p.beta->GetViewShape();
    const auto ss = p.state->GetViewShape();
    const auto ls = p.seq->GetViewShape();
    const auto is = p.indices->GetViewShape();
    if (qs.GetDimNum() != 3 || ks.GetDimNum() != 3 || vs.GetDimNum() != 3 || bs.GetDimNum() != 2 ||
        ss.GetDimNum() != 4 || ls.GetDimNum() != 1 || (is.GetDimNum() != 1 && is.GetDimNum() != 2)) {
        return false;
    }
    const int64_t t = qs.GetDim(0), nk = qs.GetDim(1), dk = qs.GetDim(2);
    const int64_t nv = vs.GetDim(1), dv = vs.GetDim(2);
    if (t <= 0 || nk <= 0 || nv <= 0 || ss.GetDim(0) <= 0 || dk != 128 || dv != 128 || qs.GetDim(0) != ks.GetDim(0) ||
        qs.GetDim(1) != ks.GetDim(1) || qs.GetDim(2) != ks.GetDim(2) || vs.GetDim(0) != t || ss.GetDim(1) != nv ||
        ss.GetDim(2) != dv || ss.GetDim(3) != dk || bs.GetDim(0) != t || bs.GetDim(1) != nv || nv % nk != 0 ||
        ls.GetDim(0) < 2) {
        return false;
    }
    const int64_t batch = ls.GetDim(0) - 1;
    if ((is.GetDimNum() == 1 && is.GetDim(0) != t) ||
        (is.GetDimNum() == 2 && (is.GetDim(0) != batch || is.GetDim(1) <= 0))) {
        return false;
    }
    if (p.accepted != nullptr && (!IsRank(p.accepted, 1) || p.accepted->GetViewShape().GetDim(0) != batch)) {
        return false;
    }
    if (p.aLog != nullptr && (!IsRank(p.aLog, 1) || p.aLog->GetViewShape().GetDim(0) != nv)) {
        return false;
    }
    if (p.dtBias != nullptr && (!IsRank(p.dtBias, 1) || p.dtBias->GetViewShape().GetDim(0) != nv)) {
        return false;
    }
    if (p.g != nullptr &&
        (!IsRank(p.g, 2) || p.g->GetViewShape().GetDim(0) != t || p.g->GetViewShape().GetDim(1) != nv)) {
        return false;
    }
    if (p.gateInKernel && (p.g == nullptr || p.aLog == nullptr)) {
        return false;
    }
    return std::isfinite(p.scale);
}

static bool CheckDtypes(const Params &p)
{
    const auto bf16 = DataType::DT_BF16;
    return p.query->GetDataType() == bf16 && p.key->GetDataType() == bf16 && p.value->GetDataType() == bf16 &&
           (p.beta->GetDataType() == bf16 || p.beta->GetDataType() == DataType::DT_FLOAT) &&
           (p.state->GetDataType() == bf16 || p.state->GetDataType() == DataType::DT_FLOAT) &&
           p.seq->GetDataType() == DataType::DT_INT32 && p.indices->GetDataType() == DataType::DT_INT32 &&
           (p.g == nullptr || p.g->GetDataType() == bf16 || p.g->GetDataType() == DataType::DT_FLOAT) &&
           (p.accepted == nullptr || p.accepted->GetDataType() == DataType::DT_INT32) &&
           (p.aLog == nullptr || p.aLog->GetDataType() == DataType::DT_FLOAT) &&
           (p.dtBias == nullptr || p.dtBias->GetDataType() == DataType::DT_FLOAT) && p.out->GetDataType() == bf16;
}

static aclnnStatus PreserveView(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(), tensor->GetViewStrides(),
                                  tensor->GetViewOffset());
    return tensor == nullptr ? ACLNN_ERR_INNER_NULLPTR : ACLNN_SUCCESS;
}

static aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    return tensor == nullptr ? ACLNN_ERR_INNER_NULLPTR : ACLNN_SUCCESS;
}

static aclnnStatus PreserveStateView(aclTensor *&tensor, aclOpExecutor *executor)
{
    if (IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(), tensor->GetViewStrides(),
                                  tensor->GetViewOffset());
    return tensor == nullptr ? ACLNN_ERR_INNER_NULLPTR : ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnRecurrentGatedDeltaRuleV2A5GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta, aclTensor *stateRef,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
    const aclTensor *numAcceptedTokens, const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    float scaleValue, bool useQkL2normInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel, bool allowNegEigval,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRecurrentGatedDeltaRuleV2A5,
                   DFX_IN(query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, gk,
                          numAcceptedTokens, aLogOptional, dtBiasOptional, scaleValue, useQkL2normInKernel,
                          useGateInKernel, useBetaSigmoidInKernel, allowNegEigval),
                   DFX_OUT(stateRef, out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(gk == nullptr, ACLNN_ERR_PARAM_INVALID);
    Params params{query,
                  key,
                  value,
                  beta,
                  stateRef,
                  actualSeqLengths,
                  ssmStateIndices,
                  g,
                  numAcceptedTokens,
                  aLogOptional,
                  dtBiasOptional,
                  scaleValue,
                  useQkL2normInKernel,
                  useGateInKernel,
                  useBetaSigmoidInKernel,
                  allowNegEigval,
                  out};
    CHECK_RET(query != nullptr && key != nullptr && value != nullptr && beta != nullptr && stateRef != nullptr &&
                  actualSeqLengths != nullptr && ssmStateIndices != nullptr && out != nullptr,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypes(params), ACLNN_ERR_PARAM_INVALID);
    if (!std::isfinite(scaleValue)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    query->SetOriginalShape(query->GetViewShape());
    key->SetOriginalShape(key->GetViewShape());
    value->SetOriginalShape(value->GetViewShape());
    beta->SetOriginalShape(beta->GetViewShape());
    stateRef->SetOriginalShape(stateRef->GetViewShape());
    actualSeqLengths->SetOriginalShape(actualSeqLengths->GetViewShape());
    ssmStateIndices->SetOriginalShape(ssmStateIndices->GetViewShape());

    const aclTensor *queryForKernel = query;
    const aclTensor *keyForKernel = key;
    const aclTensor *valueForKernel = value;
    const aclTensor *betaForKernel = beta;
    const aclTensor *seqForKernel = actualSeqLengths;
    const aclTensor *indicesForKernel = ssmStateIndices;
    const aclTensor *gForKernel = g;
    const aclTensor *acceptedForKernel = numAcceptedTokens;
    const aclTensor *aLogForKernel = aLogOptional;
    const aclTensor *dtBiasForKernel = dtBiasOptional;
    aclTensor *stateForKernel = stateRef;
    CHECK_RET(PreserveView(queryForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PreserveView(keyForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PreserveView(valueForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(betaForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(seqForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(indicesForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(gForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(acceptedForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(aLogForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(dtBiasForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(PreserveStateView(stateForKernel, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto outRet = l0op::RecurrentGatedDeltaRuleV2A5(
        queryForKernel, keyForKernel, valueForKernel, betaForKernel, stateForKernel, seqForKernel, indicesForKernel,
        gForKernel, nullptr, acceptedForKernel, aLogForKernel, dtBiasForKernel, scaleValue, useQkL2normInKernel,
        useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, uniqueExecutor.get());
    CHECK_RET(outRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outRet, out, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentGatedDeltaRuleV2A5(void *workspace, uint64_t workspaceSize,
                                             aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentGatedDeltaRuleV2A5);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
