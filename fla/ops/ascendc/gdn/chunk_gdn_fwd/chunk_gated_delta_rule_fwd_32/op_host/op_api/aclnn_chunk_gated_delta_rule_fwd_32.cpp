#include "aclnn_chunk_gated_delta_rule_fwd_32.h"

#include <cmath>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
struct Params {
    const aclTensor *q;
    const aclTensor *k;
    const aclTensor *v;
    const aclTensor *g;
    const aclTensor *beta;
    const aclTensor *initial;
    double scale;
    const aclTensor *o;
    const aclTensor *finalState;
};

aclnnStatus CheckParams(const Params &p)
{
    CHECK_COND(p.q != nullptr && p.k != nullptr && p.v != nullptr && p.g != nullptr && p.beta != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q, k, v, g and beta must not be null.");
    CHECK_COND(p.o != nullptr && p.finalState != nullptr, ACLNN_ERR_PARAM_NULLPTR, "o and final_state must not be null.");
    CHECK_COND(p.q->GetDataType() == DataType::DT_BF16 && p.k->GetDataType() == DataType::DT_BF16 &&
                   p.v->GetDataType() == DataType::DT_BF16 && p.beta->GetDataType() == DataType::DT_BF16 &&
                   p.o->GetDataType() == DataType::DT_BF16 && p.finalState->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q, k, v, beta, o and final_state must be bfloat16.");
    CHECK_COND(p.g->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID, "g must be float32.");
    const auto &q = p.q->GetViewShape();
    const auto &k = p.k->GetViewShape();
    const auto &v = p.v->GetViewShape();
    const auto &g = p.g->GetViewShape();
    const auto &beta = p.beta->GetViewShape();
    CHECK_COND(q.GetDimNum() == 4 && k.GetDimNum() == 4 && v.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
               "q, k and v must be rank-4 BNSD.");
    CHECK_COND(g.GetDimNum() == 3 && beta.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID, "g and beta must be [B, Hv, T].");
    const int64_t b = q.GetDim(0);
    const int64_t hk = q.GetDim(1);
    const int64_t t = q.GetDim(2);
    const int64_t key = q.GetDim(3);
    const int64_t hv = v.GetDim(1);
    const int64_t value = v.GetDim(3);
    CHECK_COND(key == 128 && value == 128, ACLNN_ERR_PARAM_INVALID, "K and V must be 128.");
    CHECK_COND(hv == hk && hv > 0, ACLNN_ERR_PARAM_INVALID, "the first version requires Hv == Hk > 0.");
    CHECK_COND(t > 0 && (t % 32) == 0, ACLNN_ERR_PARAM_INVALID, "T must be a positive multiple of 32.");
    CHECK_COND(k.GetDim(0) == b && k.GetDim(1) == hk && k.GetDim(2) == t && k.GetDim(3) == key,
               ACLNN_ERR_PARAM_INVALID, "k must match q.");
    CHECK_COND(v.GetDim(0) == b && v.GetDim(2) == t, ACLNN_ERR_PARAM_INVALID, "v must match q in B and T.");
    CHECK_COND(g.GetDim(0) == b && g.GetDim(1) == hv && g.GetDim(2) == t, ACLNN_ERR_PARAM_INVALID,
               "g must be [B, Hv, T].");
    CHECK_COND(beta.GetDim(0) == b && beta.GetDim(1) == hv && beta.GetDim(2) == t, ACLNN_ERR_PARAM_INVALID,
               "beta must be [B, Hv, T].");
    const auto &o = p.o->GetViewShape();
    const auto &state = p.finalState->GetViewShape();
    CHECK_COND(o.GetDimNum() == 4 && o.GetDim(0) == b && o.GetDim(1) == hv && o.GetDim(2) == t && o.GetDim(3) == value,
               ACLNN_ERR_PARAM_INVALID, "o must be [B, Hv, T, V].");
    CHECK_COND(state.GetDimNum() == 4 && state.GetDim(0) == b && state.GetDim(1) == hv && state.GetDim(2) == key &&
                   state.GetDim(3) == value,
               ACLNN_ERR_PARAM_INVALID, "final_state must be [B, Hv, K, V].");
    if (p.initial != nullptr) {
        CHECK_COND(p.initial->GetDataType() == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
                   "initial_state must be bfloat16.");
        const auto &init = p.initial->GetViewShape();
        CHECK_COND(init.GetDimNum() == 4 && init.GetDim(0) == b && init.GetDim(1) == hv && init.GetDim(2) == key &&
                       init.GetDim(3) == value,
                   ACLNN_ERR_PARAM_INVALID, "initial_state must be [B, Hv, K, V].");
    }
    CHECK_COND(std::isfinite(p.scale), ACLNN_ERR_PARAM_INVALID, "scale must be finite.");
    return ACLNN_SUCCESS;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_COND(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous conversion failed.");
    return ACLNN_SUCCESS;
}
}  // namespace

extern "C" {
aclnnStatus aclnnChunkGatedDeltaRuleFwd32GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *initialStateOptional, double scale, const aclTensor *o, const aclTensor *finalState,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be null.");
    Params params{q, k, v, g, beta, initialStateOptional, scale, o, finalState};
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwd32,
                   DFX_IN(q, k, v, g, beta, initialStateOptional, scale), DFX_OUT(o, finalState));
    auto status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();
    status = MakeContiguous(params.q, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeContiguous(params.k, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeContiguous(params.v, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeContiguous(params.g, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeContiguous(params.beta, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (params.initial != nullptr) {
        status = MakeContiguous(params.initial, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    const auto *result = l0op::ChunkGatedDeltaRuleFwd32(
        params.q, params.k, params.v, params.g, params.beta, params.initial, params.scale, params.o, params.finalState,
        executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto *oCopy = l0op::ViewCopy(result, params.o, executorPtr);
    CHECK_RET(oCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    (void)params.finalState;
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwd32(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleFwd32);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "aclnnChunkGatedDeltaRuleFwd32 failed.");
    return ACLNN_SUCCESS;
}
}
