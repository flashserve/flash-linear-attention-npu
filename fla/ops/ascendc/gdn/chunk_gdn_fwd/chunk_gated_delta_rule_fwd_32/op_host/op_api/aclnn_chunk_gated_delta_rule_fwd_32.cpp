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
    const aclIntArray *cu;
    const aclIntArray *chunks;
    double scale;
    const aclTensor *o;
    const aclTensor *finalState;
};

aclnnStatus CheckMetadata(const aclIntArray *cu, const aclIntArray *chunks, int64_t batch, int64_t tokens,
                          int64_t &seqNum)
{
    seqNum = batch;
    if (cu == nullptr && chunks == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(cu != nullptr && chunks != nullptr, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must both be present or both be absent.");
    CHECK_COND(batch == 1, ACLNN_ERR_PARAM_INVALID, "packed sequences require B = 1.");
    CHECK_COND(cu->Size() >= 2, ACLNN_ERR_PARAM_INVALID, "cu_seqlens must contain at least [0, T].");
    CHECK_COND((*cu)[0] == 0 && (*cu)[cu->Size() - 1] == tokens, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must start at 0 and end at T.");
    int64_t pair = 0;
    for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
        CHECK_COND((*cu)[seq] <= (*cu)[seq + 1], ACLNN_ERR_PARAM_INVALID, "cu_seqlens must be nondecreasing.");
        const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
        const int64_t nChunk = length == 0 ? 0 : (length + 31) / 32;
        for (int64_t local = 0; local < nChunk; ++local) {
            const size_t pos = static_cast<size_t>(pair * 2);
            CHECK_COND(pos + 1 < chunks->Size() && (*chunks)[pos] == static_cast<int64_t>(seq) &&
                           (*chunks)[pos + 1] == local,
                       ACLNN_ERR_PARAM_INVALID, "chunk_indices must be canonical sequence-major (seqId, localChunk).");
            ++pair;
        }
    }
    CHECK_COND(static_cast<int64_t>(chunks->Size()) == pair * 2, ACLNN_ERR_PARAM_INVALID,
               "chunk_indices must contain one pair per chunk.");
    seqNum = static_cast<int64_t>(cu->Size()) - 1;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const Params &p)
{
    CHECK_COND(p.q != nullptr && p.k != nullptr && p.v != nullptr && p.g != nullptr && p.beta != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q, k, v, g and beta must not be null.");
    CHECK_COND(p.o != nullptr && p.finalState != nullptr, ACLNN_ERR_PARAM_NULLPTR, "o and final_state must not be null.");
    CHECK_COND(p.q->GetDataType() == DataType::DT_BF16 && p.k->GetDataType() == DataType::DT_BF16 &&
                   p.v->GetDataType() == DataType::DT_BF16 && p.beta->GetDataType() == DataType::DT_BF16 &&
                   p.o->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q, k, v, beta and o must be bfloat16.");
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
    CHECK_COND(hk > 0 && hv > 0 && (hv % hk) == 0, ACLNN_ERR_PARAM_INVALID,
               "Hv must be a positive multiple of Hk.");
    CHECK_COND(t > 0, ACLNN_ERR_PARAM_INVALID, "T must be positive.");
    CHECK_COND(k.GetDim(0) == b && k.GetDim(1) == hk && k.GetDim(2) == t && k.GetDim(3) == key,
               ACLNN_ERR_PARAM_INVALID, "k must match q.");
    CHECK_COND(v.GetDim(0) == b && v.GetDim(2) == t, ACLNN_ERR_PARAM_INVALID, "v must match q in B and T.");
    CHECK_COND(g.GetDim(0) == b && g.GetDim(1) == hv && g.GetDim(2) == t, ACLNN_ERR_PARAM_INVALID,
               "g must be [B, Hv, T].");
    CHECK_COND(beta.GetDim(0) == b && beta.GetDim(1) == hv && beta.GetDim(2) == t, ACLNN_ERR_PARAM_INVALID,
               "beta must be [B, Hv, T].");
    int64_t seqNum = b;
    auto meta = CheckMetadata(p.cu, p.chunks, b, t, seqNum);
    if (meta != ACLNN_SUCCESS) {
        return meta;
    }
    const auto &o = p.o->GetViewShape();
    const auto &state = p.finalState->GetViewShape();
    CHECK_COND(o.GetDimNum() == 4 && o.GetDim(0) == b && o.GetDim(1) == hv && o.GetDim(2) == t && o.GetDim(3) == value,
               ACLNN_ERR_PARAM_INVALID, "o must be [B, Hv, T, V].");
    CHECK_COND(state.GetDimNum() == 4 && state.GetDim(0) == seqNum && state.GetDim(1) == hv &&
                   state.GetDim(2) == key && state.GetDim(3) == value,
               ACLNN_ERR_PARAM_INVALID, "final_state must be [N, Hv, K, V].");
    if (p.initial != nullptr) {
        CHECK_COND(p.initial->GetDataType() == DataType::DT_BF16 || p.initial->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "initial_state must be bfloat16 or float32.");
        const auto &init = p.initial->GetViewShape();
        CHECK_COND(init.GetDimNum() == 4 && init.GetDim(0) == seqNum && init.GetDim(1) == hv &&
                       init.GetDim(2) == key && init.GetDim(3) == value,
                   ACLNN_ERR_PARAM_INVALID, "initial_state must be [N, Hv, K, V].");
    }
    const auto finalDtype = p.finalState->GetDataType();
    if (p.initial == nullptr || p.initial->GetDataType() == DataType::DT_FLOAT) {
        CHECK_COND(finalDtype == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "final_state must be float32 when initial_state is absent or float32.");
    } else {
        CHECK_COND(finalDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
                   "final_state must be bfloat16 when initial_state is bfloat16.");
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
    const aclTensor *initialStateOptional, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale, const aclTensor *o, const aclTensor *finalState,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be null.");
    Params params{q, k, v, g, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional, scale, o,
                  finalState};
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwd32,
                   DFX_IN(q, k, v, g, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional, scale),
                   DFX_OUT(o, finalState));
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
        params.q, params.k, params.v, params.g, params.beta, params.initial, params.cu, params.chunks, params.scale,
        params.o, params.finalState, executorPtr);
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
