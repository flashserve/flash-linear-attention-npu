// -----------------------------------------------------------------------------------------------------------
// aclnnChunkKdaFwdFused 实现(chunk32 大融合,issue #829)
// -----------------------------------------------------------------------------------------------------------

#include "aclnn_chunk_kda_fwd_fused.h"
#include "chunk_kda_fwd_fused.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t KDA_FD32_HEAD_DIM = 128;
constexpr int64_t KDA_FD32_CHUNK_TOKENS = 128;  // kernel 每组 128 token(4×32)

aclnnStatus KdaFd32Contiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

int64_t Dim(const aclTensor *t, size_t idx)
{
    return t->GetViewShape().GetDim(idx);
}

bool SameShape3(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs->GetViewShape().GetDimNum() != 3 || rhs->GetViewShape().GetDimNum() != 3) {
        return false;
    }
    for (size_t i = 0; i < 3; ++i) {
        if (Dim(lhs, i) != Dim(rhs, i)) {
            return false;
        }
    }
    return true;
}

aclnnStatus KdaFd32CheckParams(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g,
    const aclTensor *beta, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState, const aclTensor *o, const aclTensor *finalState,
    const aclIntArray *cuSeqlens, bool headMajorVarlen)
{
    CHECK_COND(q != nullptr && k != nullptr && v != nullptr && g != nullptr && beta != nullptr &&
                   aLog != nullptr && dtBias != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q/k/v/g/beta/aLog/dtBias must not be nullptr.");
    if (cuSeqlens != nullptr) {
        if (headMajorVarlen) {
            CHECK_COND(q->GetDataType() == DataType::DT_BF16,
                       ACLNN_ERR_PARAM_INVALID, "q must be bf16.");
            const bool gOk = g->GetDataType() == DataType::DT_BF16 || g->GetDataType() == DataType::DT_FLOAT;
            const bool bOk = beta->GetDataType() == DataType::DT_BF16 || beta->GetDataType() == DataType::DT_FLOAT;
            CHECK_COND(gOk && bOk && g->GetDataType() == beta->GetDataType(),
                       ACLNN_ERR_PARAM_INVALID, "varlen g/beta must share the same dtype, bf16 or fp32.");
            CHECK_COND(q->GetViewShape().GetDimNum() == 3 && SameShape3(q, k),
                       ACLNN_ERR_PARAM_INVALID, "NTD varlen q/k must be [H, T_total, K] rank3.");
            const int64_t totalTok = Dim(q, 1);
            const int64_t headNum = Dim(q, 0);
            const int64_t kDim = Dim(q, 2);
            CHECK_COND(kDim == KDA_FD32_HEAD_DIM && totalTok > 0, ACLNN_ERR_PARAM_INVALID,
                       "NTD varlen requires K == 128 and T_total > 0.");
            CHECK_COND(v->GetViewShape().GetDimNum() == 3 && Dim(v, 1) == totalTok &&
                           Dim(v, 2) == KDA_FD32_HEAD_DIM && g->GetViewShape().GetDimNum() == 3 &&
                           Dim(g, 1) == totalTok && Dim(g, 2) == kDim && Dim(v, 0) == Dim(g, 0),
                       ACLNN_ERR_PARAM_INVALID, "NTD varlen v/g must be [HV, T_total, 128].");
            const int64_t hvNum = Dim(v, 0);
            CHECK_COND(hvNum >= headNum && hvNum % headNum == 0, ACLNN_ERR_PARAM_INVALID,
                       "NTD varlen HV must be a positive multiple of H, got H=%ld HV=%ld.", headNum, hvNum);
            CHECK_COND(beta->GetViewShape().GetDimNum() == 2 && Dim(beta, 0) == hvNum &&
                           Dim(beta, 1) == totalTok,
                       ACLNN_ERR_PARAM_INVALID, "NTD varlen beta must be [HV, T_total].");
            CHECK_COND(aLog->GetDataType() == DataType::DT_FLOAT && aLog->GetViewShape().GetDimNum() == 1 &&
                           Dim(aLog, 0) == hvNum,
                       ACLNN_ERR_PARAM_INVALID, "aLog must be fp32 [HV].");
            CHECK_COND(dtBias->GetDataType() == DataType::DT_FLOAT &&
                           dtBias->GetViewShape().GetShapeSize() == hvNum * kDim,
                       ACLNN_ERR_PARAM_INVALID, "dtBias must be fp32 [HV*K].");
            CHECK_COND(o->GetViewShape().GetDimNum() == 3 && Dim(o, 0) == totalTok &&
                           Dim(o, 1) == hvNum && Dim(o, 2) == KDA_FD32_HEAD_DIM,
                       ACLNN_ERR_PARAM_INVALID, "NTD varlen o must remain TND [T_total, HV, V].");
            const int64_t nSeq = static_cast<int64_t>(cuSeqlens->Size()) - 1;
            CHECK_COND(nSeq >= 1 && nSeq <= 1024, ACLNN_ERR_PARAM_INVALID,
                       "varlen cu_seqlens: sequence count must be in [1, 1024].");
            for (auto *state : {initialState, finalState}) {
                if (state == nullptr) {
                    continue;
                }
                CHECK_COND(state->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                           "initialState/finalState must be fp32.");
                CHECK_COND(state->GetViewShape().GetDimNum() == 3 && Dim(state, 0) == nSeq * hvNum &&
                               Dim(state, 1) == KDA_FD32_HEAD_DIM && Dim(state, 2) == KDA_FD32_HEAD_DIM,
                           ACLNN_ERR_PARAM_INVALID,
                           "varlen initialState/finalState must be [nSeq*HV, 128, 128].");
            }
            return ACLNN_SUCCESS;
        }
        // token-major varlen:q/k [T,H,K],v/g [T,HV,*],beta [T,HV],
        // o [T,HV,V],state [nSeq*HV,K,V](aclnn 已拍平)。
        CHECK_COND(q->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID, "q must be bf16.");
        const bool gOk = g->GetDataType() == DataType::DT_BF16 || g->GetDataType() == DataType::DT_FLOAT;
        const bool bOk = beta->GetDataType() == DataType::DT_BF16 || beta->GetDataType() == DataType::DT_FLOAT;
        CHECK_COND(gOk && bOk && g->GetDataType() == beta->GetDataType(),
                   ACLNN_ERR_PARAM_INVALID, "varlen g/beta must share the same dtype, bf16 or fp32.");
        CHECK_COND(q->GetViewShape().GetDimNum() == 3 && SameShape3(q, k),
                   ACLNN_ERR_PARAM_INVALID, "varlen q/k must be [T, H, K] rank3.");
        const int64_t totalTok = Dim(q, 0);
        const int64_t headNum = Dim(q, 1);
        const int64_t kDim = Dim(q, 2);
        CHECK_COND(kDim == KDA_FD32_HEAD_DIM && totalTok > 0, ACLNN_ERR_PARAM_INVALID,
                   "varlen requires K == 128 and T > 0.");
        CHECK_COND(v->GetViewShape().GetDimNum() == 3 && Dim(v, 0) == totalTok &&
                       Dim(v, 2) == KDA_FD32_HEAD_DIM && g->GetViewShape().GetDimNum() == 3 &&
                       Dim(g, 0) == totalTok && Dim(g, 2) == kDim && Dim(v, 1) == Dim(g, 1),
                   ACLNN_ERR_PARAM_INVALID, "varlen v/g must be [T, HV, 128].");
        const int64_t hvNum = Dim(v, 1);
        CHECK_COND(hvNum >= headNum && hvNum % headNum == 0, ACLNN_ERR_PARAM_INVALID,
                   "varlen HV must be a positive multiple of H, got H=%ld HV=%ld.", headNum, hvNum);
        // beta 在 aclnn varlen 路径里已被整调用一次转置为头连续 [HV, T]
        CHECK_COND(beta->GetViewShape().GetDimNum() == 2 && Dim(beta, 0) == hvNum &&
                       Dim(beta, 1) == totalTok,
                   ACLNN_ERR_PARAM_INVALID, "varlen beta must be [HV, T] (host-transposed).");
        CHECK_COND(aLog->GetDataType() == DataType::DT_FLOAT && aLog->GetViewShape().GetDimNum() == 1 &&
                       Dim(aLog, 0) == hvNum,
                   ACLNN_ERR_PARAM_INVALID, "aLog must be fp32 [HV].");
        CHECK_COND(dtBias->GetDataType() == DataType::DT_FLOAT &&
                       dtBias->GetViewShape().GetShapeSize() == hvNum * kDim,
                   ACLNN_ERR_PARAM_INVALID, "dtBias must be fp32 [HV*K].");
        CHECK_COND(SameShape3(o, v), ACLNN_ERR_PARAM_INVALID, "varlen o shape must match v [T, HV, V].");
        const int64_t nSeq = static_cast<int64_t>(cuSeqlens->Size()) - 1;
        CHECK_COND(nSeq >= 1 && nSeq <= 1024, ACLNN_ERR_PARAM_INVALID,
                   "varlen cu_seqlens: sequence count must be in [1, 1024].");
        for (auto *state : {initialState, finalState}) {
            if (state == nullptr) {
                continue;
            }
            CHECK_COND(state->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                       "initialState/finalState must be fp32.");
            CHECK_COND(state->GetViewShape().GetDimNum() == 3 && Dim(state, 0) == nSeq * hvNum &&
                           Dim(state, 1) == KDA_FD32_HEAD_DIM && Dim(state, 2) == KDA_FD32_HEAD_DIM,
                       ACLNN_ERR_PARAM_INVALID,
                       "varlen initialState/finalState must be [nSeq*HV, 128, 128].");
        }
        return ACLNN_SUCCESS;
    }
    CHECK_COND(o != nullptr, ACLNN_ERR_PARAM_NULLPTR, "o must not be nullptr.");
    CHECK_COND(q->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q must be bf16.");
    // g/beta:bf16(组合 0)或 fp32(组合 1),且两者必须同 dtype(def zip 组合只有这两种)
    const bool gOk = g->GetDataType() == DataType::DT_BF16 || g->GetDataType() == DataType::DT_FLOAT;
    const bool bOk = beta->GetDataType() == DataType::DT_BF16 || beta->GetDataType() == DataType::DT_FLOAT;
    CHECK_COND(gOk && bOk && g->GetDataType() == beta->GetDataType(),
               ACLNN_ERR_PARAM_INVALID, "g/beta must share the same dtype, bf16 or fp32.");
    CHECK_COND(q->GetViewShape().GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID,
               "q must be [H, T, K] head-contiguous rank3 (kernel contract).");
    CHECK_COND(SameShape3(q, k), ACLNN_ERR_PARAM_INVALID, "k shape must match q [H, T, K].");
    const int64_t headNum = Dim(q, 0);
    const int64_t seqLen = Dim(q, 1);
    const int64_t kDim = Dim(q, 2);
    CHECK_COND(kDim == KDA_FD32_HEAD_DIM, ACLNN_ERR_PARAM_INVALID, "K must be 128.");
    CHECK_COND(seqLen > 0, ACLNN_ERR_PARAM_INVALID,
               "T must be positive (tail chunks with T % 128 != 0 are handled in-kernel).");
    // GVA(issue #829):v/g 为 [HV, T, K/V],HV 是 H 的整数倍(H==HV 为特例)
    CHECK_COND(v->GetViewShape().GetDimNum() == 3 && g->GetViewShape().GetDimNum() == 3 &&
                   Dim(v, 1) == seqLen && Dim(g, 1) == seqLen &&
                   Dim(v, 2) == KDA_FD32_HEAD_DIM && Dim(g, 2) == kDim &&
                   Dim(v, 0) == Dim(g, 0),
               ACLNN_ERR_PARAM_INVALID, "v/g must be [HV, T, K/V] with V == 128.");
    const int64_t hvNum = Dim(v, 0);
    CHECK_COND(hvNum >= headNum && hvNum % headNum == 0, ACLNN_ERR_PARAM_INVALID,
               "HV must be a positive multiple of H (GVA), got H=%ld HV=%ld.", headNum, hvNum);
    CHECK_COND(beta->GetViewShape().GetDimNum() == 2 && Dim(beta, 0) == hvNum && Dim(beta, 1) == seqLen,
               ACLNN_ERR_PARAM_INVALID, "beta must be [HV, T].");
    CHECK_COND(aLog->GetDataType() == DataType::DT_FLOAT && aLog->GetViewShape().GetDimNum() == 1 &&
                   Dim(aLog, 0) == hvNum,
               ACLNN_ERR_PARAM_INVALID, "aLog must be fp32 [HV].");
    CHECK_COND(dtBias->GetDataType() == DataType::DT_FLOAT &&
                   dtBias->GetViewShape().GetShapeSize() == hvNum * kDim,
               ACLNN_ERR_PARAM_INVALID, "dtBias must be fp32 [HV*K].");
    CHECK_COND(SameShape3(o, v), ACLNN_ERR_PARAM_INVALID, "o shape must match v [HV, T, V].");
    for (auto *state : {initialState, finalState}) {
        if (state == nullptr) {
            continue;
        }
        CHECK_COND(state->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "initialState/finalState must be fp32.");
        CHECK_COND(state->GetViewShape().GetDimNum() == 3 && Dim(state, 0) == hvNum &&
                       Dim(state, 1) == KDA_FD32_HEAD_DIM && Dim(state, 2) == KDA_FD32_HEAD_DIM,
                   ACLNN_ERR_PARAM_INVALID, "initialState/finalState must be [HV, 128, 128].");
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkKdaFwdFusedGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLog,
    const aclTensor *dtBias,
    const aclTensor *initialState,
    double scale,
    double lowerBound,
    const aclTensor *o,
    const aclTensor *finalState,
    const aclIntArray *cuSeqlens,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnChunkKdaFwdFused,
        DFX_IN(q, k, v, g, beta, aLog, dtBias, initialState, scale, lowerBound),
        DFX_OUT(o, finalState));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(KdaFd32CheckParams(q, k, v, g, beta, aLog, dtBias, initialState, o, finalState, cuSeqlens,
                                 false) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(aLog, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(dtBias, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaFd32Contiguous(initialState, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // optional 输出缺省时框架不生成 placeholder,kernel 参数序列
    // 少一个指针 → tiling 错位 → AICore 异常。缺省时给 [1,1,1] 哑张量(tiling 侧按
    // "未请求"处理,kernel 不写)。
    const aclTensor *finalStateArg = finalState;
    if (finalStateArg == nullptr) {
        finalStateArg = executorPtr->AllocTensor(op::Shape({1, 1, 1}), DataType::DT_FLOAT,
                                                 Format::FORMAT_ND);
        CHECK_RET(finalStateArg != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 本直接入口不暴露 epsilon,固定转发默认值 1e-6(与 def attr 默认一致);
    // 需自定义 epsilon 时走 aclnnChunkKdaFwdV2(chunk_kda_fwd)入口。
    auto result = l0op::KdaChunkForwardFused(q, k, v, g, beta, aLog, dtBias, initialState,
                                             scale, lowerBound, o, finalStateArg, cuSeqlens, 1.0e-6, executorPtr, "TND");
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdFused(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwdFused);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
