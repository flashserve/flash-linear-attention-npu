// -----------------------------------------------------------------------------------------------------------
// aclnnChunkKdaFwdFused 实现(chunk32 大融合,issue #829)
// -----------------------------------------------------------------------------------------------------------

// Load structured reporters before opdev's legacy logging macros. The opdev
// headers must retain their logging convention for CHECK_RET and DFX helpers.
#include "op_common/log/log.h"
#include "aclnn_chunk_kda_fwd_fused.h"
#include "chunk_kda_fwd_fused.h"

#include <sstream>
#include <string>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "graph/utils/type_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t KDA_FD32_HEAD_DIM = 128;
constexpr int64_t KDA_FD32_CHUNK_TOKENS = 128;  // kernel 每组 128 token(4×32)
constexpr const char *OP_NAME = "aclnnChunkKdaFwdFused";

std::string ShapeText(const op::Shape &shape)
{
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) text << ',';
        text << shape.GetDim(i);
    }
    text << ']';
    return text.str();
}

aclnnStatus Missing(const char *name)
{
    OP_LOGE_WITH_INVALID_INPUT(OP_NAME, name);
    return ACLNN_ERR_PARAM_NULLPTR;
}

aclnnStatus InvalidValue(const char *name, const std::string &actual, const std::string &expected)
{
    OP_LOGE_FOR_INVALID_VALUE(OP_NAME, name, actual, expected);
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidArgument(const char *name, const std::string &reason)
{
    OP_LOGE_FOR_INVALID_VALUE(OP_NAME, name, "invalid", reason);
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidShapeText(const char *name, const std::string &actual, const std::string &expected)
{
    OP_LOGE_FOR_INVALID_SHAPE(OP_NAME, name, actual, expected);
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidShape(const aclTensor *tensor, const char *name, const std::string &expected)
{
    const auto actual = ShapeText(tensor->GetViewShape());
    // The predefined message supplies the enclosing brackets itself.
    const auto shape = actual.substr(1, actual.size() - 2);
    const auto correct = expected.size() >= 2 && expected.front() == '[' && expected.back() == ']'
        ? expected.substr(1, expected.size() - 2) : expected;
    return InvalidShapeText(name, shape, correct);
}

aclnnStatus InvalidDtype(const aclTensor *tensor, const char *name, const std::string &expected)
{
    OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, name,
        ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()), expected);
    return ACLNN_ERR_PARAM_INVALID;
}

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
    const aclTensor *required[] = {q, k, v, g, beta, aLog, dtBias};
    const char *requiredNames[] = {"q", "k", "v", "g", "beta", "aLog", "dtBias"};
    for (size_t i = 0; i < 7; ++i) {
        if (required[i] == nullptr) return Missing(requiredNames[i]);
    }
    if (cuSeqlens != nullptr && o == nullptr) return Missing("o");
    if (cuSeqlens != nullptr) {
        if (headMajorVarlen) {
            if (q->GetDataType() != DataType::DT_BF16) return InvalidDtype(q, "q", "BF16");
            const bool gOk = g->GetDataType() == DataType::DT_BF16 || g->GetDataType() == DataType::DT_FLOAT;
            const bool bOk = beta->GetDataType() == DataType::DT_BF16 || beta->GetDataType() == DataType::DT_FLOAT;
            if (!(gOk && bOk && g->GetDataType() == beta->GetDataType())) {
                if (!gOk) return InvalidDtype(g, "g", "BF16 or FP32, matching beta");
                if (!bOk) return InvalidDtype(beta, "beta", "BF16 or FP32, matching g");
                return InvalidArgument("g/beta",
                    "actual g dtype=" + ge::TypeUtils::DataTypeToSerialString(g->GetDataType()) +
                    ", beta dtype=" + ge::TypeUtils::DataTypeToSerialString(beta->GetDataType()) +
                    "; expected identical dtypes, both BF16 or both FP32");
            }
            if (!(q->GetViewShape().GetDimNum() == 3 && SameShape3(q, k))) {
                return InvalidShapeText("q/k",
                    "q=" + ShapeText(q->GetViewShape()) + ",k=" + ShapeText(k->GetViewShape()),
                    "rank-3 NTD varlen q/k with identical [H,T_total,K] shapes");
            }
            const int64_t totalTok = Dim(q, 1);
            const int64_t headNum = Dim(q, 0);
            const int64_t kDim = Dim(q, 2);
            if (!(kDim == KDA_FD32_HEAD_DIM && totalTok > 0)) {
                return InvalidValue("K/T_total",
                    "K=" + std::to_string(kDim) + ",T_total=" + std::to_string(totalTok),
                    "K=128 and T_total>0");
            }
            if (!(v->GetViewShape().GetDimNum() == 3 && Dim(v, 1) == totalTok &&
                  Dim(v, 2) == KDA_FD32_HEAD_DIM && g->GetViewShape().GetDimNum() == 3 &&
                  Dim(g, 1) == totalTok && Dim(g, 2) == kDim && Dim(v, 0) == Dim(g, 0))) {
                return InvalidShapeText("v/g",
                    "v=" + ShapeText(v->GetViewShape()) + ",g=" + ShapeText(g->GetViewShape()),
                    "v=[HV," + std::to_string(totalTok) + ",128],g=[HV," +
                    std::to_string(totalTok) + "," + std::to_string(kDim) + "] with equal HV");
            }
            const int64_t hvNum = Dim(v, 0);
            if (!(headNum > 0 && hvNum >= headNum && hvNum % headNum == 0)) {
                return InvalidValue("H/HV", "H=" + std::to_string(headNum) + ",HV=" + std::to_string(hvNum),
                    "HV>=H and HV%H==0");
            }
            if (!(beta->GetViewShape().GetDimNum() == 2 && Dim(beta, 0) == hvNum &&
                  Dim(beta, 1) == totalTok)) {
                return InvalidShape(beta, "beta",
                    "[" + std::to_string(hvNum) + "," + std::to_string(totalTok) + "]");
            }
            if (!(aLog->GetDataType() == DataType::DT_FLOAT && aLog->GetViewShape().GetDimNum() == 1 &&
                  Dim(aLog, 0) == hvNum)) {
                if (aLog->GetDataType() != DataType::DT_FLOAT) return InvalidDtype(aLog, "aLog", "FP32");
                return InvalidShape(aLog, "aLog", "[" + std::to_string(hvNum) + "]");
            }
            if (!(dtBias->GetDataType() == DataType::DT_FLOAT &&
                  dtBias->GetViewShape().GetShapeSize() == hvNum * kDim)) {
                if (dtBias->GetDataType() != DataType::DT_FLOAT) return InvalidDtype(dtBias, "dtBias", "FP32");
                return InvalidShape(dtBias, "dtBias",
                    "element count " + std::to_string(hvNum * kDim) + " ([HV*K])");
            }
            if (!(o->GetViewShape().GetDimNum() == 3 && Dim(o, 0) == totalTok &&
                  Dim(o, 1) == hvNum && Dim(o, 2) == KDA_FD32_HEAD_DIM)) {
                return InvalidShape(o, "o",
                    "[" + std::to_string(totalTok) + "," + std::to_string(hvNum) + ",128] (TND)");
            }
            const int64_t nSeq = static_cast<int64_t>(cuSeqlens->Size()) - 1;
            if (!(nSeq >= 1 && nSeq <= 1024)) {
                return InvalidValue("cuSeqlens.sequenceCount", std::to_string(nSeq), "[1,1024]");
            }
            const aclTensor *states[] = {initialState, finalState};
            const char *stateNames[] = {"initialState", "finalState"};
            for (size_t i = 0; i < 2; ++i) {
                const aclTensor *state = states[i];
                if (state == nullptr) {
                    continue;
                }
                if (state->GetDataType() != DataType::DT_FLOAT) {
                    return InvalidDtype(state, stateNames[i], "FP32");
                }
                if (!(state->GetViewShape().GetDimNum() == 3 && Dim(state, 0) == nSeq * hvNum &&
                      Dim(state, 1) == KDA_FD32_HEAD_DIM && Dim(state, 2) == KDA_FD32_HEAD_DIM)) {
                    return InvalidShape(state, stateNames[i],
                        "[" + std::to_string(nSeq * hvNum) + ",128,128]");
                }
            }
            return ACLNN_SUCCESS;
        }
        // token-major varlen:q/k [T,H,K],v/g [T,HV,*],beta [T,HV],
        // o [T,HV,V],state [nSeq*HV,K,V](aclnn 已拍平)。
        if (q->GetDataType() != DataType::DT_BF16) return InvalidDtype(q, "q", "BF16");
        const bool gOk = g->GetDataType() == DataType::DT_BF16 || g->GetDataType() == DataType::DT_FLOAT;
        const bool bOk = beta->GetDataType() == DataType::DT_BF16 || beta->GetDataType() == DataType::DT_FLOAT;
        if (!(gOk && bOk && g->GetDataType() == beta->GetDataType())) {
            if (!gOk) return InvalidDtype(g, "g", "BF16 or FP32, matching beta");
            if (!bOk) return InvalidDtype(beta, "beta", "BF16 or FP32, matching g");
            return InvalidArgument("g/beta",
                "actual g dtype=" + ge::TypeUtils::DataTypeToSerialString(g->GetDataType()) +
                ", beta dtype=" + ge::TypeUtils::DataTypeToSerialString(beta->GetDataType()) +
                "; expected identical dtypes, both BF16 or both FP32");
        }
        if (!(q->GetViewShape().GetDimNum() == 3 && SameShape3(q, k))) {
            return InvalidShapeText("q/k",
                "q=" + ShapeText(q->GetViewShape()) + ",k=" + ShapeText(k->GetViewShape()),
                "rank-3 varlen q/k with identical [T,H,K] shapes");
        }
        const int64_t totalTok = Dim(q, 0);
        const int64_t headNum = Dim(q, 1);
        const int64_t kDim = Dim(q, 2);
        if (!(kDim == KDA_FD32_HEAD_DIM && totalTok > 0)) {
            return InvalidValue("K/T", "K=" + std::to_string(kDim) + ",T=" + std::to_string(totalTok),
                "K=128 and T>0");
        }
        if (!(v->GetViewShape().GetDimNum() == 3 && Dim(v, 0) == totalTok &&
              Dim(v, 2) == KDA_FD32_HEAD_DIM && g->GetViewShape().GetDimNum() == 3 &&
              Dim(g, 0) == totalTok && Dim(g, 2) == kDim && Dim(v, 1) == Dim(g, 1))) {
            return InvalidShapeText("v/g",
                "v=" + ShapeText(v->GetViewShape()) + ",g=" + ShapeText(g->GetViewShape()),
                "v=[" + std::to_string(totalTok) + ",HV,128],g=[" + std::to_string(totalTok) +
                ",HV," + std::to_string(kDim) + "] with equal HV");
        }
        const int64_t hvNum = Dim(v, 1);
        if (!(headNum > 0 && hvNum >= headNum && hvNum % headNum == 0)) {
            return InvalidValue("H/HV", "H=" + std::to_string(headNum) + ",HV=" + std::to_string(hvNum),
                "HV>=H and HV%H==0");
        }
        // beta 在 aclnn varlen 路径里已被整调用一次转置为头连续 [HV, T]
        if (!(beta->GetViewShape().GetDimNum() == 2 && Dim(beta, 0) == hvNum &&
              Dim(beta, 1) == totalTok)) {
            return InvalidShape(beta, "beta",
                "[" + std::to_string(hvNum) + "," + std::to_string(totalTok) + "] (host-transposed)");
        }
        if (!(aLog->GetDataType() == DataType::DT_FLOAT && aLog->GetViewShape().GetDimNum() == 1 &&
              Dim(aLog, 0) == hvNum)) {
            if (aLog->GetDataType() != DataType::DT_FLOAT) return InvalidDtype(aLog, "aLog", "FP32");
            return InvalidShape(aLog, "aLog", "[" + std::to_string(hvNum) + "]");
        }
        if (!(dtBias->GetDataType() == DataType::DT_FLOAT &&
              dtBias->GetViewShape().GetShapeSize() == hvNum * kDim)) {
            if (dtBias->GetDataType() != DataType::DT_FLOAT) return InvalidDtype(dtBias, "dtBias", "FP32");
            return InvalidShape(dtBias, "dtBias",
                "element count " + std::to_string(hvNum * kDim) + " ([HV*K])");
        }
        if (!SameShape3(o, v)) return InvalidShape(o, "o", ShapeText(v->GetViewShape()));
        const int64_t nSeq = static_cast<int64_t>(cuSeqlens->Size()) - 1;
        if (!(nSeq >= 1 && nSeq <= 1024)) {
            return InvalidValue("cuSeqlens.sequenceCount", std::to_string(nSeq), "[1,1024]");
        }
        const aclTensor *states[] = {initialState, finalState};
        const char *stateNames[] = {"initialState", "finalState"};
        for (size_t i = 0; i < 2; ++i) {
            const aclTensor *state = states[i];
            if (state == nullptr) {
                continue;
            }
            if (state->GetDataType() != DataType::DT_FLOAT) {
                return InvalidDtype(state, stateNames[i], "FP32");
            }
            if (!(state->GetViewShape().GetDimNum() == 3 && Dim(state, 0) == nSeq * hvNum &&
                  Dim(state, 1) == KDA_FD32_HEAD_DIM && Dim(state, 2) == KDA_FD32_HEAD_DIM)) {
                return InvalidShape(state, stateNames[i],
                    "[" + std::to_string(nSeq * hvNum) + ",128,128]");
            }
        }
        return ACLNN_SUCCESS;
    }
    if (o == nullptr) return Missing("o");
    if (q->GetDataType() != DataType::DT_BF16) return InvalidDtype(q, "q", "BF16");
    // g/beta:bf16(组合 0)或 fp32(组合 1),且两者必须同 dtype(def zip 组合只有这两种)
    const bool gOk = g->GetDataType() == DataType::DT_BF16 || g->GetDataType() == DataType::DT_FLOAT;
    const bool bOk = beta->GetDataType() == DataType::DT_BF16 || beta->GetDataType() == DataType::DT_FLOAT;
    if (!(gOk && bOk && g->GetDataType() == beta->GetDataType())) {
        if (!gOk) return InvalidDtype(g, "g", "BF16 or FP32, matching beta");
        if (!bOk) return InvalidDtype(beta, "beta", "BF16 or FP32, matching g");
        return InvalidArgument("g/beta",
            "actual g dtype=" + ge::TypeUtils::DataTypeToSerialString(g->GetDataType()) +
            ", beta dtype=" + ge::TypeUtils::DataTypeToSerialString(beta->GetDataType()) +
            "; expected identical dtypes, both BF16 or both FP32");
    }
    if (q->GetViewShape().GetDimNum() != 3) {
        return InvalidShape(q, "q", "[H,T,K] head-contiguous rank 3 (kernel contract)");
    }
    if (!SameShape3(q, k)) return InvalidShape(k, "k", ShapeText(q->GetViewShape()));
    const int64_t headNum = Dim(q, 0);
    const int64_t seqLen = Dim(q, 1);
    const int64_t kDim = Dim(q, 2);
    if (kDim != KDA_FD32_HEAD_DIM) return InvalidValue("K", std::to_string(kDim), "128");
    if (seqLen <= 0) {
        return InvalidValue("T", std::to_string(seqLen), ">0 (tail chunks with T%128!=0 are handled in-kernel)");
    }
    // GVA(issue #829):v/g 为 [HV, T, K/V],HV 是 H 的整数倍(H==HV 为特例)
    if (!(v->GetViewShape().GetDimNum() == 3 && g->GetViewShape().GetDimNum() == 3 &&
          Dim(v, 1) == seqLen && Dim(g, 1) == seqLen &&
          Dim(v, 2) == KDA_FD32_HEAD_DIM && Dim(g, 2) == kDim &&
          Dim(v, 0) == Dim(g, 0))) {
        return InvalidShapeText("v/g",
            "v=" + ShapeText(v->GetViewShape()) + ",g=" + ShapeText(g->GetViewShape()),
            "v=[HV," + std::to_string(seqLen) + ",128],g=[HV," + std::to_string(seqLen) +
            "," + std::to_string(kDim) + "] with equal HV");
    }
    const int64_t hvNum = Dim(v, 0);
    if (!(headNum > 0 && hvNum >= headNum && hvNum % headNum == 0)) {
        return InvalidValue("H/HV", "H=" + std::to_string(headNum) + ",HV=" + std::to_string(hvNum),
            "HV>=H and HV%H==0 (GVA)");
    }
    if (!(beta->GetViewShape().GetDimNum() == 2 && Dim(beta, 0) == hvNum && Dim(beta, 1) == seqLen)) {
        return InvalidShape(beta, "beta", "[" + std::to_string(hvNum) + "," + std::to_string(seqLen) + "]");
    }
    if (!(aLog->GetDataType() == DataType::DT_FLOAT && aLog->GetViewShape().GetDimNum() == 1 &&
          Dim(aLog, 0) == hvNum)) {
        if (aLog->GetDataType() != DataType::DT_FLOAT) return InvalidDtype(aLog, "aLog", "FP32");
        return InvalidShape(aLog, "aLog", "[" + std::to_string(hvNum) + "]");
    }
    if (!(dtBias->GetDataType() == DataType::DT_FLOAT &&
          dtBias->GetViewShape().GetShapeSize() == hvNum * kDim)) {
        if (dtBias->GetDataType() != DataType::DT_FLOAT) return InvalidDtype(dtBias, "dtBias", "FP32");
        return InvalidShape(dtBias, "dtBias",
            "element count " + std::to_string(hvNum * kDim) + " ([HV*K])");
    }
    if (!SameShape3(o, v)) return InvalidShape(o, "o", ShapeText(v->GetViewShape()));
    const aclTensor *states[] = {initialState, finalState};
    const char *stateNames[] = {"initialState", "finalState"};
    for (size_t i = 0; i < 2; ++i) {
        const aclTensor *state = states[i];
        if (state == nullptr) {
            continue;
        }
        if (state->GetDataType() != DataType::DT_FLOAT) {
            return InvalidDtype(state, stateNames[i], "FP32");
        }
        if (!(state->GetViewShape().GetDimNum() == 3 && Dim(state, 0) == hvNum &&
              Dim(state, 1) == KDA_FD32_HEAD_DIM && Dim(state, 2) == KDA_FD32_HEAD_DIM)) {
            return InvalidShape(state, stateNames[i], "[" + std::to_string(hvNum) + ",128,128]");
        }
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
