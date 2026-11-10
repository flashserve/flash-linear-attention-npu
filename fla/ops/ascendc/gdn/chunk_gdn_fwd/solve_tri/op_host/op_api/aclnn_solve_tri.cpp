/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
#include "op_common/log/log.h"
#include "aclnn_solve_tri.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>
 #include "aclnn_kernels/contiguous.h"
 #include "acl/acl.h"
 #include "aclnn/aclnn_base.h"
 #include "aclnn_kernels/common/op_error_check.h"
 #include "opdev/common_types.h"
 #include "opdev/op_executor.h"
 #include "opdev/op_log.h"
 #include "opdev/op_dfx.h"
 #include "opdev/make_op_executor.h"
 #include "opdev/tensor_view_utils.h"
 
 using namespace op;

namespace {
constexpr const char *OP_NAME = "aclnnSolveTri";

template <typename Shape>
std::string ShapeText(const Shape &shape)
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

aclnnStatus InvalidValue(const char *name, const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_VALUE
    OP_LOGE_FOR_INVALID_VALUE(OP_NAME, name, actual, expected);
#else
    OP_LOGE_WITH_INVALID_ATTR(OP_NAME, name, actual.c_str(), expected.c_str());
#endif
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidDtype(const char *name, const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_DTYPE
    OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, name, actual, expected);
#else
    OP_LOGE_WITH_INVALID_ATTR(OP_NAME, name, actual.c_str(), expected.c_str());
#endif
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidRank(const char *name, const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_SHAPEDIM
    OP_LOGE_FOR_INVALID_SHAPEDIM(OP_NAME, name, actual, expected);
#else
    OP_LOGE_WITH_INVALID_ATTR(OP_NAME, name, actual.c_str(), expected.c_str());
#endif
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidShape(const char *name, const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_SHAPE
    // CANN's predefined shape message supplies the enclosing brackets.
    const auto actualDims = actual.substr(1, actual.size() - 2);
    const auto expectedDims = expected.substr(1, expected.size() - 2);
    OP_LOGE_FOR_INVALID_SHAPE(OP_NAME, name, actualDims, expectedDims);
#else
    OP_LOGE_WITH_INVALID_ATTR(OP_NAME, name, actual.c_str(), expected.c_str());
#endif
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidFormat(const char *name, const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_FORMAT
    OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, name, actual, expected);
#else
    OP_LOGE_WITH_INVALID_ATTR(OP_NAME, name, actual.c_str(), expected.c_str());
#endif
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus InvalidArgument(const char *name, const std::string &reason)
{
#ifdef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, name, reason);
#else
    OP_LOGE_WITH_INVALID_ATTR(OP_NAME, name, "invalid", reason.c_str());
#endif
    return ACLNN_ERR_PARAM_INVALID;
}
}
 
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 struct SolveTriParams {
     const aclTensor *x = nullptr;
     const aclIntArray *cuSeqlens = nullptr;
     const aclIntArray *chunkIndices = nullptr;
     const char *layout = "bsnd";
     const aclTensor *xOut = nullptr;
 };
 
 static aclnnStatus CheckNotNull(SolveTriParams params)
 {
     if (params.x == nullptr) {
         InvalidArgument("x", "actual=nullptr; expected a non-null tensor");
         return ACLNN_ERR_PARAM_NULLPTR;
     }
     if (params.xOut == nullptr) {
         InvalidArgument("xOut", "actual=nullptr; expected a non-null tensor");
         return ACLNN_ERR_PARAM_NULLPTR;
     }
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus CheckParams(SolveTriParams params)
 {
    const auto nullStatus = CheckNotNull(params);
    if (nullStatus != ACLNN_SUCCESS) return nullStatus;
    const auto dtype = params.x->GetDataType();
    if (dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16) {
        return InvalidDtype("x.dtype", op::ToString(dtype).GetString(), "FLOAT16 or BF16");
    }
    if (params.xOut->GetDataType() != dtype) {
        return InvalidDtype("xOut.dtype", op::ToString(params.xOut->GetDataType()).GetString(),
                            op::ToString(dtype).GetString());
    }
    const auto xShape = params.x->GetViewShape();
    const auto xOutShape = params.xOut->GetViewShape();
    if (xShape.GetDimNum() != 3 && xShape.GetDimNum() != 4) {
        return InvalidRank("x.rank", std::to_string(xShape.GetDimNum()), "3 or 4");
    }
    if (!(xShape == xOutShape)) {
        return InvalidShape("xOut.shape", ShapeText(xOutShape), ShapeText(xShape));
    }
    if (!IsContiguous(params.xOut)) {
        return InvalidArgument("xOut", "actual=non-contiguous; expected a contiguous tensor");
    }
    const int64_t chunkSize = xShape.GetDim(xShape.GetDimNum() - 1);
    if (chunkSize != 16 && chunkSize != 32 && chunkSize != 64 && chunkSize != 128) {
        return InvalidValue("x.chunkSize", std::to_string(chunkSize), "16, 32, 64, or 128");
    }
    const std::string layout = params.layout == nullptr ? "bsnd" : params.layout;
    const bool isDenseLayout = layout == "bsnd" || layout == "bnsd" || layout == "bhtd";
    const bool isVarlenLayout = layout == "tnd" || layout == "ntd";
    if (!isDenseLayout && !isVarlenLayout) {
        return InvalidFormat("layout", layout, "bsnd, bnsd, bhtd, tnd, or ntd");
    }
    if ((isDenseLayout && xShape.GetDimNum() != 4) || (isVarlenLayout && xShape.GetDimNum() != 3)) {
        return InvalidArgument("x.rank,layout", "actual rank=" + std::to_string(xShape.GetDimNum()) +
                               ", layout=" + layout + "; expected rank " + (isDenseLayout ? "4" : "3"));
    }
    for (size_t dim = 0; dim + 1 < xShape.GetDimNum(); ++dim) {
        if (xShape.GetDim(dim) <= 0) {
            return InvalidArgument("x.shape", "actual shape=" + ShapeText(xShape) +
                                   "; expected positive batch, head, and token dimensions");
        }
    }
    if ((params.cuSeqlens != nullptr) != (params.chunkIndices != nullptr)) {
        return InvalidArgument("cu_seqlens,chunk_indices", std::string("actual cu_seqlens=") +
            (params.cuSeqlens ? "present" : "null") + ", chunk_indices=" +
            (params.chunkIndices ? "present" : "null") + "; expected both present or both null");
    }
    if (isDenseLayout) {
        if (params.cuSeqlens != nullptr) {
            return InvalidArgument("cu_seqlens,chunk_indices", "actual=present; expected both null for layout " + layout);
        }
        return ACLNN_SUCCESS;
    }
    if (params.cuSeqlens == nullptr) {
        return InvalidArgument("cu_seqlens,chunk_indices", "actual=both null; expected both present for layout " + layout);
    }
    if (params.cuSeqlens->Size() < 2) {
        return InvalidArgument("cu_seqlens",
            "actual size=" + std::to_string(params.cuSeqlens->Size()) + "; expected at least 2");
    }
    const int64_t totalTokens = layout == "tnd" ? xShape.GetDim(0) : xShape.GetDim(1);
    if ((*params.cuSeqlens)[0] != 0 ||
        (*params.cuSeqlens)[params.cuSeqlens->Size() - 1] != totalTokens) {
        return InvalidArgument("cu_seqlens",
            "actual first=" + std::to_string((*params.cuSeqlens)[0]) + ", last=" +
            std::to_string((*params.cuSeqlens)[params.cuSeqlens->Size() - 1]) +
            "; expected first=0, last=" + std::to_string(totalTokens));
    }
    int64_t expectedChunks = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
        const int64_t begin = (*params.cuSeqlens)[seq];
        const int64_t end = (*params.cuSeqlens)[seq + 1];
        if (begin < 0 || end < begin || end > totalTokens) {
            return InvalidArgument("cu_seqlens",
                "sequence " + std::to_string(seq) + " actual=[" + std::to_string(begin) + "," +
                std::to_string(end) + "]; expected 0 <= begin <= end <= " + std::to_string(totalTokens));
        }
        const int64_t length = end - begin;
        expectedChunks += length / chunkSize + (length % chunkSize != 0);
    }
    if (params.chunkIndices->Size() % 2 != 0 ||
        params.chunkIndices->Size() / 2 != static_cast<size_t>(expectedChunks)) {
        return InvalidArgument("chunk_indices",
            "actual element count=" + std::to_string(params.chunkIndices->Size()) +
            "; expected one pair per chunk (" + std::to_string(expectedChunks) + " pairs)");
    }
    // Most callers supply canonical order. Keep that path allocation-free.
    bool canonical = true;
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
        const int64_t length = (*params.cuSeqlens)[seq + 1] - (*params.cuSeqlens)[seq];
        const int64_t chunks = length / chunkSize + (length % chunkSize != 0);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            canonical &= (*params.chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                         (*params.chunkIndices)[offset + 1] == chunk;
            offset += 2;
        }
    }
    if (!canonical) {
        using ChunkPair = std::pair<int64_t, int64_t>;
        const size_t pairCount = params.chunkIndices->Size() / 2;
        if (pairCount > std::numeric_limits<size_t>::max() / sizeof(ChunkPair)) {
            return InvalidArgument("chunk_indices", "actual pair count=" + std::to_string(pairCount) +
                "; expected a count that fits in the validation buffer");
        }
        std::unique_ptr<ChunkPair[]> sorted(new (std::nothrow) ChunkPair[pairCount]);
        if (!sorted) {
            REPORT_INNER_ERR_MSG("EZ9999", "SolveTri: cannot allocate chunk_indices validation buffer");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        for (size_t i = 0; i < pairCount; ++i) {
            sorted[i] = {(*params.chunkIndices)[i * 2], (*params.chunkIndices)[i * 2 + 1]};
        }
        std::sort(sorted.get(), sorted.get() + pairCount);
        size_t pair = 0;
        for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
            const int64_t length = (*params.cuSeqlens)[seq + 1] - (*params.cuSeqlens)[seq];
            const int64_t chunks = length / chunkSize + (length % chunkSize != 0);
            for (int64_t chunk = 0; chunk < chunks; ++chunk, ++pair) {
                if (sorted[pair] != ChunkPair{static_cast<int64_t>(seq), chunk}) {
                    return InvalidArgument("chunk_indices",
                        "sorted pair " + std::to_string(pair) + " actual=[" +
                        std::to_string(sorted[pair].first) + "," + std::to_string(sorted[pair].second) +
                        "]; expected [" + std::to_string(seq) + "," + std::to_string(chunk) +
                        "] with each chunk represented exactly once");
                }
            }
        }
    }
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
 {
     tensor = l0op::Contiguous(tensor, executor);
     CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
     return ACLNN_SUCCESS;
 }
 
 static aclnnStatus ParamsDataContiguous(SolveTriParams &params, aclOpExecutor *executorPtr)
 {
     CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                "Contiguous x failed.");
     return ACLNN_SUCCESS;
 }
 
 aclnnStatus aclnnSolveTriGetWorkspaceSize(
     const aclTensor *x,
     const aclIntArray *cuSeqlens,
     const aclIntArray *chunkIndices,
     const char *layout,
     const aclTensor *xOut,
     uint64_t *workspaceSize,
     aclOpExecutor **executor)
 {
     SolveTriParams params{x, cuSeqlens, chunkIndices, layout, xOut};

     if (workspaceSize == nullptr) {
         InvalidArgument("workspaceSize", "actual=nullptr; expected a non-null pointer");
         return ACLNN_ERR_PARAM_NULLPTR;
     }
     if (executor == nullptr) {
         InvalidArgument("executor", "actual=nullptr; expected a non-null pointer");
         return ACLNN_ERR_PARAM_NULLPTR;
     }
 
     L2_DFX_PHASE_1(aclnnSolveTri, DFX_IN(x, cuSeqlens, chunkIndices, layout), DFX_OUT(xOut));
 
     auto uniqueExecutor = CREATE_EXECUTOR();
     CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
     auto executorPtr = uniqueExecutor.get();
 
     auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
 
     CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                "ParamsDataContiguous failed.");
 
     auto result = l0op::SolveTri(params.x, params.cuSeqlens, params.chunkIndices,
                                   params.layout, params.xOut, executorPtr);
     CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
 
     auto viewCopyResult = l0op::ViewCopy(result, params.xOut, executorPtr);
     CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
 
     *workspaceSize = uniqueExecutor->GetWorkspaceSize();
     uniqueExecutor.ReleaseTo(executor);
     return ACLNN_SUCCESS;
 }
 
 aclnnStatus aclnnSolveTri(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
 {
     L2_DFX_PHASE_2(aclnnSolveTri);
     CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
                "SolveTri launch failed.");
     return ACLNN_SUCCESS;
 }
 
 #ifdef __cplusplus
 }
 #endif
