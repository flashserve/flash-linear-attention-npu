// Load structured reporters before opdev's legacy logging macros. The opdev
// headers must retain their logging convention for CHECK_RET and DFX helpers.
#include "op_common/log/log.h"
#include "aclnn_chunk_kda_bwd_prepare.h"
#include "chunk_kda_bwd_prepare.h"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include "../chunk_kda_bwd_prepare_validation.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "graph/utils/type_utils.h"

using namespace op;

namespace {
constexpr const char *OP_NAME = "aclnnChunkKdaBwdPrepare";

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

aclnnStatus InvalidShape(const aclTensor *tensor, const char *name, const std::string &expected)
{
    const auto actual = ShapeText(tensor->GetViewShape());
    // The predefined message supplies the enclosing brackets itself.
    const auto shape = actual.substr(1, actual.size() - 2);
    const auto correct = expected.size() >= 2 && expected.front() == '[' && expected.back() == ']'
        ? expected.substr(1, expected.size() - 2) : expected;
    OP_LOGE_FOR_INVALID_SHAPE(OP_NAME, name, shape, correct);
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus CheckTensor(const aclTensor *tensor, DataType dtype, const char *name)
{
    if (tensor == nullptr) return Missing(name);
    if (tensor->GetDataType() != dtype) {
        OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, name,
            ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()),
            ge::TypeUtils::DataTypeToSerialString(dtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsContiguous(tensor)) return InvalidArgument(name, "must be contiguous");
    return ACLNN_SUCCESS;
}

aclnnStatus ValidateMetadata(
    const aclIntArray *cu, const aclIntArray *indices, int64_t totalTokens,
    int64_t chunkSize, int64_t expectedChunks)
{
    if (cu->Size() < 2 || cu->Size() > 1025) {
        return InvalidValue("cu_seqlens.size", std::to_string(cu->Size()), "[2,1025]");
    }
    if ((*cu)[0] != 0 || (*cu)[cu->Size() - 1] != totalTokens) {
        return InvalidArgument("cu_seqlens", "must start at 0 and end at T=" + std::to_string(totalTokens));
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
        const int64_t begin = (*cu)[seq], end = (*cu)[seq + 1];
        if (begin < 0 || end < begin || end > totalTokens) {
            return InvalidArgument("cu_seqlens", "must be nondecreasing within [0,T]; invalid pair at " +
                std::to_string(seq) + ": [" + std::to_string(begin) + "," + std::to_string(end) + "]");
        }
        totalChunks += (end - begin + chunkSize - 1) / chunkSize;
    }
    if (totalChunks != expectedChunks || totalChunks <= 0) {
        return InvalidValue("h.total_chunks", std::to_string(expectedChunks), std::to_string(totalChunks));
    }
    if (indices == nullptr) return ACLNN_SUCCESS;
    if (indices->Size() != static_cast<size_t>(2 * totalChunks)) {
        return InvalidValue("chunk_indices.size", std::to_string(indices->Size()), std::to_string(2 * totalChunks));
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
        const int64_t count = ((*cu)[seq + 1] - (*cu)[seq] + chunkSize - 1) / chunkSize;
        for (int64_t local = 0; local < count; ++local) {
            if ((*indices)[offset] != static_cast<int64_t>(seq) || (*indices)[offset + 1] != local) {
                return InvalidArgument("chunk_indices", "must use canonical sequence-major order; pair " +
                    std::to_string(offset / 2) + " must be (" + std::to_string(seq) + "," +
                    std::to_string(local) + ")");
            }
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Check(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO, const aclTensor *h,
    const aclIntArray *cu, const aclIntArray *indices, const char *layout, double scale, int64_t chunkSize,
    const aclTensor *dAqkOut, const aclTensor *dvOut, const aclTensor *dqRawOut)
{
    const aclTensor *tensors[] = {aqk, vNew, dO, h, dAqkOut, dvOut, dqRawOut};
    const char *names[] = {"aqk", "v_new", "d_o", "h", "d_aqk", "dv", "dq_raw"};
    const DataType types[] = {DataType::DT_BF16, DataType::DT_BF16, DataType::DT_BF16,
        DataType::DT_BF16, DataType::DT_FLOAT, DataType::DT_BF16, DataType::DT_FLOAT};
    for (size_t i = 0; i < 7; ++i) {
        const auto status = CheckTensor(tensors[i], types[i], names[i]);
        if (status != ACLNN_SUCCESS) return status;
    }
    if (!std::isfinite(scale) || !std::isfinite(static_cast<float>(scale))) {
        return InvalidValue("scale", std::to_string(scale), "a finite FP32 value");
    }
    if (chunkSize != 64) return InvalidValue("chunk_size", std::to_string(chunkSize), "64");
    if (layout == nullptr) return Missing("layout");
    uint32_t inputLayout;
    if (!KDA::ParseLayout(layout, inputLayout)) {
        return InvalidValue("layout", layout, "BNSD, BSND, NTD or TND");
    }
    const bool rank3 = KDA::Rank3Layout(inputLayout);
    const bool variable = cu != nullptr;
    if (!variable && indices != nullptr) {
        return InvalidArgument("chunk_indices", "requires cu_seqlens");
    }
    const auto a = aqk->GetViewShape(), v = vNew->GetViewShape(), o = dO->GetViewShape();
    const auto state = h->GetViewShape();
    const size_t rank = rank3 ? 3 : 4;
    const char *tokenPattern = rank3 ? "rank 3, positive H/T/D" : "rank 4, positive B/H/T/D";
    for (size_t i = 0; i < 3; ++i) {
        if (tensors[i]->GetViewShape().GetDimNum() != rank) {
            return InvalidShape(tensors[i], names[i], tokenPattern);
        }
    }
    if (state.GetDimNum() != rank + 1) {
        return InvalidShape(h, "h", rank3 ? "[Nc,H,128,128]" : "[B,Nc,H,128,128]");
    }
    KDA::TokenShape aShape, vShape, oShape;
    if (!KDA::ParseTokenShape(a, inputLayout, aShape)) return InvalidShape(aqk, "aqk", tokenPattern);
    if (!KDA::ParseTokenShape(v, inputLayout, vShape)) return InvalidShape(vNew, "v_new", tokenPattern);
    if (!KDA::ParseTokenShape(o, inputLayout, oShape)) return InvalidShape(dO, "d_o", tokenPattern);
    if (aShape.D != 64) return InvalidValue("aqk.D", std::to_string(aShape.D), "64");
    for (size_t i = 1; i < 3; ++i) {
        const auto &s = i == 1 ? vShape : oShape;
        if (s.B != aShape.B || s.H != aShape.H || s.T != aShape.T || s.D != 128) {
            auto expected = a;
            expected.SetDim(rank - 1, 128);
            return InvalidShape(tensors[i], names[i], ShapeText(expected));
        }
    }
    const int64_t B = aShape.B, NV = aShape.H, T = aShape.T;
    if (variable && B != 1) return InvalidValue("B", std::to_string(B), "1 for variable lengths");
    const int64_t chunkCount = rank3 ? state.GetDim(0) : state.GetDim(1);
    const bool stateValid = rank3
        ? state.GetDim(1) == NV && state.GetDim(2) == 128 && state.GetDim(3) == 128
        : state.GetDim(0) == B && state.GetDim(2) == NV &&
              state.GetDim(3) == 128 && state.GetDim(4) == 128;
    if (!stateValid) {
        return InvalidShape(h, "h",
            (rank3 ? "[" : "[" + std::to_string(B) + ",") +
            "Nc," + std::to_string(NV) + ",128,128]");
    }
    if (variable) {
        const auto status = ValidateMetadata(cu, indices, T, chunkSize, chunkCount);
        if (status != ACLNN_SUCCESS) return status;
    } else if (chunkCount != (T + chunkSize - 1) / chunkSize) {
        return InvalidValue("h.chunks", std::to_string(chunkCount), std::to_string((T + 63) / 64));
    }
    for (size_t i = 4; i < 7; ++i) {
        const int64_t width = i == 4 ? 64 : 128;
        if (!KDA::HeadOutputShape(tensors[i]->GetViewShape(), aShape, width, rank3)) {
            return InvalidShape(tensors[i], names[i],
                (rank3 ? "[" : "[" + std::to_string(B) + ",") + std::to_string(NV) + "," +
                std::to_string(T) + "," + std::to_string(width) + "]");
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdPrepareGetWorkspaceSize(
    const aclTensor *aqk, const aclTensor *vNew, const aclTensor *dO, const aclTensor *h,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, const char *layout,
    double scale, int64_t chunkSize, bool stateVFirst,
    const aclTensor *dAqkOut, const aclTensor *dvOut, const aclTensor *dqRawOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // DFX also checks these pointers, but its generic error omits the operator
    // and parameter names. Report them before entering the DFX helper.
    if (workspaceSize == nullptr) return Missing("workspaceSize");
    if (executor == nullptr) return Missing("executor");
    L2_DFX_PHASE_1(aclnnChunkKdaBwdPrepare,
                   DFX_IN(aqk, vNew, dO, h, cuSeqlensOptional, chunkIndicesOptional,
                          layout, scale, chunkSize, stateVFirst),
                   DFX_OUT(dAqkOut, dvOut, dqRawOut));
    const auto status = Check(aqk, vNew, dO, h, cuSeqlensOptional, chunkIndicesOptional,
                             layout, scale, chunkSize, dAqkOut, dvOut, dqRawOut);
    if (status != ACLNN_SUCCESS) return status;
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto result = l0op::ChunkKdaBwdPrepare(
        aqk, vNew, dO, h, cuSeqlensOptional, chunkIndicesOptional,
        layout, scale, chunkSize, stateVFirst, dAqkOut, dvOut, dqRawOut,
        uniqueExecutor.get());
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdPrepare(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdPrepare);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
