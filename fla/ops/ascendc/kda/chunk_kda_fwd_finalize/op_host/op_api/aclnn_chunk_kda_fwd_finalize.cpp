#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
#include "aclnn_chunk_kda_fwd_finalize.h"
#include "chunk_kda_fwd_finalize.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkKdaFwdFinalize";

namespace {

constexpr int64_t FINALIZE_CHUNK_ROWS = 64;

struct FinalizeParams {
    const aclTensor *qgScaled;
    const aclTensor *aqk;
    const aclTensor *vNew;
    const aclTensor *h;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    const char *outputLayout;
    bool stateVFirst;
    const aclTensor *attnOut;
};

struct FinalizeShape {
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t seqLen = 0;
    int64_t totalChunks = 0;
    bool packed = false;
    bool sequenceMajor = false;
};

bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> dims)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (tensor->GetViewShape().GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

aclnnStatus ResolveLayout(const FinalizeParams &params, FinalizeShape &shape)
{
    if (params.outputLayout == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "outputLayout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    shape.packed = std::strcmp(params.outputLayout, "NTD") == 0 ||
                   std::strcmp(params.outputLayout, "TND") == 0;
    shape.sequenceMajor = std::strcmp(params.outputLayout, "BSND") == 0 ||
                          std::strcmp(params.outputLayout, "TND") == 0;
    if (!(shape.packed || shape.sequenceMajor || std::strcmp(params.outputLayout, "BNSD") == 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "outputLayout", params.outputLayout,
            "outputLayout must be one of BSND/BNSD/TND/NTD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypeAndFormat(const FinalizeParams &params)
{
    const aclTensor *tensors[] = {
        params.qgScaled, params.aqk, params.vNew, params.h, params.attnOut};
    const char *names[] = {
        "qgScaled", "aqk", "vNew", "h", "attnOut"};
    for (size_t index = 0; index < 5; ++index) {
        if (tensors[index]->GetDataType() != DataType::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, names[index],
                ge::TypeUtils::DataTypeToSerialString(tensors[index]->GetDataType()),
                "only BF16 is supported");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto storage = tensors[index]->GetStorageFormat();
        const auto view = tensors[index]->GetViewFormat();
        if (!(storage == Format::FORMAT_ND && view == Format::FORMAT_ND && !IsPrivateFormat(storage))) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                OP_NAME, names[index],
                std::to_string(static_cast<int>(storage)) + "/" +
                    std::to_string(static_cast<int>(view)),
                "non-private ND storage/view format only");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!IsContiguous(params.attnOut)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "attnOut", "attnOut is written by kernel and must be contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveInputShape(const FinalizeParams &params,
                              FinalizeShape &shape)
{
    const auto &q = params.qgScaled->GetViewShape();
    if (shape.packed) {
        if (!(q.GetDimNum() == 3 && q.GetDim(2) == 128)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "qgScaled", std::string(op::ToString(q).GetString()),
                "head-major [HV,T,128] for NTD/TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        shape.batch = 1;
        shape.heads = q.GetDim(0);
        shape.seqLen = q.GetDim(1);
        if (!(HasShape(params.aqk, {shape.heads, shape.seqLen, 64}) &&
              (HasShape(params.vNew, {shape.heads, shape.seqLen, 128}) ||
               HasShape(params.vNew, {1, shape.heads, shape.seqLen, 128})))) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "aqk,vNew",
                "aqk=" + std::string(op::ToString(params.aqk->GetViewShape()).GetString()) +
                    ",vNew=" + std::string(op::ToString(params.vNew->GetViewShape()).GetString()),
                "packed requires Aqk=[HV,T,64] and vNew=[HV,T,128] or [1,HV,T,128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (!(q.GetDimNum() == 4 && q.GetDim(3) == 128)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "qgScaled", std::string(op::ToString(q).GetString()),
                "head-major [B,HV,T,128] for BNSD/BSND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        shape.batch = q.GetDim(0);
        shape.heads = q.GetDim(1);
        shape.seqLen = q.GetDim(2);
        if (!(HasShape(params.aqk, {shape.batch, shape.heads, shape.seqLen, 64}) &&
              HasShape(params.vNew, {shape.batch, shape.heads, shape.seqLen, 128}))) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "aqk,vNew",
                "aqk=" + std::string(op::ToString(params.aqk->GetViewShape()).GetString()) +
                    ",vNew=" + std::string(op::ToString(params.vNew->GetViewShape()).GetString()),
                "dense requires Aqk=[B,HV,T,64] and vNew=[B,HV,T,128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!(shape.batch > 0 && shape.heads > 0 && shape.seqLen > 0 &&
          shape.batch <= std::numeric_limits<uint32_t>::max() &&
          shape.heads <= std::numeric_limits<uint32_t>::max() &&
          shape.seqLen <= std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "B,HV,T",
            std::to_string(shape.batch) + "," + std::to_string(shape.heads) + "," +
                std::to_string(shape.seqLen),
            "B/HV/T must be positive and representable by uint32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 设备端以 64 位元素索引计算 head-major 数据和 H 偏移。
    const uint64_t b = static_cast<uint64_t>(shape.batch);
    const uint64_t hv = static_cast<uint64_t>(shape.heads);
    const uint64_t t = static_cast<uint64_t>(shape.seqLen);
    if (!(b <= std::numeric_limits<uint64_t>::max() / hv / t / 128)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "qgScaled", "B*HV*T*128 exceeds uint64 element index range");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveSequenceShape(const FinalizeParams &params,
                                 FinalizeShape &shape)
{
    if (params.cuSeqlens == nullptr) {
        if (params.chunkIndices != nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "chunkIndicesOptional");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        shape.totalChunks = (shape.seqLen - 1) / FINALIZE_CHUNK_ROWS + 1;
    } else {
        if (shape.batch != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "B", std::to_string(shape.batch), "varlen input requires B=1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto *cu = params.cuSeqlens;
        if (!(cu->Size() >= 2 && (*cu)[0] == 0 && (*cu)[cu->Size() - 1] == shape.seqLen)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "cuSeqlensOptional[0],cuSeqlensOptional[last]",
                std::to_string(cu->Size()) + " entries, first=" + std::to_string((*cu)[0]) +
                    ", last=" + std::to_string((*cu)[cu->Size() - 1]),
                "cuSeqlensOptional must start at 0 and end at T=" + std::to_string(shape.seqLen) +
                    " with at least 2 entries");
            return ACLNN_ERR_PARAM_INVALID;
        }
        uint64_t total = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
            if (!(length > 0 && (*cu)[seq + 1] <= shape.seqLen)) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    OP_NAME, "cuSeqlensOptional[seq],cuSeqlensOptional[seq+1]",
                    std::to_string((*cu)[seq]) + "," + std::to_string((*cu)[seq + 1]),
                    "cuSeqlensOptional must be strictly increasing and bounded by T");
                return ACLNN_ERR_PARAM_INVALID;
            }
            total += static_cast<uint64_t>((length - 1) / FINALIZE_CHUNK_ROWS + 1);
            if (total > std::numeric_limits<uint32_t>::max()) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    OP_NAME, "totalChunks", std::to_string(total),
                    "varlen total_chunks must fit in uint32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (cu->Size() - 1 > std::numeric_limits<uint32_t>::max()) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "cuSeqlensOptional", std::to_string(cu->Size() - 1),
                "varlen sequence count must fit in uint32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        shape.totalChunks = static_cast<int64_t>(total);
        if (params.chunkIndices != nullptr) {
            if (params.chunkIndices->Size() != total * 2) {
                OP_LOGE_FOR_INVALID_LISTSIZE(
                    OP_NAME, "chunkIndicesOptional",
                    std::to_string(params.chunkIndices->Size()),
                    std::to_string(total * 2));
                return ACLNN_ERR_PARAM_INVALID;
            }
            size_t offset = 0;
            for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
                const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
                const int64_t chunks = (length - 1) / FINALIZE_CHUNK_ROWS + 1;
                for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                    if ((*params.chunkIndices)[offset] != static_cast<int64_t>(seq) ||
                        (*params.chunkIndices)[offset + 1] != chunk) {
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            OP_NAME, "chunkIndicesOptional[offset],chunkIndicesOptional[offset+1]",
                            std::to_string((*params.chunkIndices)[offset]) + "," +
                                std::to_string((*params.chunkIndices)[offset + 1]),
                            "chunkIndicesOptional must be canonical sequence-major (seq,chunk) pairs");
                        return ACLNN_ERR_PARAM_INVALID;
                    }
                    offset += 2;
                }
            }
        }
    }
    if (!HasShape(params.h, {shape.batch, shape.totalChunks, shape.heads, 128, 128})) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "h", std::string(op::ToString(params.h->GetViewShape()).GetString()),
            "[B,C,HV,128,128] with B=" + std::to_string(shape.batch) +
                ",HV=" + std::to_string(shape.heads) + ",C=" + std::to_string(shape.totalChunks));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const uint64_t b = static_cast<uint64_t>(shape.batch);
    const uint64_t hv = static_cast<uint64_t>(shape.heads);
    const uint64_t c = static_cast<uint64_t>(shape.totalChunks);
    if (!(b <= std::numeric_limits<uint64_t>::max() / hv / c / 128 / 128)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "h", "B*HV*C*128*128 exceeds uint64 element index range");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShape(const FinalizeParams &params,
                             const FinalizeShape &shape)
{
    const bool shapeValid = shape.packed
        ? (shape.sequenceMajor
            ? HasShape(params.attnOut, {shape.seqLen, shape.heads, 128})
            : HasShape(params.attnOut, {shape.heads, shape.seqLen, 128}))
        : (shape.sequenceMajor
            ? HasShape(params.attnOut, {shape.batch, shape.seqLen, shape.heads, 128})
            : HasShape(params.attnOut, {shape.batch, shape.heads, shape.seqLen, 128}));
    if (!shapeValid) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "attnOut", std::string(op::ToString(params.attnOut->GetViewShape()).GetString()),
            std::string("attnOut shape must match outputLayout=") + params.outputLayout +
                " with B=" + std::to_string(shape.batch) + ",HV=" + std::to_string(shape.heads) +
                ",T=" + std::to_string(shape.seqLen) + ",V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const FinalizeParams &params)
{
    if (!(params.qgScaled != nullptr && params.aqk != nullptr && params.vNew != nullptr &&
          params.h != nullptr && params.attnOut != nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "qgScaled,aqk,vNew,h,attnOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    FinalizeShape shape;
    aclnnStatus status = ResolveLayout(params, shape);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckDtypeAndFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = ResolveInputShape(params, shape);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = ResolveSequenceShape(params, shape);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckOutputShape(params, shape);
}

aclnnStatus MakeInputsContiguous(FinalizeParams &params,
                                 aclOpExecutor *executor)
{
    const aclTensor **inputs[] = {
        &params.qgScaled, &params.aqk, &params.vNew, &params.h};
    for (const aclTensor **input : inputs) {
        if (IsContiguous(*input)) {
            continue;
        }
        *input = l0op::Contiguous(*input, executor);
        if (!(*input != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: *input != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKdaFwdFinalizeGetWorkspaceSize(
    const aclTensor *qgScaled,
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *outputLayout,
    bool stateVFirst,
    const aclTensor *attnOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize,executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(
        aclnnChunkKdaFwdFinalize,
        DFX_IN(qgScaled, aqk, vNew, h, cuSeqlensOptional,
               chunkIndicesOptional, outputLayout, stateVFirst),
        DFX_OUT(attnOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    FinalizeParams params{qgScaled, aqk, vNew, h, cuSeqlensOptional,
                          chunkIndicesOptional, outputLayout, stateVFirst,
                          attnOut};
    aclnnStatus status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeInputsContiguous(params, uniqueExecutor.get());
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    const aclTensor *result = l0op::ChunkKdaFwdFinalize(
        params.qgScaled, params.aqk, params.vNew, params.h,
        params.cuSeqlens, params.chunkIndices, params.outputLayout,
        params.stateVFirst, params.attnOut, uniqueExecutor.get());
    if (!(result != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: result != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdFinalize(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwdFinalize);
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor,
                                   stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkKdaFwdFinalize AI Core 启动失败。", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
