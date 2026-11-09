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

#include "../../../../common/host_utils/log.h"

using namespace op;

namespace {

constexpr const char *CHUNK_KDA_FWD_FINALIZE_ENTITY = "aclnnChunkKdaFwdFinalize";
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
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "outputLayout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    shape.packed = std::strcmp(params.outputLayout, "NTD") == 0 ||
                   std::strcmp(params.outputLayout, "TND") == 0;
    shape.sequenceMajor = std::strcmp(params.outputLayout, "BSND") == 0 ||
                          std::strcmp(params.outputLayout, "TND") == 0;
    if (!(shape.packed || shape.sequenceMajor || std::strcmp(params.outputLayout, "BNSD") == 0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "outputLayout", params.outputLayout, "BSND, BNSD, TND or NTD");
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
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, names[index],
                fla_npu::host_utils::DfxDtypeString(tensors[index]->GetDataType()), "BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto storage = tensors[index]->GetStorageFormat();
        const auto view = tensors[index]->GetViewFormat();
        if (!(storage == Format::FORMAT_ND && view == Format::FORMAT_ND && !IsPrivateFormat(storage))) {
            OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, names[index],
                "storage=" + std::to_string(static_cast<int>(storage)) + ",view=" + std::to_string(static_cast<int>(view)),
                "non-private ND");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!IsContiguous(params.attnOut)) {
        OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "attnOut", "non-contiguous", "contiguous");
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
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "qgScaled",
                "rank=" + std::to_string(q.GetDimNum()), "head-major [HV,T,128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        shape.batch = 1;
        shape.heads = q.GetDim(0);
        shape.seqLen = q.GetDim(1);
        if (!(HasShape(params.aqk, {shape.heads, shape.seqLen, 64}) &&
              (HasShape(params.vNew, {shape.heads, shape.seqLen, 128}) ||
               HasShape(params.vNew, {1, shape.heads, shape.seqLen, 128})))) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "aqk,vNew", "mismatch",
                "Aqk=[HV,T,64], vNew=[HV,T,128] or [1,HV,T,128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (!(q.GetDimNum() == 4 && q.GetDim(3) == 128)) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "qgScaled",
                "rank=" + std::to_string(q.GetDimNum()), "head-major [B,HV,T,128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        shape.batch = q.GetDim(0);
        shape.heads = q.GetDim(1);
        shape.seqLen = q.GetDim(2);
        if (!(HasShape(params.aqk, {shape.batch, shape.heads, shape.seqLen, 64}) &&
              HasShape(params.vNew, {shape.batch, shape.heads, shape.seqLen, 128}))) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "aqk,vNew", "mismatch",
                "Aqk=[B,HV,T,64], vNew=[B,HV,T,128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!(shape.batch > 0 && shape.heads > 0 && shape.seqLen > 0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "qgScaled.shape[B/HV/T]",
            "B=" + std::to_string(shape.batch) + ",HV=" + std::to_string(shape.heads) + ",T=" + std::to_string(shape.seqLen),
            "positive");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 设备端以 64 位元素索引计算 head-major 数据和 H 偏移。
    const uint64_t b = static_cast<uint64_t>(shape.batch);
    const uint64_t hv = static_cast<uint64_t>(shape.heads);
    const uint64_t t = static_cast<uint64_t>(shape.seqLen);
    if (b > std::numeric_limits<uint64_t>::max() / hv / t / 128) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "qgScaled.shape", "overflow", "B*HV*T*128 in uint64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveSequenceShape(const FinalizeParams &params,
                                 FinalizeShape &shape)
{
    if (params.cuSeqlens == nullptr) {
        if (params.chunkIndices != nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "chunkIndicesOptional", "present",
                "requires cuSeqlensOptional");
            return ACLNN_ERR_PARAM_INVALID;
        }
        shape.totalChunks = (shape.seqLen - 1) / FINALIZE_CHUNK_ROWS + 1;
    } else {
        if (shape.batch != 1) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "qgScaled.shape[B]", std::to_string(shape.batch), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const auto *cu = params.cuSeqlens;
        if (!(cu->Size() >= 2 && (*cu)[0] == 0 && (*cu)[cu->Size() - 1] == shape.seqLen)) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "cuSeqlensOptional", "invalid bounds",
                "from 0 to T=" + std::to_string(shape.seqLen));
            return ACLNN_ERR_PARAM_INVALID;
        }
        uint64_t total = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
            if (!(length > 0 && (*cu)[seq + 1] <= shape.seqLen)) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "cuSeqlensOptional", "non-increasing", "strictly increasing");
                return ACLNN_ERR_PARAM_INVALID;
            }
            total += static_cast<uint64_t>((length - 1) / FINALIZE_CHUNK_ROWS + 1);
            if (total > std::numeric_limits<uint32_t>::max()) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "cuSeqlensOptional", "overflow", "total_chunks in uint32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (cu->Size() - 1 > std::numeric_limits<uint32_t>::max()) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "cuSeqlensOptional", "overflow", "sequence count in uint32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        shape.totalChunks = static_cast<int64_t>(total);
        if (params.chunkIndices != nullptr) {
            if (params.chunkIndices->Size() != total * 2) {
                OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "chunkIndicesOptional",
                    std::to_string(params.chunkIndices->Size()), std::to_string(total * 2));
                return ACLNN_ERR_PARAM_INVALID;
            }
            size_t offset = 0;
            for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
                const int64_t length = (*cu)[seq + 1] - (*cu)[seq];
                const int64_t chunks = (length - 1) / FINALIZE_CHUNK_ROWS + 1;
                for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                    if (!((*params.chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                          (*params.chunkIndices)[offset + 1] == chunk)) {
                        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "chunkIndicesOptional", "non-canonical",
                            "sequence-major (seq,chunk)");
                        return ACLNN_ERR_PARAM_INVALID;
                    }
                    offset += 2;
                }
            }
        }
    }
    if (!HasShape(params.h, {shape.batch, shape.totalChunks, shape.heads, 128, 128})) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "h", "mismatch",
            "[B=" + std::to_string(shape.batch) + ",C=" + std::to_string(shape.totalChunks) + ",HV=" + std::to_string(shape.heads) + ",128,128]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const uint64_t b = static_cast<uint64_t>(shape.batch);
    const uint64_t hv = static_cast<uint64_t>(shape.heads);
    const uint64_t c = static_cast<uint64_t>(shape.totalChunks);
    if (b > std::numeric_limits<uint64_t>::max() / hv / c / 128 / 128) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "h.shape", "overflow", "B*HV*C*128*128 in uint64");
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
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "attnOut", "mismatch",
            "outputLayout=" + std::string(params.outputLayout) + ", B=" + std::to_string(shape.batch) +
                ",HV=" + std::to_string(shape.heads) + ",T=" + std::to_string(shape.seqLen) + ",V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const FinalizeParams &params)
{
    const aclTensor *requiredT[] = {params.qgScaled, params.aqk, params.vNew, params.h, params.attnOut};
    const char *requiredN[] = {"qgScaled", "aqk", "vNew", "h", "attnOut"};
    for (size_t i = 0; i < sizeof(requiredT) / sizeof(requiredT[0]); ++i) {
        if (requiredT[i] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, requiredN[i]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
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
        CHECK_RET(*input != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_FWD_FINALIZE_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(
        aclnnChunkKdaFwdFinalize,
        DFX_IN(qgScaled, aqk, vNew, h, cuSeqlensOptional,
               chunkIndicesOptional, outputLayout, stateVFirst),
        DFX_OUT(attnOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
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
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor,
                                   stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER,
               "ChunkKdaFwdFinalize AI Core 启动失败。");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
