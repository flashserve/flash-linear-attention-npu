/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
#include "aclnn_chunk_gated_delta_rule_fwd_prepare.h"
#include "chunk_gated_delta_rule_fwd_prepare.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include <cstdint>

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkGatedDeltaRuleFwdPrepare";

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr int64_t K_SIZE = 128;
constexpr int64_t CHUNK_SIZE = 64;
constexpr int64_t DIM_4 = 4;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_1 = 1;

static bool IsA2PrepareSoc()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201;
}

struct ChunkGatedDeltaRuleFwdPrepareParams {
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
    bool outputA = true;
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

static bool IsGateDtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT || dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16;
}

static aclnnStatus CheckNotNull(const ChunkGatedDeltaRuleFwdPrepareParams &params)
{
    if (params.q == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.k == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.v == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "v");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.beta == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "beta");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.gOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "gOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.wOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "wOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.uOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "uOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.aOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "aOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    {
        const bool hasQHat = params.qHatOptional != nullptr;
        const bool hasKHat = params.kHatOptional != nullptr;
        const bool hasQRstd = params.qRstdOptional != nullptr;
        const bool hasKRstd = params.kRstdOptional != nullptr;
        if (!(hasQHat == hasKHat && hasQHat == hasQRstd && hasQHat == hasKRstd)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "qHat,kHat,qRstd,kRstd",
                std::string(hasQHat ? "qHat=present" : "qHat=absent") + "," +
                    (hasKHat ? "kHat=present" : "kHat=absent") + "," +
                    (hasQRstd ? "qRstd=present" : "qRstd=absent") + "," +
                    (hasKRstd ? "kRstd=present" : "kRstd=absent"),
                "qHat/kHat/qRstd/kRstd must all be set or all be nullptr");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.chunkSize != CHUNK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunkSize", std::to_string(params.chunkSize), "chunkSize must be 64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dtBiasOptional != nullptr && params.aLogOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "dtBiasOptional", "dtBiasOptional requires aLogOptional (use_gate_in_kernel)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.allowNegEigval && params.betaEffOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "betaEff", "allowNegEigval requires betaEff (use_beta_sigmoid_in_kernel)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional != nullptr) {
        if (params.chunkIndicesOptional == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, "chunkIndicesOptional",
                "varlen requires cuSeqlensOptional and chunkIndicesOptional together");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const ChunkGatedDeltaRuleFwdPrepareParams &params)
{
    if (params.q->GetDataType() != DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "q", ge::TypeUtils::DataTypeToSerialString(params.q->GetDataType()),
            "q must be bfloat16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.k->GetDataType() != params.q->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "k", ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()),
            "dtype must match q: " + ge::TypeUtils::DataTypeToSerialString(params.q->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.v->GetDataType() != params.q->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "v", ge::TypeUtils::DataTypeToSerialString(params.v->GetDataType()),
            "dtype must match q: " + ge::TypeUtils::DataTypeToSerialString(params.q->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsGateDtype(params.g->GetDataType())) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "g", ge::TypeUtils::DataTypeToSerialString(params.g->GetDataType()),
            "dtype must be fp32/bf16/fp16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsGateDtype(params.beta->GetDataType())) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "beta", ge::TypeUtils::DataTypeToSerialString(params.beta->GetDataType()),
            "dtype must be fp32/bf16/fp16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gOut->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "gOut", ge::TypeUtils::DataTypeToSerialString(params.gOut->GetDataType()),
            "gOut must be fp32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.wOut->GetDataType() != params.k->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "wOut", ge::TypeUtils::DataTypeToSerialString(params.wOut->GetDataType()),
            "dtype must match k: " + ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.uOut->GetDataType() != params.v->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "uOut", ge::TypeUtils::DataTypeToSerialString(params.uOut->GetDataType()),
            "dtype must match v: " + ge::TypeUtils::DataTypeToSerialString(params.v->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.aOut->GetDataType() != params.k->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "aOut", ge::TypeUtils::DataTypeToSerialString(params.aOut->GetDataType()),
            "dtype must match k: " + ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.aLogOptional != nullptr) {
        if (!IsGateDtype(params.aLogOptional->GetDataType())) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "aLogOptional", ge::TypeUtils::DataTypeToSerialString(params.aLogOptional->GetDataType()),
                "dtype must be fp32/bf16/fp16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dtBiasOptional != nullptr) {
        if (!IsGateDtype(params.dtBiasOptional->GetDataType())) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "dtBiasOptional", ge::TypeUtils::DataTypeToSerialString(params.dtBiasOptional->GetDataType()),
                "dtype must be fp32/bf16/fp16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.qHatOptional != nullptr) {
        if (params.qHatOptional->GetDataType() != params.q->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "qHatOptional", ge::TypeUtils::DataTypeToSerialString(params.qHatOptional->GetDataType()),
                "dtype must match q: " + ge::TypeUtils::DataTypeToSerialString(params.q->GetDataType()));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.kHatOptional->GetDataType() != params.k->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "kHatOptional", ge::TypeUtils::DataTypeToSerialString(params.kHatOptional->GetDataType()),
                "dtype must match k: " + ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.qRstdOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "qRstdOptional", ge::TypeUtils::DataTypeToSerialString(params.qRstdOptional->GetDataType()),
                "must be fp32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.kRstdOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "kRstdOptional", ge::TypeUtils::DataTypeToSerialString(params.kRstdOptional->GetDataType()),
                "must be fp32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.betaEffOptional != nullptr) {
        if (params.betaEffOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "betaEffOptional", ge::TypeUtils::DataTypeToSerialString(params.betaEffOptional->GetDataType()),
                "must be fp32");
            return ACLNN_ERR_PARAM_INVALID;
        }
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

static aclnnStatus CheckA2PrepareMetadata(const ChunkGatedDeltaRuleFwdPrepareParams &params, int64_t tokens)
{
    if ((params.cuSeqlensOptional == nullptr) != (params.chunkIndicesOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cuSeqlens,chunkIndices",
            std::string(params.cuSeqlensOptional == nullptr ? "cuSeqlens=absent" : "cuSeqlens=present") + "," +
                (params.chunkIndicesOptional == nullptr ? "chunkIndices=absent" : "chunkIndices=present"),
            "cuSeqlens and chunkIndices must be provided together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional == nullptr) return ACLNN_SUCCESS;
    const auto &cu = *params.cuSeqlensOptional;
    const auto &indices = *params.chunkIndicesOptional;
    if (!(cu.Size() >= 2 && cu[0] == 0 && cu[cu.Size() - 1] == tokens)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cuSeqlens[0],cuSeqlens[last]",
            std::to_string(cu.Size()) + " entries, first=" + std::to_string(cu[0]) +
                ", last=" + std::to_string(cu[cu.Size() - 1]),
            "cuSeqlens must start at 0 and end at T=" + std::to_string(tokens));
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t chunks = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        if (!(cu[seq] >= 0 && cu[seq] < cu[seq + 1] && cu[seq + 1] <= tokens)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "cuSeqlens[seq],cuSeqlens[seq+1]",
                std::to_string(cu[seq]) + "," + std::to_string(cu[seq + 1]),
                "cuSeqlens must be strictly increasing within [0,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const uint64_t length = static_cast<uint64_t>(cu[seq + 1] - cu[seq]);
        const uint64_t count = length / params.chunkSize + (length % params.chunkSize != 0);
        if (!(chunks <= static_cast<uint64_t>(INT64_MAX) - count)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "chunk_indices", std::to_string(chunks), "<= INT64_MAX - " + std::to_string(count));
        return ACLNN_ERR_PARAM_INVALID;
    }
        chunks += count;
    }
    if (!(indices.Size() % 2 == 0 && indices.Size() / 2 == chunks)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "chunkIndices", std::to_string(indices.Size()),
            "2 * chunk count = " + std::to_string(chunks * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t index = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        const int64_t length = cu[seq + 1] - cu[seq];
        const int64_t count = length / params.chunkSize + (length % params.chunkSize != 0);
        for (int64_t chunk = 0; chunk < count; ++chunk) {
            if (indices[index] != static_cast<int64_t>(seq) || indices[index + 1] != chunk) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    OP_NAME, "chunkIndices[index],chunkIndices[index+1]",
                    std::to_string(indices[index]) + "," + std::to_string(indices[index + 1]),
                    "chunkIndices must use canonical sequence-major order");
                return ACLNN_ERR_PARAM_INVALID;
            }
            index += 2;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const ChunkGatedDeltaRuleFwdPrepareParams &params)
{
    const op::Shape qShape = params.q->GetViewShape();
    const op::Shape kShape = params.k->GetViewShape();
    const op::Shape vShape = params.v->GetViewShape();
    const op::Shape gShape = params.g->GetViewShape();
    const op::Shape betaShape = params.beta->GetViewShape();
    if (qShape.GetDimNum() != DIM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "q", std::to_string(qShape.GetDimNum()), "q must be [B, Hk, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!ShapeEqual(kShape, qShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "k,q", std::string(op::ToString(kShape).GetString()),
            "k shape must match q");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (vShape.GetDimNum() != DIM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "v", std::to_string(vShape.GetDimNum()), "v must be [B, Hv, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gShape.GetDimNum() != DIM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "g", std::to_string(gShape.GetDimNum()), "g must be [B, Hv, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!ShapeEqual(betaShape, gShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "beta,g", std::string(op::ToString(betaShape).GetString()),
            "beta shape must match g");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t B = qShape.GetDim(0);
    const int64_t HK = qShape.GetDim(1);
    const int64_t T = qShape.GetDim(2);
    const int64_t K = qShape.GetDim(3);
    const int64_t HV = vShape.GetDim(1);
    const int64_t V = vShape.GetDim(3);
    if (IsA2PrepareSoc()) {
        if (!(B > 0 && HK > 0 && HV > 0 && T > 0)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "B,Hk,Hv,T",
                std::to_string(B) + "," + std::to_string(HK) + "," + std::to_string(HV) + "," +
                    std::to_string(T),
                "A2 Prepare requires positive B/Hk/Hv/T");
            return ACLNN_ERR_PARAM_INVALID;
        }
        {
            const aclnnStatus checkStatus = CheckA2PrepareMetadata(params, T);
            if (checkStatus != ACLNN_SUCCESS) {
                return checkStatus;
            }
        };
    }
    if (!(vShape.GetDim(0) == B && vShape.GetDim(2) == T)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "v.B,v.T", std::to_string(vShape.GetDim(0)) + "," + std::to_string(vShape.GetDim(2)),
            "v batch/seq must match q B=" + std::to_string(B) + ",T=" + std::to_string(T));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(gShape.GetDim(0) == B && gShape.GetDim(1) == HV && gShape.GetDim(2) == T)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "g", "[" + std::to_string(gShape.GetDim(0)) + "," + std::to_string(gShape.GetDim(1)) +
                "," + std::to_string(gShape.GetDim(2)) + "]",
            "[B,Hv,T] with B=" + std::to_string(B) + ",Hv=" + std::to_string(HV) +
                ",T=" + std::to_string(T));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (K != K_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "K", std::to_string(K), "K currently only supports " + std::to_string(K_SIZE));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(V == 128 || V == 256)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "V", std::to_string(V), "V must be 128 or 256");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HK > 0 && HV % HK == 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "Hk,Hv", std::to_string(HK) + "," + std::to_string(HV),
            "Hv must be divisible by Hk");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t ratio = HV / HK;
    if (!(ratio >= 1 && ratio <= (IsA2PrepareSoc() ? 16 : 4))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "Hv/Hk", std::to_string(ratio),
            IsA2PrepareSoc() ? "Hv/Hk must be in [1,16] on A2/A3" : "Hv/Hk must be in [1,4] on Ascend950");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (params.aLogOptional != nullptr) {
        const op::Shape aLogShape = params.aLogOptional->GetViewShape();
        if (!(aLogShape.GetDimNum() == DIM_1 && aLogShape.GetDim(0) == HV)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "aLogOptional", std::string(op::ToString(aLogShape).GetString()),
                "[Hv] with Hv=" + std::to_string(HV));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dtBiasOptional != nullptr) {
        const op::Shape dtShape = params.dtBiasOptional->GetViewShape();
        if (!(dtShape.GetDimNum() == DIM_1 && dtShape.GetDim(0) == HV)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "dtBiasOptional", std::string(op::ToString(dtShape).GetString()),
                "[Hv] with Hv=" + std::to_string(HV));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.cuSeqlensOptional != nullptr) {
        if (B != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "B", std::to_string(B), "varlen requires B=1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.qHatOptional != nullptr) {
        if (!ShapeEqual(params.qHatOptional->GetViewShape(), qShape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "qHat", std::string(op::ToString(params.qHatOptional->GetViewShape()).GetString()),
                "qHat shape must match q");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!ShapeEqual(params.kHatOptional->GetViewShape(), kShape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "kHat", std::string(op::ToString(params.kHatOptional->GetViewShape()).GetString()),
                "kHat shape must match k");
            return ACLNN_ERR_PARAM_INVALID;
        }
        {
            const op::Shape qRstd = params.qRstdOptional->GetViewShape();
            if (!(qRstd.GetDimNum() == DIM_3 && qRstd.GetDim(0) == B && qRstd.GetDim(1) == HK &&
                  qRstd.GetDim(2) == T)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    OP_NAME, "qRstd", std::string(op::ToString(qRstd).GetString()),
                    "[B,Hk,T]");
                return ACLNN_ERR_PARAM_INVALID;
            }
            const op::Shape kRstd = params.kRstdOptional->GetViewShape();
            if (!(kRstd.GetDimNum() == DIM_3 && kRstd.GetDim(0) == B && kRstd.GetDim(1) == HK &&
                  kRstd.GetDim(2) == T)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    OP_NAME, "kRstd", std::string(op::ToString(kRstd).GetString()),
                    "[B,Hk,T]");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (params.betaEffOptional != nullptr) {
        if (!ShapeEqual(params.betaEffOptional->GetViewShape(), gShape)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "betaEffOptional",
                std::string(op::ToString(params.betaEffOptional->GetViewShape()).GetString()),
                "betaEffOptional shape must match g");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!ShapeEqual(params.gOut->GetViewShape(), gShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "gOut", std::string(op::ToString(params.gOut->GetViewShape()).GetString()),
            "gOut shape must match g");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!ShapeEqual(params.uOut->GetViewShape(), vShape)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "uOut", std::string(op::ToString(params.uOut->GetViewShape()).GetString()),
            "uOut shape must match v");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const op::Shape wShape = params.wOut->GetViewShape();
    if (!(wShape.GetDimNum() == DIM_4 && wShape.GetDim(0) == B && wShape.GetDim(1) == HV &&
          wShape.GetDim(2) == T && wShape.GetDim(3) == K)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "wOut", std::string(op::ToString(wShape).GetString()), "[B,Hv,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const op::Shape aShape = params.aOut->GetViewShape();
    if (!(aShape.GetDimNum() == DIM_4 && aShape.GetDim(0) == B && aShape.GetDim(1) == HV &&
          aShape.GetDim(2) == T && aShape.GetDim(3) == params.chunkSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "aOut", std::string(op::ToString(aShape).GetString()), "[B,Hv,T,chunkSize]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus OptionalDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    return DataContiguous(tensor, executor);
}

static aclnnStatus ParamsDataContiguous(ChunkGatedDeltaRuleFwdPrepareParams &params, aclOpExecutor *executorPtr)
{
    if (!(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous q failed.", __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous k failed.", __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous v failed.", __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous g failed.", __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(DataContiguous(params.beta, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous beta failed.", __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(OptionalDataContiguous(params.aLogOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous aLog failed.", __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(OptionalDataContiguous(params.dtBiasOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous dtBias failed.", __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool A2PrepareElementCount(const op::Shape &shape, uint64_t &count)
{
    count = 1;
    if (shape.GetDimNum() == 0) return false;
    for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
        const int64_t dim = shape.GetDim(axis);
        if (dim <= 0 || count > static_cast<uint64_t>(INT64_MAX) / static_cast<uint64_t>(dim)) return false;
        count *= static_cast<uint64_t>(dim);
    }
    return true;
}

static const aclTensor *A2PrepareDenseInput(const aclTensor *tensor, bool fp32, aclOpExecutor *executor)
{
    if (tensor == nullptr) return nullptr;
    if (fp32 && tensor->GetDataType() != DataType::DT_FLOAT)
        tensor = l0op::Cast(tensor, DataType::DT_FLOAT, executor);
    if (tensor == nullptr) return nullptr;
    uint64_t count = 0, storageElements = 0;
    const int64_t base = tensor->GetViewOffset();
    if (!A2PrepareElementCount(tensor->GetViewShape(), count) ||
        !A2PrepareElementCount(tensor->GetStorageShape(), storageElements) || base < 0 ||
        static_cast<uint64_t>(base) > static_cast<uint64_t>(INT64_MAX) - count)
        return nullptr;
    // Storage shape may describe the logical view, not the backing allocation extent.
    // Preserve the offset when sharing storage; a contiguous slice needs no data copy.
    auto *view = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageShape(tensor->GetViewShape());
        view->SetOriginalShape(tensor->GetViewShape());
    }
    return view;
}

static aclnnStatus MakeA2PrepareCompute(const ChunkGatedDeltaRuleFwdPrepareParams &params,
                                        ChunkGatedDeltaRuleFwdPrepareParams &compute, aclOpExecutor *executor)
{
    compute = params;
    compute.q = A2PrepareDenseInput(params.q, false, executor);
    compute.k = A2PrepareDenseInput(params.k, false, executor);
    compute.v = A2PrepareDenseInput(params.v, false, executor);
    compute.g = A2PrepareDenseInput(params.g, true, executor);
    compute.beta = A2PrepareDenseInput(params.beta, true, executor);
    compute.aLogOptional = A2PrepareDenseInput(params.aLogOptional, true, executor);
    compute.dtBiasOptional = A2PrepareDenseInput(params.dtBiasOptional, true, executor);
    auto shadow = [executor](const aclTensor *output) -> const aclTensor * {
        return output == nullptr ? nullptr :
            executor->AllocTensor(output->GetViewShape(), output->GetDataType(), Format::FORMAT_ND);
    };
    // Prepare 内部写独立输出，全部 raw Q/K/g/beta 读取结束后才 ViewCopy 公开 view。
    compute.gOut = shadow(params.gOut); compute.wOut = shadow(params.wOut); compute.uOut = shadow(params.uOut);
    // outputA=false 时内核把 A 写入 checked scratch；此端口保持调用方占位且不写/不复制。
    compute.aOut = params.outputA ? shadow(params.aOut) : params.aOut;
    compute.qHatOptional = shadow(params.qHatOptional); compute.kHatOptional = shadow(params.kHatOptional);
    compute.qRstdOptional = shadow(params.qRstdOptional); compute.kRstdOptional = shadow(params.kRstdOptional);
    compute.betaEffOptional = shadow(params.betaEffOptional);
    if (!(compute.q && compute.k && compute.v && compute.g && compute.beta && compute.gOut && compute.wOut && compute.uOut && compute.aOut && (!params.aLogOptional || compute.aLogOptional) && (!params.dtBiasOptional || compute.dtBiasOptional) && (!params.qHatOptional || (compute.qHatOptional && compute.kHatOptional && compute.qRstdOptional && compute.kRstdOptional)) && (!params.betaEffOptional || compute.betaEffOptional))) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: (compute.q && compute.k && compute.v && compute.g && compute.beta && compute.gOut && compute.wOut && compute.uOut && compute.aOut && (!params.aLogOptional || compute.aLogOptional) && (!params.dtBiasOptional || compute.dtBiasOptional) && (!params.qHatOptional || (compute.qHatOptional && compute.kHatOptional && compute.qRstdOptional && compute.kRstdOptional)) && (!params.betaEffOptional || compute.betaEffOptional)) failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ViewCopyIfPresent(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    if (dst == nullptr || src == nullptr) {
        return ACLNN_SUCCESS;
    }
    auto copied = l0op::ViewCopy(src, dst, executor);
    if (!(copied != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: copied != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleFwdPrepareParams &params)
{
    {
        const aclnnStatus checkStatus = CheckNotNull(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckDtype(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckShape(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepareGetWorkspaceSize(
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
    bool outputA,
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
    ChunkGatedDeltaRuleFwdPrepareParams params{
        q, k, v, g, beta, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkIndicesOptional,
        chunkSize, allowNegEigval, useExp2, outputA, gOut, wOut, uOut, aOut,
        qHatOptional, kHatOptional, qRstdOptional, kRstdOptional, betaEffOptional};

    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwdPrepare,
                   DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional, cuSeqlensOptional,
                          chunkIndicesOptional, chunkSize, allowNegEigval, useExp2, outputA),
                   DFX_OUT(gOut, wOut, uOut, aOut, qHatOptional, kHatOptional, qRstdOptional,
                           kRstdOptional, betaEffOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();

    {
        const aclnnStatus checkStatus = CheckParams(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    if (!(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS)) {
        
        return ACLNN_ERR_PARAM_INVALID;
    }

    const bool a2 = IsA2PrepareSoc();
    auto compute = params;
    if (a2) {
        if (!(MakeA2PrepareCompute(params, compute, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: MakeA2PrepareCompute(params, compute, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    auto result = l0op::ChunkGatedDeltaRuleFwdPrepare(
        compute.q, compute.k, compute.v, compute.g, compute.beta, compute.aLogOptional, compute.dtBiasOptional,
        compute.cuSeqlensOptional, compute.chunkIndicesOptional, compute.chunkSize, compute.allowNegEigval,
        compute.useExp2, compute.qHatOptional != nullptr,
        compute.aLogOptional != nullptr, compute.betaEffOptional != nullptr, compute.outputA,
        compute.gOut, compute.wOut, compute.uOut, compute.aOut, compute.qHatOptional, compute.kHatOptional,
        compute.qRstdOptional, compute.kRstdOptional, compute.betaEffOptional, executorPtr);
    if (!(result[0] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(result[1] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(result[2] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(result[3] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!(ViewCopyIfPresent(result[0], params.gOut, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[0], params.gOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ViewCopyIfPresent(result[1], params.wOut, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[1], params.wOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ViewCopyIfPresent(result[2], params.uOut, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[2], params.uOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!a2 || params.outputA) {
        if (!(ViewCopyIfPresent(result[3], params.aOut, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[3], params.aOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (!(ViewCopyIfPresent(result[4], params.qHatOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[4], params.qHatOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ViewCopyIfPresent(result[5], params.kHatOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[5], params.kHatOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ViewCopyIfPresent(result[6], params.qRstdOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[6], params.qRstdOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ViewCopyIfPresent(result[7], params.kRstdOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[7], params.kRstdOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ViewCopyIfPresent(result[8], params.betaEffOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(result[8], params.betaEffOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepare(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleFwdPrepare);
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "This is an error in ChunkGatedDeltaRuleFwdPrepare launch aicore.", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
