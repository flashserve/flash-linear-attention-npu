#include "op_common/log/log.h"
#include "chunk_kda_bwd_finalize_tiling.h"

#include <cmath>
#include <cstring>
#include "graph/utils/type_utils.h"
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"
#include "chunk_kda_bwd_finalize_arch22.h"

namespace optiling {

namespace {
std::string ShapeText(const gert::StorageShape *storage)
{
    if (storage == nullptr) {
        return "absent";
    }
    const auto &shape = storage->GetStorageShape();
    std::string text = "rank=" + std::to_string(shape.GetDimNum()) + ",[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            text += ",";
        }
        text += std::to_string(shape.GetDim(i));
    }
    return text + "]";
}

ge::graphStatus FailShape(const char *node, const char *name, const gert::StorageShape *actual,
                          const std::string &expected)
{
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(node, name, ShapeText(actual), expected);
    return ge::GRAPH_FAILED;
}

ge::graphStatus FailShapes(const char *node, const char *names, const std::string &actual,
                           const std::string &reason)
{
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(node, names, actual, reason);
    return ge::GRAPH_FAILED;
}

ge::graphStatus FailDtype(const char *node, const char *name, ge::DataType actual,
                          const std::string &expected)
{
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
        node, name, ge::TypeUtils::DataTypeToSerialString(actual), expected);
    return ge::GRAPH_FAILED;
}

ge::graphStatus FailDtypes(const char *node, const char *names, const std::string &actual,
                           const std::string &reason)
{
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(node, names, actual, reason);
    return ge::GRAPH_FAILED;
}

ge::graphStatus FailMissingAttr(const char *node, const char *name)
{
    OP_LOGE_WITH_INVALID_INPUT(node, name);
    return ge::GRAPH_FAILED;
}

ge::graphStatus FailValue(const char *node, const char *name, const std::string &actual,
                          const std::string &reason)
{
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(node, name, actual, reason);
    return ge::GRAPH_FAILED;
}

ge::graphStatus FailValues(const char *node, const char *names, const std::string &actual,
                           const std::string &reason)
{
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(node, names, actual, reason);
    return ge::GRAPH_FAILED;
}

ge::graphStatus ValidateFinalize(
    gert::TilingContext *context, const ChunkKdaBwdFinalizeTilingContext &ctx,
    KDA::ChunkKdaBwdFinalizeTilingData &shape, bool arch22,
    bool useBetaSigmoid, bool allowNegEigval, const char *outputLayout)
{
    if (allowNegEigval && !useBetaSigmoid) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx.nodeName, "allowNegEigval",
            std::string("allowNegEigval=true,useBetaSigmoidInKernel=") + (useBetaSigmoid ? "true" : "false"),
            "allowNegEigval=true requires useBetaSigmoidInKernel=true");

        return ge::GRAPH_FAILED;
    }
    if (!arch22 && useBetaSigmoid) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx.nodeName, "use_beta_sigmoid_in_kernel", "true", "A5 finalize: use_beta_sigmoid_in_kernel=true is reserved but not supported");

        return ge::GRAPH_FAILED;
    }

    const bool variable = shape.isVariable != 0;
    const size_t rank = ctx.q->GetStorageShape().GetDimNum();
    const size_t base = rank == 3 ? 0 : 1;

    if (shape.NV > UINT32_MAX / (128 * sizeof(float)) ||
        static_cast<uint64_t>(shape.B) > INT64_MAX / 128 / sizeof(float) /
            static_cast<uint64_t>(shape.NV) / static_cast<uint64_t>(shape.T)) {
        return FailValues(ctx.nodeName, "B,T,NV",
            "B=" + std::to_string(shape.B) + ",T=" + std::to_string(shape.T) +
                ",NV=" + std::to_string(shape.NV),
            "tiling size exceeds the supported range");
    }

    const bool headMajor = std::strcmp(outputLayout, rank == 3 ? "NTD" : "BNSD") == 0;
    const bool tokenMajor = std::strcmp(outputLayout, rank == 3 ? "TND" : "BSND") == 0;
    if (!headMajor && !tokenMajor) {
        return FailValue(ctx.nodeName, "outputLayout", outputLayout,
            rank == 3 ? "rank-3 q requires NTD or TND" : "rank-4 q requires BNSD or BSND");
    }

    auto tokenShape = [&](const gert::StorageShape *storage, int64_t width, bool tm) {
        if (!storage || storage->GetStorageShape().GetDimNum() != rank) return false;
        const auto &s = storage->GetStorageShape();
        return (base == 0 || s.GetDim(0) == shape.B) &&
            s.GetDim(base) == (tm ? shape.T : shape.NV) &&
            s.GetDim(base + 1) == (tm ? shape.NV : shape.T) && s.GetDim(base + 2) == width;
    };
    auto scalarShape = [&](const gert::StorageShape *storage, bool tm) {
        if (!storage || storage->GetStorageShape().GetDimNum() != rank - 1) return false;
        const auto &s = storage->GetStorageShape();
        return (base == 0 || s.GetDim(0) == shape.B) && s.GetDim(base) == (tm ? shape.T : shape.NV) &&
            s.GetDim(base + 1) == (tm ? shape.NV : shape.T);
    };
    auto tokenText = [&](int64_t width, bool tm) {
        return std::string("[") + (rank == 3 ? "" : "B,") + (tm ? "T,NV," : "NV,T,") +
            std::to_string(width) + "]";
    };
    auto scalarText = [&](bool tm) {
        return std::string("[") + (rank == 3 ? "" : "B,") + (tm ? "T,NV]" : "NV,T]");
    };
    auto checkToken = [&](const char *name, const gert::StorageShape *storage, int64_t width,
                          bool tm) -> ge::graphStatus {
        if (tokenShape(storage, width, tm)) {
            return ge::GRAPH_SUCCESS;
        }
        return FailShape(ctx.nodeName, name, storage, tokenText(width, tm));
    };
    auto checkScalar = [&](const char *name, const gert::StorageShape *storage,
                           bool tm) -> ge::graphStatus {
        if (scalarShape(storage, tm)) {
            return ge::GRAPH_SUCCESS;
        }
        return FailShape(ctx.nodeName, name, storage, scalarText(tm));
    };

    const struct {
        size_t idx;
        const char *name;
        int64_t width;
        bool tm;
    } tokenInputs[] = {
        {INPUT_Q, "q", 128, shape.qkTokenMajor},
        {INPUT_K, "k", 128, shape.qkTokenMajor},
        {INPUT_V, "v", 128, shape.vTokenMajor},
        {INPUT_GK, "gk", 128, false},
        {INPUT_V_NEW, "v_new", 128, false},
        {INPUT_DV_SCAN, "dv_scan", 128, false},
        {INPUT_DQ_RAW, "dq_raw", 128, false},
        {INPUT_AKK, "akk", 64, false},
        {INPUT_D_AQK, "d_aqk", 64, false}};
    for (const auto &item : tokenInputs) {
        const ge::graphStatus status =
            checkToken(item.name, context->GetRequiredInputShape(item.idx), item.width, item.tm);
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
    }

    ge::graphStatus status = checkScalar("beta", ctx.beta, false);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    if (ctx.qRstd != nullptr) {
        status = checkScalar("q_rstd", ctx.qRstd, false);
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
        status = checkScalar("k_rstd", ctx.kRstd, false);
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
    }

    if (variable) {
        const auto *cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS);
        const auto *indicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES);
        if (cuTensor == nullptr) {
            return FailValue(ctx.nodeName, "cu_seqlens", "absent",
                "cu_seqlens is required when is_variable=1");
        }
        if (indicesTensor == nullptr) {
            return FailValue(ctx.nodeName, "chunk_indices", "absent",
                "chunk_indices is required when is_variable=1");
        }
        if (cuTensor->GetDataType() != ge::DT_INT64) {
            return FailDtype(ctx.nodeName, "cu_seqlens", cuTensor->GetDataType(), "[int64]");
        }
        if (indicesTensor->GetDataType() != ge::DT_INT64) {
            return FailDtype(ctx.nodeName, "chunk_indices", indicesTensor->GetDataType(), "[int64]");
        }
        const auto *cu = cuTensor->GetData<int64_t>();
        const auto *indices = indicesTensor->GetData<int64_t>();
        const int64_t sequences = shape.seqNum;
        if (cu == nullptr) {
            return FailValue(ctx.nodeName, "cu_seqlens", "absent", "cu_seqlens data pointer is null");
        }
        if (indices == nullptr) {
            return FailValue(ctx.nodeName, "chunk_indices", "absent", "chunk_indices data pointer is null");
        }
        if (cu[0] != 0) {
            return FailValues(ctx.nodeName, "cu_seqlens", "cu[0]=" + std::to_string(cu[0]),
                "cu_seqlens[0] must be 0");
        }
        if (cu[sequences] != shape.T) {
            return FailValues(ctx.nodeName, "cu_seqlens", "cu[-1]=" + std::to_string(cu[sequences]),
                "cu_seqlens[-1] must equal T=" + std::to_string(shape.T));
        }
        int64_t task = 0;
        for (int64_t seq = 0; seq < sequences; ++seq) {
            if (cu[seq] < 0 || cu[seq + 1] < cu[seq] || cu[seq + 1] > shape.T) {
                return FailValues(ctx.nodeName, "cu_seqlens",
                    "cu[" + std::to_string(seq) + "]=" + std::to_string(cu[seq]) +
                        ",cu[" + std::to_string(seq + 1) + "]=" + std::to_string(cu[seq + 1]),
                    "cu_seqlens must be non-decreasing and within [0,T]");
            }
            const int64_t count = (cu[seq + 1] - cu[seq] + 64 - 1) / 64;
            for (int64_t chunk = 0; chunk < count; ++chunk, ++task) {
                if (task >= shape.totalChunkNum || indices[2 * task] != seq ||
                    indices[2 * task + 1] != chunk) {
                    return FailValues(ctx.nodeName, "chunk_indices",
                        "task=" + std::to_string(task) + ",seq=" + std::to_string(seq) +
                            ",chunk=" + std::to_string(chunk),
                        "chunk_indices must be canonical sequence-major order");
                }
            }
        }
        if (task != shape.totalChunkNum) {
            return FailValues(ctx.nodeName, "chunk_indices",
                "task=" + std::to_string(task) +
                    ",totalChunkNum=" + std::to_string(shape.totalChunkNum),
                "chunk_indices length must match totalChunkNum");
        }
    }

    const struct {
        size_t idx;
        const char *name;
    } bf16Inputs[] = {
        {INPUT_Q, "q"}, {INPUT_K, "k"}, {INPUT_V, "v"}, {INPUT_AKK, "akk"},
        {INPUT_V_NEW, "v_new"}, {INPUT_H, "h"}, {INPUT_DH, "dh"}, {INPUT_DV_SCAN, "dv_scan"}};
    for (const auto &item : bf16Inputs) {
        const auto *desc = context->GetInputDesc(item.idx);
        if (desc == nullptr) {
            return FailValue(ctx.nodeName, item.name, "absent", "input descriptor is null");
        }
        if (desc->GetDataType() != ge::DT_BF16) {
            return FailDtype(ctx.nodeName, item.name, desc->GetDataType(), "[bfloat16]");
        }
    }
    const struct {
        size_t idx;
        const char *name;
    } fp32Optional[] = {
        {INPUT_GK, "gk"}, {INPUT_D_AQK, "d_aqk"}, {INPUT_DQ_RAW, "dq_raw"},
        {INPUT_Q_RSTD, "q_rstd"}, {INPUT_K_RSTD, "k_rstd"}, {INPUT_DT_BIAS, "dt_bias"}};
    for (const auto &item : fp32Optional) {
        const auto *desc = context->GetOptionalInputDesc(item.idx);
        if (desc != nullptr && desc->GetDataType() != ge::DT_FLOAT) {
            return FailDtype(ctx.nodeName, item.name, desc->GetDataType(), "[float32]");
        }
    }

    const auto betaType = context->GetInputDesc(INPUT_BETA)->GetDataType();
    const auto logType = context->GetInputDesc(INPUT_A_LOG)->GetDataType();
    if (betaType != ge::DT_FLOAT && (useBetaSigmoid || betaType != ge::DT_BF16)) {
        return FailDtype(ctx.nodeName, "beta", betaType, "[bfloat16, float32]");
    }
    if (logType != ge::DT_FLOAT) {
        return FailDtype(ctx.nodeName, "a_log", logType, "[float32]");
    }

    if (ctx.useGateInKernel) {
        const auto *rawDesc = context->GetOptionalInputDesc(INPUT_RAW_G);
        if (rawDesc == nullptr) {
            return FailValue(ctx.nodeName, "raw_g", "absent",
                "raw_g is required when use_gate_in_kernel=1");
        }
        const auto rawType = rawDesc->GetDataType();
        if (rawType != ge::DT_FLOAT && rawType != ge::DT_BF16) {
            return FailDtype(ctx.nodeName, "raw_g", rawType, "[bfloat16, float32]");
        }
        status = checkToken("raw_g", context->GetOptionalInputShape(INPUT_RAW_G), 128,
            shape.rawTokenMajor);
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
        const auto *aLogShape = context->GetOptionalInputShape(INPUT_A_LOG);
        if (aLogShape == nullptr || aLogShape->GetStorageShape().GetShapeSize() != shape.NV) {
            return FailValues(ctx.nodeName, "a_log", "size=" + std::to_string(
                aLogShape == nullptr ? -1 : aLogShape->GetStorageShape().GetShapeSize()),
                "a_log must contain NV=" + std::to_string(shape.NV) + " elements");
        }
        const auto *biasShape = context->GetOptionalInputShape(INPUT_DT_BIAS);
        if (biasShape == nullptr || biasShape->GetStorageShape().GetShapeSize() != shape.NV * 128) {
            return FailValues(ctx.nodeName, "dt_bias", "size=" + std::to_string(
                biasShape == nullptr ? -1 : biasShape->GetStorageShape().GetShapeSize()),
                "dt_bias must contain NV*128=" + std::to_string(shape.NV * 128) + " elements");
        }
    }

    const char *outputNames[3] = {"dq", "dk", "dv"};
    for (size_t i = 0; i < 3; ++i) {
        status = checkToken(outputNames[i], context->GetOutputShape(i), 128, tokenMajor);
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
    }
    status = checkScalar("d_beta", context->GetOutputShape(3), shape.rawTokenMajor || tokenMajor);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = checkToken("d_g", context->GetOutputShape(4), 128, shape.rawTokenMajor || tokenMajor);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    const auto qType = context->GetOutputDesc(0)->GetDataType();
    if (qType != ge::DT_BF16) {
        return FailDtype(ctx.nodeName, "dq", qType, "[bfloat16]");
    }
    const auto kType = context->GetOutputDesc(1)->GetDataType();
    if (kType != qType) {
        return FailDtypes(ctx.nodeName, "dq,dk",
            "dq=" + ge::TypeUtils::DataTypeToSerialString(qType) +
                ",dk=" + ge::TypeUtils::DataTypeToSerialString(kType),
            "dq and dk must have the same dtype");
    }
    const auto vType = context->GetOutputDesc(2)->GetDataType();
    if (vType != ge::DT_BF16) {
        return FailDtype(ctx.nodeName, "dv", vType, "[bfloat16]");
    }
    const auto dbType = context->GetOutputDesc(3)->GetDataType();
    // Only the A2/A3 total-interface model-layout path separates effective beta
    // storage from original beta gradient dtype. Standalone Finalize is unchanged.
    if (dbType != betaType &&
        !(arch22 && shape.rawTokenMajor && useBetaSigmoid &&
          betaType == ge::DT_FLOAT && dbType == ge::DT_BF16)) {
        return FailDtypes(ctx.nodeName, "beta,d_beta",
            "beta=" + ge::TypeUtils::DataTypeToSerialString(betaType) +
                ",d_beta=" + ge::TypeUtils::DataTypeToSerialString(dbType),
            "d_beta must match beta dtype");
    }
    const auto dgType = context->GetOutputDesc(4)->GetDataType();
    const auto rawType = context->GetInputDesc(INPUT_RAW_G)->GetDataType();
    if (dgType != rawType) {
        return FailDtypes(ctx.nodeName, "raw_g,d_g",
            "raw_g=" + ge::TypeUtils::DataTypeToSerialString(rawType) +
                ",d_g=" + ge::TypeUtils::DataTypeToSerialString(dgType),
            "d_g must match raw_g dtype");
    }
    const auto daLogType = context->GetOutputDesc(5)->GetDataType();
    if (daLogType != ge::DT_FLOAT) {
        return FailDtype(ctx.nodeName, "d_a_log", daLogType, "[float32]");
    }
    const auto ddtBiasType = context->GetOutputDesc(6)->GetDataType();
    if (ddtBiasType != ge::DT_FLOAT) {
        return FailDtype(ctx.nodeName, "d_dt_bias", ddtBiasType, "[float32]");
    }

    const char *shapePairs[2][2] = {{"a_log", "d_a_log"}, {"dt_bias", "d_dt_bias"}};
    for (size_t i = 0; i < 2; ++i) {
        const auto *inputShape = context->GetRequiredInputShape(INPUT_A_LOG + i);
        const auto *outputShape = context->GetOutputShape(5 + i);
        if (inputShape == nullptr || outputShape == nullptr ||
            inputShape->GetStorageShape().GetDimNum() != outputShape->GetStorageShape().GetDimNum()) {
            return FailShapes(ctx.nodeName, i == 0 ? "a_log,d_a_log" : "dt_bias,d_dt_bias",
                "input=" + ShapeText(inputShape) + ",output=" + ShapeText(outputShape),
                "input and gradient shape must match");
        }
        for (size_t axis = 0; axis < inputShape->GetStorageShape().GetDimNum(); ++axis) {
            if (inputShape->GetStorageShape().GetDim(axis) != outputShape->GetStorageShape().GetDim(axis)) {
                return FailShapes(ctx.nodeName, i == 0 ? "a_log,d_a_log" : "dt_bias,d_dt_bias",
                    "input=" + ShapeText(inputShape) + ",output=" + ShapeText(outputShape),
                    "input and gradient shape must match");
            }
        }
    }

    shape.outputTokenMajor = tokenMajor;
    shape.gateOutputTokenMajor = shape.rawTokenMajor || tokenMajor;
    shape.betaMode = useBetaSigmoid ? (allowNegEigval ? 2 : 1) : 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace


ge::graphStatus Tiling4ChunkKdaBwdFinalize(gert::TilingContext *context)
{
    KDA::ChunkKdaBwdFinalizeTilingData shape{};
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const double *scale = attrs->GetAttrPointer<double>(ATTR_SCALE);
    const double *lowerBound = attrs->GetAttrPointer<double>(ATTR_LOWER_BOUND);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const bool *safeGate = attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE);
    const bool *useGate = attrs->GetAttrPointer<bool>(ATTR_USE_GATE_IN_KERNEL);
    const bool *useExp2 = attrs->GetAttrPointer<bool>(ATTR_USE_EXP2);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(ATTR_STATE_V_FIRST);
    const bool *useBetaSigmoid = attrs->GetAttrPointer<bool>(ATTR_USE_BETA_SIGMOID);
    const bool *allowNegEigval = attrs->GetAttrPointer<bool>(ATTR_ALLOW_NEG_EIGVAL);
    const char *outputLayout = attrs->GetStr(ATTR_OUTPUT_LAYOUT);
    if (useBetaSigmoid == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "use_beta_sigmoid");
        return ge::GRAPH_FAILED;
    }
    if (allowNegEigval == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "allow_neg_eigval");
        return ge::GRAPH_FAILED;
    }
    if (outputLayout == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "output_layout");
        return ge::GRAPH_FAILED;
    }
    if (scale == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "scale");
        return ge::GRAPH_FAILED;
    }
    if (lowerBound == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "lower_bound");
        return ge::GRAPH_FAILED;
    }
    if (chunkSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "chunkSize");
        return ge::GRAPH_FAILED;
    }
    if (safeGate == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "safe_gate");
        return ge::GRAPH_FAILED;
    }
    if (useGate == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "use_gate");
        return ge::GRAPH_FAILED;
    }
    if (useExp2 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "use_exp2");
        return ge::GRAPH_FAILED;
    }
    if (stateVFirst == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "state_v_first");
        return ge::GRAPH_FAILED;
    }
    if (!std::isfinite(static_cast<float>(*scale)) || !std::isfinite(static_cast<float>(*lowerBound)) ||
        *lowerBound < -5.0 || static_cast<float>(*lowerBound) >= 0.0f) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "scale,lower_bound",
            std::to_string(static_cast<float>(*scale)) + "," +
                std::to_string(static_cast<float>(*lowerBound)),
            "scale/lower_bound must be finite");

        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    if (aicCoreNum == 0) {
        // The A5 aclnn direct-launch context may omit platform info.  This
        // checkpoint is A5-only and must still expose all 28 physical AICs.
        aicCoreNum = platform.GetCurNpuArch() == NpuArch::DAV_3510 ? 28U : 1U;
        OP_LOGW(context->GetNodeName(),
                "platform info did not report AIC count; fallback to %u", aicCoreNum);
    }
    const char *modelLayout = attrs->GetStr(10);
    const bool mixed = modelLayout && modelLayout[0];
    const bool modelTm = mixed && (std::strcmp(modelLayout,"BSND") == 0 || std::strcmp(modelLayout,"TND") == 0);
    if (mixed && !modelTm && std::strcmp(modelLayout,"BNSD") && std::strcmp(modelLayout,"NTD")) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "model_layout", modelLayout, "BSND/BNSD/TND/NTD, or empty for standalone Finalize");
        return ge::GRAPH_FAILED;
    }
    ChunkKdaBwdFinalizeTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(mixed ? INPUT_GK : INPUT_Q),
        context->GetRequiredInputShape(mixed ? INPUT_GK : INPUT_K),
        context->GetRequiredInputShape(mixed ? INPUT_GK : INPUT_V),
        context->GetRequiredInputShape(INPUT_GK),
        context->GetRequiredInputShape(INPUT_BETA),
        context->GetRequiredInputShape(INPUT_AKK),
        context->GetRequiredInputShape(INPUT_H),
        context->GetRequiredInputShape(INPUT_DH),
        context->GetRequiredInputShape(INPUT_DV_SCAN),
        context->GetOptionalInputShape(INPUT_Q_RSTD),
        context->GetOptionalInputShape(INPUT_K_RSTD),
        context->GetOptionalInputShape(INPUT_CU_SEQLENS),
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES),
        *scale, *lowerBound, *chunkSize, *safeGate, *useGate, *useExp2, *stateVFirst,
        aicCoreNum,
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize())};
    auto fail = [&]() { return ge::GRAPH_FAILED; };
    // Check dimensions before the shared shape processor multiplies them.
    if (!ctx.k || !ctx.h) return fail();
    const auto &kShape = ctx.k->GetStorageShape();
    const auto &hShape = ctx.h->GetStorageShape();
    uint64_t elements = 1;
    for (size_t i = 0; i < kShape.GetDimNum(); ++i) {
        const auto dim = kShape.GetDim(i);
        if (dim <= 0 || static_cast<uint64_t>(dim) > (INT64_MAX - 512) / 4U / elements) return fail();
        elements *= static_cast<uint64_t>(dim);
    }
    elements = 1;
    for (size_t i = 0; i < hShape.GetDimNum(); ++i) {
        const auto dim = hShape.GetDim(i);
        if (dim <= 0 || static_cast<uint64_t>(dim) > (INT64_MAX - 512) / 4U / elements) return fail();
        elements *= static_cast<uint64_t>(dim);
    }
    ChunkKdaBwdFinalizeTilingProcessor processor(ctx, shape);
    if (processor.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    shape.qkTokenMajor = modelTm && !ctx.qRstd;
    shape.vTokenMajor = modelTm;
    shape.rawTokenMajor = mixed;
    const auto npuArch = platform.GetCurNpuArch();
    OP_CHECK_IF(npuArch != NpuArch::DAV_2201 && npuArch != NpuArch::DAV_3510,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
    context->GetNodeName(), "platform",
    "unsupported NpuArch for ChunkKdaBwdFinalize tiling"),
                return ge::GRAPH_FAILED);
    const bool arch22 = npuArch == NpuArch::DAV_2201;
    if (ValidateFinalize(context, ctx, shape, arch22, *useBetaSigmoid,
                         *allowNegEigval, outputLayout) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (arch22) return TilingFinalizeArch22(context, ctx, shape);
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdFinalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = shape;
    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    rawTiling->SetDataSize(sizeof(KDA::ChunkKdaBwdFinalizeTilingData) + 8);
    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaBwdFinalize(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdFinalize)
    .Tiling(Tiling4ChunkKdaBwdFinalize)
    .TilingParse<ChunkKdaBwdFinalizeCompileInfo>(TilingPrepareForChunkKdaBwdFinalize);

} // namespace optiling
