/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "op_common/log/log.h"
#include "chunk_kda_fwd_prepare_tiling.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include "graph/utils/type_utils.h"

#include "chunk_kda_fwd_prepare_tiling_processor.h"
#include "../op_kernel/chunk_kda_fwd_prepare_tiling_key.h"
#include "../op_kernel/chunk_kda_fwd_prepare_runtime_key.h"
#include "../op_kernel/chunk_kda_fwd_prepare_runtime_policy.h"
#include "../op_kernel/arch22/chunk_kda_fwd_prepare_resource.h"
#include "platform/soc_spec.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace {

constexpr int64_t PREPARE_CHUNK_SIZE = 64;
constexpr int64_t PREPARE_K_DIM = 128;
constexpr int64_t PREPARE_V_DIM = 128;
constexpr uint32_t PREPARE_MIX_BATCH_MODE = 1;

static_assert(PREPARE_OUTPUT_MODE_NONE == CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE);
static_assert(PREPARE_OUTPUT_MODE_RECOMPUTE ==
              CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE);
static_assert(PREPARE_OUTPUT_MODE_SAVE == CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE);
static_assert(PREPARE_OUTPUT_MODE_FORWARD ==
              CHUNK_KDA_FWD_PREPARE_OUTPUT_FORWARD);

enum class PrepareLayout {
    BNSD,
    BSND,
    NTD,
    TND,
};

struct PrepareShapeInfo {
    bool packed = false;
    bool sequenceMajor = false;
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t qkHeadNum = 0;
    int64_t valueHeadNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
};

template <typename T>
std::string ToText(T value)
{
    std::ostringstream text;
    text << value;
    return text.str();
}

std::string ShapeText(const gert::Shape &shape)
{
    std::string text = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0)
            text += ',';
        text += ToText(shape.GetDim(i));
    }
    return text + ']';
}

std::string DtypeText(ge::DataType type)
{
    return ge::TypeUtils::DataTypeToSerialString(type);
}

std::string LogicalText(const PrepareShapeInfo &shape)
{
    return "B=" + ToText(shape.batch) + ",T=" + ToText(shape.seqLen) + ",HK=" + ToText(shape.qkHeadNum) +
           ",HV=" + ToText(shape.valueHeadNum) + ",K=" + ToText(shape.kDim) + ",V=" + ToText(shape.vDim);
}

void ReportTilingError(gert::TilingContext *context, const std::string &reason)
{
    const char *name =
        context == nullptr || context->GetNodeName() == nullptr ? "ChunkKdaFwdPrepare" : context->GetNodeName();
    OP_LOGE_LIBOPAPI_REPORT(name, "%s", reason.c_str());
    const std::vector<const char *> keys = {"opName", "reason"};
    const std::vector<const char *> values = {name, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}

bool ParseLayout(const char *layout, PrepareLayout &parsed)
{
    if (layout == nullptr || std::strcmp(layout, "BNSD") == 0) {
        parsed = PrepareLayout::BNSD;
        return true;
    }
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = PrepareLayout::BSND;
        return true;
    }
    if (std::strcmp(layout, "NTD") == 0) {
        parsed = PrepareLayout::NTD;
        return true;
    }
    if (std::strcmp(layout, "TND") == 0) {
        parsed = PrepareLayout::TND;
        return true;
    }
    return false;
}

bool SameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t dim = 0; dim < lhs.GetDimNum(); ++dim) {
        if (lhs.GetDim(dim) != rhs.GetDim(dim)) {
            return false;
        }
    }
    return true;
}

bool HasShape(const gert::Shape &shape,
              std::initializer_list<int64_t> expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    size_t dim = 0;
    for (int64_t value : expected) {
        if (shape.GetDim(dim++) != value) {
            return false;
        }
    }
    return true;
}

bool ResolveShape(gert::TilingContext *context, PrepareLayout layout,
                  PrepareShapeInfo &info)
{
    const auto *qShapePtr = context->GetInputShape(PREPARE_INPUT_Q);
    const auto *kShapePtr = context->GetInputShape(PREPARE_INPUT_K);
    const auto *vShapePtr = context->GetInputShape(PREPARE_INPUT_V);
    const auto *gShapePtr = context->GetInputShape(PREPARE_INPUT_G);
    const auto *betaShapePtr = context->GetInputShape(PREPARE_INPUT_BETA);
    if (qShapePtr == nullptr || kShapePtr == nullptr || vShapePtr == nullptr ||
        gShapePtr == nullptr || betaShapePtr == nullptr) {
        ReportTilingError(context, "missing mandatory q/k/v/g/beta storage shape in TilingContext");
        return false;
    }

    const auto invalidShapes = [&]() {
        const std::string actual =
            ShapeText(qShapePtr->GetStorageShape()) + "," + ShapeText(kShapePtr->GetStorageShape()) + "," +
            ShapeText(vShapePtr->GetStorageShape()) + "," + ShapeText(gShapePtr->GetStorageShape()) + "," +
            ShapeText(betaShapePtr->GetStorageShape());
        const bool packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
        const bool sequenceMajor = layout == PrepareLayout::BSND || layout == PrepareLayout::TND;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context), "q,k,v,g,beta", actual,
            "q/k must match; q/k/v/g rank=" + ToText(packed ? 3 : 4) + ", beta rank=" + ToText(packed ? 2 : 3) +
                "; shared B/T/K and value-head axes must agree; packed=" + ToText(packed) +
                ", sequenceMajor=" + ToText(sequenceMajor));
        return false;
    };
    const auto &q = qShapePtr->GetStorageShape();
    const auto &k = kShapePtr->GetStorageShape();
    const auto &v = vShapePtr->GetStorageShape();
    const auto &g = gShapePtr->GetStorageShape();
    const auto &beta = betaShapePtr->GetStorageShape();
    if (!SameShape(q, k)) {
        return invalidShapes();
    }

    info.packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
    info.sequenceMajor = layout == PrepareLayout::BSND || layout == PrepareLayout::TND;
    if (info.packed) {
        if (q.GetDimNum() != 3 || v.GetDimNum() != 3 || g.GetDimNum() != 3) {
            return invalidShapes();
        }
        info.batch = 1;
        if (layout == PrepareLayout::TND) {
            info.seqLen = q.GetDim(0);
            info.qkHeadNum = q.GetDim(1);
            info.valueHeadNum = v.GetDim(1);
            info.kDim = q.GetDim(2);
            info.vDim = v.GetDim(2);
            if (!HasShape(v, {info.seqLen, info.valueHeadNum, info.vDim}) ||
                !HasShape(g, {info.seqLen, info.valueHeadNum, info.kDim})) {
                return invalidShapes();
            }
        } else {
            info.qkHeadNum = q.GetDim(0);
            info.valueHeadNum = v.GetDim(0);
            info.seqLen = q.GetDim(1);
            info.kDim = q.GetDim(2);
            info.vDim = v.GetDim(2);
            if (!HasShape(v, {info.valueHeadNum, info.seqLen, info.vDim}) ||
                !HasShape(g, {info.valueHeadNum, info.seqLen, info.kDim})) {
                return invalidShapes();
            }
        }
        if (layout == PrepareLayout::TND) {
            return HasShape(beta, {info.seqLen, info.valueHeadNum}) ? true : invalidShapes();
        }
        return HasShape(beta, {info.valueHeadNum, info.seqLen}) ? true : invalidShapes();
    }

    if (q.GetDimNum() != 4 || v.GetDimNum() != 4 || g.GetDimNum() != 4) {
        return invalidShapes();
    }
    info.batch = q.GetDim(0);
    if (layout == PrepareLayout::BSND) {
        info.seqLen = q.GetDim(1);
        info.qkHeadNum = q.GetDim(2);
        info.valueHeadNum = v.GetDim(2);
        info.kDim = q.GetDim(3);
        info.vDim = v.GetDim(3);
        if (!HasShape(v, {info.batch, info.seqLen, info.valueHeadNum, info.vDim}) ||
            !HasShape(g, {info.batch, info.seqLen, info.valueHeadNum, info.kDim})) {
            return invalidShapes();
        }
    } else {
        info.qkHeadNum = q.GetDim(1);
        info.valueHeadNum = v.GetDim(1);
        info.seqLen = q.GetDim(2);
        info.kDim = q.GetDim(3);
        info.vDim = v.GetDim(3);
        if (!HasShape(v, {info.batch, info.valueHeadNum, info.seqLen, info.vDim}) ||
            !HasShape(g, {info.batch, info.valueHeadNum, info.seqLen, info.kDim})) {
            return invalidShapes();
        }
    }
    if (layout == PrepareLayout::BSND) {
        return HasShape(beta, {info.batch, info.seqLen, info.valueHeadNum}) ? true : invalidShapes();
    }
    return HasShape(beta, {info.batch, info.valueHeadNum, info.seqLen}) ? true : invalidShapes();
}

bool IsGateDtype(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT;
}

bool DtypeToTemplateToken(ge::DataType dtype, uint64_t &token)
{
    if (dtype == ge::DT_BF16) {
        token = CHUNK_KDA_FWD_PREPARE_TPL_BF16;
        return true;
    }
    if (dtype == ge::DT_FLOAT) {
        token = CHUNK_KDA_FWD_PREPARE_TPL_FP32;
        return true;
    }
    return false;
}

bool FitsUint32(int64_t value)
{
    return value > 0 &&
        static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

bool FitsDmaSourceStride(int64_t headNum, uint64_t rowBytes)
{
    return headNum > 0 && rowBytes > 0 &&
        static_cast<uint64_t>(headNum - 1) <=
            std::numeric_limits<uint32_t>::max() / rowBytes;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

bool ResolveSequenceInfo(gert::TilingContext *context,
                         const PrepareShapeInfo &shape, uint64_t &seqNum,
                         uint64_t &totalChunks, bool &isVarLen)
{
    const auto *cuTensor =
        context->GetOptionalInputTensor(PREPARE_INPUT_CU_SEQLENS);
    const auto *chunkTensor =
        context->GetOptionalInputTensor(PREPARE_INPUT_CHUNK_INDICES);
    isVarLen = cuTensor != nullptr;
    if (!isVarLen) {
        if (chunkTensor != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                Ops::Base::GetOpInfo(context), "chunk_indices",
                "present without cu_seqlens; provide both metadata inputs together");
            return false;
        }
        seqNum = static_cast<uint64_t>(shape.batch);
        totalChunks = CeilDiv(static_cast<uint64_t>(shape.seqLen),
                              PREPARE_CHUNK_SIZE);
        return totalChunks != 0;
    }

    if (!shape.packed && shape.batch != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Ops::Base::GetOpInfo(context), "B", ToText(shape.batch),
                                              "rank-4 variable-length input requires B=1");
        return false;
    }
    const auto &cuShape = cuTensor->GetStorageShape();
    if (cuShape.GetDimNum() != 1 || cuShape.GetDim(0) < 2) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context), "cu_seqlens", ShapeText(cuShape),
                                              "expected a rank-1 tensor with at least two elements [0,total_tokens]");
        return false;
    }
    const int64_t *cu = cuTensor->GetData<int64_t>();
    if (cu == nullptr) {
        ReportTilingError(context, "GetData<int64_t>(cu_seqlens) returned null after descriptor validation; check host "
                                   "metadata conversion and placement");
        return false;
    }
    seqNum = static_cast<uint64_t>(cuShape.GetDim(0) - 1);
    if (cu[0] != 0 || cu[seqNum] != shape.seqLen) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(Ops::Base::GetOpInfo(context), "cu_seqlens endpoints",
                                               ToText(cu[0]) + "," + ToText(cu[seqNum]),
                                               "expected 0," + ToText(shape.seqLen));
        return false;
    }

    totalChunks = 0;
    for (uint64_t seq = 0; seq < seqNum; ++seq) {
        if (cu[seq] < 0 || cu[seq + 1] < cu[seq] ||
            cu[seq + 1] > shape.seqLen) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(Ops::Base::GetOpInfo(context), "cu_seqlens adjacent elements",
                                                   ToText(cu[seq]) + "," + ToText(cu[seq + 1]),
                                                   "invalid pair at index=" + ToText(seq) +
                                                       "; require 0<=begin<=end<=T=" + ToText(shape.seqLen));
            return false;
        }
        const uint64_t length = static_cast<uint64_t>(cu[seq + 1] - cu[seq]);
        const uint64_t sequenceChunks = CeilDiv(length, PREPARE_CHUNK_SIZE);
        if (totalChunks > std::numeric_limits<uint64_t>::max() - sequenceChunks) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(Ops::Base::GetOpInfo(context), "total_chunks,sequence_chunks",
                                                   ToText(totalChunks) + "," + ToText(sequenceChunks),
                                                   "their sum exceeds uint64");
            return false;
        }
        totalChunks += sequenceChunks;
    }
    if (totalChunks == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Ops::Base::GetOpInfo(context), "total_chunks", ToText(totalChunks),
                                              "provide at least one non-empty sequence");
        return false;
    }

    if (chunkTensor == nullptr) {
        return true;
    }
    const auto &chunkShape = chunkTensor->GetStorageShape();
    if (chunkShape.GetDimNum() != 1 ||
        static_cast<uint64_t>(chunkShape.GetDim(0)) != totalChunks * 2) {
        OP_LOGE_FOR_INVALID_SHAPE(Ops::Base::GetOpInfo(context), "chunk_indices", ShapeText(chunkShape),
                                  "[" + ToText(totalChunks * 2) + "]");
        return false;
    }
    const int64_t *indices = chunkTensor->GetData<int64_t>();
    if (indices == nullptr) {
        ReportTilingError(context, "GetData<int64_t>(chunk_indices) returned null after descriptor validation; check "
                                   "host metadata conversion and placement");
        return false;
    }
    uint64_t offset = 0;
    for (uint64_t seq = 0; seq < seqNum; ++seq) {
        const uint64_t length = static_cast<uint64_t>(cu[seq + 1] - cu[seq]);
        const uint64_t chunks = CeilDiv(length, PREPARE_CHUNK_SIZE);
        for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
            if (indices[offset] != static_cast<int64_t>(seq) ||
                indices[offset + 1] != static_cast<int64_t>(chunk)) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(Ops::Base::GetOpInfo(context), "chunk_indices pair",
                                                       ToText(indices[offset]) + "," + ToText(indices[offset + 1]),
                                                       "at offset=" + ToText(offset) +
                                                           ", expected sequence-major pair=" + ToText(seq) + "," +
                                                           ToText(chunk));
                return false;
            }
            offset += 2;
        }
    }
    return true;
}

bool CheckOptionalInput(gert::TilingContext *context,
                        const PrepareShapeInfo &shape, bool useGateInKernel,
                        bool safeGate, bool &hasDtBias)
{
    const auto *aLogDesc = context->GetOptionalInputDesc(PREPARE_INPUT_A_LOG);
    const auto *dtBiasDesc = context->GetOptionalInputDesc(PREPARE_INPUT_DT_BIAS);
    const auto *aLogShape = context->GetOptionalInputShape(PREPARE_INPUT_A_LOG);
    const auto *dtBiasShape = context->GetOptionalInputShape(PREPARE_INPUT_DT_BIAS);
    if ((aLogDesc == nullptr) != (aLogShape == nullptr) ||
        (dtBiasDesc == nullptr) != (dtBiasShape == nullptr)) {
        ReportTilingError(context,
                          "a_log/dt_bias descriptor and shape metadata must exist in pairs; a_log desc/shape=" +
                              ToText(aLogDesc != nullptr) + "/" + ToText(aLogShape != nullptr) +
                              ", dt_bias desc/shape=" + ToText(dtBiasDesc != nullptr) + "/" +
                              ToText(dtBiasShape != nullptr));
        return false;
    }
    const bool hasALog = aLogDesc != nullptr;
    hasDtBias = dtBiasDesc != nullptr;

    if (!useGateInKernel) {
        if (hasALog || hasDtBias || safeGate) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                Ops::Base::GetOpInfo(context), "use_gate_in_kernel,a_log,dt_bias,safe_gate",
                "false," + ToText(hasALog) + "," + ToText(hasDtBias) + "," + ToText(safeGate),
                "with gate disabled, a_log/dt_bias must be absent and safe_gate "
                "must be false; pointer fields describe presence");
            return false;
        }
        return true;
    }
    if (!hasALog) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context), "a_log",
                                                 "required when use_gate_in_kernel=true; provide FP32 [HV]");
        return false;
    }
    if (aLogDesc->GetDataType() != ge::DT_FLOAT ||
        (dtBiasDesc != nullptr && dtBiasDesc->GetDataType() != ge::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            Ops::Base::GetOpInfo(context), "a_log,dt_bias",
            DtypeText(aLogDesc->GetDataType()) + "," +
                (dtBiasDesc == nullptr ? "absent" : DtypeText(dtBiasDesc->GetDataType())),
            "each present tensor must have dtype FLOAT (FP32)");
        return false;
    }
    const auto &aLog = aLogShape->GetStorageShape();
    if (!HasShape(aLog, {shape.valueHeadNum})) {
        OP_LOGE_FOR_INVALID_SHAPE(Ops::Base::GetOpInfo(context), "a_log", ShapeText(aLog),
                                  "[" + ToText(shape.valueHeadNum) + "]");
        return false;
    }
    if (hasDtBias) {
        const auto &dtBias = dtBiasShape->GetStorageShape();
        if (!HasShape(dtBias, {shape.valueHeadNum * shape.kDim})) {
            OP_LOGE_FOR_INVALID_SHAPE(Ops::Base::GetOpInfo(context), "dt_bias", ShapeText(dtBias),
                                      "[" + ToText(shape.valueHeadNum * shape.kDim) + "] (flattened HV*K)");
            return false;
        }
    }
    return true;
}

void PrintTiling(gert::TilingContext *context,
                 ChunkKdaFwdPrepareTilingData &tiling,
                 uint64_t outputMode, uint64_t tilingKey,
                 uint64_t workspaceBytes)
{
    OP_LOGD(context->GetNodeName(),
            "ChunkKdaFwdPrepare tiling: B=%u, N=%u, T=%u, HK=%u, HV=%u, "
            "chunks=%u, cores=%u, headsPerPartition=%u, outputMode=%lu, "
            "varlen=%d, sequenceMajor=%d, tilingKey=%lu, "
            "workspace=%lu.",
            tiling.get_batch(), tiling.get_seqNum(), tiling.get_seqLen(),
            tiling.get_qkHeadNum(), tiling.get_valueHeadNum(),
            tiling.get_totalChunks(), tiling.get_usedCoreNum(),
            tiling.get_headsPerPartition(),
            static_cast<unsigned long>(outputMode),
            static_cast<int>(tiling.get_isVarLen()),
            static_cast<int>(tiling.get_inputSequenceMajor()),
            static_cast<unsigned long>(tilingKey),
            static_cast<unsigned long>(workspaceBytes));
}

} // namespace

ge::graphStatus Tiling4ChunkKdaFwdPrepare(gert::TilingContext *context)
{
    if (context == nullptr) {
        ReportTilingError(nullptr, "Tiling4ChunkKdaFwdPrepare received a null TilingContext");
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    const auto *qDesc = context->GetInputDesc(PREPARE_INPUT_Q);
    const auto *kDesc = context->GetInputDesc(PREPARE_INPUT_K);
    const auto *vDesc = context->GetInputDesc(PREPARE_INPUT_V);
    const auto *gDesc = context->GetInputDesc(PREPARE_INPUT_G);
    const auto *betaDesc = context->GetInputDesc(PREPARE_INPUT_BETA);
    if (attrs == nullptr || qDesc == nullptr || kDesc == nullptr ||
        vDesc == nullptr || gDesc == nullptr || betaDesc == nullptr) {
        ReportTilingError(context, "missing mandatory input descriptor or runtime attrs; q/k/v/g/beta=" +
                                       ToText(qDesc != nullptr) + "/" + ToText(kDesc != nullptr) + "/" +
                                       ToText(vDesc != nullptr) + "/" + ToText(gDesc != nullptr) + "/" +
                                       ToText(betaDesc != nullptr) + ", attrs=" + ToText(attrs != nullptr));
        return ge::GRAPH_FAILED;
    }

    const char *layout = attrs->GetStr(PREPARE_ATTR_LAYOUT);
    const auto *scalePtr = attrs->GetAttrPointer<float>(PREPARE_ATTR_SCALE);
    const auto *chunkSizePtr =
        attrs->GetAttrPointer<int64_t>(PREPARE_ATTR_CHUNK_SIZE);
    const auto *epsilonPtr = attrs->GetAttrPointer<float>(PREPARE_ATTR_EPSILON);
    const auto *useNormPtr =
        attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_QK_L2NORM);
    const auto *useGatePtr = attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_GATE);
    const auto *useBetaPtr =
        attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_BETA_SIGMOID);
    const auto *allowNegPtr =
        attrs->GetAttrPointer<bool>(PREPARE_ATTR_ALLOW_NEG_EIGVAL);
    const auto *safeGatePtr = attrs->GetAttrPointer<bool>(PREPARE_ATTR_SAFE_GATE);
    const auto *lowerBoundPtr =
        attrs->GetAttrPointer<float>(PREPARE_ATTR_LOWER_BOUND);
    const auto *useExp2Ptr = attrs->GetAttrPointer<bool>(PREPARE_ATTR_USE_EXP2);
    if (scalePtr == nullptr || chunkSizePtr == nullptr || epsilonPtr == nullptr ||
        useNormPtr == nullptr || useGatePtr == nullptr || useBetaPtr == nullptr ||
        allowNegPtr == nullptr || safeGatePtr == nullptr || lowerBoundPtr == nullptr ||
        useExp2Ptr == nullptr) {
        ReportTilingError(
            context,
            "runtime attribute table is incomplete; expected "
            "scale/chunk_size/epsilon/norm/gate/beta/allow_neg_eigval/safe_gate/lower_bound/use_exp2 attributes");
        return ge::GRAPH_FAILED;
    }

    PrepareLayout parsedLayout;
    if (!ParseLayout(layout, parsedLayout)) {
        OP_LOGE_FOR_INVALID_FORMAT(Ops::Base::GetOpInfo(context), "layout", layout == nullptr ? "<null>" : layout,
                                   "BNSD, BSND, NTD or TND (case-sensitive)");
        return ge::GRAPH_FAILED;
    }
    if (*chunkSizePtr != PREPARE_CHUNK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(Ops::Base::GetOpInfo(context), "chunk_size", ToText(*chunkSizePtr), "64");
        return ge::GRAPH_FAILED;
    }
    if (!std::isfinite(*scalePtr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Ops::Base::GetOpInfo(context), "scale", ToText(*scalePtr),
                                              "FP32 value must be finite");
        return ge::GRAPH_FAILED;
    }
    if (!std::isfinite(*epsilonPtr) || *epsilonPtr <= 0.0F) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Ops::Base::GetOpInfo(context), "epsilon", ToText(*epsilonPtr),
                                              "FP32 value must be positive and finite");
        return ge::GRAPH_FAILED;
    }
    if (!std::isfinite(*lowerBoundPtr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Ops::Base::GetOpInfo(context), "lower_bound", ToText(*lowerBoundPtr),
                                              "FP32 value must be finite");
        return ge::GRAPH_FAILED;
    }
    if (*allowNegPtr && !*useBetaPtr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(Ops::Base::GetOpInfo(context),
                                               "allow_neg_eigval,use_beta_sigmoid_in_kernel",
                                               ToText(*allowNegPtr) + "," + ToText(*useBetaPtr),
                                               "allow_neg_eigval=true requires use_beta_sigmoid_in_kernel=true");
        return ge::GRAPH_FAILED;
    }
    if (*safeGatePtr && (*lowerBoundPtr < -5.0F || *lowerBoundPtr >= 0.0F)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(Ops::Base::GetOpInfo(context), "lower_bound", ToText(*lowerBoundPtr),
                                              "safe_gate=true requires [-5,0)");
        return ge::GRAPH_FAILED;
    }

    PrepareShapeInfo shape;
    if (!ResolveShape(context, parsedLayout, shape)) {
        OP_LOGE_WITHOUT_REPORT(context, "Input shape validation failed; preserve the first structured message.");
        return ge::GRAPH_FAILED;
    }
    const auto *outputModePtr =
        attrs->GetAttrPointer<int64_t>(PREPARE_ATTR_OUTPUT_MODE);
    if (outputModePtr == nullptr || *outputModePtr < PREPARE_OUTPUT_MODE_NONE ||
        *outputModePtr > PREPARE_OUTPUT_MODE_FORWARD) {
        OP_LOGE_FOR_INVALID_VALUE(Ops::Base::GetOpInfo(context), "output_mode",
                                  outputModePtr == nullptr ? "<null>" : ToText(*outputModePtr),
                                  "0(none), 1(recompute), 2(save), 3(forward)");
        return ge::GRAPH_FAILED;
    }
    const uint64_t outputMode = static_cast<uint64_t>(*outputModePtr);
    if (!FitsUint32(shape.batch) || !FitsUint32(shape.seqLen) ||
        !FitsUint32(shape.qkHeadNum) || !FitsUint32(shape.valueHeadNum) ||
        shape.kDim != PREPARE_K_DIM || shape.vDim != PREPARE_V_DIM ||
        shape.valueHeadNum < shape.qkHeadNum ||
        shape.valueHeadNum % shape.qkHeadNum != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            Ops::Base::GetOpInfo(context), "B,T,HK,HV,K,V", LogicalText(shape),
            "B/T/HK/HV must be in [1,4294967295], K=V=128, HV>=HK and HV%HK=0; layout=" +
                std::string(layout == nullptr ? "BNSD" : layout));
        return ge::GRAPH_FAILED;
    }
    if (qDesc->GetDataType() != ge::DT_BF16 ||
        kDesc->GetDataType() != ge::DT_BF16 ||
        vDesc->GetDataType() != ge::DT_BF16 ||
        !IsGateDtype(gDesc->GetDataType()) ||
        !IsGateDtype(betaDesc->GetDataType())) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(Ops::Base::GetOpInfo(context), "q,k,v,g,beta",
                                               DtypeText(qDesc->GetDataType()) + "," + DtypeText(kDesc->GetDataType()) +
                                                   "," + DtypeText(vDesc->GetDataType()) + "," +
                                                   DtypeText(gDesc->GetDataType()) + "," +
                                                   DtypeText(betaDesc->GetDataType()),
                                               "q/k/v must be BF16; g and beta each must be BF16 or FLOAT (FP32)");
        return ge::GRAPH_FAILED;
    }
    if (shape.sequenceMajor) {
        const uint64_t gateElementBytes =
            gDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : 2U;
        const uint64_t betaElementBytes =
            betaDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : 2U;
        const bool strideValid =
            FitsDmaSourceStride(shape.qkHeadNum,
                                PREPARE_K_DIM * 2U) &&
            FitsDmaSourceStride(shape.valueHeadNum,
                                PREPARE_V_DIM * 2U) &&
            FitsDmaSourceStride(shape.valueHeadNum,
                                PREPARE_K_DIM * gateElementBytes) &&
            FitsDmaSourceStride(shape.valueHeadNum, betaElementBytes);
        if (!strideValid) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                Ops::Base::GetOpInfo(context), "HK,HV", ToText(shape.qkHeadNum) + "," + ToText(shape.valueHeadNum),
                "layout=" + std::string(layout) +
                    "; each (heads-1)*rowBytes DMA source stride must fit uint32; reduce the head counts");
            return ge::GRAPH_FAILED;
        }
    }

    bool hasDtBias = false;
    if (!CheckOptionalInput(context, shape, *useGatePtr, *safeGatePtr,
                            hasDtBias)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t seqNum = 0;
    uint64_t totalChunks = 0;
    bool isVarLen = false;
    if (!ResolveSequenceInfo(context, shape, seqNum, totalChunks, isVarLen)) {
        return ge::GRAPH_FAILED; // Preserve the metadata discovery-point error.
    }
    if (seqNum > std::numeric_limits<uint32_t>::max() || totalChunks > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(Ops::Base::GetOpInfo(context), "sequence_count,total_chunks",
                                               ToText(seqNum) + "," + ToText(totalChunks),
                                               "each count must fit uint32");
        return ge::GRAPH_FAILED;
    }

    if (context->GetPlatformInfo() == nullptr) {
        ReportTilingError(
            context,
            "platform information is null; check matching SoC/OPP installation and platform loading diagnostics");
        return ge::GRAPH_FAILED;
    }
    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t aicCoreNum = platform.GetCoreNumAic();
    const bool isAscend950 =
        platform.GetCurNpuArch() == NpuArch::DAV_3510;
    ChunkKdaFwdPrepareScheduleContext scheduleContext;
    scheduleContext.batch = static_cast<uint64_t>(shape.batch);
    scheduleContext.qkHeadNum = static_cast<uint64_t>(shape.qkHeadNum);
    scheduleContext.valueHeadNum = static_cast<uint64_t>(shape.valueHeadNum);
    scheduleContext.chunksPerSequence = totalChunks;
    scheduleContext.totalVarLenChunks = totalChunks;
    scheduleContext.aicCoreNum = aicCoreNum;
    scheduleContext.libApiWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
    scheduleContext.isVarLen = isVarLen;
    scheduleContext.isArch22 = !isAscend950;
    // These are input-derived work-domain limits, not an internal Tiling failure.
    const uint64_t chunkWorkItems = isVarLen ? totalChunks : static_cast<uint64_t>(shape.batch) * totalChunks;
    const uint64_t partitions = !isAscend950 ?
                                    (static_cast<uint64_t>(shape.valueHeadNum) + 1) / 2 :
                                    ((chunkWorkItems < aicCoreNum && shape.qkHeadNum > 1) ? shape.qkHeadNum : 1);
    if (chunkWorkItems > std::numeric_limits<uint32_t>::max() ||
        (chunkWorkItems != 0 && partitions > std::numeric_limits<uint32_t>::max() / chunkWorkItems)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            Ops::Base::GetOpInfo(context), "chunk_work_items,head_partitions",
            ToText(chunkWorkItems) + "," + ToText(partitions),
            "chunk work items and their product with head partitions must fit uint32; reduce B/T/HV");
        return ge::GRAPH_FAILED;
    }
    ChunkKdaFwdPrepareSchedule schedule;
    if (!ChunkKdaFwdPrepareTilingProcessor(scheduleContext).Process(schedule)) {
        ReportTilingError(
            context,
            "ChunkKdaFwdPrepareTilingProcessor::Process failed after input/work-domain validation; AIC cores=" +
                ToText(aicCoreNum) + ", " + LogicalText(shape) + ", chunks=" + ToText(totalChunks) +
                ", libApiWorkspaceBytes=" + ToText(scheduleContext.libApiWorkspaceBytes));
        return ge::GRAPH_FAILED;
    }

    uint64_t gateDtypeToken = 0;
    uint64_t betaDtypeToken = 0;
    if (!DtypeToTemplateToken(gDesc->GetDataType(), gateDtypeToken) ||
        !DtypeToTemplateToken(betaDesc->GetDataType(), betaDtypeToken)) {
        ReportTilingError(context, "validated g/beta dtype cannot be encoded into template tokens");
        return ge::GRAPH_FAILED;
    }
    const uint64_t normMode = *useNormPtr
        ? CHUNK_KDA_FWD_PREPARE_NORM_L2
        : CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY;
    const uint64_t betaMode = !*useBetaPtr
        ? CHUNK_KDA_FWD_PREPARE_BETA_RAW
        : (*allowNegPtr ? CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID
                        : CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID);
    const uint64_t gateMode = !*useGatePtr
        ? CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP
        : (*safeGatePtr ? CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID
                        : CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS);
    using namespace KdaPrepare;
    const uint64_t legacyTilingKey = GET_TPL_TILING_KEY(
        gateDtypeToken, betaDtypeToken, normMode, betaMode, gateMode,
        static_cast<uint64_t>(*useExp2Ptr),
        static_cast<uint64_t>(*safeGatePtr),
        outputMode);
    const auto pipeline = PreparePipeline::Generic;
    const uint64_t tilingKey = isAscend950 ? legacyTilingKey
        : RuntimeKey::Encode(normMode, gateMode);

    ChunkKdaFwdPrepareTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(shape.batch));
    tiling.set_seqNum(static_cast<uint32_t>(seqNum));
    tiling.set_seqLen(static_cast<uint32_t>(shape.seqLen));
    tiling.set_qkHeadNum(static_cast<uint32_t>(shape.qkHeadNum));
    tiling.set_valueHeadNum(static_cast<uint32_t>(shape.valueHeadNum));
    tiling.set_totalChunks(static_cast<uint32_t>(totalChunks));
    tiling.set_usedCoreNum(schedule.usedCoreNum);
    tiling.set_headsPerPartition(schedule.headsPerPartition);
    tiling.set_epsilon(*epsilonPtr);
    tiling.set_lowerBound(*lowerBoundPtr);
    tiling.set_scale(*scalePtr);
    tiling.set_isVarLen(isVarLen);
    tiling.set_inputSequenceMajor(shape.sequenceMajor);
    tiling.set_hasDtBias(hasDtBias);
    tiling.set_normMode(normMode);
    tiling.set_betaMode(betaMode);
    tiling.set_gateMode(gateMode);
    tiling.set_outputMode(outputMode);
    tiling.set_pipelineVariant(static_cast<uint32_t>(pipeline));
    tiling.set_useExp2(*useExp2Ptr);

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(schedule.usedCoreNum);
    if (context->SetScheduleMode(PREPARE_MIX_BATCH_MODE) != ge::GRAPH_SUCCESS) {
        ReportTilingError(context, "SetScheduleMode(MIX AIC/AIV batch mode=1) failed; tilingKey=" + ToText(tilingKey) +
                                       ", blockDim=" + ToText(schedule.usedCoreNum));
        return ge::GRAPH_FAILED;
    }
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        ReportTilingError(context, "GetWorkspaceSizes(1) returned null; expected one TilingContext output slot, not a "
                                   "user device workspace allocation");
        return ge::GRAPH_FAILED;
    }
    if (schedule.workspaceBytes > std::numeric_limits<size_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            Ops::Base::GetOpInfo(context), "workspace_bytes", ToText(schedule.workspaceBytes),
            "must fit platform size_t; maximum=" + ToText(std::numeric_limits<size_t>::max()));
        return ge::GRAPH_FAILED;
    }
    auto *rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr || rawTiling->GetCapacity() < tiling.GetDataSize()) {
        ReportTilingError(context, "raw tiling output buffer is missing or too small; required bytes=" +
                                       ToText(tiling.GetDataSize()) +
                                       ", capacity=" + ToText(rawTiling == nullptr ? 0 : rawTiling->GetCapacity()));
        return ge::GRAPH_FAILED;
    }
    workspace[0] = static_cast<size_t>(schedule.workspaceBytes);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    PrintTiling(context, tiling, outputMode, tilingKey,
                schedule.workspaceBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaFwdPrepare(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaFwdPrepare)
    .Tiling(Tiling4ChunkKdaFwdPrepare)
    .TilingParse<ChunkKdaFwdPrepareCompileInfo>(
        TilingPrepareForChunkKdaFwdPrepare);

} // namespace optiling
