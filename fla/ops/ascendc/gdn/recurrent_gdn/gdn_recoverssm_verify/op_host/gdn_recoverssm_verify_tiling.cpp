/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_verify_tiling.h"

#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gdn_recoverssm_verify_tiling_key.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace optiling {
namespace {
constexpr size_t A_LOG = 0;
constexpr size_t A = 1;
constexpr size_t B = 2;
constexpr size_t DT_BIAS = 3;
constexpr size_t QUERY = 4;
constexpr size_t KEY = 5;
constexpr size_t VALUE = 6;
constexpr size_t CHECKPOINT = 7;
constexpr size_t REPLAY = 8;
constexpr size_t QUERY_START = 9;
constexpr size_t STATE_INDICES = 10;

std::string ShapeText(const gert::Shape &shape)
{
    std::string text = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            text += ",";
        }
        text += std::to_string(shape.GetDim(i));
    }
    return text + "]";
}
} // namespace

REGISTER_OPS_TILING_TEMPLATE(GdnRecoverssmVerify, GdnRecoverssmVerifyTiling, 0);

void GdnRecoverssmVerifyTiling::InitCompileInfo()
{
    auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return;
    }
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = platform.GetCoreNumAiv();
    compileInfo_.isA5 = platform.GetCurNpuArch() == NpuArch::DAV_3510;
}

ge::graphStatus GdnRecoverssmVerifyTiling::GetPlatformInfo()
{
    if (!compileInfo_.isA5 || compileInfo_.aivNum == 0 || compileInfo_.ubSize == 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "platform",
            "GdnRecoverssmVerify requires Ascend 950 with runtime AIV and UB information");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::CheckContext()
{
    for (size_t index = 0; index <= STATE_INDICES; ++index) {
        if (context_->GetInputShape(index) == nullptr || context_->GetInputDesc(index) == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "input[" + std::to_string(index) + "]",
                                                     "input shape and descriptor must not be nullptr");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::AnalyzeDtype()
{
    const auto dtype = [&](size_t index) { return context_->GetInputDesc(index)->GetDataType(); };
    if (dtype(A_LOG) != ge::DT_FLOAT || dtype(DT_BIAS) != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "A_log/dt_bias", "non-float", "[float32]");
        return ge::GRAPH_FAILED;
    }
    for (size_t index : {A, B, QUERY, KEY, VALUE}) {
        if (dtype(index) != ge::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "activation",
                                      ge::TypeUtils::DataTypeToSerialString(dtype(index)), "[bfloat16]");
            return ge::GRAPH_FAILED;
        }
    }
    if (dtype(CHECKPOINT) != ge::DT_BF16 && dtype(CHECKPOINT) != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "checkpoint_state",
                                  ge::TypeUtils::DataTypeToSerialString(dtype(CHECKPOINT)), "[bfloat16,float32]");
        return ge::GRAPH_FAILED;
    }
    if (dtype(REPLAY) != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "replay_cache",
                                  ge::TypeUtils::DataTypeToSerialString(dtype(REPLAY)), "[float32]");
        return ge::GRAPH_FAILED;
    }
    if (dtype(QUERY_START) != dtype(STATE_INDICES) ||
        (dtype(QUERY_START) != ge::DT_INT32 && dtype(QUERY_START) != ge::DT_INT64)) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "query_start_loc/state_indices",
                                  "mismatched or unsupported metadata", "[int32|int64]");
        return ge::GRAPH_FAILED;
    }
    tilingData_.checkpointDtype = dtype(CHECKPOINT) == ge::DT_FLOAT ? GRSV_TPL_FP32 : GRSV_TPL_BF16;
    tilingData_.metadataDtype = dtype(QUERY_START) == ge::DT_INT64 ? 64U : 32U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::AnalyzeShapesAndStrides()
{
    const auto &q = context_->GetInputShape(QUERY)->GetOriginShape();
    const auto &k = context_->GetInputShape(KEY)->GetOriginShape();
    const auto &v = context_->GetInputShape(VALUE)->GetOriginShape();
    const auto &a = context_->GetInputShape(A)->GetOriginShape();
    const auto &b = context_->GetInputShape(B)->GetOriginShape();
    const auto &state = context_->GetInputShape(CHECKPOINT)->GetOriginShape();
    const auto &replay = context_->GetInputShape(REPLAY)->GetOriginShape();
    const auto &start = context_->GetInputShape(QUERY_START)->GetOriginShape();
    const auto &indices = context_->GetInputShape(STATE_INDICES)->GetOriginShape();

    if (q.GetDimNum() != 4 || k.GetDimNum() != 4 || v.GetDimNum() != 4 || state.GetDimNum() != 4 ||
        replay.GetDimNum() != 4 || start.GetDimNum() != 1 || indices.GetDimNum() != 1 ||
        (a.GetDimNum() != 2 && a.GetDimNum() != 3) || (b.GetDimNum() != 2 && b.GetDimNum() != 3)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "shapes",
            "q=" + ShapeText(q) + ", k=" + ShapeText(k) + ", v=" + ShapeText(v) + ", a=" + ShapeText(a) +
                ", b=" + ShapeText(b) + ", checkpoint=" + ShapeText(state) + ", replay=" + ShapeText(replay) +
                ", start=" + ShapeText(start) + ", indices=" + ShapeText(indices));
        return ge::GRAPH_FAILED;
    }
    const int64_t total = q.GetDim(1);
    const int64_t hk = q.GetDim(2);
    const int64_t kd = q.GetDim(3);
    const int64_t hv = v.GetDim(2);
    const int64_t vd = v.GetDim(3);
    const int64_t batch = indices.GetDim(0);
    if (q.GetDim(0) != 1 || v.GetDim(0) != 1 || v.GetDim(1) != total || k.GetDim(0) != 1 || k.GetDim(1) != total ||
        k.GetDim(2) != hk || k.GetDim(3) != kd || total < 0 || hk <= 0 || hv <= 0 || hv % hk != 0 || hv / hk > 16 ||
        kd != 128 || vd != 128 || start.GetDim(0) != batch + 1 || batch < 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "shapes",
                                                 "q=" + ShapeText(q) + ", k=" + ShapeText(k) + ", v=" + ShapeText(v) +
                                                     ", start=" + ShapeText(start) + ", indices=" + ShapeText(indices));
        return ge::GRAPH_FAILED;
    }
    if (static_cast<int64_t>(a.GetShapeSize()) != total * hv || static_cast<int64_t>(b.GetShapeSize()) != total * hv) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "a/b",
                                                 "numel=" + std::to_string(a.GetShapeSize()) + "/" +
                                                     std::to_string(b.GetShapeSize()) +
                                                     ", expected=" + std::to_string(total * hv));
        return ge::GRAPH_FAILED;
    }
    if (state.GetDim(1) != hv || state.GetDim(2) != vd || state.GetDim(3) != kd ||
        replay.GetDim(0) != state.GetDim(0) || replay.GetDim(1) != hv || replay.GetDim(3) != vd + kd + 1 ||
        replay.GetDim(2) <= 0 || replay.GetDim(2) > 32) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "checkpoint/replay",
                                                 "state=" + ShapeText(state) + ", replay=" + ShapeText(replay));
        return ge::GRAPH_FAILED;
    }
    const auto readStride = [&](size_t index, size_t rank, uint64_t *out) -> bool {
        const auto *shape = context_->GetInputShape(index);
        const auto *desc = context_->GetInputDesc(index);
        const auto *stride = context_->GetInputStride(index);
        if (stride != nullptr && stride->GetDimNum() != 0 && stride->GetDimNum() != rank) {
            return false;
        }
        const uint64_t elemBytes = desc->GetDataType() == ge::DT_FLOAT ? 4U : 2U;
        uint64_t dense = 1;
        for (size_t dim = rank; dim > 0; --dim) {
            const int64_t value = (stride != nullptr && stride->GetDimNum() != 0) ? stride->GetStride(dim - 1) :
                                                                                    static_cast<int64_t>(dense);
            if (value <= 0) {
                return false;
            }
            out[dim - 1] = static_cast<uint64_t>(value);
            dense *= static_cast<uint64_t>(shape->GetOriginShape().GetDim(dim - 1));
        }
        (void)elemBytes;
        return true;
    };
    if (!readStride(QUERY, 4, tilingData_.queryStrides) || !readStride(KEY, 4, tilingData_.keyStrides) ||
        !readStride(VALUE, 4, tilingData_.valueStrides) || !readStride(CHECKPOINT, 4, tilingData_.checkpointStrides) ||
        !readStride(REPLAY, 4, tilingData_.replayStrides)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "strides", "positive strides required");
        return ge::GRAPH_FAILED;
    }
    const auto &outShape = context_->GetOutputShape(0)->GetOriginShape();
    if (outShape.GetDimNum() != 4) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "out shape",
                                                 "rank4 required, actual=" + ShapeText(outShape));
        return ge::GRAPH_FAILED;
    }
    const auto *outStride = context_->GetOutputStride(0);
    uint64_t dense = 1;
    for (size_t dim = 4; dim > 0; --dim) {
        const int64_t value = outStride != nullptr && outStride->GetDimNum() == 4 ? outStride->GetStride(dim - 1) :
                                                                                    static_cast<int64_t>(dense);
        if (value <= 0) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "out strides",
                                                     "all strides must be positive");
            return ge::GRAPH_FAILED;
        }
        tilingData_.outStrides[dim - 1] = static_cast<uint64_t>(value);
        const int64_t extent = outShape.GetDim(dim - 1);
        dense *= static_cast<uint64_t>(extent > 0 ? extent : 1);
    }
    if (tilingData_.queryStrides[3] != 1 || tilingData_.keyStrides[3] != 1 || tilingData_.valueStrides[3] != 1 ||
        tilingData_.checkpointStrides[3] != 1 || tilingData_.replayStrides[3] != 1) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "inner strides",
                                                 "K, V, checkpoint K, and replay payload must be contiguous");
        return ge::GRAPH_FAILED;
    }
    const auto analyzeScalarLayout = [&](size_t index, uint64_t *strides) -> bool {
        const auto &shape = context_->GetInputShape(index)->GetOriginShape();
        if (shape.GetDimNum() == 3 && (shape.GetDim(0) != 1 || shape.GetDim(1) != total || shape.GetDim(2) != hv)) {
            return false;
        }
        const auto *stride = context_->GetInputStride(index);
        const size_t rank = shape.GetDimNum();
        strides[0] =
            stride != nullptr && stride->GetDimNum() != 0 ? static_cast<uint64_t>(stride->GetStride(rank - 2)) : hv;
        strides[1] =
            stride != nullptr && stride->GetDimNum() != 0 ? static_cast<uint64_t>(stride->GetStride(rank - 1)) : 1U;
        return strides[0] > 0 && strides[1] == 1;
    };
    if (!analyzeScalarLayout(A, tilingData_.aStrides) || !analyzeScalarLayout(B, tilingData_.bStrides)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "a/b",
                                                 "a=" + ShapeText(a) + ", b=" + ShapeText(b) +
                                                     "; expected [T,HV] or [1,T,HV] with contiguous heads");
        return ge::GRAPH_FAILED;
    }
    tilingData_.queryStartStride =
        context_->GetInputStride(QUERY_START) != nullptr && context_->GetInputStride(QUERY_START)->GetDimNum() != 0 ?
            context_->GetInputStride(QUERY_START)->GetStride(0) :
            1U;
    tilingData_.stateIndexStride = context_->GetInputStride(STATE_INDICES) != nullptr &&
                                           context_->GetInputStride(STATE_INDICES)->GetDimNum() != 0 ?
                                       context_->GetInputStride(STATE_INDICES)->GetStride(0) :
                                       1U;
    tilingData_.batchSize = static_cast<uint32_t>(batch);
    tilingData_.totalTokens = static_cast<uint32_t>(total);
    tilingData_.keyHeads = static_cast<uint32_t>(hk);
    tilingData_.valueHeads = static_cast<uint32_t>(hv);
    tilingData_.keyDim = static_cast<uint32_t>(kd);
    tilingData_.valueDim = static_cast<uint32_t>(vd);
    tilingData_.specQueryLen = static_cast<uint32_t>(replay.GetDim(2));
    tilingData_.taskCount = static_cast<uint32_t>(batch * hv);
    const uint64_t rounds =
        std::max<uint64_t>(1U, (tilingData_.taskCount + compileInfo_.aivNum - 1U) / compileInfo_.aivNum);
    const uint64_t used = std::max<uint64_t>(1U, (tilingData_.taskCount + rounds - 1U) / rounds);
    tilingData_.vectorCoreNum = static_cast<uint32_t>(used);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::AnalyzeAttributes()
{
    const auto *attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "attrs", "null");
        return ge::GRAPH_FAILED;
    }
    const float *scale = attrs->GetAttrPointer<float>(0);
    const float *beta = attrs->GetAttrPointer<float>(1);
    const float *threshold = attrs->GetAttrPointer<float>(2);
    const bool *norm = attrs->GetAttrPointer<bool>(3);
    if (scale == nullptr || beta == nullptr || threshold == nullptr || norm == nullptr || !std::isfinite(*scale) ||
        !std::isfinite(*beta) || *beta <= 0.0f || !std::isfinite(*threshold)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "attributes",
                                                 "scale/threshold finite, softplus_beta > 0");
        return ge::GRAPH_FAILED;
    }
    tilingData_.scale = *scale;
    tilingData_.softplusBeta = *beta;
    tilingData_.softplusThreshold = *threshold;
    tilingData_.useQkL2NormInKernel = *norm ? 1U : 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::CheckUb()
{
    constexpr uint64_t stateBytes = 3ULL * 128ULL * 128ULL * sizeof(float);
    constexpr uint64_t qkvBytes = 2ULL * 4ULL * 3ULL * 128ULL * 2ULL;
    constexpr uint64_t scalarBytes = 2ULL * 2ULL * 4ULL * 16ULL * 2ULL;
    constexpr uint64_t outBytes = 2ULL * 4ULL * 128ULL * 2ULL;
    constexpr uint64_t correctionBytes = 2ULL * 4ULL * 128ULL * sizeof(float);
    constexpr uint64_t keyBytes = 2ULL * 4ULL * 128ULL * sizeof(float);
    constexpr uint64_t decayBytes = 2ULL * 4ULL * 8ULL * sizeof(float);
    constexpr uint64_t guard = 4096U;
    const uint64_t total = stateBytes + qkvBytes + scalarBytes + outBytes + correctionBytes + keyBytes + decayBytes;
    if (total + guard > compileInfo_.ubSize) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "UB",
                                                 "required=" + std::to_string(total + guard) +
                                                     ", available=" + std::to_string(compileInfo_.ubSize));
        return ge::GRAPH_FAILED;
    }
    tilingData_.ubCalSize = static_cast<uint32_t>(total);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::GetShapeAttrsInfo()
{
    if (CheckContext() != ge::GRAPH_SUCCESS || AnalyzeDtype() != ge::GRAPH_SUCCESS ||
        AnalyzeShapesAndStrides() != ge::GRAPH_SUCCESS || AnalyzeAttributes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::DoOpTiling()
{
    return CheckUb();
}

ge::graphStatus GdnRecoverssmVerifyTiling::DoLibApiTiling()
{
    tilingKey_ = GET_TPL_TILING_KEY(tilingData_.checkpointDtype, tilingData_.useQkL2NormInKernel != 0);
    return ge::GRAPH_SUCCESS;
}

uint64_t GdnRecoverssmVerifyTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus GdnRecoverssmVerifyTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<int64_t>(GRSV_SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmVerifyTiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    const size_t size = sizeof(tilingData_);
    if (memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), &tilingData_,
                 size) != EOK) {
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(size);
    size_t *workspace = context_->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = static_cast<size_t>(workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnRecoverssmVerify).Tiling(TilingFunc).TilingParse<GdnRecoverssmVerifyCompileInfo>(TilingPrepare);
} // namespace optiling
