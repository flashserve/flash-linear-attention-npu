/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_commit_tiling.h"

#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gdn_recoverssm_commit_tiling_key.h"

#include <algorithm>
#include <limits>
#include <string>

namespace optiling {
namespace {
constexpr size_t STATE = 0;
constexpr size_t REPLAY = 1;
constexpr size_t TABLE = 2;
constexpr size_t STATE_INDICES = 3;
constexpr size_t COMMIT_LENS = 4;
constexpr size_t FINAL_INDICES = 5;
constexpr size_t BOUNDARY_INDICES = 6;
constexpr size_t BOUNDARY_LENS = 7;

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

bool ReadStride(const gert::TilingContext *context, size_t index, size_t rank, uint64_t *out)
{
    const auto *shape = context->GetInputShape(index);
    const auto *stride = context->GetInputStride(index);
    if (shape == nullptr || (stride != nullptr && stride->GetDimNum() != 0 &&
                             stride->GetDimNum() != rank)) {
        return false;
    }
    uint64_t dense = 1U;
    for (size_t dim = rank; dim > 0; --dim) {
        const int64_t value =
            stride != nullptr && stride->GetDimNum() != 0 ?
                stride->GetStride(dim - 1) :
                static_cast<int64_t>(dense);
        if (value <= 0) {
            return false;
        }
        out[dim - 1] = static_cast<uint64_t>(value);
        const int64_t extent = shape->GetOriginShape().GetDim(dim - 1);
        dense *= static_cast<uint64_t>(extent > 0 ? extent : 1);
    }
    return true;
}
} // namespace

REGISTER_OPS_TILING_TEMPLATE(GdnRecoverssmCommit, GdnRecoverssmCommitTiling, 0);

void GdnRecoverssmCommitTiling::InitCompileInfo()
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

ge::graphStatus GdnRecoverssmCommitTiling::GetPlatformInfo()
{
    if (!compileInfo_.isA5 || compileInfo_.aivNum == 0 || compileInfo_.ubSize == 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "platform",
            "GdnRecoverssmCommit requires Ascend 950 with runtime AIV and UB information");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::CheckContext()
{
    for (size_t index = STATE; index <= BOUNDARY_LENS; ++index) {
        if (context_->GetInputShape(index) == nullptr ||
            context_->GetInputDesc(index) == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context_->GetNodeName(), "input[" + std::to_string(index) + "]",
                "input shape and descriptor must not be nullptr");
            return ge::GRAPH_FAILED;
        }
    }
    if (context_->GetOutputShape(0) == nullptr ||
        context_->GetOutputDesc(0) == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "state_out", "output must not be nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::AnalyzeDtype()
{
    const auto dtype = [&](size_t index) {
        return context_->GetInputDesc(index)->GetDataType();
    };
    const auto stateType = dtype(STATE);
    if (stateType != ge::DT_BF16 && stateType != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "state",
                                  ge::TypeUtils::DataTypeToSerialString(stateType),
                                  "[bfloat16,float32]");
        return ge::GRAPH_FAILED;
    }
    if (dtype(REPLAY) != ge::DT_FLOAT || dtype(TABLE) != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "replay/table",
                                  "unsupported", "[replay=float32,table=int64]");
        return ge::GRAPH_FAILED;
    }
    const auto isIntegerMeta = [&](size_t index) {
        const auto value = dtype(index);
        return value == ge::DT_INT32 || value == ge::DT_INT64;
    };
    if (!isIntegerMeta(STATE_INDICES) || !isIntegerMeta(COMMIT_LENS) ||
        !isIntegerMeta(FINAL_INDICES) || !isIntegerMeta(BOUNDARY_INDICES) ||
        !isIntegerMeta(BOUNDARY_LENS)) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "metadata",
                                  "unsupported", "[int32,int64]");
        return ge::GRAPH_FAILED;
    }
    const auto planType = dtype(COMMIT_LENS);
    if (dtype(FINAL_INDICES) != planType || dtype(BOUNDARY_INDICES) != planType ||
        dtype(BOUNDARY_LENS) != planType) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "plan metadata",
                                  "mismatched", "all plan tensors share int32 or int64");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOutputDesc(0)->GetDataType() != stateType) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "state_out",
                                  ge::TypeUtils::DataTypeToSerialString(
                                      context_->GetOutputDesc(0)->GetDataType()),
                                  "same as state");
        return ge::GRAPH_FAILED;
    }
    stateDtypeTemplate_ =
        stateType == ge::DT_FLOAT ? GRSC_TPL_FP32 : GRSC_TPL_BF16;
    tilingData_.stateIndicesDtype = dtype(STATE_INDICES) == ge::DT_INT64 ? 64U : 32U;
    tilingData_.planDtype = planType == ge::DT_INT64 ? 64U : 32U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::AnalyzeShapesAndStrides()
{
    const auto &state = context_->GetInputShape(STATE)->GetOriginShape();
    const auto &replay = context_->GetInputShape(REPLAY)->GetOriginShape();
    const auto &table = context_->GetInputShape(TABLE)->GetOriginShape();
    const auto &source = context_->GetInputShape(STATE_INDICES)->GetOriginShape();
    const auto &lengths = context_->GetInputShape(COMMIT_LENS)->GetOriginShape();
    const auto &final = context_->GetInputShape(FINAL_INDICES)->GetOriginShape();
    const auto &boundary =
        context_->GetInputShape(BOUNDARY_INDICES)->GetOriginShape();
    const auto &recovery =
        context_->GetInputShape(BOUNDARY_LENS)->GetOriginShape();

    if (state.GetDimNum() != 4 || replay.GetDimNum() != 4 ||
        table.GetDimNum() != 3 || source.GetDimNum() != 1 ||
        lengths.GetDimNum() != 1 || final.GetDimNum() != 1 ||
        boundary.GetDimNum() != 1 || recovery.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "shapes",
            "state=" + ShapeText(state) + ", replay=" + ShapeText(replay) +
                ", table=" + ShapeText(table) + ", source=" + ShapeText(source));
        return ge::GRAPH_FAILED;
    }

    const int64_t pages = state.GetDim(0);
    const int64_t heads = state.GetDim(1);
    const int64_t valueDim = state.GetDim(2);
    const int64_t keyDim = state.GetDim(3);
    const int64_t window = replay.GetDim(2);
    const int64_t layers = table.GetDim(2);
    const int64_t batch = source.GetDim(0);
    if (pages <= 0 || heads <= 0 || valueDim != 128 || keyDim != 128 ||
        replay.GetDim(0) != pages || replay.GetDim(1) != heads ||
        window < 1 || window > 32 ||
        replay.GetDim(3) != valueDim + keyDim + 1 ||
        table.GetDim(0) != 2 || table.GetDim(1) != 2 || layers <= 0 ||
        batch < 0 || lengths.GetDim(0) != batch ||
        final.GetDim(0) != batch || boundary.GetDim(0) != batch ||
        recovery.GetDim(0) != batch) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "shape contract",
            "state=" + ShapeText(state) + ", replay=" + ShapeText(replay) +
                ", table=" + ShapeText(table) + ", batch=" +
                std::to_string(batch));
        return ge::GRAPH_FAILED;
    }

    uint64_t stateStrides[4] = {};
    uint64_t replayStrides[4] = {};
    uint64_t tableStrides[3] = {};
    if (!ReadStride(context_, STATE, 4, stateStrides) ||
        !ReadStride(context_, REPLAY, 4, replayStrides) ||
        !ReadStride(context_, TABLE, 3, tableStrides)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "strides", "state/replay/table strides invalid");
        return ge::GRAPH_FAILED;
    }
    if (stateStrides[3] != 1U || replayStrides[3] != 1U ||
        tableStrides[0] != static_cast<uint64_t>(2 * layers) ||
        tableStrides[1] != static_cast<uint64_t>(layers) ||
        tableStrides[2] != 1U) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "inner strides",
            "state key, replay payload must be contiguous and the address table must be contiguous");
        return ge::GRAPH_FAILED;
    }

    const auto metaStride = [&](size_t index, uint64_t *value) -> bool {
        const auto *stride = context_->GetInputStride(index);
        *value = stride != nullptr && stride->GetDimNum() != 0 ?
                     static_cast<uint64_t>(stride->GetStride(0)) :
                     1U;
        return *value > 0;
    };
    if (!metaStride(STATE_INDICES, &tilingData_.stateIndicesStride) ||
        !metaStride(COMMIT_LENS, &tilingData_.commitLensStride) ||
        !metaStride(FINAL_INDICES, &tilingData_.finalStateIndicesStride) ||
        !metaStride(BOUNDARY_INDICES,
                    &tilingData_.boundaryStateIndicesStride) ||
        !metaStride(BOUNDARY_LENS, &tilingData_.boundaryRecoveryLensStride)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "metadata strides", "positive stride required");
        return ge::GRAPH_FAILED;
    }

    const uint64_t taskCount =
        static_cast<uint64_t>(batch) * static_cast<uint64_t>(heads) *
        static_cast<uint64_t>(layers);
    if (taskCount > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "task count", "B*HV*layers exceeds uint32");
        return ge::GRAPH_FAILED;
    }

    tilingData_.batchSize = static_cast<uint32_t>(batch);
    tilingData_.valueHeads = static_cast<uint32_t>(heads);
    tilingData_.valueDim = static_cast<uint32_t>(valueDim);
    tilingData_.keyDim = static_cast<uint32_t>(keyDim);
    tilingData_.layerCount = static_cast<uint32_t>(layers);
    tilingData_.taskCount = static_cast<uint32_t>(taskCount);
    tilingData_.stateHeadStride = stateStrides[1];
    tilingData_.stateValueStride = stateStrides[2];
    tilingData_.stateKeyStride = stateStrides[3];
    tilingData_.replayHeadStride = replayStrides[1];
    tilingData_.replayPosStride = replayStrides[2];
    tilingData_.replayDimStride = replayStrides[3];

    const uint64_t rounds =
        std::max<uint64_t>(1U, (taskCount + compileInfo_.aivNum - 1U) /
                                   compileInfo_.aivNum);
    const uint64_t used =
        std::max<uint64_t>(1U, (taskCount + rounds - 1U) / rounds);
    tilingData_.vectorCoreNum =
        static_cast<uint32_t>(std::min<uint64_t>(used, compileInfo_.aivNum));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::AnalyzeAttributes()
{
    const auto *attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "attrs", "null");
        return ge::GRAPH_FAILED;
    }
    const int64_t *window =
        attrs->GetAttrPointer<int64_t>(0);
    const bool *align = attrs->GetAttrPointer<bool>(1);
    const auto &replay = context_->GetInputShape(REPLAY)->GetOriginShape();
    if (window == nullptr || align == nullptr || *window < 1 || *window > 32 ||
        *window != replay.GetDim(2)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "attributes",
            "spec_query_len must equal replay.shape[2] and be in [1,32]");
        return ge::GRAPH_FAILED;
    }
    tilingData_.specQueryLen = static_cast<uint32_t>(*window);
    tilingData_.alignMode = *align ? 1U : 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::GetShapeAttrsInfo()
{
    if (CheckContext() != ge::GRAPH_SUCCESS ||
        AnalyzeDtype() != ge::GRAPH_SUCCESS ||
        AnalyzeShapesAndStrides() != ge::GRAPH_SUCCESS ||
        AnalyzeAttributes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::CheckUb()
{
    constexpr uint64_t matrixBytes =
        128ULL * 128ULL * sizeof(float);
    constexpr uint64_t rowBytes = 128ULL * sizeof(float);
    uint64_t total = matrixBytes + 1028U + 5U * rowBytes + 4096U;
    if (tilingData_.alignMode != 0U) {
        total += matrixBytes + 4U * rowBytes;
    }
    if (total > compileInfo_.ubSize) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "UB",
            "required=" + std::to_string(total) +
                ", available=" + std::to_string(compileInfo_.ubSize));
        return ge::GRAPH_FAILED;
    }
    tilingData_.ubCalSize = static_cast<uint32_t>(total);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::DoOpTiling()
{
    return CheckUb();
}

ge::graphStatus GdnRecoverssmCommitTiling::DoLibApiTiling()
{
    tilingKey_ =
        GET_TPL_TILING_KEY(stateDtypeTemplate_, tilingData_.alignMode != 0U);
    return ge::GRAPH_SUCCESS;
}

uint64_t GdnRecoverssmCommitTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus GdnRecoverssmCommitTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<int64_t>(GRSC_SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GdnRecoverssmCommitTiling::PostTiling()
{
    context_->SetBlockDim(std::max<uint32_t>(1U, tilingData_.vectorCoreNum));
    const size_t size = sizeof(tilingData_);
    if (memcpy_s(context_->GetRawTilingData()->GetData(),
                 context_->GetRawTilingData()->GetCapacity(), &tilingData_,
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
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(
        context);
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GdnRecoverssmCommit)
    .Tiling(TilingFunc)
    .TilingParse<GdnRecoverssmCommitCompileInfo>(TilingPrepare);

} // namespace optiling
