/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "compact_conv_state_tiling.h"

#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/compact_conv_state_tiling_key.h"

#include <algorithm>
#include <limits>
#include <string>

namespace optiling {
namespace {
constexpr size_t CONV_STATE = 0;
constexpr size_t BASE_ADDRS = 1;
constexpr size_t BLOCK_STRIDES = 2;
constexpr size_t DIM_STRIDES = 3;
constexpr size_t TOKEN_STRIDES = 4;
constexpr size_t STATE_INDICES = 5;
constexpr size_t COMMIT_LENS = 6;
constexpr size_t FINAL_INDICES = 7;
constexpr size_t BOUNDARY_INDICES = 8;
constexpr size_t BOUNDARY_LENS = 9;

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

bool IsDenseVector(const gert::TilingContext *context, size_t index)
{
    const auto *shape = context->GetInputShape(index);
    const auto *stride = context->GetInputStride(index);
    if (shape == nullptr) {
        return false;
    }
    const int64_t batch = shape->GetOriginShape().GetDim(0);
    if (batch == 0) {
        return true;
    }
    return stride == nullptr || stride->GetDimNum() == 0 || stride->GetStride(0) == 1;
}
} // namespace

REGISTER_OPS_TILING_TEMPLATE(CompactConvState, CompactConvStateTiling, 0);

ge::graphStatus CompactConvStateTiling::GetPlatformInfo()
{
    auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (platform.GetCurNpuArch() != NpuArch::DAV_3510 || platform.GetCoreNumAiv() == 0U || ubSize == 0U) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompactConvStateTiling::CheckContext()
{
    for (size_t index = CONV_STATE; index <= BOUNDARY_LENS; ++index) {
        if (context_->GetInputShape(index) == nullptr || context_->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    return context_->GetOutputShape(0) != nullptr && context_->GetOutputDesc(0) != nullptr ? ge::GRAPH_SUCCESS :
                                                                                             ge::GRAPH_FAILED;
}

ge::graphStatus CompactConvStateTiling::AnalyzeDtype()
{
    const auto dtype = [&](size_t index) { return context_->GetInputDesc(index)->GetDataType(); };
    const auto stateType = dtype(CONV_STATE);
    if (stateType != ge::DT_BF16 && stateType != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const auto isInteger = [&](size_t index) { return dtype(index) == ge::DT_INT32 || dtype(index) == ge::DT_INT64; };
    if (!isInteger(STATE_INDICES) || !isInteger(COMMIT_LENS) || !isInteger(FINAL_INDICES) ||
        !isInteger(BOUNDARY_INDICES) || !isInteger(BOUNDARY_LENS)) {
        return ge::GRAPH_FAILED;
    }
    const auto planType = dtype(COMMIT_LENS);
    if (dtype(FINAL_INDICES) != planType || dtype(BOUNDARY_INDICES) != planType || dtype(BOUNDARY_LENS) != planType) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = BASE_ADDRS; index <= TOKEN_STRIDES; ++index) {
        if (dtype(index) != ge::DT_INT64) {
            return ge::GRAPH_FAILED;
        }
    }
    if (context_->GetOutputDesc(0)->GetDataType() != stateType) {
        return ge::GRAPH_FAILED;
    }
    stateDtypeTemplate_ = stateType == ge::DT_FLOAT ? CCS_TPL_FP32 : CCS_TPL_BF16;
    stateIndexDtypeTemplate_ = dtype(STATE_INDICES) == ge::DT_INT64 ? CCS_TPL_INT64 : CCS_TPL_INT32;
    planDtypeTemplate_ = planType == ge::DT_INT64 ? CCS_TPL_INT64 : CCS_TPL_INT32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompactConvStateTiling::AnalyzeShapesAndStrides()
{
    const auto &state = context_->GetInputShape(CONV_STATE)->GetOriginShape();
    if (state.GetDimNum() != 3 || state.GetDim(0) <= 0 || state.GetDim(1) <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t channels = state.GetDim(1);
    if (state.GetDim(2) < static_cast<int64_t>(tilingData_.historyLen)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t layers = context_->GetInputShape(BASE_ADDRS)->GetOriginShape().GetDim(0);
    if (layers <= 0) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = BASE_ADDRS; index <= TOKEN_STRIDES; ++index) {
        const auto &shape = context_->GetInputShape(index)->GetOriginShape();
        if (shape.GetDimNum() != 1 || shape.GetDim(0) != layers || !IsDenseVector(context_, index)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto &source = context_->GetInputShape(STATE_INDICES)->GetOriginShape();
    if (source.GetDimNum() != 1 || source.GetDim(0) < 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = source.GetDim(0);
    for (size_t index = COMMIT_LENS; index <= BOUNDARY_LENS; ++index) {
        const auto &shape = context_->GetInputShape(index)->GetOriginShape();
        if (shape.GetDimNum() != 1 || shape.GetDim(0) != batch || !IsDenseVector(context_, index)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *sourceStride = context_->GetInputStride(STATE_INDICES);
    const int64_t rawSourceStride =
        sourceStride != nullptr && sourceStride->GetDimNum() != 0 ? sourceStride->GetStride(0) : 1;
    if (batch > 0 && rawSourceStride <= 0) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.stateIndicesStride = rawSourceStride > 0 ? static_cast<uint64_t>(rawSourceStride) : 1U;
    if (channels > std::numeric_limits<uint32_t>::max() || layers > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const auto *stateStride = context_->GetInputStride(CONV_STATE);
    if (stateStride != nullptr && stateStride->GetDimNum() != 0) {
        if (stateStride->GetDimNum() != 3 || stateStride->GetStride(1) <= 0 || stateStride->GetStride(2) <= 0) {
            return ge::GRAPH_FAILED;
        }
    }
    const uint64_t tileChannels =
        std::min<uint64_t>(static_cast<uint64_t>(channels), static_cast<uint64_t>(CCS_CHANNEL_TILE_256));
    constexpr uint64_t copyAlignment = 32U;
    const uint64_t elementBytes = stateDtypeTemplate_ == CCS_TPL_FP32 ? sizeof(float) : sizeof(uint16_t);
    const auto alignUp = [](uint64_t value, uint64_t alignment) {
        return (value + alignment - 1U) / alignment * alignment;
    };
    const uint64_t channelLastBytes =
        static_cast<uint64_t>(tilingData_.historyLen) * alignUp(tileChannels * elementBytes, copyAlignment);
    const uint64_t channelFirstBytes =
        tileChannels * alignUp(static_cast<uint64_t>(tilingData_.historyLen) * elementBytes, copyAlignment);
    tilingData_.bufferElements = std::max(channelLastBytes, channelFirstBytes) / elementBytes;
    if (tilingData_.bufferElements > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tilesPerRowLayer =
        (static_cast<uint32_t>(channels) + CCS_CHANNEL_TILE_256 - 1U) / CCS_CHANNEL_TILE_256;
    const uint64_t taskCount = static_cast<uint64_t>(batch) * static_cast<uint64_t>(layers) * tilesPerRowLayer;
    if (taskCount > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.taskCount = static_cast<uint32_t>(taskCount);
    tilingData_.channelCount = static_cast<uint32_t>(channels);
    tilingData_.layerCount = static_cast<uint32_t>(layers);
    tilingData_.tilesPerRowLayer = tilesPerRowLayer;

    auto *platformInfo = context_->GetPlatformInfo();
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint64_t aivNum = platform.GetCoreNumAiv();
    const uint64_t rounds = std::max<uint64_t>(1U, (taskCount + aivNum - 1U) / aivNum);
    const uint64_t used = std::max<uint64_t>(1U, (taskCount + rounds - 1U) / rounds);
    context_->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(used, aivNum)));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompactConvStateTiling::AnalyzeAttributes()
{
    const auto *attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *historyLen = attrs->GetAttrPointer<int64_t>(0);
    const bool *alignMode = attrs->GetAttrPointer<bool>(1);
    const int64_t *dimStride = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *nullBlockId = attrs->GetAttrPointer<int64_t>(3);
    if (historyLen == nullptr || alignMode == nullptr || *historyLen < 1 ||
        *historyLen > static_cast<int64_t>(CCS_HISTORY_MAX_32)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t resolvedDimStride = dimStride == nullptr ? -1 : *dimStride;
    const int64_t resolvedNullBlockId = nullBlockId == nullptr ? 0 : *nullBlockId;
    if (resolvedDimStride == 0 || resolvedDimStride < -1 || resolvedNullBlockId < 0) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.historyLen = static_cast<uint32_t>(*historyLen);
    alignMode_ = *alignMode;
    dimStride_ = resolvedDimStride;
    sharedDimStride_ = resolvedDimStride > 0;
    tilingData_.sharedDimStride = sharedDimStride_ ? static_cast<uint64_t>(resolvedDimStride) : 0U;
    tilingData_.nullBlockId = resolvedNullBlockId;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompactConvStateTiling::GetShapeAttrsInfo()
{
    if (CheckContext() != ge::GRAPH_SUCCESS || AnalyzeDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (AnalyzeAttributes() != ge::GRAPH_SUCCESS || AnalyzeShapesAndStrides() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompactConvStateTiling::CheckUb()
{
    const uint64_t elementBytes = stateDtypeTemplate_ == CCS_TPL_FP32 ? sizeof(float) : sizeof(uint16_t);
    constexpr uint64_t buffers = CCS_BUFFERS_2 * CCS_ROLE_BUFFERS_2;
    const uint64_t required = buffers * tilingData_.bufferElements * elementBytes + 4096U;
    uint64_t ubSize = 0U;
    const auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    return required <= ubSize ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus CompactConvStateTiling::DoOpTiling()
{
    return CheckUb();
}

ge::graphStatus CompactConvStateTiling::DoLibApiTiling()
{
    tilingKey_ = GET_TPL_TILING_KEY(stateDtypeTemplate_, alignMode_, planDtypeTemplate_, stateIndexDtypeTemplate_,
                                    sharedDimStride_);
    return ge::GRAPH_SUCCESS;
}

uint64_t CompactConvStateTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus CompactConvStateTiling::GetWorkspaceSize()
{
    workspaceSize_ = static_cast<int64_t>(CCS_SYS_WORKSPACE_SIZE);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompactConvStateTiling::PostTiling()
{
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
    return context == nullptr ? ge::GRAPH_FAILED :
                                Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CompactConvState).Tiling(TilingFunc).TilingParse<CompactConvStateCompileInfo>(TilingPrepare);

} // namespace optiling
