/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_tiling_validation.h
 * \brief CausalConv1d tiling input validation and shape/dtype parsing.
 */

#ifndef CAUSAL_CONV1D_TILING_VALIDATION_H
#define CAUSAL_CONV1D_TILING_VALIDATION_H

#include <cstring>
#include <limits>

#include "platform/platform_infos_def.h"
#include "causal_conv1d_tiling_utils.h"
#include "../op_kernel/causal_conv1d_tiling_data.h"
#include "op_common/log/log.h"
#include "../../../../common/host_utils/log.h"

namespace optiling::causal_conv1d_host {

using namespace Ops::Transformer::OpTiling;

inline ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, uint32_t &coreNum)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    coreNum = platformInfoPtr->GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    platformInfoPtr->GetLocalMemSize(fe::LocalMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SetWorkspaceSize(gert::TilingContext *context, size_t workspaceSize)
{
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

inline bool IsOptionalInputPresent(gert::TilingContext *context, uint32_t index)
{
    const auto *shapePtr = context->GetOptionalInputShape(index);
    if (shapePtr == nullptr) {
        return false;
    }
    const auto shape = shapePtr->GetStorageShape();
    const int64_t dimNum = shape.GetDimNum();
    return dimNum != 0 && !(dimNum == 1 && shape.GetDim(0) <= 0);
}

inline ge::graphStatus ResolveMetadataPair(gert::TilingContext *context, uint32_t deviceIndex, uint32_t cpuIndex,
                                           const char *deviceName, const char *cpuName, uint32_t &resolvedIndex,
                                           bool &useCpu)
{
    const bool hasDevice = IsOptionalInputPresent(context, deviceIndex);
    const bool hasCpu = IsOptionalInputPresent(context, cpuIndex);
    if (hasDevice && hasCpu) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(context->GetNodeName(), "device,cpu metadata", "both present",
            "provide only one");
        return ge::GRAPH_FAILED;
    }
    useCpu = hasCpu;
    resolvedIndex = hasCpu ? cpuIndex : deviceIndex;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ResolveMetadataInputs(gert::TilingContext *context, ResolvedMetadataInputs &inputs)
{
    OP_CHECK_IF(ResolveMetadataPair(context, QUERY_START_LOC_INDEX, QUERY_START_LOC_CPU_INDEX, "queryStartLoc",
                                    "queryStartLocCpu", inputs.queryStartLocIndex,
                                    inputs.queryStartLocUseCpu) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to resolve queryStartLoc inputs"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ResolveMetadataPair(context, CACHE_INDICES_INDEX, CACHE_INDICES_CPU_INDEX, "cacheIndices",
                                    "cacheIndicesCpu", inputs.cacheIndicesIndex,
                                    inputs.cacheIndicesUseCpu) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to resolve cacheIndices inputs"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ResolveMetadataPair(context, HAS_INITIAL_STATE_INDEX, HAS_INITIAL_STATE_CPU_INDEX, "hasInitialState",
                                    "hasInitialStateCpu", inputs.hasInitialStateIndex,
                                    inputs.hasInitialStateUseCpu) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to resolve hasInitialState inputs"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ResolveMetadataPair(context, NUM_ACCEPTED_TOKENS_INDEX, NUM_ACCEPTED_TOKENS_CPU_INDEX,
                                    "numAcceptedTokens", "numAcceptedTokensCpu", inputs.numAcceptedTokensIndex,
                                    inputs.numAcceptedTokensUseCpu) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to resolve numAcceptedTokens inputs"), return ge::GRAPH_FAILED);
    if (inputs.queryStartLocUseCpu) {
        OP_LOGW(context,
                "CausalConv1d host-array input queryStartLocCpu is deprecated and will be removed after "
                "December 2026; use the device Tensor input queryStartLoc instead.");
    }
    if (inputs.cacheIndicesUseCpu) {
        OP_LOGW(context,
                "CausalConv1d host-array input cacheIndicesCpu is deprecated and will be removed after "
                "December 2026; use the device Tensor input cacheIndices instead.");
    }
    if (inputs.hasInitialStateUseCpu) {
        OP_LOGW(context,
                "CausalConv1d host-array input hasInitialStateCpu is deprecated and will be removed after "
                "December 2026; use the device Tensor input hasInitialState instead.");
    }
    if (inputs.numAcceptedTokensUseCpu) {
        OP_LOGW(context,
                "CausalConv1d host-array input numAcceptedTokensCpu is deprecated and will be removed after "
                "December 2026; use the device Tensor input numAcceptedTokens instead.");
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus EncodeMetadataDtype(gert::TilingContext *context, uint32_t index, const char *inputName,
                                           bool allowBool, bool useCpu, int64_t &encoded)
{
    const auto *desc = context->GetOptionalInputDesc(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    const ge::DataType dtype = desc->GetDataType();
    if (useCpu) {
        if (dtype == ge::DT_INT64) {
            encoded = METADATA_DTYPE_INT64;
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(context->GetNodeName(), (std::string(inputName) + "Cpu").c_str(), "not int64", "INT64");
        return ge::GRAPH_FAILED;
    }
    if (dtype == ge::DT_INT32) {
        encoded = METADATA_DTYPE_INT32;
        return ge::GRAPH_SUCCESS;
    }
    if (allowBool && dtype == ge::DT_BOOL) {
        encoded = METADATA_DTYPE_BOOL;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(context->GetNodeName(), inputName, "not int32/bool", allowBool ? "bool/int32" : "int32");
    return ge::GRAPH_FAILED;
}

inline bool HasVisibleMetadataData(const gert::Tensor *tensor, int64_t dtype)
{
    if (tensor == nullptr) {
        return false;
    }
    if (dtype == METADATA_DTYPE_INT32) {
        return tensor->GetData<int32_t>() != nullptr;
    }
    if (dtype == METADATA_DTYPE_BOOL) {
        return tensor->GetData<uint8_t>() != nullptr;
    }
    return tensor->GetData<int64_t>() != nullptr;
}

inline int64_t ReadVisibleMetadataValue(const gert::Tensor *tensor, int64_t dtype, int64_t index)
{
    if (dtype == METADATA_DTYPE_INT32) {
        return static_cast<int64_t>(tensor->GetData<int32_t>()[index]);
    }
    if (dtype == METADATA_DTYPE_BOOL) {
        return static_cast<int64_t>(tensor->GetData<uint8_t>()[index]);
    }
    return tensor->GetData<int64_t>()[index];
}

inline ge::graphStatus GetAttrsInfo(gert::TilingContext *context, CausalConv1dAttrInfo &attrInfo)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char *activationPtr = attrs->GetAttrPointer<char>(ATTR_ACTIVATION_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, activationPtr);
    if (std::strcmp(activationPtr, "none") == 0) {
        attrInfo.activationMode = 0;
    } else if (std::strcmp(activationPtr, "silu") == 0 || std::strcmp(activationPtr, "swish") == 0) {
        attrInfo.activationMode = 1;
    } else {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "activation", activationPtr, "none/silu/swish");
        return ge::GRAPH_FAILED;
    }

    const int64_t *padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    attrInfo.padSlotId = (padSlotIdPtr == nullptr) ? -1 : *padSlotIdPtr;
    const int64_t *nullBlockIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_NULL_BLOCK_ID_INDEX);
    attrInfo.nullBlockId = (nullBlockIdPtr == nullptr) ? -1 : *nullBlockIdPtr;

    const int64_t *runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
    attrInfo.runMode = (runModePtr == nullptr) ? 0 : *runModePtr;
    if (attrInfo.runMode != 0 && attrInfo.runMode != 1) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "runMode", std::to_string(attrInfo.runMode), "0/1");
        return ge::GRAPH_FAILED;
    }

    const int64_t *headNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_NUM_INDEX);
    attrInfo.headNum = (headNumPtr == nullptr) ? 0 : *headNumPtr;
    if (attrInfo.headNum < 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "headNum", std::to_string(attrInfo.headNum), ">=0");
        return ge::GRAPH_FAILED;
    }

    const int64_t *maxQueryLenPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_QUERY_LEN_INDEX);
    attrInfo.maxQueryLen = (maxQueryLenPtr == nullptr) ? -1 : *maxQueryLenPtr;
    if (attrInfo.maxQueryLen < -1 ||
        attrInfo.maxQueryLen > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "maxQueryLen", std::to_string(attrInfo.maxQueryLen),
            "-1 or int32 range");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ValidateAlignedDim(gert::TilingContext *context, int64_t dim)
{
    if (dim % DIM_ALIGN_ELEMS != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "dim", std::to_string(dim),
            "multiple of " + std::to_string(DIM_ALIGN_ELEMS));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GetShapeDtypeInfo(gert::TilingContext *context, const CausalConv1dAttrInfo &attrInfo,
                                         CausalConv1dTilingData &tiling, bool &hasBias)
{
    const bool isDecodeMode = (attrInfo.runMode == 1);
    ResolvedMetadataInputs metadataInputs;
    OP_CHECK_IF(ResolveMetadataInputs(context, metadataInputs) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ResolveMetadataInputs error"), return ge::GRAPH_FAILED);
    tiling.activationMode = attrInfo.activationMode;
    tiling.padSlotId = attrInfo.padSlotId;
    tiling.hasNullBlock = (attrInfo.nullBlockId >= 0) ? 1 : 0;
    tiling.nullBlockId = attrInfo.nullBlockId;
    tiling.headNum = attrInfo.headNum;
    tiling.headDim = 0;
    tiling.maxQueryLen = -1;
    tiling.queryStartLocUseCpu = metadataInputs.queryStartLocUseCpu ? 1 : 0;
    tiling.cacheIndicesUseCpu = metadataInputs.cacheIndicesUseCpu ? 1 : 0;
    tiling.hasInitialStateUseCpu = metadataInputs.hasInitialStateUseCpu ? 1 : 0;
    tiling.numAcceptedTokensUseCpu = metadataInputs.numAcceptedTokensUseCpu ? 1 : 0;
    tiling.queryStartLocDtype = METADATA_DTYPE_INT32;
    tiling.cacheIndicesDtype = METADATA_DTYPE_INT32;
    tiling.hasInitialStateDtype = METADATA_DTYPE_INT32;
    tiling.numAcceptedTokensDtype = METADATA_DTYPE_INT32;

    auto xShapePtr = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    int64_t dim = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t batch = 0;
    int64_t inputMode = 0;

    if (xShape.GetDimNum() == 2) {
        if (isDecodeMode) {
            inputMode = 2;
            batch = xShape.GetDim(0);
            dim = xShape.GetDim(1);
            seqLen = 1;
            cuSeqlen = batch;
            if (batch <= 0 || dim <= 0) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "x", "non-positive", "positive batch/dim");
                return ge::GRAPH_FAILED;
            }
        } else {
            inputMode = 0;
            cuSeqlen = xShape.GetDim(0);
            dim = xShape.GetDim(1);
            seqLen = 0;
            if (dim <= 0 || cuSeqlen < 0) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "x", "invalid", "dim>0, cuSeqlen>=0");
                return ge::GRAPH_FAILED;
            }
        }
    } else if (xShape.GetDimNum() == 3) {
        inputMode = 1;
        batch = xShape.GetDim(0);
        seqLen = xShape.GetDim(1);
        dim = xShape.GetDim(2);
        cuSeqlen = batch * seqLen;
        if (batch <= 0 || dim <= 0 || seqLen <= 0) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "x", "non-positive", "positive batch/dim/seqLen");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(context->GetNodeName(), "x", "rank mismatch", "2D or 3D");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ValidateAlignedDim(context, dim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "dim alignment validation failed"),
                return ge::GRAPH_FAILED);
    if (tiling.headNum > 0) {
        if (isDecodeMode) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "headNum", ">0", "only runMode=0");
            return ge::GRAPH_FAILED;
        }
        if (tiling.headNum > dim || dim % tiling.headNum != 0) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "headNum", std::to_string(tiling.headNum),
                "in (0, dim=" + std::to_string(dim) + "] and divide dim");
            return ge::GRAPH_FAILED;
        }
        tiling.headDim = dim / tiling.headNum;
        if (tiling.headDim % DIM_ALIGN_ELEMS != 0) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "headDim", std::to_string(tiling.headDim),
                "multiple of " + std::to_string(DIM_ALIGN_ELEMS));
            return ge::GRAPH_FAILED;
        }
    }

    auto wShapePtr = context->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShapePtr);
    auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    if (wShape.GetDimNum() != 2) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "weight", "rank mismatch", "2D (width, dim)");
        return ge::GRAPH_FAILED;
    }
    const int64_t width = wShape.GetDim(0);
    const int64_t wDim = wShape.GetDim(1);
    if (wDim != dim) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "weight", "dim mismatch", "shape[1]=dim");
        return ge::GRAPH_FAILED;
    }
    if (width < 2 || width > 4) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "width", std::to_string(width), "[2,4]");
        return ge::GRAPH_FAILED;
    }

    const auto *sShapePtr = context->GetOptionalInputShape(CONV_STATES_INDEX);
    const bool hasConvStates =
        (sShapePtr != nullptr) && (sShapePtr->GetStorageShape().GetShapeSize() != 0);
    if (!hasConvStates && isDecodeMode) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(context->GetNodeName(), "convStates");
        return ge::GRAPH_FAILED;
    }
    int64_t numCacheLines = 0;
    int64_t stateLen = width - 1;
    if (hasConvStates) {
        auto sShape = EnsureNotScalar(sShapePtr->GetStorageShape());
        if (sShape.GetDimNum() != 3) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "convStates", "rank mismatch",
                "3D (num_cache_lines, state_len, dim)");
            return ge::GRAPH_FAILED;
        }
        numCacheLines = sShape.GetDim(0);
        stateLen = sShape.GetDim(1);
        const int64_t sDim = sShape.GetDim(2);
        if (numCacheLines <= 0) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "convStates.shape[0]", std::to_string(numCacheLines), ">0");
            return ge::GRAPH_FAILED;
        }
        if (sDim != dim) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "convStates", "dim mismatch", "shape[2]=dim");
            return ge::GRAPH_FAILED;
        }
        if (stateLen < (width - 1)) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "convStates.shape[1]", std::to_string(stateLen),
                ">=width-1=" + std::to_string(width - 1));
            return ge::GRAPH_FAILED;
        }

        const auto inputStride = context->GetOptionalInputStride(CONV_STATES_INDEX);
        if (inputStride != nullptr && inputStride->GetDimNum() == 3) {
            const int64_t stride0 = inputStride->GetStride(0);
            const int64_t stride1 = inputStride->GetStride(1);
            const int64_t stride2 = inputStride->GetStride(2);
            if (stride0 <= 0 || stride1 <= 0 || stride2 != 1) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "convStates stride", "invalid", "stride0>0, stride1>0, stride2=1");
                return ge::GRAPH_FAILED;
            }
            tiling.convStateStride0 = stride0;
            tiling.convStateStride1 = stride1;
        } else {
            tiling.convStateStride0 = stateLen * sDim;
            tiling.convStateStride1 = sDim;
        }
    } else {

        tiling.convStateStride0 = 0;
        tiling.convStateStride1 = 0;
    }

    auto qslShapePtr = context->GetOptionalInputShape(metadataInputs.queryStartLocIndex);
    const gert::CompileTimeTensorDesc *qslDesc = context->GetOptionalInputDesc(metadataInputs.queryStartLocIndex);
    bool qslAbsent = true;
    int64_t qslSize = 0;
    if (qslShapePtr != nullptr) {
        const auto qslStorageShape = qslShapePtr->GetStorageShape();
        const int64_t qslDimNum = qslStorageShape.GetDimNum();
        qslAbsent = (qslDimNum == 0) || (qslDimNum == 1 && qslStorageShape.GetDim(0) <= 0);
        if (!qslAbsent) {
            auto qslShape = EnsureNotScalar(qslStorageShape);
            if (qslShape.GetDimNum() != 1) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "queryStartLoc", "rank mismatch", "1D");
                return ge::GRAPH_FAILED;
            }
            qslSize = qslShape.GetDim(0);
            if (qslSize < 1) {
                OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(context->GetNodeName(), "queryStartLoc", std::to_string(qslSize), ">=1");
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
            OP_CHECK_IF(EncodeMetadataDtype(context, metadataInputs.queryStartLocIndex, "queryStartLoc", false,
                                            metadataInputs.queryStartLocUseCpu,
                                            tiling.queryStartLocDtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context, "invalid queryStartLoc dtype"), return ge::GRAPH_FAILED);
        }
    }

    if (qslAbsent) {
        if (inputMode == 0) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(context->GetNodeName(), "queryStartLoc");
            return ge::GRAPH_FAILED;
        }
        qslSize = batch + 1;
    }

    if (cuSeqlen > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "cuSeqlen", std::to_string(cuSeqlen), "int32 range");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor *qslTensor = nullptr;
    bool qslDataVisible = false;
    int64_t observedMaxQueryLen = 0;
    if (!qslAbsent && metadataInputs.queryStartLocUseCpu) {
        qslTensor = context->GetOptionalInputTensor(metadataInputs.queryStartLocIndex);
        qslDataVisible = HasVisibleMetadataData(qslTensor, tiling.queryStartLocDtype);
        if (qslDataVisible) {
            if (ReadVisibleMetadataValue(qslTensor, tiling.queryStartLocDtype, 0) != 0) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "queryStartLoc[0]", "not 0", "0");
                return ge::GRAPH_FAILED;
            }
            const int64_t qslLast = ReadVisibleMetadataValue(qslTensor, tiling.queryStartLocDtype, qslSize - 1);
            if (qslLast != cuSeqlen) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "queryStartLoc[last]", std::to_string(qslLast),
                    std::to_string(cuSeqlen));
                return ge::GRAPH_FAILED;
            }
            for (int64_t i = 0; i + 1 < qslSize; ++i) {
                const int64_t cur = ReadVisibleMetadataValue(qslTensor, tiling.queryStartLocDtype, i);
                const int64_t nxt = ReadVisibleMetadataValue(qslTensor, tiling.queryStartLocDtype, i + 1);
                if (cur < 0 || cur > cuSeqlen) {
                    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "queryStartLoc[i]", std::to_string(cur),
                        "in [0, cuSeqlen=" + std::to_string(cuSeqlen) + "]");
                    return ge::GRAPH_FAILED;
                }
                if (nxt < 0 || nxt > cuSeqlen) {
                    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "queryStartLoc[i+1]", std::to_string(nxt),
                        "in [0, cuSeqlen=" + std::to_string(cuSeqlen) + "]");
                    return ge::GRAPH_FAILED;
                }
                if (nxt < cur) {
                    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "queryStartLoc",
                    "[" + std::to_string(i + 1) + "]=" + std::to_string(nxt),
                    "non-decreasing, >= queryStartLoc[" + std::to_string(i) + "]=" + std::to_string(cur));
                    return ge::GRAPH_FAILED;
                }
                const int64_t segmentLen = nxt - cur;
                observedMaxQueryLen = (segmentLen > observedMaxQueryLen) ? segmentLen : observedMaxQueryLen;
            }
        }
    }

    if (!qslAbsent && isDecodeMode && inputMode == 2) {
        const int64_t batchFromQsl = qslSize - 1;
        if (batchFromQsl != batch) {
            inputMode = 0;
            cuSeqlen = xShape.GetDim(0);
            batch = batchFromQsl;
            seqLen = 0;
            if (dim <= 0 || cuSeqlen < 0 || batch < 0) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "x/queryStartLoc", "invalid",
                    "dim>0, cuSeqlen>=0, batch>=0");
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (inputMode == 0) {
        batch = qslSize - 1;
    }
    if (!qslAbsent && (inputMode == 1 || inputMode == 2)) {
        if (qslSize != batch + 1) {
            OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(context->GetNodeName(), "queryStartLoc", std::to_string(qslSize),
                std::to_string(batch + 1));
            return ge::GRAPH_FAILED;
        }
    }
    if (isDecodeMode) {
        const int64_t decodeSeqLen = (inputMode == 1) ? seqLen : 1;
        if (decodeSeqLen < 1) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "seqlen", std::to_string(decodeSeqLen), ">=1");
            return ge::GRAPH_FAILED;
        }
    }
    if (isDecodeMode && inputMode == 0) {
        tiling.maxQueryLen = attrInfo.maxQueryLen;
        if (qslDataVisible) {
            if (attrInfo.maxQueryLen >= 0 && attrInfo.maxQueryLen < observedMaxQueryLen) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "maxQueryLen", std::to_string(attrInfo.maxQueryLen),
                    ">= observed max segment " + std::to_string(observedMaxQueryLen));
                return ge::GRAPH_FAILED;
            }
            // Preserve raw callers that omit the new attr when deprecated host
            // metadata exposes the exact upper bound at tiling time.
            if (tiling.maxQueryLen < 0) {
                tiling.maxQueryLen = observedMaxQueryLen;
            }
        }
    }

    tiling.hasCacheIndices = 0;
    bool ciAbsent = true;
    auto ciShapePtr = context->GetOptionalInputShape(metadataInputs.cacheIndicesIndex);
    if (ciShapePtr != nullptr) {
        const auto ciStorageShape = ciShapePtr->GetStorageShape();
        const int64_t ciDimNum = ciStorageShape.GetDimNum();
        ciAbsent = (ciDimNum == 0) || (ciDimNum == 1 && ciStorageShape.GetDim(0) <= 0);
        if (!ciAbsent) {
            auto ciShape = EnsureNotScalar(ciStorageShape);
            if (ciShape.GetDimNum() != 1) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "cacheIndices", "rank mismatch", "1D");
                return ge::GRAPH_FAILED;
            }
            if (ciShape.GetDim(0) != batch) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "cacheIndices", "size mismatch", "=batch");
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(EncodeMetadataDtype(context, metadataInputs.cacheIndicesIndex, "cacheIndices", false,
                                            metadataInputs.cacheIndicesUseCpu,
                                            tiling.cacheIndicesDtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context, "invalid cacheIndices dtype"), return ge::GRAPH_FAILED);
            tiling.hasCacheIndices = 1;

            const gert::Tensor *ciTensor = metadataInputs.cacheIndicesUseCpu
                                               ? context->GetOptionalInputTensor(metadataInputs.cacheIndicesIndex)
                                               : nullptr;
            if (HasVisibleMetadataData(ciTensor, tiling.cacheIndicesDtype)) {
                for (int64_t i = 0; i < batch; ++i) {
                    const int64_t v = ReadVisibleMetadataValue(ciTensor, tiling.cacheIndicesDtype, i);
                    const bool isPadSlot = (v == tiling.padSlotId);
                    const bool isNullBlock = (tiling.hasNullBlock != 0 && v == tiling.nullBlockId);
                    if (isPadSlot || isNullBlock) {
                        continue;
                    }
                    if (hasConvStates) {
                        if (v > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
                            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "cacheIndices[i]", std::to_string(v),
                                "int32 range");
                            return ge::GRAPH_FAILED;
                        }
                        if (v < 0 || v >= numCacheLines) {
                            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "cacheIndices[i]", std::to_string(v),
                                "in [0, num_cache_lines=" + std::to_string(numCacheLines) + ")");
                            return ge::GRAPH_FAILED;
                        }
                    }
                }
            }
        }
    }
    if (ciAbsent) {
        if (hasConvStates && numCacheLines < batch) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "num_cache_lines", std::to_string(numCacheLines),
                ">= batch=" + std::to_string(batch));
            return ge::GRAPH_FAILED;
        }
    }

    tiling.hasInitialState = 0;
    auto hisShapePtr = context->GetOptionalInputShape(metadataInputs.hasInitialStateIndex);
    if (hisShapePtr != nullptr) {
        const auto hisStorageShape = hisShapePtr->GetStorageShape();
        const int64_t hisDimNum = hisStorageShape.GetDimNum();
        const bool hisAbsent = (hisDimNum == 0) || (hisDimNum == 1 && hisStorageShape.GetDim(0) <= 0);
        if (!hisAbsent) {
            if (isDecodeMode) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "hasInitialState", "present", "only runMode=0");
                return ge::GRAPH_FAILED;
            }
            auto hisShape = EnsureNotScalar(hisStorageShape);
            if (hisShape.GetDimNum() != 1) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "hasInitialState", "rank mismatch", "1D");
                return ge::GRAPH_FAILED;
            }
            if (hisShape.GetDim(0) != batch) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "hasInitialState", "size mismatch", "=batch");
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(EncodeMetadataDtype(context, metadataInputs.hasInitialStateIndex, "hasInitialState", true,
                                            metadataInputs.hasInitialStateUseCpu,
                                            tiling.hasInitialStateDtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context, "invalid hasInitialState dtype"), return ge::GRAPH_FAILED);

            const gert::Tensor *hisTensor = metadataInputs.hasInitialStateUseCpu
                                                ? context->GetOptionalInputTensor(metadataInputs.hasInitialStateIndex)
                                                : nullptr;
            if (HasVisibleMetadataData(hisTensor, tiling.hasInitialStateDtype)) {
                bool hasTrueInitialState = false;
                for (int64_t i = 0; i < batch; ++i) {
                    const int64_t v = ReadVisibleMetadataValue(hisTensor, tiling.hasInitialStateDtype, i);
                    if (v != 0 && v != 1) {
                        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "hasInitialState[i]", std::to_string(v), "0/1");
                        return ge::GRAPH_FAILED;
                    }
                    hasTrueInitialState = hasTrueInitialState || (v != 0);
                }
                tiling.hasInitialState = hasTrueInitialState ? 1 : 0;
            } else {

                tiling.hasInitialState = 1;
            }
        }
    }
    if (!hasConvStates) {

        tiling.hasInitialState = 0;
    }

    tiling.hasNumAcceptedTokens = 0;
    auto natShapePtr = context->GetOptionalInputShape(metadataInputs.numAcceptedTokensIndex);
    if (natShapePtr != nullptr) {
        const auto natStorageShape = natShapePtr->GetStorageShape();
        const int64_t natDimNum = natStorageShape.GetDimNum();
        const bool natAbsent = (natDimNum == 0) || (natDimNum == 1 && natStorageShape.GetDim(0) <= 0);
        if (!natAbsent) {
            if (!isDecodeMode) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "numAcceptedTokens", "present", "only runMode=1");
                return ge::GRAPH_FAILED;
            }
            auto natShape = EnsureNotScalar(natStorageShape);
            if (natShape.GetDimNum() != 1) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "numAcceptedTokens", "rank mismatch", "1D");
                return ge::GRAPH_FAILED;
            }
            if (natShape.GetDim(0) != batch) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "numAcceptedTokens", "size mismatch", "=batch");
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(EncodeMetadataDtype(context, metadataInputs.numAcceptedTokensIndex, "numAcceptedTokens", false,
                                            metadataInputs.numAcceptedTokensUseCpu,
                                            tiling.numAcceptedTokensDtype) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context, "invalid numAcceptedTokens dtype"), return ge::GRAPH_FAILED);

            if (inputMode == 1 || (inputMode == 0 && tiling.maxQueryLen >= 0)) {
                const int64_t effectiveSeqLen = (inputMode == 1) ? seqLen : tiling.maxQueryLen;
                const int64_t reqStateLen = (width - 1) + ((effectiveSeqLen > 0) ? (effectiveSeqLen - 1) : 0);
                if (stateLen < reqStateLen) {
                    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "stateLen", std::to_string(stateLen),
                        ">= " + std::to_string(reqStateLen));
                    return ge::GRAPH_FAILED;
                }
            }

            const gert::Tensor *natTensor = metadataInputs.numAcceptedTokensUseCpu
                                                ? context->GetOptionalInputTensor(metadataInputs.numAcceptedTokensIndex)
                                                : nullptr;
            if (HasVisibleMetadataData(natTensor, tiling.numAcceptedTokensDtype)) {
                for (int64_t i = 0; i < batch; ++i) {
                    const int64_t a = ReadVisibleMetadataValue(natTensor, tiling.numAcceptedTokensDtype, i);
                    if (a < 0) {
                        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "numAcceptedTokens[i]", std::to_string(a), ">=0");
                        return ge::GRAPH_FAILED;
                    }

                    if (inputMode == 2) {
                        if (a > 1) {
                            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "numAcceptedTokens[i]", std::to_string(a), "<=1");
                            return ge::GRAPH_FAILED;
                        }
                    } else if (inputMode == 1) {
                        if (a > seqLen) {
                            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "numAcceptedTokens[i]", std::to_string(a),
                                "<=seqlen=" + std::to_string(seqLen));
                            return ge::GRAPH_FAILED;
                        }
                    } else if (inputMode == 0 && qslDataVisible) {
                        const int64_t lenI =
                            ReadVisibleMetadataValue(qslTensor, tiling.queryStartLocDtype, i + 1) -
                            ReadVisibleMetadataValue(qslTensor, tiling.queryStartLocDtype, i);
                        if (a > lenI) {
                            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "numAcceptedTokens[i]", std::to_string(a),
                                "<=segment length=" + std::to_string(lenI));
                            return ge::GRAPH_FAILED;
                        }
                    }
                }
            }

            tiling.hasNumAcceptedTokens = 1;
        }
    }

    tiling.hasBias = 0;
    hasBias = false;
    auto biasShapePtr = context->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr != nullptr) {
        const auto biasStorageShape = biasShapePtr->GetStorageShape();
        const int64_t biasDimNum = biasStorageShape.GetDimNum();
        const bool biasAbsent = (biasDimNum == 0) || (biasDimNum == 1 && biasStorageShape.GetDim(0) <= 0);
        if (!biasAbsent) {
            auto biasShape = EnsureNotScalar(biasStorageShape);
            if (biasShape.GetDimNum() != 1) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "bias", "rank mismatch", "1D (dim,)");
                return ge::GRAPH_FAILED;
            }
            if (biasShape.GetDim(0) != dim) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(context->GetNodeName(), "bias", "size mismatch", "=dim");
                return ge::GRAPH_FAILED;
            }
            tiling.hasBias = 1;
            hasBias = true;
        }
    }

    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    if (xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(context->GetNodeName(), "x", fla_npu::host_utils::DfxDtypeString(xDtype), "BF16,FP16");
        return ge::GRAPH_FAILED;
    }

    auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    if (wDesc->GetDataType() != xDtype) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(context->GetNodeName(), "weight,x", "dtype mismatch", "equal");
        return ge::GRAPH_FAILED;
    }

    if (hasBias) {
        auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
        if (biasDesc->GetDataType() != xDtype) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(context->GetNodeName(), "bias,x", "dtype mismatch", "equal");
            return ge::GRAPH_FAILED;
        }
    }

    const auto *sDesc = context->GetOptionalInputDesc(CONV_STATES_INDEX);
    if (sDesc != nullptr) {
        if (sDesc->GetDataType() != xDtype) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(context->GetNodeName(), "convStates,x", "dtype mismatch", "equal");
            return ge::GRAPH_FAILED;
        }
    }

    if (tiling.hasNumAcceptedTokens == 1) {
        if (width != 4) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(context->GetNodeName(), "width", std::to_string(width), "4");
            return ge::GRAPH_FAILED;
        }
    }

    tiling.dim = dim;
    tiling.cuSeqlen = cuSeqlen;
    tiling.seqLen = seqLen;
    tiling.inputMode = inputMode;
    tiling.width = width;
    tiling.stateLen = stateLen;
    tiling.numCacheLines = numCacheLines;
    tiling.batch = batch;
    return ge::GRAPH_SUCCESS;
}

}

#endif
