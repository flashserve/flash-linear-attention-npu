/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_bwd_tiling.cpp
 * \brief Causal Conv1D backward tiling implementation
 *
 * Tiling keys:
 *   key=0 : Main kernel. Activation is selected by tiling data at runtime.
 */

#include "op_common/log/log.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "platform/soc_spec.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "causal_conv1d_bwd_tiling.h"
#include <algorithm>
#include <cstring>
#include <string>

namespace {
    constexpr const char *OP_NAME = "CausalConv1dBwd";
    constexpr uint32_t BYTE_BLOCK = 32;
    constexpr uint32_t X_INPUT_INDEX = 0;
    constexpr uint32_t Y_INPUT_INDEX = 1;
    constexpr uint32_t WEIGHT_INPUT_INDEX = 2;
    constexpr uint32_t DY_INPUT_INDEX = 3;
    constexpr uint32_t INITIAL_STATE_INPUT_INDEX = 4;
    constexpr uint32_t DHT_INPUT_INDEX = 5;
    constexpr uint32_t QUERY_START_LOC_INPUT_INDEX = 6;
    constexpr uint32_t DIM_INDEX0 = 0;
    constexpr uint32_t DIM_INDEX1 = 1;
    constexpr uint32_t DIM_INDEX2 = 2;
    constexpr uint32_t DIM_INDEX3 = 3;
    constexpr uint32_t ATTR_ACTIVATION_INDEX = 0;
    constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 1;
    constexpr uint32_t RESERVED_UB = 16 * 1024;
    constexpr uint32_t B16_DTYPE_SIZE = 2U;
    constexpr uint32_t FP32_DTYPE_SIZE = 4U;
    constexpr uint32_t BLOCK_ALIGN_NUM = 8U;
    constexpr uint32_t DIRECT_FAST_BT_MAX = 224U;
    constexpr uint32_t DIRECT_FAST_BT_MIN = 64U;
    constexpr uint32_t DIRECT_FAST_BT_STEP = 32U;
    constexpr uint32_t DIRECT_910B_BT_BD64 = 104U;
    constexpr uint32_t DIRECT_910B_BT_BD128 = 55U;
    constexpr uint32_t ARCH35_DIRECT_W_MAX = 8U;
    constexpr uint32_t ACTIVATION_NONE = 0;
    constexpr uint32_t ACTIVATION_SILU = 1;
    constexpr uint32_t ACTIVATION_SWISH = 2;
    constexpr uint32_t INPUT_LAYOUT_BSND = 0; // y/dy: [B, T, N*D].
    constexpr uint32_t INPUT_LAYOUT_TND = 1;  // y/dy: [total_tokens, N*D].
    constexpr uint32_t INPUT_LAYOUT_BNSD = 2; // y/dy: [B, N, T, D].
    constexpr uint32_t INPUT_LAYOUT_NTD = 3;  // y/dy: [N, total_tokens, D].
}

namespace optiling {
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using namespace Ops::Transformer::OpTiling;

static std::string ShapeToString(const gert::Shape &shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape.GetDim(i));
    }
    return result + "]";
}

static ge::graphStatus CheckDtypes(gert::TilingContext *context, ge::DataType xDtype)
{
    const uint32_t requiredInputIndices[] = {WEIGHT_INPUT_INDEX, DY_INPUT_INDEX};
    const char *requiredInputNames[] = {"weight", "dy"};
    for (size_t i = 0; i < sizeof(requiredInputIndices) / sizeof(requiredInputIndices[0]); i++) {
        const auto *desc = context->GetInputDesc(requiredInputIndices[i]);
        OP_CHECK_IF(
            desc == nullptr,
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: required input %s descriptor is unavailable", __func__,
                                   requiredInputNames[i]),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(desc->GetDataType() != xDtype,
                    OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, requiredInputNames[i],
                                              std::to_string(static_cast<int>(desc->GetDataType())),
                                              std::to_string(static_cast<int>(xDtype))),
                    return ge::GRAPH_FAILED);
    }

    const uint32_t optionalInputIndices[] = {Y_INPUT_INDEX, INITIAL_STATE_INPUT_INDEX, DHT_INPUT_INDEX};
    const char *optionalInputNames[] = {"y", "initial_state", "dht"};
    for (size_t i = 0; i < sizeof(optionalInputIndices) / sizeof(optionalInputIndices[0]); i++) {
        const auto *desc = context->GetOptionalInputDesc(optionalInputIndices[i]);
        OP_CHECK_IF(desc != nullptr && desc->GetDataType() != xDtype,
                    OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, optionalInputNames[i],
                                              std::to_string(static_cast<int>(desc->GetDataType())),
                                              std::to_string(static_cast<int>(xDtype))),
                    return ge::GRAPH_FAILED);
    }

    const char *outputNames[] = {"dx", "dw", "db", "dh0"};
    for (size_t i = 0; i < sizeof(outputNames) / sizeof(outputNames[0]); i++) {
        const auto *desc = context->GetOutputDesc(i);
        OP_CHECK_IF(desc != nullptr && desc->IsExist() && desc->GetDataType() != xDtype,
                    OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, outputNames[i],
                                              std::to_string(static_cast<int>(desc->GetDataType())),
                                              std::to_string(static_cast<int>(xDtype))),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CausalConv1dBwdTilingFunc(gert::TilingContext *context)
{
    CausalConv1dBwdTilingData *tiling = context->GetTilingData<CausalConv1dBwdTilingData>();
    OP_CHECK_IF(
        tiling == nullptr,
        OP_LOGE(OP_NAME, "%s",
                std::string("tiling data: actual=null; expected=valid CausalConv1dBwdTilingData buffer").c_str()),
        return ge::GRAPH_FAILED);
    const auto memsetRc = memset_s(tiling, sizeof(CausalConv1dBwdTilingData), 0, sizeof(CausalConv1dBwdTilingData));
    OP_CHECK_IF(memsetRc != EOK,
                OP_LOGE(OP_NAME, "%s",
                        std::string("memset_s(tiling): rc=" + std::to_string(memsetRc) +
                                    "; expected rc=0 (EOK); failed to initialize tiling data")
                            .c_str()),
                return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(X_INPUT_INDEX)->GetStorageShape();
    auto dyShape = context->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    auto wShape = context->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto xDesc = context->GetInputDesc(X_INPUT_INDEX);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: required input x descriptor is unavailable", __func__),
                return ge::GRAPH_FAILED);
    auto xDtype = xDesc->GetDataType();
    if (CheckDtypes(context, xDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    bool isB16Input = (xDtype == ge::DT_BF16 || xDtype == ge::DT_FLOAT16);
    bool isFp32Input = (xDtype == ge::DT_FLOAT);
    uint32_t B = 0;
    uint32_t T = 0;
    uint32_t D = 0;
    uint32_t totalTokens = 0;
    uint32_t inputMode = 1;
    uint32_t inputLayout = INPUT_LAYOUT_BSND;
    uint32_t inputN = 1;
    uint32_t inputHeadDim = 0;

    uint32_t activation = ACTIVATION_NONE;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > ATTR_ACTIVATION_INDEX) {
        const int64_t *activationPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_INDEX);
        OP_CHECK_IF(activationPtr == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "activation"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(*activationPtr != ACTIVATION_NONE && *activationPtr != ACTIVATION_SILU &&
                        *activationPtr != ACTIVATION_SWISH,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "activation", std::to_string(*activationPtr),
                                                          "expected 0 (None), 1 (SiLU) or 2 (Swish)"),
                    return ge::GRAPH_FAILED);
        activation = static_cast<uint32_t>(*activationPtr);
    }
    const char *inputLayoutStr = "BSND";
    if (attrs != nullptr && attrs->GetAttrNum() > ATTR_INPUT_LAYOUT_INDEX) {
        const char *attrLayout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
        if (attrLayout != nullptr) {
            inputLayoutStr = attrLayout;
        }
    }
    if (std::strcmp(inputLayoutStr, "BNSD") == 0) {
        inputLayout = INPUT_LAYOUT_BNSD;
    } else if (std::strcmp(inputLayoutStr, "NTD") == 0) {
        inputLayout = INPUT_LAYOUT_NTD;
    } else if (std::strcmp(inputLayoutStr, "TND") == 0) {
        inputLayout = INPUT_LAYOUT_TND;
    } else if (std::strcmp(inputLayoutStr, "BSND") == 0 || std::strcmp(inputLayoutStr, "BSH") == 0) {
        inputLayout = INPUT_LAYOUT_BSND;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "inputLayout", inputLayoutStr, "expected BSND/BSH/TND/BNSD/NTD");
        return ge::GRAPH_FAILED;
    }

    if (inputLayout == INPUT_LAYOUT_BNSD) {
        OP_CHECK_IF(xShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", ShapeToString(xShape),
                                                          "expected logical rank 3 [B, T, D] for BNSD"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dyShape.GetDimNum() != 4,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "dy", ShapeToString(dyShape),
                                                          "expected rank 4 [B, N, T, Dh] for BNSD"),
                    return ge::GRAPH_FAILED);
        inputMode = 1;
        B = xShape.GetDim(DIM_INDEX0);
        T = xShape.GetDim(DIM_INDEX1);
        D = xShape.GetDim(DIM_INDEX2);
        inputN = dyShape.GetDim(DIM_INDEX1);
        inputHeadDim = dyShape.GetDim(DIM_INDEX3);
        OP_CHECK_IF(static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != B ||
                        static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX2)) != T || inputN * inputHeadDim != D,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, "dy", ShapeToString(dyShape),
                        "expected [B, N, T, Dh] with D=N*Dh for x=" + ShapeToString(xShape) + " (BNSD)"),
                    return ge::GRAPH_FAILED);
        totalTokens = B * T;
    } else if (inputLayout == INPUT_LAYOUT_NTD) {
        OP_CHECK_IF(xShape.GetDimNum() != 2,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", ShapeToString(xShape),
                                                          "expected logical rank 2 [total_tokens, D] for NTD"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dyShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "dy", ShapeToString(dyShape),
                                                          "expected rank 3 [N, total_tokens, Dh] for NTD"),
                    return ge::GRAPH_FAILED);
        inputMode = 0;
        totalTokens = xShape.GetDim(DIM_INDEX0);
        D = xShape.GetDim(DIM_INDEX1);
        inputN = dyShape.GetDim(DIM_INDEX0);
        inputHeadDim = dyShape.GetDim(DIM_INDEX2);
        OP_CHECK_IF(static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != totalTokens || inputN * inputHeadDim != D,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, "dy", ShapeToString(dyShape),
                        "expected [N, total_tokens, Dh] with D=N*Dh for x=" + ShapeToString(xShape) + " (NTD)"),
                    return ge::GRAPH_FAILED);
    } else if (inputLayout == INPUT_LAYOUT_TND) {
        OP_CHECK_IF(xShape.GetDimNum() != 2 || dyShape.GetDimNum() != 2,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, "x/dy", "x=" + ShapeToString(xShape) + "; dy=" + ShapeToString(dyShape),
                        "expected rank 2 [total_tokens, D] for TND"),
                    return ge::GRAPH_FAILED);
        inputMode = 0;
        totalTokens = xShape.GetDim(DIM_INDEX0);
        D = xShape.GetDim(DIM_INDEX1);
        OP_CHECK_IF(static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != totalTokens ||
                        static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != D,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "dy", ShapeToString(dyShape),
                                                          "expected x shape=" + ShapeToString(xShape) + " for TND"),
                    return ge::GRAPH_FAILED);
        inputN = 1;
        inputHeadDim = D;
    } else if (inputLayout == INPUT_LAYOUT_BSND) {
        OP_CHECK_IF(xShape.GetDimNum() != 3 || dyShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        OP_NAME, "x/dy", "x=" + ShapeToString(xShape) + "; dy=" + ShapeToString(dyShape),
                        "expected rank 3 [B, T, D] for BSND/BSH"),
                    return ge::GRAPH_FAILED);
        inputMode = 1;
        B = xShape.GetDim(DIM_INDEX0);
        T = xShape.GetDim(DIM_INDEX1);
        D = xShape.GetDim(DIM_INDEX2);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != B ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != T ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX2)) != D,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "dy", ShapeToString(dyShape),
                                                  "expected x shape=" + ShapeToString(xShape) + " for BSND/BSH"),
            return ge::GRAPH_FAILED);
        totalTokens = B * T;
        inputN = 1;
        inputHeadDim = D;
    } else {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", ShapeToString(xShape),
                                              std::string("expected shape compatible with inputLayout=") +
                                                  inputLayoutStr);
        return ge::GRAPH_FAILED;
    }

    auto yShapePtr = context->GetOptionalInputShape(Y_INPUT_INDEX);
    if (yShapePtr != nullptr && yShapePtr->GetStorageShape().GetDimNum() > 0) {
        auto yShape = yShapePtr->GetStorageShape();
        OP_CHECK_IF(yShape.GetDimNum() != dyShape.GetDimNum(),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "y", ShapeToString(yShape),
                                                          "expected dy shape=" + ShapeToString(dyShape)),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < dyShape.GetDimNum(); i++) {
            OP_CHECK_IF(yShape.GetDim(i) != dyShape.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "y", ShapeToString(yShape),
                                                              "expected dy shape=" + ShapeToString(dyShape)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(
            activation != ACTIVATION_NONE,
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "y"),
            return ge::GRAPH_FAILED);
    }
    uint32_t W = wShape.GetDim(DIM_INDEX0);
    OP_CHECK_IF(wShape.GetDimNum() != 2 || static_cast<uint32_t>(wShape.GetDim(DIM_INDEX1)) != D,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", ShapeToString(wShape),
                                                      "expected rank 2 [W, D_total] with D_total=" + std::to_string(D)),
                return ge::GRAPH_FAILED);

    uint32_t numChunks = 0;
    if (inputMode == 0) {
        auto qslShapePtr = context->GetOptionalInputShape(QUERY_START_LOC_INPUT_INDEX);
        OP_CHECK_IF(
            qslShapePtr == nullptr,
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "queryStartLoc"),
            return ge::GRAPH_FAILED);
        auto qslShape = qslShapePtr->GetStorageShape();
        OP_CHECK_IF(qslShape.GetDimNum() != 1 || qslShape.GetDim(0) < 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "queryStartLoc", ShapeToString(qslShape),
                                                          "expected rank 1 and at least one element in varlen mode"),
                    return ge::GRAPH_FAILED);
        B = qslShape.GetDim(0) - 1;
        const gert::CompileTimeTensorDesc *qslDesc = context->GetOptionalInputDesc(QUERY_START_LOC_INPUT_INDEX);
        OP_CHECK_IF(
            qslDesc == nullptr,
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: queryStartLoc descriptor is unavailable in varlen mode", __func__),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE(OP_NAME, "queryStartLoc",
                                              std::to_string(static_cast<int>(qslDesc->GetDataType())),
                                              std::to_string(static_cast<int>(ge::DT_INT64))),
                    return ge::GRAPH_FAILED);
        const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INPUT_INDEX);
        const int64_t *qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
        OP_CHECK_IF(qslData == nullptr,
                    OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: queryStartLoc host tensor data is unavailable in varlen mode",
                                           __func__),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(qslData[0] != 0 || qslData[B] != static_cast<int64_t>(totalTokens),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        OP_NAME, "queryStartLoc",
                        "first=" + std::to_string(qslData[0]) + "; last=" + std::to_string(qslData[B]),
                        "expected first=0 and last=total_tokens=" + std::to_string(totalTokens)),
                    return ge::GRAPH_FAILED);
        uint32_t maxSeqLen = 0;
        for (uint32_t i = 0; i < B; i++) {
            OP_CHECK_IF(qslData[i + 1] < qslData[i],
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            OP_NAME, "queryStartLoc",
                            "[" + std::to_string(i) + "]=" + std::to_string(qslData[i]) + "; [" +
                                std::to_string(i + 1) + "]=" + std::to_string(qslData[i + 1]),
                            "expected non-decreasing sequence offsets"),
                        return ge::GRAPH_FAILED);
            uint32_t seqLen = static_cast<uint32_t>(qslData[i + 1] - qslData[i]);
            maxSeqLen = std::max(maxSeqLen, seqLen);
            numChunks += CeilDiv(seqLen, static_cast<uint32_t>(64));
        }
        T = maxSeqLen;
    }

    bool useInitialState = false;
    bool useFinalState = false;
    {
        auto initStateShape = context->GetInputShape(INITIAL_STATE_INPUT_INDEX);
        if (initStateShape != nullptr && initStateShape->GetStorageShape().GetDimNum() > 0) {
            auto shape = initStateShape->GetStorageShape();
            OP_CHECK_IF(shape.GetDimNum() != 3 || static_cast<uint32_t>(shape.GetDim(0)) != B ||
                            static_cast<uint32_t>(shape.GetDim(1)) != W || static_cast<uint32_t>(shape.GetDim(2)) != D,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "initial_state", ShapeToString(shape),
                                                              "expected [B, W, D]=[" + std::to_string(B) + ", " +
                                                                  std::to_string(W) + ", " + std::to_string(D) + "]"),
                        return ge::GRAPH_FAILED);
            useInitialState = true;
        }
        auto dhtShape = context->GetInputShape(DHT_INPUT_INDEX);
        if (dhtShape != nullptr && dhtShape->GetStorageShape().GetDimNum() > 0) {
            auto shape = dhtShape->GetStorageShape();
            OP_CHECK_IF(shape.GetDimNum() != 3 || static_cast<uint32_t>(shape.GetDim(0)) != B ||
                            static_cast<uint32_t>(shape.GetDim(1)) != W || static_cast<uint32_t>(shape.GetDim(2)) != D,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "dht", ShapeToString(shape),
                                                              "expected [B, W, D]=[" + std::to_string(B) + ", " +
                                                                  std::to_string(W) + ", " + std::to_string(D) + "]"),
                        return ge::GRAPH_FAILED);
            useFinalState = true;
        }
    }

    uint64_t ubSize = 0;
    uint64_t sysWorkspaceSize = 0;
    uint32_t coreNum = 0;
    bool isAscend910B = false;
    bool isAscend950 = false;
    {
        fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
        OP_CHECK_IF(
            platformInfoPtr == nullptr,
            OP_LOGE(OP_NAME, "%s",
                    std::string("platformInfo: actual=null; expected=valid platform metadata for resource queries")
                        .c_str()),
            return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        isAscend910B = (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_2201);
        isAscend950 = (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF(coreNum == 0,
                    OP_LOGE(OP_NAME, "%s",
                            std::string("coreNum: actual=0; expected > 0 AIV cores from platform metadata").c_str()),
                    return ge::GRAPH_FAILED);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == 0,
                    OP_LOGE(OP_NAME, "%s",
                            std::string("ubSize: actual=0 bytes; expected > 0 bytes from platform metadata").c_str()),
                    return ge::GRAPH_FAILED);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }

    ubSize = (ubSize - RESERVED_UB) / BYTE_BLOCK * BYTE_BLOCK;

    uint32_t alignDim = (inputLayout == INPUT_LAYOUT_BNSD || inputLayout == INPUT_LAYOUT_NTD) ? inputHeadDim : D;
    OP_CHECK_IF(alignDim % 16 != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    OP_NAME, "D/headDim", std::to_string(alignDim),
                    std::string("expected dimension divisible by 16 for inputLayout=") + inputLayoutStr),
                return ge::GRAPH_FAILED);
    uint32_t BD = (D % 64 == 0) ? 64 : 16;
    bool wideBlockLayoutSupported =
        (inputLayout != INPUT_LAYOUT_BNSD && inputLayout != INPUT_LAYOUT_NTD) ||
        (inputHeadDim >= 128 && inputHeadDim % 128 == 0);
    bool use910BWideBlock = isAscend910B && isB16Input &&
                            activation == ACTIVATION_SILU &&
                            !useInitialState && !useFinalState &&
                            W == 4 && D % 128 == 0 && wideBlockLayoutSupported;
    if (use910BWideBlock) {
        BD = 128;
    }
    uint32_t numBlksD = D / BD;

    // i_d 切分（useIdSplit）：每个核独占若干 i_d 的全部 chunk，核间无 dw/db 重叠。
    bool useIdSplit = isAscend950 && numBlksD >= coreNum;

    // A5(arch35) regbase 快路径：b16、BD∈{64,16}、W∈[1,8]、无 initial_state/dht。
    bool arch35WOk = (W >= 1 && W <= ARCH35_DIRECT_W_MAX);
    if (activation != ACTIVATION_NONE) {
        arch35WOk = arch35WOk && (W == 4 || numBlksD <= 4 || useIdSplit);
    }
    bool useArch35DirectFast = isAscend950 &&
                               ((isB16Input && (BD == 64 ||
                                                (BD == 16 && (useIdSplit || (numBlksD == 1 && T <= 64))))) ||
                                (isFp32Input && BD == 16 && useIdSplit)) &&
                               (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH ||
                                activation == ACTIVATION_NONE) &&
                               !useInitialState && !useFinalState &&
                               arch35WOk;
    bool use910BDirectFast = isAscend910B && isB16Input &&
                             activation == ACTIVATION_SILU &&
                             !useInitialState && !useFinalState &&
                             W == 4 && (BD == 64 || BD == 128);
    uint32_t rawDtypeSize = isB16Input ? B16_DTYPE_SIZE : FP32_DTYPE_SIZE;
    auto CalcArch35DirectNeed = [BD, W, rawDtypeSize, isB16Input](uint32_t bt) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        uint32_t regPad = (BD < 64U) ? 64U : 0U;
        uint32_t dbAl = (BD < 64U) ? 64U : bdAl;
        uint32_t need = 0;
        need += (2 * btBdAl + 4 * dyBdAl + wBdAl + regPad) * rawDtypeSize;
        if (isB16Input) {
            need += (std::max(btBdAl, wBdAl) + btBdAl + bdAl) * rawDtypeSize;
        }
        need += (btBdAl + dyBdAl + wBdAl + 2 * regPad + dbAl) * FP32_DTYPE_SIZE;
        return need;
    };
    auto Calc910BDirectNeed = [BD, W](uint32_t bt) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        uint32_t xRawAl = std::max(btBdAl, wBdAl);
        uint32_t need = (btBdAl + 3 * dyBdAl + 2 * wBdAl + 2 * bdAl) * FP32_DTYPE_SIZE;
        need += (xRawAl + 2 * dyBdAl + 2 * btBdAl) * B16_DTYPE_SIZE;
        return need;
    };
    // 普通路径（chunk 切分 / i_d 切分）的 UB 预算，不含两级归约 stage buffer（该部分在预算内按需追加）。
    auto CalcNormalNeed = [BD, W, rawDtypeSize, isB16Input, isAscend950, activation,
                           useInitialState, useFinalState](uint32_t bt) -> uint32_t {
        uint32_t btBdAl = CeilAlign(bt * BD, BLOCK_ALIGN_NUM);
        uint32_t dyBdAl = CeilAlign((bt + W - 1) * BD, BLOCK_ALIGN_NUM);
        uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
        uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
        uint32_t calcBdAl = std::max(std::max(btBdAl, wBdAl), dyBdAl);
        uint32_t wBtBdAl = CeilAlign(W * bt * BD, BLOCK_ALIGN_NUM);
        const bool hasWeight = true;
        const bool hasBias = true;
        uint32_t need = (3 * btBdAl + 2 * calcBdAl) * FP32_DTYPE_SIZE;
        need += (dyBdAl - btBdAl) * FP32_DTYPE_SIZE;
        if (hasWeight) need += 2 * wBdAl * FP32_DTYPE_SIZE;
        if (hasWeight && activation == ACTIVATION_NONE) need += wBtBdAl * FP32_DTYPE_SIZE;
        if (hasBias)   need += bdAl * FP32_DTYPE_SIZE;
        if (hasBias && activation == ACTIVATION_NONE) need += btBdAl * FP32_DTYPE_SIZE;
        if (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH)
            need += 2 * dyBdAl * FP32_DTYPE_SIZE;
        if (useInitialState || useFinalState) {
            need += bdAl * FP32_DTYPE_SIZE;
        }
        if (isB16Input) {
            need += calcBdAl * rawDtypeSize;
            if (isAscend950) {
                need += (btBdAl + bdAl) * rawDtypeSize;
            }
        }
        return need;
    };

    uint32_t BT = 64;
    if (useArch35DirectFast) {
        BT = std::min(T, DIRECT_FAST_BT_MAX);
        while (BT > DIRECT_FAST_BT_MIN && CalcArch35DirectNeed(BT) > ubSize) {
            BT = (BT > DIRECT_FAST_BT_MIN + DIRECT_FAST_BT_STEP) ? (BT - DIRECT_FAST_BT_STEP) : DIRECT_FAST_BT_MIN;
        }
    } else if (use910BDirectFast) {
        uint32_t maxBt = (BD == 128) ? DIRECT_910B_BT_BD128 : DIRECT_910B_BT_BD64;
        BT = std::min(T, maxBt);
        while (BT > 1 && Calc910BDirectNeed(BT) > ubSize) {
            BT--;
        }
    } else if (T < BT) {
        BT = T;
    }
    OP_CHECK_IF(
        BT == 0 || totalTokens == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "sequence length",
                                              "T=" + std::to_string(T) + "; BT=" + std::to_string(BT) +
                                                  "; totalTokens=" + std::to_string(totalTokens),
                                              "expected BT > 0 and totalTokens > 0; empty sequence is not supported"),
        return ge::GRAPH_FAILED);
    auto CalcNumChunks = [&](uint32_t bt) -> uint32_t {
        if (bt == 0) {
            return 0;
        }
        if (inputMode == 0) {
            const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INPUT_INDEX);
            const int64_t *qslData = qslTensor->GetData<int64_t>();
            uint32_t n = 0;
            for (uint32_t i = 0; i < B; i++) {
                n += CeilDiv(static_cast<uint32_t>(qslData[i + 1] - qslData[i]), bt);
            }
            return n;
        }
        return CeilDiv(T, bt) * B;
    };
    numChunks = CalcNumChunks(BT);

    // 小 shape 核数填充：!useIdSplit 时核间并行度只来自 chunk 数，D 方向再多 i_d 也只能每核串行遍历。
    constexpr uint32_t BT_FILL_STEP = 32U;
    constexpr uint32_t BT_FILL_FLOOR = 16U;
    constexpr uint32_t BT_FILL_CAP = 8U;
    constexpr uint32_t BT_FILL_ELEMS_PER_CORE = 8192U;
    constexpr uint32_t BT_FILL_MIN_TARGET = 4U;
    if (isAscend950 && !useIdSplit && numBlksD >= 2) {
        uint32_t target = std::min(coreNum, BT_FILL_CAP);
        if (useArch35DirectFast) {
            uint64_t totalElems = static_cast<uint64_t>(totalTokens) * D;
            uint64_t byWork = (totalElems >= BT_FILL_ELEMS_PER_CORE)
                                  ? (totalElems / BT_FILL_ELEMS_PER_CORE) : 1ULL;
            target = static_cast<uint32_t>(
                std::min<uint64_t>(target, std::max<uint64_t>(BT_FILL_MIN_TARGET, byWork)));
        }
        while (BT > BT_FILL_FLOOR && numChunks < target) {
            BT = (BT > BT_FILL_FLOOR + BT_FILL_STEP) ? (BT - BT_FILL_STEP) : BT_FILL_FLOOR;
            numChunks = CalcNumChunks(BT);
        }
    }

    // BT=64→32 的两档选择（仅 ascend950 正常路径，非 idSplit / 非快路径）：
    if (isAscend950 && !useIdSplit && !useArch35DirectFast && !use910BDirectFast && numBlksD >= 2 &&
        BT == 64U && numChunks < coreNum) {
        uint32_t numChunksHalf = CalcNumChunks(BT >> 1U);
        if (numChunksHalf == coreNum) {
            BT >>= 1U;
            numChunks = numChunksHalf;
        }
    }

    if (!useArch35DirectFast && !use910BDirectFast) {
        while (BT > 1U && CalcNormalNeed(BT) > ubSize) {
            BT >>= 1U;
        }
        numChunks = CalcNumChunks(BT);
    }

    // chunk × i_d 二维切分：把 numBlksD 个 i_d 与 numChunks 个 chunk 一起铺到 AIV 核上
    const bool twoDByUnit = static_cast<uint64_t>(numBlksD) * numChunks < coreNum;
    uint32_t splitG = 0;
    bool use2DSplit = false;
    if (isAscend950 && !useIdSplit && !useArch35DirectFast && numBlksD >= 2 && numChunks >= 1) {
        if (twoDByUnit) {
            splitG = numChunks; // 每核恰好 1 个 (i_d, chunk) 单元，沿用历史行为
            use2DSplit = true;
        } else if (numBlksD >= 4 && numChunks < coreNum) {
            splitG = std::min(numChunks, coreNum / numBlksD);
            use2DSplit = (splitG >= 2); // splitG<2 时 2D 无额外并行度，退回 1D
        }
    }

    uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
    uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
    uint32_t need = 0;
    uint32_t dwStageRows = 0;
    if (useArch35DirectFast) {
        need = CalcArch35DirectNeed(BT);
    } else if (use910BDirectFast) {
        need = Calc910BDirectNeed(BT);
    } else {
        need = CalcNormalNeed(BT);

        if (useIdSplit && (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH)) {
            uint32_t rMax = 1;
            while ((rMax << 1) <= BT) rMax <<= 1;
            for (uint32_t r = rMax; r >= 2; r >>= 1) {
                uint32_t dwStageAl = CeilAlign(W * r * BD, BLOCK_ALIGN_NUM);
                uint32_t dbStageAl = CeilAlign(r * BD, BLOCK_ALIGN_NUM);
                uint32_t stageBytes = (dwStageAl + dbStageAl) * FP32_DTYPE_SIZE;
                if (need + stageBytes <= ubSize) {
                    dwStageRows = r;
                    need += stageBytes;
                    break;
                }
            }
        }
    }

    OP_CHECK_IF(need > ubSize,
                OP_LOGE(OP_NAME, "%s",
                        std::string("UB capacity: requiredBytes=" + std::to_string(need) +
                                    "; availableBytes=" + std::to_string(ubSize) +
                                    "; expected requiredBytes <= availableBytes for selected tile")
                            .c_str()),
                return ge::GRAPH_FAILED);

    // The kernel entry is not template-specialized by activation. Keep one
    // function entry and let tiling->activation select the runtime path.
    uint64_t tilingKey = CAUSAL_CONV1D_BWD_TPL_ACTIVATION_NONE;

    // Compute block dimension
    //
    // A5(arch35) 上有两个核间切分方案：
    //  - i_d 切分（useIdSplit）：每个核独占若干 i_d 的全部 chunk。
    //  - chunk 切分（原方案）：每个核负责部分 chunk、遍历全部 i_d，需要 partial 归约。
    uint32_t chunkPerCore = 0;
    uint32_t tailChunk = 0;
    uint32_t totalChunks = numChunks;
    uint32_t mainCoreNum = useIdSplit ? coreNum
                          : (use2DSplit ? (numBlksD * splitG)
                                        : std::min(coreNum, totalChunks));
    if (useIdSplit || use2DSplit) {
        chunkPerCore = 0;
        tailChunk = 0;
    } else if (totalChunks <= mainCoreNum) {
        chunkPerCore = 1;
        tailChunk = 0;
    } else {
        chunkPerCore = totalChunks / mainCoreNum;
        tailChunk = totalChunks % mainCoreNum;
    }

    constexpr uint32_t RED_STAGE_ELEMS = 2048U; // 2048 个 fp32 = 8KB，与 kernel 侧同一上限
    uint32_t redStageRows = 0;
    if (isAscend950 && mainCoreNum > 1U && !useIdSplit && !use2DSplit) {
        uint32_t rowsCap = RED_STAGE_ELEMS / (W * BD);
        if (rowsCap < 1U) {
            rowsCap = 1U;
        }
        redStageRows = (mainCoreNum < rowsCap) ? mainCoreNum : rowsCap;
        if (redStageRows < 2U) {
            redStageRows = 0U;
        }
    }
    if (redStageRows >= 2U) {
        uint64_t redBytes =
            static_cast<uint64_t>(CeilAlign(redStageRows * W * BD, BLOCK_ALIGN_NUM)) * FP32_DTYPE_SIZE;
        OP_CHECK_IF(redBytes > RESERVED_UB,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "ubSize", std::to_string(redBytes).c_str(),
                        "the partial-reduce staging buffer exceeds the reserved UB region"),
                    return ge::GRAPH_FAILED);
    }

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(mainCoreNum);
    const auto scheduleRc = context->SetScheduleMode(1);
    OP_CHECK_IF(scheduleRc != ge::GRAPH_SUCCESS,
                OP_LOGE(OP_NAME, "%s",
                        std::string("SetScheduleMode: mode=1; rc=" + std::to_string(scheduleRc) +
                                    "; expected rc=0 (GRAPH_SUCCESS)")
                            .c_str()),
                return ge::GRAPH_FAILED);

    tiling->B = B;
    tiling->T = T;
    tiling->D = D;
    tiling->W = W;
    tiling->activation = activation;
    tiling->hasWeight = 1;
    tiling->hasBias = 1;
    tiling->useInitialState = useInitialState ? 1 : 0;
    tiling->useFinalState = useFinalState ? 1 : 0;
    tiling->inputMode = inputMode;
    tiling->inputLayout = inputLayout;
    tiling->inputN = inputN;
    tiling->inputHeadDim = inputHeadDim;
    tiling->totalTokens = totalTokens;
    tiling->blockNum = mainCoreNum;
    tiling->BT = BT;
    tiling->BD = BD;
    tiling->numBlksD = numBlksD;
    tiling->numChunks = numChunks;
    tiling->batchPerCore = 1;
    tiling->tailBatch = 0;
    tiling->chunkPerCore = chunkPerCore;
    tiling->tailChunk = tailChunk;
    tiling->useIdSplit = useIdSplit ? 1 : 0;
    tiling->use2DSplit = use2DSplit ? 1 : 0;
    tiling->dwStageRows = dwStageRows;
    tiling->redStageRows = redStageRows;

    // useIdSplit 时每个核独占 i_d 并直接写回 dw/db，不再需要 partial 归约缓冲。
    uint64_t partialDwBytes = useIdSplit ? 0ULL
        : static_cast<uint64_t>(mainCoreNum) * numBlksD * wBdAl * FP32_DTYPE_SIZE;
    uint64_t partialDbBytes = useIdSplit ? 0ULL
        : static_cast<uint64_t>(mainCoreNum) * numBlksD * bdAl * FP32_DTYPE_SIZE;
    uint64_t wsSize = sysWorkspaceSize + partialDwBytes + partialDbBytes;
    tiling->workspaceSize = sysWorkspaceSize;

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = wsSize;

    OP_LOGD(context,
            "Tiling result: B[%ld] T[%ld] totalTokens[%ld] D[%ld] W[%ld] inputMode[%ld] activation[%ld] "
            "inputLayout[%ld] inputN[%ld] inputHeadDim[%ld] "
            "BT[%ld] BD[%ld] numBlksD[%ld] numChunks[%ld] "
            "mainCore[%d] chunkPerCore[%ld] tail[%ld] sysWs[%ld] wsSize[%ld] idSplit[%d] split2D[%d] dwStageRows[%u] redStageRows[%u]",
            B, T, totalTokens, D, W, inputMode, activation,
            inputLayout, inputN, inputHeadDim,
            BT, BD, numBlksD, numChunks,
            mainCoreNum, chunkPerCore, tailChunk, sysWorkspaceSize, wsSize,
            useIdSplit ? 1 : 0, use2DSplit ? 1 : 0, dwStageRows, redStageRows);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCausalConv1dBwd(gert::TilingParseContext *context)
{
    OP_LOGD(context, "Enter TilingParseForCausalConv1dBwd.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1dBwd)
    .Tiling(CausalConv1dBwdTilingFunc)
    .TilingParse<CausalConv1dBwdCompileInfo>(TilingParseForCausalConv1dBwd);

} // namespace optiling
