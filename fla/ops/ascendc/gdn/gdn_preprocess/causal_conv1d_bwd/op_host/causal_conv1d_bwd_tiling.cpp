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

#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "causal_conv1d_bwd_tiling.h"
#include <algorithm>
#include <cstring>

namespace {
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
    constexpr uint32_t FP32_DTYPE_SIZE = 4U;
    constexpr uint32_t BLOCK_ALIGN_NUM = 8U;
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

static ge::graphStatus CausalConv1dBwdTilingFunc(gert::TilingContext *context)
{
    CausalConv1dBwdTilingData *tiling = context->GetTilingData<CausalConv1dBwdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CausalConv1dBwdTilingData), 0, sizeof(CausalConv1dBwdTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    auto xShape = context->GetInputShape(X_INPUT_INDEX)->GetStorageShape();
    auto dyShape = context->GetInputShape(DY_INPUT_INDEX)->GetStorageShape();
    auto wShape = context->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
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
        activation = *(attrs->GetAttrPointer<uint32_t>(ATTR_ACTIVATION_INDEX));
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
        OP_LOGE(context, "unsupported inputLayout %s, expect BSND/TND/BNSD/NTD", inputLayoutStr);
        return ge::GRAPH_FAILED;
    }

    if (inputLayout == INPUT_LAYOUT_BNSD) {
        OP_CHECK_IF(xShape.GetDimNum() != 3,
                    OP_LOGE(context, "x must be logical 3D [B, T, D] when inputLayout is BNSD"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dyShape.GetDimNum() != 4,
                    OP_LOGE(context, "dy must be 4D [B, N, T, Dh] when inputLayout is BNSD"),
                    return ge::GRAPH_FAILED);
        inputMode = 1;
        B = xShape.GetDim(DIM_INDEX0);
        T = xShape.GetDim(DIM_INDEX1);
        D = xShape.GetDim(DIM_INDEX2);
        inputN = dyShape.GetDim(DIM_INDEX1);
        inputHeadDim = dyShape.GetDim(DIM_INDEX3);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != B ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX2)) != T ||
                inputN * inputHeadDim != D,
            OP_LOGE(context, "dy must be [B, N, T, Dh] with D=N*Dh for logical x [B, T, D]"),
            return ge::GRAPH_FAILED);
        totalTokens = B * T;
    } else if (inputLayout == INPUT_LAYOUT_NTD) {
        OP_CHECK_IF(xShape.GetDimNum() != 2,
                    OP_LOGE(context, "x must be logical 2D [total_tokens, D] when inputLayout is NTD"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(dyShape.GetDimNum() != 3,
                    OP_LOGE(context, "dy must be 3D [N, total_tokens, Dh] when inputLayout is NTD"),
                    return ge::GRAPH_FAILED);
        inputMode = 0;
        totalTokens = xShape.GetDim(DIM_INDEX0);
        D = xShape.GetDim(DIM_INDEX1);
        inputN = dyShape.GetDim(DIM_INDEX0);
        inputHeadDim = dyShape.GetDim(DIM_INDEX2);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != totalTokens ||
                inputN * inputHeadDim != D,
            OP_LOGE(context, "dy must be [N, total_tokens, Dh] with D=N*Dh for logical x [total_tokens, D]"),
            return ge::GRAPH_FAILED);
    } else if (inputLayout == INPUT_LAYOUT_TND) {
        OP_CHECK_IF(xShape.GetDimNum() != 2 || dyShape.GetDimNum() != 2,
                    OP_LOGE(context, "x and dy must be [total_tokens, D] when inputLayout is TND"),
                    return ge::GRAPH_FAILED);
        inputMode = 0;
        totalTokens = xShape.GetDim(DIM_INDEX0);
        D = xShape.GetDim(DIM_INDEX1);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != totalTokens ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != D,
            OP_LOGE(context, "dy shape must match x shape when inputLayout is TND"),
            return ge::GRAPH_FAILED);
        inputN = 1;
        inputHeadDim = D;
    } else if (inputLayout == INPUT_LAYOUT_BSND) {
        OP_CHECK_IF(xShape.GetDimNum() != 3 || dyShape.GetDimNum() != 3,
                    OP_LOGE(context, "x and dy must be [B, T, D] when inputLayout is BSND/BSH"),
                    return ge::GRAPH_FAILED);
        inputMode = 1;
        B = xShape.GetDim(DIM_INDEX0);
        T = xShape.GetDim(DIM_INDEX1);
        D = xShape.GetDim(DIM_INDEX2);
        OP_CHECK_IF(
            static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX0)) != B ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX1)) != T ||
                static_cast<uint32_t>(dyShape.GetDim(DIM_INDEX2)) != D,
            OP_LOGE(context, "dy shape must match x shape when inputLayout is BSND/BSH"),
            return ge::GRAPH_FAILED);
        totalTokens = B * T;
        inputN = 1;
        inputHeadDim = D;
    } else {
        OP_LOGE(context, "x shape does not match inputLayout %s", inputLayoutStr);
        return ge::GRAPH_FAILED;
    }

    auto yShapePtr = context->GetOptionalInputShape(Y_INPUT_INDEX);
    if (yShapePtr != nullptr && yShapePtr->GetStorageShape().GetDimNum() > 0) {
        auto yShape = yShapePtr->GetStorageShape();
        OP_CHECK_IF(yShape.GetDimNum() != dyShape.GetDimNum(),
                    OP_LOGE(context, "y and dy must have the same shape"),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < dyShape.GetDimNum(); i++) {
            OP_CHECK_IF(yShape.GetDim(i) != dyShape.GetDim(i),
                        OP_LOGE(context, "y and dy must have the same shape"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(activation != ACTIVATION_NONE,
                    OP_LOGE(context, "y is required when activation is enabled"),
                    return ge::GRAPH_FAILED);
    }
    uint32_t W = wShape.GetDim(DIM_INDEX0);
    OP_CHECK_IF(wShape.GetDimNum() != 2 || static_cast<uint32_t>(wShape.GetDim(DIM_INDEX1)) != D,
                OP_LOGE(context, "weight must be [W, D_total], got dim1 %ld expect %u",
                        wShape.GetDimNum() > DIM_INDEX1 ? wShape.GetDim(DIM_INDEX1) : 0, D),
                return ge::GRAPH_FAILED);

    uint32_t numChunks = 0;
    if (inputMode == 0) {
        auto qslShapePtr = context->GetOptionalInputShape(QUERY_START_LOC_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, qslShapePtr);
        auto qslShape = qslShapePtr->GetStorageShape();
        OP_CHECK_IF(qslShape.GetDimNum() != 1 || qslShape.GetDim(0) < 1,
                    OP_LOGE(context, "queryStartLoc must be 1D and non-empty in varlen mode"),
                    return ge::GRAPH_FAILED);
        B = qslShape.GetDim(0) - 1;
        const gert::CompileTimeTensorDesc *qslDesc = context->GetOptionalInputDesc(QUERY_START_LOC_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, qslDesc);
        OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64,
                    OP_LOGE(context, "queryStartLoc dtype must be int64"),
                    return ge::GRAPH_FAILED);
        const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INPUT_INDEX);
        const int64_t *qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
        OP_CHECK_NULL_WITH_CONTEXT(context, qslData);
        OP_CHECK_IF(qslData[0] != 0 || qslData[B] != static_cast<int64_t>(totalTokens),
                    OP_LOGE(context, "queryStartLoc must start with 0 and end with total_tokens"),
                    return ge::GRAPH_FAILED);
        uint32_t maxSeqLen = 0;
        for (uint32_t i = 0; i < B; i++) {
            OP_CHECK_IF(qslData[i + 1] < qslData[i],
                        OP_LOGE(context, "queryStartLoc must be non-decreasing"),
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
            OP_CHECK_IF(
                shape.GetDimNum() != 3 ||
                    static_cast<uint32_t>(shape.GetDim(0)) != B ||
                    static_cast<uint32_t>(shape.GetDim(1)) != W ||
                    static_cast<uint32_t>(shape.GetDim(2)) != D,
                OP_LOGE(context, "initial_state must be [B, W, D]=[%u, %u, %u]", B, W, D),
                return ge::GRAPH_FAILED);
            useInitialState = true;
        }
        auto dhtShape = context->GetInputShape(DHT_INPUT_INDEX);
        if (dhtShape != nullptr && dhtShape->GetStorageShape().GetDimNum() > 0) {
            auto shape = dhtShape->GetStorageShape();
            OP_CHECK_IF(
                shape.GetDimNum() != 3 ||
                    static_cast<uint32_t>(shape.GetDim(0)) != B ||
                    static_cast<uint32_t>(shape.GetDim(1)) != W ||
                    static_cast<uint32_t>(shape.GetDim(2)) != D,
                OP_LOGE(context, "dht must be [B, W, D]=[%u, %u, %u]", B, W, D),
                return ge::GRAPH_FAILED);
            useFinalState = true;
        }
    }

    uint64_t ubSize = 0;
    uint64_t sysWorkspaceSize = 0;
    uint32_t coreNum = 0;
    {
        fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }

    ubSize = (ubSize - RESERVED_UB) / BYTE_BLOCK * BYTE_BLOCK;

    uint32_t alignDim = (inputLayout == INPUT_LAYOUT_BNSD || inputLayout == INPUT_LAYOUT_NTD) ? inputHeadDim : D;
    OP_CHECK_IF(alignDim % 16 != 0,
                OP_LOGE(context, "%s D %u must be divisible by 16", inputLayoutStr, alignDim),
                return ge::GRAPH_FAILED);
    uint32_t BD = (D % 64 == 0) ? 64 : 16;
    uint32_t numBlksD = D / BD;

    uint32_t BT = 64;
    if (T < BT) {
        BT = T;
    }
    OP_CHECK_IF(BT == 0 || totalTokens == 0,
                OP_LOGE(context, "empty sequence is not supported: T %u totalTokens %u", T, totalTokens),
                return ge::GRAPH_FAILED);
    if (inputMode == 0) {
        numChunks = 0;
        const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INPUT_INDEX);
        const int64_t *qslData = qslTensor->GetData<int64_t>();
        for (uint32_t i = 0; i < B; i++) {
            numChunks += CeilDiv(static_cast<uint32_t>(qslData[i + 1] - qslData[i]), BT);
        }
    } else {
        numChunks = CeilDiv(T, BT) * B;
    }

    uint32_t btBdAl = CeilAlign(BT * BD, BLOCK_ALIGN_NUM);
    uint32_t dyBdAl = CeilAlign((BT + W - 1) * BD, BLOCK_ALIGN_NUM);
    uint32_t wBdAl = CeilAlign(W * BD, BLOCK_ALIGN_NUM);
    uint32_t calcBdAl = std::max(btBdAl, wBdAl);
    uint32_t wBtBdAl = CeilAlign(W * BT * BD, BLOCK_ALIGN_NUM);
    uint32_t bdAl = CeilAlign(BD, BLOCK_ALIGN_NUM);
    uint32_t need = 0;
    need += (3 * btBdAl + 2 * calcBdAl) * FP32_DTYPE_SIZE;
    need += (dyBdAl - btBdAl) * FP32_DTYPE_SIZE;
    const bool hasWeight = true;
    const bool hasBias = true;
    if (hasWeight) need += 2 * wBdAl * FP32_DTYPE_SIZE;
    if (hasWeight && activation == ACTIVATION_NONE) need += wBtBdAl * FP32_DTYPE_SIZE;
    if (hasBias)   need += bdAl * FP32_DTYPE_SIZE;
    if (hasBias && activation == ACTIVATION_NONE) need += btBdAl * FP32_DTYPE_SIZE;
    if (activation == ACTIVATION_SILU || activation == ACTIVATION_SWISH)
        need += 2 * btBdAl * FP32_DTYPE_SIZE;
    if (useInitialState || useFinalState) {
        need += bdAl * FP32_DTYPE_SIZE;
    }

    OP_CHECK_IF(need > ubSize,
                OP_LOGE(context, "UB overflow: need %u > available %lu", need, ubSize),
                return ge::GRAPH_FAILED);

    // The kernel entry is not template-specialized by activation. Keep one
    // function entry and let tiling->activation select the runtime path.
    uint64_t tilingKey = CAUSAL_CONV1D_BWD_TPL_ACTIVATION_NONE;

    // Compute block dimension
    uint32_t chunkPerCore = 0;
    uint32_t tailChunk = 0;
    uint32_t totalChunks = numChunks;
    uint32_t mainCoreNum = std::min(coreNum, totalChunks);
    if (totalChunks <= mainCoreNum) {
        chunkPerCore = 1;
        tailChunk = 0;
    } else {
        chunkPerCore = totalChunks / mainCoreNum;
        tailChunk = totalChunks % mainCoreNum;
    }

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(mainCoreNum);
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetScheduleMode(1) error"), return ge::GRAPH_FAILED);

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

    uint64_t partialDwBytes = static_cast<uint64_t>(mainCoreNum) * numBlksD * wBdAl * FP32_DTYPE_SIZE;
    uint64_t partialDbBytes = static_cast<uint64_t>(mainCoreNum) * numBlksD * bdAl * FP32_DTYPE_SIZE;
    uint64_t wsSize = sysWorkspaceSize + partialDwBytes + partialDbBytes;
    tiling->workspaceSize = sysWorkspaceSize;

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = wsSize;

    OP_LOGD(context,
            "Tiling result: B[%ld] T[%ld] totalTokens[%ld] D[%ld] W[%ld] inputMode[%ld] activation[%ld] "
            "inputLayout[%ld] inputN[%ld] inputHeadDim[%ld] "
            "BT[%ld] BD[%ld] numBlksD[%ld] numChunks[%ld] "
            "mainCore[%d] chunkPerCore[%ld] tail[%ld] sysWs[%ld] wsSize[%ld]",
            B, T, totalTokens, D, W, inputMode, activation,
            inputLayout, inputN, inputHeadDim,
            BT, BD, numBlksD, numChunks,
            mainCoreNum, chunkPerCore, tailChunk, sysWorkspaceSize, wsSize);

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
