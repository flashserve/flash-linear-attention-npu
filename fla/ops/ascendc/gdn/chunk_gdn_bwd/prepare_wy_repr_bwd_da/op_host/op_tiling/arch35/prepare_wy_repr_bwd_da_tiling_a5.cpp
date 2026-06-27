/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_da_tiling_a5.cpp
 * \brief
 */

#include "prepare_wy_repr_bwd_da_tiling_a5.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

bool PrepareWyReprBwdDaTilingA5::SetTiling(gert::TilingContext *context)
{
    context_ = context;
    CommonTiling();
    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    if (cuSeqlensTensor == nullptr) {
        OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return false);
        tiling_.isVariable = 0;
    } else {
        OP_CHECK_IF(tiling_.B != VAR_LEN_B_DIM_1,
                    OP_LOGE(context->GetNodeName(),
                            "If cu_seqlens is not nullptr, the dim 0 of q needs to be 1, but now is %ld.",
                            tiling_.B),
                    return false);
        auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesTensor);
        OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return false);
        tiling_.isVariable = 1;
    }
    context->SetTilingKey(1);
    OP_LOGD(context->GetNodeName(), "tilingKey: %d", context->GetTilingKey());
    PrepareWyReprBwdDaTilingDataPrint();

    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(context_->GetNodeName(), "RawTilingData is nullptr."),
                return false);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tiling_), sizeof(tiling_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return false;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(tiling_));

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());

    auto sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "=== sysWorkspaceSize: %ld", sysWorkspaceSize);
    int64_t maxKV = std::max(tiling_.K, tiling_.V);
    size_t userWorkspaceSize = 2 * tiling_.B * tiling_.HV * tiling_.T * (tiling_.chunkSize + maxKV);
    OP_LOGD(context->GetNodeName(), "=== userWorkspaceSize: %ld", userWorkspaceSize);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize + userWorkspaceSize);
    OP_LOGD(context->GetNodeName(), "=== currentWorkspace[0]: %ld", currentWorkspace[0]);
    context->SetScheduleMode(1); // set as batchmod for template using SyncAll
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwdDa end.");
    return true;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::RequiredInputDimNumCheck(const gert::StorageShape *curShape,
                                                                     size_t validDimNum, const char *inputName)
{
    OP_CHECK_IF(curShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "Input %s is required, but got nullptr.", inputName),
                return ge::GRAPH_FAILED);
    const gert::Shape storageShape = curShape->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    OP_CHECK_IF(dimNum != validDimNum,
                OP_LOGE(context_->GetNodeName(),
                        "Check input %s shape failed, the dim num should be %zu, but get %zu.", inputName, validDimNum,
                        dimNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::CompareShape(const gert::Shape &shape1, const gert::Shape &shape2,
                                                         const char *inputName1, const char *inputName2,
                                                         size_t compareDimNum)
{
    size_t shapeDim1 = 0;
    size_t shapeDim2 = 0;
    for (size_t dimIndex = 0; dimIndex < compareDimNum; dimIndex++) {
        shapeDim1 = shape1.GetDim(dimIndex);
        shapeDim2 = shape2.GetDim(dimIndex);
        OP_CHECK_IF(shapeDim1 != shapeDim2,
                    OP_LOGE(context_->GetNodeName(),
                            "Compare input shape of %s and %s failed, the length of dim %zu should be same,but got "
                            "%zu and %zu.",
                            inputName1, inputName2, dimIndex, shapeDim1, shapeDim2),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::SetRowNumKBetaGRegbase(uint64_t ubSize, ge::DataType kType,
                                                                   ge::DataType betaType)
{
    uint64_t rowNum = chunkSize / 2;
    uint64_t sizeofKType = SIZE_FP32;
    uint64_t sizeofBetaType = SIZE_FP32;
    if (kType != ge::DataType::DT_FLOAT) {
        sizeofKType = SIZE_HALF;
    }
    if (betaType != ge::DataType::DT_FLOAT) {
        sizeofBetaType = SIZE_HALF;
    }
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * K * sizeofBetaType;
        useUbSize += 2 * rowNum * K * sizeofKType;

        if (useUbSize <= ubSize) {
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.rowNumKBetaG = rowNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::SetRowNumVBetaRegbase(uint64_t ubSize, ge::DataType kType,
                                                                  ge::DataType betaType)
{
    uint64_t rowNum = chunkSize / 2;
    uint64_t sizeofKType = SIZE_FP32;
    uint64_t sizeofBetaType = SIZE_FP32;
    if (kType != ge::DataType::DT_FLOAT) {
        sizeofKType = SIZE_HALF;
    }
    if (betaType != ge::DataType::DT_FLOAT) {
        sizeofBetaType = SIZE_HALF;
    }
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * rowNum * V * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * V * sizeofKType;

        if (useUbSize <= ubSize) {
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.rowNumVBeta = rowNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::SetRowNumMDuDwRegbase(uint64_t ubSize, ge::DataType kType,
                                                                  ge::DataType betaType)
{
    uint64_t rowNum = chunkSize / 2;
    uint64_t sizeofKType = SIZE_FP32;
    uint64_t sizeofBetaType = SIZE_FP32;
    if (kType != ge::DataType::DT_FLOAT) {
        sizeofKType = SIZE_HALF;
    }
    if (betaType != ge::DataType::DT_FLOAT) {
        sizeofBetaType = SIZE_HALF;
    }
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * rowNum * chunkSize * sizeofKType;
        useUbSize += 2 * rowNum * chunkSize * sizeofKType;
        useUbSize += 2 * rowNum * chunkSize * sizeofKType;

        if (useUbSize <= ubSize) {
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.rowNumMDuDw = rowNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::SetRowNumGRegbase(uint64_t ubSize, ge::DataType kType,
                                                              ge::DataType betaType)
{
    uint64_t rowNum = chunkSize / 2;
    uint64_t sizeofKType = SIZE_FP32;
    uint64_t sizeofBetaType = SIZE_FP32;
    if (kType != ge::DataType::DT_FLOAT) {
        sizeofKType = SIZE_HALF;
    }
    if (betaType != ge::DataType::DT_FLOAT) {
        sizeofBetaType = SIZE_HALF;
    }
    uint64_t gCVNum = 2;
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * chunkSize * sizeofBetaType;
        useUbSize += 2 * rowNum * chunkSize * sizeofKType;
        useUbSize += 2 * rowNum * chunkSize * sizeofKType;

        if (useUbSize <= ubSize) {
            gCVNum += (ubSize - useUbSize) / (rowNum * chunkSize * sizeofKType);
            gCVNum = std::min(gCVNum, MAX_CUBE_VEC_SYNC_NUM);
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.rowNumG = rowNum;
    tiling_.gCVNum = gCVNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::CommonTiling()
{
    const gert::Shape kStorageShape = context_->GetRequiredInputShape(INPUT_K_IDX)->GetStorageShape();
    const gert::Shape vStorageShape = context_->GetRequiredInputShape(INPUT_V_IDX)->GetStorageShape();
    const gert::Shape betaStorageShape = context_->GetRequiredInputShape(INPUT_BETA_IDX)->GetStorageShape();
    const gert::Shape AStorageShape = context_->GetRequiredInputShape(INPUT_A_IDX)->GetStorageShape();
    const gert::Shape dwStorageShape = context_->GetRequiredInputShape(INPUT_DW_IDX)->GetStorageShape();
    const gert::Shape duStorageShape = context_->GetRequiredInputShape(INPUT_DU_IDX)->GetStorageShape();
    const gert::Shape gStorageShape = context_->GetRequiredInputShape(INPUT_G_IDX)->GetStorageShape();
    OP_CHECK_IF(CompareShape(vStorageShape, duStorageShape, INPUT_V_NAME, INPUT_DU_NAME, DIM_NUM_4) !=
                    ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);
    OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, INPUT_BETA_NAME, INPUT_G_NAME, DIM_NUM_3) !=
                    ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);

    // GVA: K uses HK heads while the other tensors use HV heads. Batch/time dimensions must still match.
    OP_CHECK_IF(vStorageShape.GetDim(DIM_0) != kStorageShape.GetDim(DIM_0) ||
                    vStorageShape.GetDim(DIM_2) != kStorageShape.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: batch/time dims must match.", INPUT_V_NAME,
                        INPUT_K_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != gStorageShape.GetDim(DIM_0) ||
                    kStorageShape.GetDim(DIM_2) != gStorageShape.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: batch/time dims must match.", INPUT_K_NAME,
                        INPUT_G_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dwStorageShape.GetDim(DIM_0) != kStorageShape.GetDim(DIM_0) ||
                    dwStorageShape.GetDim(DIM_2) != kStorageShape.GetDim(DIM_2) ||
                    dwStorageShape.GetDim(DIM_3) != kStorageShape.GetDim(DIM_3),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: batch/time/K dims must match.", INPUT_DW_NAME,
                        INPUT_K_NAME),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != AStorageShape.GetDim(DIM_0) ||
                    kStorageShape.GetDim(DIM_2) != AStorageShape.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: batch/time dims must match.", INPUT_K_NAME,
                        INPUT_A_NAME),
                return ge::GRAPH_FAILED);

    B = static_cast<int64_t>(vStorageShape.GetDim(DIM_0));
    HV = static_cast<int64_t>(vStorageShape.GetDim(DIM_1));
    HK = static_cast<int64_t>(kStorageShape.GetDim(DIM_1));
    OP_CHECK_IF(HK <= 0 || HV % HK != 0,
                OP_LOGE(context_->GetNodeName(),
                        "HV (%ld) must be a positive multiple of HK (%ld) for GVA.", HV, HK),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaStorageShape.GetDim(DIM_1) != vStorageShape.GetDim(DIM_1) ||
                    AStorageShape.GetDim(DIM_1) != vStorageShape.GetDim(DIM_1) ||
                    dwStorageShape.GetDim(DIM_1) != vStorageShape.GetDim(DIM_1),
                OP_LOGE(context_->GetNodeName(),
                        "The head dim of beta/A/dw must match HV (%ld).", HV),
                return ge::GRAPH_FAILED);
    T = static_cast<int64_t>(vStorageShape.GetDim(DIM_2));
    K = static_cast<int64_t>(kStorageShape.GetDim(DIM_3));
    V = static_cast<int64_t>(vStorageShape.GetDim(DIM_3));
    tiling_.B = B;
    tiling_.HV = HV;
    tiling_.HK = HK;
    tiling_.T = T;
    tiling_.K = K;
    tiling_.V = V;
    auto attrPtr = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
    chunkSize = static_cast<int64_t>(*(attrPtr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_IDX)));
    OP_CHECK_IF(chunkSize != CHUNK_SIZE_64 && chunkSize != CHUNK_SIZE_128,
                OP_LOGE(context_->GetNodeName(),
                        "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.", chunkSize),
                return ge::GRAPH_FAILED);
    tiling_.chunkSize = chunkSize;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto kDesc = context_->GetDynamicInputDesc(INPUT_K_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kDesc);
    auto kDType = kDesc->GetDataType();
    auto betaDesc = context_->GetDynamicInputDesc(INPUT_BETA_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, betaDesc);
    auto betaDType = betaDesc->GetDataType();

    OP_CHECK_IF(SetRowNumKBetaGRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "SetRowNumKBetaGRegbase Failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetRowNumVBetaRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "SetRowNumVBetaRegbase Failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetRowNumMDuDwRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "SetRowNumMDuDwRegbase Failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetRowNumGRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "SetRowNumGRegbase Failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::FixLenTiling()
{
    tiling_.chunkNum = tiling_.B * CeilDiv(tiling_.T, tiling_.chunkSize);
    return ge::GRAPH_SUCCESS;
}

int64_t PrepareWyReprBwdDaTilingA5::CeilDiv(int64_t a, int64_t b)
{
    if (unlikely(b == 0)) {
        return 0;
    }
    return (a + b - 1) / b;
}

ge::graphStatus PrepareWyReprBwdDaTilingA5::VariableLenTiling()
{
    const gert::StorageShape *chunkIndicesShape = context_->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, chunkIndicesShape);
    OP_CHECK_IF(RequiredInputDimNumCheck(chunkIndicesShape, DIM_1, INPUT_CHUNK_INDICES_NAME) != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);

    const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();
    int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(DIM_0);
    OP_CHECK_IF(chunkIndicesDim0 % 2 != 0,
                    OP_LOGE(context_->GetNodeName(),
                            "Check chunk_indices shape failed, the dim 0 of chunk_indices should be even, but get %ld.",
                            chunkIndicesDim0),
                    return ge::GRAPH_FAILED);
    tiling_.chunkNum = chunkIndicesDim0 / 2;

    return ge::GRAPH_SUCCESS;
}

void PrepareWyReprBwdDaTilingA5::PrepareWyReprBwdDaTilingDataPrint()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print PrepareWyReprBwdDaA5 tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling_.B);
    OP_LOGD(nodeName, "=== HV: %ld", tiling_.HV);
    OP_LOGD(nodeName, "=== HK: %ld", tiling_.HK);
    OP_LOGD(nodeName, "=== T: %ld", tiling_.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling_.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling_.V);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling_.chunkSize);
    OP_LOGD(nodeName, "=== chunkNum: %ld", tiling_.chunkNum);
    OP_LOGD(nodeName, "=== rowNumKBetaG: %ld", tiling_.rowNumKBetaG);
    OP_LOGD(nodeName, "=== rowNumVBeta: %ld", tiling_.rowNumVBeta);
    OP_LOGD(nodeName, "=== rowNumMDuDw: %ld", tiling_.rowNumMDuDw);
    OP_LOGD(nodeName, "=== rowNumG: %ld", tiling_.rowNumG);
    OP_LOGD(nodeName, "=== isVariable: %ld", tiling_.isVariable);
    OP_LOGD(nodeName, "=== gCVNum: %ld", tiling_.gCVNum);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print PrepareWyReprBwdDaA5 tiling data end <<<<<<<<<<<<<<<<");
}

} // namespace optiling
