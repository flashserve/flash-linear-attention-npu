/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local_tiling.cpp
 * \brief
 */

#include "prepare_wy_repr_bwd_full_tiling_a5.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
   
bool PrepareWyReprBwdFullTilingA5::SetTiling(gert::TilingContext *context)
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
    PrepareWyReprBwdFullTilingDataPrint();

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

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t userWorkspaceSize = 2 * tiling_.B * tiling_.HV * tiling_.T * tiling_.V;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    context->SetScheduleMode(1); // set as batchmod for template using SyncAll
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwdFull end.");
    return true;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::CommonTiling()
{
    const gert::Shape kStorageShape = context_->GetRequiredInputShape(INPUT_K_IDX)->GetStorageShape();
    const gert::Shape vStorageShape = context_->GetRequiredInputShape(INPUT_V_IDX)->GetStorageShape();
    const gert::Shape betaStorageShape = context_->GetRequiredInputShape(INPUT_BETA_IDX)->GetStorageShape();
    const gert::Shape AStorageShape = context_->GetRequiredInputShape(INPUT_A_IDX)->GetStorageShape();
    const gert::Shape dAStorageShape = context_->GetRequiredInputShape(INPUT_DA_IDX)->GetStorageShape();
    const gert::Shape dwStorageShape = context_->GetRequiredInputShape(INPUT_DW_IDX)->GetStorageShape();
    const gert::Shape duStorageShape = context_->GetRequiredInputShape(INPUT_DU_IDX)->GetStorageShape();
    const gert::Shape gStorageShape = context_->GetRequiredInputShape(INPUT_G_IDX)->GetStorageShape();
    HV = static_cast<int64_t>(vStorageShape.GetDim(DIM_1));
    HK = static_cast<int64_t>(kStorageShape.GetDim(DIM_1));
    OP_CHECK_IF(vStorageShape.GetDim(DIM_0) != kStorageShape.GetDim(DIM_0),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: batch dim 0 must match (got %zu vs %zu).",
                        INPUT_V_NAME, INPUT_K_NAME, vStorageShape.GetDim(DIM_0), kStorageShape.GetDim(DIM_0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vStorageShape.GetDim(DIM_2) != kStorageShape.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: time dim 2 must match (got %zu vs %zu).",
                        INPUT_V_NAME, INPUT_K_NAME, vStorageShape.GetDim(DIM_2), kStorageShape.GetDim(DIM_2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(HV <= 0 || HK <= 0,
                OP_LOGE(context_->GetNodeName(), "HV and HK must be positive (HV=%ld, HK=%ld).", HV, HK),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(HV % HK != 0,
                OP_LOGE(context_->GetNodeName(),
                        "HV (%ld) must be a positive multiple of HK (%ld) for GQA (remainder %ld).", HV, HK,
                        HV % HK),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CompareShape(vStorageShape, duStorageShape, INPUT_V_NAME, INPUT_DU_NAME, DIM_NUM_4) !=
                    ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);
    OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, INPUT_BETA_NAME, INPUT_G_NAME, DIM_NUM_3) !=
                    ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != gStorageShape.GetDim(DIM_0),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: batch dim 0 must match (got %zu vs %zu).",
                        INPUT_K_NAME, INPUT_G_NAME, kStorageShape.GetDim(DIM_0), gStorageShape.GetDim(DIM_0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStorageShape.GetDim(DIM_2) != gStorageShape.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: time dim 2 must match (got %zu vs %zu).",
                        INPUT_K_NAME, INPUT_G_NAME, kStorageShape.GetDim(DIM_2), gStorageShape.GetDim(DIM_2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vStorageShape.GetDim(DIM_1) != gStorageShape.GetDim(DIM_1),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: head dim 1 must match HV (got %zu vs %zu).",
                        INPUT_V_NAME, INPUT_G_NAME, vStorageShape.GetDim(DIM_1), gStorageShape.GetDim(DIM_1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CompareShape(AStorageShape, dAStorageShape, INPUT_A_NAME, INPUT_DA_NAME, DIM_NUM_4) !=
                    ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);
    OP_CHECK_IF(CompareShape(dwStorageShape, vStorageShape, INPUT_DW_NAME, INPUT_V_NAME, DIM_NUM_3) !=
                    ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);
    OP_CHECK_IF(dwStorageShape.GetDim(DIM_3) != kStorageShape.GetDim(DIM_3),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: last dim K must match (got %zu vs %zu).",
                        INPUT_DW_NAME, INPUT_K_NAME, dwStorageShape.GetDim(DIM_3), kStorageShape.GetDim(DIM_3)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != AStorageShape.GetDim(DIM_0),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: batch dim 0 must match (got %zu vs %zu).",
                        INPUT_K_NAME, INPUT_A_NAME, kStorageShape.GetDim(DIM_0), AStorageShape.GetDim(DIM_0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kStorageShape.GetDim(DIM_2) != AStorageShape.GetDim(DIM_2),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: time dim 2 must match (got %zu vs %zu).",
                        INPUT_K_NAME, INPUT_A_NAME, kStorageShape.GetDim(DIM_2), AStorageShape.GetDim(DIM_2)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vStorageShape.GetDim(DIM_1) != AStorageShape.GetDim(DIM_1),
                OP_LOGE(context_->GetNodeName(),
                        "Compare input shape of %s and %s failed: head dim 1 must match HV (got %zu vs %zu).",
                        INPUT_V_NAME, INPUT_A_NAME, vStorageShape.GetDim(DIM_1), AStorageShape.GetDim(DIM_1)),
                return ge::GRAPH_FAILED);
    B = static_cast<int64_t>(vStorageShape.GetDim(DIM_0));
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
    OP_CHECK_NULL_WITH_CONTEXT(context_, kDesc); // check xDesc is not null
    auto kDType = kDesc->GetDataType();
    auto betaDesc = context_->GetDynamicInputDesc(INPUT_BETA_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, betaDesc); // check xDesc is not null
    auto betaDType = betaDesc->GetDataType();

    OP_CHECK_IF(SetKBeteVecRowRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "SetKBeteVecRow Failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetDkbVecRowRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetDkbVecRow Failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetDkbgVecRowRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetDkbgVecRow Failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetDvbVecRowRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetDvbVecRow Failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetKKTVecRowRegbase(ubSize, kDType, betaDType) != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetKKTVecRow Failed."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::FixLenTiling()
{
    tiling_.chunkNum = tiling_.B * CeilDiv(tiling_.T, tiling_.chunkSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::VariableLenTiling()
{
    const gert::StorageShape *chunkIndicesShape = context_->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    const gert::StorageShape *seqlensShape = context_->GetOptionalInputShape(INPUT_SEQLENS_IDX);
    auto cuSeqlensTensor = context_->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    auto chunkIndicesTensor = context_->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cuSeqlensTensor);
    OP_CHECK_NULL_WITH_CONTEXT(context_, chunkIndicesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqlensShape);
    OP_CHECK_IF(RequiredInputDimNumCheck(chunkIndicesShape, DIM_1, INPUT_CHUNK_INDICES_NAME) != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(RequiredInputDimNumCheck(seqlensShape, DIM_1, INPUT_SEQLENS_NAME) != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);

    const gert::Shape seqlensStorageShape = seqlensShape->GetStorageShape();
    int64_t seqlensDim0 = seqlensStorageShape.GetDim(DIM_0);
    OP_CHECK_IF(seqlensDim0 < 2,
                OP_LOGE(context_->GetNodeName(),
                        "Check seqlens shape failed, the dim 0 of seqlens should be larger than 1, but get %ld.",
                        seqlensDim0),
                return ge::GRAPH_FAILED);
    const int64_t* cuSeqlens = cuSeqlensTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, cuSeqlens);
    if(cuSeqlens[0] != 0){
        OP_LOGE(context_->GetNodeName(),
                            "Check seqlens data failed, the seqlens[0] should be 0, but get %ld.",
                            cuSeqlens[0]);
        return ge::GRAPH_FAILED;
    }
    std::vector<int64_t> expectChunkIndices;
    for(int64_t i = 1; i < seqlensDim0; i++){
        int64_t curSeqLen = cuSeqlens[i] - cuSeqlens[i - 1];
        OP_CHECK_IF(curSeqLen <= 0,
                    OP_LOGE(context_->GetNodeName(),
                            "Check seqlens data failed, the seqlens[%ld]:[%ld] should be larger than seqlens[%ld]:[%ld]",
                            i, cuSeqlens[i], i - 1, cuSeqlens[i - 1]),
                    return ge::GRAPH_FAILED);
        for(int64_t j = 0;j < curSeqLen; j += chunkSize){
            expectChunkIndices.push_back(i - 1);
            expectChunkIndices.push_back(j / chunkSize);
        }
    }

    
    const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();
    int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(DIM_0);
    OP_CHECK_IF(chunkIndicesDim0 != expectChunkIndices.size(),
                OP_LOGE(context_->GetNodeName(),
                        "Check chunk_indices shape failed, the len of chunk_indices should be %ld, but get %ld.",
                        expectChunkIndices.size(), chunkIndicesDim0),
                return ge::GRAPH_FAILED);
    const int64_t* chunkIndices = chunkIndicesTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, chunkIndices);
    for(int64_t i = 0; i < expectChunkIndices.size(); i++){
        OP_CHECK_IF(expectChunkIndices[i] != chunkIndices[i],
                    OP_LOGE(context_->GetNodeName(),
                            "Check chunk_indices data failed, the chunk_indices[%ld] should be %ld, but get %ld.",
                            i, expectChunkIndices[i], chunkIndices[i]),
                    return ge::GRAPH_FAILED);
    }
    tiling_.chunkNum = chunkIndicesStorageShape.GetDim(DIM_0) / 2;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                            const char *inputName)
{
    OP_CHECK_IF(curShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "Input %s is required, but got nullptr.", inputName),
                return ge::GRAPH_FAILED);
    const gert::Shape storageShape = curShape->GetStorageShape();
    size_t dimNum = storageShape.GetDimNum();
    OP_CHECK_IF(dimNum != validDimNum,
                OP_LOGE(context_->GetNodeName(),
                        "Check input %s shape failed, the dim num should be %zu, but get %zu.", inputName,
                        validDimNum, dimNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                const char *inputName2, size_t compareDimNum)
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

ge::graphStatus PrepareWyReprBwdFullTilingA5::SetKBeteVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
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
        useUbSize += 2 * rowNum * K * sizeofKType;
        
        if (useUbSize <= ubSize) {
            break;
        }
        rowNum = rowNum / 2;
        
    }
    tiling_.kBeteVecRow = rowNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::SetDkbVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
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
    uint64_t dkbCVNum = 2;
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;

        if (useUbSize <= ubSize) {
            dkbCVNum += (ubSize - useUbSize) / (rowNum * K * sizeofKType);
            dkbCVNum = std::min(dkbCVNum, MAX_CUBE_VEC_SYNC_NUM);
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.dkbVecRow = rowNum;
    tiling_.dkbCVNum = dkbCVNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::SetDkbgVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
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
    uint64_t dkbgCVNum = 2;
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * K * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * sizeofBetaType;

        if (useUbSize <= ubSize) {
            dkbgCVNum += (ubSize - useUbSize) / (rowNum * K * sizeofKType);
            dkbgCVNum = std::min(dkbgCVNum, MAX_CUBE_VEC_SYNC_NUM);
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.dkbgVecRow = rowNum;
    tiling_.dkbgCVNum = dkbgCVNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::SetDvbVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
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
    uint64_t dvbCVNum = 2;
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * rowNum * V * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * sizeofBetaType;
        useUbSize += 2 * rowNum * V * sizeofKType;
        useUbSize += 2 * rowNum * V * sizeofKType;
        useUbSize += 2 * rowNum * sizeofBetaType;

        if (useUbSize <= ubSize) {
            dvbCVNum += (ubSize - useUbSize) / (rowNum * V * sizeofKType);
            dvbCVNum = std::min(dvbCVNum, MAX_CUBE_VEC_SYNC_NUM);
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.dvbVecRow = rowNum;
    tiling_.dvbCVNum = dvbCVNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareWyReprBwdFullTilingA5::SetKKTVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType)
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
    uint64_t kktCVNum = 2;
    while (rowNum >= 8) {
        uint64_t useUbSize = 0;
        useUbSize += 2 * chunkSize * sizeofBetaType;
        useUbSize += 2 * chunkSize * sizeofBetaType;
        useUbSize += 2 * rowNum * chunkSize * sizeofKType;
        useUbSize += 2 * rowNum * chunkSize * sizeofKType;
        useUbSize += 2 * chunkSize * sizeofBetaType;

        useUbSize += chunkSize * sizeof(float);
        useUbSize += chunkSize * sizeof(float);

        if (useUbSize <= ubSize) {
            kktCVNum += (ubSize - useUbSize) / (rowNum * chunkSize * sizeofKType);
            kktCVNum = std::min(kktCVNum, MAX_CUBE_VEC_SYNC_NUM);
            break;
        }
        rowNum = rowNum / 2;
    }
    tiling_.kktVecRow = rowNum;
    tiling_.kktCVNum = kktCVNum;
    return ge::GRAPH_SUCCESS;
}

int64_t PrepareWyReprBwdFullTilingA5::CeilDiv(int64_t a, int64_t b)
{
    if (unlikely(b == 0)) {
        return 0;
    }
    return (a + b - 1) / b;
}

void PrepareWyReprBwdFullTilingA5::PrepareWyReprBwdFullTilingDataPrint()
{
    auto nodeName = context_->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print PrepareWyReprBwdFullA5 tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling_.B);
    OP_LOGD(nodeName, "=== HV: %ld", tiling_.HV);
    OP_LOGD(nodeName, "=== HK: %ld", tiling_.HK);
    OP_LOGD(nodeName, "=== T: %ld", tiling_.T);
    OP_LOGD(nodeName, "=== K: %ld", tiling_.K);
    OP_LOGD(nodeName, "=== V: %ld", tiling_.V);
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling_.chunkSize);
    OP_LOGD(nodeName, "=== chunkNum: %ld", tiling_.chunkNum);
    OP_LOGD(nodeName, "=== kBeteVecRow: %ld", tiling_.kBeteVecRow);
    OP_LOGD(nodeName, "=== dkbVecRow: %ld", tiling_.dkbVecRow);
    OP_LOGD(nodeName, "=== dkbgVecRow: %ld", tiling_.dkbgVecRow);
    OP_LOGD(nodeName, "=== dvbVecRow: %ld", tiling_.dvbVecRow);
    OP_LOGD(nodeName, "=== kktVecRow: %ld", tiling_.kktVecRow);
    OP_LOGD(nodeName, "=== isVariable: %ld", tiling_.isVariable);
    OP_LOGD(nodeName, "=== kBetaCVNum: %ld", tiling_.kBetaCVNum);
    OP_LOGD(nodeName, "=== dkbCVNum: %ld", tiling_.dkbCVNum);
    OP_LOGD(nodeName, "=== dkbgCVNum: %ld", tiling_.dkbgCVNum);
    OP_LOGD(nodeName, "=== dvbCVNum: %ld", tiling_.dvbCVNum);
    OP_LOGD(nodeName, "=== kktCVNum: %ld", tiling_.kktCVNum);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print PrepareWyReprBwdFullA5 tiling data end <<<<<<<<<<<<<<<<");
}

} // namespace optiling
