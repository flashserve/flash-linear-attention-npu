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

#include "prepare_wy_repr_bwd_full_tiling.h"
#include "arch35/prepare_wy_repr_bwd_full_tiling_a5.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {


class PrepareWyReprBwdFullTilingProcessor {
    gert::TilingContext *context_;
    PrepareWyReprBwdFullTilingData &tiling_;
    int64_t B = 0;
    int64_t HK = 0;
    int64_t HV = 0;
    int64_t K = 0;
    int64_t V = 0;
    int64_t T = 0;
    int64_t chunkSize = 0;

public:
    explicit PrepareWyReprBwdFullTilingProcessor(gert::TilingContext *context, PrepareWyReprBwdFullTilingData &tiling)
        : context_(context), tiling_(tiling)
    {
    }

    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
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

    ge::graphStatus PreCheck()
    {
        const ge::char_t *nodeName = context_->GetNodeName();
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_K_IDX), DIM_NUM_4, INPUT_K_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_V_IDX), DIM_NUM_4, INPUT_V_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_BETA_IDX), DIM_NUM_3,
                                             INPUT_BETA_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_A_IDX), DIM_NUM_4, INPUT_A_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_DA_IDX), DIM_NUM_4, INPUT_DA_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_DW_IDX), DIM_NUM_4, INPUT_DW_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_DU_IDX), DIM_NUM_4, INPUT_DU_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_G_IDX), DIM_NUM_3, INPUT_G_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
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
    uint64_t AlignUp(uint64_t value, uint64_t align)
    {
        return (value + align - 1) / align * align;
    }

    uint64_t GetTypeSize(ge::DataType dtype)
    {
        return dtype == ge::DataType::DT_FLOAT ? SIZE_FP32 : SIZE_HALF;
    }

    uint64_t GetFusedUseUbSize(uint64_t rowNum, uint64_t ubK, uint64_t ubV, uint64_t sizeofKType)
    {
        constexpr uint64_t SAFETY_MARGIN = 16 * 1024;
        uint64_t persistentBytes = 0;
        persistentBytes += AlignUp(chunkSize * SIZE_FP32, ONE_BLOCK_32); // beta fp32
        persistentBytes += AlignUp(chunkSize * SIZE_FP32, ONE_BLOCK_32); // exp(g) fp32
        persistentBytes += AlignUp(chunkSize * SIZE_FP32, ONE_BLOCK_32); // dbeta accumulator
        persistentBytes += AlignUp(chunkSize * SIZE_FP32, ONE_BLOCK_32); // dg accumulator

        uint64_t queueBytes = 0;
        queueBytes += 2 * AlignUp(rowNum * ubK * sizeofKType, ONE_BLOCK_32);
        queueBytes += 2 * AlignUp(rowNum * ubV * sizeofKType, ONE_BLOCK_32);

        uint64_t group1TempBytes = 0;
        group1TempBytes += AlignUp(5 * rowNum * K * SIZE_FP32, ONE_BLOCK_32);
        group1TempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32); // beta brcb
        group1TempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32); // exp(g) brcb
        group1TempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32); // reduce scratch

        uint64_t group2VTempBytes = 0;
        group2VTempBytes += AlignUp(2 * rowNum * V * SIZE_FP32, ONE_BLOCK_32);
        group2VTempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32);

        uint64_t group2KktTempBytes = 0;
        group2KktTempBytes += AlignUp(chunkSize * chunkSize * SIZE_FP32, ONE_BLOCK_32);
        group2KktTempBytes += AlignUp(2 * rowNum * chunkSize * SIZE_FP32, ONE_BLOCK_32);
        group2KktTempBytes += AlignUp(chunkSize * ONE_BLOCK_32, ONE_BLOCK_32);

        uint64_t maxTempBytes = group1TempBytes > group2VTempBytes ? group1TempBytes : group2VTempBytes;
        maxTempBytes = maxTempBytes > group2KktTempBytes ? maxTempBytes : group2KktTempBytes;
        return persistentBytes + queueBytes + maxTempBytes + SAFETY_MARGIN;
    }

    uint64_t GetFusedRowNum(uint64_t ubSize, ge::DataType kType)
    {
        uint64_t rowNum = chunkSize / 2;
        uint64_t sizeofKType = GetTypeSize(kType);
        uint64_t ubK = K > chunkSize ? K : chunkSize;
        uint64_t ubV = V > chunkSize ? V : chunkSize;
        while (rowNum >= 8) {
            uint64_t useUbSize = GetFusedUseUbSize(rowNum, ubK, ubV, sizeofKType);
            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        return rowNum;
    }

    ge::graphStatus SetFusedTiling(uint64_t ubSize, ge::DataType kType, uint64_t usedCoreNum)
    {
        uint64_t fusedRowNum = GetFusedRowNum(ubSize, kType);
        tiling_.set_fusedKVecRow(fusedRowNum);
        tiling_.set_fusedVVecRow(fusedRowNum);
        tiling_.set_fusedKktVecRow(fusedRowNum);
        tiling_.set_workspaceBufferCount(2);
        tiling_.set_usedCoreNum(usedCoreNum);

        uint64_t sizeofKType = GetTypeSize(kType);
        uint64_t maxKV = K > V ? K : V;
        uint64_t slotSize = 0;
        slotSize += 3 * AlignUp(chunkSize * K * sizeofKType, ONE_BLOCK_32);
        slotSize += AlignUp(chunkSize * maxKV * sizeofKType, ONE_BLOCK_32);
        slotSize += AlignUp(chunkSize * chunkSize * sizeofKType, ONE_BLOCK_32);
        tiling_.set_workspaceSlotSize(slotSize);
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus CommonTiling()
    {
        const gert::Shape kStorageShape = context_->GetRequiredInputShape(INPUT_K_IDX)->GetStorageShape();
        const gert::Shape vStorageShape = context_->GetRequiredInputShape(INPUT_V_IDX)->GetStorageShape();
        const gert::Shape betaStorageShape = context_->GetRequiredInputShape(INPUT_BETA_IDX)->GetStorageShape();
        const gert::Shape AStorageShape = context_->GetRequiredInputShape(INPUT_A_IDX)->GetStorageShape();
        const gert::Shape dAStorageShape = context_->GetRequiredInputShape(INPUT_DA_IDX)->GetStorageShape();
        const gert::Shape dwStorageShape = context_->GetRequiredInputShape(INPUT_DW_IDX)->GetStorageShape();
        const gert::Shape duStorageShape = context_->GetRequiredInputShape(INPUT_DU_IDX)->GetStorageShape();
        const gert::Shape gStorageShape = context_->GetRequiredInputShape(INPUT_G_IDX)->GetStorageShape();
        /* Benchmark (test_npu_prepare_wy_repr_bwd_full): k [B,HK,T,K], v/dw/… [B,HV,T,…], HV % HK == 0. */
        const int64_t HV = static_cast<int64_t>(vStorageShape.GetDim(DIM_1));
        const int64_t HK = static_cast<int64_t>(kStorageShape.GetDim(DIM_1));
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
        OP_CHECK_IF(CompareShape(dwStorageShape, kStorageShape, INPUT_DW_NAME, INPUT_K_NAME, DIM_NUM_3) !=
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
        tiling_.set_B(B);
        tiling_.set_HV(HV);
        tiling_.set_HK(HK);
        tiling_.set_T(T);
        tiling_.set_K(K);
        tiling_.set_V(V);
        OP_CHECK_IF(V != V_DIM_128 && V != V_DIM_256,
                    OP_LOGE(context_->GetNodeName(),
                            "Check value dim V failed: only %ld or %ld is supported, but get %ld.", V_DIM_128, V_DIM_256, V),
                    return ge::GRAPH_FAILED);
        auto attrPtr = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
        chunkSize = static_cast<int64_t>(*(attrPtr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_IDX)));
        OP_CHECK_IF(chunkSize != CHUNK_SIZE_64 && chunkSize != CHUNK_SIZE_128,
                    OP_LOGE(context_->GetNodeName(),
                            "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.", chunkSize),
                    return ge::GRAPH_FAILED);
        tiling_.set_chunkSize(chunkSize);
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        auto kDesc = context_->GetDynamicInputDesc(INPUT_K_IDX, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kDesc); // check xDesc is not null
        auto kDType = kDesc->GetDataType();
        OP_CHECK_IF(SetFusedTiling(ubSize, kDType, platform_ascendc::PlatformAscendC(context_->GetPlatformInfo()).GetCoreNumAic()) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "SetFusedTiling Failed."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    int64_t CeilDiv(int64_t a, int64_t b)
    {
        if (unlikely(b == 0)) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    ge::graphStatus FixLenTiling()
    {
        tiling_.set_chunkNum(tiling_.get_B() * CeilDiv(tiling_.get_T(), tiling_.get_chunkSize()));
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
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
        tiling_.set_chunkNum(chunkIndicesStorageShape.GetDim(DIM_0) / 2);
        return ge::GRAPH_SUCCESS;
    }
};

static void PrepareWyReprBwdFullTilingDataPrint(gert::TilingContext *context, PrepareWyReprBwdFullTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print PrepareWyReprBwdFull tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.get_B());
    OP_LOGD(nodeName, "=== HV: %ld", tiling.get_HV());
    OP_LOGD(nodeName, "=== HK: %ld", tiling.get_HK());
    OP_LOGD(nodeName, "=== T: %ld", tiling.get_T());
    OP_LOGD(nodeName, "=== K: %ld", tiling.get_K());
    OP_LOGD(nodeName, "=== V: %ld", tiling.get_V());
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.get_chunkSize());
    OP_LOGD(nodeName, "=== fusedKVecRow: %ld", tiling.get_fusedKVecRow());
    OP_LOGD(nodeName, "=== workspaceSlotSize: %ld", tiling.get_workspaceSlotSize());
    OP_LOGD(nodeName, "=== workspaceBufferCount: %ld", tiling.get_workspaceBufferCount());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print PrepareWyReprBwdFull tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4PrepareWyReprBwdFull(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwdFull start.");
    PrepareWyReprBwdFullTilingData tiling;
    PrepareWyReprBwdFullTilingProcessor processor(context, tiling);

    OP_CHECK_IF(processor.PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        PrepareWyReprBwdFullTilingA5 prepareWyReprBwdFullTilingA5;
        OP_CHECK_IF(!prepareWyReprBwdFullTilingA5.SetTiling(context),
                    OP_LOGE(context->GetNodeName(), "SetTiling failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    
    OP_CHECK_IF(processor.CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    if (cuSeqlensTensor == nullptr) {
        OP_CHECK_IF(processor.FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        tiling.set_isVariable(0);
    } else {
        OP_CHECK_IF(tiling.get_B() != VAR_LEN_B_DIM_1,
                    OP_LOGE(context->GetNodeName(),
                            "If cu_seqlens is not nullptr, the dim 0 of q needs to be 1, but now is %ld.",
                            tiling.get_B()),
                    return ge::GRAPH_FAILED);
        auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesTensor);
        OP_CHECK_IF(processor.VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        tiling.set_isVariable(1);
    }
    if (tiling.get_V() == V_DIM_256) {
        context->SetTilingKey(2);
    } else {
        context->SetTilingKey(1);
    }
    OP_LOGD(context->GetNodeName(), "tilingKey: %d (V=%ld)", context->GetTilingKey(), tiling.get_V());
    PrepareWyReprBwdFullTilingDataPrint(context, tiling);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t userWorkspaceSize =
        tiling.get_usedCoreNum() * tiling.get_workspaceBufferCount() * tiling.get_workspaceSlotSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    context->SetScheduleMode(1); // set as batchmod for template using SyncAll
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwdFull end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForPrepareWyReprBwdFull(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct PrepareWyReprBwdFullCompileInfo {};

IMPL_OP_OPTILING(PrepareWyReprBwdFull)
    .Tiling(Tiling4PrepareWyReprBwdFull)
    .TilingParse<PrepareWyReprBwdFullCompileInfo>(TilingPrepareForPrepareWyReprBwdFull);

} // namespace optiling
