/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_bwd_dv_local_tiling.cpp
 * \brief
 */

#include "prepare_wy_repr_bwd_full_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
static constexpr size_t INPUT_K_IDX = 0;
static constexpr size_t INPUT_V_IDX = 1;
static constexpr size_t INPUT_BETA_IDX = 2;
static constexpr size_t INPUT_A_IDX = 3;
static constexpr size_t INPUT_DA_IDX = 4;
static constexpr size_t INPUT_DW_IDX = 5;
static constexpr size_t INPUT_DU_IDX = 6;
static constexpr size_t INPUT_G_IDX = 7;
static constexpr size_t INPUT_SEQLENS_IDX = 8;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 9;

static constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;

static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_NUM_4 = 4;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr int64_t CHUNK_INDICES_DIM_1_SIZE = 2;

static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;

static constexpr int64_t VAR_LEN_B_DIM_1 = 1;

static constexpr const char *const INPUT_K_NAME = "k";
static constexpr const char *const INPUT_V_NAME = "v";
static constexpr const char *const INPUT_BETA_NAME = "beta";
static constexpr const char *const INPUT_A_NAME = "A";
static constexpr const char *const INPUT_DA_NAME = "dA";
static constexpr const char *const INPUT_DW_NAME = "dw";
static constexpr const char *const INPUT_DU_NAME = "du";
static constexpr const char *const INPUT_G_NAME = "g";
static constexpr const char *const INPUT_CHUNK_INDICES_NAME = "chunk_indices";


class PrepareWyReprBwdFullTilingProcessor {
    gert::TilingContext *context_;
    PrepareWyReprBwdFullTilingData &tiling_;

public:
    explicit PrepareWyReprBwdFullTilingProcessor(gert::TilingContext *context, PrepareWyReprBwdFullTilingData &tiling)
        : context_(context), tiling_(tiling)
    {
    }

    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                             const char *inputName)
    {
        OP_CHECK_IF(curShape == nullptr, OP_LOGE(context_->GetNodeName(), "Input %s is required, but got nullptr.", inputName), return ge::GRAPH_FAILED);
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
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetRequiredInputShape(INPUT_BETA_IDX), DIM_NUM_3, INPUT_BETA_NAME) !=
                        ge::GRAPH_SUCCESS,
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
        OP_CHECK_IF(CompareShape(vStorageShape, kStorageShape, INPUT_V_NAME, INPUT_K_NAME, DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(vStorageShape, duStorageShape, INPUT_V_NAME, INPUT_DU_NAME, DIM_NUM_4) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, INPUT_BETA_NAME, INPUT_G_NAME, DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(kStorageShape, gStorageShape, INPUT_K_NAME, INPUT_G_NAME, DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(AStorageShape, dAStorageShape, INPUT_A_NAME, INPUT_DA_NAME, DIM_NUM_4) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(dwStorageShape, kStorageShape, INPUT_DW_NAME, INPUT_K_NAME, DIM_NUM_4) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(kStorageShape, AStorageShape, INPUT_K_NAME, INPUT_A_NAME, DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        tiling_.set_B(static_cast<int64_t>(vStorageShape.GetDim(DIM_0)));
        tiling_.set_H(static_cast<int64_t>(vStorageShape.GetDim(DIM_1)));
        tiling_.set_T(static_cast<int64_t>(vStorageShape.GetDim(DIM_2)));
        tiling_.set_K(static_cast<int64_t>(kStorageShape.GetDim(DIM_3)));
        tiling_.set_V(static_cast<int64_t>(vStorageShape.GetDim(DIM_3)));
        tiling_.set_kBeteVecRow(32);
        tiling_.set_dkbVecRow(32);
        tiling_.set_dkbgVecRow(16);
        tiling_.set_dvbVecRow(32);
        tiling_.set_kktVecRow(32);

        auto attrPtr = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
        int64_t chunkSize = static_cast<int64_t>(*(attrPtr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_IDX)));
        OP_CHECK_IF(chunkSize != CHUNK_SIZE_64 && chunkSize != CHUNK_SIZE_128,
                    OP_LOGE(context_->GetNodeName(),
                            "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.", chunkSize),
                    return ge::GRAPH_FAILED);
        tiling_.set_chunkSize(chunkSize);

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
        tiling_.set_chunkNum(CeilDiv(tiling_.get_T(), tiling_.get_chunkSize()));
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        const gert::StorageShape *chunkIndicesShape = context_->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, chunkIndicesShape);
        OP_CHECK_IF(RequiredInputDimNumCheck(chunkIndicesShape, DIM_2, INPUT_CHUNK_INDICES_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim1 = chunkIndicesStorageShape.GetDim(DIM_1);
        OP_CHECK_IF(chunkIndicesDim1 != CHUNK_INDICES_DIM_1_SIZE,
                    OP_LOGE(context_->GetNodeName(),
                            "Check chunk_indices shape failed, the dim 1 of chunk_indices should be 2, but get %ld.",
                            chunkIndicesDim1),
                    return ge::GRAPH_FAILED);
        tiling_.set_chunkNum(chunkIndicesStorageShape.GetDim(DIM_0));

        return ge::GRAPH_SUCCESS;
    }
};

static void PrepareWyReprBwdFullTilingDataPrint(gert::TilingContext *context, PrepareWyReprBwdFullTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print PrepareWyReprBwdFull tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== B: %ld", tiling.get_B());
    OP_LOGD(nodeName, "=== G: %ld", tiling.get_H());
    OP_LOGD(nodeName, "=== T: %ld", tiling.get_T());
    OP_LOGD(nodeName, "=== K: %ld", tiling.get_K());
    OP_LOGD(nodeName, "=== V: %ld", tiling.get_V());
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.get_chunkSize());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print PrepareWyReprBwdFull tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4PrepareWyReprBwdFull(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4PrepareWyReprBwdFull start.");
    PrepareWyReprBwdFullTilingData tiling;
    PrepareWyReprBwdFullTilingProcessor processor(context, tiling);

    OP_CHECK_IF(processor.PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
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
    context->SetTilingKey(1);
    OP_LOGD(context->GetNodeName(), "tilingKey: %d", context->GetTilingKey());
    PrepareWyReprBwdFullTilingDataPrint(context, tiling);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());


    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t userWorkspaceSize = 2 * tiling.get_B() * tiling.get_H() * tiling.get_T() * tiling.get_V();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

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
