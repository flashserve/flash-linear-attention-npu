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

#include "chunk_bwd_dv_local_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
static constexpr size_t INPUT_Q_IDX = 0;
static constexpr size_t INPUT_K_IDX = 1;
static constexpr size_t INPUT_DO_IDX = 2;
static constexpr size_t INPUT_G_IDX = 3;
static constexpr size_t INPUT_TRI_MATRIX_IDX = 4;
static constexpr size_t INPUT_SEQLENS_IDX = 7;
static constexpr size_t INPUT_CHUNK_INDICES_IDX = 8;
static constexpr size_t ATTR_SCALE_IDX = 0;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 1;

static constexpr size_t Q_K_DO_DIM_NUM = 4;
static constexpr size_t G_DIM_NUM = 3;
static constexpr size_t TRI_MATRIX_DIM_NUM = 2;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;

static constexpr int64_t V_L_B = 1;
static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;

static constexpr const char *const INPUT_Q_NAME = "q";
static constexpr const char *const INPUT_K_NAME = "k";
static constexpr const char *const INPUT_DO_NAME = "do";
static constexpr const char *const INPUT_G_NAME = "g";
static constexpr const char *const INPUT_TRI_MATRIX_NAME = "upper_tri_matrix";

class ChunkBwdDvLocalTilingProcessor {
    gert::TilingContext *context_;
    ChunkBwdDvLocalTilingData &tiling_;

public:
    explicit ChunkBwdDvLocalTilingProcessor(gert::TilingContext *context, ChunkBwdDvLocalTilingData &tiling)
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
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetInputShape(INPUT_Q_IDX), Q_K_DO_DIM_NUM, INPUT_Q_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetInputShape(INPUT_K_IDX), Q_K_DO_DIM_NUM, INPUT_K_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetInputShape(INPUT_DO_IDX), Q_K_DO_DIM_NUM, INPUT_DO_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetInputShape(INPUT_G_IDX), G_DIM_NUM, INPUT_G_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        // OP_CHECK_IF(RequiredInputDimNumCheck(context_->GetInputShape(INPUT_TRI_MATRIX_IDX), TRI_MATRIX_DIM_NUM,
        //                                      INPUT_TRI_MATRIX_NAME) != ge::GRAPH_SUCCESS,
        //             , return ge::GRAPH_FAILED);
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
        const gert::Shape qStorageShape = context_->GetInputShape(INPUT_Q_IDX)->GetStorageShape();
        const gert::Shape kStorageShape = context_->GetInputShape(INPUT_K_IDX)->GetStorageShape();
        const gert::Shape dOStorageShape = context_->GetInputShape(INPUT_DO_IDX)->GetStorageShape();
        const gert::Shape gStorageShape = context_->GetInputShape(INPUT_G_IDX)->GetStorageShape();
        OP_CHECK_IF(CompareShape(qStorageShape, kStorageShape, INPUT_Q_NAME, INPUT_K_NAME, Q_K_DO_DIM_NUM) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(qStorageShape, dOStorageShape, INPUT_Q_NAME, INPUT_DO_NAME, G_DIM_NUM) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(qStorageShape, gStorageShape, INPUT_Q_NAME, INPUT_G_NAME, G_DIM_NUM) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        tiling_.set_b(static_cast<int64_t>(qStorageShape.GetDim(DIM_0)));
        tiling_.set_h(static_cast<int64_t>(qStorageShape.GetDim(DIM_1)));
        tiling_.set_t(static_cast<int64_t>(qStorageShape.GetDim(DIM_2)));
        tiling_.set_k(static_cast<int64_t>(qStorageShape.GetDim(DIM_3)));
        tiling_.set_v(static_cast<int64_t>(dOStorageShape.GetDim(DIM_3)));

        auto attrPtr = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrPtr);
        float scale = *(attrPtr->GetAttrPointer<float>(ATTR_SCALE_IDX));
        tiling_.set_scale(scale);
        int64_t chunkSize = static_cast<int64_t>(*(attrPtr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_IDX)));
        OP_CHECK_IF(chunkSize != CHUNK_SIZE_64 && chunkSize != CHUNK_SIZE_128,
                    OP_LOGE(context_->GetNodeName(),
                            "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.", chunkSize),
                    return ge::GRAPH_FAILED);
        tiling_.set_chunkSize(chunkSize);

        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        tiling_.set_totalCoreNum(static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv()));
        return ge::GRAPH_SUCCESS;
    }

    int64_t CeilDiv(int64_t a, int64_t b)
    {
        if (unlikely(b == 0)) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    void InterKernelSplit(int64_t chunkNumForT, int64_t totalCoreNum)
    {
        int64_t allChunkNum = chunkNumForT * tiling_.get_b(); // b*t , h核内循环
        int64_t chunkNumTailCore = allChunkNum / totalCoreNum;
        int64_t chunkNumPreCore = chunkNumTailCore + 1;
        int64_t preCoreNum = allChunkNum % totalCoreNum;
        int64_t tailCoreNum = chunkNumTailCore == 0 ? 0 : totalCoreNum - preCoreNum;

        tiling_.set_chunkNumForT(chunkNumForT);
        tiling_.set_chunkNumPreCore(chunkNumPreCore);
        tiling_.set_chunkNumTailCore(chunkNumTailCore);
        tiling_.set_preCoreNum(preCoreNum);
        tiling_.set_tailCoreNum(tailCoreNum);
        tiling_.set_totalCoreNum(preCoreNum + tailCoreNum);
    }

    ge::graphStatus FixLenTiling()
    {
        InterKernelSplit(CeilDiv(tiling_.get_t(), tiling_.get_chunkSize()), tiling_.get_totalCoreNum());

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        const gert::StorageShape *chunkIndicesShape = context_->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, chunkIndicesShape);
        const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();

        InterKernelSplit(static_cast<int64_t>(chunkIndicesStorageShape.GetDim(DIM_0)), tiling_.get_totalCoreNum());

        return ge::GRAPH_SUCCESS;
    }
};

static void ChunkBwdDvLocalTilingDataPrint(gert::TilingContext *context, ChunkBwdDvLocalTilingData &tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ChunkBwdDvLocal tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, "=== b: %ld", tiling.get_b());
    OP_LOGD(nodeName, "=== h: %ld", tiling.get_h());
    OP_LOGD(nodeName, "=== t: %ld", tiling.get_t());
    OP_LOGD(nodeName, "=== k: %ld", tiling.get_k());
    OP_LOGD(nodeName, "=== v: %ld", tiling.get_v());
    OP_LOGD(nodeName, "=== chunkNumForT: %ld", tiling.get_chunkNumForT());
    OP_LOGD(nodeName, "=== chunkNumPreCore: %ld", tiling.get_chunkNumPreCore());
    OP_LOGD(nodeName, "=== chunkNumTailCore: %ld", tiling.get_chunkNumTailCore());
    OP_LOGD(nodeName, "=== preCoreNum: %ld", tiling.get_preCoreNum());
    OP_LOGD(nodeName, "=== tailCoreNum: %ld", tiling.get_tailCoreNum());
    OP_LOGD(nodeName, "=== totalCoreNum: %ld", tiling.get_totalCoreNum());
    OP_LOGD(nodeName, "=== chunkSize: %ld", tiling.get_chunkSize());
    OP_LOGD(nodeName, "=== scale: %f", tiling.get_scale());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ChunkBwdDvLocal tiling data end <<<<<<<<<<<<<<<<");
}

ge::graphStatus Tiling4ChunkBwdDvLocal(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkBwdDvLocal start.");
    ChunkBwdDvLocalTilingData tiling;
    ChunkBwdDvLocalTilingProcessor processor(context, tiling);

    OP_CHECK_IF(processor.PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(processor.CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_SEQLENS_IDX);
    if (cuSeqlensTensor == nullptr) {
        OP_CHECK_IF(processor.FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        tiling.set_isVariable(false);
    } else {
        OP_CHECK_IF(tiling.get_b() != V_L_B,
                    OP_LOGE(context->GetNodeName(),
                            "If cu_seqlens is not nullptr, the dim 0 of q needs to be 1, but now is %ld.",
                            tiling.get_b()),
                    return ge::GRAPH_FAILED);
        auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesTensor);
        OP_CHECK_IF(processor.VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        tiling.set_isVariable(true);
    }
    context->SetTilingKey(1);
    OP_LOGD(context->GetNodeName(), "tilingKey: %d", context->GetTilingKey());
    ChunkBwdDvLocalTilingDataPrint(context, tiling);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(tiling.get_totalCoreNum());

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t userWorkspaceSize = 2 * tiling.get_b() * tiling.get_h() * tiling.get_t() * tiling.get_chunkSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    OP_LOGD(context->GetNodeName(), "Tiling4ChunkBwdDvLocal end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkBwdDvLocal(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkBwdDvLocal)
    .Tiling(Tiling4ChunkBwdDvLocal)
    .TilingParse<ChunkBwdDvLocalCompileInfo>(TilingPrepareForChunkBwdDvLocal);

} // namespace optiling
