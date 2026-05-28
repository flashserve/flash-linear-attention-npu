/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local_tiling_processor.h
 * \brief Tiling processor decoupled from gert::TilingContext, reusable in both aclnn and kernel launch modes.
 */

#ifndef CHUNK_BWD_DV_LOCAL_TILING_PROCESSOR_H
#define CHUNK_BWD_DV_LOCAL_TILING_PROCESSOR_H

#include "../op_kernel/chunk_bwd_dv_local_struct.h"
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

using GDN::ChunkBwdDvLocalTilingData;

namespace optiling {

static constexpr size_t CHUNK_BWD_DV_LOCAL_INPUT_Q_IDX = 0;
static constexpr size_t CHUNK_BWD_DV_LOCAL_INPUT_K_IDX = 1;
static constexpr size_t CHUNK_BWD_DV_LOCAL_INPUT_DO_IDX = 2;
static constexpr size_t CHUNK_BWD_DV_LOCAL_INPUT_G_IDX = 3;
static constexpr size_t CHUNK_BWD_DV_LOCAL_INPUT_SEQLENS_IDX = 6;
static constexpr size_t CHUNK_BWD_DV_LOCAL_INPUT_CHUNK_INDICES_IDX = 7;
static constexpr size_t CHUNK_BWD_DV_LOCAL_ATTR_SCALE_IDX = 0;
static constexpr size_t CHUNK_BWD_DV_LOCAL_ATTR_CHUNK_SIZE_IDX = 1;

static constexpr size_t Q_K_DO_DIM_NUM = 4;
static constexpr size_t G_DIM_NUM = 3;
static constexpr size_t SEQLENS_DIM_NUM = 1;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;

static constexpr uint32_t QKV_DTYPE_SIZE = 2;

static constexpr int64_t V_L_B = 1;
static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;
static constexpr int64_t CHUNK_INDICES_DIM_1_SIZE = 2;

static constexpr int64_t K_SIZE_128 = 128;
static constexpr int64_t V_SIZE_128 = 128;
static constexpr int64_t V_SIZE_256 = 256;

static constexpr const char *const INPUT_Q_NAME = "q";
static constexpr const char *const INPUT_K_NAME = "k";
static constexpr const char *const INPUT_DO_NAME = "do";
static constexpr const char *const INPUT_G_NAME = "g";
static constexpr const char *const INPUT_TRI_MATRIX_NAME = "upper_tri_matrix";
static constexpr const char *const INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const INPUT_SEQLENS_NAME = "cu_seqlens";

struct ChunkBwdDvLocalTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *dOShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    double scale;
    int32_t chunkSize;
};

class ChunkBwdDvLocalTilingProcessor {
    ChunkBwdDvLocalTilingContext &ctx_;
    ChunkBwdDvLocalTilingData &tiling_;

public:
    explicit ChunkBwdDvLocalTilingProcessor(ChunkBwdDvLocalTilingContext &ctx, ChunkBwdDvLocalTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                             const char *inputName)
    {
        OP_CHECK_IF(curShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", inputName),
                    return ge::GRAPH_FAILED);
        const gert::Shape storageShape = curShape->GetStorageShape();
        size_t dimNum = storageShape.GetDimNum();
        OP_CHECK_IF(dimNum != validDimNum,
                    OP_LOGE(ctx_.nodeName,
                            "Check input %s shape failed, the dim num should be %zu, but get %zu.", inputName,
                            validDimNum, dimNum),
                    return ge::GRAPH_FAILED);
        for (size_t dimIndex = 0; dimIndex < dimNum; dimIndex++) {
            OP_CHECK_IF(storageShape.GetDim(dimIndex) == 0,
                        OP_LOGE(ctx_.nodeName,
                                "Check input %s shape failed, the dim %zu should be non-zero, but get 0.", inputName,
                                dimIndex),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.qShape, Q_K_DO_DIM_NUM,
                                             INPUT_Q_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, Q_K_DO_DIM_NUM,
                                             INPUT_K_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dOShape, Q_K_DO_DIM_NUM,
                                             INPUT_DO_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, G_DIM_NUM, INPUT_G_NAME) !=
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
                        OP_LOGE(ctx_.nodeName,
                                "Compare input shape of %s and %s failed, the length of dim %zu should be same,but got "
                                "%zu and %zu.",
                                inputName1, inputName2, dimIndex, shapeDim1, shapeDim2),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape qStorageShape = ctx_.qShape->GetStorageShape();
        const gert::Shape kStorageShape = ctx_.kShape->GetStorageShape();
        const gert::Shape dOStorageShape = ctx_.dOShape->GetStorageShape();
        const gert::Shape gStorageShape = ctx_.gShape->GetStorageShape();
        OP_CHECK_IF(CompareShape(qStorageShape, kStorageShape, INPUT_Q_NAME, INPUT_K_NAME, Q_K_DO_DIM_NUM) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(qStorageShape, dOStorageShape, INPUT_Q_NAME, INPUT_DO_NAME, G_DIM_NUM) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(qStorageShape, gStorageShape, INPUT_Q_NAME, INPUT_G_NAME, G_DIM_NUM) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        tiling_.b = static_cast<int64_t>(qStorageShape.GetDim(DIM_0));
        tiling_.h = static_cast<int64_t>(qStorageShape.GetDim(DIM_1));
        tiling_.t = static_cast<int64_t>(qStorageShape.GetDim(DIM_2));
        tiling_.k = static_cast<int64_t>(qStorageShape.GetDim(DIM_3));
        tiling_.v = static_cast<int64_t>(dOStorageShape.GetDim(DIM_3));

        OP_LOGI(ctx_.nodeName, "=== K/V dimension check: K=%ld, V=%ld", tiling_.k, tiling_.v);
        OP_CHECK_IF(tiling_.k != K_SIZE_128,
                    OP_LOGE(ctx_.nodeName,
                            "Check input k shape failed, the k dimension should be 128, but get %ld.", tiling_.k),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.v != V_SIZE_128 && tiling_.v != V_SIZE_256,
                    OP_LOGE(ctx_.nodeName,
                            "Check input v shape failed, the v dimension should be 128 or 256, but get %ld.", tiling_.v),
                    return ge::GRAPH_FAILED);

        tiling_.scale = static_cast<float>(ctx_.scale);
        int64_t chunkSize = static_cast<int64_t>(ctx_.chunkSize);
        OP_CHECK_IF(chunkSize != CHUNK_SIZE_64 && chunkSize != CHUNK_SIZE_128,
                    OP_LOGE(ctx_.nodeName,
                            "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.", chunkSize),
                    return ge::GRAPH_FAILED);
        tiling_.chunkSize = chunkSize;
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
        tiling_.chunkNumForT = CeilDiv(tiling_.t, tiling_.chunkSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        OP_CHECK_IF(ctx_.cuSeqlensShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", INPUT_SEQLENS_NAME),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, SEQLENS_DIM_NUM, INPUT_SEQLENS_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);

        OP_CHECK_IF(ctx_.chunkIndicesShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", INPUT_CHUNK_INDICES_NAME),
                    return ge::GRAPH_FAILED);
        const gert::Shape chunkIndicesStorageShape = ctx_.chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(DIM_0);
        OP_CHECK_IF(chunkIndicesDim0 % CHUNK_INDICES_DIM_1_SIZE != 0,
                    OP_LOGE(ctx_.nodeName,
                            "Check chunk_indices shape failed, the dim 0 of chunk_indices needs to be divisible by 2, but get %ld.",
                            chunkIndicesDim0),
                    return ge::GRAPH_FAILED);

        tiling_.chunkNumForT = chunkIndicesDim0 / CHUNK_INDICES_DIM_1_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        if (IsVariableLength()) {
            OP_CHECK_IF(tiling_.b != V_L_B,
                        OP_LOGE(ctx_.nodeName, "B cannot be 1 when cu_seqlens is nullptr."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // CHUNK_BWD_DV_LOCAL_TILING_PROCESSOR_H
