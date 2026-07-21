/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef PREPARE_WY_REPR_BWD_TILING_PROCESSOR_H
#define PREPARE_WY_REPR_BWD_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/prepare_wy_repr_bwd_struct.h"
#include "../op_kernel/prepare_wy_repr_bwd_tiling_key.h"

using GDN::PrepareWyReprBwdTilingData;

namespace optiling {

static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_K_IDX = 0;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_V_IDX = 1;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_BETA_IDX = 2;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_A_IDX = 3;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_DW_IDX = 4;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_DU_IDX = 5;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_G_IDX = 6;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_SEQLENS_IDX = 7;
static constexpr size_t PREPARE_WY_REPR_BWD_INPUT_CHUNK_INDICES_IDX = 8;

static constexpr size_t PREPARE_WY_REPR_BWD_ATTR_CHUNK_SIZE_IDX = 0;

static constexpr size_t PREPARE_WY_REPR_BWD_DIM_NUM_1 = 1;
static constexpr size_t PREPARE_WY_REPR_BWD_DIM_NUM_3 = 3;
static constexpr size_t PREPARE_WY_REPR_BWD_DIM_NUM_4 = 4;

static constexpr size_t PREPARE_WY_REPR_BWD_DIM_0 = 0;
static constexpr size_t PREPARE_WY_REPR_BWD_DIM_1 = 1;
static constexpr size_t PREPARE_WY_REPR_BWD_DIM_2 = 2;
static constexpr size_t PREPARE_WY_REPR_BWD_DIM_3 = 3;

static constexpr int64_t PREPARE_WY_REPR_BWD_CHUNK_SIZE_64 = 64;
static constexpr int64_t PREPARE_WY_REPR_BWD_CHUNK_SIZE_128 = 128;
static constexpr int64_t K_DIM_128 = 128;
static constexpr int64_t PREPARE_WY_REPR_BWD_V_DIM_128 = 128;
static constexpr int64_t PREPARE_WY_REPR_BWD_V_DIM_256 = 256;
static constexpr int64_t PREPARE_WY_REPR_BWD_VAR_LEN_B_DIM_1 = 1;

static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_K_NAME = "k";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_V_NAME = "v";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_BETA_NAME = "beta";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_A_NAME = "A";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_DW_NAME = "dw";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_DU_NAME = "du";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_G_NAME = "g";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const PREPARE_WY_REPR_BWD_INPUT_SEQLENS_NAME = "cu_seqlens";

static constexpr uint64_t PREPARE_WY_REPR_BWD_SIZE_HALF = 2;
static constexpr uint64_t PREPARE_WY_REPR_BWD_SIZE_FP32 = 4;
static constexpr uint64_t PREPARE_WY_REPR_BWD_ONE_BLOCK_32 = 32;
static constexpr uint64_t PREPARE_WY_REPR_BWD_UB_IO_BYTES = 16 * 1024;
static constexpr uint64_t PREPARE_WY_REPR_BWD_FIXED_IO_BYTES = 4 * PREPARE_WY_REPR_BWD_UB_IO_BYTES;
static constexpr uint64_t PREPARE_WY_REPR_BWD_VEC_COMPUTE_BYTES = 128 * 1024;
static constexpr uint64_t PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT = 2;

struct PrepareWyReprBwdTilingContext {
    const char *nodeName;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *aShape;
    const gert::StorageShape *dwShape;
    const gert::StorageShape *duShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    const int64_t *cuSeqlensData;
    const int64_t *chunkIndicesData;
    int64_t chunkSize;
    ge::DataType kDataType;
    ge::DataType gDataType;
    ge::DataType betaDataType;
    uint64_t ubSize;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class PrepareWyReprBwdTilingProcessor {
    PrepareWyReprBwdTilingContext &ctx_;
    PrepareWyReprBwdTilingData &tiling_;
    size_t workspaceSize_ = 0;
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;

public:
    explicit PrepareWyReprBwdTilingProcessor(PrepareWyReprBwdTilingContext &ctx, PrepareWyReprBwdTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

    uint32_t GetBlockDim() const
    {
        return blockDim_;
    }

    uint64_t GetTilingKey() const
    {
        return tilingKey_;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr || ctx_.chunkIndicesShape != nullptr;
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
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                 const char *inputName2, size_t compareDimNum)
    {
        for (size_t dimIndex = 0; dimIndex < compareDimNum; dimIndex++) {
            size_t shapeDim1 = shape1.GetDim(dimIndex);
            size_t shapeDim2 = shape2.GetDim(dimIndex);
            OP_CHECK_IF(shapeDim1 != shapeDim2,
                        OP_LOGE(ctx_.nodeName,
                                "Compare input shape of %s and %s failed, the length of dim %zu should be same, but "
                                "got %zu and %zu.",
                                inputName1, inputName2, dimIndex, shapeDim1, shapeDim2),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, PREPARE_WY_REPR_BWD_DIM_NUM_4,
                                             PREPARE_WY_REPR_BWD_INPUT_K_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, PREPARE_WY_REPR_BWD_DIM_NUM_4,
                                             PREPARE_WY_REPR_BWD_INPUT_V_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.betaShape, PREPARE_WY_REPR_BWD_DIM_NUM_3,
                                             PREPARE_WY_REPR_BWD_INPUT_BETA_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.aShape, PREPARE_WY_REPR_BWD_DIM_NUM_4,
                                             PREPARE_WY_REPR_BWD_INPUT_A_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dwShape, PREPARE_WY_REPR_BWD_DIM_NUM_4,
                                             PREPARE_WY_REPR_BWD_INPUT_DW_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.duShape, PREPARE_WY_REPR_BWD_DIM_NUM_4,
                                             PREPARE_WY_REPR_BWD_INPUT_DU_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, PREPARE_WY_REPR_BWD_DIM_NUM_3,
                                             PREPARE_WY_REPR_BWD_INPUT_G_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.betaDataType != ctx_.gDataType,
                    OP_LOGE(ctx_.nodeName, "beta and g dtype must be same in first fused prepare_wy_repr_bwd."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    uint64_t AlignUp(uint64_t value, uint64_t align) const
    {
        return (value + align - 1) / align * align;
    }

    uint64_t CeilDiv(uint64_t value, uint64_t divisor) const
    {
        if (divisor == 0) {
            return 0;
        }
        return (value + divisor - 1) / divisor;
    }

    uint64_t DtypeSize(ge::DataType dtype) const
    {
        return dtype == ge::DT_FLOAT ? PREPARE_WY_REPR_BWD_SIZE_FP32 : PREPARE_WY_REPR_BWD_SIZE_HALF;
    }

    uint32_t DtypeKey(ge::DataType dtype) const
    {
        if (dtype == ge::DT_BF16) {
            return TPL_BF16;
        }
        if (dtype == ge::DT_FLOAT) {
            return TPL_FP32;
        }
        return TPL_FP16;
    }

    ge::graphStatus SetTilingKey()
    {
        OP_CHECK_IF(ctx_.kDataType != ge::DT_FLOAT16 && ctx_.kDataType != ge::DT_BF16,
                    OP_LOGE(ctx_.nodeName, "k dtype only supports fp16/bf16."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.gDataType != ge::DT_FLOAT16 && ctx_.gDataType != ge::DT_BF16 &&
                        ctx_.gDataType != ge::DT_FLOAT,
                    OP_LOGE(ctx_.nodeName, "g dtype only supports fp16/bf16/fp32."), return ge::GRAPH_FAILED);

        uint32_t kTypeKey = DtypeKey(ctx_.kDataType);
        uint32_t gTypeKey = DtypeKey(ctx_.gDataType);
        tilingKey_ = GET_TPL_TILING_KEY(kTypeKey, gTypeKey, static_cast<uint32_t>(tiling_.V),
                                        static_cast<uint32_t>(tiling_.chunkSize));
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape kShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vShape = ctx_.vShape->GetStorageShape();
        const gert::Shape betaShape = ctx_.betaShape->GetStorageShape();
        const gert::Shape aShape = ctx_.aShape->GetStorageShape();
        const gert::Shape dwShape = ctx_.dwShape->GetStorageShape();
        const gert::Shape duShape = ctx_.duShape->GetStorageShape();
        const gert::Shape gShape = ctx_.gShape->GetStorageShape();

        OP_CHECK_IF(CompareShape(vShape, duShape, PREPARE_WY_REPR_BWD_INPUT_V_NAME,
                                 PREPARE_WY_REPR_BWD_INPUT_DU_NAME,
                                 PREPARE_WY_REPR_BWD_DIM_NUM_4) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(betaShape, gShape, PREPARE_WY_REPR_BWD_INPUT_BETA_NAME,
                                 PREPARE_WY_REPR_BWD_INPUT_G_NAME,
                                 PREPARE_WY_REPR_BWD_DIM_NUM_3) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(dwShape, vShape, PREPARE_WY_REPR_BWD_INPUT_DW_NAME,
                                 PREPARE_WY_REPR_BWD_INPUT_V_NAME, PREPARE_WY_REPR_BWD_DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0) != vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0) ||
                        kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_2) != vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_2),
                    OP_LOGE(ctx_.nodeName, "k/v batch and time dims must match."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0) != aShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0) ||
                        kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_2) != aShape.GetDim(PREPARE_WY_REPR_BWD_DIM_2),
                    OP_LOGE(ctx_.nodeName, "k/A batch and time dims must match."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_1) != aShape.GetDim(PREPARE_WY_REPR_BWD_DIM_1) ||
                        vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_1) != betaShape.GetDim(PREPARE_WY_REPR_BWD_DIM_1),
                    OP_LOGE(ctx_.nodeName, "v/A/beta head dims must match HV."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0) != betaShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0) ||
                        kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_2) != betaShape.GetDim(PREPARE_WY_REPR_BWD_DIM_2),
                    OP_LOGE(ctx_.nodeName, "k/beta batch and time dims must match."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(dwShape.GetDim(PREPARE_WY_REPR_BWD_DIM_3) != kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_3),
                    OP_LOGE(ctx_.nodeName, "dw last dim must match k last dim K."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(aShape.GetDim(PREPARE_WY_REPR_BWD_DIM_3) != static_cast<size_t>(ctx_.chunkSize),
                    OP_LOGE(ctx_.nodeName, "A last dim must equal chunk_size."), return ge::GRAPH_FAILED);

        tiling_.B = static_cast<int64_t>(vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0));
        tiling_.HV = static_cast<int64_t>(vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_1));
        tiling_.HK = static_cast<int64_t>(kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_1));
        tiling_.T = static_cast<int64_t>(vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_2));
        tiling_.K = static_cast<int64_t>(kShape.GetDim(PREPARE_WY_REPR_BWD_DIM_3));
        tiling_.V = static_cast<int64_t>(vShape.GetDim(PREPARE_WY_REPR_BWD_DIM_3));
        tiling_.chunkSize = ctx_.chunkSize;

        OP_CHECK_IF(tiling_.K != K_DIM_128,
                    OP_LOGE(ctx_.nodeName, "K must be 128, but got %ld.", tiling_.K), return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.V != PREPARE_WY_REPR_BWD_V_DIM_128 &&
                        tiling_.V != PREPARE_WY_REPR_BWD_V_DIM_256,
                    OP_LOGE(ctx_.nodeName, "V must be 128 or 256, but got %ld.", tiling_.V),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.chunkSize != PREPARE_WY_REPR_BWD_CHUNK_SIZE_64 &&
                        tiling_.chunkSize != PREPARE_WY_REPR_BWD_CHUNK_SIZE_128,
                    OP_LOGE(ctx_.nodeName, "chunk_size must be 64 or 128, but got %ld.", tiling_.chunkSize),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.HV <= 0 || tiling_.HK <= 0 || tiling_.HV % tiling_.HK != 0,
                    OP_LOGE(ctx_.nodeName, "HV must be a positive multiple of HK, got HV=%ld, HK=%ld.", tiling_.HV,
                            tiling_.HK),
                    return ge::GRAPH_FAILED);
        tiling_.groupSize = tiling_.HV / tiling_.HK;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ChunkTiling()
    {
        if (!IsVariableLength()) {
            tiling_.isVariable = 0;
            tiling_.chunkNumPerB = static_cast<int64_t>(CeilDiv(static_cast<uint64_t>(tiling_.T),
                                                                static_cast<uint64_t>(tiling_.chunkSize)));
            tiling_.chunkNum = tiling_.B * tiling_.chunkNumPerB;
            return ge::GRAPH_SUCCESS;
        }

        OP_CHECK_IF(ctx_.cuSeqlensShape == nullptr || ctx_.chunkIndicesShape == nullptr,
                    OP_LOGE(ctx_.nodeName,
                            "Variable-length tiling requires both cu_seqlens and chunk_indices to be provided."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(tiling_.B != PREPARE_WY_REPR_BWD_VAR_LEN_B_DIM_1,
                    OP_LOGE(ctx_.nodeName, "Variable-length mode requires B=1, but got %ld.", tiling_.B),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, PREPARE_WY_REPR_BWD_DIM_NUM_1,
                                             PREPARE_WY_REPR_BWD_INPUT_SEQLENS_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.chunkIndicesShape, PREPARE_WY_REPR_BWD_DIM_NUM_1,
                                             PREPARE_WY_REPR_BWD_INPUT_CHUNK_INDICES_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData == nullptr || ctx_.chunkIndicesData == nullptr,
                    OP_LOGE(ctx_.nodeName, "cu_seqlens and chunk_indices const data are required for tiling."),
                    return ge::GRAPH_FAILED);

        const gert::Shape chunkIndicesShape = ctx_.chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim0 = static_cast<int64_t>(chunkIndicesShape.GetDim(PREPARE_WY_REPR_BWD_DIM_0));
        OP_CHECK_IF(chunkIndicesDim0 <= 0 || (chunkIndicesDim0 % 2) != 0,
                    OP_LOGE(ctx_.nodeName, "chunk_indices length must be positive even, but got %ld.",
                            chunkIndicesDim0),
                    return ge::GRAPH_FAILED);
        tiling_.isVariable = 1;
        tiling_.chunkNumPerB = 0;
        tiling_.chunkNum = chunkIndicesDim0 / 2;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VectorRowTiling()
    {
        uint64_t kTypeBytes = DtypeSize(ctx_.kDataType);
        uint64_t gTypeBytes = DtypeSize(ctx_.gDataType);
        uint64_t computeBytes = ctx_.ubSize > PREPARE_WY_REPR_BWD_FIXED_IO_BYTES
                                    ? ctx_.ubSize - PREPARE_WY_REPR_BWD_FIXED_IO_BYTES
                                    : PREPARE_WY_REPR_BWD_VEC_COMPUTE_BYTES;
        computeBytes = std::min(computeBytes, PREPARE_WY_REPR_BWD_VEC_COMPUTE_BYTES);

        auto fitRow = [&](uint64_t cols, uint64_t tempFactor) -> int64_t {
            uint64_t row = static_cast<uint64_t>(tiling_.chunkSize) / 2;
            while (row >= 8) {
                uint64_t matrixBytes = row * cols * kTypeBytes;
                uint64_t scaleBytes = row * gTypeBytes;
                uint64_t brcbBytes = row * PREPARE_WY_REPR_BWD_ONE_BLOCK_32;
                uint64_t tempBytes = tempFactor * row * cols * PREPARE_WY_REPR_BWD_SIZE_FP32 +
                                     4 * AlignUp(row * PREPARE_WY_REPR_BWD_SIZE_FP32,
                                                 PREPARE_WY_REPR_BWD_ONE_BLOCK_32) +
                                     brcbBytes + 2 * AlignUp(scaleBytes, PREPARE_WY_REPR_BWD_ONE_BLOCK_32);
                if (matrixBytes <= PREPARE_WY_REPR_BWD_UB_IO_BYTES &&
                    tempBytes <= computeBytes) {
                    break;
                }
                row = row / 2;
            }
            return static_cast<int64_t>(row);
        };

        tiling_.kVecRow = fitRow(static_cast<uint64_t>(tiling_.K), 2);
        tiling_.vVecRow = fitRow(static_cast<uint64_t>(tiling_.V), 2);
        tiling_.mVecRow = fitRow(static_cast<uint64_t>(tiling_.chunkSize), 4);
        tiling_.kktVecRow = fitRow(static_cast<uint64_t>(tiling_.chunkSize), 3);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        uint64_t kTypeBytes = DtypeSize(ctx_.kDataType);
        tiling_.workspaceBufferCount = static_cast<int64_t>(PREPARE_WY_REPR_BWD_WORKSPACE_BUFFER_COUNT);
        tiling_.kBytes = static_cast<int64_t>(
            AlignUp(static_cast<uint64_t>(tiling_.chunkSize * tiling_.K) * kTypeBytes,
                    PREPARE_WY_REPR_BWD_ONE_BLOCK_32));
        tiling_.vBytes = static_cast<int64_t>(
            AlignUp(static_cast<uint64_t>(tiling_.chunkSize * tiling_.V) * kTypeBytes,
                    PREPARE_WY_REPR_BWD_ONE_BLOCK_32));
        tiling_.mBytes = static_cast<int64_t>(
            AlignUp(static_cast<uint64_t>(tiling_.chunkSize * tiling_.chunkSize) * kTypeBytes,
                    PREPARE_WY_REPR_BWD_ONE_BLOCK_32));

        int64_t offset = 0;
        tiling_.kbgOffset = offset;
        offset += tiling_.kBytes;
        tiling_.vbOffset = offset;
        offset += tiling_.vBytes;
        tiling_.kbetaOffset = offset;
        offset += tiling_.kBytes;
        tiling_.dkbgOffset = offset;
        offset += tiling_.kBytes;
        tiling_.dvbOffset = offset;
        offset += tiling_.vBytes;
        tiling_.da1Offset = offset;
        offset += tiling_.mBytes;
        tiling_.da2Offset = offset;
        offset += tiling_.mBytes;
        tiling_.da4Offset = offset;
        offset += tiling_.mBytes;
        tiling_.da5Offset = offset;
        offset += tiling_.mBytes;
        tiling_.da6Offset = offset;
        offset += tiling_.mBytes;
        tiling_.dOffset = offset;
        offset += tiling_.mBytes;
        tiling_.dkbOffset = offset;
        offset += tiling_.kBytes;
        tiling_.dkOffset = offset;
        offset += tiling_.kBytes;

        tiling_.workspaceSlotSize =
            static_cast<int64_t>(AlignUp(static_cast<uint64_t>(offset), PREPARE_WY_REPR_BWD_ONE_BLOCK_32));
        tiling_.kktOffset = tiling_.workspaceBufferCount * tiling_.workspaceSlotSize;
        tiling_.workspaceCoreSize = tiling_.kktOffset + tiling_.workspaceBufferCount * tiling_.mBytes;
        blockDim_ = ctx_.aicCoreNum;

        uint64_t userWorkspaceSize = static_cast<uint64_t>(blockDim_) *
                                     static_cast<uint64_t>(tiling_.workspaceCoreSize);
        workspaceSize_ = ctx_.sysWorkspaceSize + userWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(ChunkTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(VectorRowTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(WorkspaceTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetTilingKey() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // PREPARE_WY_REPR_BWD_TILING_PROCESSOR_H
