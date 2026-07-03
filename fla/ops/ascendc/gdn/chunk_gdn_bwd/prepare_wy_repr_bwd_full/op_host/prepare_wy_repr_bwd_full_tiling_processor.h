/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef PREPARE_WY_REPR_BWD_FULL_TILING_PROCESSOR_H
#define PREPARE_WY_REPR_BWD_FULL_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/prepare_wy_repr_bwd_full_struct.h"

using GDN::PrepareWyReprBwdFullTilingData;

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

static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;
static constexpr int64_t V_DIM_128 = 128;
static constexpr int64_t V_DIM_256 = 256;
static constexpr int64_t VAR_LEN_B_DIM_1 = 1;

static constexpr int64_t PREPARE_WY_REPR_BWD_FULL_DTYPE_FP16 = 0;
static constexpr int64_t PREPARE_WY_REPR_BWD_FULL_DTYPE_BF16 = 1;
static constexpr int64_t PREPARE_WY_REPR_BWD_FULL_DTYPE_FP32 = 2;

static constexpr const char *const INPUT_K_NAME = "k";
static constexpr const char *const INPUT_V_NAME = "v";
static constexpr const char *const INPUT_BETA_NAME = "beta";
static constexpr const char *const INPUT_A_NAME = "A";
static constexpr const char *const INPUT_DA_NAME = "dA";
static constexpr const char *const INPUT_DW_NAME = "dw";
static constexpr const char *const INPUT_DU_NAME = "du";
static constexpr const char *const INPUT_G_NAME = "g";
static constexpr const char *const INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const INPUT_SEQLENS_NAME = "cu_seqlens";

static constexpr uint64_t SIZE_HALF = 2;
static constexpr uint64_t SIZE_FP32 = 4;
static constexpr uint64_t ONE_BLOCK_32 = 32;
static constexpr uint64_t MAX_CUBE_VEC_SYNC_NUM = 5;

struct PrepareWyReprBwdFullTilingContext {
    const char *nodeName;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *aShape;
    const gert::StorageShape *dAShape;
    const gert::StorageShape *dwShape;
    const gert::StorageShape *duShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    const int64_t *cuSeqlensData;
    const int64_t *chunkIndicesData;
    int64_t chunkSize;
    int64_t kDataType;
    int64_t betaDataType;
    uint64_t ubSize;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class PrepareWyReprBwdFullTilingProcessor {
    PrepareWyReprBwdFullTilingContext &ctx_;
    PrepareWyReprBwdFullTilingData &tiling_;
    size_t workspaceSize_ = 0;
    uint32_t blockDim_ = 0;

public:
    explicit PrepareWyReprBwdFullTilingProcessor(PrepareWyReprBwdFullTilingContext &ctx,
                                                 PrepareWyReprBwdFullTilingData &tiling)
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

    uint32_t GetTilingKey() const
    {
        return 1U;
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

    ge::graphStatus PreCheck()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, DIM_NUM_4, INPUT_K_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, DIM_NUM_4, INPUT_V_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.betaShape, DIM_NUM_3, INPUT_BETA_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.aShape, DIM_NUM_4, INPUT_A_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dAShape, DIM_NUM_4, INPUT_DA_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dwShape, DIM_NUM_4, INPUT_DW_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.duShape, DIM_NUM_4, INPUT_DU_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, DIM_NUM_3, INPUT_G_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
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

    uint64_t AlignUp(uint64_t value, uint64_t align) const
    {
        return (value + align - 1) / align * align;
    }

    uint64_t DtypeSize(int64_t dtype) const
    {
        return dtype == PREPARE_WY_REPR_BWD_FULL_DTYPE_FP32 ? SIZE_FP32 : SIZE_HALF;
    }

    uint64_t GetFusedUseUbSize(uint64_t rowNum, uint64_t ubK, uint64_t ubV, uint64_t sizeofKType)
    {
        constexpr uint64_t SAFETY_MARGIN = 16 * 1024;
        uint64_t persistentBytes = 0;
        persistentBytes += AlignUp(tiling_.chunkSize * SIZE_FP32, ONE_BLOCK_32);
        persistentBytes += AlignUp(tiling_.chunkSize * SIZE_FP32, ONE_BLOCK_32);
        persistentBytes += AlignUp(tiling_.chunkSize * SIZE_FP32, ONE_BLOCK_32);
        persistentBytes += AlignUp(tiling_.chunkSize * SIZE_FP32, ONE_BLOCK_32);

        uint64_t queueBytes = 0;
        queueBytes += 2 * AlignUp(rowNum * ubK * sizeofKType, ONE_BLOCK_32);
        queueBytes += 2 * AlignUp(rowNum * ubV * sizeofKType, ONE_BLOCK_32);

        uint64_t group1TempBytes = 0;
        group1TempBytes += AlignUp(5 * rowNum * tiling_.K * SIZE_FP32, ONE_BLOCK_32);
        group1TempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32);
        group1TempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32);
        group1TempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32);

        uint64_t group2VTempBytes = 0;
        group2VTempBytes += AlignUp(2 * rowNum * tiling_.V * SIZE_FP32, ONE_BLOCK_32);
        group2VTempBytes += AlignUp(rowNum * ONE_BLOCK_32, ONE_BLOCK_32);

        uint64_t group2KktTempBytes = 0;
        group2KktTempBytes += AlignUp(tiling_.chunkSize * tiling_.chunkSize * SIZE_FP32, ONE_BLOCK_32);
        group2KktTempBytes += AlignUp(2 * rowNum * tiling_.chunkSize * SIZE_FP32, ONE_BLOCK_32);
        group2KktTempBytes += AlignUp(tiling_.chunkSize * ONE_BLOCK_32, ONE_BLOCK_32);

        uint64_t maxTempBytes = std::max(group1TempBytes, group2VTempBytes);
        maxTempBytes = std::max(maxTempBytes, group2KktTempBytes);
        return persistentBytes + queueBytes + maxTempBytes + SAFETY_MARGIN;
    }

    uint64_t GetFusedRowNum(uint64_t ubSize)
    {
        uint64_t rowNum = tiling_.chunkSize / 2;
        uint64_t sizeofKType = DtypeSize(ctx_.kDataType);
        uint64_t ubK = static_cast<uint64_t>(std::max(tiling_.K, tiling_.chunkSize));
        uint64_t ubV = static_cast<uint64_t>(std::max(tiling_.V, tiling_.chunkSize));
        while (rowNum >= 8) {
            uint64_t useUbSize = GetFusedUseUbSize(rowNum, ubK, ubV, sizeofKType);
            if (useUbSize <= ubSize) {
                break;
            }
            rowNum = rowNum / 2;
        }
        return rowNum;
    }

    ge::graphStatus SetFusedTiling()
    {
        uint64_t fusedRowNum = GetFusedRowNum(ctx_.ubSize);
        tiling_.fusedKVecRow = static_cast<int64_t>(fusedRowNum);
        tiling_.fusedVVecRow = static_cast<int64_t>(fusedRowNum);
        tiling_.fusedKktVecRow = static_cast<int64_t>(fusedRowNum);
        tiling_.workspaceBufferCount = 2;
        tiling_.usedCoreNum = static_cast<int64_t>(ctx_.aicCoreNum);

        uint64_t sizeofKType = DtypeSize(ctx_.kDataType);
        uint64_t maxKV = static_cast<uint64_t>(std::max(tiling_.K, tiling_.V));
        uint64_t slotSize = 0;
        slotSize += 3 * AlignUp(tiling_.chunkSize * tiling_.K * sizeofKType, ONE_BLOCK_32);
        slotSize += AlignUp(tiling_.chunkSize * maxKV * sizeofKType, ONE_BLOCK_32);
        slotSize += AlignUp(tiling_.chunkSize * tiling_.chunkSize * sizeofKType, ONE_BLOCK_32);
        tiling_.workspaceSlotSize = static_cast<int64_t>(slotSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape kStorageShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vStorageShape = ctx_.vShape->GetStorageShape();
        const gert::Shape betaStorageShape = ctx_.betaShape->GetStorageShape();
        const gert::Shape aStorageShape = ctx_.aShape->GetStorageShape();
        const gert::Shape dAStorageShape = ctx_.dAShape->GetStorageShape();
        const gert::Shape dwStorageShape = ctx_.dwShape->GetStorageShape();
        const gert::Shape duStorageShape = ctx_.duShape->GetStorageShape();
        const gert::Shape gStorageShape = ctx_.gShape->GetStorageShape();

        const int64_t hv = static_cast<int64_t>(vStorageShape.GetDim(DIM_1));
        const int64_t hk = static_cast<int64_t>(kStorageShape.GetDim(DIM_1));
        OP_CHECK_IF(vStorageShape.GetDim(DIM_0) != kStorageShape.GetDim(DIM_0),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: batch dim 0 must match (got %zu vs %zu).",
                            INPUT_V_NAME, INPUT_K_NAME, vStorageShape.GetDim(DIM_0), kStorageShape.GetDim(DIM_0)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(vStorageShape.GetDim(DIM_2) != kStorageShape.GetDim(DIM_2),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: time dim 2 must match (got %zu vs %zu).",
                            INPUT_V_NAME, INPUT_K_NAME, vStorageShape.GetDim(DIM_2), kStorageShape.GetDim(DIM_2)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hv <= 0 || hk <= 0,
                    OP_LOGE(ctx_.nodeName, "HV and HK must be positive (HV=%ld, HK=%ld).", hv, hk),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hv % hk != 0,
                    OP_LOGE(ctx_.nodeName,
                            "HV (%ld) must be a positive multiple of HK (%ld) for GQA (remainder %ld).", hv, hk,
                            hv % hk),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(vStorageShape, duStorageShape, INPUT_V_NAME, INPUT_DU_NAME, DIM_NUM_4) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(betaStorageShape, gStorageShape, INPUT_BETA_NAME, INPUT_G_NAME, DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != gStorageShape.GetDim(DIM_0),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: batch dim 0 must match (got %zu vs %zu).",
                            INPUT_K_NAME, INPUT_G_NAME, kStorageShape.GetDim(DIM_0), gStorageShape.GetDim(DIM_0)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(DIM_2) != gStorageShape.GetDim(DIM_2),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: time dim 2 must match (got %zu vs %zu).",
                            INPUT_K_NAME, INPUT_G_NAME, kStorageShape.GetDim(DIM_2), gStorageShape.GetDim(DIM_2)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(vStorageShape.GetDim(DIM_1) != gStorageShape.GetDim(DIM_1),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: head dim 1 must match HV (got %zu vs %zu).",
                            INPUT_V_NAME, INPUT_G_NAME, vStorageShape.GetDim(DIM_1), gStorageShape.GetDim(DIM_1)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(aStorageShape, dAStorageShape, INPUT_A_NAME, INPUT_DA_NAME, DIM_NUM_4) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CompareShape(dwStorageShape, vStorageShape, INPUT_DW_NAME, INPUT_V_NAME, DIM_NUM_3) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(dwStorageShape.GetDim(DIM_3) != kStorageShape.GetDim(DIM_3),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: last dim K must match (got %zu vs %zu).",
                            INPUT_DW_NAME, INPUT_K_NAME, dwStorageShape.GetDim(DIM_3), kStorageShape.GetDim(DIM_3)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(DIM_0) != aStorageShape.GetDim(DIM_0),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: batch dim 0 must match (got %zu vs %zu).",
                            INPUT_K_NAME, INPUT_A_NAME, kStorageShape.GetDim(DIM_0), aStorageShape.GetDim(DIM_0)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kStorageShape.GetDim(DIM_2) != aStorageShape.GetDim(DIM_2),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: time dim 2 must match (got %zu vs %zu).",
                            INPUT_K_NAME, INPUT_A_NAME, kStorageShape.GetDim(DIM_2), aStorageShape.GetDim(DIM_2)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(vStorageShape.GetDim(DIM_1) != aStorageShape.GetDim(DIM_1),
                    OP_LOGE(ctx_.nodeName,
                            "Compare input shape of %s and %s failed: head dim 1 must match HV (got %zu vs %zu).",
                            INPUT_V_NAME, INPUT_A_NAME, vStorageShape.GetDim(DIM_1), aStorageShape.GetDim(DIM_1)),
                    return ge::GRAPH_FAILED);

        tiling_.B = static_cast<int64_t>(vStorageShape.GetDim(DIM_0));
        tiling_.HV = hv;
        tiling_.HK = hk;
        tiling_.T = static_cast<int64_t>(vStorageShape.GetDim(DIM_2));
        tiling_.K = static_cast<int64_t>(kStorageShape.GetDim(DIM_3));
        tiling_.V = static_cast<int64_t>(vStorageShape.GetDim(DIM_3));

        OP_CHECK_IF(tiling_.V != V_DIM_128 && tiling_.V != V_DIM_256,
                    OP_LOGE(ctx_.nodeName,
                            "Check value dim V failed: only %ld or %ld is supported, but get %ld.", V_DIM_128,
                            V_DIM_256, tiling_.V),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkSize != CHUNK_SIZE_64 && ctx_.chunkSize != CHUNK_SIZE_128,
                    OP_LOGE(ctx_.nodeName,
                            "Check attr chunkSize failed, the chunkSize should be 64 or 128, but get %ld.",
                            ctx_.chunkSize),
                    return ge::GRAPH_FAILED);
        tiling_.chunkSize = ctx_.chunkSize;

        OP_CHECK_IF(SetFusedTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(ctx_.nodeName, "SetFusedTiling Failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    int64_t CeilDiv(int64_t a, int64_t b) const
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    ge::graphStatus FixLenTiling()
    {
        tiling_.chunkNum = tiling_.B * CeilDiv(tiling_.T, tiling_.chunkSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.chunkIndicesShape, DIM_1, INPUT_CHUNK_INDICES_NAME) !=
                        ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, DIM_1, INPUT_SEQLENS_NAME) != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s data is required for variable-length tiling, but got nullptr.",
                            INPUT_SEQLENS_NAME),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkIndicesData == nullptr,
                    OP_LOGE(ctx_.nodeName, "Input %s data is required for variable-length tiling, but got nullptr.",
                            INPUT_CHUNK_INDICES_NAME),
                    return ge::GRAPH_FAILED);

        const gert::Shape seqlensStorageShape = ctx_.cuSeqlensShape->GetStorageShape();
        int64_t seqlensDim0 = seqlensStorageShape.GetDim(DIM_0);
        OP_CHECK_IF(seqlensDim0 < 2,
                    OP_LOGE(ctx_.nodeName,
                            "Check seqlens shape failed, the dim 0 of seqlens should be larger than 1, but get %ld.",
                            seqlensDim0),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.cuSeqlensData[0] != 0,
                    OP_LOGE(ctx_.nodeName, "Check seqlens data failed, the seqlens[0] should be 0, but get %ld.",
                            ctx_.cuSeqlensData[0]),
                    return ge::GRAPH_FAILED);

        std::vector<int64_t> expectChunkIndices;
        for (int64_t i = 1; i < seqlensDim0; i++) {
            int64_t curSeqLen = ctx_.cuSeqlensData[i] - ctx_.cuSeqlensData[i - 1];
            OP_CHECK_IF(curSeqLen <= 0,
                        OP_LOGE(ctx_.nodeName,
                                "Check seqlens data failed, the seqlens[%ld]:[%ld] should be larger than seqlens[%ld]:[%ld]",
                                i, ctx_.cuSeqlensData[i], i - 1, ctx_.cuSeqlensData[i - 1]),
                        return ge::GRAPH_FAILED);
            for (int64_t j = 0; j < curSeqLen; j += tiling_.chunkSize) {
                expectChunkIndices.push_back(i - 1);
                expectChunkIndices.push_back(j / tiling_.chunkSize);
            }
        }

        const gert::Shape chunkIndicesStorageShape = ctx_.chunkIndicesShape->GetStorageShape();
        int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(DIM_0);
        OP_CHECK_IF(chunkIndicesDim0 != static_cast<int64_t>(expectChunkIndices.size()),
                    OP_LOGE(ctx_.nodeName,
                            "Check chunk_indices shape failed, the len of chunk_indices should be %zu, but get %ld.",
                            expectChunkIndices.size(), chunkIndicesDim0),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < static_cast<int64_t>(expectChunkIndices.size()); i++) {
            OP_CHECK_IF(expectChunkIndices[i] != ctx_.chunkIndicesData[i],
                        OP_LOGE(ctx_.nodeName,
                                "Check chunk_indices data failed, the chunk_indices[%ld] should be %ld, but get %ld.",
                                i, expectChunkIndices[i], ctx_.chunkIndicesData[i]),
                        return ge::GRAPH_FAILED);
        }
        tiling_.chunkNum = chunkIndicesStorageShape.GetDim(DIM_0) / 2;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        blockDim_ = ctx_.aicCoreNum;
        uint64_t userWorkspaceSize = static_cast<uint64_t>(tiling_.usedCoreNum) *
                                     static_cast<uint64_t>(tiling_.workspaceBufferCount) *
                                     static_cast<uint64_t>(tiling_.workspaceSlotSize);
        workspaceSize_ = ctx_.sysWorkspaceSize + userWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(PreCheck() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CommonTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        if (IsVariableLength()) {
            OP_CHECK_IF(ctx_.cuSeqlensShape == nullptr || ctx_.chunkIndicesShape == nullptr,
                        OP_LOGE(ctx_.nodeName,
                                "Variable-length tiling requires both cu_seqlens and chunk_indices to be provided."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(tiling_.B != VAR_LEN_B_DIM_1,
                        OP_LOGE(ctx_.nodeName,
                                "If cu_seqlens is not nullptr, the dim 0 of q needs to be 1, but now is %ld.",
                                tiling_.B),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 1;
        } else {
            OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVariable = 0;
        }
        OP_CHECK_IF(WorkspaceTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // PREPARE_WY_REPR_BWD_FULL_TILING_PROCESSOR_H
