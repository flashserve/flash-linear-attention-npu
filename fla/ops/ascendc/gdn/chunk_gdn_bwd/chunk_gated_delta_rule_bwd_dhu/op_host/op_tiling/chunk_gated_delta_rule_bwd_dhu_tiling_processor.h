/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_tiling_processor.h
 * \brief Tiling processor shared by aclnn tiling and fast kernel launch.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_PROCESSOR_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../../op_kernel/chunk_gated_delta_rule_bwd_dhu_struct.h"

using GDN::ChunkGatedDeltaRuleBwdDhuTilingData;

namespace optiling {

static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_NUM_4 = 4;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;

static constexpr int64_t K_SIZE_128 = 128;
static constexpr int64_t V_SIZE_128 = 128;
static constexpr int64_t V_SIZE_256 = 256;
static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;
static constexpr int64_t CHUNK_INDICES_PAIR = 2;
static constexpr int64_t VAR_LEN_B = 1;
static constexpr int64_t HEADS_PER_TASK = 4;
static constexpr int64_t WORKSPACE_BUFFER_COUNT = 8;
static constexpr uint64_t VECTOR_SUB_BLOCK_NUM = 2;
static constexpr uint64_t DTYPE_SIZE_HALF = 2;
static constexpr uint64_t DTYPE_SIZE_FLOAT = 4;
static constexpr uint64_t ALIGN_BYTES_32 = 32;
static constexpr uint64_t ALIGN_BYTES_512 = 512;
static constexpr uint64_t UB_GUARD_BYTES = 16 * 1024;

static constexpr const char *const CGDR_BWD_DHU_INPUT_Q_NAME = "q";
static constexpr const char *const CGDR_BWD_DHU_INPUT_K_NAME = "k";
static constexpr const char *const CGDR_BWD_DHU_INPUT_W_NAME = "w";
static constexpr const char *const CGDR_BWD_DHU_INPUT_DO_NAME = "d_o";
static constexpr const char *const CGDR_BWD_DHU_INPUT_DV_NAME = "dv";
static constexpr const char *const CGDR_BWD_DHU_INPUT_G_NAME = "g";
static constexpr const char *const CGDR_BWD_DHU_INPUT_GK_NAME = "gk";
static constexpr const char *const CGDR_BWD_DHU_INPUT_SEQLENS_NAME = "cu_seqlens";
static constexpr const char *const CGDR_BWD_DHU_INPUT_CHUNK_INDICES_NAME = "chunk_indices";

struct ChunkGatedDeltaRuleBwdDhuTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *wShape;
    const gert::StorageShape *dOShape;
    const gert::StorageShape *dvShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *gkShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    ge::DataType qDataType;
    ge::DataType gDataType;
    bool hasG;
    bool hasGk;
    bool hasDh0;
    bool stage0Debug;
    double scale;
    int32_t chunkSize;
    uint64_t ubSize;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class ChunkGatedDeltaRuleBwdDhuTilingProcessor {
public:
    explicit ChunkGatedDeltaRuleBwdDhuTilingProcessor(ChunkGatedDeltaRuleBwdDhuTilingContext &ctx,
                                                       ChunkGatedDeltaRuleBwdDhuTilingData &tiling)
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
        return tilingKey_;
    }

    bool IsVariableLength() const
    {
        return ctx_.cuSeqlensShape != nullptr || ctx_.chunkIndicesShape != nullptr;
    }

    ge::graphStatus Process()
    {
        if (PreCheck() != ge::GRAPH_SUCCESS) {
            OP_LOGE(ctx_.nodeName, "ChunkGatedDeltaRuleBwdDhu PreCheck failed.");
            return ge::GRAPH_FAILED;
        }
        if (CommonTiling() != ge::GRAPH_SUCCESS) {
            OP_LOGE(ctx_.nodeName, "ChunkGatedDeltaRuleBwdDhu CommonTiling failed.");
            return ge::GRAPH_FAILED;
        }
        if (IsVariableLength()) {
            if (VariableLenTiling() != ge::GRAPH_SUCCESS) {
                OP_LOGE(ctx_.nodeName, "ChunkGatedDeltaRuleBwdDhu VariableLenTiling failed.");
                return ge::GRAPH_FAILED;
            }
            tiling_.isVariable = 1;
        } else {
            tiling_.chunkNumForT = CeilDiv(tiling_.T, tiling_.chunkSize);
            tiling_.totalChunkNum = tiling_.chunkNumForT;
            tiling_.chunkTaskNum = tiling_.B * tiling_.chunkNumForT;
            tiling_.seqNum = tiling_.B;
            tiling_.isVariable = 0;
        }
        if (WorkspaceTiling() != ge::GRAPH_SUCCESS) {
            OP_LOGE(ctx_.nodeName, "ChunkGatedDeltaRuleBwdDhu WorkspaceTiling failed.");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

private:
    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                             const char *inputName)
    {
        if (curShape == nullptr) {
            OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.", inputName);
            return ge::GRAPH_FAILED;
        }
        const gert::Shape storageShape = curShape->GetStorageShape();
        size_t dimNum = storageShape.GetDimNum();
        if (dimNum != validDimNum) {
            OP_LOGE(ctx_.nodeName, "Check input %s shape failed, dim num should be %zu, but got %zu.",
                    inputName, validDimNum, dimNum);
            return ge::GRAPH_FAILED;
        }
        for (size_t dimIndex = 0; dimIndex < dimNum; ++dimIndex) {
            if (storageShape.GetDim(dimIndex) == 0) {
                OP_LOGE(ctx_.nodeName, "Check input %s shape failed, dim %zu is 0.", inputName, dimIndex);
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareDim(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                               const char *inputName2, size_t dimIndex)
    {
        const size_t dim1 = shape1.GetDim(dimIndex);
        const size_t dim2 = shape2.GetDim(dimIndex);
        if (dim1 != dim2) {
            OP_LOGE(ctx_.nodeName,
                    "Compare input shape of %s and %s failed, dim %zu should be same, but got %zu and %zu.",
                    inputName1, inputName2, dimIndex, dim1, dim2);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                 const char *inputName2, size_t compareDimNum)
    {
        for (size_t dimIndex = 0; dimIndex < compareDimNum; ++dimIndex) {
            if (CompareDim(shape1, shape2, inputName1, inputName2, dimIndex) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    int64_t CeilDiv(int64_t a, int64_t b) const
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    uint64_t DtypeSize(ge::DataType dtype) const
    {
        return dtype == ge::DT_FLOAT ? DTYPE_SIZE_FLOAT : DTYPE_SIZE_HALF;
    }

    uint64_t Align32(uint64_t bytes) const
    {
        return (bytes + ALIGN_BYTES_32 - 1) / ALIGN_BYTES_32 * ALIGN_BYTES_32;
    }

    uint64_t AlignDown(uint64_t bytes, uint64_t align) const
    {
        return align == 0 ? bytes : bytes / align * align;
    }

    uint64_t VectorTileBytes(uint64_t row, uint64_t maxDim, uint64_t qSize) const
    {
        uint64_t bytes = 4 * Align32(row * maxDim * qSize) +
                         2 * Align32(row * maxDim * DTYPE_SIZE_FLOAT) +
                         2 * Align32(row * static_cast<uint64_t>(tiling_.V) * DTYPE_SIZE_FLOAT);
        if (ctx_.qDataType == ge::DT_BF16) {
            bytes += 2 * Align32(row * static_cast<uint64_t>(tiling_.V) * qSize);
        }
        return bytes;
    }

    int64_t GetVecRow(uint64_t qSize) const
    {
        const uint64_t maxDim = static_cast<uint64_t>(std::max(tiling_.K, tiling_.V));
        const uint64_t gateElems = static_cast<uint64_t>(std::max(tiling_.K, tiling_.chunkSize));
        const uint64_t gateSize = DtypeSize(ctx_.gDataType);
        const uint64_t gateFactorResidentCount = ctx_.hasG ? 3UL : 1UL;
        const uint64_t maxRows = static_cast<uint64_t>(tiling_.K);
        uint64_t row = maxRows;
        if (ctx_.ubSize > UB_GUARD_BYTES) {
            while (row > 8) {
                const uint64_t fixedBytes =
                    2 * Align32(gateElems * gateSize) +
                    gateFactorResidentCount *
                        Align32(static_cast<uint64_t>(HEADS_PER_TASK) * gateElems * DTYPE_SIZE_FLOAT);
                if (fixedBytes + VectorTileBytes(row, maxDim, qSize) + UB_GUARD_BYTES <= ctx_.ubSize) {
                    break;
                }
                row /= 2;
            }
        } else {
            row = 8;
        }
        if (row < 8) {
            row = 8;
        }
        return static_cast<int64_t>(row);
    }

    int DtypeKey(ge::DataType dtype) const
    {
        if (dtype == ge::DT_BF16) {
            return TPL_BF16;
        }
        if (dtype == ge::DT_FLOAT) {
            return TPL_FP32;
        }
        return TPL_FP16;
    }

    ge::graphStatus PreCheck()
    {
        if (RequiredInputDimNumCheck(ctx_.qShape, DIM_NUM_4,
                                     CGDR_BWD_DHU_INPUT_Q_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (RequiredInputDimNumCheck(ctx_.kShape, DIM_NUM_4,
                                     CGDR_BWD_DHU_INPUT_K_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (RequiredInputDimNumCheck(ctx_.wShape, DIM_NUM_4,
                                     CGDR_BWD_DHU_INPUT_W_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (RequiredInputDimNumCheck(ctx_.dOShape, DIM_NUM_4,
                                     CGDR_BWD_DHU_INPUT_DO_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (RequiredInputDimNumCheck(ctx_.dvShape, DIM_NUM_4,
                                     CGDR_BWD_DHU_INPUT_DV_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (ctx_.hasG == ctx_.hasGk) {
            OP_LOGE(ctx_.nodeName, "Exactly one of g and gk must be provided, but hasG=%d hasGk=%d.",
                    static_cast<int>(ctx_.hasG), static_cast<int>(ctx_.hasGk));
            return ge::GRAPH_FAILED;
        }
        if (ctx_.gDataType != ge::DT_FLOAT && ctx_.gDataType != ctx_.qDataType) {
            OP_LOGE(ctx_.nodeName, "The dtype of g or gk must be float32 or match q and k.");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.hasG) {
            if (RequiredInputDimNumCheck(ctx_.gShape, DIM_NUM_3,
                                         CGDR_BWD_DHU_INPUT_G_NAME) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            if (RequiredInputDimNumCheck(ctx_.gkShape, DIM_NUM_4,
                                         CGDR_BWD_DHU_INPUT_GK_NAME) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape qShape = ctx_.qShape->GetStorageShape();
        const gert::Shape kShape = ctx_.kShape->GetStorageShape();
        const gert::Shape wShape = ctx_.wShape->GetStorageShape();
        const gert::Shape dOShape = ctx_.dOShape->GetStorageShape();
        const gert::Shape dvShape = ctx_.dvShape->GetStorageShape();

        if (CompareShape(qShape, kShape, CGDR_BWD_DHU_INPUT_Q_NAME,
                         CGDR_BWD_DHU_INPUT_K_NAME, DIM_NUM_4) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CompareShape(dOShape, dvShape, CGDR_BWD_DHU_INPUT_DO_NAME,
                         CGDR_BWD_DHU_INPUT_DV_NAME, DIM_NUM_4) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        tiling_.B = static_cast<int64_t>(qShape.GetDim(DIM_0));
        tiling_.HK = static_cast<int64_t>(qShape.GetDim(DIM_1));
        tiling_.T = static_cast<int64_t>(qShape.GetDim(DIM_2));
        tiling_.K = static_cast<int64_t>(qShape.GetDim(DIM_3));
        tiling_.HV = static_cast<int64_t>(dvShape.GetDim(DIM_1));
        tiling_.V = static_cast<int64_t>(dvShape.GetDim(DIM_3));
        tiling_.scale = static_cast<float>(ctx_.scale);

        if (CompareDim(qShape, wShape, CGDR_BWD_DHU_INPUT_Q_NAME,
                       CGDR_BWD_DHU_INPUT_W_NAME, DIM_0) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CompareDim(qShape, wShape, CGDR_BWD_DHU_INPUT_Q_NAME,
                       CGDR_BWD_DHU_INPUT_W_NAME, DIM_2) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CompareDim(qShape, wShape, CGDR_BWD_DHU_INPUT_Q_NAME,
                       CGDR_BWD_DHU_INPUT_W_NAME, DIM_3) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (static_cast<int64_t>(wShape.GetDim(DIM_1)) != tiling_.HV) {
            return ge::GRAPH_FAILED;
        }
        if (CompareDim(qShape, dOShape, CGDR_BWD_DHU_INPUT_Q_NAME,
                       CGDR_BWD_DHU_INPUT_DO_NAME, DIM_0) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CompareDim(qShape, dOShape, CGDR_BWD_DHU_INPUT_Q_NAME,
                       CGDR_BWD_DHU_INPUT_DO_NAME, DIM_2) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (ctx_.hasG) {
            const gert::Shape gShape = ctx_.gShape->GetStorageShape();
            if (CompareShape(dOShape, gShape, CGDR_BWD_DHU_INPUT_DO_NAME,
                             CGDR_BWD_DHU_INPUT_G_NAME, DIM_NUM_3) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            const gert::Shape gkShape = ctx_.gkShape->GetStorageShape();
            if (CompareShape(dOShape, gkShape, CGDR_BWD_DHU_INPUT_DO_NAME,
                             CGDR_BWD_DHU_INPUT_GK_NAME, DIM_NUM_3) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (CompareDim(qShape, gkShape, CGDR_BWD_DHU_INPUT_Q_NAME,
                           CGDR_BWD_DHU_INPUT_GK_NAME, DIM_3) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }

        if (tiling_.HV % tiling_.HK != 0) {
            return ge::GRAPH_FAILED;
        }
        tiling_.HRatio = tiling_.HV / tiling_.HK;
        tiling_.hasDh0 = ctx_.hasDh0 ? 1 : 0;
        tiling_.hasGk = ctx_.hasGk ? 1 : 0;

        if (tiling_.K != K_SIZE_128) {
            return ge::GRAPH_FAILED;
        }
        if (tiling_.V != V_SIZE_128 && tiling_.V != V_SIZE_256) {
            return ge::GRAPH_FAILED;
        }

        const int64_t chunkSize = static_cast<int64_t>(ctx_.chunkSize);
        if (chunkSize != CHUNK_SIZE_64 && chunkSize != CHUNK_SIZE_128) {
            return ge::GRAPH_FAILED;
        }
        tiling_.chunkSize = chunkSize;
        tiling_.headWindowNum = CeilDiv(tiling_.HV, HEADS_PER_TASK);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
        if (ctx_.cuSeqlensShape == nullptr || ctx_.chunkIndicesShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (tiling_.B != VAR_LEN_B) {
            return ge::GRAPH_FAILED;
        }
        if (RequiredInputDimNumCheck(ctx_.cuSeqlensShape, DIM_NUM_1,
                                     CGDR_BWD_DHU_INPUT_SEQLENS_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (RequiredInputDimNumCheck(ctx_.chunkIndicesShape, DIM_NUM_1,
                                     CGDR_BWD_DHU_INPUT_CHUNK_INDICES_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        const gert::Shape cuSeqlensShape = ctx_.cuSeqlensShape->GetStorageShape();
        const gert::Shape chunkIndicesShape = ctx_.chunkIndicesShape->GetStorageShape();
        const int64_t cuDim0 = static_cast<int64_t>(cuSeqlensShape.GetDim(DIM_0));
        const int64_t chunkIndicesDim0 =
            static_cast<int64_t>(chunkIndicesShape.GetDim(DIM_0));
        if (cuDim0 < 2) {
            return ge::GRAPH_FAILED;
        }
        if (chunkIndicesDim0 % CHUNK_INDICES_PAIR != 0) {
            return ge::GRAPH_FAILED;
        }
        tiling_.seqNum = cuDim0 - 1;
        tiling_.chunkNumForT = CeilDiv(tiling_.T, tiling_.chunkSize);
        tiling_.totalChunkNum = chunkIndicesDim0 / CHUNK_INDICES_PAIR;
        tiling_.chunkTaskNum = tiling_.totalChunkNum;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus WorkspaceTiling()
    {
        tiling_.taskNum = tiling_.seqNum * tiling_.headWindowNum;
        blockDim_ = ctx_.aicCoreNum == 0 ? 1U : ctx_.aicCoreNum;

        const uint64_t qSize = DtypeSize(ctx_.qDataType);
        tiling_.dh0ClearCoreNum = 0;
        tiling_.dh0ClearElemsPerCore = 0;
        tiling_.dh0ClearTailElems = 0;
        if (ctx_.hasDh0) {
            const uint64_t dh0Elems =
                static_cast<uint64_t>(tiling_.B) * static_cast<uint64_t>(tiling_.HV) *
                static_cast<uint64_t>(tiling_.totalChunkNum) * static_cast<uint64_t>(tiling_.K) *
                static_cast<uint64_t>(tiling_.V);
            const uint64_t dh0Bytes = dh0Elems * qSize;
            if (dh0Bytes > 0) {
                const uint64_t maxVecCoreNum =
                    static_cast<uint64_t>(blockDim_) * VECTOR_SUB_BLOCK_NUM;
                uint64_t clearCoreNum = dh0Bytes / ALIGN_BYTES_512;
                if (clearCoreNum == 0) {
                    clearCoreNum = 1;
                }
                clearCoreNum = std::min(clearCoreNum, maxVecCoreNum);
                uint64_t clearBytesPerCore = 0;
                if (clearCoreNum > 1) {
                    clearBytesPerCore = AlignDown(dh0Bytes / clearCoreNum, ALIGN_BYTES_512);
                    if (clearBytesPerCore == 0) {
                        clearCoreNum = 1;
                    }
                }
                const uint64_t clearTailBytes = dh0Bytes - clearBytesPerCore * (clearCoreNum - 1);
                tiling_.dh0ClearCoreNum = static_cast<int64_t>(clearCoreNum);
                tiling_.dh0ClearElemsPerCore = static_cast<int64_t>(clearBytesPerCore / qSize);
                tiling_.dh0ClearTailElems = static_cast<int64_t>(clearTailBytes / qSize);
            }
        }
        tiling_.vecRow = GetVecRow(qSize);
        tiling_.qgWorkspaceElems = tiling_.chunkSize * tiling_.K;
        tiling_.stateWorkspaceElems = static_cast<int64_t>(
            Align32(static_cast<uint64_t>(tiling_.K) * static_cast<uint64_t>(tiling_.V) * DTYPE_SIZE_FLOAT) / qSize);
        tiling_.dvStateWorkspaceElems = tiling_.chunkSize * tiling_.V;
        tiling_.termQWorkspaceElems = tiling_.K * tiling_.V;
        tiling_.dv2WorkspaceElems = 0;
        tiling_.termWWorkspaceElems = tiling_.K * tiling_.V;

        int64_t offset = 0;
        tiling_.qgWorkspaceOffset = offset;
        offset += tiling_.qgWorkspaceElems;
        offset = static_cast<int64_t>(Align32(static_cast<uint64_t>(offset) * qSize) / qSize);
        tiling_.stateWorkspaceOffset = offset;
        offset += tiling_.stateWorkspaceElems;
        tiling_.dvStateWorkspaceOffset = offset;
        offset += tiling_.dvStateWorkspaceElems;
        tiling_.termQWorkspaceOffset = offset;
        offset += tiling_.termQWorkspaceElems;
        tiling_.dv2WorkspaceOffset = offset;
        offset += tiling_.dv2WorkspaceElems;
        tiling_.termWWorkspaceOffset = offset;
        offset += tiling_.termWWorkspaceElems;
        tiling_.workspaceElemsPerSubBlock = offset;

        const uint64_t workspaceSlotNum =
            static_cast<uint64_t>(blockDim_) * WORKSPACE_BUFFER_COUNT;
        const uint64_t userWorkspaceBytes =
            workspaceSlotNum *
            static_cast<uint64_t>(tiling_.workspaceElemsPerSubBlock) * qSize;
        workspaceSize_ = ctx_.sysWorkspaceSize + static_cast<size_t>(userWorkspaceBytes);

        using namespace GDN;
        tilingKey_ = GET_TPL_TILING_KEY(
            static_cast<uint64_t>(DtypeKey(ctx_.qDataType)), static_cast<uint64_t>(DtypeKey(ctx_.gDataType)),
            static_cast<uint64_t>(tiling_.V), static_cast<uint64_t>(ctx_.hasGk ? 1 : 0));
        return ge::GRAPH_SUCCESS;
    }

    ChunkGatedDeltaRuleBwdDhuTilingContext &ctx_;
    ChunkGatedDeltaRuleBwdDhuTilingData &tiling_;
    size_t workspaceSize_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t tilingKey_ = 1U;
};

} // namespace optiling

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_PROCESSOR_H
