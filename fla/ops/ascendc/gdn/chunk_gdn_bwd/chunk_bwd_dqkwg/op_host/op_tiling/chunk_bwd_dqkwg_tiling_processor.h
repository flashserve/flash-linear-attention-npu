/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_tiling_processor.h
 * \brief Tiling processor decoupled from gert::TilingContext, reusable in both aclnn and kernel launch modes.
 */

#ifndef CHUNK_BWD_DQKWG_TILING_PROCESSOR_H
#define CHUNK_BWD_DQKWG_TILING_PROCESSOR_H

#include "../../op_kernel/chunk_bwd_dqkwg_struct.h"
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

using GDN::ChunkBwdDqkwgTilingData;

namespace optiling {

static constexpr size_t CHUNK_BWD_DQKWG_INPUT_Q_IDX = 0;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_K_IDX = 1;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_V_IDX = 2;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_G_IDX = 3;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_H_IDX = 4;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_DO_IDX = 5;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_DH_IDX = 6;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_DV_IDX = 7;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_CUSEQLENS_IDX = 8;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_CHUNK_INDICES_IDX = 9;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_W_IDX = 10;
static constexpr size_t CHUNK_BWD_DQKWG_INPUT_G_GAMMA_IDX = 11;
static constexpr size_t CHUNK_BWD_DQKWG_ATTR_SCALE_IDX = 0;
static constexpr size_t CHUNK_BWD_DQKWG_ATTR_CHUNK_SIZE_IDX = 1;

static constexpr size_t QK_DIM_NUM = 4;
static constexpr size_t V_DIM_NUM = 4;
static constexpr size_t G_DIM_NUM = 3;
static constexpr size_t H_DIM_NUM = 5;
static constexpr size_t SEQLENS_DIM_NUM = 1;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t DIM_4 = 4;

static constexpr size_t FP16_SIZE = 2;
static constexpr size_t FP32_SIZE = 4;

static constexpr int64_t CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_SIZE_128 = 128;
static constexpr int64_t CHUNK_INDICES_DIM_1_SIZE = 2;
static constexpr int64_t K_SIZE_128 = 128;
static constexpr int64_t V_SIZE_128 = 128;
static constexpr int64_t V_SIZE_256 = 256;

static constexpr const char *const INPUT_Q_NAME = "q";
static constexpr const char *const INPUT_K_NAME = "k";
static constexpr const char *const INPUT_V_NAME = "v";
static constexpr const char *const INPUT_G_NAME = "g";
static constexpr const char *const INPUT_H_NAME = "h";
static constexpr const char *const INPUT_DO_NAME = "do";
static constexpr const char *const INPUT_DH_NAME = "dh";
static constexpr const char *const INPUT_DV_NAME = "dv";
static constexpr const char *const INPUT_CUSEQLENS_NAME = "cu_seqlens";
static constexpr const char *const INPUT_CHUNK_INDICES_NAME = "chunk_indices";

struct ChunkBwdDqkwgTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *vShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *hShape;
    const gert::StorageShape *doShape;
    const gert::StorageShape *dhShape;
    const gert::StorageShape *dvShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    float scale;
    int32_t chunkSize;
    uint32_t coreNum;
};

class ChunkBwdDqkwgTilingProcessor {
    ChunkBwdDqkwgTilingContext &ctx_;
    ChunkBwdDqkwgTilingData &tiling_;

public:
    explicit ChunkBwdDqkwgTilingProcessor(ChunkBwdDqkwgTilingContext &ctx, ChunkBwdDqkwgTilingData &tiling)
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
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.qShape, QK_DIM_NUM, INPUT_Q_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.kShape, QK_DIM_NUM, INPUT_K_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.vShape, V_DIM_NUM, INPUT_V_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.gShape, G_DIM_NUM, INPUT_G_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.hShape, H_DIM_NUM, INPUT_H_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.doShape, V_DIM_NUM, INPUT_DO_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dhShape, H_DIM_NUM, INPUT_DH_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.dvShape, V_DIM_NUM, INPUT_DV_NAME) != ge::GRAPH_SUCCESS,
                    , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    int64_t CeilDiv(int64_t a, int64_t b)
    {
        if (unlikely(b == 0)) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    ge::graphStatus CommonTiling()
    {
        const gert::Shape qStorageShape = ctx_.qShape->GetStorageShape();
        const gert::Shape kStorageShape = ctx_.kShape->GetStorageShape();
        const gert::Shape vStorageShape = ctx_.vShape->GetStorageShape();
        const gert::Shape gStorageShape = ctx_.gShape->GetStorageShape();
        const gert::Shape hStorageShape = ctx_.hShape->GetStorageShape();
        const gert::Shape doStorageShape = ctx_.doShape->GetStorageShape();
        const gert::Shape dhStorageShape = ctx_.dhShape->GetStorageShape();
        const gert::Shape dvStorageShape = ctx_.dvShape->GetStorageShape();

        int64_t B = vStorageShape.GetDim(DIM_0);
        int64_t HV = vStorageShape.GetDim(DIM_1);
        int64_t T = vStorageShape.GetDim(DIM_2);
        int64_t HK = kStorageShape.GetDim(DIM_1);
        int64_t K = kStorageShape.GetDim(DIM_3);
        int64_t V = vStorageShape.GetDim(DIM_3);

        OP_CHECK_IF(HK == 0 || HV % HK != 0,
                    OP_LOGE(ctx_.nodeName, "HV must be a multiple of HK, but HV = %ld, HK = %ld.", HV, HK),
                    return ge::GRAPH_FAILED);

        int64_t BT = static_cast<int64_t>(ctx_.chunkSize);
        OP_CHECK_IF(BT != CHUNK_SIZE_64 && BT != CHUNK_SIZE_128,
                    OP_LOGE(ctx_.nodeName, "BT should be 64 or 128, but get %ld.", BT),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(K != K_SIZE_128,
                    OP_LOGE(ctx_.nodeName, "K should be 128, but now K = %ld.", K),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(V != V_SIZE_128 && V != V_SIZE_256,
                    OP_LOGE(ctx_.nodeName, "V should be 128 or 256, but now V = %ld.", V),
                    return ge::GRAPH_FAILED);

        int64_t numChunks = CeilDiv(T, BT);
        int64_t isVarLen = 0;
        if (ctx_.cuSeqlensShape != nullptr) {
            OP_CHECK_IF(RequiredInputDimNumCheck(ctx_.cuSeqlensShape, SEQLENS_DIM_NUM, INPUT_CUSEQLENS_NAME) !=
                            ge::GRAPH_SUCCESS,
                        , return ge::GRAPH_FAILED);
            OP_CHECK_IF(ctx_.chunkIndicesShape == nullptr,
                        OP_LOGE(ctx_.nodeName, "Input %s is required, but got nullptr.",
                                INPUT_CHUNK_INDICES_NAME),
                        return ge::GRAPH_FAILED);
            const gert::Shape chunkIndicesStorageShape = ctx_.chunkIndicesShape->GetStorageShape();
            int64_t chunkIndicesDim0 = chunkIndicesStorageShape.GetDim(DIM_0);
            OP_CHECK_IF(chunkIndicesDim0 % CHUNK_INDICES_DIM_1_SIZE != 0,
                        OP_LOGE(ctx_.nodeName,
                                "Check chunk_indices shape failed, the dim 0 of chunk_indices needs to be divisible "
                                "by 2, but get %ld.",
                                chunkIndicesDim0),
                        return ge::GRAPH_FAILED);
            numChunks = chunkIndicesDim0 / CHUNK_INDICES_DIM_1_SIZE;
            isVarLen = 1;
        }

        auto align32 = [](size_t value) -> size_t {
            return ((value + 31) / 32) * 32;
        };

        const int64_t coreLoops = B * numChunks;
        int64_t aicNum = static_cast<int64_t>(ctx_.coreNum);
        if (aicNum < 1) {
            aicNum = 1;
        }
        int64_t usedAicNum = std::min(aicNum, coreLoops);
        if (usedAicNum < 1) {
            usedAicNum = 1;
        }
        const int64_t ringCoreSlots = usedAicNum;

        const size_t mainDgLastSize = align32(static_cast<size_t>(B) * HV * numChunks * FP32_SIZE);
        const size_t mainMm5Size = static_cast<size_t>(B) * HV * T * K * FP16_SIZE;
        const size_t mainDsTempSize = static_cast<size_t>(B) * HV * T * BT * FP16_SIZE;
        const size_t mainWorkspaceSize = mainDgLastSize + mainMm5Size + mainDsTempSize;

        auto actualWorkspaceForDepth = [&](int64_t depth) -> size_t {
            int64_t shortDepth = (depth / 2 >= 2) ? (depth / 2) : 2;
            size_t shortBtxK =
                align32(static_cast<size_t>(ringCoreSlots) * shortDepth * HV * BT * K * FP16_SIZE);
            size_t sharedBtxK =
                align32(static_cast<size_t>(ringCoreSlots) * depth * HV * BT * K * FP16_SIZE);
            size_t groupBtb =
                align32(static_cast<size_t>(ringCoreSlots) * depth * HV * BT * BT * FP16_SIZE);
            size_t shortBtb =
                align32(static_cast<size_t>(ringCoreSlots) * shortDepth * HV * BT * BT * FP16_SIZE);
            size_t dgLast = align32(static_cast<size_t>(ringCoreSlots) * depth * HV * FP32_SIZE);
            return shortBtxK + sharedBtxK + groupBtb + shortBtb + dgLast;
        };

        const size_t l2RingBudget = static_cast<size_t>(512) * 1024 * 1024;
        const size_t ringBudget = std::min(mainWorkspaceSize, l2RingBudget);
        int64_t groupRingDepth = 4;
        for (int64_t candidate : {16, 8, 4}) {
            if (actualWorkspaceForDepth(candidate) <= ringBudget) {
                groupRingDepth = candidate;
                break;
            }
        }

        const int64_t minDepthForOverlap = 8;
        if (groupRingDepth < minDepthForOverlap &&
            actualWorkspaceForDepth(minDepthForOverlap) <= mainWorkspaceSize) {
            groupRingDepth = minDepthForOverlap;
        }

        const int64_t adaptiveShortDepth = (groupRingDepth / 2 >= 2) ? (groupRingDepth / 2) : 2;
        const size_t shortBtxKSize =
            align32(static_cast<size_t>(ringCoreSlots) * adaptiveShortDepth * HV * BT * K * FP16_SIZE);
        const size_t sharedBtxKSize =
            align32(static_cast<size_t>(ringCoreSlots) * groupRingDepth * HV * BT * K * FP16_SIZE);
        const size_t groupBtbSize =
            align32(static_cast<size_t>(ringCoreSlots) * groupRingDepth * HV * BT * BT * FP16_SIZE);
        const size_t shortBtbSize =
            align32(static_cast<size_t>(ringCoreSlots) * adaptiveShortDepth * HV * BT * BT * FP16_SIZE);
        const size_t dgLastSize =
            align32(static_cast<size_t>(ringCoreSlots) * groupRingDepth * HV * FP32_SIZE);

        size_t offset = 0;
        const size_t wsDwOffset = offset;
        offset += shortBtxKSize;
        const size_t wsMm5Offset = offset;
        offset += sharedBtxKSize;
        const size_t wsDsTempOffset = offset;
        offset += groupBtbSize;
        const size_t wsMul1Offset = offset;
        offset += shortBtbSize;
        const size_t wsDgLastOffset = offset;
        offset += dgLastSize;

        tiling_.B = static_cast<uint64_t>(B);
        tiling_.HV = static_cast<uint64_t>(HV);
        tiling_.HK = static_cast<uint64_t>(HK);
        tiling_.T = static_cast<uint64_t>(T);
        tiling_.K = static_cast<uint64_t>(K);
        tiling_.V = static_cast<uint64_t>(V);
        tiling_.BT = static_cast<uint64_t>(BT);
        tiling_.numChunks = static_cast<uint64_t>(numChunks);
        tiling_.scale = ctx_.scale;
        tiling_.mul0RowNum = (V == V_SIZE_256) ? 16 : 32;
        tiling_.aicCoreNum = static_cast<uint32_t>(usedAicNum);
        tiling_.wsDwOffset = static_cast<uint64_t>(wsDwOffset);
        tiling_.wsBtxKSyncSlotsPerHead = static_cast<uint64_t>(groupRingDepth);
        tiling_.wsDgLastOffset = static_cast<uint64_t>(wsDgLastOffset);
        tiling_.dgLastSize = static_cast<uint64_t>(dgLastSize);
        tiling_.wsMm5Offset = static_cast<uint64_t>(wsMm5Offset);
        tiling_.wsDsTempOffset = static_cast<uint64_t>(wsDsTempOffset);
        tiling_.wsMm6Offset = static_cast<uint64_t>(wsDwOffset);
        tiling_.wsMm7Offset = static_cast<uint64_t>(wsMm5Offset);
        tiling_.wsMul1Offset = static_cast<uint64_t>(wsMul1Offset);
        tiling_.totalWorkspaceSize = static_cast<uint64_t>(offset);
        tiling_.isVarLen = static_cast<uint64_t>(isVarLen);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus FixLenTiling()
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VariableLenTiling()
    {
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
            OP_CHECK_IF(tiling_.B != 1,
                        OP_LOGE(ctx_.nodeName, "varlen mode only support B = 1, but now B = %ld.", tiling_.B),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(VariableLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVarLen = 1;
        } else {
            OP_CHECK_IF(FixLenTiling() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
            tiling_.isVarLen = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // CHUNK_BWD_DQKWG_TILING_PROCESSOR_H
