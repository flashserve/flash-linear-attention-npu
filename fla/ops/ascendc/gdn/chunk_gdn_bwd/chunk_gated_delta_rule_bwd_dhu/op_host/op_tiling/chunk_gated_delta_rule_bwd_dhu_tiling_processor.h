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
 * \brief Tiling processor decoupled from gert::TilingContext, reusable in both aclnn and kernel launch modes.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_PROCESSOR_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "exe_graph/runtime/storage_shape.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../../op_kernel/chunk_gated_delta_rule_bwd_dhu_struct.h"

using GDN::ChunkGatedDeltaRuleBwdDhuTilingData;

namespace optiling {

static constexpr size_t CHUNK_GDR_BWD_DHU_INPUT_Q_IDX = 0;
static constexpr size_t CHUNK_GDR_BWD_DHU_INPUT_K_IDX = 1;
static constexpr size_t CHUNK_GDR_BWD_DHU_INPUT_W_IDX = 2;
static constexpr size_t CHUNK_GDR_BWD_DHU_INPUT_DO_IDX = 3;
static constexpr size_t CHUNK_GDR_BWD_DHU_INPUT_DV_IDX = 4;
static constexpr size_t CHUNK_GDR_BWD_DHU_INPUT_G_IDX = 5;

static constexpr size_t CHUNK_GDR_BWD_DHU_DIM_0 = 0;
static constexpr size_t CHUNK_GDR_BWD_DHU_DIM_1 = 1;
static constexpr size_t CHUNK_GDR_BWD_DHU_DIM_2 = 2;
static constexpr size_t CHUNK_GDR_BWD_DHU_DIM_3 = 3;

static constexpr uint32_t CHUNK_GDR_BWD_DHU_NUM_64 = 64;
static constexpr uint32_t CHUNK_GDR_BWD_DHU_NUM_128 = 128;
static constexpr uint32_t CHUNK_GDR_BWD_DHU_NUM_2 = 2;
static constexpr uint32_t CHUNK_GDR_BWD_DHU_NUM_3 = 3;
static constexpr uint32_t CHUNK_GDR_BWD_DHU_BLOCK_SIZE = 32;

static constexpr uint32_t CHUNK_GDR_BWD_DHU_HALF_DTYPE_SIZE = 2;
static constexpr uint32_t CHUNK_GDR_BWD_DHU_FP32_DTYPE_SIZE = 4;

static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_Q_NAME = "q";
static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_K_NAME = "k";
static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_W_NAME = "w";
static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_DO_NAME = "d_o";
static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_DV_NAME = "dv";
static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_G_NAME = "g";
static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_CHUNK_INDICES_NAME = "chunk_indices";
static constexpr const char *const CHUNK_GDR_BWD_DHU_INPUT_SEQLENS_NAME = "cu_seqlens";

struct ChunkGatedDeltaRuleBwdDhuTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *wShape;
    const gert::StorageShape *doShape;
    const gert::StorageShape *dvShape;
    const gert::StorageShape *gShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    ge::DataType qDtype;
    ge::DataType gDtype;
    bool hasG;
    bool hasScaleAttr;
    double scaleAttr;
    int32_t chunkSize;
    uint64_t ubSize;
    uint32_t totalCoreNum;
    size_t sysWorkspaceSize;
};

class ChunkGatedDeltaRuleBwdDhuTilingProcessor {
    ChunkGatedDeltaRuleBwdDhuTilingContext &ctx_;
    ChunkGatedDeltaRuleBwdDhuTilingData &tiling_;

    bool isVariableLen_ = false;
    uint32_t tilingKey_ = GDN::CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_KEY;
    size_t workspaceSize_ = 0;
    uint32_t blockDim_ = 0;

    uint64_t B_ = 0;
    uint64_t Hv_ = 0;
    uint64_t Hk_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t chunkSize_ = CHUNK_GDR_BWD_DHU_NUM_64;

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
        return isVariableLen_;
    }

    template <typename T>
    static T CeilDiv(T a, T b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }

    ge::graphStatus Init()
    {
        const gert::Shape qShape = ctx_.qShape->GetStorageShape();
        const gert::Shape kShape = ctx_.kShape->GetStorageShape();
        const gert::Shape wShape = ctx_.wShape->GetStorageShape();
        const gert::Shape doShape = ctx_.doShape->GetStorageShape();
        const gert::Shape dvShape = ctx_.dvShape->GetStorageShape();

        B_ = static_cast<uint64_t>(qShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0));
        Hk_ = static_cast<uint64_t>(qShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_1));
        T_ = static_cast<uint64_t>(qShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_2));
        K_ = static_cast<uint64_t>(qShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_3));
        Hv_ = static_cast<uint64_t>(doShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_1));
        V_ = static_cast<uint64_t>(doShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_3));

        OP_CHECK_IF(
            kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0) != static_cast<int64_t>(B_) ||
                kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_1) != static_cast<int64_t>(Hk_) ||
                kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_2) != static_cast<int64_t>(T_) ||
                kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_3) != static_cast<int64_t>(K_),
            OP_LOGE(ctx_.nodeName,
                    "k must match q as [B,Hk,T,K]; q [%lu,%lu,%lu,%lu], k [%ld,%ld,%ld,%ld].", B_, Hk_, T_, K_,
                    kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0), kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_1),
                    kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_2), kShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_3)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0) != static_cast<int64_t>(B_) ||
                wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_1) != static_cast<int64_t>(Hv_) ||
                wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_2) != static_cast<int64_t>(T_) ||
                wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_3) != static_cast<int64_t>(K_),
            OP_LOGE(ctx_.nodeName,
                    "w must be [B,Hv,T,K] with Hv=dO.dim1; expect [%lu,%lu,%lu,%lu], got [%ld,%ld,%ld,%ld].", B_, Hv_,
                    T_, K_, wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0), wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_1),
                    wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_2), wShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_3)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(doShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0) != static_cast<int64_t>(B_) ||
                        doShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_2) != static_cast<int64_t>(T_),
                    OP_LOGE(ctx_.nodeName, "dO batch/time must match q."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            dvShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0) != static_cast<int64_t>(B_) ||
                dvShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_1) != static_cast<int64_t>(Hv_) ||
                dvShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_2) != static_cast<int64_t>(T_) ||
                dvShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_3) != static_cast<int64_t>(V_),
            OP_LOGE(ctx_.nodeName, "dv must be [B,Hv,T,V] aligned with dO."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(Hv_ == 0 || Hk_ == 0 || (Hv_ % Hk_) != 0,
                    OP_LOGE(ctx_.nodeName,
                            "GVA: Hv (value heads) must be an integer multiple of Hk (q/k heads); require Hv mod Hk "
                            "== 0; got Hk=%lu Hv=%lu.",
                            Hk_, Hv_),
                    return ge::GRAPH_FAILED);

        const bool isScale = ctx_.hasScaleAttr;
        const float scale = isScale ? static_cast<float>(ctx_.scaleAttr) : 1.0f;
        chunkSize_ = static_cast<uint64_t>(ctx_.chunkSize);
        OP_CHECK_IF(!(chunkSize_ == CHUNK_GDR_BWD_DHU_NUM_64 || chunkSize_ == CHUNK_GDR_BWD_DHU_NUM_128),
                    OP_LOGE(ctx_.nodeName, "chunk_size should be 64 or 128, but got %lu.", chunkSize_),
                    return ge::GRAPH_FAILED);

        tiling_.B = B_;
        tiling_.Hv = Hv_;
        tiling_.Hk = Hk_;
        tiling_.T = T_;
        tiling_.K = K_;
        tiling_.V = V_;
        tiling_.isScale = isScale ? 1 : 0;
        tiling_.scale = scale;
        tiling_.chunkSize = chunkSize_;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus VarLenSetting()
    {
        const bool hasCuSeqlens = ctx_.cuSeqlensShape != nullptr;
        const bool hasChunkIndices = ctx_.chunkIndicesShape != nullptr;
        if (hasCuSeqlens && hasChunkIndices) {
            isVariableLen_ = true;
        } else if (!hasCuSeqlens && !hasChunkIndices) {
            isVariableLen_ = false;
        } else {
            OP_LOGE(ctx_.nodeName, "cu_seqlens and chunk_indices must both be provided or both be omitted.");
            return ge::GRAPH_FAILED;
        }
        tiling_.isVarLen = isVariableLen_ ? 1 : 0;

        if (!isVariableLen_) {
            tiling_.chunkNum = static_cast<uint64_t>(CeilDiv(T_, chunkSize_));
            tiling_.seqNum = 1;
        } else {
            const gert::Shape cuSeqlensShape = ctx_.cuSeqlensShape->GetStorageShape();
            const gert::Shape chunkIndicesShape = ctx_.chunkIndicesShape->GetStorageShape();
            auto seqNum = cuSeqlensShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0) - 1;
            auto chunkNum = chunkIndicesShape.GetDim(CHUNK_GDR_BWD_DHU_DIM_0) / CHUNK_GDR_BWD_DHU_NUM_2;
            tiling_.seqNum = static_cast<uint64_t>(seqNum);
            tiling_.chunkNum = static_cast<uint64_t>(chunkNum);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckInputShape()
    {
        OP_CHECK_IF(isVariableLen_ && B_ != 1,
                    OP_LOGE(ctx_.nodeName, "B must be 1 when sequence is variable len, but got %lu.", B_),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckInputDtype()
    {
        if (!ctx_.hasG) {
            OP_LOGE(ctx_.nodeName, "Input g is required for chunk_gated_delta_rule_bwd_dhu kernel.");
            return ge::GRAPH_FAILED;
        }
        const ge::DataType qDtype = ctx_.qDtype;
        const ge::DataType gDtype = ctx_.gDtype;
        if (gDtype != qDtype && gDtype != ge::DT_FLOAT) {
            OP_LOGE(ctx_.nodeName, "gDtype must be DT_FLOAT or as same as qDtype");
            return ge::GRAPH_FAILED;
        }
        if (gDtype == ge::DT_FLOAT) {
            tilingKey_ = GDN::CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_KEY_G_FP32;
        } else {
            tilingKey_ = GDN::CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_KEY;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CalcUb()
    {
        const uint32_t halfBT = CeilDiv(static_cast<uint32_t>(chunkSize_), CHUNK_GDR_BWD_DHU_NUM_2);
        const uint32_t halfK = CeilDiv(static_cast<uint32_t>(K_), CHUNK_GDR_BWD_DHU_NUM_2);
        const uint32_t gBrcbBufByte = halfBT * CHUNK_GDR_BWD_DHU_BLOCK_SIZE;
        const uint32_t dvBufByte = halfBT * static_cast<uint32_t>(V_) * CHUNK_GDR_BWD_DHU_HALF_DTYPE_SIZE;
        const uint32_t dvCastBufByte = halfBT * static_cast<uint32_t>(V_) * CHUNK_GDR_BWD_DHU_FP32_DTYPE_SIZE;
        const uint32_t dqkBufByte = halfBT * static_cast<uint32_t>(K_) * CHUNK_GDR_BWD_DHU_HALF_DTYPE_SIZE;
        const uint32_t dqkCastBufByte = halfBT * static_cast<uint32_t>(K_) * CHUNK_GDR_BWD_DHU_FP32_DTYPE_SIZE;
        const uint32_t dhCastBufByte = halfK * static_cast<uint32_t>(V_) * CHUNK_GDR_BWD_DHU_FP32_DTYPE_SIZE;

        const uint32_t dvPeak =
            static_cast<uint32_t>(chunkSize_) * CHUNK_GDR_BWD_DHU_FP32_DTYPE_SIZE + gBrcbBufByte + dvBufByte +
            CHUNK_GDR_BWD_DHU_NUM_2 * dvCastBufByte;
        const uint32_t gatedQPeak = CHUNK_GDR_BWD_DHU_NUM_2 * static_cast<uint32_t>(chunkSize_) *
                                        CHUNK_GDR_BWD_DHU_FP32_DTYPE_SIZE +
                                    dqkBufByte + dqkCastBufByte + gBrcbBufByte;
        const uint32_t dhPeak = CHUNK_GDR_BWD_DHU_NUM_2 * dhCastBufByte;
        const uint32_t tBufByte = std::max(dhPeak, std::max(dvPeak, gatedQPeak));

        OP_CHECK_IF(tBufByte > ctx_.ubSize,
                    OP_LOGE(ctx_.nodeName, "K/V is too large, K should less than 128 and V should less than 256."),
                    return ge::GRAPH_FAILED);
        tiling_.gBufSize = halfBT;
        tiling_.dvBufSize = halfBT * static_cast<uint64_t>(V_);
        tiling_.qBufSize = halfBT * static_cast<uint64_t>(K_);
        tiling_.dhBufSize = halfK * static_cast<uint64_t>(V_);
        tiling_.totalTbufByte = tBufByte;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetWorkspaceSize()
    {
        const uint32_t taskNum = static_cast<uint32_t>(B_ * Hk_ * tiling_.seqNum);
        const uint32_t usedCoreNum = taskNum > ctx_.totalCoreNum ? ctx_.totalCoreNum : taskNum;
        tiling_.usedCoreNum = usedCoreNum;
        blockDim_ = usedCoreNum;

        const uint64_t bdvWs = chunkSize_ * V_ * usedCoreNum;
        const uint64_t qWs = K_ * chunkSize_ * usedCoreNum;
        const uint64_t wDv2Ws = K_ * V_ * usedCoreNum;
        const uint64_t qDoWs = K_ * V_ * usedCoreNum;
        const size_t usrWsSize =
            static_cast<size_t>((bdvWs + qWs + wDv2Ws + qDoWs) * CHUNK_GDR_BWD_DHU_HALF_DTYPE_SIZE);

        workspaceSize_ = usrWsSize + ctx_.sysWorkspaceSize;
        tiling_.bdvWs = bdvWs;
        tiling_.qWs = qWs;
        tiling_.wDv2Ws = wDv2Ws;
        tiling_.qDoWs = qDoWs;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(Init() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(VarLenSetting() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckInputShape() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(CalcUb() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(SetWorkspaceSize() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
};

} // namespace optiling

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_TILING_PROCESSOR_H
