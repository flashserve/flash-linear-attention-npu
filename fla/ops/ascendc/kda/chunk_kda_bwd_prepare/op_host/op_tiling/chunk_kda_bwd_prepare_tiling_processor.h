/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_PREPARE_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../../op_kernel/chunk_kda_bwd_prepare_struct.h"

namespace optiling {

struct ChunkKdaBwdPrepareTilingContext {
    const char *nodeName;
    const gert::StorageShape *aqkShape;
    const gert::StorageShape *vNewShape;
    const gert::StorageShape *dOShape;
    const gert::StorageShape *hShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    ge::DataType aqkDataType;
    ge::DataType vNewDataType;
    ge::DataType dODataType;
    ge::DataType hDataType;
    double scale;
    int64_t chunkSize;
    bool stateVFirst;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
};

class ChunkKdaBwdPrepareTilingProcessor {
public:
    ChunkKdaBwdPrepareTilingProcessor(
        ChunkKdaBwdPrepareTilingContext &ctx,
        KDA::ChunkKdaBwdPrepareTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    ge::graphStatus Process()
    {
        if (ctx_.aqkShape == nullptr || ctx_.vNewShape == nullptr ||
            ctx_.dOShape == nullptr || ctx_.hShape == nullptr) {
            OP_LOGE(ctx_.nodeName, "required input shape is null");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.aqkDataType != ge::DT_BF16 || ctx_.vNewDataType != ge::DT_BF16 ||
            ctx_.dODataType != ge::DT_BF16 || ctx_.hDataType != ge::DT_BF16) {
            OP_LOGE(ctx_.nodeName, "KernelA A5 path requires BF16 inputs");
            return ge::GRAPH_FAILED;
        }
        if (ctx_.chunkSize != 64) {
            OP_LOGE(ctx_.nodeName, "chunk_size must be 64");
            return ge::GRAPH_FAILED;
        }
        const bool hasCu = ctx_.cuSeqlensShape != nullptr;
        const bool hasIndices = ctx_.chunkIndicesShape != nullptr;
        if (hasCu != hasIndices) {
            OP_LOGE(ctx_.nodeName, "cu_seqlens and chunk_indices must appear together");
            return ge::GRAPH_FAILED;
        }

        const gert::Shape aqk = ctx_.aqkShape->GetStorageShape();
        const gert::Shape vNew = ctx_.vNewShape->GetStorageShape();
        const gert::Shape dO = ctx_.dOShape->GetStorageShape();
        const gert::Shape h = ctx_.hShape->GetStorageShape();
        const bool isVariable = hasCu;
        const size_t tokenRank = isVariable ? 3 : 4;
        const size_t stateRank = isVariable ? 4 : 5;
        if (aqk.GetDimNum() != tokenRank || vNew.GetDimNum() != tokenRank ||
            dO.GetDimNum() != tokenRank || h.GetDimNum() != stateRank) {
            OP_LOGE(ctx_.nodeName, "input rank does not match dense/varlen contract");
            return ge::GRAPH_FAILED;
        }

        if (isVariable) {
            tiling_.B = 1;
            tiling_.NV = aqk.GetDim(0);
            tiling_.T = aqk.GetDim(1);
            tiling_.K = dO.GetDim(2);
            tiling_.V = vNew.GetDim(2);
            tiling_.denseChunkNum = 0;
            tiling_.totalChunkNum = h.GetDim(1);
            const gert::Shape cu = ctx_.cuSeqlensShape->GetStorageShape();
            const gert::Shape indices = ctx_.chunkIndicesShape->GetStorageShape();
            if (cu.GetDimNum() != 1 || indices.GetDimNum() != 1 ||
                cu.GetDim(0) < 2 || indices.GetDim(0) != 2 * tiling_.totalChunkNum) {
                OP_LOGE(ctx_.nodeName, "invalid varlen metadata shape");
                return ge::GRAPH_FAILED;
            }
            tiling_.seqNum = cu.GetDim(0) - 1;
        } else {
            tiling_.B = aqk.GetDim(0);
            tiling_.NV = aqk.GetDim(1);
            tiling_.T = aqk.GetDim(2);
            tiling_.K = dO.GetDim(3);
            tiling_.V = vNew.GetDim(3);
            tiling_.denseChunkNum = h.GetDim(2);
            tiling_.totalChunkNum = tiling_.B * tiling_.denseChunkNum;
            tiling_.seqNum = tiling_.B;
        }
        tiling_.chunkTaskNum = tiling_.totalChunkNum;
        constexpr int64_t HEADS_PER_WORK_TASK = 4;
        tiling_.headWindowNum =
            (tiling_.NV + HEADS_PER_WORK_TASK - 1) / HEADS_PER_WORK_TASK;
        tiling_.workTaskNum = tiling_.chunkTaskNum * tiling_.headWindowNum;
        tiling_.chunkSize = ctx_.chunkSize;
        tiling_.stateVFirst = ctx_.stateVFirst ? 1U : 0U;
        tiling_.isVariable = isVariable ? 1U : 0U;
        tiling_.scale = static_cast<float>(ctx_.scale);

        if (tiling_.B <= 0 || tiling_.NV <= 0 || tiling_.T <= 0 ||
            tiling_.K != 128 || tiling_.V != 128 ||
            tiling_.totalChunkNum <= 0 || tiling_.chunkTaskNum <= 0 ||
            tiling_.headWindowNum <= 0 || tiling_.workTaskNum <= 0) {
            OP_LOGE(ctx_.nodeName, "unsupported KernelA shape");
            return ge::GRAPH_FAILED;
        }
        blockDim_ = std::min(
            ctx_.aicCoreNum == 0 ? 1U : ctx_.aicCoreNum,
            static_cast<uint32_t>(tiling_.workTaskNum));
        tilingKey_ = isVariable ? 2U : 1U;
        workspaceSize_ = ctx_.sysWorkspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetBlockDim() const { return blockDim_; }
    uint32_t GetTilingKey() const { return tilingKey_; }
    size_t GetWorkspaceSize() const { return workspaceSize_; }

private:
    ChunkKdaBwdPrepareTilingContext &ctx_;
    KDA::ChunkKdaBwdPrepareTilingData &tiling_;
    uint32_t blockDim_ = 1;
    uint32_t tilingKey_ = 1;
    size_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_PREPARE_TILING_PROCESSOR_H
