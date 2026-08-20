#ifndef CHUNK_KDA_BWD_A_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_A_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/chunk_kda_bwd_common.h"

namespace optiling {

constexpr size_t KDA_BWD_A_AQK_IDX = 0;
constexpr size_t KDA_BWD_A_V_NEW_IDX = 1;
constexpr size_t KDA_BWD_A_H_IDX = 2;
constexpr size_t KDA_BWD_A_DO_IDX = 3;
constexpr size_t KDA_BWD_A_CU_SEQLENS_IDX = 4;
constexpr size_t KDA_BWD_A_CHUNK_INDICES_IDX = 5;
constexpr size_t KDA_BWD_A_CHUNK_SIZE_ATTR_IDX = 0;

struct ChunkKdaBwdATilingContext {
    const char *nodeName;
    const gert::StorageShape *aqkShape;
    const gert::StorageShape *vNewShape;
    const gert::StorageShape *hShape;
    const gert::StorageShape *doShape;
    const gert::StorageShape *cuSeqlensShape;
    const gert::StorageShape *chunkIndicesShape;
    ge::DataType dataType;
    ge::DataType vNewType;
    ge::DataType hType;
    ge::DataType doType;
    int64_t chunkSize;
    uint32_t aicCoreNum;
    size_t systemWorkspaceSize;
};

class ChunkKdaBwdATilingProcessor {
public:
    ChunkKdaBwdATilingProcessor(
        ChunkKdaBwdATilingContext &ctx, KDA::ChunkKdaBwdATilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(ctx_.aqkShape == nullptr ||
                        ctx_.vNewShape == nullptr || ctx_.hShape == nullptr ||
                        ctx_.doShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "Aqk/v_new/h/d_o are required."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.dataType != ge::DT_FLOAT16 &&
                        ctx_.dataType != ge::DT_BF16,
                    OP_LOGE(ctx_.nodeName, "Kernel A supports FP16/BF16 inputs."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.vNewType != ctx_.dataType ||
                        ctx_.hType != ctx_.dataType ||
                        ctx_.doType != ctx_.dataType,
                    OP_LOGE(ctx_.nodeName,
                            "Aqk/v_new/h/d_o must use one common dtype."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkSize != 64,
                    OP_LOGE(ctx_.nodeName, "Kernel A requires chunk_size=64."),
                    return ge::GRAPH_FAILED);

        const bool hasCu = ctx_.cuSeqlensShape != nullptr;
        const bool hasChunks = ctx_.chunkIndicesShape != nullptr;
        OP_CHECK_IF(hasCu != hasChunks,
                    OP_LOGE(ctx_.nodeName,
                            "cu_seqlens and chunk_indices must be both present or both absent."),
                    return ge::GRAPH_FAILED);
        tiling_.isVarLen = hasCu ? 1 : 0;

        const gert::Shape aqk = ctx_.aqkShape->GetOriginShape();
        const gert::Shape vNew = ctx_.vNewShape->GetOriginShape();
        const gert::Shape h = ctx_.hShape->GetOriginShape();
        const gert::Shape dO = ctx_.doShape->GetOriginShape();
        const size_t tokenRank = hasCu ? 3 : 4;
        const size_t stateRank = hasCu ? 4 : 5;
        OP_CHECK_IF(aqk.GetDimNum() != tokenRank ||
                        vNew.GetDimNum() != tokenRank ||
                        dO.GetDimNum() != tokenRank ||
                        h.GetDimNum() != stateRank,
                    OP_LOGE(ctx_.nodeName,
                            "Kernel A expects dense BNSD rank-4/rank-5 or varlen NTD rank-3/rank-4."),
                    return ge::GRAPH_FAILED);

        const size_t headAxis = hasCu ? 0 : 1;
        const size_t tokenAxis = hasCu ? 1 : 2;
        const size_t dimAxis = hasCu ? 2 : 3;
        const int64_t batch = hasCu ? 1 : static_cast<int64_t>(aqk.GetDim(0));
        tiling_.headNum = static_cast<int64_t>(aqk.GetDim(headAxis));
        tiling_.seqlen = static_cast<int64_t>(aqk.GetDim(tokenAxis));
        const int64_t valueDim = static_cast<int64_t>(dO.GetDim(dimAxis));
        tiling_.chunkSize = ctx_.chunkSize;
        OP_CHECK_IF(aqk.GetDim(dimAxis) != 64,
                    OP_LOGE(ctx_.nodeName, "Kernel A requires C=64 and K=128."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(valueDim != 128 && valueDim != 256,
                    OP_LOGE(ctx_.nodeName, "Kernel A requires V=128 or V=256."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(vNew.GetDim(dimAxis) != static_cast<size_t>(valueDim) ||
                        dO.GetDim(dimAxis) != static_cast<size_t>(valueDim),
                    OP_LOGE(ctx_.nodeName, "v_new and d_o must have the same V."),
                    return ge::GRAPH_FAILED);
        for (size_t axis = 0; axis < dimAxis; ++axis) {
            OP_CHECK_IF(aqk.GetDim(axis) != vNew.GetDim(axis) ||
                            aqk.GetDim(axis) != dO.GetDim(axis),
                        OP_LOGE(ctx_.nodeName,
                                "Aqk/v_new/d_o leading dimensions must match."),
                        return ge::GRAPH_FAILED);
        }

        if (!hasCu) {
            tiling_.chunkNumPerBatch =
                (tiling_.seqlen + tiling_.chunkSize - 1) / tiling_.chunkSize;
            tiling_.chunkNum = batch * tiling_.chunkNumPerBatch;
            OP_CHECK_IF(h.GetDim(0) != static_cast<size_t>(batch) ||
                            h.GetDim(1) != static_cast<size_t>(tiling_.chunkNumPerBatch) ||
                            h.GetDim(2) != static_cast<size_t>(tiling_.headNum) ||
                            h.GetDim(3) != 128 ||
                            h.GetDim(4) != static_cast<size_t>(valueDim),
                        OP_LOGE(ctx_.nodeName,
                                "Dense h must be [B,chunkNum,H,K,V]."),
                        return ge::GRAPH_FAILED);
        } else {
            const gert::Shape cu = ctx_.cuSeqlensShape->GetOriginShape();
            const gert::Shape chunks = ctx_.chunkIndicesShape->GetOriginShape();
            OP_CHECK_IF(cu.GetDimNum() != 1 || cu.GetDim(0) < 2 ||
                            chunks.GetDimNum() != 1 ||
                            chunks.GetDim(0) == 0 || (chunks.GetDim(0) % 2) != 0,
                        OP_LOGE(ctx_.nodeName,
                                "Varlen metadata requires cu_seqlens[N+1] and flattened chunk_indices[2*Nc]."),
                        return ge::GRAPH_FAILED);
            tiling_.chunkNumPerBatch = 0;
            tiling_.chunkNum = static_cast<int64_t>(chunks.GetDim(0) / 2);
            OP_CHECK_IF(h.GetDim(0) != static_cast<size_t>(tiling_.chunkNum) ||
                            h.GetDim(1) != static_cast<size_t>(tiling_.headNum) ||
                            h.GetDim(2) != 128 ||
                            h.GetDim(3) != static_cast<size_t>(valueDim),
                        OP_LOGE(ctx_.nodeName,
                                "Varlen h must be [totalChunks,H,K,V]."),
                        return ge::GRAPH_FAILED);
        }

        const uint64_t ownerCount =
            static_cast<uint64_t>(tiling_.chunkNum) * tiling_.headNum;
        blockDim_ = static_cast<uint32_t>(std::min<uint64_t>(
            ownerCount, static_cast<uint64_t>(ctx_.aicCoreNum)));
        if (blockDim_ == 0) {
            blockDim_ = 1;
        }
        tiling_.usedCoreNum = blockDim_;
        workspaceSize_ = ctx_.systemWorkspaceSize;
        tilingKey_ = (ctx_.dataType == ge::DT_BF16 ? 2U : 0U) +
                     (valueDim == 256 ? 2U : 1U);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetBlockDim() const { return blockDim_; }
    size_t GetWorkspaceSize() const { return workspaceSize_; }
    uint64_t GetTilingKey() const { return tilingKey_; }

private:
    uint64_t AlignUp(uint64_t value, uint64_t align) const
    {
        return (value + align - 1) / align * align;
    }

    ChunkKdaBwdATilingContext &ctx_;
    KDA::ChunkKdaBwdATilingData &tiling_;
    uint32_t blockDim_ = 1;
    size_t workspaceSize_ = 0;
    uint64_t tilingKey_ = 1;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_A_TILING_PROCESSOR_H
