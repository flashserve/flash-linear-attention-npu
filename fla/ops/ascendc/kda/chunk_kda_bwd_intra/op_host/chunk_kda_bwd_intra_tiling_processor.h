/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_INTRA_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_INTRA_TILING_PROCESSOR_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"

#include "../op_kernel/chunk_kda_bwd_intra_struct.h"
#include "../op_kernel/chunk_kda_bwd_intra_tiling_key.h"

namespace optiling {

using KDA::ChunkKdaBwdIntraTilingData;

constexpr size_t KDA_BWD_Q_IDX = 0;
constexpr size_t KDA_BWD_K_IDX = 1;
constexpr size_t KDA_BWD_GK_IDX = 2;
constexpr size_t KDA_BWD_BETA_IDX = 3;
constexpr size_t KDA_BWD_DAQK_IDX = 4;
constexpr size_t KDA_BWD_DAKK_IDX = 5;
constexpr size_t KDA_BWD_DQ_IDX = 6;
constexpr size_t KDA_BWD_DK_IDX = 7;
constexpr size_t KDA_BWD_DB_IDX = 8;
constexpr size_t KDA_BWD_DG_IDX = 9;
constexpr size_t KDA_BWD_CU_SEQLENS_IDX = 10;
constexpr size_t KDA_BWD_CHUNK_METADATA_IDX = 11;
constexpr size_t KDA_BWD_CHUNK_SIZE_ATTR_IDX = 0;
constexpr size_t KDA_BWD_SAFE_GATE_ATTR_IDX = 1;
constexpr size_t KDA_BWD_LAYOUT_MODE_ATTR_IDX = 2;
constexpr size_t KDA_BWD_TOTAL_CHUNKS_ATTR_IDX = 3;
constexpr uint64_t KDA_BWD_HEADS_PER_WINDOW = 2;
constexpr int64_t KDA_BWD_WORKSPACE_SLOT_COUNT = 4;
constexpr int64_t KDA_BWD_LAYOUT_DENSE_BNSD = 0;
constexpr int64_t KDA_BWD_LAYOUT_VARLEN_TND = 1;

struct ChunkKdaBwdIntraTilingContext {
    const char *nodeName;
    const gert::StorageShape *qShape;
    const gert::StorageShape *kShape;
    const gert::StorageShape *gkShape;
    const gert::StorageShape *betaShape;
    const gert::StorageShape *dAqkShape;
    const gert::StorageShape *dAkkShape;
    const gert::StorageShape *dqShape;
    const gert::StorageShape *dkShape;
    const gert::StorageShape *dbShape;
    const gert::StorageShape *dgShape;
    ge::DataType qDataType;
    ge::DataType betaDataType;
    int64_t chunkSize;
    bool safeGate;
    int64_t layoutMode;
    int64_t totalChunks;
    bool hasCuSeqlens;
    bool hasChunkMetadata;
    int64_t cuSeqlensElements;
    int64_t chunkMetadataElements;
    uint32_t aicCoreNum;
    size_t systemWorkspaceSize;
    bool isA5;
};

class ChunkKdaBwdIntraTilingProcessor {
public:
    ChunkKdaBwdIntraTilingProcessor(
        ChunkKdaBwdIntraTilingContext &ctx, ChunkKdaBwdIntraTilingData &tiling)
        : ctx_(ctx), tiling_(tiling)
    {
    }

    ge::graphStatus Process()
    {
        OP_CHECK_IF(CheckSpec() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        OP_CHECK_IF(BuildWorkspace() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetBlockDim() const
    {
        return blockDim_;
    }

    uint64_t GetTilingKey() const
    {
        return tilingKey_;
    }

    size_t GetWorkspaceSize() const
    {
        return workspaceSize_;
    }

private:
    static uint64_t Align512(uint64_t value)
    {
        return (value + 511U) / 512U * 512U;
    }

    ge::graphStatus RequireRank(const gert::StorageShape *shape, size_t rank, const char *name) const
    {
        OP_CHECK_IF(shape == nullptr, OP_LOGE(ctx_.nodeName, "%s is required.", name),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != rank,
                    OP_LOGE(ctx_.nodeName, "%s must be rank %zu.", name, rank),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    bool SameShape(const gert::StorageShape *lhs, const gert::StorageShape *rhs) const
    {
        const gert::Shape a = lhs->GetStorageShape();
        const gert::Shape b = rhs->GetStorageShape();
        if (a.GetDimNum() != b.GetDimNum()) {
            return false;
        }
        for (size_t i = 0; i < a.GetDimNum(); ++i) {
            if (a.GetDim(i) != b.GetDim(i)) {
                return false;
            }
        }
        return true;
    }

    ge::graphStatus CheckSpec()
    {
        OP_CHECK_IF(ctx_.qDataType != ge::DT_BF16,
                    OP_LOGE(ctx_.nodeName, "ChunkKdaBwdIntra requires BF16 q/k."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.betaDataType != ge::DT_BF16 &&
                        ctx_.betaDataType != ge::DT_FLOAT,
                    OP_LOGE(ctx_.nodeName, "ChunkKdaBwdIntra beta must be BF16 or FP32."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!ctx_.safeGate,
                    OP_LOGE(ctx_.nodeName,
                            "safe_gate=false is reserved but not supported by the first kernel."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.chunkSize != 64,
                    OP_LOGE(ctx_.nodeName, "chunk_size must be 64."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_.layoutMode != KDA_BWD_LAYOUT_DENSE_BNSD &&
                        ctx_.layoutMode != KDA_BWD_LAYOUT_VARLEN_TND,
                    OP_LOGE(ctx_.nodeName, "layout_mode must be dense BNSD or varlen TND."),
                    return ge::GRAPH_FAILED);

        const bool isVarLen = ctx_.layoutMode == KDA_BWD_LAYOUT_VARLEN_TND;

        OP_CHECK_IF(ctx_.qShape == nullptr,
                    OP_LOGE(ctx_.nodeName, "q is required."),
                    return ge::GRAPH_FAILED);
        const gert::StorageShape *vectorShapes[] = {
            ctx_.qShape, ctx_.kShape, ctx_.gkShape, ctx_.dAqkShape, ctx_.dAkkShape,
            ctx_.dqShape, ctx_.dkShape, ctx_.dgShape
        };
        const gert::Shape q = ctx_.qShape->GetStorageShape();
        const size_t qRank = q.GetDimNum();
        OP_CHECK_IF((isVarLen && qRank != 3 && qRank != 4) ||
                        (!isVarLen && qRank != 4),
                    OP_LOGE(ctx_.nodeName, "q rank does not match layout_mode."),
                    return ge::GRAPH_FAILED);
        const size_t vectorRank = isVarLen ? qRank : 4;
        const size_t scalarRank = vectorRank - 1;
        for (size_t i = 0; i < 8; ++i) {
            const char *names[] = {"q", "k", "gk", "dAqk", "dAkk", "dq", "dk", "dg"};
            OP_CHECK_IF(RequireRank(vectorShapes[i], vectorRank, names[i]) != ge::GRAPH_SUCCESS, ,
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(RequireRank(ctx_.betaShape, scalarRank, "beta") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(RequireRank(ctx_.dbShape, scalarRank, "db") != ge::GRAPH_SUCCESS, ,
                    return ge::GRAPH_FAILED);

        if (isVarLen) {
            OP_CHECK_IF(qRank == 4 && q.GetDim(0) != 1,
                        OP_LOGE(ctx_.nodeName, "varlen BSND compatibility requires B=1."),
                        return ge::GRAPH_FAILED);
            const size_t tokenDim = qRank == 3 ? 0 : 1;
            const size_t headDim = qRank == 3 ? 1 : 2;
            const size_t featureDim = qRank == 3 ? 2 : 3;
            tiling_.batch = 1;
            tiling_.seqlen = static_cast<int64_t>(q.GetDim(tokenDim));
            tiling_.headNum = static_cast<int64_t>(q.GetDim(headDim));
            tiling_.headDim = static_cast<int64_t>(q.GetDim(featureDim));
        } else {
            tiling_.batch = static_cast<int64_t>(q.GetDim(0));
            tiling_.headNum = static_cast<int64_t>(q.GetDim(1));
            tiling_.seqlen = static_cast<int64_t>(q.GetDim(2));
            tiling_.headDim = static_cast<int64_t>(q.GetDim(3));
        }
        tiling_.chunkSize = ctx_.chunkSize;

        OP_CHECK_IF(tiling_.batch <= 0 || tiling_.headNum <= 0 || tiling_.seqlen <= 0,
                    OP_LOGE(ctx_.nodeName, "B, H and T must be positive."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((isVarLen && tiling_.headDim != 128) ||
                        (!isVarLen && tiling_.headDim != 64 &&
                         tiling_.headDim != 128 && tiling_.headDim != 256),
                    OP_LOGE(ctx_.nodeName,
                            "varlen supports K=128; dense supports K=64/128/256."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!SameShape(ctx_.qShape, ctx_.kShape) || !SameShape(ctx_.qShape, ctx_.gkShape) ||
                        !SameShape(ctx_.qShape, ctx_.dqShape) || !SameShape(ctx_.qShape, ctx_.dkShape) ||
                        !SameShape(ctx_.qShape, ctx_.dgShape),
                    OP_LOGE(ctx_.nodeName, "q/k/gk/dq/dk/dg shapes must match."),
                    return ge::GRAPH_FAILED);
        const gert::Shape beta = ctx_.betaShape->GetStorageShape();
        const gert::Shape dAqk = ctx_.dAqkShape->GetStorageShape();
        OP_CHECK_IF(!SameShape(ctx_.betaShape, ctx_.dbShape) ||
                        !SameShape(ctx_.dAqkShape, ctx_.dAkkShape),
                    OP_LOGE(ctx_.nodeName, "beta/db and dAqk/dAkk shapes must match."),
                    return ge::GRAPH_FAILED);
        if (isVarLen) {
            const size_t tokenDim = qRank == 3 ? 0 : 1;
            const size_t headDim = qRank == 3 ? 1 : 2;
            const size_t scalarTokenDim = qRank == 3 ? 0 : 1;
            const size_t scalarHeadDim = qRank == 3 ? 1 : 2;
            const size_t matrixTokenDim = tokenDim;
            const size_t matrixHeadDim = headDim;
            const size_t matrixChunkDim = qRank == 3 ? 2 : 3;
            OP_CHECK_IF(beta.GetDim(scalarTokenDim) != static_cast<size_t>(tiling_.seqlen) ||
                            beta.GetDim(scalarHeadDim) != static_cast<size_t>(tiling_.headNum),
                        OP_LOGE(ctx_.nodeName, "varlen beta/db must be [T,H] or [1,T,H]."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(dAqk.GetDim(matrixTokenDim) != static_cast<size_t>(tiling_.seqlen) ||
                            dAqk.GetDim(matrixHeadDim) != static_cast<size_t>(tiling_.headNum) ||
                            dAqk.GetDim(matrixChunkDim) !=
                                static_cast<size_t>(tiling_.chunkSize),
                        OP_LOGE(ctx_.nodeName,
                                "varlen dAqk/dAkk must be [T,H,64] or [1,T,H,64]."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(!ctx_.hasCuSeqlens || !ctx_.hasChunkMetadata ||
                            ctx_.totalChunks <= 0 ||
                            ctx_.cuSeqlensElements < 2 ||
                            ctx_.cuSeqlensElements > 65 ||
                            ctx_.chunkMetadataElements != ctx_.totalChunks * 4,
                        OP_LOGE(ctx_.nodeName,
                                "invalid varlen cu_seqlens or packed metadata."),
                        return ge::GRAPH_FAILED);
            tiling_.chunkNumPerBatch = 0;
            tiling_.chunkNum = ctx_.totalChunks;
        } else {
            OP_CHECK_IF(ctx_.hasCuSeqlens || ctx_.hasChunkMetadata || ctx_.totalChunks != 0,
                        OP_LOGE(ctx_.nodeName, "dense mode must not carry varlen metadata."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(beta.GetDim(0) != static_cast<size_t>(tiling_.batch) ||
                            beta.GetDim(1) != static_cast<size_t>(tiling_.headNum) ||
                            beta.GetDim(2) != static_cast<size_t>(tiling_.seqlen),
                        OP_LOGE(ctx_.nodeName, "dense beta/db must be [B,H,T]."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(dAqk.GetDim(0) != static_cast<size_t>(tiling_.batch) ||
                            dAqk.GetDim(1) != static_cast<size_t>(tiling_.headNum) ||
                            dAqk.GetDim(2) != static_cast<size_t>(tiling_.seqlen) ||
                            dAqk.GetDim(3) != static_cast<size_t>(tiling_.chunkSize),
                        OP_LOGE(ctx_.nodeName,
                                "dense dAqk/dAkk must be [B,H,T,chunk_size]."),
                        return ge::GRAPH_FAILED);
            tiling_.chunkNumPerBatch =
                (tiling_.seqlen + tiling_.chunkSize - 1) / tiling_.chunkSize;
            tiling_.chunkNum = tiling_.batch * tiling_.chunkNumPerBatch;
        }

        const uint64_t headWindowCount =
            (static_cast<uint64_t>(tiling_.headNum) + KDA_BWD_HEADS_PER_WINDOW - 1) /
            KDA_BWD_HEADS_PER_WINDOW;
        const uint64_t taskGroupCount =
            static_cast<uint64_t>(tiling_.chunkNum) * headWindowCount;
        blockDim_ = static_cast<uint32_t>(
            std::min<uint64_t>(taskGroupCount, static_cast<uint64_t>(ctx_.aicCoreNum)));
        if (blockDim_ == 0) {
            blockDim_ = 1;
        }

        const uint32_t kKey = tiling_.headDim == 64 ? CHUNK_KDA_BWD_INTRA_K64 :
                              (tiling_.headDim == 128 ? CHUNK_KDA_BWD_INTRA_K128 :
                                                       CHUNK_KDA_BWD_INTRA_K256);
        const uint32_t layoutKey = isVarLen ? CHUNK_KDA_BWD_INTRA_VARLEN_TND :
                                             CHUNK_KDA_BWD_INTRA_DENSE_BNSD;
        const uint32_t betaKey =
            ctx_.betaDataType == ge::DT_FLOAT ? CHUNK_KDA_BWD_INTRA_BETA_FP32 :
                                               CHUNK_KDA_BWD_INTRA_BETA_BF16;
        tilingKey_ = GET_TPL_TILING_KEY(
            kKey, CHUNK_KDA_BWD_INTRA_SAFE, layoutKey, betaKey);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus BuildWorkspace()
    {
        const uint64_t bt = static_cast<uint64_t>(tiling_.chunkSize);
        const uint64_t k = static_cast<uint64_t>(tiling_.headDim);
        const bool useA5DenseK128Row32 =
            ctx_.isA5 && ctx_.layoutMode == KDA_BWD_LAYOUT_DENSE_BNSD &&
            tiling_.headDim == 128 && tiling_.chunkSize == 64;
        const uint64_t bc = useA5DenseK128Row32 ? 32 : 16;

        tiling_.aLowerOffset = 0;
        tiling_.bLowerOffset = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(tiling_.aLowerOffset) + 2 * bc * bt * sizeof(float)));
        tiling_.aUpperOffset = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(tiling_.bLowerOffset) + bt * k * sizeof(float)));
        tiling_.bUpperOffset = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(tiling_.aUpperOffset) + 2 * bt * bc * sizeof(float)));
        const uint64_t inputRegionSize =
            Align512(static_cast<uint64_t>(tiling_.bUpperOffset) + 2 * bt * k * sizeof(float));

        tiling_.resultDqOffset = 0;
        tiling_.resultDkLowerOffset = static_cast<int64_t>(
            Align512(bc * k * sizeof(float)));
        tiling_.resultDkUpperOffset = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(tiling_.resultDkLowerOffset) + bc * k * sizeof(float)));
        const uint64_t resultRegionSize =
            Align512(static_cast<uint64_t>(tiling_.resultDkUpperOffset) + bc * k * sizeof(float));
        tiling_.resultRegionOffset = static_cast<int64_t>(
            Align512(inputRegionSize));
        tiling_.workspaceSlotSize = static_cast<int64_t>(
            Align512(static_cast<uint64_t>(tiling_.resultRegionOffset) +
                     resultRegionSize));
        tiling_.workspaceCoreSize =
            KDA_BWD_WORKSPACE_SLOT_COUNT * tiling_.workspaceSlotSize;

        const uint64_t userBytes = static_cast<uint64_t>(blockDim_) *
                                   static_cast<uint64_t>(tiling_.workspaceCoreSize);
        workspaceSize_ = ctx_.systemWorkspaceSize + userBytes;
        return ge::GRAPH_SUCCESS;
    }

    ChunkKdaBwdIntraTilingContext &ctx_;
    ChunkKdaBwdIntraTilingData &tiling_;
    uint32_t blockDim_ = 1;
    uint64_t tilingKey_ = 0;
    size_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_INTRA_TILING_PROCESSOR_H
