/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_PREPARE_TILING_PROCESSOR_H
#include <algorithm>
#include <cmath>
#include <limits>
#include <exe_graph/runtime/storage_shape.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "../chunk_kda_bwd_prepare_validation.h"
#include "../../op_kernel/chunk_kda_bwd_prepare_struct.h"
#include "../../op_kernel/chunk_kda_bwd_prepare_tiling_key.h"
namespace optiling {
struct ChunkKdaBwdPrepareTilingContext {
    const char *nodeName;
    const gert::StorageShape *aqkShape, *vNewShape, *dOShape, *hShape;
    const gert::StorageShape *cuSeqlensShape, *chunkIndicesShape;
    ge::DataType aqkDataType, vNewDataType, dODataType, hDataType;
    double scale;
    int64_t chunkSize;
    bool stateVFirst;
    const char *layout;
    uint32_t aicCoreNum;
    size_t sysWorkspaceSize;
    bool isA5 = false;
};
class ChunkKdaBwdPrepareTilingProcessor {
public:
    ChunkKdaBwdPrepareTilingProcessor(ChunkKdaBwdPrepareTilingContext &ctx,
                                     KDA::ChunkKdaBwdPrepareTilingData &tiling)
        : ctx_(ctx), tiling_(tiling) {}
    ge::graphStatus Process()
    {
        if (!ctx_.aqkShape || !ctx_.vNewShape || !ctx_.dOShape || !ctx_.hShape ||
            ctx_.aqkDataType != ge::DT_BF16 || ctx_.vNewDataType != ge::DT_BF16 ||
            ctx_.dODataType != ge::DT_BF16 || ctx_.hDataType != ge::DT_BF16 ||
            ctx_.chunkSize != 64 || !std::isfinite(ctx_.scale) ||
            !std::isfinite(static_cast<float>(ctx_.scale)) ||
            !KDA::ParseLayout(ctx_.layout, tiling_.inputLayout)) return Fail("dtype/layout/attributes");
        const bool rank3 = KDA::Rank3Layout(tiling_.inputLayout);
        const bool variable = ctx_.cuSeqlensShape != nullptr;
        if ((ctx_.cuSeqlensShape != nullptr) != variable ||
            (ctx_.chunkIndicesShape != nullptr) != variable) return Fail("metadata presence");
        KDA::TokenShape a, v, o;
        if (!KDA::ParseTokenShape(ctx_.aqkShape->GetStorageShape(), tiling_.inputLayout, a) ||
            !KDA::ParseTokenShape(ctx_.vNewShape->GetStorageShape(), tiling_.inputLayout, v) ||
            !KDA::ParseTokenShape(ctx_.dOShape->GetStorageShape(), tiling_.inputLayout, o) ||
            a.D != 64 || v.D != 128 || o.D != 128 || a.B != v.B || a.B != o.B ||
            a.H != v.H || a.H != o.H || a.T != v.T || a.T != o.T) return Fail("token shapes");
        if (variable && a.B != 1) return Fail("variable lengths require B=1");
        if (ctx_.isA5 && (KDA::SequenceMajorLayout(tiling_.inputLayout) || variable != rank3))
            return Fail("A5 currently supports fixed BNSD and variable NTD only");
        // CATLASS A5 TileCopy currently narrows the GM row stride to uint32_t.
        if (ctx_.isA5 && KDA::SequenceMajorLayout(tiling_.inputLayout) &&
            a.H > std::numeric_limits<uint32_t>::max() / 128) return Fail("A5 token row stride");
        auto h = ctx_.hShape->GetStorageShape();
        if (h.GetDimNum() != (rank3 ? 4 : 5)) return Fail("state rank");
        tiling_.B = a.B; tiling_.NV = a.H; tiling_.T = a.T;
        tiling_.K = 128; tiling_.V = 128; tiling_.chunkSize = 64;
        tiling_.isVariable = variable; tiling_.stateVFirst = ctx_.stateVFirst;
        tiling_.scale = static_cast<float>(ctx_.scale);
        tiling_.denseChunkNum = variable ? 0 : (a.T + 63) / 64;
        const int64_t chunks = h.GetDim(rank3 ? 0 : 1);
        const size_t stateBase = rank3 ? 1 : 2;
        if (chunks <= 0 || h.GetDim(stateBase) != a.H ||
            h.GetDim(stateBase + 1) != 128 || h.GetDim(stateBase + 2) != 128 ||
            (!rank3 && h.GetDim(0) != a.B) ||
            (!variable && chunks != tiling_.denseChunkNum)) return Fail("state shapes");
        if (chunks > std::numeric_limits<int64_t>::max() / a.B) return Fail("chunk overflow");
        tiling_.totalChunkNum = chunks * a.B;
        tiling_.seqNum = a.B;
        if (variable) {
            auto cu = ctx_.cuSeqlensShape->GetStorageShape();
            auto indices = ctx_.chunkIndicesShape->GetStorageShape();
            if (cu.GetDimNum() != 1 || indices.GetDimNum() != 1 ||
                cu.GetDim(0) < 2 || cu.GetDim(0) > 1025 ||
                chunks > std::numeric_limits<int64_t>::max() / 2 ||
                indices.GetDim(0) != 2 * chunks) return Fail("metadata shapes");
            tiling_.seqNum = cu.GetDim(0) - 1;
        }
        tiling_.chunkTaskNum = tiling_.totalChunkNum;
        tiling_.headWindowNum = a.H / 4 + (a.H % 4 != 0);
        if (tiling_.chunkTaskNum > std::numeric_limits<int64_t>::max() / tiling_.headWindowNum ||
            tiling_.totalChunkNum > std::numeric_limits<int64_t>::max() / a.H / (128 * 128 * 2)) return Fail("work overflow");
        tiling_.workTaskNum = tiling_.chunkTaskNum * tiling_.headWindowNum;
        if (ctx_.aicCoreNum == 0) return Fail("no AIC cores");
        blockDim_ = static_cast<uint32_t>(std::min<int64_t>(ctx_.aicCoreNum, tiling_.workTaskNum));
        tilingKey_ = GET_TPL_TILING_KEY(tiling_.inputLayout, tiling_.isVariable);
        const size_t scratchSize = ctx_.isA5 ? 0 : static_cast<size_t>(blockDim_) * 65536;
        workspaceSize_ = ctx_.sysWorkspaceSize + scratchSize;
        OP_LOGD(ctx_.nodeName, "layout=%u variable=%u stateVFirst=%u key=%lu blocks=%u userScratch=%lu",
                tiling_.inputLayout, tiling_.isVariable, tiling_.stateVFirst, tilingKey_,
                blockDim_, scratchSize);
        return ge::GRAPH_SUCCESS;
    }
    uint32_t GetBlockDim() const { return blockDim_; }
    uint64_t GetTilingKey() const { return tilingKey_; }
    size_t GetWorkspaceSize() const { return workspaceSize_; }
private:
    ge::graphStatus Fail(const char *reason)
    {
        OP_LOGE(ctx_.nodeName, "invalid KernelA %s: layout=%s scale=%g chunk=%ld cores=%u cu=%u indices=%u",
                reason, ctx_.layout ? ctx_.layout : "null", ctx_.scale, ctx_.chunkSize,
                ctx_.aicCoreNum, ctx_.cuSeqlensShape != nullptr, ctx_.chunkIndicesShape != nullptr);
        return ge::GRAPH_FAILED;
    }
    ChunkKdaBwdPrepareTilingContext &ctx_;
    KDA::ChunkKdaBwdPrepareTilingData &tiling_;
    uint32_t blockDim_ = 0;
    uint64_t tilingKey_ = 0;
    size_t workspaceSize_ = 0;
};
}
#endif
