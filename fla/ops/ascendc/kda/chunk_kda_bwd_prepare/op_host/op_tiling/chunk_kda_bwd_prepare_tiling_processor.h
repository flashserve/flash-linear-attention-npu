/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_TILING_PROCESSOR_H
#define CHUNK_KDA_BWD_PREPARE_TILING_PROCESSOR_H
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include "graph/utils/type_utils.h"
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
    bool doTokenMajor = false;
};
class ChunkKdaBwdPrepareTilingProcessor {
public:
    ChunkKdaBwdPrepareTilingProcessor(ChunkKdaBwdPrepareTilingContext &ctx,
                                     KDA::ChunkKdaBwdPrepareTilingData &tiling)
        : ctx_(ctx), tiling_(tiling) {}
    ge::graphStatus Process()
    {
        if (!ctx_.aqkShape || !ctx_.vNewShape || !ctx_.dOShape || !ctx_.hShape) {
            return FailNull(!ctx_.aqkShape ? "aqk"
                            : (!ctx_.vNewShape ? "v_new"
                               : (!ctx_.dOShape ? "d_o" : "h")));
        }
        if (ctx_.aqkDataType != ge::DT_BF16 || ctx_.vNewDataType != ge::DT_BF16 ||
            ctx_.dODataType != ge::DT_BF16 || ctx_.hDataType != ge::DT_BF16) {
            return FailDtypes(
                "aqk,v_new,d_o,h",
                std::string("aqk=") + ge::TypeUtils::DataTypeToSerialString(ctx_.aqkDataType) +
                    ",v_new=" + ge::TypeUtils::DataTypeToSerialString(ctx_.vNewDataType) +
                    ",d_o=" + ge::TypeUtils::DataTypeToSerialString(ctx_.dODataType) +
                    ",h=" + ge::TypeUtils::DataTypeToSerialString(ctx_.hDataType),
                "must all be bfloat16");
        }
        if (ctx_.chunkSize != 64) {
            return FailValue("chunkSize", std::to_string(ctx_.chunkSize), "must be 64");
        }
        if (!std::isfinite(ctx_.scale) || !std::isfinite(static_cast<float>(ctx_.scale))) {
            return FailValue("scale", std::to_string(ctx_.scale), "must be a finite float32 value");
        }
        if (!KDA::ParseLayout(ctx_.layout, tiling_.inputLayout)) {
            return FailValue("layout", ctx_.layout == nullptr ? "absent" : std::string(ctx_.layout),
                             "must be one of BNSD, BSND, NTD or TND");
        }
        const bool rank3 = KDA::Rank3Layout(tiling_.inputLayout);
        const bool variable = ctx_.cuSeqlensShape != nullptr;
        if ((ctx_.cuSeqlensShape != nullptr) != variable ||
            (ctx_.chunkIndicesShape != nullptr) != variable) {
            return FailNull(variable ? "chunk_indices" : "cu_seqlens");
        }
        const auto vNewShape = ctx_.vNewShape->GetStorageShape();
        const auto dOShape = ctx_.dOShape->GetStorageShape();
        const size_t rank = vNewShape.GetDimNum();
        const size_t base = rank == 3 ? 0 : 1;
        const bool transposedNonSquare =
            (rank == 3 || rank == 4) && dOShape.GetDimNum() == rank &&
            (base == 0 || dOShape.GetDim(0) == vNewShape.GetDim(0)) &&
            dOShape.GetDim(base) == vNewShape.GetDim(base + 1) &&
            dOShape.GetDim(base + 1) == vNewShape.GetDim(base) &&
            dOShape.GetDim(base + 2) == vNewShape.GetDim(base + 2) &&
            vNewShape.GetDim(base) != vNewShape.GetDim(base + 1);
        ctx_.doTokenMajor = ctx_.isA5 && transposedNonSquare &&
            ((tiling_.inputLayout == KDA::BNSD && !variable) ||
             (tiling_.inputLayout == KDA::NTD && variable));
        KDA::TokenShape a, v, o;
        if (!KDA::ParseTokenShape(ctx_.aqkShape->GetStorageShape(), tiling_.inputLayout, a) ||
            !KDA::ParseTokenShape(vNewShape, tiling_.inputLayout, v)) {
            return FailShape("aqk,v_new", "rank or layout parse failed",
                             "token tensors must be 4D BNSD or 3D NTD");
        }
        if (ctx_.doTokenMajor) {
            o = v;
        } else if (!KDA::ParseTokenShape(dOShape, tiling_.inputLayout, o)) {
            return FailShape("d_o", "rank or layout parse failed",
                             "d_o must be 4D BNSD or 3D NTD like v_new");
        }
        if (a.D != 64 || v.D != 128 || o.D != 128 || a.B != v.B || a.B != o.B ||
            a.H != v.H || a.H != o.H || a.T != v.T || a.T != o.T) {
            return FailShape("aqk,v_new,d_o",
                             "aqk(B=" + std::to_string(a.B) + ",H=" + std::to_string(a.H) +
                                 ",T=" + std::to_string(a.T) + ",D=" + std::to_string(a.D) +
                                 "),v_new(B=" + std::to_string(v.B) + ",H=" + std::to_string(v.H) +
                                 ",T=" + std::to_string(v.T) + ",D=" + std::to_string(v.D) +
                                 "),d_o(B=" + std::to_string(o.B) + ",H=" + std::to_string(o.H) +
                                 ",T=" + std::to_string(o.T) + ",D=" + std::to_string(o.D) + ")",
                             "token dims must match, with D=64 for aqk and D=128 for v_new/d_o");
        }
        if (variable && a.B != 1) {
            return FailValue("B", std::to_string(a.B), "variable-length inputs require B=1");
        }
        if (ctx_.isA5 && (KDA::SequenceMajorLayout(tiling_.inputLayout) || variable != rank3))
            return FailCapability("layout",
                                  KDA::SequenceMajorLayout(tiling_.inputLayout) ? "sequence-major" : "batch-major",
                                  "A5 supports fixed BNSD or variable NTD only");
        // CATLASS A5 TileCopy currently narrows the GM row stride to uint32_t.
        if (ctx_.isA5 && (KDA::SequenceMajorLayout(tiling_.inputLayout) || ctx_.doTokenMajor) &&
             a.H > std::numeric_limits<uint32_t>::max() / 128) {
            return FailValue("HK", std::to_string(a.H),
                             "token row stride exceeds the uint32 range on A5");
        }
        auto h = ctx_.hShape->GetStorageShape();
        if (h.GetDimNum() != (rank3 ? 4 : 5)) {
            return FailShapeDim("h", std::to_string(h.GetDimNum()),
                                rank3 ? "4 for NTD layout" : "5 for BNSD layout");
        }
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
            (!variable && chunks != tiling_.denseChunkNum)) {
            return FailShape("h", "chunks=" + std::to_string(chunks) +
                                 ",B=" + std::to_string(a.B) +
                                 ",H=" + std::to_string(a.H) +
                                 ",expected_chunks=" + std::to_string(tiling_.denseChunkNum),
                             "h state shape must match the chunk schedule");
        }
        if (chunks > std::numeric_limits<int64_t>::max() / a.B) {
            return FailValue("chunks,B", std::to_string(chunks) + "," + std::to_string(a.B),
                             "chunk count overflow");
        }
        tiling_.totalChunkNum = chunks * a.B;
        tiling_.seqNum = a.B;
        if (variable) {
            auto cu = ctx_.cuSeqlensShape->GetStorageShape();
            auto indices = ctx_.chunkIndicesShape->GetStorageShape();
            if (cu.GetDimNum() != 1 || indices.GetDimNum() != 1 ||
                cu.GetDim(0) < 2 ||
                chunks > std::numeric_limits<int64_t>::max() / 2 ||
                indices.GetDim(0) != 2 * chunks) {
            return FailShape("cu_seqlens,chunk_indices",
                             "cu=" + std::to_string(cu.GetDim(0)) +
                                 ",chunk_indices=" + std::to_string(indices.GetDim(0)) +
                                 ",expected_chunk_indices=" + std::to_string(2 * chunks),
                             "cu_seqlens/chunk_indices must match the chunk schedule");
        }
            tiling_.seqNum = cu.GetDim(0) - 1;
        }
        tiling_.chunkTaskNum = tiling_.totalChunkNum;
        tiling_.headWindowNum = a.H / 4 + (a.H % 4 != 0);
        if (tiling_.chunkTaskNum > std::numeric_limits<int64_t>::max() / tiling_.headWindowNum ||
            tiling_.totalChunkNum > std::numeric_limits<int64_t>::max() / a.H / (128 * 128 * 2)) {
            return FailValue("workTaskNum", std::to_string(tiling_.chunkTaskNum) + "," +
                                                 std::to_string(tiling_.headWindowNum),
                             "tiling task count overflow");
        }
        tiling_.workTaskNum = tiling_.chunkTaskNum * tiling_.headWindowNum;
        if (ctx_.aicCoreNum == 0) {
            return FailCapability("aic_num", "0", "platform must report at least one AIC core");
        }
        blockDim_ = static_cast<uint32_t>(std::min<int64_t>(ctx_.aicCoreNum, tiling_.workTaskNum));
        tilingKey_ = GET_TPL_TILING_KEY(
            tiling_.inputLayout, tiling_.isVariable,
            static_cast<uint64_t>(ctx_.doTokenMajor ? 1 : 0));
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
    ge::graphStatus FailNull(const char *name)
    {
        OP_LOGE_WITH_INVALID_INPUT(ctx_.nodeName, name);
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus FailValue(const char *name, const std::string &actual, const char *expected)
    {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName, name, actual, expected);
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus FailShape(const char *name, const std::string &actual, const char *reason)
    {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ctx_.nodeName, name, actual, reason);
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus FailShapeDim(const char *name, const std::string &actual, const std::string &expected)
    {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ctx_.nodeName, name, actual, expected);
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus FailDtypes(const char *names, const std::string &actual, const char *expected)
    {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(ctx_.nodeName, names, actual, expected);
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus FailCapability(const char *name, const std::string &actual, const char *reason)
    {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx_.nodeName, name, actual, reason);
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
