/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "chunk_kda_bwd_prepare_tiling.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus Tiling4ChunkKdaBwdPrepare(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdPrepareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto *aqkDesc = context->GetInputDesc(KDA_PREPARE_INPUT_AQK);
    auto *vNewDesc = context->GetInputDesc(KDA_PREPARE_INPUT_V_NEW);
    auto *dODesc = context->GetInputDesc(KDA_PREPARE_INPUT_D_O);
    auto *hDesc = context->GetInputDesc(KDA_PREPARE_INPUT_H);
    OP_CHECK_NULL_WITH_CONTEXT(context, aqkDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vNewDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dODesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, hDesc);

    const float *scale = attrs->GetAttrPointer<float>(KDA_PREPARE_ATTR_SCALE);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(KDA_PREPARE_ATTR_CHUNK_SIZE);
    const char *layout = attrs->GetStr(KDA_PREPARE_ATTR_LAYOUT);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(KDA_PREPARE_ATTR_STATE_V_FIRST);
    const bool *doTokenMajorPtr = attrs->GetAttrPointer<bool>(KDA_PREPARE_ATTR_DO_TOKEN_MAJOR);
    const bool doTokenMajor = doTokenMajorPtr != nullptr && *doTokenMajorPtr;
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    if (!std::isfinite(*scale)) {
        OP_LOGE(context->GetNodeName(), "scale must be finite");
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto *cuShape = context->GetOptionalInputShape(KDA_PREPARE_INPUT_CU_SEQLENS);
    const auto *indicesShape = context->GetOptionalInputShape(KDA_PREPARE_INPUT_CHUNK_INDICES);
    // Opaque ACLNN may represent an absent optional input with an empty shape.
    if (cuShape && cuShape->GetStorageShape().GetDimNum() == 0) cuShape = nullptr;
    if (indicesShape && indicesShape->GetStorageShape().GetDimNum() == 0) indicesShape = nullptr;
    if (doTokenMajor) {
        if (!layout || (std::strcmp(layout, "BNSD") != 0 && std::strcmp(layout, "NTD") != 0)) {
            OP_LOGE(context->GetNodeName(), "do_token_major requires internal BNSD or NTD layout");
            return ge::GRAPH_FAILED;
        }
        const auto &v = context->GetRequiredInputShape(KDA_PREPARE_INPUT_V_NEW)->GetStorageShape();
        const auto &o = context->GetRequiredInputShape(KDA_PREPARE_INPUT_D_O)->GetStorageShape();
        const size_t rank = v.GetDimNum(), base = rank == 3 ? 0 : 1;
        if ((rank != 3 && rank != 4) || o.GetDimNum() != rank ||
            (base && v.GetDim(0) != o.GetDim(0)) || v.GetDim(base) != o.GetDim(base + 1) ||
            v.GetDim(base + 1) != o.GetDim(base) || v.GetDim(base + 2) != o.GetDim(base + 2)) {
            OP_LOGE_WITH_INVALID_INPUT_SHAPE(context->GetNodeName(), KDA_PREPARE_INPUT_D_O,
                "head-major dO or mismatched dimensions", "token-major dO matching vNew");
            return ge::GRAPH_FAILED;
        }
    }
    const bool isA5 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    OP_CHECK_IF(doTokenMajor && !isA5,
                OP_LOGE(context->GetNodeName(), "do_token_major is only supported on Ascend950"),
                return ge::GRAPH_FAILED);
    ChunkKdaBwdPrepareTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(KDA_PREPARE_INPUT_AQK),
        context->GetRequiredInputShape(KDA_PREPARE_INPUT_V_NEW),
        context->GetRequiredInputShape(doTokenMajor ? KDA_PREPARE_INPUT_V_NEW : KDA_PREPARE_INPUT_D_O),
        context->GetRequiredInputShape(KDA_PREPARE_INPUT_H),
        cuShape, indicesShape,
        aqkDesc->GetDataType(), vNewDesc->GetDataType(), dODesc->GetDataType(), hDesc->GetDataType(),
        *scale, *chunkSize, stateVFirst != nullptr && *stateVFirst,
        layout, static_cast<uint32_t>(platform.GetCoreNumAic()),
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize()),
        isA5,
    };
    ChunkKdaBwdPrepareTilingProcessor processor(ctx, *tiling);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(doTokenMajor && isA5 &&
                    tiling->NV > std::numeric_limits<uint32_t>::max() / tiling->V,
                OP_LOGE_WITH_INVALID_INPUT_SHAPE(context->GetNodeName(), KDA_PREPARE_INPUT_D_O,
                    "token-major row stride exceeds uint32_t", "NV*V within uint32_t"),
                return ge::GRAPH_FAILED);
    tiling->doTokenMajor = doTokenMajor;
    if (isA5) {
        auto *rawTiling = context->GetRawTilingData();
        OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
        rawTiling->SetDataSize(sizeof(KDA::ChunkKdaBwdPrepareTilingData) + 8);
    }
    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaBwdPrepare(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdPrepare)
    .Tiling(Tiling4ChunkKdaBwdPrepare)
    .TilingParse<ChunkKdaBwdPrepareCompileInfo>(TilingPrepareForChunkKdaBwdPrepare);

} // namespace optiling
