#include "op_common/log/log.h"
#include "chunk_kda_bwd_finalize_tiling.h"

#include <cmath>
#include <cstring>
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"
#include "chunk_kda_bwd_finalize_arch22.h"

namespace optiling {

namespace {
ge::graphStatus ValidateFinalize(
    gert::TilingContext *context, const ChunkKdaBwdFinalizeTilingContext &ctx,
    KDA::ChunkKdaBwdFinalizeTilingData &shape, bool arch22,
    bool useBetaSigmoid, bool allowNegEigval, const char *outputLayout,
    bool gateOutputHeadMajor)
{
    if (allowNegEigval && !useBetaSigmoid) {
        OP_LOGE(ctx.nodeName, "allowNegEigval=true requires useBetaSigmoidInKernel=true");
        return ge::GRAPH_FAILED;
    }
    if (!arch22 && useBetaSigmoid) {
        OP_LOGE(ctx.nodeName,
                "A5 finalize: use_beta_sigmoid_in_kernel=true is reserved but not supported");
        return ge::GRAPH_FAILED;
    }
    auto fail = [&]() {
        OP_LOGE(ctx.nodeName, "invalid finalize shape, dtype, layout or metadata");
        return ge::GRAPH_FAILED;
    };
    if (arch22 && gateOutputHeadMajor) {
        OP_LOGE_WITH_INVALID_ATTR(ctx.nodeName, "gate_output_head_major", "true",
            "false on A2/A3 because the arch22 finalize kernel supports only token-major gate outputs");
        return ge::GRAPH_FAILED;
    }
    const bool gateTokenMajor = !gateOutputHeadMajor;
    const bool variable = shape.isVariable != 0;
    const size_t rank = ctx.q->GetStorageShape().GetDimNum();
    const size_t base = rank == 3 ? 0 : 1;
    if (shape.NV > UINT32_MAX / (128 * sizeof(float)) ||
        static_cast<uint64_t>(shape.B) > INT64_MAX / 128 / sizeof(float) /
            static_cast<uint64_t>(shape.NV) / static_cast<uint64_t>(shape.T)) return fail();
    const bool headMajor = std::strcmp(outputLayout, rank == 3 ? "NTD" : "BNSD") == 0;
    const bool tokenMajor = std::strcmp(outputLayout, rank == 3 ? "TND" : "BSND") == 0;
    if (!headMajor && !tokenMajor) return fail();
    auto tokenShape = [&](const gert::StorageShape *storage, int64_t width, bool tm) {
        if (!storage || storage->GetStorageShape().GetDimNum() != rank) return false;
        const auto &s = storage->GetStorageShape();
        return (base == 0 || s.GetDim(0) == shape.B) &&
            s.GetDim(base) == (tm ? shape.T : shape.NV) &&
            s.GetDim(base + 1) == (tm ? shape.NV : shape.T) && s.GetDim(base + 2) == width;
    };
    auto scalarShape = [&](const gert::StorageShape *storage, bool tm) {
        if (!storage || storage->GetStorageShape().GetDimNum() != rank - 1) return false;
        const auto &s = storage->GetStorageShape();
        return (base == 0 || s.GetDim(0) == shape.B) && s.GetDim(base) == (tm ? shape.T : shape.NV) &&
            s.GetDim(base + 1) == (tm ? shape.NV : shape.T);
    };
    for (size_t i : {INPUT_Q, INPUT_K, INPUT_V, INPUT_GK, INPUT_V_NEW, INPUT_DV_SCAN, INPUT_DQ_RAW}) {
        if (!tokenShape(context->GetRequiredInputShape(i), 128, false)) return fail();
    }
    for (size_t i : {INPUT_AKK, INPUT_D_AQK}) {
        if (!tokenShape(context->GetRequiredInputShape(i), 64, false)) return fail();
    }
    if (!scalarShape(ctx.beta, false) ||
        (ctx.qRstd && (!scalarShape(ctx.qRstd, false) || !scalarShape(ctx.kRstd, false)))) return fail();
    if (variable) {
        const auto *cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS);
        const auto *indicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES);
        if (!cuTensor || !indicesTensor ||
            cuTensor->GetDataType() != ge::DT_INT64 || indicesTensor->GetDataType() != ge::DT_INT64) return fail();
        const auto *cu = cuTensor->GetData<int64_t>();
        const auto *indices = indicesTensor->GetData<int64_t>();
        const int64_t sequences = shape.seqNum;
        if (!cu || !indices || cu[0] != 0 || cu[sequences] != shape.T) return fail();
        int64_t task = 0;
        for (int64_t seq = 0; seq < sequences; ++seq) {
            if (cu[seq] < 0 || cu[seq + 1] < cu[seq] || cu[seq + 1] > shape.T) return fail();
            const int64_t count = (cu[seq + 1] - cu[seq] + 64 - 1) / 64;
            for (int64_t chunk = 0; chunk < count; ++chunk, ++task) {
                if (task >= shape.totalChunkNum || indices[2 * task] != seq || indices[2 * task + 1] != chunk) return fail();
            }
        }
        if (task != shape.totalChunkNum) return fail();
    }
    for (size_t i : {INPUT_Q, INPUT_K, INPUT_V, INPUT_AKK, INPUT_V_NEW, INPUT_H, INPUT_DH, INPUT_DV_SCAN}) {
        const auto *d = context->GetInputDesc(i);
        if (!d || d->GetDataType() != ge::DT_BF16) return fail();
    }
    for (size_t i : {INPUT_GK, INPUT_D_AQK, INPUT_DQ_RAW, INPUT_Q_RSTD, INPUT_K_RSTD}) {
        const auto *d = context->GetOptionalInputDesc(i);
        if (d && d->GetDataType() != ge::DT_FLOAT) return fail();
    }
    const auto betaType = context->GetInputDesc(INPUT_BETA)->GetDataType();
    const auto logType = context->GetInputDesc(INPUT_A_LOG)->GetDataType();
    if ((betaType != ge::DT_FLOAT && (useBetaSigmoid || betaType != ge::DT_BF16)) ||
        (logType != ge::DT_FLOAT && (arch22 || logType != ge::DT_BF16))) return fail();
    if (ctx.useGateInKernel) {
        for (size_t i : {INPUT_RAW_G}) {
            const auto *d = context->GetOptionalInputDesc(i);
            if (!d || d->GetDataType() != ge::DT_FLOAT) return fail();
        }
        const auto *a = context->GetOptionalInputShape(INPUT_A_LOG);
        const auto *bias = context->GetOptionalInputShape(INPUT_DT_BIAS);
        if (!tokenShape(context->GetOptionalInputShape(INPUT_RAW_G), 128, false) ||
            !a || a->GetStorageShape().GetShapeSize() != shape.NV ||
            (!bias || (bias->GetStorageShape().GetShapeSize() != shape.NV * 128 ||
                      context->GetOptionalInputDesc(INPUT_DT_BIAS)->GetDataType() != ge::DT_FLOAT))) return fail();
    }
    for (size_t i = 0; i < 3; ++i) {
        if (!tokenShape(context->GetOutputShape(i), 128, tokenMajor)) return fail();
    }
    if (!scalarShape(context->GetOutputShape(3), gateTokenMajor) ||
        !tokenShape(context->GetOutputShape(4), 128, gateTokenMajor)) return fail();
    const auto qType = context->GetOutputDesc(0)->GetDataType();
    if (qType != ge::DT_BF16 || context->GetOutputDesc(1)->GetDataType() != qType ||
        context->GetOutputDesc(2)->GetDataType() != ge::DT_BF16) return fail();
    if (context->GetOutputDesc(3)->GetDataType() != betaType) return fail();
    for (size_t i = 4; i < 7; ++i) {
        if (context->GetOutputDesc(i)->GetDataType() != ge::DT_FLOAT) return fail();
    }
    for (size_t i = 0; i < 2; ++i) {
        const auto &input = context->GetRequiredInputShape(INPUT_A_LOG + i)->GetStorageShape();
        const auto &output = context->GetOutputShape(5 + i)->GetStorageShape();
        if (input.GetDimNum() != output.GetDimNum()) return fail();
        for (size_t axis = 0; axis < input.GetDimNum(); ++axis) {
            if (input.GetDim(axis) != output.GetDim(axis)) return fail();
        }
    }
    shape.outputTokenMajor = tokenMajor;
    shape.outputGateTokenMajor = gateTokenMajor;
    shape.betaMode = useBetaSigmoid ? (allowNegEigval ? 2 : 1) : 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace


ge::graphStatus Tiling4ChunkKdaBwdFinalize(gert::TilingContext *context)
{
    KDA::ChunkKdaBwdFinalizeTilingData shape{};
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const double *scale = attrs->GetAttrPointer<double>(ATTR_SCALE);
    const double *lowerBound = attrs->GetAttrPointer<double>(ATTR_LOWER_BOUND);
    const int64_t *chunkSize = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const bool *safeGate = attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE);
    const bool *useGate = attrs->GetAttrPointer<bool>(ATTR_USE_GATE_IN_KERNEL);
    const bool *useExp2 = attrs->GetAttrPointer<bool>(ATTR_USE_EXP2);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(ATTR_STATE_V_FIRST);
    const bool *useBetaSigmoid = attrs->GetAttrPointer<bool>(ATTR_USE_BETA_SIGMOID);
    const bool *allowNegEigval = attrs->GetAttrPointer<bool>(ATTR_ALLOW_NEG_EIGVAL);
    const char *outputLayout = attrs->GetStr(ATTR_OUTPUT_LAYOUT);
    const bool *gateOutputHeadMajor = attrs->GetAttrPointer<bool>(ATTR_GATE_OUTPUT_HEAD_MAJOR);
    OP_CHECK_NULL_WITH_CONTEXT(context, useBetaSigmoid);
    OP_CHECK_NULL_WITH_CONTEXT(context, allowNegEigval);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputLayout);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateOutputHeadMajor);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, lowerBound);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, safeGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, useGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, useExp2);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateVFirst);
    if (!std::isfinite(static_cast<float>(*scale)) || !std::isfinite(static_cast<float>(*lowerBound)) ||
        *lowerBound < -5.0 || static_cast<float>(*lowerBound) >= 0.0f) {
        OP_LOGE(context->GetNodeName(), "scale/lower_bound must be finite");
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicCoreNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    if (aicCoreNum == 0) {
        // The A5 aclnn direct-launch context may omit platform info.  This
        // checkpoint is A5-only and must still expose all 28 physical AICs.
        aicCoreNum = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950 ? 28U : 1U;
        OP_LOGW(context->GetNodeName(),
                "platform info did not report AIC count; fallback to %u", aicCoreNum);
    }
    ChunkKdaBwdFinalizeTilingContext ctx{
        context->GetNodeName(),
        context->GetRequiredInputShape(INPUT_Q),
        context->GetRequiredInputShape(INPUT_K),
        context->GetRequiredInputShape(INPUT_V),
        context->GetRequiredInputShape(INPUT_GK),
        context->GetRequiredInputShape(INPUT_BETA),
        context->GetRequiredInputShape(INPUT_AKK),
        context->GetRequiredInputShape(INPUT_H),
        context->GetRequiredInputShape(INPUT_DH),
        context->GetRequiredInputShape(INPUT_DV_SCAN),
        context->GetOptionalInputShape(INPUT_Q_RSTD),
        context->GetOptionalInputShape(INPUT_K_RSTD),
        context->GetOptionalInputShape(INPUT_CU_SEQLENS),
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES),
        *scale, *lowerBound, *chunkSize, *safeGate, *useGate, *useExp2, *stateVFirst,
        aicCoreNum,
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize())};
    auto fail = [&]() { return ge::GRAPH_FAILED; };
    // Check dimensions before the shared shape processor multiplies them.
    if (!ctx.k || !ctx.h) return fail();
    const auto &kShape = ctx.k->GetStorageShape();
    const auto &hShape = ctx.h->GetStorageShape();
    uint64_t elements = 1;
    for (size_t i = 0; i < kShape.GetDimNum(); ++i) {
        const auto dim = kShape.GetDim(i);
        if (dim <= 0 || static_cast<uint64_t>(dim) > (INT64_MAX - 512) / 4U / elements) return fail();
        elements *= static_cast<uint64_t>(dim);
    }
    elements = 1;
    for (size_t i = 0; i < hShape.GetDimNum(); ++i) {
        const auto dim = hShape.GetDim(i);
        if (dim <= 0 || static_cast<uint64_t>(dim) > (INT64_MAX - 512) / 4U / elements) return fail();
        elements *= static_cast<uint64_t>(dim);
    }
    ChunkKdaBwdFinalizeTilingProcessor processor(ctx, shape);
    OP_CHECK_IF(processor.Process() != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    const bool arch22 = platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND950;
    if (ValidateFinalize(context, ctx, shape, arch22, *useBetaSigmoid,
                         *allowNegEigval, outputLayout, *gateOutputHeadMajor) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (arch22) return TilingFinalizeArch22(context, ctx, shape);
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdFinalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = shape;
    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    rawTiling->SetDataSize(sizeof(KDA::ChunkKdaBwdFinalizeTilingData) + 8);
    context->SetTilingKey(processor.GetTilingKey());
    context->SetBlockDim(processor.GetBlockDim());
    context->GetWorkspaceSizes(1)[0] = processor.GetWorkspaceSize();
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkKdaBwdFinalize(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwdFinalize)
    .Tiling(Tiling4ChunkKdaBwdFinalize)
    .TilingParse<ChunkKdaBwdFinalizeCompileInfo>(TilingPrepareForChunkKdaBwdFinalize);

} // namespace optiling
