// Copyright (c) 2026 Tianjin University, Ltd.
// BSD-3-Clause. A2 独立 Prepare 包装，只消费既有公开端口。
#ifndef GDN_PREPARE_HOST_ARCH22_TILING_H
#define GDN_PREPARE_HOST_ARCH22_TILING_H
#include "../../op_kernel/arch22/chunk_gated_delta_rule_fwd_prepare_standalone_tiling.h"
#include "securec.h"
#include <initializer_list>

namespace optiling {
inline bool A2PrepareShapeEquals(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != dims.size()) return false;
    size_t axis = 0;
    for (int64_t dim : dims) if (shape->GetStorageShape().GetDim(axis++) != dim) return false;
    return true;
}

inline ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdPrepareArch22(gert::TilingContext *context)
{
    const auto *qShape = context->GetRequiredInputShape(PREPARE_INPUT_Q);
    const auto *vShape = context->GetRequiredInputShape(PREPARE_INPUT_V);
    if (qShape == nullptr || vShape == nullptr || qShape->GetStorageShape().GetDimNum() != 4 ||
        vShape->GetStorageShape().GetDimNum() != 4) return ge::GRAPH_FAILED;
    const auto &q = qShape->GetStorageShape();
    const auto &v = vShape->GetStorageShape();
    const int64_t batch = q.GetDim(0), hk = q.GetDim(1), tokens = q.GetDim(2);
    const int64_t hv = v.GetDim(1), valueDim = v.GetDim(3);
    if (batch <= 0 || hk <= 0 || hv <= 0 || tokens <= 0 || q.GetDim(3) != 128 ||
        !A2PrepareShapeEquals(context->GetRequiredInputShape(PREPARE_INPUT_K), {batch, hk, tokens, 128}) ||
        !A2PrepareShapeEquals(vShape, {batch, hv, tokens, valueDim}) ||
        !A2PrepareShapeEquals(context->GetRequiredInputShape(PREPARE_INPUT_G), {batch, hv, tokens}) ||
        !A2PrepareShapeEquals(context->GetRequiredInputShape(PREPARE_INPUT_BETA), {batch, hv, tokens}))
        return ge::GRAPH_FAILED;

    const auto *attrs = context->GetAttrs();
    auto flag = [attrs](size_t index, bool fallback) {
        const bool *value = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(index);
        return value == nullptr ? fallback : *value;
    };
    const auto *btAttr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(PREPARE_ATTR_CHUNK_SIZE);
    const int64_t bt = btAttr == nullptr ? 64 : *btAttr;
    const bool norm = flag(PREPARE_ATTR_USE_QK_L2NORM, false);
    const bool sigmoid = flag(PREPARE_ATTR_USE_BETA_SIGMOID, false);
    const bool neg = flag(PREPARE_ATTR_ALLOW_NEG_EIGVAL, false);
    const bool exp2 = flag(PREPARE_ATTR_USE_EXP2, false);
    const bool outputA = flag(PREPARE_ATTR_OUTPUT_A, true);
    const bool hasALog = context->GetOptionalInputDesc(PREPARE_INPUT_A_LOG) != nullptr;
    const bool hasBias = context->GetOptionalInputDesc(PREPARE_INPUT_DT_BIAS) != nullptr;
    const bool gate = flag(PREPARE_ATTR_USE_GATE, false) || hasALog;
    if ((gate && !hasALog) || (hasBias && !gate) || (neg && !sigmoid)) return ge::GRAPH_FAILED;

    const auto *qDesc = context->GetInputDesc(PREPARE_INPUT_Q);
    const auto *kDesc = context->GetInputDesc(PREPARE_INPUT_K);
    const auto *vDesc = context->GetInputDesc(PREPARE_INPUT_V);
    const auto *gDesc = context->GetInputDesc(PREPARE_INPUT_G);
    const auto *betaDesc = context->GetInputDesc(PREPARE_INPUT_BETA);
    if (!qDesc || !kDesc || !vDesc || !gDesc || !betaDesc) return ge::GRAPH_FAILED;
    const auto dtype = qDesc->GetDataType();
    if ((dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16) || kDesc->GetDataType() != dtype ||
        vDesc->GetDataType() != dtype || gDesc->GetDataType() != ge::DT_FLOAT ||
        betaDesc->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    for (size_t input : {PREPARE_INPUT_A_LOG, PREPARE_INPUT_DT_BIAS}) {
        const auto *desc = context->GetOptionalInputDesc(input);
        if (desc && (desc->GetDataType() != ge::DT_FLOAT ||
                     !A2PrepareShapeEquals(context->GetOptionalInputShape(input), {hv})))
            return ge::GRAPH_FAILED;
    }

    const auto *cu = context->GetOptionalInputShape(PREPARE_INPUT_CU_SEQLENS);
    const auto *indices = context->GetOptionalInputShape(PREPARE_INPUT_CHUNK_INDICES);
    if ((cu != nullptr) != (indices != nullptr) || (bt != 64 && bt != 128)) return ge::GRAPH_FAILED;
    const bool varlen = cu != nullptr;
    uint64_t nt = GdnUnifiedPrepare::CeilDiv(static_cast<uint64_t>(tokens), static_cast<uint64_t>(bt));
    if (varlen) {
        const auto *cuDesc = context->GetOptionalInputDesc(PREPARE_INPUT_CU_SEQLENS);
        const auto *idxDesc = context->GetOptionalInputDesc(PREPARE_INPUT_CHUNK_INDICES);
        const auto &cs = cu->GetStorageShape();
        const auto &is = indices->GetStorageShape();
        if (!cuDesc || !idxDesc || cuDesc->GetDataType() != ge::DT_INT64 ||
            idxDesc->GetDataType() != ge::DT_INT64 || cs.GetDimNum() != 1 || cs.GetDim(0) < 2)
            return ge::GRAPH_FAILED;
        if (is.GetDimNum() == 1 && is.GetDim(0) > 0 && is.GetDim(0) % 2 == 0)
            nt = static_cast<uint64_t>(is.GetDim(0) / 2);
        else if (is.GetDimNum() == 2 && is.GetDim(0) > 0 && is.GetDim(1) == 2)
            nt = static_cast<uint64_t>(is.GetDim(0));
        else return ge::GRAPH_FAILED;
    }
    auto output = [context](size_t index, ge::DataType type, std::initializer_list<int64_t> shape) {
        const auto *desc = context->GetOutputDesc(index);
        return desc && desc->GetDataType() == type && A2PrepareShapeEquals(context->GetOutputShape(index), shape);
    };
    if (!output(PREPARE_OUTPUT_G, ge::DT_FLOAT, {batch, hv, tokens}) ||
        !output(PREPARE_OUTPUT_W, dtype, {batch, hv, tokens, 128}) ||
        !output(PREPARE_OUTPUT_U, dtype, {batch, hv, tokens, valueDim}) ||
        (outputA && !output(PREPARE_OUTPUT_A, dtype, {batch, hv, tokens, bt})) ||
        (norm && (!output(PREPARE_OUTPUT_Q_HAT, dtype, {batch, hk, tokens, 128}) ||
                  !output(PREPARE_OUTPUT_K_HAT, dtype, {batch, hk, tokens, 128}) ||
                  !output(PREPARE_OUTPUT_Q_RSTD, ge::DT_FLOAT, {batch, hk, tokens}) ||
                  !output(PREPARE_OUTPUT_K_RSTD, ge::DT_FLOAT, {batch, hk, tokens}))) ||
        (sigmoid && !output(PREPARE_OUTPUT_BETA_EFF, ge::DT_FLOAT, {batch, hv, tokens})))
        return ge::GRAPH_FAILED;

    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    GdnUnifiedPrepare::StandaloneTiling tiling{};
    if (!GdnUnifiedPrepare::MakeStandalonePlan(batch, hk, hv, tokens, bt, valueDim, nt, varlen,
            platform.GetCoreNumAic(), platform.GetCoreNumAiv(), outputA, tiling) || tiling.cores > UINT32_MAX)
        return ge::GRAPH_FAILED;
    tiling.useQkNorm = norm; tiling.useGate = gate; tiling.useDtBias = hasBias;
    tiling.useBetaSigmoid = sigmoid; tiling.allowNegEigval = neg; tiling.useExp2 = exp2;
    uint64_t workspaceBytes = 0;
    if (!GdnUnifiedPrepare::StandaloneAdd(platform.GetLibApiWorkSpaceSize(), tiling.userWorkspaceBytes, workspaceBytes))
        return ge::GRAPH_FAILED;
    auto *raw = context->GetRawTilingData();
    auto *workspaceSizes = context->GetWorkspaceSizes(1);
    if (!raw || !workspaceSizes || raw->GetCapacity() < sizeof(tiling) ||
        memcpy_s(raw->GetData(), raw->GetCapacity(), &tiling, sizeof(tiling)) != EOK)
        return ge::GRAPH_FAILED;
    raw->SetDataSize(sizeof(tiling));
    context->SetTilingKey(22);
    context->SetBlockDim(static_cast<uint32_t>(tiling.cores));
    workspaceSizes[0] = static_cast<size_t>(workspaceBytes);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
#endif
