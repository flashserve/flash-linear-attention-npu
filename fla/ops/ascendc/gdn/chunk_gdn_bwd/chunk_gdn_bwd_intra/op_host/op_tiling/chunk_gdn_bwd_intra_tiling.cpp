/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "chunk_gdn_bwd_intra_tiling.h"

#include <algorithm>
#include <limits>
#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"
#include "../chunk_gdn_bwd_intra_schedule.h"
#include "../../op_kernel/arch22/chunk_gdn_bwd_intra_config.h"
#include "../../op_kernel/chunk_gdn_bwd_intra_tiling_key.h"

namespace optiling {
namespace {

uint64_t DataTypeKey(ge::DataType dtype, ge::DataType fp32Type,
                     uint64_t lowPrecisionKey, uint64_t fp32Key)
{
    return dtype == fp32Type ? fp32Key : lowPrecisionKey;
}

const char *DtypeName(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16: return "bf16";
        case ge::DT_FLOAT16: return "fp16";
        case ge::DT_FLOAT: return "fp32";
        case ge::DT_DOUBLE: return "fp64";
        case ge::DT_INT8: return "int8";
        case ge::DT_UINT8: return "uint8";
        case ge::DT_INT16: return "int16";
        case ge::DT_UINT16: return "uint16";
        case ge::DT_INT32: return "int32";
        case ge::DT_INT64: return "int64";
        case ge::DT_BOOL: return "bool";
        default: return "unknown";
    }
}

} // namespace

ge::graphStatus Tiling4ChunkGdnBwdIntra(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<GDN::ChunkGdnBwdIntraTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto *scale = attrs->GetAttrPointer<double>(INTRA_SCALE_ATTR_IDX);
    const auto *chunkSize = attrs->GetAttrPointer<int64_t>(INTRA_CHUNK_SIZE_ATTR_IDX);
    const auto *useExp2 = attrs->GetAttrPointer<bool>(INTRA_USE_EXP2_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, useExp2);

    const auto *qShape = context->GetRequiredInputShape(INTRA_Q_IDX);
    const auto *vShape = context->GetRequiredInputShape(INTRA_V_IDX);
    const auto *qDesc = context->GetInputDesc(INTRA_Q_IDX);
    const auto *gDesc = context->GetInputDesc(INTRA_G_IDX);
    const auto *betaDesc = context->GetInputDesc(INTRA_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    const ge::DataType mainDtype = qDesc->GetDataType();
    const ge::DataType gDtype = gDesc->GetDataType();
    const ge::DataType betaDtype = betaDesc->GetDataType();
    OP_CHECK_IF(mainDtype != ge::DT_BF16 && mainDtype != ge::DT_FLOAT16,
                OP_LOGE(context->GetNodeName(),
                        "mainDtype dtype is %s, expect dtype is bf16 or fp16.",
                        DtypeName(mainDtype)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gDtype != ge::DT_BF16 && gDtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(),
                        "gDtype dtype is %s, expect dtype is bf16 or fp32.",
                        DtypeName(gDtype)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaDtype != ge::DT_BF16 && betaDtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(),
                        "betaDtype dtype is %s, expect dtype is bf16 or fp32.",
                        DtypeName(betaDtype)),
                return ge::GRAPH_FAILED);

    const auto qStorage = qShape->GetStorageShape();
    const auto vStorage = vShape->GetStorageShape();
    OP_CHECK_IF(qStorage.GetDimNum() != 4 || vStorage.GetDimNum() != 4 || *chunkSize != 64,
                OP_LOGE(context->GetNodeName(), "q/v rank must be 4 and chunk_size must be 64."),
                return ge::GRAPH_FAILED);
    const int64_t batch = qStorage.GetDim(0);
    const int64_t qkHeads = qStorage.GetDim(1);
    const int64_t seqlen = qStorage.GetDim(2);
    const int64_t keyDim = qStorage.GetDim(3);
    const int64_t valueHeads = vStorage.GetDim(1);
    const int64_t valueDim = vStorage.GetDim(3);
    OP_CHECK_IF(batch <= 0 || seqlen <= 0 || qkHeads <= 0 || valueHeads <= 0,
                OP_LOGE(context->GetNodeName(), "B/T/HK/HV must be positive."),
                return ge::GRAPH_FAILED);
    const int64_t headRatio = valueHeads / qkHeads;
    OP_CHECK_IF(valueHeads % qkHeads != 0 || headRatio > 4,
                OP_LOGE(context->GetNodeName(), "unsupported HV/HK head ratio, must be 1/2/3/4."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDim != 128 || valueDim != 128,
                OP_LOGE(context->GetNodeName(), "K and V dims must be 128."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vStorage.GetDim(0) != batch || vStorage.GetDim(2) != seqlen,
                OP_LOGE(context->GetNodeName(), "v batch and T dims must match q."),
                return ge::GRAPH_FAILED);
    uint64_t tensorBytes = 0;
    OP_CHECK_IF(!IntraCheckedMultiply(batch, valueHeads, tensorBytes) ||
                    !IntraCheckedMultiply(tensorBytes, seqlen, tensorBytes) ||
                    !IntraCheckedMultiply(tensorBytes, 128 * sizeof(uint16_t), tensorBytes) ||
                    tensorBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context->GetNodeName(), "tensor address range overflow."),
                return ge::GRAPH_FAILED);

    const auto *cuTensor = context->GetOptionalInputTensor(INTRA_CU_SEQLENS_IDX);
    const auto *chunkTensor = context->GetOptionalInputTensor(INTRA_CHUNK_INDICES_IDX);
    const bool isVarlen = cuTensor != nullptr;
    OP_CHECK_IF(isVarlen != (chunkTensor != nullptr) || (isVarlen && batch != 1),
                OP_LOGE(context->GetNodeName(), "varlen metadata must be paired and requires B=1."),
                return ge::GRAPH_FAILED);
    const int64_t chunksPerBatch = seqlen / *chunkSize + (seqlen % *chunkSize != 0);
    uint64_t fixedChunks = 0;
    OP_CHECK_IF(!IntraCheckedMultiply(batch, chunksPerBatch, fixedChunks) ||
                    fixedChunks > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context->GetNodeName(), "chunk count overflow."),
                return ge::GRAPH_FAILED);
    int64_t chunkCount = static_cast<int64_t>(fixedChunks);
    if (isVarlen) {
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkTensor);
        const auto *chunkShape = context->GetOptionalInputShape(INTRA_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkShape);
        const auto metadataShape = chunkShape->GetStorageShape();
        const bool matrixMetadata = metadataShape.GetDimNum() == 2 && metadataShape.GetDim(1) == 2;
        const bool flatMetadata = metadataShape.GetDimNum() == 1 && metadataShape.GetDim(0) % 2 == 0;
        OP_CHECK_IF(!matrixMetadata && !flatMetadata,
                    OP_LOGE(context->GetNodeName(), "chunk_indices must contain flattened pairs or [Nchunk,2]."),
                    return ge::GRAPH_FAILED);
        chunkCount = matrixMetadata ? metadataShape.GetDim(0) : metadataShape.GetDim(0) / 2;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto arch = platform.GetCurNpuArch();
    const bool arch22 = arch == NpuArch::DAV_2201;
    OP_CHECK_IF(!arch22 && arch != NpuArch::DAV_3510,
                OP_LOGE(context->GetNodeName(), "only DAV_2201 and DAV_3510 are supported."),
                return ge::GRAPH_FAILED);
    const uint32_t aicCores = platform.GetCoreNumAic();
    IntraSchedule schedule;
    OP_CHECK_IF(!BuildIntraSchedule(chunkCount, valueHeads, qkHeads, aicCores,
                                    arch22, platform.GetLibApiWorkSpaceSize(), schedule),
                OP_LOGE(context->GetNodeName(), "invalid schedule or workspace overflow."),
                return ge::GRAPH_FAILED);
    if (arch22) {
        uint64_t ubBytes = 0;
        uint64_t l1Bytes = 0;
        uint64_t l0aBytes = 0;
        uint64_t l0bBytes = 0;
        uint64_t l0cBytes = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Bytes);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aBytes);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bBytes);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cBytes);
        OP_CHECK_IF(ubBytes < GDN::Arch22::UB_BYTES ||
                        l1Bytes < GDN::Arch22::L1_SCORE_BYTES + schedule.cg * GDN::Arch22::L1_HEAD_BYTES ||
                        l0aBytes < GDN::Arch22::L0_HOST_MIN_BYTES ||
                        l0bBytes < GDN::Arch22::L0_HOST_MIN_BYTES ||
                        l0cBytes < GDN::Arch22::L0_HOST_MIN_BYTES,
                    OP_LOGE(context->GetNodeName(), "insufficient on-chip memory for arch22 schedule."),
                    return ge::GRAPH_FAILED);
    }
    tiling->batch = batch;
    tiling->qkHeads = qkHeads;
    tiling->valueHeads = valueHeads;
    tiling->seqlen = seqlen;
    tiling->keyDim = keyDim;
    tiling->valueDim = valueDim;
    tiling->chunkSize = *chunkSize;
    tiling->chunksPerBatch = chunksPerBatch;
    tiling->chunkCount = chunkCount;
    tiling->cg = schedule.cg;
    tiling->hvSliceCount = schedule.hvSlices;
    tiling->workCount = schedule.workCount;
    tiling->blockDim = schedule.blockDim;
    tiling->isVarlen = isVarlen ? 1 : 0;
    tiling->useExp2 = *useExp2 ? 1 : 0;
    tiling->scale = static_cast<float>(*scale);
    tiling->reserved = arch22 ? 2201 : 0;
    const uint64_t strategyKey = isVarlen ?
        CHUNK_GDN_BWD_INTRA_VARLEN : CHUNK_GDN_BWD_INTRA_FIXED;
    const uint64_t gKey = static_cast<uint64_t>(headRatio);
    const uint64_t mainKey = mainDtype == ge::DT_BF16 ?
        CHUNK_GDN_BWD_INTRA_MAIN_BF16 : CHUNK_GDN_BWD_INTRA_MAIN_FP16;
    const uint64_t gateKey = DataTypeKey(
        gDtype, ge::DT_FLOAT,
        CHUNK_GDN_BWD_INTRA_GATE_BF16, CHUNK_GDN_BWD_INTRA_GATE_FP32);
    const uint64_t betaKey = DataTypeKey(
        betaDtype, ge::DT_FLOAT,
        CHUNK_GDN_BWD_INTRA_BETA_BF16, CHUNK_GDN_BWD_INTRA_BETA_FP32);
    context->SetTilingKey(GET_TPL_TILING_KEY(
        strategyKey, gKey, mainKey, gateKey, betaKey));
    context->SetBlockDim(schedule.blockDim);
    size_t *ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = schedule.workspaceBytes;
    OP_LOGD(context->GetNodeName(),"workspace = [%zu]", ws[0]);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkGdnBwdIntra(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGdnBwdIntra)
    .Tiling(Tiling4ChunkGdnBwdIntra)
    .TilingParse<ChunkGdnBwdIntraCompileInfo>(TilingPrepare4ChunkGdnBwdIntra);

} // namespace optiling
