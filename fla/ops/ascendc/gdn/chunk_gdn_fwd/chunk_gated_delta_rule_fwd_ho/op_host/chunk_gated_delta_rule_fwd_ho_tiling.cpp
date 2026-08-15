/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_ho_tiling.h"

#include "../../chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.h"
#include "../../chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"
#include "../../chunk_fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../op_kernel/chunk_gated_delta_rule_fwd_ho_struct.h"

#include "securec.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>
#include <register/op_impl_registry.h>

namespace optiling {
namespace {

constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_K = 1;
constexpr size_t INPUT_W = 2;
constexpr size_t INPUT_U = 3;
constexpr size_t INPUT_G = 4;
constexpr size_t INPUT_GK = 5;
constexpr size_t INPUT_INITIAL_STATE = 6;
constexpr size_t INPUT_CU_SEQLENS = 7;
constexpr size_t INPUT_CHUNK_INDICES = 8;

constexpr size_t ATTR_OUTPUT_FINAL_STATE = 0;
constexpr size_t ATTR_CHUNK_SIZE = 1;
constexpr size_t ATTR_SCALE = 2;

constexpr int64_t DIM_BATCH = 0;
constexpr int64_t DIM_HEAD = 1;
constexpr int64_t DIM_TOKEN = 2;
constexpr int64_t DIM_CHANNEL = 3;
constexpr int64_t SUPPORTED_K = 128;
constexpr int64_t SUPPORTED_V128 = 128;
constexpr int64_t SUPPORTED_V256 = 256;
constexpr int64_t CHUNK_64 = 64;
constexpr int64_t CHUNK_128 = 128;
constexpr uint32_t TILING_KEY_V128 = 1;
constexpr uint32_t TILING_KEY_V256 = 2;
constexpr size_t TILING_ALIGNMENT = 8;
constexpr size_t WORKSPACE_ALIGNMENT = 512;
constexpr size_t WORKSPACE_RESERVE = 16 * 1024 * 1024;
constexpr int64_t PING_PONG_STAGES = 2;
constexpr int64_t HO_PIPELINE_VALUE_HEADS = 8;
constexpr int64_t HO_PIPELINE_VALUE_DIM = 128;
constexpr uint32_t HO_PIPELINE_CUBE_CORES = 24;
constexpr size_t HO_PIPELINE_EVENT_COUNT = 2;
constexpr size_t HO_PIPELINE_EVENT_BYTES_PER_CORE = 32;

size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

int64_t DtypeToEnum(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return GDN_FWD_H_DTYPE_BF16;
    }
    if (dtype == ge::DT_FLOAT16) {
        return GDN_FWD_H_DTYPE_FP16;
    }
    return GDN_FWD_H_DTYPE_FP32;
}

bool IsRank(const gert::StorageShape *shape, size_t rank)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == rank;
}

void FillHMacroTiling(const ::ChunkGatedDeltaRuleFwdHTilingData &src,
                      ChunkGatedDeltaRuleFwdHTilingData &dst)
{
    dst.set_batch(src.batch);
    dst.set_seqlen(src.seqlen);
    dst.set_kNumHead(src.kNumHead);
    dst.set_vNumHead(src.vNumHead);
    dst.set_kHeadDim(src.kHeadDim);
    dst.set_vHeadDim(src.vHeadDim);
    dst.set_chunkSize(src.chunkSize);
    dst.set_useInitialState(src.useInitialState);
    dst.set_storeFinalState(src.storeFinalState);
    dst.set_dataType(src.dataType);
    dst.set_gDataType(src.gDataType);
    dst.set_stateDataType(src.stateDataType);
    dst.set_isVariedLen(src.isVariedLen);
    dst.set_shapeBatch(src.shapeBatch);
    dst.set_tokenBatch(src.tokenBatch);
    dst.set_useGk(src.useGk);
    dst.set_vWorkspaceOffset(src.vWorkspaceOffset);
    dst.set_vUpdateWorkspaceOffset(src.vUpdateWorkspaceOffset);
    dst.set_kDecayWorkspaceOffset(src.kDecayWorkspaceOffset);
    dst.set_hWorkspaceOffset(src.hWorkspaceOffset);
    dst.set_numSeqWorkspaceOffset(src.numSeqWorkspaceOffset);
    dst.set_numChunksWorkspaceOffset(src.numChunksWorkspaceOffset);
}

size_t FillOTilingWorkspace(GDN::ChunkFwdOTilingData &tiling, uint32_t aicCoreNum,
                            size_t sysWorkspaceSize)
{
    size_t offset = sysWorkspaceSize + WORKSPACE_RESERVE;
    tiling.vWorkspaceOffset = static_cast<int64_t>(offset);
    offset += AlignUp(static_cast<size_t>(aicCoreNum) * tiling.chunkSize * tiling.vHeadDim *
                          sizeof(float) * PING_PONG_STAGES,
                      WORKSPACE_ALIGNMENT);
    tiling.hWorkspaceOffset = static_cast<int64_t>(offset);
    offset += AlignUp(static_cast<size_t>(aicCoreNum) * tiling.chunkSize * tiling.vHeadDim *
                          sizeof(float) * PING_PONG_STAGES,
                      WORKSPACE_ALIGNMENT);
    tiling.attnWorkspaceOffset = static_cast<int64_t>(offset);
    offset += AlignUp(static_cast<size_t>(aicCoreNum) * tiling.chunkSize * tiling.chunkSize *
                          sizeof(float) * PING_PONG_STAGES,
                      WORKSPACE_ALIGNMENT);
    tiling.aftermaskWorkspaceOffset = static_cast<int64_t>(offset);
    offset += AlignUp(static_cast<size_t>(aicCoreNum) * tiling.chunkSize * tiling.chunkSize *
                          sizeof(float) * PING_PONG_STAGES,
                      WORKSPACE_ALIGNMENT);
    tiling.maskWorkspaceOffset = static_cast<int64_t>(offset);
    offset += AlignUp(static_cast<size_t>(tiling.chunkSize) * tiling.chunkSize, WORKSPACE_ALIGNMENT);
    return offset + WORKSPACE_RESERVE;
}

} // namespace

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdHO(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdHO start.");
    const auto *qShapePtr = context->GetOptionalInputShape(INPUT_Q);
    const auto *kShapePtr = context->GetOptionalInputShape(INPUT_K);
    const auto *wShapePtr = context->GetOptionalInputShape(INPUT_W);
    const auto *uShapePtr = context->GetOptionalInputShape(INPUT_U);
    const auto *gShapePtr = context->GetOptionalInputShape(INPUT_G);
    const auto *cuShapePtr = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
    const auto *chunkShapePtr = context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
    OP_CHECK_IF(!IsRank(qShapePtr, 4) || !IsRank(kShapePtr, 4) || !IsRank(wShapePtr, 4) ||
                    !IsRank(uShapePtr, 4) || !IsRank(gShapePtr, 3),
                OP_LOGE(context->GetNodeName(), "q/k/w/u must be rank 4 and g must be rank 3."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((cuShapePtr == nullptr) != (chunkShapePtr == nullptr),
                OP_LOGE(context->GetNodeName(),
                        "cu_seqlens and chunk_indices must be both present or both absent."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuShapePtr != nullptr && (!IsRank(cuShapePtr, 1) || !IsRank(chunkShapePtr, 1)),
                OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must be rank 1."),
                return ge::GRAPH_FAILED);

    const gert::Shape qShape = qShapePtr->GetStorageShape();
    const gert::Shape kShape = kShapePtr->GetStorageShape();
    const gert::Shape wShape = wShapePtr->GetStorageShape();
    const gert::Shape uShape = uShapePtr->GetStorageShape();
    const gert::Shape gShape = gShapePtr->GetStorageShape();
    const int64_t batch = qShape.GetDim(DIM_BATCH);
    const int64_t kNumHead = qShape.GetDim(DIM_HEAD);
    const int64_t seqlen = qShape.GetDim(DIM_TOKEN);
    const int64_t kHeadDim = qShape.GetDim(DIM_CHANNEL);
    const int64_t vNumHead = uShape.GetDim(DIM_HEAD);
    const int64_t vHeadDim = uShape.GetDim(DIM_CHANNEL);

    OP_CHECK_IF(batch <= 0 || kNumHead <= 0 || vNumHead <= 0 || seqlen <= 0,
                OP_LOGE(context->GetNodeName(), "B/H/T dimensions must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kShape.GetDim(DIM_BATCH) != batch || kShape.GetDim(DIM_HEAD) != kNumHead ||
                    kShape.GetDim(DIM_TOKEN) != seqlen || kShape.GetDim(DIM_CHANNEL) != kHeadDim,
                OP_LOGE(context->GetNodeName(), "q and k must have the same shape."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(wShape.GetDim(DIM_BATCH) != batch || wShape.GetDim(DIM_HEAD) != vNumHead ||
                    wShape.GetDim(DIM_TOKEN) != seqlen || wShape.GetDim(DIM_CHANNEL) != kHeadDim ||
                    uShape.GetDim(DIM_BATCH) != batch || uShape.GetDim(DIM_TOKEN) != seqlen,
                OP_LOGE(context->GetNodeName(), "w/u must match q/k in B/T and use value heads."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gShape.GetDim(0) != batch || gShape.GetDim(1) != vNumHead ||
                    gShape.GetDim(2) != seqlen,
                OP_LOGE(context->GetNodeName(), "g must have shape [B,HV,T]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vNumHead % kNumHead != 0,
                OP_LOGE(context->GetNodeName(), "vNumHead must be divisible by kNumHead."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kHeadDim != SUPPORTED_K ||
                    (vHeadDim != SUPPORTED_V128 && vHeadDim != SUPPORTED_V256),
                OP_LOGE(context->GetNodeName(), "Only K=128 and V=128/256 are supported."),
                return ge::GRAPH_FAILED);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool outputFinalState = *(attrs->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE));
    const int64_t chunkSize = *(attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE));
    const double scale = *(attrs->GetAttrPointer<double>(ATTR_SCALE));
    OP_CHECK_IF(chunkSize != CHUNK_64 && chunkSize != CHUNK_128,
                OP_LOGE(context->GetNodeName(), "chunk_size must be 64 or 128."),
                return ge::GRAPH_FAILED);

    const bool isVarlen = cuShapePtr != nullptr;
    const int64_t tokenBatch = isVarlen ? cuShapePtr->GetStorageShape().GetDim(0) - 1 : 1;
    OP_CHECK_IF(tokenBatch <= 0 || (isVarlen && batch != 1),
                OP_LOGE(context->GetNodeName(), "Varlen input requires physical B=1 and at least one sequence."),
                return ge::GRAPH_FAILED);
    const int64_t totalChunks = isVarlen
                                    ? chunkShapePtr->GetStorageShape().GetDim(0) / 2
                                    : (seqlen + chunkSize - 1) / chunkSize;
    OP_CHECK_IF(totalChunks <= 0 ||
                    (isVarlen && chunkShapePtr->GetStorageShape().GetDim(0) % 2 != 0),
                OP_LOGE(context->GetNodeName(), "chunk_indices must contain (seq,chunk) pairs."),
                return ge::GRAPH_FAILED);

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicCoreNum = platform.GetCoreNumAic();
    const size_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    const auto *qDesc = context->GetInputDesc(INPUT_Q);
    const auto *gDesc = context->GetInputDesc(INPUT_G);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    const auto *initialDesc = context->GetOptionalInputDesc(INPUT_INITIAL_STATE);
    const bool useInitialState = initialDesc != nullptr;
    const bool useGk = context->GetOptionalInputDesc(INPUT_GK) != nullptr;

    ChunkGatedDeltaRuleFwdHTilingContext hContext{};
    hContext.seqlen = seqlen;
    hContext.kNumHead = kNumHead;
    hContext.kHeadDim = kHeadDim;
    hContext.vNumHead = vNumHead;
    hContext.vHeadDim = vHeadDim;
    hContext.shapeBatchDim = batch;
    hContext.hasCuSeqlens = isVarlen;
    hContext.cuSeqlensDim0 = isVarlen ? tokenBatch + 1 : 0;
    hContext.dataType = DtypeToEnum(qDesc->GetDataType());
    hContext.gDataType = DtypeToEnum(gDesc->GetDataType());
    hContext.useInitialState = useInitialState;
    hContext.stateDataType = useInitialState ? DtypeToEnum(initialDesc->GetDataType()) : GDN_FWD_H_DTYPE_FP32;
    hContext.useGk = useGk;
    hContext.storeFinalState = outputFinalState;
    hContext.chunkSize = chunkSize;
    hContext.aicCoreNum = aicCoreNum;
    hContext.libApiWorkSpaceSize = sysWorkspaceSize;

    ::ChunkGatedDeltaRuleFwdHTilingData hPlain{};
    uint32_t hBlockDim = 0;
    size_t hWorkspaceSize = 0;
    ChunkGatedDeltaRuleFwdHTilingProcessor hProcessor(hContext);
    hProcessor.Process(hPlain, hBlockDim, hWorkspaceSize);

    ChunkGatedDeltaRuleFwdHTilingData hTiling;
    FillHMacroTiling(hPlain, hTiling);
    GDN::ChunkFwdOTilingData oTiling{};
    oTiling.shapeBatch = isVarlen ? 1 : batch;
    oTiling.seqlen = seqlen;
    oTiling.kNumHead = kNumHead;
    oTiling.vNumHead = vNumHead;
    oTiling.kHeadDim = kHeadDim;
    oTiling.vHeadDim = vHeadDim;
    oTiling.chunkSize = chunkSize;
    oTiling.isVariedLen = isVarlen ? 1 : 0;
    oTiling.tokenBatch = tokenBatch;
    oTiling.dataType = hContext.dataType;
    oTiling.gDataType = hContext.gDataType;
    oTiling.scale = static_cast<float>(scale);
    const size_t oWorkspaceSize = FillOTilingWorkspace(oTiling, aicCoreNum, sysWorkspaceSize);

    const size_t elementSize = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    size_t workspaceOffset = AlignUp(std::max(hWorkspaceSize, oWorkspaceSize), WORKSPACE_ALIGNMENT);
    GDN::ChunkGatedDeltaRuleFwdHOTrailer trailer{};
    trailer.hIntermediateOffset = static_cast<int64_t>(workspaceOffset);
    workspaceOffset += AlignUp(static_cast<size_t>(isVarlen ? 1 : batch) * vNumHead * totalChunks *
                                   kHeadDim * vHeadDim * elementSize,
                               WORKSPACE_ALIGNMENT);
    trailer.vNewIntermediateOffset = static_cast<int64_t>(workspaceOffset);
    workspaceOffset += AlignUp(static_cast<size_t>(batch) * vNumHead * seqlen * vHeadDim * elementSize,
                               WORKSPACE_ALIGNMENT);
    const bool enableChunkPipeline = !isVarlen && batch == 1 &&
                                     vNumHead == HO_PIPELINE_VALUE_HEADS &&
                                     vHeadDim == HO_PIPELINE_VALUE_DIM &&
                                     (chunkSize == CHUNK_64 || chunkSize == CHUNK_128) &&
                                     aicCoreNum == HO_PIPELINE_CUBE_CORES;
    if (enableChunkPipeline) {
        workspaceOffset += AlignUp(static_cast<size_t>(aicCoreNum) * PING_PONG_STAGES *
                                       HO_PIPELINE_EVENT_COUNT * HO_PIPELINE_EVENT_BYTES_PER_CORE,
                                   WORKSPACE_ALIGNMENT);
    }
    workspaceOffset += WORKSPACE_RESERVE;

    const size_t hTilingSize = hTiling.GetDataSize();
    const size_t oTilingOffset = AlignUp(hTilingSize, TILING_ALIGNMENT);
    const size_t trailerOffset = oTilingOffset + sizeof(oTiling);
    const size_t rawTilingSize = trailerOffset + sizeof(trailer);
    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    OP_CHECK_IF(rawTilingSize > rawTiling->GetCapacity(),
                OP_LOGE(context->GetNodeName(), "Combined tiling data exceeds raw tiling capacity."),
                return ge::GRAPH_FAILED);
    void *rawData = rawTiling->GetData();
    OP_CHECK_IF(memset_s(rawData, rawTiling->GetCapacity(), 0, rawTilingSize) != EOK,
                OP_LOGE(context->GetNodeName(), "Clear raw tiling data failed."),
                return ge::GRAPH_FAILED);
    hTiling.SaveToBuffer(rawData, rawTiling->GetCapacity());
    OP_CHECK_IF(memcpy_s(static_cast<uint8_t *>(rawData) + oTilingOffset,
                         rawTiling->GetCapacity() - oTilingOffset, &oTiling, sizeof(oTiling)) != EOK ||
                    memcpy_s(static_cast<uint8_t *>(rawData) + trailerOffset,
                             rawTiling->GetCapacity() - trailerOffset, &trailer, sizeof(trailer)) != EOK,
                OP_LOGE(context->GetNodeName(), "Serialize combined tiling data failed."),
                return ge::GRAPH_FAILED);
    rawTiling->SetDataSize(rawTilingSize);

    context->SetTilingKey(vHeadDim > SUPPORTED_V128 ? TILING_KEY_V256 : TILING_KEY_V128);
    context->SetBlockDim(hBlockDim);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = workspaceOffset;
    OP_LOGD(context->GetNodeName(),
            "FwdHO tiling: h=%zu, o=%zu, hIntermediate=%ld, vNewIntermediate=%ld, total=%zu.",
            hWorkspaceSize, oWorkspaceSize, trailer.hIntermediateOffset,
            trailer.vNewIntermediateOffset, workspaceOffset);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleFwdHO(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwdHO)
    .Tiling(Tiling4ChunkGatedDeltaRuleFwdHO)
    .TilingParse<ChunkGatedDeltaRuleFwdHOCompileInfo>(TilingPrepareForChunkGatedDeltaRuleFwdHO);

} // namespace optiling
