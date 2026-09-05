/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "../../../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.h"
#include "../../../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"
#include "../../../op_kernel/internal/operators/chunk_fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../../../op_kernel/internal/gated_delta_rule_state_update_output/chunk_gated_delta_rule_state_update_output_struct.h"
#include "../../../op_kernel/internal/operators/recompute_w_u_fwd/op_host/op_tiling/recompute_w_u_fwd_tiling_processor.h"

#include "securec.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>
#include <register/op_impl_registry.h>

namespace optiling {
namespace {

constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_K = 1;
constexpr size_t INPUT_V = 2;
constexpr size_t INPUT_BETA = 3;
constexpr size_t INPUT_A = 4;
constexpr size_t INPUT_G = 5;
constexpr size_t INPUT_GK = 6;
constexpr size_t INPUT_INITIAL_STATE = 7;
constexpr size_t INPUT_CU_SEQLENS = 8;
constexpr size_t INPUT_CHUNK_INDICES = 9;

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
                            size_t baseOffset)
{
    size_t offset = baseOffset;
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

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdArch35StateOutput(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdArch35StateOutput start.");
    const auto *qShapePtr = context->GetOptionalInputShape(INPUT_Q);
    const auto *kShapePtr = context->GetOptionalInputShape(INPUT_K);
    const auto *vShapePtr = context->GetOptionalInputShape(INPUT_V);
    const auto *betaShapePtr = context->GetOptionalInputShape(INPUT_BETA);
    const auto *aShapePtr = context->GetOptionalInputShape(INPUT_A);
    const auto *gShapePtr = context->GetOptionalInputShape(INPUT_G);
    const auto *cuShapePtr = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
    const auto *chunkShapePtr = context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
    OP_CHECK_IF(!IsRank(qShapePtr, 4) || !IsRank(kShapePtr, 4) || !IsRank(vShapePtr, 4) ||
                    !IsRank(betaShapePtr, 3) || !IsRank(aShapePtr, 4) || !IsRank(gShapePtr, 3),
                OP_LOGE(context->GetNodeName(), "q/k/v/A must be rank 4 and beta/g rank 3."),
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
    const gert::Shape vShape = vShapePtr->GetStorageShape();
    const gert::Shape betaShape = betaShapePtr->GetStorageShape();
    const gert::Shape aShape = aShapePtr->GetStorageShape();
    const gert::Shape gShape = gShapePtr->GetStorageShape();
    const int64_t batch = qShape.GetDim(DIM_BATCH);
    const int64_t kNumHead = qShape.GetDim(DIM_HEAD);
    const int64_t seqlen = qShape.GetDim(DIM_TOKEN);
    const int64_t kHeadDim = qShape.GetDim(DIM_CHANNEL);
    const int64_t vNumHead = vShape.GetDim(DIM_HEAD);
    const int64_t vHeadDim = vShape.GetDim(DIM_CHANNEL);

    OP_CHECK_IF(batch <= 0 || kNumHead <= 0 || vNumHead <= 0 || seqlen <= 0,
                OP_LOGE(context->GetNodeName(), "B/H/T dimensions must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kShape.GetDim(DIM_BATCH) != batch || kShape.GetDim(DIM_HEAD) != kNumHead ||
                    kShape.GetDim(DIM_TOKEN) != seqlen || kShape.GetDim(DIM_CHANNEL) != kHeadDim,
                OP_LOGE(context->GetNodeName(), "q and k must have the same shape."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape.GetDim(DIM_BATCH) != batch || vShape.GetDim(DIM_TOKEN) != seqlen ||
                    betaShape.GetDim(0) != batch || betaShape.GetDim(1) != vNumHead ||
                    betaShape.GetDim(2) != seqlen || aShape.GetDim(DIM_BATCH) != batch ||
                    aShape.GetDim(DIM_HEAD) != vNumHead || aShape.GetDim(DIM_TOKEN) != seqlen ||
                    aShape.GetDim(DIM_CHANNEL) <= 0,
                OP_LOGE(context->GetNodeName(), "v/beta/A must match q/k in B/T and value heads."),
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
                OP_LOGE(context->GetNodeName(), "Phase 5 fused path supports K=128 and V=128/256."),
                return ge::GRAPH_FAILED);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool outputFinalState = *(attrs->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE));
    const int64_t chunkSize = *(attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE));
    const double scale = *(attrs->GetAttrPointer<double>(ATTR_SCALE));
    OP_CHECK_IF(chunkSize != CHUNK_64 && chunkSize != CHUNK_128,
                OP_LOGE(context->GetNodeName(), "chunk_size must be 64 or 128."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(aShape.GetDim(DIM_CHANNEL) != chunkSize,
                OP_LOGE(context->GetNodeName(), "A last dimension must equal chunk_size."),
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
    const auto *kDesc = context->GetInputDesc(INPUT_K);
    const auto *vDesc = context->GetInputDesc(INPUT_V);
    const auto *betaDesc = context->GetInputDesc(INPUT_BETA);
    const auto *aDesc = context->GetInputDesc(INPUT_A);
    const auto *gDesc = context->GetInputDesc(INPUT_G);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, aDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    OP_CHECK_IF(qDesc->GetDataType() != ge::DT_FLOAT16 && qDesc->GetDataType() != ge::DT_BF16,
                OP_LOGE(context->GetNodeName(), "q/k/v must be float16 or bfloat16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDesc->GetDataType() != qDesc->GetDataType() ||
                    vDesc->GetDataType() != qDesc->GetDataType() ||
                    aDesc->GetDataType() != qDesc->GetDataType(),
                OP_LOGE(context->GetNodeName(), "q/k/v/A must use the same dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaDesc->GetDataType() != ge::DT_FLOAT || gDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "Phase 5 fused path requires float32 beta and g."),
                return ge::GRAPH_FAILED);
    const auto *initialDesc = context->GetOptionalInputDesc(INPUT_INITIAL_STATE);
    const bool useInitialState = initialDesc != nullptr;
    const bool useGk = context->GetOptionalInputDesc(INPUT_GK) != nullptr;

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS);
    auto chunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES);
    const int64_t *cuSeqlensData = cuSeqlensTensor == nullptr ? nullptr : cuSeqlensTensor->GetData<int64_t>();
    const int64_t *chunkIndicesData =
        chunkIndicesTensor == nullptr ? nullptr : chunkIndicesTensor->GetData<int64_t>();
    GDN::RecomputeWUFwdTilingData recomputeTiling{};
    RecomputeWUFwdTilingContext recomputeContext{
        context->GetNodeName(),
        context->GetRequiredInputShape(INPUT_K),
        context->GetRequiredInputShape(INPUT_V),
        context->GetRequiredInputShape(INPUT_BETA),
        context->GetRequiredInputShape(INPUT_A),
        context->GetRequiredInputShape(INPUT_G),
        context->GetOptionalInputShape(INPUT_CU_SEQLENS),
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES),
        cuSeqlensData,
        chunkIndicesData,
        static_cast<int32_t>(chunkSize),
        kDesc->GetDataType(),
        betaDesc->GetDataType(),
        0,
        sysWorkspaceSize,
    };
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, recomputeContext.ubSize);
    RecomputeWUFwdTilingProcessor recomputeProcessor(recomputeContext, recomputeTiling);
    OP_CHECK_IF(recomputeProcessor.Process() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RecomputeWUFwd tiling failed."),
                return ge::GRAPH_FAILED);

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
    const size_t elementSize = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const size_t recomputeScratchBytes =
        recomputeProcessor.GetWorkspaceSize() - sysWorkspaceSize;
    const size_t wBytes = static_cast<size_t>(batch) * vNumHead * seqlen * kHeadDim * elementSize;
    const size_t uBytes = static_cast<size_t>(batch) * vNumHead * seqlen * vHeadDim * elementSize;
    const size_t wOffset = AlignUp(recomputeScratchBytes, WORKSPACE_ALIGNMENT);
    const size_t uOffset = wOffset + AlignUp(wBytes, WORKSPACE_ALIGNMENT);
    const size_t wuEnd = uOffset + AlignUp(uBytes, WORKSPACE_ALIGNMENT);
    const size_t defaultHoBase = sysWorkspaceSize + WORKSPACE_RESERVE;
    const size_t hoBase = AlignUp(std::max(defaultHoBase, wuEnd), WORKSPACE_ALIGNMENT);
    const size_t hoShift = hoBase - defaultHoBase;
    auto ShiftHWorkspace = [hoShift](ChunkGatedDeltaRuleFwdHTilingData &tiling) {
        tiling.set_vWorkspaceOffset(tiling.get_vWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_vUpdateWorkspaceOffset(tiling.get_vUpdateWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_kDecayWorkspaceOffset(tiling.get_kDecayWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_hWorkspaceOffset(tiling.get_hWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_numSeqWorkspaceOffset(tiling.get_numSeqWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_numChunksWorkspaceOffset(tiling.get_numChunksWorkspaceOffset() + static_cast<int64_t>(hoShift));
    };
    ShiftHWorkspace(hTiling);
    const size_t oWorkspaceSize = FillOTilingWorkspace(oTiling, aicCoreNum, hoBase);
    hWorkspaceSize += hoShift;

    size_t workspaceOffset = AlignUp(std::max(hWorkspaceSize, oWorkspaceSize), WORKSPACE_ALIGNMENT);
    GDN::ChunkGatedDeltaRuleStateOutputTrailer trailer{};
    trailer.recompute = recomputeTiling;
    trailer.recomputeWorkspaceOffset = 0;
    trailer.wIntermediateOffset = static_cast<int64_t>(wOffset);
    trailer.uIntermediateOffset = static_cast<int64_t>(uOffset);
    trailer.qDataType = DtypeToEnum(qDesc->GetDataType());
    trailer.betaDataType = DtypeToEnum(betaDesc->GetDataType());
    trailer.hIntermediateOffset = static_cast<int64_t>(workspaceOffset);
    workspaceOffset += AlignUp(static_cast<size_t>(isVarlen ? 1 : batch) * vNumHead * totalChunks *
                                   kHeadDim * vHeadDim * elementSize,
                               WORKSPACE_ALIGNMENT);
    trailer.vNewIntermediateOffset = static_cast<int64_t>(workspaceOffset);
    workspaceOffset += AlignUp(static_cast<size_t>(batch) * vNumHead * seqlen * vHeadDim * elementSize,
                               WORKSPACE_ALIGNMENT);
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
            "Fused D+HO tiling: wuScratch=%zu, h=%zu, o=%zu, w=%ld, u=%ld, total=%zu.",
            recomputeScratchBytes, hWorkspaceSize, oWorkspaceSize,
            trailer.wIntermediateOffset, trailer.uIntermediateOffset, workspaceOffset);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
