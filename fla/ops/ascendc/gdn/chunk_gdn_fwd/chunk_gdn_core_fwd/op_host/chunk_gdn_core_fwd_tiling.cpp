/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gdn_core_fwd_tiling.h"

#include "../op_kernel/internal/operators/chunk_fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.h"
#include "../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#include "../op_kernel/internal/state_update_output/chunk_gdn_core_state_update_output_struct.h"
#include "../op_kernel/chunk_gdn_core_fwd_struct.h"

#include "securec.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>
#include <register/op_impl_registry.h>

namespace optiling {

ge::graphStatus Tiling4ChunkGdnCoreStateOutput(gert::TilingContext *context);

namespace {

constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_K = 1;
constexpr size_t INPUT_V = 2;
constexpr size_t INPUT_BETA = 3;
constexpr size_t INPUT_A_STORAGE = 4;
constexpr size_t INPUT_RAW_G = 5;
constexpr size_t INPUT_GK = 6;
constexpr size_t INPUT_INITIAL_STATE = 7;
constexpr size_t INPUT_CU_SEQLENS = 8;
constexpr size_t INPUT_CHUNK_INDICES = 9;

constexpr size_t ATTR_OUTPUT_FINAL_STATE = 0;
constexpr size_t ATTR_CHUNK_SIZE = 1;
constexpr size_t ATTR_OUTPUT_MASK = 3;
constexpr int64_t GDN_CORE_OUTPUT_MASK_ALL = static_cast<int64_t>(
    GDN::GDN_CORE_OUTPUT_G_CUMSUM | GDN::GDN_CORE_OUTPUT_A);

constexpr int64_t SUPPORTED_K_DIM = 128;
constexpr int64_t SUPPORTED_V_DIM_128 = 128;
constexpr int64_t SUPPORTED_V_DIM_256 = 256;
constexpr int64_t CHUNK_64 = 64;
constexpr int64_t CHUNK_128 = 128;
constexpr uint32_t TILING_KEY_V128 = 1;
constexpr uint32_t TILING_KEY_V256 = 2;
constexpr uint64_t WORKSPACE_ALIGNMENT = 512;
constexpr uint64_t TILING_ALIGNMENT = 8;
constexpr uint64_t FP32_BLOCK_ELEMS = 8;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : CeilDiv(value, alignment) * alignment;
}

bool IsShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (shape->GetStorageShape().GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

bool GetChunkCount(const gert::StorageShape *shape, uint64_t *count)
{
    if (shape == nullptr || count == nullptr) {
        return false;
    }
    const auto &storage = shape->GetStorageShape();
    if (storage.GetDimNum() == 1 && storage.GetDim(0) > 0 && storage.GetDim(0) % 2 == 0) {
        *count = static_cast<uint64_t>(storage.GetDim(0) / 2);
        return true;
    }
    if (storage.GetDimNum() == 2 && storage.GetDim(0) > 0 && storage.GetDim(1) == 2) {
        *count = static_cast<uint64_t>(storage.GetDim(0));
        return true;
    }
    return false;
}

ge::graphStatus BuildCoefficientCubeTiling(uint64_t bt, uint64_t k, ge::DataType dtype,
                                           AscendC::tiling::TCubeTiling &tiling)
{
    matmul_tiling::MatmulApiTiling mm;
    const auto inputType = dtype == ge::DT_BF16
                               ? matmul_tiling::DataType::DT_BF16
                               : matmul_tiling::DataType::DT_FLOAT16;
    if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    inputType, false) != 0 ||
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    inputType, true) != 0 ||
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT) != 0 ||
        mm.EnableBias(false) != 0) {
        return ge::GRAPH_FAILED;
    }
    const int32_t btI32 = static_cast<int32_t>(bt);
    const int32_t kI32 = static_cast<int32_t>(k);
    if (mm.SetShape(btI32, btI32, kI32) != 0 ||
        mm.SetOrgShape(btI32, btI32, kI32) != 0 ||
        mm.SetFixSplit(btI32, btI32, -1) != 0 ||
        mm.SetBufferSpace(-1, -1, -1, -1) != 0) {
        return ge::GRAPH_FAILED;
    }
    return mm.GetTiling(tiling) == -1 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus Tiling4ChunkGdnCoreFwd(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr || context->GetAttrs() == nullptr,
                OP_LOGE("ChunkGdnCoreFwd", "Invalid tiling context."),
                return ge::GRAPH_FAILED);
    const auto *qShape = context->GetOptionalInputShape(INPUT_Q);
    const auto *kShape = context->GetOptionalInputShape(INPUT_K);
    const auto *vShape = context->GetOptionalInputShape(INPUT_V);
    const auto *betaShape = context->GetOptionalInputShape(INPUT_BETA);
    const auto *aShape = context->GetOptionalInputShape(INPUT_A_STORAGE);
    const auto *gShape = context->GetOptionalInputShape(INPUT_RAW_G);
    OP_CHECK_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr || betaShape == nullptr ||
                    aShape == nullptr || gShape == nullptr ||
                    qShape->GetStorageShape().GetDimNum() != 4 ||
                    kShape->GetStorageShape().GetDimNum() != 4 ||
                    vShape->GetStorageShape().GetDimNum() != 4 ||
                    betaShape->GetStorageShape().GetDimNum() != 3 ||
                    aShape->GetStorageShape().GetDimNum() != 4 ||
                    gShape->GetStorageShape().GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "Phase 6 requires rank-4 q/k/v/A and rank-3 beta/raw_g."),
                return ge::GRAPH_FAILED);
    const auto *qDesc = context->GetInputDesc(INPUT_Q);
    const auto *kDesc = context->GetInputDesc(INPUT_K);
    const auto *vDesc = context->GetInputDesc(INPUT_V);
    const auto *betaDesc = context->GetInputDesc(INPUT_BETA);
    const auto *aDesc = context->GetInputDesc(INPUT_A_STORAGE);
    const auto *gDesc = context->GetInputDesc(INPUT_RAW_G);
    OP_CHECK_IF(qDesc == nullptr || kDesc == nullptr || vDesc == nullptr || betaDesc == nullptr ||
                    aDesc == nullptr || gDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "Phase 6 requires valid input descriptors."),
                return ge::GRAPH_FAILED);
    const ge::DataType inputDtype = qDesc->GetDataType();
    const bool isFp16 = inputDtype == ge::DT_FLOAT16;
    const bool isBf16 = inputDtype == ge::DT_BF16;
    const gert::Shape qStorage = qShape->GetStorageShape();
    const int64_t batch = qStorage.GetDim(0);
    const int64_t heads = qStorage.GetDim(1);
    const int64_t tokens = qStorage.GetDim(2);
    const int64_t kDim = qStorage.GetDim(3);
    const gert::Shape vStorage = vShape->GetStorageShape();
    const int64_t valueHeads = vStorage.GetDim(1);
    const int64_t vDim = vStorage.GetDim(3);
    const auto *cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
    const auto *chunkShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
    const auto *cuDesc = context->GetOptionalInputDesc(INPUT_CU_SEQLENS);
    const auto *chunkDesc = context->GetOptionalInputDesc(INPUT_CHUNK_INDICES);
    const bool hasCu = cuDesc != nullptr && cuShape != nullptr;
    const bool hasChunks = chunkDesc != nullptr && chunkShape != nullptr;
    const bool isVarlen = hasCu || hasChunks;
    OP_CHECK_IF((!isFp16 && !isBf16) ||
                    batch <= 0 || heads <= 0 || tokens <= 0 || kDim != SUPPORTED_K_DIM ||
                    (vDim != SUPPORTED_V_DIM_128 && vDim != SUPPORTED_V_DIM_256) ||
                    valueHeads <= 0 || (valueHeads % heads) != 0,
                OP_LOGE(context->GetNodeName(),
                        "Phase 6 requires positive B/Hk/T, Hk divides Hv, K=128, and V=128/256; dense T may be arbitrary."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape(kShape, {batch, heads, tokens, kDim}),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires k to match q in [B,H,T,K]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape(vShape, {batch, valueHeads, tokens, vDim}),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires v=[B,Hv,T,V] with Hv divisible by Hk."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape(betaShape, {batch, valueHeads, tokens}),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires beta=[B,Hv,T]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape(gShape, {batch, valueHeads, tokens}),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires raw_g=[B,Hv,T]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDesc->GetDataType() != inputDtype ||
                    vDesc->GetDataType() != inputDtype ||
                    aDesc->GetDataType() != inputDtype ||
                    betaDesc->GetDataType() != ge::DT_FLOAT ||
                    gDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(),
                        "Phase 6 requires matching FP16/BF16 inputs/A and FP32 beta/g."),
                return ge::GRAPH_FAILED);

    const bool *outputFinalState =
        context->GetAttrs()->GetAttrPointer<bool>(ATTR_OUTPUT_FINAL_STATE);
    const int64_t *chunkSize = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const int64_t *outputMask = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_OUTPUT_MASK);
    uint64_t varlenChunks = 0;
    OP_CHECK_IF(outputFinalState == nullptr || chunkSize == nullptr || outputMask == nullptr ||
                    *outputMask < 0 || *outputMask > GDN_CORE_OUTPUT_MASK_ALL ||
                    (*chunkSize != CHUNK_64 && *chunkSize != CHUNK_128) ||
                    context->GetOptionalInputDesc(INPUT_GK) != nullptr || hasCu != hasChunks ||
                    (isVarlen && (cuDesc->GetDataType() != ge::DT_INT64 ||
                                  chunkDesc->GetDataType() != ge::DT_INT64 ||
                                  cuShape->GetStorageShape().GetDimNum() != 1 ||
                                  cuShape->GetStorageShape().GetDim(0) < 2 ||
                                  !GetChunkCount(chunkShape, &varlenChunks))),
                OP_LOGE(context->GetNodeName(),
                        "Phase 6 requires output_mask in [0,3], chunk_size=64/128, and paired valid varlen metadata."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsShape(aShape, {batch, valueHeads, tokens, *chunkSize}),
                OP_LOGE(context->GetNodeName(), "Phase 6 requires a_storage=[B,Hv,T,chunk_size]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(Tiling4ChunkGdnCoreStateOutput(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Reuse of the accepted Phase 5 suffix tiling failed."),
                return ge::GRAPH_FAILED);

    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint64_t aicCoreNum = std::max<uint64_t>(1, platform.GetCoreNumAic());
    const uint64_t aivCoreNum = std::max<uint64_t>(1, platform.GetCoreNumAiv());
    const uint64_t systemWorkspace = platform.GetLibApiWorkSpaceSize();
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    OP_CHECK_IF(workspaceSizes[0] < systemWorkspace,
                OP_LOGE(context->GetNodeName(), "Phase 5 workspace is smaller than system workspace."),
                return ge::GRAPH_FAILED);

    GDN::ChunkGdnCoreFwdTrailer trailer{};
    trailer.outputMask = static_cast<uint64_t>(*outputMask);
    auto &coefficient = trailer.coefficient;
    coefficient.B = static_cast<uint64_t>(batch);
    coefficient.Hk = static_cast<uint64_t>(heads);
    coefficient.Hv = static_cast<uint64_t>(valueHeads);
    coefficient.hvPerHk = static_cast<uint64_t>(valueHeads / heads);
    coefficient.T = static_cast<uint64_t>(tokens);
    coefficient.K = SUPPORTED_K_DIM;
    coefficient.BT = static_cast<uint64_t>(*chunkSize);
    coefficient.NT = isVarlen ? varlenChunks : CeilDiv(coefficient.T, coefficient.BT);
    // Coefficient generation produces one KKT/solve tile per value head. K is shared by the
    // contiguous group of hvPerHk value heads mapped to one logical K head.
    coefficient.taskNum = coefficient.B * coefficient.Hv * coefficient.NT;
    coefficient.usedAicNum = aicCoreNum;
    coefficient.usedAivNum = std::min<uint64_t>(aivCoreNum, aicCoreNum * 2);
    coefficient.btAlign = AlignUp(coefficient.BT, FP32_BLOCK_ELEMS);
    coefficient.isVarlen = isVarlen ? 1 : 0;
    coefficient.scoreWorkspaceBytes =
        AlignUp(coefficient.taskNum * coefficient.BT * coefficient.BT * sizeof(float), WORKSPACE_ALIGNMENT);
    coefficient.aWorkspaceBytes = AlignUp(
        coefficient.B * coefficient.Hv * coefficient.T * coefficient.BT * sizeof(uint16_t), WORKSPACE_ALIGNMENT);
    coefficient.solveWorkspacePerCoreBytes = AlignUp(
        5 * coefficient.BT * coefficient.BT * sizeof(uint16_t), WORKSPACE_ALIGNMENT);
    coefficient.totalTiles = static_cast<int64_t>(coefficient.taskNum);
    coefficient.matrixSize = *chunkSize;
    coefficient.numHeads = valueHeads;
    coefficient.seqLen = tokens;
    coefficient.batchSize = batch;
    coefficient.isLower = 1;
    coefficient.hasCuSeqlens = isVarlen ? 1 : 0;
    coefficient.tilesPerCore = static_cast<int64_t>(CeilDiv(coefficient.taskNum, aicCoreNum));
    coefficient.chunkSize = *chunkSize;
    coefficient.numChunks = isVarlen ? 0 : static_cast<int64_t>(coefficient.NT);
    coefficient.lastChunkValidSize = isVarlen ? 0 :
        (tokens % *chunkSize == 0 ? *chunkSize : tokens % *chunkSize);
    coefficient.totalChunks = static_cast<int64_t>(coefficient.NT);
    coefficient.layoutMode = isVarlen ? 3 : 0;
    coefficient.dtypeMode = isBf16 ? 1 : 0;
    coefficient.totalTokens = isVarlen ? tokens : 0;
    OP_CHECK_IF(BuildCoefficientCubeTiling(coefficient.BT, coefficient.K, inputDtype,
                                           coefficient.cubeTilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(),
                        "Failed to build the Phase 6 KKT Matmul tiling."),
                return ge::GRAPH_FAILED);
    uint64_t workspaceOffset = AlignUp(workspaceSizes[0] - systemWorkspace, WORKSPACE_ALIGNMENT);
    trailer.scoreWorkspaceOffset = workspaceOffset;
    workspaceOffset += coefficient.scoreWorkspaceBytes;
    trailer.aWorkspaceOffset = workspaceOffset;
    workspaceOffset += coefficient.aWorkspaceBytes;
    trailer.solveWorkspaceOffset = workspaceOffset;
    workspaceOffset += aicCoreNum * coefficient.solveWorkspacePerCoreBytes;
    trailer.gCumsumBhtOffset = workspaceOffset;
    workspaceOffset += AlignUp(coefficient.B * coefficient.Hv * coefficient.T * sizeof(float), WORKSPACE_ALIGNMENT);
    workspaceSizes[0] = systemWorkspace + workspaceOffset;

    ChunkGatedDeltaRuleFwdHTilingData hTiling;
    const uint64_t hTilingSize = hTiling.GetDataSize();
    OP_CHECK_IF(hTilingSize != sizeof(::ChunkGatedDeltaRuleFwdHTilingData),
                OP_LOGE(context->GetNodeName(),
                        "FwdH host/kernel tiling size mismatch: host=%lu, kernel=%zu.",
                        hTilingSize, sizeof(::ChunkGatedDeltaRuleFwdHTilingData)),
                return ge::GRAPH_FAILED);
    const uint64_t oTilingOffset = AlignUp(hTilingSize, TILING_ALIGNMENT);
    const uint64_t stateOutputTilingEnd = oTilingOffset + sizeof(GDN::ChunkFwdOTilingData) +
                                          sizeof(GDN::ChunkGdnCoreStateOutputTrailer);
    const uint64_t phase6TrailerOffset = AlignUp(stateOutputTilingEnd, TILING_ALIGNMENT);
    auto *rawTiling = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTiling);
    const uint64_t rawTilingSize = phase6TrailerOffset + sizeof(trailer);
    OP_CHECK_IF(rawTilingSize > rawTiling->GetCapacity(),
                OP_LOGE(context->GetNodeName(), "Phase 6 combined tiling exceeds raw tiling capacity."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(static_cast<uint8_t *>(rawTiling->GetData()) + phase6TrailerOffset,
                         rawTiling->GetCapacity() - phase6TrailerOffset,
                         &trailer, sizeof(trailer)) != EOK,
                OP_LOGE(context->GetNodeName(), "Serialize Phase 6 coefficient trailer failed."),
                return ge::GRAPH_FAILED);
    rawTiling->SetDataSize(rawTilingSize);
    context->SetTilingKey(vDim == SUPPORTED_V_DIM_256 ? TILING_KEY_V256 : TILING_KEY_V128);
    context->SetScheduleMode(1);
    OP_LOGD(context->GetNodeName(),
            "Phase 6 tiling: B=%ld, Hk=%ld, Hv=%ld, T=%ld, K=%ld, V=%ld, blocks=%lu, tasks=%lu, suffix=%zu, total=%zu.",
            batch, heads, valueHeads, tokens, kDim, vDim,
            aicCoreNum, coefficient.taskNum, workspaceSizes[0] - systemWorkspace,
            workspaceSizes[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGdnCoreFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGdnCoreFwd)
    .Tiling(Tiling4ChunkGdnCoreFwd)
    .TilingParse<ChunkGdnCoreFwdCompileInfo>(TilingPrepareForChunkGdnCoreFwd);

} // namespace optiling
