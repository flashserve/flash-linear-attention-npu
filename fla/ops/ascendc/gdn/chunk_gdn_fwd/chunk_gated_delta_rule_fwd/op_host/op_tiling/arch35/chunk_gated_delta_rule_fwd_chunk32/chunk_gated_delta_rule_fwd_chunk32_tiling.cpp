/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_32_tiling_data.h"

#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>

namespace optiling {
namespace {

constexpr size_t INPUT_Q = 0;
constexpr size_t INPUT_K = 1;
constexpr size_t INPUT_V = 2;
// The public low-level op reserves input 3 for FP32 beta. Chunk32 keeps its
// fixed BF16 beta in the otherwise-unused a_storage slot to preserve that ABI.
constexpr size_t INPUT_BETA = 4;
constexpr size_t INPUT_RAW_G = 5;
constexpr size_t INPUT_INITIAL_STATE = 7;
constexpr size_t INPUT_CU_SEQLENS = 8;
constexpr size_t INPUT_CHUNK_INDICES = 9;
constexpr size_t OUTPUT_O = 0;
constexpr size_t OUTPUT_FINAL_STATE = 1;
constexpr size_t ATTR_CHUNK_SIZE = 1;
constexpr size_t ATTR_SCALE = 2;
constexpr size_t ATTR_STATE_V_FIRST = 7;
constexpr size_t ATTR_ALLOW_NEG_EIGVAL = 8;

constexpr int64_t CHUNK_SIZE = 32;
constexpr int64_t HEAD_DIM = 128;
constexpr uint64_t DEFAULT_WORKSPACE_BYTES = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t TILING_KEY = 32;

bool IsShape(const gert::StorageShape *shape, std::initializer_list<int64_t> dims)
{
    if (shape == nullptr) {
        return false;
    }
    const auto matches = [&dims](const gert::Shape &candidate) {
        if (candidate.GetDimNum() != dims.size()) {
            return false;
        }
        size_t index = 0;
        for (int64_t dim : dims) {
            if (candidate.GetDim(index++) != dim) {
                return false;
            }
        }
        return true;
    };
    return matches(shape->GetStorageShape()) || matches(shape->GetOriginShape());
}

} // namespace

ge::graphStatus TilingForChunkGatedDeltaRuleFwdChunk32(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr || context->GetAttrs() == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid chunk32 tiling context."),
                return ge::GRAPH_FAILED);

    const int64_t *chunkSize = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const double *scale = context->GetAttrs()->GetAttrPointer<double>(ATTR_SCALE);
    OP_CHECK_IF(chunkSize == nullptr || *chunkSize != CHUNK_SIZE || scale == nullptr,
                OP_LOGE(context->GetNodeName(), "Chunk32 tiling requires chunk_size=32 and a scale attribute."),
                return ge::GRAPH_FAILED);

    const auto *qShape = context->GetOptionalInputShape(INPUT_Q);
    const auto *kShape = context->GetOptionalInputShape(INPUT_K);
    const auto *vShape = context->GetOptionalInputShape(INPUT_V);
    const auto *betaShape = context->GetOptionalInputShape(INPUT_BETA);
    const auto *gShape = context->GetOptionalInputShape(INPUT_RAW_G);
    const auto *oShape = context->GetOutputShape(OUTPUT_O);
    const auto *stateShape = context->GetOutputShape(OUTPUT_FINAL_STATE);
    const auto *qDesc = context->GetInputDesc(INPUT_Q);
    const auto *kDesc = context->GetInputDesc(INPUT_K);
    const auto *vDesc = context->GetInputDesc(INPUT_V);
    const auto *betaDesc = context->GetInputDesc(INPUT_BETA);
    const auto *gDesc = context->GetInputDesc(INPUT_RAW_G);
    const auto *stateDesc = context->GetOutputDesc(OUTPUT_FINAL_STATE);
    OP_CHECK_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr ||
                    betaShape == nullptr || gShape == nullptr || oShape == nullptr ||
                    stateShape == nullptr || qDesc == nullptr || kDesc == nullptr ||
                    vDesc == nullptr || betaDesc == nullptr || gDesc == nullptr ||
                    stateDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "Chunk32 requires valid q/k/v/beta/g and output descriptors."),
                return ge::GRAPH_FAILED);

    const gert::Shape &q = qShape->GetStorageShape();
    const gert::Shape &v = vShape->GetStorageShape();
    OP_CHECK_IF(q.GetDimNum() != 4 || v.GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "Chunk32 q/k/v must use BNSD rank-4 storage."),
                return ge::GRAPH_FAILED);
    const int64_t batch = q.GetDim(0);
    const int64_t keyHeads = q.GetDim(1);
    const int64_t tokens = q.GetDim(2);
    const int64_t keyDim = q.GetDim(3);
    const int64_t valueHeads = v.GetDim(1);
    const int64_t valueDim = v.GetDim(3);
    OP_CHECK_IF(batch <= 0 || keyHeads <= 0 || valueHeads <= 0 || tokens <= 0 ||
                    valueHeads % keyHeads != 0 || keyDim != HEAD_DIM || valueDim != HEAD_DIM,
                OP_LOGE(context->GetNodeName(),
                        "Chunk32 requires positive B/H/T, Hv divisible by Hk, and K=V=128."),
                return ge::GRAPH_FAILED);
    // A caller-owned contiguous o can arrive with its storage collapsed to
    // numel. The logical BNSD layout is already checked in ACLNN.
    const int64_t oNumel = batch * valueHeads * tokens * valueDim;
    const gert::Shape &oStorage = oShape->GetStorageShape();
    const bool oFlat = oStorage.GetDimNum() == 1 && oStorage.GetDim(0) == oNumel;
    OP_CHECK_IF(!IsShape(kShape, {batch, keyHeads, tokens, keyDim}) ||
                    !IsShape(vShape, {batch, valueHeads, tokens, valueDim}) ||
                    !IsShape(betaShape, {batch, valueHeads, tokens}) ||
                    !IsShape(gShape, {batch, valueHeads, tokens}) ||
                    !(IsShape(oShape, {batch, valueHeads, tokens, valueDim}) || oFlat),
                OP_LOGE(context->GetNodeName(), "Chunk32 input or output storage shape is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qDesc->GetDataType() != ge::DT_BF16 ||
                    kDesc->GetDataType() != ge::DT_BF16 ||
                    vDesc->GetDataType() != ge::DT_BF16 ||
                    betaDesc->GetDataType() != ge::DT_BF16 ||
                    gDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "Chunk32 requires BF16 q/k/v/beta and FP32 g."),
                return ge::GRAPH_FAILED);

    const auto *cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
    const auto *chunkShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
    const auto *cuDesc = context->GetOptionalInputDesc(INPUT_CU_SEQLENS);
    const auto *chunkDesc = context->GetOptionalInputDesc(INPUT_CHUNK_INDICES);
    const bool hasCu = cuShape != nullptr && cuDesc != nullptr;
    const bool hasChunks = chunkShape != nullptr && chunkDesc != nullptr;
    OP_CHECK_IF(hasCu != hasChunks,
                OP_LOGE(context->GetNodeName(), "Chunk32 varlen metadata must be paired."),
                return ge::GRAPH_FAILED);
    uint32_t seqNum = 0;
    int64_t stateBatch = batch;
    if (hasCu) {
        const gert::Shape &cu = cuShape->GetStorageShape();
        const gert::Shape &chunks = chunkShape->GetStorageShape();
        OP_CHECK_IF(batch != 1 || cuDesc->GetDataType() != ge::DT_INT64 ||
                        chunkDesc->GetDataType() != ge::DT_INT64 ||
                        cu.GetDimNum() != 1 || cu.GetDim(0) < 2 ||
                        chunks.GetDimNum() != 1 || chunks.GetDim(0) < 0 ||
                        chunks.GetDim(0) % 2 != 0,
                    OP_LOGE(context->GetNodeName(), "Chunk32 varlen metadata descriptor is invalid."),
                    return ge::GRAPH_FAILED);
        stateBatch = cu.GetDim(0) - 1;
        seqNum = static_cast<uint32_t>(stateBatch);
    }
    // ACLNN validates the logical state shape. Executor-owned/output tensors
    // may reach tiling with a flattened storage descriptor, so do not repeat
    // that logical-shape check against StorageShape here.

    uint32_t hasInitial = 0;
    uint32_t initialFp32 = 0;
    const auto *initialShape = context->GetOptionalInputShape(INPUT_INITIAL_STATE);
    const auto *initialDesc = context->GetOptionalInputDesc(INPUT_INITIAL_STATE);
    if (initialShape != nullptr || initialDesc != nullptr) {
        OP_CHECK_IF(initialShape == nullptr || initialDesc == nullptr ||
                        (initialDesc->GetDataType() != ge::DT_BF16 &&
                         initialDesc->GetDataType() != ge::DT_FLOAT),
                    OP_LOGE(context->GetNodeName(), "Chunk32 initial state descriptor is invalid."),
                    return ge::GRAPH_FAILED);
        hasInitial = 1;
        initialFp32 = initialDesc->GetDataType() == ge::DT_FLOAT ? 1U : 0U;
    }
    const ge::DataType stateDtype = stateDesc->GetDataType();
    if (hasInitial == 0 || initialFp32 != 0) {
        OP_CHECK_IF(stateDtype != ge::DT_FLOAT,
                    OP_LOGE(context->GetNodeName(),
                            "Chunk32 final state must be FP32 when initial state is absent or FP32."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(stateDtype != ge::DT_BF16,
                    OP_LOGE(context->GetNodeName(),
                            "Chunk32 final state must be BF16 when initial state is BF16."),
                    return ge::GRAPH_FAILED);
    }

    auto *tiling = context->GetTilingData<ChunkGatedDeltaRuleFwd32TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->B = static_cast<uint32_t>(batch);
    tiling->HV = static_cast<uint32_t>(valueHeads);
    tiling->HK = static_cast<uint32_t>(keyHeads);
    tiling->T = static_cast<uint32_t>(tokens);
    tiling->hasInitial = hasInitial;
    tiling->scale = static_cast<float>(*scale);
    tiling->seqNum = seqNum;
    tiling->initialFp32 = initialFp32;
    tiling->finalFp32 = stateDtype == ge::DT_FLOAT ? 1U : 0U;
    const bool *stateVFirst = context->GetAttrs()->GetAttrPointer<bool>(ATTR_STATE_V_FIRST);
    tiling->stateVFirst = stateVFirst != nullptr && *stateVFirst ? 1U : 0U;
    const bool *allowNeg = context->GetAttrs()->GetAttrPointer<bool>(ATTR_ALLOW_NEG_EIGVAL);
    tiling->allowNegEigval = allowNeg != nullptr && *allowNeg ? 1U : 0U;

    uint32_t aicNum = 1;
    uint64_t workspaceBytes = DEFAULT_WORKSPACE_BYTES;
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    workspaceBytes = platform.GetLibApiWorkSpaceSize();
    aicNum = std::max<uint32_t>(static_cast<uint32_t>(platform.GetCoreNumAic()), 1U);
    const uint32_t blockDim =
        std::min<uint32_t>(static_cast<uint32_t>(batch * valueHeads), aicNum);
    tiling->coreNum = blockDim;
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = workspaceBytes;
    context->SetTilingKey(TILING_KEY);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
