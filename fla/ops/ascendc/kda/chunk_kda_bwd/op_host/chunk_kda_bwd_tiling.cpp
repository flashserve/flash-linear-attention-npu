#include "chunk_kda_bwd_tiling.h"

#include <algorithm>
#include <cstdint>

#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"

#include "chunk_kda_bwd_a_tiling_processor.h"
#include "chunk_kda_bwd_c_tiling_processor.h"
#include "../op_kernel/kda_gate_bwd_post_common.h"

namespace optiling {
namespace {

enum InputIndex : size_t {
    INPUT_Q = 0,
    INPUT_K,
    INPUT_V,
    INPUT_BETA,
    INPUT_GK,
    INPUT_AQK,
    INPUT_AKK,
    INPUT_W,
    INPUT_QG,
    INPUT_KG,
    INPUT_V_NEW,
    INPUT_H,
    INPUT_DO,
    INPUT_RAW_G,
    INPUT_A_LOG,
    INPUT_DT_BIAS,
    INPUT_CU_SEQLENS,
    INPUT_CHUNK_INDICES,
};

enum OutputIndex : size_t {
    OUTPUT_DQ = 0,
    OUTPUT_DK,
    OUTPUT_DV,
    OUTPUT_DB,
    OUTPUT_DG,
    OUTPUT_DA,
    OUTPUT_DBIAS,
};

enum AttrIndex : size_t {
    ATTR_SCALE = 0,
    ATTR_CHUNK_SIZE,
    ATTR_SAFE_GATE,
    ATTR_USE_GATE,
    ATTR_LOWER_BOUND,
    ATTR_DISABLE_RECOMPUTE,
    ATTR_USE_EXP2,
    ATTR_STATE_V_FIRST,
    ATTR_DEFER_GATE_POST,
};

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

uint64_t DtypeSize(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT ? 4U : 2U;
}

int64_t CeilDiv(int64_t a, int64_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

ge::graphStatus BuildKernelBTiling(
    gert::TilingContext *context, const KDA::ChunkKdaBwdATilingData &a,
    uint64_t ubSize, uint32_t blockDim,
    GDN::ChunkGatedDeltaRuleBwdDhuTilingData &b,
    uint64_t &userWorkspaceBytes, float scale)
{
    const auto *qgDesc = context->GetInputDesc(INPUT_QG);
    const auto *gkDesc = context->GetInputDesc(INPUT_GK);
    const auto *qShapeStorage = context->GetRequiredInputShape(INPUT_Q);
    const auto *vShapeStorage = context->GetRequiredInputShape(INPUT_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, qgDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gkDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShapeStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShapeStorage);
    OP_CHECK_IF(qgDesc->GetDataType() != ge::DT_FLOAT16 &&
                    qgDesc->GetDataType() != ge::DT_BF16,
                OP_LOGE(context->GetNodeName(),
                        "Kernel B qg/kg/w inputs must be FP16 or BF16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gkDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(),
                        "single-launch Kernel B requires saved gk in FP32"),
                return ge::GRAPH_FAILED);

    const gert::Shape qShape = qShapeStorage->GetOriginShape();
    const gert::Shape vShape = vShapeStorage->GetOriginShape();
    const size_t dimAxis = a.isVarLen ? 2U : 3U;
    b.B = a.isVarLen ? 1 : static_cast<int64_t>(qShape.GetDim(0));
    b.HK = a.headNum;
    b.HV = a.headNum;
    b.T = a.seqlen;
    b.K = static_cast<int64_t>(qShape.GetDim(dimAxis));
    b.V = static_cast<int64_t>(vShape.GetDim(dimAxis));
    b.HRatio = 1;
    b.chunkSize = a.chunkSize;
    b.chunkNumForT = CeilDiv(b.T, b.chunkSize);
    b.totalChunkNum = a.isVarLen ? a.chunkNum : b.chunkNumForT;
    b.chunkTaskNum = a.isVarLen ? a.chunkNum : b.B * b.chunkNumForT;
    b.seqNum = b.B;
    if (a.isVarLen) {
        const auto *cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS);
        OP_CHECK_NULL_WITH_CONTEXT(context, cuShape);
        b.seqNum = static_cast<int64_t>(
            cuShape->GetOriginShape().GetDim(0) - 1);
    }
    b.headWindowNum = CeilDiv(b.HV, 4);
    b.taskNum = b.seqNum * b.headWindowNum;
    b.isVariable = a.isVarLen;
    b.hasDh0 = 0;
    b.dh0ClearCoreNum = 0;
    b.dh0ClearElemsPerCore = 0;
    b.dh0ClearTailElems = 0;
    b.hasGk = 1;
    b.scale = scale;

    const uint64_t qSize = DtypeSize(qgDesc->GetDataType());
    const uint64_t gateSize = DtypeSize(gkDesc->GetDataType());
    const uint64_t maxDim = static_cast<uint64_t>(std::max(b.K, b.V));
    const uint64_t gateElems = static_cast<uint64_t>(
        std::max<int32_t>(b.K, b.chunkSize));
    uint64_t row = static_cast<uint64_t>(b.K);
    const auto align32 = [](uint64_t bytes) { return AlignUp(bytes, 32); };
    while (row > 8) {
        const uint64_t fixedBytes =
            2 * align32(gateElems * gateSize) +
            align32(4U * gateElems * sizeof(float));
        uint64_t vectorBytes =
            4 * align32(row * maxDim * qSize) +
            2 * align32(row * maxDim * sizeof(float)) +
            2 * align32(row * static_cast<uint64_t>(b.V) * sizeof(float));
        if (qgDesc->GetDataType() == ge::DT_BF16) {
            vectorBytes +=
                2 * align32(row * static_cast<uint64_t>(b.V) * qSize);
        }
        if (fixedBytes + vectorBytes + 16U * 1024U <= ubSize) {
            break;
        }
        row /= 2;
    }
    b.vecRow = static_cast<int64_t>(std::max<uint64_t>(row, 8));

    b.qgWorkspaceElems = b.chunkSize * b.K;
    b.stateWorkspaceElems = static_cast<int64_t>(
        align32(static_cast<uint64_t>(b.K) * b.V * sizeof(float)) / qSize);
    b.dvStateWorkspaceElems = b.chunkSize * b.V;
    b.termQWorkspaceElems = b.K * b.V;
    b.dv2WorkspaceElems = 0;
    b.termWWorkspaceElems = b.K * b.V;

    int64_t offset = 0;
    b.qgWorkspaceOffset = offset;
    offset += b.qgWorkspaceElems;
    offset = static_cast<int64_t>(align32(static_cast<uint64_t>(offset) * qSize) / qSize);
    b.stateWorkspaceOffset = offset;
    offset += b.stateWorkspaceElems;
    b.dvStateWorkspaceOffset = offset;
    offset += b.dvStateWorkspaceElems;
    b.termQWorkspaceOffset = offset;
    offset += b.termQWorkspaceElems;
    b.dv2WorkspaceOffset = offset;
    offset += b.dv2WorkspaceElems;
    b.termWWorkspaceOffset = offset;
    offset += b.termWWorkspaceElems;
    b.workspaceElemsPerSubBlock = offset;

    userWorkspaceBytes = static_cast<uint64_t>(blockDim) * 8U *
        static_cast<uint64_t>(b.workspaceElemsPerSubBlock) * qSize;
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus Tiling4ChunkKdaBwd(gert::TilingContext *context)
{
    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto *scale = attrs->GetAttrPointer<float>(ATTR_SCALE);
    const auto *chunkSize = attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const auto *safeGate = attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE);
    const auto *useGate = attrs->GetAttrPointer<bool>(ATTR_USE_GATE);
    const auto *lowerBound = attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND);
    const auto *disableRecompute =
        attrs->GetAttrPointer<bool>(ATTR_DISABLE_RECOMPUTE);
    const auto *useExp2 = attrs->GetAttrPointer<bool>(ATTR_USE_EXP2);
    const auto *stateVFirst =
        attrs->GetAttrPointer<bool>(ATTR_STATE_V_FIRST);
    const auto *deferGatePost =
        attrs->GetAttrPointer<bool>(ATTR_DEFER_GATE_POST);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, safeGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, useGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, lowerBound);
    OP_CHECK_NULL_WITH_CONTEXT(context, disableRecompute);
    OP_CHECK_NULL_WITH_CONTEXT(context, useExp2);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateVFirst);
    OP_CHECK_NULL_WITH_CONTEXT(context, deferGatePost);
    OP_CHECK_IF(!*disableRecompute,
                OP_LOGE(context->GetNodeName(),
                        "disable_recompute=false is reserved but not supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!*useExp2,
                OP_LOGE(context->GetNodeName(),
                        "use_exp2=false is reserved but not supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*stateVFirst,
                OP_LOGE(context->GetNodeName(),
                        "state_v_first=true is reserved but not supported"),
                return ge::GRAPH_FAILED);

    const auto *wShape = context->GetOptionalInputShape(INPUT_W);
    const auto *qgShape = context->GetOptionalInputShape(INPUT_QG);
    const auto *kgShape = context->GetOptionalInputShape(INPUT_KG);
    const auto *vNewShape = context->GetOptionalInputShape(INPUT_V_NEW);
    const auto *hShape = context->GetOptionalInputShape(INPUT_H);
    const auto *wDesc = context->GetInputDesc(INPUT_W);
    const auto *qgDesc = context->GetInputDesc(INPUT_QG);
    const auto *kgDesc = context->GetInputDesc(INPUT_KG);
    const auto *vNewDesc = context->GetInputDesc(INPUT_V_NEW);
    const auto *hDesc = context->GetInputDesc(INPUT_H);
    OP_CHECK_IF(wShape == nullptr || qgShape == nullptr ||
                    kgShape == nullptr || vNewShape == nullptr ||
                    hShape == nullptr || wDesc == nullptr ||
                    qgDesc == nullptr || kgDesc == nullptr ||
                    vNewDesc == nullptr || hDesc == nullptr,
                OP_LOGE(context->GetNodeName(),
                        "w, qg, kg, v_new and h are required when disable_recompute=true"),
                return ge::GRAPH_FAILED);

    const uint32_t blockDim =
        std::max<uint32_t>(static_cast<uint32_t>(platform.GetCoreNumAic()), 1U);
    const bool isAscend950 =
        platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const size_t systemWorkspace =
        static_cast<size_t>(platform.GetLibApiWorkSpaceSize());

    ChunkKdaBwdATilingContext aCtx{
        context->GetNodeName(),
        context->GetRequiredInputShape(INPUT_AQK),
        vNewShape,
        hShape,
        context->GetRequiredInputShape(INPUT_DO),
        context->GetOptionalInputShape(INPUT_CU_SEQLENS),
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES),
        context->GetInputDesc(INPUT_AQK)->GetDataType(),
        vNewDesc->GetDataType(),
        hDesc->GetDataType(),
        context->GetInputDesc(INPUT_DO)->GetDataType(),
        *chunkSize, blockDim, systemWorkspace};
    KDA::ChunkKdaBwdATilingData aTiling{};
    ChunkKdaBwdATilingProcessor aProcessor(aCtx, aTiling);
    OP_CHECK_IF(aProcessor.Process() != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t bUserWorkspace = 0;
    OP_CHECK_IF(BuildKernelBTiling(
                    context, aTiling, ubSize, blockDim,
                    tiling->kernelB, bUserWorkspace, *scale) !=
                    ge::GRAPH_SUCCESS,
                , return ge::GRAPH_FAILED);

    ChunkKdaBwdCTilingContext cCtx{};
    cCtx.nodeName = context->GetNodeName();
    const size_t cInputMap[KDA_C_INPUT_COUNT] = {
        INPUT_Q, INPUT_K, INPUT_V, INPUT_V_NEW, INPUT_GK, INPUT_BETA,
        INPUT_AKK, INPUT_H, INPUT_H, INPUT_V, INPUT_Q,
        INPUT_AKK, INPUT_RAW_G, INPUT_A_LOG, INPUT_DT_BIAS,
        INPUT_CU_SEQLENS, INPUT_CHUNK_INDICES};
    for (size_t i = 0; i < KDA_C_INPUT_COUNT; ++i) {
        const size_t inputIndex = cInputMap[i];
        const bool schemaOptional =
            inputIndex == INPUT_W || inputIndex == INPUT_QG ||
            inputIndex == INPUT_KG || inputIndex == INPUT_V_NEW ||
            inputIndex == INPUT_H || inputIndex == INPUT_RAW_G ||
            inputIndex == INPUT_A_LOG || inputIndex == INPUT_DT_BIAS ||
            inputIndex == INPUT_CU_SEQLENS ||
            inputIndex == INPUT_CHUNK_INDICES;
        cCtx.shapes[i] = schemaOptional ?
            context->GetOptionalInputShape(inputIndex) :
            context->GetRequiredInputShape(inputIndex);
        const auto *desc = context->GetInputDesc(inputIndex);
        if (desc != nullptr) {
            cCtx.types[i] = desc->GetDataType();
        }
    }
    cCtx.types[KDA_C_DQ_RAW] = ge::DT_FLOAT;
    cCtx.types[KDA_C_DAQK] = ge::DT_FLOAT;
    cCtx.scale = *scale;
    cCtx.chunkSize = *chunkSize;
    cCtx.safeGate = *safeGate;
    const gert::Shape publicQShape =
        context->GetRequiredInputShape(INPUT_Q)->GetOriginShape();
    const bool denseGateFusion =
        isAscend950 && *useGate && *safeGate && aTiling.isVarLen == 0 &&
        publicQShape.GetDimNum() == 4 &&
        publicQShape.GetDim(1) > 0 &&
        publicQShape.GetDim(2) % *chunkSize == 0;
    cCtx.useGateInKernel = denseGateFusion;
    cCtx.lowerBound = *lowerBound;
    cCtx.dhHeadMajor = true;
    cCtx.validateIntermediateShapes = false;
    cCtx.aicCoreNum = blockDim;
    cCtx.systemWorkspaceSize = systemWorkspace;
    ChunkKdaBwdCTilingProcessor cProcessor(cCtx, tiling->kernelC);
    OP_CHECK_IF(cProcessor.Process() != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED);
    tiling->kernelC.deferGatePost = *deferGatePost ? 1 : 0;

    const uint64_t cTotalWorkspace = cProcessor.GetWorkspaceSize();
    OP_CHECK_IF(cTotalWorkspace < systemWorkspace,
                OP_LOGE(context->GetNodeName(),
                        "Kernel C workspace is smaller than system workspace"),
                return ge::GRAPH_FAILED);
    const uint64_t cUserWorkspace = cTotalWorkspace - systemWorkspace;
    const gert::Shape qShape =
        context->GetRequiredInputShape(INPUT_Q)->GetOriginShape();
    const gert::Shape vShape =
        context->GetRequiredInputShape(INPUT_V)->GetOriginShape();
    const gert::Shape akkShape =
        context->GetRequiredInputShape(INPUT_AKK)->GetOriginShape();
    const bool varlen = aTiling.isVarLen != 0;
    const uint64_t tokenCount = static_cast<uint64_t>(
        qShape.GetShapeSize() / qShape.GetDim(varlen ? 2U : 3U));
    const uint64_t dataBytes = DtypeSize(
        context->GetInputDesc(INPUT_Q)->GetDataType());
    const uint64_t kDim = tiling->kernelB.K;
    const uint64_t vDim = tiling->kernelB.V;
    const uint64_t headNum = tiling->kernelB.HV;
    const uint64_t chunkTasks = tiling->kernelB.chunkTaskNum;
    uint64_t cursor = 0;
    const auto reserve = [&cursor](uint64_t bytes) {
        const uint64_t offset = cursor;
        cursor = AlignUp(cursor + bytes, 512);
        return offset;
    };
    tiling->dv0Offset = static_cast<uint32_t>(
        reserve(tokenCount * vDim * dataBytes));
    tiling->dqRawOffset = static_cast<uint32_t>(
        reserve(tokenCount * kDim * sizeof(float)));
    tiling->dAqkOffset = static_cast<uint32_t>(
        reserve(static_cast<uint64_t>(akkShape.GetShapeSize()) * sizeof(float)));
    tiling->dhOffset = static_cast<uint32_t>(reserve(
        chunkTasks * headNum * kDim * vDim * dataBytes));
    tiling->dvScanOffset = static_cast<uint32_t>(
        reserve(tokenCount * vDim * dataBytes));
    tiling->dAkkOffset = static_cast<uint32_t>(
        reserve(static_cast<uint64_t>(akkShape.GetShapeSize()) * sizeof(float)));
    tiling->kernelBWorkspaceOffset = static_cast<uint32_t>(cursor);
    cursor = AlignUp(cursor + bUserWorkspace, 512);
    tiling->kernelCWorkspaceOffset = static_cast<uint32_t>(cursor);
    cursor += cUserWorkspace;
    OP_CHECK_IF(cursor > UINT32_MAX,
                OP_LOGE(context->GetNodeName(),
                        "single-launch private workspace exceeds 4 GiB"),
                return ge::GRAPH_FAILED);
    const uint64_t totalWorkspace = systemWorkspace + cursor;

    context->SetBlockDim(blockDim);
    context->SetTilingKey(cProcessor.GetTilingKey());
    context->SetScheduleMode(1);
    context->GetWorkspaceSizes(1)[0] = totalWorkspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ChunkKdaBwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKdaBwd)
    .Tiling(Tiling4ChunkKdaBwd)
    .TilingParse<ChunkKdaBwdCompileInfo>(TilingPrepare4ChunkKdaBwd);

ge::graphStatus Tiling4KdaGateBwdPost(gert::TilingContext *context)
{
    constexpr size_t INPUT_DG_ACT = 0;
    constexpr size_t INPUT_RAW_G = 1;
    constexpr size_t INPUT_A_LOG = 2;
    constexpr size_t INPUT_DT_BIAS = 3;
    constexpr size_t INPUT_CU_SEQLENS = 4;
    constexpr size_t INPUT_CHUNK_INDICES = 5;
    constexpr size_t ATTR_CHUNK_SIZE = 0;
    constexpr size_t ATTR_SAFE_GATE = 1;
    constexpr size_t ATTR_LOWER_BOUND = 2;
    constexpr size_t ATTR_INPUT_SCANNED = 3;

    const auto *attrs = context->GetAttrs();
    const auto *dgShapeStorage =
        context->GetRequiredInputShape(INPUT_DG_ACT);
    const auto *rawShapeStorage =
        context->GetRequiredInputShape(INPUT_RAW_G);
    const auto *aLogShapeStorage =
        context->GetRequiredInputShape(INPUT_A_LOG);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, dgShapeStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, rawShapeStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, aLogShapeStorage);

    const auto *chunkSize =
        attrs->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE);
    const auto *safeGate = attrs->GetAttrPointer<bool>(ATTR_SAFE_GATE);
    const auto *lowerBound =
        attrs->GetAttrPointer<float>(ATTR_LOWER_BOUND);
    const auto *inputScanned =
        attrs->GetAttrPointer<bool>(ATTR_INPUT_SCANNED);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, safeGate);
    OP_CHECK_NULL_WITH_CONTEXT(context, lowerBound);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputScanned);
    OP_CHECK_IF(*chunkSize != 64,
                OP_LOGE(context->GetNodeName(),
                        "KdaGateBwdPost requires chunk_size=64"),
                return ge::GRAPH_FAILED);

    const gert::Shape dgShape = dgShapeStorage->GetOriginShape();
    const gert::Shape rawShape = rawShapeStorage->GetOriginShape();
    const bool isVarLen = dgShape.GetDimNum() == 3U;
    const bool hasCuSeqlens =
        context->GetOptionalInputShape(INPUT_CU_SEQLENS) != nullptr;
    const bool hasChunkIndices =
        context->GetOptionalInputShape(INPUT_CHUNK_INDICES) != nullptr;
    OP_CHECK_IF(hasCuSeqlens != hasChunkIndices ||
                    isVarLen != hasCuSeqlens,
                OP_LOGE(context->GetNodeName(),
                        "rank-3 GatePost requires cu_seqlens/chunk_indices and rank-4 GatePost forbids them"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dgShape.GetDimNum() != (isVarLen ? 3U : 4U) ||
                    rawShape.GetDimNum() != dgShape.GetDimNum() ||
                    rawShape.GetShapeSize() != dgShape.GetShapeSize(),
                OP_LOGE(context->GetNodeName(),
                        "dg_act/raw_g shapes must match canonical BHTK/HTK layout"),
                return ge::GRAPH_FAILED);

    const size_t headAxis = isVarLen ? 0U : 1U;
    const size_t tokenAxis = isVarLen ? 1U : 2U;
    const size_t keyAxis = isVarLen ? 2U : 3U;
    const int64_t batch = isVarLen ? 1 : dgShape.GetDim(0);
    const int64_t heads = dgShape.GetDim(headAxis);
    const int64_t seqlen = dgShape.GetDim(tokenAxis);
    const int64_t keyDim = dgShape.GetDim(keyAxis);
    OP_CHECK_IF(batch <= 0 || heads <= 0 || seqlen <= 0 || keyDim != 128,
                OP_LOGE(context->GetNodeName(),
                        "KdaGateBwdPost requires positive B/H/T and K=128"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(aLogShapeStorage->GetOriginShape().GetShapeSize() != heads,
                OP_LOGE(context->GetNodeName(),
                        "a_log must contain one FP32 value per head"),
                return ge::GRAPH_FAILED);

    int64_t chunkNum = 0;
    const int64_t chunkNumPerBatch = CeilDiv(seqlen, *chunkSize);
    if (isVarLen) {
        const auto *chunkShape =
            context->GetOptionalInputShape(INPUT_CHUNK_INDICES);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkShape);
        const int64_t metadataElems =
            chunkShape->GetOriginShape().GetShapeSize();
        OP_CHECK_IF(metadataElems <= 0 || metadataElems % 2 != 0,
                    OP_LOGE(context->GetNodeName(),
                            "chunk_indices must contain (sequence, local_chunk) pairs"),
                return ge::GRAPH_FAILED);
        chunkNum = metadataElems / 2;
        OP_CHECK_IF(chunkNum > KDA::KDA_GATE_POST_MAX_CHUNKS,
                    OP_LOGE(context->GetNodeName(),
                            "KdaGateBwdPost supports at most %d packed chunks",
                            KDA::KDA_GATE_POST_MAX_CHUNKS),
                    return ge::GRAPH_FAILED);
    } else {
        chunkNum = batch * chunkNumPerBatch;
    }

    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableAiv = std::max<uint32_t>(
        static_cast<uint32_t>(platform.GetCoreNumAiv()), 1U);
    const uint32_t usedCoreNum = std::max<uint32_t>(
        std::min<uint32_t>(static_cast<uint32_t>(heads), availableAiv), 1U);
    auto *tiling =
        context->GetTilingData<KDA::KdaGateBwdPostTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->batch = static_cast<int32_t>(batch);
    tiling->seqlen = static_cast<int32_t>(seqlen);
    tiling->headNum = static_cast<int32_t>(heads);
    tiling->keyDim = static_cast<int32_t>(keyDim);
    tiling->chunkSize = static_cast<int32_t>(*chunkSize);
    tiling->chunkNum = static_cast<int32_t>(chunkNum);
    tiling->chunkNumPerBatch = static_cast<int32_t>(chunkNumPerBatch);
    tiling->usedCoreNum = static_cast<int32_t>(usedCoreNum);
    tiling->isVarLen = isVarLen ? 1 : 0;
    tiling->hasDtBias =
        context->GetOptionalInputShape(INPUT_DT_BIAS) != nullptr ? 1 : 0;
    tiling->inputScanned = *inputScanned ? 1 : 0;
    tiling->reserved1 = 0;
    tiling->lowerBound = *lowerBound;
    if (isVarLen) {
        const auto *cuTensor =
            context->GetOptionalInputTensor(INPUT_CU_SEQLENS);
        const auto *chunkTensor =
            context->GetOptionalInputTensor(INPUT_CHUNK_INDICES);
        OP_CHECK_NULL_WITH_CONTEXT(context, cuTensor);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkTensor);
        const int64_t *cu = cuTensor->GetData<int64_t>();
        const int64_t *chunks = chunkTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, cu);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunks);
        const int64_t cuElems =
            cuTensor->GetStorageShape().GetShapeSize();
        int64_t packedCursor = 0;
        for (int64_t task = 0; task < chunkNum; ++task) {
            const int64_t sequence = chunks[2 * task];
            const int64_t localChunk = chunks[2 * task + 1];
            OP_CHECK_IF(sequence < 0 || sequence + 1 >= cuElems ||
                            localChunk < 0,
                        OP_LOGE(context->GetNodeName(),
                                "invalid chunk_indices entry at task %ld", task),
                        return ge::GRAPH_FAILED);
            const int64_t seqBegin = cu[sequence];
            const int64_t seqEnd = cu[sequence + 1];
            const int64_t begin = seqBegin + localChunk * *chunkSize;
            const int64_t end = std::min<int64_t>(begin + *chunkSize, seqEnd);
            OP_CHECK_IF(seqBegin < 0 || seqEnd < seqBegin ||
                            seqEnd > seqlen || begin < seqBegin ||
                            begin >= seqEnd || end <= begin ||
                            begin != packedCursor,
                        OP_LOGE(context->GetNodeName(),
                                "chunk_indices must use canonical packed order at task %ld",
                                task),
                        return ge::GRAPH_FAILED);
            packedCursor = end;
        }
        OP_CHECK_IF(packedCursor != seqlen,
                    OP_LOGE(context->GetNodeName(),
                            "canonical packed chunks must cover all tokens"),
                    return ge::GRAPH_FAILED);
    }

    context->SetBlockDim(usedCoreNum);
    // Task decoding and token addressing use tiling.isVarLen at runtime.
    // Reuse the same device specialization for dense and packed layouts so
    // the proven A5 reverse-scan pipeline is compiled only once.
    context->SetTilingKey(*safeGate ? 1U : 3U);
    context->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4KdaGateBwdPost(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KdaGateBwdPost)
    .Tiling(Tiling4KdaGateBwdPost)
    .TilingParse<ChunkKdaBwdCompileInfo>(TilingPrepare4KdaGateBwdPost);

IMPL_OP_OPTILING(KdaGateBwdPostVarlen)
    .Tiling(Tiling4KdaGateBwdPost)
    .TilingParse<ChunkKdaBwdCompileInfo>(TilingPrepare4KdaGateBwdPost);

} // namespace optiling
