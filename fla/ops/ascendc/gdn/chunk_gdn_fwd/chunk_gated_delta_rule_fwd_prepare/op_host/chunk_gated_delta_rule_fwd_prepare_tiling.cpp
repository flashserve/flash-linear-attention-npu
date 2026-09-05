/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "chunk_gated_delta_rule_fwd_prepare_tiling.h"

#include <cstdio>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {

bool HasTensor(const gert::StorageShape *shape)
{
    return shape != nullptr;
}

const gert::Shape &LogicalShape(const gert::StorageShape *shape)
{
    const gert::Shape &origin = shape->GetOriginShape();
    const gert::Shape &storage = shape->GetStorageShape();
    return origin.GetDimNum() >= storage.GetDimNum() ? origin : storage;
}

void PrintShape(const char *name, const gert::StorageShape *shape)
{
    if (shape == nullptr) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] %s=null\n", name);
        return;
    }
    const gert::Shape &origin = shape->GetOriginShape();
    const gert::Shape &storage = shape->GetStorageShape();
    printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] %s originRank=%zu storageRank=%zu origin=[",
           name, origin.GetDimNum(), storage.GetDimNum());
    for (size_t i = 0; i < origin.GetDimNum(); ++i) {
        printf("%s%ld", i == 0 ? "" : ",", origin.GetDim(i));
    }
    printf("] storage=[");
    for (size_t i = 0; i < storage.GetDimNum(); ++i) {
        printf("%s%ld", i == 0 ? "" : ",", storage.GetDim(i));
    }
    printf("]\n");
}

bool HasOutput(const gert::TilingContext *context, size_t index)
{
    return context->GetOutputDesc(index) != nullptr && context->GetOutputShape(index) != nullptr;
}

} // namespace

static ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdPrepare(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = ascendcPlatform.GetCoreNumAic();
    if (coreNum <= 0) {
        coreNum = ascendcPlatform.GetCoreNumAiv();
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }

    const gert::StorageShape *qShape = context->GetRequiredInputShape(PREPARE_INPUT_Q);
    const gert::StorageShape *kShape = context->GetRequiredInputShape(PREPARE_INPUT_K);
    const gert::StorageShape *vShape = context->GetRequiredInputShape(PREPARE_INPUT_V);
    const gert::StorageShape *gShape = context->GetRequiredInputShape(PREPARE_INPUT_G);
    const gert::StorageShape *betaShape = context->GetRequiredInputShape(PREPARE_INPUT_BETA);
    if (qShape == nullptr || kShape == nullptr || vShape == nullptr || gShape == nullptr || betaShape == nullptr) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] required input shape is null\n");
        return ge::GRAPH_FAILED;
    }

    PrintShape("q", qShape);
    PrintShape("k", kShape);
    PrintShape("v", vShape);
    PrintShape("g", gShape);
    PrintShape("beta", betaShape);
    PrintShape("a_log", context->GetOptionalInputShape(PREPARE_INPUT_A_LOG));
    PrintShape("dt_bias", context->GetOptionalInputShape(PREPARE_INPUT_DT_BIAS));
    PrintShape("cu_seqlens", context->GetOptionalInputShape(PREPARE_INPUT_CU_SEQLENS));
    PrintShape("chunk_indices", context->GetOptionalInputShape(PREPARE_INPUT_CHUNK_INDICES));

    const auto &q = LogicalShape(qShape);
    const auto &v = LogicalShape(vShape);
    if (q.GetDimNum() != 4 || v.GetDimNum() != 4) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] q/v rank must be 4, got q=%zu v=%zu\n",
               q.GetDimNum(), v.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t B = q.GetDim(0);
    const int64_t HK = q.GetDim(1);
    const int64_t T = q.GetDim(2);
    const int64_t K = q.GetDim(3);
    const int64_t HV = v.GetDim(1);
    const int64_t V = v.GetDim(3);

    auto attrPtr = context->GetAttrs();
    int64_t chunkSize = 64;
    bool allowNegEigval = false;
    bool useExp2 = false;
    bool useQkL2normAttr = false;
    bool useGateAttr = false;
    bool useBetaAttr = false;
    if (attrPtr != nullptr) {
        const int64_t *chunkSizePtr = attrPtr->GetAttrPointer<int64_t>(PREPARE_ATTR_CHUNK_SIZE);
        const bool *allowNegPtr = attrPtr->GetAttrPointer<bool>(PREPARE_ATTR_ALLOW_NEG_EIGVAL);
        const bool *useExp2Ptr = attrPtr->GetAttrPointer<bool>(PREPARE_ATTR_USE_EXP2);
        const bool *useQkPtr = attrPtr->GetAttrPointer<bool>(PREPARE_ATTR_USE_QK_L2NORM);
        const bool *useGatePtr = attrPtr->GetAttrPointer<bool>(PREPARE_ATTR_USE_GATE);
        const bool *useBetaPtr = attrPtr->GetAttrPointer<bool>(PREPARE_ATTR_USE_BETA_SIGMOID);
        if (chunkSizePtr != nullptr) {
            chunkSize = *chunkSizePtr;
        }
        if (allowNegPtr != nullptr) {
            allowNegEigval = *allowNegPtr;
        }
        if (useExp2Ptr != nullptr) {
            useExp2 = *useExp2Ptr;
        }
        if (useQkPtr != nullptr) {
            useQkL2normAttr = *useQkPtr;
        }
        if (useGatePtr != nullptr) {
            useGateAttr = *useGatePtr;
        }
        if (useBetaPtr != nullptr) {
            useBetaAttr = *useBetaPtr;
        }
    }

    const bool hasALog = HasTensor(context->GetOptionalInputShape(PREPARE_INPUT_A_LOG));
    const bool hasDtBias = HasTensor(context->GetOptionalInputShape(PREPARE_INPUT_DT_BIAS));
    const bool hasCuSeqlens = HasTensor(context->GetOptionalInputShape(PREPARE_INPUT_CU_SEQLENS));
    const bool hasChunkIndices = HasTensor(context->GetOptionalInputShape(PREPARE_INPUT_CHUNK_INDICES));
    const bool hasQHat = HasOutput(context, PREPARE_OUTPUT_Q_HAT);
    const bool hasKHat = HasOutput(context, PREPARE_OUTPUT_K_HAT);
    const bool hasQRstd = HasOutput(context, PREPARE_OUTPUT_Q_RSTD);
    const bool hasKRstd = HasOutput(context, PREPARE_OUTPUT_K_RSTD);
    const bool hasBetaEff = HasOutput(context, PREPARE_OUTPUT_BETA_EFF);
    if (!hasQHat || !hasKHat || !hasQRstd || !hasKRstd) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] q_hat/k_hat/q_rstd/k_rstd are required "
               "(this version always runs Q/K L2Norm). qHat=%d kHat=%d qRstd=%d kRstd=%d\n",
               static_cast<int>(hasQHat), static_cast<int>(hasKHat),
               static_cast<int>(hasQRstd), static_cast<int>(hasKRstd));
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }

    // Prefer ACLNN-inferred attrs: optional outputs can be packed and indices shift.
    // This version always runs in-kernel Q/K L2Norm; the attr is not a kernel switch.
    const bool useGateInKernel = useGateAttr || hasALog;
    const bool useBetaSigmoid = useBetaAttr;

    int64_t seqNum = B;
    if (hasCuSeqlens) {
        const auto *cuShape = context->GetOptionalInputShape(PREPARE_INPUT_CU_SEQLENS);
        const auto &cu = LogicalShape(cuShape);
        seqNum = cu.GetDimNum() >= 1 ? (cu.GetDim(0) - 1) : 1;
    }

    int64_t totalChunks = ((T + chunkSize - 1) / chunkSize) * B * HV;
    if (hasChunkIndices) {
        const auto *idxShape = context->GetOptionalInputShape(PREPARE_INPUT_CHUNK_INDICES);
        if (idxShape != nullptr) {
            const auto &idx = LogicalShape(idxShape);
            int64_t n = idx.GetDimNum() >= 1 ? idx.GetDim(0) : 0;
            if (n > 0) {
                totalChunks = (n / 2) * HV;
            }
        }
    }

    const auto *qDesc = context->GetInputDesc(PREPARE_INPUT_Q);
    const auto *gDesc = context->GetInputDesc(PREPARE_INPUT_G);
    const auto *betaDesc = context->GetInputDesc(PREPARE_INPUT_BETA);
    if (qDesc == nullptr || gDesc == nullptr || betaDesc == nullptr) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] required input desc is null\n");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType qDtype = qDesc->GetDataType();
    const ge::DataType gDtype = gDesc->GetDataType();
    const ge::DataType betaDtype = betaDesc->GetDataType();

    if (K != 128 || (V != 128 && V != 256) || chunkSize != 64) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] unsupported K=%ld V=%ld chunkSize=%ld "
               "(need K=128 V=128/256 chunkSize=64)\n",
               K, V, chunkSize);
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (HK <= 0 || HV % HK != 0) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] Hv must be divisible by Hk, Hk=%ld Hv=%ld\n", HK, HV);
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    const int64_t hRatio = HV / HK;
    if (hRatio < 1 || hRatio > 4) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] Hv/Hk=%ld not in {1,2,3,4}\n", hRatio);
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (qDtype != ge::DT_BF16) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] q/k/v must be bf16, got dtype=%d\n",
               static_cast<int>(qDtype));
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (!useExp2) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] use_exp2 currently must be true\n");
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (!useQkL2normAttr) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] use_qk_l2norm currently must be true\n");
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (useGateInKernel) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] use_gate_in_kernel currently must be false\n");
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (hasCuSeqlens && B != 1) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] varlen requires B=1, got B=%ld\n", B);
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (hasCuSeqlens && !hasChunkIndices) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] varlen requires chunk_indices with cu_seqlens\n");
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }
    if (allowNegEigval && !useBetaSigmoid) {
        printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] allow_neg_eigval requires use_beta_sigmoid\n");
        fflush(stdout);
        return ge::GRAPH_FAILED;
    }

    printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] B=%ld Hk=%ld Hv=%ld T=%ld K=%ld V=%ld chunkSize=%ld seqNum=%ld totalChunks=%ld\n",
           B, HK, HV, T, K, V, chunkSize, seqNum, totalChunks);
    printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] flags: useQkL2norm=1 (qHat=%d kHat=%d qRstd=%d kRstd=%d) "
           "useGate=%d (aLog=%d dtBias=%d) useBetaSigmoid=%d allowNegEigval=%d useExp2=%d "
           "isVarlen=%d chunkIndices=%d\n",
           static_cast<int>(hasQHat), static_cast<int>(hasKHat),
           static_cast<int>(hasQRstd), static_cast<int>(hasKRstd),
           static_cast<int>(useGateInKernel), static_cast<int>(hasALog), static_cast<int>(hasDtBias),
           static_cast<int>(useBetaSigmoid), static_cast<int>(allowNegEigval), static_cast<int>(useExp2),
           static_cast<int>(hasCuSeqlens), static_cast<int>(hasChunkIndices));
    printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] dtype: q=%d g=%d beta=%d HRatio=%ld\n",
           static_cast<int>(qDtype), static_cast<int>(gDtype), static_cast<int>(betaDtype),
           HK == 0 ? 0 : (HV / HK));
    fflush(stdout);

    ChunkGatedDeltaRuleFwdPrepareTilingData tiling;
    tiling.set_inputBatchSize(B);
    tiling.set_queryKeyHeadCount(HK);
    tiling.set_valueHeadCount(HV);
    tiling.set_sequenceTokenLength(T);
    tiling.set_queryKeyHeadDim(K);
    tiling.set_valueHeadDim(V);
    tiling.set_valueHeadsPerQueryKeyHead(HK == 0 ? 0 : (HV / HK));
    tiling.set_tokensPerChunk(chunkSize);
    tiling.set_packedSequenceCount(seqNum);
    tiling.set_isVariableLengthPacked(hasCuSeqlens ? 1 : 0);
    tiling.set_hasChunkIndexTable(hasChunkIndices ? 1 : 0);
    tiling.set_enableQueryKeyL2NormInKernel(1);
    tiling.set_enableFusedGateSoftplus(useGateInKernel ? 1 : 0);
    tiling.set_enableBetaSigmoid(useBetaSigmoid ? 1 : 0);
    tiling.set_hasGateDtBias(hasDtBias ? 1 : 0);
    tiling.set_hasQueryHatGmOutput(hasQHat ? 1 : 0);
    tiling.set_hasKeyHatGmOutput(hasKHat ? 1 : 0);
    tiling.set_scaleBetaByTwoWhenNegEigval(allowNegEigval ? 1 : 0);
    tiling.set_useExp2ForGateCumsum(useExp2 ? 1 : 0);
    tiling.set_queryKeyStorageDtype(static_cast<int64_t>(qDtype));
    tiling.set_gateStorageDtype(static_cast<int64_t>(gDtype));
    tiling.set_betaStorageDtype(static_cast<int64_t>(betaDtype));
    tiling.set_totalChunkTileCount(totalChunks);
    tiling.set_reservedEightByteAlignPad(0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    // Kernel GET_TILING_DATA opParaSize is GetDataSize()+8 on this CANN; mismatch
    // leaves the tiling GM unbound (MTE 0x80000000).
    const size_t tilingBytes = tiling.GetDataSize();
    const size_t tilingBytesDevice = tilingBytes + 8;
    context->GetRawTilingData()->SetDataSize(tilingBytesDevice);
    printf("[ChunkGatedDeltaRuleFwdPrepare][Tiling] tilingBytes=%zu deviceBytes=%zu capacity=%zu coreNum=%ld\n",
           tilingBytes, tilingBytesDevice, context->GetRawTilingData()->GetCapacity(), coreNum);
    fflush(stdout);
    context->SetTilingKey(0);
    context->SetBlockDim(static_cast<uint32_t>(coreNum > 0 ? coreNum : 1));

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Per-core user WS: gmWsY 16 KiB + gmWsA 4×16 KiB, padded to 128 KiB.
    // Must match kWsPerCoreBytes.
    ws[0] = sysWorkspaceSize + static_cast<size_t>(coreNum) * 128 * 1024;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4ChunkGatedDeltaRuleFwdPrepare(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct ChunkGatedDeltaRuleFwdPrepareCompileInfo {};

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwdPrepare)
    .Tiling(Tiling4ChunkGatedDeltaRuleFwdPrepare)
    .TilingParse<ChunkGatedDeltaRuleFwdPrepareCompileInfo>(TilingParse4ChunkGatedDeltaRuleFwdPrepare);

} // namespace optiling
