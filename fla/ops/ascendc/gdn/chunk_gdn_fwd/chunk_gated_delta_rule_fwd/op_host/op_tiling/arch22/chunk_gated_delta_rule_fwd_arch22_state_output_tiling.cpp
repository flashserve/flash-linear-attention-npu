/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "../../../op_kernel/internal/arch22/pipeline/wu_ho/chunk_recompute_wu_fwd_ho_tiling.h"

#include "../../../op_kernel/internal/arch22/stages/fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.h"
#include "../../../op_kernel/internal/arch22/stages/fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"
#include "../../../op_kernel/internal/arch22/stages/fwd_o/op_kernel/chunk_fwd_o_struct.h"
#include "../../../op_kernel/internal/arch22/stages/recompute_wu/op_host/op_tiling/recompute_w_u_fwd_tiling_processor.h"
#include "../../../op_kernel/internal/arch22/pipeline/wu_ho/chunk_recompute_wu_fwd_ho_struct.h"

#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <algorithm>
#include <cinttypes>
#include <initializer_list>
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
constexpr size_t ATTR_RAW_G_LAYOUT = 4;
constexpr size_t ATTR_QKV_LAYOUT = 5;
constexpr size_t ATTR_O_LAYOUT = 6;

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
// HO 分离布局的 O 基址最终写入 int64 tiling 字段，需按 int64 上界核查。
constexpr size_t MAX_INT64_AS_SIZE_T = 0x7fffffffffffffffULL;

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

void FillHMacroTiling(const ::GdnMegaArch22FwdHTilingData &src,
                      GdnMegaArch22FwdHTilingData &dst)
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

bool CheckedWorkspaceBytes(std::initializer_list<size_t> factors, size_t &bytes)
{
    bytes = 1;
    for (const size_t factor : factors) {
        if (factor != 0 && bytes > MAX_INT64_AS_SIZE_T / factor) {
            return false;
        }
        bytes *= factor;
    }
    return true;
}

bool AppendWorkspace(size_t bytes, size_t &offset)
{
    if (bytes > MAX_INT64_AS_SIZE_T - (WORKSPACE_ALIGNMENT - 1)) {
        return false;
    }
    const size_t alignedBytes = AlignUp(bytes, WORKSPACE_ALIGNMENT);
    if (offset > MAX_INT64_AS_SIZE_T - alignedBytes) {
        return false;
    }
    offset += alignedBytes;
    return true;
}

bool FillOTilingWorkspace(GDN::GdnMegaArch22FwdOTilingData &tiling, uint32_t aicCoreNum,
                         size_t hWorkStages, size_t baseOffset, size_t &endOffset)
{
    size_t vBytes = 0;
    size_t hBytes = 0;
    size_t attnBytes = 0;
    size_t maskBytes = 0;
    if (baseOffset > MAX_INT64_AS_SIZE_T ||
        !CheckedWorkspaceBytes({aicCoreNum, static_cast<size_t>(tiling.chunkSize),
                                static_cast<size_t>(tiling.vHeadDim), sizeof(float), PING_PONG_STAGES}, vBytes) ||
        !CheckedWorkspaceBytes({aicCoreNum, static_cast<size_t>(tiling.chunkSize),
                                static_cast<size_t>(tiling.vHeadDim), sizeof(float), hWorkStages}, hBytes) ||
        !CheckedWorkspaceBytes({aicCoreNum, static_cast<size_t>(tiling.chunkSize),
                                static_cast<size_t>(tiling.chunkSize), sizeof(float), PING_PONG_STAGES}, attnBytes) ||
        !CheckedWorkspaceBytes({static_cast<size_t>(tiling.chunkSize),
                                static_cast<size_t>(tiling.chunkSize)}, maskBytes)) {
        return false;
    }
    size_t offset = baseOffset;
    tiling.vWorkspaceOffset = static_cast<int64_t>(offset);
    if (!AppendWorkspace(vBytes, offset)) { return false; }
    tiling.hWorkspaceOffset = static_cast<int64_t>(offset);
    if (!AppendWorkspace(hBytes, offset)) { return false; }
    tiling.attnWorkspaceOffset = static_cast<int64_t>(offset);
    if (!AppendWorkspace(attnBytes, offset)) { return false; }
    tiling.aftermaskWorkspaceOffset = static_cast<int64_t>(offset);
    if (!AppendWorkspace(attnBytes, offset)) { return false; }
    tiling.maskWorkspaceOffset = static_cast<int64_t>(offset);
    if (!AppendWorkspace(maskBytes, offset) || !AppendWorkspace(WORKSPACE_RESERVE, offset)) {
        return false;
    }
    endOffset = offset;
    return true;
}

struct OwnerHWorkspace {
    size_t v{0}, decay{0}, kDecay{0}, h{0}, numSeq{0}, numChunks{0}, end{0}, shiftedEnd{0};
};

bool AddOwnerBytes(size_t bytes, size_t &offset)
{
    if (offset > MAX_INT64_AS_SIZE_T || bytes > MAX_INT64_AS_SIZE_T - offset) { return false; }
    offset += bytes;
    return true;
}

bool AppendOwnerH(size_t bytes, size_t &offset)
{
    // 原H的AH(n)=(n+512)/512*512；整除时也多512，不能换普通AlignUp。
    if (bytes > MAX_INT64_AS_SIZE_T - WORKSPACE_ALIGNMENT) { return false; }
    return AddOwnerBytes((bytes + WORKSPACE_ALIGNMENT) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT, offset);
}

bool CheckOwnerHBeforeProcessor(const ChunkGatedDeltaRuleFwdHTilingContext &ctx,
                                size_t recomputeWorkspace, size_t elementSize, OwnerHWorkspace &expected)
{
    if (ctx.seqlen <= 0 || ctx.vNumHead <= 0 || ctx.shapeBatchDim <= 0 ||
        ctx.kHeadDim != 128 || ctx.vHeadDim != 128 || ctx.chunkSize != 64 || ctx.useGk ||
        (ctx.hasCuSeqlens && ctx.cuSeqlensDim0 <= 1) ||
        ctx.libApiWorkSpaceSize > MAX_INT64_AS_SIZE_T ||
        recomputeWorkspace < ctx.libApiWorkSpaceSize || recomputeWorkspace > MAX_INT64_AS_SIZE_T) { return false; }
    const size_t tokenBatch = ctx.hasCuSeqlens ? static_cast<size_t>(ctx.cuSeqlensDim0 - 1) : 1;
    size_t wh = 0, delta = 0, meta = 0;
    if (tokenBatch >= MAX_INT64_AS_SIZE_T ||
        !CheckedWorkspaceBytes({ctx.aicCoreNum, 2, 64, 128, sizeof(float)}, wh) ||
        !CheckedWorkspaceBytes({ctx.aicCoreNum, 2, 128, 128, sizeof(float)}, delta) ||
        !CheckedWorkspaceBytes({tokenBatch + 1, sizeof(int64_t)}, meta)) { return false; }
    size_t off = ctx.libApiWorkSpaceSize;
    if (!AddOwnerBytes(WORKSPACE_RESERVE, off)) { return false; }
    expected.v = off;
    if (!AppendOwnerH(wh, off)) { return false; }
    expected.decay = off;
    if (!AppendOwnerH(wh, off)) { return false; }
    expected.kDecay = expected.h = off;
    if (!AppendOwnerH(delta, off)) { return false; }
    expected.numSeq = off;
    if (!AppendOwnerH(meta, off)) { return false; }
    expected.numChunks = off;
    if (!AppendOwnerH(meta, off) || !AddOwnerBytes(WORKSPACE_RESERVE, off)) { return false; }
    expected.end = off;
    // 旧W/U布局的每项乘、对齐、相加在执行原表达式前预检，包含hoShift。
    size_t wBytes = 0, uBytes = 0;
    if (!CheckedWorkspaceBytes({static_cast<size_t>(ctx.shapeBatchDim), static_cast<size_t>(ctx.vNumHead),
                                static_cast<size_t>(ctx.seqlen), 128, elementSize}, wBytes) ||
        !CheckedWorkspaceBytes({static_cast<size_t>(ctx.shapeBatchDim), static_cast<size_t>(ctx.vNumHead),
                                static_cast<size_t>(ctx.seqlen), 128, elementSize}, uBytes)) { return false; }
    size_t wuEnd = 0;
    if (!AppendWorkspace(recomputeWorkspace - ctx.libApiWorkSpaceSize, wuEnd) ||
        !AppendWorkspace(wBytes, wuEnd) || !AppendWorkspace(uBytes, wuEnd)) { return false; }
    size_t defaultBase = ctx.libApiWorkSpaceSize;
    if (!AddOwnerBytes(WORKSPACE_RESERVE, defaultBase)) { return false; }
    size_t hoBase = 0;
    if (!AppendWorkspace(std::max(defaultBase, wuEnd), hoBase)) { return false; }
    expected.shiftedEnd = expected.end;
    return AddOwnerBytes(hoBase - defaultBase, expected.shiftedEnd);
}

} // namespace

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdArch22StateOutput(gert::TilingContext *context,
                                                               bool separateHoWorkspace,
                                                               bool ownerLocalWorkspace)
{
    OP_LOGD(context->GetNodeName(), "Tiling4ChunkGatedDeltaRuleFwdArch22StateOutput start.");
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

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *qkvLayoutAttr = attrs->GetAttrPointer<int64_t>(ATTR_QKV_LAYOUT);
    const int64_t qkvLayout = qkvLayoutAttr == nullptr ? 0 : *qkvLayoutAttr;
    OP_CHECK_IF(qkvLayout != 0 && qkvLayout != 1,
                OP_LOGE(context->GetNodeName(), "qkv_layout must be 0 or 1."), return ge::GRAPH_FAILED);
    auto LogicalQkvShape = [qkvLayout](const gert::StorageShape *input) {
        gert::StorageShape logical = *input;
        if (qkvLayout == 1) {
            auto &shape = logical.MutableStorageShape();
            const int64_t tokens = shape.GetDim(1);
            shape.SetDim(1, shape.GetDim(2));
            shape.SetDim(2, tokens);
        }
        return logical;
    };
    const auto logicalQ = LogicalQkvShape(qShapePtr);
    const auto logicalK = LogicalQkvShape(kShapePtr);
    const auto logicalV = LogicalQkvShape(vShapePtr);
    const gert::Shape qShape = logicalQ.GetStorageShape();
    const gert::Shape kShape = logicalK.GetStorageShape();
    const gert::Shape vShape = logicalV.GetStorageShape();
    const gert::Shape betaShape = betaShapePtr->GetStorageShape();
    const gert::Shape aShape = aShapePtr->GetStorageShape();
    const gert::Shape gShape = gShapePtr->GetStorageShape();
    const int64_t batch = qShape.GetDim(DIM_BATCH);
    const int64_t kNumHead = qShape.GetDim(DIM_HEAD);
    const int64_t seqlen = qShape.GetDim(DIM_TOKEN);
    const int64_t kHeadDim = qShape.GetDim(DIM_CHANNEL);
    const int64_t vNumHead = vShape.GetDim(DIM_HEAD);
    const int64_t vHeadDim = vShape.GetDim(DIM_CHANNEL);
    const int64_t *rawGLayoutAttr = attrs->GetAttrPointer<int64_t>(ATTR_RAW_G_LAYOUT);
    const int64_t rawGLayout = rawGLayoutAttr == nullptr ? 0 : *rawGLayoutAttr;
    OP_CHECK_IF(rawGLayout != 0 && rawGLayout != 1,
                OP_LOGE(context->GetNodeName(), "raw_g_layout must be 0 (BHT) or 1 (BTH)."),
                return ge::GRAPH_FAILED);
    const int64_t *oLayoutAttr = attrs->GetAttrPointer<int64_t>(ATTR_O_LAYOUT);
    const int64_t oLayout = oLayoutAttr == nullptr ? 0 : *oLayoutAttr;
    OP_CHECK_IF(oLayout != 0 && oLayout != 1,
                OP_LOGE(context->GetNodeName(), "o_layout must be 0 or 1."), return ge::GRAPH_FAILED);

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
    OP_CHECK_IF((rawGLayout == 0 &&
                 (gShape.GetDim(0) != batch || gShape.GetDim(1) != vNumHead || gShape.GetDim(2) != seqlen)) ||
                    (rawGLayout == 1 &&
                     (gShape.GetDim(0) != batch || gShape.GetDim(1) != seqlen ||
                      gShape.GetDim(2) != vNumHead)),
                OP_LOGE(context->GetNodeName(),
                        "g layout contract is 0:[B,HV,T] or 1:[B,T,HV]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vNumHead % kNumHead != 0,
                OP_LOGE(context->GetNodeName(), "vNumHead must be divisible by kNumHead."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kHeadDim != SUPPORTED_K ||
                    (vHeadDim != SUPPORTED_V128 && vHeadDim != SUPPORTED_V256),
                OP_LOGE(context->GetNodeName(), "Phase 5 fused path supports K=128 and V=128/256."),
                return ge::GRAPH_FAILED);

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
    GDN::GdnMegaArch22RecomputeWUTilingData recomputeTiling{};
    GdnArch22RecomputeWUFwdTilingContext recomputeContext{
        context->GetNodeName(),
        &logicalK,
        &logicalV,
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
    recomputeContext.gIsBth = rawGLayout == 1;
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, recomputeContext.ubSize);
    GdnArch22RecomputeWUFwdTilingProcessor recomputeProcessor(recomputeContext, recomputeTiling);
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

    const size_t elementSize = qDesc->GetDataType() == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    OwnerHWorkspace expectedOwner{};
    OP_CHECK_IF(ownerLocalWorkspace &&
                    !CheckOwnerHBeforeProcessor(hContext, recomputeProcessor.GetWorkspaceSize(), elementSize, expectedOwner),
                OP_LOGE(context->GetNodeName(),
                        "Owner-local H/WU workspace precheck failed: region=H/WU, B=%" PRId64 ", Hv=%" PRId64
                        ", T=%" PRId64 ", N=%" PRId64 ", K=%" PRId64 ", V=%" PRId64 ", BT=%" PRId64
                        ", cores=%u, elementBytes=%zu, sysBytes=%zu, recomputeBytes=%zu; require K=V=128, BT=64,"
                        " no gk, sysBytes<=recomputeBytes and each size/offset <= INT64_MAX=%zu.",
                        batch, vNumHead, seqlen, tokenBatch, kHeadDim, vHeadDim, chunkSize, aicCoreNum,
                        elementSize, sysWorkspaceSize, recomputeProcessor.GetWorkspaceSize(), MAX_INT64_AS_SIZE_T),
                return ge::GRAPH_FAILED);
    ::GdnMegaArch22FwdHTilingData hPlain{};
    uint32_t hBlockDim = 0;
    size_t hWorkspaceSize = 0;
    ChunkGatedDeltaRuleFwdHTilingProcessor hProcessor(hContext);
    hProcessor.Process(hPlain, hBlockDim, hWorkspaceSize);
    OP_CHECK_IF(ownerLocalWorkspace &&
                    (hPlain.vWorkspaceOffset != static_cast<int64_t>(expectedOwner.v) ||
                     hPlain.vUpdateWorkspaceOffset != static_cast<int64_t>(expectedOwner.decay) ||
                     hPlain.kDecayWorkspaceOffset != static_cast<int64_t>(expectedOwner.kDecay) ||
                     hPlain.hWorkspaceOffset != static_cast<int64_t>(expectedOwner.h) ||
                     hPlain.numSeqWorkspaceOffset != static_cast<int64_t>(expectedOwner.numSeq) ||
                     hPlain.numChunksWorkspaceOffset != static_cast<int64_t>(expectedOwner.numChunks) ||
                     hWorkspaceSize != expectedOwner.end),
                OP_LOGE(context->GetNodeName(), "Owner-local H workspace differs from the checked original layout."),
                return ge::GRAPH_FAILED);

    GdnMegaArch22FwdHTilingData hTiling;
    FillHMacroTiling(hPlain, hTiling);
    GDN::GdnMegaArch22FwdOTilingData oTiling{};
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
    auto ShiftHWorkspace = [hoShift](GdnMegaArch22FwdHTilingData &tiling) {
        tiling.set_vWorkspaceOffset(tiling.get_vWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_vUpdateWorkspaceOffset(tiling.get_vUpdateWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_kDecayWorkspaceOffset(tiling.get_kDecayWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_hWorkspaceOffset(tiling.get_hWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_numSeqWorkspaceOffset(tiling.get_numSeqWorkspaceOffset() + static_cast<int64_t>(hoShift));
        tiling.set_numChunksWorkspaceOffset(tiling.get_numChunksWorkspaceOffset() + static_cast<int64_t>(hoShift));
    };
    ShiftHWorkspace(hTiling);
    hWorkspaceSize += hoShift;
    OP_CHECK_IF(ownerLocalWorkspace && hWorkspaceSize != expectedOwner.shiftedEnd,
                OP_LOGE(context->GetNodeName(), "Owner-local shifted H workspace differs from the checked end."),
                return ge::GRAPH_FAILED);
    // 只扩静态 prepared BT64/V128 的专用 H 区，后续 O 区及完整 reserve 原大小顺延。
    // Device 仍以实际 H/attn offset 间距决定 H4；旧路径可继续使用前两槽。
    const size_t oHWorkStages = platform.GetCurNpuArch() == NpuArch::DAV_2201 &&
                                   rawGLayout == 1 && oLayout == 1 && chunkSize == CHUNK_64 &&
                                   vHeadDim == SUPPORTED_V128 && kHeadDim == SUPPORTED_K ? 4 : PING_PONG_STAGES;
    size_t oBase = hoBase;
    if (separateHoWorkspace) {
        size_t oSpanWithReserve = 0;
        OP_CHECK_IF(!FillOTilingWorkspace(oTiling, aicCoreNum, oHWorkStages, 0, oSpanWithReserve),
                    OP_LOGE(context->GetNodeName(), "Separate HO O span exceeds int64 range."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(hWorkspaceSize > MAX_INT64_AS_SIZE_T - WORKSPACE_ALIGNMENT,
                    OP_LOGE(context->GetNodeName(), "Separate HO O-base offset exceeds int64 range."),
                    return ge::GRAPH_FAILED);
        oBase = AlignUp(hWorkspaceSize, WORKSPACE_ALIGNMENT);
        OP_CHECK_IF(oBase > MAX_INT64_AS_SIZE_T - oSpanWithReserve,
                    OP_LOGE(context->GetNodeName(), "Separate HO O region exceeds int64 range."),
                    return ge::GRAPH_FAILED);
    }
    size_t oWorkspaceSize = 0;
    OP_CHECK_IF(!FillOTilingWorkspace(oTiling, aicCoreNum, oHWorkStages, oBase, oWorkspaceSize),
                OP_LOGE(context->GetNodeName(),
                        "O workspace limit exceeded: region=O, base=%zu, cores=%u, BT=%" PRId64 ", V=%" PRId64
                        ", hStages=%zu, pingPongStages=%" PRId64 ", elementBytes=%zu, alignment=%zu, reserveBytes=%zu;"
                        " require base+aligned(v,h,attn,aftermask,mask,reserve) <= INT64_MAX=%zu.",
                        oBase, aicCoreNum, chunkSize, vHeadDim, oHWorkStages, PING_PONG_STAGES,
                        sizeof(float), WORKSPACE_ALIGNMENT, WORKSPACE_RESERVE, MAX_INT64_AS_SIZE_T),
                return ge::GRAPH_FAILED);

    size_t workspaceOffset = 0;
    OP_CHECK_IF(!AppendWorkspace(std::max(hWorkspaceSize, oWorkspaceSize), workspaceOffset),
                OP_LOGE(context->GetNodeName(),
                        "HO workspace alignment limit exceeded: region=HO, hEnd=%zu, oEnd=%zu, alignment=%zu;"
                        " require AlignUp(max(hEnd,oEnd),alignment) <= INT64_MAX=%zu.",
                        hWorkspaceSize, oWorkspaceSize, WORKSPACE_ALIGNMENT, MAX_INT64_AS_SIZE_T),
                return ge::GRAPH_FAILED);
    GDN::ChunkRecomputeWUFwdHOTrailer trailer{};
    trailer.recompute = recomputeTiling;
    trailer.recomputeWorkspaceOffset = 0;
    trailer.wIntermediateOffset = static_cast<int64_t>(wOffset);
    trailer.uIntermediateOffset = static_cast<int64_t>(uOffset);
    trailer.qDataType = DtypeToEnum(qDesc->GetDataType());
    trailer.betaDataType = DtypeToEnum(betaDesc->GetDataType());
    size_t hIntermediateBytes = 0;
    size_t vNewIntermediateBytes = 0;
    OP_CHECK_IF(!CheckedWorkspaceBytes({static_cast<size_t>(isVarlen ? 1 : batch),
                                        static_cast<size_t>(vNumHead), static_cast<size_t>(totalChunks),
                                        static_cast<size_t>(kHeadDim), static_cast<size_t>(vHeadDim), elementSize},
                                       hIntermediateBytes) ||
                    !CheckedWorkspaceBytes({static_cast<size_t>(batch), static_cast<size_t>(vNumHead),
                                            static_cast<size_t>(seqlen), static_cast<size_t>(vHeadDim), elementSize},
                                           vNewIntermediateBytes),
                OP_LOGE(context->GetNodeName(),
                        "HO persistent workspace size limit exceeded: region=%s, B=%" PRId64 ", hBatch=%" PRId64
                        ", Hv=%" PRId64 ", T=%" PRId64 ", NT=%" PRId64 ", K=%" PRId64 ", V=%" PRId64
                        ", elementBytes=%zu; require hBatch*Hv*NT*K*V*elementBytes and B*Hv*T*V*elementBytes"
                        " <= INT64_MAX=%zu.",
                        vNewIntermediateBytes == 0 ? "h" : "vNew", batch, isVarlen ? int64_t{1} : batch,
                        vNumHead, seqlen, totalChunks, kHeadDim, vHeadDim, elementSize, MAX_INT64_AS_SIZE_T),
                return ge::GRAPH_FAILED);
    trailer.hIntermediateOffset = static_cast<int64_t>(workspaceOffset);
    OP_CHECK_IF(!AppendWorkspace(hIntermediateBytes, workspaceOffset),
                OP_LOGE(context->GetNodeName(),
                        "Persistent workspace end limit exceeded: region=h, offset=%zu, bytes=%zu, alignment=%zu;"
                        " require bytes<=INT64_MAX-(alignment-1) and offset+AlignUp(bytes,alignment) <= INT64_MAX=%zu.",
                        workspaceOffset, hIntermediateBytes, WORKSPACE_ALIGNMENT, MAX_INT64_AS_SIZE_T),
                return ge::GRAPH_FAILED);
    trailer.vNewIntermediateOffset = static_cast<int64_t>(workspaceOffset);
    OP_CHECK_IF(!AppendWorkspace(vNewIntermediateBytes, workspaceOffset) ||
                    !AppendWorkspace(WORKSPACE_RESERVE, workspaceOffset),
                OP_LOGE(context->GetNodeName(),
                        "Persistent workspace end limit exceeded: region=%s, offset=%zu, vNewBase=%" PRId64
                        ", vNewBytes=%zu, reserveBytes=%zu, alignment=%zu; require each appended bytes"
                        " <= INT64_MAX-(alignment-1) and offset+AlignUp(bytes,alignment) <= INT64_MAX=%zu.",
                        workspaceOffset == static_cast<size_t>(trailer.vNewIntermediateOffset) ? "vNew" : "reserve",
                        workspaceOffset, trailer.vNewIntermediateOffset, vNewIntermediateBytes,
                        WORKSPACE_RESERVE, WORKSPACE_ALIGNMENT, MAX_INT64_AS_SIZE_T),
                return ge::GRAPH_FAILED);

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
