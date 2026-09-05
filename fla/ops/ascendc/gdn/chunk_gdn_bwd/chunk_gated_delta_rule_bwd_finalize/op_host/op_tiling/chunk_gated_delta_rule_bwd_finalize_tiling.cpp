#include "chunk_gated_delta_rule_bwd_finalize_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace GDN;

namespace optiling {
namespace {

enum InputIndex : size_t {
    INPUT_Q_IDX = 0,
    INPUT_K_IDX,
    INPUT_V_IDX,
    INPUT_V_NEW_IDX,
    INPUT_DO_IDX,
    INPUT_DU_IDX,
    INPUT_G_IDX,
    INPUT_BETA_IDX,
    INPUT_H_IDX,
    INPUT_DH_IDX,
    INPUT_A_IDX,
    INPUT_Q_RSTD_IDX,
    INPUT_K_RSTD_IDX,
    INPUT_BETA_RAW_IDX,
    INPUT_CU_SEQLENS_IDX,
    INPUT_CHUNK_INDICES_IDX,
};

constexpr int64_t CHUNK_SIZE_64 = 64;
constexpr int64_t K_SIZE_128 = 128;
constexpr int64_t V_SIZE_128 = 128;
int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int DTypeKey(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return TPL_BF16;
    }
    if (dtype == ge::DT_FLOAT) {
        return TPL_FP32;
    }
    return -1;
}

bool CheckRank(const gert::StorageShape *shape, size_t rank)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == rank;
}

ge::graphStatus ValidateShapes(gert::TilingContext *context, GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData &tiling)
{
    // 先从真实输入 shape 提取运行时规格，再校验当前仅支持的 64/128 档位；
    // 禁止用编译期常量代替 shape 推导结果写入 tiling data。
    const auto qShape = context->GetRequiredInputShape(INPUT_Q_IDX);
    const auto kShape = context->GetRequiredInputShape(INPUT_K_IDX);
    const auto vShape = context->GetRequiredInputShape(INPUT_V_IDX);
    const auto vNewShape = context->GetRequiredInputShape(INPUT_V_NEW_IDX);
    const auto dOShape = context->GetRequiredInputShape(INPUT_DO_IDX);
    const auto duShape = context->GetRequiredInputShape(INPUT_DU_IDX);
    const auto gShape = context->GetRequiredInputShape(INPUT_G_IDX);
    const auto betaShape = context->GetRequiredInputShape(INPUT_BETA_IDX);
    const auto hShape = context->GetRequiredInputShape(INPUT_H_IDX);
    const auto dhShape = context->GetRequiredInputShape(INPUT_DH_IDX);
    const auto aShape = context->GetRequiredInputShape(INPUT_A_IDX);
    OP_CHECK_IF(!CheckRank(qShape, 4) || !CheckRank(kShape, 4) || !CheckRank(vShape, 4) ||
                    !CheckRank(vNewShape, 4) || !CheckRank(dOShape, 4) || !CheckRank(duShape, 4) ||
                    !CheckRank(gShape, 3) || !CheckRank(betaShape, 3) || !CheckRank(hShape, 5) ||
                    !CheckRank(dhShape, 5) || !CheckRank(aShape, 4),
                OP_LOGE(context->GetNodeName(), "Required input rank check failed."), return ge::GRAPH_FAILED);

    const gert::Shape q = qShape->GetStorageShape();
    const gert::Shape k = kShape->GetStorageShape();
    const gert::Shape v = vShape->GetStorageShape();
    const gert::Shape vNew = vNewShape->GetStorageShape();
    const gert::Shape dO = dOShape->GetStorageShape();
    const gert::Shape du = duShape->GetStorageShape();
    const gert::Shape a = aShape->GetStorageShape();
    const gert::Shape h = hShape->GetStorageShape();
    const gert::Shape dh = dhShape->GetStorageShape();
    tiling.B = q.GetDim(0);
    tiling.HK = q.GetDim(1);
    tiling.T = q.GetDim(2);
    tiling.K = q.GetDim(3);
    tiling.HV = v.GetDim(1);
    tiling.V = v.GetDim(3);
    tiling.chunkSize = a.GetDim(3);
    OP_CHECK_IF(tiling.B <= 0 || tiling.HK <= 0 || tiling.HV <= 0 || tiling.T <= 0,
                OP_LOGE(context->GetNodeName(), "B, HK, HV and T must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.K != K_SIZE_128 || tiling.V != V_SIZE_128,
                OP_LOGE(context->GetNodeName(), "K and V must both be 128."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.chunkSize != CHUNK_SIZE_64,
                OP_LOGE(context->GetNodeName(), "BT derived from A must be 64."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.HV % tiling.HK != 0,
                OP_LOGE(context->GetNodeName(), "HV must be divisible by HK."), return ge::GRAPH_FAILED);
    tiling.headRatio = tiling.HV / tiling.HK;
    OP_CHECK_IF(tiling.headRatio < 1 || tiling.headRatio > 4,
                OP_LOGE(context->GetNodeName(), "G=HV/HK must be in [1, 4]."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(k.GetDim(0) != tiling.B || k.GetDim(1) != tiling.HK ||
                    k.GetDim(2) != tiling.T || k.GetDim(3) != tiling.K,
                OP_LOGE(context->GetNodeName(), "k must have the same shape as q."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(v.GetDim(0) != tiling.B || v.GetDim(2) != tiling.T ||
                    vNew.GetDim(0) != tiling.B || vNew.GetDim(1) != tiling.HV ||
                    vNew.GetDim(2) != tiling.T || vNew.GetDim(3) != tiling.V ||
                    dO.GetDim(0) != tiling.B || dO.GetDim(1) != tiling.HV ||
                    dO.GetDim(2) != tiling.T || dO.GetDim(3) != tiling.V ||
                    du.GetDim(0) != tiling.B || du.GetDim(1) != tiling.HV ||
                    du.GetDim(2) != tiling.T || du.GetDim(3) != tiling.V,
                OP_LOGE(context->GetNodeName(), "v, v_new, do and du must have shape [B, HV, T, V]."),
                return ge::GRAPH_FAILED);

    const gert::Shape g = gShape->GetStorageShape();
    const gert::Shape beta = betaShape->GetStorageShape();
    OP_CHECK_IF(g.GetDim(0) != tiling.B || g.GetDim(1) != tiling.HV || g.GetDim(2) != tiling.T ||
                    beta.GetDim(0) != tiling.B || beta.GetDim(1) != tiling.HV || beta.GetDim(2) != tiling.T,
                OP_LOGE(context->GetNodeName(), "g and beta must have shape [B, HV, T]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(a.GetDim(0) != tiling.B || a.GetDim(1) != tiling.HV || a.GetDim(2) != tiling.T,
                OP_LOGE(context->GetNodeName(), "A must have shape [B, HV, T, 64]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(h.GetDim(0) != tiling.B || h.GetDim(1) != tiling.HV ||
                    h.GetDim(3) != tiling.K || h.GetDim(4) != tiling.V ||
                    dh.GetDim(0) != tiling.B || dh.GetDim(1) != tiling.HV ||
                    dh.GetDim(2) != h.GetDim(2) || dh.GetDim(3) != tiling.K || dh.GetDim(4) != tiling.V,
                OP_LOGE(context->GetNodeName(), "h and dh must have shape [B, HV, NT, K, V]."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus Tiling4ChunkGatedDeltaRuleBwdFinalize(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(ValidateShapes(context, *tiling) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const double *scale = attrs->GetAttrPointer<double>(0);
    const int32_t *chunkSize = attrs->GetAttrPointer<int32_t>(1);
    const bool *useL2Norm = attrs->GetAttrPointer<bool>(2);
    const bool *useBetaSigmoid = attrs->GetAttrPointer<bool>(3);
    const bool *useGate = attrs->GetAttrPointer<bool>(4);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(5);
    const bool *useExp2 = attrs->GetAttrPointer<bool>(6);
    const bool useL2NormValue = useL2Norm == nullptr ? false : *useL2Norm;
    const bool useBetaSigmoidValue = useBetaSigmoid == nullptr ? false : *useBetaSigmoid;
    const bool useGateValue = useGate == nullptr ? false : *useGate;
    const bool useExp2Value = useExp2 == nullptr ? true : *useExp2;
    const bool stateVFirstValue = stateVFirst == nullptr ? false : *stateVFirst;
    const int64_t actualChunkSize = chunkSize == nullptr ? tiling->chunkSize : *chunkSize;
    OP_CHECK_IF(actualChunkSize != tiling->chunkSize,
                OP_LOGE(context->GetNodeName(), "chunk_size only supports 64."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(useGateValue,
                OP_LOGE(context->GetNodeName(), "use_gate_in_kernel only supports false."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!useExp2Value,
                OP_LOGE(context->GetNodeName(), "use_exp2 only supports true."),
                return ge::GRAPH_FAILED);

    const auto cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    const auto chunkIndicesShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    OP_CHECK_IF((cuShape == nullptr) != (chunkIndicesShape == nullptr),
                OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must be both present or absent."),
                return ge::GRAPH_FAILED);
    tiling->isVariable = cuShape != nullptr ? 1 : 0;
    if (tiling->isVariable != 0) {
        OP_CHECK_IF(!CheckRank(cuShape, 1) || !CheckRank(chunkIndicesShape, 1) || tiling->B != 1,
                    OP_LOGE(context->GetNodeName(),
                            "Varlen requires B=1 and rank-1 cu_seqlens/chunk_indices."),
                    return ge::GRAPH_FAILED);
        const gert::Shape cu = cuShape->GetStorageShape();
        const gert::Shape chunks = chunkIndicesShape->GetStorageShape();
        // aclIntArray 转成的 chunk_indices 是一维扁平数组，按
        // [seqIdx0, chunkIdx0, seqIdx1, chunkIdx1, ...] 两两分组。
        OP_CHECK_IF(cu.GetDim(0) < 2 || chunks.GetDim(0) <= 0 || chunks.GetDim(0) % 2 != 0,
                    OP_LOGE(context->GetNodeName(), "Invalid varlen index shapes."), return ge::GRAPH_FAILED);
        tiling->seqNum = cu.GetDim(0) - 1;
        tiling->totalChunkNum = chunks.GetDim(0) / 2;
        tiling->chunkNumForT = 0;
    } else {
        tiling->seqNum = tiling->B;
        tiling->chunkNumForT = CeilDiv(tiling->T, tiling->chunkSize);
        tiling->totalChunkNum = tiling->B * tiling->chunkNumForT;
    }
    const auto hShape = context->GetRequiredInputShape(INPUT_H_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    const int64_t stateChunkNum = tiling->isVariable != 0 ? tiling->totalChunkNum : tiling->chunkNumForT;
    OP_CHECK_IF(hShape->GetStorageShape().GetDim(2) != stateChunkNum,
                OP_LOGE(context->GetNodeName(), "h/dh NT does not match the chunk schedule."),
                return ge::GRAPH_FAILED);

    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t aicCoreNum = std::max<uint32_t>(1, platform.GetCoreNumAic());
    constexpr int64_t maxHeadsPerTask = 4;
    const int64_t maxGvaPerTask = maxHeadsPerTask / tiling->headRatio;
    OP_CHECK_IF(tiling->totalChunkNum > std::numeric_limits<int64_t>::max() / tiling->HK,
                OP_LOGE(context->GetNodeName(), "The chunk/GVA task count overflows."),
                return ge::GRAPH_FAILED);
    const int64_t totalGvaTaskNum = tiling->totalChunkNum * tiling->HK;
    int64_t gvaPerTask = std::min(
        maxGvaPerTask, CeilDiv(totalGvaTaskNum, static_cast<int64_t>(aicCoreNum)));
    // 同一 HK 对应的 GVA 个 HV 是不可拆分原子。在 dhu 类似的
    // 目标 group size 下向下取能整除 HK 的最大值，避免轻量尾组。
    while (gvaPerTask > 1 && tiling->HK % gvaPerTask != 0) {
        --gvaPerTask;
    }
    tiling->taskGroupSize = gvaPerTask * tiling->headRatio;
    tiling->headGroupNum = tiling->HK / gvaPerTask;
    OP_CHECK_IF(tiling->totalChunkNum > std::numeric_limits<int64_t>::max() / tiling->headGroupNum,
                OP_LOGE(context->GetNodeName(), "The flattened chunk/head-group task count overflows."),
                return ge::GRAPH_FAILED);
    tiling->taskNum = tiling->totalChunkNum * tiling->headGroupNum;

    const auto qDesc = context->GetInputDesc(INPUT_Q_IDX);
    const auto kDesc = context->GetInputDesc(INPUT_K_IDX);
    const auto vDesc = context->GetInputDesc(INPUT_V_IDX);
    const auto gDesc = context->GetInputDesc(INPUT_G_IDX);
    const auto betaDesc = context->GetInputDesc(INPUT_BETA_IDX);
    const auto vNewDesc = context->GetInputDesc(INPUT_V_NEW_IDX);
    const auto dODesc = context->GetInputDesc(INPUT_DO_IDX);
    const auto duDesc = context->GetInputDesc(INPUT_DU_IDX);
    const auto hDesc = context->GetInputDesc(INPUT_H_IDX);
    const auto dhDesc = context->GetInputDesc(INPUT_DH_IDX);
    const auto aDesc = context->GetInputDesc(INPUT_A_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vNewDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dODesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, duDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, hDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, dhDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, aDesc);
    const int qKey = DTypeKey(qDesc->GetDataType());
    const int gKey = DTypeKey(gDesc->GetDataType());
    const int betaKey = DTypeKey(betaDesc->GetDataType());
    OP_CHECK_IF(qKey != TPL_BF16 ||
                    (gKey != TPL_BF16 && gKey != TPL_FP32) ||
                    (betaKey != TPL_BF16 && betaKey != TPL_FP32),
                OP_LOGE(context->GetNodeName(), "Unsupported q, g or beta dtype."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gDesc->GetDataType() != betaDesc->GetDataType(),
                OP_LOGE(context->GetNodeName(), "g and beta must use the same dtype."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDesc->GetDataType() != qDesc->GetDataType() ||
                    vDesc->GetDataType() != qDesc->GetDataType() ||
                    vNewDesc->GetDataType() != qDesc->GetDataType() ||
                    dODesc->GetDataType() != qDesc->GetDataType() ||
                    duDesc->GetDataType() != qDesc->GetDataType() ||
                    hDesc->GetDataType() != qDesc->GetDataType() ||
                    dhDesc->GetDataType() != qDesc->GetDataType() ||
                    aDesc->GetDataType() != qDesc->GetDataType(),
                OP_LOGE(context->GetNodeName(), "Main tensors must use BF16."),
                return ge::GRAPH_FAILED);

    // 可选输入按实际启用的 Stage 15/Stage 12 功能校验。
    const auto qRstdShape = context->GetOptionalInputShape(INPUT_Q_RSTD_IDX);
    const auto kRstdShape = context->GetOptionalInputShape(INPUT_K_RSTD_IDX);
    const auto betaRawShape = context->GetOptionalInputShape(INPUT_BETA_RAW_IDX);
    const auto qRstdDesc = context->GetOptionalInputDesc(INPUT_Q_RSTD_IDX);
    const auto kRstdDesc = context->GetOptionalInputDesc(INPUT_K_RSTD_IDX);
    const auto betaRawDesc = context->GetOptionalInputDesc(INPUT_BETA_RAW_IDX);
    if (useL2NormValue) {
        OP_CHECK_IF(!CheckRank(qRstdShape, 3) || !CheckRank(kRstdShape, 3) ||
                        qRstdDesc == nullptr || kRstdDesc == nullptr,
                    OP_LOGE(context->GetNodeName(),
                            "q_rstd and k_rstd are required when qk L2Norm is enabled."),
                    return ge::GRAPH_FAILED);
        const gert::Shape qRstd = qRstdShape->GetStorageShape();
        const gert::Shape kRstd = kRstdShape->GetStorageShape();
        OP_CHECK_IF(qRstd.GetDim(0) != tiling->B || qRstd.GetDim(1) != tiling->HK ||
                        qRstd.GetDim(2) != tiling->T ||
                        kRstd.GetDim(0) != tiling->B || kRstd.GetDim(1) != tiling->HK ||
                        kRstd.GetDim(2) != tiling->T ||
                        qRstdDesc->GetDataType() != ge::DT_FLOAT ||
                        kRstdDesc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE(context->GetNodeName(),
                            "q_rstd and k_rstd must be fp32 [B, HK, T]."),
                    return ge::GRAPH_FAILED);
    }
    if (useBetaSigmoidValue) {
        OP_CHECK_IF(!CheckRank(betaRawShape, 3) || betaRawDesc == nullptr,
                    OP_LOGE(context->GetNodeName(),
                            "beta_raw is required when beta sigmoid backward is enabled."),
                    return ge::GRAPH_FAILED);
        const gert::Shape betaRaw = betaRawShape->GetStorageShape();
        OP_CHECK_IF(betaRaw.GetDim(0) != tiling->B || betaRaw.GetDim(1) != tiling->HV ||
                        betaRaw.GetDim(2) != tiling->T ||
                        betaRawDesc->GetDataType() != betaDesc->GetDataType(),
                    OP_LOGE(context->GetNodeName(),
                            "beta_raw must match beta shape and dtype."),
                    return ge::GRAPH_FAILED);
    }

    tiling->scale = scale == nullptr ? 1.0f : static_cast<float>(*scale);
    tiling->stateVFirst = stateVFirstValue ? 1 : 0;

    const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint64_t>(aicCoreNum, tiling->taskNum));
    // AIC/AIV 最多跨一个 task group 重叠。每个 AIC 使用两套固定
    // HEAD window 奇偶轮转，避免 workspace 随逻辑 task 数增长。
    const uint64_t vectorBytes = static_cast<uint64_t>(tiling->chunkSize) * tiling->K * sizeof(uint16_t);
    constexpr uint64_t workspaceBufferCount = 8; // 2 windows * 4 HEADs
    constexpr uint64_t workspaceRegionCount = 7;
    const uint64_t userWorkspace = static_cast<uint64_t>(blockDim) *
        workspaceBufferCount * workspaceRegionCount * vectorBytes;
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() + userWorkspace;

    // 主张量固定 BF16，g/beta 共用一个 BF16 或 FP32 模板参数；
    // 两个 backward 开关独立控制输入搬运和 VF 公式，共 8 个模板。
    // state_v_first 只选择 state GM 布局解释，作为运行时 tiling 数据不扩展 key。
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(qKey), static_cast<uint64_t>(gKey),
        static_cast<uint64_t>(useL2NormValue), static_cast<uint64_t>(useBetaSigmoidValue));
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleBwdFinalize(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleBwdFinalize)
    .Tiling(Tiling4ChunkGatedDeltaRuleBwdFinalize)
    .TilingParse<ChunkGatedDeltaRuleBwdFinalizeCompileInfo>(TilingPrepareForChunkGatedDeltaRuleBwdFinalize);

} // namespace optiling
