#include <cstring>
#include "../../op_kernel/arch92/chunk_gated_delta_rule_bwd_finalize_policy.h"
#include "chunk_gated_delta_rule_bwd_finalize_tiling.h"
#include "../chunk_gated_delta_rule_bwd_finalize_validation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <initializer_list>

#include <register/op_impl_registry.h>
#include "platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"

#include "../arch22/chunk_gated_delta_rule_bwd_finalize_tiling_arch22.h"
#include <string>
#include "graph/utils/type_utils.h"

using namespace GDN;

namespace optiling {
namespace {

std::string ShapeText(const gert::StorageShape *storage)
{
    if (storage == nullptr) {
        return "absent";
    }
    const auto &shape = storage->GetStorageShape();
    std::string text = "rank=" + std::to_string(shape.GetDimNum()) + ",[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            text += ",";
        }
        text += std::to_string(shape.GetDim(i));
    }
    return text + "]";
}


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
bool AddressFits(std::initializer_list<int64_t> factors)
{
    uint64_t bytes = sizeof(uint16_t);
    for (const int64_t factor : factors) {
        if (factor < 0 || (factor != 0 &&
            bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / factor)) {
            return false;
        }
        bytes *= factor;
    }
    return true;
}

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + (value % divisor != 0);
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

ge::graphStatus ValidateShapes(gert::TilingContext *context,
                               GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData &tiling,
                               bool stateVFirst, int64_t inputLayout)
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
    if (!CheckRank(qShape, 4) || !CheckRank(kShape, 4) || !CheckRank(vShape, 4) ||
                    !CheckRank(vNewShape, 4) || !CheckRank(dOShape, 4) || !CheckRank(duShape, 4) ||
                    !CheckRank(gShape, 3) || !CheckRank(betaShape, 3) || !CheckRank(hShape, 5) ||
                    !CheckRank(dhShape, 5) || !CheckRank(aShape, 4)) {
        const auto rankText = [](const gert::StorageShape *shape) {
            return shape == nullptr ? std::string("null")
                                    : std::to_string(shape->GetStorageShape().GetDimNum());
        };
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(),
            "q,k,v,v_new,d_o,du,g,beta,h,dh,A",
            rankText(qShape) + "," + rankText(kShape) + "," + rankText(vShape) + "," +
                rankText(vNewShape) + "," + rankText(dOShape) + "," + rankText(duShape) + "," +
                rankText(gShape) + "," + rankText(betaShape) + "," + rankText(hShape) + "," +
                rankText(dhShape) + "," + rankText(aShape),
            "Required input rank check failed");
        return ge::GRAPH_FAILED;
    }

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
    tiling.HK = (inputLayout == GDN::FinalizeLayout::BSND) ? q.GetDim(2) : q.GetDim(1);
    tiling.T = (inputLayout == GDN::FinalizeLayout::BSND) ? q.GetDim(1) : q.GetDim(2);
    tiling.K = q.GetDim(3);
    tiling.HV = (inputLayout == GDN::FinalizeLayout::BSND) ? v.GetDim(2) : v.GetDim(1);
    tiling.V = v.GetDim(3);
    tiling.chunkSize = a.GetDim(3);
    if (tiling.B <= 0 || tiling.HK <= 0 || tiling.HV <= 0 || tiling.T <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "B/HK/HV/T",
            std::to_string(tiling.B) + "," + std::to_string(tiling.HK) + "," +
                std::to_string(tiling.HV) + "," + std::to_string(tiling.T),
            "B, HK, HV and T must be positive");
        return ge::GRAPH_FAILED;
    }
    if (tiling.K != K_SIZE_128 || tiling.V != V_SIZE_128) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "K/V",
            std::to_string(tiling.K) + "," + std::to_string(tiling.V),
            "K and V must both be 128");
        return ge::GRAPH_FAILED;
    }
    if (tiling.chunkSize != CHUNK_SIZE_64) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "A",
            std::to_string(tiling.chunkSize), "BT derived from A must be 64");
        return ge::GRAPH_FAILED;
    }
    tiling.headRatio = tiling.HV / tiling.HK;
    if (tiling.HV % tiling.HK != 0 || tiling.headRatio < 1 || tiling.headRatio > FinalizeHost::kA5PublishedMaxG) {
        const std::string actual = "HK=" + std::to_string(tiling.HK) + ", HV=" + std::to_string(tiling.HV);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "HV/HK", actual, "integer in [1, 16] on A5/A6");
        return ge::GRAPH_FAILED;
    }

    if (k.GetDim(0) != tiling.B ||
                    ((inputLayout == GDN::FinalizeLayout::BSND) ?
                         (k.GetDim(1) != tiling.T || k.GetDim(2) != tiling.HK) :
                         (k.GetDim(1) != tiling.HK || k.GetDim(2) != tiling.T)) ||
                    k.GetDim(3) != tiling.K) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "q/k",
                    std::string("q=") +
                        (std::string("[") + std::to_string(q.GetDim(0)) + "," + std::to_string(q.GetDim(1)) + "," +
                         std::to_string(q.GetDim(2)) + "," + std::to_string(q.GetDim(3)) + "]") +
                        ", k=" +
                        (std::string("[") + std::to_string(k.GetDim(0)) + "," + std::to_string(k.GetDim(1)) + "," +
                         std::to_string(k.GetDim(2)) + "," + std::to_string(k.GetDim(3)) + "]"),
                    "k must have the same shape as q in the selected layout");
        return ge::GRAPH_FAILED;
    }
    if (v.GetDim(0) != tiling.B ||
                    ((inputLayout == GDN::FinalizeLayout::BSND) ? (v.GetDim(1) != tiling.T || v.GetDim(2) != tiling.HV)
                                       : (v.GetDim(1) != tiling.HV || v.GetDim(2) != tiling.T)) ||
                    vNew.GetDim(0) != tiling.B || vNew.GetDim(1) != tiling.HV ||
                    vNew.GetDim(2) != tiling.T || vNew.GetDim(3) != tiling.V ||
                    dO.GetDim(0) != tiling.B || dO.GetDim(1) != tiling.HV ||
                    dO.GetDim(2) != tiling.T || dO.GetDim(3) != tiling.V ||
                    du.GetDim(0) != tiling.B || du.GetDim(1) != tiling.HV ||
                    du.GetDim(2) != tiling.T || du.GetDim(3) != tiling.V) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "v",
            std::to_string(v.GetDim(0)) + "," + std::to_string(v.GetDim(1)) + "," +
                std::to_string(v.GetDim(2)) + "," + std::to_string(v.GetDim(3)),
            "v, v_new, do and du must have shape [B, HV, T, V]");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape g = gShape->GetStorageShape();
    const gert::Shape beta = betaShape->GetStorageShape();
    if (g.GetDim(0) != tiling.B || g.GetDim(1) != tiling.HV || g.GetDim(2) != tiling.T ||
                    beta.GetDim(0) != tiling.B || beta.GetDim(1) != tiling.HV || beta.GetDim(2) != tiling.T) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "g,beta",
            std::to_string(g.GetDim(0)) + "," + std::to_string(g.GetDim(1)) + "," +
                std::to_string(g.GetDim(2)) + "," + std::to_string(beta.GetDim(0)) + "," +
                std::to_string(beta.GetDim(1)) + "," + std::to_string(beta.GetDim(2)),
            "g and beta must have shape [B, HV, T]");
        return ge::GRAPH_FAILED;
    }
    if (a.GetDim(0) != tiling.B || a.GetDim(1) != tiling.HV || a.GetDim(2) != tiling.T) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "A",
            std::to_string(a.GetDim(0)) + "," + std::to_string(a.GetDim(1)) + "," +
                std::to_string(a.GetDim(2)) + "," + std::to_string(a.GetDim(3)),
            "A must have shape [B, HV, T, 64]");
        return ge::GRAPH_FAILED;
    }
    const int64_t stateDim3 = stateVFirst ? tiling.V : tiling.K;
    const int64_t stateDim4 = stateVFirst ? tiling.K : tiling.V;
    if (h.GetDim(0) != tiling.B || h.GetDim(2) != tiling.HV ||
                    h.GetDim(3) != stateDim3 || h.GetDim(4) != stateDim4 ||
                    dh.GetDim(0) != tiling.B || dh.GetDim(2) != tiling.HV ||
                    dh.GetDim(1) != h.GetDim(1) || dh.GetDim(3) != stateDim3 ||
                    dh.GetDim(4) != stateDim4) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "h/dh",
            "state_v_first=" + std::to_string(static_cast<int>(stateVFirst)),
            "h and dh state dimensions do not match for the given state_v_first");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus Tiling4ChunkGatedDeltaRuleBwdFinalize(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *stateVFirst = attrs->GetAttrPointer<bool>(5);
    const bool stateVFirstValue = stateVFirst == nullptr ? false : *stateVFirst;
    const char *qkvLayout = attrs->GetStr(7);
    if (qkvLayout == nullptr) {
        qkvLayout = "BNSD";
    }
    if (std::strcmp(qkvLayout, "BNSD") != 0 && std::strcmp(qkvLayout, "BSND") != 0 &&
                    std::strcmp(qkvLayout, "NTD") != 0 && std::strcmp(qkvLayout, "TND") != 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "layout", qkvLayout, "BNSD, BSND, NTD or TND");
        return ge::GRAPH_FAILED;
    }
    const int64_t layoutValue =
        (std::strcmp(qkvLayout, "BSND") == 0 || std::strcmp(qkvLayout, "TND") == 0)
            ? GDN::FinalizeLayout::BSND : GDN::FinalizeLayout::BNSD;
    tiling->layout = layoutValue;
    // A2/A3 使用私有 tiling；A5 保留基线属性读取、shape 和 workspace 流程。
    const platform_ascendc::PlatformAscendC routePlatform(context->GetPlatformInfo());
    const auto npuArch = routePlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        return gdn_finalize_arch22::Tiling(context, routePlatform);
    }
    bool isArch92 = false;
#ifdef GDN_FINALIZE_ENABLE_ARCH92
    isArch92 = npuArch == NpuArch::DAV_9201;
#endif
    if (npuArch != NpuArch::DAV_3510 && !isArch92) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "npuArch", std::to_string(static_cast<int32_t>(npuArch)),
            "DAV_3510 or DAV_9201 required; DAV_2201 uses the arch22 private tiling");
        return ge::GRAPH_FAILED;
    }
    // 必选输入判空必须先于 ValidateShapes 的 rank 检查：
    // 缺参 → 161001/EZ0004，rank 不符 → 161002。
    const auto hShape = context->GetRequiredInputShape(INPUT_H_IDX);
    if (hShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "h");
        return ge::GRAPH_FAILED;
    }
    const auto dhShape = context->GetRequiredInputShape(INPUT_DH_IDX);
    if (dhShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "dh");
        return ge::GRAPH_FAILED;
    }
    if (ValidateShapes(context, *tiling, stateVFirstValue, layoutValue) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const double *scale = attrs->GetAttrPointer<double>(0);
    const int32_t *chunkSize = attrs->GetAttrPointer<int32_t>(1);
    const bool *useL2Norm = attrs->GetAttrPointer<bool>(2);
    const bool *useBetaSigmoid = attrs->GetAttrPointer<bool>(3);
    const bool *useGate = attrs->GetAttrPointer<bool>(4);
    const bool *useExp2 = attrs->GetAttrPointer<bool>(6);
    const bool useL2NormValue = useL2Norm == nullptr ? false : *useL2Norm;
    const bool useBetaSigmoidValue = useBetaSigmoid == nullptr ? false : *useBetaSigmoid;
    const bool useGateValue = useGate == nullptr ? false : *useGate;
    const bool useExp2Value = useExp2 == nullptr ? true : *useExp2;
    const int64_t actualChunkSize = chunkSize == nullptr ? tiling->chunkSize : *chunkSize;
    if (actualChunkSize != tiling->chunkSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "chunkSize",
            std::to_string(actualChunkSize), "chunkSize only supports 64");
        return ge::GRAPH_FAILED;
    }
    if (useGateValue) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "use_gate_in_kernel", useGateValue ? "true" : "false", "use_gate_in_kernel only supports false");
        return ge::GRAPH_FAILED;
    }
    const auto cuShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
    const auto chunkIndicesShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
    if ((cuShape == nullptr) != (chunkIndicesShape == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "cu_seqlens,chunk_indices",
            std::string("cu=") + (cuShape == nullptr ? "absent" : "present") + ",chunk=" +
                (chunkIndicesShape == nullptr ? "absent" : "present"),
            "cu_seqlens and chunk_indices must be both present or absent");
        return ge::GRAPH_FAILED;
    }
    tiling->isVariable = cuShape != nullptr ? 1 : 0;
    if (tiling->isVariable != 0) {
    if (!CheckRank(cuShape, 1) || !CheckRank(chunkIndicesShape, 1) || tiling->B != 1) {
          OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "B,cu_seqlens,chunk_indices",
          std::to_string(tiling->B) + "," +
          std::to_string(cuShape->GetStorageShape().GetDimNum()) + "," +
          std::to_string(chunkIndicesShape->GetStorageShape().GetDimNum()),
          "Varlen requires B=1 and rank-1 cu_seqlens/chunk_indices");
          return ge::GRAPH_FAILED;
    }
        const gert::Shape cu = cuShape->GetStorageShape();
        const gert::Shape chunks = chunkIndicesShape->GetStorageShape();
        // aclIntArray 转成的 chunk_indices 是一维扁平数组，按
        // [seqIdx0, chunkIdx0, seqIdx1, chunkIdx1, ...] 两两分组。
    if (cu.GetDim(0) < 2 || chunks.GetDim(0) <= 0 || chunks.GetDim(0) % 2 != 0) {
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "cu_seqlens/chunk_indices",
          std::to_string(cu.GetDim(0)) + "," + std::to_string(chunks.GetDim(0)),
          "Invalid varlen index shapes");
          return ge::GRAPH_FAILED;
    }
        tiling->seqNum = cu.GetDim(0) - 1;
        tiling->totalChunkNum = chunks.GetDim(0) / 2;
        tiling->chunkNumForT = 0;
    } else {
        tiling->seqNum = tiling->B;
        tiling->chunkNumForT = CeilDiv(tiling->T, tiling->chunkSize);
        if ((isArch92 || tiling->headRatio > 4) && tiling->B > std::numeric_limits<int64_t>::max() / tiling->chunkNumForT) {
            const std::string actual = "B=" + std::to_string(tiling->B) +
                ", chunksPerSequence=" + std::to_string(tiling->chunkNumForT);
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "chunk count", actual,
                "B * chunksPerSequence must fit in int64_t");
            return ge::GRAPH_FAILED;
        }
        tiling->totalChunkNum = tiling->B * tiling->chunkNumForT;
    }
    const int64_t stateChunkNum = tiling->isVariable != 0 ? tiling->totalChunkNum : tiling->chunkNumForT;
    if (hShape->GetStorageShape().GetDim(1) != stateChunkNum) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "h/dh",
            std::to_string(hShape->GetStorageShape().GetDim(1)) + "," + std::to_string(stateChunkNum),
            "h/dh NT does not match the chunk schedule");
        return ge::GRAPH_FAILED;
    }

    if ((isArch92 || tiling->headRatio > 4) &&
        (!AddressFits({tiling->B, tiling->HV, tiling->T, tiling->K}) ||
         !AddressFits({tiling->B, stateChunkNum, tiling->HV, tiling->K, tiling->V}))) {
        const std::string actual = "B=" + std::to_string(tiling->B) + ", HV=" + std::to_string(tiling->HV) +
            ", T=" + std::to_string(tiling->T) + ", K=" + std::to_string(tiling->K) +
            ", NT=" + std::to_string(stateChunkNum) + ", V=" + std::to_string(tiling->V);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tensor byte count", actual,
            "2*B*HV*T*K and 2*B*NT*HV*K*V must each fit in int64_t");
        return ge::GRAPH_FAILED;
    }

    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t aicCoreNum = std::max<uint32_t>(1, platform.GetCoreNumAic());
    if (isArch92) {
        aicCoreNum = std::min(platform.GetCoreNumAic(), platform.GetCoreNumAiv());
        uint64_t ubBytes = 0;
        uint64_t l1Bytes = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Bytes);
        if (aicCoreNum == 0 || ubBytes < FinalizeArch92::UB_TOTAL_BYTES ||
            l1Bytes < FinalizeArch92::L1_TOTAL_BYTES) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "A6 core/UB/L1 capacity",
                std::to_string(aicCoreNum) + "," + std::to_string(ubBytes) + "," + std::to_string(l1Bytes),
                "A6 requires paired AIC/AIV cores, at least 372 KiB UB and 456 KiB L1");
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t maxHeadsPerTask = isArch92 ? FinalizeArch92::HEAD_COUNT : 4;
    if (tiling->totalChunkNum > std::numeric_limits<int64_t>::max() / tiling->HK) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "chunkCount,HK",
            std::to_string(tiling->totalChunkNum) + "," + std::to_string(tiling->HK),
            "The chunk/GVA task count overflows");
        return ge::GRAPH_FAILED;
    }
    const int64_t totalGvaTaskNum = tiling->totalChunkNum * tiling->HK;
    if (tiling->headRatio > maxHeadsPerTask) {
        // One logical task owns a complete (chunk, HK); maxHeadsPerTask HVs per window.
        tiling->taskGroupSize = maxHeadsPerTask;
        tiling->headGroupNum = tiling->HK;
        tiling->taskNum = totalGvaTaskNum;
    } else {
        const int64_t maxGvaPerTask = maxHeadsPerTask / tiling->headRatio;
        int64_t gvaPerTask = std::min(
            maxGvaPerTask, CeilDiv(totalGvaTaskNum, static_cast<int64_t>(aicCoreNum)));
        // 同一 HK 对应的 GVA 个 HV 是不可拆分原子。在 dhu 类似的
        // 目标 group size 下向下取能整除 HK 的最大值，避免轻量尾组。
        while (gvaPerTask > 1 && tiling->HK % gvaPerTask != 0) {
            --gvaPerTask;
        }
        tiling->taskGroupSize = gvaPerTask * tiling->headRatio;
        tiling->headGroupNum = tiling->HK / gvaPerTask;
        if (tiling->totalChunkNum > std::numeric_limits<int64_t>::max() / tiling->headGroupNum) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "chunkTaskNum",
                "totalChunkNum=" + std::to_string(tiling->totalChunkNum) +
                    ",headGroupNum=" + std::to_string(tiling->headGroupNum),
                "The flattened chunk/head-group task count overflows");
            return ge::GRAPH_FAILED;
        }
        tiling->taskNum = tiling->totalChunkNum * tiling->headGroupNum;
    }

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
    if (qDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "q");
        return ge::GRAPH_FAILED;
    }
    if (kDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "k");
        return ge::GRAPH_FAILED;
    }
    if (vDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "v");
        return ge::GRAPH_FAILED;
    }
    if (gDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "g");
        return ge::GRAPH_FAILED;
    }
    if (betaDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "beta");
        return ge::GRAPH_FAILED;
    }
    if (vNewDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "v_new");
        return ge::GRAPH_FAILED;
    }
    if (dODesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "dO");
        return ge::GRAPH_FAILED;
    }
    if (duDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "du");
        return ge::GRAPH_FAILED;
    }
    if (hDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "h");
        return ge::GRAPH_FAILED;
    }
    if (dhDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "dh");
        return ge::GRAPH_FAILED;
    }
    if (aDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "a");
        return ge::GRAPH_FAILED;
    }
    const int qKey = DTypeKey(qDesc->GetDataType());
    const int gKey = DTypeKey(gDesc->GetDataType());
    const int betaKey = DTypeKey(betaDesc->GetDataType());
    if (qKey != TPL_BF16 ||
                    (gKey != TPL_BF16 && gKey != TPL_FP32) ||
                    (betaKey != TPL_BF16 && betaKey != TPL_FP32)) {
        const std::string qDtypeName = ge::TypeUtils::DataTypeToSerialString(qDesc->GetDataType());
        const std::string gDtypeName = ge::TypeUtils::DataTypeToSerialString(gDesc->GetDataType());
        const std::string betaDtypeName = ge::TypeUtils::DataTypeToSerialString(betaDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "q,g,beta",
            qDtypeName + "," + gDtypeName + "," + betaDtypeName,
            "Unsupported q, g or beta dtype");
        return ge::GRAPH_FAILED;
    }
    if (gDesc->GetDataType() != betaDesc->GetDataType()) {
        const std::string gBetaDtypeName = ge::TypeUtils::DataTypeToSerialString(gDesc->GetDataType()) + "," +
            ge::TypeUtils::DataTypeToSerialString(betaDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "g,beta",
            gBetaDtypeName, "g and beta must use the same dtype");
        return ge::GRAPH_FAILED;
    }
    if (kDesc->GetDataType() != qDesc->GetDataType() ||
                    vDesc->GetDataType() != qDesc->GetDataType() ||
                    vNewDesc->GetDataType() != qDesc->GetDataType() ||
                    dODesc->GetDataType() != qDesc->GetDataType() ||
                    duDesc->GetDataType() != qDesc->GetDataType() ||
                    hDesc->GetDataType() != qDesc->GetDataType() ||
                    dhDesc->GetDataType() != qDesc->GetDataType() ||
                    aDesc->GetDataType() != qDesc->GetDataType()) {
        const std::string actual =
            std::string("q=") + ge::TypeUtils::DataTypeToSerialString(qDesc->GetDataType()) +
            ",k=" + ge::TypeUtils::DataTypeToSerialString(kDesc->GetDataType()) +
            ",v=" + ge::TypeUtils::DataTypeToSerialString(vDesc->GetDataType()) +
            ",v_new=" + ge::TypeUtils::DataTypeToSerialString(vNewDesc->GetDataType()) +
            ",dO=" + ge::TypeUtils::DataTypeToSerialString(dODesc->GetDataType()) +
            ",du=" + ge::TypeUtils::DataTypeToSerialString(duDesc->GetDataType()) +
            ",h=" + ge::TypeUtils::DataTypeToSerialString(hDesc->GetDataType()) +
            ",dh=" + ge::TypeUtils::DataTypeToSerialString(dhDesc->GetDataType()) +
            ",A=" + ge::TypeUtils::DataTypeToSerialString(aDesc->GetDataType());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "q,k,v,v_new,dO,du,h,dh,A",
            actual, "Main tensors must use BF16");
        return ge::GRAPH_FAILED;
    }

    // 可选输入按实际启用的 Stage 15/Stage 12 功能校验。
    const auto qRstdShape = context->GetOptionalInputShape(INPUT_Q_RSTD_IDX);
    const auto kRstdShape = context->GetOptionalInputShape(INPUT_K_RSTD_IDX);
    const auto betaRawShape = context->GetOptionalInputShape(INPUT_BETA_RAW_IDX);
    const auto qRstdDesc = context->GetOptionalInputDesc(INPUT_Q_RSTD_IDX);
    const auto kRstdDesc = context->GetOptionalInputDesc(INPUT_K_RSTD_IDX);
    const auto betaRawDesc = context->GetOptionalInputDesc(INPUT_BETA_RAW_IDX);
    if (useL2NormValue) {
    if (!CheckRank(qRstdShape, 3) || !CheckRank(kRstdShape, 3) ||
                        qRstdDesc == nullptr || kRstdDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "q_rstd,k_rstd",
            std::string("q_rstd=") + (qRstdDesc == nullptr ? "absent" : "present") + ",k_rstd=" +
                (kRstdDesc == nullptr ? "absent" : "present"),
            "q_rstd and k_rstd are required when qk L2Norm is enabled");
        return ge::GRAPH_FAILED;
    }
        const gert::Shape qRstd = qRstdShape->GetStorageShape();
        const gert::Shape kRstd = kRstdShape->GetStorageShape();
    if (qRstd.GetDim(0) != tiling->B || qRstd.GetDim(1) != tiling->HK ||
          qRstd.GetDim(2) != tiling->T ||
          kRstd.GetDim(0) != tiling->B || kRstd.GetDim(1) != tiling->HK ||
                        kRstd.GetDim(2) != tiling->T) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "q_rstd,k_rstd",
            ShapeText(qRstdShape) + "," + ShapeText(kRstdShape),
            "q_rstd and k_rstd must both be [B, HK, T]");
        return ge::GRAPH_FAILED;
    }
    if (qRstdDesc->GetDataType() != ge::DT_FLOAT || kRstdDesc->GetDataType() != ge::DT_FLOAT) {
          OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "q_rstd,k_rstd",
          ge::TypeUtils::DataTypeToSerialString(qRstdDesc->GetDataType()) + "," +
          ge::TypeUtils::DataTypeToSerialString(kRstdDesc->GetDataType()),
          "q_rstd and k_rstd must be fp32");
          return ge::GRAPH_FAILED;
    }
    }
    if (useBetaSigmoidValue) {
    if (!CheckRank(betaRawShape, 3) || betaRawDesc == nullptr) {
          OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "beta_raw");
          return ge::GRAPH_FAILED;
    }
        const gert::Shape betaRaw = betaRawShape->GetStorageShape();
    if (betaRaw.GetDim(0) != tiling->B || betaRaw.GetDim(1) != tiling->HV ||
                        betaRaw.GetDim(2) != tiling->T) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "beta_raw,beta",
            ShapeText(betaRawShape) + "," + ShapeText(context->GetRequiredInputShape(INPUT_BETA_IDX)),
            "beta_raw must be [B, HV, T] and match beta shape");
        return ge::GRAPH_FAILED;
    }
    if (betaRawDesc->GetDataType() != betaDesc->GetDataType()) {
          OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "beta_raw,beta",
          ge::TypeUtils::DataTypeToSerialString(betaRawDesc->GetDataType()) + "," +
          ge::TypeUtils::DataTypeToSerialString(betaDesc->GetDataType()),
          "beta_raw dtype must match beta dtype");
          return ge::GRAPH_FAILED;
    }
    }

    tiling->scale = scale == nullptr ? 1.0f : static_cast<float>(*scale);
    tiling->stateVFirst = stateVFirstValue ? 1 : 0;

    const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint64_t>(aicCoreNum, tiling->taskNum));
    // A5 uses two four-head banks; A6 uses three three-head banks. Every
    // valid physical window advances the bank serial on both engines.
    const uint64_t vectorBytes = static_cast<uint64_t>(tiling->chunkSize) * tiling->K * sizeof(uint16_t);
    const uint64_t workspaceBufferCount = isArch92 ? FinalizeArch92::WORKSPACE_BUFFER_COUNT : 8;
    const uint64_t workspaceRegionCount = isArch92 ? FinalizeArch92::WORKSPACE_REGION_COUNT : 7;
    const uint64_t privateBytes = tiling->headRatio > 4 ?
        static_cast<uint64_t>(blockDim) * 2 * GDN::GVA_ACC_CORE_BYTES : 0;
    const uint64_t regionBytesPerCore = workspaceBufferCount * workspaceRegionCount * vectorBytes;
    const uint64_t accumulatorBytesPerCore = tiling->headRatio > 4 ? 2 * GDN::GVA_ACC_CORE_BYTES : 0;
    const uint64_t bytesPerCore = regionBytesPerCore + accumulatorBytesPerCore;
    if (isArch92 && static_cast<uint64_t>(blockDim) >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / bytesPerCore) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace bytes",
            std::to_string(blockDim), "A6 workspace byte offsets must fit in int64_t");
        return ge::GRAPH_FAILED;
    }
    const uint64_t userWorkspace = static_cast<uint64_t>(blockDim) * regionBytesPerCore + privateBytes;
    if (userWorkspace > std::numeric_limits<size_t>::max() - platform.GetLibApiWorkSpaceSize()) {
        const std::string actual = "userBytes=" + std::to_string(userWorkspace) +
            ", systemBytes=" + std::to_string(platform.GetLibApiWorkSpaceSize());
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "workspace bytes", actual,
            "userBytes + systemBytes must fit in size_t");
        return ge::GRAPH_FAILED;
    }
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() + userWorkspace;
    OP_LOGD(context->GetNodeName(),"workspace = [%zu]", workspaceSizes[0]);

    // 主张量固定 BF16，g/beta 共用一个 BF16 或 FP32 模板参数；
    // 三个 backward/指数开关独立控制输入搬运和 VF 公式，共 16 个模板。
    // state_v_first 只选择 state GM 布局解释，作为运行时 tiling 数据不扩展 key。
    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(qKey), static_cast<uint64_t>(gKey),
        static_cast<uint64_t>(useL2NormValue), static_cast<uint64_t>(useBetaSigmoidValue),
        static_cast<uint64_t>(useExp2Value));
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
