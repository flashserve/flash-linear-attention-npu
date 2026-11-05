#include "recurrent_gated_delta_rule_v2_tiling.h"

#include "err/ops_err.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

namespace optiling {
namespace {
constexpr size_t Q = 0, K = 1, V = 2, BETA = 3, STATE = 4, SEQ = 5, IDX = 6;
constexpr size_t G = 7, GK = 8, ACCEPTED = 9, ALOG = 10, DTBIAS = 11;
constexpr size_t ATTR_SCALE = 0, ATTR_QK_NORM = 1, ATTR_GATE = 2, ATTR_BETA_SIGMOID = 3, ATTR_ALLOW_NEG = 4;

template <typename S>
bool CopyStride(const S *src, uint64_t *dst, size_t n)
{
    if (src == nullptr || src->GetDimNum() != n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        const int64_t stride = src->GetStride(i);
        dst[i] = stride > 0 ? static_cast<uint64_t>(stride) : 0;
    }
    return true;
}

bool SpanFits(uint64_t count, uint64_t stride, uint64_t innerSpan, uint64_t &span)
{
    if (count == 0 || innerSpan == 0) {
        return false;
    }
    if (count == 1) {
        span = innerSpan;
        return true;
    }
    constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    if (stride > (maxValue - innerSpan) / (count - 1)) {
        return false;
    }
    span = (count - 1) * stride + innerSpan;
    return true;
}

bool DmaStrideFits(uint64_t tokenStride, uint64_t featureStride, uint64_t featureCount)
{
    constexpr uint64_t elementBytes = sizeof(uint16_t);
    constexpr uint64_t maxGapBytes = std::numeric_limits<uint32_t>::max();
    if (featureStride == 1) {
        return tokenStride >= featureCount && (tokenStride - featureCount) <= maxGapBytes / elementBytes;
    }
    return featureStride > 1 && (featureStride - 1) <= maxGapBytes / elementBytes;
}

} // namespace

REGISTER_OPS_TILING_TEMPLATE(RecurrentGatedDeltaRuleV2, RecurrentGatedDeltaRuleV2Tiling, 0);

ge::graphStatus RecurrentGatedDeltaRuleV2Tiling::CheckAndFill()
{
    const auto q = context_->GetInputShape(Q)->GetOriginShape();
    const auto k = context_->GetInputShape(K)->GetOriginShape();
    const auto v = context_->GetInputShape(V)->GetOriginShape();
    const auto beta = context_->GetInputShape(BETA)->GetOriginShape();
    const auto state = context_->GetInputShape(STATE)->GetOriginShape();
    const auto seq = context_->GetInputShape(SEQ)->GetOriginShape();
    const auto idx = context_->GetInputShape(IDX)->GetOriginShape();
    if (q.GetDimNum() != 3 || k.GetDimNum() != 3 || v.GetDimNum() != 3 || beta.GetDimNum() != 2 ||
        state.GetDimNum() != 4 || seq.GetDimNum() != 1 || (idx.GetDimNum() != 1 && idx.GetDimNum() != 2)) {
        OP_LOGE(context_->GetNodeName(), "V2 expects Q/K/V rank 3, beta rank 2, state rank 4, and index rank 1 or 2");
        return ge::GRAPH_FAILED;
    }
    if (q.GetDim(0) != k.GetDim(0) || q.GetDim(0) != v.GetDim(0) || q.GetDim(1) != k.GetDim(1) ||
        q.GetDim(2) != state.GetDim(3) || v.GetDim(1) != state.GetDim(1) || v.GetDim(2) != state.GetDim(2) ||
        beta.GetDim(0) != q.GetDim(0) || beta.GetDim(1) != v.GetDim(1) || q.GetDim(2) != 128 || v.GetDim(2) != 128 ||
        q.GetDim(1) <= 0 || v.GetDim(1) <= 0 || v.GetDim(1) % q.GetDim(1) != 0 || seq.GetDim(0) < 2) {
        OP_LOGE(context_->GetNodeName(), "V2 shape relation or A5 K=V=128 constraint failed");
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = seq.GetDim(0) - 1;
    if ((idx.GetDimNum() == 1 && idx.GetDim(0) != q.GetDim(0)) ||
        (idx.GetDimNum() == 2 && (idx.GetDim(0) != batch || idx.GetDim(1) <= 0))) {
        OP_LOGE(context_->GetNodeName(), "ssm_state_indices must be [T] or [B, W]");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOptionalInputDesc(GK) != nullptr) {
        OP_LOGE(context_->GetNodeName(), "gk is not supported by RecurrentGatedDeltaRuleV2");
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOptionalInputDesc(G) == nullptr) {
        OP_LOGE(context_->GetNodeName(), "RecurrentGatedDeltaRuleV2 requires g");
        return ge::GRAPH_FAILED;
    }
    const auto attrs = context_->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrPointer<float>(ATTR_SCALE) == nullptr ||
        !std::isfinite(*attrs->GetAttrPointer<float>(ATTR_SCALE))) {
        OP_LOGE(context_->GetNodeName(), "scale must be finite");
        return ge::GRAPH_FAILED;
    }
    tilingData_.t = static_cast<uint32_t>(q.GetDim(0));
    tilingData_.nk = static_cast<uint32_t>(q.GetDim(1));
    tilingData_.dk = static_cast<uint32_t>(q.GetDim(2));
    tilingData_.nv = static_cast<uint32_t>(v.GetDim(1));
    tilingData_.dv = static_cast<uint32_t>(v.GetDim(2));
    tilingData_.sBlockNum = static_cast<uint32_t>(state.GetDim(0));
    tilingData_.b = static_cast<uint32_t>(seq.GetDim(0) - 1);
    tilingData_.ssmStateStride = idx.GetDimNum() == 2 ? static_cast<uint32_t>(idx.GetDim(1)) : 0;
    tilingData_.scale = *attrs->GetAttrPointer<float>(ATTR_SCALE);
    tilingData_.useQkL2norm = *attrs->GetAttrPointer<bool>(ATTR_QK_NORM) ? 1 : 0;
    tilingData_.useGateInKernel = *attrs->GetAttrPointer<bool>(ATTR_GATE) ? 1 : 0;
    tilingData_.useBetaSigmoid = *attrs->GetAttrPointer<bool>(ATTR_BETA_SIGMOID) ? 1 : 0;
    tilingData_.allowNegEigval = *attrs->GetAttrPointer<bool>(ATTR_ALLOW_NEG) ? 1 : 0;
    tilingData_.hasAcceptedTokens = context_->GetOptionalInputDesc(ACCEPTED) == nullptr ? 0 : 1;
    tilingData_.hasDtBias = context_->GetOptionalInputDesc(DTBIAS) == nullptr ? 0 : 1;
    if (tilingData_.useGateInKernel && context_->GetOptionalInputDesc(ALOG) == nullptr) {
        OP_LOGE(context_->GetNodeName(), "use_gate_in_kernel requires g and A_log");
        return ge::GRAPH_FAILED;
    }
    if (tilingData_.useBetaSigmoid == 0) {
        tilingData_.allowNegEigval = 0;
    }
    return FillStrides();
}

ge::graphStatus RecurrentGatedDeltaRuleV2Tiling::FillStrides()
{
    uint64_t q[3]{}, k[3]{}, v[3]{}, state[4]{}, beta[2]{}, out[3]{};
    if (!CopyStride(context_->GetInputStride(Q), q, 3)) {
        q[0] = static_cast<uint64_t>(tilingData_.nk) * tilingData_.dk;
        q[1] = tilingData_.dk;
        q[2] = 1;
    }
    if (!CopyStride(context_->GetInputStride(K), k, 3)) {
        k[0] = static_cast<uint64_t>(tilingData_.nk) * tilingData_.dk;
        k[1] = tilingData_.dk;
        k[2] = 1;
    }
    if (!CopyStride(context_->GetInputStride(V), v, 3)) {
        v[0] = static_cast<uint64_t>(tilingData_.nv) * tilingData_.dv;
        v[1] = tilingData_.dv;
        v[2] = 1;
    }
    if (!CopyStride(context_->GetInputStride(BETA), beta, 2)) {
        beta[0] = tilingData_.nv;
        beta[1] = 1;
    }
    if (!CopyStride(context_->GetInputStride(STATE), state, 4)) {
        state[0] = static_cast<uint64_t>(tilingData_.nv) * tilingData_.dv * tilingData_.dk;
        state[1] = static_cast<uint64_t>(tilingData_.dv) * tilingData_.dk;
        state[2] = tilingData_.dk;
        state[3] = 1;
    }
    if (!CopyStride(context_->GetOutputStride(0), out, 3)) {
        out[0] = static_cast<uint64_t>(tilingData_.nv) * tilingData_.dv;
        out[1] = tilingData_.dv;
        out[2] = 1;
    }
    tilingData_.queryTokenStride = q[0];
    tilingData_.queryHeadStride = q[1];
    tilingData_.queryFeatureStride = q[2];
    tilingData_.keyTokenStride = k[0];
    tilingData_.keyHeadStride = k[1];
    tilingData_.keyFeatureStride = k[2];
    tilingData_.valueTokenStride = v[0];
    tilingData_.valueHeadStride = v[1];
    tilingData_.valueFeatureStride = v[2];
    tilingData_.stateInStride0 = state[0];
    tilingData_.stateInStride1 = state[1];
    tilingData_.stateInStride2 = state[2];
    tilingData_.stateInStride3 = state[3];
    tilingData_.stateOutStride0 = state[0];
    tilingData_.stateOutStride1 = state[1];
    tilingData_.stateOutStride2 = state[2];
    tilingData_.stateOutStride3 = state[3];
    uint64_t qSpan = 0, kSpan = 0, vSpan = 0;
    uint64_t qTokenSpan = 0, kTokenSpan = 0, vTokenSpan = 0;
    const bool qSpanOk = SpanFits(tilingData_.dk, q[2], 1, qSpan) && SpanFits(tilingData_.nk, q[1], qSpan, qTokenSpan);
    const bool kSpanOk = SpanFits(tilingData_.dk, k[2], 1, kSpan) && SpanFits(tilingData_.nk, k[1], kSpan, kTokenSpan);
    const bool vSpanOk = SpanFits(tilingData_.dv, v[2], 1, vSpan) && SpanFits(tilingData_.nv, v[1], vSpan, vTokenSpan);
    const uint64_t stateRowSpan = static_cast<uint64_t>(tilingData_.dv) * tilingData_.dk;
    uint64_t stateBlockSpan = 0, stateHeadSpan = 0;
    const bool stateBlockMajor = state[1] >= stateRowSpan &&
                                 SpanFits(tilingData_.nv, state[1], stateRowSpan, stateBlockSpan) &&
                                 state[0] >= stateBlockSpan;
    const bool stateHeadMajor = state[0] >= stateRowSpan &&
                                SpanFits(tilingData_.sBlockNum, state[0], stateRowSpan, stateHeadSpan) &&
                                state[1] >= stateHeadSpan;
    if (q[2] != 1 || k[2] != 1 || v[2] != 1 || state[3] != 1 || !qSpanOk || !kSpanOk || !vSpanOk || q[1] < qSpan ||
        k[1] < kSpan || v[1] < vSpan || q[0] < qTokenSpan || k[0] < kTokenSpan || v[0] < vTokenSpan ||
        !DmaStrideFits(q[0], q[2], tilingData_.dk) || !DmaStrideFits(k[0], k[2], tilingData_.dk) ||
        !DmaStrideFits(v[0], v[2], tilingData_.dv) || state[2] != tilingData_.dk ||
        (!stateBlockMajor && !stateHeadMajor) || beta[0] != static_cast<uint64_t>(tilingData_.nv) || beta[1] != 1 ||
        out[0] != static_cast<uint64_t>(tilingData_.nv) * tilingData_.dv || out[1] != tilingData_.dv || out[2] != 1) {
        OP_LOGE(context_->GetNodeName(),
                "unsupported strides: q=[%lu,%lu,%lu], k=[%lu,%lu,%lu], v=[%lu,%lu,%lu], "
                "state=[%lu,%lu,%lu,%lu], beta=[%lu,%lu], out=[%lu,%lu,%lu]",
                q[0], q[1], q[2], k[0], k[1], k[2], v[0], v[1], v[2], state[0], state[1], state[2], state[3], beta[0],
                beta[1], out[0], out[1], out[2]);
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOptionalInputDesc(G) != nullptr) {
        uint64_t g[2]{};
        if (!CopyStride(context_->GetOptionalInputStride(G), g, 2)) {
            g[0] = tilingData_.nv;
            g[1] = 1;
        }
        if (g[0] != static_cast<uint64_t>(tilingData_.nv) || g[1] != 1) {
            OP_LOGE(context_->GetNodeName(), "V2 scalar g requires dense head stride");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleV2Tiling::GetShapeAttrsInfo()
{
    return CheckAndFill();
}

ge::graphStatus RecurrentGatedDeltaRuleV2Tiling::DoOpTiling()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t aiv = platform.GetCoreNumAiv();
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint64_t tasks = static_cast<uint64_t>(tilingData_.b) * tilingData_.nv;
    tilingData_.vectorCoreNum = static_cast<uint32_t>(tasks == 0 ? 1 : std::min(aiv, tasks));
    tilingData_.stateOutBufferNum = 1;
    tilingData_.attnOutBufferNum = 1;
    tilingData_.stateVFirst = 1;

    // Every supported dtype/layout uses the same three-bank, four-token pipeline.
    constexpr uint64_t TOKEN_TILE_4 = 4;
    constexpr uint64_t STATE_BANKS_3 = 3;
    constexpr uint64_t INPUT_BUFFERS_2 = 2;
    constexpr uint64_t bf16Bytes = 2;
    constexpr uint64_t fp32Bytes = 4;
    constexpr uint64_t dataBlockBytes = 32;
    constexpr uint64_t ubSafetyBytes = 4U * 1024U;
    constexpr uint64_t dim = 128;
    const uint64_t pipelineBytes =
        STATE_BANKS_3 * dim * dim * fp32Bytes + INPUT_BUFFERS_2 * TOKEN_TILE_4 * 3U * dim * bf16Bytes +
        INPUT_BUFFERS_2 * TOKEN_TILE_4 * dataBlockBytes + INPUT_BUFFERS_2 * TOKEN_TILE_4 * dataBlockBytes +
        INPUT_BUFFERS_2 * TOKEN_TILE_4 * dim * bf16Bytes + dim * fp32Bytes;
    if (pipelineBytes + ubSafetyBytes > ub) {
        OP_LOGE(context_->GetNodeName(), "insufficient UB for unified V2 pipeline: need=%lu, ub=%lu",
                pipelineBytes + ubSafetyBytes, ub);
        return ge::GRAPH_FAILED;
    }
    tilingData_.vStep = static_cast<uint32_t>(dim);
    tilingData_.ubRestBytes = 0;
    tilingData_.outputFinalState = 1;
    tilingData_.inplaceFinalState = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleV2Tiling::DoLibApiTiling()
{
    const uint64_t stateKey =
        context_->GetInputDesc(STATE)->GetDataType() == ge::DT_FLOAT ? RGDR_V2_TPL_FP32 : RGDR_V2_TPL_BF16;
    const uint64_t gKey = context_->GetOptionalInputDesc(G) != nullptr &&
                                  context_->GetOptionalInputDesc(G)->GetDataType() == ge::DT_FLOAT ?
                              RGDR_V2_TPL_FP32 :
                              RGDR_V2_TPL_BF16;
    const uint64_t betaKey =
        context_->GetInputDesc(BETA)->GetDataType() == ge::DT_FLOAT ? RGDR_V2_TPL_FP32 : RGDR_V2_TPL_BF16;
    const uint64_t qkL2normKey = tilingData_.useQkL2norm != 0 ? 1 : 0;
    const uint64_t gateInKernelKey = tilingData_.useGateInKernel != 0 ? 1 : 0;
    const uint64_t betaSigmoidKey = tilingData_.useBetaSigmoid != 0 ? 1 : 0;
    const uint64_t allowNegEigvalKey = tilingData_.allowNegEigval != 0 ? 1 : 0;
    tilingKey_ =
        GET_TPL_TILING_KEY(stateKey, gKey, betaKey, qkL2normKey, gateInKernelKey, betaSigmoidKey, allowNegEigvalKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleV2Tiling::GetWorkspaceSize()
{
    workspaceSize_ = 16U * 1024U * 1024U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RecurrentGatedDeltaRuleV2Tiling::PostTiling()
{
    context_->SetBlockDim(tilingData_.vectorCoreNum);
    auto *rawTiling = context_->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetCapacity() < sizeof(tilingData_)) {
        return ge::GRAPH_FAILED;
    }
    errno_t ret = memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), &tilingData_, sizeof(tilingData_));
    if (ret != EOK) {
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(sizeof(tilingData_));
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RecurrentGatedDeltaRuleV2TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus RecurrentGatedDeltaRuleV2TilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecurrentGatedDeltaRuleV2)
    .Tiling(RecurrentGatedDeltaRuleV2TilingFunc)
    .TilingParse<RecurrentGatedDeltaRuleV2CompileInfo>(RecurrentGatedDeltaRuleV2TilingParse);
} // namespace optiling
