#include "chunk_gated_delta_rule_fwd_32_tiling.h"

#include <algorithm>

#include "../op_kernel/arch35/chunk_gated_delta_rule_fwd_32_tiling_data.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kDim = 128;
constexpr uint32_t kChunk = 32;
constexpr uint64_t kDefaultWorkspace = 16ULL * 1024ULL * 1024ULL;

bool Same4(const gert::Shape &shape, int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    return shape.GetDimNum() == 4 && shape.GetDim(0) == d0 && shape.GetDim(1) == d1 && shape.GetDim(2) == d2 &&
           shape.GetDim(3) == d3;
}
}  // namespace

ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *qShape = context->GetInputShape(0);
    const auto *kShape = context->GetInputShape(1);
    const auto *vShape = context->GetInputShape(2);
    const auto *gShape = context->GetInputShape(3);
    const auto *betaShape = context->GetInputShape(4);
    const auto *oShape = context->GetOutputShape(0);
    const auto *stateShape = context->GetOutputShape(1);
    const auto *qDesc = context->GetInputDesc(0);
    const auto *kDesc = context->GetInputDesc(1);
    const auto *vDesc = context->GetInputDesc(2);
    const auto *gDesc = context->GetInputDesc(3);
    const auto *betaDesc = context->GetInputDesc(4);
    if (qShape == nullptr || kShape == nullptr || vShape == nullptr || gShape == nullptr || betaShape == nullptr ||
        oShape == nullptr || stateShape == nullptr || qDesc == nullptr || kDesc == nullptr || vDesc == nullptr ||
        gDesc == nullptr ||
        betaDesc == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &q = qShape->GetStorageShape();
    const gert::Shape &k = kShape->GetStorageShape();
    const gert::Shape &v = vShape->GetStorageShape();
    const gert::Shape &g = gShape->GetStorageShape();
    const gert::Shape &beta = betaShape->GetStorageShape();
    if (q.GetDimNum() != 4 || v.GetDimNum() != 4 || g.GetDimNum() != 3 || beta.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t b = q.GetDim(0);
    const int64_t hk = q.GetDim(1);
    const int64_t t = q.GetDim(2);
    const int64_t key = q.GetDim(3);
    const int64_t hv = v.GetDim(1);
    const int64_t value = v.GetDim(3);
    if (b <= 0 || hk <= 0 || hv != hk || t <= 0 || (t % kChunk) != 0 || key != kDim || value != kDim) {
        return ge::GRAPH_FAILED;
    }
    if (!Same4(k, b, hk, t, key) || !Same4(v, b, hv, t, value) || g.GetDim(0) != b || g.GetDim(1) != hv ||
        g.GetDim(2) != t || beta.GetDim(0) != b || beta.GetDim(1) != hv || beta.GetDim(2) != t) {
        return ge::GRAPH_FAILED;
    }
    if (!Same4(oShape->GetStorageShape(), b, hv, t, value) ||
        !Same4(stateShape->GetStorageShape(), b, hv, key, value)) {
        return ge::GRAPH_FAILED;
    }
    if (qDesc->GetDataType() != ge::DT_BF16 || kDesc->GetDataType() != ge::DT_BF16 ||
        vDesc->GetDataType() != ge::DT_BF16 || gDesc->GetDataType() != ge::DT_FLOAT ||
        betaDesc->GetDataType() != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    const auto *scaleAttr = context->GetAttrs()->GetAttrPointer<double>(0);
    if (scaleAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t hasInitial = 0;
    const auto *initialDesc = context->GetOptionalInputDesc(5);
    const auto *initialShape = context->GetOptionalInputShape(5);
    if (initialDesc != nullptr && initialShape != nullptr) {
        if (initialDesc->GetDataType() != ge::DT_BF16 ||
            !Same4(initialShape->GetStorageShape(), b, hv, key, value)) {
            return ge::GRAPH_FAILED;
        }
        hasInitial = 1;
    }

    auto *tiling = context->GetTilingData<ChunkGatedDeltaRuleFwd32TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->B = static_cast<uint32_t>(b);
    tiling->HV = static_cast<uint32_t>(hv);
    tiling->T = static_cast<uint32_t>(t);
    tiling->hasInitial = hasInitial;
    tiling->scale = static_cast<float>(*scaleAttr);
    uint32_t aicNum = 1;
    uint64_t workspaceBytes = kDefaultWorkspace;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        workspaceBytes = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
        aicNum = std::max<uint32_t>(static_cast<uint32_t>(platform.GetCoreNumAic()), 1U);
    }
    const uint32_t heads = static_cast<uint32_t>(b * hv);
    const uint32_t blockDim = std::min(heads, aicNum);
    tiling->coreNum = blockDim;
    context->SetTilingKey(0);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = workspaceBytes;
    tiling->workspaceOffset = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct ChunkGatedDeltaRuleFwd32CompileInfo {};

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwd32)
    .Tiling(TilingFunc)
    .TilingParse<ChunkGatedDeltaRuleFwd32CompileInfo>(TilingPrepare);
}  // namespace optiling
