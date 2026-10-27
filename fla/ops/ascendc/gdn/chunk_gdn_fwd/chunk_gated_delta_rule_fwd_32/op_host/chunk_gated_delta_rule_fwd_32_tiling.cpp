#include "chunk_gated_delta_rule_fwd_32_tiling.h"

#include <algorithm>

#include "../op_kernel/arch35/chunk_gated_delta_rule_fwd_32_tiling_data.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kDim = 128;
constexpr uint64_t kDefaultWorkspace = 16ULL * 1024ULL * 1024ULL;

bool Same4(const gert::Shape &shape, int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    return shape.GetDimNum() == 4 && shape.GetDim(0) == d0 && shape.GetDim(1) == d1 && shape.GetDim(2) == d2 &&
           shape.GetDim(3) == d3;
}

// chunk_indices must be the canonical sequence-major (seqId, localChunkId) list.
bool CanonicalChunks(const int64_t *cu, int64_t cuCount, const int64_t *indices, int64_t indexCount, int64_t tokens)
{
    if (cu == nullptr || indices == nullptr || cuCount < 2 || cu[0] != 0 || cu[cuCount - 1] != tokens) {
        return false;
    }
    int64_t pair = 0;
    for (int64_t seq = 0; seq + 1 < cuCount; ++seq) {
        if (cu[seq] > cu[seq + 1]) {
            return false;
        }
        const int64_t length = cu[seq + 1] - cu[seq];
        const int64_t chunks = length == 0 ? 0 : (length + 31) / 32;
        for (int64_t local = 0; local < chunks; ++local) {
            const int64_t pos = pair * 2;
            if (pos + 1 >= indexCount || indices[pos] != seq || indices[pos + 1] != local) {
                return false;
            }
            ++pair;
        }
    }
    return indexCount == pair * 2;
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
    const auto *stateDesc = context->GetOutputDesc(1);
    const auto *qDesc = context->GetInputDesc(0);
    const auto *kDesc = context->GetInputDesc(1);
    const auto *vDesc = context->GetInputDesc(2);
    const auto *gDesc = context->GetInputDesc(3);
    const auto *betaDesc = context->GetInputDesc(4);
    if (qShape == nullptr || kShape == nullptr || vShape == nullptr || gShape == nullptr || betaShape == nullptr ||
        oShape == nullptr || stateShape == nullptr || stateDesc == nullptr || qDesc == nullptr || kDesc == nullptr ||
        vDesc == nullptr || gDesc == nullptr ||
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
    if (b <= 0 || hk <= 0 || hv <= 0 || (hv % hk) != 0 || t <= 0 || key != kDim || value != kDim) {
        return ge::GRAPH_FAILED;
    }
    if (!Same4(k, b, hk, t, key) || !Same4(v, b, hv, t, value) || g.GetDim(0) != b || g.GetDim(1) != hv ||
        g.GetDim(2) != t || beta.GetDim(0) != b || beta.GetDim(1) != hv || beta.GetDim(2) != t) {
        return ge::GRAPH_FAILED;
    }
    const auto *cuTensor = context->GetOptionalInputTensor(6);
    const auto *chunkTensor = context->GetOptionalInputTensor(7);
    if ((cuTensor == nullptr) != (chunkTensor == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t seqNum = 0;
    int64_t stateBatch = b;
    if (cuTensor != nullptr) {
        const int64_t *cu = cuTensor->GetData<int64_t>();
        const int64_t *indices = chunkTensor->GetData<int64_t>();
        const auto &cuShape = cuTensor->GetStorageShape();
        const auto &chunkShape = chunkTensor->GetStorageShape();
        if (b != 1 || cu == nullptr || indices == nullptr || cuShape.GetDimNum() != 1 ||
            chunkShape.GetDimNum() != 1) {
            return ge::GRAPH_FAILED;
        }
        const int64_t cuCount = cuShape.GetDim(0);
        const int64_t indexCount = chunkShape.GetDim(0);
        if (!CanonicalChunks(cu, cuCount, indices, indexCount, t)) {
            return ge::GRAPH_FAILED;
        }
        seqNum = static_cast<uint32_t>(cuCount - 1);
        stateBatch = cuCount - 1;
    }
    if (!Same4(oShape->GetStorageShape(), b, hv, t, value) ||
        !Same4(stateShape->GetStorageShape(), stateBatch, hv, key, value)) {
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
    uint32_t initialFp32 = 0;
    const auto *initialDesc = context->GetOptionalInputDesc(5);
    const auto *initialShape = context->GetOptionalInputShape(5);
    if (initialDesc != nullptr && initialShape != nullptr) {
        const auto dtype = initialDesc->GetDataType();
        if ((dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT) ||
            !Same4(initialShape->GetStorageShape(), stateBatch, hv, key, value)) {
            return ge::GRAPH_FAILED;
        }
        hasInitial = 1;
        initialFp32 = dtype == ge::DT_FLOAT ? 1U : 0U;
    }
    const auto finalDtype = stateDesc->GetDataType();
    uint32_t finalFp32 = 0;
    if (hasInitial == 0 || initialFp32 != 0) {
        if (finalDtype != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }
        finalFp32 = 1;
    } else if (finalDtype != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<ChunkGatedDeltaRuleFwd32TilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->B = static_cast<uint32_t>(b);
    tiling->HV = static_cast<uint32_t>(hv);
    tiling->HK = static_cast<uint32_t>(hk);
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
    tiling->seqNum = seqNum;
    tiling->initialFp32 = initialFp32;
    tiling->finalFp32 = finalFp32;
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
