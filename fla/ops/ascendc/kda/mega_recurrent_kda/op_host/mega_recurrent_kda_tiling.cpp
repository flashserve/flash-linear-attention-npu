#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "err/ops_err.h"
#include "kda_metadata.h"
#include "kda_workspace.h"
#include "../op_kernel/arch22/tiling_data.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace {
using kda::Input;
namespace geometry = kda::geometry;
using Dims = std::vector<int64_t>;
constexpr int64_t kHiddenWidth = 7168;
constexpr int64_t kHeadWidth = 128;
Dims GetDimensions(gert::TilingContext *ctx, Input input)
{
    const auto index = static_cast<size_t>(input);
    auto tensor = input == Input::CuSeqlens ? ctx->GetOptionalInputShape(index) : ctx->GetInputShape(index);
    if (!tensor) {
        return {};
    }
    const auto &origin = tensor->GetOriginShape();
    const auto &s = origin.GetDimNum() ? origin : tensor->GetStorageShape();
    Dims dims(s.GetDimNum());
    for (size_t i = 0; i < dims.size(); ++i) {
        dims[i] = s.GetDim(i);
        if (dims[i] <= 0) {
            throw std::invalid_argument("expected positive input dimensions");
        }
    }
    return dims;
}
uint32_t CheckedDimension(int64_t value)
{
    if (value <= 0 || uint64_t(value) > UINT32_MAX) {
        throw std::invalid_argument("unsupported index span");
    }
    return static_cast<uint32_t>(value);
}
void CheckShape(const Dims &actual, const Dims &required, const char *name)
{
    if (actual != required) {
        throw std::invalid_argument(name);
    }
}
void CheckProjectionShapes(gert::TilingContext *ctx, int64_t batch, int64_t n)
{
    const int64_t width = n * kHeadWidth;
    CheckShape(GetDimensions(ctx, Input::AcceptedTokens), {batch}, "accepted-token shape");
    CheckShape(GetDimensions(ctx, Input::QkvWeight), {3 * width, kHiddenWidth}, "QKV projection shape");
    CheckShape(GetDimensions(ctx, Input::DecayAWeight), {kHeadWidth, kHiddenWidth}, "decay-A shape");
    CheckShape(GetDimensions(ctx, Input::DecayBWeight), {width, kHeadWidth}, "decay-B shape");
    CheckShape(GetDimensions(ctx, Input::BetaWeight), {n, kHiddenWidth}, "beta shape");
    CheckShape(GetDimensions(ctx, Input::GateWeight), {width, kHiddenWidth}, "gate shape");
    CheckShape(GetDimensions(ctx, Input::NormWeight), {kHeadWidth}, "norm weight shape");
    CheckShape(GetDimensions(ctx, Input::OutputWeight), {kHiddenWidth, width}, "output weight shape");
    CheckShape(GetDimensions(ctx, Input::ALog), {n}, "A_log shape");
    CheckShape(GetDimensions(ctx, Input::DtBias), {n, kHeadWidth}, "dt_bias shape");
}
uint32_t GetConvLayout(gert::TilingContext *ctx, int64_t n)
{
    const int64_t width = n * kHeadWidth;
    const Dims weights = GetDimensions(ctx, Input::ConvWeight);
    if (weights == Dims{4, 3 * width}) {
        return 0;
    }
    if (weights == Dims{3, n, 4, kHeadWidth}) {
        return 1;
    }
    CheckShape(weights, {3, 4, kHeadWidth}, "conv1d weight shape");
    return 2;
}
MegaRecurrentKdaTilingData ReadCommonTiling(gert::TilingContext *ctx, const int64_t *geometryData)
{
    const auto *attrs = ctx->GetAttrs();
    Dims x = GetDimensions(ctx, Input::HiddenStates);
    if ((x.size() != 2 && x.size() != 3) || x.back() != kHiddenWidth) {
        throw std::invalid_argument("hidden_states must be BSH or TH with H=7168");
    }
    Dims batchDim = GetDimensions(ctx, Input::StateIndices);
    if (batchDim.size() != 1) {
        throw std::invalid_argument("state_indices must be a vector");
    }
    const int64_t batch = batchDim[0], n = geometryData[geometry::Heads],
                  capacity = geometryData[geometry::ReplayCapacity];
    if (n > UINT32_MAX / (6 * kHeadWidth)) {
        throw std::invalid_argument("head count exceeds device stride range");
    }
    CheckProjectionShapes(ctx, batch, n);
    uint32_t layout = GetConvLayout(ctx, n);
    const bool packed = x.size() == 2;
    uint32_t seq = packed ? 0 : CheckedDimension(x[1]);
    if (!packed && x[0] != batch) {
        throw std::invalid_argument("BSH batch mismatch");
    }
    if (packed) {
        CheckShape(GetDimensions(ctx, Input::CuSeqlens), {batch + 1}, "cu_seqlens shape");
    }
    uint32_t rows = CheckedDimension(packed ? x[0] : kda::checkedProduct(batch, seq));
    if (geometryData[geometry::WorkspaceBytes] <= 0 ||
        (geometryData[geometry::CrossesTile] != 0 && geometryData[geometry::CrossesTile] != 1)) {
        throw std::invalid_argument("invalid workspace plan");
    }
    auto scale = attrs ? attrs->GetAttrPointer<float>(0) : nullptr;
    auto lower = attrs ? attrs->GetAttrPointer<float>(1) : nullptr;
    auto eps = attrs ? attrs->GetAttrPointer<float>(2) : nullptr;
    if (!scale || !lower || !eps || !std::isfinite(*scale) || !std::isfinite(*lower) || *lower < -5 || *lower > 0 ||
        !std::isfinite(*eps) || *eps < std::numeric_limits<float>::min()) {
        throw std::invalid_argument("invalid scale, lower_bound or rms_norm_eps");
    }
    MegaRecurrentKdaTilingData common{};
    common.convStride = uint64_t(geometryData[geometry::ConvStride]);
    common.checkpointStride = uint64_t(geometryData[geometry::CheckpointStride]);
    common.uStride = uint64_t(geometryData[geometry::UStride]);
    common.keyStride = uint64_t(geometryData[geometry::KStride]);
    common.decayStride = uint64_t(geometryData[geometry::DecayStride]);
    common.batch = CheckedDimension(batch);
    common.sequence = seq;
    common.tokens = rows;
    common.heads = CheckedDimension(n);
    common.capacity = CheckedDimension(capacity);
    common.prefix64 = packed;
    common.convLayout = layout;
    common.crosses = bool(geometryData[geometry::CrossesTile]);
    common.scale = *scale;
    common.lowerBound = *lower;
    common.rmsNormEps = *eps;
    return common;
}

template <typename TilingData>
ge::graphStatus PublishTiling(gert::TilingContext *ctx, const TilingData &data, uint64_t workspaceBytes,
                              uint64_t systemBytes, uint32_t cores, uint64_t key)
{
    auto *raw = ctx->GetRawTilingData();
    auto *sizes = ctx->GetWorkspaceSizes(1);
    if (!raw || !sizes || raw->GetCapacity() < sizeof(data) || workspaceBytes > UINT64_MAX - systemBytes) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    sizes[0] = workspaceBytes + systemBytes;
    ctx->SetBlockDim(cores);
    ctx->SetTilingKey(key);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetDeviceTiling(gert::TilingContext *ctx, const MegaRecurrentKdaTilingData &common,
                                const int64_t *geometryData)
{
    auto platformInfo = ctx->GetPlatformInfo();
    if (!platformInfo) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t cores = platform.GetCoreNumAic();
    if (!cores) {
        return ge::GRAPH_FAILED;
    }
    const auto soc = platform.GetSocVersion();
    const auto systemBytes = platform.GetLibApiWorkSpaceSize();
    if (soc == platform_ascendc::SocVersion::ASCEND950) {
        return PublishTiling(ctx, common, geometryData[geometry::WorkspaceBytes], systemBytes, cores,
                             common.crosses ? 2 : 1);
    }
    if (soc != platform_ascendc::SocVersion::ASCEND910B && soc != platform_ascendc::SocVersion::ASCEND910_93) {
        return ge::GRAPH_FAILED;
    }

    using namespace MegaRecurrentKdaV2;
    MegaRecurrentKdaArch22TilingData data{};
    data.common = common;
    data.poolSlots = CheckedDimension(geometryData[geometry::PoolSlots]);
    data.convLength = CheckedDimension(geometryData[geometry::ConvLength]);
    // Arch22 stores workspace offsets in uint32_t and uses a 16-bit panel stride.
    if (common.tokens > INT32_MAX || common.heads > 65535 / 24 ||
        (common.crosses &&
         uint64_t(common.batch) * common.heads * kHeadWidth * kHeadWidth > UINT32_MAX / sizeof(float)) ||
        uint64_t(common.tokens) * common.heads * 1536 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    KdaProblem shape{common.batch,    common.sequence, common.heads,  data.poolSlots,
                     data.convLength, common.capacity, common.tokens, kTileRows};
    uint64_t l2Bytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
    data.arch22 = ComputeTiling(shape, cores, 2 * cores, common.crosses, l2Bytes);
    if (data.arch22.overflow) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t prefixBytes = (uint64_t(common.batch) + 1) * sizeof(int32_t);
    return PublishTiling(ctx, data, data.arch22.workspaceBytes + prefixBytes, systemBytes, cores, 0);
}

ge::graphStatus TileMegaRecurrentKda(gert::TilingContext *ctx)
{
    if (ctx == nullptr) {
        return ge::GRAPH_FAILED;
    }
    try {
        auto attrs = ctx->GetAttrs();
        auto metadata = attrs ? attrs->GetListInt(3) : nullptr;
        if (!metadata || metadata->GetSize() != geometry::FieldCount) {
            throw std::invalid_argument("missing logical state geometry");
        }
        const int64_t *geometryData = metadata->GetData();
        const auto common = ReadCommonTiling(ctx, geometryData);
        return SetDeviceTiling(ctx, common, geometryData);
    } catch (const std::exception &e) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ctx->GetNodeName(), "MegaRecurrentKda", e.what());
        return ge::GRAPH_FAILED;
    }
}

struct CompileInfo {};
ge::graphStatus PrepareTiling(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_OPTILING(MegaRecurrentKda).Tiling(TileMegaRecurrentKda).TilingParse<CompileInfo>(PrepareTiling);
