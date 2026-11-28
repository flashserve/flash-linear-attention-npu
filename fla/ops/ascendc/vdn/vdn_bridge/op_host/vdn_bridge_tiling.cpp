// SPDX-License-Identifier: BSD-3-Clause
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "vdn_bridge_tiling_processor.h"
#include "../op_kernel/vdn_bridge_tiling_key.h"

namespace optiling {
namespace {
ge::graphStatus Tiling(gert::TilingContext *context)
{
    using namespace VdnBridgeHost;
    using namespace VdnBridgeKernel;
    const auto *prefix = context->GetInputShape(0);
    const auto *suffix = context->GetInputShape(1);
    const auto *bounds = context->GetInputShape(2);
    const auto *logs = context->GetInputShape(3);
    const auto *out = context->GetOutputShape(0);
    const auto *outDesc = context->GetOutputDesc(0);
    const auto *attrs = context->GetAttrs();
    if (prefix == nullptr || suffix == nullptr || bounds == nullptr || logs == nullptr ||
        out == nullptr || outDesc == nullptr || attrs == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *start = attrs->GetAttrPointer<int64_t>(0);
    const auto *count = attrs->GetAttrPointer<int64_t>(1);
    const auto *bf16 = attrs->GetAttrPointer<bool>(2);
    if (start == nullptr || count == nullptr || bf16 == nullptr || prefix->GetOriginShape().GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const auto &bank = prefix->GetOriginShape();
    const int64_t bankRows = bank.GetDim(0);
    const int64_t heads = bank.GetDim(1);
    const auto dtype = outDesc->GetDataType();
    if (!ValidTile(bankRows, heads, *start, *count) || !HasShape(bank, {bankRows, heads, HEAD_DIM, HEAD_DIM}) ||
        !HasShape(suffix->GetOriginShape(), {bankRows, heads, HEAD_DIM, HEAD_DIM}) ||
        !HasShape(bounds->GetOriginShape(), {bankRows - 1, 2}) ||
        !HasShape(logs->GetOriginShape(), {bankRows, heads, HEAD_DIM}) ||
        !HasShape(out->GetOriginShape(), {*count, heads, HEAD_DIM, HEAD_DIM}) ||
        dtype != (*bf16 ? ge::DT_BF16 : ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<VdnBridgeTilingData>();
    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    // The 910B device binary reads the struct plus one CANN runtime word (opParaSize=40).
    constexpr size_t tilingBytes = sizeof(VdnBridgeTilingData) + sizeof(uint64_t);
    if (data == nullptr || workspace == nullptr || raw == nullptr || raw->GetCapacity() < tilingBytes) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto cores = BlockDim(heads, *count, platform.GetCoreNumAiv());
    if (cores == 0) {
        return ge::GRAPH_FAILED;
    }
    *data = {bankRows - 1, heads, *start, *count};
    context->SetBlockDim(cores);
    context->SetTilingKey(GET_TPL_TILING_KEY(*bf16));
    std::fill_n(static_cast<uint8_t *>(raw->GetData()) + sizeof(VdnBridgeTilingData), sizeof(uint64_t), 0);
    raw->SetDataSize(tilingBytes);
    // This vector kernel only accesses its explicit GM tensors and UB.
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
ge::graphStatus Parse(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_OPTILING(VdnBridge).Tiling(Tiling).TilingParse<CompileInfo>(Parse);
} // namespace optiling
