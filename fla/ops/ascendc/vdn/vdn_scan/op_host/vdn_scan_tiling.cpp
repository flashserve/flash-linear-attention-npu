// SPDX-License-Identifier: BSD-3-Clause
#include "vdn_scan_tiling_processor.h"

#include "../op_kernel/vdn_scan_struct.h"
#include "../op_kernel/vdn_scan_tiling_key.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
using namespace VdnScanHost;

ge::graphStatus Tile(gert::TilingContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *data = context->GetTilingData<VdnScanTilingData>();
    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    // The 910B binary's opParaSize includes one runtime word after the tiling struct.
    constexpr size_t tilingBytes = sizeof(VdnScanTilingData) + sizeof(uint64_t);
    if (input == nullptr || data == nullptr || workspace == nullptr || context->GetPlatformInfo() == nullptr ||
        raw == nullptr || raw->GetCapacity() < tilingBytes) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = input->GetOriginShape();
    if (shape.GetDimNum() != 4 || shape.GetDim(2) != kHeadDim || shape.GetDim(3) != kHeadDim ||
        !ValidDimensions(shape.GetDim(0), shape.GetDim(1))) {
        return ge::GRAPH_FAILED;
    }
    data->frames = static_cast<int32_t>(shape.GetDim(0));
    data->heads = static_cast<int32_t>(shape.GetDim(1));
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blocks = BlockDim(data->heads, platform.GetCoreNumAic());
    if (blocks == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(blocks);
    context->SetTilingKey(GET_TPL_TILING_KEY(VDN_SCAN_TPL_DIM));
    std::fill_n(static_cast<uint8_t *>(raw->GetData()) + sizeof(VdnScanTilingData), sizeof(uint64_t), 0);
    raw->SetDataSize(tilingBytes);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

struct CompileInfo {};
ge::graphStatus Parse(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(VdnScan).Tiling(Tile).TilingParse<CompileInfo>(Parse);
} // namespace optiling
