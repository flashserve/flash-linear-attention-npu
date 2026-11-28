// SPDX-License-Identifier: BSD-3-Clause
#include "vdn_add_diagonal_tiling_processor.h"

#include "../op_kernel/vdn_add_diagonal_struct.h"
#include "../op_kernel/vdn_add_diagonal_tiling_key.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
using namespace VdnAddDiagonalHost;

ge::graphStatus Tile(gert::TilingContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *data = context->GetTilingData<VdnAddDiagonalTilingData>();
    auto *workspace = context->GetWorkspaceSizes(1);
    auto *raw = context->GetRawTilingData();
    // The 910B binary's opParaSize includes one runtime word after the tiling struct.
    constexpr size_t tilingBytes = sizeof(VdnAddDiagonalTilingData) + sizeof(uint64_t);
    if (input == nullptr || data == nullptr || workspace == nullptr || context->GetPlatformInfo() == nullptr ||
        raw == nullptr || raw->GetCapacity() < tilingBytes) {
        return ge::GRAPH_FAILED;
    }
    const auto &shape = input->GetOriginShape();
    if (shape.GetDimNum() != 3 || shape.GetDim(1) != kHeadDim || shape.GetDim(2) != kHeadDim ||
        !ValidBatch(shape.GetDim(0))) {
        return ge::GRAPH_FAILED;
    }
    data->count = shape.GetDim(0);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t blocks = BlockDim(data->count, platform.GetCoreNumAiv());
    if (blocks == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(blocks);
    context->SetTilingKey(GET_TPL_TILING_KEY(VDN_ADD_DIAGONAL_TPL_SECTOR));
    std::fill_n(static_cast<uint8_t *>(raw->GetData()) + sizeof(VdnAddDiagonalTilingData), sizeof(uint64_t), 0);
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

IMPL_OP_OPTILING(VdnAddDiagonal).Tiling(Tile).TilingParse<CompileInfo>(Parse);
} // namespace optiling
