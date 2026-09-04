/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "w_mmad_demo_tiling.h"

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus Tiling4WMmadDemo(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    WMmadDemoTilingData tiling;
    tiling.set_dummy(3);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(0);
    context->SetBlockDim(1);

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = sysWorkspaceSize + 128 * 1024;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse4WMmadDemo(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct WMmadDemoCompileInfo {};

IMPL_OP_OPTILING(WMmadDemo)
    .Tiling(Tiling4WMmadDemo)
    .TilingParse<WMmadDemoCompileInfo>(TilingParse4WMmadDemo);

} // namespace optiling
