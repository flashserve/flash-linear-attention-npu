/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_tiling.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_templates_registry.h"
#include <register/op_impl_registry.h>

namespace optiling {

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdArch22(gert::TilingContext *context);
ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdArch35(gert::TilingContext *context);

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwd(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("ChunkGatedDeltaRuleFwd", "Invalid tiling context."),
                return ge::GRAPH_FAILED);
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        return Tiling4ChunkGatedDeltaRuleFwdArch35(context);
    }
    return Tiling4ChunkGatedDeltaRuleFwdArch22(context);
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleFwd(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwd)
    .Tiling(Tiling4ChunkGatedDeltaRuleFwd)
    .TilingParse<ChunkGatedDeltaRuleFwdCompileInfo>(TilingPrepareForChunkGatedDeltaRuleFwd);

} // namespace optiling
