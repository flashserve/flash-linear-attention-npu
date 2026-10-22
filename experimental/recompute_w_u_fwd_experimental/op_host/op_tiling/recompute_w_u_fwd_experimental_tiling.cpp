#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "../../op_kernel/recompute_w_u_fwd_experimental_tiling.h"

namespace optiling {
struct RecomputeWUFwdExperimentalCompileInfo {};

ge::graphStatus Tiling4RecomputeWUFwdExperimental(gert::TilingContext *context)
{
    auto *data = context->GetTilingData<Experimental::RecomputeWUFwdExperimentalTilingData>();
    if (data == nullptr) {
        return ge::GRAPH_FAILED;
    }
    data->elements = 8;
    context->SetBlockDim(1);
    context->SetTilingKey(1);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseRecomputeWUFwdExperimental(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RecomputeWUFwdExperimental)
    .Tiling(Tiling4RecomputeWUFwdExperimental)
    .TilingParse<RecomputeWUFwdExperimentalCompileInfo>(ParseRecomputeWUFwdExperimental);
} // namespace optiling
