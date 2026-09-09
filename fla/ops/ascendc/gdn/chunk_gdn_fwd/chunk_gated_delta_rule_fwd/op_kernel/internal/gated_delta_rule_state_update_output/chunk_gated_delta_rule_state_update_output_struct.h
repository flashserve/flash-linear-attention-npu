/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GATED_DELTA_RULE_STATE_UPDATE_OUTPUT_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_STATE_UPDATE_OUTPUT_STRUCT_H

#include <cstdint>

#include "../operators/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_struct.h"

namespace GDN {

struct ChunkGatedDeltaRuleStateOutputTrailer {
    RecomputeWUFwdTilingData recompute;
    int64_t recomputeWorkspaceOffset;
    int64_t wIntermediateOffset;
    int64_t uIntermediateOffset;
    int64_t hIntermediateOffset;
    int64_t vNewIntermediateOffset;
    int64_t qDataType;
    int64_t betaDataType;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_STATE_UPDATE_OUTPUT_STRUCT_H
