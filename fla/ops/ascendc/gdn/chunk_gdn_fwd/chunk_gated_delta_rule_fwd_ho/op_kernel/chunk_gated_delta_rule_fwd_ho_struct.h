/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GATED_DELTA_RULE_FWD_HO_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_FWD_HO_STRUCT_H

#include <cstdint>

namespace GDN {

struct ChunkGatedDeltaRuleFwdHOTrailer {
    int64_t hIntermediateOffset;
    int64_t vNewIntermediateOffset;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_FWD_HO_STRUCT_H
