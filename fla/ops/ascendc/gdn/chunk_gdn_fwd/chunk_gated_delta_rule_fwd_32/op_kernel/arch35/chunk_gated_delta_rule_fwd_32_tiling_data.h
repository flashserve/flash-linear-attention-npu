#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_TILING_DATA_H

#include <cstdint>

struct ChunkGatedDeltaRuleFwd32TilingData {
    uint32_t B;
    uint32_t HV;
    uint32_t T;
    uint32_t hasInitial;
    float scale;
    uint32_t workspaceOffset;
    uint32_t coreNum;
};

#endif
