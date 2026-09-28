#ifndef CHUNK_KDA_BWD_FINALIZE_TILING_H
#define CHUNK_KDA_BWD_FINALIZE_TILING_H

#include <exe_graph/runtime/tiling_context.h>

#include "chunk_kda_bwd_finalize_tiling_processor.h"

namespace optiling {

struct ChunkKdaBwdFinalizeCompileInfo {};

enum ChunkKdaBwdFinalizeInput : size_t {
    INPUT_Q = 0, INPUT_K, INPUT_V, INPUT_GK, INPUT_RAW_G, INPUT_BETA,
    INPUT_A_LOG, INPUT_DT_BIAS, INPUT_AKK, INPUT_V_NEW, INPUT_H, INPUT_DH,
    INPUT_DV_SCAN, INPUT_D_AQK, INPUT_DQ_RAW, INPUT_Q_RSTD, INPUT_K_RSTD,
    INPUT_CU_SEQLENS, INPUT_CHUNK_INDICES
};

enum ChunkKdaBwdFinalizeAttr : size_t {
    ATTR_SCALE = 0, ATTR_LOWER_BOUND, ATTR_CHUNK_SIZE, ATTR_SAFE_GATE,
    ATTR_USE_GATE_IN_KERNEL, ATTR_USE_EXP2, ATTR_STATE_V_FIRST
};

} // namespace optiling

#endif // CHUNK_KDA_BWD_FINALIZE_TILING_H
