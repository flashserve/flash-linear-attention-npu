#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>

namespace optiling {

// 与 op_kernel/arch35/chunk_kda_fwd_fused 的 KdaFd32TilingData 一致。
BEGIN_TILING_DATA_DEF(ChunkKdaFwdFusedTilingData)
TILING_DATA_FIELD_DEF(uint64_t, nChunks);
TILING_DATA_FIELD_DEF(uint64_t, scaleBf16);
TILING_DATA_FIELD_DEF(uint64_t, gateScaleBits);
TILING_DATA_FIELD_DEF(uint64_t, headNum);
TILING_DATA_FIELD_DEF(uint64_t, seqLen);
TILING_DATA_FIELD_DEF(uint64_t, hasInitialState);
TILING_DATA_FIELD_DEF(uint64_t, outputFinalState);
TILING_DATA_FIELD_DEF(uint64_t, qkGroup);
TILING_DATA_FIELD_DEF(uint64_t, taskCount);
TILING_DATA_FIELD_DEF(uint64_t, hvNumReal);
TILING_DATA_FIELD_DEF(uint64_t, blockNum);
TILING_DATA_FIELD_DEF(uint64_t, varlenMode);
TILING_DATA_FIELD_DEF(uint64_t, epsilonBits);
TILING_DATA_FIELD_DEF(uint64_t, gateFlags);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 1025, cuData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKdaFwdFused, ChunkKdaFwdFusedTilingData)

struct ChunkKdaFwdFusedCompileInfo {};
} // namespace optiling
