#ifndef KDA_BWD_FINALIZE_ARCH22_TILING_H
#define KDA_BWD_FINALIZE_ARCH22_TILING_H
#if __has_include("../../../chunk_kda_bwd/op_kernel/chunk_kda_bwd_common.h")
#include "../../../chunk_kda_bwd/op_kernel/chunk_kda_bwd_common.h"
#else
#include "chunk_kda_bwd/op_kernel/chunk_kda_bwd_common.h"
#endif
namespace KDA {
struct ChunkKdaBwdFinalizeArch22TilingData {
    ChunkKdaBwdCTilingData phaseC;
    uint64_t dqOffset;
    uint64_t dkOffset;
    uint64_t dAkkOffset;
    uint64_t dAqkOffset;
    uint64_t phaseWorkspaceOffset;
    uint32_t fp32Output;
    uint32_t safeGate;
};
}
#endif
