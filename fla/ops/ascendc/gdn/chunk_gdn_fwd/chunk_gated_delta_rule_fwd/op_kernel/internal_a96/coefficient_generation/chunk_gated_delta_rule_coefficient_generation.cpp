#include "chunk_gated_delta_rule_kkt_cube.h"
#include "chunk_gated_delta_rule_cumsum_kkt.h"

#include "gated_delta_rule_solve_tri/arch35/solve_tri_ascend950.h"

using namespace AscendC;

namespace {
constexpr uint64_t KKT_READY_FLAG = 3;
constexpr uint64_t KKT_SOLVE_RELEASE_FLAG = 0;

template <typename T, int MATRIX_SIZE, GDN::Arch35GdnSyncVariant Variant, typename TilingData>
__aicore__ inline void RunSolvePhase(GM_ADDR a, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                     GM_ADDR out, GM_ADDR workspace,
                                     const TilingData *tilingData)
{
    using Sync = GDN::Arch35GdnSyncTraits<Variant>;
    if constexpr (MATRIX_SIZE == 64 && Sync::kKktToSolveGroup) {

        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<0x4, PIPE_FIX>(KKT_READY_FLAG);
            CrossCoreSetFlag<0x4, PIPE_FIX>(KKT_SOLVE_RELEASE_FLAG);
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<0x4, PIPE_MTE3>(KKT_READY_FLAG);
            CrossCoreWaitFlag(KKT_SOLVE_RELEASE_FLAG);
        }
    } else {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<0x4, PIPE_FIX>(KKT_READY_FLAG);
        }
        if ASCEND_IS_AIV {

            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            CrossCoreSetFlag<0x4, PIPE_MTE3>(KKT_READY_FLAG);
        }
    }

    if constexpr (MATRIX_SIZE == 64) {
        SolveTri64<T, T, Sync::kHeadMajorSolve64> solve;
        solve.Init(a, cuSeqlens, chunkIndices, out, workspace, tilingData, true, true);
        solve.Process();
    } else {
        SolveTri128<T, T> solve;
        solve.Init(a, cuSeqlens, chunkIndices, out, workspace, tilingData, true, true);
        solve.Process();
    }
}
}
