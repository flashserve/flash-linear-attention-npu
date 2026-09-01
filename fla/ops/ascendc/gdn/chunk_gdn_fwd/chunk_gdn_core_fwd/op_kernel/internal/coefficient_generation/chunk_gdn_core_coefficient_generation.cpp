#include "chunk_gdn_core_kkt_cube.h"
#include "chunk_gdn_core_cumsum_kkt.h"
// This translation unit uses the private PR340 SolveTri copy below. The public
// solve_tri operator remains independently registered and keeps the same
// high-precision implementation.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "gdn_core_solve_tri/arch35/solve_tri_ascend950.h"
#else
#include "gdn_core_solve_tri/solve_tri_cube.h"
#include "gdn_core_solve_tri/solve_tri_vector.h"
#endif

using namespace AscendC;

namespace {
constexpr uint64_t KKT_READY_FLAG = 3;

template <typename T, int MATRIX_SIZE, typename TilingData>
__aicore__ inline void RunSolvePhase(GM_ADDR a, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                     GM_ADDR out, GM_ADDR workspace,
                                     const TilingData *tilingData)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if (tilingData->isVarlen != 0) {
        // KKT uses head-major ownership while varlen SolveTri remaps the same
        // tiles chunk-major. A paired event can therefore miss a producer on
        // another core; wait for every KKT writer before SolveTri starts.
        AscendC::SyncAll<false>();
    } else {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(KKT_READY_FLAG);
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG);
        }
    }
    // Phase6 passes a per-core user-workspace slice and its KKT epilogue uses
    // contiguous tile ownership.  Keep those policies explicit instead of
    // silently inheriting the standalone round-robin/default-workspace path.
    if constexpr (MATRIX_SIZE == 64) {
        SolveTri64<T, T> solve;
        solve.Init(a, cuSeqlens, chunkIndices, out, workspace, tilingData, true, true);
        solve.Process();
    } else {
        SolveTri128<T, T> solve;
        solve.Init(a, cuSeqlens, chunkIndices, out, workspace, tilingData, true, true);
        solve.Process();
    }
#else
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(KKT_READY_FLAG);
        NsSolveTri::SolveTriCube<MATRIX_SIZE, T> solve;
        solve.Init(a, cuSeqlens, chunkIndices, out, workspace, tilingData, true);
        solve.Process(false);
    }
    if ASCEND_IS_AIV {
        if (GetSubBlockIdx() == 0) {
            NsSolveTri::SolveTriVector<MATRIX_SIZE, T> constants;
            constants.Init(workspace, tilingData->totalTiles, tilingData->matrixSize);
            constants.Process(false, true);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG);
    }
#endif
}
}  // namespace
