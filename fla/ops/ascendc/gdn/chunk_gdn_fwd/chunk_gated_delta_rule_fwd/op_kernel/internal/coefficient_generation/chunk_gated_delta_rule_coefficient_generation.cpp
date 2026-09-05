#include "chunk_gated_delta_rule_kkt_cube.h"
#include "chunk_gated_delta_rule_cumsum_kkt.h"
// This translation unit uses the private PR340 SolveTri copy below. The public
// solve_tri operator remains independently registered and keeps the same
// high-precision implementation.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "gated_delta_rule_solve_tri/arch35/solve_tri_ascend950.h"
#else
#include "gated_delta_rule_solve_tri/solve_tri_cube.h"
#include "gated_delta_rule_solve_tri/solve_tri_vector.h"
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
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(KKT_READY_FLAG);
    }
    if ASCEND_IS_AIV {
        // A short task range can leave an entire coefficient tile on an AIV
        // group whose writeback finishes after an unrelated group has already
        // released its AIC.  Publish the complete AIV generation before any
        // AIC enters SolveTri; a paired-subblock barrier is not sufficient for
        // this cross-group dependency.
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG);
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
