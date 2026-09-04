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
constexpr uint64_t KKT_SOLVE_RELEASE_FLAG = 0;

// Event generations in the fused suffix are deliberately disjoint in time:
// P uses AIV->AIC flag3 and AIC->both-AIV flag0, Solve uses flags1/2/3,
// Q uses flag5, WU drains flags3/4/5, and only then may H reuse flags0..7.
// Target CANN reserves flags6/7, so coefficient generation must not use them.
// For mode2, the two AIV notifications are aggregated by FFTS and the paired
// AIC performs one wait, matching the official cross-core set/wait example.

template <typename T, int MATRIX_SIZE, GDN::GdnCoreSyncVariant kSyncVariant,
          typename TilingData>
__aicore__ inline void RunSolvePhase(GM_ADDR a, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                     GM_ADDR out, GM_ADDR workspace,
                                     const TilingData *tilingData)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using SyncTraits = GDN::GdnCoreSyncVariantTraits<kSyncVariant>;
    constexpr bool kUseKktToSolveGroupHandoff =
        MATRIX_SIZE == 64 && SyncTraits::kKktToSolveGroupHandoff;
    if (tilingData->isVarlen != 0) {
        if constexpr (kUseKktToSolveGroupHandoff) {
            // Both AIV subblocks publish their MTE3 completion. Mode2 folds
            // those two notifications into one AIC wait; a second wait would
            // consume a nonexistent generation and can deadlock.
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag(KKT_READY_FLAG);
                CrossCoreSetFlag<0x2, PIPE_FIX>(KKT_SOLVE_RELEASE_FLAG);
            }
            // Every AIV, including an idle subblock, participates so flag0 is
            // fully consumed before Solve starts and can later be reused by H.
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(KKT_READY_FLAG);
                CrossCoreWaitFlag(KKT_SOLVE_RELEASE_FLAG);
            }
        } else {
            // KKT uses head-major ownership while legacy varlen SolveTri
            // remaps the same tiles chunk-major. Wait for every writer unless
            // the head-major paired protocol is selected explicitly.
            AscendC::SyncAll<false>();
        }
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
        SolveTri64<T, T, SyncTraits::kUseImmediateMte2Mte1,
                   SyncTraits::kDeferMte2Mte1Wait,
                   SyncTraits::kHeadMajorSolve64Ownership> solve;
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
