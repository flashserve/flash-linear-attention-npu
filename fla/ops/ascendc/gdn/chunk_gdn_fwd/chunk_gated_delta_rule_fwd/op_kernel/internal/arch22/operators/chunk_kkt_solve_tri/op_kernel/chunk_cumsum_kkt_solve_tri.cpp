#include "chunk_kkt_cube.h"
#include "../../chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h"
#include "../../solve_tri/op_kernel/solve_tri_cube.h"
#include "../../solve_tri/op_kernel/solve_tri_vector.h"

using namespace AscendC;

namespace {
constexpr uint64_t SCORE_READY_FLAG = 2;
constexpr uint64_t KKT_READY_FLAG = 3;

template <typename T, int MATRIX_SIZE, typename TilingData>
__aicore__ inline void RunSolvePhase(GM_ADDR a, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                     GM_ADDR out, GM_ADDR workspace,
                                     const TilingData *tilingData)
{
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
}
}  // namespace

#ifndef GDN_CHUNK_CUMSUM_KKT_SOLVE_IMPL_ONLY
template <uint32_t D_T_K, uint32_t CHUNK_KEY>
__global__ __aicore__ void chunk_cumsum_kkt_solve_tri(GM_ADDR k,
                                                       GM_ADDR g,
                                                       GM_ADDR beta,
                                                       GM_ADDR cuSeqlens,
                                                       GM_ADDR chunkIndices,
                                                       GM_ADDR gCumsum,
                                                       GM_ADDR A,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(ChunkKktSolveTriTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    GM_ADDR scoreWorkspace = userWorkspace;
    GM_ADDR aWorkspace = userWorkspace + tilingData.scoreWorkspaceBytes;
    GM_ADDR solveWorkspaceBase = aWorkspace + tilingData.aWorkspaceBytes;
    uint64_t coreGroup = static_cast<uint64_t>(GetBlockIdx());
    if ASCEND_IS_AIV {
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        coreGroup = subBlockNum == 0 ? coreGroup : coreGroup / subBlockNum;
    }
    GM_ADDR solveWorkspace = solveWorkspaceBase + coreGroup * tilingData.solveWorkspacePerCoreBytes;

    if ASCEND_IS_AIC {
        NsChunkKktCube::ChunkKktCube<DTYPE_K> kktCube;
        kktCube.Process(k, cuSeqlens, chunkIndices, scoreWorkspace, &tilingData);
        CrossCoreSetFlag<0x2, PIPE_FIX>(SCORE_READY_FLAG);
    }
    if ASCEND_IS_AIV {
        TPipe kktPipe;
        NsChunkScaledDotKkt::ChunkScaledDotKkt<DTYPE_K, DTYPE_K> kkt;
        kkt.InitFusedCumsum(
            k, g, beta, cuSeqlens, chunkIndices, gCumsum, aWorkspace, scoreWorkspace,
            tilingData.B, tilingData.Hk, tilingData.Hv, tilingData.hvPerHk,
            tilingData.T, tilingData.K, tilingData.BT, tilingData.NT,
            tilingData.taskNum, tilingData.usedAicNum, tilingData.usedAivNum,
            tilingData.btAlign, tilingData.isVarlen, &kktPipe);
        CrossCoreWaitFlag(SCORE_READY_FLAG);
        kkt.ProcessEpilogueForSolve(tilingData.tilesPerCore);
        kktPipe.Reset();
    }
    if (tilingData.BT == 64) {
        RunSolvePhase<DTYPE_K, 64>(aWorkspace, cuSeqlens, chunkIndices, A, solveWorkspace, &tilingData);
    } else {
        RunSolvePhase<DTYPE_K, 128>(aWorkspace, cuSeqlens, chunkIndices, A, solveWorkspace, &tilingData);
    }
}
#endif
