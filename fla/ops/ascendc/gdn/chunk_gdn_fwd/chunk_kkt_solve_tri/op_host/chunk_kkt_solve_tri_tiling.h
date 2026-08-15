#ifndef CHUNK_KKT_SOLVE_TRI_TILING_H
#define CHUNK_KKT_SOLVE_TRI_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkKktSolveTriTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, B);
    TILING_DATA_FIELD_DEF(uint64_t, Hk);
    TILING_DATA_FIELD_DEF(uint64_t, Hv);
    TILING_DATA_FIELD_DEF(uint64_t, hvPerHk);
    TILING_DATA_FIELD_DEF(uint64_t, T);
    TILING_DATA_FIELD_DEF(uint64_t, K);
    TILING_DATA_FIELD_DEF(uint64_t, BT);
    TILING_DATA_FIELD_DEF(uint64_t, NT);
    TILING_DATA_FIELD_DEF(uint64_t, taskNum);
    TILING_DATA_FIELD_DEF(uint64_t, usedAicNum);
    TILING_DATA_FIELD_DEF(uint64_t, usedAivNum);
    TILING_DATA_FIELD_DEF(uint64_t, btAlign);
    TILING_DATA_FIELD_DEF(uint64_t, isVarlen);
    TILING_DATA_FIELD_DEF(uint64_t, scoreWorkspaceBytes);
    TILING_DATA_FIELD_DEF(uint64_t, aWorkspaceBytes);
    TILING_DATA_FIELD_DEF(uint64_t, solveWorkspacePerCoreBytes);
    TILING_DATA_FIELD_DEF(int64_t, totalTiles);
    TILING_DATA_FIELD_DEF(int64_t, matrixSize);
    TILING_DATA_FIELD_DEF(int64_t, numHeads);
    TILING_DATA_FIELD_DEF(int64_t, seqLen);
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(int64_t, isLower);
    TILING_DATA_FIELD_DEF(int64_t, hasCuSeqlens);
    TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
    TILING_DATA_FIELD_DEF(int64_t, chunkSize);
    TILING_DATA_FIELD_DEF(int64_t, numChunks);
    TILING_DATA_FIELD_DEF(int64_t, lastChunkValidSize);
    TILING_DATA_FIELD_DEF(int64_t, totalChunks);
    TILING_DATA_FIELD_DEF(int64_t, layoutMode);
    TILING_DATA_FIELD_DEF(int64_t, dtypeMode);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkKktSolveTri, ChunkKktSolveTriTilingData)
REGISTER_TILING_DATA_CLASS(ChunkCumsumKktSolveTri, ChunkKktSolveTriTilingData)
}  // namespace optiling

#endif
