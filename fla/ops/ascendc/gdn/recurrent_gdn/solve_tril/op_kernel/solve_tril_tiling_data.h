#ifndef __SOLVE_TRIL_TILING_DATA_H__
#define __SOLVE_TRIL_TILING_DATA_H__

#include <cstdint>

struct SolveTrilTilingData {
    int64_t n;
    int64_t batchSize;
    int64_t numLeafBlocks;
    int64_t mbhLevels;
    int64_t blockDim;
    int64_t taskPerCore;
    int64_t workspaceOffset;
    int64_t reserved;  // padding to 64 bytes (opParaSize)
};

#endif // __SOLVE_TRIL_TILING_DATA_H__
