#ifndef __SOLVE_TRIL_TILING_DATA_H__
#define __SOLVE_TRIL_TILING_DATA_H__

#include <cstdint>

struct SolveTrilTilingData {
    int64_t batchSize;       // BSND -> num of fixed-length seqs; TND -> 1
    int64_t seqLength;       // BSND -> length of each fixed seq; TND -> invalid
    int64_t numHead;         // num heads (no difference between modes)
    int64_t chunkSize;       // chunk block size BT in {16,32,64,128}
    int64_t chunkNumInSeq;   // BSND -> CeilDiv(seqLength, chunkSize); TND -> invalid
    int64_t chunkNumTotal;   // BSND -> batchSize * numHead * chunkNumInSeq; TND -> chunk_indices.shape(0) * numHead
    int32_t mode;            // 0 = BSND, 1 = TND
    int64_t blockDim;        // num AI cores used
    int64_t taskPerCore;     // tasks assigned per core
    int64_t rowStride;       // GM stride between rows in a chunk = numHead * chunkSize
    int64_t mbhLevels;       // 0/1/2/3 for BT={16,32,64,128}
};

#endif // __SOLVE_TRIL_TILING_DATA_H__
