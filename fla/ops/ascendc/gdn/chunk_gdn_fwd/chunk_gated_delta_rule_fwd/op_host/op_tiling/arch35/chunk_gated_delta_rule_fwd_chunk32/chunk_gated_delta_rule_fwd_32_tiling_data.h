#ifndef CHUNK_GATED_DELTA_RULE_FWD_32_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_FWD_32_TILING_DATA_H

#include <cstdint>

struct ChunkGatedDeltaRuleFwd32TilingData {
    uint32_t B;
    uint32_t HV;
    uint32_t HK;
    uint32_t T;
    uint32_t hasInitial;
    float scale;
    uint32_t coreNum;
    // 0: one sequence per batch row. Otherwise B is 1 and T is packed by cu_seqlens.
    uint32_t seqNum;
    // 1: initial_state is FP32 and enters the recurrence unchanged.
    uint32_t initialFp32;
    // 1: final_state is FP32. That is the absent-initial case and the FP32-initial case.
    // A BF16 initial writes a BF16 final. The on-chip recurrence stays FP32 either way.
    uint32_t finalFp32;
    // 1: GM initial/final state is physical [V,K], and the on-chip H uses that
    // same axis order. 0 keeps the historical [K,V] schedule.
    uint32_t stateVFirst;
    // 1: beta_eff = 2 * sigmoid(beta). 0 keeps sigmoid(beta).
    uint32_t allowNegEigval;
};

#endif
