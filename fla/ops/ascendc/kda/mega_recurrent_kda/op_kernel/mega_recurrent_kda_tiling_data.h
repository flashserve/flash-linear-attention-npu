#pragma once
#include <stdint.h>

struct MegaRecurrentKdaTilingData {
    uint64_t convStride, checkpointStride, uStride, keyStride, decayStride;
    uint32_t batch, sequence, tokens, heads, capacity, prefix64, convLayout, crosses;
    float scale, lowerBound, rmsNormEps;
    uint32_t mxAlgorithm;
};
