#pragma once
#include "../mega_recurrent_kda_tiling_data.h"
#include "mega_recurrent_kda_v2_tiling.h"

struct MegaRecurrentKdaArch22TilingData {
    MegaRecurrentKdaTilingData common;
    uint64_t sourceConvStride;
    uint32_t inputTokens, poolSlots, convLength;
    MegaRecurrentKdaV2::KdaTiling arch22;
};
