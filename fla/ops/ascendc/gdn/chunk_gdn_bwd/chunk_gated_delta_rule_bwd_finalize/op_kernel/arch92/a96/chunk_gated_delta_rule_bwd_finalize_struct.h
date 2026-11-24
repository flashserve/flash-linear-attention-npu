#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

constexpr int64_t GVA_ACC_CORE_BYTES = 2 * 64 * 128 * sizeof(float);

#define TPL_BF16 10
#define TPL_FP32 30

#ifndef TORCH_MODE

ASCENDC_TPL_ARGS_DECL(ChunkGatedDeltaRuleBwdFinalize,
    ASCENDC_TPL_DTYPE_DECL(D_T_Q, TPL_BF16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, TPL_BF16, TPL_FP32),
    ASCENDC_TPL_BOOL_DECL(USE_QK_L2NORM, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_BETA_SIGMOID, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1),
);

#define TPL_SEL_ONE(Q, G, USE_QK_L2NORM_VALUE, USE_BETA_SIGMOID_VALUE, USE_EXP2_VALUE) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DTYPE_SEL(D_T_Q, Q), \
        ASCENDC_TPL_DTYPE_SEL(D_T_G, G), \
        ASCENDC_TPL_BOOL_SEL(USE_QK_L2NORM, USE_QK_L2NORM_VALUE), \
        ASCENDC_TPL_BOOL_SEL(USE_BETA_SIGMOID, USE_BETA_SIGMOID_VALUE), \
        ASCENDC_TPL_BOOL_SEL(USE_EXP2, USE_EXP2_VALUE), \
    )

#define TPL_SEL_FOR_DTYPE(Q, G) \
    TPL_SEL_ONE(Q, G, 0, 0, 0), \
    TPL_SEL_ONE(Q, G, 0, 0, 1), \
    TPL_SEL_ONE(Q, G, 0, 1, 0), \
    TPL_SEL_ONE(Q, G, 0, 1, 1), \
    TPL_SEL_ONE(Q, G, 1, 0, 0), \
    TPL_SEL_ONE(Q, G, 1, 0, 1), \
    TPL_SEL_ONE(Q, G, 1, 1, 0), \
    TPL_SEL_ONE(Q, G, 1, 1, 1)

ASCENDC_TPL_SEL(

    TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_BF16),
    TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_FP32),
);

#undef TPL_SEL_FOR_DTYPE
#undef TPL_SEL_ONE
#endif

namespace FinalizeLayout {
enum Layout : int64_t {
    BNSD = 0,
    BSND = 1,
};
}

struct ChunkGatedDeltaRuleBwdFinalizeTilingData {

    int64_t B;
    int64_t HK;
    int64_t HV;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t chunkSize;

    int64_t chunkNumForT;
    int64_t totalChunkNum;
    int64_t taskNum;
    int64_t seqNum;
    int64_t isVariable;
    int64_t stateVFirst;

    int64_t headRatio;
    int64_t taskGroupSize;
    int64_t headGroupNum;

    float scale;

    int64_t layout;
};

}

#endif
