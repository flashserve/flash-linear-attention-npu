#ifndef GDN_BWD_FINALIZE_A2_WORKSPACE_H
#define GDN_BWD_FINALIZE_A2_WORKSPACE_H

#include <cstdint>
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "chunk_gated_delta_rule_bwd_finalize_struct.h"
#else
#include "../chunk_gated_delta_rule_bwd_finalize_struct.h"
#endif

#if defined(__CCE_AICORE__)
#define GDN_A2_HOST_DEVICE __aicore__ inline
#else
#define GDN_A2_HOST_DEVICE inline
#endif

namespace GDN {

namespace FinalizeArch22 {
#ifndef TPL_BF16
#define TPL_BF16 10
#define TPL_FP32 30
#endif
#ifndef TORCH_MODE
// Host and arch22 kernels share the complete template declaration.
ASCENDC_TPL_ARGS_DECL(ChunkGatedDeltaRuleBwdFinalize,
    ASCENDC_TPL_DTYPE_DECL(D_T_Q, TPL_BF16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, TPL_BF16, TPL_FP32),
    ASCENDC_TPL_BOOL_DECL(USE_QK_L2NORM, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_BETA_SIGMOID, 0, 1),
    ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1),
    ASCENDC_TPL_BOOL_DECL(ENABLE_MHA_OWNER, 0, 1),
);

#define GDN_A2_TPL_SEL_ONE(Q, G, NORM, SIGMOID, EXP2, MHA_OWNER) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DTYPE_SEL(D_T_Q, Q), \
        ASCENDC_TPL_DTYPE_SEL(D_T_G, G), \
        ASCENDC_TPL_BOOL_SEL(USE_QK_L2NORM, NORM), \
        ASCENDC_TPL_BOOL_SEL(USE_BETA_SIGMOID, SIGMOID), \
        ASCENDC_TPL_BOOL_SEL(USE_EXP2, EXP2), \
        ASCENDC_TPL_BOOL_SEL(ENABLE_MHA_OWNER, MHA_OWNER), \
    )

#define GDN_A2_TPL_SEL_FOR_DTYPE(Q, G, MHA_OWNER) \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 0, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 0, 1, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 1, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 0, 1, 1, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 0, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 0, 1, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 1, 0, MHA_OWNER), \
    GDN_A2_TPL_SEL_ONE(Q, G, 1, 1, 1, MHA_OWNER)

ASCENDC_TPL_SEL(
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_BF16, 0),
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_FP32, 0),
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_BF16, 1),
    GDN_A2_TPL_SEL_FOR_DTYPE(TPL_BF16, TPL_FP32, 1),
);

#undef GDN_A2_TPL_SEL_FOR_DTYPE
#undef GDN_A2_TPL_SEL_ONE
#endif

} // namespace FinalizeArch22

// Host 和 device 共用同一份字节布局。DQ/WY 的临时区只在全核 barrier 后复用。
struct A2FinalizeWorkspaceLayout {
    int64_t gOffset;
    int64_t betaOffset;
    int64_t hOffset;
    int64_t dhOffset;
    int64_t dqOffset;
    int64_t dkOffset;
    int64_t dkWyOffset;
    int64_t dwOffset;
    int64_t dgDqOffset;
    int64_t dgWyOffset;
    int64_t dbetaOffset;
    int64_t stageOffset;
    int64_t dqWorkspaceBytes;
    int64_t wyWorkspaceBytes;
    int64_t totalBytes;

    GDN_A2_HOST_DEVICE static int64_t Align(int64_t bytes)
    {
        return (bytes + 511) / 512 * 512;
    }

    GDN_A2_HOST_DEVICE A2FinalizeWorkspaceLayout(
        const ChunkGatedDeltaRuleBwdFinalizeTilingData &t, int64_t blocks)
    {
        const int64_t scalar = t.B * t.HV * t.T;
        const int64_t qBytes = t.B * t.HK * t.T * t.K * 2;
        const int64_t stateBytes = t.stateVFirst != 0 ?
            t.totalChunkNum * t.HV * t.K * t.V * 2 : 0;
        int64_t offset = 0;
        gOffset = offset; offset += Align(scalar * 4);
        betaOffset = offset; offset += Align(scalar * 4);
        hOffset = offset; offset += Align(stateBytes);
        dhOffset = offset; offset += Align(stateBytes);
        dqOffset = offset; offset += Align(qBytes);
        dkOffset = offset; offset += Align(qBytes);
        dkWyOffset = offset; offset += Align(qBytes);
        dwOffset = offset; offset += Align(scalar * t.K * 2);
        dgDqOffset = offset; offset += Align(scalar * 4);
        dgWyOffset = offset; offset += Align(scalar * 4);
        dbetaOffset = offset; offset += Align(scalar * 4);
        stageOffset = offset;

        const int64_t btk = blocks * t.HV * t.chunkSize * t.K * 2;
        const int64_t btb = blocks * t.HV * t.chunkSize * t.chunkSize * 2;
        dqWorkspaceBytes = 4 * btk + 2 * btb + Align(blocks * t.HV * 4);
        const int64_t kBytes = t.chunkSize * t.K * 2;
        const int64_t vBytes = t.chunkSize * t.V * 2;
        const int64_t mBytes = t.chunkSize * t.chunkSize * 2;
        wyWorkspaceBytes = blocks * 4 * (5 * kBytes + 2 * vBytes + 7 * mBytes);
        offset += Align(dqWorkspaceBytes > wyWorkspaceBytes ? dqWorkspaceBytes : wyWorkspaceBytes);
        totalBytes = offset;
    }
};

} // namespace GDN

#undef GDN_A2_HOST_DEVICE
#endif
