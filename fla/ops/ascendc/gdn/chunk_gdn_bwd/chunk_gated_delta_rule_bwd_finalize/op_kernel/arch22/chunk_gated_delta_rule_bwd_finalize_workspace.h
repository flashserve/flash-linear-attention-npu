#ifndef GDN_BWD_FINALIZE_A2_WORKSPACE_H
#define GDN_BWD_FINALIZE_A2_WORKSPACE_H

#include <cstdint>
#include "../chunk_gated_delta_rule_bwd_finalize_struct.h"

#if defined(__CCE_AICORE__)
#define GDN_A2_HOST_DEVICE __aicore__ inline
#else
#define GDN_A2_HOST_DEVICE inline
#endif

namespace GDN {

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
