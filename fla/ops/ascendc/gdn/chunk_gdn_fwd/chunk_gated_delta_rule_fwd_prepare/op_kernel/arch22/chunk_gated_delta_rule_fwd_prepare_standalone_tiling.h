// Copyright (c) 2026 Tianjin University, Ltd.
// BSD-3-Clause. 独立 Prepare 的 A2 host/device 私有合同；不改变公开参数。
#ifndef GDN_ARCH22_PREPARE_STANDALONE_TILING_H
#define GDN_ARCH22_PREPARE_STANDALONE_TILING_H
#include "chunk_gated_delta_rule_fwd_prepare_policy.h"

namespace GdnUnifiedPrepare {
struct StandaloneTiling {
    uint64_t batch, hk, hv, tokens, bt, valueDim, nt, isVarlen;
    uint64_t cores, scratchBytes, internalAOffset, internalABytes, userWorkspaceBytes;
    uint64_t useQkNorm, useGate, useDtBias, useBetaSigmoid, allowNegEigval, useExp2, outputA;
};

#if !defined(__CCE_AICORE__)
inline bool StandaloneMul(uint64_t a, uint64_t b, uint64_t &out)
{
    if (b != 0 && a > static_cast<uint64_t>(INT64_MAX) / b) return false;
    out = a * b;
    return true;
}
inline bool StandaloneAdd(uint64_t a, uint64_t b, uint64_t &out)
{
    if (b > static_cast<uint64_t>(INT64_MAX) - a) return false;
    out = a + b;
    return true;
}

// WorkItems 与每核 arena 直接复用当前核心，先 checked 所有进入内核的总量。
inline bool MakeStandalonePlan(uint64_t batch, uint64_t hk, uint64_t hv, uint64_t tokens,
                              uint64_t bt, uint64_t valueDim, uint64_t nt, bool isVarlen,
                              uint64_t aicCores, uint64_t aivCores, bool outputA,
                              StandaloneTiling &out)
{
    out = {};
    if (!batch || !hk || !hv || !tokens || !nt || hv % hk ||
        (bt != 64 && bt != 128) || (valueDim != 128 && valueDim != 256) ||
        (isVarlen && batch != 1) || !aicCores || aivCores < 2) return false;
    const uint64_t group = hv / hk;
    const auto layout = MakeWorkspaceLayout(bt, valueDim, group);
    uint64_t tasks = 0, rows = 0, count = 0;
    if (!StandaloneMul(batch, nt, count) ||
        !StandaloneMul(count, CeilDiv(hk, layout.hkPerTask), tasks) ||
        !StandaloneMul(batch, hv, rows) || !StandaloneMul(rows, tokens, rows) ||
        !StandaloneMul(rows, Max(128, valueDim) * 2, count) ||
        !StandaloneMul(batch, hk, count) || !StandaloneMul(count, tokens, count) ||
        !StandaloneMul(count, 128 * 2, count)) return false;
    const uint64_t cores = Min(tasks, Min(aicCores, aivCores / 2));
    uint64_t scratch = 0, aBytes = 0, total = 0;
    if (!cores || tasks != WorkItems(batch, nt, hk, group, bt) ||
        !StandaloneMul(layout.coreBytes, cores, scratch)) return false;
    if (!outputA) {
        if (!StandaloneMul(rows, bt * 2, aBytes) ||
            aBytes > static_cast<uint64_t>(INT64_MAX) - 511) return false;
        aBytes = Align512(aBytes);
    }
    if (!StandaloneAdd(scratch, aBytes, total)) return false;
    out.batch = batch; out.hk = hk; out.hv = hv; out.tokens = tokens;
    out.bt = bt; out.valueDim = valueDim; out.nt = nt; out.isVarlen = isVarlen;
    out.cores = cores; out.scratchBytes = scratch;
    out.internalAOffset = scratch; out.internalABytes = aBytes; out.userWorkspaceBytes = total;
    out.outputA = outputA;
    return true;
}
#endif
} // namespace GdnUnifiedPrepare
#endif
