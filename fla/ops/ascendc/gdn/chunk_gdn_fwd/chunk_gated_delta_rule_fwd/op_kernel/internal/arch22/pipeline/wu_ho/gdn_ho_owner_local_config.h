/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#pragma once

namespace GDN::HoOwnerLocal {

// 只用于新的 arch220/static BT64/V128 组合入口，不占用旧 idle ready 字段。
struct Config {
    bool enabled{false};
    bool outputSequenceMajor{false};
    GM_ADDR q{nullptr};
    GM_ADDR o{nullptr};
    GM_ADDR userWorkspace{nullptr};
    GdnMegaArch22FwdOTilingData oTiling{};
};

struct Layout {
    static constexpr uint32_t KIB = 1024;
    static constexpr uint32_t MASK = 0;
    static constexpr uint32_t CALC = 16 * KIB;
    static constexpr uint32_t DELTA = 48 * KIB;
    static constexpr uint32_t LOW0 = 83 * KIB;
    static constexpr uint32_t LOW1 = 99 * KIB;
    static constexpr uint32_t CARRY0 = 123 * KIB;
    static constexpr uint32_t CARRY1 = 155 * KIB;
    static constexpr uint32_t END = 187 * KIB;
    static constexpr uint32_t SYNC = 188 * KIB;
    static constexpr uint32_t W_L1 = 160 * KIB;
    static constexpr uint32_t B_L1 = 224 * KIB;
    static constexpr uint32_t B_SLOT = 48 * KIB;
    static constexpr uint32_t H_IN_B = 16 * KIB;
    static constexpr uint32_t L1_END = 320 * KIB;
    // META: HV1 G/Last 115/116、117/118，HV2 S写独占119/120。
    static constexpr uint32_t G0 = 115 * KIB;
    static constexpr uint32_t LAST0 = 116 * KIB;
    static constexpr uint32_t G1 = 117 * KIB;
    static constexpr uint32_t LAST1 = 118 * KIB;
    static constexpr uint32_t UPDATE_LAST0 = 119 * KIB;
    static constexpr uint32_t UPDATE_LAST1 = 120 * KIB;
    static constexpr uint32_t HV1_SCRATCH = 121 * KIB;
};

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
__aicore__ inline bool Add(uint64_t a, uint64_t b, uint64_t &out)
{
    constexpr uint64_t LIMIT = 0x7fffffffffffffffULL;
    if (a > LIMIT || b > LIMIT - a) { return false; }
    out = a + b;
    return true;
}

__aicore__ inline bool Mul(uint64_t a, uint64_t b, uint64_t &out)
{
    constexpr uint64_t LIMIT = 0x7fffffffffffffffULL;
    if (b != 0 && a > LIMIT / b) { return false; }
    out = a * b;
    return true;
}

__aicore__ inline bool Aligned(uint64_t value, uint64_t extra, uint64_t &out)
{
    uint64_t sum = 0;
    if (!Add(value, extra, sum)) { return false; }
    out = sum / 512 * 512;
    return true;
}

// 所有核只读取同一tiling/cu几何。拒绝只影响优化，绝不按本core active数回退。
__aicore__ inline Config Resolve(
    GM_ADDR hTilingAddr, GM_ADDR cuSeqlens, GM_ADDR q, GM_ADDR o, GM_ADDR userWorkspace,
    const GdnMegaArch22FwdOTilingData &ot, int64_t hPersistent, int64_t vPersistent,
    bool idleEnabled, bool outputSequenceMajor)
{
    Config cfg{};
    const auto *ht = reinterpret_cast<const __gm__ GdnMegaArch22FwdHTilingData *>(hTilingAddr);
    const uint64_t cores = AscendC::GetBlockNum();
    if (idleEnabled || !outputSequenceMajor || cores == 0 || ht->useGk != 0 ||
        ht->stateDataType != 2 || ht->kHeadDim != 128 || ht->vHeadDim != 128 ||
        ht->chunkSize != 64 || ht->shapeBatch <= 0 || ht->batch <= 0 || ht->seqlen <= 0 ||
        ht->vNumHead <= 0 || ht->kNumHead <= 0 || ht->vNumHead % ht->kNumHead != 0 ||
        ht->tokenBatch <= 0 || ht->tokenBatch > 0xffffffffLL ||
        ht->seqlen > 0xffffffffLL || ht->batch > 0xffffffffLL ||
        ht->vNumHead > 0xffffffffLL || ht->kNumHead > 0xffffffffLL ||
        ot.shapeBatch != ht->shapeBatch || ot.seqlen != ht->seqlen || ot.kNumHead != ht->kNumHead ||
        ot.vNumHead != ht->vNumHead || ot.kHeadDim != 128 || ot.vHeadDim != 128 || ot.chunkSize != 64 ||
        ot.isVariedLen != ht->isVariedLen || ot.tokenBatch != ht->tokenBatch) {
        return cfg;
    }
    uint64_t sequences = static_cast<uint64_t>(ht->batch);
    uint64_t chunks = ht->seqlen / 64 + (ht->seqlen % 64 != 0);
    if (ht->isVariedLen != 0) {
        if (cuSeqlens == nullptr || ht->shapeBatch != 1) { return cfg; }
        AscendC::GlobalTensor<int64_t> cu;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        if (cu.GetValue(0) != 0) { return cfg; }
        sequences = 0;
        chunks = 0;
        int64_t prev = 0;
        for (int64_t i = 1; i <= ht->tokenBatch; ++i) {
            const int64_t end = cu.GetValue(i);
            if (end < prev || end > ht->seqlen) { return cfg; }
            const uint64_t n = static_cast<uint64_t>(end - prev);
            if (n != 0) {
                ++sequences;
                if (!Add(chunks, n / 64 + (n % 64 != 0), chunks)) { return cfg; }
            }
            prev = end;
        }
        if (prev != ht->seqlen) { return cfg; }
    }
    uint64_t owners = 0;
    if (sequences == 0 || !Mul(sequences, ht->vNumHead, owners) ||
        owners <= cores || owners > 0xffffffffULL || chunks == 0 || chunks > 0xffffffffULL) { return cfg; }
    // 原chunk pipeline要求dense batch1/Hv8/C24，其P<C；这里P>C，双方互斥。
    uint64_t stateElements = 0, uvElements = 0, finalElements = 0, x = 0;
    if (!Mul(ht->shapeBatch, chunks, x) || !Mul(x, ht->vNumHead, x) ||
        !Mul(x, 128 * 128, stateElements) || stateElements > 0xffffffffULL ||
        !Mul(ht->shapeBatch, ht->vNumHead, x) || !Mul(x, ht->seqlen, x) ||
        !Mul(x, 128, uvElements) || uvElements > 0xffffffffULL ||
        !Mul(owners, 128 * 128, finalElements) || finalElements > 0xffffffffULL) { return cfg; }
    const int64_t hs[] = {ht->vWorkspaceOffset, ht->vUpdateWorkspaceOffset, ht->kDecayWorkspaceOffset,
                         ht->hWorkspaceOffset, ht->numSeqWorkspaceOffset, ht->numChunksWorkspaceOffset};
    for (uint32_t i = 0; i < 6; ++i) {
        if (hs[i] < 0 || static_cast<uint64_t>(hs[i]) > 0xffffffffULL) { return cfg; }
    }
    if (ot.vWorkspaceOffset < 0 || ot.hWorkspaceOffset < 0 || ot.attnWorkspaceOffset < 0 ||
        ot.aftermaskWorkspaceOffset < 0 || ot.maskWorkspaceOffset < 0 || hPersistent < 0 || vPersistent < 0) { return cfg; }
    uint64_t wh = 0, delta = 0, meta = 0, end = 0, expected = 0;
    if (!Mul(cores, 2 * 64 * 128 * 4, wh) || !Aligned(wh, 512, wh) ||
        !Mul(cores, 2 * 128 * 128 * 4, delta) || !Aligned(delta, 512, delta) ||
        !Add(ht->tokenBatch, 1, meta) || !Mul(meta, 8, meta) || !Aligned(meta, 512, meta) ||
        !Add(hs[0], wh, expected) || expected != static_cast<uint64_t>(hs[1]) ||
        !Add(expected, wh, expected) || expected != static_cast<uint64_t>(hs[2]) || hs[2] != hs[3] ||
        !Add(hs[3], delta, expected) || expected != static_cast<uint64_t>(hs[4]) ||
        !Add(expected, meta, expected) || expected != static_cast<uint64_t>(hs[5]) ||
        !Add(expected, meta, end) || !Add(end, 16 * 1024 * 1024, end) ||
        !Aligned(end, 511, expected) || static_cast<uint64_t>(ot.vWorkspaceOffset) < expected) { return cfg; }
    uint64_t vBytes = 0, qhBytes = 0, rawBytes = 0;
    if (!Mul(cores, 2 * 64 * 128 * 4, vBytes) || !Mul(cores, 4 * 64 * 128 * 4, qhBytes) ||
        !Mul(cores, 2 * 64 * 64 * 4, rawBytes) ||
        !Add(ot.vWorkspaceOffset, vBytes, end) || static_cast<uint64_t>(ot.hWorkspaceOffset) < end ||
        !Add(ot.hWorkspaceOffset, qhBytes, end) || static_cast<uint64_t>(ot.attnWorkspaceOffset) < end ||
        !Add(ot.attnWorkspaceOffset, rawBytes, end) || static_cast<uint64_t>(ot.aftermaskWorkspaceOffset) < end ||
        !Add(ot.aftermaskWorkspaceOffset, rawBytes, end) || static_cast<uint64_t>(ot.maskWorkspaceOffset) < end ||
        !Add(ot.maskWorkspaceOffset, 64 * 64 + 16 * 1024 * 1024, end) ||
        end > static_cast<uint64_t>(hPersistent) || !Mul(stateElements, 2, x) ||
        !Aligned(x, 511, x) || !Add(hPersistent, x, end) || end > static_cast<uint64_t>(vPersistent) ||
        !Mul(uvElements, 2, x) || !Add(vPersistent, x, end)) { return cfg; }
    cfg.enabled = true;
    cfg.outputSequenceMajor = outputSequenceMajor;
    cfg.q = q;
    cfg.o = o;
    cfg.userWorkspace = userWorkspace;
    cfg.oTiling = ot;
    return cfg;
}
#endif

} // namespace GDN::HoOwnerLocal
