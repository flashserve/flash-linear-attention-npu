// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. A2 GDN Prepare：单资源、持久 work 循环、双版本 GM 的统一入口。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_H
#define GDN_ARCH22_UNIFIED_PREPARE_H

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 220
#error "Unified GDN Prepare is a DAV_2201 implementation"
#endif

#include "chunk_gated_delta_rule_fwd_prepare_cube.h"
#include "chunk_gated_delta_rule_fwd_prepare_vector.h"

namespace GdnUnifiedPrepare {

// 只改变同 owner 的 Scores 发射时点；所有证明失败都沿原顺序。
template <typename InputT, bool Prepared, int V, int BT>
class ScoreLookahead {
    struct Range { GM_ADDR address; uint64_t bytes; };
    const Args &a_;
    Range writes_[9]{};
    uint32_t writeCount_ = 0;
    bool enabled_ = false;

    __aicore__ static inline bool ScaleBytes(uint64_t &bytes, uint64_t factor)
    {
        const uint32_t aLo = static_cast<uint32_t>(bytes);
        const uint32_t aHi = static_cast<uint32_t>(bytes >> 32);
        const uint32_t bLo = static_cast<uint32_t>(factor);
        const uint32_t bHi = static_cast<uint32_t>(factor >> 32);
        if (aHi != 0 && bHi != 0) return false;
        // 两个32位分肢相乘均能放入uint64；高分肢同时非零已在上方拒绝。
        const uint64_t low = static_cast<uint64_t>(aLo) * bLo;
        const uint64_t cross = aHi != 0 ? static_cast<uint64_t>(aHi) * bLo :
            static_cast<uint64_t>(aLo) * bHi;
        const uint64_t high = (low >> 32) + cross;
        if (high > 0xFFFFFFFFULL) return false;
        bytes = (high << 32) | static_cast<uint32_t>(low);
        return true;
    }

    __aicore__ static inline bool RangeFits(GM_ADDR address, uint64_t bytes)
    {
        return bytes == 0 || (address != nullptr &&
            reinterpret_cast<uint64_t>(address) <= ~uint64_t(0) - bytes);
    }

    __aicore__ inline bool AddWrite(GM_ADDR address, uint64_t bytes)
    {
        if (!RangeFits(address, bytes) || writeCount_ == 9) return false;
        writes_[writeCount_++] = {address, bytes};
        return true;
    }

    __aicore__ inline bool SafeRead(GM_ADDR address, uint64_t bytes) const
    {
        if (!RangeFits(address, bytes)) return false;
        const uint64_t x = reinterpret_cast<uint64_t>(address);
        for (uint32_t i = 0; i < writeCount_; ++i) {
            const uint64_t y = reinterpret_cast<uint64_t>(writes_[i].address);
            if (bytes != 0 && writes_[i].bytes != 0 &&
                !(x <= y ? y - x >= bytes : x - y >= writes_[i].bytes)) return false;
        }
        return true;
    }

    __aicore__ inline bool MetadataPair(GM_ADDR base, uint64_t first, GM_ADDR &pair) const
    {
        if (base == nullptr) return false;
        uint64_t offset = first;
        if (!ScaleBytes(offset, sizeof(int64_t))) return false;
        const uint64_t address = reinterpret_cast<uint64_t>(base);
        if (address > ~uint64_t(0) - offset) return false;
        pair = reinterpret_cast<GM_ADDR>(address + offset);
        return SafeRead(pair, 2 * sizeof(int64_t));
    }

public:
    __aicore__ inline ScoreLookahead(const Args &a, const WorkspaceLayout &p,
                                     uint64_t cores, bool needed) : a_(a)
    {
        if (!needed || a.hk == 0 || a.nt == 0) return;
        uint64_t keys = a.batch, rows = a.batch;
        if (!ScaleBytes(keys, a.hk) || !ScaleBytes(keys, a.tokens) ||
            !ScaleBytes(keys, 128 * sizeof(InputT)) ||
            !ScaleBytes(rows, a.hv) || !ScaleBytes(rows, a.tokens)) return;
        uint64_t gate = rows, matrix = rows, w = rows, u = rows, scratch = p.coreBytes;
        if (!ScaleBytes(gate, sizeof(float)) || !ScaleBytes(matrix, BT * sizeof(InputT)) ||
            !ScaleBytes(w, 128 * sizeof(InputT)) || !ScaleBytes(u, V * sizeof(InputT)) ||
            !ScaleBytes(scratch, cores) || !AddWrite(a.a, matrix) || !AddWrite(a.w, w) ||
            !AddWrite(a.u, u) || !AddWrite(a.gBht, gate) || !AddWrite(a.scratch, scratch)) return;
        if constexpr (Prepared) {
            uint64_t last = a.batch;
            if (!ScaleBytes(last, a.nt) || !ScaleBytes(last, CeilDiv(a.hv, 8)) ||
                !ScaleBytes(last, 8 * sizeof(float)) ||
                !AddWrite(a.gDecay, gate) || !AddWrite(a.gLastExp, last)) return;
        }
        if (a.outputG != 0 && a.gBth != nullptr && !AddWrite(a.gBth, gate)) return;
        if (a.betaEff != nullptr && !AddWrite(a.betaEff, gate)) return;
        enabled_ = SafeRead(a.k, keys);
    }

    __aicore__ inline bool Next(uint64_t work, uint64_t hksPerTask, Chunk &c) const
    {
        if (!enabled_) return false;
        const uint64_t partitions = CeilDiv(a_.hk, hksPerTask);
        const uint64_t globalChunk = work / partitions;
        c = {};
        c.batch = globalChunk / a_.nt;
        c.index = globalChunk % a_.nt;
        if (a_.isVarlen == 0) {
            c.row = c.index;
            if (!ScaleBytes(c.row, BT)) return false;
            c.valid = c.row < a_.tokens ? Min(BT, a_.tokens - c.row) : 0;
        } else {
            // 先证明实际 indices 两项可读，不能先 GetValue 再检查别名。
            uint64_t first = c.index;
            GM_ADDR pair{};
            if (!ScaleBytes(first, 2) || !MetadataPair(a_.chunkIndices, first, pair)) return false;
            auto indices = Gm<int64_t>(pair, 2);
            const int64_t sequence = indices.GetValue(0);
            const int64_t localChunk = indices.GetValue(1);
            if (sequence < 0 || localChunk < 0 ||
                !MetadataPair(a_.cuSeqlens, static_cast<uint64_t>(sequence), pair)) return false;
            // Args 没有 cu 总长；只保护原有效输入合同中的真实 cu[sequence:sequence+2]。
            auto cu = Gm<int64_t>(pair, 2);
            const int64_t begin = cu.GetValue(0);
            const int64_t end = cu.GetValue(1);
            if (begin < 0 || end < begin) return false;
            uint64_t offset = static_cast<uint64_t>(localChunk);
            if (!ScaleBytes(offset, BT) ||
                offset > uint64_t(0x7FFFFFFFFFFFFFFFULL) - static_cast<uint64_t>(begin)) return false;
            c.batch = 0;
            c.row = static_cast<uint64_t>(begin) + offset;
            c.valid = static_cast<uint64_t>(end) > c.row && c.row < a_.tokens
                ? Min(BT, Min(static_cast<uint64_t>(end) - c.row, a_.tokens - c.row)) : 0;
        }
        c.hkBegin = work % partitions;
        if (!ScaleBytes(c.hkBegin, hksPerTask) || c.hkBegin >= a_.hk) return false;
        c.hkEnd = c.hkBegin + Min(hksPerTask, a_.hk - c.hkBegin);
        c.headBegin = c.hkBegin;
        c.headEnd = c.hkEnd;
        const uint64_t g = a_.hv / a_.hk;
        if (!ScaleBytes(c.headBegin, g) || !ScaleBytes(c.headEnd, g)) return false;
        return c.valid != 0;
    }
};

template <typename InputT, bool Prepared, int V, int BT, int QkvLayout, bool QkNorm = false>
__aicore__ inline void Run(const Args &args)
{
    static_assert(sizeof(InputT) == 2, "GDN preserves its FP16/BF16 intermediate rounding");
    static_assert(V == 128 || V == 256);
    static_assert(BT == 64 || BT == 128);
    static_assert(QkvLayout == 0 || QkvLayout == 1);
    constexpr uint32_t Window = kHeadsPerWindow;
    const uint64_t g = args.hv / args.hk;
    const auto p = MakeWorkspaceLayout(BT, V, g);
    const uint64_t total = WorkItems(args.batch, args.nt, args.hk, g, BT);
    const uint64_t group = GroupId();
    if (group >= total) return; // 空物理组不建立资源、不 seed、不等待集合。
    Resources owner(Cube<InputT, V, BT, QkvLayout>::kCSlots); // 跨 task/wave 仅此一个资源拥有者。
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        Cube<InputT, V, BT, QkvLayout> cube(owner, args, p);
        // 整个持久 work 循环仅 seed 一次。Norm模板专用0，1仍留给其它kernel。
        CrossCoreSetFlag<0x2, PIPE_MTE2>(6);
        CrossCoreSetFlag<0x2, PIPE_MTE2>(7);
        uint64_t wave = 0;
        const uint64_t step = GetBlockNum();
        ScoreLookahead<InputT, Prepared, V, BT> lookahead(args, p, step, step < total - group);
        bool primed = false;
        uint64_t pendingWork = 0;
        Chunk pendingChunk{};
        for (uint64_t work = group; work < total; work += GetBlockNum()) {
            const bool fromPending = primed && pendingWork == work;
            const Chunk c = fromPending ? pendingChunk : Decode<BT>(args, work, p.hkPerTask);
            primed = false;
            if (c.valid == 0) continue;
            if (!fromPending) {
                if constexpr (QkNorm) CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
                cube.Scores(c); // 每 task 仍一次 raw Scores。
            }
            for (uint64_t begin = c.headBegin; begin < c.headEnd;) {
                const uint32_t windows = static_cast<uint32_t>(Min(2, CeilDiv(c.headEnd - begin, Window)));
                // 必须先发布本组全部 raw-ready，AIV 才能完成两窗 KKT 并进入 Assemble16。
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    cube.template Merge<16>(c, slot, active);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    cube.template Merge<32>(c, slot, active);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                }
                if constexpr (BT == 128) {
                    for (uint32_t i = 0; i < windows; ++i) {
                        const uint64_t head = begin + i * Window;
                        const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                        const uint32_t slot = (wave + i) & 1;
                        CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                        owner.template Fence<HardEvent::MTE2_S>();
                        cube.template Merge<64>(c, slot, active);
                        CrossCoreSetFlag<0x2, PIPE_FIX>(4 + slot);
                    }
                }
                // 当前 raw score 的两窗读者已关闭；不提前发布下一 work 的 4/5。
                // Norm 分支先发起当前 WU，下一轮再等待下一个 work 的归一化。
                if (!QkNorm && c.headEnd - begin <= windows * Window && step < total - work) {
                    const uint64_t nextWork = work + step;
                    Chunk nextChunk{};
                    if (lookahead.Next(nextWork, p.hkPerTask, nextChunk)) {
                        if constexpr (QkNorm) CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
                        cube.Scores(nextChunk);
                        pendingChunk = nextChunk;
                        pendingWork = nextWork;
                        primed = true;
                    }
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(2 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    cube.WU(c, head, active, slot);
                }
                begin += windows * Window;
                wave += windows;
            }
        }
    }
    if ASCEND_IS_AIV {
        Vector<InputT, Prepared, V, BT, QkvLayout, QkNorm> vector(owner, args, p);
        uint64_t wave = 0;
        for (uint64_t work = group; work < total; work += GetBlockNum()) {
            const Chunk c = Decode<BT>(args, work, p.hkPerTask);
            if (c.valid == 0) continue;
            if constexpr (QkNorm) {
                vector.Norm(c);
                // 两sub均发，覆盖本地MTE3帽子写；必须先于下一次6/7等待。
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
            }
            for (uint64_t begin = c.headBegin; begin < c.headEnd;) {
                const uint32_t windows = static_cast<uint32_t>(Min(2, CeilDiv(c.headEnd - begin, Window)));
                const bool prefixPair = Prepared && windows == 2 && vector.CanPrefixPair();
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    // 空 AIV 仍参加全部集合；wave 只对本组实际窗口递增。
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(6 + slot);
                    if (prefixPair) {
                        if (i == 0) vector.PrefixPair(c, begin,
                            static_cast<uint32_t>(Min(2 * Window, c.headEnd - begin)), slot);
                    } else {
                        if constexpr (!Prepared && BT == 64 && QkvLayout == 0) {
                            if (windows == 2) {
                                if (i == 0) vector.PrefixPairPublic(c, begin,
                                    static_cast<uint32_t>(Min(2 * Window, c.headEnd - begin)), slot);
                            } else {
                                vector.Prefix(c, head, active, slot);
                            }
                        } else {
                            vector.Prefix(c, head, active, slot);
                        }
                    }
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    vector.KktAndLeaves(c, head, active, slot);
                    vector.Publish(slot);
                }
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    vector.template Assemble<16>(c, head, active, slot);
                    vector.Publish(slot);
                }
                if constexpr (BT == 128) {
                    for (uint32_t i = 0; i < windows; ++i) {
                        const uint64_t head = begin + i * Window;
                        const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                        const uint32_t slot = (wave + i) & 1;
                        CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                        owner.template Fence<HardEvent::MTE2_S>();
                        vector.template Assemble<32>(c, head, active, slot);
                        vector.Publish(slot);
                    }
                }
                // 本 AIV 的两槽 Assemble16 都已完成，才可用整个 leaf 区做 Final/RHS。
                for (uint32_t i = 0; i < windows; ++i) {
                    const uint64_t head = begin + i * Window;
                    const uint32_t active = static_cast<uint32_t>(Min(Window, c.headEnd - head));
                    const uint32_t slot = (wave + i) & 1;
                    vector.Rhs(c, head, active, slot);
                    // 完整 RHS 已写独立 GM；先关闭 MTE3，再进入 Final 的 Scalar DCCI。
                    owner.template Fence<HardEvent::MTE3_S>();
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(4 + slot);
                    owner.template Fence<HardEvent::MTE2_S>();
                    vector.template Assemble<BT / 2>(c, head, active, slot);
                    vector.Publish(slot);
                }
                begin += windows * Window;
                wave += windows;
            }
        }
        // 奇数 wave/未使用版本也消费最后的 free，seed 与每个发布均闭合。
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(6);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(7);
    }
}
} // namespace GdnUnifiedPrepare
#endif
