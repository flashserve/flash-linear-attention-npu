// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. 只复用 Catlass 的搬运/MMAD 原语，整段调度属于统一 Prepare。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_CUBE_H
#define GDN_ARCH22_UNIFIED_PREPARE_CUBE_H

#include <type_traits>
#include "chunk_gated_delta_rule_fwd_prepare_resources.h"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GdnUnifiedPrepare {
template <typename InputT, int V, int BT, int QkvLayout>
class Cube {
public:
    static constexpr uint64_t kFullCBytes = CeilDiv(BT, 16) * 16 *
        CeilDiv(Max(BT, Max(V, 128)), 16) * 16 * sizeof(float);
    static constexpr uint32_t kCSlots = kFullCBytes <= Local::kL0CBytes / 2 ? 2 : 1;
private:
    using Tag = Catlass::Arch::AtlasA2;
    template <class T, class C, bool TransB>
    using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Tag, T, Catlass::layout::RowMajor,
        T, std::conditional_t<TransB, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>,
        C, Catlass::layout::RowMajor>;
    Resources &r_;
    const Args &a_;
    WorkspaceLayout p_;
    GM_ADDR core_;
    uint32_t cSlot_{};

    static constexpr uint32_t kMergeBankOffset = 32 * 1024;
    static constexpr uint32_t kMergeBankBytes = 16 * 1024;
    static_assert(CeilDiv(BT, 16) * 16 * 128 * sizeof(InputT) <= kMergeBankOffset);
    static_assert(64 * 64 * sizeof(float) <= kMergeBankBytes);
    static_assert(Local::kGemmA + kMergeBankOffset + kMergeBankBytes <= Local::kGemmB);
    static_assert(Local::kGemmB + kMergeBankOffset + kMergeBankBytes <= Local::kRhsA);

    template <class T, class C, bool TransB, class LA, class LB>
    __aicore__ inline void MultiplyL1(const LA &left, const LB &right, GlobalTensor<C> output,
                                    uint32_t m, uint32_t n, uint32_t k, uint64_t stride,
                                    uint32_t kStep, bool releaseGeneral, uint32_t bank = 0)
    {
        using TC = Copy<T, C, TransB>;
        typename TC::CopyL1ToL0A loadA;
        typename TC::CopyL1ToL0B loadB;
        Catlass::Gemm::Tile::TileMmadTla<Tag, T, typename TC::LayoutTagL1A> mm;
        const uint32_t mmRows = m == 1 ? 16 : m;
        auto c0 = tla::MakeTensor(r_.L0C(cSlot_), tla::MakeLayoutL0C(mmRows, n), Catlass::Arch::PositionL0C{});
        WaitFlag<HardEvent::FIX_M>(r_.cFree[cSlot_]);
        for (uint32_t ko = 0; ko < k; ko += kStep) {
            const uint32_t kk = static_cast<uint32_t>(Min(kStep, k - ko));
            auto al0 = tla::MakeTensor(r_.template L0A<T>(),
                tla::MakeLayout<T, typename TC::LayoutTagL0A>(mmRows, kk), Catlass::Arch::PositionL0A{});
            auto bl0 = tla::MakeTensor(r_.template L0B<T>(),
                tla::MakeLayout<T, typename TC::LayoutTagL0B>(kk, n), Catlass::Arch::PositionL0B{});
            auto al1 = tla::GetTile(left, tla::MakeCoord(0, ko), tla::MakeShape(mmRows, kk));
            auto bl1 = tla::GetTile(right, tla::MakeCoord(ko, 0), tla::MakeShape(kk, n));
            WaitFlag<HardEvent::M_MTE1>(r_.l0Free);
            loadA(al0, al1);
            loadB(bl0, bl1);
            if (releaseGeneral && ko + kk == k) {
                // 这是两个通用 L1 区间真正的末次 MTE1 读，独立于 GM/C 的生命周期。
                SetFlag<HardEvent::MTE1_MTE2>(r_.GeneralFree(bank, 0));
                SetFlag<HardEvent::MTE1_MTE2>(r_.GeneralFree(bank, 1));
            }
            SetFlag<HardEvent::MTE1_M>(r_.l0Ready);
            WaitFlag<HardEvent::MTE1_M>(r_.l0Ready);
            mm(c0, al0, bl0, mmRows, n, kk, ko == 0,
                kCSlots == 2 ? 0 : (ko + kk == k ? 0b11 : 0b10));
            SetFlag<HardEvent::M_MTE1>(r_.l0Free);
        }
        SetFlag<HardEvent::M_FIX>(r_.cReady[cSlot_]);
        WaitFlag<HardEvent::M_FIX>(r_.cReady[cSlot_]);
        auto cg = tla::MakeTensor(output,
            tla::MakeLayout(tla::MakeShape(m, n), tla::MakeStride(stride, tla::Int<1>{})),
            Catlass::Arch::PositionGM{});
        typename TC::template CopyL0CToGm<decltype(cg)> store;
        store(cg, c0, kCSlots == 2 ? 0 : 0b11);
        SetFlag<HardEvent::FIX_M>(r_.cFree[cSlot_]);
        if constexpr (kCSlots == 2) cSlot_ ^= 1;
    }

    template <class T, class C, bool TransB = false>
    __aicore__ inline void PrefetchGmProduct(GlobalTensor<T> left, GlobalTensor<T> right,
                                   uint32_t m, uint32_t n, uint32_t k,
                                    uint64_t sa, uint64_t sb, uint32_t bank)
    {
        using TC = Copy<T, C, TransB>;
        using LayoutB = std::conditional_t<TransB, Catlass::layout::ColumnMajor, Catlass::layout::RowMajor>;
        auto ag = tla::MakeTensor(left,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(m, k, sa)), Catlass::Arch::PositionGM{});
        auto bg = tla::MakeTensor(right,
            tla::MakeLayoutFromTag(LayoutB(k, n, sb)), Catlass::Arch::PositionGM{});
        const uint32_t mmRows = m == 1 ? 16 : m;
        auto al1 = tla::MakeTensor(r_.template L1<T>(Local::kGemmA + bank * kMergeBankOffset),
            tla::MakeLayout<T, typename TC::LayoutTagL1A>(mmRows, k), Catlass::Arch::PositionL1{});
        auto bl1 = tla::MakeTensor(r_.template L1<T>(Local::kGemmB + bank * kMergeBankOffset),
            tla::MakeLayout<T, typename TC::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
        typename TC::template CopyGmToL1A<decltype(ag)> copyA;
        typename TC::template CopyGmToL1B<decltype(bg)> copyB;
        WaitFlag<HardEvent::MTE1_MTE2>(r_.GeneralFree(bank, 0));
        WaitFlag<HardEvent::MTE1_MTE2>(r_.GeneralFree(bank, 1));
        copyA(al1, ag);
        copyB(bl1, bg);
        SetFlag<HardEvent::MTE2_MTE1>(r_.GeneralReady(bank));
    }

    template <class T, class C, bool TransB = false>
    __aicore__ inline void MultiplyGmLoaded(GlobalTensor<C> output,
                                          uint32_t m, uint32_t n, uint32_t k, uint64_t sc, uint32_t bank)
    {
        using TC = Copy<T, C, TransB>;
        const uint32_t mmRows = m == 1 ? 16 : m;
        auto al1 = tla::MakeTensor(r_.template L1<T>(Local::kGemmA + bank * kMergeBankOffset),
            tla::MakeLayout<T, typename TC::LayoutTagL1A>(mmRows, k), Catlass::Arch::PositionL1{});
        auto bl1 = tla::MakeTensor(r_.template L1<T>(Local::kGemmB + bank * kMergeBankOffset),
            tla::MakeLayout<T, typename TC::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
        MultiplyL1<T, C, TransB>(al1, bl1, output, m, n, k, sc, k, true, bank);
    }

    template <class T, class C, bool TransB = false>
    __aicore__ inline void GmProduct(GlobalTensor<T> left, GlobalTensor<T> right, GlobalTensor<C> output,
                                   uint32_t m, uint32_t n, uint32_t k,
                                   uint64_t sa, uint64_t sb, uint64_t sc)
    {
        PrefetchGmProduct<T, C, TransB>(left, right, m, n, k, sa, sb, 0);
        WaitFlag<HardEvent::MTE2_MTE1>(r_.l1Ready);
        MultiplyGmLoaded<T, C, TransB>(output, m, n, k, sc, 0);
    }

    __aicore__ inline void PrefetchPair16Product(GlobalTensor<float> left0, GlobalTensor<float> left1,
                                          GlobalTensor<float> right0, GlobalTensor<float> right1,
                                           uint64_t sa, uint64_t sb, uint32_t bank)
    {
        using TC = Copy<float, float, false>;
        auto ag0 = tla::MakeTensor(left0,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(16, 16, sa)), Catlass::Arch::PositionGM{});
        auto ag1 = tla::MakeTensor(left1,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(16, 16, sa)), Catlass::Arch::PositionGM{});
        auto bg0 = tla::MakeTensor(right0,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(16, 16, sb)), Catlass::Arch::PositionGM{});
        auto bg1 = tla::MakeTensor(right1,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(16, 16, sb)), Catlass::Arch::PositionGM{});
        // 在完整 zN 布局中按元素坐标拼接；独立 16x16 NZ 数组不能直接竖向相邻摆放。
        auto al1 = tla::MakeTensor(r_.template L1<float>(Local::kGemmA + bank * kMergeBankOffset),
            tla::MakeLayout<float, typename TC::LayoutTagL1A>(32, 16), Catlass::Arch::PositionL1{});
        auto bl1 = tla::MakeTensor(r_.template L1<float>(Local::kGemmB + bank * kMergeBankOffset),
            tla::MakeLayout<float, typename TC::LayoutTagL1B>(16, 32), Catlass::Arch::PositionL1{});
        auto a0 = tla::GetTile(al1, tla::MakeCoord(0, 0), tla::MakeShape(16, 16));
        auto a1 = tla::GetTile(al1, tla::MakeCoord(16, 0), tla::MakeShape(16, 16));
        auto b0 = tla::GetTile(bl1, tla::MakeCoord(0, 0), tla::MakeShape(16, 16));
        auto b1 = tla::GetTile(bl1, tla::MakeCoord(0, 16), tla::MakeShape(16, 16));
        typename TC::template CopyGmToL1A<decltype(ag0)> copyA;
        typename TC::template CopyGmToL1B<decltype(bg0)> copyB;
        WaitFlag<HardEvent::MTE1_MTE2>(r_.GeneralFree(bank, 0));
        WaitFlag<HardEvent::MTE1_MTE2>(r_.GeneralFree(bank, 1));
        copyA(a0, ag0);
        copyA(a1, ag1);
        copyB(b0, bg0);
        copyB(b1, bg1);
        SetFlag<HardEvent::MTE2_MTE1>(r_.GeneralReady(bank));
    }

    __aicore__ inline GM_ADDR Head(uint32_t slot, uint32_t localHead) const
    { return core_ + p_.slotsOffset + slot * p_.slotBytes + localHead * p_.headBytes; }

    struct MergeProduct { uint32_t head, row, pair, n0, n1; };
    struct MergeCursor { uint32_t head{}, row{}, pair{}; };

    template <int S>
    __aicore__ inline bool NextMergeProduct(const Chunk &c, uint32_t activeHeads,
                                           MergeCursor &cursor, MergeProduct &product) const
    {
        if constexpr (BT == 64 && S == 16) {
            if (c.valid == BT) {
                if (cursor.head == activeHeads) return false;
                product = {cursor.head++, 0, 0, 16, 16};
                return true;
            }
        }
        while (cursor.head < activeHeads) {
            if (cursor.row >= c.valid) {
                ++cursor.head;
                cursor.row = cursor.pair = 0;
                continue;
            }
            const uint32_t row = cursor.row, pair = cursor.pair;
            cursor.row += 2 * S;
            ++cursor.pair;
            const uint32_t n0 = static_cast<uint32_t>(Min(S, c.valid - row));
            const uint32_t n1 = c.valid > row + S ? static_cast<uint32_t>(Min(S, c.valid - row - S)) : 0;
            if (n1 == 0) continue;
            product = {cursor.head, row, pair, n0, n1};
            return true;
        }
        return false;
    }

    template <int S, bool Second>
    __aicore__ inline void PrefetchMergeProduct(const Chunk &c, uint32_t slot,
                                               const MergeProduct &product, uint32_t bank)
    {
        GM_ADDR head = Head(slot, product.head);
        auto x = Gm<float>(head + p_.xOffset, BT * BT);
        auto diag = Gm<float>(head + (S == 32 ? p_.d1Offset : p_.d0Offset), BT * S);
        if constexpr (BT == 64 && S == 16) {
            if (c.valid == BT) {
                if constexpr (Second) {
                    PrefetchPair16Product(x[32], x[16 * BT + 48], diag[0], diag[32 * S], BT, S, bank);
                } else {
                    PrefetchPair16Product(diag[16 * S], diag[48 * S], x[16 * BT], x[48 * BT + 32], S, BT, bank);
                }
                return;
            }
        }
        const uint32_t row = product.row;
        if constexpr (Second) {
            PrefetchGmProduct<float, float>(x[(row + S) * BT + row], diag[row * S],
                product.n1, product.n0, product.n0, BT, S, bank);
        } else {
            PrefetchGmProduct<float, float>(diag[(row + S) * S], x[(row + S) * BT + row],
                product.n1, product.n0, product.n1, S, BT, bank);
        }
    }

    template <int S, bool Second>
    __aicore__ inline void ConsumeMergeProduct(const Chunk &c, uint32_t slot,
                                              const MergeProduct &product, uint32_t bank)
    {
        GM_ADDR head = Head(slot, product.head);
        auto x = Gm<float>(head + p_.xOffset, BT * BT);
        auto result = Gm<float>(head + p_.resultOffset, BT * BT / 4);
        if constexpr (BT == 64 && S == 16) {
            if (c.valid == BT) {
                // 保持完整父 zN 布局、各16项归约和首乘 FP32 GM 重载边界。
                MultiplyGmLoaded<float, float>(Second ? result : x[32],
                    32, 32, 16, Second ? 2 * S : BT, bank);
                return;
            }
        }
        MultiplyGmLoaded<float, float>(Second ? result[product.pair * S * S] : x[(product.row + S) * BT + product.row],
            product.n1, product.n0, Second ? product.n0 : product.n1, Second ? S : BT, bank);
    }

    template <int S, bool Second>
    __aicore__ inline void MergeProducts(const Chunk &c, uint32_t slot, uint32_t activeHeads)
    {
        MergeCursor cursor{};
        MergeProduct current{}, next{};
        if (!NextMergeProduct<S>(c, activeHeads, cursor, current)) return;
        uint32_t bank = 0;
        PrefetchMergeProduct<S, Second>(c, slot, current, bank);
        while (true) {
            WaitFlag<HardEvent::MTE2_MTE1>(r_.GeneralReady(bank));
            const bool hasNext = NextMergeProduct<S>(c, activeHeads, cursor, next);
            // 当前 ready 的 Wait 已发出，下一异址装载不挡住当前产品的 MTE1 读。
            if (hasNext) PrefetchMergeProduct<S, Second>(c, slot, next, bank ^ 1);
            ConsumeMergeProduct<S, Second>(c, slot, current, bank);
            if (!hasNext) break;
            current = next;
            bank ^= 1;
        }
    }

    template <class T, bool IsA>
    __aicore__ inline void ResidentCopy(GlobalTensor<T> input, uint32_t offset,
                                       uint32_t m, uint32_t n, uint64_t stride)
    {
        using TC = Copy<T, T, false>;
        auto gm = tla::MakeTensor(input,
            tla::MakeLayoutFromTag(Catlass::layout::RowMajor(m, n, stride)), Catlass::Arch::PositionGM{});
        if constexpr (IsA) {
            auto l1 = tla::MakeTensor(r_.template L1<T>(offset),
                tla::MakeLayout<T, typename TC::LayoutTagL1A>(m == 1 ? 16 : m, n), Catlass::Arch::PositionL1{});
            typename TC::template CopyGmToL1A<decltype(gm)> copy;
            copy(l1, gm);
        } else {
            auto l1 = tla::MakeTensor(r_.template L1<T>(offset),
                tla::MakeLayout<T, typename TC::LayoutTagL1B>(m, n), Catlass::Arch::PositionL1{});
            typename TC::template CopyGmToL1B<decltype(gm)> copy;
            copy(l1, gm);
        }
    }

public:
    __aicore__ inline Cube(Resources &r, const Args &args, WorkspaceLayout p)
        : r_(r), a_(args), p_(p), core_(args.scratch + GroupId() * p.coreBytes) {}

    __aicore__ inline void Scores(const Chunk &c)
    {
        auto kg = Gm<InputT>(a_.k, a_.batch * a_.hk * a_.tokens * 128);
        auto scores = Gm<float>(core_, p_.scoreBytes / sizeof(float));
        for (uint64_t hk = c.hkBegin; hk < c.hkEnd; ++hk) {
            const uint64_t offset = InputOffset<QkvLayout>(a_, c, hk, a_.hk, 128);
            const uint64_t stride = QkvLayout == 1 ? a_.hk * 128 : 128;
            GmProduct<InputT, float, true>(kg[offset], kg[offset], scores[(hk - c.hkBegin) * BT * BT],
                c.valid, c.valid, 128, stride, stride, BT);
        }
        // raw score 仅在下一个 HK task 开始时覆盖；本 task 的所有 G>pack waves 只读它。
    }

    template <int S>
    __aicore__ inline void Merge(const Chunk &c, uint32_t slot, uint32_t activeHeads)
    {
        static_assert(S == 16 || S == 32 || S == 64);
        // 整包 AIV 写已由入口 ready 关闭；ENTIRE_DATA_CACHE 覆盖所有 head，
        // 无需在互不重叠的 head 记录之间重复清理和 DDR 屏障。
        Invalidate(Gm<float>(Head(slot, 0), p_.slotBytes / sizeof(float)));
        MergeProducts<S, false>(c, slot, activeHeads);
        // 整包首乘已发出后统一交接；不在每个子块之间做 first→second 串行等待。
        r_.template Fence<HardEvent::FIX_MTE2>();
        r_.template Fence<HardEvent::MTE2_S>(); // FIX→MTE2→Scalar，之后的 DCCI 才可能执行。
        // 首乘的新 tempX 全部完成后，整包次乘统一建立读侧可见性。
        Invalidate(Gm<float>(Head(slot, 0), p_.slotBytes / sizeof(float)));
        MergeProducts<S, true>(c, slot, activeHeads);
    }

    __aicore__ inline void WU(const Chunk &c, uint64_t headBegin, uint32_t activeHeads, uint32_t slot)
    {
        using TC = Copy<InputT, InputT, false>;
        auto ag = Gm<InputT>(a_.a, a_.batch * a_.hv * a_.tokens * BT);
        auto wg = Gm<InputT>(a_.w, a_.batch * a_.hv * a_.tokens * 128);
        auto ug = Gm<InputT>(a_.u, a_.batch * a_.hv * a_.tokens * V);
        const uint32_t valid = static_cast<uint32_t>(c.valid);
        // 按完整 BT 包络判断，不能因 BT128 尾块较短而越过原 L1 分区。
        constexpr bool fitsFourHeads =
            4ULL * BT * BT * sizeof(InputT) <= Local::kRhsK - Local::kRhsA &&
            4ULL * BT * 128 * sizeof(InputT) <= Local::kRhsV - Local::kRhsK &&
            4ULL * BT * V * sizeof(InputT) <= Local::kL1Bytes - Local::kRhsV;
        const bool prefetchSecond = fitsFourHeads && activeHeads > 2 && activeHeads <= 4;
        // A/RHS 已整包发布，整数据缓存清理同时覆盖公开 A 和私有 RHS。
        // GM slot 在最后一对 head 搬入完成前仍由本消费者持有。
        Invalidate(Gm<InputT>(Head(slot, 0), p_.slotBytes / sizeof(InputT)));
        for (uint32_t first = 0; first < activeHeads; first += 2) {
            const uint32_t count = static_cast<uint32_t>(Min(2, activeHeads - first));
            // 此许可属于 L1 lane；GM slot 的许可只在下面整包最后 pair 的读取后归还。
            WaitFlag<HardEvent::MTE1_MTE2>(r_.rhsFree);
            if (!prefetchSecond || first == 0) {
                for (uint32_t pair = 0; pair < count; ++pair) {
                    const uint32_t h = first + pair;
                    const uint64_t globalHead = headBegin + h;
                    auto rhs = Gm<InputT>(Head(slot, h), p_.headBytes / sizeof(InputT));
                    ResidentCopy<InputT, true>(ag[HeadOffset(a_, c, globalHead, BT)],
                        Local::kRhsA + pair * BT * BT * sizeof(InputT), valid, valid, BT);
                    ResidentCopy<InputT, false>(rhs[p_.kbgOffset / sizeof(InputT)],
                        Local::kRhsK + pair * BT * 128 * sizeof(InputT), valid, 128, 128);
                    ResidentCopy<InputT, false>(rhs[p_.vbOffset / sizeof(InputT)],
                        Local::kRhsV + pair * BT * V * sizeof(InputT), valid, V, V);
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(r_.l1Ready);
            if (first + count == activeHeads) {
                // 整包的全部 A/Kbg/Vb GM 读取已发出；从 MTE2 发布保证最后一个读已完成。
                // 此后最后 pair 的 MMAD/FIX 仍可执行，下一 pack 可重用另一个/此 GM 版本。
                CrossCoreSetFlag<0x2, PIPE_MTE2>(6 + slot);
            }
            WaitFlag<HardEvent::MTE2_MTE1>(r_.l1Ready);
            if (prefetchSecond && first == 0) {
                // 已发出 ready0 等待；异址 DMA 可先行，ready1 仍受下一轮 rhsFree Wait 约束。
                for (uint32_t h = 2; h < activeHeads; ++h) {
                    auto rhs = Gm<InputT>(Head(slot, h), p_.headBytes / sizeof(InputT));
                    ResidentCopy<InputT, true>(ag[HeadOffset(a_, c, headBegin + h, BT)],
                        Local::kRhsA + h * BT * BT * sizeof(InputT), valid, valid, BT);
                    ResidentCopy<InputT, false>(rhs[p_.kbgOffset / sizeof(InputT)],
                        Local::kRhsK + h * BT * 128 * sizeof(InputT), valid, 128, 128);
                    ResidentCopy<InputT, false>(rhs[p_.vbOffset / sizeof(InputT)],
                        Local::kRhsV + h * BT * V * sizeof(InputT), valid, V, V);
                }
            }
            for (uint32_t pair = 0; pair < count; ++pair) {
                const uint64_t globalHead = headBegin + first + pair;
                const uint32_t resident = prefetchSecond ? first + pair : pair;
                const uint32_t mmRows = valid == 1 ? 16 : valid;
                auto left = tla::MakeTensor(
                    r_.template L1<InputT>(Local::kRhsA + resident * BT * BT * sizeof(InputT)),
                    tla::MakeLayout<InputT, typename TC::LayoutTagL1A>(mmRows, valid), Catlass::Arch::PositionL1{});
                auto vb = tla::MakeTensor(
                    r_.template L1<InputT>(Local::kRhsV + resident * BT * V * sizeof(InputT)),
                    tla::MakeLayout<InputT, typename TC::LayoutTagL1B>(valid, V), Catlass::Arch::PositionL1{});
                auto kbg = tla::MakeTensor(
                    r_.template L1<InputT>(Local::kRhsK + resident * BT * 128 * sizeof(InputT)),
                    tla::MakeLayout<InputT, typename TC::LayoutTagL1B>(valid, 128), Catlass::Arch::PositionL1{});
                // 原 V256 路径的累加边界为 64，原 V128 路径为 128；W 与 U 使用同一边界。
                MultiplyL1<InputT, InputT, false>(left, vb, ug[HeadOffset(a_, c, globalHead, V)],
                    valid, V, valid, V, V == 256 ? 64 : 128, false);
                MultiplyL1<InputT, InputT, false>(left, kbg, wg[HeadOffset(a_, c, globalHead, 128)],
                    valid, 128, valid, 128, V == 256 ? 64 : 128, false);
            }
            SetFlag<HardEvent::MTE1_MTE2>(r_.rhsFree);
        }
    }
};
} // namespace GdnUnifiedPrepare
#endif
