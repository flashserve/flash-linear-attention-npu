#ifndef GDN_ARCH22_FP32_SOLVE_PIPELINE_H
#define GDN_ARCH22_FP32_SOLVE_PIPELINE_H
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. DAV_2201 分层 FP32 Solve 的私有迁移组件。
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 220
#error "This private Solve pipeline is only compiled for DAV_2201"
#endif
#include "kernel_operator.h"
#include "../../common/gdn_raw_front_resources.h"
#include "../../common/front_task_order.h"
#include "solve_tri_fp32.h"
namespace GdnFp32Solve {
using namespace AscendC;
using namespace Catlass;


struct FullProblem {
    int64_t batch, tokens, heads, bt, headFirst, sequences;
    int64_t tasks32, tasks64, tasks128;
    // 仅 RunOwnedBatch 使用：物理组内的完整 parent 子区间，GM 全局形状不变。
    int64_t localParentBegin = 0;
    int64_t localParentCount = -1;
    // Parent=true 仅由 A2 Front 调用；GM layout 和原分核范围不变。
    int64_t frontHeadsPerKey = 1;
    GM_ADDR frontChunkIndices = nullptr;
};
struct WorkItem {
    int64_t b, h, t, base, length;
};

template <int kStaticBT = 0>
__aicore__ inline int64_t ChunkSize(FullProblem p)
{
    if constexpr (kStaticBT != 0) {
        return static_cast<int64_t>(kStaticBT);
    } else {
        return p.bt;
    }
}

// 同一 AIC 与两个 paired AIV 使用相同的 CG=1 grid-stride 所有权。
// 传入的是本地任务序号；ring/leaf 版本也按本地序号推进。
__aicore__ inline int64_t CoreGroup()
{
    int64_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= GetSubBlockNum();
    }
    return core;
}

// Parent 策略：本核组按连续块持有的 parent（最终层任务）数，可为 0。
template <int kStaticBT = 0>
__aicore__ inline int64_t ParentOwnedCount(FullProblem p)
{
    if (p.localParentCount >= 0) {
        return p.localParentCount;
    }
    int64_t n = ChunkSize<kStaticBT>(p) == 64 ? p.tasks64 : p.tasks128;
    int64_t per = (n + GetBlockNum() - 1) / GetBlockNum();
    int64_t rest = n - CoreGroup() * per;
    return per < rest ? per : (rest > 0 ? rest : 0);
}

// Parent 策略下本核组在 span 层的本地任务数；非 Parent 保持原 q/rem 路径。
template <bool Parent, int kStaticBT = 0>
__aicore__ inline int64_t SpanTaskEnd(FullProblem p, int span, int64_t tasks, int64_t workers, int64_t core)
{
    if constexpr (Parent) {
        return ParentOwnedCount<kStaticBT>(p) * (ChunkSize<kStaticBT>(p) / span);
    } else {
        return tasks / workers + (core < tasks % workers ? 1 : 0);
    }
}

// 默认 false 走原 grid-stride 定位；Parent=true 时本地任务按同 parent 连续块映射，
// 仅 RunOwnedBatch 的 Front 实例使用同一私有 helper，不增加运行时顺序选择。
template <bool Parent = false, int kStaticBT = 0>
__aicore__ inline WorkItem locate(int64_t task, int span, FullProblem p, GlobalTensor<int64_t> cu)
{
    if constexpr (Parent) {
        int64_t n = ChunkSize<kStaticBT>(p) == 64 ? p.tasks64 : p.tasks128;
        int64_t subs = ChunkSize<kStaticBT>(p) / span;
        if (n <= 0 || task < 0 || task >= ParentOwnedCount<kStaticBT>(p) * subs) {
            return {0, 0, 0, 0, 0};
        }
        int64_t per = (n + GetBlockNum() - 1) / GetBlockNum();
        int64_t parent = CoreGroup() * per + p.localParentBegin + task / subs;
        int64_t subtile = task % subs;
        int64_t nt = n / (p.batch * p.heads);
        const auto coord = GdnFrontTaskOrder::Decode(parent, p.heads, p.frontHeadsPerKey, nt);
        if (p.sequences == 0) {
            return {coord.batch, coord.head, coord.chunk * ChunkSize<kStaticBT>(p) + subtile * span,
                    0, p.tokens};
        }
        GlobalTensor<int64_t> indices;
        indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(p.frontChunkIndices));
        const int64_t seq = indices.GetValue(coord.chunk * 2);
        const int64_t chunk = indices.GetValue(coord.chunk * 2 + 1);
        const int64_t base = cu.GetValue(seq);
        return {0, coord.head, chunk * ChunkSize<kStaticBT>(p) + subtile * span,
                base, cu.GetValue(seq + 1) - base};
    }
    task = CoreGroup() + task * GetBlockNum();
    if (p.sequences == 0) {
        int64_t bh = task % (p.batch * p.heads);
        return {bh / p.heads, bh % p.heads, task / (p.batch * p.heads) * span, 0, p.tokens};
    }
    for (int64_t seq = 0; seq < p.sequences; ++seq) {
        int64_t base = cu.GetValue(seq), length = cu.GetValue(seq + 1) - base;
        int64_t count = ((length + span - 1) / span) * p.heads;
        if (task < count) {
            return {0, task % p.heads, task / p.heads * span, base, length};
        }
        task -= count;
    }
    return {0, 0, 0, 0, 0};
}
__aicore__ inline int64_t pos(FullProblem p, WorkItem w, int64_t row, int width)
{
    int64_t t = w.base + row;
    return (p.headFirst ? (w.b * p.heads + w.h) * p.tokens + t : (w.b * p.tokens + t) * p.heads + w.h) * width;
}
__aicore__ inline int64_t stride(FullProblem p, int width)
{
    return (p.headFirst ? 1 : p.heads) * width;
}

template <class MM>
__aicore__ inline void full_gemm(MM &mm, GlobalTensor<float> a, GlobalTensor<float> b, GlobalTensor<float> c, int m,
                                 int n, int k, int64_t sa, int64_t sb, int64_t sc)
{
    auto ta = tla::MakeTensor(a, tla::MakeLayout(tla::MakeShape(m, k), tla::MakeStride(sa, tla::Int<1>{})),
                              Arch::PositionGM{});
    auto tb = tla::MakeTensor(b, tla::MakeLayout(tla::MakeShape(k, n), tla::MakeStride(sb, tla::Int<1>{})),
                              Arch::PositionGM{});
    auto tc = tla::MakeTensor(c, tla::MakeLayout(tla::MakeShape(m, n), tla::MakeStride(sc, tla::Int<1>{})),
                              Arch::PositionGM{});
    mm(ta, tb, tc, GemmCoord{uint32_t(m), uint32_t(n), uint32_t(k)});
}

template <int S, class Out, int Group = 16, bool Parent = false, bool Raw = false, int kStaticBT = 0>
__aicore__ inline void pipeline_merge(GdnRawFront::Resource &resource,
                                      GM_ADDR input, GM_ADDR prev, GM_ADDR output, GM_ADDR workspace, GM_ADDR cuAddr,
                                      FullProblem info, int64_t tasks, GdnRawFront::VectorResources *resources)
{
    static_assert(S == 64, "pipeline_merge only supports S=64");
    constexpr int W = S > 32 ? S : 32;
    constexpr int E = W * W;
    constexpr int BATCH = GDN::FP32_SOLVE_MERGE_BATCH_SIZE;
    constexpr int TEMP = S == 16 ? GDN::FP32_SOLVE_SMALL_TEMP_SLOT_COUNT
                                 : GDN::FP32_SOLVE_LARGE_TEMP_SLOT_COUNT;
    static_assert(Group > 0 && Group <= 16 && 16 % Group == 0);
    int64_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= GetSubBlockNum();
    }
    int64_t workers = GetBlockNum();
    int64_t begin = 0, end = SpanTaskEnd<Parent, kStaticBT>(info, S * 2, tasks, workers, core);
    GlobalTensor<float> x, d, ws, batchWs;
    GlobalTensor<Out> y;
    GlobalTensor<int64_t> cu;
    x.SetGlobalBuffer((__gm__ float *)input);
    d.SetGlobalBuffer((__gm__ float *)prev);
    // 同组的所有层复用同一个 arena；slot 是组内下标，组间区间恒不相交。
    const int64_t arenaElements = ChunkSize<kStaticBT>(info) == 128
        ? GDN::FP32_SOLVE_ARENA128_ELEMENTS : GDN::FP32_SOLVE_ARENA64_ELEMENTS;
    ws.SetGlobalBuffer((__gm__ float *)workspace + core * arenaElements);
    batchWs.SetGlobalBuffer((__gm__ float *)workspace + core * arenaElements +
                           GDN::FP32_SOLVE_TEMP_ELEMENTS);
    static_assert(TEMP * E <= GDN::FP32_SOLVE_TEMP_ELEMENTS);
    y.SetGlobalBuffer((__gm__ Out *)output);
    cu.SetGlobalBuffer((__gm__ int64_t *)cuAddr);
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        using Tag = Arch::AtlasA2;
        using Policy = Gemm::MmadPingpong<Tag, true, false>;
        using Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
        using Copy = NsSolveTri::SolveTriTileCopy<Tag>;
        using MM = Gemm::Block::BlockMmadTla<Policy, Shape, Shape, float, float, float, void, Copy>;
        MM mm(resource);
        for (int64_t task = begin; task < end; ++task) {
            int64_t index = task - begin, slot = index % (BATCH * 2);
            if (index % BATCH == 0 && index >= 2 * BATCH)
                CrossCoreWaitFlag(2 + (index / BATCH) % 2);
            auto w = locate<Parent, kStaticBT>(task, S * 2, info, cu);
            int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
            if (n1 > 0) {
                auto d0 = d[pos(info, w, w.t, S)], d1 = d[pos(info, w, w.t + S, S)];
                auto off = x[pos(info, w, w.t + S, ChunkSize<kStaticBT>(info)) + w.t % ChunkSize<kStaticBT>(info)];
                full_gemm(mm, d1, off, ws, n1, n0, n1, stride(info, S), stride(info, ChunkSize<kStaticBT>(info)), W);
                SetFlag<HardEvent::FIX_MTE2>(0);
                WaitFlag<HardEvent::FIX_MTE2>(0);
                full_gemm(mm, ws, d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
            }
            if ((index + 1) % BATCH == 0 || task + 1 == end)
                CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
        }
        int64_t batches = (end - begin + BATCH - 1) / BATCH;
        for (int64_t batch = batches > 2 ? batches - 2 : 0; batch < batches; ++batch)
            CrossCoreWaitFlag(2 + batch % 2);
    }
    if ASCEND_IS_AIV {
        GdnRawFront::SolveStage<Raw> pipe(resources);
        GdnRawFront::Buffer<Raw> fb, ob;
        pipe.BindBuffer(fb, S * 2 * S * 4);
        if constexpr (sizeof(Out) != 4)
            pipe.BindBuffer(ob, S * 2 * S * sizeof(Out));
        auto f = fb.template Get<float>();
        LocalTensor<Out> o;
        if constexpr (sizeof(Out) == 4)
            o = f;
        else
            o = ob.template Get<Out>();
        int sub = GetSubBlockIdx();
        for (int64_t task = begin; task < end; ++task) {
            int64_t index = task - begin, slot = index % (BATCH * 2);
            if (index % BATCH == 0)
                CrossCoreWaitFlag(4 + (index / BATCH) % 2);
            auto w = locate<Parent, kStaticBT>(task, S * 2, info, cu);
            int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
            int rows = sub == 0 ? n0 : n1;
            Duplicate(f, 0.0f, S * 2 * S);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources));
            WaitFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources));
            if (rows > 0) {
                int rounded = (rows + 7) / 8 * 8;
                DataCopyPadExtParams<float> pad{true, 0, (uint8_t)(rounded - rows), 0};
                DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(rows * 4), (uint32_t)((stride(info, S) - rows) * 4),
                                     (uint32_t)((2 * S - rounded) / 8), 0};
                DataCopyPad(f[sub * S], d[pos(info, w, w.t + sub * S, S)], cp, pad);
                if (sub == 1) {
                    DataCopyExtParams cross{(uint16_t)rows, (uint32_t)(n0 * 4), (uint32_t)((W - n0) * 4),
                                            (uint32_t)((2 * S - n0) / 8), 0};
                    DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
                    DataCopyPad(f, batchWs[slot * E], cross, noPad);
                }
            }
            SetFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources));
            WaitFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources));
            if (sub == 1 && rows > 0) {
                Muls(f, f, -1.0f, S, (uint8_t)rows, {1, 1, (uint8_t)(2 * S / 8), (uint8_t)(2 * S / 8)});
                PipeBarrier<PIPE_V>();
            }
            if constexpr (sizeof(Out) != 4)
                Cast(o, f, RoundMode::CAST_RINT, S * 2 * S);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources));
            WaitFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources));
            if (rows > 0) {
                DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(2 * S * sizeof(Out)), 0,
                                     (uint32_t)((stride(info, 2 * S) - 2 * S) * sizeof(Out)), 0};
                DataCopyPad(y[pos(info, w, w.t + sub * S, 2 * S)], o, cp);
            }
            SetFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources));
            WaitFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources));
            if ((index + 1) % BATCH == 0 || task + 1 == end)
                CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + (index / BATCH) % 2);
        }
    }
}

template <int S, class Out, int Group = 8, bool Parent = false, bool Raw = false, int kStaticBT = 0>
__aicore__ inline void stage64_merge(GdnRawFront::Resource &resource,
                                     GM_ADDR input, GM_ADDR prev, GM_ADDR output, GM_ADDR workspace, GM_ADDR cuAddr,
                                     FullProblem info, int64_t tasks, GdnRawFront::VectorResources *resources)
{
    static_assert(S == 32, "stage64_merge only supports S=32");
    constexpr int W = S > 32 ? S : 32;
    constexpr int E = W * W;
    constexpr int BATCH = GDN::FP32_SOLVE_MERGE_BATCH_SIZE;
    constexpr int TEMP = S == 32 ? GDN::FP32_SOLVE_SMALL_TEMP_SLOT_COUNT
                                 : GDN::FP32_SOLVE_LARGE_TEMP_SLOT_COUNT;
    static_assert(Group > 0 && Group <= 16 && 16 % Group == 0);
    int64_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= GetSubBlockNum();
    }
    int64_t workers = GetBlockNum();
    int64_t begin = 0, end = SpanTaskEnd<Parent, kStaticBT>(info, S * 2, tasks, workers, core);
    GlobalTensor<float> x, d, ws, batchWs;
    GlobalTensor<Out> y;
    GlobalTensor<int64_t> cu;
    x.SetGlobalBuffer((__gm__ float *)input);
    d.SetGlobalBuffer((__gm__ float *)prev);
    // 同组的所有层复用同一个 arena；slot 是组内下标，组间区间恒不相交。
    const int64_t arenaElements = ChunkSize<kStaticBT>(info) == 128
        ? GDN::FP32_SOLVE_ARENA128_ELEMENTS : GDN::FP32_SOLVE_ARENA64_ELEMENTS;
    ws.SetGlobalBuffer((__gm__ float *)workspace + core * arenaElements);
    batchWs.SetGlobalBuffer((__gm__ float *)workspace + core * arenaElements +
                           GDN::FP32_SOLVE_TEMP_ELEMENTS);
    static_assert(TEMP * E <= GDN::FP32_SOLVE_TEMP_ELEMENTS);
    y.SetGlobalBuffer((__gm__ Out *)output);
    cu.SetGlobalBuffer((__gm__ int64_t *)cuAddr);
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        using Tag = Arch::AtlasA2;
        using Policy = Gemm::MmadPingpong<Tag, true, false>;
        using Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
        using Copy = NsSolveTri::SolveTriTileCopy<Tag>;
        using MM = Gemm::Block::BlockMmadTla<Policy, Shape, Shape, float, float, float, void, Copy>;
        MM mm(resource);
        for (int64_t task = begin; task < end; task += Group) {
            int64_t index = task - begin;
            if (index % BATCH == 0 && index >= 2 * BATCH)
                CrossCoreWaitFlag(2 + (index / BATCH) % 2);
            int count = end - task < Group ? int(end - task) : Group;
            WorkItem jobs[Group];
            int rows0[Group], rows1[Group];
            for (int j = 0; j < count; ++j) {
                auto w = locate<Parent, kStaticBT>(task + j, S * 2, info, cu);
                jobs[j] = w;
                int n0 = w.length - w.t < S ? w.length - w.t : S;
                int n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                rows0[j] = n0;
                rows1[j] = n1;
                if (n1 > 0) {
                    auto d1 = d[pos(info, w, w.t + S, S)];
                    auto off = x[pos(info, w, w.t + S, ChunkSize<kStaticBT>(info)) + w.t % ChunkSize<kStaticBT>(info)];
                    full_gemm(mm, d1, off, ws[j * E], n1, n0, n1, stride(info, S), stride(info, ChunkSize<kStaticBT>(info)), W);
                }
            }

            SetFlag<HardEvent::FIX_MTE2>(0);
            WaitFlag<HardEvent::FIX_MTE2>(0);
            for (int j = 0; j < count; ++j) {
                auto w = jobs[j];
                int n0 = rows0[j], n1 = rows1[j];
                if (n1 > 0) {
                    int64_t slot = (index + j) % (BATCH * 2);
                    auto d0 = d[pos(info, w, w.t, S)];
                    full_gemm(mm, ws[j * E], d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                }
            }

            // 第二遍 MTE2 读完 ws 后再允许下一组 FIX 覆盖；结果仍经 PIPE_FIX 发布。
            SetFlag<HardEvent::MTE2_FIX>(0);
            WaitFlag<HardEvent::MTE2_FIX>(0);
            if ((index + count) % BATCH == 0 || task + count == end)
                CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
        }
        int64_t batches = (end - begin + BATCH - 1) / BATCH;
        for (int64_t batch = batches > 2 ? batches - 2 : 0; batch < batches; ++batch)
            CrossCoreWaitFlag(2 + batch % 2);
    }
    if ASCEND_IS_AIV {
        GdnRawFront::SolveStage<Raw> pipe(resources);
        GdnRawFront::Buffer<Raw> fb, ob;
        // 成对批处理：单轮 V/MTE2/MTE3 协议服务相邻两任务，flag 对数量减半；
        // BATCH=16 与步长2对齐，批边界恒落在对边界，不跨批。
        pipe.BindBuffer(fb, 2 * S * 2 * S * 4);
        if constexpr (sizeof(Out) != 4)
            pipe.BindBuffer(ob, 2 * S * 2 * S * sizeof(Out));
        auto f = fb.template Get<float>();
        LocalTensor<Out> o;
        if constexpr (sizeof(Out) == 4)
            o = f;
        else
            o = ob.template Get<Out>();
        int sub = GetSubBlockIdx();
        constexpr int64_t taskElems = S * 2 * S;
        for (int64_t task = begin; task < end; task += 2) {
            const int count = end - task >= 2 ? 2 : 1;
            WorkItem jobs[2];
            int rowsArr[2], n0Arr[2];
            int64_t slots[2];
            for (int j = 0; j < count; ++j) {
                int64_t index = task + j - begin;
                if (index % BATCH == 0)
                    CrossCoreWaitFlag(4 + (index / BATCH) % 2);
                auto w = locate<Parent, kStaticBT>(task + j, S * 2, info, cu);
                jobs[j] = w;
                int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                n0Arr[j] = n0;
                rowsArr[j] = sub == 0 ? n0 : n1;
                slots[j] = index % (BATCH * 2);
                Duplicate(f[j * taskElems], 0.0f, S * 2 * S);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources));
            WaitFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources));
            for (int j = 0; j < count; ++j) {
                auto w = jobs[j];
                int rows = rowsArr[j], n0 = n0Arr[j];
                if (rows > 0) {
                    int rounded = (rows + 7) / 8 * 8;
                    DataCopyPadExtParams<float> pad{true, 0, (uint8_t)(rounded - rows), 0};
                    DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(rows * 4), (uint32_t)((stride(info, S) - rows) * 4),
                                         (uint32_t)((2 * S - rounded) / 8), 0};
                    DataCopyPad(f[j * taskElems + sub * S], d[pos(info, w, w.t + sub * S, S)], cp, pad);
                    if (sub == 1) {
                        DataCopyExtParams cross{(uint16_t)rows, (uint32_t)(n0 * 4), (uint32_t)((W - n0) * 4),
                                                (uint32_t)((2 * S - n0) / 8), 0};
                        DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
                        DataCopyPad(f[j * taskElems], batchWs[slots[j] * E], cross, noPad);
                    }
                }
            }
            SetFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources));
            WaitFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources));
            for (int j = 0; j < count; ++j) {
                if (sub == 1 && rowsArr[j] > 0) {
                    Muls(f[j * taskElems], f[j * taskElems], -1.0f, S, (uint8_t)rowsArr[j],
                         {1, 1, (uint8_t)(2 * S / 8), (uint8_t)(2 * S / 8)});
                    PipeBarrier<PIPE_V>();
                }
            }
            if constexpr (sizeof(Out) != 4) {
                for (int j = 0; j < count; ++j)
                    Cast(o[j * taskElems], f[j * taskElems], RoundMode::CAST_RINT, S * 2 * S);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources));
            WaitFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources));
            for (int j = 0; j < count; ++j) {
                auto w = jobs[j];
                if (rowsArr[j] > 0) {
                    DataCopyExtParams cp{(uint16_t)rowsArr[j], (uint32_t)(2 * S * sizeof(Out)), 0,
                                         (uint32_t)((stride(info, 2 * S) - 2 * S) * sizeof(Out)), 0};
                    DataCopyPad(y[pos(info, w, w.t + sub * S, 2 * S)], o[j * taskElems], cp);
                }
            }
            SetFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources));
            WaitFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources));
            for (int j = 0; j < count; ++j) {
                int64_t index = task + j - begin;
                if ((index + 1) % BATCH == 0 || task + j + 1 == end)
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + (index / BATCH) % 2);
            }
        }
    }
}


template <int L, bool Parent = false, bool Raw = false, int kStaticBT = 0>
class LeafProducer {
    static constexpr int E = L * 256;
    GdnRawFront::Buffer<Raw> ab, pb, rb;
    GdnRawFront::VectorResources *resources_ = nullptr;
    LocalTensor<float> a, prod, row;
    GlobalTensor<float> in, out;
    GlobalTensor<int64_t> cu;
    FullProblem info;
    int sub;

public:
    __aicore__ inline void Init(GdnRawFront::SolveStage<Raw> &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR cuAddr, FullProblem p, int s, GdnRawFront::VectorResources *resources)
    {
        resources_ = resources;
        info = p;
        sub = s;
        pipe.BindBuffer(ab, E * 4);
        pipe.BindBuffer(pb, E * 4);
        pipe.BindBuffer(rb, 2 * L * 16 * 4);
        a = ab.template Get<float>();
        prod = pb.template Get<float>();
        row = rb.template Get<float>();
        in.SetGlobalBuffer((__gm__ float *)input);
        out.SetGlobalBuffer((__gm__ float *)output);
        cu.SetGlobalBuffer((__gm__ int64_t *)cuAddr);
    }
    __aicore__ inline void Produce(int64_t begin, int64_t end, int64_t version)
    {
        int count = end - begin < L ? int(end - begin) : L;
        WorkItem jobs[L];
        int rows[L];
        Duplicate(a, 0.0f, E);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources_));
        WaitFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources_));
        for (int leaf = 0; leaf < count; ++leaf) {
            auto w = locate<Parent, kStaticBT>(begin + leaf, 32, info, cu);
            jobs[leaf] = w;
            int64_t t = w.t + sub * 16;
            int n = w.length - t < 16 ? int(w.length - t) : 16;
            rows[leaf] = n;
            if (n <= 0)
                continue;
            int rounded = (n + 7) / 8 * 8;
            DataCopyExtParams cp{(uint16_t)n, (uint32_t)(n * 4), (uint32_t)((stride(info, ChunkSize<kStaticBT>(info)) - n) * 4),
                                 (uint32_t)((16 - rounded) / 8), 0};
            DataCopyPadExtParams<float> pad{true, 0, (uint8_t)(rounded - n), 0};
            DataCopyPad(a[leaf * 256], in[pos(info, w, t, ChunkSize<kStaticBT>(info)) + t % ChunkSize<kStaticBT>(info)], cp, pad);
        }
        SetFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources_));
        WaitFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources_));
        Muls(a, a, -1.0f, E);
        PipeBarrier<PIPE_V>();
        // 行预取双缓冲：i+1 轮的取数/广播与 i 轮的乘加树无 UB 冲突，
        // 相邻屏障间的独立指令数翻倍，摊薄每轮 8 屏障的等待。
        DataCopy(row, a[2 * 16], {(uint16_t)L, 2, 30, 0});
        PipeBarrier<PIPE_V>();
        for (int i = 2; i < 16; ++i) {
            LocalTensor<float> cur = row[(i % 2) * L * 16];
            if (i + 1 < 16) {
                DataCopy(row[((i + 1) % 2) * L * 16], a[(i + 1) * 16], {(uint16_t)L, 2, 30, 0});
            }
            PipeBarrier<PIPE_V>();
            // 每个系数广播到对应16元素行；两次写入交错的32B块。
            // prod先存系数，乘法原位覆盖后继续使用原来的加法树。
            Brcb(prod, cur, L * 2, {2, 16});
            Brcb(prod[8], cur, L * 2, {2, 16});
            PipeBarrier<PIPE_V>();
            Mul(prod, a, prod, E);
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[128], 64, L, {1, 1, 1, 32, 32, 32});
            Add(prod[64], prod[64], prod[192], 64, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[64], 64, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[32], 32, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(prod, prod, prod[16], 16, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
            Add(a[i * 16], a[i * 16], prod, 16, L, {1, 1, 1, 32, 32, 32});
            PipeBarrier<PIPE_V>();
        }
        for (int i = 0; i < 16; ++i) {
            uint64_t mask[2] = {1ULL << i, 0};
            Duplicate(a[i * 16], 1.0f, mask, L, 1, 32);
        }
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources_));
        WaitFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources_));
        for (int leaf = 0; leaf < count; ++leaf) {
            if (rows[leaf] <= 0)
                continue;
            auto w = jobs[leaf];
            DataCopyExtParams cp{(uint16_t)rows[leaf], 64, 0, (uint32_t)((stride(info, 16) - 16) * 4), 0};
            DataCopyPad(out[pos(info, w, w.t + sub * 16, 16)], a[leaf * 256], cp);
        }

        CrossCoreSetFlag<0x2, PIPE_MTE3>(version % 2);

        SetFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources_));
        WaitFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources_));
    }
};

template <int L, bool Ahead, bool Parent = false, bool Raw = false, int kStaticBT = 0>
__aicore__ inline void leaf_merge_pipeline(GdnRawFront::Resource &resource,
                                           GM_ADDR input, GM_ADDR prev, GM_ADDR output, GM_ADDR workspace,
                                           GM_ADDR cuAddr, FullProblem info, int64_t tasks, GdnRawFront::VectorResources *resources)
{
    constexpr int S = 16, Group = 16;
    using Out = float;
    constexpr int W = S > 32 ? S : 32;
    constexpr int E = W * W;
    constexpr int BATCH = GDN::FP32_SOLVE_MERGE_BATCH_SIZE;
    constexpr int TEMP = GDN::FP32_SOLVE_SMALL_TEMP_SLOT_COUNT;
    static_assert(Group > 0 && Group <= 16 && 16 % Group == 0);
    int64_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= GetSubBlockNum();
    }
    int64_t workers = GetBlockNum();
    int64_t begin = 0, end = SpanTaskEnd<Parent, kStaticBT>(info, S * 2, tasks, workers, core);
    GlobalTensor<float> x, d, ws, batchWs;
    GlobalTensor<Out> y;
    GlobalTensor<int64_t> cu;
    x.SetGlobalBuffer((__gm__ float *)input);
    d.SetGlobalBuffer((__gm__ float *)prev);
    // 同组的所有层复用同一个 arena；slot 是组内下标，组间区间恒不相交。
    const int64_t arenaElements = ChunkSize<kStaticBT>(info) == 128
        ? GDN::FP32_SOLVE_ARENA128_ELEMENTS : GDN::FP32_SOLVE_ARENA64_ELEMENTS;
    ws.SetGlobalBuffer((__gm__ float *)workspace + core * arenaElements);
    batchWs.SetGlobalBuffer((__gm__ float *)workspace + core * arenaElements +
                           GDN::FP32_SOLVE_TEMP_ELEMENTS);
    static_assert(TEMP * E <= GDN::FP32_SOLVE_TEMP_ELEMENTS);
    y.SetGlobalBuffer((__gm__ Out *)output);
    cu.SetGlobalBuffer((__gm__ int64_t *)cuAddr);
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        using Tag = Arch::AtlasA2;
        using Policy = Gemm::MmadPingpong<Tag, true, false>;
        using Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
        using Copy = NsSolveTri::SolveTriTileCopy<Tag>;
        using MM = Gemm::Block::BlockMmadTla<Policy, Shape, Shape, float, float, float, void, Copy>;
        MM mm(resource);
        for (int64_t task = begin; task < end; task += Group) {
            int64_t index = task - begin;
            if (index % L == 0)
                CrossCoreWaitFlag((index / L) % 2);
            if (index % BATCH == 0 && index >= 2 * BATCH)
                CrossCoreWaitFlag(2 + (index / BATCH) % 2);
            int count = end - task < Group ? int(end - task) : Group;
            WorkItem jobs[Group];
            int rows0[Group], rows1[Group];
            for (int j = 0; j < count; ++j) {
                auto w = locate<Parent, kStaticBT>(task + j, S * 2, info, cu);
                jobs[j] = w;
                int n0 = w.length - w.t < S ? w.length - w.t : S;
                int n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                rows0[j] = n0;
                rows1[j] = n1;
                if (n1 > 0) {
                    auto d1 = d[pos(info, w, w.t + S, S)];
                    auto off = x[pos(info, w, w.t + S, ChunkSize<kStaticBT>(info)) + w.t % ChunkSize<kStaticBT>(info)];
                    full_gemm(mm, d1, off, ws[j * E], n1, n0, n1, stride(info, S), stride(info, ChunkSize<kStaticBT>(info)), W);
                }
            }

            SetFlag<HardEvent::FIX_MTE2>(0);
            WaitFlag<HardEvent::FIX_MTE2>(0);
            for (int j = 0; j < count; ++j) {
                auto w = jobs[j];
                int n0 = rows0[j], n1 = rows1[j];
                if (n1 > 0) {
                    int64_t slot = (index + j) % (BATCH * 2);
                    auto d0 = d[pos(info, w, w.t, S)];
                    full_gemm(mm, ws[j * E], d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                }
            }

            // 第二遍 MTE2 读完 ws 后再允许下一组 FIX 覆盖；结果仍经 PIPE_FIX 发布。
            SetFlag<HardEvent::MTE2_FIX>(0);
            WaitFlag<HardEvent::MTE2_FIX>(0);
            if ((index + count) % BATCH == 0 || task + count == end)
                CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
        }
        int64_t batches = (end - begin + BATCH - 1) / BATCH;
        for (int64_t batch = batches > 2 ? batches - 2 : 0; batch < batches; ++batch)
            CrossCoreWaitFlag(2 + batch % 2);
    }
    if ASCEND_IS_AIV {
        GdnRawFront::SolveStage<Raw> pipe(resources);
        GdnRawFront::Buffer<Raw> fb, ob;
        pipe.BindBuffer(fb, 2 * S * 2 * S * 4);
        if constexpr (sizeof(Out) != 4)
            pipe.BindBuffer(ob, 2 * S * 2 * S * sizeof(Out));
        auto f = fb.template Get<float>();
        LocalTensor<Out> o;
        if constexpr (sizeof(Out) == 4)
            o = f;
        else
            o = ob.template Get<Out>();
        int sub = GetSubBlockIdx();
        LeafProducer<L, Parent, Raw, kStaticBT> leaf;
        leaf.Init(pipe, input, prev, cuAddr, info, sub, resources);
        if constexpr (Ahead) {
            if (begin < end)
                leaf.Produce(begin, end, 0);
        }
        for (int64_t chunk = begin; chunk < end; chunk += L) {
            int64_t version = (chunk - begin) / L;
            if constexpr (!Ahead)
                leaf.Produce(chunk, end, version);
            if constexpr (Ahead) {
                if (chunk + L < end)
                    leaf.Produce(chunk + L, end, version + 1);
            }
            // 成对批处理：单轮 V/MTE2/MTE3 协议服务相邻两任务，flag 对数量减半；
            // BATCH=16 与步长2对齐，批边界恒落在对边界，不跨批。
            constexpr int64_t taskElems = S * 2 * S;
            for (int64_t task = chunk; task < end && task < chunk + L; task += 2) {
                const int count = task + 1 < end && task + 1 < chunk + L ? 2 : 1;
                WorkItem jobs[2];
                int rowsArr[2], n0Arr[2];
                int64_t slots[2];
                for (int j = 0; j < count; ++j) {
                    int64_t index = task + j - begin;
                    if (index % BATCH == 0)
                        CrossCoreWaitFlag(4 + (index / BATCH) % 2);
                    auto w = locate<Parent, kStaticBT>(task + j, S * 2, info, cu);
                    jobs[j] = w;
                    int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                    n0Arr[j] = n0;
                    rowsArr[j] = sub == 0 ? n0 : n1;
                    slots[j] = index % (BATCH * 2);
                    Duplicate(f[j * taskElems], 0.0f, S * 2 * S);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources));
                WaitFlag<HardEvent::V_MTE2>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE2>(resources));
                for (int j = 0; j < count; ++j) {
                    auto w = jobs[j];
                    int rows = rowsArr[j], n0 = n0Arr[j];
                    if (rows > 0) {
                        int rounded = (rows + 7) / 8 * 8;
                        DataCopyPadExtParams<float> pad{true, 0, (uint8_t)(rounded - rows), 0};
                        DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(rows * 4), (uint32_t)((stride(info, S) - rows) * 4),
                                             (uint32_t)((2 * S - rounded) / 8), 0};
                        DataCopyPad(f[j * taskElems + sub * S], d[pos(info, w, w.t + sub * S, S)], cp, pad);
                        if (sub == 1) {
                            DataCopyExtParams cross{(uint16_t)rows, (uint32_t)(n0 * 4), (uint32_t)((W - n0) * 4),
                                                    (uint32_t)((2 * S - n0) / 8), 0};
                            DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
                            DataCopyPad(f[j * taskElems], batchWs[slots[j] * E], cross, noPad);
                        }
                    }
                }
                SetFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources));
                WaitFlag<HardEvent::MTE2_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE2_V>(resources));
                for (int j = 0; j < count; ++j) {
                    if (sub == 1 && rowsArr[j] > 0) {
                        Muls(f[j * taskElems], f[j * taskElems], -1.0f, S, (uint8_t)rowsArr[j],
                             {1, 1, (uint8_t)(2 * S / 8), (uint8_t)(2 * S / 8)});
                        PipeBarrier<PIPE_V>();
                    }
                }
                if constexpr (sizeof(Out) != 4) {
                    for (int j = 0; j < count; ++j)
                        Cast(o[j * taskElems], f[j * taskElems], RoundMode::CAST_RINT, S * 2 * S);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources));
                WaitFlag<HardEvent::V_MTE3>(GdnRawFront::SolveEvent<Raw, HardEvent::V_MTE3>(resources));
                for (int j = 0; j < count; ++j) {
                    auto w = jobs[j];
                    if (rowsArr[j] > 0) {
                        DataCopyExtParams cp{(uint16_t)rowsArr[j], (uint32_t)(2 * S * sizeof(Out)), 0,
                                             (uint32_t)((stride(info, 2 * S) - 2 * S) * sizeof(Out)), 0};
                        DataCopyPad(y[pos(info, w, w.t + sub * S, 2 * S)], o[j * taskElems], cp);
                    }
                }
                SetFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources));
                WaitFlag<HardEvent::MTE3_V>(GdnRawFront::SolveEvent<Raw, HardEvent::MTE3_V>(resources));
                for (int j = 0; j < count; ++j) {
                    int64_t index = task + j - begin;
                    if ((index + 1) % BATCH == 0 || task + j + 1 == end)
                        CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + (index / BATCH) % 2);
                }
            }
        }
    }
}


__aicore__ inline void OrderMmadCreditReuse()
{
    if ASCEND_IS_AIC {
        // ready/free 已闭合 GM/FIX；串联旧析构的 credit 消费，避免新 MM 提前预置同号事件。
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
    }
}

// 调用方已经通过配对AIV→AIC→AIV发布本批x。所有子层属于同一物理组，
// 保留 ready/free 并有向交接 MM credit；Raw AIV 仅绑定视图，不构造局部 TPipe。
template <class Out, bool Raw = false, int kStaticBT = 0>
__aicore__ inline void RunOwnedBatch(GdnRawFront::Resource &resource,
                                    GM_ADDR x, GM_ADDR d16, GM_ADDR d32, GM_ADDR d64,
                                    GM_ADDR y, GM_ADDR ws, GM_ADDR cu, FullProblem p,
                                    GdnRawFront::VectorResources *resources)
{
    leaf_merge_pipeline<32, true, true, Raw, kStaticBT>(resource, x, d16, d32, ws, cu, p, p.tasks32, resources);
    OrderMmadCreditReuse();
    if constexpr (kStaticBT == 128) {
        stage64_merge<32, float, 8, true, Raw, kStaticBT>(resource, x, d32, d64, ws, cu, p, p.tasks64, resources);
        OrderMmadCreditReuse();
        pipeline_merge<64, Out, 16, true, Raw, kStaticBT>(resource, x, d64, y, ws, cu, p, p.tasks128, resources);
    } else if constexpr (kStaticBT == 64) {
        stage64_merge<32, Out, 8, true, Raw, kStaticBT>(resource, x, d32, y, ws, cu, p, p.tasks64, resources);
    } else if (p.bt == 128) {
        stage64_merge<32, float, 8, true, Raw, kStaticBT>(resource, x, d32, d64, ws, cu, p, p.tasks64, resources);
        OrderMmadCreditReuse();
        pipeline_merge<64, Out, 16, true, Raw, kStaticBT>(resource, x, d64, y, ws, cu, p, p.tasks128, resources);
    } else {
        stage64_merge<32, Out, 8, true, Raw, kStaticBT>(resource, x, d32, y, ws, cu, p, p.tasks64, resources);
    }
    OrderMmadCreditReuse();
    if ASCEND_IS_AIV {
        // 末次 MTE3_V 的 V 侧消费完成后，WU 才能重新预置同方向的 credit。
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
    }
}

} // namespace GdnFp32Solve

#endif
