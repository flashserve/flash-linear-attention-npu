#ifndef GDN_ARCH22_FP32_SOLVE_PIPELINE_H
#define GDN_ARCH22_FP32_SOLVE_PIPELINE_H
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. DAV_2201 分层 FP32 Solve 的私有迁移组件。
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 220
#error "This private Solve pipeline is only compiled for DAV_2201"
#endif
#include "kernel_operator.h"
#include "solve_tri_fp32.h"
namespace GdnFp32Solve {
using namespace AscendC;
using namespace Catlass;


struct FullProblem {
    int64_t batch, tokens, heads, bt, headFirst, sequences;
    int64_t tasks32, tasks64, tasks128;
};
struct WorkItem {
    int64_t b, h, t, base, length;
};

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

__aicore__ inline WorkItem locate(int64_t task, int span, FullProblem p, GlobalTensor<int64_t> cu)
{
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

template <class In>
__aicore__ inline void full_convert(GM_ADDR x, GM_ADDR y, int64_t count)
{
    constexpr int E = 16384;
    TPipe pipe;
    TBuf<TPosition::VECCALC> ib, ob;
    pipe.InitBuffer(ib, E * sizeof(In));
    pipe.InitBuffer(ob, E * 4);
    auto a = ib.Get<In>();
    auto b = ob.Get<float>();
    GlobalTensor<In> src;
    GlobalTensor<float> dst;
    src.SetGlobalBuffer((__gm__ In *)x);
    dst.SetGlobalBuffer((__gm__ float *)y);
    for (int64_t start = GetBlockIdx() * E; start < count; start += GetBlockNum() * 2 * E) {
        int n = count - start < E ? count - start : E;
        DataCopy(a, src[start], n);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Cast(b, a, RoundMode::CAST_NONE, n);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(dst[start], b, n);
        PipeBarrier<PIPE_ALL>();
    }
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

template <int S, class Out, int Group = 16>
__aicore__ inline void pipeline_merge(GM_ADDR input, GM_ADDR prev, GM_ADDR output, GM_ADDR workspace, GM_ADDR cuAddr,
                                      FullProblem info, int64_t tasks)
{
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
    int64_t workers = GetBlockNum(), q = tasks / workers, rem = tasks % workers;
    int64_t begin = 0, end = q + (core < rem ? 1 : 0);
    GlobalTensor<float> x, d, ws, batchWs;
    GlobalTensor<Out> y;
    GlobalTensor<int64_t> cu;
    x.SetGlobalBuffer((__gm__ float *)input);
    d.SetGlobalBuffer((__gm__ float *)prev);
    ws.SetGlobalBuffer((__gm__ float *)workspace + core * TEMP * E);
    batchWs.SetGlobalBuffer((__gm__ float *)workspace + workers * TEMP * E);
    y.SetGlobalBuffer((__gm__ Out *)output);
    cu.SetGlobalBuffer((__gm__ int64_t *)cuAddr);
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        using Tag = Arch::AtlasA2;
        using Policy = Gemm::MmadPingpong<Tag, true, false>;
        using Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
        using Copy = NsSolveTri::SolveTriTileCopy<Tag>;
        using MM = Gemm::Block::BlockMmadTla<Policy, Shape, Shape, float, float, float, void, Copy>;
        Arch::Resource<Tag> resource;
        MM mm(resource);
        if constexpr (S == 16) {
            for (int64_t task = begin; task < end; task += Group) {
                int64_t index = task - begin;
                if (index % BATCH == 0 && index >= 2 * BATCH)
                    CrossCoreWaitFlag(2 + (index / BATCH) % 2);
                int count = end - task < Group ? int(end - task) : Group;
                WorkItem jobs[Group];
                int rows0[Group], rows1[Group];
                for (int j = 0; j < count; ++j) {
                    auto w = locate(task + j, S * 2, info, cu);
                    jobs[j] = w;
                    int n0 = w.length - w.t < S ? w.length - w.t : S;
                    int n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                    rows0[j] = n0;
                    rows1[j] = n1;
                    if (n1 > 0) {
                        auto d1 = d[pos(info, w, w.t + S, S)];
                        auto off = x[pos(info, w, w.t + S, info.bt) + w.t % info.bt];
                        full_gemm(mm, d1, off, ws[j * E], n1, n0, n1, stride(info, S), stride(info, info.bt), W);
                    }
                }

                PipeBarrier<PIPE_ALL>();
                for (int j = 0; j < count; ++j) {
                    auto w = jobs[j];
                    int n0 = rows0[j], n1 = rows1[j];
                    if (n1 > 0) {
                        int64_t slot = core * BATCH * 2 + (index + j) % (BATCH * 2);
                        auto d0 = d[pos(info, w, w.t, S)];
                        full_gemm(mm, ws[j * E], d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                    }
                }

                PipeBarrier<PIPE_ALL>();
                if ((index + count) % BATCH == 0 || task + count == end)
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
            }
        } else {
            for (int64_t task = begin; task < end; ++task) {
                int64_t index = task - begin, slot = core * BATCH * 2 + index % (BATCH * 2);
                if (index % BATCH == 0 && index >= 2 * BATCH)
                    CrossCoreWaitFlag(2 + (index / BATCH) % 2);
                auto w = locate(task, S * 2, info, cu);
                int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                if (n1 > 0) {
                    auto d0 = d[pos(info, w, w.t, S)], d1 = d[pos(info, w, w.t + S, S)];
                    auto off = x[pos(info, w, w.t + S, info.bt) + w.t % info.bt];
                    full_gemm(mm, d1, off, ws, n1, n0, n1, stride(info, S), stride(info, info.bt), W);
                    PipeBarrier<PIPE_ALL>();
                    full_gemm(mm, ws, d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                }
                if ((index + 1) % BATCH == 0 || task + 1 == end)
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
            }
        }
        int64_t batches = (end - begin + BATCH - 1) / BATCH;
        for (int64_t batch = batches > 2 ? batches - 2 : 0; batch < batches; ++batch)
            CrossCoreWaitFlag(2 + batch % 2);
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> fb, ob;
        pipe.InitBuffer(fb, S * 2 * S * 4);
        if constexpr (sizeof(Out) != 4)
            pipe.InitBuffer(ob, S * 2 * S * sizeof(Out));
        auto f = fb.Get<float>();
        LocalTensor<Out> o;
        if constexpr (sizeof(Out) == 4)
            o = f;
        else
            o = ob.Get<Out>();
        int sub = GetSubBlockIdx();
        for (int64_t task = begin; task < end; ++task) {
            int64_t index = task - begin, slot = core * BATCH * 2 + index % (BATCH * 2);
            if (index % BATCH == 0)
                CrossCoreWaitFlag(4 + (index / BATCH) % 2);
            auto w = locate(task, S * 2, info, cu);
            int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
            int rows = sub == 0 ? n0 : n1;
            Duplicate(f, 0.0f, S * 2 * S);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
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
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if (sub == 1 && rows > 0) {
                Muls(f, f, -1.0f, S, (uint8_t)rows, {1, 1, (uint8_t)(2 * S / 8), (uint8_t)(2 * S / 8)});
                PipeBarrier<PIPE_V>();
            }
            if constexpr (sizeof(Out) != 4)
                Cast(o, f, RoundMode::CAST_RINT, S * 2 * S);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            if (rows > 0) {
                DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(2 * S * sizeof(Out)), 0,
                                     (uint32_t)((stride(info, 2 * S) - 2 * S) * sizeof(Out)), 0};
                DataCopyPad(y[pos(info, w, w.t + sub * S, 2 * S)], o, cp);
            }
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            if ((index + 1) % BATCH == 0 || task + 1 == end)
                CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + (index / BATCH) % 2);
        }
    }
}

template <int S, class Out, int Group = 8>
__aicore__ inline void stage64_merge(GM_ADDR input, GM_ADDR prev, GM_ADDR output, GM_ADDR workspace, GM_ADDR cuAddr,
                                     FullProblem info, int64_t tasks)
{
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
    int64_t workers = GetBlockNum(), q = tasks / workers, rem = tasks % workers;
    int64_t begin = 0, end = q + (core < rem ? 1 : 0);
    GlobalTensor<float> x, d, ws, batchWs;
    GlobalTensor<Out> y;
    GlobalTensor<int64_t> cu;
    x.SetGlobalBuffer((__gm__ float *)input);
    d.SetGlobalBuffer((__gm__ float *)prev);
    ws.SetGlobalBuffer((__gm__ float *)workspace + core * TEMP * E);
    batchWs.SetGlobalBuffer((__gm__ float *)workspace + workers * TEMP * E);
    y.SetGlobalBuffer((__gm__ Out *)output);
    cu.SetGlobalBuffer((__gm__ int64_t *)cuAddr);
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        using Tag = Arch::AtlasA2;
        using Policy = Gemm::MmadPingpong<Tag, true, false>;
        using Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
        using Copy = NsSolveTri::SolveTriTileCopy<Tag>;
        using MM = Gemm::Block::BlockMmadTla<Policy, Shape, Shape, float, float, float, void, Copy>;
        Arch::Resource<Tag> resource;
        MM mm(resource);
        if constexpr (S == 32) {
            for (int64_t task = begin; task < end; task += Group) {
                int64_t index = task - begin;
                if (index % BATCH == 0 && index >= 2 * BATCH)
                    CrossCoreWaitFlag(2 + (index / BATCH) % 2);
                int count = end - task < Group ? int(end - task) : Group;
                WorkItem jobs[Group];
                int rows0[Group], rows1[Group];
                for (int j = 0; j < count; ++j) {
                    auto w = locate(task + j, S * 2, info, cu);
                    jobs[j] = w;
                    int n0 = w.length - w.t < S ? w.length - w.t : S;
                    int n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                    rows0[j] = n0;
                    rows1[j] = n1;
                    if (n1 > 0) {
                        auto d1 = d[pos(info, w, w.t + S, S)];
                        auto off = x[pos(info, w, w.t + S, info.bt) + w.t % info.bt];
                        full_gemm(mm, d1, off, ws[j * E], n1, n0, n1, stride(info, S), stride(info, info.bt), W);
                    }
                }

                PipeBarrier<PIPE_ALL>();
                for (int j = 0; j < count; ++j) {
                    auto w = jobs[j];
                    int n0 = rows0[j], n1 = rows1[j];
                    if (n1 > 0) {
                        int64_t slot = core * BATCH * 2 + (index + j) % (BATCH * 2);
                        auto d0 = d[pos(info, w, w.t, S)];
                        full_gemm(mm, ws[j * E], d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                    }
                }

                PipeBarrier<PIPE_ALL>();
                if ((index + count) % BATCH == 0 || task + count == end)
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
            }
        } else {
            for (int64_t task = begin; task < end; ++task) {
                int64_t index = task - begin, slot = core * BATCH * 2 + index % (BATCH * 2);
                if (index % BATCH == 0 && index >= 2 * BATCH)
                    CrossCoreWaitFlag(2 + (index / BATCH) % 2);
                auto w = locate(task, S * 2, info, cu);
                int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                if (n1 > 0) {
                    auto d0 = d[pos(info, w, w.t, S)], d1 = d[pos(info, w, w.t + S, S)];
                    auto off = x[pos(info, w, w.t + S, info.bt) + w.t % info.bt];
                    full_gemm(mm, d1, off, ws, n1, n0, n1, stride(info, S), stride(info, info.bt), W);
                    PipeBarrier<PIPE_ALL>();
                    full_gemm(mm, ws, d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                }
                if ((index + 1) % BATCH == 0 || task + 1 == end)
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
            }
        }
        int64_t batches = (end - begin + BATCH - 1) / BATCH;
        for (int64_t batch = batches > 2 ? batches - 2 : 0; batch < batches; ++batch)
            CrossCoreWaitFlag(2 + batch % 2);
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> fb, ob;
        pipe.InitBuffer(fb, S * 2 * S * 4);
        if constexpr (sizeof(Out) != 4)
            pipe.InitBuffer(ob, S * 2 * S * sizeof(Out));
        auto f = fb.Get<float>();
        LocalTensor<Out> o;
        if constexpr (sizeof(Out) == 4)
            o = f;
        else
            o = ob.Get<Out>();
        int sub = GetSubBlockIdx();
        for (int64_t task = begin; task < end; ++task) {
            int64_t index = task - begin, slot = core * BATCH * 2 + index % (BATCH * 2);
            if (index % BATCH == 0)
                CrossCoreWaitFlag(4 + (index / BATCH) % 2);
            auto w = locate(task, S * 2, info, cu);
            int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
            int rows = sub == 0 ? n0 : n1;
            Duplicate(f, 0.0f, S * 2 * S);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
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
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if (sub == 1 && rows > 0) {
                Muls(f, f, -1.0f, S, (uint8_t)rows, {1, 1, (uint8_t)(2 * S / 8), (uint8_t)(2 * S / 8)});
                PipeBarrier<PIPE_V>();
            }
            if constexpr (sizeof(Out) != 4)
                Cast(o, f, RoundMode::CAST_RINT, S * 2 * S);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            if (rows > 0) {
                DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(2 * S * sizeof(Out)), 0,
                                     (uint32_t)((stride(info, 2 * S) - 2 * S) * sizeof(Out)), 0};
                DataCopyPad(y[pos(info, w, w.t + sub * S, 2 * S)], o, cp);
            }
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            if ((index + 1) % BATCH == 0 || task + 1 == end)
                CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + (index / BATCH) % 2);
        }
    }
}


template <int L>
class LeafProducer {
    static constexpr int E = L * 256;
    TBuf<TPosition::VECCALC> ab, pb, rb, cb;
    LocalTensor<float> a, prod, row, coeff;
    GlobalTensor<float> in, out;
    GlobalTensor<int64_t> cu;
    FullProblem info;
    int sub;

public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR input, GM_ADDR output, GM_ADDR cuAddr, FullProblem p, int s)
    {
        info = p;
        sub = s;
        pipe.InitBuffer(ab, E * 4);
        pipe.InitBuffer(pb, E * 4);
        pipe.InitBuffer(rb, L * 16 * 4);
        pipe.InitBuffer(cb, L * 128 * 4);
        a = ab.Get<float>();
        prod = pb.Get<float>();
        row = rb.Get<float>();
        coeff = cb.Get<float>();
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
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        for (int leaf = 0; leaf < count; ++leaf) {
            auto w = locate(begin + leaf, 32, info, cu);
            jobs[leaf] = w;
            int64_t t = w.t + sub * 16;
            int n = w.length - t < 16 ? int(w.length - t) : 16;
            rows[leaf] = n;
            if (n <= 0)
                continue;
            int rounded = (n + 7) / 8 * 8;
            DataCopyExtParams cp{(uint16_t)n, (uint32_t)(n * 4), (uint32_t)((stride(info, info.bt) - n) * 4),
                                 (uint32_t)((16 - rounded) / 8), 0};
            DataCopyPadExtParams<float> pad{true, 0, (uint8_t)(rounded - n), 0};
            DataCopyPad(a[leaf * 256], in[pos(info, w, t, info.bt) + t % info.bt], cp, pad);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Muls(a, a, -1.0f, E);
        PipeBarrier<PIPE_V>();
        for (int i = 2; i < 16; ++i) {
            DataCopy(row, a[i * 16], {(uint16_t)L, 2, 30, 0});
            PipeBarrier<PIPE_V>();
            Brcb(coeff, row, L * 2, {1, 8});
            PipeBarrier<PIPE_V>();
            for (int r = 0; r < L * 16; r += 240) {
                uint8_t n = L * 16 - r < 240 ? L * 16 - r : 240;
                Mul(prod[r * 16], a[r * 16], coeff[r * 8], 16, n, {1, 1, 0, 2, 2, 1});
            }
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

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (int leaf = 0; leaf < count; ++leaf) {
            if (rows[leaf] <= 0)
                continue;
            auto w = jobs[leaf];
            DataCopyExtParams cp{(uint16_t)rows[leaf], 64, 0, (uint32_t)((stride(info, 16) - 16) * 4), 0};
            DataCopyPad(out[pos(info, w, w.t + sub * 16, 16)], a[leaf * 256], cp);
        }

        CrossCoreSetFlag<0x2, PIPE_MTE3>(version % 2);

        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
};

template <int L, bool Ahead>
__aicore__ inline void leaf_merge_pipeline(GM_ADDR input, GM_ADDR prev, GM_ADDR output, GM_ADDR workspace,
                                           GM_ADDR cuAddr, FullProblem info, int64_t tasks)
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
    int64_t workers = GetBlockNum(), q = tasks / workers, rem = tasks % workers;
    int64_t begin = 0, end = q + (core < rem ? 1 : 0);
    GlobalTensor<float> x, d, ws, batchWs;
    GlobalTensor<Out> y;
    GlobalTensor<int64_t> cu;
    x.SetGlobalBuffer((__gm__ float *)input);
    d.SetGlobalBuffer((__gm__ float *)prev);
    ws.SetGlobalBuffer((__gm__ float *)workspace + core * TEMP * E);
    batchWs.SetGlobalBuffer((__gm__ float *)workspace + workers * TEMP * E);
    y.SetGlobalBuffer((__gm__ Out *)output);
    cu.SetGlobalBuffer((__gm__ int64_t *)cuAddr);
    if ASCEND_IS_AIC {
        SetHF32Mode(false);
        using Tag = Arch::AtlasA2;
        using Policy = Gemm::MmadPingpong<Tag, true, false>;
        using Shape = tla::Shape<tla::Int<64>, tla::Int<64>, tla::Int<64>>;
        using Copy = NsSolveTri::SolveTriTileCopy<Tag>;
        using MM = Gemm::Block::BlockMmadTla<Policy, Shape, Shape, float, float, float, void, Copy>;
        Arch::Resource<Tag> resource;
        MM mm(resource);
        if constexpr (S == 16) {
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
                    auto w = locate(task + j, S * 2, info, cu);
                    jobs[j] = w;
                    int n0 = w.length - w.t < S ? w.length - w.t : S;
                    int n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                    rows0[j] = n0;
                    rows1[j] = n1;
                    if (n1 > 0) {
                        auto d1 = d[pos(info, w, w.t + S, S)];
                        auto off = x[pos(info, w, w.t + S, info.bt) + w.t % info.bt];
                        full_gemm(mm, d1, off, ws[j * E], n1, n0, n1, stride(info, S), stride(info, info.bt), W);
                    }
                }

                PipeBarrier<PIPE_ALL>();
                for (int j = 0; j < count; ++j) {
                    auto w = jobs[j];
                    int n0 = rows0[j], n1 = rows1[j];
                    if (n1 > 0) {
                        int64_t slot = core * BATCH * 2 + (index + j) % (BATCH * 2);
                        auto d0 = d[pos(info, w, w.t, S)];
                        full_gemm(mm, ws[j * E], d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                    }
                }

                PipeBarrier<PIPE_ALL>();
                if ((index + count) % BATCH == 0 || task + count == end)
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
            }
        } else {
            for (int64_t task = begin; task < end; ++task) {
                int64_t index = task - begin, slot = core * BATCH * 2 + index % (BATCH * 2);
                if (index % BATCH == 0 && index >= 2 * BATCH)
                    CrossCoreWaitFlag(2 + (index / BATCH) % 2);
                auto w = locate(task, S * 2, info, cu);
                int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                if (n1 > 0) {
                    auto d0 = d[pos(info, w, w.t, S)], d1 = d[pos(info, w, w.t + S, S)];
                    auto off = x[pos(info, w, w.t + S, info.bt) + w.t % info.bt];
                    full_gemm(mm, d1, off, ws, n1, n0, n1, stride(info, S), stride(info, info.bt), W);
                    PipeBarrier<PIPE_ALL>();
                    full_gemm(mm, ws, d0, batchWs[slot * E], n1, n0, n0, W, stride(info, S), W);
                }
                if ((index + 1) % BATCH == 0 || task + 1 == end)
                    CrossCoreSetFlag<0x2, PIPE_FIX>(4 + (index / BATCH) % 2);
            }
        }
        int64_t batches = (end - begin + BATCH - 1) / BATCH;
        for (int64_t batch = batches > 2 ? batches - 2 : 0; batch < batches; ++batch)
            CrossCoreWaitFlag(2 + batch % 2);
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        TBuf<TPosition::VECCALC> fb, ob;
        pipe.InitBuffer(fb, S * 2 * S * 4);
        if constexpr (sizeof(Out) != 4)
            pipe.InitBuffer(ob, S * 2 * S * sizeof(Out));
        auto f = fb.Get<float>();
        LocalTensor<Out> o;
        if constexpr (sizeof(Out) == 4)
            o = f;
        else
            o = ob.Get<Out>();
        int sub = GetSubBlockIdx();
        LeafProducer<L> leaf;
        leaf.Init(pipe, input, prev, cuAddr, info, sub);
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
            for (int64_t task = chunk; task < end && task < chunk + L; ++task) {
                int64_t index = task - begin, slot = core * BATCH * 2 + index % (BATCH * 2);
                if (index % BATCH == 0)
                    CrossCoreWaitFlag(4 + (index / BATCH) % 2);
                auto w = locate(task, S * 2, info, cu);
                int n0 = w.length - w.t < S ? w.length - w.t : S, n1 = w.length - w.t - S < S ? w.length - w.t - S : S;
                int rows = sub == 0 ? n0 : n1;
                Duplicate(f, 0.0f, S * 2 * S);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
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
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                if (sub == 1 && rows > 0) {
                    Muls(f, f, -1.0f, S, (uint8_t)rows, {1, 1, (uint8_t)(2 * S / 8), (uint8_t)(2 * S / 8)});
                    PipeBarrier<PIPE_V>();
                }
                if constexpr (sizeof(Out) != 4)
                    Cast(o, f, RoundMode::CAST_RINT, S * 2 * S);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                if (rows > 0) {
                    DataCopyExtParams cp{(uint16_t)rows, (uint32_t)(2 * S * sizeof(Out)), 0,
                                         (uint32_t)((stride(info, 2 * S) - 2 * S) * sizeof(Out)), 0};
                    DataCopyPad(y[pos(info, w, w.t + sub * S, 2 * S)], o, cp);
                }
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                if ((index + 1) % BATCH == 0 || task + 1 == end)
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(2 + (index / BATCH) % 2);
            }
        }
    }
}


// mixed 边界覆盖所有物理组，包括零任务组。各阶段先完成 ready/free
// 和 CATLASS 析构 drain，再复用 UB/L1/scratch/flag。
template <class In, class Out>
__aicore__ inline void Run(GM_ADDR raw, GM_ADDR x, GM_ADDR d16, GM_ADDR d32, GM_ADDR d64, GM_ADDR y, GM_ADDR ws,
                           GM_ADDR cu, FullProblem p)
{
    // KKT/staging 的跨组 GM 写入对所有 Solve 消费者可见。
    SyncAll<false>();
    if constexpr (sizeof(In) != 4) {
        if ASCEND_IS_AIV {
            full_convert<In>(raw, x, p.batch * p.tokens * p.heads * p.bt);
        }
        SyncAll<false>();
    }
    leaf_merge_pipeline<32, true>(x, d16, d32, ws, cu, p, p.tasks32);
    SyncAll<false>();
    if (p.bt == 128) {
        stage64_merge<32, float>(x, d32, d64, ws, cu, p, p.tasks64);
        SyncAll<false>();
        pipeline_merge<64, Out>(x, d64, y, ws, cu, p, p.tasks128);
    } else {
        stage64_merge<32, Out>(x, d32, y, ws, cu, p, p.tasks64);
    }

    // 最终 A 由 AIV/MTE3 写回，不使用旧 AIC/FIX 发布协议。
    SyncAll<false>();
}

} // namespace GdnFp32Solve

#endif
