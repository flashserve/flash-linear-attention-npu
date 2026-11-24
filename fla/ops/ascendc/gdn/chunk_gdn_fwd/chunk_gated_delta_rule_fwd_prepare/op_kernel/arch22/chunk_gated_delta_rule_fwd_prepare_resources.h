// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. Prepare 唯一的物理资源拥有者；计算步骤只有裸偏移视图。
#ifndef GDN_ARCH22_UNIFIED_PREPARE_RESOURCES_H
#define GDN_ARCH22_UNIFIED_PREPARE_RESOURCES_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_fwd_prepare_policy.h"

namespace GdnUnifiedPrepare {
using namespace AscendC;
using Args = Arguments<GM_ADDR>;

struct Chunk {
    uint64_t batch, index, row, valid, hkBegin, hkEnd, headBegin, headEnd;
};

__aicore__ inline uint64_t GroupId()
{
    uint64_t group = GetBlockIdx();
    if ASCEND_IS_AIV { group /= GetSubBlockNum(); }
    return group;
}

template <int BT>
__aicore__ inline Chunk Decode(const Args &a, uint64_t work, uint64_t hksPerTask)
{
    const uint64_t partitions = CeilDiv(a.hk, hksPerTask);
    const uint64_t globalChunk = work / partitions;
    Chunk c{};
    c.batch = globalChunk / a.nt;
    c.index = globalChunk % a.nt;
    c.row = c.index * BT;
    c.valid = c.row < a.tokens ? Min(BT, a.tokens - c.row) : 0;
    if (a.isVarlen != 0) {
        GlobalTensor<int64_t> cu, indices;
        cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(a.cuSeqlens));
        indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(a.chunkIndices), a.nt * 2);
        const int64_t sequence = indices.GetValue(c.index * 2);
        const int64_t localChunk = indices.GetValue(c.index * 2 + 1);
        const int64_t begin = cu.GetValue(sequence);
        const int64_t end = cu.GetValue(sequence + 1);
        const int64_t row = begin + localChunk * BT;
        c.batch = 0;
        c.row = row > 0 ? static_cast<uint64_t>(row) : 0;
        c.valid = end > row && row >= 0 && static_cast<uint64_t>(row) < a.tokens
            ? Min(BT, Min(static_cast<uint64_t>(end - row), a.tokens - c.row)) : 0;
    }
    const uint64_t g = a.hv / a.hk;
    c.hkBegin = (work % partitions) * hksPerTask;
    c.hkEnd = Min(a.hk, c.hkBegin + hksPerTask);
    c.headBegin = c.hkBegin * g;
    c.headEnd = c.hkEnd * g;
    return c;
}

template <int QkvLayout>
__aicore__ inline uint64_t InputOffset(const Args &a, const Chunk &c,
                                      uint64_t head, uint64_t heads, uint64_t width, uint64_t row = 0)
{
    if constexpr (QkvLayout == 1) {
        return ((c.batch * a.tokens + c.row + row) * heads + head) * width;
    } else {
        return ((c.batch * heads + head) * a.tokens + c.row + row) * width;
    }
}

__aicore__ inline uint64_t HeadOffset(const Args &a, const Chunk &c, uint64_t head, uint64_t width = 1)
{
    return ((c.batch * a.hv + head) * a.tokens + c.row) * width;
}

class Resources {
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ub_;
    TBuf<TPosition::A1> l1_;
    TBuf<TPosition::A2> l0a_;
    TBuf<TPosition::B2> l0b_;
    TBuf<TPosition::CO1> l0c_;
    uint32_t cSlots_;
public:
    TEventID m2v{}, v2m2{}, v2m3{}, m3v{}, m3m2{}, s2v{}, m3s{}, m2s{};
    TEventID l1Free[2]{}, rhsFree{}, l1Ready{}, l0Free{}, l0Ready{}, cFree[2]{}, cReady[2]{}, fixM2{}, m2Fix{};
    TEventID mergeFree[2]{}, mergeReady{};

    __aicore__ inline Resources(uint32_t cSlots = 1) : cSlots_(cSlots)
    {
        m2s = pipe_.AllocEventID<HardEvent::MTE2_S>();
        if ASCEND_IS_AIV {
            pipe_.InitBuffer(ub_, Local::kUbBytes);
            m2v = pipe_.AllocEventID<HardEvent::MTE2_V>();
            v2m2 = pipe_.AllocEventID<HardEvent::V_MTE2>();
            v2m3 = pipe_.AllocEventID<HardEvent::V_MTE3>();
            m3v = pipe_.AllocEventID<HardEvent::MTE3_V>();
            m3m2 = pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
            s2v = pipe_.AllocEventID<HardEvent::S_V>();
            m3s = pipe_.AllocEventID<HardEvent::MTE3_S>();
        }
        if ASCEND_IS_AIC {
            pipe_.InitBuffer(l1_, Local::kL1Bytes);
            pipe_.InitBuffer(l0a_, Local::kL0Bytes);
            pipe_.InitBuffer(l0b_, Local::kL0Bytes);
            pipe_.InitBuffer(l0c_, Local::kL0CBytes);
            l1Free[0] = pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
            l1Free[1] = pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
            rhsFree = pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
            l1Ready = pipe_.AllocEventID<HardEvent::MTE2_MTE1>();
            mergeFree[0] = pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
            mergeFree[1] = pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
            mergeReady = pipe_.AllocEventID<HardEvent::MTE2_MTE1>();
            l0Free = pipe_.AllocEventID<HardEvent::M_MTE1>();
            l0Ready = pipe_.AllocEventID<HardEvent::MTE1_M>();
            for (uint32_t i = 0; i < cSlots_; ++i) {
                cFree[i] = pipe_.AllocEventID<HardEvent::FIX_M>();
                cReady[i] = pipe_.AllocEventID<HardEvent::M_FIX>();
            }
            fixM2 = pipe_.AllocEventID<HardEvent::FIX_MTE2>();
            m2Fix = pipe_.AllocEventID<HardEvent::MTE2_FIX>();
            SetFlag<HardEvent::MTE1_MTE2>(l1Free[0]);
            SetFlag<HardEvent::MTE1_MTE2>(l1Free[1]);
            SetFlag<HardEvent::MTE1_MTE2>(rhsFree);
            SetFlag<HardEvent::MTE1_MTE2>(mergeFree[0]);
            SetFlag<HardEvent::MTE1_MTE2>(mergeFree[1]);
            SetFlag<HardEvent::M_MTE1>(l0Free);
            for (uint32_t i = 0; i < cSlots_; ++i) SetFlag<HardEvent::FIX_M>(cFree[i]);
        }
    }

    template <class T> __aicore__ inline LocalTensor<T> UB(uint32_t offset)
    { return ub_.Get<uint8_t>()[offset].template ReinterpretCast<T>(); }
    template <class T> __aicore__ inline LocalTensor<T> L1(uint32_t offset)
    { return l1_.Get<uint8_t>()[offset].template ReinterpretCast<T>(); }
    template <class T> __aicore__ inline LocalTensor<T> L0A() { return l0a_.Get<T>(); }
    template <class T> __aicore__ inline LocalTensor<T> L0B() { return l0b_.Get<T>(); }
    __aicore__ inline LocalTensor<float> L0C(uint32_t slot = 0)
    { return l0c_.Get<float>()[slot * (Local::kL0CBytes / (2 * sizeof(float)))]; }

    __aicore__ inline TEventID GeneralFree(uint32_t bank, uint32_t operand) const
    { return bank == 0 ? l1Free[operand] : mergeFree[operand]; }
    __aicore__ inline TEventID GeneralReady(uint32_t bank) const
    { return bank == 0 ? l1Ready : mergeReady; }

    template <HardEvent E> __aicore__ inline void Fence()
    {
        TEventID id{};
        if constexpr (E == HardEvent::MTE2_V) id = m2v;
        if constexpr (E == HardEvent::V_MTE2) id = v2m2;
        if constexpr (E == HardEvent::V_MTE3) id = v2m3;
        if constexpr (E == HardEvent::MTE3_V) id = m3v;
        if constexpr (E == HardEvent::MTE3_MTE2) id = m3m2;
        if constexpr (E == HardEvent::S_V) id = s2v;
        if constexpr (E == HardEvent::MTE3_S) id = m3s;
        if constexpr (E == HardEvent::MTE2_S) id = m2s;
        if constexpr (E == HardEvent::FIX_MTE2) id = fixM2;
        if constexpr (E == HardEvent::MTE2_FIX) id = m2Fix;
        SetFlag<E>(id);
        WaitFlag<E>(id);
    }

    __aicore__ inline ~Resources()
    {
        if ASCEND_IS_AIV {
            Fence<HardEvent::MTE3_V>();
            Fence<HardEvent::MTE3_MTE2>();
            Fence<HardEvent::V_MTE2>();
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(m2v);
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(v2m2);
            pipe_.ReleaseEventID<HardEvent::V_MTE3>(v2m3);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(m3v);
            pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(m3m2);
            pipe_.ReleaseEventID<HardEvent::S_V>(s2v);
            pipe_.ReleaseEventID<HardEvent::MTE3_S>(m3s);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(m2s);
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free[1]);
            WaitFlag<HardEvent::MTE1_MTE2>(mergeFree[0]);
            WaitFlag<HardEvent::MTE1_MTE2>(mergeFree[1]);
            WaitFlag<HardEvent::MTE1_MTE2>(rhsFree);
            WaitFlag<HardEvent::M_MTE1>(l0Free);
            for (uint32_t i = 0; i < cSlots_; ++i) WaitFlag<HardEvent::FIX_M>(cFree[i]);
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(l1Free[0]);
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(l1Free[1]);
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(rhsFree);
            pipe_.ReleaseEventID<HardEvent::MTE2_MTE1>(l1Ready);
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(mergeFree[0]);
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(mergeFree[1]);
            pipe_.ReleaseEventID<HardEvent::MTE2_MTE1>(mergeReady);
            pipe_.ReleaseEventID<HardEvent::M_MTE1>(l0Free);
            pipe_.ReleaseEventID<HardEvent::MTE1_M>(l0Ready);
            for (uint32_t i = 0; i < cSlots_; ++i) {
                pipe_.ReleaseEventID<HardEvent::FIX_M>(cFree[i]);
                pipe_.ReleaseEventID<HardEvent::M_FIX>(cReady[i]);
            }
            pipe_.ReleaseEventID<HardEvent::FIX_MTE2>(fixM2);
            pipe_.ReleaseEventID<HardEvent::MTE2_FIX>(m2Fix);
        }
    }
};

template <class T> __aicore__ inline GlobalTensor<T> Gm(GM_ADDR address, uint64_t elements)
{
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address), elements);
    return tensor;
}

template <class T> __aicore__ inline void Invalidate(GlobalTensor<T> tensor)
{
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(tensor);
    DataSyncBarrier<MemDsbT::DDR>();
}
} // namespace GdnUnifiedPrepare
#endif
