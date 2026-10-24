// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// BSD-3-Clause. 共用顶层资源的 UB 裸视图和显式事件，不拥有存储或事件池。
#ifndef GDN_ARCH22_RAW_FRONT_RESOURCES_H
#define GDN_ARCH22_RAW_FRONT_RESOURCES_H

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

namespace GdnRawFront {
using namespace AscendC;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using Resource = Catlass::Arch::Resource<Catlass::Arch::Ascend950>;
#else
using Resource = Catlass::Arch::Resource<Catlass::Arch::AtlasA2>;
#endif

__aicore__ inline uint32_t Align(uint32_t bytes) { return (bytes + 31) / 32 * 32; }
__aicore__ inline uint32_t Reserve(uint32_t &cursor, uint32_t bytes)
{
    uint32_t offset = cursor;
    cursor += Align(bytes);
    return offset;
}

struct KktLayout {
    uint32_t g, beta, score, out, typed, gate, row, bytes;
    __aicore__ inline void Init(uint32_t bt, uint32_t align, uint32_t outputBytes)
    {
        uint32_t cursor = 0;
        g = Reserve(cursor, align * 4);
        beta = Reserve(cursor, align * 4);
        score = Reserve(cursor, bt * align * 4);
        out = Reserve(cursor, bt * align * 4);
        typed = outputBytes == sizeof(float) ? cursor : Reserve(cursor, bt * align * outputBytes);
        gate = Reserve(cursor, 32 * align * 4);
        row = Reserve(cursor, 32 * 8 * 4);
        bytes = cursor;
    }
};

struct WuLayout {
    uint32_t input, beta, g, out, fp32, betaFp32, brcb, gFp32, bytes;
    __aicore__ inline void Init(uint32_t rows, uint32_t width, uint32_t inputBytes, bool gated)
    {
        uint32_t cursor = 0;
        input = Reserve(cursor, 2 * Align(rows * width * inputBytes));
        beta = Reserve(cursor, 2 * Align(rows * 4));
        g = gated ? Reserve(cursor, 2 * Align(rows * 4)) : 0;
        out = Reserve(cursor, 2 * Align(rows * width * inputBytes));
        fp32 = Reserve(cursor, rows * width * 4);
        betaFp32 = Reserve(cursor, rows * 4);
        brcb = Reserve(cursor, rows * 32);
        gFp32 = gated ? Reserve(cursor, rows * 4) : 0;
        bytes = cursor;
    }
};

struct PrepareLayout {
    uint32_t input, prefixBth, prefixBht, offsets, acc, bytes;
    __aicore__ inline void Init(uint32_t bt, uint64_t heads)
    {
        const uint32_t width = heads < 64 ? static_cast<uint32_t>(heads) : 64;
        const uint32_t stride = (width + 7) / 8 * 8;
        const uint32_t tileBytes = bt * stride * sizeof(float);
        uint32_t cursor = 0;
        input = Reserve(cursor, tileBytes);
        prefixBth = Reserve(cursor, tileBytes);
        prefixBht = Reserve(cursor, tileBytes);
        offsets = Reserve(cursor, tileBytes);
        acc = Reserve(cursor, stride * sizeof(float));
        bytes = cursor;
    }
};

class VectorResources {
    Resource *owner_ = nullptr;
public:
    KktLayout kkt;
    PrepareLayout prepare;

    __aicore__ inline void Init(Resource &owner, uint32_t bt, uint32_t btAlign,
                               uint64_t heads, uint32_t inputBytes)
    {
        owner_ = &owner;
        kkt.Init(bt, btAlign, inputBytes);
        prepare.Init(bt, heads);
    }

    template <class T> __aicore__ inline LocalTensor<T> View(uint32_t offset, uint32_t bytes)
    {
        auto tensor = owner_->ubBuf.template GetBufferByByte<T>(offset);
        tensor.SetSize(Align(bytes) / sizeof(T));
        return tensor;
    }

    template <HardEvent Direction> __aicore__ inline event_t Event(uint32_t slot = 0) const
    {
        // Resource 构造已 Destroy 其 TPipe；此后只使用固定 ID。
        // 同方向槽 0/1 的 seed、循环和 drain 由各阶段成对闭合。
        return static_cast<event_t>(slot);
    }

    __aicore__ inline void BeginWuPhase()
    {
        // 空闲槽的可写 credit；未使用槽也由 EndWuPhase 消费，不留下孤立 token。
        for (uint32_t slot = 0; slot < 2; ++slot) {
            SetFlag<HardEvent::V_MTE2>(Event<HardEvent::V_MTE2>(slot));
            SetFlag<HardEvent::MTE3_V>(Event<HardEvent::MTE3_V>(slot));
        }
    }
    __aicore__ inline void EndWuPhase()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(Event<HardEvent::V_MTE2>(slot));
            WaitFlag<HardEvent::MTE3_V>(Event<HardEvent::MTE3_V>(slot));
        }
    }
};

// Raw 模板参数仅兼容原 Solve 接口；两种实例均为非拥有视图。
template <bool Raw> class Buffer {
    LocalTensor<uint8_t> buffer_;
public:
    __aicore__ inline void BindView(VectorResources *resources, uint32_t offset, uint32_t bytes)
    { buffer_ = resources->View<uint8_t>(offset, bytes); }
    template <class T> __aicore__ inline LocalTensor<T> Get()
    { return buffer_.template ReinterpretCast<T>(); }
};

template <bool Raw> class SolveStage {
    VectorResources *resources_;
    uint32_t cursor_ = 0;
public:
    __aicore__ inline explicit SolveStage(VectorResources *resources) : resources_(resources) {}
    __aicore__ inline void BindBuffer(Buffer<Raw> &buffer, uint32_t bytes)
    { buffer.BindView(resources_, Reserve(cursor_, bytes), bytes); }
};

template <bool Raw, HardEvent Direction>
__aicore__ inline event_t SolveEvent(VectorResources *resources)
{
    return static_cast<event_t>(0);
}
} // namespace GdnRawFront
#endif
