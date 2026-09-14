/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_CUBE_H
#define CHUNK_KDA_FWD_FINALIZE_CUBE_H

#include <type_traits>
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif
#else
#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif
#endif
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "chunk_kda_fwd_finalize_struct.h"

namespace KdaFinalize {

using namespace AscendC;

// A5 的 Fixpipe 把两个独立的 FP32 结果写到对应 AIV 的 UB；
// A2/A3 的 Fixpipe 写到每 AIC 私有的 GM 中转区。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using FinalizeArch = Catlass::Arch::Ascend950;
constexpr bool kFinalizeA5 = true;
constexpr FixpipeConfig kFinalizeUbFixpipe = {CO2Layout::ROW_MAJOR, true};
#else
using FinalizeArch = Catlass::Arch::AtlasA2;
constexpr bool kFinalizeA5 = false;
#endif

template <bool StateVFirst>
class FinalizeCube {
    using Element = bfloat16_t;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using HLayout = std::conditional_t<StateVFirst, LayoutCM, LayoutRM>;
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<
        FinalizeArch, Element, LayoutRM, Element, HLayout, float, LayoutRM>;
    using TileCopyAV = Catlass::Gemm::Tile::PackedTileCopyTla<
        FinalizeArch, Element, LayoutRM, Element, LayoutRM, float, LayoutRM>;

    static constexpr uint32_t kQOffset = 0;
    static constexpr uint32_t kHOffset = kQOffset + 16 * 1024;
    static constexpr uint32_t kAOffset = kHOffset + 32 * 1024;
    static constexpr uint32_t kVOffset = kAOffset + 8 * 1024;
    static constexpr uint32_t kL1HeadBytes = 72 * 1024;
    static constexpr uint32_t kUbSlotBytes = 80 * 1024;
    static constexpr uint32_t kL0AOffset[2] = {0, 16 * 1024};
    static constexpr uint32_t kL0BOffset[2] = {0, 32 * 1024};

public:
    __aicore__ inline void Init(const FinalizeArgs &args)
    {
        args_ = args;
        core_ = WorkgroupId();
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.qgScaled));
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.aqk));
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.vNew));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(args.h));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if ASCEND_IS_AIC {
            SetLoadDataPaddingValue<Element>(static_cast<Element>(0));
        }
#endif
    }

    __aicore__ inline void Process()
    {
        if (core_ >= args_.tiling.usedCoreNum || args_.tiling.usedCoreNum == 0) {
            return;
        }
        constexpr uint16_t kA5Ready[4] = {0, 1, 16, 17};
        constexpr uint16_t kA5Free[4] = {4, 5, 20, 21};
        constexpr uint16_t kA2Ready[2] = {0, 1};
        constexpr uint16_t kA2Free[2] = {2, 3};
        bool slotInFlight[4] = {false, false, false, false};
        bool pairInFlight[2] = {false, false};
        const uint32_t partitions = CeilDiv(args_.tiling.valueHeadNum,
                                            args_.tiling.headsPerPartition);
        const uint32_t total = TotalWorkItems(args_);
        const uint32_t begin = WorkBegin(total, core_, args_.tiling.usedCoreNum);
        const uint32_t end = WorkEnd(total, core_, args_.tiling.usedCoreNum);
        for (uint32_t work = begin; work < end; ++work) {
            FinalizeChunk chunk{};
            if (!ResolveChunk(args_, work / partitions, chunk)) {
                continue;
            }
            const uint32_t headBegin = (work % partitions) * args_.tiling.headsPerPartition;
            uint32_t headEnd = headBegin + args_.tiling.headsPerPartition;
            if (headEnd > args_.tiling.valueHeadNum) {
                headEnd = args_.tiling.valueHeadNum;
            }
            for (uint32_t group = headBegin; group < headEnd; group += Shape::kHeadsPerGroup) {
                uint32_t active = headEnd - group;
                if (active > Shape::kHeadsPerGroup) {
                    active = Shape::kHeadsPerGroup;
                }
                if constexpr (kFinalizeA5) {
                    for (uint32_t localHead = 0; localHead < active; ++localHead) {
                        if (slotInFlight[localHead]) {
                            CrossCoreWaitFlag<0x4, PIPE_MTE2>(kA5Free[localHead]);
                        }
                        // C0: P=Q_g_scaled@H，R=Aqk@V_new；互不依赖。
                        StageC0(chunk, group + localHead, localHead);
                        CrossCoreSetFlag<0x4, PIPE_FIX>(kA5Ready[localHead]);
                        slotInFlight[localHead] = true;
                    }
                } else {
                    for (uint32_t pair = 0; pair < 2; ++pair) {
                        if (pairInFlight[pair]) {
                            CrossCoreWaitFlag<0x2, PIPE_MTE2>(kA2Free[pair]);
                        }
                        for (uint32_t member = 0; member < 2; ++member) {
                            const uint32_t localHead = pair * 2 + member;
                            if (localHead < active) {
                                StageC0(chunk, group + localHead, localHead);
                            }
                        }
                        // mode 2 汇聚两个 AIV；空闲 AIV 也参与本 pair。
                        CrossCoreSetFlag<0x2, PIPE_FIX>(kA2Ready[pair]);
                        pairInFlight[pair] = true;
                    }
                }
            }
        }
        if constexpr (kFinalizeA5) {
            for (uint32_t i = 0; i < 4; ++i) {
                if (slotInFlight[i]) {
                    CrossCoreWaitFlag<0x4, PIPE_MTE2>(kA5Free[i]);
                }
            }
        } else {
            for (uint32_t pair = 0; pair < 2; ++pair) {
                if (pairInFlight[pair]) {
                    CrossCoreWaitFlag<0x2, PIPE_MTE2>(kA2Free[pair]);
                }
            }
            for (uint32_t plane = 0; plane < 2; ++plane) {
                if (l0OperandInFlight_[plane]) {
                    WaitFlag<HardEvent::M_MTE1>(plane);
                }
            }
        }
    }

private:
    template <typename TileCopy, typename LayoutB>
    __aicore__ inline void LoadOperands(GlobalTensor<Element> &sourceA,
                                        GlobalTensor<Element> &sourceB,
                                        uint64_t offsetA, uint64_t offsetB,
                                        uint32_t l1AOffset, uint32_t l1BOffset,
                                        uint32_t m, uint32_t k, uint32_t aStride,
                                        uint32_t bRows, uint32_t n)
    {
        using L1A = typename TileCopy::LayoutTagL1A;
        using L1B = typename TileCopy::LayoutTagL1B;
        auto gmALayout = tla::MakeLayout<Element, LayoutRM>(Shape::kChunkRows, aStride);
        auto gmBLayout = tla::MakeLayout<Element, LayoutB>(bRows, n);
        auto gmATensor = tla::MakeTensor(sourceA[offsetA], gmALayout, Catlass::Arch::PositionGM{});
        auto gmBTensor = tla::MakeTensor(sourceB[offsetB], gmBLayout, Catlass::Arch::PositionGM{});
        auto gmABlock = GetTile(gmATensor, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto gmBBlock = GetTile(gmBTensor, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        using CopyGmA = typename TileCopy::template CopyGmToL1A<decltype(gmABlock)>;
        using CopyGmB = typename TileCopy::template CopyGmToL1B<decltype(gmBBlock)>;
        auto l1A = resource_.l1Buf.template GetBufferByByte<Element>(l1AOffset);
        auto l1B = resource_.l1Buf.template GetBufferByByte<Element>(l1BOffset);
        auto l1ATensor = tla::MakeTensor(l1A, tla::MakeLayout<Element, L1A>(Shape::kChunkRows, aStride),
                                        Catlass::Arch::PositionL1{});
        auto l1BTensor = tla::MakeTensor(l1B, tla::MakeLayout<Element, L1B>(bRows, n),
                                        Catlass::Arch::PositionL1{});
        CopyGmA{}(l1ATensor, gmABlock);
        if constexpr (!kFinalizeA5) {
            if (m < 16) {
                // A2/A3 MMAD 最少读取 16 行；zN 每个 K 分形占 64 个 32B 块。
                Fill(l1A[m * 16], InitConstValueParams<Element>(
                    static_cast<uint16_t>(CeilDiv(k, 16)), static_cast<uint16_t>(16 - m),
                    static_cast<uint16_t>(48 + m), static_cast<Element>(0.0f)));
            }
        }
        CopyGmB{}(l1BTensor, gmBBlock);
    }

    template <typename TileCopy>
    __aicore__ inline void ComputeProduct(uint32_t l1AOffset, uint32_t l1BOffset,
                                          uint32_t m, uint32_t k, uint32_t aStride,
                                          uint32_t bRows, uint32_t n, uint32_t plane)
    {
        using L1A = typename TileCopy::LayoutTagL1A;
        using L1B = typename TileCopy::LayoutTagL1B;
        using L0A = typename TileCopy::LayoutTagL0A;
        using L0B = typename TileCopy::LayoutTagL0B;
        using CopyA = typename TileCopy::CopyL1ToL0A;
        using CopyB = typename TileCopy::CopyL1ToL0B;
        using Mmad = Catlass::Gemm::Tile::TileMmadTla<FinalizeArch, Element, L1A>;

        auto l1A = resource_.l1Buf.template GetBufferByByte<Element>(l1AOffset);
        auto l1B = resource_.l1Buf.template GetBufferByByte<Element>(l1BOffset);
        auto l1ATensor = tla::MakeTensor(l1A, tla::MakeLayout<Element, L1A>(Shape::kChunkRows, aStride),
                                        Catlass::Arch::PositionL1{});
        auto l1BTensor = tla::MakeTensor(l1B, tla::MakeLayout<Element, L1B>(bRows, n),
                                        Catlass::Arch::PositionL1{});
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Lock<PIPE_MTE1>(0);
        Mutex::Lock<PIPE_MTE1>(static_cast<uint8_t>(4 + plane));
#endif
        auto l0A = resource_.l0ABuf.template GetBufferByByte<Element>(kL0AOffset[plane]);
        auto l0B = resource_.l0BBuf.template GetBufferByByte<Element>(kL0BOffset[plane]);
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(plane * Shape::kProductBytes);
        // 尾块的 L0 分形必须按真实 M/K 紧凑排列；以 64 行布局裁成 1 行
        // 会让 MMAD 把后续 K 分形误读为尚未写入的 L0 地址。
        auto tensorL0A = tla::MakeTensor(l0A, tla::MakeLayout<Element, L0A>(m, k),
                                        Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, tla::MakeLayout<Element, L0B>(k, n),
                                        Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n),
                                        Catlass::Arch::PositionL0C{});
#if !(defined(__CCE_AICORE__) && __CCE_AICORE__ == 310)
        if (l0OperandInFlight_[plane]) {
            WaitFlag<HardEvent::M_MTE1>(plane);
        }
#endif
        auto tileL1A = GetTile(l1ATensor, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = GetTile(l1BTensor, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        CopyA{}(tileL0A, tileL1A);
        CopyB{}(tileL0B, tileL1B);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_MTE1>(static_cast<uint8_t>(4 + plane));
        Mutex::Unlock<PIPE_MTE1>(0);
        Mutex::Lock<PIPE_M>(static_cast<uint8_t>(4 + plane));
        Mutex::Lock<PIPE_M>(static_cast<uint8_t>(2 + plane));
#else
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        SetFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(0);
#endif
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        // A2/A3 的 MMAD 至少处理一个 16 行分形，尾块仍只写回有效行。
        const uint32_t madM = kFinalizeA5 ? m : (m < 16 ? 16 : m);
        Mmad{}(tileL0C, tileL0A, tileL0B, madM, n, k, true, 0);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_M>(static_cast<uint8_t>(2 + plane));
        Mutex::Unlock<PIPE_M>(static_cast<uint8_t>(4 + plane));
#else
        SetFlag<HardEvent::M_MTE1>(plane);
        l0OperandInFlight_[plane] = true;
        SetFlag<HardEvent::M_FIX>(plane);
#endif
    }

    __aicore__ inline void StoreProduct(uint32_t m, uint32_t n,
                                         uint32_t localHead, uint32_t plane)
    {
        auto l0C = resource_.l0CBuf.template GetBufferByByte<float>(plane * Shape::kProductBytes);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Lock<PIPE_FIX>(static_cast<uint8_t>(2 + plane));
#else
        WaitFlag<HardEvent::M_FIX>(plane);
#endif
        if constexpr (kFinalizeA5) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            const uint32_t aiv = localHead / 2;
            const uint32_t slot = localHead % 2;
            auto ub = resource_.ubBuf.template GetBufferByByte<float>(
                slot * kUbSlotBytes + plane * Shape::kProductBytes);
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fix{};
            fix.nSize = n;
            fix.mSize = m;
            fix.srcStride = CeilDiv(m, 16) * 16;
            fix.dstStride = n;
            fix.quantPre = QuantMode_t::NoQuant;
            fix.subBlockId = static_cast<uint8_t>(aiv);
            Fixpipe<float, float, kFinalizeUbFixpipe>(ub, l0C, fix);
#endif
        } else {
            auto relay = GlobalTensor<float>{};
            relay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                args_.workspace + static_cast<uint64_t>(core_) * Shape::kRelayCoreBytes +
                localHead * Shape::kRelayHeadBytes + plane * Shape::kProductBytes));
            auto fix = FixpipeParamsV220(n, m, CeilDiv(m, 16) * 16, n, false);
            fix.quantPre = QuantMode_t::NoQuant;
            Fixpipe<float, float, CFG_ROW_MAJOR>(relay, l0C, fix);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_FIX>(static_cast<uint8_t>(2 + plane));
#else
        SetFlag<HardEvent::FIX_M>(plane);
        WaitFlag<HardEvent::FIX_M>(plane);
#endif
    }

    __aicore__ inline void StageC0(const FinalizeChunk &chunk,
                                    uint32_t head, uint32_t localHead)
    {
        const uint32_t lane = localHead * kL1HeadBytes;
        const uint64_t qOffset = InputOffset(args_, chunk, head, Shape::kHeadDim);
        const uint64_t aOffset = InputOffset(args_, chunk, head, Shape::kChunkRows);
        const uint64_t hOffset = StateOffset(args_, chunk, head);
        // C0 入口将本 head 的四个输入搬入独立 L1 区，之后两次 MMAD 不读 GM。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Lock<PIPE_MTE2>(0);
#endif
        LoadOperands<TileCopyQH, HLayout>(
            q_, h_, qOffset, hOffset, lane + kQOffset, lane + kHOffset,
            chunk.validRows, Shape::kHeadDim, Shape::kHeadDim,
            Shape::kHeadDim, Shape::kHeadDim);
        LoadOperands<TileCopyAV, LayoutRM>(
            a_, v_, aOffset, qOffset, lane + kAOffset, lane + kVOffset,
            chunk.validRows, chunk.validRows, Shape::kChunkRows,
            Shape::kChunkRows, Shape::kHeadDim);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Mutex::Unlock<PIPE_MTE2>(0);
#else
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
#endif
        ComputeProduct<TileCopyQH>(lane + kQOffset, lane + kHOffset,
                                   chunk.validRows, Shape::kHeadDim,
                                   Shape::kHeadDim, Shape::kHeadDim,
                                   Shape::kHeadDim, 0);
        ComputeProduct<TileCopyAV>(lane + kAOffset, lane + kVOffset,
                                   chunk.validRows, chunk.validRows,
                                   Shape::kChunkRows, Shape::kChunkRows,
                                   Shape::kHeadDim, 1);
        StoreProduct(chunk.validRows, Shape::kHeadDim, localHead, 0);
        StoreProduct(chunk.validRows, Shape::kHeadDim, localHead, 1);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // 同一 FIX pipe 的两次 Unlock 不保证先后完成；ready 必须覆盖两个 plane。
        PipeBarrier<PIPE_FIX>();
#endif
    }

    FinalizeArgs args_{};
    uint32_t core_ = 0;
    bool l0OperandInFlight_[2] = {false, false};
    GlobalTensor<Element> q_;
    GlobalTensor<Element> a_;
    GlobalTensor<Element> v_;
    GlobalTensor<Element> h_;
    Catlass::Arch::Resource<FinalizeArch> resource_;
};

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_CUBE_H
