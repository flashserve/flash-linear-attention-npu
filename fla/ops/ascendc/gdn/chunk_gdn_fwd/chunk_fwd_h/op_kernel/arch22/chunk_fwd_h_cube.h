/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_CHUNK_FWD_H_CUBE_H
#define ARCH22_CHUNK_FWD_H_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include <type_traits>

#include "../chunk_fwd_h_policy.h"
#include "../chunk_fwd_h_utils.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "chunk_fwd_h_utils.h"

namespace GDN {

template <typename CompilePolicy, bool STATE_V_FIRST>
class ChunkFwdHCubeArch22 {
public:
    __aicore__ inline void Init(const FwdHKernelArgs &args)
    {
        args_ = args;
        coreIdx_ = AscendC::GetBlockIdx();
        coreNum_ = AscendC::GetBlockNum();
        hasCubeWork_ = args_.tiling.useInitialState || CompilePolicy::STORE_FINAL ||
            args_.tiling.seqlen > static_cast<int64_t>(FWD_H_CHUNK);
        if (!hasCubeWork_) {
            return;
        }
        InitBuffers();
        InitEvents();
    }

    __aicore__ inline void Process()
    {
        const FwdHCoreHeadRange range =
            FwdHResolveCoreHeadRange(args_.tiling, coreIdx_, coreNum_);
        const uint32_t headsPerSequence = static_cast<uint32_t>(args_.tiling.vNumHead);
        uint32_t cachedSequence = FwdHSequenceCount(args_.tiling);
        FwdHSequenceSpan sequenceSpan{};
        FwdHSequenceCursor sequenceCursor{};
        for (uint32_t cursor = range.begin; cursor < range.end;) {
            const uint32_t sequence = cursor / headsPerSequence;
            const uint32_t hvBegin = cursor - sequence * headsPerSequence;
            uint32_t unitHeads = range.end - cursor;
            const uint32_t sequenceRemain = headsPerSequence - hvBegin;
            if (unitHeads > sequenceRemain) {
                unitHeads = sequenceRemain;
            }
            if (unitHeads > FWD_H_AIC_HEAD_SLOTS) {
                unitHeads = FWD_H_AIC_HEAD_SLOTS;
            }
            if (sequence != cachedSequence) {
                sequenceSpan = FwdHResolveSequence(args_, sequence, sequenceCursor);
                cachedSequence = sequence;
            }
            const FwdHWorkUnit unit{
                sequenceSpan,
                FwdHBuildHeadRange<CompilePolicy::GATE_MODE>(args_.tiling, hvBegin, unitHeads)};
            cursor += unitHeads;
            if (unit.sequence.chunkCount != 0 && unit.headRound.activeHeadCount != 0) {
                ProcessWorkUnit(unit, cursor < range.end);
            }
        }
        if (hasCubeWork_) {
            DrainEvents();
        }
    }

private:
    using ArchTag = Catlass::Arch::AtlasA2;
    using PType = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = std::conditional_t<STATE_V_FIRST, Catlass::layout::ColumnMajor,
                                       Catlass::layout::RowMajor>;
    using LayoutLeftS2 = Catlass::layout::ColumnMajor;
    using LayoutRightS2 = Catlass::layout::RowMajor;
    using LayoutOutput = Catlass::layout::RowMajor;
    using TileS0 = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, bfloat16_t, LayoutW, bfloat16_t, LayoutH, PType, LayoutOutput>;
    using TileS2 = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, bfloat16_t, LayoutLeftS2, bfloat16_t, LayoutRightS2, float, LayoutOutput>;
    using ElementAccumulator = typename TileS2::ElementAccumulator;
    using CopyL1ToL0AS0 = typename TileS0::CopyL1ToL0A;
    using CopyL1ToL0BS0 = typename TileS0::CopyL1ToL0B;
    using CopyL1ToL0AS2 = typename TileS2::CopyL1ToL0A;
    using CopyL1ToL0BS2 = typename TileS2::CopyL1ToL0B;
    using TileMmadS0 = Catlass::Gemm::Tile::TileMmadTla<ArchTag, bfloat16_t,
                                                        typename TileS0::LayoutTagL1A>;
    using TileMmadS2 = Catlass::Gemm::Tile::TileMmadTla<ArchTag, bfloat16_t,
                                                        typename TileS2::LayoutTagL1A>;
    template <typename Tensor>
    using CopyGmToL1AS0 = typename TileS0::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1BS0 = typename TileS0::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyGmToL1AS2 = typename TileS2::template CopyGmToL1A<Tensor>;
    template <typename Tensor>
    using CopyGmToL1BS2 = typename TileS2::template CopyGmToL1B<Tensor>;
    template <typename Tensor>
    using CopyL0CToGmS0 = typename TileS0::template CopyL0CToGm<Tensor>;
    template <typename Tensor>
    using CopyL0CToGmS2 = typename TileS2::template CopyL0CToGm<Tensor>;

    static constexpr auto L1_W_LAYOUT = tla::MakeLayout<bfloat16_t, typename TileS0::LayoutTagL1A>(
        tla::Int<FWD_H_CHUNK>{}, tla::Int<FWD_H_K>{});
    static constexpr auto L1_H_LAYOUT = tla::MakeLayout<bfloat16_t, typename TileS0::LayoutTagL1B>(
        tla::Int<FWD_H_K>{}, tla::Int<FWD_H_V>{});
    static constexpr auto L1_LEFT_S2_LAYOUT =
        tla::MakeLayout<bfloat16_t, typename TileS2::LayoutTagL1A>(
            tla::Int<FWD_H_K>{}, tla::Int<FWD_H_CHUNK>{});
    static constexpr auto L1_RIGHT_S2_LAYOUT =
        tla::MakeLayout<bfloat16_t, typename TileS2::LayoutTagL1B>(
            tla::Int<FWD_H_CHUNK>{}, tla::Int<FWD_H_V>{});
    static constexpr uint32_t L0_SLOTS = 2;
    static constexpr uint32_t L0A_SLOT_BYTES = FWD_H_CHUNK * FWD_H_K * sizeof(bfloat16_t);
    static constexpr uint32_t L0B_SLOT_BYTES = FWD_H_K * FWD_H_V * sizeof(bfloat16_t);
    static constexpr uint32_t L0C_SLOT_BYTES = FWD_H_K * FWD_H_V * sizeof(ElementAccumulator);

    __aicore__ inline AscendC::TEventID WReadyEvent(uint32_t slot) const
    {
        return wReadyEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID WDoneEvent(uint32_t slot) const
    {
        return wDoneEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID HRightReadyEvent(uint32_t slot) const
    {
        return hRightReadyEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID HRightDoneEvent(uint32_t slot) const
    {
        return hRightDoneEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID L0AFreeEvent(uint32_t slot) const
    {
        return l0AFreeEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID L0AReadyEvent(uint32_t slot) const
    {
        return l0AReadyEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID L0BFreeEvent(uint32_t slot) const
    {
        return l0BFreeEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID L0BReadyEvent(uint32_t slot) const
    {
        return l0BReadyEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID FixFreeEvent(uint32_t slot) const
    {
        return fixFreeEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID FixDoneEvent(uint32_t slot) const
    {
        return fixDoneEvent_[slot];
    }

    __aicore__ inline uint64_t ScratchByteOffset(int64_t base, uint32_t slot,
                                                  uint32_t slotBytes) const
    {
        return static_cast<uint64_t>(base) +
            (static_cast<uint64_t>(coreIdx_) * FWD_H_AIC_HEAD_SLOTS + slot) * slotBytes;
    }

    __aicore__ inline void ClearL1(AscendC::LocalTensor<bfloat16_t> tensor,
                                   uint32_t bytes) const
    {
        AscendC::InitConstValueParams<bfloat16_t> params(
            1, static_cast<uint16_t>(bytes / 32U), 0, static_cast<bfloat16_t>(0));
        AscendC::InitConstValue(tensor, params);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void InitBuffers()
    {
        GetTPipePtr()->InitBuffer(l1Buf_, ArchTag::L1_SIZE);
        GetTPipePtr()->InitBuffer(l0ABuf_, ArchTag::L0A_SIZE);
        GetTPipePtr()->InitBuffer(l0BBuf_, ArchTag::L0B_SIZE);
        GetTPipePtr()->InitBuffer(l0CBuf_, ArchTag::L0C_SIZE);
        GetTPipePtr()->InitBuffer(fixBuf_, ArchTag::FIXBUF_SIZE);
        for (uint32_t slot = 0; slot < FWD_H_AIC_HEAD_SLOTS; ++slot) {
            l1W_[slot] = l1Buf_.Get<uint8_t>()[FWD_H_L1_W_BASE + slot * FWD_H_L1_W_SLOT_BYTES].template ReinterpretCast<bfloat16_t>();
            l1HRight_[slot] = l1Buf_.Get<uint8_t>()[FWD_H_L1_H_RIGHT_BASE + slot * FWD_H_L1_H_RIGHT_SLOT_BYTES].template ReinterpretCast<bfloat16_t>();
            l1Kg_[slot] = l1Buf_.Get<uint8_t>()[FWD_H_L1_KG_BASE + slot * FWD_H_L1_KG_SLOT_BYTES].template ReinterpretCast<bfloat16_t>();
        }
        for (uint32_t slot = 0; slot < L0_SLOTS; ++slot) {
            l0A_[slot] = l0ABuf_.Get<uint8_t>()[slot * L0A_SLOT_BYTES].template ReinterpretCast<bfloat16_t>();
            l0B_[slot] = l0BBuf_.Get<uint8_t>()[slot * L0B_SLOT_BYTES].template ReinterpretCast<bfloat16_t>();
            l0C_[slot] = l0CBuf_.Get<uint8_t>()[slot * L0C_SLOT_BYTES].template ReinterpretCast<ElementAccumulator>();
        }
    }

    __aicore__ inline void InitEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIC_HEAD_SLOTS; ++slot) {
            wReadyEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            wDoneEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
            hRightReadyEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            hRightDoneEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HRightReadyEvent(slot));
        }
        for (uint32_t slot = 0; slot < L0_SLOTS; ++slot) {
            l0AFreeEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
            l0AReadyEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_M>();
            l0BFreeEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_MTE1>();
            l0BReadyEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE1_M>();
            fixFreeEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::FIX_M>();
            fixDoneEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::M_FIX>();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(slot));
        }
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIC_HEAD_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HRightReadyEvent(slot));
        }
        for (uint32_t slot = 0; slot < L0_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(slot));
        }
        for (uint32_t slot = 0; slot < FWD_H_AIC_HEAD_SLOTS; ++slot) {
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(WDoneEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(HRightReadyEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(HRightDoneEvent(slot));
        }
        for (uint32_t slot = 0; slot < L0_SLOTS; ++slot) {
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_M>(L0AReadyEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE1_M>(L0BReadyEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::FIX_M>(FixFreeEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::M_FIX>(FixDoneEvent(slot));
        }
    }

    __aicore__ inline uint64_t WOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, FWD_H_K);
    }

    __aicore__ inline uint64_t HOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHHOffset(args_.tiling, unit.sequence, head.hv, chunk.globalChunk);
    }

    __aicore__ inline void LoadStage0Head(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head, bool wPreloaded)
    {
        // Stage0 搬运：W_c,h[M,K] 与 H_c,h[K,V] 分别进入四个 roundHead L1 resident 槽。
        // wPreloaded：W 已由上一 chunk 的 S2 尾部 PrefetchNextW 装入（WDone 已 set），
        // 此处只装 H——W 的 wait/copy/set 全部跳过，flag 记账保持 1:1。
        const uint32_t slot = head.roundHead;
        if (!wPreloaded) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));
            AscendC::GlobalTensor<bfloat16_t> gmW;
            gmW.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.w) + WOffset(unit, chunk, head));
            if (chunk.validTokens < FWD_H_CHUNK) {
                ClearL1(l1W_[slot], FWD_H_L1_W_SLOT_BYTES);
            }
            auto gmWLayout = tla::MakeLayout<bfloat16_t, LayoutW>(chunk.validTokens, FWD_H_K);
            auto tensorW = tla::MakeTensor(gmW, gmWLayout, Catlass::Arch::PositionGM{});
            auto blockW = tla::GetTile(tensorW, tla::MakeCoord(0, 0),
                                       tla::MakeShape(chunk.validTokens, FWD_H_K));
            auto tensorL1W = tla::MakeTensor(l1W_[slot], L1_W_LAYOUT, Catlass::Arch::PositionL1{});
            CopyGmToL1AS0<decltype(blockW)> copyW;
            copyW(tensorL1W, blockW);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HRightReadyEvent(slot));

        AscendC::GlobalTensor<bfloat16_t> gmH;
        if (chunk.first && args_.tiling.useInitialState != 0 && !CompilePolicy::STATE_FP32) {
            const uint64_t stateOffset = FwdHStateOffset<STATE_V_FIRST>(
                args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
            gmH.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.initialState) + stateOffset);
        } else {
            gmH.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h) + HOffset(unit, chunk, head));
        }
        auto gmHLayout = tla::MakeLayout<bfloat16_t, LayoutH>(FWD_H_K, FWD_H_V);
        auto tensorH = tla::MakeTensor(gmH, gmHLayout, Catlass::Arch::PositionGM{});
        auto blockH = tla::GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(FWD_H_K, FWD_H_V));
        auto tensorL1H = tla::MakeTensor(l1HRight_[slot], L1_H_LAYOUT, Catlass::Arch::PositionL1{});
        CopyGmToL1BS0<decltype(blockH)> copyH;
        copyH(tensorL1H, blockH);
        if (!wPreloaded) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(WDoneEvent(slot));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(HRightDoneEvent(slot));
    }

    __aicore__ inline void ComputeStage0Head(const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head, uint32_t pipelineSlot)
    {
        // Stage0 计算：Pacc=W@H；A2/A3 不支持 L0C->UB，PType 结果由 Fixpipe 写 P GM scratch。
        const uint32_t slot = head.roundHead;
        const uint32_t m = FwdHAlignCube(chunk.validTokens);
        auto tensorL1W = tla::MakeTensor(l1W_[slot], L1_W_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1H = tla::MakeTensor(l1HRight_[slot], L1_H_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A_[pipelineSlot],
            tla::MakeLayout<bfloat16_t, typename TileS0::LayoutTagL0A>(m, FWD_H_K),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B_[pipelineSlot],
            tla::MakeLayout<bfloat16_t, typename TileS0::LayoutTagL0B>(FWD_H_K, FWD_H_V),
            Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C_[pipelineSlot], tla::MakeLayoutL0C(m, FWD_H_V),
                                         Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(WDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(HRightDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(pipelineSlot));
        CopyL1ToL0AS0 copyA;
        CopyL1ToL0BS0 copyB;
        copyA(tensorL0A, tla::GetTile(tensorL1W, tla::MakeCoord(0, 0), tla::MakeShape(m, FWD_H_K)));
        copyB(tensorL0B, tla::GetTile(tensorL1H, tla::MakeCoord(0, 0),
                                     tla::MakeShape(FWD_H_K, FWD_H_V)));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HRightReadyEvent(slot));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0AReadyEvent(pipelineSlot));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0BReadyEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0AReadyEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0BReadyEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(pipelineSlot));
        TileMmadS0 mmad;
        mmad(tensorL0C, tensorL0A, tensorL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(pipelineSlot));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(pipelineSlot));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(FixDoneEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(FixDoneEvent(pipelineSlot));
        const uint64_t pByteOffset = ScratchByteOffset(args_.tiling.vWorkspaceOffset,
                                                       head.roundHead,
                                                       FWD_H_TOKEN_MATRIX_FP32_BYTES);
        AscendC::GlobalTensor<PType> gmP;
        gmP.SetGlobalBuffer(reinterpret_cast<__gm__ PType *>(args_.workspace + pByteOffset));
        auto tensorP = tla::MakeTensor(gmP, tla::MakeLayout<PType, LayoutOutput>(m, FWD_H_V),
                                       Catlass::Arch::PositionGM{});
        // A2/A3 的 FP32 Fixpipe 按 16 行块写出。尾 chunk 的 W 已在 L1 补零，
        // 因此写满对齐后的 m 行，并由 AIV 只读取 validTokens 行。
        CopyL0CToGmS0<decltype(tensorP)> copyP;
        copyP(tensorP, tensorL0C, 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(pipelineSlot));
    }

    __aicore__ inline void RunStage0(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage0：H_c=cast_BF16(R_c)，P_c=W_c@H_c；Cube 使用 BF16 输入、FP32 累加。
        if (chunk.first && args_.tiling.useInitialState == 0) {
            return;
        }
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        const bool waitHReady = !(chunk.first && args_.tiling.useInitialState != 0 &&
                                  !CompilePolicy::STATE_FP32);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (waitHReady) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                    FwdHAicPeerFlag(FWD_H_H_READY_FLAG, pairSlot, 0));
            }
            if (!chunk.first) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(
                    FwdHAicPeerFlag(FWD_H_P_FREE_FLAG, pairSlot, 0));
            }
            const uint32_t firstHead = pairSlot * 2U;
            const uint32_t pairHeads =
                unit.headRound.activeHeadCount - firstHead > 1U ? 2U : 1U;
            for (uint32_t pairHead = 0; pairHead < pairHeads; ++pairHead) {
                const uint32_t headId = firstHead + pairHead;
                LoadStage0Head(unit, chunk, unit.headRound.heads[headId],
                               wPreloadChunk_ == chunk.chunk);
                if (pairHead > 0) {
                    const uint32_t previous = headId - 1U;
                    ComputeStage0Head(chunk, unit.headRound.heads[previous], previous & 1U);
                }
            }
            const uint32_t last = firstHead + pairHeads - 1U;
            ComputeStage0Head(chunk, unit.headRound.heads[last], last & 1U);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                FwdHAicPeerFlag(FWD_H_P_READY_FLAG, pairSlot, 0));
        }
    }

    __aicore__ inline void LoadKg(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                  const FwdHKgBinding &binding)
    {
        // Stage2 搬运：按 requiredKhCount 加载当前 chunk/round 的实际 k_raw/kg，不跨 round 缓存。
        const uint32_t slot = binding.slot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));
        if (chunk.validTokens < FWD_H_CHUNK) {
            ClearL1(l1Kg_[slot], FWD_H_L1_KG_SLOT_BYTES);
        }
        AscendC::GlobalTensor<bfloat16_t> gmKg;
        gmKg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.k) +
            FwdHKOffset(args_.tiling, unit.sequence.physicalBatch, binding.kh, chunk.tokenBegin));
        if constexpr (!STATE_V_FIRST) {
            auto tensorGm = tla::MakeTensor(gmKg,
                tla::MakeLayout<bfloat16_t, LayoutLeftS2>(FWD_H_K, chunk.validTokens),
                Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(FWD_H_K, chunk.validTokens));
            auto tensorL1 = tla::MakeTensor(l1Kg_[slot], L1_LEFT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1AS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        } else {
            auto tensorGm = tla::MakeTensor(gmKg,
                tla::MakeLayout<bfloat16_t, LayoutRightS2>(chunk.validTokens, FWD_H_K),
                Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(chunk.validTokens, FWD_H_K));
            auto tensorL1 = tla::MakeTensor(l1Kg_[slot], L1_RIGHT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1BS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(WDoneEvent(slot));
    }

    __aicore__ inline void LoadRight(const FwdHChunkSpan &chunk, const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.roundHead;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(HRightReadyEvent(slot));
        if (chunk.validTokens < FWD_H_CHUNK) {
            ClearL1(l1HRight_[slot], FWD_H_L1_KG_SLOT_BYTES);
        }
        const uint64_t rightOffset = args_.tiling.vUpdateWorkspaceOffset / sizeof(bfloat16_t) +
            FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_CHUNK * FWD_H_V);
        AscendC::GlobalTensor<bfloat16_t> gmRight;
        gmRight.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace) + rightOffset);
        if constexpr (!STATE_V_FIRST) {
            auto tensorGm = tla::MakeTensor(gmRight,
                tla::MakeLayout<bfloat16_t, LayoutRightS2>(chunk.validTokens, FWD_H_V),
                Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(chunk.validTokens, FWD_H_V));
            auto tensorL1 = tla::MakeTensor(l1HRight_[slot], L1_RIGHT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1BS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        } else {
            auto tensorGm = tla::MakeTensor(gmRight,
                tla::MakeLayout<bfloat16_t, LayoutLeftS2>(FWD_H_V, chunk.validTokens),
                Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(FWD_H_V, chunk.validTokens));
            auto tensorL1 = tla::MakeTensor(l1HRight_[slot], L1_LEFT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1AS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(HRightDoneEvent(slot));
    }

    __aicore__ inline void ComputeStage2Head(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head,
                                             uint32_t pipelineSlot)
    {
        // Stage2 计算：state_v_first=false 时 D=kg^T@right，输出 [K,V]；
        // state_v_first=true 时交换两个输入，计算 right^T@kg，输出物理 [V,K]；
        // A2/A3 的 FP32 D 经 Fixpipe 写 GM scratch。
        const FwdHKgBinding binding = FwdHBuildKgBinding(unit.headRound, head.kgSlot);
        const uint32_t m = FWD_H_K;
        const uint32_t k = FwdHAlignCube(chunk.validTokens);
        auto tensorL0A = tla::MakeTensor(l0A_[pipelineSlot],
            tla::MakeLayout<bfloat16_t, typename TileS2::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B_[pipelineSlot],
            tla::MakeLayout<bfloat16_t, typename TileS2::LayoutTagL0B>(k, FWD_H_V),
            Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C_[pipelineSlot], tla::MakeLayoutL0C(m, FWD_H_V),
                                         Catlass::Arch::PositionL0C{});
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(pipelineSlot));
        if (head.roundHead == binding.firstConsumer) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(WDoneEvent(binding.slot));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(HRightDoneEvent(head.roundHead));
        CopyL1ToL0AS2 copyA;
        CopyL1ToL0BS2 copyB;
        if constexpr (STATE_V_FIRST) {
            // LoadRight 已将 GM ND right[M,V] 按 ColumnMajor 映射为 L1[V,M]，
            // kg 保持 GM ND[M,K]，Cube 直接执行 [V,M]@[M,K] -> [V,K]。
            auto tensorRight = tla::MakeTensor(l1HRight_[head.roundHead], L1_LEFT_S2_LAYOUT,
                                               Catlass::Arch::PositionL1{});
            auto tensorKg = tla::MakeTensor(l1Kg_[binding.slot], L1_RIGHT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            copyA(tensorL0A, tla::GetTile(tensorRight, tla::MakeCoord(0, 0),
                                          tla::MakeShape(m, k)));
            copyB(tensorL0B, tla::GetTile(tensorKg, tla::MakeCoord(0, 0),
                                          tla::MakeShape(k, FWD_H_V)));
        } else {
            auto tensorKg = tla::MakeTensor(l1Kg_[binding.slot], L1_LEFT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            auto tensorRight = tla::MakeTensor(l1HRight_[head.roundHead], L1_RIGHT_S2_LAYOUT,
                                               Catlass::Arch::PositionL1{});
            copyA(tensorL0A, tla::GetTile(tensorKg, tla::MakeCoord(0, 0),
                                          tla::MakeShape(m, k)));
            copyB(tensorL0B, tla::GetTile(tensorRight, tla::MakeCoord(0, 0),
                                          tla::MakeShape(k, FWD_H_V)));
        }
        if (head.roundHead == binding.lastConsumer) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(binding.slot));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(HRightReadyEvent(head.roundHead));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0AReadyEvent(pipelineSlot));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0BReadyEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0AReadyEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0BReadyEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(pipelineSlot));
        TileMmadS2 mmad;
        mmad(tensorL0C, tensorL0A, tensorL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(pipelineSlot));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(pipelineSlot));
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(FixDoneEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(FixDoneEvent(pipelineSlot));
        const uint64_t dByteOffset = ScratchByteOffset(args_.tiling.hWorkspaceOffset,
                                                       head.roundHead,
                                                       FWD_H_STATE_FP32_BYTES);
        AscendC::GlobalTensor<float> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + dByteOffset));
        auto tensorD = tla::MakeTensor(gmD,
            tla::MakeLayout<float, LayoutOutput>(FWD_H_K, FWD_H_V), Catlass::Arch::PositionGM{});
        CopyL0CToGmS2<decltype(tensorD)> copyD;
        copyD(tensorD, tensorL0C, 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(pipelineSlot));
    }

    __aicore__ inline void RunStage2(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage2：g-only 计算 D_c=k_raw_c^T@V_new_g；gk-only 计算 D_c=kg_c^T@V_new。
        for (uint32_t kgSlot = 0; kgSlot < unit.headRound.requiredKhCount; ++kgSlot) {
            LoadKg(unit, chunk, FwdHBuildKgBinding(unit.headRound, kgSlot));
        }
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                FwdHAicPeerFlag(FWD_H_RIGHT_READY_FLAG, pairSlot, 0));
            if (!chunk.first) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(
                    FwdHAicPeerFlag(FWD_H_D_FREE_FLAG, pairSlot, 0));
            }
            const uint32_t firstHead = pairSlot * 2U;
            const uint32_t pairHeads =
                unit.headRound.activeHeadCount - firstHead > 1U ? 2U : 1U;
            for (uint32_t pairHead = 0; pairHead < pairHeads; ++pairHead) {
                const uint32_t headId = firstHead + pairHead;
                LoadRight(chunk, unit.headRound.heads[headId]);
                if (pairHead > 0) {
                    const uint32_t previous = headId - 1U;
                    ComputeStage2Head(unit, chunk, unit.headRound.heads[previous], previous & 1U);
                }
            }
            const uint32_t last = firstHead + pairHeads - 1U;
            ComputeStage2Head(unit, chunk, unit.headRound.heads[last], last & 1U);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(
                FwdHAicPeerFlag(FWD_H_RIGHT_FREE_FLAG, pairSlot, 0));
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                FwdHAicPeerFlag(FWD_H_D_READY_FLAG, pairSlot, 0));
        }
        PrefetchNextW(unit, chunk);
    }

    __aicore__ inline void PrefetchNextW(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // W(c+1) 预取：S2 right 装载发出后发射（MTE2 FIFO 不越队），与 S2 计算及
        // S3 窗口重叠；wait/set 记账 1:1 不变，跨 work unit 边界不预取。
        const uint32_t nextChunk = chunk.chunk + 1U;
        if (nextChunk >= unit.sequence.chunkCount) {
            return;
        }
        const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, nextChunk);
        for (uint32_t headId = 0; headId < unit.headRound.activeHeadCount; ++headId) {
            const FwdHHeadBinding &head = unit.headRound.heads[headId];
            const uint32_t slot = head.roundHead;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(WReadyEvent(slot));
            AscendC::GlobalTensor<bfloat16_t> gmW;
            gmW.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.w) +
                WOffset(unit, next, head));
            if (next.validTokens < FWD_H_CHUNK) {
                ClearL1(l1W_[slot], FWD_H_L1_W_SLOT_BYTES);
            }
            auto gmWLayout = tla::MakeLayout<bfloat16_t, LayoutW>(next.validTokens, FWD_H_K);
            auto tensorW = tla::MakeTensor(gmW, gmWLayout, Catlass::Arch::PositionGM{});
            auto blockW = tla::GetTile(tensorW, tla::MakeCoord(0, 0),
                                       tla::MakeShape(next.validTokens, FWD_H_K));
            auto tensorL1W = tla::MakeTensor(l1W_[slot], L1_W_LAYOUT, Catlass::Arch::PositionL1{});
            CopyGmToL1AS0<decltype(blockW)> copyW;
            copyW(tensorL1W, blockW);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(WDoneEvent(slot));
        }
        wPreloadChunk_ = nextChunk;
    }

    __aicore__ inline void ProcessWorkUnit(const FwdHWorkUnit &unit, bool hasNextWorkUnit)
    {
        if (!hasCubeWork_) {
            return;
        }
        // chunk 编号每 work unit 从 0 重启，预取标记须同步复位。
        wPreloadChunk_ = NO_W_PRELOAD;
        for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
            const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
            RunStage0(unit, chunk);
            if (CompilePolicy::STORE_FINAL || !chunk.last) {
                RunStage2(unit, chunk);
            }
        }
        const bool terminalStage2 = CompilePolicy::STORE_FINAL ||
            unit.sequence.chunkCount > 1;
        const bool terminalStage0 = args_.tiling.useInitialState != 0 || unit.sequence.chunkCount > 1;
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (terminalStage2) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(
                    FwdHAicPeerFlag(FWD_H_D_FREE_FLAG, pairSlot, 0));
            }
            if (terminalStage0) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(
                    FwdHAicPeerFlag(FWD_H_P_FREE_FLAG, pairSlot, 0));
            }
        }
        if (hasNextWorkUnit) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_FIX>(
                FwdHAicPeerFlag(FWD_H_ROUND_DONE_FLAG, 0, 0));
            // ACK 表示 AIC 已完成本轮消费；AIV 收到后才可复用下一轮的本地槽。
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(
                FwdHAicPeerFlag(FWD_H_ROUND_ACK_FLAG, 0, 0));
        }
    }

    FwdHKernelArgs args_{};
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    bool hasCubeWork_ = false;
    static constexpr uint32_t NO_W_PRELOAD = 0xFFFFFFFFU;
    uint32_t wPreloadChunk_ = NO_W_PRELOAD;
    AscendC::TBuf<AscendC::TPosition::A1> l1Buf_{};
    AscendC::TBuf<AscendC::TPosition::A2> l0ABuf_{};
    AscendC::TBuf<AscendC::TPosition::B2> l0BBuf_{};
    AscendC::TBuf<AscendC::TPosition::CO1> l0CBuf_{};
    AscendC::TBuf<AscendC::TPosition::C2PIPE2GM> fixBuf_{};
    AscendC::TEventID wReadyEvent_[FWD_H_AIC_HEAD_SLOTS]{};
    AscendC::TEventID wDoneEvent_[FWD_H_AIC_HEAD_SLOTS]{};
    AscendC::TEventID hRightReadyEvent_[FWD_H_AIC_HEAD_SLOTS]{};
    AscendC::TEventID hRightDoneEvent_[FWD_H_AIC_HEAD_SLOTS]{};
    AscendC::TEventID l0AFreeEvent_[L0_SLOTS]{};
    AscendC::TEventID l0AReadyEvent_[L0_SLOTS]{};
    AscendC::TEventID l0BFreeEvent_[L0_SLOTS]{};
    AscendC::TEventID l0BReadyEvent_[L0_SLOTS]{};
    AscendC::TEventID fixFreeEvent_[L0_SLOTS]{};
    AscendC::TEventID fixDoneEvent_[L0_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> l1W_[FWD_H_AIC_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> l1HRight_[FWD_H_AIC_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> l1Kg_[FWD_H_AIC_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> l0A_[L0_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> l0B_[L0_SLOTS]{};
    AscendC::LocalTensor<ElementAccumulator> l0C_[L0_SLOTS]{};
};

} // namespace GDN

// ---- gated 复用引擎（AIC 侧）：自 arch22/chunk_fwd_h_utils.h 的
// GDNFwdHKernelPreload 拆出（Init 保留 AIC 调度器分支，Process = 原
// ProcessCube 体；纯搬移，指令序列不变）。
namespace Catlass::Gemm::Kernel {

using namespace Catlass;
using namespace tla;

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE,
    bool USE_EXP2_
>
class ChunkFwdHGatedCubeArch22 {
public:
    
    using ArchTag = Arch::AtlasA2;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHPreloadCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHPreloadVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaPreload<ArchTag, true, false>;
    using L1TileShapeTla = Shape<_128, _128, _128>;
    using L0TileShapeTla = L1TileShapeTla;

    using WType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;

    // cube 1
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;

    // V1/V2 epilogue 已解体入 vec 引擎（本类无 epilogue 别名）。

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsetsPreload;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;
    
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;

    
    uint32_t batch{0};
    uint32_t seqlen{0};
    uint32_t kNumHead{0};
    uint32_t vNumHead{0};
    uint32_t kHeadDim{0};
    uint32_t vHeadDim{0};
    uint32_t chunkSize{0};
    bool useInitialState{false};
    bool storeFinalState{false};
    uint32_t isVariedLen{0};
    uint32_t shapeBatch{0};
    uint32_t tokenBatch{0};
    uint64_t vWorkspaceOffset{0};
    uint64_t vUpdateWorkspaceOffset{0};
    uint64_t hWorkspaceOffset{0};
    uint64_t numSeqWorkspaceOffset{0};
    uint64_t numChunksWorkspaceOffset{0};
    
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;
    
    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    CubeScheduler cubeBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline ChunkFwdHGatedCubeArch22() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, 
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {
        
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = gdnFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = gdnFwdHTilingData->hWorkspaceOffset;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;
        
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }
    }

    __aicore__ inline void Process()
    {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadWH blockMmadWH(resource);
            BlockMmadKV blockMmadKV(resource);

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(coreNum * kHeadDim * PING_PONG_STAGES, vHeadDim);
            AscendC::SyncAll<false>();    
            uint32_t currStage = 0; // 0: C1, 1: C2
            blockMmadWH.preSetFlags();
            while (cubeBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* C1: v_work = w @ h[i] */                    
                    cubeBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }

                        const GDNFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                        int64_t cube1OffsetW = cube1Offsets.wOffset;
                        int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                        int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                        auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                        auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                        auto tensorV = tla::MakeTensor(gmVWorkspace[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                        GemmCoord cube1Shape {cube1Offsets.blockTokens, vHeadDim, kHeadDim};
                        auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                        auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                        auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                        blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape, cubeBlockScheduler.vec2Done[i]);
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);
                    }
                } else {
                    /* C2: h[i+1] = k.T @ v_work */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);

                        if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                            // step 3: h[i+1] = k.T @ v_work
                            int64_t cube2OffsetK = cube2Offsets.wkOffset;
                            int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                            int64_t cube2OffsetH = cube2Offsets.hWorkOffset;
                            auto tensorK = tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                            auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vworkLayout, Catlass::Arch::PositionGM{});
                            auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2OffsetH], hworkLayout, Catlass::Arch::PositionGM{});
                            GemmCoord cube2Shape{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
                            auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                            auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                            auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                            blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape, cubeBlockScheduler.vec1Done);
                        } else {
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
                    }
                }
                currStage ^= 0x01;
            }
            blockMmadKV.finalWaitFlags();
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);

    }

};

} // namespace Catlass::Gemm::Kernel

#endif // ARCH22_CHUNK_FWD_H_CUBE_H
