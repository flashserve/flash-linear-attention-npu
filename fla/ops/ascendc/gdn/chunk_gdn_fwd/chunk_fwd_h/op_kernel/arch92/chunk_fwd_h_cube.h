/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH92_CHUNK_FWD_H_CUBE_H
#define ARCH92_CHUNK_FWD_H_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 9201
#endif

#include <type_traits>

#include "../chunk_fwd_h_policy.h"
#include "../chunk_fwd_h_utils.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN {

template <typename CompilePolicy, bool STATE_V_FIRST>
class ChunkFwdHCubeArch92 {
public:
    __aicore__ inline void Init(const FwdHKernelArgs &args)
    {
        args_ = args;
        coreIdx_ = AscendC::GetBlockIdx();
        coreNum_ = AscendC::GetBlockNum();
        hasCubeWork_ = args_.tiling.useInitialState || args_.tiling.storeFinalState ||
            args_.tiling.seqlen > static_cast<int64_t>(FWD_H_CHUNK);
        if (!hasCubeWork_) {
            return;
        }
        InitEvents();
    }

    __aicore__ inline void Process()
    {
        const FwdHCoreHeadRange range =
            FwdHResolveCoreHeadRange(args_.tiling, coreIdx_, coreNum_);
        const uint32_t headsPerSequence = static_cast<uint32_t>(args_.tiling.vNumHead);
        uint32_t cachedSequence = FwdHSequenceCount(args_.tiling);
        FwdHSequenceSpan sequenceSpan{};
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
                sequenceSpan = FwdHResolveSequence(args_, sequence);
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
    using ArchTag = Catlass::Arch::Ascend960DT;
    using PType = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = std::conditional_t<STATE_V_FIRST, Catlass::layout::ColumnMajor,
                                       Catlass::layout::RowMajor>;
    using LayoutLeftS2 = Catlass::layout::ColumnMajor;
    using LayoutRightS2 = Catlass::layout::RowMajor;
    using LayoutOutput = Catlass::layout::RowMajor;

    using TileS0 = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, bfloat16_t, LayoutW, bfloat16_t, LayoutH, PType, LayoutOutput>;
    using TileS2 = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, bfloat16_t, LayoutLeftS2, bfloat16_t, LayoutRightS2, float, LayoutOutput>;
    using ElementAccumulator = typename TileS2::ElementAccumulator;

    __aicore__ inline uint32_t DataSlot(const FwdHHeadBinding &head) const
    {
        return head.localSlot;
    }

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
    using CopyL0CToUbS0 = typename TileS0::template CopyL0CToDst<Tensor>;
    template <typename Tensor>
    using CopyL0CToUbS2 = typename TileS2::template CopyL0CToDst<Tensor>;

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
    static_assert(FwdHArch92::L1_USED <= ArchTag::L1_SIZE);
    static_assert(FwdHArch92::UB_USED <= ArchTag::UB_SIZE);
    static constexpr uint32_t L0A_SLOT_BYTES = FWD_H_CHUNK * FWD_H_K * sizeof(bfloat16_t);
    static constexpr uint32_t L0B_SLOT_BYTES = FWD_H_K * FWD_H_V * sizeof(bfloat16_t);
    static constexpr uint32_t L0C_SLOT_BYTES = FWD_H_K * FWD_H_V * sizeof(ElementAccumulator);

    // MTE1_MTE2: 0/1 = whole-bank free. MTE2_MTE1: 0/1 = K ready,
    // 2..5 = bank/head operands ready (S0 then S2). Never use reserved IDs 6/7.
    __aicore__ inline AscendC::TEventID BankEvent(uint32_t bank) const
    {
        return static_cast<AscendC::TEventID>(bank);
    }

    __aicore__ inline AscendC::TEventID OperandEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(2 + slot);
    }

    __aicore__ inline AscendC::TEventID L0AFreeEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::TEventID L0AReadyEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::TEventID L0BFreeEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(L0_SLOTS + slot);
    }

    __aicore__ inline AscendC::TEventID L0BReadyEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(L0_SLOTS + slot);
    }

    __aicore__ inline AscendC::TEventID FixFreeEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::TEventID FixDoneEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> L1Buffer() const
    {
        return AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A1, 0, ArchTag::L1_SIZE);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> L0ABuffer() const
    {
        return AscendC::LocalTensor<uint8_t>(AscendC::TPosition::A2, 0, ArchTag::L0A_SIZE);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> L0BBuffer() const
    {
        return AscendC::LocalTensor<uint8_t>(AscendC::TPosition::B2, 0, ArchTag::L0B_SIZE);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> L0CBuffer() const
    {
        return AscendC::LocalTensor<uint8_t>(AscendC::TPosition::CO1, 0, ArchTag::L0C_SIZE);
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> UbBuffer() const
    {
        return FwdHArch92::UbBuffer();
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1At(uint32_t slot,
        uint32_t base, uint32_t bytes)
    {
        return L1Buffer()[(slot / FwdHArch92::HEADS) * FwdHArch92::L1_BANK_BYTES +
                          base + (slot % FwdHArch92::HEADS) * bytes]
            .template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1W(uint32_t slot)
    {
        return L1At(slot, FwdHArch92::L1_W_BASE, FWD_H_TOKEN_MATRIX_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1H(uint32_t slot)
    {
        return L1At(slot, FwdHArch92::L1_H_BASE, FWD_H_STATE_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1Right(uint32_t slot)
    {
        return L1At(slot, FwdHArch92::L1_RIGHT_BASE, FWD_H_TOKEN_MATRIX_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L1Kg(uint32_t slot)
    {
        return L1At(slot, FwdHArch92::L1_K_BASE, FWD_H_TOKEN_MATRIX_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L0A(uint32_t slot)
    {
        return L0ABuffer()[slot * L0A_SLOT_BYTES].template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> L0B(uint32_t slot)
    {
        return L0BBuffer()[slot * L0B_SLOT_BYTES].template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline AscendC::LocalTensor<ElementAccumulator> L0C(uint32_t slot)
    {
        return L0CBuffer()[slot * L0C_SLOT_BYTES]
            .template ReinterpretCast<ElementAccumulator>();
    }

    __aicore__ inline void InitEvents()
    {
        for (uint32_t bank = 0; bank < FwdHArch92::BANKS; ++bank) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BankEvent(bank));
        }
        for (uint32_t slot = 0; slot < L0_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(slot));
        }
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t bank = 0; bank < FwdHArch92::BANKS; ++bank) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BankEvent(bank));
        }
        for (uint32_t slot = 0; slot < L0_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(slot));
        }
        // 固定 EventID 已全部回收到初始 free 状态，下一次 kernel launch 可重新使用。
    }

    __aicore__ inline uint64_t WOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, FWD_H_K);
    }

    __aicore__ inline void ClearL1(AscendC::LocalTensor<bfloat16_t> tensor,
                                   uint32_t bytes) const
    {
        AscendC::InitConstValueParams<bfloat16_t> params(
            1, static_cast<uint16_t>(bytes / 32U), 0, static_cast<bfloat16_t>(0));
        AscendC::InitConstValue(tensor, params);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline uint64_t HOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHHOffset(args_.tiling, unit.sequence, head.hv, chunk.globalChunk);
    }

    // 保留 leaf stage 调用边界，使 TLA 临时对象在各 head/stage 之间复用标量栈区。
    __aicore__ inline void LoadStage0W(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head,
        uint32_t wSlot)
    {
        // W has a dedicated L1 slot. The combined operand-ready is published after H.
        AscendC::GlobalTensor<bfloat16_t> gmW;
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.w) + WOffset(unit, chunk, head));
        gmW.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (chunk.validTokens < FWD_H_CHUNK) {
            ClearL1(L1W(wSlot), FWD_H_L1_W_SLOT_BYTES);
        }
        auto gmWLayout = tla::MakeLayout<bfloat16_t, LayoutW>(chunk.validTokens, FWD_H_K);
        auto tensorW = tla::MakeTensor(gmW, gmWLayout, Catlass::Arch::PositionGM{});
        auto blockW = tla::GetTile(tensorW, tla::MakeCoord(0, 0),
                                   tla::MakeShape(chunk.validTokens, FWD_H_K));
        auto tensorL1W = tla::MakeTensor(L1W(wSlot), L1_W_LAYOUT, Catlass::Arch::PositionL1{});
        CopyGmToL1AS0<decltype(blockW)> copyW;
        copyW(tensorL1W, blockW);
    }

    __aicore__ inline void LoadStage0H(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head,
        uint32_t hSlot)
    {
        AscendC::GlobalTensor<bfloat16_t> gmH;
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(FwdHArch92::H_FLAG + head.localSlot);
        gmH.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h) + HOffset(unit, chunk, head));
        auto gmHLayout = tla::MakeLayout<bfloat16_t, LayoutH>(FWD_H_K, FWD_H_V);
        auto tensorH = tla::MakeTensor(gmH, gmHLayout, Catlass::Arch::PositionGM{});
        auto blockH = tla::GetTile(tensorH, tla::MakeCoord(0, 0),
                                   tla::MakeShape(FWD_H_K, FWD_H_V));
        auto tensorL1H = tla::MakeTensor(L1H(hSlot), L1_H_LAYOUT, Catlass::Arch::PositionL1{});
        CopyGmToL1BS0<decltype(blockH)> copyH;
        copyH(tensorL1H, blockH);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(OperandEvent(hSlot));
    }

    __aicore__ inline void ComputeStage0Head(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head,
        uint32_t pipelineSlot, uint32_t wSlot, uint32_t hSlot)
    {
        // Stage0 计算：Pacc_c,h = W_c,h @ H_c,h，BF16 x BF16 -> FP32；
        // 随后按 StateT 转为 PType，Fixpipe 直接写入该 head 所属 AIV 的 local slot。
        const uint32_t m = FwdHAlignCube(chunk.validTokens);
        auto tensorL1W = tla::MakeTensor(L1W(wSlot), L1_W_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1H = tla::MakeTensor(L1H(hSlot), L1_H_LAYOUT, Catlass::Arch::PositionL1{});
        auto layoutL0A = tla::MakeLayout<bfloat16_t, typename TileS0::LayoutTagL0A>(m, FWD_H_K);
        auto layoutL0B = tla::MakeLayout<bfloat16_t, typename TileS0::LayoutTagL0B>(FWD_H_K, FWD_H_V);
        auto layoutL0C = tla::MakeLayoutL0C(m, FWD_H_V);
        auto tensorL0A = tla::MakeTensor(L0A(pipelineSlot), layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(L0B(pipelineSlot), layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(L0C(pipelineSlot), layoutL0C, Catlass::Arch::PositionL0C{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(OperandEvent(hSlot));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(pipelineSlot));
        CopyL1ToL0AS0 copyA;
        CopyL1ToL0BS0 copyB;
        copyA(tensorL0A, tla::GetTile(tensorL1W, tla::MakeCoord(0, 0), tla::MakeShape(m, FWD_H_K)));
        copyB(tensorL0B, tla::GetTile(tensorL1H, tla::MakeCoord(0, 0),
                                     tla::MakeShape(FWD_H_K, FWD_H_V)));
        if (!NeedsStage2(chunk) && head.roundHead + 1 == unit.headRound.activeHeadCount) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BankEvent(chunk.chunk & 1U));
        }
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

        if (!chunk.first) {
            // Only this head's hNext MTE3 owns its PD slot until PD_FREE.
            AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(FwdHArch92::D_FLAG + head.localSlot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(FixDoneEvent(pipelineSlot));
        AscendC::LocalTensor<PType> pUb = UbBuffer()[
            FwdHLocalSlotBase(DataSlot(head))].template ReinterpretCast<PType>();
        auto ubLayout = tla::MakeLayout<PType, LayoutOutput>(m, FWD_H_V);
        auto tensorUb = tla::MakeTensor(pUb, ubLayout, Catlass::Arch::PositionUB{});
        auto blockUb = tla::GetTile(tensorUb, tla::MakeCoord(0, 0),
                                    tla::MakeShape(chunk.validTokens, FWD_H_V));
        CopyL0CToUbS0<decltype(blockUb)> copyP;
        copyP(blockUb, tensorL0C, static_cast<uint8_t>(head.aiv), 0);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
            FwdHArch92::P_FLAG + head.localSlot);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(pipelineSlot));
    }

    __aicore__ inline void RunStage0(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage0：H_c=cast_BF16(R_c)，P_c=W_c@H_c；Cube 使用 BF16 输入、FP32 累加。
        if (chunk.first && args_.tiling.useInitialState == 0) {
            return;
        }
        for (uint32_t roundHead = 0; roundHead < unit.headRound.activeHeadCount; ++roundHead) {
            const FwdHHeadBinding &head = unit.headRound.heads[roundHead];
            const uint32_t slot = FwdHArch92::BankSlot(chunk.chunk, roundHead);
            // Startup keeps W/H contiguous for each head. Later W was prefetched.
            if (chunk.first) {
                LoadStage0W(unit, chunk, head, slot);
            }
            LoadStage0H(unit, chunk, head, slot);
            if (roundHead > 0) {
                ComputeStage0Head(unit, chunk, unit.headRound.heads[roundHead - 1],
                                  (roundHead - 1) & 1U, slot - 1, slot - 1);
            }
        }
        const uint32_t last = unit.headRound.activeHeadCount - 1;
        const uint32_t slot = FwdHArch92::BankSlot(chunk.chunk, last);
        ComputeStage0Head(unit, chunk, unit.headRound.heads[last], last & 1U, slot, slot);
    }

    __aicore__ inline void LoadKg(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHKgBinding &binding)
    {
        // Each bank loads only the distinct kh required by this work unit.
        const uint32_t slot = FwdHArch92::BankSlot(chunk.chunk, binding.slot);
        if (chunk.validTokens < FWD_H_CHUNK) {
            ClearL1(L1Kg(slot), FWD_H_L1_KG_SLOT_BYTES);
        }
        AscendC::GlobalTensor<bfloat16_t> gmKg;
        const uint64_t offset = FwdHKOffset(args_.tiling, unit.sequence.physicalBatch,
                                            binding.kh, chunk.tokenBegin);
        gmKg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.k) + offset);
        bool readOnce = true;
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            const uint32_t groupSize = static_cast<uint32_t>(args_.tiling.vNumHead) /
                                       static_cast<uint32_t>(args_.tiling.kNumHead);
            const uint32_t groupBegin = binding.kh * groupSize;
            const uint32_t unitBegin = unit.headRound.heads[0].hv;
            const uint32_t unitEnd = unitBegin + unit.headRound.activeHeadCount;
            readOnce = unitBegin <= groupBegin && unitEnd >= groupBegin + groupSize;
        }
        if (readOnce) {
            gmKg.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if constexpr (!STATE_V_FIRST) {
            auto gmLayout = tla::MakeLayout<bfloat16_t, LayoutLeftS2>(FWD_H_K, chunk.validTokens);
            auto tensorGm = tla::MakeTensor(gmKg, gmLayout, Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(FWD_H_K, chunk.validTokens));
            auto tensorL1 = tla::MakeTensor(L1Kg(slot), L1_LEFT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1AS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        } else {
            auto gmLayout = tla::MakeLayout<bfloat16_t, LayoutRightS2>(chunk.validTokens, FWD_H_K);
            auto tensorGm = tla::MakeTensor(gmKg, gmLayout, Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(chunk.validTokens, FWD_H_K));
            auto tensorL1 = tla::MakeTensor(L1Kg(slot), L1_RIGHT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1BS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        }
    }

    __aicore__ inline void LoadRight(
        const FwdHChunkSpan &chunk, const FwdHHeadBinding &head, uint32_t hSlot)
    {
        // The right slot is separate from H, and its ready belongs to this bank/head.
        AscendC::CrossCoreWaitFlag<0x4, PIPE_MTE2>(
            FwdHArch92::RIGHT_FLAG + FwdHArch92::BankSlot(chunk.chunk, head.localSlot));
        if (chunk.validTokens < FWD_H_CHUNK) {
            ClearL1(L1Right(hSlot), FWD_H_L1_KG_SLOT_BYTES);
        }
        AscendC::GlobalTensor<bfloat16_t> gmRight;
        const uint64_t offset = args_.tiling.vUpdateWorkspaceOffset / sizeof(bfloat16_t) +
            FwdHArch92::RightOffset(coreIdx_, chunk.chunk, head.localSlot);
        gmRight.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace) + offset);
        if constexpr (!STATE_V_FIRST) {
            auto gmLayout = tla::MakeLayout<bfloat16_t, LayoutRightS2>(chunk.validTokens, FWD_H_V);
            auto tensorGm = tla::MakeTensor(gmRight, gmLayout, Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(chunk.validTokens, FWD_H_V));
            auto tensorL1 = tla::MakeTensor(L1Right(hSlot), L1_RIGHT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1BS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        } else {
            auto gmLayout = tla::MakeLayout<bfloat16_t, LayoutLeftS2>(FWD_H_V, chunk.validTokens);
            auto tensorGm = tla::MakeTensor(gmRight, gmLayout, Catlass::Arch::PositionGM{});
            auto blockGm = tla::GetTile(tensorGm, tla::MakeCoord(0, 0),
                                        tla::MakeShape(FWD_H_V, chunk.validTokens));
            auto tensorL1 = tla::MakeTensor(L1Right(hSlot), L1_LEFT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            CopyGmToL1AS2<decltype(blockGm)> copy;
            copy(tensorL1, blockGm);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(OperandEvent(hSlot));
    }

    __aicore__ inline void ComputeStage2Head(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head,
        const FwdHKgBinding &binding, uint32_t hSlot, uint32_t pipelineSlot, bool stage0Ran)
    {
        // Stage2 计算：state_v_first=false 时 D=kg^T@right，输出 [K,V]；
        // state_v_first=true 时交换两个输入，计算 right^T@kg，输出物理 [V,K]。
        const uint32_t m = FWD_H_K;
        const uint32_t k = FwdHAlignCube(chunk.validTokens);
        auto layoutL0A = tla::MakeLayout<bfloat16_t, typename TileS2::LayoutTagL0A>(m, k);
        auto layoutL0B = tla::MakeLayout<bfloat16_t, typename TileS2::LayoutTagL0B>(k, FWD_H_V);
        auto layoutL0C = tla::MakeLayoutL0C(m, FWD_H_V);
        auto tensorL0A = tla::MakeTensor(L0A(pipelineSlot), layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(L0B(pipelineSlot), layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(L0C(pipelineSlot), layoutL0C, Catlass::Arch::PositionL0C{});

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0AFreeEvent(pipelineSlot));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BFreeEvent(pipelineSlot));
        if (head.roundHead == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(BankEvent(chunk.chunk & 1U));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(OperandEvent(hSlot));
        CopyL1ToL0AS2 copyA;
        CopyL1ToL0BS2 copyB;
        if constexpr (STATE_V_FIRST) {
            // LoadRight 已将 GM ND right[M,V] 按 ColumnMajor 映射为 L1[V,M]，
            // kg 保持 GM ND[M,K]，Cube 直接执行 [V,M]@[M,K] -> [V,K]。
            auto tensorRight = tla::MakeTensor(L1Right(hSlot), L1_LEFT_S2_LAYOUT,
                                               Catlass::Arch::PositionL1{});
            auto tensorKg = tla::MakeTensor(L1Kg(FwdHArch92::BankSlot(chunk.chunk, binding.slot)),
                                            L1_RIGHT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            copyA(tensorL0A, tla::GetTile(tensorRight, tla::MakeCoord(0, 0),
                                          tla::MakeShape(m, k)));
            copyB(tensorL0B, tla::GetTile(tensorKg, tla::MakeCoord(0, 0),
                                          tla::MakeShape(k, FWD_H_V)));
        } else {
            auto tensorKg = tla::MakeTensor(L1Kg(FwdHArch92::BankSlot(chunk.chunk, binding.slot)),
                                            L1_LEFT_S2_LAYOUT,
                                            Catlass::Arch::PositionL1{});
            auto tensorRight = tla::MakeTensor(L1Right(hSlot), L1_RIGHT_S2_LAYOUT,
                                               Catlass::Arch::PositionL1{});
            copyA(tensorL0A, tla::GetTile(tensorKg, tla::MakeCoord(0, 0),
                                          tla::MakeShape(m, k)));
            copyB(tensorL0B, tla::GetTile(tensorRight, tla::MakeCoord(0, 0),
                                          tla::MakeShape(k, FWD_H_V)));
        }
        if (head.roundHead + 1 == unit.headRound.activeHeadCount) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BankEvent(chunk.chunk & 1U));
        }
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE1>(
            FwdHArch92::RIGHT_FLAG + FwdHArch92::BankSlot(chunk.chunk, head.localSlot));
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

        if (stage0Ran) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(FwdHArch92::P_FLAG + head.localSlot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(FixDoneEvent(pipelineSlot));
        AscendC::LocalTensor<float> dUb =
            UbBuffer()[FwdHLocalSlotBase(DataSlot(head))].template ReinterpretCast<float>();
        auto ubLayout = tla::MakeLayout<float, LayoutOutput>(FWD_H_K, FWD_H_V);
        auto tensorUb = tla::MakeTensor(dUb, ubLayout, Catlass::Arch::PositionUB{});
        CopyL0CToUbS2<decltype(tensorUb)> copyD;
        copyD(tensorUb, tensorL0C, static_cast<uint8_t>(head.aiv), 0);
        AscendC::CrossCoreSetFlag<0x4, PIPE_FIX>(
            FwdHArch92::D_FLAG + head.localSlot);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FixFreeEvent(pipelineSlot));
    }

    __aicore__ inline bool NeedsStage2(const FwdHChunkSpan &chunk) const
    {
        return args_.tiling.storeFinalState != 0 || !chunk.last;
    }

    __aicore__ inline void PrefetchK(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        if (!NeedsStage2(chunk)) {
            return;
        }
        for (uint32_t kgSlot = 0; kgSlot < unit.headRound.requiredKhCount; ++kgSlot) {
            LoadKg(unit, chunk, FwdHBuildKgBinding(unit.headRound, kgSlot));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(BankEvent(chunk.chunk & 1U));
    }

    __aicore__ inline void PrefetchWK(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BankEvent(chunk.chunk & 1U));
        for (uint32_t roundHead = 0; roundHead < unit.headRound.activeHeadCount; ++roundHead) {
            LoadStage0W(unit, chunk, unit.headRound.heads[roundHead],
                        FwdHArch92::BankSlot(chunk.chunk, roundHead));
        }
        PrefetchK(unit, chunk);
    }

    __aicore__ inline void RunStage2(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                     bool stage0Ran)
    {
        for (uint32_t roundHead = 0; roundHead < unit.headRound.activeHeadCount; ++roundHead) {
            const FwdHHeadBinding &head = unit.headRound.heads[roundHead];
            LoadRight(chunk, head, FwdHArch92::BankSlot(chunk.chunk, roundHead));
            if (roundHead > 0) {
                const FwdHHeadBinding &previous = unit.headRound.heads[roundHead - 1];
                const FwdHKgBinding binding =
                    FwdHBuildKgBinding(unit.headRound, previous.kgSlot);
                ComputeStage2Head(unit, chunk, previous, binding,
                                  FwdHArch92::BankSlot(chunk.chunk, previous.roundHead),
                                  (roundHead - 1) & 1U, stage0Ran);
            }
        }
        const uint32_t last = unit.headRound.activeHeadCount - 1;
        const FwdHHeadBinding &head = unit.headRound.heads[last];
        const FwdHKgBinding binding = FwdHBuildKgBinding(unit.headRound, head.kgSlot);
        ComputeStage2Head(unit, chunk, head, binding,
                          FwdHArch92::BankSlot(chunk.chunk, head.roundHead), last & 1U, stage0Ran);
    }

    __aicore__ inline void SyncNextWorkUnit(bool hasNextWorkUnit)
    {
        if (hasNextWorkUnit) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(FwdHArch92::ROUND_FLAG);
            AscendC::CrossCoreSetFlag<0x4, PIPE_S>(FwdHArch92::ROUND_FLAG);
        }
    }

    __aicore__ inline void ProcessWorkUnit(const FwdHWorkUnit &unit, bool hasNextWorkUnit)
    {
        if (!hasCubeWork_) {
            return;
        }
        if (unit.sequence.chunkCount == 1 && !args_.tiling.useInitialState &&
            !args_.tiling.storeFinalState) {
            // A short varlen unit has no Cube consumer. Do not acquire/reseed a bank
            // on unrelated pipes: that would permit a second set before the first wait.
            SyncNextWorkUnit(hasNextWorkUnit);
            return;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BankEvent(0));
        for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
            const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
            const bool stage0Ran = !(chunk.first && args_.tiling.useInitialState == 0);
            RunStage0(unit, chunk);
            if (chunk.first) {
                // Do not put K traffic between the startup W/H pairs.
                PrefetchK(unit, chunk);
            }
            if (!chunk.last) {
                // Queue lookahead before RIGHT_READY, which can block MTE2.
                PrefetchWK(unit, FwdHBuildChunk(unit.sequence, chunkId + 1));
            }
            if (NeedsStage2(chunk)) {
                RunStage2(unit, chunk, stage0Ran);
            }
        }
        const bool terminalStage2 = args_.tiling.storeFinalState != 0;
        const bool terminalStage0 = args_.tiling.useInitialState != 0 || unit.sequence.chunkCount > 1;
        for (uint32_t roundHead = 0; roundHead < unit.headRound.activeHeadCount; ++roundHead) {
            if (terminalStage2) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(FwdHArch92::D_FLAG + roundHead);
            } else if (terminalStage0) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_FIX>(FwdHArch92::P_FLAG + roundHead);
            }
        }
        SyncNextWorkUnit(hasNextWorkUnit);
    }

    FwdHKernelArgs args_{};
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    bool hasCubeWork_ = false;
};

} // namespace GDN

#endif // ARCH92_CHUNK_FWD_H_CUBE_H
