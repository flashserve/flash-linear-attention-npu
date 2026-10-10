/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_CHUNK_FWD_H_VEC_H
#define ARCH22_CHUNK_FWD_H_VEC_H

#include <type_traits>

#include "../chunk_fwd_h_policy.h"
#include "../chunk_fwd_h_utils.h"

namespace GDN {

constexpr float FWD_H_ARCH22_LN2 = 0.69314718055994530942f;

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
class ChunkFwdHVecArch22 {
public:
    __aicore__ inline void Init(const FwdHKernelArgs &args)
    {
        args_ = args;
        coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        coreNum_ = AscendC::GetBlockNum();
        aiv_ = AscendC::GetSubBlockIdx();
        InitBuffers();
        InitEvents();
    }

    __aicore__ inline void InitEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            for (uint32_t buf = 0; buf < TILE_BUFS; ++buf) {
                ioFreeEvent_[slot][buf] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
                ioReadyEvent_[slot][buf] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
                ioDoneEvent_[slot][buf] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
            }
            scalarReadEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
            scalarWriteEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::S_V>();
        }
    }

    __aicore__ inline AscendC::TEventID IoFreeEvent(uint32_t slot, uint32_t buf) const
    {
        return ioFreeEvent_[slot][buf];
    }

    __aicore__ inline AscendC::TEventID IoReadyEvent(uint32_t slot, uint32_t buf) const
    {
        return ioReadyEvent_[slot][buf];
    }

    __aicore__ inline AscendC::TEventID IoDoneEvent(uint32_t slot, uint32_t buf) const
    {
        return ioDoneEvent_[slot][buf];
    }

    __aicore__ inline AscendC::TEventID ScalarReadEvent(uint32_t slot) const
    {
        return scalarReadEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID ScalarWriteEvent(uint32_t slot) const
    {
        return scalarWriteEvent_[slot];
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
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            for (uint32_t buf = 0; buf < TILE_BUFS; ++buf) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
            }
        }
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            for (uint32_t buf = 0; buf < TILE_BUFS; ++buf) {
                GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
                GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
                GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
            }
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(ScalarReadEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(ScalarWriteEvent(slot));
        }
    }

private:
    using PType = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
    static constexpr uint32_t TILE_ROWS = 8;
    static constexpr uint32_t TILE_ELEMENTS = TILE_ROWS * FWD_H_V;
    static constexpr uint32_t LOCAL_SLOT_BYTES = 32 * 1024;
    static constexpr uint32_t LOCAL0_BASE = 0;
    static constexpr uint32_t LOCAL1_BASE = LOCAL_SLOT_BYTES;
    static constexpr uint32_t STATE0_BASE = 64 * 1024;
    static constexpr uint32_t STATE1_BASE = 96 * 1024;
    // STATE_FP32 常驻布局：两个 local slot 的 FP32 rolling state 各 64KB，占
    // [64K,192K)；该模板下 stateBf16_ 是死代码，不再初始化（BF16 路径布局不变）。
    static constexpr uint32_t FP32_STATE0_BASE = 64 * 1024;
    static constexpr uint32_t FP32_STATE1_BASE = 128 * 1024;
    // tile 双缓冲布局（每 slot 32K 内）：raw_[0,8K) 双份（FP32 P tile 4K/份）、
    // cast [8K,12K)、calc [12K,16K)、io [16K,20K) 双份（2K/份）、right [20K,24K)
    // 双份（2K/份）、gate/alpha/share [24K,28K)。MTE2(t+1) 只需等 MTE3(t-1)
    // 排空（同 buf 上一次消费者），与 V(t)/MTE3(t) 并行——解除 tile 链全串行。
    static constexpr uint32_t RAW_BASE = 0;
    static constexpr uint32_t RAW_BUF_BYTES = 4 * 1024;
    static constexpr uint32_t CAST_BASE = 8 * 1024;
    static constexpr uint32_t CALC_BASE = 12 * 1024;
    static constexpr uint32_t IO_BASE = 16 * 1024;
    static constexpr uint32_t IO_BUF_ELEMS = TILE_ELEMENTS;
    static constexpr uint32_t RIGHT_BASE = 20 * 1024;
    static constexpr uint32_t RIGHT_BUF_ELEMS = TILE_ELEMENTS;
    static constexpr uint32_t GATE_RAW_BASE = 24 * 1024;
    static constexpr uint32_t GATE_FP32_BASE = 25 * 1024;
    static constexpr uint32_t ALPHA_BASE = 26 * 1024;
    static constexpr uint32_t SHARE_BASE = 27 * 1024;
    static constexpr uint32_t TILE_BUFS = 2;

    __aicore__ inline uint32_t TileBuf(uint32_t row) const
    {
        return (row / TILE_ROWS) & 1U;
    }

    __aicore__ inline uint32_t LocalBase(uint32_t slot) const
    {
        return slot == 0 ? LOCAL0_BASE : LOCAL1_BASE;
    }

    __aicore__ inline uint32_t TileRows(uint32_t rows) const
    {
        return rows < TILE_ROWS ? rows : TILE_ROWS;
    }

    __aicore__ inline void InitBuffers()
    {
        GetTPipePtr()->InitBuffer(ubBuf_, 192 * 1024);
        AscendC::LocalTensor<uint8_t> ub = ubBuf_.Get<uint8_t>();
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            const uint32_t base = LocalBase(slot);
            raw_[slot] = ub[base + RAW_BASE].template ReinterpretCast<uint8_t>();
            cast_[slot] = ub[base + CAST_BASE].template ReinterpretCast<float>();
            calc_[slot] = ub[base + CALC_BASE].template ReinterpretCast<float>();
            for (uint32_t buf = 0; buf < TILE_BUFS; ++buf) {
                io_[slot][buf] = ub[base + IO_BASE + buf * IO_BUF_ELEMS * sizeof(bfloat16_t)]
                                     .template ReinterpretCast<bfloat16_t>();
                right_[slot][buf] = ub[base + RIGHT_BASE + buf * RIGHT_BUF_ELEMS * sizeof(bfloat16_t)]
                                        .template ReinterpretCast<bfloat16_t>();
            }
            gateRaw_[slot] = ub[base + GATE_RAW_BASE].template ReinterpretCast<GateT>();
            gateFp32_[slot] = ub[base + GATE_FP32_BASE].template ReinterpretCast<float>();
            alpha_[slot] = ub[base + ALPHA_BASE].template ReinterpretCast<float>();
            share_[slot] = ub[base + SHARE_BASE].template ReinterpretCast<uint8_t>();
        }
        if constexpr (CompilePolicy::STATE_FP32) {
            stateFp32_[0] = ub[FP32_STATE0_BASE].template ReinterpretCast<float>();
            stateFp32_[1] = ub[FP32_STATE1_BASE].template ReinterpretCast<float>();
        } else {
            stateBf16_[0] = ub[STATE0_BASE].template ReinterpretCast<bfloat16_t>();
            stateBf16_[1] = ub[STATE1_BASE].template ReinterpretCast<bfloat16_t>();
        }
    }

    __aicore__ inline uint64_t UOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head, uint32_t row) const
    {
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin + row, FWD_H_V);
    }

    __aicore__ inline uint64_t GateOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head) const
    {
        const uint32_t dim = CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? 1 : FWD_H_K;
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, dim);
    }

    __aicore__ inline uint64_t HOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHHOffset(args_.tiling, unit.sequence, head.hv, chunk.globalChunk);
    }

    __aicore__ inline uint64_t ScratchByteOffset(int64_t base, uint32_t slot,
                                                  uint32_t slotBytes) const
    {
        return static_cast<uint64_t>(base) +
            (static_cast<uint64_t>(coreIdx_) * FWD_H_AIC_HEAD_SLOTS + slot) * slotBytes;
    }

    __aicore__ inline float ReadScalar(const AscendC::LocalTensor<float> &tensor,
                                       uint32_t index, uint32_t slot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(ScalarReadEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(ScalarReadEvent(slot));
        const float value = tensor.GetValue(index);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(ScalarWriteEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(ScalarWriteEvent(slot));
        return value;
    }

    __aicore__ inline void PrepareScalarGate(uint32_t slot, uint32_t validTokens)
    {
        if constexpr (std::is_same<GateT, float>::value) {
            AscendC::Adds(gateFp32_[slot], gateRaw_[slot], 0.0f, validTokens);
        } else {
            AscendC::Cast(gateFp32_[slot], gateRaw_[slot], AscendC::RoundMode::CAST_NONE,
                          validTokens);
        }
        AscendC::PipeBarrier<PIPE_V>();
        const float last = ReadScalar(gateFp32_[slot], validTokens - 1, slot);
        AscendC::Duplicate(cast_[slot], last, validTokens);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(gateFp32_[slot], cast_[slot], gateFp32_[slot], validTokens);
        AscendC::Duplicate(alpha_[slot], last, 1);
        if constexpr (CompilePolicy::USE_EXP2) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(gateFp32_[slot], gateFp32_[slot], FWD_H_ARCH22_LN2, validTokens);
            AscendC::Muls(alpha_[slot], alpha_[slot], FWD_H_ARCH22_LN2, 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gateFp32_[slot], gateFp32_[slot], validTokens);
        AscendC::Exp(alpha_[slot], alpha_[slot], 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareKeyGate(uint32_t slot)
    {
        if constexpr (std::is_same<GateT, float>::value) {
            AscendC::Adds(gateFp32_[slot], gateRaw_[slot], 0.0f, FWD_H_K);
        } else {
            AscendC::Cast(gateFp32_[slot], gateRaw_[slot], AscendC::RoundMode::CAST_NONE,
                          FWD_H_K);
        }
        if constexpr (CompilePolicy::USE_EXP2) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(gateFp32_[slot], gateFp32_[slot], FWD_H_ARCH22_LN2, FWD_H_K);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gateFp32_[slot], gateFp32_[slot], FWD_H_K);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunSMinusOne(const FwdHWorkUnit &unit)
    {
        // S-1：StateT=FP32 且存在 initial_state 时，逐 tile 执行 H0=cast_BF16(R0)，
        // GM 物理顺序原生遵循 state_v_first，不依赖外部转置。
        if constexpr (!CompilePolicy::STATE_FP32) {
            return;
        }
        if (args_.tiling.useInitialState == 0) {
            return;
        }
        AscendC::GlobalTensor<float> initial;
        AscendC::GlobalTensor<bfloat16_t> h;
        initial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
                const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
                const uint32_t slot = head.localSlot;
                const uint64_t stateBase = FwdHStateOffset<STATE_V_FIRST>(
                    args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                const FwdHChunkSpan first = FwdHBuildChunk(unit.sequence, 0);
                for (uint32_t row = 0; row < FWD_H_K; row += TILE_ROWS) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
                    // FP32 state 常驻：initial_state 直接读入常驻区，S3 全程不再经 GM。
                    AscendC::DataCopy(stateFp32_[slot][row * FWD_H_V],
                                      initial[stateBase + row * FWD_H_V], TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, 0));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, 0));
                    AscendC::Cast(io_[slot][0], stateFp32_[slot][row * FWD_H_V],
                                  AscendC::RoundMode::CAST_RINT, TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                    AscendC::DataCopy(h[HOffset(unit, first, head) + row * FWD_H_V],
                                      io_[slot][0], TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_H_READY_FLAG, pairSlot));
        }
    }

    template <bool HAS_P>
    __aicore__ inline void PrefetchStage1Head(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
        AscendC::GlobalTensor<bfloat16_t> u;
        u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
        const uint32_t firstRows = TileRows(chunk.validTokens);
        AscendC::DataCopy(io_[slot][0], u[UOffset(unit, chunk, head, 0)], firstRows * FWD_H_V);
        if constexpr (HAS_P) {
            const uint64_t pByteOffset = ScratchByteOffset(args_.tiling.vWorkspaceOffset,
                                                           head.roundHead,
                                                           FWD_H_TOKEN_MATRIX_FP32_BYTES);
            AscendC::GlobalTensor<PType> p;
            p.SetGlobalBuffer(reinterpret_cast<__gm__ PType *>(args_.workspace + pByteOffset));
            AscendC::LocalTensor<PType> rawP =
                raw_[slot].template ReinterpretCast<PType>();
            AscendC::DataCopy(rawP, p, firstRows * FWD_H_V);
        }
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            if (args_.tiling.storeFinalState != 0 || !chunk.last) {
                AscendC::GlobalTensor<GateT> gate;
                gate.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args_.g));
                AscendC::DataCopyPadExtParams<GateT> pad{false, 0, 0, 0};
                const uint32_t gateBytes = static_cast<uint32_t>(chunk.validTokens * sizeof(GateT));
                AscendC::DataCopyExtParams copy{1, gateBytes, 0, 0, 0};
                AscendC::DataCopyPad(gateRaw_[slot], gate[GateOffset(unit, chunk, head)], copy, pad);
            }
        }
        if (chunk.first && !CompilePolicy::STATE_FP32 && args_.tiling.useInitialState != 0) {
            AscendC::GlobalTensor<bfloat16_t> initial;
            initial.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.initialState));
            const uint64_t offset = FwdHStateOffset<STATE_V_FIRST>(
                args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
            AscendC::DataCopy(stateBf16_[slot], initial[offset], FWD_H_K * FWD_H_V);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, 0));
    }

    template <bool HAS_P>
    __aicore__ inline void LoadStage1Tile(const FwdHWorkUnit &unit,
                                          const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head,
                                          uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t rows = TileRows(chunk.validTokens - row);
        const uint32_t buf = TileBuf(row);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
        AscendC::GlobalTensor<bfloat16_t> u;
        u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
        AscendC::DataCopy(io_[slot][buf], u[UOffset(unit, chunk, head, row)], rows * FWD_H_V);
        if constexpr (HAS_P) {
            const uint64_t pByteOffset = ScratchByteOffset(args_.tiling.vWorkspaceOffset,
                                                           head.roundHead,
                                                           FWD_H_TOKEN_MATRIX_FP32_BYTES);
            AscendC::GlobalTensor<PType> p;
            p.SetGlobalBuffer(reinterpret_cast<__gm__ PType *>(args_.workspace + pByteOffset));
            AscendC::LocalTensor<PType> rawP =
                raw_[slot].template ReinterpretCast<PType>();
            AscendC::DataCopy(rawP[buf * (RAW_BUF_BYTES / sizeof(PType))],
                              p[row * FWD_H_V], rows * FWD_H_V);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
    }

    template <bool HAS_P, bool WRITE_RIGHT, bool WAIT_READY>
    __aicore__ inline void ComputeStage1Tile(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head,
                                             uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t rows = TileRows(chunk.validTokens - row);
        const uint32_t elements = rows * FWD_H_V;
        const uint32_t buf = TileBuf(row);
        if constexpr (WAIT_READY) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
        }
        AscendC::Cast(calc_[slot], io_[slot][buf], AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (HAS_P) {
            if constexpr (std::is_same<PType, float>::value) {
                AscendC::LocalTensor<float> rawF =
                    raw_[slot].template ReinterpretCast<float>();
                AscendC::Sub(calc_[slot], calc_[slot], rawF[buf * (RAW_BUF_BYTES / sizeof(float))],
                             elements);
            } else {
                AscendC::LocalTensor<bfloat16_t> rawB =
                    raw_[slot].template ReinterpretCast<bfloat16_t>();
                AscendC::Cast(cast_[slot], rawB[buf * (RAW_BUF_BYTES / sizeof(bfloat16_t))],
                              AscendC::RoundMode::CAST_NONE, elements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(calc_[slot], calc_[slot], cast_[slot], elements);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(io_[slot][buf], calc_[slot], AscendC::RoundMode::CAST_RINT, elements);
        if constexpr (WRITE_RIGHT) {
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                uint32_t dstShape[2] = {rows, FWD_H_V};
                uint32_t srcShape[2] = {rows, 1};
                // Broadcast 的 gate 广播临时必须随 tile buf 走（P 已被上方 Sub 消费，
                // 同 buf 复用安全）：双缓冲下 tile t 的 MTE2 与 compute(t-1) 并行，
                // 固定写 buf0 会覆盖 tile t 的在途 P 数据（R3 冒烟 v_new/final_state
                // 双杆爆错根因）。
                AscendC::Broadcast<float, 2, 1>(
                    raw_[slot].template ReinterpretCast<float>()[buf * (RAW_BUF_BYTES / sizeof(float))],
                    gateFp32_[slot][row], dstShape, srcShape, share_[slot]);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(calc_[slot], calc_[slot],
                             raw_[slot].template ReinterpretCast<float>()[buf * (RAW_BUF_BYTES / sizeof(float))],
                             elements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(right_[slot][buf], calc_[slot], AscendC::RoundMode::CAST_RINT,
                              elements);
            }
        }

        AscendC::GlobalTensor<bfloat16_t> vNew;
        vNew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.vNew));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
        AscendC::DataCopy(vNew[UOffset(unit, chunk, head, row)], io_[slot][buf], elements);
        if constexpr (WRITE_RIGHT) {
            const uint64_t rightOffset = args_.tiling.vUpdateWorkspaceOffset / sizeof(bfloat16_t) +
                FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_CHUNK * FWD_H_V) +
                row * FWD_H_V;
            AscendC::GlobalTensor<bfloat16_t> rightGm;
            rightGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace));
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                AscendC::DataCopy(rightGm[rightOffset], right_[slot][buf], elements);
            } else {
                AscendC::DataCopy(rightGm[rightOffset], io_[slot][buf], elements);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
    }

    template <bool HAS_P, bool WRITE_RIGHT>
    __aicore__ inline void ConsumeStage1Head(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head)
    {
        // Stage1：V_new=cast_BF16(fp32(U)-fp32(P))；g-only 同时生成
        // V_new_g=cast_BF16(E(g_last-g_i)*V_new_fp32)，gk-only 的 right 即 V_new。
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, 0));
        if constexpr (WRITE_RIGHT && CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            PrepareScalarGate(slot, chunk.validTokens);
        }
        if (chunk.first && args_.tiling.useInitialState == 0) {
            if constexpr (!CompilePolicy::STATE_FP32) {
                AscendC::Duplicate(stateBf16_[slot], static_cast<bfloat16_t>(0), FWD_H_K * FWD_H_V);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        ComputeStage1Tile<HAS_P, WRITE_RIGHT, false>(unit, chunk, head, 0);
        for (uint32_t row = TILE_ROWS; row < chunk.validTokens; row += TILE_ROWS) {
            LoadStage1Tile<HAS_P>(unit, chunk, head, row);
            ComputeStage1Tile<HAS_P, WRITE_RIGHT, true>(unit, chunk, head, row);
        }
        if (chunk.first && (!CompilePolicy::STATE_FP32 || args_.tiling.useInitialState == 0)) {
            AscendC::GlobalTensor<bfloat16_t> h;
            h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
            if constexpr (!CompilePolicy::STATE_FP32) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                AscendC::DataCopy(h[HOffset(unit, chunk, head)], stateBf16_[slot], FWD_H_K * FWD_H_V);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
            } else {
                // STATE_FP32：stateBf16_ 区已被常驻 state 占用，H0=0 的 h 输出改以
                // raw_（P tile 缓冲，S1 结束后空闲且无在途 MTE3 读者）作零值中转：
                // V 写 raw_ → V_MTE3 桥 → MTE3 写 h。禁止经 io_/calc_——末 tile 的
                // v_new MTE3 仍在途，同核 WaitFlag 只装载对应 pipe、不阻塞 scalar，
                // V 写 io_ 会与其竞态（冒烟实证：chunk 0 末 tile 的 v_new 整块清零）。
                for (uint32_t row = 0; row < FWD_H_K; row += TILE_ROWS) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
                    AscendC::LocalTensor<bfloat16_t> zeroTile =
                        raw_[slot].template ReinterpretCast<bfloat16_t>();
                    AscendC::Duplicate(zeroTile, static_cast<bfloat16_t>(0), TILE_ELEMENTS);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                    AscendC::DataCopy(h[HOffset(unit, chunk, head) + row * FWD_H_V],
                                      zeroTile, TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
                }
            }
        }
    }

    __aicore__ inline void RunStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage1：V_new=cast_BF16(fp32(U)-fp32(P))；g-only 同时生成
        // V_new_g=cast_BF16(E(g_last-g_i)*V_new_fp32)，gk-only 的右矩阵就是 V_new。
        const bool hasP = !(chunk.first && args_.tiling.useInitialState == 0);
        const bool writeRight = args_.tiling.storeFinalState != 0 || !chunk.last;
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (hasP && pairSlot > 0) {
                // A2/A3 的跨核 wait 会阻塞全部流水；先消费已 ready 的 ping，不能让它
                // 等待下一个 pair 的 P_READY。无 P 时保留 pong MTE2 与 ping VEC 的重叠。
                CompleteStage1Pair(unit, chunk, pairSlot - 1U, hasP, writeRight);
            }
            if (hasP) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                    FwdHAivLocalFlag(FWD_H_P_READY_FLAG, pairSlot));
            }
            if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
                const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
                if (hasP) {
                    PrefetchStage1Head<true>(unit, chunk, head);
                } else {
                    PrefetchStage1Head<false>(unit, chunk, head);
                }
            }
            if (!hasP && pairSlot > 0) {
                CompleteStage1Pair(unit, chunk, pairSlot - 1U, hasP, writeRight);
            }
        }
        if (pairCount > 0) {
            CompleteStage1Pair(unit, chunk, pairCount - 1U, hasP, writeRight);
        }
    }

    __aicore__ inline void CompleteStage1Pair(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              uint32_t pairSlot, bool hasP,
                                              bool writeRight)
    {
        if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
            const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
            DispatchStage1(unit, chunk, head, hasP, writeRight);
        }
        if (hasP) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_V>(
                FwdHAivLocalFlag(FWD_H_P_FREE_FLAG, pairSlot));
        }
        if (writeRight) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_RIGHT_READY_FLAG, pairSlot));
        }
    }

    __aicore__ inline void DispatchStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head, bool hasP, bool writeRight)
    {
        if (hasP) {
            writeRight ? ConsumeStage1Head<true, true>(unit, chunk, head)
                       : ConsumeStage1Head<true, false>(unit, chunk, head);
        } else {
            writeRight ? ConsumeStage1Head<false, true>(unit, chunk, head)
                       : ConsumeStage1Head<false, false>(unit, chunk, head);
        }
    }

    __aicore__ inline void PrefetchStage3Head(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
        const uint64_t dByteOffset = ScratchByteOffset(args_.tiling.hWorkspaceOffset,
                                                       head.roundHead,
                                                       FWD_H_STATE_FP32_BYTES);
        AscendC::GlobalTensor<float> d;
        d.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + dByteOffset));
        AscendC::DataCopy(raw_[slot].template ReinterpretCast<float>(), d, TILE_ELEMENTS);
        // STATE_FP32 的 rolling state 常驻 UB，不再从 GM（initial/kDecayWorkspace）读取。
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
            AscendC::GlobalTensor<GateT> gk;
            gk.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args_.gk));
            const uint64_t offset = GateOffset(unit, chunk, head) +
                static_cast<uint64_t>(chunk.validTokens - 1) * FWD_H_K;
            AscendC::DataCopy(gateRaw_[slot], gk[offset], FWD_H_K);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, 0));
    }

    __aicore__ inline void LoadStage3Tile(const FwdHWorkUnit &unit,
                                          const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head,
                                          uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t buf = TileBuf(row);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
        const uint64_t dByteOffset = ScratchByteOffset(args_.tiling.hWorkspaceOffset,
                                                       head.roundHead,
                                                       FWD_H_STATE_FP32_BYTES);
        AscendC::GlobalTensor<float> d;
        d.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + dByteOffset));
        AscendC::LocalTensor<float> rawF = raw_[slot].template ReinterpretCast<float>();
        AscendC::DataCopy(rawF[buf * (RAW_BUF_BYTES / sizeof(float))],
                          d[row * FWD_H_V], TILE_ELEMENTS);
        // STATE_FP32 的 rolling state 常驻 UB，不再从 GM（kDecayWorkspace）读取。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
    }

    template <bool WRITE_H, bool WAIT_READY>
    __aicore__ inline void ComputeStage3Tile(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head,
                                             uint32_t row, float alphaValue)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t buf = TileBuf(row);
        if constexpr (WAIT_READY) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
        }
        if constexpr (CompilePolicy::STATE_FP32) {
            if (chunk.first && args_.tiling.useInitialState == 0) {
                AscendC::Duplicate(calc_[slot], 0.0f, TILE_ELEMENTS);
            } else {
                // rolling state 常驻 UB：V pipe 内同序拷贝（RAW 于上一 tile 写回，
                // 由同 pipe FIFO 保序），不再经 GM kDecayWorkspace。
                AscendC::Adds(calc_[slot], stateFp32_[slot][row * FWD_H_V], 0.0f, TILE_ELEMENTS);
            }
        }
        if constexpr (!CompilePolicy::STATE_FP32) {
            AscendC::Cast(calc_[slot], stateBf16_[slot][row * FWD_H_V],
                          AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            AscendC::Muls(calc_[slot], calc_[slot], alphaValue, TILE_ELEMENTS);
        } else if constexpr (STATE_V_FIRST) {
            for (uint32_t localRow = 0; localRow < TILE_ROWS; ++localRow) {
                AscendC::Mul(calc_[slot][localRow * FWD_H_V],
                             calc_[slot][localRow * FWD_H_V], gateFp32_[slot], FWD_H_V);
            }
        } else {
            uint32_t dstShape[2] = {TILE_ROWS, FWD_H_V};
            uint32_t srcShape[2] = {TILE_ROWS, 1};
            AscendC::Broadcast<float, 2, 1>(cast_[slot], gateFp32_[slot][row],
                                            dstShape, srcShape, share_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(calc_[slot], calc_[slot], cast_[slot], TILE_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        {
            AscendC::LocalTensor<float> rawF = raw_[slot].template ReinterpretCast<float>();
            AscendC::Add(calc_[slot], calc_[slot],
                         rawF[buf * (RAW_BUF_BYTES / sizeof(float))], TILE_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (CompilePolicy::STATE_FP32) {
            // 结果写回常驻区（V pipe 写）；final_state 输出改走小缓冲 tile 循环，
            // 不再占用 MTE3 窗口（S3 尾部只剩 H 的写出）。末 tile 写回后 V pipe FIFO
            // 保证后续（同 head 下一 chunk S3、跨 head 的 S-1 种子）读到最终值，
            // 不再需要 MTE3_GM→MTE2 的跨 pipe 往返同步。
            AscendC::Adds(stateFp32_[slot][row * FWD_H_V], calc_[slot], 0.0f, TILE_ELEMENTS);
            if constexpr (!WRITE_H) {
                if (args_.tiling.storeFinalState != 0) {
                    // calc_ 不得直接作 MTE3 源：双缓冲后 Load(t+1) 只等 t-1 的 MTE3
                    // 排空，V 写 calc_(t+1) 与 MTE3 读 calc_(t) 失去排序（末 chunk
                    // final_state 竞态根因）。经 raw_ buf（4KB；其中 D 已被本 tile
                    // 的 Add 消费完毕）中转：V 拷入 → 桥接 → MTE3 读出；raw_[buf] 的
                    // 下一写者 Load(t+2) 由 free(buf) 门控在本拷贝排空之后。
                    AscendC::LocalTensor<float> rawF = raw_[slot].template ReinterpretCast<float>();
                    AscendC::Adds(rawF[buf * (RAW_BUF_BYTES / sizeof(float))],
                                  calc_[slot], 0.0f, TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
                    AscendC::GlobalTensor<float> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.finalState));
                    const uint64_t finalOffset = FwdHStateOffset<STATE_V_FIRST>(
                        args_.tiling, unit.sequence.sequence, head.hv, 0, 0) + row * FWD_H_V;
                    AscendC::DataCopy(finalState[finalOffset],
                                      rawF[buf * (RAW_BUF_BYTES / sizeof(float))],
                                      TILE_ELEMENTS);
                }
            }
        } else {
            AscendC::Cast(stateBf16_[slot][row * FWD_H_V], calc_[slot],
                          AscendC::RoundMode::CAST_RINT, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (!WRITE_H) {
                // Terminal BF16 state has no per-tile MTE3 copy. Bridge V to MTE3 before
                // releasing the slot so the next MTE2 cannot overwrite raw_ early.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
            }
        }
        if constexpr (WRITE_H) {
            AscendC::Cast(io_[slot][buf], calc_[slot], AscendC::RoundMode::CAST_RINT, TILE_ELEMENTS);
            AscendC::GlobalTensor<bfloat16_t> h;
            h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
            const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunk.chunk + 1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
            AscendC::DataCopy(h[HOffset(unit, next, head) + row * FWD_H_V], io_[slot][buf], TILE_ELEMENTS);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
    }

    template <bool WRITE_H>
    __aicore__ inline void ConsumeStage3Head(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head)
    {
        // Stage3：g-only 为 R_next=E(g_last)R+D；gk-only 为
        // R_next[k,v]=E(gk_last[k])R[k,v]+D[k,v]，所有算术均为 FP32。
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, 0));
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
            PrepareKeyGate(slot);
        }
        // alpha 是 chunk·head 常量（Stage1 的 PrepareScalarGate 写入）：整个 head 的
        // 16 个 tile 只需一次 V_S/S_V 同步读取，避免逐 tile ReadScalar 阻塞 scalar
        // 线程并拖慢 MTE3 的 H 写出发射。
        float alphaValue = 0.0f;
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            alphaValue = ReadScalar(alpha_[slot], 0, slot);
        }
        ComputeStage3Tile<WRITE_H, false>(unit, chunk, head, 0, alphaValue);
        for (uint32_t row = TILE_ROWS; row < FWD_H_K; row += TILE_ROWS) {
            LoadStage3Tile(unit, chunk, head, row);
            ComputeStage3Tile<WRITE_H, true>(unit, chunk, head, row, alphaValue);
        }
        if constexpr (!WRITE_H) {
            if (args_.tiling.storeFinalState != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
                const uint64_t finalOffset = FwdHStateOffset<STATE_V_FIRST>(
                    args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                if constexpr (!CompilePolicy::STATE_FP32) {
                    AscendC::GlobalTensor<bfloat16_t> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.finalState));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, 0));
                    AscendC::DataCopy(finalState[finalOffset], stateBf16_[slot], FWD_H_K * FWD_H_V);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, 0));
            }
        }
    }

    __aicore__ inline void CompleteStage3Pair(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              uint32_t pairSlot, bool writeH)
    {
        if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
            const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
            writeH ? ConsumeStage3Head<true>(unit, chunk, head)
                   : ConsumeStage3Head<false>(unit, chunk, head);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_V>(
            FwdHAivLocalFlag(FWD_H_D_FREE_FLAG, pairSlot));
        if (writeH) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_H_READY_FLAG, pairSlot));
        }
    }

    __aicore__ inline void RunStage3(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage3：g-only 为 R_next=E(g_last)R+D；gk-only 为
        // R_next[k,v]=E(gk_last[k])R[k,v]+D[k,v]，并按需生成下一 H 或 final_state。
        const bool writeH = !chunk.last;
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (pairSlot > 0) {
                // 先消费已完成 MTE2 的 ping，再等待下一 pair 的 D_READY。
                CompleteStage3Pair(unit, chunk, pairSlot - 1U, writeH);
            }
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                FwdHAivLocalFlag(FWD_H_D_READY_FLAG, pairSlot));
            if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
                const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
                PrefetchStage3Head(unit, chunk, head);
            }
        }
        if (pairCount > 0) {
            CompleteStage3Pair(unit, chunk, pairCount - 1U, writeH);
        }
    }

    __aicore__ inline void ProcessWorkUnit(const FwdHWorkUnit &unit, bool hasNextWorkUnit)
    {
        const bool hasCubeWork = args_.tiling.useInitialState || args_.tiling.storeFinalState ||
            args_.tiling.seqlen > static_cast<int64_t>(FWD_H_CHUNK);
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        if (args_.tiling.useInitialState == 0 && unit.sequence.chunkCount > 1) {
            // 首 chunk 没有 Stage0/P 消费链。为第二个 chunk 的 Stage0 提供一次 P scratch
            // 初始 free credit；后续 chunk 的 credit 都由前一 chunk 的 Stage1 产生。
            for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_P_FREE_FLAG, pairSlot));
            }
        }
        RunSMinusOne(unit);
        for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
            const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
            if (chunkId > 0) {
                for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(
                        FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, pairSlot));
                }
            }
            RunStage1(unit, chunk);
            if (args_.tiling.storeFinalState != 0 || !chunk.last) {
                RunStage3(unit, chunk);
            }
        }
        if (args_.tiling.storeFinalState != 0) {
            for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, pairSlot));
            }
        }
        // 跨 head_round 前必须把本轮所有 VEC->MTE3 写回收口。AIC 收到 ROUND_DONE 后
        // 才能开始下一轮 kg/H/W 的 MTE2，避免上一轮 MTE3 与下一轮预取跨 round 交叠。
        for (uint32_t slot = 0; slot < localHeads; ++slot) {
            for (uint32_t buf = 0; buf < TILE_BUFS; ++buf) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot, buf));
            }
        }
        if (hasCubeWork && hasNextWorkUnit) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(FwdHAivLocalFlag(FWD_H_ROUND_DONE_FLAG, 0));
            AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(FwdHAivLocalFlag(FWD_H_ROUND_ACK_FLAG, 0));
        }
    }

    FwdHKernelArgs args_{};
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t aiv_ = 0;
    AscendC::TEventID ioFreeEvent_[FWD_H_AIV_HEAD_SLOTS][TILE_BUFS]{};
    AscendC::TEventID ioReadyEvent_[FWD_H_AIV_HEAD_SLOTS][TILE_BUFS]{};
    AscendC::TEventID ioDoneEvent_[FWD_H_AIV_HEAD_SLOTS][TILE_BUFS]{};
    AscendC::TEventID scalarReadEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TEventID scalarWriteEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_{};
    AscendC::LocalTensor<uint8_t> raw_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> cast_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> calc_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> io_[FWD_H_AIV_HEAD_SLOTS][TILE_BUFS]{};
    AscendC::LocalTensor<bfloat16_t> right_[FWD_H_AIV_HEAD_SLOTS][TILE_BUFS]{};
    AscendC::LocalTensor<GateT> gateRaw_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> gateFp32_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> alpha_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<uint8_t> share_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> stateBf16_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> stateFp32_[FWD_H_AIV_HEAD_SLOTS]{};
};

} // namespace GDN

#endif // ARCH22_CHUNK_FWD_H_VEC_H
