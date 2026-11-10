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
#include "chunk_fwd_h_utils.h"

namespace GDN {

constexpr float FWD_H_ARCH22_LN2 = 0.69314718055994530942f;

// A2 向量 API 间无硬件 RAW 追踪，S1/S3 计算路径的中间屏障为必要保护，勿剪除。
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
    // T16：dense 专用 TILE_ROWS=16（host tiling key 保证 dense 且每 AIV 单 head 链才选）。
    static constexpr uint32_t TILE_ROWS = CompilePolicy::DENSE_T16 ? 16 : 8;
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
    // tile 双缓冲布局（每 slot 32K 内）：raw 双份 [0,8K)、cast [8K,12K)、calc
    // [12K,16K)、io/right 双份 [16K,24K)、gate/alpha/share [24K,28K)。
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
    // T16 单 slot 大缓冲布局（slot1 与 FP32 state1 静态闲置）：raw 双份 8K/份、
    // io/right 双份 4K/份、calc 双份 8K/份（跨 tile RAW 保护）、cast 8K；
    // [64K,128K) 为本 head 的 FP32 state。
    static constexpr uint32_t T16_RAW_BASE = 0;
    static constexpr uint32_t T16_RAW_BUF_BYTES = 8 * 1024;
    static constexpr uint32_t T16_IO_BASE = 16 * 1024;
    static constexpr uint32_t T16_IO_BUF_BYTES = 4 * 1024;
    static constexpr uint32_t T16_RIGHT_BASE = 24 * 1024;
    static constexpr uint32_t T16_RIGHT_BUF_BYTES = 4 * 1024;
    static constexpr uint32_t T16_CAST_BASE = 32 * 1024;
    static constexpr uint32_t T16_CALC_BASE = 40 * 1024;
    static constexpr uint32_t T16_CALC_BYTES = 8 * 1024;
    // T16 尾部区 [56K,64K)：gate/gateFp32/alpha 各 1K、share 4K——不得与 calc
    // 双份 [40K,56K) 重叠（重叠会互相覆写）。
    static constexpr uint32_t T16_GATE_RAW_BASE = 56 * 1024;
    static constexpr uint32_t T16_GATE_FP32_BASE = 57 * 1024;
    static constexpr uint32_t T16_ALPHA_BASE = 58 * 1024;
    static constexpr uint32_t T16_SHARE_BASE = 59 * 1024;
    static constexpr uint32_t TILE_BUFS = 2;
    // N3：g(scalar)+FP32 的 S3 融合开关（三步压两步，Axpy dst=src·scalar+dst）。
    static constexpr bool G_FP32_AXPY =
        CompilePolicy::STATE_FP32 && (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G);

    __aicore__ inline uint32_t TileBuf(uint32_t row) const
    {
        return (row / TILE_ROWS) & 1U;
    }

    // calc tile 寻址：T16 双份（跨 tile RAW 保护）；T8 恒用 [0]（[1] 未初始化）。
    __aicore__ inline AscendC::LocalTensor<float> CalcBuf(uint32_t slot,
                                                          uint32_t buf) const
    {
        if constexpr (CompilePolicy::DENSE_T16) {
            return calc_[slot][buf];
        } else {
            return calc_[slot][0];
        }
    }

    // raw_ 双份 per-buf 元素步进：T16 8K/份、T8 4K/份。
    static constexpr uint32_t RAW_BUF_ELEMS =
        (CompilePolicy::DENSE_T16 ? T16_RAW_BUF_BYTES : RAW_BUF_BYTES) / sizeof(float);
    static constexpr uint32_t T16_RAW_ELEMS_P =
        (CompilePolicy::DENSE_T16 ? T16_RAW_BUF_BYTES : RAW_BUF_BYTES) / sizeof(PType);
    static constexpr uint32_t T16_RAW_ELEMS_B =
        (CompilePolicy::DENSE_T16 ? T16_RAW_BUF_BYTES : RAW_BUF_BYTES) / sizeof(bfloat16_t);

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
        if constexpr (CompilePolicy::DENSE_T16) {
            // T16 单 slot 大缓冲：本 AIV 绑定的 head 恒落 localSlot 0。
            static_assert(T16_RAW_BASE == 0 && T16_IO_BASE == 16 * 1024 &&
                              T16_RIGHT_BASE == 24 * 1024 && T16_CAST_BASE == 32 * 1024 &&
                              T16_CALC_BASE == 40 * 1024 && T16_CALC_BYTES == 8 * 1024 &&
                              T16_GATE_RAW_BASE == 56 * 1024,
                          "T16 layout must stay within the 64K single-slot window");
            for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
                const uint32_t base = LocalBase(slot);
                raw_[slot] = ub[base + T16_RAW_BASE].template ReinterpretCast<uint8_t>();
                cast_[slot] = ub[base + T16_CAST_BASE].template ReinterpretCast<float>();
                for (uint32_t buf = 0; buf < TILE_BUFS; ++buf) {
                    io_[slot][buf] =
                        ub[base + T16_IO_BASE + buf * T16_IO_BUF_BYTES]
                            .template ReinterpretCast<bfloat16_t>();
                    right_[slot][buf] =
                        ub[base + T16_RIGHT_BASE + buf * T16_RIGHT_BUF_BYTES]
                            .template ReinterpretCast<bfloat16_t>();
                    calc_[slot][buf] =
                        ub[base + T16_CALC_BASE + buf * T16_CALC_BYTES]
                            .template ReinterpretCast<float>();
                }
                gateRaw_[slot] = ub[base + T16_GATE_RAW_BASE].template ReinterpretCast<GateT>();
                gateFp32_[slot] = ub[base + T16_GATE_FP32_BASE].template ReinterpretCast<float>();
                alpha_[slot] = ub[base + T16_ALPHA_BASE].template ReinterpretCast<float>();
                share_[slot] = ub[base + T16_SHARE_BASE].template ReinterpretCast<uint8_t>();
            }
        } else {
            for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
                const uint32_t base = LocalBase(slot);
                raw_[slot] = ub[base + RAW_BASE].template ReinterpretCast<uint8_t>();
                cast_[slot] = ub[base + CAST_BASE].template ReinterpretCast<float>();
                calc_[slot][0] = ub[base + CALC_BASE].template ReinterpretCast<float>();
                for (uint32_t buf = 0; buf < TILE_BUFS; ++buf) {
                    io_[slot][buf] = ub[base + IO_BASE + buf * IO_BUF_ELEMS * sizeof(bfloat16_t)]
                                         .template ReinterpretCast<bfloat16_t>();
                    right_[slot][buf] =
                        ub[base + RIGHT_BASE + buf * RIGHT_BUF_ELEMS * sizeof(bfloat16_t)]
                            .template ReinterpretCast<bfloat16_t>();
                }
                gateRaw_[slot] = ub[base + GATE_RAW_BASE].template ReinterpretCast<GateT>();
                gateFp32_[slot] = ub[base + GATE_FP32_BASE].template ReinterpretCast<float>();
                alpha_[slot] = ub[base + ALPHA_BASE].template ReinterpretCast<float>();
                share_[slot] = ub[base + SHARE_BASE].template ReinterpretCast<uint8_t>();
            }
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
            if (CompilePolicy::STORE_FINAL || !chunk.last) {
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
            AscendC::DataCopy(rawP[buf * (T16_RAW_ELEMS_P)],
                              p[row * FWD_H_V], rows * FWD_H_V);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
    }

    // tile 0 的 IoReady wait 与 gate 预处理（读 MTE2 装载的 gateRaw_，须在 wait
    // 之后）一并内联进 Compute；wait 总数与 set/wait 配对不变。
    template <bool HAS_P, bool WRITE_RIGHT>
    __aicore__ inline void ComputeStage1Tile(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head,
                                             uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t rows = TileRows(chunk.validTokens - row);
        const uint32_t elements = rows * FWD_H_V;
        const uint32_t buf = TileBuf(row);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
        if (row == 0) {
            if constexpr (WRITE_RIGHT && CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                PrepareScalarGate(slot, chunk.validTokens);
            }
        }
        AscendC::Cast(CalcBuf(slot, buf), io_[slot][buf], AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr (HAS_P) {
            if constexpr (std::is_same<PType, float>::value) {
                AscendC::LocalTensor<float> rawF =
                    raw_[slot].template ReinterpretCast<float>();
                AscendC::Sub(CalcBuf(slot, buf), CalcBuf(slot, buf),
                             rawF[buf * RAW_BUF_ELEMS],
                             elements);
            } else {
                AscendC::LocalTensor<bfloat16_t> rawB =
                    raw_[slot].template ReinterpretCast<bfloat16_t>();
                AscendC::Cast(cast_[slot], rawB[buf * (T16_RAW_ELEMS_B)],
                              AscendC::RoundMode::CAST_NONE, elements);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Sub(CalcBuf(slot, buf), CalcBuf(slot, buf), cast_[slot], elements);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(io_[slot][buf], CalcBuf(slot, buf), AscendC::RoundMode::CAST_RINT, elements);
        if constexpr (WRITE_RIGHT) {
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                uint32_t dstShape[2] = {rows, FWD_H_V};
                uint32_t srcShape[2] = {rows, 1};
                // Broadcast 的 gate 广播临时须随 tile buf 走：双缓冲下 tile t 的
                // MTE2 与 compute(t-1) 并行，固定写 buf0 会覆盖在途数据。
                AscendC::Broadcast<float, 2, 1>(
                    raw_[slot].template ReinterpretCast<float>()[buf * RAW_BUF_ELEMS],
                    gateFp32_[slot][row], dstShape, srcShape, share_[slot]);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Mul(CalcBuf(slot, buf), CalcBuf(slot, buf),
                             raw_[slot].template ReinterpretCast<float>()[buf * RAW_BUF_ELEMS],
                             elements);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Cast(right_[slot][buf], CalcBuf(slot, buf),
                              AscendC::RoundMode::CAST_RINT, elements);
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
        // stateBf16_ 零化在 tile 0 进 Compute 前发射（与 io_/gateRaw_ 区域不相交）。
        if (chunk.first && args_.tiling.useInitialState == 0) {
            if constexpr (!CompilePolicy::STATE_FP32) {
                AscendC::Duplicate(stateBf16_[slot], static_cast<bfloat16_t>(0), FWD_H_K * FWD_H_V);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        ComputeStage1Tile<HAS_P, WRITE_RIGHT>(unit, chunk, head, 0);
        for (uint32_t row = TILE_ROWS; row < chunk.validTokens; row += TILE_ROWS) {
            LoadStage1Tile<HAS_P>(unit, chunk, head, row);
            ComputeStage1Tile<HAS_P, WRITE_RIGHT>(unit, chunk, head, row);
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
                // STATE_FP32：stateBf16_ 区已被常驻 state 占用，H0=0 的 h 输出经
                // raw_（S1 结束后空闲）作零值中转；禁止经 io_/calc_（末 tile 的
                // v_new MTE3 仍在途，V 写会与其竞态）。
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
        const bool writeRight = CompilePolicy::STORE_FINAL || !chunk.last;
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
        AscendC::DataCopy(rawF[buf * RAW_BUF_ELEMS],
                          d[row * FWD_H_V], TILE_ELEMENTS);
        // STATE_FP32 的 rolling state 常驻 UB，不再从 GM（kDecayWorkspace）读取。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
    }

    // 同 S1：WAIT_READY 维收掉（4→2 实例），tile 0 的 IoReady wait 与 KEY_GK 的
    // gate 预处理（读 MTE2 装载的 gk 尾行）内联进 Compute。
    template <bool WRITE_H>
    __aicore__ inline void ComputeStage3Tile(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head,
                                             uint32_t row, float alphaValue)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t buf = TileBuf(row);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot, buf));
        if (row == 0) {
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
                PrepareKeyGate(slot);
            }
        }
        if constexpr (G_FP32_AXPY) {
            // N3 融合路径：calc 先装 D（raw_ 已由 Load/Prefetch 经 IoReadyEvent 就绪），
            // Axpy 原地并入 state·α，替代旧三步两屏障（省一次向量算子+一次屏障）。
            AscendC::LocalTensor<float> rawF = raw_[slot].template ReinterpretCast<float>();
            AscendC::Adds(CalcBuf(slot, buf), rawF[buf * RAW_BUF_ELEMS],
                          0.0f, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            if (!(chunk.first && args_.tiling.useInitialState == 0)) {
                // rolling state 跨 tile RAW 同旧路径由 V pipe FIFO 保序；首 chunk 无
                // 初态时 state 分量为 0，Axpy 跳过（calc=D 即终值）。
                AscendC::Axpy(CalcBuf(slot, buf), stateFp32_[slot][row * FWD_H_V],
                              alphaValue, TILE_ELEMENTS);
            }
        }
        if constexpr (!G_FP32_AXPY && CompilePolicy::STATE_FP32) {
            if (chunk.first && args_.tiling.useInitialState == 0) {
                AscendC::Duplicate(CalcBuf(slot, buf), 0.0f, TILE_ELEMENTS);
            } else {
                // rolling state 常驻 UB：V pipe 内同序拷贝（RAW 于上一 tile 写回，
                // 由同 pipe FIFO 保序），不再经 GM kDecayWorkspace。
                AscendC::Adds(CalcBuf(slot, buf), stateFp32_[slot][row * FWD_H_V], 0.0f,
                              TILE_ELEMENTS);
            }
        }
        if constexpr (!CompilePolicy::STATE_FP32) {
            AscendC::Cast(CalcBuf(slot, buf), stateBf16_[slot][row * FWD_H_V],
                          AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr (!G_FP32_AXPY) {
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                AscendC::Muls(CalcBuf(slot, buf), CalcBuf(slot, buf), alphaValue, TILE_ELEMENTS);
            } else if constexpr (STATE_V_FIRST) {
                for (uint32_t localRow = 0; localRow < TILE_ROWS; ++localRow) {
                    AscendC::Mul(CalcBuf(slot, buf)[localRow * FWD_H_V],
                                 CalcBuf(slot, buf)[localRow * FWD_H_V], gateFp32_[slot], FWD_H_V);
                }
            } else {
                uint32_t dstShape[2] = {TILE_ROWS, FWD_H_V};
                uint32_t srcShape[2] = {TILE_ROWS, 1};
                AscendC::Broadcast<float, 2, 1>(cast_[slot], gateFp32_[slot][row],
                                                dstShape, srcShape, share_[slot]);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Mul(CalcBuf(slot, buf), CalcBuf(slot, buf), cast_[slot], TILE_ELEMENTS);
            }
            AscendC::PipeBarrier<PIPE_V>();

            {
                AscendC::LocalTensor<float> rawF = raw_[slot].template ReinterpretCast<float>();
                AscendC::Add(CalcBuf(slot, buf), CalcBuf(slot, buf),
                             rawF[buf * RAW_BUF_ELEMS], TILE_ELEMENTS);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        if constexpr (CompilePolicy::STATE_FP32) {
            // 结果写回常驻区（V pipe 写）；final_state 输出改走小缓冲 tile 循环，
            // 不再占用 MTE3 窗口（S3 尾部只剩 H 的写出）。末 tile 写回后 V pipe FIFO
            // 保证后续（同 head 下一 chunk S3、跨 head 的 S-1 种子）读到最终值，
            // 不再需要 MTE3_GM→MTE2 的跨 pipe 往返同步。
            AscendC::Adds(stateFp32_[slot][row * FWD_H_V], CalcBuf(slot, buf), 0.0f,
                          TILE_ELEMENTS);
            if constexpr (!WRITE_H) {
                if constexpr (CompilePolicy::STORE_FINAL) {
                    // calc_ 不得直接作 MTE3 源：双缓冲下 V 写 calc_(t+1) 与 MTE3 读
                    // calc_(t) 失去排序，须经 raw_[buf] 中转（D 已被本 tile 消费完）。
                    AscendC::LocalTensor<float> rawF = raw_[slot].template ReinterpretCast<float>();
                    AscendC::Adds(rawF[buf * RAW_BUF_ELEMS],
                                  CalcBuf(slot, buf), 0.0f, TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot, buf));
                    AscendC::GlobalTensor<float> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.finalState));
                    const uint64_t finalOffset = FwdHStateOffset<STATE_V_FIRST>(
                        args_.tiling, unit.sequence.sequence, head.hv, 0, 0) + row * FWD_H_V;
                    AscendC::DataCopy(finalState[finalOffset],
                                      rawF[buf * RAW_BUF_ELEMS],
                                      TILE_ELEMENTS);
                }
            }
        } else {
            AscendC::Cast(stateBf16_[slot][row * FWD_H_V], CalcBuf(slot, buf),
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
            AscendC::Cast(io_[slot][buf], CalcBuf(slot, buf), AscendC::RoundMode::CAST_RINT,
                          TILE_ELEMENTS);
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
        // tile 0 的 IoReady wait 已内联 ComputeStage3Tile（其后的 KEY_GK gate 预处理
        // 读 MTE2 装载的 gk 尾行，依赖该 wait）——此处不再预 wait（消费型事件，
        // 双 wait 死锁）。alpha 是 chunk·head 常量（Stage1 的 PrepareScalarGate 经
        // V pipe 写入，V_S/S_V 自同步）：整个 head 的 16 个 tile 只需一次标量读。
        float alphaValue = 0.0f;
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            alphaValue = ReadScalar(alpha_[slot], 0, slot);
        }
        ComputeStage3Tile<WRITE_H>(unit, chunk, head, 0, alphaValue);
        for (uint32_t row = TILE_ROWS; row < FWD_H_K; row += TILE_ROWS) {
            LoadStage3Tile(unit, chunk, head, row);
            ComputeStage3Tile<WRITE_H>(unit, chunk, head, row, alphaValue);
        }
        if constexpr (!WRITE_H) {
            if constexpr (CompilePolicy::STORE_FINAL) {
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
        const bool hasCubeWork = args_.tiling.useInitialState || CompilePolicy::STORE_FINAL ||
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
            if (CompilePolicy::STORE_FINAL || !chunk.last) {
                RunStage3(unit, chunk);
            }
        }
        if constexpr (CompilePolicy::STORE_FINAL) {
            for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, pairSlot));
            }
        }
        // 跨 head_round 前把本轮所有 VEC->MTE3 写回收口，避免与下一轮 AIC 预取交叠。
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
    // calc 双份：T16 布局下 tile 内单 buf 无法靠 buf 交替规避跨 tile RAW
    //（A2 向量指令无硬件 RAW 追踪，vec.h 头部 R6 警示）；T8 布局恒用 [0]
    //（第二份不初始化，行为与单份等价——InitBuffers 的 else 分支只填 [0]）。
    AscendC::LocalTensor<float> calc_[FWD_H_AIV_HEAD_SLOTS][TILE_BUFS]{};
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

// ---- gated 复用引擎（AIV 侧）：自 arch22/chunk_fwd_h_utils.h 的
// GDNFwdHKernelPreload 拆出（Init 保留 AIV 调度器分支，Process = 原
// ProcessVec 体；纯搬移，指令序列不变）。V1/V2 epilogue 仍以
// BlockEpilogue 特化形式经 arch22/chunk_fwd_h_utils.h 引入（后续步骤解体）。
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
class ChunkFwdHGatedVecArch22 {
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

    // vec 1
    using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasGDNFwdHVnewPreload<USE_EXP2_>;
    using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, VworkType, FinalStateType>;

    // vec 2
    using DispatchPolicyGDNFwdHUpdate = Epilogue::EpilogueAtlasGDNFwdHUpdatePreload<USE_EXP2_>;
    using EpilogueGDNFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHUpdate, HType, GType, HType, HworkType, FinalStateType>;

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

    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline ChunkFwdHGatedVecArch22() {}

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

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }
    }


    __aicore__ inline void Process()
    {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
            uint32_t totalChunks = isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
            uint32_t transferCount = isVariedLen ? (vecBlockScheduler.tokenBatch * vNumHead / coreNum) : (shapeBatch * vNumHead / coreNum);
            uint32_t remainderFlag = isVariedLen ? (((vecBlockScheduler.tokenBatch * vNumHead) % coreNum) != 0): (((shapeBatch * vNumHead) % coreNum) != 0);
            uint32_t step = transferCount + remainderFlag;
            uint32_t stateBlockSize = kHeadDim * vHeadDim;
            uint32_t pingpongFlag = 1;
            uint32_t start = coreIdx * step;
            uint32_t end = start + step;
            uint32_t maxLimit = isVariedLen ? vecBlockScheduler.tokenBatch * vNumHead : shapeBatch * vNumHead;
            uint32_t realEnd = min(end, maxLimit);
            if (useInitialState) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                AscendC::Duplicate(hUbTensorPing, static_cast<ElementH>(0), stateBlockSize);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            }
            for(uint32_t initialStateBlockOffset = start ;initialStateBlockOffset >= start && initialStateBlockOffset < realEnd;initialStateBlockOffset++)
            {   
                uint32_t batchIdx = initialStateBlockOffset / vNumHead;
                uint32_t vHeadIdx = initialStateBlockOffset % vNumHead;
                uint32_t chunkOffset = isVariedLen ? gmNumChunks.GetValue(batchIdx) : 0; 
                uint64_t initialStateOffset = initialStateBlockOffset * stateBlockSize;
                uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                uint64_t hOffset = ((shapeBatchIdx * totalChunks + chunkOffset) * vNumHead + vHeadIdx) * stateBlockSize;
                if (useInitialState) {
                    AscendC::LocalTensor<ElementInitialState> stateUbTensor = pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                    AscendC::LocalTensor<ElementH> hUbTensor = pingpongFlag ? hUbTensorPing : hUbTensorPong;
                    auto event_id = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    if constexpr(!std::is_same<ElementInitialState, ElementH>::value) {
                        AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateBlockSize);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                        AscendC::Cast(hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT, stateBlockSize);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                        AscendC::DataCopy(gmH[hOffset], hUbTensor, stateBlockSize);
                    } else {
                        AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateBlockSize);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                        AscendC::DataCopy(gmH[hOffset], stateUbTensor, stateBlockSize);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    pingpongFlag = 1 - pingpongFlag;
                } else {
                    AscendC::DataCopy(gmH[hOffset], hUbTensorPing, stateBlockSize);
                }


            }
                
            if (useInitialState) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            
            AscendC::SyncAll<false>();

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);
            EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
            uint32_t pongBaseEvent = 4;

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); // preset v
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset v
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);

            uint32_t currStage = 0; // 0: V1, 1: V2
            while (vecBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* V1:
                     * gmV = gmU - gmVWorkspace
                     * g_buf = gmG[-1] - gmG
                     * g_buf = exp(g_buf)
                     * gmVWorkspace = g_buf * gmV
                     */
                    vecBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        epilogueGDNFwdHVnew(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset], 
                            gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset], 
                            vec1Offsets.blockTokens, kHeadDim, vHeadDim, 
                            vecBlockScheduler.cube1Done, vecBlockScheduler.vec1Done,
                            vec1Offsets.isInitialState, vec1Offsets.isFinalState, storeFinalState, (i == 0)
                        );
                    }
                } else {
                    /* V2: h[i+1] += h_work if i < num_chunks - 1 else None */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);

                        if (vecBlockScheduler.NeedProcessStage2(stream)) {
                            // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                            epilogueGDNFwdHUpdate(
                                gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                gmG[vec2Offsets.gOffset],
                                gmH[vec2Offsets.hSrcOffset],
                                gmHWorkspace[vec2Offsets.hWorkOffset],
                                vec2Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube2Done,
                                vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState, (i == 0)
                            );
                        } else {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[i]);
                    }
                }
                currStage ^= 0x01;
            }

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); // preset v
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset v
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
    }

};

} // namespace Catlass::Gemm::Kernel

#endif // ARCH22_CHUNK_FWD_H_VEC_H
