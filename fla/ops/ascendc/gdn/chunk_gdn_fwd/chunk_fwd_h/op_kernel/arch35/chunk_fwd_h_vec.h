/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH35_CHUNK_FWD_H_VEC_H
#define ARCH35_CHUNK_FWD_H_VEC_H

#include <type_traits>

#include "../chunk_fwd_h_policy.h"
#include "../chunk_fwd_h_utils.h"
#include "kernel_utils/vector/regbase.hpp"

namespace GDN {

using namespace AscendC::MicroAPI;

constexpr float FWD_H_LN2 = 0.69314718055994530942f;

template <typename T>
__simd_callee__ inline void FwdHLoadAsFloat(RegTensor<float> &zero, RegTensor<float> &one,
                                             __ubuf__ T *src, MaskReg &mask16)
{
    RegTensor<T> raw;
    LoadIn<T, false>(raw, src);
    CastHalf2Float<T>(zero, one, raw, mask16);
}

template <typename T>
__simd_callee__ inline void FwdHLoadScalar(RegTensor<float> &dst, __ubuf__ T *src,
                                            MaskReg &mask16, MaskReg &mask32)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadIn<float, true>(dst, src);
    } else {
        RegTensor<T> raw;
        RegTensor<float> unused;
        LoadIn<T, true>(raw, src);
        CastHalf2Float<T>(dst, unused, raw, mask16);
    }
    (void)mask32;
}

__simd_callee__ inline void FwdHStoreBf16(__ubuf__ bfloat16_t *dst,
                                          RegTensor<float> &zero, RegTensor<float> &one,
                                          MaskReg &mask32, MaskReg &mask16)
{
    RegTensor<bfloat16_t> output;
    CastFloat2Half<bfloat16_t>(output, zero, one, mask32);
    StoreAlign(dst, output, mask16);
}

template <bool HAS_P, bool SCALAR_G, bool WRITE_RIGHT, bool ZERO_STATE, bool USE_EXP2,
          typename PType, typename GateT>
__simd_vf__ inline void FwdHStage1Arch35Vf(__ubuf__ bfloat16_t *uAndVNew,
                                           __ubuf__ PType *p,
                                           __ubuf__ GateT *gate,
                                           __ubuf__ bfloat16_t *right,
                                           __ubuf__ bfloat16_t *state,
                                           __ubuf__ float *alpha,
                                           uint16_t validTokens)
{
    // Stage1:
    //   V_new_fp32[i,:] = fp32(U[i,:]) - fp32(P[i,:])，HAS_P=false 时 P=0；
    //   V_new[i,:] = cast_BF16(V_new_fp32[i,:])；
    //   g-only: right[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32[i,:])，alpha=E(g_last)；
    //   gk-only: right=V_new。E 由 USE_EXP2 在编译期选择。
    // 本 VF 覆盖当前 head 的全部 MTE2 后向量计算，内部没有运行期分支。
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t BF16_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint16_t STATE_ROWS = FWD_H_STATE_BF16_BYTES / sizeof(bfloat16_t) / BF16_PER_REG;
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> lastGate;
    if constexpr (SCALAR_G && WRITE_RIGHT) {
        FwdHLoadScalar<GateT>(lastGate, gate + validTokens - 1, mask16, mask32);
        RegTensor<float> alphaReg;
        FwdHLoadScalar<GateT>(alphaReg, gate + validTokens - 1, mask16, mask32);
        if constexpr (USE_EXP2) {
            Muls(alphaReg, alphaReg, FWD_H_LN2, mask32);
        }
        Exp(alphaReg, alphaReg, mask32);
        uint32_t scalarCount = 1;
        MaskReg scalarMask = UpdateMask<float>(scalarCount);
        StoreAlign(alpha, alphaReg, scalarMask);
    }

    if constexpr (ZERO_STATE) {
        RegTensor<bfloat16_t> zeroState;
        Duplicate(zeroState, static_cast<bfloat16_t>(0), mask16);
        #pragma unroll 2
        for (uint16_t row = 0; row < STATE_ROWS; ++row) {
            StoreAlign(state + static_cast<uint32_t>(row) * BF16_PER_REG, zeroState, mask16);
        }
    }

    RegTensor<float> u0;
    RegTensor<float> u1;
    RegTensor<float> p0;
    RegTensor<float> p1;
    RegTensor<float> gateReg;
    #pragma unroll 2
    for (uint16_t row = 0; row < validTokens; ++row) {
        const uint32_t rowBf16Offset = static_cast<uint32_t>(row) * FWD_H_V;
        const uint32_t rowFp32Offset = static_cast<uint32_t>(row) * FWD_H_V;
        for (uint16_t col = 0; col < FWD_H_V; col += BF16_PER_REG) {
            const uint32_t bf16Offset = rowBf16Offset + col;
            const uint32_t fp32Offset = rowFp32Offset + col;
            FwdHLoadAsFloat<bfloat16_t>(u0, u1, uAndVNew + bf16Offset, mask16);
            if constexpr (HAS_P) {
                if constexpr (std::is_same<PType, float>::value) {
                    LoadAlign(p0, p + fp32Offset);
                    LoadAlign(p1, p + fp32Offset + FP32_PER_REG);
                } else {
                    FwdHLoadAsFloat<PType>(p0, p1, p + bf16Offset, mask16);
                }
            } else {
                Duplicate(p0, 0.0f, mask32);
                Duplicate(p1, 0.0f, mask32);
            }
            Sub(u0, u0, p0, mask32);
            Sub(u1, u1, p1, mask32);
            FwdHStoreBf16(uAndVNew + bf16Offset, u0, u1, mask32, mask16);

            if constexpr (WRITE_RIGHT) {
                if constexpr (SCALAR_G) {
                    FwdHLoadScalar<GateT>(gateReg, gate + row, mask16, mask32);
                    Sub(gateReg, lastGate, gateReg, mask32);
                    if constexpr (USE_EXP2) {
                        Muls(gateReg, gateReg, FWD_H_LN2, mask32);
                    }
                    Exp(gateReg, gateReg, mask32);
                    Mul(u0, u0, gateReg, mask32);
                    Mul(u1, u1, gateReg, mask32);
                }
                FwdHStoreBf16(right + bf16Offset, u0, u1, mask32, mask16);
            }
        }
    }
}

template <bool STATE_FP32, bool SCALAR_G, bool STATE_V_FIRST, bool WRITE_H,
          bool ZERO_STATE, bool USE_EXP2, typename GateT>
__simd_vf__ inline void FwdHStage3Arch35Vf(__ubuf__ bfloat16_t *stateBf16,
                                           __ubuf__ float *stateFp32,
                                           __ubuf__ float *d,
                                           __ubuf__ GateT *gkLast,
                                           __ubuf__ float *alpha,
                                           __ubuf__ bfloat16_t *hNext)
{
    // Stage3:
    //   g-only: R_next = alpha * R + D，alpha=E(g_last)；
    //   gk-only: R_next[k,v] = E(gk_last[k]) * R[k,v] + D[k,v]；
    //   WRITE_H 时 H_next = cast_BF16(R_next)。
    // STATE_V_FIRST=true 时 state、D 和 H 都以物理 [V,K] 顺序处理，公式语义仍为 [K,V]。
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t BF16_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint16_t ROWS = 128;
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> state0;
    RegTensor<float> state1;
    RegTensor<float> d0;
    RegTensor<float> d1;
    RegTensor<float> gate0;
    RegTensor<float> gate1;

    if constexpr (SCALAR_G) {
        LoadIn<float, true>(gate0, alpha);
        Adds(gate1, gate0, 0.0f, mask32);
    }

    #pragma unroll 2
    for (uint16_t row = 0; row < ROWS; ++row) {
        for (uint16_t col = 0; col < FWD_H_V; col += BF16_PER_REG) {
            const uint32_t bf16Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            const uint32_t fp32Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            if constexpr (STATE_FP32) {
                if constexpr (ZERO_STATE) {
                    Duplicate(state0, 0.0f, mask32);
                    Duplicate(state1, 0.0f, mask32);
                } else {
                    LoadAlign(state0, stateFp32 + fp32Offset);
                    LoadAlign(state1, stateFp32 + fp32Offset + FP32_PER_REG);
                }
            } else {
                FwdHLoadAsFloat<bfloat16_t>(state0, state1, stateBf16 + bf16Offset, mask16);
            }
            LoadAlign(d0, d + fp32Offset);
            LoadAlign(d1, d + fp32Offset + FP32_PER_REG);

            if constexpr (!SCALAR_G) {
                if constexpr (STATE_V_FIRST) {
                    if constexpr (std::is_same<GateT, float>::value) {
                        LoadAlign(gate0, gkLast + col);
                        LoadAlign(gate1, gkLast + col + FP32_PER_REG);
                    } else {
                        FwdHLoadAsFloat<GateT>(gate0, gate1, gkLast + col, mask16);
                    }
                    if constexpr (USE_EXP2) {
                        Muls(gate0, gate0, FWD_H_LN2, mask32);
                        Muls(gate1, gate1, FWD_H_LN2, mask32);
                    }
                    Exp(gate0, gate0, mask32);
                    Exp(gate1, gate1, mask32);
                } else {
                    FwdHLoadScalar<GateT>(gate0, gkLast + row, mask16, mask32);
                    if constexpr (USE_EXP2) {
                        Muls(gate0, gate0, FWD_H_LN2, mask32);
                    }
                    Exp(gate0, gate0, mask32);
                    Adds(gate1, gate0, 0.0f, mask32);
                }
            }
            Mul(state0, state0, gate0, mask32);
            Mul(state1, state1, gate1, mask32);
            Add(state0, state0, d0, mask32);
            Add(state1, state1, d1, mask32);

            if constexpr (STATE_FP32) {
                StoreAlign(stateFp32 + fp32Offset, state0, mask32);
                StoreAlign(stateFp32 + fp32Offset + FP32_PER_REG, state1, mask32);
            } else {
                FwdHStoreBf16(stateBf16 + bf16Offset, state0, state1, mask32, mask16);
            }
            if constexpr (WRITE_H && STATE_FP32) {
                // FP32 state 没有独立的 32 KiB H bank。当前 D 行读入寄存器后，允许把
                // BF16 H 原位写到 D slot 的低半区；写地址只覆盖已经消费的 D 行。
                FwdHStoreBf16(hNext + bf16Offset, state0, state1, mask32, mask16);
            }
        }
    }
}

__simd_vf__ inline void FwdHSMinusOneArch35Vf(__ubuf__ float *state,
                                               __ubuf__ bfloat16_t *h)
{
    // S-1: H0 = cast_BF16(initial_state)。输入和输出保持 state_v_first 指定的同一物理顺序。
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t BF16_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> state0;
    RegTensor<float> state1;
    #pragma unroll 2
    for (uint16_t row = 0; row < 128; ++row) {
        for (uint16_t col = 0; col < FWD_H_V; col += BF16_PER_REG) {
            const uint32_t fp32Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            const uint32_t bf16Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            LoadAlign(state0, state + fp32Offset);
            LoadAlign(state1, state + fp32Offset + FP32_PER_REG);
            FwdHStoreBf16(h + bf16Offset, state0, state1, mask32, mask16);
        }
    }
}

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
class ChunkFwdHVecArch35 {
public:
    __aicore__ inline void Init(const FwdHKernelArgs &args)
    {
        args_ = args;
        coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        coreNum_ = AscendC::GetBlockNum();
        aiv_ = AscendC::GetSubBlockIdx();
        InitBuffers();
        InitLocalEvents();
    }

    __aicore__ inline void Process()
    {
        const uint32_t sequenceCount = args_.tiling.isVariedLen != 0
                                           ? static_cast<uint32_t>(args_.tiling.tokenBatch)
                                           : static_cast<uint32_t>(args_.tiling.shapeBatch);
        const uint32_t rounds = FwdHHeadRoundsPerSequence<CompilePolicy::GATE_MODE>(args_.tiling);
        // 同一 sequence 的所有 head-round 绑定到同一核，round 内串行收口；
        // ProcessWorkUnit 返回时已完成本轮 ROUND_DONE/ACK，下一轮才允许预取。
        for (uint32_t sequence = coreIdx_; sequence < sequenceCount; sequence += coreNum_) {
            const FwdHSequenceSpan sequenceSpan = FwdHResolveSequence(args_, sequence);
            for (uint32_t round = 0; round < rounds; ++round) {
                const FwdHWorkUnit unit{sequenceSpan,
                                        FwdHBuildHeadRound<CompilePolicy::GATE_MODE>(args_.tiling,
                                                                                      round)};
                if (unit.sequence.chunkCount != 0 && unit.headRound.activeHeadCount != 0) {
                    ProcessWorkUnit(unit);
                }
            }
        }
        DrainLocalEvents();
    }

private:
    __aicore__ inline void InitBuffers()
    {
        GetTPipePtr()->InitBuffer(ubBuf_, 248 * 1024);
        AscendC::LocalTensor<uint8_t> ub = ubBuf_.Get<uint8_t>();
        local_[0] = ub[FWD_H_UB_LOCAL0_BASE].template ReinterpretCast<uint8_t>();
        local_[1] = ub[FWD_H_UB_LOCAL1_BASE].template ReinterpretCast<uint8_t>();
        right_[0] = ub[FWD_H_UB_LOCAL0_BASE + FWD_H_TOKEN_MATRIX_FP32_BYTES].template ReinterpretCast<bfloat16_t>();
        right_[1] = ub[FWD_H_UB_LOCAL1_BASE + FWD_H_TOKEN_MATRIX_FP32_BYTES].template ReinterpretCast<bfloat16_t>();
        stateBf16_[0] = ub[FWD_H_UB_BF16_STATE_BASE].template ReinterpretCast<bfloat16_t>();
        stateBf16_[1] = ub[FWD_H_UB_BF16_STATE_BASE + FWD_H_STATE_BF16_BYTES].template ReinterpretCast<bfloat16_t>();
        workBf16_[0] = ub[FWD_H_UB_BF16_WORK_BASE].template ReinterpretCast<bfloat16_t>();
        workBf16_[1] = ub[FWD_H_UB_BF16_WORK_BASE + FWD_H_TOKEN_MATRIX_BF16_BYTES].template ReinterpretCast<bfloat16_t>();
        stateFp32_ = ub[FWD_H_UB_FP32_STATE_BASE].template ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            sminusH_[slot] = ub[slot * FWD_H_STATE_BF16_BYTES].template ReinterpretCast<bfloat16_t>();
            sminusState_[slot] = ub[64 * 1024 + slot * FWD_H_STATE_FP32_BYTES].template ReinterpretCast<float>();
        }
        gate_[0] = ub[FWD_H_UB_GATE_BASE].template ReinterpretCast<GateT>();
        gate_[1] = ub[FWD_H_UB_GATE_BASE + 512].template ReinterpretCast<GateT>();
        alpha_[0] = ub[FWD_H_UB_GATE_BASE + 1024].template ReinterpretCast<float>();
        alpha_[1] = ub[FWD_H_UB_GATE_BASE + 1056].template ReinterpretCast<float>();
    }

    __aicore__ inline void InitLocalEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            ioFreeEvent_[slot] = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
            ioReadyEvent_[slot] = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
            ioDoneEvent_[slot] = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
    }

    __aicore__ inline AscendC::TEventID IoFreeEvent(uint32_t slot) const
    {
        return ioFreeEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID IoReadyEvent(uint32_t slot) const
    {
        return ioReadyEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID IoDoneEvent(uint32_t slot) const
    {
        return ioDoneEvent_[slot];
    }

    __aicore__ inline void DrainLocalEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
    }

    __aicore__ inline uint64_t HOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHHOffset(args_.tiling, unit.sequence, head.hv, chunk.globalChunk);
    }

    __aicore__ inline uint64_t UOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, FWD_H_V);
    }

    __aicore__ inline uint64_t GateOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head) const
    {
        const uint32_t dim = CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? 1 : FWD_H_K;
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, dim);
    }

    __aicore__ inline void CopyGateToUb(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                        const FwdHHeadBinding &head)
    {
        AscendC::GlobalTensor<GateT> gateGm;
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(
            CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? args_.g : args_.gk));
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            AscendC::DataCopyPadExtParams<GateT> pad{false, 0, 0, 0};
            const uint32_t gateBytes = static_cast<uint32_t>(chunk.validTokens * sizeof(GateT));
            AscendC::DataCopyExtParams copy{1, gateBytes, 0, 0, 0};
            AscendC::DataCopyPad(gate_[head.localSlot], gateGm[GateOffset(unit, chunk, head)], copy, pad);
        } else {
            const uint64_t lastOffset = GateOffset(unit, chunk, head) +
                                        static_cast<uint64_t>(chunk.validTokens - 1) * FWD_H_K;
            AscendC::DataCopy(gate_[head.localSlot], gateGm[lastOffset], FWD_H_K);
        }
    }

    __aicore__ inline void PrefetchSMinusOneHead(const FwdHWorkUnit &unit,
                                                 const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        AscendC::GlobalTensor<float> initial;
        initial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
        const uint64_t stateBase = FwdHStateOffset(args_.tiling, unit.sequence.sequence,
                                                   head.hv, 0, 0);
        AscendC::DataCopy(sminusState_[slot], initial[stateBase], FWD_H_K * FWD_H_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
    }

    __aicore__ inline void ConsumeSMinusOneHead(const FwdHWorkUnit &unit,
                                                const FwdHHeadBinding &head)
    {
        // S-1：H0=cast_BF16(initial_state)。输入输出保持 state_v_first 指定的物理顺序。
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        AscendC::VF_CALL<FwdHSMinusOneArch35Vf>(
            reinterpret_cast<__ubuf__ float *>(sminusState_[slot].GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(sminusH_[slot].GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::GlobalTensor<bfloat16_t> h;
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        const FwdHChunkSpan firstChunk = FwdHBuildChunk(unit.sequence, 0);
        AscendC::DataCopy(h[HOffset(unit, firstChunk, head)], sminusH_[slot],
                          FWD_H_K * FWD_H_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    __aicore__ inline void RunSMinusOne(const FwdHWorkUnit &unit)
    {
        // S-1：StateT=FP32 且存在 initial_state 时，H0=cast_BF16(R0)；
        // 输入输出保持 state_v_first 指定的物理顺序，不依赖外部转置。
        if constexpr (!CompilePolicy::STATE_FP32) {
            return;
        }
        if (args_.tiling.useInitialState == 0) {
            return;
        }
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
            PrefetchSMinusOneHead(unit, head);
            if (coreHeadId > 0) {
                ConsumeSMinusOneHead(
                    unit, unit.headRound.heads[aiv_ + 2 * (coreHeadId - 1)]);
            }
        }
        if (localHeads > 0) {
            ConsumeSMinusOneHead(
                unit, unit.headRound.heads[aiv_ + 2 * (localHeads - 1)]);
        }
        // S-1 与主循环复用 [0,192) KiB。先 drain 全部 active bank，再统一发布 H ready。
        for (uint32_t slot = 0; slot < localHeads; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_H_READY_FLAG, head.localSlot));
        }
    }

    template <bool HAS_P, bool WRITE_RIGHT, bool ZERO_STATE>
    __aicore__ inline void ConsumeStage1Head(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        if constexpr (HAS_P) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(FwdHAivLocalFlag(FWD_H_P_READY_FLAG, slot));
        }
        using PType = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
        __ubuf__ PType *p = reinterpret_cast<__ubuf__ PType *>(local_[slot].GetPhyAddr());
        __ubuf__ bfloat16_t *right = reinterpret_cast<__ubuf__ bfloat16_t *>(right_[slot].GetPhyAddr());
        __ubuf__ bfloat16_t *state = reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf16_[slot].GetPhyAddr());
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            AscendC::VF_CALL<FwdHStage1Arch35Vf<HAS_P, true, WRITE_RIGHT, ZERO_STATE,
                                                CompilePolicy::USE_EXP2, PType, GateT>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(workBf16_[slot].GetPhyAddr()), p,
                reinterpret_cast<__ubuf__ GateT *>(gate_[slot].GetPhyAddr()), right, state,
                reinterpret_cast<__ubuf__ float *>(alpha_[slot].GetPhyAddr()),
                static_cast<uint16_t>(chunk.validTokens));
        } else {
            AscendC::VF_CALL<FwdHStage1Arch35Vf<HAS_P, false, WRITE_RIGHT, ZERO_STATE,
                                                CompilePolicy::USE_EXP2, PType, GateT>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(workBf16_[slot].GetPhyAddr()), p,
                reinterpret_cast<__ubuf__ GateT *>(gate_[slot].GetPhyAddr()), right, state,
                reinterpret_cast<__ubuf__ float *>(alpha_[slot].GetPhyAddr()),
                static_cast<uint16_t>(chunk.validTokens));
        }
        if constexpr (HAS_P) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_V>(FwdHAivLocalFlag(FWD_H_P_FREE_FLAG, slot));
        }

        AscendC::GlobalTensor<bfloat16_t> vNew;
        AscendC::GlobalTensor<bfloat16_t> rightGm;
        AscendC::GlobalTensor<bfloat16_t> h;
        vNew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.vNew));
        rightGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + args_.tiling.vUpdateWorkspaceOffset));
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::DataCopy(vNew[UOffset(unit, chunk, head)], workBf16_[slot],
                          chunk.validTokens * FWD_H_V);
        if constexpr (WRITE_RIGHT) {
            const uint64_t rightOffset = FwdHCoreSlotOffset(coreIdx_, head.roundHead,
                                                            FWD_H_CHUNK * FWD_H_V);
            AscendC::DataCopy(rightGm[rightOffset], right_[slot], chunk.validTokens * FWD_H_V);
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_RIGHT_READY_FLAG, slot));
        }
        if constexpr (!CompilePolicy::STATE_FP32) {
            if (chunk.first) {
                AscendC::DataCopy(h[HOffset(unit, chunk, head)], stateBf16_[slot], FWD_H_K * FWD_H_V);
            }
        } else if constexpr (ZERO_STATE) {
            AscendC::DataCopy(h[HOffset(unit, chunk, head)], stateBf16_[slot], FWD_H_K * FWD_H_V);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    __aicore__ inline void RunStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage1：V_new=cast_BF16(fp32(U)-fp32(P))；g-only 同时生成
        // V_new_g=cast_BF16(E(g_last-g_i)*V_new_fp32)，gk-only 的右矩阵就是 V_new。
        const bool hasP = !(chunk.first && args_.tiling.useInitialState == 0);
        const bool writeRight = args_.tiling.storeFinalState != 0 || !chunk.last;
        const bool zeroState = chunk.first && args_.tiling.useInitialState == 0;
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
            const uint32_t slot = head.localSlot;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::GlobalTensor<bfloat16_t> u;
            u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
            AscendC::DataCopy(workBf16_[slot], u[UOffset(unit, chunk, head)], chunk.validTokens * FWD_H_V);
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                if (writeRight) {
                    CopyGateToUb(unit, chunk, head);
                }
            }
            if (!CompilePolicy::STATE_FP32 && chunk.first && args_.tiling.useInitialState != 0) {
                AscendC::GlobalTensor<bfloat16_t> initial;
                initial.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.initialState));
                const uint64_t stateOffset = FwdHStateOffset(args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                AscendC::DataCopy(stateBf16_[slot], initial[stateOffset], FWD_H_K * FWD_H_V);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
            if (coreHeadId > 0) {
                const FwdHHeadBinding &previous = unit.headRound.heads[aiv_ + 2 * (coreHeadId - 1)];
                DispatchStage1(unit, chunk, previous, hasP, writeRight, zeroState);
            }
        }
        if (localHeads > 0) {
            const FwdHHeadBinding &last = unit.headRound.heads[aiv_ + 2 * (localHeads - 1)];
            DispatchStage1(unit, chunk, last, hasP, writeRight, zeroState);
        }
    }

    __aicore__ inline void DispatchStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head, bool hasP,
                                          bool writeRight, bool zeroState)
    {
        if (hasP) {
            if (writeRight) {
                zeroState ? ConsumeStage1Head<true, true, true>(unit, chunk, head)
                          : ConsumeStage1Head<true, true, false>(unit, chunk, head);
            } else {
                zeroState ? ConsumeStage1Head<true, false, true>(unit, chunk, head)
                          : ConsumeStage1Head<true, false, false>(unit, chunk, head);
            }
        } else if (writeRight) {
            zeroState ? ConsumeStage1Head<false, true, true>(unit, chunk, head)
                      : ConsumeStage1Head<false, true, false>(unit, chunk, head);
        } else {
            zeroState ? ConsumeStage1Head<false, false, true>(unit, chunk, head)
                      : ConsumeStage1Head<false, false, false>(unit, chunk, head);
        }
    }

    template <bool WRITE_H, bool ZERO_STATE>
    __aicore__ inline void ConsumeStage3Head(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(FwdHAivLocalFlag(FWD_H_D_READY_FLAG, slot));
        __ubuf__ float *d = reinterpret_cast<__ubuf__ float *>(local_[slot].GetPhyAddr());
        __ubuf__ bfloat16_t *hNext = nullptr;
        if constexpr (CompilePolicy::STATE_FP32) {
            hNext = reinterpret_cast<__ubuf__ bfloat16_t *>(local_[slot].GetPhyAddr());
        } else {
            hNext = reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf16_[slot].GetPhyAddr());
        }
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            AscendC::VF_CALL<FwdHStage3Arch35Vf<CompilePolicy::STATE_FP32, true, STATE_V_FIRST,
                                                WRITE_H, ZERO_STATE, CompilePolicy::USE_EXP2, GateT>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf16_[slot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(stateFp32_.GetPhyAddr()), d,
                reinterpret_cast<__ubuf__ GateT *>(gate_[slot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(alpha_[slot].GetPhyAddr()), hNext);
        } else {
            AscendC::VF_CALL<FwdHStage3Arch35Vf<CompilePolicy::STATE_FP32, false, STATE_V_FIRST,
                                                WRITE_H, ZERO_STATE, CompilePolicy::USE_EXP2, GateT>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(stateBf16_[slot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(stateFp32_.GetPhyAddr()), d,
                reinterpret_cast<__ubuf__ GateT *>(gate_[slot].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(alpha_[slot].GetPhyAddr()), hNext);
        }
        if constexpr (!WRITE_H || !CompilePolicy::STATE_FP32) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_V>(FwdHAivLocalFlag(FWD_H_D_FREE_FLAG, slot));
        }

        AscendC::GlobalTensor<bfloat16_t> h;
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        if constexpr (WRITE_H) {
            if constexpr (CompilePolicy::STATE_FP32) {
                AscendC::GlobalTensor<float> state;
                state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace));
                const uint64_t stateOffset = args_.tiling.kDecayWorkspaceOffset / sizeof(float) +
                    FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_K * FWD_H_V);
                AscendC::DataCopy(state[stateOffset], stateFp32_, FWD_H_K * FWD_H_V);
            }
            const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunk.chunk + 1);
            if constexpr (CompilePolicy::STATE_FP32) {
                AscendC::LocalTensor<bfloat16_t> hLocal =
                    local_[slot].template ReinterpretCast<bfloat16_t>();
                AscendC::DataCopy(h[HOffset(unit, next, head)], hLocal, FWD_H_K * FWD_H_V);
                // H 的 MTE3 完成前，D slot 仍是 MTE3 输入，不能交给下一 Stage0/Stage2。
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    FwdHAivLocalFlag(FWD_H_D_FREE_FLAG, slot));
            } else {
                AscendC::DataCopy(h[HOffset(unit, next, head)], stateBf16_[slot],
                                  FWD_H_K * FWD_H_V);
            }
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(FwdHAivLocalFlag(FWD_H_H_READY_FLAG, slot));
        } else {
            if (args_.tiling.storeFinalState != 0) {
                if constexpr (CompilePolicy::STATE_FP32) {
                    AscendC::GlobalTensor<float> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.finalState));
                    const uint64_t offset = FwdHStateOffset(args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                    AscendC::DataCopy(finalState[offset], stateFp32_, FWD_H_K * FWD_H_V);
                } else {
                    AscendC::GlobalTensor<bfloat16_t> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.finalState));
                    const uint64_t offset = FwdHStateOffset(args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                    AscendC::DataCopy(finalState[offset], stateBf16_[slot], FWD_H_K * FWD_H_V);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    __aicore__ inline void RunStage3(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage3：g-only 为 R_next=E(g_last)R+D；gk-only 为
        // R_next[k,v]=E(gk_last[k])R[k,v]+D[k,v]，并按需生成下一 H 或 final_state。
        const bool writeH = !chunk.last;
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        if constexpr (CompilePolicy::STATE_FP32) {
            // FP32 state 与仅用于 H0 的两个 BF16 state slot 复用 [128,192) KiB。
            // 首个 head 覆写共享区前，必须确认本 AIV 当前 round 的全部 MTE3 已完成。
            for (uint32_t slot = 0; slot < localHeads; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            }
        }
        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
            const uint32_t slot = head.localSlot;
            if constexpr (CompilePolicy::STATE_FP32) {
                if (coreHeadId > 0) {
                    const uint32_t previousSlot = unit.headRound.heads[
                        aiv_ + 2 * (coreHeadId - 1)].localSlot;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(previousSlot));
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(previousSlot));
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            if constexpr (CompilePolicy::STATE_FP32) {
                AscendC::GlobalTensor<float> state;
                const uint64_t workspaceBase = args_.tiling.kDecayWorkspaceOffset / sizeof(float);
                state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace));
                const uint64_t offset = workspaceBase +
                    FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_K * FWD_H_V);
                if (chunk.first) {
                    if (args_.tiling.useInitialState != 0) {
                        AscendC::GlobalTensor<float> initial;
                        initial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
                        const uint64_t initialOffset = FwdHStateOffset(args_.tiling, unit.sequence.sequence,
                                                                      head.hv, 0, 0);
                        AscendC::DataCopy(stateFp32_, initial[initialOffset], FWD_H_K * FWD_H_V);
                    }
                } else {
                    AscendC::DataCopy(stateFp32_, state[offset], FWD_H_K * FWD_H_V);
                }
            }
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
                CopyGateToUb(unit, chunk, head);
            }
            const bool zeroState = chunk.first && args_.tiling.useInitialState == 0;
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
            } else if constexpr (CompilePolicy::STATE_FP32) {
                if (!zeroState) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
                }
            }
            if (writeH) {
                zeroState ? ConsumeStage3Head<true, true>(unit, chunk, head)
                          : ConsumeStage3Head<true, false>(unit, chunk, head);
            } else {
                zeroState ? ConsumeStage3Head<false, true>(unit, chunk, head)
                          : ConsumeStage3Head<false, false>(unit, chunk, head);
            }
        }
    }

    __aicore__ inline void ProcessWorkUnit(const FwdHWorkUnit &unit)
    {
        const bool hasCubeWork = args_.tiling.useInitialState || args_.tiling.storeFinalState ||
            args_.tiling.seqlen > static_cast<int64_t>(FWD_H_CHUNK);
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        RunSMinusOne(unit);
        for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
            const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
            if (chunkId > 0) {
                for (uint32_t slot = 0; slot < localHeads; ++slot) {
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                        FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, slot));
                }
            }
            RunStage1(unit, chunk);
            if (args_.tiling.storeFinalState != 0 || !chunk.last) {
                RunStage3(unit, chunk);
            }
        }
        const bool lastHasStage2 = args_.tiling.storeFinalState != 0;
        if (lastHasStage2) {
            for (uint32_t slot = 0; slot < localHeads; ++slot) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, slot));
            }
        }
        // 跨 head_round 前必须把本轮所有 VEC->MTE3 写回收口。AIC 收到 ROUND_DONE 后
        // 才能开始下一轮 kg/H/W 的 MTE2，避免上一轮 MTE3 与下一轮预取跨 round 交叠。
        for (uint32_t slot = 0; slot < localHeads; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
        if (hasCubeWork) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(FwdHAivLocalFlag(FWD_H_ROUND_DONE_FLAG, 0));
            AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(FwdHAivLocalFlag(FWD_H_H_READY_FLAG, 0));
        }
    }

    FwdHKernelArgs args_{};
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t aiv_ = 0;
    AscendC::TEventID ioFreeEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TEventID ioReadyEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TEventID ioDoneEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_{};
    AscendC::LocalTensor<uint8_t> local_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> right_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> stateBf16_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> workBf16_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> sminusH_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> sminusState_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> stateFp32_{};
    AscendC::LocalTensor<GateT> gate_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> alpha_[FWD_H_AIV_HEAD_SLOTS]{};
};

} // namespace GDN

#endif // ARCH35_CHUNK_FWD_H_VEC_H
