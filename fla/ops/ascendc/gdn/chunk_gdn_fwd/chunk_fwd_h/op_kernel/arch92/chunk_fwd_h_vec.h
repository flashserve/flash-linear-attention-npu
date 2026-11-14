/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH92_CHUNK_FWD_H_VEC_H
#define ARCH92_CHUNK_FWD_H_VEC_H

#include <type_traits>

#include "../chunk_fwd_h_policy.h"
#include "../chunk_fwd_h_utils.h"
#include "kernel_utils/vector/regbase.hpp"

namespace GDN {

using namespace AscendC::MicroAPI;

constexpr float FWD_H_LN2 = 0.69314718055994530942f;
constexpr uint32_t FWD_H_ARCH92_GATE_SLOT_BYTES = 512;
constexpr uint32_t FWD_H_ARCH92_GATE_SCALE_OFFSET = 256;

constexpr CastTrait FWD_H_B16_TO_F32_ZERO = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__simd_callee__ inline void FwdHLoadFloatPair(RegTensor<float> &zero, RegTensor<float> &one,
                                               __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void FwdHStoreFloatPair(__ubuf__ float *dst,
                                                RegTensor<float> &zero, RegTensor<float> &one,
                                                MaskReg &mask32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, mask32);
}

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

template <typename T>
__simd_callee__ inline void FwdHLoadGateVector(RegTensor<float> &dst,
                                                __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        Cast<float, T, FWD_H_B16_TO_F32_ZERO>(dst, raw, mask);
    }
}

__simd_callee__ inline void FwdHStoreGateVector(__ubuf__ float *dst,
                                                 RegTensor<float> &src, MaskReg &mask)
{
    DataCopy<float, StoreDist::DIST_NORM_B32>(dst, src, mask);
}

__simd_callee__ inline void FwdHStoreGateScalar(__ubuf__ float *dst,
                                                 RegTensor<float> &src, MaskReg &mask)
{
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst, src, mask);
}

__simd_callee__ inline void FwdHStoreBf16(__ubuf__ bfloat16_t *dst,
                                          RegTensor<float> &zero, RegTensor<float> &one,
                                          MaskReg &mask32, MaskReg &mask16)
{
    RegTensor<bfloat16_t> output;
    CastFloat2Half<bfloat16_t>(output, zero, one, mask32);
    StoreAlign(dst, output, mask16);
}

template <bool USE_EXP2, typename GateT>
__simd_vf__ inline void FwdHPrepareScalarGateArch92Vf(__ubuf__ float *gateScale,
                                                       __ubuf__ GateT *gate,
                                                       __ubuf__ float *alpha,
                                                       uint16_t validTokens)
{
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> lastGate;
    FwdHLoadScalar<GateT>(lastGate, gate + validTokens - 1, mask16, mask32);

    RegTensor<float> alphaReg;
    Adds(alphaReg, lastGate, 0.0f, mask32);
    if constexpr (USE_EXP2) {
        Muls(alphaReg, alphaReg, FWD_H_LN2, mask32);
    }
    Exp(alphaReg, alphaReg, mask32);
    uint32_t scalarCount = 1;
    MaskReg scalarMask = UpdateMask<float>(scalarCount);
    FwdHStoreGateVector(alpha, alphaReg, scalarMask);

    RegTensor<float> gateReg;
    if (validTokens == FWD_H_CHUNK) {
        uint32_t gateCount = validTokens;
        MaskReg gateMask = UpdateMask<float>(gateCount);
        FwdHLoadGateVector<GateT>(gateReg, gate, gateMask);
        Sub(gateReg, lastGate, gateReg, gateMask);
        if constexpr (USE_EXP2) {
            Muls(gateReg, gateReg, FWD_H_LN2, gateMask);
        }
        Exp(gateReg, gateReg, gateMask);
        FwdHStoreGateVector(gateScale, gateReg, gateMask);
    } else {
        // DataCopyPad initializes only the valid tail and its final 32-byte block.
        // Keep tail loads scalar so the VF never reads the uninitialized remainder
        // of the 256-byte raw-gate bank. Full chunks retain the single vector Exp.
        for (uint16_t row = 0; row < validTokens; ++row) {
            FwdHLoadScalar<GateT>(gateReg, gate + row, mask16, mask32);
            Sub(gateReg, lastGate, gateReg, mask32);
            if constexpr (USE_EXP2) {
                Muls(gateReg, gateReg, FWD_H_LN2, mask32);
            }
            Exp(gateReg, gateReg, mask32);
            uint32_t one = 1;
            MaskReg oneMask = UpdateMask<float>(one);
            FwdHStoreGateScalar(gateScale + row, gateReg, oneMask);
        }
    }
}

template <bool HAS_P, bool SCALAR_G, bool WRITE_RIGHT, bool ZERO_STATE, typename PType>
__simd_vf__ inline void FwdHStage1Arch92Vf(__ubuf__ bfloat16_t *uAndVNew,
                                           __ubuf__ PType *p,
                                           __ubuf__ float *gateScale,
                                           __ubuf__ bfloat16_t *right,
                                           __ubuf__ bfloat16_t *state,
                                           uint16_t validTokens)
{
    // Stage1:
    //   V_new_fp32[i,:] = fp32(U[i,:]) - fp32(P[i,:])，HAS_P=false 时 P=0；
    //   V_new[i,:] = cast_BF16(V_new_fp32[i,:])；
    //   g-only: right[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32[i,:])，alpha=E(g_last)；
    //   gk-only: right=V_new。E 由 USE_EXP2 在编译期选择。
    // 本 VF 覆盖当前 head 的全部 MTE2 后向量计算，内部没有运行期分支。
    constexpr uint16_t BF16_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint16_t STATE_ROWS = FWD_H_STATE_BF16_BYTES / sizeof(bfloat16_t) / BF16_PER_REG;
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();

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
    RegTensor<float> scaleReg;
    #pragma unroll 2
    for (uint16_t row = 0; row < validTokens; ++row) {
        const uint32_t rowBf16Offset = static_cast<uint32_t>(row) * FWD_H_V;
        const uint32_t rowFp32Offset = static_cast<uint32_t>(row) * FWD_H_V;
        if constexpr (SCALAR_G && WRITE_RIGHT) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, gateScale + row);
        }
        for (uint16_t col = 0; col < FWD_H_V; col += BF16_PER_REG) {
            const uint32_t bf16Offset = rowBf16Offset + col;
            const uint32_t fp32Offset = rowFp32Offset + col;
            FwdHLoadAsFloat<bfloat16_t>(u0, u1, uAndVNew + bf16Offset, mask16);
            if constexpr (HAS_P) {
                if constexpr (std::is_same<PType, float>::value) {
                    FwdHLoadFloatPair(p0, p1, p + fp32Offset);
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
                    Mul(u0, u0, scaleReg, mask32);
                    Mul(u1, u1, scaleReg, mask32);
                }
                FwdHStoreBf16(right + bf16Offset, u0, u1, mask32, mask16);
            }
        }
    }
}

template <bool STATE_FP32, bool SCALAR_G, bool STATE_V_FIRST, bool WRITE_H,
          bool ZERO_STATE, bool USE_EXP2, typename GateT>
__simd_vf__ inline void FwdHStage3Arch92Vf(__ubuf__ bfloat16_t *stateBf16,
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
    } else if constexpr (STATE_V_FIRST) {
        // 物理 [V,K] 布局的 128 列恰好由一对 FP32 寄存器覆盖。
        // 所有 V 行使用同一份 E(gk_last)，只计算一次并跨行复用。
        if constexpr (std::is_same<GateT, float>::value) {
            FwdHLoadFloatPair(gate0, gate1, gkLast);
        } else {
            FwdHLoadAsFloat<GateT>(gate0, gate1, gkLast, mask16);
        }
        if constexpr (USE_EXP2) {
            Muls(gate0, gate0, FWD_H_LN2, mask32);
            Muls(gate1, gate1, FWD_H_LN2, mask32);
        }
        Exp(gate0, gate0, mask32);
        Exp(gate1, gate1, mask32);
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
                    FwdHLoadFloatPair(state0, state1, stateFp32 + fp32Offset);
                }
            } else {
                FwdHLoadAsFloat<bfloat16_t>(state0, state1, stateBf16 + bf16Offset, mask16);
            }
            FwdHLoadFloatPair(d0, d1, d + fp32Offset);

            if constexpr (!SCALAR_G) {
                if constexpr (!STATE_V_FIRST) {
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
                FwdHStoreFloatPair(stateFp32 + fp32Offset, state0, state1, mask32);
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

__simd_vf__ inline void FwdHSMinusOneArch92Vf(__ubuf__ float *state,
                                               __ubuf__ bfloat16_t *h)
{
    // S-1: H0 = cast_BF16(initial_state)。输入和输出保持 state_v_first 指定的同一物理顺序。
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
            FwdHLoadFloatPair(state0, state1, state + fp32Offset);
            FwdHStoreBf16(h + bf16Offset, state0, state1, mask32, mask16);
        }
    }
}

template <typename StateT>
__simd_vf__ inline void FwdHZeroStateArch92Vf(__ubuf__ StateT *state)
{
    constexpr uint16_t ELEMENTS = AscendC::VECTOR_REG_WIDTH / sizeof(StateT);
    MaskReg mask = CreateMask<StateT, MaskPattern::ALL>();
    RegTensor<StateT> zero;
    Duplicate(zero, static_cast<StateT>(0), mask);
    for (uint16_t i = 0; i < FWD_H_K * FWD_H_V / ELEMENTS; ++i) {
        StoreAlign(state + static_cast<uint32_t>(i) * ELEMENTS, zero, mask);
    }
}

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
class ChunkFwdHVecArch92 {
public:
    __aicore__ inline void Init(const FwdHKernelArgs &args)
    {
        args_ = args;
        coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        coreNum_ = AscendC::GetBlockNum();
        ub_ = FwdHArch92::UbBuffer();
        InitLocalEvents();
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
            if (unit.sequence.chunkCount == 0) {
                StoreEmptySequenceFinalState(unit);
            } else if (unit.headRound.activeHeadCount != 0) {
                ProcessWorkUnit(unit, cursor < range.end);
            }
        }
        DrainLocalEvents();
    }

private:
    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> UbAt(uint32_t byteOffset)
    {
        return ub_[byteOffset].template ReinterpretCast<T>();
    }

    __aicore__ inline AscendC::LocalTensor<float> PdSlot(uint32_t slot)
    {
        return UbAt<float>(FwdHArch92::PD_BASE + slot * FWD_H_STATE_FP32_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<float> StateFp32(uint32_t slot)
    {
        return UbAt<float>(FwdHArch92::STATE_BASE + slot * FWD_H_STATE_FP32_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> StateBf16Slot(uint32_t slot)
    {
        // BF16 state and the zero-H0 temporary use the low half of this head's state.
        return StateFp32(slot).template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> WorkBf16Slot(uint32_t slot)
    {
        return UbAt<bfloat16_t>(FwdHArch92::INPUT_BASE + slot * FWD_H_TOKEN_MATRIX_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> RightSlot(uint32_t slot)
    {
        return UbAt<bfloat16_t>(FwdHArch92::RIGHT_BASE + slot * FWD_H_TOKEN_MATRIX_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<GateT> GateSlot(uint32_t slot)
    {
        return UbAt<GateT>(FwdHArch92::GATE_BASE + slot * FWD_H_ARCH92_GATE_SLOT_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<float> GateScaleSlot(uint32_t slot)
    {
        return UbAt<float>(FwdHArch92::GATE_BASE + slot * FWD_H_ARCH92_GATE_SLOT_BYTES +
                           FWD_H_ARCH92_GATE_SCALE_OFFSET);
    }

    __aicore__ inline AscendC::LocalTensor<float> AlphaSlot(uint32_t chunk, uint32_t slot)
    {
        return UbAt<float>(FwdHArch92::ALPHA_BASE + FwdHArch92::BankSlot(chunk, slot) * 32);
    }

    // Event numbers are local to each direction. MTE3_V 0/1 protects right,
    // 2/3 protects state; MTE3_MTE2 0/1 protects initialization, 2/3 U/vNew.
    static constexpr uint32_t WORK_EVENT_BASE = 2;

    __aicore__ inline void InitLocalEvents()
    {
        for (uint32_t slot = 0; slot < FwdHArch92::HEADS; ++slot) {
            const auto ioEvent = static_cast<AscendC::TEventID>(slot);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(WORK_EVENT_BASE + slot));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ioEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ioEvent);
        }
    }

    __aicore__ inline void DrainLocalEvents()
    {
        for (uint32_t slot = 0; slot < FwdHArch92::HEADS; ++slot) {
            const auto ioEvent = static_cast<AscendC::TEventID>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(WORK_EVENT_BASE + slot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ioEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ioEvent);
        }
    }

    __aicore__ inline bool NeedsStage3(const FwdHChunkSpan &chunk) const
    {
        return args_.tiling.storeFinalState != 0 || !chunk.last;
    }

    __aicore__ inline void StoreEmptySequenceFinalState(const FwdHWorkUnit &unit)
    {
        if (!args_.tiling.storeFinalState) {
            return;
        }
        using StateT = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
        AscendC::GlobalTensor<StateT> finalState;
        finalState.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(args_.finalState));
        for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
            const auto ioEvent = static_cast<AscendC::TEventID>(coreHeadId);
            const uint64_t offset = FwdHStateOffset<STATE_V_FIRST>(
                args_.tiling, unit.sequence.sequence, unit.headRound.heads[coreHeadId].hv, 0, 0);
            auto state = StateFp32(coreHeadId).template ReinterpretCast<StateT>();
            // The previous unit may still be writing from this state bank. Empty
            // units use only the local IO credit; the AIC skips them entirely.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioEvent);
            if (args_.tiling.useInitialState) {
                AscendC::GlobalTensor<StateT> initial;
                initial.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(args_.initialState));
                AscendC::DataCopy(state, initial[offset], FWD_H_K * FWD_H_V);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ioEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ioEvent);
            if (!args_.tiling.useInitialState) {
                AscendC::VF_CALL<FwdHZeroStateArch92Vf<StateT>>(
                    reinterpret_cast<__ubuf__ StateT *>(state.GetPhyAddr()));
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ioEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ioEvent);
            // Copy the physical state verbatim: no transpose or BF16 round-trip.
            AscendC::DataCopy(finalState[offset], state, FWD_H_K * FWD_H_V);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ioEvent);
        }
    }

    __aicore__ inline void LoadInitialState(const FwdHWorkUnit &unit, const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        const auto ioEvent = static_cast<AscendC::TEventID>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ioEvent);
        if (args_.tiling.useInitialState) {
            const uint64_t offset = FwdHStateOffset<STATE_V_FIRST>(
                args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
            if constexpr (CompilePolicy::STATE_FP32) {
                AscendC::GlobalTensor<float> initial;
                initial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
                AscendC::DataCopy(StateFp32(slot), initial[offset], FWD_H_K * FWD_H_V);
            } else {
                AscendC::GlobalTensor<bfloat16_t> initial;
                initial.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.initialState));
                AscendC::DataCopy(StateBf16Slot(slot), initial[offset], FWD_H_K * FWD_H_V);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ioEvent);
    }

    __aicore__ inline void StoreInitialH(const FwdHWorkUnit &unit, const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        const auto ioEvent = static_cast<AscendC::TEventID>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ioEvent);
        auto hLocal = StateBf16Slot(slot);
        if (args_.tiling.useInitialState) {
            if constexpr (CompilePolicy::STATE_FP32) {
                hLocal = PdSlot(slot).template ReinterpretCast<bfloat16_t>();
                AscendC::VF_CALL<FwdHSMinusOneArch92Vf>(
                    reinterpret_cast<__ubuf__ float *>(StateFp32(slot).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t *>(hLocal.GetPhyAddr()));
            }
        } else {
            // Do not touch PD: the first D may arrive before this zero-H0 store.
            // FP32 rolling state is initialized by ZERO_STATE in the first S3.
            AscendC::VF_CALL<FwdHZeroStateArch92Vf<bfloat16_t>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(hLocal.GetPhyAddr()));
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ioEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ioEvent);
        AscendC::GlobalTensor<bfloat16_t> h;
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        const uint64_t offset = FwdHHOffset(args_.tiling, unit.sequence, head.hv, unit.sequence.chunkPrefix);
        AscendC::DataCopy(h[offset], hLocal, FWD_H_K * FWD_H_V);
        // Return the state slot after H0 is written.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<AscendC::TEventID>(WORK_EVENT_BASE + slot));
        if (args_.tiling.useInitialState) {
            // Also orders the initial PD->H store before the first Fixpipe P write.
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(FwdHArch92::H_FLAG + slot);
        }
    }

    __aicore__ inline void InitState(const FwdHWorkUnit &unit)
    {
        for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
            LoadInitialState(unit, unit.headRound.heads[coreHeadId]);
            if (coreHeadId > 0) {
                StoreInitialH(unit, unit.headRound.heads[coreHeadId - 1]);
            }
        }
        StoreInitialH(unit, unit.headRound.heads[unit.headRound.activeHeadCount - 1]);
    }

    __aicore__ inline void CopyGateToUb(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                        const FwdHHeadBinding &head, uint32_t gateSlot)
    {
        AscendC::GlobalTensor<GateT> gateGm;
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(
            CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? args_.g : args_.gk));
        gateGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        constexpr uint32_t gateDim = CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? 1 : FWD_H_K;
        const uint64_t gateOffset = FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                                                    chunk.tokenBegin, gateDim);
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            if (chunk.validTokens == FWD_H_CHUNK) {
                AscendC::DataCopy(GateSlot(gateSlot), gateGm[gateOffset], FWD_H_CHUNK);
            } else {
                AscendC::DataCopyPadExtParams<GateT> pad{false, 0, 0, 0};
                const uint32_t gateBytes = static_cast<uint32_t>(chunk.validTokens * sizeof(GateT));
                AscendC::DataCopyExtParams copy{1, gateBytes, 0, 0, 0};
                AscendC::DataCopyPad(GateSlot(gateSlot), gateGm[gateOffset], copy, pad);
            }
        } else {
            const uint64_t lastOffset = gateOffset + static_cast<uint64_t>(chunk.validTokens - 1) * FWD_H_K;
            AscendC::DataCopy(GateSlot(gateSlot), gateGm[lastOffset], FWD_H_K);
        }
    }

    __aicore__ inline void PrefetchInputs(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
            const auto workEvent = static_cast<AscendC::TEventID>(WORK_EVENT_BASE + coreHeadId);
            const FwdHHeadBinding &head = unit.headRound.heads[coreHeadId];
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(workEvent);
            if (NeedsStage3(chunk)) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<AscendC::TEventID>(coreHeadId));
            }
            AscendC::GlobalTensor<bfloat16_t> u;
            u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
            u.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            const uint64_t offset = FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                                                     chunk.tokenBegin, FWD_H_V);
            AscendC::DataCopy(WorkBf16Slot(coreHeadId), u[offset], chunk.validTokens * FWD_H_V);
            if (NeedsStage3(chunk)) {
                CopyGateToUb(unit, chunk, head, head.localSlot);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(workEvent);
        }
    }

    __aicore__ inline void PrepareGateForAllHeads(const FwdHWorkUnit &unit,
                                                  const FwdHChunkSpan &chunk)
    {
        for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
            const auto workEvent = static_cast<AscendC::TEventID>(WORK_EVENT_BASE + coreHeadId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(workEvent);
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                if (NeedsStage3(chunk)) {
                    AscendC::VF_CALL<FwdHPrepareScalarGateArch92Vf<CompilePolicy::USE_EXP2, GateT>>(
                        reinterpret_cast<__ubuf__ float *>(GateScaleSlot(coreHeadId).GetPhyAddr()),
                        reinterpret_cast<__ubuf__ GateT *>(GateSlot(coreHeadId).GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(AlphaSlot(chunk.chunk, coreHeadId).GetPhyAddr()),
                        static_cast<uint16_t>(chunk.validTokens));
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool HAS_P, bool WRITE_RIGHT>
    // 保留 leaf stage 调用边界，使各 stage 的临时标量对象复用同一栈区。
    __aicore__ inline void ConsumeStage1Head(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        const auto workEvent = static_cast<AscendC::TEventID>(WORK_EVENT_BASE + slot);
        const auto ioEvent = static_cast<AscendC::TEventID>(slot);
        using PType = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
        if constexpr (HAS_P) {
            AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(FwdHArch92::P_FLAG + slot);
        }
        if constexpr (WRITE_RIGHT) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ioEvent);
            if (chunk.chunk >= FwdHArch92::BANKS) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    FwdHArch92::RIGHT_FLAG + FwdHArch92::BankSlot(chunk.chunk, slot));
            }
        }
        AscendC::VF_CALL<FwdHStage1Arch92Vf<HAS_P, CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G,
                                          WRITE_RIGHT, false, PType>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(WorkBf16Slot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ PType *>(PdSlot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(GateScaleSlot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(RightSlot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(StateBf16Slot(slot).GetPhyAddr()),
            static_cast<uint16_t>(chunk.validTokens));
        if constexpr (HAS_P) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_V>(FwdHArch92::P_FLAG + slot);
        }
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G && WRITE_RIGHT) {
            // alpha is separate; only rawGate/scale may be overwritten by c+1 now.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ioEvent);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(workEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(workEvent);
        AscendC::GlobalTensor<bfloat16_t> vNew;
        vNew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.vNew));
        const uint64_t offset = FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                                                 chunk.tokenBegin, FWD_H_V);
        AscendC::DataCopy(vNew[offset], WorkBf16Slot(slot), chunk.validTokens * FWD_H_V);
        if constexpr (WRITE_RIGHT) {
            AscendC::GlobalTensor<bfloat16_t> rightGm;
            rightGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                args_.workspace + args_.tiling.vUpdateWorkspaceOffset));
            AscendC::DataCopy(rightGm[FwdHArch92::RightOffset(coreIdx_, chunk.chunk, slot)], RightSlot(slot),
                              chunk.validTokens * FWD_H_V);
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                FwdHArch92::RIGHT_FLAG + FwdHArch92::BankSlot(chunk.chunk, slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ioEvent);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(workEvent);
    }

    __aicore__ inline void RunStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage1：V_new=cast_BF16(fp32(U)-fp32(P))；g-only 同时生成
        // V_new_g=cast_BF16(E(g_last-g_i)*V_new_fp32)，gk-only 的右矩阵就是 V_new。
        const bool hasP = !(chunk.first && args_.tiling.useInitialState == 0);
        for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[coreHeadId];
            if (NeedsStage3(chunk)) {
                if (hasP) {
                    ConsumeStage1Head<true, true>(unit, chunk, head);
                } else {
                    ConsumeStage1Head<false, true>(unit, chunk, head);
                }
            } else if (hasP) {
                ConsumeStage1Head<true, false>(unit, chunk, head);
            } else {
                ConsumeStage1Head<false, false>(unit, chunk, head);
            }
        }
    }

    template <bool WRITE_H, bool ZERO_STATE>
    __aicore__ inline void ConsumeStage3Head(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        const auto workEvent = static_cast<AscendC::TEventID>(WORK_EVENT_BASE + slot);
        const auto ioEvent = static_cast<AscendC::TEventID>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(workEvent);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(FwdHArch92::D_FLAG + slot);
        auto hLocal = StateBf16Slot(slot);
        if constexpr (CompilePolicy::STATE_FP32) {
            hLocal = PdSlot(slot).template ReinterpretCast<bfloat16_t>();
        }
        // The existing VF reads D in increasing address order before compressing H
        // into the already-consumed low half. Preserve its Mul/Add and cast order.
        AscendC::VF_CALL<FwdHStage3Arch92Vf<CompilePolicy::STATE_FP32,
                                          CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G, STATE_V_FIRST,
                                          WRITE_H, ZERO_STATE, CompilePolicy::USE_EXP2, GateT>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(StateBf16Slot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(StateFp32(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(PdSlot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ GateT *>(GateSlot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(AlphaSlot(chunk.chunk, slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(hLocal.GetPhyAddr()));
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
            // gk_last is consumed by S3, so it cannot be prefetched at the scalar-g point.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ioEvent);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ioEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ioEvent);
        if constexpr (WRITE_H) {
            AscendC::GlobalTensor<bfloat16_t> h;
            h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
            const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunk.chunk + 1);
            const uint64_t offset = FwdHHOffset(args_.tiling, unit.sequence, head.hv, next.globalChunk);
            AscendC::DataCopy(h[offset], hLocal, FWD_H_K * FWD_H_V);
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(FwdHArch92::H_FLAG + slot);
        } else {
            const uint64_t offset = FwdHStateOffset<STATE_V_FIRST>(
                args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
            if constexpr (CompilePolicy::STATE_FP32) {
                AscendC::GlobalTensor<float> finalState;
                finalState.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.finalState));
                AscendC::DataCopy(finalState[offset], StateFp32(slot), FWD_H_K * FWD_H_V);
            } else {
                AscendC::GlobalTensor<bfloat16_t> finalState;
                finalState.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.finalState));
                AscendC::DataCopy(finalState[offset], StateBf16Slot(slot), FWD_H_K * FWD_H_V);
            }
        }
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(FwdHArch92::D_FLAG + slot);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(workEvent);
    }

    __aicore__ inline void RunStage3(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        const bool zeroState = chunk.first && args_.tiling.useInitialState == 0;
        for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[coreHeadId];
            if (!chunk.last) {
                if (zeroState) {
                    ConsumeStage3Head<true, true>(unit, chunk, head);
                } else {
                    ConsumeStage3Head<true, false>(unit, chunk, head);
                }
            } else if (zeroState) {
                ConsumeStage3Head<false, true>(unit, chunk, head);
            } else {
                ConsumeStage3Head<false, false>(unit, chunk, head);
            }
        }
    }

    __aicore__ inline void ProcessWorkUnit(const FwdHWorkUnit &unit, bool hasNextWorkUnit)
    {
        InitState(unit);
        const FwdHChunkSpan first = FwdHBuildChunk(unit.sequence, 0);
        PrefetchInputs(unit, first);
        PrepareGateForAllHeads(unit, first);
        for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
            const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
            RunStage1(unit, chunk);
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                if (!chunk.last) {
                    const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunkId + 1);
                    PrefetchInputs(unit, next);
                    PrepareGateForAllHeads(unit, next);
                }
            }
            if (NeedsStage3(chunk)) {
                RunStage3(unit, chunk);
            }
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
                if (!chunk.last) {
                    const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunkId + 1);
                    PrefetchInputs(unit, next);
                    PrepareGateForAllHeads(unit, next);
                }
            }
        }
        const uint32_t updates = unit.sequence.chunkCount - (args_.tiling.storeFinalState ? 0 : 1);
        const uint32_t firstPending = updates > FwdHArch92::BANKS ? updates - FwdHArch92::BANKS : 0;
        for (uint32_t chunkId = firstPending; chunkId < updates; ++chunkId) {
            for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    FwdHArch92::RIGHT_FLAG + FwdHArch92::BankSlot(chunkId, coreHeadId));
            }
        }
        for (uint32_t coreHeadId = 0; coreHeadId < unit.headRound.activeHeadCount; ++coreHeadId) {
            const auto workEvent = static_cast<AscendC::TEventID>(WORK_EVENT_BASE + coreHeadId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(workEvent);
            // Return the initialization credit only after the entire unit. The next
            // unit cannot load state until all current state writebacks complete.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(coreHeadId));
        }
        const bool hasCubeWork = args_.tiling.useInitialState || args_.tiling.storeFinalState ||
            args_.tiling.seqlen > static_cast<int64_t>(FWD_H_CHUNK);
        if (hasCubeWork && hasNextWorkUnit) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(FwdHArch92::ROUND_FLAG);
            AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(FwdHArch92::ROUND_FLAG);
        }
    }

    FwdHKernelArgs args_{};
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    AscendC::LocalTensor<uint8_t> ub_{};
};

} // namespace GDN

#endif // ARCH92_CHUNK_FWD_H_VEC_H
