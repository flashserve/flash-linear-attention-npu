// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "recurrent_kda_struct.h"
namespace RecurrentKda {
using namespace AscendC;
using namespace AscendC::MicroAPI;
// 输入：已就绪UB输入、参数及State bank；输出：更新State和一个token的out。
// 复用：原VF寄存器、512B scratch；同步：保留VST→VLD屏障；VF外以资源Mutex保护bank。
template <class Gate, class Beta, bool vFirst, bool fused = false,
              bool twoPass = true,
          typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage1LaunchRecurrentKdaVF(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint32_t base, uint32_t slot,
                                uint32_t parameter);

template <class Gate, class Beta, bool vFirst, bool fused, bool twoPass,
          typename inType, typename outType, typename stateType, bool stateVFirst>
__simd_vf__ inline void RecurrentKdaTokenVF(
    __ubuf__ uint8_t *memory, __ubuf__ inType *capturedQuery, __ubuf__ inType *capturedKey,
    __ubuf__ inType *capturedValue, __ubuf__ Gate *capturedGate, __ubuf__ Beta *capturedBeta,
    __ubuf__ outType *capturedOutput, __ubuf__ float *capturedParams,
    __ubuf__ volatile uint32_t *capturedEvents, uint32_t base, uint32_t launch, float scale, float lower)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    const uint32_t config = launch;
    const uint32_t features = config >> 8;
    const bool norm = fused || (features & RKDA_VF_NORM), rawGate = fused || (features & RKDA_VF_RAW_GATE);
    const bool bias = fused || (features & RKDA_VF_BIAS), sigmoid = fused || (features & RKDA_VF_SIGMOID);
    const bool negative = !fused && (features & RKDA_VF_NEGATIVE), safe = !fused && (features & RKDA_VF_SAFE);
    // Keep the captured descriptor invariant for a whole task. Rewriting a
    // spilled capture every token otherwise makes the compiler insert V->S.
    uint32_t cur = (config & 3) + base % 3;
    if (cur >= 3) cur -= 3;
    const uint32_t next = cur == 2 ? 0 : cur + 1;
    auto query = capturedQuery;
    auto key = capturedKey;
    auto value = capturedValue;
    auto gate = capturedGate;
    auto beta = capturedBeta;
    auto output = capturedOutput;
    auto params = capturedParams;
    auto events = capturedEvents;
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> q0, q1, k0, k1, alpha0, alpha1, strength, a;
    RegTensor<float> temporary;
    RegTensor<uint32_t> outputRotation, columns, pairColumns, lowHalfMask;
    Arange((RegTensor<int32_t> &)outputRotation, 4);
    uint32_t firstCount = 4;
    MaskReg firstFour = UpdateMask<float>(firstCount);
    if constexpr (!vFirst) {
        Arange((RegTensor<int32_t> &)columns, 0);
        Muls(columns, columns, StateLayout::PITCH, mask);
        if constexpr (!std::is_same<stateType, float>::value) {
            ShiftRights(pairColumns, columns, int16_t(1), mask);
            Duplicate(lowHalfMask, uint32_t(0xFFFF));
        }
    }
    // Per-head parameters are loaded for this token and kept in registers.
    if (rawGate) {
        RegBase::Broadcast(a, params, mask);
        Exp(a, a, mask);
    }
    const uint32_t inputToken = base % RKDA_A5_WINDOW_TOKENS;
    auto source = (__ubuf__ stateType *)(memory + cur * BANK);
    auto destination = (__ubuf__ stateType *)(memory + next * BANK);
    RegBase::Load(q0, query + inputToken * RKDA_DIM_128, mask);
    RegBase::Load(q1, query + inputToken * RKDA_DIM_128 + 64, mask);
    RegBase::Load(k0, key + inputToken * RKDA_DIM_128, mask);
    RegBase::Load(k1, key + inputToken * RKDA_DIM_128 + 64, mask);
    if (norm) { RegBase::Normalize(q0, q1, mask); RegBase::Normalize(k0, k1, mask); }
    Muls(q0, q0, scale, mask);
    Muls(q1, q1, scale, mask);
    RegBase::Broadcast(strength, beta + inputToken * (32 / sizeof(Beta)), mask);
    if (sigmoid) {
        RegBase::Sigmoid(strength, mask);
        if (negative) Muls(strength, strength, 2.0f, mask);
    }
    RegTensor<float> updateKey0, updateKey1;
    if constexpr (vFirst || !std::is_same<stateType, float>::value) {
        Mul(updateKey0, k0, strength, mask);
        Mul(updateKey1, k1, strength, mask);
    }
    RegBase::Load(alpha0, gate + inputToken * RKDA_DIM_128, mask);
    RegBase::Load(alpha1, gate + inputToken * RKDA_DIM_128 + 64, mask);
    if (rawGate && bias) {
        DataCopy(temporary, params + 8); Add(alpha0, alpha0, temporary, mask);
        DataCopy(temporary, params + 8 + 64); Add(alpha1, alpha1, temporary, mask);
    }
    RegBase::Decay(alpha0, a, rawGate, safe, lower, mask);
    RegBase::Decay(alpha1, a, rawGate, safe, lower, mask);
    // Batch the exact BF16->FP32 conversion once per token. The
    // scratch lies in the reserved tail, beyond both event words;
    // only this VF uses it, after all rows of the prior token retire.
    RegTensor<float> valueLo, valueHi, outLo, outHi;
    RegBase::Load(valueLo, value + inputToken * RKDA_DIM_128, mask);
    RegBase::Load(valueHi, value + inputToken * RKDA_DIM_128 + RegBase::LANES, mask);
    if constexpr (!vFirst) {
        RegBase::UpdateKFirstColumns(destination, source, valueLo, valueHi,
            outLo, outHi, q0, q1, k0, k1, alpha0, alpha1, strength,
            (__ubuf__ float *)(events + 64), mask);
    } else if constexpr (twoPass) {
        RegTensor<float> predictionLo, predictionHi, predictionKey0, predictionKey1;
        Mul(predictionKey0, alpha0, k0, mask);
        Mul(predictionKey1, alpha1, k1, mask);
        // Collect 128 reductions without four-row Interleave/Gather appends.
        // Prediction -> residual -> output reuse the same 512-byte region.
        // Output overwrites only residual elements already read by this pair.
        auto valueFloat = (__ubuf__ float *)(events + 64);
        #pragma unroll 1
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> sum0, sum1;
            RegBase::PredictIndependentPair(sum0, sum1, source + row * RegBase::DIM,
                predictionKey0, predictionKey1, mask);
            RegBase::Store<float, true>(valueFloat + row, sum0, mask);
            RegBase::Store<float, true>(valueFloat + row + 1, sum1, mask);
        }
        #pragma unroll 1
        for (uint16_t row = 64; row < 128; row += 2) {
            RegTensor<float> sum0, sum1;
            RegBase::PredictIndependentPair(sum0, sum1, source + row * RegBase::DIM,
                predictionKey0, predictionKey1, mask);
            RegBase::Store<float, true>(valueFloat + row, sum0, mask);
            RegBase::Store<float, true>(valueFloat + row + 1, sum1, mask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(predictionLo, valueFloat);
        DataCopy(predictionHi, valueFloat + RegBase::LANES);
        Sub(valueLo, valueLo, predictionLo, mask);
        Sub(valueHi, valueHi, predictionHi, mask);
        DataCopy(valueFloat, valueLo, mask);
        DataCopy(valueFloat + RegBase::LANES, valueHi, mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        #pragma unroll 1
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> sum0, sum1;
            RegBase::UpdateIndependentPair(destination, source, valueFloat, sum0, sum1, row,
                q0, q1, alpha0, alpha1, updateKey0, updateKey1, mask);
            RegBase::Store<float, true>(valueFloat + row, sum0, mask);
            RegBase::Store<float, true>(valueFloat + row + 1, sum1, mask);
        }
        #pragma unroll 1
        for (uint16_t row = 64; row < 128; row += 2) {
            RegTensor<float> sum0, sum1;
            RegBase::UpdateIndependentPair(destination, source, valueFloat, sum0, sum1, row,
                q0, q1, alpha0, alpha1, updateKey0, updateKey1, mask);
            RegBase::Store<float, true>(valueFloat + row, sum0, mask);
            RegBase::Store<float, true>(valueFloat + row + 1, sum1, mask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy(outLo, valueFloat);
        DataCopy(outHi, valueFloat + RegBase::LANES);
    } else {
        auto valueFloat = (__ubuf__ float *)(events + 64);
        DataCopy(valueFloat, valueLo, mask);
        DataCopy(valueFloat + RegBase::LANES, valueHi, mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        Duplicate(outLo, 0.0f);
        #pragma unroll 1
        for (uint16_t row = 0; row < RegBase::LANES; row += 4) {
            RegBase::UpdateRows4<stateType, float, vFirst>(
                destination, source, valueFloat, outLo, outputRotation, firstFour, row,
                q0, q1, k0, k1, alpha0, alpha1, strength, updateKey0, updateKey1,
                columns, pairColumns, lowHalfMask, mask);
        }
        Duplicate(outHi, 0.0f);
        #pragma unroll 1
        for (uint16_t row = RegBase::LANES; row < RKDA_DIM_128; row += 4) {
            RegBase::UpdateRows4<stateType, float, vFirst>(
                destination, source, valueFloat, outHi, outputRotation, firstFour, row,
                q0, q1, k0, k1, alpha0, alpha1, strength, updateKey0, updateKey1,
                columns, pairColumns, lowHalfMask, mask);
        }
    }
    RegBase::Store(output + inputToken * RKDA_DIM_128, outLo, mask);
    RegBase::Store(output + inputToken * RKDA_DIM_128 + RegBase::LANES, outHi, mask);

}

template <class Gate, class Beta, bool vFirst, bool fused,
              bool twoPass,
          typename inType, typename outType, typename stateType, bool stateVFirst, bool fusedMode>
__aicore__ inline void Stage1LaunchRecurrentKdaVF(RKDAVectorContext<inType, outType, stateType, stateVFirst, fusedMode> &ctx, uint32_t base, uint32_t slot,
                                uint32_t parameter)
{
    auto memory = (__ubuf__ uint8_t *)ctx.ub_.GetPhyAddr();
    auto capturedQuery = (__ubuf__ inType *)(memory + ctx.qOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedKey = (__ubuf__ inType *)(memory + ctx.kOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedValue = (__ubuf__ inType *)(memory + ctx.vOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedGate = (__ubuf__ Gate *)(memory + ctx.gateOffset_ + slot * RKDA_ROW_FP32_BYTES_512 * ctx.x_);
    auto capturedBeta = (__ubuf__ Beta *)(memory + ctx.betaOffset_ + slot * ctx.betaSlot_);
    auto capturedOutput = (__ubuf__ outType *)(memory + ctx.outOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedParams = (__ubuf__ float *)(memory + parameter);
    auto capturedEvents = (__ubuf__ volatile uint32_t *)(memory + ctx.eventOffset_);
    // One launch computes exactly one token. Input/output pointers already
    // select the tile slot; only the bank pair and semantic flags are packed.
    const uint32_t launch = ctx.initialBank_ | (ctx.vfFlags_ << 8);
    const float scale = ctx.td_->scale, lower = ctx.td_->lowerBound;
    RecurrentKdaTokenVF<Gate, Beta, vFirst, fused, twoPass,
        inType, outType, stateType, stateVFirst>(memory, capturedQuery, capturedKey, capturedValue,
        capturedGate, capturedBeta, capturedOutput, capturedParams, capturedEvents, base, launch, scale, lower);

}
} // namespace RecurrentKda
