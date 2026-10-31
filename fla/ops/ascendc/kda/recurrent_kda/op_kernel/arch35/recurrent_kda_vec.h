// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "recurrent_kda_struct.h"
namespace RecurrentKda {
using namespace AscendC;
using namespace AscendC::MicroAPI;
// 输入：已就绪UB输入、参数及State bank；输出：更新State和完整tile out。
// 复用：原VF寄存器、512B scratch；同步：保留VST→VLD与token完成发布屏障。
template <class Gate, class Beta, bool vFirst, bool fused = false, uint32_t fixedWindow = 0,
              bool twoPass = true,
          typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage1LaunchRecurrentKdaVF(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint32_t base, uint32_t valid, uint32_t slot,
                                uint32_t parameter, bool last);

template <class Gate, class Beta, bool vFirst, bool fused, uint32_t fixedWindow,
              bool twoPass,
          typename inType, typename outType, typename stateType, bool stateVFirst>
__aicore__ inline void Stage1LaunchRecurrentKdaVF(RKDAVectorContext<inType, outType, stateType, stateVFirst> &ctx, uint32_t base, uint32_t valid, uint32_t slot,
                                uint32_t parameter, bool last)
{
    using StateLayout = RegBase::StateLayout<stateType, stateVFirst>;
    constexpr uint32_t BANK = StateLayout::BYTES;
    auto memory = (__ubuf__ uint8_t *)ctx.ub_.GetPhyAddr();
    auto capturedQuery = (__ubuf__ inType *)(memory + ctx.qOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedKey = (__ubuf__ inType *)(memory + ctx.kOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedValue = (__ubuf__ inType *)(memory + ctx.vOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedGate = (__ubuf__ Gate *)(memory + ctx.gateOffset_ + slot * RKDA_ROW_FP32_BYTES_512 * ctx.x_);
    auto capturedBeta = (__ubuf__ Beta *)(memory + ctx.betaOffset_ + slot * ctx.betaSlot_);
    auto capturedOutput = (__ubuf__ outType *)(memory + ctx.outOffset_ + slot * RKDA_ROW_BF16_BYTES_256 * ctx.x_);
    auto capturedParams = (__ubuf__ float *)(memory + parameter);
    auto capturedEvents = (__ubuf__ volatile uint32_t *)(memory + ctx.eventOffset_);
    const uint32_t bankSlot = ctx.initialBank_ | (ctx.alternateBank_ << 2) | (slot << 4);
    // The three-bit field encodes 8 as zero for the fixed X=8 path.
    // General paths keep the runtime valid count unchanged.
    const uint32_t launch = bankSlot | (valid << 5) | (parameter << 8);
    const bool norm = fused || ctx.td_->useQkL2norm, rawGate = fused || ctx.td_->hasALog;
    const bool bias = fused || ctx.td_->hasDtBias;
    const bool sigmoid = fused || ctx.td_->useBetaSigmoid;
    const bool negative = fused ? false : ctx.td_->allowNegEigval, safe = fused ? false : ctx.td_->safeGate;
    const bool write = ctx.td_->inplaceFinalState || ctx.td_->outputFinalState, indexed = fused ? false : ctx.td_->hasSsmStateIndices;
    const float scale = ctx.td_->scale, lower = ctx.td_->lowerBound;
    const uint16_t tokenCount = static_cast<uint16_t>(valid);
    __VEC_SCOPE__
    {
        const uint32_t config = fixedWindow ? launch : bankSlot;
        const uint32_t initialBank = config & 3, alternateBank = (config >> 2) & 3;
        const uint32_t inputSlot = (config >> 4) & 1;
        const uint16_t encodedCount = static_cast<uint16_t>((config >> 5) & 7);
        const uint16_t count = fixedWindow ?
            static_cast<uint16_t>(encodedCount == 0 ? fixedWindow : encodedCount) : tokenCount;
        // Fixed-window specialization leaves general strides/dtypes on the original path.
        constexpr uint32_t betaStride = fixedWindow < 4 ? 128 : 32 * fixedWindow;
        const uint32_t rowSlot = inputSlot * RKDA_ROW_BF16_BYTES_256 * fixedWindow;
        constexpr uint32_t betaOffset = 3 * BANK + 2560 * fixedWindow;
        constexpr uint32_t outputOffset = betaOffset + 2 * betaStride + RKDA_PARAM_DOUBLE_BYTES_1088;
        auto query = fixedWindow ? (__ubuf__ inType *)(memory + 3 * BANK + rowSlot) : capturedQuery;
        auto key = fixedWindow ? (__ubuf__ inType *)(memory + 3 * BANK + RKDA_ROW_FP32_BYTES_512 * fixedWindow + rowSlot) : capturedKey;
        auto value = fixedWindow ? (__ubuf__ inType *)(memory + 3 * BANK + RKDA_GATE_DOUBLE_BYTES_1024 * fixedWindow + rowSlot) : capturedValue;
        auto gate = fixedWindow ? (__ubuf__ Gate *)(memory + 3 * BANK + 1536 * fixedWindow + 2 * rowSlot) : capturedGate;
        auto beta = fixedWindow ? (__ubuf__ Beta *)(memory + betaOffset + inputSlot * betaStride) : capturedBeta;
        auto output = fixedWindow ? (__ubuf__ outType *)(memory + outputOffset + rowSlot) : capturedOutput;
        // parameter slots are RKDA_PARAM_SLOT_BYTES_544 bytes apart and the second slot is not
        // RKDA_ROW_BF16_BYTES_256-byte aligned; keep the captured pointer instead of packing
        // its low address bits into the launch word.
        auto params = capturedParams;
        auto events = fixedWindow ? (__ubuf__ volatile uint32_t *)(memory + outputOffset + RKDA_ROW_FP32_BYTES_512 * fixedWindow) : capturedEvents;
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
        // Per-head parameters are loaded once per tile and kept in registers.
        if (rawGate) {
            RegBase::Broadcast(a, params, mask);
            Exp(a, a, mask);
        }
        #pragma unroll 1
        for (uint16_t t = 0; t < count; ++t) {
            const uint32_t token = base + t;
            const uint32_t cur = (token & 1) ? alternateBank : initialBank;
            const uint32_t next = (token & 1) ? initialBank : alternateBank;
            auto source = (__ubuf__ stateType *)(memory + cur * BANK);
            auto destination = (__ubuf__ stateType *)(memory + next * BANK);
            // Only a bank still owned by an earlier snapshot needs a wait.
            if (write && indexed) {
                if (token >= 2) while (events[8] < token - 1) {}
            }
            LocalMemBar<MemType::SCALAR_STORE, MemType::VEC_LOAD>();
            RegBase::Load(q0, query + t * RKDA_DIM_128, mask);
            RegBase::Load(q1, query + t * RKDA_DIM_128 + 64, mask);
            RegBase::Load(k0, key + t * RKDA_DIM_128, mask);
            RegBase::Load(k1, key + t * RKDA_DIM_128 + 64, mask);
            if (norm) { RegBase::Normalize(q0, q1, mask); RegBase::Normalize(k0, k1, mask); }
            Muls(q0, q0, scale, mask);
            Muls(q1, q1, scale, mask);
            RegBase::Broadcast(strength, beta + t * (32 / sizeof(Beta)), mask);
            if (sigmoid) {
                RegBase::Sigmoid(strength, mask);
                if (negative) Muls(strength, strength, 2.0f, mask);
            }
            RegTensor<float> updateKey0, updateKey1;
            if constexpr (vFirst || !std::is_same<stateType, float>::value) {
                Mul(updateKey0, k0, strength, mask);
                Mul(updateKey1, k1, strength, mask);
            }
            RegBase::Load(alpha0, gate + t * RKDA_DIM_128, mask);
            RegBase::Load(alpha1, gate + t * RKDA_DIM_128 + 64, mask);
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
            RegBase::Load(valueLo, value + t * RKDA_DIM_128, mask);
            RegBase::Load(valueHi, value + t * RKDA_DIM_128 + RegBase::LANES, mask);
            if constexpr (!vFirst) {
                RegBase::UpdateKFirstColumns(destination, source, valueLo, valueHi,
                    outLo, outHi, q0, q1, k0, k1, alpha0, alpha1, strength,
                    (__ubuf__ float *)(events + 64), mask);
            } else if constexpr (twoPass) {
                RegTensor<float> predictionLo, predictionHi, predictionKey0, predictionKey1;
                Mul(predictionKey0, alpha0, k0, mask);
                Mul(predictionKey1, alpha1, k1, mask);
                Duplicate(predictionLo, 0.0f);
                #pragma unroll 1
                for (uint16_t row = 0; row < RegBase::LANES; row += 4) {
                    RegBase::PredictVFirstRows4(source, predictionLo, outputRotation, firstFour,
                        row, predictionKey0, predictionKey1, mask);
                }
                Duplicate(predictionHi, 0.0f);
                #pragma unroll 1
                for (uint16_t row = RegBase::LANES; row < RKDA_DIM_128; row += 4) {
                    RegBase::PredictVFirstRows4(source, predictionHi, outputRotation, firstFour,
                        row, predictionKey0, predictionKey1, mask);
                }
                Sub(valueLo, valueLo, predictionLo, mask);
                Sub(valueHi, valueHi, predictionHi, mask);
                // Reuse the existing RKDA_ROW_FP32_BYTES_512-byte scratch for residuals, not State.
                auto valueFloat = (__ubuf__ float *)(events + 64);
                DataCopy(valueFloat, valueLo, mask);
                DataCopy(valueFloat + RegBase::LANES, valueHi, mask);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // Both layouts collect a full token in two FP32 output registers.
                Duplicate(outLo, 0.0f);
                #pragma unroll 1
                for (uint16_t row = 0; row < RegBase::LANES; row += 4) {
                    RegBase::UpdateVFirstRows4Second(
                        destination, source, valueFloat, outLo, outputRotation, firstFour, row,
                        q0, q1, alpha0, alpha1, updateKey0, updateKey1, mask);
                }
                Duplicate(outHi, 0.0f);
                #pragma unroll 1
                for (uint16_t row = RegBase::LANES; row < RKDA_DIM_128; row += 4) {
                    RegBase::UpdateVFirstRows4Second(
                        destination, source, valueFloat, outHi, outputRotation, firstFour, row,
                        q0, q1, alpha0, alpha1, updateKey0, updateKey1, mask);
                }
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
            RegBase::Store(output + t * RKDA_DIM_128, outLo, mask);
            RegBase::Store(output + t * RKDA_DIM_128 + RegBase::LANES, outHi, mask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LocalMemBar<MemType::VEC_STORE, MemType::SCALAR_STORE>();
            events[0] = token + 1;
        }
    }

}
} // namespace RecurrentKda
