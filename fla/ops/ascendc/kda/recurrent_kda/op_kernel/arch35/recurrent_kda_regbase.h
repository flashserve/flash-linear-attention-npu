// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef RECURRENT_KDA_REGBASE_H
#define RECURRENT_KDA_REGBASE_H

#include "kernel_operator.h"

namespace RecurrentKda {
namespace RegBase {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t DIM = 128;
constexpr uint32_t LANES = 64;
constexpr uint32_t STATE_BYTES = DIM * DIM * sizeof(float);

// Only K-first rows are padded. DMA row gaps must be multiples of 32 bytes.
// A 32-byte gap rotates Gather/Scatter lanes across UB banks instead of
// repeating the same bank/subbank at the old power-of-two row stride.
template <class State, bool vFirst>
struct StateLayout {
    static constexpr uint32_t PITCH = vFirst ? DIM : DIM + 32 / sizeof(State);
    static constexpr uint32_t BYTES = !vFirst && std::is_same<State, float>::value ?
        DIM * PITCH * sizeof(State) : STATE_BYTES;
    static constexpr uint32_t ROW_GAP = (PITCH - DIM) * sizeof(State) / 32;
};

// Each register contains 64 FP32 columns. Low precision inputs stay raw in UB.
template <class T>
__simd_callee__ inline void Load(RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy(dst, src);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, T, cast>(dst, raw, mask);
    }
}

template <class T>
__simd_callee__ inline void Broadcast(RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, src);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_BRC_B16>(raw, src);
        uint32_t one = 1;
        MaskReg first = UpdateMask<float>(one);
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, T, cast>(dst, raw, first);
        Duplicate(dst, dst, mask);
    }
}

template <class T, bool firstOnly = false>
__simd_callee__ inline void Store(__ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        if constexpr (firstOnly) DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst, src, mask);
        else DataCopy(dst, src, mask);
    } else {
        RegTensor<T> raw;
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::NO_SAT,
                                    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        if constexpr (firstOnly) {
            uint32_t one = 1;
            MaskReg first = UpdateMask<float>(one);
            Cast<T, float, cast>(raw, src, first);
            DataCopy<T, StoreDist::DIST_FIRST_ELEMENT_B16>(dst, raw, mask);
        } else {
            Cast<T, float, cast>(raw, src, mask);
            DataCopy<T, StoreDist::DIST_PACK_B32>(dst, raw, mask);
        }
    }
}

// Gather offsets are in elements; K-first physical storage is preserved in all banks.
template <class T>
__simd_callee__ inline void GatherRow(RegTensor<float> &dst, __ubuf__ T *src,
                                     RegTensor<uint32_t> &columns, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        Gather(dst, src, columns, mask);
    } else {
        RegTensor<uint16_t> offsets;
        RegTensor<T> packed;
        RegTensor<uint32_t> unpacked;
        Pack(offsets, columns);
        uint32_t count = LANES;
        MaskReg halfMask = UpdateMask<T>(count);
        Gather(packed, src, offsets, halfMask);
        UnPack(unpacked, (RegTensor<uint16_t> &)packed);
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, T, cast>(dst, (RegTensor<T> &)unpacked, mask);
    }
}

// K-first BF16 rows share 32-bit words. Access adjacent V rows as pairs,
// retaining BF16 bits in registers while each row's FP32 math stays sequential.
__simd_callee__ inline void LoadPairRow(RegTensor<float> &dst, RegTensor<uint32_t> &packed,
                                      __ubuf__ bfloat16_t *src, RegTensor<uint32_t> &columns,
                                      bool odd, MaskReg &mask)
{
    static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                      MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    if (odd) {
        RegTensor<uint32_t> high;
        ShiftRights(high, packed, int16_t(16), mask);
        Cast<float, bfloat16_t, cast>(dst, (RegTensor<bfloat16_t> &)high, mask);
    } else {
        Gather(packed, (__ubuf__ uint32_t *)src, columns, mask);
        Cast<float, bfloat16_t, cast>(dst, (RegTensor<bfloat16_t> &)packed, mask);
    }
}

__simd_callee__ inline void StorePairRow(__ubuf__ bfloat16_t *dst, RegTensor<uint32_t> &low,
                                       RegTensor<float> &src, RegTensor<uint32_t> &columns,
                                       RegTensor<uint32_t> &lowHalfMask, bool odd, MaskReg &mask)
{
    static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::NO_SAT,
                                      MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if (odd) {
        RegTensor<bfloat16_t> raw;
        RegTensor<uint32_t> high, pair;
        Cast<bfloat16_t, float, cast>(raw, src, mask);
        ShiftLefts(high, (RegTensor<uint32_t> &)raw, int16_t(16), mask);
        Or(pair, low, high, mask);
        Scatter((__ubuf__ uint32_t *)dst, pair, columns, mask);
    } else {
        Cast<bfloat16_t, float, cast>((RegTensor<bfloat16_t> &)low, src, mask);
        And(low, low, lowHalfMask, mask);
    }
}

__simd_callee__ inline void Sum128(RegTensor<float> &sum, RegTensor<float> &a,
                                  RegTensor<float> &b, MaskReg &mask)
{
    Add(sum, a, b, mask);
    ReduceSum(sum, sum, mask);
    Duplicate(sum, sum, mask);
}

template <bool broadcast>
__simd_callee__ inline void FusedSum128(
    RegTensor<float> &sum, RegTensor<float> &lowProduct,
    RegTensor<float> &highState, RegTensor<float> &highCoefficient, MaskReg &mask)
{
    MulAddDst(lowProduct, highState, highCoefficient, mask);
    ReduceSum(sum, lowProduct, mask);
    if constexpr (broadcast) Duplicate(sum, sum, mask);
}

// Pack four valid reduction lanes, append the block, then rotate by four lanes.
// Sixteen appends produce one naturally ordered 64-element output half.
__simd_callee__ inline void CollectOutputBlock(
    RegTensor<float> &output, RegTensor<float> &sum0, RegTensor<float> &sum1,
    RegTensor<float> &sum2, RegTensor<float> &sum3,
    RegTensor<uint32_t> &rotation, MaskReg &firstFour)
{
    Interleave(sum0, sum2, sum0, sum2);
    Interleave(sum1, sum3, sum1, sum3);
    Interleave(sum0, sum1, sum0, sum1);
    Move(output, sum0, firstFour);
    Gather(output, output, rotation);
}

// Interleave each contiguous State fragment's load/conversion and prediction.
template <class State>
__simd_callee__ inline void LoadDecayPrediction(
    RegTensor<float> &state, RegTensor<float> &product, __ubuf__ State *source,
    RegTensor<float> &alpha, RegTensor<float> &key, MaskReg &mask)
{
    Load(state, source, mask);
    Mul(state, state, alpha, mask);
    Mul(product, state, key, mask);
}

__simd_callee__ inline void LoadPairDecayPrediction(
    RegTensor<float> &state, RegTensor<float> &product, RegTensor<uint32_t> &packed,
    __ubuf__ bfloat16_t *source, RegTensor<uint32_t> &columns, bool odd,
    RegTensor<float> &alpha, RegTensor<float> &key, MaskReg &mask)
{
    LoadPairRow(state, packed, source, columns, odd, mask);
    Mul(state, state, alpha, mask);
    Mul(product, state, key, mask);
}

// K-first State keeps the contiguous V dimension in each register. The first
// pass produces all predictions; the second updates State and accumulates out.
// Both passes read the original bank. Only the second writes the alternate bank.
template <class State>
__simd_callee__ inline void PredictKFirstHalf(
    RegTensor<float> &predictionLo, RegTensor<float> &predictionHi,
    __ubuf__ State *source, __ubuf__ float *predictionKey, MaskReg &mask)
{
    constexpr uint32_t PITCH = StateLayout<State, false>::PITCH;
    #pragma unroll 1
    for (uint16_t k = 0; k < LANES; ++k) {
        RegTensor<float> coefficient, lo, hi;
        Broadcast(coefficient, predictionKey + k, mask);
        Load(lo, source + k * PITCH, mask);
        Load(hi, source + k * PITCH + LANES, mask);
        MulAddDst(predictionLo, lo, coefficient, mask);
        MulAddDst(predictionHi, hi, coefficient, mask);
    }
}

template <class State>
__simd_callee__ inline void UpdateKFirstHalf(
    __ubuf__ State *destination, __ubuf__ State *source,
    RegTensor<float> &deltaLo, RegTensor<float> &deltaHi,
    RegTensor<float> &outputLo, RegTensor<float> &outputHi,
    RegTensor<float> &query, RegTensor<float> &key, RegTensor<float> &decay,
    MaskReg &mask)
{
    constexpr uint32_t PITCH = StateLayout<State, false>::PITCH;
    RegTensor<uint32_t> lane;
    Duplicate(lane, uint32_t(0));
    #pragma unroll 1
    for (uint16_t k = 0; k < LANES; ++k) {
        RegTensor<float> q, keyValue, alpha, lo, hi;
        Gather(alpha, decay, lane);
        Gather(keyValue, key, lane);
        Gather(q, query, lane);
        Load(lo, source + k * PITCH, mask);
        Load(hi, source + k * PITCH + LANES, mask);
        Mul(lo, lo, alpha, mask);
        Mul(hi, hi, alpha, mask);
        MulAddDst(lo, deltaLo, keyValue, mask);
        MulAddDst(hi, deltaHi, keyValue, mask);
        MulAddDst(outputLo, lo, q, mask);
        MulAddDst(outputHi, hi, q, mask);
        // Accumulate out from FP32 State before rounding for the next token.
        Store(destination + k * PITCH, lo, mask);
        Store(destination + k * PITCH + LANES, hi, mask);
        Adds(lane, lane, uint32_t(1), mask);
    }
}

template <class State>
__simd_callee__ inline void UpdateKFirstColumns(
    __ubuf__ State *destination, __ubuf__ State *source,
    RegTensor<float> &valueLo, RegTensor<float> &valueHi,
    RegTensor<float> &outputLo, RegTensor<float> &outputHi,
    RegTensor<float> &q0, RegTensor<float> &q1, RegTensor<float> &k0, RegTensor<float> &k1,
    RegTensor<float> &alpha0, RegTensor<float> &alpha1, RegTensor<float> &strength,
    __ubuf__ float *coefficientScratch, MaskReg &mask)
{
    constexpr uint32_t OFFSET = LANES * StateLayout<State, false>::PITCH;
    RegTensor<float> predictionLo, predictionHi, predictionKey0, predictionKey1;
    Duplicate(predictionLo, 0.0f);
    Duplicate(predictionHi, 0.0f);
    Mul(predictionKey0, alpha0, k0, mask);
    Mul(predictionKey1, alpha1, k1, mask);
    // Prediction coefficients live until both K halves have consumed them.
    // The reserved 512-byte tail is disjoint from event words and DMA buffers.
    DataCopy(coefficientScratch, predictionKey0, mask);
    DataCopy(coefficientScratch + LANES, predictionKey1, mask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    PredictKFirstHalf(predictionLo, predictionHi, source, coefficientScratch, mask);
    PredictKFirstHalf(predictionLo, predictionHi, source + OFFSET, coefficientScratch + LANES, mask);
    Sub(predictionLo, valueLo, predictionLo, mask);
    Sub(predictionHi, valueHi, predictionHi, mask);
    Mul(predictionLo, predictionLo, strength, mask);
    Mul(predictionHi, predictionHi, strength, mask);
    Duplicate(outputLo, 0.0f);
    Duplicate(outputHi, 0.0f);
    UpdateKFirstHalf(destination, source, predictionLo, predictionHi,
                          outputLo, outputHi, q0, k0, alpha0, mask);
    UpdateKFirstHalf(destination + OFFSET, source + OFFSET, predictionLo, predictionHi,
                          outputLo, outputHi, q1, k1, alpha1, mask);
}

// Two independent loads precede conversion; keep only a rolling pair live.
template <class State>
__simd_callee__ inline void LoadIndependentPair(
    RegTensor<float> &a, RegTensor<float> &b,
    __ubuf__ State *sourceA, __ubuf__ State *sourceB, MaskReg &mask)
{
    if constexpr (std::is_same<State, float>::value) {
        DataCopy(a, sourceA);
        DataCopy(b, sourceB);
    } else {
        RegTensor<State> rawA, rawB;
        DataCopy<State, LoadDist::DIST_UNPACK_B16>(rawA, sourceA);
        DataCopy<State, LoadDist::DIST_UNPACK_B16>(rawB, sourceB);
        static constexpr CastTrait trait = {RegLayout::ZERO, SatMode::UNKNOWN,
            MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, State, trait>(a, rawA, mask);
        Cast<float, State, trait>(b, rawB, mask);
    }
}

// First V-first pass consumes each State row without retaining a State shadow.
template <class State>
__simd_callee__ inline void PredictVFirstRow(
    RegTensor<float> &sum, __ubuf__ State *source,
    RegTensor<float> &predictionKey0, RegTensor<float> &predictionKey1, MaskReg &mask)
{
    RegTensor<float> low, high;
    Load(low, source, mask);
    Mul(low, low, predictionKey0, mask);
    Load(high, source + LANES, mask);
    FusedSum128<false>(sum, low, high, predictionKey1, mask);
}

template <class State>
__simd_callee__ inline void PredictIndependentPair(
    RegTensor<float> &sumA, RegTensor<float> &sumB,
    __ubuf__ State *source, RegTensor<float> &key0,
    RegTensor<float> &key1, MaskReg &mask)
{
    RegTensor<float> lowA, lowB, highA, highB;
    LoadIndependentPair(lowA, lowB, source, source + DIM, mask);
    Mul(lowA, lowA, key0, mask);
    Mul(lowB, lowB, key0, mask);
    LoadIndependentPair(highA, highB, source + LANES, source + DIM + LANES, mask);
    MulAddDst(lowA, highA, key1, mask);
    MulAddDst(lowB, highB, key1, mask);
    ReduceSum(sumA, lowA, mask);
    ReduceSum(sumB, lowB, mask);
}

template <class State>
__simd_callee__ inline void PredictVFirstRows4(
    __ubuf__ State *source, RegTensor<float> &prediction,
    RegTensor<uint32_t> &rotation, MaskReg &firstFour, uint16_t row,
    RegTensor<float> &predictionKey0, RegTensor<float> &predictionKey1, MaskReg &mask)
{
    RegTensor<float> sum0, sum1, sum2, sum3;
    if constexpr (!std::is_same<State, float>::value) {
    PredictIndependentPair(sum0, sum1, source + row * DIM, predictionKey0, predictionKey1, mask);
    PredictIndependentPair(sum2, sum3, source + (row + 2) * DIM, predictionKey0, predictionKey1, mask);
    } else {
    PredictVFirstRow(sum0, source + row * DIM, predictionKey0, predictionKey1, mask);
    PredictVFirstRow(sum1, source + (row + 1) * DIM, predictionKey0, predictionKey1, mask);
    PredictVFirstRow(sum2, source + (row + 2) * DIM, predictionKey0, predictionKey1, mask);
    PredictVFirstRow(sum3, source + (row + 3) * DIM, predictionKey0, predictionKey1, mask);
    }
    CollectOutputBlock(prediction, sum0, sum1, sum2, sum3, rotation, firstFour);
}

template <class State>
__simd_callee__ inline void UpdateIndependentPair(
    __ubuf__ State *destination, __ubuf__ State *source, __ubuf__ float *residual,
    RegTensor<float> &sumA, RegTensor<float> &sumB, uint16_t row,
    RegTensor<float> &q0, RegTensor<float> &q1,
    RegTensor<float> &alpha0, RegTensor<float> &alpha1,
    RegTensor<float> &key0, RegTensor<float> &key1, MaskReg &mask)
{
    RegTensor<float> lowA, lowB, highA, highB, deltaA, deltaB, productA, productB;
    Broadcast(deltaA, residual + row, mask);
    Broadcast(deltaB, residual + row + 1, mask);
    LoadIndependentPair(lowA, lowB, source + row * DIM, source + (row + 1) * DIM, mask);
    Mul(lowA, lowA, alpha0, mask);
    Mul(lowB, lowB, alpha0, mask);
    MulAddDst(lowA, deltaA, key0, mask);
    MulAddDst(lowB, deltaB, key0, mask);
    LoadIndependentPair(highA, highB, source + row * DIM + LANES, source + (row + 1) * DIM + LANES, mask);
    Mul(productA, lowA, q0, mask);
    Mul(productB, lowB, q0, mask);
    // Low State is no longer needed after its output product; release it early.
    Store(destination + row * DIM, lowA, mask);
    Store(destination + (row + 1) * DIM, lowB, mask);
    Mul(highA, highA, alpha1, mask);
    Mul(highB, highB, alpha1, mask);
    MulAddDst(highA, deltaA, key1, mask);
    MulAddDst(highB, deltaB, key1, mask);
    MulAddDst(productA, highA, q1, mask);
    MulAddDst(productB, highB, q1, mask);
    Store(destination + row * DIM + LANES, highA, mask);
    Store(destination + (row + 1) * DIM + LANES, highB, mask);
    ReduceSum(sumA, productA, mask);
    ReduceSum(sumB, productB, mask);
}

// Second pass rereads the original bank and consumes the complete residual vector.
template <class State>
__simd_callee__ inline void UpdateVFirstRows4Second(
    __ubuf__ State *destination, __ubuf__ State *source, __ubuf__ float *residual,
    RegTensor<float> &output, RegTensor<uint32_t> &rotation, MaskReg &firstFour,
    uint16_t row, RegTensor<float> &q0, RegTensor<float> &q1,
    RegTensor<float> &alpha0, RegTensor<float> &alpha1,
    RegTensor<float> &updateKey0, RegTensor<float> &updateKey1, MaskReg &mask)
{
    if constexpr (std::is_same<State, float>::value) {
    RegTensor<float> sum0, sum1, sum2, sum3;
    UpdateIndependentPair(destination, source, residual, sum0, sum1, row,
        q0, q1, alpha0, alpha1, updateKey0, updateKey1, mask);
    UpdateIndependentPair(destination, source, residual, sum2, sum3, row + 2,
        q0, q1, alpha0, alpha1, updateKey0, updateKey1, mask);
    CollectOutputBlock(output, sum0, sum1, sum2, sum3, rotation, firstFour);
    } else {
    RegTensor<float> s00, s01, s10, s11, s20, s21, s30, s31;
    Load(s00, source + row * DIM, mask); Mul(s00, s00, alpha0, mask);
    Load(s01, source + row * DIM + LANES, mask); Mul(s01, s01, alpha1, mask);
    Load(s10, source + (row + 1) * DIM, mask); Mul(s10, s10, alpha0, mask);
    Load(s11, source + (row + 1) * DIM + LANES, mask); Mul(s11, s11, alpha1, mask);
    Load(s20, source + (row + 2) * DIM, mask); Mul(s20, s20, alpha0, mask);
    Load(s21, source + (row + 2) * DIM + LANES, mask); Mul(s21, s21, alpha1, mask);
    Load(s30, source + (row + 3) * DIM, mask); Mul(s30, s30, alpha0, mask);
    Load(s31, source + (row + 3) * DIM + LANES, mask); Mul(s31, s31, alpha1, mask);
    RegTensor<float> delta0, delta1, delta2, delta3;
    Broadcast(delta0, residual + row, mask); Broadcast(delta1, residual + row + 1, mask);
    Broadcast(delta2, residual + row + 2, mask); Broadcast(delta3, residual + row + 3, mask);
    MulAddDst(s00, delta0, updateKey0, mask); MulAddDst(s10, delta1, updateKey0, mask);
    MulAddDst(s20, delta2, updateKey0, mask); MulAddDst(s30, delta3, updateKey0, mask);
    MulAddDst(s01, delta0, updateKey1, mask); MulAddDst(s11, delta1, updateKey1, mask);
    MulAddDst(s21, delta2, updateKey1, mask); MulAddDst(s31, delta3, updateKey1, mask);
    RegTensor<float> p0, p1, p2, p3, sum0, sum1, sum2, sum3;
    Mul(p0, s00, q0, mask); Mul(p1, s10, q0, mask);
    Mul(p2, s20, q0, mask); Mul(p3, s30, q0, mask);
    FusedSum128<false>(sum0, p0, s01, q1, mask);
    FusedSum128<false>(sum1, p1, s11, q1, mask);
    FusedSum128<false>(sum2, p2, s21, q1, mask);
    FusedSum128<false>(sum3, p3, s31, q1, mask);
    // Output reductions consume unrounded FP32 State before the BF16 stores.
    Store(destination + row * DIM, s00, mask);
    Store(destination + row * DIM + LANES, s01, mask);
    Store(destination + (row + 1) * DIM, s10, mask);
    Store(destination + (row + 1) * DIM + LANES, s11, mask);
    Store(destination + (row + 2) * DIM, s20, mask);
    Store(destination + (row + 2) * DIM + LANES, s21, mask);
    Store(destination + (row + 3) * DIM, s30, mask);
    Store(destination + (row + 3) * DIM + LANES, s31, mask);
    CollectOutputBlock(output, sum0, sum1, sum2, sum3, rotation, firstFour);
    }
}

// Keep four complete mathematical V rows live. State results reach UB per block;
// FP32 outputs stay in a token-wide register accumulator until all rows finish.
template <class State, class Value, bool vFirst>
__simd_callee__ inline void UpdateRows4(
    __ubuf__ State *destination, __ubuf__ State *source,
    __ubuf__ Value *value, RegTensor<float> &output,
    RegTensor<uint32_t> &outputRotation, MaskReg &firstFour, uint16_t row,
    RegTensor<float> &q0, RegTensor<float> &q1,
    RegTensor<float> &k0, RegTensor<float> &k1,
    RegTensor<float> &alpha0, RegTensor<float> &alpha1,
    RegTensor<float> &strength, RegTensor<float> &updateKey0,
    RegTensor<float> &updateKey1, RegTensor<uint32_t> &columns,
    RegTensor<uint32_t> &pairColumns, RegTensor<uint32_t> &lowHalfMask, MaskReg &mask)
{
    constexpr uint32_t PITCH = StateLayout<State, vFirst>::PITCH;
    // K-first FP32 is faster with the original unfused dot products.
    constexpr bool fusedDot = vFirst || !std::is_same<State, float>::value;
    RegTensor<float> s00, s01, s10, s11, s20, s21, s30, s31;
    RegTensor<float> p00, p01, p10, p11, p20, p21, p30, p31;
    if constexpr (vFirst) {
        LoadDecayPrediction(s00, p00, source + row * DIM, alpha0, k0, mask);
        LoadDecayPrediction(s01, p01, source + row * DIM + LANES, alpha1, k1, mask);
        LoadDecayPrediction(s10, p10, source + (row + 1) * DIM, alpha0, k0, mask);
        LoadDecayPrediction(s11, p11, source + (row + 1) * DIM + LANES, alpha1, k1, mask);
        LoadDecayPrediction(s20, p20, source + (row + 2) * DIM, alpha0, k0, mask);
        LoadDecayPrediction(s21, p21, source + (row + 2) * DIM + LANES, alpha1, k1, mask);
        LoadDecayPrediction(s30, p30, source + (row + 3) * DIM, alpha0, k0, mask);
        LoadDecayPrediction(s31, p31, source + (row + 3) * DIM + LANES, alpha1, k1, mask);
    } else if constexpr (std::is_same<State, float>::value) {
        GatherRow(s00, source + row + 0 + 0 * LANES * PITCH, columns, mask);
        GatherRow(s01, source + row + 0 + 1 * LANES * PITCH, columns, mask);
        GatherRow(s10, source + row + 1 + 0 * LANES * PITCH, columns, mask);
        GatherRow(s11, source + row + 1 + 1 * LANES * PITCH, columns, mask);
        GatherRow(s20, source + row + 2 + 0 * LANES * PITCH, columns, mask);
        GatherRow(s21, source + row + 2 + 1 * LANES * PITCH, columns, mask);
        GatherRow(s30, source + row + 3 + 0 * LANES * PITCH, columns, mask);
        GatherRow(s31, source + row + 3 + 1 * LANES * PITCH, columns, mask);
        Mul(s00, s00, alpha0, mask); Mul(s10, s10, alpha0, mask);
        Mul(s20, s20, alpha0, mask); Mul(s30, s30, alpha0, mask);
        Mul(s01, s01, alpha1, mask); Mul(s11, s11, alpha1, mask);
        Mul(s21, s21, alpha1, mask); Mul(s31, s31, alpha1, mask);
        Mul(p00, s00, k0, mask); Mul(p10, s10, k0, mask);
        Mul(p20, s20, k0, mask); Mul(p30, s30, k0, mask);
        Mul(p01, s01, k1, mask); Mul(p11, s11, k1, mask);
        Mul(p21, s21, k1, mask); Mul(p31, s31, k1, mask);
    } else {
        RegTensor<uint32_t> packed0, packed1;
        LoadPairDecayPrediction(s00, p00, packed0, source + row + 0 + 0 * LANES * PITCH,
                               pairColumns, false, alpha0, k0, mask);
        LoadPairDecayPrediction(s01, p01, packed1, source + row + 0 + 1 * LANES * PITCH,
                               pairColumns, false, alpha1, k1, mask);
        LoadPairDecayPrediction(s10, p10, packed0, source + row + 0 + 0 * LANES * PITCH,
                               pairColumns, true, alpha0, k0, mask);
        LoadPairDecayPrediction(s11, p11, packed1, source + row + 0 + 1 * LANES * PITCH,
                               pairColumns, true, alpha1, k1, mask);
        LoadPairDecayPrediction(s20, p20, packed0, source + row + 2 + 0 * LANES * PITCH,
                               pairColumns, false, alpha0, k0, mask);
        LoadPairDecayPrediction(s21, p21, packed1, source + row + 2 + 1 * LANES * PITCH,
                               pairColumns, false, alpha1, k1, mask);
        LoadPairDecayPrediction(s30, p30, packed0, source + row + 2 + 0 * LANES * PITCH,
                               pairColumns, true, alpha0, k0, mask);
        LoadPairDecayPrediction(s31, p31, packed1, source + row + 2 + 1 * LANES * PITCH,
                               pairColumns, true, alpha1, k1, mask);
    }
    RegTensor<float> prediction0, prediction1, prediction2, prediction3;
    RegTensor<float> delta0, delta1, delta2, delta3;
    if constexpr (fusedDot) {
        FusedSum128<true>(prediction0, p00, s01, k1, mask);
        FusedSum128<true>(prediction1, p10, s11, k1, mask);
        FusedSum128<true>(prediction2, p20, s21, k1, mask);
        FusedSum128<true>(prediction3, p30, s31, k1, mask);
    } else {
        Sum128(prediction0, p00, p01, mask);
        Sum128(prediction1, p10, p11, mask);
        Sum128(prediction2, p20, p21, mask);
        Sum128(prediction3, p30, p31, mask);
    }
    Broadcast(delta0, value + row, mask);
    Broadcast(delta1, value + row + 1, mask);
    Broadcast(delta2, value + row + 2, mask);
    Broadcast(delta3, value + row + 3, mask);
    Sub(delta0, delta0, prediction0, mask); Sub(delta1, delta1, prediction1, mask);
    Sub(delta2, delta2, prediction2, mask); Sub(delta3, delta3, prediction3, mask);
    // K-first FP32 retains the separate update: its Gather/Scatter schedule
    // did not benefit from the fused/reassociated arithmetic on A5.
    if constexpr (vFirst || !std::is_same<State, float>::value) {
        MulAddDst(s00, delta0, updateKey0, mask); MulAddDst(s10, delta1, updateKey0, mask);
        MulAddDst(s20, delta2, updateKey0, mask); MulAddDst(s30, delta3, updateKey0, mask);
        MulAddDst(s01, delta0, updateKey1, mask); MulAddDst(s11, delta1, updateKey1, mask);
        MulAddDst(s21, delta2, updateKey1, mask); MulAddDst(s31, delta3, updateKey1, mask);
    } else {
        Mul(delta0, delta0, strength, mask); Mul(delta1, delta1, strength, mask);
        Mul(delta2, delta2, strength, mask); Mul(delta3, delta3, strength, mask);

        // Reuse product registers after prediction instead of allocating a State shadow.
        Mul(p00, delta0, k0, mask); Mul(p10, delta1, k0, mask);
        Mul(p20, delta2, k0, mask); Mul(p30, delta3, k0, mask);
        Add(s00, s00, p00, mask); Add(s10, s10, p10, mask);
        Add(s20, s20, p20, mask); Add(s30, s30, p30, mask);
        Mul(p01, delta0, k1, mask); Mul(p11, delta1, k1, mask);
        Mul(p21, delta2, k1, mask); Mul(p31, delta3, k1, mask);
        Add(s01, s01, p01, mask); Add(s11, s11, p11, mask);
        Add(s21, s21, p21, mask); Add(s31, s31, p31, mask);
    }

    // All four outputs consume FP32 State before any State row is rounded to BF16.
    Mul(p00, s00, q0, mask); Mul(p10, s10, q0, mask);
    Mul(p20, s20, q0, mask); Mul(p30, s30, q0, mask);
    if constexpr (fusedDot) {
        FusedSum128<false>(prediction0, p00, s01, q1, mask);
        FusedSum128<false>(prediction1, p10, s11, q1, mask);
        FusedSum128<false>(prediction2, p20, s21, q1, mask);
        FusedSum128<false>(prediction3, p30, s31, q1, mask);
    } else {
        Mul(p01, s01, q1, mask); Mul(p11, s11, q1, mask);
        Mul(p21, s21, q1, mask); Mul(p31, s31, q1, mask);
        Add(prediction0, p00, p01, mask); Add(prediction1, p10, p11, mask);
        Add(prediction2, p20, p21, mask); Add(prediction3, p30, p31, mask);
        ReduceSum(prediction0, prediction0, mask); ReduceSum(prediction1, prediction1, mask);
        ReduceSum(prediction2, prediction2, mask); ReduceSum(prediction3, prediction3, mask);
    }
    if constexpr (vFirst) {
        Store(destination + row * DIM, s00, mask);
        Store(destination + row * DIM + LANES, s01, mask);
        Store(destination + (row + 1) * DIM, s10, mask);
        Store(destination + (row + 1) * DIM + LANES, s11, mask);
        Store(destination + (row + 2) * DIM, s20, mask);
        Store(destination + (row + 2) * DIM + LANES, s21, mask);
        Store(destination + (row + 3) * DIM, s30, mask);
        Store(destination + (row + 3) * DIM + LANES, s31, mask);
    } else if constexpr (std::is_same<State, float>::value) {
        Scatter(destination + row + 0 + 0 * LANES * PITCH, s00, columns, mask);
        Scatter(destination + row + 0 + 1 * LANES * PITCH, s01, columns, mask);
        Scatter(destination + row + 1 + 0 * LANES * PITCH, s10, columns, mask);
        Scatter(destination + row + 1 + 1 * LANES * PITCH, s11, columns, mask);
        Scatter(destination + row + 2 + 0 * LANES * PITCH, s20, columns, mask);
        Scatter(destination + row + 2 + 1 * LANES * PITCH, s21, columns, mask);
        Scatter(destination + row + 3 + 0 * LANES * PITCH, s30, columns, mask);
        Scatter(destination + row + 3 + 1 * LANES * PITCH, s31, columns, mask);
    } else {
        RegTensor<uint32_t> packed0, packed1;
        StorePairRow(destination + row + 0 + 0 * LANES * PITCH, packed0, s00,
                     pairColumns, lowHalfMask, false, mask);
        StorePairRow(destination + row + 0 + 1 * LANES * PITCH, packed1, s01,
                     pairColumns, lowHalfMask, false, mask);
        StorePairRow(destination + row + 0 + 0 * LANES * PITCH, packed0, s10,
                     pairColumns, lowHalfMask, true, mask);
        StorePairRow(destination + row + 0 + 1 * LANES * PITCH, packed1, s11,
                     pairColumns, lowHalfMask, true, mask);
        StorePairRow(destination + row + 2 + 0 * LANES * PITCH, packed0, s20,
                     pairColumns, lowHalfMask, false, mask);
        StorePairRow(destination + row + 2 + 1 * LANES * PITCH, packed1, s21,
                     pairColumns, lowHalfMask, false, mask);
        StorePairRow(destination + row + 2 + 0 * LANES * PITCH, packed0, s30,
                     pairColumns, lowHalfMask, true, mask);
        StorePairRow(destination + row + 2 + 1 * LANES * PITCH, packed1, s31,
                     pairColumns, lowHalfMask, true, mask);
    }
    // Retire the independent State stores before extending the output chain.
    // The FP32 reductions remain live; State rounding cannot change the output.
    CollectOutputBlock(output, prediction0, prediction1, prediction2, prediction3,
                       outputRotation, firstFour);
}

__simd_callee__ inline void Normalize(RegTensor<float> &lo, RegTensor<float> &hi, MaskReg &mask)
{
    RegTensor<float> a, b, norm;
    Mul(a, lo, lo, mask);
    Mul(b, hi, hi, mask);
    Sum128(norm, a, b, mask);
    Adds(norm, norm, 1.e-6f, mask);
    Sqrt(norm, norm, mask);
    Div(lo, lo, norm, mask);
    Div(hi, hi, norm, mask);
}

__simd_callee__ inline void Sigmoid(RegTensor<float> &dst, MaskReg &mask)
{
    RegTensor<float> magnitude, numerator, denominator, one, negative;
    MaskReg positive;
    Duplicate(one, 1.0f);
    Abs(magnitude, dst, mask);
    Muls(magnitude, magnitude, -1.0f, mask);
    Exp(magnitude, magnitude, mask);
    Adds(denominator, magnitude, 1.0f, mask);
    Div(numerator, one, denominator, mask);
    Div(negative, magnitude, denominator, mask);
    CompareScalar<float, CMPMODE::GE>(positive, dst, 0.0f, mask);
    Select(dst, numerator, negative, positive);
}

__simd_callee__ inline void Decay(RegTensor<float> &gate, RegTensor<float> &a,
                                 bool rawGate, bool safeGate, float lowerBound, MaskReg &mask)
{
    if (rawGate) {
        if (safeGate) {
            Mul(gate, gate, a, mask);
            Sigmoid(gate, mask);
            Muls(gate, gate, lowerBound, mask);
        } else {
            RegTensor<float> tail, positive;
            Abs(tail, gate, mask);
            Muls(tail, tail, -1.0f, mask);
            Exp(tail, tail, mask);
            Adds(tail, tail, 1.0f, mask);
            Ln(tail, tail, mask);
            Maxs(positive, gate, 0.0f, mask);
            Add(gate, positive, tail, mask);
            Mul(gate, gate, a, mask);
            Muls(gate, gate, -1.0f, mask);
        }
    }
    Exp(gate, gate, mask);
}

} // namespace RegBase
} // namespace RecurrentKda
#endif
