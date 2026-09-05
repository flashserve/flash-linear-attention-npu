/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef RECURRENT_GATED_DELTA_RULE_REGBASE_H
#define RECURRENT_GATED_DELTA_RULE_REGBASE_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"

namespace RecurrentGatedDeltaRule {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t RGDR_K128 = 128;
constexpr CastTrait RGDR_F32_TO_B16_ZERO = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, AscendC::RoundMode::CAST_RINT};
constexpr CastTrait RGDR_F32_TO_B16_ONE = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void RgdrLoadPairAsFloat(RegTensor<float> &zero, RegTensor<float> &one,
                                                __ubuf__ T *src, MaskReg &mask16)
{
    if constexpr (std::is_same<T, float>::value) {
        (void)mask16;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
    } else {
        RegTensor<T> raw;
        LoadIn<T, false>(raw, src);
        CastHalf2Float<T>(zero, one, raw, mask16);
    }
}

template <typename T>
__simd_callee__ inline void RgdrLoadScalarAsFloat(RegTensor<float> &dst, __ubuf__ T *src,
                                                  MaskReg &mask16)
{
    if constexpr (std::is_same<T, float>::value) {
        (void)mask16;
        LoadIn<float, true>(dst, src);
    } else {
        RegTensor<T> raw;
        RegTensor<float> unused;
        LoadIn<T, true>(raw, src);
        CastHalf2Float<T>(dst, unused, raw, mask16);
    }
}

template <typename T>
__simd_callee__ inline void RgdrStorePairFromFloat(__ubuf__ T *dst, RegTensor<float> &zero,
                                                   RegTensor<float> &one, MaskReg &mask32, MaskReg &mask16)
{
    if constexpr (std::is_same<T, float>::value) {
        (void)mask16;
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, mask32);
    } else {
        RegTensor<T> raw;
        Cast<T, float, RGDR_F32_TO_B16_ONE>(raw, one, mask32);
        Cast<T, float, RGDR_F32_TO_B16_ZERO>(raw, zero, mask32);
        StoreAlign(dst, raw, mask16);
    }
}

template <typename T>
__simd_callee__ inline void RgdrStoreScalarFromFloat(__ubuf__ T *&dst, RegTensor<float> &src,
                                                     MaskReg &mask32, UnalignRegForStore &unalignStore)
{
    RegTensor<T> raw;
    Cast<T, float, RGDR_F32_TO_B16_ZERO>(raw, src, mask32);
    StoreUnAlign(dst, raw, unalignStore, 1);
}

template <typename StateT, typename OutT, bool HAS_G, bool HAS_GK, bool READ_INITIAL, bool WRITE_RECURRENT>
__simd_callee__ inline void RgdrProcessK128Row(
    __ubuf__ float *recurrentState, __ubuf__ StateT *initialState, __ubuf__ StateT *stateOut,
    __ubuf__ OutT *&attnOut, __ubuf__ bfloat16_t *value, uint16_t row,
    RegTensor<float> &query0, RegTensor<float> &query1, RegTensor<float> &key0, RegTensor<float> &key1,
    RegTensor<float> &gk0, RegTensor<float> &gk1, RegTensor<float> &gamma, RegTensor<float> &beta,
    MaskReg &mask32, MaskReg &mask16, UnalignRegForStore &unalignStore)
{
    const uint32_t stateOffset = static_cast<uint32_t>(row) * RGDR_K128;
    RegTensor<float> state0;
    RegTensor<float> state1;
    if constexpr (READ_INITIAL) {
        RgdrLoadPairAsFloat<StateT>(state0, state1, initialState + stateOffset, mask16);
    } else {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(state0, state1, recurrentState + stateOffset);
    }

    if constexpr (HAS_G) {
        Mul(state0, state0, gamma, mask32);
        Mul(state1, state1, gamma, mask32);
    }
    if constexpr (HAS_GK) {
        Mul(state0, state0, gk0, mask32);
        Mul(state1, state1, gk1, mask32);
    }

    RegTensor<float> work0;
    RegTensor<float> work1;
    RegTensor<float> dotK;
    RegTensor<float> dotKBroadcast;
    Mul(work0, state0, key0, mask32);
    Mul(work1, state1, key1, mask32);
    Add(work0, work0, work1, mask32);
    ReduceSum(dotK, work0, mask32);
    Duplicate(dotKBroadcast, dotK, mask32);

    RegTensor<float> valueScalar;
    RgdrLoadScalarAsFloat<bfloat16_t>(valueScalar, value + row, mask16);
    RegTensor<float> delta;
    Sub(delta, valueScalar, dotKBroadcast, mask32);
    Mul(delta, delta, beta, mask32);

    Mul(work0, delta, key0, mask32);
    Mul(work1, delta, key1, mask32);
    Add(state0, state0, work0, mask32);
    Add(state1, state1, work1, mask32);

    RegTensor<float> attn;
    Mul(work0, state0, query0, mask32);
    Mul(work1, state1, query1, mask32);
    Add(work0, work0, work1, mask32);
    ReduceSum(attn, work0, mask32);

    if constexpr (WRITE_RECURRENT) {
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(recurrentState + stateOffset, state0, state1, mask32);
    }
    RgdrStorePairFromFloat<StateT>(stateOut + stateOffset, state0, state1, mask32, mask16);
    RgdrStoreScalarFromFloat<OutT>(attnOut, attn, mask32, unalignStore);
}

/**
 * K=128 recurrent token hot path.
 *
 * Q/K/GK are loaded once and retained in registers while two V rows are consumed per loop. Gate mode and
 * first/last-token state behavior are compile-time policies, so the VF body has no runtime feature branch.
 */
template <typename StateT, typename OutT, bool HAS_G, bool HAS_GK, bool READ_INITIAL, bool WRITE_RECURRENT>
__simd_vf__ inline void RgdrRecurrentToken128Vf(
    __ubuf__ float *recurrentState, __ubuf__ StateT *initialState, __ubuf__ StateT *stateOut,
    __ubuf__ OutT *attnOut, __ubuf__ bfloat16_t *query, __ubuf__ bfloat16_t *key,
    __ubuf__ bfloat16_t *value, __ubuf__ float *gammaInput, __ubuf__ float *gkInput,
    __ubuf__ bfloat16_t *betaInput, float scale, uint16_t rows)
{
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> query0;
    RegTensor<float> query1;
    RegTensor<float> key0;
    RegTensor<float> key1;
    RgdrLoadPairAsFloat<bfloat16_t>(query0, query1, query, mask16);
    RgdrLoadPairAsFloat<bfloat16_t>(key0, key1, key, mask16);
    Muls(query0, query0, scale, mask32);
    Muls(query1, query1, scale, mask32);

    RegTensor<float> gamma;
    if constexpr (HAS_G) {
        RgdrLoadScalarAsFloat<float>(gamma, gammaInput, mask16);
        Exp(gamma, gamma, mask32);
    }

    RegTensor<float> gk0;
    RegTensor<float> gk1;
    if constexpr (HAS_GK) {
        RgdrLoadPairAsFloat<float>(gk0, gk1, gkInput, mask16);
        Exp(gk0, gk0, mask32);
        Exp(gk1, gk1, mask32);
    }

    RegTensor<float> beta;
    RgdrLoadScalarAsFloat<bfloat16_t>(beta, betaInput, mask16);
    UnalignRegForStore unalignStore;
    __ubuf__ OutT *attnCursor = attnOut;

    // Keep the VF loop in the compiler-preferred uint16/start-0/step-1 form. vStep is 8-element aligned,
    // therefore each logical iteration can consume two rows without a runtime tail branch.
    const uint16_t rowPairs = rows >> 1;
    for (uint16_t pair = 0; pair < rowPairs; ++pair) {
        const uint16_t row = pair << 1;
        RgdrProcessK128Row<StateT, OutT, HAS_G, HAS_GK, READ_INITIAL, WRITE_RECURRENT>(
            recurrentState, initialState, stateOut, attnCursor, value, row,
            query0, query1, key0, key1, gk0, gk1, gamma, beta, mask32, mask16, unalignStore);
        RgdrProcessK128Row<StateT, OutT, HAS_G, HAS_GK, READ_INITIAL, WRITE_RECURRENT>(
            recurrentState, initialState, stateOut, attnCursor, value, static_cast<uint16_t>(row + 1),
            query0, query1, key0, key1, gk0, gk1, gamma, beta, mask32, mask16, unalignStore);
    }
    // Preserve the unaligned-store cache across all contiguous scalar outputs and flush the tail only once.
    StoreUnAlignPost(attnCursor, unalignStore, 0);
}

} // namespace RecurrentGatedDeltaRule

#endif // RECURRENT_GATED_DELTA_RULE_REGBASE_H
