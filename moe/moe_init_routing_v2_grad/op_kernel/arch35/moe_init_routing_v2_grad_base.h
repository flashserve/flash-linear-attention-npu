/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_full_load.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V2_GRAD_BASE_COMPUTE_H
#define MOE_INIT_ROUTING_V2_GRAD_BASE_COMPUTE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV2Grad {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr int64_t DROP_PAD_MODE = 1;
constexpr int64_t ACTIVE_MODE = 2;
constexpr uint32_t VL_F32 = Ops::Base::GetVRegSize() / sizeof(float);
constexpr int64_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t SCALE_COEF_TWO = 2;
constexpr int64_t SCALE_COEF_FOUR = 4;
constexpr int64_t SCALE_COEF_EIGHT = 8;

constexpr uint16_t ROW_ZERO = 0;
constexpr uint16_t ROW_ONE = 1;
constexpr uint16_t ROW_TWO = 2;
constexpr uint16_t ROW_THREE = 3;
constexpr uint16_t ROW_FOUR = 4;
constexpr uint16_t ROW_FIVE = 5;
constexpr uint16_t ROW_SIX = 6;
constexpr uint16_t ROW_SEVEN = 7;

constexpr uint32_t ROW_TWO_OFFSET = 2;
constexpr uint32_t ROW_THREE_OFFSET = 3;
constexpr uint32_t ROW_FOUR_OFFSET = 4;
constexpr uint32_t ROW_FIVE_OFFSET = 5;
constexpr uint32_t ROW_SIX_OFFSET = 6;
constexpr uint32_t ROW_SEVEN_OFFSET = 7;

constexpr int64_t MAX_BINARY_ADD_BUFFER_CNT = 64;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr uint16_t BINARY_ADD_LOOP_COUNT = 4;

constexpr int64_t UNROLL_COEF = 4;
constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline void BinaryAddVF(
    __local_mem__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint32_t offset, uint16_t binaryAddKLoop,
    uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop, RegTensor<float>& x1,
    RegTensor<float>& x2, RegTensor<float>& x3, RegTensor<float>& x4)
{
    uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
        for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
            DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
            DataCopy(x2, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            DataCopy(
                x3, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride +
                     offset));
            DataCopy(
                x4, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride +
                     offset));
            Add(x3, x3, x4, pregLoop);
            Add(x1, x1, x3, pregLoop);
            DataCopy(((__local_mem__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
        DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + offset));
        DataCopy(x2, ((__local_mem__ float*)binaryAddTmpAddr + rLoopStride + offset));
        Add(x1, x1, x2, pregLoop);
        DataCopy(((__local_mem__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <typename T>
__aicore__ inline void LoadOneTensorForDtypeT(
    __local_mem__ T* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
    }
}

template <typename T>
__aicore__ inline void StoreOneTensorForDtypeT(
    __local_mem__ T* output, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> yFp16;
        Cast<half, float, castTraitB322B16>(yFp16, src, preg);
        DataCopy<half, StoreDist::DIST_PACK_B32>(((__local_mem__ half*)output + offset), yFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(((__local_mem__ bfloat16_t*)output + offset), xBf16, preg);
    } else {
        DataCopy(((__local_mem__ float*)output + offset), src, preg);
    }
}

template <typename T>
__aicore__ inline void TwoRowAddWithTail(
    RegTensor<float>& dst, __local_mem__ T* input, MaskReg& preg, uint32_t offset1, uint32_t offset2, uint32_t offset3,
    uint32_t offset4, RegTensor<float>& rem, RegTensor<float>& nextRow, RegTensor<float>& remNextRow)
{
    LoadOneTensorForDtypeT(input, dst, preg, offset1);
    LoadOneTensorForDtypeT(input, rem, preg, offset2);
    Add(dst, dst, rem, preg);
    LoadOneTensorForDtypeT(input, nextRow, preg, offset3);
    LoadOneTensorForDtypeT(input, remNextRow, preg, offset4);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

template <typename T>
__aicore__ inline void TwoRowAdd(
    RegTensor<float>& dst, __local_mem__ T* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
    RegTensor<float>& nextRow)
{
    LoadOneTensorForDtypeT(input, dst, preg, offset1);
    LoadOneTensorForDtypeT(input, nextRow, preg, offset2);
    Add(dst, dst, nextRow, preg);
}

template <typename T>
__aicore__ inline void SixteenRowAddWithTail(
    RegTensor<float>& dst, __local_mem__ T* input, MaskReg& pregLoop, uint32_t quotOffset, uint32_t remOffset,
    uint32_t stride, RegTensor<float>& rem, RegTensor<float>& nextRow, RegTensor<float>& remNextRow,
    RegTensor<float>& x2, RegTensor<float>& x3, RegTensor<float>& x4)
{
    TwoRowAddWithTail(
        dst, input, pregLoop, quotOffset, remOffset, quotOffset + stride, remOffset + stride, rem, nextRow, remNextRow);
    TwoRowAddWithTail(
        x2, input, pregLoop, quotOffset + ROW_TWO_OFFSET * stride, remOffset + ROW_TWO_OFFSET * stride,
        quotOffset + ROW_THREE_OFFSET * stride, remOffset + ROW_THREE_OFFSET * stride, rem, nextRow, remNextRow);
    Add(dst, dst, x2, pregLoop);
    TwoRowAddWithTail(
        x3, input, pregLoop, quotOffset + ROW_FOUR_OFFSET * stride, remOffset + ROW_FOUR_OFFSET * stride,
        quotOffset + ROW_FIVE_OFFSET * stride, remOffset + ROW_FIVE_OFFSET * stride, rem, nextRow, remNextRow);
    TwoRowAddWithTail(
        x4, input, pregLoop, quotOffset + ROW_SIX_OFFSET * stride, remOffset + ROW_SIX_OFFSET * stride,
        quotOffset + ROW_SEVEN_OFFSET * stride, remOffset + ROW_SEVEN_OFFSET * stride, rem, nextRow, remNextRow);
    Add(x3, x3, x4, pregLoop);
    Add(dst, dst, x3, pregLoop);
}

template <typename T, typename U, bool withInit>
__aicore__ inline void SequenceReduceSum(
    __local_mem__ T* inputAddr, __local_mem__ U* outputAddr, int64_t currentN, int64_t currentK, int64_t currentH,
    int64_t currentHAlign, int64_t currentKStride)
{
    int64_t hQuotCount = currentH / VL_F32 / UNROLL_COEF;
    int64_t hRem = currentH - hQuotCount * VL_F32 * UNROLL_COEF;

    uint16_t hQuotLoopCount = hQuotCount;
    uint32_t hRemCount = hRem;
    uint16_t hRemLoopCount = Ops::Base::CeilDiv(hRemCount, VL_F32);
    uint16_t nLoopCount = currentN;
    uint16_t kLoopCount = currentK;

    uint32_t nInputStride = currentKStride * currentHAlign;
    uint32_t nOutputStride = currentHAlign;
    uint32_t hRemOffset = hQuotCount * VL_F32 * UNROLL_COEF;

    __VEC_SCOPE__
    {
        RegTensor<float> sum0;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> sum3;
        RegTensor<float> x0;
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> x3;

        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        // unroll
        for (uint16_t hIdx = 0; hIdx < hQuotLoopCount; hIdx++) {
            uint32_t hOffset = hIdx * VL_F32 * UNROLL_COEF;
            for (uint16_t nIdx = 0; nIdx < nLoopCount; nIdx++) {
                uint32_t nInputOffset = nIdx * nInputStride;
                if constexpr (withInit) {
                    Duplicate(sum0, 0.0, pregMain);
                    Duplicate(sum1, 0.0, pregMain);
                    Duplicate(sum2, 0.0, pregMain);
                    Duplicate(sum3, 0.0, pregMain);
                } else {
                    LoadOneTensorForDtypeT(outputAddr, sum0, pregMain, nIdx * nOutputStride + hOffset);
                    LoadOneTensorForDtypeT(outputAddr, sum1, pregMain, nIdx * nOutputStride + hOffset + VL_F32);
                    LoadOneTensorForDtypeT(
                        outputAddr, sum2, pregMain, nIdx * nOutputStride + hOffset + VL_F32 * ROW_TWO_OFFSET);
                    LoadOneTensorForDtypeT(
                        outputAddr, sum3, pregMain, nIdx * nOutputStride + hOffset + VL_F32 * ROW_THREE_OFFSET);
                }
                for (uint16_t kIdx = 0; kIdx < kLoopCount; kIdx++) {
                    LoadOneTensorForDtypeT(inputAddr, x0, pregMain, nInputOffset + hOffset + kIdx * nOutputStride);
                    LoadOneTensorForDtypeT(
                        inputAddr, x1, pregMain, nInputOffset + hOffset + kIdx * nOutputStride + VL_F32);
                    LoadOneTensorForDtypeT(
                        inputAddr, x2, pregMain,
                        nInputOffset + hOffset + kIdx * nOutputStride + VL_F32 * ROW_TWO_OFFSET);
                    LoadOneTensorForDtypeT(
                        inputAddr, x3, pregMain,
                        nInputOffset + hOffset + kIdx * nOutputStride + VL_F32 * ROW_THREE_OFFSET);
                    Add(sum0, sum0, x0, pregMain);
                    Add(sum1, sum1, x1, pregMain);
                    Add(sum2, sum2, x2, pregMain);
                    Add(sum3, sum3, x3, pregMain);
                }
                StoreOneTensorForDtypeT(outputAddr, sum0, pregMain, nIdx * nOutputStride + hOffset);
                StoreOneTensorForDtypeT(outputAddr, sum1, pregMain, nIdx * nOutputStride + hOffset + VL_F32);
                StoreOneTensorForDtypeT(
                    outputAddr, sum2, pregMain, nIdx * nOutputStride + hOffset + VL_F32 * ROW_TWO_OFFSET);
                StoreOneTensorForDtypeT(
                    outputAddr, sum3, pregMain, nIdx * nOutputStride + hOffset + VL_F32 * ROW_THREE_OFFSET);
            }
        }
        uint32_t sreg = hRem;
        for (uint16_t hIdx = 0; hIdx < hRemLoopCount; hIdx++) {
            pregLoop = UpdateMask<float>(sreg);
            uint32_t hOffset = hIdx * VL_F32 + hRemOffset;
            for (uint16_t nIdx = 0; nIdx < nLoopCount; nIdx++) {
                uint32_t nInputOffset = nIdx * nInputStride;
                if constexpr (withInit) {
                    Duplicate(sum0, 0.0, pregMain);
                } else {
                    LoadOneTensorForDtypeT(outputAddr, sum0, pregLoop, nIdx * nOutputStride + hOffset);
                }
                for (uint16_t kIdx = 0; kIdx < kLoopCount; kIdx++) {
                    LoadOneTensorForDtypeT(inputAddr, x0, pregLoop, nInputOffset + hOffset + kIdx * nOutputStride);
                    Add(sum0, sum0, x0, pregLoop);
                }
                StoreOneTensorForDtypeT(outputAddr, sum0, pregLoop, nIdx * nOutputStride + hOffset);
            }
        }
    }
}
} // namespace MoeInitRoutingV2Grad
#endif // MOE_INIT_ROUTING_V2_GRAD_BASE_COMPUTE_H