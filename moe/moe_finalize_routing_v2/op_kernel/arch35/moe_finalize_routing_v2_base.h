/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_v2_partial_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_BASE_H_
#define MOE_FINALIZE_ROUTING_V2_BASE_H_
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"	
#include "op_kernel/math_util.h"

namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;

constexpr int32_t DROPLESS_COLUMN = 0; // 按列读取
constexpr int32_t DROP_PAD_COLUMN = 1; // 按列读取
constexpr int32_t DROPLESS_ROW = 2;    // 按行读取
constexpr int32_t DROP_PAD_ROW = 3;    // 按行读取
constexpr int32_t INVALID_IDX = -1;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);	
    return Ops::Base::CeilAlign(num, elemNum);
}

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataUnalign(
    RegTensor<float>& dst, __local_mem__ T*& src, UnalignReg& uSrc, MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyUnAlign(dst, uSrc, src, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpUnPack;
        DataCopyUnAlign(tmp, uSrc, src, postUpdateStride);
        UnPack((RegTensor<uint32_t>&)tmpUnPack, (RegTensor<uint16_t>&)tmp);
        Cast<float, T, castTraitB162B32Even>(dst, tmpUnPack, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataWithBrc(
    RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOuputDataUnalign(
    RegTensor<float>& src, __local_mem__ T*& dst, UnalignReg& uDst, MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyUnAlign(dst, src, uDst, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpPack;
        Cast<T, float, castTraitB322B16Even>(tmpPack, src, pregLoop);
        Pack((RegTensor<uint16_t>&)tmpPack, (RegTensor<uint32_t>&)tmp);
        DataCopyUnAlign(dst, tmpPack, uDst, postUpdateStride);
    }
}

template <typename T, bool hasX2>
__aicore__ inline void VFProcessX1AndX2(
    const LocalTensor<float>& yLocal, const LocalTensor<T>& x1Local, const LocalTensor<T>& x2Local, uint16_t processLen)
{
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ T* x1LocalAddr = (__local_mem__ T*)x1Local.GetPhyAddr();
    __local_mem__ T* x2LocalAddr = (__local_mem__ T*)x2Local.GetPhyAddr();
    uint16_t loopCount = Ops::Base::CeilDiv<uint16_t>(processLen, VL_FP32);	
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> y;
        MaskReg pregLoop;
        uint32_t sreg = processLen;
        for (uint16_t i = 0; i < loopCount; i += 1) {
            pregLoop = UpdateMask<float>(sreg);
            LoadInputData<T>(x1, x1LocalAddr, pregLoop, i * VL_FP32);
            if constexpr (hasX2) {
                LoadInputData<T>(x2, x2LocalAddr, pregLoop, i * VL_FP32);
                Add(y, x1, x2, pregLoop);
                DataCopy(yLocalAddr + i * VL_FP32, y, pregLoop);
            } else {
                DataCopy(yLocalAddr + i * VL_FP32, x1, pregLoop);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ProcessX1AndX2(
    const LocalTensor<float>& yLocal, const LocalTensor<T>& x1Local, const LocalTensor<T>& x2Local, uint16_t processLen,
    bool hasX1, bool hasX2)
{
    if (hasX1 && hasX2) {
        VFProcessX1AndX2<T, true>(yLocal, x1Local, x2Local, processLen);
    } else if (hasX1 && !hasX2) {
        VFProcessX1AndX2<T, false>(yLocal, x1Local, x2Local, processLen);
    } else {
        AscendC::Duplicate(yLocal, static_cast<float>(0.0f), processLen);
    }
}

template <typename T, typename S, bool hasBias, bool hasScale>
__aicore__ inline void VFProcessExpandXBiasScale(
    const LocalTensor<float> yLocal, const LocalTensor<T>& expandedXLocal, const LocalTensor<T>& biasLocal, S scale,
    uint16_t processLen)
{
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ float* srcLocalAddr = yLocalAddr;
    __local_mem__ T* expandedXLocalAddr = (__local_mem__ T*)expandedXLocal.GetPhyAddr();
    __local_mem__ T* biasLocalAddr = hasBias ? (__local_mem__ T*)biasLocal.GetPhyAddr() : nullptr;

    uint16_t loopCount = processLen / VL_FP32;
    uint16_t tailNum = processLen - loopCount * VL_FP32;
    uint16_t tailLoop = Ops::Base::CeilDiv<uint16_t>(tailNum, VL_FP32);	
    __VEC_SCOPE__
    {
        RegTensor<float> expandedX;
        RegTensor<float> bias;
        RegTensor<float> y;
        RegTensor<float> src;
        RegTensor<float> scaleReg;
        RegTensor<S> tmp;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<S, AscendC::MicroAPI::MaskPattern::ALL>();
        UnalignReg uSrc;
        UnalignReg uExpandedX;
        UnalignReg uBias;
        UnalignReg uDst;
        DataCopyUnAlignPre<float>(uSrc, srcLocalAddr);
        DataCopyUnAlignPre<T>(uExpandedX, expandedXLocalAddr);
        if constexpr (hasBias) {
            DataCopyUnAlignPre<T>(uBias, biasLocalAddr);
        }
        if constexpr (hasScale) {
            if constexpr (IsSameType<S, float>::value) {
                Duplicate(scaleReg, scale, pregMain);
            } else if constexpr (IsSameType<S, half>::value || IsSameType<S, bfloat16_t>::value) {
                Duplicate(tmp, scale, pregMain);
                Cast<float, S, castTraitB162B32Even>(scaleReg, tmp, pregMain);
            }
        }
        for (uint16_t i = 0; i < loopCount; i++) {
            LoadInputDataUnalign<T>(expandedX, expandedXLocalAddr, uExpandedX, pregMain, VL_FP32);
            LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregMain, VL_FP32);
            if constexpr (hasBias) {
                LoadInputDataUnalign<T>(bias, biasLocalAddr, uBias, pregMain, VL_FP32);
                Add(expandedX, expandedX, bias, pregMain);
            }
            if constexpr (hasScale) {
                Mul(expandedX, expandedX, scaleReg, pregMain);
            }
            Add(y, src, expandedX, pregMain);
            StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregMain, VL_FP32);
        }

        uint32_t sreg = tailNum;
        for (uint16_t i = 0; i < tailLoop; i++) {
            pregLoop = UpdateMask<float>(sreg);
            LoadInputDataUnalign<T>(expandedX, expandedXLocalAddr, uExpandedX, pregLoop, tailNum);
            LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregLoop, tailNum);
            if constexpr (hasBias) {
                LoadInputDataUnalign<T>(bias, biasLocalAddr, uBias, pregLoop, tailNum);
                Add(expandedX, expandedX, bias, pregLoop);
            }
            if constexpr (hasScale) {
                Mul(expandedX, expandedX, scaleReg, pregLoop);
            }
            Add(y, src, expandedX, pregLoop);
            StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregLoop, tailNum);
        }
        DataCopyUnAlignPost(yLocalAddr, uDst, 0);
    }
}

template <typename T, typename S>
__aicore__ inline void ProcessExpandXBiasScale(
    const LocalTensor<float> yLocal, const LocalTensor<T>& expandedXLocal, const LocalTensor<T>& biasLocal, S scale,
    uint16_t processLen, bool hasBias, bool hasScale)
{
    if (hasBias && hasScale) {
        VFProcessExpandXBiasScale<T, S, true, true>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    } else if (hasBias && !hasScale) {
        VFProcessExpandXBiasScale<T, S, true, false>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    } else if (!hasBias && hasScale) {
        VFProcessExpandXBiasScale<T, S, false, true>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    } else {
        VFProcessExpandXBiasScale<T, S, false, false>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    }
}

template <typename T, typename S, bool hasBias, bool hasScale>
__aicore__ inline void VFProcessExpandXBiasScaleOptimized(
    const LocalTensor<float>& yLocal, const LocalTensor<T>& expandedXLocal, const LocalTensor<T>& biasLocal,
    const LocalTensor<S>& scalesLocal, uint16_t validK, uint16_t processLen)
{
    __local_mem__ float* yOriginAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ float* yLocalAddr = yOriginAddr;
    __local_mem__ float* srcLocalAddr = yOriginAddr;
    __local_mem__ T* expandedXLocalAddr = (__local_mem__ T*)expandedXLocal.GetPhyAddr();
    __local_mem__ T* biasLocalAddr = hasBias ? (__local_mem__ T*)biasLocal.GetPhyAddr() : nullptr;
    __local_mem__ S* scalesLocalAddr = hasScale ? (__local_mem__ S*)scalesLocal.GetPhyAddr() : nullptr;

    uint16_t loopCount = processLen / VL_FP32;
    uint16_t tailNum = processLen - loopCount * VL_FP32;
    uint16_t tailLoop = Ops::Base::CeilDiv<uint16_t>(tailNum, VL_FP32);	
    uint16_t processLenAlign = RoundUp<T>(processLen);
    __VEC_SCOPE__
    {
        RegTensor<float> expandedX;
        RegTensor<float> bias;
        RegTensor<float> y;
        RegTensor<float> src;
        RegTensor<float> scaleReg;
        RegTensor<T> tmp;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
        UnalignReg uSrc;
        UnalignReg uExpandedX;
        UnalignReg uBias;
        UnalignReg uDst;

        for (uint16_t k = 0; k < validK; k++) {
            srcLocalAddr = yOriginAddr;
            yLocalAddr = yOriginAddr;
            DataCopyUnAlignPre<float>(uSrc, srcLocalAddr);
            if constexpr (hasScale) {
                LoadInputDataWithBrc<S>(scaleReg, scalesLocalAddr, pregMain, k);
            }
            for (uint16_t i = 0; i < loopCount; i++) {
                LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregMain, VL_FP32);
                LoadInputData<T>(expandedX, expandedXLocalAddr, pregMain, k * processLenAlign + i * VL_FP32);
                if constexpr (hasBias) {
                    LoadInputData<T>(bias, biasLocalAddr, pregMain, k * processLenAlign + i * VL_FP32);
                    Add(expandedX, expandedX, bias, pregMain);
                }
                if constexpr (hasScale) {
                    Mul(expandedX, expandedX, scaleReg, pregMain);
                }
                Add(y, src, expandedX, pregMain);
                StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregMain, VL_FP32);
            }
            uint32_t sreg = tailNum;
            for (uint16_t i = 0; i < tailLoop; i++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregLoop, tailNum);
                LoadInputData<T>(expandedX, expandedXLocalAddr, pregLoop, k * processLenAlign + loopCount * VL_FP32);
                if constexpr (hasBias) {
                    LoadInputData<T>(bias, biasLocalAddr, pregMain, k * processLenAlign + loopCount * VL_FP32);
                    Add(expandedX, expandedX, bias, pregLoop);
                }
                if constexpr (hasScale) {
                    Mul(expandedX, expandedX, scaleReg, pregLoop);
                }
                Add(y, src, expandedX, pregLoop);
                StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregLoop, tailNum);
            }
            DataCopyUnAlignPost(yLocalAddr, uDst, 0);
            LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }
}

template <typename T, typename S>
__aicore__ inline void ProcessExpandXBiasScaleOptimized(
    const LocalTensor<float>& yLocal, const LocalTensor<T>& expandedXLocal, const LocalTensor<T>& biasLocal,
    const LocalTensor<S>& scalesLocal, uint16_t validK, uint16_t processLen, bool hasBias, bool hasScale)
{
    if (hasBias && hasScale) {
        VFProcessExpandXBiasScaleOptimized<T, S, true, true>(
            yLocal, expandedXLocal, biasLocal, scalesLocal, validK, processLen);
    } else if (hasBias && !hasScale) {
        VFProcessExpandXBiasScaleOptimized<T, S, true, false>(
            yLocal, expandedXLocal, biasLocal, scalesLocal, validK, processLen);
    } else if (!hasBias && hasScale) {
        VFProcessExpandXBiasScaleOptimized<T, S, false, true>(
            yLocal, expandedXLocal, biasLocal, scalesLocal, validK, processLen);
    } else {
        VFProcessExpandXBiasScaleOptimized<T, S, false, false>(
            yLocal, expandedXLocal, biasLocal, scalesLocal, validK, processLen);
    }
}

template <typename T>
__aicore__ inline void CopyIn(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst, const uint32_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyOut(
    const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm, const uint16_t nBurst, const uint32_t copyLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(outputGm, outputTensor, dataCopyParams);
}
} // namespace MoeFinalizeRoutingV2Regbase

#endif