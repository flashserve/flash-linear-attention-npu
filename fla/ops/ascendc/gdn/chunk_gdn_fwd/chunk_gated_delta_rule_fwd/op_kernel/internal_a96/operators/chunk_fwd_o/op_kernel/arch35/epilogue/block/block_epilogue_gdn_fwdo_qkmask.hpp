/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_QKMASK_HPP

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 9201
#endif

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_o_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif

__simd_vf__ inline void QkmaskCausalMaskVf(
    __ubuf__ float* __restrict__ upAddr,
    __ubuf__ float* __restrict__ maskBase,
    uint32_t mActualThisStage,
    uint32_t alignedNActual,
    uint32_t gbrcStart,
    uint32_t gbrcEffStart,
    uint32_t gbrcRealStart,
    uint32_t gbrcRealEnd)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    (void)gbrcEffStart;

    RegTensor<float> vregUp;
    RegTensor<float> vregMask;
    RegTensor<float> vregZero;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();

    Duplicate(vregZero, 0.0f, maskFull);

    uint32_t gbrcEnd = gbrcStart + mActualThisStage;

    for (uint32_t absRow = gbrcStart; absRow < (gbrcEnd > 64 ? 64 : gbrcEnd); ++absRow) {
        uint32_t row = absRow - gbrcStart;
        __ubuf__ float* rowAddr = upAddr + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase + absRow * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mins(vregUp, vregUp, 0.0f, maskFull);
        Exp(vregUp, vregUp, maskFull);
        Mul(vregUp, vregUp, vregMask, maskFull);
        StoreAlign(rowAddr, vregUp, maskFull);

        StoreAlign(rowAddr + VL, vregZero, maskFull);
    }

    const uint32_t secondLoopBegin = gbrcStart > VL ? gbrcStart : VL;
    for (uint32_t absRow = secondLoopBegin; absRow < gbrcEnd; ++absRow) {
        uint32_t row = absRow - gbrcStart;
        __ubuf__ float* rowAddr = upAddr + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase + (absRow - 64) * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        Mins(vregUp, vregUp, 0.0f, maskFull);
        Exp(vregUp, vregUp, maskFull);
        StoreAlign(rowAddr, vregUp, maskFull);

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr + VL);
        Mins(vregUp, vregUp, 0.0f, maskFull);
        Exp(vregUp, vregUp, maskFull);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mul(vregUp, vregUp, vregMask, maskFull);
        StoreAlign(rowAddr + VL, vregUp, maskFull);
    }
}

__simd_vf__ inline void QkmaskCausalMaskVfTail(
    __ubuf__ float* __restrict__ upAddr,
    __ubuf__ float* __restrict__ maskBase,
    uint32_t mActualThisStage,
    uint32_t alignedNActual,
    uint32_t gbrcStart,
    uint32_t gbrcEffStart,
    uint32_t gbrcRealStart,
    uint32_t gbrcRealEnd)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    (void)gbrcEffStart;
    (void)gbrcRealStart;
    (void)gbrcRealEnd;

    RegTensor<float> vregUp;
    RegTensor<float> vregMask;
    RegTensor<float> vregZero;
    MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();

    Duplicate(vregZero, 0.0f, maskFull);

    uint32_t gbrcEnd = gbrcStart + mActualThisStage;
    uint32_t tailWidth = alignedNActual - VL;
    uint32_t tailWidth1 = alignedNActual > VL ? VL : alignedNActual;
    uint32_t tailWidth2 = alignedNActual > tailWidth1 ? alignedNActual - tailWidth1 : 0;
    MaskReg maskTail1 = UpdateMask<float>(tailWidth1);
    MaskReg maskTail2 = UpdateMask<float>(tailWidth2);

    for (uint32_t absRow = gbrcStart; absRow < (gbrcEnd > 64 ? 64 : gbrcEnd); ++absRow) {
        uint32_t row = absRow - gbrcStart;
        __ubuf__ float* rowAddr = upAddr + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase + absRow * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mins(vregUp, vregUp, 0.0f, maskTail1);
        Exp(vregUp, vregUp, maskTail1);
        Mul(vregUp, vregUp, vregMask, maskTail1);
        StoreAlign(rowAddr, vregUp, maskTail1);
        StoreAlign(rowAddr + VL, vregZero, maskTail2);
    }

    const uint32_t secondLoopBegin = gbrcStart > VL ? gbrcStart : VL;
    for (uint32_t absRow = secondLoopBegin; absRow < gbrcEnd; ++absRow) {
        uint32_t row = absRow - gbrcStart;
        __ubuf__ float* rowAddr = upAddr + row * alignedNActual;
        __ubuf__ float* maskRowAddr = maskBase + (absRow - 64) * VL;

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr);
        Mins(vregUp, vregUp, 0.0f, maskTail1);
        Exp(vregUp, vregUp, maskTail1);
        StoreAlign(rowAddr, vregUp, maskTail1);

        LoadAlign<float, LoadDist::DIST_NORM>(vregUp, rowAddr + VL);
        Mins(vregUp, vregUp, 0.0f, maskTail2);
        Exp(vregUp, vregUp, maskTail2);
        LoadAlign<float, LoadDist::DIST_NORM>(vregMask, maskRowAddr);
        Mul(vregUp, vregUp, vregMask, maskTail2);
        StoreAlign(rowAddr + VL, vregUp, maskTail2);
    }
}

namespace Catlass::Epilogue::Block {

template <
    class AOutputType_,
    class GInputType_,
    class AInputType_,
    class MaskInputType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdOQkmask,
    AOutputType_,
    GInputType_,
    AInputType_,
    MaskInputType_
> {
public:

    using DispatchPolicy = EpilogueAtlasGDNFwdOQkmask;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using AElementOutput = typename AOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    using MaskElementInput = typename MaskInputType_::Element;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_ELENUM_PER_BLK = 8;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t UB_TILE_SIZE = 16384;
    static constexpr uint32_t UB_LINE_SIZE = 512;
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;
    static constexpr uint32_t MULTIPLIER = 2;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t BASE = 0;
        constexpr uint32_t MASK_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_SIZE = 40 * UB_LINE_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 16 * UB_LINE_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t G_FLOAT_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_OFFSET = MASK_UB_TENSOR_OFFSET + MASK_UB_TENSOR_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_OFFSET = GBRCLEFTCAST_UB_TENSOR_OFFSET + GBRCLEFTCAST_UB_TENSOR_SIZE;
        constexpr uint32_t GCOMP_UB_TENSOR_OFFSET = GBRCUP_UB_TENSOR_OFFSET + GBRCUP_UB_TENSOR_SIZE;
        constexpr uint32_t SHARE_UB_TENSOR_OFFSET = GCOMP_UB_TENSOR_OFFSET + G_FLOAT_UB_TENSOR_SIZE;

        maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(MASK_UB_TENSOR_OFFSET);
        gbrcLeftcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCLEFTCAST_UB_TENSOR_OFFSET);
        gbrcUpUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCUP_UB_TENSOR_OFFSET);
        gcompUbTensor = resource.ubBuf.template GetBufferByByte<float>(GCOMP_UB_TENSOR_OFFSET);
        shareUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_UB_TENSOR_OFFSET);

        constexpr uint32_t G_UB_TENSOR_OFFSET_PING = SHARE_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET_PING = G_UB_TENSOR_OFFSET_PING + G_FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t A_UB_TENSOR_OFFSET_PING = G_HALF_UB_TENSOR_OFFSET_PING + G_HALF_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_UB_TENSOR_OFFSET_PING = A_UB_TENSOR_OFFSET_PING + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_HALF_UB_TENSOR_OFFSET_PING = OUT_UB_TENSOR_OFFSET_PING + FLOAT_UB_TENSOR_SIZE;

        gUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(G_UB_TENSOR_OFFSET_PING);
        gUbFPTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PING);
        gUbBFTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PING);
        aUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(A_UB_TENSOR_OFFSET_PING);
        outUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(OUT_UB_TENSOR_OFFSET_PING);
        outUbFPTensorPing = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PING);
        outUbBFTensorPing = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PING);

        constexpr uint32_t G_UB_TENSOR_OFFSET_PONG = 32 * UB_LINE_SIZE + OUT_HALF_UB_TENSOR_OFFSET_PING + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET_PONG = G_UB_TENSOR_OFFSET_PONG + G_FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t A_UB_TENSOR_OFFSET_PONG = G_HALF_UB_TENSOR_OFFSET_PONG + G_HALF_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_UB_TENSOR_OFFSET_PONG = A_UB_TENSOR_OFFSET_PONG + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_HALF_UB_TENSOR_OFFSET_PONG = OUT_UB_TENSOR_OFFSET_PONG + FLOAT_UB_TENSOR_SIZE;

        gUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(G_UB_TENSOR_OFFSET_PONG);
        gUbFPTensorPong = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PONG);
        gUbBFTensorPong = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PONG);
        aUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(A_UB_TENSOR_OFFSET_PONG);
        outUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(OUT_UB_TENSOR_OFFSET_PONG);
        outUbFPTensorPong = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PONG);
        outUbBFTensorPong = resource.ubBuf.template GetBufferByByte<AElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PONG);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<AElementOutput> maskOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<AElementInput> attnInput,
        AscendC::GlobalTensor<MaskElementInput> boolInput,
        uint32_t fullChunkSize,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        uint32_t &pingpongFlag
        , uint32_t batchIdx, uint32_t headIdx, uint32_t chunkIdx,
        Arch::CrossCoreFlag* waitFlag = nullptr
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nActual = chunkSize;
        uint32_t alignedNActual = CeilDiv(nActual, 16) * 16;
        bool isContiguousFullTile = chunkSize == fullChunkSize && nActual == alignedNActual;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetA = mOffset * nActual + nOffset;
        uint16_t aInputDstStride;
        if((nActual - 1) % 16 <= 7) aInputDstStride = 1;
        else aInputDstStride = 0;

        uint32_t gbrcStart, gbrcRealStart, gbrcRealEnd, gbrcRealProcess, gbrcEffStart, gbrcEffEnd, mulsRemain, mulsRemainIdx;
        if(mActualThisSubBlock <= 32)
        {   if(subBlockIdx == 0)
            {
                gbrcStart = 0;
                gbrcRealStart = 0;
                gbrcRealProcess = mActualThisSubBlock;
            }
            else
            {
                gbrcStart = mActualPerSubBlock;
                gbrcRealStart = gbrcStart & ~7;
                gbrcRealProcess = mActual - gbrcRealStart;
            }

            gbrcEffStart = gbrcStart - gbrcRealStart;
            gbrcEffEnd = gbrcEffStart + mActualThisSubBlock;

            uint32_t dstUpShape_[2] = {mActualThisSubBlock, alignedNActual};
            uint32_t srcUpShape_[2] = {1, alignedNActual};
            uint32_t dstLeftShape_[2] = {gbrcRealProcess, alignedNActual};
            uint32_t srcLeftShape_[2] = {gbrcRealProcess, 1};

            AscendC::ResetMask();
            AscendC::GlobalTensor<AElementOutput> maskOutputThisSubBlock = maskOutput[gbrcStart * nActual];
            AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[gbrcStart * nActual];
            AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

            AscendC::DataCopyParams aInputUbParams{(uint16_t)mActualThisSubBlock, (uint16_t)(nActual*sizeof(float)), 0, aInputDstStride};
            AscendC::DataCopyPadParams aInputUbPadParams{false, 0, 0, 0};

            AscendC::DataCopyExtParams aOutputUbParams{(uint16_t)mActualThisSubBlock, (uint32_t)(nActual*sizeof(half)), 0, 0, 0};

            AscendC::DataCopyParams gfloatUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
            AscendC::DataCopyParams ghalfUbParams{1, (uint16_t)(mActual*sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

            AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
            AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
            AscendC::LocalTensor<AElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
            AscendC::LocalTensor<AElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;
            AscendC::LocalTensor<float> gUbTensor = (pingpongFlag == 0) ? gUbTensorPing : gUbTensorPong;
            AscendC::LocalTensor<GElementInput> gUbFPTensor = (pingpongFlag == 0) ? gUbFPTensorPing : gUbFPTensorPong;
            AscendC::LocalTensor<GElementInput> gUbBFTensor = (pingpongFlag == 0) ? gUbBFTensorPing : gUbBFTensorPong;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            if constexpr(std::is_same<GElementInput, float>::value) {
                AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gfloatUbParams, gUbPadParams);
            } else {
                AscendC::DataCopyPad(gUbFPTensor, gInputThisSubBlock, ghalfUbParams, gUbPadParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            if constexpr(!std::is_same<GElementInput, float>::value) {
                AscendC::Cast(gUbTensor, gUbFPTensor, AscendC::RoundMode::CAST_NONE, mActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Copy(gcompUbTensor, gUbTensor, 64, 2, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Broadcast<float, 2, 0>(gbrcUpUbTensor, gcompUbTensor, dstUpShape_, srcUpShape_, shareUbTensor);
            AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gcompUbTensor[gbrcRealStart], dstLeftShape_, srcLeftShape_, shareUbTensor);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(gbrcUpUbTensor, gbrcLeftcastUbTensor[gbrcEffStart*alignedNActual], gbrcUpUbTensor, mActualThisSubBlock * alignedNActual);
            AscendC::PipeBarrier<PIPE_V>();
            gbrcRealEnd = CeilDiv(gbrcStart + mActualThisSubBlock, 8) * 8;

            {
                auto upAddr = reinterpret_cast<uint64_t>(gbrcUpUbTensor.GetPhyAddr());
                auto maskBase = reinterpret_cast<uint64_t>(maskUbTensor.GetPhyAddr());
                QkmaskCausalMaskVfTail((__ubuf__ float*)upAddr,
                                       (__ubuf__ float*)maskBase,
                                       mActualThisSubBlock, alignedNActual,
                                       gbrcStart, gbrcEffStart, gbrcRealStart, gbrcRealEnd);
                AscendC::PipeBarrier<PIPE_V>();
            }
            (void)gbrcRealEnd;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            if (waitFlag) Arch::CrossCoreWaitFlag(*waitFlag);
            if(isContiguousFullTile) AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mActualThisSubBlock*nActual);
            else AscendC::DataCopyPad(aUbTensor, attnInputThisSubBlock, aInputUbParams, aInputUbPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::Mul(outUbTensor, aUbTensor, gbrcUpUbTensor, mActualThisSubBlock * alignedNActual);
            AscendC::PipeBarrier<PIPE_V>();

            if(std::is_same<AElementOutput, half>::value)
            {
                AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbFPTensor, mActualThisSubBlock*nActual);
                else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbFPTensor, aOutputUbParams);
            }
            else
            {
                AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbBFTensor, mActualThisSubBlock*nActual);
                else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbBFTensor, aOutputUbParams);
            }
            pingpongFlag = 1 - pingpongFlag;
        }
        else
        {
            AscendC::ResetMask();
            AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

            AscendC::DataCopyParams gfloatUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
            AscendC::DataCopyParams ghalfUbParams{1, (uint16_t)(mActual*sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

            AscendC::LocalTensor<float> gUbTensor = (pingpongFlag == 0) ? gUbTensorPing : gUbTensorPong;
            AscendC::LocalTensor<GElementInput> gUbFPTensor = (pingpongFlag == 0) ? gUbFPTensorPing : gUbFPTensorPong;
            AscendC::LocalTensor<GElementInput> gUbBFTensor = (pingpongFlag == 0) ? gUbBFTensorPing : gUbBFTensorPong;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            if constexpr(std::is_same<GElementInput, float>::value) {
                AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gfloatUbParams, gUbPadParams);
            } else {
                AscendC::DataCopyPad(gUbFPTensor, gInputThisSubBlock, ghalfUbParams, gUbPadParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            if constexpr(!std::is_same<GElementInput, float>::value) {
                AscendC::Cast(gUbTensor, gUbFPTensor, AscendC::RoundMode::CAST_NONE, mActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Copy(gcompUbTensor, gUbTensor, 64, 2, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t mActualPerStage = CeilDiv(mActualThisSubBlock, 2);
            uint32_t mActualThisStage = 0;
            for(uint32_t stage = 0; stage < 2; ++stage)
            {
                if(stage==0) mActualThisStage = mActualPerStage;
                else mActualThisStage = mActualThisSubBlock - mActualPerStage;

                if(subBlockIdx == 0 && stage == 0)
                {
                    gbrcStart = 0;
                    gbrcRealStart = 0;
                    gbrcRealProcess = mActualThisStage;
                }
                else if(subBlockIdx == 0 && stage == 1)
                {
                    gbrcStart = mActualPerStage;
                    gbrcRealStart = gbrcStart & ~7;
                    gbrcRealProcess = mActualThisSubBlock - gbrcRealStart;
                }
                else if(subBlockIdx == 1 && stage == 0)
                {
                    gbrcStart = mActualPerSubBlock;
                    gbrcRealStart = gbrcStart & ~7;
                    gbrcRealProcess = mActualPerSubBlock + mActualThisStage - gbrcRealStart;
                }
                else if(subBlockIdx == 1 && stage == 1)
                {
                    gbrcStart = mActualPerSubBlock + mActualPerStage;
                    gbrcRealStart = gbrcStart & ~7;
                    gbrcRealProcess = mActual - gbrcRealStart;
                }

                gbrcEffStart = gbrcStart - gbrcRealStart;

                AscendC::GlobalTensor<AElementOutput> maskOutputThisSubBlock = maskOutput[gbrcStart * nActual];
                AscendC::GlobalTensor<AElementInput> attnInputThisSubBlock = attnInput[gbrcStart * nActual];

                AscendC::DataCopyParams aInputUbParams{(uint16_t)mActualThisStage, (uint16_t)(nActual*sizeof(float)), 0, aInputDstStride};
                AscendC::DataCopyPadParams aInputUbPadParams{false, 0, 0, 0};

                AscendC::DataCopyExtParams aOutputUbParams{(uint16_t)mActualThisStage, (uint32_t)(nActual*sizeof(half)), 0, 0, 0};

                AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
                AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
                AscendC::LocalTensor<AElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
                AscendC::LocalTensor<AElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                if (waitFlag && stage == 0) Arch::CrossCoreWaitFlag(*waitFlag);
                if(isContiguousFullTile) AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mActualThisStage*nActual);
                else AscendC::DataCopyPad(aUbTensor, attnInputThisSubBlock, aInputUbParams, aInputUbPadParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);

                uint32_t dstUpShape_[2] = {mActualThisStage, alignedNActual};
                uint32_t srcUpShape_[2] = {1, alignedNActual};
                uint32_t dstLeftShape_[2] = {gbrcRealProcess, alignedNActual};
                uint32_t srcLeftShape_[2] = {gbrcRealProcess, 1};

                AscendC::Broadcast<float, 2, 0>(gbrcUpUbTensor, gcompUbTensor, dstUpShape_, srcUpShape_, shareUbTensor);
                AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gcompUbTensor[gbrcRealStart], dstLeftShape_, srcLeftShape_, shareUbTensor);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(gbrcUpUbTensor, gbrcLeftcastUbTensor[gbrcEffStart*alignedNActual], gbrcUpUbTensor, mActualThisStage * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();

                gbrcRealEnd = CeilDiv(gbrcStart + mActualThisStage, 8) * 8;

                {
                    auto upAddr = reinterpret_cast<uint64_t>(gbrcUpUbTensor.GetPhyAddr());
                    auto maskBase = reinterpret_cast<uint64_t>(maskUbTensor.GetPhyAddr());
                    if (alignedNActual == 128) {
                        QkmaskCausalMaskVf((__ubuf__ float*)upAddr,
                                           (__ubuf__ float*)maskBase,
                                           mActualThisStage, alignedNActual,
                                           gbrcStart, gbrcEffStart, gbrcRealStart, gbrcRealEnd);
                    } else {
                        QkmaskCausalMaskVfTail((__ubuf__ float*)upAddr,
                                               (__ubuf__ float*)maskBase,
                                               mActualThisStage, alignedNActual,
                                               gbrcStart, gbrcEffStart, gbrcRealStart, gbrcRealEnd);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                (void)gbrcRealEnd;
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
                AscendC::Mul(outUbTensor, aUbTensor, gbrcUpUbTensor, mActualThisStage * alignedNActual);
                AscendC::PipeBarrier<PIPE_V>();
                if(std::is_same<AElementOutput, half>::value)
                {
                    AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisStage * alignedNActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbFPTensor, mActualThisStage*nActual);
                    else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbFPTensor, aOutputUbParams);
                }
                else
                {
                    AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisStage * alignedNActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    if(isContiguousFullTile) AscendC::DataCopy(maskOutputThisSubBlock, outUbBFTensor, mActualThisStage*nActual);
                    else AscendC::DataCopyPad(maskOutputThisSubBlock, outUbBFTensor, aOutputUbParams);
                }
                pingpongFlag = 1 - pingpongFlag;
            }
        }

    }

private:
    AscendC::LocalTensor<float> maskUbTensor;
    AscendC::LocalTensor<float> gbrcLeftcastUbTensor;
    AscendC::LocalTensor<float> gbrcUpUbTensor;
    AscendC::LocalTensor<float> gcompUbTensor;
    AscendC::LocalTensor<uint8_t> shareUbTensor;

    AscendC::LocalTensor<float> gUbTensorPing;
    AscendC::LocalTensor<GElementInput> gUbFPTensorPing;
    AscendC::LocalTensor<GElementInput> gUbBFTensorPing;
    AscendC::LocalTensor<float> aUbTensorPing;
    AscendC::LocalTensor<float> outUbTensorPing;
    AscendC::LocalTensor<AElementOutput> outUbFPTensorPing;
    AscendC::LocalTensor<AElementOutput> outUbBFTensorPing;

    AscendC::LocalTensor<float> gUbTensorPong;
    AscendC::LocalTensor<GElementInput> gUbFPTensorPong;
    AscendC::LocalTensor<GElementInput> gUbBFTensorPong;
    AscendC::LocalTensor<float> aUbTensorPong;
    AscendC::LocalTensor<float> outUbTensorPong;
    AscendC::LocalTensor<AElementOutput> outUbFPTensorPong;
    AscendC::LocalTensor<AElementOutput> outUbBFTensorPong;

};
}

#endif
