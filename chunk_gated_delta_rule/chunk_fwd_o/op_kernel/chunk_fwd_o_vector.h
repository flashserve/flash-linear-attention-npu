/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_fwd_o_vector.h
 * \brief Vector (AIV) operations for chunk_fwd_o, including epilogue implementations
 */

#ifndef CHUNK_FWD_O_VECTOR_H
#define CHUNK_FWD_O_VECTOR_H

#include "chunk_fwd_o_common.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass;

// ============================================================================
// GDNFwdOQkmaskEpilogue: applies causal mask + gate scaling to QK attention
// (moved from catlass/epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp)
// ============================================================================
template <
    class AElementOutput_,
    class GElementInput_,
    class AElementInput_,
    class MaskElementInput_
>
class GDNFwdOQkmaskEpilogue {
public:
    using ArchTag = Arch::AtlasA2;

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
    GDNFwdOQkmaskEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t BASE = 0;
        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t G_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_SIZE = 80 * UB_LINE_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_OFFSET = MASK_UB_TENSOR_OFFSET + MASK_UB_TENSOR_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_OFFSET = GBRCLEFTCAST_UB_TENSOR_OFFSET + GBRCLEFTCAST_UB_TENSOR_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET = GBRCUP_UB_TENSOR_OFFSET + GBRCUP_UB_TENSOR_SIZE;
        constexpr uint32_t SHARE_TENSOR_OFFSET = G_HALF_UB_TENSOR_OFFSET + G_HALF_UB_TENSOR_SIZE;

        maskUbTensor = resource.ubBuf.template GetBufferByByte<half>(MASK_UB_TENSOR_OFFSET);
        gbrcleftcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCLEFTCAST_UB_TENSOR_OFFSET);
        gbrcuphalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(GBRCUP_UB_TENSOR_OFFSET);
        gbrcupfloatUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCUP_UB_TENSOR_OFFSET);
        ghalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(G_HALF_UB_TENSOR_OFFSET);
        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_TENSOR_OFFSET);

        constexpr uint32_t G_UB_TENSOR_OFFSET_PING = SHARE_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_TENSOR_OFFSET_PING = G_UB_TENSOR_OFFSET_PING + G_UB_TENSOR_SIZE;

        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput_>(G_UB_TENSOR_OFFSET_PING);
        outputFPUbTensor_ping = resource.ubBuf.template GetBufferByByte<half>(OUT_TENSOR_OFFSET_PING);
        outputBFUbTensor_ping = resource.ubBuf.template GetBufferByByte<AElementOutput_>(OUT_TENSOR_OFFSET_PING);

        constexpr uint32_t G_UB_TENSOR_OFFSET_PONG = OUT_TENSOR_OFFSET_PING + HALF_UB_TENSOR_SIZE + 64 * UB_LINE_SIZE;
        constexpr uint32_t OUT_TENSOR_OFFSET_PONG = G_UB_TENSOR_OFFSET_PONG + G_UB_TENSOR_SIZE;

        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput_>(G_UB_TENSOR_OFFSET_PONG);
        outputFPUbTensor_pong = resource.ubBuf.template GetBufferByByte<half>(OUT_TENSOR_OFFSET_PONG);
        outputBFUbTensor_pong = resource.ubBuf.template GetBufferByByte<AElementOutput_>(OUT_TENSOR_OFFSET_PONG);
    }

    CATLASS_DEVICE
    ~GDNFwdOQkmaskEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<AElementOutput_> maskOutput,
        AscendC::GlobalTensor<GElementInput_> gInput,
        AscendC::GlobalTensor<AElementInput_> attnInput,
        AscendC::GlobalTensor<MaskElementInput_> boolInput,
        uint32_t fullChunkSize,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim)
    {
        uint32_t mActual = chunkSize;
        uint32_t nActual = chunkSize;
        uint32_t alignedNActual = (nActual+15)/16*16;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetA = mOffset * nActual + nOffset;

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0) {
            gbrcStart = 0;
            gbrcRealStart = 0;
            gbrcReptime = (mActualThisSubBlock + 8 - 1) / 8;
        } else {
            gbrcStart = mActualPerSubBlock;
            gbrcRealStart = gbrcStart & ~15;
            gbrcReptime = (mActual - gbrcRealStart + 8 - 1) / 8;
        }
        gbrcEffStart = gbrcStart-gbrcRealStart;
        gbrcEffEnd = gbrcEffStart + mActualThisSubBlock;

        uint32_t dstUpShape_[2] = {mActualThisSubBlock, alignedNActual};
        uint32_t srcUpShape_[2] = {1, alignedNActual};
        uint32_t dstLeftShape_[2] = {gbrcReptime*8, alignedNActual};
        uint32_t srcLeftShape_[2] = {gbrcReptime*8, 1};

        AscendC::ResetMask();
        AscendC::GlobalTensor<AElementOutput_> maskOutputThisSubBlock = maskOutput[offsetA];
        AscendC::GlobalTensor<AElementInput_> attnInputThisSubBlock = attnInput[offsetA];
        AscendC::GlobalTensor<GElementInput_> gInputThisSubBlock = gInput;

        AscendC::DataCopyParams aInputUbParams{(uint16_t)mActualThisSubBlock, (uint16_t)(nActual*sizeof(half)), 0, 0};
        AscendC::DataCopyPadParams aInputUbPadParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams aOutputUbParams{(uint16_t)mActualThisSubBlock, (uint32_t)(nActual*sizeof(half)), 0, 0, 0};

        AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
        AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

        pingpongFlag = isFirst ? 0 : 4;
        AscendC::LocalTensor<GElementInput_> gUbTensor = pingpongFlag == 0 ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<half> outputFPUbTensor = pingpongFlag == 0 ? outputFPUbTensor_ping : outputFPUbTensor_pong;
        AscendC::LocalTensor<AElementOutput_> outputBFUbTensor = pingpongFlag == 0 ? outputBFUbTensor_ping : outputBFUbTensor_pong;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::Broadcast<float, 2, 0>(gbrcupfloatUbTensor, gUbTensor, dstUpShape_, srcUpShape_, shareBuffer_);
        AscendC::Broadcast<float, 2, 1>(gbrcleftcastUbTensor, gUbTensor[gbrcRealStart], dstLeftShape_, srcLeftShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(gbrcupfloatUbTensor, gbrcleftcastUbTensor[gbrcEffStart*alignedNActual], gbrcupfloatUbTensor, mActualThisSubBlock * alignedNActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gbrcuphalfUbTensor, gbrcupfloatUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * alignedNActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(gbrcuphalfUbTensor, gbrcuphalfUbTensor, (half)0.0, mActualThisSubBlock * alignedNActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gbrcuphalfUbTensor, gbrcuphalfUbTensor, mActualThisSubBlock * alignedNActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gbrcuphalfUbTensor, gbrcuphalfUbTensor, maskUbTensor[gbrcStart*fullChunkSize], alignedNActual, mActualThisSubBlock,
        {1,1,1, static_cast<uint8_t>(alignedNActual/16), static_cast<uint8_t>(alignedNActual/16), static_cast<uint8_t>(fullChunkSize/16)});
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        if(chunkSize==fullChunkSize) AscendC::DataCopy(outputFPUbTensor, attnInputThisSubBlock, mActualThisSubBlock*nActual);
        else AscendC::DataCopyPad(outputFPUbTensor, attnInputThisSubBlock, aInputUbParams, aInputUbPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::Mul(outputFPUbTensor, outputFPUbTensor, gbrcuphalfUbTensor, mActualThisSubBlock * alignedNActual);

        if constexpr(!std::is_same<AElementOutput_, half>::value) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(gbrcleftcastUbTensor, outputFPUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * alignedNActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputBFUbTensor, gbrcleftcastUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * alignedNActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            if(chunkSize==fullChunkSize) AscendC::DataCopy(maskOutputThisSubBlock, outputBFUbTensor, mActualThisSubBlock*nActual);
            else AscendC::DataCopyPad(maskOutputThisSubBlock, outputBFUbTensor, aOutputUbParams);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            if(chunkSize==fullChunkSize) AscendC::DataCopy(maskOutputThisSubBlock, outputFPUbTensor, mActualThisSubBlock*nActual);
            else AscendC::DataCopyPad(maskOutputThisSubBlock, outputFPUbTensor, aOutputUbParams);
        }
        isFirst = false;
    }

private:
    uint32_t pingpongFlag = 0;
    bool isFirst = true;
    AscendC::LocalTensor<half> maskUbTensor;
    AscendC::LocalTensor<float> gbrcleftcastUbTensor;
    AscendC::LocalTensor<half> gbrcuphalfUbTensor;
    AscendC::LocalTensor<float> gbrcupfloatUbTensor;
    AscendC::LocalTensor<half> ghalfUbTensor;
    AscendC::LocalTensor<uint8_t> shareBuffer_;

    AscendC::LocalTensor<GElementInput_> gUbTensor_ping;
    AscendC::LocalTensor<half> outputFPUbTensor_ping;
    AscendC::LocalTensor<AElementOutput_> outputBFUbTensor_ping;

    AscendC::LocalTensor<GElementInput_> gUbTensor_pong;
    AscendC::LocalTensor<half> outputFPUbTensor_pong;
    AscendC::LocalTensor<AElementOutput_> outputBFUbTensor_pong;
};

// ============================================================================
// GDNFwdOOutputEpilogue: combines V_workspace + H_workspace, applies gate, outputs O
// (moved from catlass/epilogue/block/block_epilogue_gdn_fwdo_output.hpp)
// ============================================================================
template <
    class HElementOutput_,
    class GElementInput_,
    class AElementInput_,
    class HElementInput_
>
class GDNFwdOOutputEpilogue {
public:
    using ArchTag = Arch::AtlasA2;

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
    GDNFwdOOutputEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t BASE = 0;
        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t G_UB_TENSOR_SIZE = 2 * UB_LINE_SIZE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_SIZE = 80 * UB_LINE_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t MASK_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;

        constexpr uint32_t MASK_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t GBRCLEFTCAST_UB_TENSOR_OFFSET = MASK_UB_TENSOR_OFFSET + MASK_UB_TENSOR_SIZE;
        constexpr uint32_t GBRCUP_UB_TENSOR_OFFSET = GBRCLEFTCAST_UB_TENSOR_OFFSET + GBRCLEFTCAST_UB_TENSOR_SIZE;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET = GBRCUP_UB_TENSOR_OFFSET + GBRCUP_UB_TENSOR_SIZE;
        constexpr uint32_t SHARE_TENSOR_OFFSET = G_HALF_UB_TENSOR_OFFSET + G_HALF_UB_TENSOR_SIZE;

        gbrcleftcastUbTensor = resource.ubBuf.template GetBufferByByte<float>(GBRCLEFTCAST_UB_TENSOR_OFFSET);
        gbrcuphalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(GBRCUP_UB_TENSOR_OFFSET);
        ghalfUbTensor = resource.ubBuf.template GetBufferByByte<half>(G_HALF_UB_TENSOR_OFFSET);
        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_TENSOR_OFFSET);

        constexpr uint32_t ATTN_UB_TENSOR_OFFSET = SHARE_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t H_UB_TENSOR_OFFSET = ATTN_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET = H_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + G_UB_TENSOR_SIZE;

        aUbTensor = resource.ubBuf.template GetBufferByByte<AElementInput_>(ATTN_UB_TENSOR_OFFSET);
        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementInput_>(H_UB_TENSOR_OFFSET);
        gUbTensor = resource.ubBuf.template GetBufferByByte<GElementInput_>(G_UB_TENSOR_OFFSET);
        outputFPUbTensor = resource.ubBuf.template GetBufferByByte<half>(OUT_TENSOR_OFFSET);
        outputBFUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput_>(OUT_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~GDNFwdOOutputEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput_> hOutput,
        AscendC::GlobalTensor<GElementInput_> gInput,
        AscendC::GlobalTensor<AElementInput_> attnInput,
        AscendC::GlobalTensor<HElementInput_> hInput,
        float scale,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim)
    {
        uint32_t mCVActual = chunkSize;
        uint32_t nCVActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mCVActualPerSubBlock = CeilDiv(mCVActual, subBlockNum);
        uint32_t mCVActualThisSubBlock = (subBlockIdx == 0) ? mCVActualPerSubBlock : (mCVActual - mCVActualPerSubBlock);
        uint32_t mCVOffset = subBlockIdx * mCVActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetCV = mCVOffset * nCVActual + nOffset;

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0) {
            gbrcStart = 0;
            gbrcRealStart = 0;
            gbrcReptime = (mCVActualThisSubBlock + 8 - 1) / 8;
        } else {
            gbrcStart = mCVActualPerSubBlock;
            gbrcRealStart = gbrcStart & ~15;
            gbrcReptime = (mCVActual - gbrcRealStart + 8 - 1) / 8;
        }
        gbrcEffStart = gbrcStart-gbrcRealStart;
        gbrcEffEnd = gbrcEffStart + mCVActualThisSubBlock;
        uint32_t dstShape_[2] = {gbrcReptime*8, nCVActual};
        uint32_t srcShape_[2] = {gbrcReptime*8, 1};

        AscendC::ResetMask();
        AscendC::GlobalTensor<HElementOutput_> hOutputThisSubBlock = hOutput[offsetCV];
        AscendC::GlobalTensor<AElementInput_> attnInputThisSubBlock = attnInput[offsetCV];
        AscendC::GlobalTensor<HElementInput_> hInputThisSubBlock = hInput[offsetCV];
        AscendC::GlobalTensor<GElementInput_> gInputThisSubBlock = gInput;
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(mCVActualThisSubBlock * nCVActual * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyParams gUbParams{1, (uint16_t)(mCVActual*sizeof(float)), 0, 0};
        AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast(ghalfUbTensor, gUbTensor, AscendC::RoundMode::CAST_NONE, mCVActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(ghalfUbTensor, ghalfUbTensor, mCVActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Broadcast<half, 2, 1>(gbrcuphalfUbTensor, ghalfUbTensor[gbrcRealStart], dstShape_, srcShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mCVActualThisSubBlock * nCVActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Mul(gbrcuphalfUbTensor, hUbTensor, gbrcuphalfUbTensor[gbrcEffStart*nCVActual], mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::DataCopy(aUbTensor, attnInputThisSubBlock, mCVActualThisSubBlock * nCVActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        AscendC::Add(gbrcuphalfUbTensor, aUbTensor, gbrcuphalfUbTensor, mCVActualThisSubBlock * nCVActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(outputFPUbTensor, gbrcuphalfUbTensor, (half)scale, mCVActualThisSubBlock * nCVActual);

        if constexpr(!std::is_same<HElementOutput_, half>::value) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(gbrcleftcastUbTensor, outputFPUbTensor, AscendC::RoundMode::CAST_NONE, mCVActualThisSubBlock * nCVActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputBFUbTensor, gbrcleftcastUbTensor, AscendC::RoundMode::CAST_RINT, mCVActualThisSubBlock * nCVActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(hOutputThisSubBlock, outputBFUbTensor, copyParams);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(hOutputThisSubBlock, outputFPUbTensor, copyParams);
        }
    }

private:
    AscendC::LocalTensor<float> gbrcleftcastUbTensor;
    AscendC::LocalTensor<half> gbrcuphalfUbTensor;
    AscendC::LocalTensor<half> ghalfUbTensor;
    AscendC::LocalTensor<uint8_t> shareBuffer_;

    AscendC::LocalTensor<AElementInput_> aUbTensor;
    AscendC::LocalTensor<HElementInput_> hUbTensor;
    AscendC::LocalTensor<GElementInput_> gUbTensor;
    AscendC::LocalTensor<half> outputFPUbTensor;
    AscendC::LocalTensor<HElementOutput_> outputBFUbTensor;
};

// ============================================================================
// ChunkFwdOVectorProcess: Vector (AIV) process class
// ============================================================================
template <typename ElementType>
class ChunkFwdOVectorProcess {
public:
    using ArchTag = Arch::AtlasA2;

    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    float scale;
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t attnWorkspaceOffset;
    uint32_t aftermaskWorkspaceOffset;
    uint32_t maskWorkspaceOffset;

    AscendC::GlobalTensor<ElementType> gmQ;
    AscendC::GlobalTensor<ElementType> gmK;
    AscendC::GlobalTensor<ElementType> gmV;
    AscendC::GlobalTensor<ElementType> gmH;
    AscendC::GlobalTensor<float> gmG;
    AscendC::GlobalTensor<ElementType> gmO;
    AscendC::GlobalTensor<half> gmVWorkspace;
    AscendC::GlobalTensor<half> gmHWorkspace;
    AscendC::GlobalTensor<half> gmAttnWorkspace;
    AscendC::GlobalTensor<ElementType> gmAftermaskWorkspace;
    AscendC::GlobalTensor<bool> gmMask;

    BlockSchedulerGdnFwdOVec vecBlockScheduler;

    __aicore__ inline ChunkFwdOVectorProcess() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
        GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR tiling, GM_ADDR user) {

        __gm__ ChunkFwdOTilingData *__restrict tilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);

        shapeBatch = tilingData->shapeBatch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        scale = tilingData->scale;
        chunkSize = tilingData->chunkSize;
        vWorkspaceOffset = tilingData->vWorkspaceOffset;
        hWorkspaceOffset = tilingData->hWorkspaceOffset;
        attnWorkspaceOffset = tilingData->attnWorkspaceOffset;
        aftermaskWorkspaceOffset = tilingData->aftermaskWorkspaceOffset;
        maskWorkspaceOffset = tilingData->maskWorkspaceOffset;

        gmQ.SetGlobalBuffer((__gm__ ElementType *)q);
        gmK.SetGlobalBuffer((__gm__ ElementType *)k);
        gmV.SetGlobalBuffer((__gm__ ElementType *)v);
        gmH.SetGlobalBuffer((__gm__ ElementType *)h);
        gmG.SetGlobalBuffer((__gm__ float *)g);
        gmO.SetGlobalBuffer((__gm__ ElementType *)o);
        gmVWorkspace.SetGlobalBuffer((__gm__ half *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ half *)(user + hWorkspaceOffset));
        gmAttnWorkspace.SetGlobalBuffer((__gm__ half *)(user + attnWorkspaceOffset));
        gmAftermaskWorkspace.SetGlobalBuffer((__gm__ ElementType *)(user + aftermaskWorkspaceOffset));
        gmMask.SetGlobalBuffer((__gm__ bool *)(user + maskWorkspaceOffset));

        vecBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
    }

    __aicore__ inline void Process() {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        Arch::Resource<ArchTag> resource;

        AscendC::LocalTensor<half> maskUbTensor = resource.ubBuf.template GetBufferByByte<half>(0);
        for(uint32_t i = 0; i < chunkSize; ++i) {
            for(uint32_t j = 0 ; j < chunkSize; ++j) {
                if(i>=j) maskUbTensor.SetValue(i*chunkSize+j, (half)1.0);
                else maskUbTensor.SetValue(i*chunkSize+j, (half)0.0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        bool needRun = false;

        if (coreIdx < coreNum * subBlockNum) {
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
        }

        while (vecBlockScheduler.isRunning) {
            vecBlockScheduler.InitTask();

            // Vec1: QK mask + gate scaling
            if (vecBlockScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done);
                GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                int64_t vec1OffsetG = vec1Offsets.gOffset;
                int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                GDNFwdOQkmaskEpilogue<ElementType, float, half, bool> epilogueQkmask(resource);
                epilogueQkmask(
                    gmAftermaskWorkspace[vec1OffsetAttnMask],
                    gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                    chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim
                );
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
            }

            AscendC::PipeBarrier<PIPE_ALL>();

            // Vec2: combine output
            if (needRun && coreIdx < coreNum * subBlockNum) {
                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube3Done);
                GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                int64_t vec2OffsetO = vec2Offsets.ovOffset;
                int64_t vec2OffsetG = vec2Offsets.gOffset;
                int64_t vec2OffsetVWork = vec2Offsets.hvWorkOffset;
                int64_t vec2OffsetHWork = vec2Offsets.hvWorkOffset;
                GDNFwdOOutputEpilogue<ElementType, float, half, half> epilogueOutput(resource);
                epilogueOutput(
                    gmO[vec2OffsetO],
                    gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork],
                    scale, vec2Offsets.blockTokens, kHeadDim, vHeadDim
                );
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
            }

            AscendC::PipeBarrier<PIPE_ALL>();

            needRun = true;
        }
    }
};

#endif // CHUNK_FWD_O_VECTOR_H
