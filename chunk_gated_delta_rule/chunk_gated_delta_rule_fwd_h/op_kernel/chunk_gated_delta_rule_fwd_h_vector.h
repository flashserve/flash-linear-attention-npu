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
 * \file chunk_gated_delta_rule_fwd_h_vector.h
 * \brief Vector (AIV) operations for chunk_gated_delta_rule_fwd_h, including epilogue implementations
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_VECTOR_H

#include "chunk_gated_delta_rule_fwd_h_common.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

using namespace Catlass;

// ============================================================================
// GDNFwdHVnewEpilogue: v_new = u - ws, v_new_decay = v_new * gate_decay
// (moved from catlass/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp)
// ============================================================================
template <
    class VElementOutput_,
    class GElementInput_,
    class UElementInput_,
    class WSElementInput_
>
class GDNFwdHVnewEpilogue {
public:
    using ArchTag = Arch::AtlasA2;

    static constexpr uint32_t HALF_ELENUM_PER_BLK = 16;
    static constexpr uint32_t FLOAT_ELENUM_PER_BLK = 8;
    static constexpr uint32_t HALF_ELENUM_PER_VECCALC = 128;
    static constexpr uint32_t FLOAT_ELENUM_PER_VECCALC = 64;
    static constexpr uint32_t UB_TILE_SIZE = 16384;  // 64 * 128 * 2B
    static constexpr uint32_t UB_LINE_SIZE = 512;   // 128 * 2 * 2B
    static constexpr uint32_t HALF_ELENUM_PER_LINE = 256;    // 128 * 2
    static constexpr uint32_t FLOAT_ELENUM_PER_LINE = 128;   // 128
    static constexpr uint32_t MULTIPLIER = 2;

    CATLASS_DEVICE
    GDNFwdHVnewEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t BASE = 0;

        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;
        constexpr uint32_t GSUB_UB_TENSOR_SIZE = 1 * UB_LINE_SIZE;
        constexpr uint32_t SHARE_TENSOR_SIZE = 1 * UB_LINE_SIZE;

        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t HALF_UB_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t U_UB_TENSOR_OFFSET = HALF_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t WS_UB_TENSOR_OFFSET = U_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET = WS_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_OUTPUT_UB_TENSOR_OFFSET = G_UB_TENSOR_OFFSET + GSUB_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_DECAY_UB_TENSOR_OFFSET = VNEW_OUTPUT_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;

        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
        halfUbTensor = resource.ubBuf.template GetBufferByByte<half>(HALF_UB_TENSOR_OFFSET);

        uUbTensor_ping = resource.ubBuf.template GetBufferByByte<UElementInput_>(U_UB_TENSOR_OFFSET);
        uUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(U_UB_TENSOR_OFFSET);
        wsUbTensor_ping = resource.ubBuf.template GetBufferByByte<half>(WS_UB_TENSOR_OFFSET);
        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput_>(G_UB_TENSOR_OFFSET);
        gUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(G_UB_TENSOR_OFFSET);
        vNewOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput_>(VNEW_OUTPUT_UB_TENSOR_OFFSET);
        vNewOutputUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(VNEW_OUTPUT_UB_TENSOR_OFFSET);
        vNewDecayUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput_>(VNEW_DECAY_UB_TENSOR_OFFSET);
        vNewDecayUbHalfTensor_ping = resource.ubBuf.template GetBufferByByte<half>(VNEW_DECAY_UB_TENSOR_OFFSET);

        constexpr uint32_t U_UB_TENSOR_OFFSET_pong = VNEW_DECAY_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t WS_UB_TENSOR_OFFSET_pong = U_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_UB_TENSOR_OFFSET_pong = WS_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_OUTPUT_UB_TENSOR_OFFSET_pong = G_UB_TENSOR_OFFSET_pong + GSUB_UB_TENSOR_SIZE;
        constexpr uint32_t VNEW_DECAY_UB_TENSOR_OFFSET_pong = VNEW_OUTPUT_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;

        constexpr uint32_t SHARE_TENSOR_OFFSET = VNEW_DECAY_UB_TENSOR_OFFSET_pong + HALF_UB_TENSOR_SIZE;

        uUbTensor_pong = resource.ubBuf.template GetBufferByByte<UElementInput_>(U_UB_TENSOR_OFFSET_pong);
        uUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(U_UB_TENSOR_OFFSET_pong);
        wsUbTensor_pong = resource.ubBuf.template GetBufferByByte<half>(WS_UB_TENSOR_OFFSET_pong);
        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput_>(G_UB_TENSOR_OFFSET_pong);
        gUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(G_UB_TENSOR_OFFSET_pong);
        vNewOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput_>(VNEW_OUTPUT_UB_TENSOR_OFFSET_pong);
        vNewOutputUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(VNEW_OUTPUT_UB_TENSOR_OFFSET_pong);
        vNewDecayUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput_>(VNEW_DECAY_UB_TENSOR_OFFSET_pong);
        vNewDecayUbHalfTensor_pong = resource.ubBuf.template GetBufferByByte<half>(VNEW_DECAY_UB_TENSOR_OFFSET_pong);

        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~GDNFwdHVnewEpilogue() {}

    float taylor_exp(float x, int n) {
        float sum = 1.0;
        float term = 1.0;
        for (int i = 1; i <= n; i++) {
            term = term * x / i;
            sum += term;
        }
        return sum;
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput_> vnewOutput,
        AscendC::GlobalTensor<VElementOutput_> vnewdecayOutput,
        AscendC::GlobalTensor<float> gInput,
        AscendC::GlobalTensor<UElementInput_> uInput,
        AscendC::GlobalTensor<WSElementInput_> wsInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nkActual = kHeadDim;
        uint32_t nvActual = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetK = mOffset * nvActual + nOffset;
        int64_t offsetD = 0;

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0)
        {
            gbrcStart = 0;
            gbrcRealStart = 0;
            gbrcReptime = (mActualThisSubBlock + 8 - 1) / 8;
        }
        else
        {
            gbrcStart = mActualPerSubBlock;
            gbrcRealStart = gbrcStart & ~15;
            gbrcReptime = (mActual - gbrcRealStart + 8 - 1) / 8;
        }
        gbrcEffStart = gbrcStart-gbrcRealStart;
        gbrcEffEnd = gbrcEffStart + mActualThisSubBlock;

        AscendC::ResetMask();

        AscendC::GlobalTensor<VElementOutput_> vnewOutputThisSubBlock = vnewOutput[offsetK];
        AscendC::GlobalTensor<VElementOutput_> vnewdecayOutputThisSubBlock = vnewdecayOutput[offsetK];
        AscendC::GlobalTensor<float> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<UElementInput_> uInputThisSubBlock = uInput[offsetK];
        AscendC::GlobalTensor<WSElementInput_> wsInputThisSubBlock = wsInput[offsetK];
        AscendC::LocalTensor<half> vNewUbHalfTensor;

        pingpongFlag = isFirst ? 0 : 4;
        AscendC::LocalTensor<UElementInput_> uUbTensor = pingpongFlag == 0 ? uUbTensor_ping : uUbTensor_pong;
        AscendC::LocalTensor<half> uUbHalfTensor = pingpongFlag == 0 ? uUbHalfTensor_ping : uUbHalfTensor_pong;
        AscendC::LocalTensor<half> wsUbTensor = pingpongFlag == 0 ? wsUbTensor_ping : wsUbTensor_pong;
        AscendC::LocalTensor<float> gUbTensor = pingpongFlag == 0 ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<half> gUbHalfTensor = pingpongFlag == 0 ? gUbHalfTensor_ping : gUbHalfTensor_pong;
        AscendC::LocalTensor<VElementOutput_> vNewOutputUbTensor = pingpongFlag == 0 ? vNewOutputUbTensor_ping : vNewOutputUbTensor_pong;
        AscendC::LocalTensor<half> vNewOutputUbHalfTensor = pingpongFlag == 0 ? vNewOutputUbHalfTensor_ping : vNewOutputUbHalfTensor_pong;
        AscendC::LocalTensor<VElementOutput_> vNewDecayUbTensor = pingpongFlag == 0 ? vNewDecayUbTensor_ping : vNewDecayUbTensor_pong;
        AscendC::LocalTensor<half> vNewDecayUbHalfTensor = pingpongFlag == 0 ? vNewDecayUbHalfTensor_ping : vNewDecayUbHalfTensor_pong;

        AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
        AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2 + pingpongFlag);
        float inputVal = gUbTensor.GetValue(mActual-1);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID2 + pingpongFlag);

        AscendC::Duplicate<float>(floatUbTensor, inputVal, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub<float>(gUbTensor, floatUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gUbHalfTensor, gUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t dstShape_[2] = {gbrcReptime*8, nvActual};
        uint32_t srcShape_[2] = {gbrcReptime*8, 1};
        AscendC::Broadcast<half, 2, 1>(halfUbTensor, gUbHalfTensor[gbrcRealStart], dstShape_, srcShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube1Done);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        if constexpr(!std::is_same<UElementInput_, half>::value) {
            AscendC::DataCopy(uUbTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::Cast(floatUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(uUbHalfTensor, floatUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(uUbHalfTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        AscendC::DataCopy(wsUbTensor, wsInputThisSubBlock, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);

        if constexpr(!std::is_same<VElementOutput_, half>::value) {
            AscendC::Sub<half>(uUbHalfTensor, uUbHalfTensor, wsUbTensor, mActualThisSubBlock * nvActual);
            vNewUbHalfTensor = uUbHalfTensor;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(floatUbTensor, uUbHalfTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vNewOutputUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock * nvActual);
        } else {
            AscendC::Sub<half>(vNewOutputUbHalfTensor, uUbHalfTensor, wsUbTensor, mActualThisSubBlock * nvActual);
            vNewUbHalfTensor = vNewOutputUbHalfTensor;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbHalfTensor, mActualThisSubBlock * nvActual);
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(vNewDecayUbHalfTensor, vNewUbHalfTensor, halfUbTensor[gbrcEffStart*nvActual], mActualThisSubBlock * nvActual);

        if constexpr(!std::is_same<VElementOutput_, half>::value) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(floatUbTensor, vNewDecayUbHalfTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(vNewDecayUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewdecayOutputThisSubBlock, vNewDecayUbTensor, mActualThisSubBlock * nvActual);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(vnewdecayOutputThisSubBlock, vNewDecayUbHalfTensor, mActualThisSubBlock * nvActual);
        }

        isFirst = false;
    }

private:
    uint32_t pingpongFlag = 0;
    bool isFirst = true;

    AscendC::LocalTensor<float> floatUbTensor;
    AscendC::LocalTensor<half> halfUbTensor;

    AscendC::LocalTensor<UElementInput_> uUbTensor_ping;
    AscendC::LocalTensor<half> uUbHalfTensor_ping;
    AscendC::LocalTensor<half> wsUbTensor_ping;
    AscendC::LocalTensor<float> gUbTensor_ping;
    AscendC::LocalTensor<half> gUbHalfTensor_ping;
    AscendC::LocalTensor<VElementOutput_> vNewOutputUbTensor_ping;
    AscendC::LocalTensor<half> vNewOutputUbHalfTensor_ping;
    AscendC::LocalTensor<VElementOutput_> vNewDecayUbTensor_ping;
    AscendC::LocalTensor<half> vNewDecayUbHalfTensor_ping;

    AscendC::LocalTensor<UElementInput_> uUbTensor_pong;
    AscendC::LocalTensor<half> uUbHalfTensor_pong;
    AscendC::LocalTensor<half> wsUbTensor_pong;
    AscendC::LocalTensor<float> gUbTensor_pong;
    AscendC::LocalTensor<half> gUbHalfTensor_pong;
    AscendC::LocalTensor<VElementOutput_> vNewOutputUbTensor_pong;
    AscendC::LocalTensor<half> vNewOutputUbHalfTensor_pong;
    AscendC::LocalTensor<VElementOutput_> vNewDecayUbTensor_pong;
    AscendC::LocalTensor<half> vNewDecayUbHalfTensor_pong;

    AscendC::LocalTensor<uint8_t> shareBuffer_;
};

// ============================================================================
// GDNFwdHUpdateEpilogue: h[i+1] = h[i] * exp(g_last) + h_update
// (moved from catlass/epilogue/block/block_epilogue_gdn_fwdh_update.hpp)
// ============================================================================
template <
    class HElementOutput_,
    class GElementInput_,
    class HElementInput_,
    class HUpdateElementInput_
>
class GDNFwdHUpdateEpilogue {
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
    GDNFwdHUpdateEpilogue(Arch::Resource<ArchTag> &resource)
    {
        constexpr uint32_t FLOAT_UB_TENSOR_SIZE = 64 * UB_LINE_SIZE;
        constexpr uint32_t HALF_UB_TENSOR_SIZE = 32 * UB_LINE_SIZE;

        constexpr uint32_t BASE = 0;
        constexpr uint32_t FLOAT_UB_TENSOR_OFFSET = BASE;
        constexpr uint32_t HALF_UB_TENSOR_OFFSET = FLOAT_UB_TENSOR_OFFSET + FLOAT_UB_TENSOR_SIZE;

        constexpr uint32_t H_TENSOR_OFFSET = HALF_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t HUPDATE_UB_TENSOR_OFFSET = H_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t HOUTPUT_UB_TENSOR_OFFSET = HUPDATE_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;
        constexpr uint32_t G_TENSOR_OFFSET = HOUTPUT_UB_TENSOR_OFFSET + HALF_UB_TENSOR_SIZE;

        floatUbTensor = resource.ubBuf.template GetBufferByByte<float>(FLOAT_UB_TENSOR_OFFSET);
        halfUbTensor = resource.ubBuf.template GetBufferByByte<half>(HALF_UB_TENSOR_OFFSET);

        hUbTensor = resource.ubBuf.template GetBufferByByte<HElementInput_>(H_TENSOR_OFFSET);
        hUbHalfTensor = resource.ubBuf.template GetBufferByByte<half>(H_TENSOR_OFFSET);
        hUpdateUbHalfTensor = resource.ubBuf.template GetBufferByByte<half>(HUPDATE_UB_TENSOR_OFFSET);

        hOutputUbTensor = resource.ubBuf.template GetBufferByByte<HElementOutput_>(HOUTPUT_UB_TENSOR_OFFSET);
        hOutputUbHalfTensor = resource.ubBuf.template GetBufferByByte<half>(HOUTPUT_UB_TENSOR_OFFSET);

        glastUbTensor = resource.ubBuf.template GetBufferByByte<half>(G_TENSOR_OFFSET);
    }

    CATLASS_DEVICE
    ~GDNFwdHUpdateEpilogue() {}

    float taylor_exp(float x, int n) {
        float sum = 1.0;
        float term = 1.0;
        for (int i = 1; i <= n; i++) {
            term = term * x / i;
            sum += term;
        }
        return sum;
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput_> hOutput,
        AscendC::GlobalTensor<float> gInput,
        AscendC::GlobalTensor<HElementInput_> hInput,
        AscendC::GlobalTensor<HUpdateElementInput_> hUpdateInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done
    )
    {
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetH = mOffset * nActual + nOffset;

        AscendC::ResetMask();

        AscendC::GlobalTensor<HElementOutput_> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<float> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput_> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<HUpdateElementInput_> hUpdateInputThisSubBlock = hUpdateInput[offsetH];

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        if constexpr(!std::is_same<HElementInput_, half>::value) {
            AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(floatUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(hUbHalfTensor, floatUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(hUbHalfTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        glastUbTensor.SetValue(0, (half)gInputThisSubBlock.GetValue(chunkSize-1));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Exp(glastUbTensor, glastUbTensor, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        half muls = glastUbTensor.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Muls(hUbHalfTensor, hUbHalfTensor, muls, mActualThisSubBlock * nActual);

        Arch::CrossCoreWaitFlag(cube2Done);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::DataCopy(hUpdateUbHalfTensor, hUpdateInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Add<half>(hOutputUbHalfTensor, hUbHalfTensor, hUpdateUbHalfTensor, mActualThisSubBlock * nActual);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

        if constexpr(!std::is_same<HElementOutput_, half>::value) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(floatUbTensor, hOutputUbHalfTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(hOutputUbTensor, floatUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(hOutputThisSubBlock, hOutputUbTensor, mActualThisSubBlock * nActual);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(hOutputThisSubBlock, hOutputUbHalfTensor, mActualThisSubBlock * nActual);
        }
    }

private:
    AscendC::LocalTensor<float> floatUbTensor;
    AscendC::LocalTensor<half> halfUbTensor;

    AscendC::LocalTensor<HElementInput_> hUbTensor;
    AscendC::LocalTensor<half> hUbHalfTensor;
    AscendC::LocalTensor<half> hUpdateUbHalfTensor;

    AscendC::LocalTensor<HElementOutput_> hOutputUbTensor;
    AscendC::LocalTensor<half> hOutputUbHalfTensor;

    AscendC::LocalTensor<half> glastUbTensor;
};

// ============================================================================
// ChunkGatedDeltaRuleFwdHVectorProcess: Vector (AIV) main process
// ============================================================================
template <typename ElementType>
class ChunkGatedDeltaRuleFwdHVectorProcess {
public:
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementVWork = half;
    using ElementHWork = half;

    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;

    AscendC::GlobalTensor<ElementType> gmU;
    AscendC::GlobalTensor<float> gmG;
    AscendC::GlobalTensor<ElementType> gmH;
    AscendC::GlobalTensor<ElementType> gmV;
    AscendC::GlobalTensor<ElementType> gmVWorkspace;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspaceHalf;
    AscendC::GlobalTensor<ElementType> gmHWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspaceHalf;

    BlockSchedulerGdnFwdHVec vecBlockScheduler;

    __aicore__ inline ChunkGatedDeltaRuleFwdHVectorProcess() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state,
        GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state,
        GM_ADDR tiling, GM_ADDR user) {

        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict tilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = tilingData->batch;
        seqlen = tilingData->seqlen;
        kNumHead = tilingData->kNumHead;
        vNumHead = tilingData->vNumHead;
        kHeadDim = tilingData->kHeadDim;
        vHeadDim = tilingData->vHeadDim;
        chunkSize = tilingData->chunkSize;
        useInitialState = tilingData->useInitialState;
        storeFinalState = tilingData->storeFinalState;
        isVariedLen = tilingData->isVariedLen;
        shapeBatch = tilingData->shapeBatch;
        tokenBatch = tilingData->tokenBatch;
        vWorkspaceOffset = tilingData->vWorkspaceOffset;
        hWorkspaceOffset = tilingData->hWorkspaceOffset;

        gmU.SetGlobalBuffer((__gm__ ElementType *)u);
        gmG.SetGlobalBuffer((__gm__ float *)g);
        gmH.SetGlobalBuffer((__gm__ ElementType *)h);
        gmV.SetGlobalBuffer((__gm__ ElementType *)v_new);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementType *)(user + vWorkspaceOffset));
        gmVWorkspaceHalf.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementType *)(user + hWorkspaceOffset));
        gmHWorkspaceHalf.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));

        vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling);
    }

    __aicore__ inline void Process() {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();

        Arch::Resource<ArchTag> resource;

        using VType = Catlass::Gemm::GemmType<ElementType, Catlass::layout::RowMajor>;
        using GType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
        using UType = Catlass::Gemm::GemmType<ElementType, Catlass::layout::RowMajor>;
        using VworkType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
        using HType = Catlass::Gemm::GemmType<ElementType, Catlass::layout::RowMajor>;
        using HworkType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;

        GDNFwdHVnewEpilogue<ElementType, float, ElementType, half> epilogueGDNFwdHVnew(resource);
        bool needRun = false;

        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
        while (vecBlockScheduler.isRunning) {
            vecBlockScheduler.InitTask();
            // step 2:
            GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
            if (!vec1Offsets.isDummyHead) {
                epilogueGDNFwdHVnew(
                    gmV[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset],
                    gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspaceHalf[vec1Offsets.vWorkOffset],
                    vec1Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube1Done
                );
            } else {
                Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done);
            }
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);

            GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
            if (!vec2Offsets.isFinalState && needRun) {
                // step 4: h[i+1] += h_work if i < num_chunks - 1 else None
                if (!vec2Offsets.isDummyHead) {
                    GDNFwdHUpdateEpilogue<ElementType, float, ElementType, half> epilogueGDNFwdHUpdate(resource);
                    epilogueGDNFwdHUpdate(
                        gmH[vec2Offsets.hDstOffset],
                        gmG[vec2Offsets.gOffset],
                        gmH[vec2Offsets.hSrcOffset],
                        gmHWorkspaceHalf[vec2Offsets.hWorkOffset],
                        vec2Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube2Done
                    );
                } else {
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                }
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
            }
            needRun = true;
        }
    }
};

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_VECTOR_H
