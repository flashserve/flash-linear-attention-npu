/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_VNEW_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"



namespace Catlass::Epilogue::Block {

template <
    class VOutputType_,
    class GInputType_,
    class UInputType_,
    class WSInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHVnew,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_,
    FinalStateType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHVnew;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using WSElementInput = typename WSInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t PING_BUF_0_OFFSET = 32 * 1024;
        constexpr uint32_t PING_BUF_1_OFFSET = 48 * 1024;
        constexpr uint32_t PING_BUF_2_OFFSET = 64 * 1024;
        constexpr uint32_t PING_BUF_3_OFFSET = 80 * 1024;
        constexpr uint32_t PONG_BUF_0_OFFSET = 96 * 1024;
        constexpr uint32_t PONG_BUF_1_OFFSET = 112 * 1024;
        constexpr uint32_t PONG_BUF_2_OFFSET = 128 * 1024;
        constexpr uint32_t PONG_BUF_3_OFFSET = 144 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = 161 * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = 162 * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = 163 * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = 164 * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = 165 * 1024;
        constexpr uint32_t SHARE_BUF_OFFSET = 166 * 1024;

        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        uUbTensor_ping = resource.ubBuf.template GetBufferByByte<UElementInput>(PING_BUF_2_OFFSET);
        wsUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        gUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_BUF_OFFSET);
        gLastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_SUB_BUF_OFFSET);
        gInputUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(PING_G_SUB_BUF_OFFSET);
        vNewOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);
        vNewDecayUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);

        uUbTensor_pong = resource.ubBuf.template GetBufferByByte<UElementInput>(PONG_BUF_2_OFFSET);
        wsUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_BUF_OFFSET);
        gLastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_SUB_BUF_OFFSET);
        gInputUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_G_SUB_BUF_OFFSET);
        vNewOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_2_OFFSET);
        vNewDecayUbTensor_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_2_OFFSET);

        shareBuffer_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_BUF_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    template <typename Element>
    CATLASS_DEVICE
    void CopyGmToUb(
        AscendC::LocalTensor<Element> dst,
        AscendC::GlobalTensor<Element> src,
        uint32_t rows,
        uint32_t cols,
        uint32_t srcStride)
    {
        if (cols == srcStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)),
            static_cast<uint32_t>((srcStride - cols) * sizeof(Element)),
            0,
            0};
        AscendC::DataCopyPadExtParams<Element> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename Element>
    CATLASS_DEVICE
    void CopyUbToGm(
        AscendC::GlobalTensor<Element> dst,
        AscendC::LocalTensor<Element> src,
        uint32_t rows,
        uint32_t cols,
        uint32_t dstStride)
    {
        if (cols == dstStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(Element)),
            0,
            static_cast<uint32_t>((dstStride - cols) * sizeof(Element)),
            0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput> vnewOutput,
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<UElementInput> uInput,
        AscendC::GlobalTensor<float> wsInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vBlockDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done,
        Arch::CrossCoreFlag vec1Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool isPing
    )
    {
        static constexpr uint32_t ROW_TILE = 16;
        uint32_t mActual = chunkSize;
        uint32_t nvActual = vBlockDim;
        uint32_t inputStride = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = rowBegin + rowsPerSubBlock;
        if (rowEnd > mActual) {
            rowEnd = mActual;
        }
        if (rowBegin >= mActual) {
            return;
        }

        AscendC::ResetMask();

        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<UElementInput> uUbTensor = isPing ? uUbTensor_ping : uUbTensor_pong;
        AscendC::LocalTensor<float> wsUbTensor = isPing ? wsUbTensor_ping : wsUbTensor_pong;
        AscendC::LocalTensor<float> gUbTensor = isPing ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<float> gLastUbTensor = isPing ? gLastUbTensor_ping : gLastUbTensor_pong;
        AscendC::LocalTensor<GElementInput> gInputUbTensor = isPing ? gInputUbTensor_ping : gInputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor = isPing ? vNewOutputUbTensor_ping : vNewOutputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor = isPing ? vNewDecayUbTensor_ping : vNewDecayUbTensor_pong;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);
        if constexpr(std::is_same<GElementInput, float>::value) {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        } else {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(GElementInput)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gInputUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + pingpongFlag);
        if constexpr(!std::is_same<GElementInput, float>::value) {
            AscendC::Cast(gUbTensor, gInputUbTensor, AscendC::RoundMode::CAST_NONE, mActual);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        float inputVal = gUbTensor.GetValue(mActual-1);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<float>(gLastUbTensor, inputVal, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub<float>(gUbTensor, gLastUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube1Done);

        bool waitWsFromMte3 = storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value;
        for (uint32_t rowStart = rowBegin; rowStart < rowEnd; rowStart += ROW_TILE) {
            uint32_t rowsThisTile = rowEnd - rowStart;
            if (rowsThisTile > ROW_TILE) {
                rowsThisTile = ROW_TILE;
            }

            AscendC::GlobalTensor<VElementOutput> vnewOutputThisTile = vnewOutput[rowStart * inputStride];
            AscendC::GlobalTensor<VElementOutput> vnewdecayOutputThisTile = vnewdecayOutput[rowStart * nvActual];
            AscendC::GlobalTensor<UElementInput> uInputThisTile = uInput[rowStart * inputStride];
            AscendC::GlobalTensor<float> wsInputThisTile = wsInput[rowStart * nvActual];

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
            CopyGmToUb(uUbTensor, uInputThisTile, rowsThisTile, nvActual, inputStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::Cast(calcUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();

            if (waitWsFromMte3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            }
            AscendC::DataCopy(wsUbTensor, wsInputThisTile, rowsThisTile * nvActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            waitWsFromMte3 = false;

            AscendC::Sub<float>(wsUbTensor, calcUbTensor, wsUbTensor, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t row = 0; row < rowsThisTile; ++row) {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
                float gScale = gUbTensor.GetValue(rowStart + row);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
                AscendC::Muls(calcUbTensor[row * nvActual], wsUbTensor[row * nvActual], gScale, nvActual);
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(vNewDecayUbTensor, calcUbTensor, AscendC::RoundMode::CAST_RINT, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::DataCopy(vnewdecayOutputThisTile, vNewDecayUbTensor, rowsThisTile * nvActual);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
            AscendC::Cast(vNewOutputUbTensor, wsUbTensor, AscendC::RoundMode::CAST_RINT, rowsThisTile * nvActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            CopyUbToGm(vnewOutputThisTile, vNewOutputUbTensor, rowsThisTile, nvActual, inputStride);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);

    }

private:
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<UElementInput> uUbTensor_ping;
    AscendC::LocalTensor<float> wsUbTensor_ping;
    AscendC::LocalTensor<float> gUbTensor_ping;
    AscendC::LocalTensor<float> gLastUbTensor_ping;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_ping;

    AscendC::LocalTensor<UElementInput> uUbTensor_pong;
    AscendC::LocalTensor<float> wsUbTensor_pong;
    AscendC::LocalTensor<float> gUbTensor_pong;
    AscendC::LocalTensor<float> gLastUbTensor_pong;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_pong;

    AscendC::LocalTensor<uint8_t> shareBuffer_;

};
}

#endif
