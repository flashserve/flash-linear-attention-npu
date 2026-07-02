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
    class VUpdateType_,
    class FinalStateType_,
    class KGatedTag
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHVnew,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_,
    VUpdateType_,
    FinalStateType_,
    KGatedTag
> {
    static constexpr bool kGated = KGatedTag::value;
public:
    using DispatchPolicy = EpilogueAtlasGDNFwdHVnew;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using WSElementInput = typename WSInputType_::Element;
    using VElementUpdate = typename VUpdateType_::Element;
    using VLayoutUpdate = typename VUpdateType_::Layout;
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
        // gLastUbTensor: 1KB at PING_G_SUB_BUF_OFFSET (162KB).
        // Holds g_last (mActual floats, Duplicate output) for g-section, then reused for gk_last
        // (nkActual=128 floats=512B) in kGated section. Capacity OK for K<=256; K>256 requires reallocation.
        // gInputUbTensor shares same offset (temporally safe: used sequentially).
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

        if constexpr (kGated) {
            gkInputUb_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(PING_BUF_2_OFFSET);
            gkInputUb_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_BUF_2_OFFSET);
            gkDecayOutputUb_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);
            gkDecayOutputUb_pong = resource.ubBuf.template GetBufferByByte<VElementOutput>(PONG_BUF_2_OFFSET);
        }

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput> vnewOutput,
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutput,
        AscendC::LocalTensor<VElementUpdate> l1VUpdate,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<UElementInput> uInput,
        AscendC::GlobalTensor<float> wsInput,
        AscendC::GlobalTensor<GElementInput> gkInput,
        AscendC::GlobalTensor<VElementOutput> kInput,
        AscendC::GlobalTensor<VElementOutput> kDecayWorkspace,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done,
        Arch::CrossCoreFlag vec1Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool isPing
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
        int64_t offsetGk = mOffset * nkActual;

        uint32_t gbrcStart, gbrcRealStart, gbrcReptime, gbrcEffStart, gbrcEffEnd;
        if(subBlockIdx==0)
        {
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
        uint32_t dstShape_[2] = {gbrcReptime*8, nvActual};
        uint32_t srcShape_[2] = {gbrcReptime*8, 1};

        AscendC::ResetMask();

        AscendC::GlobalTensor<VElementOutput> vnewOutputThisSubBlock = vnewOutput[offsetK];
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutputThisSubBlock = vnewdecayOutput[offsetK];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<UElementInput> uInputThisSubBlock = uInput[offsetK];
        AscendC::GlobalTensor<float> wsInputThisSubBlock = wsInput[offsetK];

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<UElementInput> uUbTensor = isPing ? uUbTensor_ping : uUbTensor_pong;
        AscendC::LocalTensor<float> wsUbTensor = isPing ? wsUbTensor_ping : wsUbTensor_pong;
        AscendC::LocalTensor<float> gUbTensor = isPing ? gUbTensor_ping : gUbTensor_pong;
        AscendC::LocalTensor<float> gLastUbTensor = isPing ? gLastUbTensor_ping : gLastUbTensor_pong;
        AscendC::LocalTensor<GElementInput> gInputUbTensor = isPing ? gInputUbTensor_ping : gInputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor = isPing ? vNewOutputUbTensor_ping : vNewOutputUbTensor_pong;
        AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor = isPing ? vNewDecayUbTensor_ping : vNewDecayUbTensor_pong;


        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag); // wait v_c2
        AscendC::DataCopy(uUbTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual); // mte2 u
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag); // set u
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag); // wait u
        AscendC::Cast(calcUbTensor, uUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nvActual); // cast u

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag); // wait last g
        if constexpr(std::is_same<GElementInput, float>::value) {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(float)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};
            AscendC::DataCopyPad(gUbTensor, gInputThisSubBlock, gUbParams, gUbPadParams); // copy g
        } else {
            AscendC::DataCopyParams gUbParams{1, (uint16_t)(mActual * sizeof(half)), 0, 0};
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

        AscendC::Duplicate<float>(gLastUbTensor, inputVal, mActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub<float>(gUbTensor, gLastUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreWaitFlag(cube1Done);

        if (storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        }
        AscendC::Sub<float>(wsUbTensor, calcUbTensor, wsUbTensor, mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Broadcast<float, 2, 1>(calcUbTensor, gUbTensor[gbrcRealStart], dstShape_, srcShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);

        AscendC::Mul(calcUbTensor[gbrcEffStart*nvActual], wsUbTensor, calcUbTensor[gbrcEffStart*nvActual], mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t nvLoops = vHeadDim / FLOAT_NUM_PER_REPEAT;
        for (uint32_t nLoop = 0; nLoop < nvLoops; nLoop++) {
            uint32_t castSrcOffset = gbrcEffStart * nvActual + nLoop * FLOAT_NUM_PER_REPEAT;
            uint32_t castDstOffset = nLoop * mActualThisSubBlock * FLOAT_NUM_PER_REPEAT;
            AscendC::Cast(vNewDecayUbTensor[castDstOffset], calcUbTensor[castSrcOffset], AscendC::RoundMode::CAST_RINT, FLOAT_NUM_PER_REPEAT, mActualThisSubBlock, {(uint16_t)mActualThisSubBlock, 1, 1, (uint8_t)(nvLoops * 8)});
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);

        uint32_t ubToL1Loops = nvActual / SIZE_16_NUM_PER_C0;
        uint32_t mActualPadded = (mActual + NZ_BLOCK_SIZE - 1) / NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = ubToL1Loops;
        intriParams.blockLen = mActualThisSubBlock;
        intriParams.srcGap = 0;
        intriParams.dstGap = mActualPadded - mActualThisSubBlock;
        uint32_t l1Addr = mOffset * SIZE_16_NUM_PER_C0;
        AscendC::DataCopy(l1VUpdate[l1Addr], vNewDecayUbTensor, intriParams);

        if constexpr (!kGated) {
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
        AscendC::Cast(vNewOutputUbTensor, wsUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock * nvActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);

        if constexpr (kGated) {
            AscendC::GlobalTensor<GElementInput> gkInputThisSubBlock = gkInput[offsetGk];
            AscendC::GlobalTensor<VElementOutput> kInputThisSubBlock = kInput[offsetGk];
            AscendC::GlobalTensor<VElementOutput> kDecayWorkspaceThisSubBlock = kDecayWorkspace[offsetGk];

            AscendC::LocalTensor<GElementInput> gkInputUb = isPing ? gkInputUb_ping : gkInputUb_pong;
            AscendC::LocalTensor<VElementOutput> gkDecayOutputUb = isPing ? gkDecayOutputUb_ping : gkDecayOutputUb_pong;

            uint32_t gkDataCount = mActualThisSubBlock * nkActual;

            // Phase A: wsUbTensor holds gk float data (MTE2→V→Cast)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag); // reuse wsUB
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag); // reuse uUB
            if constexpr(std::is_same<GElementInput, float>::value) {
                AscendC::DataCopy(wsUbTensor, gkInputThisSubBlock, gkDataCount);
            } else {
                AscendC::DataCopy(gkInputUb, gkInputThisSubBlock, gkDataCount);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            if constexpr(!std::is_same<GElementInput, float>::value) {
                AscendC::Cast(wsUbTensor, gkInputUb, AscendC::RoundMode::CAST_NONE, gkDataCount);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);

            // Phase B: gLastUbTensor holds gk_last [K] (reuse gInputUb for half→float cast)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag); // wait glastUB free
            AscendC::GlobalTensor<GElementInput> gkLastInput = gkInput[(mActual - 1) * nkActual];
            if constexpr(std::is_same<GElementInput, float>::value) {
                AscendC::DataCopy(gLastUbTensor, gkLastInput, nkActual);
            } else {
                AscendC::PipeBarrier<PIPE_MTE2>(); // the same memory
                AscendC::DataCopy(gkInputUb, gkLastInput, nkActual);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            if constexpr(!std::is_same<GElementInput, float>::value) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(gLastUbTensor, gkInputUb, AscendC::RoundMode::CAST_NONE, nkActual);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);

            // Phase C: calcUbTensor holds gk_decay (Broadcast→Sub→Exp)
            //          wsUbTensor still holds gk float (read by Sub, then freed)
            uint32_t gkBrcReptime = (mActualThisSubBlock + 8 - 1) / 8;
            uint32_t dstShapeGk[2] = {gkBrcReptime * 8, nkActual};
            uint32_t srcShapeGk[2] = {1, nkActual};
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Broadcast<float, 2, 0>(calcUbTensor, gLastUbTensor, dstShapeGk, srcShapeGk, shareBuffer_);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub<float>(calcUbTensor, calcUbTensor, wsUbTensor, gkDataCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);

            AscendC::Exp<float>(calcUbTensor, calcUbTensor, gkDataCount);
            AscendC::PipeBarrier<PIPE_V>();
            // wsUbTensor freed after Sub — gk float data consumed

            // Phase D: k_decay = k * gk_decay, output to gkDecayOutputUb
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            if constexpr(std::is_same<VElementOutput, float>::value) {
                AscendC::DataCopy(wsUbTensor, kInputThisSubBlock, gkDataCount);
            } else {
                AscendC::PipeBarrier<PIPE_MTE2>(); // same UB offset as wsUb
                AscendC::DataCopy(gkDecayOutputUb, kInputThisSubBlock, gkDataCount);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + pingpongFlag);
            if constexpr(!std::is_same<VElementOutput, float>::value) {
                AscendC::Cast(wsUbTensor, gkDecayOutputUb, AscendC::RoundMode::CAST_NONE, gkDataCount);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::PipeBarrier<PIPE_V>();

            // k_decay = k_float * gk_decay
            AscendC::Mul<float>(calcUbTensor, wsUbTensor, calcUbTensor, gkDataCount);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(gkDecayOutputUb, calcUbTensor, AscendC::RoundMode::CAST_RINT, gkDataCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
            AscendC::DataCopy(kDecayWorkspaceThisSubBlock, gkDecayOutputUb, gkDataCount);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);
        }

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

    AscendC::LocalTensor<GElementInput> gkInputUb_ping;
    AscendC::LocalTensor<GElementInput> gkInputUb_pong;
    AscendC::LocalTensor<VElementOutput> gkDecayOutputUb_ping;
    AscendC::LocalTensor<VElementOutput> gkDecayOutputUb_pong;

};
}
#endif