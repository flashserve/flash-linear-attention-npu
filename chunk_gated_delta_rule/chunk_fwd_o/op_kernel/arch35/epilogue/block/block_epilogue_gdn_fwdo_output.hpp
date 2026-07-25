/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDO_OUTPUT_HPP

#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_o_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

static __simd_vf__ inline void GdnFwdoOutputMulAddScaleVf(
    __ubuf__ float* dst,
    __ubuf__ float* attn,
    __ubuf__ float* hidden,
    __ubuf__ float* gate,
    float scale,
    uint32_t elementCount)
{
    AscendC::Reg::RegTensor<float> attnReg0;
    AscendC::Reg::RegTensor<float> attnReg1;
    AscendC::Reg::RegTensor<float> hiddenReg0;
    AscendC::Reg::RegTensor<float> hiddenReg1;
    AscendC::Reg::RegTensor<float> gateReg0;
    AscendC::Reg::RegTensor<float> gateReg1;
    AscendC::Reg::RegTensor<float> dstReg0;
    AscendC::Reg::RegTensor<float> dstReg1;
    constexpr uint32_t ELEMS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t ELEMS_PER_DUAL = ELEMS_PER_REG * 2;
    auto maskFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    uint16_t dualRepeats = static_cast<uint16_t>(elementCount / ELEMS_PER_DUAL);

    for (uint16_t i = 0; i < dualRepeats; ++i) {
        uint32_t offset = static_cast<uint32_t>(i) * ELEMS_PER_DUAL;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(attnReg0, attnReg1, attn + offset);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(hiddenReg0, hiddenReg1, hidden + offset);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(gateReg0, gateReg1, gate + offset);
        AscendC::Reg::Mul(dstReg0, hiddenReg0, gateReg0, maskFull);
        AscendC::Reg::Mul(dstReg1, hiddenReg1, gateReg1, maskFull);
        AscendC::Reg::Add(dstReg0, attnReg0, dstReg0, maskFull);
        AscendC::Reg::Add(dstReg1, attnReg1, dstReg1, maskFull);
        AscendC::Reg::Muls(dstReg0, dstReg0, scale, maskFull);
        AscendC::Reg::Muls(dstReg1, dstReg1, scale, maskFull);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_INTLV_B32>(
            dst + offset, dstReg0, dstReg1, maskFull);
    }

}

template <
    class HOutputType_,
    class GInputType_,
    class AInputType_,
    class HInputType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdOOutput,
    HOutputType_,
    GInputType_,
    AInputType_,
    HInputType_
> {
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdOOutput;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using AElementInput = typename AInputType_::Element;
    using HElementInput = typename HInputType_::Element;

    // using CopyGmToUbInput = Tile::CopyGm2Ub<ArchTag, InputType_>;
    // using CopyUbToGmOutput = Tile::CopyUb2Gm<ArchTag, OutputType_>;

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
        constexpr uint32_t UB_WORK_SLOT_BYTES = 64 * 128 * sizeof(float);

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
        constexpr uint32_t A_UB_TENSOR_OFFSET_PING = 71 * 1024;
        constexpr uint32_t H_UB_TENSOR_OFFSET_PING = A_UB_TENSOR_OFFSET_PING + UB_WORK_SLOT_BYTES;
        constexpr uint32_t A_UB_TENSOR_OFFSET_PONG = H_UB_TENSOR_OFFSET_PING + UB_WORK_SLOT_BYTES;
        constexpr uint32_t H_UB_TENSOR_OFFSET_PONG = A_UB_TENSOR_OFFSET_PONG + UB_WORK_SLOT_BYTES;
        constexpr uint32_t OUT_UB_TENSOR_OFFSET_PING = H_UB_TENSOR_OFFSET_PONG + UB_WORK_SLOT_BYTES;
        constexpr uint32_t OUT_HALF_UB_TENSOR_OFFSET_PING = OUT_UB_TENSOR_OFFSET_PING + FLOAT_UB_TENSOR_SIZE;

        gUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(G_UB_TENSOR_OFFSET_PING);
        gUbFPTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PING);
        gUbBFTensorPing = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PING);
        aUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(A_UB_TENSOR_OFFSET_PING);
        hUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(H_UB_TENSOR_OFFSET_PING);
        outUbTensorPing = resource.ubBuf.template GetBufferByByte<float>(OUT_UB_TENSOR_OFFSET_PING);
        outUbFPTensorPing = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PING);
        outUbBFTensorPing = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PING);

        constexpr uint32_t G_UB_TENSOR_OFFSET_PONG = G_UB_TENSOR_OFFSET_PING;
        constexpr uint32_t G_HALF_UB_TENSOR_OFFSET_PONG = G_UB_TENSOR_OFFSET_PONG + G_FLOAT_UB_TENSOR_SIZE;
        constexpr uint32_t OUT_UB_TENSOR_OFFSET_PONG = OUT_UB_TENSOR_OFFSET_PING;
        constexpr uint32_t OUT_HALF_UB_TENSOR_OFFSET_PONG = OUT_UB_TENSOR_OFFSET_PONG + FLOAT_UB_TENSOR_SIZE;

        gUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(G_UB_TENSOR_OFFSET_PONG);
        gUbFPTensorPong = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PONG);
        gUbBFTensorPong = resource.ubBuf.template GetBufferByByte<GElementInput>(G_HALF_UB_TENSOR_OFFSET_PONG);
        aUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(A_UB_TENSOR_OFFSET_PONG);
        hUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(H_UB_TENSOR_OFFSET_PONG);
        outUbTensorPong = resource.ubBuf.template GetBufferByByte<float>(OUT_UB_TENSOR_OFFSET_PONG);
        outUbFPTensorPong = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PONG);
        outUbBFTensorPong = resource.ubBuf.template GetBufferByByte<HElementOutput>(OUT_HALF_UB_TENSOR_OFFSET_PONG);
    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::LocalTensor<AElementInput> attnInput,
        AscendC::LocalTensor<HElementInput> hInput,
        float scale,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        uint32_t &pingpongFlag,
        bool &mte3Pending,
        uint32_t &mte3PendingEvent
        , uint32_t batchIdx, uint32_t headIdx, uint32_t chunkIdx
        )
    {
        uint32_t mActual = chunkSize;
        uint32_t nActual = vHeadDim;
        uint32_t alignedM = CeilDiv(nActual, 8) * 8;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        int64_t offsetA = mOffset * nActual + nOffset;

        uint32_t gbrcStart, gbrcRealStart, gbrcRealEnd, gbrcRealProcess, gbrcEffStart, gbrcEffEnd, mulsRemain, mulsRemainIdx;
        if(mActualThisSubBlock <= 32)
        {
            if(subBlockIdx == 0)
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
            uint32_t dstShape_[2] = {gbrcRealProcess, nActual};
            uint32_t srcShape_[2] = {gbrcRealProcess, 1};

            AscendC::ResetMask();
            AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
            AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[gbrcStart * nActual];

            AscendC::DataCopyParams gfloatUbParams{1, (uint16_t)(mActual*sizeof(float)), 0, 0};
            AscendC::DataCopyParams ghalfUbParams{1, (uint16_t)(mActual*sizeof(half)), 0, 0};
            AscendC::DataCopyPadParams gUbPadParams{false, 0, 0, 0};

            AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
            AscendC::LocalTensor<float> hUbTensor = (pingpongFlag == 0) ? hUbTensorPing : hUbTensorPong;
            AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
            AscendC::LocalTensor<HElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
            AscendC::LocalTensor<HElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;
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


            AscendC::Exp(gcompUbTensor, gcompUbTensor, mActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gcompUbTensor[gbrcRealStart], dstShape_, srcShape_, shareUbTensor);
            AscendC::PipeBarrier<PIPE_V>();

            __ubuf__ float* outAddr = (__ubuf__ float*)outUbTensor.GetPhyAddr();
            __ubuf__ float* attnAddr = (__ubuf__ float*)aUbTensor.GetPhyAddr();
            __ubuf__ float* hiddenAddr = (__ubuf__ float*)hUbTensor.GetPhyAddr();
            __ubuf__ float* gateAddr =
                (__ubuf__ float*)gbrcLeftcastUbTensor[gbrcEffStart * nActual].GetPhyAddr();
            constexpr uint32_t VF_DUAL_ELEMENTS = 2 * AscendC::VECTOR_REG_WIDTH / sizeof(float);
            uint32_t totalElements = mActualThisSubBlock * nActual;
            uint32_t vfElements = totalElements / VF_DUAL_ELEMENTS * VF_DUAL_ELEMENTS;
            if (vfElements != 0) {
                AscendC::VF_CALL<GdnFwdoOutputMulAddScaleVf>(
                    outAddr, attnAddr, hiddenAddr, gateAddr, (float)scale, vfElements);
            }
            uint32_t tailElements = totalElements - vfElements;
            if (tailElements != 0) {
                AscendC::Mul(outUbTensor[vfElements], hUbTensor[vfElements],
                    gbrcLeftcastUbTensor[gbrcEffStart * nActual + vfElements], tailElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(outUbTensor[vfElements], aUbTensor[vfElements],
                    outUbTensor[vfElements], tailElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(outUbTensor[vfElements], outUbTensor[vfElements], (float)scale, tailElements);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if(std::is_same<HElementOutput, half>::value)
            {
                AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::DataCopy(hOutputThisSubBlock, outUbFPTensor, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
                mte3Pending = true;
                mte3PendingEvent = EVENT_ID0 + pingpongFlag;
            }
            else
            {
                AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::DataCopy(hOutputThisSubBlock, outUbBFTensor, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
                mte3Pending = true;
                mte3PendingEvent = EVENT_ID0 + pingpongFlag;
            }
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
            AscendC::Exp(gcompUbTensor, gcompUbTensor, mActual);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t mActualPerStage = CeilDiv(mActualThisSubBlock, 2);
            uint32_t mActualThisStage = 0;
            for(uint32_t stage = 0; stage < 2; stage++)
            {
                if(stage == 0) mActualThisStage = mActualPerStage;
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
                uint32_t ubStageOffset = stage * mActualPerStage * nActual;
                uint32_t dstShape_[2] = {gbrcRealProcess, nActual};
                uint32_t srcShape_[2] = {gbrcRealProcess, 1};

                AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[gbrcStart * nActual];
                AscendC::LocalTensor<float> aUbTensor = (pingpongFlag == 0) ? aUbTensorPing : aUbTensorPong;
                AscendC::LocalTensor<float> hUbTensor = (pingpongFlag == 0) ? hUbTensorPing : hUbTensorPong;
                AscendC::LocalTensor<float> outUbTensor = (pingpongFlag == 0) ? outUbTensorPing : outUbTensorPong;
                AscendC::LocalTensor<HElementOutput> outUbFPTensor = (pingpongFlag == 0) ? outUbFPTensorPing : outUbFPTensorPong;
                AscendC::LocalTensor<HElementOutput> outUbBFTensor = (pingpongFlag == 0) ? outUbBFTensorPing : outUbBFTensorPong;

                AscendC::Broadcast<float, 2, 1>(gbrcLeftcastUbTensor, gcompUbTensor[gbrcRealStart], dstShape_, srcShape_, shareUbTensor);
                AscendC::PipeBarrier<PIPE_V>();

                __ubuf__ float* outAddr = (__ubuf__ float*)outUbTensor.GetPhyAddr();
                __ubuf__ float* attnAddr = (__ubuf__ float*)aUbTensor[ubStageOffset].GetPhyAddr();
                __ubuf__ float* hiddenAddr = (__ubuf__ float*)hUbTensor[ubStageOffset].GetPhyAddr();
                __ubuf__ float* gateAddr =
                    (__ubuf__ float*)gbrcLeftcastUbTensor[gbrcEffStart * nActual].GetPhyAddr();
                constexpr uint32_t VF_DUAL_ELEMENTS = 2 * AscendC::VECTOR_REG_WIDTH / sizeof(float);
                uint32_t totalElements = mActualThisStage * nActual;
                uint32_t vfElements = totalElements / VF_DUAL_ELEMENTS * VF_DUAL_ELEMENTS;
                if (vfElements != 0) {
                    AscendC::VF_CALL<GdnFwdoOutputMulAddScaleVf>(
                        outAddr, attnAddr, hiddenAddr, gateAddr, (float)scale, vfElements);
                }
                uint32_t tailElements = totalElements - vfElements;
                if (tailElements != 0) {
                    AscendC::Mul(outUbTensor[vfElements], hUbTensor[ubStageOffset + vfElements],
                        gbrcLeftcastUbTensor[gbrcEffStart * nActual + vfElements], tailElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(outUbTensor[vfElements], aUbTensor[ubStageOffset + vfElements],
                        outUbTensor[vfElements], tailElements);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(outUbTensor[vfElements], outUbTensor[vfElements], (float)scale, tailElements);
                }
                AscendC::PipeBarrier<PIPE_V>();

                if (mte3Pending) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3PendingEvent);
                    mte3Pending = false;
                }

                if(std::is_same<HElementOutput, half>::value)
                {
                    AscendC::Cast(outUbFPTensor, outUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisStage * nActual);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::DataCopy(hOutputThisSubBlock, outUbFPTensor, mActualThisStage * nActual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
                    mte3Pending = true;
                    mte3PendingEvent = EVENT_ID0 + pingpongFlag;
                }
                else
                {
                    AscendC::Cast(outUbBFTensor, outUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisStage * nActual);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                    AscendC::DataCopy(hOutputThisSubBlock, outUbBFTensor, mActualThisStage * nActual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
                    mte3Pending = true;
                    mte3PendingEvent = EVENT_ID0 + pingpongFlag;
                }
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
    AscendC::LocalTensor<float> hUbTensorPing;
    AscendC::LocalTensor<float> outUbTensorPing;
    AscendC::LocalTensor<HElementOutput> outUbFPTensorPing;
    AscendC::LocalTensor<HElementOutput> outUbBFTensorPing;

    AscendC::LocalTensor<float> gUbTensorPong;
    AscendC::LocalTensor<GElementInput> gUbFPTensorPong;
    AscendC::LocalTensor<GElementInput> gUbBFTensorPong;
    AscendC::LocalTensor<float> aUbTensorPong;
    AscendC::LocalTensor<float> hUbTensorPong;
    AscendC::LocalTensor<float> outUbTensorPong;
    AscendC::LocalTensor<HElementOutput> outUbFPTensorPong;
    AscendC::LocalTensor<HElementOutput> outUbBFTensorPong;


};
}

#endif
