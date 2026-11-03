/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_GDN_FWDH_UPDATE_HPP
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "../gdn_fwd_h_epilogue_policies.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

struct GDNFwdHNoFinalRefill {
    CATLASS_DEVICE void operator()() const {}
};

template <
    class HOutputType_,
    class GInputType_,
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_,
    class KGatedTag
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHUpdate,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_,
    KGatedTag
> {
    static constexpr bool kGated = KGatedTag::value;
    static constexpr bool scalarGated = KGatedTag::scalarGated;
    static constexpr bool useExp2 = KGatedTag::useExp2;
    static constexpr float LN2 = 0.6931471805599453f;
public:
    using DispatchPolicy = EpilogueAtlasGDNFwdHUpdate;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using HElementInput = typename HInputType_::Element;
    using HUpdateElementInput = typename HUpdateInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    static constexpr bool kSharedLowResident = !kGated && scalarGated &&
        std::is_same<FinalStateElement, float>::value &&
        (std::is_same<HElementInput, half>::value || std::is_same<HElementInput, bfloat16_t>::value) &&
        (std::is_same<HElementOutput, half>::value || std::is_same<HElementOutput, bfloat16_t>::value);
#else
    static constexpr bool kSharedLowResident = false;
#endif

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t SHARED_LOW_OFFSET = 160 * 1024;
        constexpr uint32_t PING_BUF_0_OFFSET = 32 * 1024;
        constexpr uint32_t PING_BUF_1_OFFSET = 48 * 1024;
        constexpr uint32_t PING_BUF_2_OFFSET = 64 * 1024;
        constexpr uint32_t PING_BUF_3_OFFSET = 80 * 1024;
        constexpr uint32_t PONG_BUF_0_OFFSET = 96 * 1024;
        constexpr uint32_t PONG_BUF_1_OFFSET = 112 * 1024;
        constexpr uint32_t PONG_BUF_2_OFFSET = 128 * 1024;
        constexpr uint32_t PONG_BUF_3_OFFSET = 144 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = (kSharedLowResident ? 176 : 160) * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = (kSharedLowResident ? 177 : 161) * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = (kSharedLowResident ? 178 : 162) * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = (kSharedLowResident ? 179 : 163) * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = (kSharedLowResident ? 180 : 164) * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = (kSharedLowResident ? 181 : 165) * 1024;
        constexpr uint32_t SHARE_BUF_OFFSET = (kSharedLowResident ? 182 : 166) * 1024;


        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);
        if constexpr (kSharedLowResident) {
            // 每路完整 FP32 carry 复用旧 U16KiB + H16KiB；V1 不写此槽。
            residentStateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_2_OFFSET);
            residentStateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_2_OFFSET);
        }

        hUpdateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        gkBroadcastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_1_OFFSET);
        hUbTensor_ping = resource.ubBuf.template GetBufferByByte<HElementOutput>(kSharedLowResident ? SHARED_LOW_OFFSET : PING_BUF_3_OFFSET);
        finalOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<FinalStateElement>(PING_BUF_3_OFFSET);
        glastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_INPUT_BUF_OFFSET);

        hUpdateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        gkBroadcastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_1_OFFSET);
        hUbTensor_pong = resource.ubBuf.template GetBufferByByte<HElementOutput>(kSharedLowResident ? SHARED_LOW_OFFSET : PONG_BUF_3_OFFSET);
        finalOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<FinalStateElement>(PONG_BUF_3_OFFSET);
        glastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_INPUT_BUF_OFFSET);

        if constexpr (kGated) {
            gkLastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_SUB_BUF_OFFSET);
            gkLastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_SUB_BUF_OFFSET);
            gkInputUbTensor_ping = resource.ubBuf.template GetBufferByByte<GElementInput>(PING_G_INPUT_BUF_OFFSET);
            gkInputUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_G_INPUT_BUF_OFFSET);
            shareBufferGk_ = resource.ubBuf.template GetBufferByByte<uint8_t>(SHARE_BUF_OFFSET);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    CATLASS_DEVICE void ConfigureOwnerLocal(Arch::Resource<ArchTag> &resource)
    {
        static_assert(kSharedLowResident, "Owner-local HV2 preserves the original shared-low recurrence");
        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(16 * 1024);
        hUpdateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(48 * 1024);
        hUpdateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(48 * 1024);
        residentStateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(123 * 1024);
        residentStateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(155 * 1024);
        hUbTensor_ping = resource.ubBuf.template GetBufferByByte<HElementOutput>(83 * 1024);
        hUbTensor_pong = resource.ubBuf.template GetBufferByByte<HElementOutput>(99 * 1024);
        finalOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<FinalStateElement>(123 * 1024);
        finalOutputUbTensor_pong = resource.ubBuf.template GetBufferByByte<FinalStateElement>(155 * 1024);
        // S.SetValue可能越过Q的MTE3_V；不能与Q/O Gcomp(117KiB)别名。
        glastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(119 * 1024);
        glastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(120 * 1024);
    }
#endif

    __aicore__ inline void ConfigureGLastExp(GM_ADDR addr)
    {
        if (addr != nullptr) {
            gLastExpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addr));
            hasGLastExp_ = true;
        }
    }

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

    __aicore__ inline void StateCopyReady(uint32_t pingpongFlag)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
    }

    __aicore__ inline void RetireLowSnapshot(uint32_t pingpongFlag)
    {
        if constexpr (kSharedLowResident) {
            // V 等快照最后读者；原 caller V_MTE2 再保护下一 Low 搬入。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
        }
    }

    template <class FinalRefill = GDNFwdHNoFinalRefill>
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<float> hUpdateInput,
        AscendC::GlobalTensor<GElementInput> gkInput,
        AscendC::GlobalTensor<FinalStateElement> initialState,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vBlockDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool useInitialState,
        bool isPing,
        bool cube2AlreadyWaited,
        uint64_t gLastExpIndex,
        const FinalRefill& finalRefill = FinalRefill{}
    )
    {
        static constexpr uint32_t ROW_TILE = 16;
        uint32_t mActual = kHeadDim;
        uint32_t nActual = vBlockDim;
        uint32_t outputStride = vHeadDim;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t rowBegin = subBlockIdx * rowsPerSubBlock;
        uint32_t rowEnd = rowBegin + rowsPerSubBlock;
        if (rowEnd > mActual) {
            rowEnd = mActual;
        }
        if (rowBegin >= mActual) {
            if (!cube2AlreadyWaited) {
                Arch::CrossCoreWaitFlag(cube2Done);
            }
            return;
        }

        AscendC::ResetMask();

        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<float> hUpdateUbTensor = isPing ? hUpdateUbTensor_ping : hUpdateUbTensor_pong;
        AscendC::LocalTensor<float> gkBroadcastUbTensor =
            isPing ? gkBroadcastUbTensor_ping : gkBroadcastUbTensor_pong;
        AscendC::LocalTensor<HElementOutput> hUbTensor = isPing ? hUbTensor_ping : hUbTensor_pong;
        AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor = isPing ? finalOutputUbTensor_ping : finalOutputUbTensor_pong;
        AscendC::LocalTensor<float> glastUbTensor = isPing ? glastUbTensor_ping : glastUbTensor_pong;
        AscendC::LocalTensor<float> stateUbTensor = calcUbTensor;
        AscendC::LocalTensor<float> updatedStateUbTensor = hUpdateUbTensor;
        if constexpr (kSharedLowResident) {
            stateUbTensor = isPing ? residentStateUbTensor_ping : residentStateUbTensor_pong;
            updatedStateUbTensor = stateUbTensor;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pingpongFlag);
        bool useFp32StateUpdate = storeFinalState && std::is_same<FinalStateElement, float>::value &&
                                 (!isInitialState || useInitialState);

        float muls = 1.0f;
        if constexpr (scalarGated) {
            if (hasGLastExp_) {
                muls = gLastExpGm_.GetValue(gLastExpIndex);
            } else {
                GElementInput gLastVal = gInputThisSubBlock.GetValue(chunkSize-1);
                float gLastFloat = 0.0f;
                if constexpr(std::is_same<GElementInput, float>::value) {
                    gLastFloat = gLastVal;
                } else if constexpr(std::is_same<GElementInput, half>::value) {
                    gLastFloat = (float)gLastVal;
                } else if constexpr(std::is_same<GElementInput, bfloat16_t>::value) {
                    gLastFloat = AscendC::ToFloat(gLastVal);
                }
                glastUbTensor.SetValue(0, gLastFloat);

                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
                if constexpr (useExp2) {
                    AscendC::Muls(glastUbTensor, glastUbTensor, LN2, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Exp(glastUbTensor, glastUbTensor, 1);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
                muls = glastUbTensor.GetValue(0);
            }
        }
        if constexpr (kGated) {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2 + pingpongFlag);
        }
        if constexpr (scalarGated) {
            if (!hasGLastExp_) {
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
            }
        }

        if (nActual <= 128 && nActual == outputStride) {
            uint32_t mActualThisSubBlock = rowEnd - rowBegin;
            AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[rowBegin * outputStride];
            AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[rowBegin * outputStride];
            AscendC::GlobalTensor<float> hUpdateInputThisSubBlock = hUpdateInput[rowBegin * nActual];
            AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[rowBegin * outputStride];

            if (storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
            if constexpr (std::is_same<FinalStateElement, float>::value) {
                if (useFp32StateUpdate) {
                    if (isInitialState) {
                        AscendC::DataCopy(stateUbTensor, initialState[rowBegin * outputStride],
                                          mActualThisSubBlock * nActual);
                        if constexpr (kSharedLowResident) {
                            StateCopyReady(pingpongFlag);
                        }
                    } else if constexpr (!kSharedLowResident) {
                        AscendC::DataCopy(stateUbTensor, finalStateThisSubBlock,
                                          mActualThisSubBlock * nActual);
                    }
                } else {
                    AscendC::DataCopy(hUbTensor, hInputThisSubBlock,
                                      mActualThisSubBlock * nActual);
                    if constexpr (kSharedLowResident) {
                        StateCopyReady(pingpongFlag);
                    }
                }
            } else {
                AscendC::DataCopy(hUbTensor, hInputThisSubBlock,
                                  mActualThisSubBlock * nActual);
            }
            if constexpr (!kSharedLowResident) {
                StateCopyReady(pingpongFlag);
            }

            if (!useFp32StateUpdate) {
                AscendC::Cast(stateUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE,
                              mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Muls(calcUbTensor, stateUbTensor, muls, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();

            if constexpr (kGated) {
                AscendC::GlobalTensor<GElementInput> gkLastInput = gkInput[(chunkSize - 1) * kHeadDim + rowBegin];
                AscendC::LocalTensor<float> gkLastUbTensor = isPing ? gkLastUbTensor_ping : gkLastUbTensor_pong;
                AscendC::LocalTensor<GElementInput> gkInputUbTensor = isPing ? gkInputUbTensor_ping : gkInputUbTensor_pong;

                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2 + pingpongFlag);
                if constexpr(std::is_same<GElementInput, float>::value) {
                    AscendC::DataCopy(gkLastUbTensor, gkLastInput, mActualThisSubBlock);
                } else {
                    AscendC::DataCopy(gkInputUbTensor, gkLastInput, mActualThisSubBlock);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                if constexpr(!std::is_same<GElementInput, float>::value) {
                    AscendC::Cast(gkLastUbTensor, gkInputUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(gkLastUbTensor, gkLastUbTensor, LN2, mActualThisSubBlock);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(gkLastUbTensor, gkLastUbTensor, mActualThisSubBlock);
                AscendC::PipeBarrier<PIPE_V>();

                uint32_t gkBrcReptime = (mActualThisSubBlock + 8 - 1) / 8;
                uint32_t dstShapeGk[2] = {gkBrcReptime * 8, nActual};
                uint32_t srcShapeGk[2] = {gkBrcReptime * 8, 1};
                AscendC::Broadcast<float, 2, 1>(hUpdateUbTensor, gkLastUbTensor,
                                                dstShapeGk, srcShapeGk, shareBufferGk_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(calcUbTensor, calcUbTensor, hUpdateUbTensor,
                             mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            }

            if (!cube2AlreadyWaited) {
                Arch::CrossCoreWaitFlag(cube2Done);
            }

            if constexpr (kGated) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::DataCopy(hUpdateUbTensor, hUpdateInputThisSubBlock, mActualThisSubBlock * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::Add<float>(updatedStateUbTensor, calcUbTensor, hUpdateUbTensor, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            if (storeFinalState && isFinalState && std::is_same<FinalStateElement, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
            }

            if constexpr(std::is_same<FinalStateElement, float>::value) {
                if (storeFinalState) {
                    constexpr bool overlapStateStoreCast = !kGated && scalarGated &&
                        (std::is_same<HElementOutput, half>::value ||
                         std::is_same<HElementOutput, bfloat16_t>::value);
                    if constexpr (overlapStateStoreCast) {
                        // Let MTE3 read FP32 state while V casts the snapshot.
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        if constexpr (!kSharedLowResident) {
                            AscendC::DataCopy(finalStateThisSubBlock, updatedStateUbTensor, mActualThisSubBlock * nActual);
                        }
                    }
                    if (!isFinalState) {
                        AscendC::Cast(hUbTensor, updatedStateUbTensor, AscendC::RoundMode::CAST_RINT,
                                      mActualThisSubBlock * nActual);
                        AscendC::PipeBarrier<PIPE_V>();
                        if constexpr (overlapStateStoreCast) {
                            // Join the Cast reader before EVENT0 can release the FP32 source.
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                        }
                    } else if constexpr (kSharedLowResident) {
                        AscendC::DataCopy(finalStateThisSubBlock, updatedStateUbTensor, mActualThisSubBlock * nActual);
                        // 仅原末 V2 分支：H0 完成后才发布原 EVENT0 和 vec2Done。
                        finalRefill();
                    }
                    if constexpr (!overlapStateStoreCast) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        AscendC::DataCopy(finalStateThisSubBlock, updatedStateUbTensor, mActualThisSubBlock * nActual);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
                    if (!isFinalState) {
                        AscendC::DataCopy(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                        RetireLowSnapshot(pingpongFlag);
                    }
                } else {
                    AscendC::Cast(hUbTensor, updatedStateUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::DataCopy(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                    RetireLowSnapshot(pingpongFlag);
                }
            } else {
                if (storeFinalState && isFinalState) {
                    AscendC::Cast(finalOutputUbTensor, updatedStateUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::DataCopy(finalStateThisSubBlock, finalOutputUbTensor, mActualThisSubBlock * nActual);
                } else {
                    AscendC::Cast(hUbTensor, updatedStateUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::DataCopy(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pingpongFlag);
            return;
        }

        if (!cube2AlreadyWaited) {
            Arch::CrossCoreWaitFlag(cube2Done);
        }

        bool waitHFromV = storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value;
        bool waitUpdateFromMte3 = false;
        uint32_t updateReadyEvent = EVENT_ID3 + pingpongFlag;
        bool waitGkFromScalar = true;
        for (uint32_t rowStart = rowBegin; rowStart < rowEnd; rowStart += ROW_TILE) {
            uint32_t rowsThisTile = rowEnd - rowStart;
            if (rowsThisTile > ROW_TILE) {
                rowsThisTile = ROW_TILE;
            }

            AscendC::GlobalTensor<HElementOutput> hOutputThisTile = hOutput[rowStart * outputStride];
            AscendC::GlobalTensor<HElementInput> hInputThisTile = hInput[rowStart * outputStride];
            AscendC::GlobalTensor<float> hUpdateInputThisTile = hUpdateInput[rowStart * nActual];
            AscendC::GlobalTensor<FinalStateElement> finalStateThisTile = finalState[rowStart * outputStride];
            AscendC::LocalTensor<float> stateThisTile = stateUbTensor;
            AscendC::LocalTensor<float> updatedStateThisTile = updatedStateUbTensor;
            if constexpr (kSharedLowResident) {
                uint32_t stateRowOffset = (rowStart - rowBegin) * nActual;
                stateThisTile = stateUbTensor[stateRowOffset];
                updatedStateThisTile = updatedStateUbTensor[stateRowOffset];
            }

            if (waitHFromV) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
            if constexpr (std::is_same<FinalStateElement, float>::value) {
                if (useFp32StateUpdate) {
                    if (isInitialState) {
                        CopyGmToUb(stateThisTile, initialState[rowStart * outputStride],
                                   rowsThisTile, nActual, outputStride);
                        if constexpr (kSharedLowResident) {
                            StateCopyReady(pingpongFlag);
                        }
                    } else if constexpr (!kSharedLowResident) {
                        CopyGmToUb(stateThisTile, finalStateThisTile, rowsThisTile, nActual,
                                   outputStride);
                    }
                } else {
                    CopyGmToUb(hUbTensor, hInputThisTile, rowsThisTile, nActual,
                               outputStride);
                    if constexpr (kSharedLowResident) {
                        StateCopyReady(pingpongFlag);
                    }
                }
            } else {
                CopyGmToUb(hUbTensor, hInputThisTile, rowsThisTile, nActual, outputStride);
            }
            if constexpr (!kSharedLowResident) {
                StateCopyReady(pingpongFlag);
            }

            if (!useFp32StateUpdate) {
                AscendC::Cast(stateThisTile, hUbTensor, AscendC::RoundMode::CAST_NONE, rowsThisTile * nActual);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Muls(calcUbTensor, stateThisTile, muls, rowsThisTile * nActual);
            AscendC::PipeBarrier<PIPE_V>();

            if constexpr (kGated) {
                AscendC::GlobalTensor<GElementInput> gkLastInput = gkInput[(chunkSize - 1) * kHeadDim + rowStart];
                AscendC::LocalTensor<float> gkLastUbTensor = isPing ? gkLastUbTensor_ping : gkLastUbTensor_pong;
                AscendC::LocalTensor<GElementInput> gkInputUbTensor = isPing ? gkInputUbTensor_ping : gkInputUbTensor_pong;

                if (waitGkFromScalar) {
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2 + pingpongFlag);
                    waitGkFromScalar = false;
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
                }
                if constexpr(std::is_same<GElementInput, float>::value) {
                    AscendC::DataCopy(gkLastUbTensor, gkLastInput, rowsThisTile);
                } else {
                    AscendC::DataCopy(gkInputUbTensor, gkLastInput, rowsThisTile);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
                if constexpr(!std::is_same<GElementInput, float>::value) {
                    AscendC::Cast(gkLastUbTensor, gkInputUbTensor, AscendC::RoundMode::CAST_NONE, rowsThisTile);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(gkLastUbTensor, gkLastUbTensor, LN2, rowsThisTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(gkLastUbTensor, gkLastUbTensor, rowsThisTile);
                AscendC::PipeBarrier<PIPE_V>();

                uint32_t gkBrcReptime = (rowsThisTile + 8 - 1) / 8;
                uint32_t dstShapeGk[2] = {gkBrcReptime * 8, nActual};
                uint32_t srcShapeGk[2] = {gkBrcReptime * 8, 1};
                AscendC::Broadcast<float, 2, 1>(gkBroadcastUbTensor, gkLastUbTensor,
                                                dstShapeGk, srcShapeGk, shareBufferGk_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(calcUbTensor, calcUbTensor, gkBroadcastUbTensor, rowsThisTile * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
            }

            if (waitUpdateFromMte3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            }
            AscendC::DataCopy(hUpdateUbTensor, hUpdateInputThisTile, rowsThisTile * nActual);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
            AscendC::Add<float>(updatedStateThisTile, calcUbTensor, hUpdateUbTensor, rowsThisTile * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            if (storeFinalState && isFinalState && std::is_same<FinalStateElement, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
                waitHFromV = true;
            } else {
                waitHFromV = false;
            }

            if constexpr(std::is_same<FinalStateElement, float>::value) {
                if (storeFinalState) {
                    if (!isFinalState) {
                        AscendC::Cast(hUbTensor, updatedStateThisTile, AscendC::RoundMode::CAST_RINT,
                                      rowsThisTile * nActual);
                        AscendC::PipeBarrier<PIPE_V>();
                        if constexpr (kSharedLowResident) {
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        }
                    } else if constexpr (kSharedLowResident) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        CopyUbToGm(finalStateThisTile, updatedStateThisTile, rowsThisTile, nActual, outputStride);
                    }
                    if constexpr (!kSharedLowResident) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                        CopyUbToGm(finalStateThisTile, updatedStateThisTile, rowsThisTile, nActual, outputStride);
                    }
                    // MTE3_V 退休下一 V 写者；MTE3_MTE2 由下一 row 或收尾消费，保护 MTE2 复用。
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                    waitUpdateFromMte3 = true;
                    if (!isFinalState) {
                        CopyUbToGm(hOutputThisTile, hUbTensor, rowsThisTile, nActual, outputStride);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                        RetireLowSnapshot(pingpongFlag);
                    }
                } else {
                    AscendC::Cast(hUbTensor, updatedStateThisTile, AscendC::RoundMode::CAST_RINT, rowsThisTile * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    CopyUbToGm(hOutputThisTile, hUbTensor, rowsThisTile, nActual, outputStride);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                    RetireLowSnapshot(pingpongFlag);
                    waitUpdateFromMte3 = false;
                }
            } else {
                if (storeFinalState && isFinalState) {
                    AscendC::Cast(finalOutputUbTensor, updatedStateThisTile, AscendC::RoundMode::CAST_RINT, rowsThisTile * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    CopyUbToGm(finalStateThisTile, finalOutputUbTensor, rowsThisTile, nActual, outputStride);
                } else {
                    AscendC::Cast(hUbTensor, updatedStateThisTile, AscendC::RoundMode::CAST_RINT, rowsThisTile * nActual);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                    CopyUbToGm(hOutputThisTile, hUbTensor, rowsThisTile, nActual, outputStride);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                waitUpdateFromMte3 = false;
            }
        }
        if (storeFinalState && std::is_same<FinalStateElement, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
            if (!isFinalState) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        }
        if constexpr (kGated) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + pingpongFlag);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + pingpongFlag);

    }

private:
    bool hasGLastExp_ = false;
    AscendC::GlobalTensor<float> gLastExpGm_;
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;
    AscendC::LocalTensor<float> residentStateUbTensor_ping;
    AscendC::LocalTensor<float> residentStateUbTensor_pong;

    AscendC::LocalTensor<float> hUpdateUbTensor_ping;
    AscendC::LocalTensor<float> gkBroadcastUbTensor_ping;
    AscendC::LocalTensor<HElementOutput> hUbTensor_ping;
    AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor_ping;
    AscendC::LocalTensor<float> glastUbTensor_ping;

    AscendC::LocalTensor<float> hUpdateUbTensor_pong;
    AscendC::LocalTensor<float> gkBroadcastUbTensor_pong;
    AscendC::LocalTensor<HElementOutput> hUbTensor_pong;
    AscendC::LocalTensor<FinalStateElement> finalOutputUbTensor_pong;
    AscendC::LocalTensor<float> glastUbTensor_pong;

    AscendC::LocalTensor<float> gkLastUbTensor_ping;
    AscendC::LocalTensor<float> gkLastUbTensor_pong;
    AscendC::LocalTensor<GElementInput> gkInputUbTensor_ping;
    AscendC::LocalTensor<GElementInput> gkInputUbTensor_pong;
    AscendC::LocalTensor<uint8_t> shareBufferGk_;
};
}

#endif
