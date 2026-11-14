/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file arch22/chunk_fwd_h_utils.h
 * \brief gated 复用引擎（仅 A2/arch22，由 chunk_fwd_h.cpp 非 arch35 分支引入）：
 * 复用 chunk_gated_delta_rule_fwd_h 的 preload kernel 链。原六个文件
 * （chunk_fwd_h_gated_struct.h、epilogue/gdn_fwd_h_epilogue_policies.hpp、
 * epilogue/block/block_epilogue_gdn_fwdh_{update,vnew}_preload.hpp、
 * gemm/block/block_scheduler_gdn_fwd_h_preload.hpp、gemm/kernel/gdn_fwd_h_kernel_preload.hpp）
 * 按原 include 展开顺序内联为单文件，芯片目录回归 cube.h/vec.h + 本文件布局；
 * 另将 chunk_fwd_h.cpp 的 gated 复用谓词/入口/分流移至文末。纯搬移，无代码语义变化。
 */

#ifndef ARCH22_CHUNK_FWD_H_UTILS_H
#define ARCH22_CHUNK_FWD_H_UTILS_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

// ---- 内联自原 chunk_fwd_h_gated_struct.h：本尊算子 kernel 侧 tiling 结构体镜像
// （字段序/类型对齐框架 BEGIN_TILING_DATA_DEF 宏生成的同名结构；fast kernel launch
// 扩展独立编译 kernel 时无该自动生成头，故提供同构 plain struct，可按值经 <<<>>> 传入）
struct ChunkGatedDeltaRuleFwdHTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t dataType;
    int64_t gDataType;
    int64_t stateDataType;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    bool useG;
    bool useGk;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
};

// ---- 内联自原 epilogue/gdn_fwd_h_epilogue_policies.hpp（仅保留 preload 策略——
// 非 preload 两策略 EpilogueAtlasGDNFwdH{Vnew,Update} 供原版非 preload 链使用，
// 本拷贝未引入该链，零引用，已删）
namespace Catlass::Epilogue {

// 域模板维（2026-11-10 exp 域复用释放）：USE_EXP2_=true（log2 域，2^x）/
// false（ln 域，e^x）——epilogue 内两处 LN2 换算按此 if constexpr 取舍。
template <bool USE_EXP2_>
struct EpilogueAtlasGDNFwdHVnewPreload {
    using ArchTag = Arch::AtlasA2;
};

template <bool USE_EXP2_>
struct EpilogueAtlasGDNFwdHUpdatePreload {
    using ArchTag = Arch::AtlasA2;
};

}  // namespace Catlass::Epilogue

// ---- 内联自原 epilogue/block/block_epilogue_gdn_fwdh_update_preload.hpp
namespace Catlass::Epilogue::Block {

template <
    bool USE_EXP2_,
    class HOutputType_,
    class GInputType_,
    class HInputType_,
    class HUpdateInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHUpdatePreload<USE_EXP2_>,
    HOutputType_,
    GInputType_,
    HInputType_,
    HUpdateInputType_,
    FinalStateType_
> {
    static constexpr float LN2 = 0.6931471805599453f;
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHUpdatePreload<USE_EXP2_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using HElementOutput = typename HOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using HElementInput = typename HInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t PING_BUF_0_OFFSET = 32 * 1024;
        constexpr uint32_t PING_BUF_3_OFFSET = 80 * 1024;
        constexpr uint32_t PONG_BUF_0_OFFSET = 96 * 1024;
        constexpr uint32_t PONG_BUF_3_OFFSET = 144 * 1024;
        constexpr uint32_t PING_G_INPUT_BUF_OFFSET = 164 * 1024;
        constexpr uint32_t PONG_G_INPUT_BUF_OFFSET = 165 * 1024;

        calcUbTensor = resource.ubBuf.template GetBufferByByte<float>(CALC_BUF_OFFSET);

        hUpdateUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        hUbTensor_ping = resource.ubBuf.template GetBufferByByte<HElementOutput>(PING_BUF_3_OFFSET);
        glastUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_G_INPUT_BUF_OFFSET);

        hUpdateUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        hUbTensor_pong = resource.ubBuf.template GetBufferByByte<HElementOutput>(PONG_BUF_3_OFFSET);
        glastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_INPUT_BUF_OFFSET);

    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<HElementOutput> hOutput,
        AscendC::GlobalTensor<FinalStateElement> finalState,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<HElementInput> hInput,
        AscendC::GlobalTensor<float> hUpdateInput,
        uint32_t chunkSize,
        uint32_t kHeadDim,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube2Done,
        bool isInitialState,
        bool isFinalState,
        bool storeFinalState,
        bool isPing
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

        AscendC::GlobalTensor<HElementOutput> hOutputThisSubBlock = hOutput[offsetH];
        AscendC::GlobalTensor<GElementInput> gInputThisSubBlock = gInput;
        AscendC::GlobalTensor<HElementInput> hInputThisSubBlock = hInput[offsetH];
        AscendC::GlobalTensor<float> hUpdateInputThisSubBlock = hUpdateInput[offsetH];
        AscendC::GlobalTensor<FinalStateElement> finalStateThisSubBlock = finalState[offsetH];

        uint32_t pingpongFlag = isPing ? 0 : pongBaseEvent;
        AscendC::LocalTensor<float> hUpdateUbTensor = isPing ? hUpdateUbTensor_ping : hUpdateUbTensor_pong;
        AscendC::LocalTensor<HElementOutput> hUbTensor = isPing ? hUbTensor_ping : hUbTensor_pong;
        AscendC::LocalTensor<float> glastUbTensor = isPing ? glastUbTensor_ping : glastUbTensor_pong;

        if (storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        }
        AscendC::DataCopy(hUbTensor, hInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + pingpongFlag);

        AscendC::Cast(calcUbTensor, hUbTensor, AscendC::RoundMode::CAST_NONE, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();
        if (storeFinalState && isFinalState && std::is_same<FinalStateElement, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pingpongFlag);
        }
        
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
        // 域换算（对齐主线 vec 的 CompilePolicy::USE_EXP2 分支写法）：log2 域
        // （USE_EXP2_=true）×LN2 换算到 ln 域（2^x = e^(LN2·x)）；ln 域（false）
        // e^g 直取免换算。
        if constexpr (USE_EXP2_) {
            AscendC::Muls(glastUbTensor, glastUbTensor, LN2, 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(glastUbTensor, glastUbTensor, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + pingpongFlag);
        float muls = glastUbTensor.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + pingpongFlag);
        AscendC::Muls(calcUbTensor, calcUbTensor, muls, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();
        Arch::CrossCoreWaitFlag(cube2Done);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::DataCopy(hUpdateUbTensor, hUpdateInputThisSubBlock, mActualThisSubBlock * nActual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::Add<float>(hUpdateUbTensor, calcUbTensor, hUpdateUbTensor, mActualThisSubBlock * nActual);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr(std::is_same<FinalStateElement, float>::value) {
            if (storeFinalState && isFinalState) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + pingpongFlag);
                AscendC::DataCopy(finalStateThisSubBlock, hUpdateUbTensor, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
            } else {
                AscendC::Cast(hUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
                AscendC::DataCopy(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
            }
        } else {
            AscendC::Cast(hUbTensor, hUpdateUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nActual);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + pingpongFlag);
            if (isFinalState) {
                AscendC::DataCopy(finalStateThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
            } else {
                AscendC::DataCopy(hOutputThisSubBlock, hUbTensor, mActualThisSubBlock * nActual);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pingpongFlag);
        }
        
    }

private:
    uint32_t pongBaseEvent = 4;

    AscendC::LocalTensor<float> calcUbTensor;

    AscendC::LocalTensor<float> hUpdateUbTensor_ping;
    AscendC::LocalTensor<HElementOutput> hUbTensor_ping;
    AscendC::LocalTensor<float> glastUbTensor_ping;

    AscendC::LocalTensor<float> hUpdateUbTensor_pong;
    AscendC::LocalTensor<HElementOutput> hUbTensor_pong;
    AscendC::LocalTensor<float> glastUbTensor_pong;

};
}

// ---- 内联自原 epilogue/block/block_epilogue_gdn_fwdh_vnew_preload.hpp
namespace Catlass::Epilogue::Block {

template <
    bool USE_EXP2_,
    class VOutputType_,
    class GInputType_,
    class UInputType_,
    class WSInputType_,
    class FinalStateType_
>
class BlockEpilogue <
    EpilogueAtlasGDNFwdHVnewPreload<USE_EXP2_>,
    VOutputType_,
    GInputType_,
    UInputType_,
    WSInputType_,
    FinalStateType_
> {
    static constexpr float LN2 = 0.6931471805599453f;
public:
    // Type aliases
    using DispatchPolicy = EpilogueAtlasGDNFwdHVnewPreload<USE_EXP2_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using VElementOutput = typename VOutputType_::Element;
    using GElementInput = typename GInputType_::Element;
    using UElementInput = typename UInputType_::Element;
    using FinalStateElement = typename FinalStateType_::Element;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource)
    {

        constexpr uint32_t CALC_BUF_OFFSET = 0;
        constexpr uint32_t PING_BUF_0_OFFSET = 32 * 1024;
        constexpr uint32_t PING_BUF_2_OFFSET = 64 * 1024;
        constexpr uint32_t PONG_BUF_0_OFFSET = 96 * 1024;
        constexpr uint32_t PONG_BUF_2_OFFSET = 128 * 1024;
        constexpr uint32_t PING_G_BUF_OFFSET = 160 * 1024;
        constexpr uint32_t PONG_G_BUF_OFFSET = 161 * 1024;
        constexpr uint32_t PING_G_SUB_BUF_OFFSET = 162 * 1024;
        constexpr uint32_t PONG_G_SUB_BUF_OFFSET = 163 * 1024;
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

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<VElementOutput> vnewOutput,
        AscendC::GlobalTensor<VElementOutput> vnewdecayOutput,
        AscendC::GlobalTensor<GElementInput> gInput,
        AscendC::GlobalTensor<UElementInput> uInput,
        AscendC::GlobalTensor<float> wsInput,
        uint32_t chunkSize,
        uint32_t vHeadDim,
        Arch::CrossCoreFlag cube1Done,
        Arch::CrossCoreFlag vec1Done,
        bool isInitialState,
        bool storeFinalState,
        bool isPing
    )
    {
        uint32_t mActual = chunkSize;
        uint32_t nvActual = vHeadDim;

        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t mActualPerSubBlock = CeilDiv(mActual, subBlockNum);
        uint32_t mActualThisSubBlock = (subBlockIdx == 0) ? mActualPerSubBlock : (mActual - mActualPerSubBlock);
        uint32_t mOffset = subBlockIdx * mActualPerSubBlock;
        uint32_t nOffset = 0;
        // 当前场景内部一定连续
        // k [B, H, T, D]
        // g [B, H, T]
        // 在外部offset的基础上进一步offset
        // 当前asset kdim == vHeadDim
        int64_t offsetK = mOffset * nvActual + nOffset;

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
        // 尾块 mActual=1 时 sub1 计数为 0：零长度 DataCopy 会在 GM 张量末边界发出退化访问
        if (mActualThisSubBlock > 0) {
            AscendC::DataCopy(uUbTensor, uInputThisSubBlock, mActualThisSubBlock * nvActual); // mte2 u
        }
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

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<float>(gLastUbTensor, inputVal, mActual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sub<float>(gUbTensor, gLastUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();
        // 域换算（同 update epilogue）：log2 域 ×LN2；ln 域 e^g 直取免换算。
        if constexpr (USE_EXP2_) {
            AscendC::Muls(gUbTensor, gUbTensor, LN2, mActual);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Exp(gUbTensor, gUbTensor, mActual);
        AscendC::PipeBarrier<PIPE_V>();
        Arch::CrossCoreWaitFlag(cube1Done);

        if (storeFinalState && isInitialState && std::is_same<FinalStateElement, float>::value) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pingpongFlag);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        }
        if (mActualThisSubBlock > 0) {
            AscendC::DataCopy(wsUbTensor, wsInputThisSubBlock, mActualThisSubBlock * nvActual);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + pingpongFlag);
        
        AscendC::Sub<float>(wsUbTensor, calcUbTensor, wsUbTensor, mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Broadcast<float, 2, 1>(calcUbTensor, gUbTensor[gbrcRealStart], dstShape_, srcShape_, shareBuffer_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pingpongFlag);

        AscendC::Mul(calcUbTensor[gbrcEffStart*nvActual], wsUbTensor, calcUbTensor[gbrcEffStart*nvActual], mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::Cast(vNewDecayUbTensor, calcUbTensor[gbrcEffStart*nvActual], AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        if (mActualThisSubBlock > 0) {
            AscendC::DataCopy(vnewdecayOutputThisSubBlock, vNewDecayUbTensor, mActualThisSubBlock * nvActual);
        }
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vec1Done);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + pingpongFlag);
        AscendC::Cast(vNewOutputUbTensor, wsUbTensor, AscendC::RoundMode::CAST_RINT, mActualThisSubBlock * nvActual);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pingpongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + pingpongFlag);
        if (mActualThisSubBlock > 0) {
            AscendC::DataCopy(vnewOutputThisSubBlock, vNewOutputUbTensor, mActualThisSubBlock * nvActual);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pingpongFlag);

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


// ---- 内联自原 gemm/block/block_scheduler_gdn_fwd_h_preload.hpp
using namespace Catlass;

// gated 复用拷贝仅引入 preload 链：原版经非 preload scheduler（本拷贝未引入）
// 传递的 PING_PONG_STAGES 在此兜底定义。
#ifndef PING_PONG_STAGES
constexpr uint32_t PING_PONG_STAGES = 2;
#endif

namespace Catlass::Gemm::Block {

struct GDNFwdHOffsetsPreload {
    uint64_t hSrcOffset{0};
    uint64_t hDstOffset{0};
    uint64_t uvOffset{0};
    uint64_t wkOffset{0};
    uint64_t wOffset{0};
    uint64_t gOffset{0};
    uint64_t hWorkOffset{0};
    uint64_t vWorkOffset{0};
    uint64_t initialStateOffset{0};
    uint64_t finalStateOffset{0};
    bool isInitialState{false};
    bool isFinalState{false};
    uint32_t blockTokens{0};
};

struct GDNFwdHStreamPreload {
    uint32_t vIdx{0};
    uint32_t batchIdx{0};
    uint32_t chunkIdx{0};
    uint32_t vHeadIdx{0};
    uint32_t kHeadIdx{0};
    uint32_t shapeBatchIdx{0};
    uint32_t tokenBatchIdx{0};
     
    uint32_t chunkOffset{0};
    uint32_t tokenOffset{0};
    uint32_t batchChunks{0};
    uint32_t batchTokens{0};
     
    GDNFwdHOffsetsPreload offset;
};
     
struct GDNFwdHRunningQPreload {
    GDNFwdHStreamPreload streams[PING_PONG_STAGES];
    uint32_t head{0};
};

struct BlockSchedulerGdnFwdHPreload {
    uint32_t batch{0};
    uint32_t seqlen{0};
    uint32_t kNumHead{0};
    uint32_t vNumHead{0};
    uint32_t kHeadDim{0};
    uint32_t vHeadDim{0};
    uint32_t chunkSize{0};
    uint32_t vBlockSize{128};
    uint32_t isVariedLen{0};
    uint32_t tokenBatch{0};
    bool useInitialState{false};
    bool storeFinalState{false};
    uint64_t numSeqWorkspaceOffset{0};
    uint64_t numChunksWorkspaceOffset{0};

    uint32_t taskIdx{0};
    uint32_t taskLoops{0};
    uint32_t cubeCoreIdx{0};
    uint32_t cubeCoreNum{0};
    uint32_t vLoops{0};
    uint32_t taskNum{0};
    uint32_t headGroups{0};
    uint32_t totalChunks{0};
    uint32_t totalTokens{0};
    bool hasDummyHead{false};

    GDNFwdHRunningQPreload runningQ;
    int64_t curLoopIdx{-1};
    uint32_t curLoopTaskBegin{0};
    uint32_t curLoopTaskCnt{0};
    uint32_t lastLoopTaskCnt{0};

    bool isRunning{false};

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    Arch::CrossCoreFlag cube1Done{0};
    Arch::CrossCoreFlag vec1Done{1};
    Arch::CrossCoreFlag cube2Done{2};
    Arch::CrossCoreFlag vec2Done[PING_PONG_STAGES] = {3, 4};

    CATLASS_DEVICE
    BlockSchedulerGdnFwdHPreload() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR tiling, GM_ADDR user, uint32_t coreIdx, uint32_t coreNum) {
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if (isVariedLen) {
            gmNumChunks.SetValue(0, 0);
            gmNumSeq.SetValue(0, 0);
            uint32_t actualBatch = 0;
            int64_t prevSeq = 0, currSeq;
            for (uint32_t b = 1; b <= tokenBatch; b++) {
                currSeq = gmSeqlen.GetValue(b);
                int64_t batchSeqLen = currSeq - prevSeq;
                if (batchSeqLen > 0) {
                    actualBatch++;
                    gmNumSeq.SetValue(actualBatch, currSeq);
                    int64_t batchChunk = (batchSeqLen + chunkSize - 1) / chunkSize;
                    gmNumChunks.SetValue(actualBatch, gmNumChunks.GetValue(actualBatch - 1) + batchChunk);
                }
                prevSeq = currSeq;
            }
            tokenBatch = actualBatch;
            batch = actualBatch;
            totalChunks = gmNumChunks.GetValue(tokenBatch);
            totalTokens = gmNumSeq.GetValue(tokenBatch);
        } else {
            totalChunks = (seqlen + chunkSize - 1) / chunkSize;
            totalTokens = seqlen;
        }

        cubeCoreIdx = coreIdx;
        cubeCoreNum = coreNum;
        vLoops = vHeadDim / vBlockSize;
        taskNum = vLoops * batch * vNumHead;
        headGroups = vNumHead / kNumHead;
        hasDummyHead = (taskNum % (PING_PONG_STAGES * cubeCoreNum) <= cubeCoreNum) && (taskNum % (PING_PONG_STAGES * cubeCoreNum) > 0);
        taskLoops = (taskNum + cubeCoreNum * PING_PONG_STAGES - 1) / (cubeCoreNum * PING_PONG_STAGES);
        uint32_t maxTaskCntPerLoop = taskNum > cubeCoreNum ? PING_PONG_STAGES : 1;
        curLoopTaskBegin = cubeCoreIdx * maxTaskCntPerLoop;
        uint32_t lastLoopTaskBegin = curLoopTaskBegin + (taskLoops - 1) * maxTaskCntPerLoop * cubeCoreNum;
        if (lastLoopTaskBegin >= taskNum) {
            lastLoopTaskCnt = 0;
        } else {
            uint32_t maxTaskCntLastLoop = hasDummyHead ? 1 : PING_PONG_STAGES;
            lastLoopTaskCnt = taskNum - lastLoopTaskBegin;
            if (lastLoopTaskCnt >= maxTaskCntLastLoop) {
                lastLoopTaskCnt = maxTaskCntLastLoop;
            }
        }
        curLoopTaskCnt = taskLoops > 1 ? PING_PONG_STAGES : lastLoopTaskCnt;
        curLoopIdx = -1; // -1: 第一次创建task时会将curLoopIdx加1
        taskIdx = curLoopTaskBegin + PING_PONG_STAGES; // 第一次创建task时重新初始化taskIdx
        isRunning = curLoopTaskBegin < taskNum;

    }


    CATLASS_DEVICE
    void InitNewStream(GDNFwdHStreamPreload& newStream) {
        newStream.vIdx = taskIdx / (batch * vNumHead);
        newStream.batchIdx = (taskIdx - newStream.vIdx * batch * vNumHead) / vNumHead;
        newStream.vHeadIdx = taskIdx % vNumHead;
        newStream.kHeadIdx = newStream.vHeadIdx / headGroups;
        newStream.shapeBatchIdx = isVariedLen ? 0 : newStream.batchIdx;
        newStream.tokenBatchIdx = isVariedLen ? newStream.batchIdx : 0;
        newStream.chunkOffset = isVariedLen ? gmNumChunks.GetValue(newStream.tokenBatchIdx) : 0;
        newStream.batchChunks = isVariedLen ? (gmNumChunks.GetValue(newStream.tokenBatchIdx + 1) - newStream.chunkOffset) : totalChunks;
        newStream.tokenOffset = isVariedLen ? gmNumSeq.GetValue(newStream.tokenBatchIdx) : 0;
        newStream.batchTokens = isVariedLen ? (gmNumSeq.GetValue(newStream.tokenBatchIdx + 1) - newStream.tokenOffset) : totalTokens;
        newStream.chunkIdx = 0;
    }
     
    CATLASS_DEVICE
    void UpdateTask(uint32_t streamId) {
        auto& stream = runningQ.streams[streamId];
        auto& offset = stream.offset;
     
        offset.isInitialState = stream.chunkIdx == 0; 
        offset.isFinalState = stream.chunkIdx == (stream.batchChunks - 1); 
        offset.initialStateOffset = (stream.batchIdx * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim; 
        offset.finalStateOffset = (stream.batchIdx * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim; 
        offset.hSrcOffset = ((stream.shapeBatchIdx * totalChunks + stream.chunkOffset + stream.chunkIdx) * vNumHead + stream.vHeadIdx) * kHeadDim * vHeadDim;
        offset.hDstOffset = offset.hSrcOffset + vNumHead * kHeadDim * vHeadDim;
        offset.uvOffset = (stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * vHeadDim;
        offset.wkOffset = (stream.shapeBatchIdx * kNumHead * totalTokens + stream.kHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * kHeadDim;
        offset.wOffset = (stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize) * kHeadDim;
        offset.gOffset = stream.shapeBatchIdx * vNumHead * totalTokens + stream.vHeadIdx * totalTokens + stream.tokenOffset + stream.chunkIdx * chunkSize;
        offset.hWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + streamId) * kHeadDim * vHeadDim;
        offset.vWorkOffset = (cubeCoreIdx * PING_PONG_STAGES + streamId) * chunkSize * vHeadDim;
        offset.blockTokens = offset.isFinalState ? (stream.batchTokens - stream.chunkIdx * chunkSize) : chunkSize;
    }
     
    CATLASS_DEVICE
    void InitTasks() {
        //auto oldHead = runningQ.head;
        auto streamId = runningQ.head;
        for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
            //auto streamId = (oldHead + i) % PING_PONG_STAGES;
            auto& stream = runningQ.streams[streamId];
            stream.chunkIdx += 1;
            if (StreamIsDone(stream)) {
                // 当前stream已完成，用一个新stream替换它
                taskIdx += 1;
                if (taskIdx >= (curLoopTaskBegin + curLoopTaskCnt)) {
                    curLoopIdx += 1;
                    // curLoopTaskBegin = curLoopIdx * PING_PONG_STAGES * cubeCoreNum + PING_PONG_STAGES * cubeCoreIdx;
                    // if (curLoopIdx + 1 >= taskLoops) {
                    //     curLoopTaskCnt = lastLoopTaskCnt;
                    // }
                    if (curLoopIdx + 1 < taskLoops) {
                        curLoopTaskBegin = curLoopIdx * PING_PONG_STAGES * cubeCoreNum + PING_PONG_STAGES * cubeCoreIdx;
                    } else {
                        curLoopTaskBegin = curLoopIdx * PING_PONG_STAGES * cubeCoreNum + (hasDummyHead ? 1 : PING_PONG_STAGES) * cubeCoreIdx;
                        curLoopTaskCnt = lastLoopTaskCnt;
                    }
                    taskIdx = curLoopTaskBegin;
                }
     
                //runningQ.head = (streamId + 1) % PING_PONG_STAGES;
                stream.batchChunks = 0;
                if (taskIdx < taskNum) {
                    InitNewStream(stream);
                    UpdateTask(streamId);
                } 

                if (streamId == runningQ.head) {
                    runningQ.head = (runningQ.head + 1) % PING_PONG_STAGES;
                    if (taskIdx >= taskNum) {
                        // 没有新stream了，将head推进到下一个未完成的stream上
                        for (uint32_t j = 0; j < PING_PONG_STAGES && StreamIsDone(runningQ.streams[runningQ.head]); ++j) {
                            runningQ.head = (runningQ.head + 1) % PING_PONG_STAGES;
                        }
                        isRunning = ! StreamIsDone(runningQ.streams[runningQ.head]);
                    }
                    streamId = runningQ.head;
                }
                else {
                    streamId = (streamId + 1) % PING_PONG_STAGES;
                }

            } else {
                UpdateTask(streamId);
                streamId = (streamId + 1) % PING_PONG_STAGES;
            }
        }
    }
    
    CATLASS_DEVICE
    const GDNFwdHStreamPreload& GetStream(uint32_t i) const {
        return runningQ.streams[(runningQ.head + i) % PING_PONG_STAGES];
    }

    // GetStream(i) 的物理槽位。流迁移时 i 变而槽位不变，epilogue 的事件组与
    // UB 组须按槽位选取，否则迁移流的事件配对断裂，kernel 死锁。
    CATLASS_DEVICE
    uint32_t GetStreamSlot(uint32_t i) const {
        return (runningQ.head + i) % PING_PONG_STAGES;
    }

    CATLASS_DEVICE
    const GDNFwdHOffsetsPreload& GetCurTaskOffsets(const GDNFwdHStreamPreload& stream) const {
        return stream.offset;
    }

    CATLASS_DEVICE
    bool StreamIsDone(const GDNFwdHStreamPreload& stream) const {
        return stream.chunkIdx >= stream.batchChunks;
    }

    CATLASS_DEVICE
    bool NeedProcessStage2(const GDNFwdHStreamPreload& stream) {
        return storeFinalState || !stream.offset.isFinalState;
    }
};

struct BlockSchedulerGdnFwdHPreloadCube : public BlockSchedulerGdnFwdHPreload {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHPreloadCube() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdHPreload::Init(cu_seqlens, tiling, user, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

};

struct BlockSchedulerGdnFwdHPreloadVec : public BlockSchedulerGdnFwdHPreload {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHPreloadVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdHPreload::Init(cu_seqlens, tiling, user, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
    }

};

}  // namespace Catlass::Gemm::Block

// ---- 内联自原 gemm/kernel/gdn_fwd_h_kernel_preload.hpp（using 与类主体）
using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm {

// ===== 本地 MMAD 策略结构体（fork 自 kernel_utils/block/
// block_mmad_pingpong_tla_preload.hpp 的 MmadPingpongTlaPreload 并增
// WAIT_A_BEFORE_LOAD_ 位）：公共块零改动（评审约定），本文件与其解耦——
// 下方 BlockMmadTla 为共享特化的本地 fork，按策略位的 vecDone 等待位置
// 取舍（true=前置到 A 装载前，供 A=Vwork 依赖 V1 输出的 vfirst C2；
// false=共享默认，A 装载与 vecDone 尾部重叠）。勿再 include 原共享头。
template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, bool USE_HF32_MODE_ = false, uint32_t L0C_STAGES_ = 1,
    bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2, uint32_t L0A_STAGES_ = 2,
    uint32_t L0B_STAGES_ = 2, bool WAIT_A_BEFORE_LOAD_ = false>
struct MmadPingpongTlaPreload {
    using ArchTag = ArchTag_;
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool USE_HF32_MODE = USE_HF32_MODE_;
    static constexpr bool ENABLE_L1_RESIDENT = ENABLE_L1_RESIDENT_;
    // WAIT_A_BEFORE_LOAD=true 时先等 vecDone 再装载 A，供 A 依赖 vec 输出的
    // 形态使用；默认 false，A 装载与 vecDone 尾部重叠。
    static constexpr bool WAIT_A_BEFORE_LOAD = WAIT_A_BEFORE_LOAD_;
};

}  // namespace Catlass::Gemm

namespace Catlass::Gemm::Block {

// ===== 本地 MMAD 块（2026-11-12 评审约定：公共块
// block_mmad_pingpong_tla_preload.hpp 零改动、与本文件解耦）：
// fork 共享 BlockMmadTla 的 MmadPingpongTlaPreload 特化并增策略位
// WAIT_A_BEFORE_LOAD（见上方本地 MmadPingpongTlaPreload 注释）——
// A=Vwork（V1 输出）形态等待前置到 A 装载前，其余行为与共享实现逐行
// 一致，本文件三处 MMAD 使用（WH/KV/KVVFirst）全部走本副本。
template <
    class DispatchPolicy_,
    class L1TileShape_,
    class L0TileShape_,
    class ElementA_,
    class ElementB_,
    class ElementC_,
    class ElementBias_ = void,
    class TileCopy_ = Catlass::Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy_::ArchTag, ElementA_, Catlass::layout::RowMajor,
        ElementB_, Catlass::layout::RowMajor, ElementC_, Catlass::layout::RowMajor, ElementBias_>,
    class TileMmad_ =
        Catlass::Gemm::Tile::TileMmadTla<typename DispatchPolicy_::ArchTag, ElementA_, typename TileCopy_::LayoutTagL1A>
>
struct BlockMmadTla {
    // Type Aliases
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using CopyL1ToBT = typename TileCopy::CopyL1ToBT;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool HAS_BIAS = TileCopy::HAS_BIAS;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    using L1AAlignHelper = typename TileCopy_::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy_::L1BAlignHelper;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool USE_HF32_MODE = DispatchPolicy::USE_HF32_MODE;
    static constexpr bool ENABLE_L1_RESIDENT = DispatchPolicy::ENABLE_L1_RESIDENT;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // Check HF32_MODE
    static_assert(
        !USE_HF32_MODE || (USE_HF32_MODE && std::is_same_v<ElementA, float> && std::is_same_v<ElementB, float>),
        "HF32 MODE only supports in float!"
    );

    // Check L0C_STAGES
    static_assert(!(ENABLE_UNIT_FLAG && L0C_STAGES != 1), "L0C_STAGES must be 1 when UnitFlag is true!");

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value ||
                      ((std::is_same_v<ElementC, half> || std::is_same_v<ElementC, bfloat16_t> ||
                          std::is_same_v<ElementC, float>) && tla::detail::iszN<ElementC, LayoutC>::value),
        "LayoutC only supports zN in half or bfloat16 or float, RowMajor in all dtype yet!");

    // Check L1TileShape
    static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static constexpr uint32_t _32B = 32*8; // in bits
    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 2201)
    static_assert(L1_TILE_M * SizeOfBits<ElementA>::value % _32B == 0, "L1TileShape::M must be 32B aligned.");
    static_assert(L1_TILE_K * SizeOfBits<ElementA>::value % _32B == 0, "L1TileShape::K must be 32B aligned.");
    static_assert(L1_TILE_K * SizeOfBits<ElementB>::value % _32B == 0, "L1TileShape::K must be 32B aligned.");
    static_assert(L1_TILE_N * SizeOfBits<ElementB>::value % _32B == 0, "L1TileShape::N must be 32B aligned.");
    static_assert(L0_TILE_K * SizeOfBits<ElementB>::value % _32B == 0, "L0TileShape::K must be 32B aligned.");
#endif

    static_assert((!HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 8) || (HAS_BIAS && (L1A_STAGES + L1B_STAGES) <= 7), 
        "L1 Buffer overflow: Exceeds the supported range of EVENT(0~7)");

    static_assert((!HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 8) || (HAS_BIAS && (L0A_STAGES + L0B_STAGES) <= 7), 
        "L0 Buffer overflow: Exceeds the supported range of EVENT_ID(0~7)");

    static constexpr auto L1A_LAYOUT =
        tla::MakeLayout<ElementA, LayoutTagL1A>(tla::Int<L1_TILE_M>{}, tla::Int<L1_TILE_K>{});
    static constexpr auto L1B_LAYOUT =
        tla::MakeLayout<ElementB, LayoutTagL1B>(tla::Int<L1_TILE_K>{}, tla::Int<L1_TILE_N>{});
    static constexpr auto L1BIAS_LAYOUT = tla::MakeLayout(tla::Int<L1_TILE_N>{});
    static constexpr auto L0BIAS_LAYOUT = tla::MakeLayout(tla::Int<L0_TILE_N>{});

    // When enabling L1 resident mode, restore the pointer and coordinates that record the last state
    // to the initial state. if two blockmmad instances need to be consecutively invoked at the kernel layer,
    // RestoreStatus() must be inserted between them.
    CATLASS_DEVICE
    void RestoreStatus()
    {
        for (int i = 0; i < L1A_STAGES; ++i) {
            lastAddrA[i] = nullptr;
            lastCoordA[i] = MatrixCoord{0U, 0U};
        }
        for (int i = 0; i < L1B_STAGES; ++i) {
            lastAddrB[i] = nullptr;
            lastCoordB[i] = MatrixCoord{0U, 0U};
        }
    }

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        if ASCEND_IS_AIC {
            uint32_t l1AOffset = l1BufAddrStart;
            uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1A_STAGES;
            // Init buffers
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
                // Assign event ID for each stages
                l1AEventList[i] = i;
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
                // Assign event ID for each stages
                l1BEventList[i] = i + L1A_STAGES;
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
                // Assign event ID for each stages
                l0AEventList[i] = i;
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
                // Assign event ID for each stages
                l0BEventList[i] = i + L0A_STAGES;
            }
            if constexpr(!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
                    l0CEventList[i] = i;
                }
            } else {
                l0CTensorList[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            }
            if constexpr (HAS_BIAS) {
                uint32_t l1BiasOffset = l1BOffset + L1B_TILE_SIZE * L1B_STAGES;
                l1BiasTensor = resource.l1Buf.template GetBufferByByte<uint8_t>(l1BiasOffset);
                l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla() {}

    CATLASS_DEVICE
    void preSetFlags() {

        if ASCEND_IS_AIC {
            // use HF32 when USE_HF32_MODE is true
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(true);
            } else {
                AscendC::SetHF32Mode(false);
            }
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(true);
            }
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            if constexpr(!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
                }
            }
            if constexpr (HAS_BIAS) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }

            if constexpr (ENABLE_L1_RESIDENT) {
                RestoreStatus();
            }
        }
    }

    CATLASS_DEVICE
    void finalWaitFlags() {
        if ASCEND_IS_AIC {
            if constexpr (USE_HF32_MODE) {
                AscendC::SetHF32Mode(false);
            }
            if constexpr (ENABLE_UNIT_FLAG && tla::detail::isRowMajor<LayoutC>::value) {
                AscendC::SetMMLayoutTransform(false);
            }
            for (uint32_t i = 0; i < L1A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            for (uint32_t i = 0; i < L1B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            if constexpr(!ENABLE_UNIT_FLAG) {
                for (uint32_t i = 0; i < L0C_STAGES; i++) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
                }
            }
            if constexpr (HAS_BIAS) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
            }
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    template <class TensorA, class TensorB, class TensorC, class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, GemmCoord const &actualShape, Arch::CrossCoreFlag vecDone,
        TensorBias const &tensorBias = {})
    {
        // Check L1TileShape
        if constexpr (HAS_BIAS) {
            static constexpr uint32_t BIAS_BUF_SIZE = L0_TILE_N * sizeof(ElementAccumulator);
            static constexpr uint32_t L1BIAS_SIZE = L1_TILE_N * sizeof(ElementBias);
            static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE,
                "BIAS_BUF_SIZE exceeding the BT space! Reduce L0_TILE_N");
            static_assert(L1A_TILE_SIZE * L1A_STAGES + L1B_TILE_SIZE * L1B_STAGES + L1BIAS_SIZE <= ArchTag::L1_SIZE,
                "L1TileShape exceeding the L1 space!");
        }

        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 2201)
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToGm<TensorC>;
        CopyL0CToGm copyL0CToDst;
#endif        
#if (defined (CATLASS_ARCH) && CATLASS_ARCH == 3510)
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
#endif        

        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();

        uint32_t mL1Actual = mBlockActual;
        if constexpr (std::is_same_v<ArchTag, Arch::AtlasA2>) {
            // Avoid using the gemv mode in mmad
            if (mL1Actual == 1) {
                mL1Actual = 16;
            }
        }
        uint32_t nL1Actual = nBlockActual;

        auto layoutInL0C = tla::MakeLayoutL0C(mL1Actual, nL1Actual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});
        auto tensorL0Bias = tla::MakeTensor(l0BiasTensor, L0BIAS_LAYOUT, Arch::PositionBias{});

        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);
        // load first matrix A tile from GM to L1
        if constexpr (DispatchPolicy::WAIT_A_BEFORE_LOAD) {
            // A 依赖 vecDone 输出（A=Vwork 为 V1 输出）：把等待放到 A 装载（MTE2
            // 异步发射）之前，确保依赖数据已写完。
            Arch::CrossCoreWaitFlag(vecDone);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        auto tensorL1A = tla::MakeTensor(l1ATensorList[l1AListId], L1A_LAYOUT, Arch::PositionL1{});
        auto tensorTileA = GetTileA(tensorA, 0, 0, mBlockActual, kL1Actual);
        if constexpr (ENABLE_L1_RESIDENT) {
            // If the currently loaded GM pointer and block coordinates are the same as the last loaded ones,
            // skip this loading.
            if (lastAddrA[l1AListId] != tensorTileA.data().GetPhyAddr()
                || tla::get<0>(tensorTileA.coord()) != lastCoordA[l1AListId].row()
                || tla::get<1>(tensorTileA.coord()) != lastCoordA[l1AListId].column()) {
                copyGmToL1A(tensorL1A, tensorTileA);
                lastCoordA[l1AListId] = MatrixCoord{tla::get<0>(tensorTileA.coord()), tla::get<1>(tensorTileA.coord())};
                lastAddrA[l1AListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                    tensorTileA.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1A(tensorL1A, tensorTileA);
        }
        if constexpr (!DispatchPolicy::WAIT_A_BEFORE_LOAD) {
            Arch::CrossCoreWaitFlag(vecDone);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        // load first matrix B tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        auto tensorL1B = tla::MakeTensor(l1BTensorList[l1BListId], L1B_LAYOUT, Arch::PositionL1{});
        auto tensorTileB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(kL1Actual, nBlockActual));
        if constexpr (ENABLE_L1_RESIDENT) {
            if (lastAddrB[l1BListId] != tensorTileB.data().GetPhyAddr()
                || tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListId].row()
                || tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListId].column()) {
                copyGmToL1B(tensorL1B, tensorTileB);
                lastCoordB[l1BListId] = MatrixCoord{tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                lastAddrB[l1BListId] = const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                    tensorTileB.data().GetPhyAddr()
                );
            }
        } else {
            copyGmToL1B(tensorL1B, tensorTileB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
            using CopyGmToL1Bias = typename TileCopy::template CopyGmToL1Bias<TensorBias>;
            CopyGmToL1Bias copyGmToL1Bias;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
            auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
            auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
            copyGmToL1Bias(tensorL1Bias, tensorBias);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        }

        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(mL1Actual);
        uint32_t nL0Loop = CeilDiv<L0_TILE_N>(nL1Actual);

        // main loop
        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
            uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
            uint32_t kL1ActualNext{0};
            // preload next tile from GM to L1
            if (kL1Idx < kL1Loop - 1) {
                uint32_t kL1IdxNext = kL1Idx + 1;
                kL1ActualNext = (kL1IdxNext < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);

                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1AListIdNext];
                auto l1BTensor = l1BTensorList[l1BListIdNext];
                auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
                auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
                // Get GM tile for next stage
                auto tensorTileA = GetTileA(tensorA, 0, kL1IdxNext * L1_TILE_K, mBlockActual, kL1ActualNext);
                auto tensorTileB = GetTile(tensorB, tla::MakeCoord(kL1IdxNext * L1_TILE_K, 0),
                    tla::MakeShape(kL1ActualNext, nBlockActual));

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                if constexpr (ENABLE_L1_RESIDENT) {
                    if (lastAddrA[l1AListIdNext] != tensorTileA.data().GetPhyAddr()
                        || tla::get<0>(tensorTileA.coord()) != lastCoordA[l1AListIdNext].row()
                        || tla::get<1>(tensorTileA.coord()) != lastCoordA[l1AListIdNext].column()) {
                        copyGmToL1A(tensorL1A, tensorTileA);
                        lastCoordA[l1AListIdNext] =
                            MatrixCoord{tla::get<0>(tensorTileA.coord()), tla::get<1>(tensorTileA.coord())};
                        lastAddrA[l1AListIdNext] =
                            const_cast<__gm__ typename AscendC::GlobalTensor<ElementA>::PrimType *>(
                                tensorTileA.data().GetPhyAddr()
                            );
                    }
                } else {
                    copyGmToL1A(tensorL1A, tensorTileA);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                if constexpr (ENABLE_L1_RESIDENT) {
                    if (lastAddrB[l1BListIdNext] != tensorTileB.data().GetPhyAddr()
                        || tla::get<0>(tensorTileB.coord()) != lastCoordB[l1BListIdNext].row()
                        || tla::get<1>(tensorTileB.coord()) != lastCoordB[l1BListIdNext].column()) {
                        copyGmToL1B(tensorL1B, tensorTileB);
                        lastCoordB[l1BListIdNext] =
                            MatrixCoord{tla::get<0>(tensorTileB.coord()), tla::get<1>(tensorTileB.coord())};
                        lastAddrB[l1BListIdNext] =
                            const_cast<__gm__ typename AscendC::GlobalTensor<ElementB>::PrimType *>(
                                tensorTileB.data().GetPhyAddr()
                            );
                    }
                } else {
                    copyGmToL1B(tensorL1B, tensorTileB);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1AListId];
            auto l1BTensor = l1BTensorList[l1BListId];
            tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
            tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
            // Get the loop nums on L0
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

            for (int mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0_TILE_M : (mL1Actual - mL0Idx * L0_TILE_M);

                for (int kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    auto layoutAInL0 = tla::MakeLayout<ElementA, LayoutTagL0A>(mL0Actual, kL0Actual);
                    auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
                    // Locate the current tile of matrix A on L1
                    auto tensorTileL1A = GetTileA(tensorL1A, mL0Idx * L0_TILE_M, kL0Idx * L0_TILE_K, mL0Actual, kL0Actual);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mL0Idx == 0) && (kL0Idx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                    }

                    bool initC = ((kL1Idx == 0) && (kL0Idx == 0));
                    for (int nL0Idx = 0; nL0Idx < nL0Loop; nL0Idx++) {
                        uint32_t nL0Actual = (nL0Idx < nL0Loop - 1) ? L0_TILE_N : (nL1Actual - nL0Idx * L0_TILE_N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        auto layoutBInL0 = tla::MakeLayout<ElementB, LayoutTagL0B>(kL0Actual, nL0Actual);
                        auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
                        // Locate the current tile of matrix B on L1
                        auto tensorTileL1B = GetTile(tensorL1B,
                                                     tla::MakeCoord(kL0Idx * L0_TILE_K, nL0Idx * L0_TILE_N),
                                                     tla::MakeShape(kL0Actual, nL0Actual));

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((mL0Idx == 0) && (kL0Idx == 0) && (nL0Idx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(tensorL0B, tensorTileL1B);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                        }

                        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                            if (initC) {
                                if (nL0Idx == 0) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                                auto l1Bias = l1BiasTensor.template ReinterpretCast<ElementBias>();
                                auto tensorL1Bias = tla::MakeTensor(l1Bias, L1BIAS_LAYOUT, Arch::PositionL1{});
                                auto tensorTileL1Bias = GetTile(tensorL1Bias,
                                                                tla::MakeCoord(nL0Idx * L0_TILE_N),
                                                                tla::MakeShape(nL0Actual));
                                // Load bias to l0 biasTable
                                copyL1ToBT(tensorL0Bias, tensorTileL1Bias);
                                if (nL0Idx == nL0Loop - 1) {
                                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                                }
                            }
                        }

                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                        // Locate the current tile on L0C
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                     tla::MakeCoord(mL0Idx * L0_TILE_M, nL0Idx * L0_TILE_N),
                                                     tla::MakeShape(mL0Actual, nL0Actual));

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kL1Idx == kL1Loop - 1) && (mL0Idx == mL0Loop - 1) &&
                                (kL0Idx == kL0Loop - 1) && (nL0Idx == nL0Loop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }

                        if constexpr (HAS_BIAS && !std::is_same_v<TensorBias, EmptyClass>) {
                            if (initC) {
                                tileMmad(tensorTileL0C, tensorL0A, tensorL0B, tensorL0Bias,
                                    mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                            } else {
                                tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                                    mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                            }
                        } else {
                            tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                                mL0Actual, nL0Actual, kL0Actual, initC, unitFlag);
                        }

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1AListId = l1AListIdNext;
            l1BListId = l1BListIdNext;
            kL1Actual = kL1ActualNext;
        }

        // copy block out
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
            copyL0CToDst(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        } else {
            copyL0CToDst(tensorC, tensorL0C, 0b11);
        }
    }

protected:
    template<class TensorA>
    CATLASS_DEVICE auto GetTileA(TensorA &tensorA, uint32_t mIndex, uint32_t kIndex, uint32_t mSize, uint32_t kSize)
    {
        if constexpr(tla::detail::isVector<LayoutA>::value) {
            return GetTile(tensorA, tla::MakeCoord(kIndex), tla::MakeShape(kSize));
        } else {
            return GetTile(tensorA, tla::MakeCoord(mIndex, kIndex), tla::MakeShape(mSize, kSize));
        }
    }

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    AscendC::LocalTensor<uint8_t> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    __gm__ typename AscendC::GlobalTensor<ElementA>::PrimType* lastAddrA[L1A_STAGES];
    __gm__ typename AscendC::GlobalTensor<ElementB>::PrimType* lastAddrB[L1B_STAGES];
    MatrixCoord lastCoordA[L1A_STAGES];
    MatrixCoord lastCoordB[L1B_STAGES];
    
    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
};

}  // namespace Catlass::Gemm::Block

namespace Catlass::Gemm::Kernel {

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE,
    bool USE_EXP2_,
    bool STATE_V_FIRST_
>
class GDNFwdHKernelPreload {
public:
    
    using ArchTag = Arch::AtlasA2;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHPreloadCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHPreloadVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaPreload<ArchTag, true, false>;
    using L1TileShapeTla = Shape<_128, _128, _128>;
    using L0TileShapeTla = L1TileShapeTla;

    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;

    // cube 1
    // vfirst 时 state 物理 [V,K]：C1 的 H 以 [K,V] ColumnMajor 视角消费（零搬运）；
    // LayoutH 仅作用于 C1 输入视图，C2 输出恒 RowMajor（vfirst 时形状翻转）。
    using LayoutH = std::conditional_t<STATE_V_FIRST_, Catlass::layout::ColumnMajor,
                                       Catlass::layout::RowMajor>;
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, LayoutH, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;

    // 等待前置策略实例（WAIT_A_BEFORE_LOAD=true）：见本地 MmadPingpongTlaPreload 注释
    using DispatchPolicyKVVFirst = Gemm::MmadPingpongTlaPreload<ArchTag, true, false, 1, false, 2, 2, 2, 2, true>;
    using BlockMmadKVVFirst = Gemm::Block::BlockMmadTla<DispatchPolicyKVVFirst, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;


    // vec 1
    using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasGDNFwdHVnewPreload<USE_EXP2_>;
    using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, VworkType, FinalStateType>;

    // vec 2
    using DispatchPolicyGDNFwdHUpdate = Epilogue::EpilogueAtlasGDNFwdHUpdatePreload<USE_EXP2_>;
    using EpilogueGDNFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHUpdate, HType, GType, HType, HworkType, FinalStateType>;

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsetsPreload;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;
    
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;

    uint32_t seqlen{0};
    uint32_t kNumHead{0};
    uint32_t vNumHead{0};
    uint32_t kHeadDim{0};
    uint32_t vHeadDim{0};
    uint32_t chunkSize{0};
    bool useInitialState{false};
    bool storeFinalState{false};
    uint32_t isVariedLen{0};
    uint32_t shapeBatch{0};
    uint64_t vWorkspaceOffset{0};
    uint64_t vUpdateWorkspaceOffset{0};
    uint64_t hWorkspaceOffset{0};
    uint64_t numChunksWorkspaceOffset{0};
    
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;

    AscendC::GlobalTensor<int64_t> gmNumChunks;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline GDNFwdHKernelPreload() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state, GM_ADDR cu_seqlens,
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {
        
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        vWorkspaceOffset = gdnFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = gdnFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = gdnFwdHTilingData->hWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;
        
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));

        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, tiling, user);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, tiling, user);
        }
    }
    
    __aicore__ inline void Process() {

        if ASCEND_IS_AIC {
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadWH blockMmadWH(resource);
            BlockMmadKV blockMmadKV(resource);
            BlockMmadKVVFirst blockMmadKVVFirst(resource);

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            // C1 的 H per-tile 视角：vfirst 时以 [K,V] 视角索引 [V,K] 物理 state（零搬运）
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(kHeadDim, vHeadDim);
            auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);

            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto vworkLayout = tla::MakeLayout<ElementV, LayoutV>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            // C2 的 hWorkspace 输出恒 RowMajor；vfirst 时形状翻转 [V,K]（等价转置式产出）
            auto hworkLayout = STATE_V_FIRST_
                ? tla::MakeLayout<ElementHWork, Catlass::layout::RowMajor>(vHeadDim, kHeadDim)
                : tla::MakeLayout<ElementHWork, Catlass::layout::RowMajor>(kHeadDim, vHeadDim);
            AscendC::SyncAll<false>();    
            uint32_t currStage = 0; // 0: C1, 1: C2
            blockMmadWH.preSetFlags();
            while (cubeBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* C1: v_work = w @ h[i] */                    
                    cubeBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }

                        const GDNFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                        int64_t cube1OffsetW = cube1Offsets.wOffset;
                        int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                        int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                        auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                        auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                        auto tensorV = tla::MakeTensor(gmVWorkspace[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                        GemmCoord cube1Shape {cube1Offsets.blockTokens, vHeadDim, kHeadDim};
                        auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                        auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                        auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                        // vec2Done 保证同流上一迭代 V2 的 h 写/读完成，按槽位（流）
                        // 配对而非按 i：流迁移换 i 后按 i 等待会漏等，读写 h 竞态。
                        blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape, cubeBlockScheduler.vec2Done[cubeBlockScheduler.GetStreamSlot(i)]);
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);
                    }
                } else {
                    /* C2: h[i+1] = k.T @ v_work */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);

                        if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                            // step 3: h[i+1] = k.T @ v_work
                            int64_t cube2OffsetK = cube2Offsets.wkOffset;
                            int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                            int64_t cube2OffsetH = cube2Offsets.hWorkOffset;
                            auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2OffsetH], hworkLayout, Catlass::Arch::PositionGM{});
                            if constexpr (STATE_V_FIRST_) {
                                // 等价转置式：(k^T@v)^T = v^T@k，A/B 换视角后 L0C 直接
                                // 产出物理 [V,K] 零转置开销。A=Vwork 依赖 V1 输出，
                                // 用等待前置变体 BlockMmadKVVFirst 保证 A 读序。
                                auto vworkALayout = tla::MakeLayout<ElementV, Catlass::layout::ColumnMajor>(vHeadDim, chunkSize);
                                auto tensorVworkA = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vworkALayout, Catlass::Arch::PositionGM{});
                                auto kRowLayout = tla::MakeLayout<ElementK, Catlass::layout::RowMajor>(chunkSize, kHeadDim);
                                auto tensorKRow = tla::MakeTensor(gmK[cube2OffsetK], kRowLayout, Catlass::Arch::PositionGM{});
                                GemmCoord cube2ShapeV{vHeadDim, kHeadDim, cube2Offsets.blockTokens};
                                auto tensorBlockA = GetTile(tensorVworkA, tla::MakeCoord(0, 0), tla::MakeShape(cube2ShapeV.m(), cube2ShapeV.k()));
                                auto tensorBlockB = GetTile(tensorKRow, tla::MakeCoord(0, 0), tla::MakeShape(cube2ShapeV.k(), cube2ShapeV.n()));
                                auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2ShapeV.m(), cube2ShapeV.n()));
                                blockMmadKVVFirst(tensorBlockA, tensorBlockB, tensorBlockHwork, cube2ShapeV, cubeBlockScheduler.vec1Done);
                            } else {
                                auto tensorK = tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                                auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vworkLayout, Catlass::Arch::PositionGM{});
                                GemmCoord cube2Shape{kHeadDim, vHeadDim, cube2Offsets.blockTokens};
                                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                                auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                                auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                                blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape, cubeBlockScheduler.vec1Done);
                            }
                        } else {
                            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
                    }
                }
                currStage ^= 0x01;
            }
            if constexpr (STATE_V_FIRST_) {
                blockMmadKVVFirst.finalWaitFlags();
            } else {
                blockMmadKV.finalWaitFlags();
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);

        }

        if ASCEND_IS_AIV {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
            uint32_t totalChunks = isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
            uint32_t transferCount = isVariedLen ? (vecBlockScheduler.tokenBatch * vNumHead / coreNum) : (shapeBatch * vNumHead / coreNum);
            uint32_t remainderFlag = isVariedLen ? (((vecBlockScheduler.tokenBatch * vNumHead) % coreNum) != 0): (((shapeBatch * vNumHead) % coreNum) != 0);
            uint32_t step = transferCount + remainderFlag;
            uint32_t stateBlockSize = kHeadDim * vHeadDim;
            uint32_t pingpongFlag = 1;
            uint32_t start = coreIdx * step;
            uint32_t end = start + step;
            uint32_t maxLimit = isVariedLen ? vecBlockScheduler.tokenBatch * vNumHead : shapeBatch * vNumHead;
            uint32_t realEnd = min(end, maxLimit);
            if (useInitialState) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                AscendC::Duplicate(hUbTensorPing, static_cast<ElementH>(0), stateBlockSize);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            }
            for(uint32_t initialStateBlockOffset = start ;initialStateBlockOffset >= start && initialStateBlockOffset < realEnd;initialStateBlockOffset++)
            {   
                uint32_t batchIdx = initialStateBlockOffset / vNumHead;
                uint32_t vHeadIdx = initialStateBlockOffset % vNumHead;
                uint32_t chunkOffset = isVariedLen ? gmNumChunks.GetValue(batchIdx) : 0; 
                uint64_t initialStateOffset = initialStateBlockOffset * stateBlockSize;
                uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                uint64_t hOffset = ((shapeBatchIdx * totalChunks + chunkOffset) * vNumHead + vHeadIdx) * stateBlockSize;
                if (useInitialState) {
                    AscendC::LocalTensor<ElementInitialState> stateUbTensor = pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                    AscendC::LocalTensor<ElementH> hUbTensor = pingpongFlag ? hUbTensorPing : hUbTensorPong;
                    auto event_id = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    if constexpr(!std::is_same<ElementInitialState, ElementH>::value) {
                        AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateBlockSize);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                        AscendC::Cast(hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT, stateBlockSize);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                        AscendC::DataCopy(gmH[hOffset], hUbTensor, stateBlockSize);
                    } else {
                        AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateBlockSize);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                        AscendC::DataCopy(gmH[hOffset], stateUbTensor, stateBlockSize);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                    pingpongFlag = 1 - pingpongFlag;
                } else {
                    AscendC::DataCopy(gmH[hOffset], hUbTensorPing, stateBlockSize);
                }


            }
                
            if (useInitialState) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            
            AscendC::SyncAll<false>();

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);
            EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
            uint32_t pongBaseEvent = 4;

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); // preset v
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset v
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);

            uint32_t currStage = 0; // 0: V1, 1: V2
            while (vecBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* V1:
                     * gmV = gmU - gmVWorkspace
                     * g_buf = gmG[-1] - gmG
                     * g_buf = exp(g_buf)
                     * gmVWorkspace = g_buf * gmV
                     */
                    vecBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        epilogueGDNFwdHVnew(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset],
                            gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset],
                            vec1Offsets.blockTokens, vHeadDim,
                            vecBlockScheduler.cube1Done, vecBlockScheduler.vec1Done,
                            // isPing 按槽位取（事件组/UB 组与 vWork 槽同键，
                            // 防流迁移换组断链，见 GetStreamSlot 注释）
                            vec1Offsets.isInitialState, storeFinalState,
                            (vecBlockScheduler.GetStreamSlot(i) == 0)
                        );
                    }
                } else {
                    /* V2: h[i+1] += h_work if i < num_chunks - 1 else None */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);

                        if (vecBlockScheduler.NeedProcessStage2(stream)) {
                            // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                            epilogueGDNFwdHUpdate(
                                gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                gmG[vec2Offsets.gOffset],
                                gmH[vec2Offsets.hSrcOffset],
                                gmHWorkspace[vec2Offsets.hWorkOffset],
                                vec2Offsets.blockTokens, kHeadDim, vHeadDim, vecBlockScheduler.cube2Done,
                                // isPing 按槽位取（同 V1，防流迁移换组断链）
                                vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState,
                                (vecBlockScheduler.GetStreamSlot(i) == 0)
                            );
                        } else {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                        }
                        // 置位侧同 C1 等待侧按槽位配对（见 C1 处注释）
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[vecBlockScheduler.GetStreamSlot(i)]);
                    }
                }
                currStage ^= 0x01;
            }

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); // preset v
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset v
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);

        }
    }

};

}

// ---- 自原 chunk_fwd_h.cpp 移入：gated 复用谓词与入口（RunFwdHTyped 分流调用）
namespace GDN {
// gated 复用谓词：g 模式即复用（vfirst 经 STATE_V_FIRST 模板维支持；storeFinal
// 与 fp32/bf16 gate/state 均可）；g dtype 由 host 入口唯一校验。
template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline constexpr bool FwdHGatedReusable()
{
    // 域不限（2026-11-10 exp 域复用释放）：log2（USE_EXP2=true）/ln（false）均
    // 可——域经模板维进 gated kernel（epilogue 内 LN2 换算 if constexpr 取舍）；
    // 与 host 侧 gatedEligible 两处同条件。
    return CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G;
}

// gated 复用分支：构造 gated tiling 镜像（workspace gatedTilingOffset 段，各 AIC
// 写入同值）后调 preload kernel；USE_EXP2 携带 g 域，STATE_V_FIRST 携带布局。
template <typename GateT, typename StateT, bool USE_EXP2, bool STATE_V_FIRST>
__aicore__ inline void RunFwdHGatedReuse(const FwdHKernelArgs &args)
{
    using DataT = bfloat16_t;
    using GatedKernel = Catlass::Gemm::Kernel::GDNFwdHKernelPreload<DataT, GateT, StateT, float, USE_EXP2, STATE_V_FIRST>;
    __gm__ uint8_t *user = args.workspace;
    __gm__ ChunkGatedDeltaRuleFwdHTilingData *gtiling = reinterpret_cast<
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *>(user + args.tiling.gatedTilingOffset);
    {
        gtiling->batch = args.tiling.batch;
        gtiling->seqlen = args.tiling.seqlen;
        gtiling->kNumHead = args.tiling.kNumHead;
        gtiling->vNumHead = args.tiling.vNumHead;
        gtiling->kHeadDim = args.tiling.kHeadDim;
        gtiling->vHeadDim = args.tiling.vHeadDim;
        gtiling->chunkSize = args.tiling.chunkSize;
        gtiling->useInitialState = args.tiling.useInitialState;
        gtiling->storeFinalState = args.tiling.storeFinalState;
        gtiling->dataType = 1; // bf16 输入
        gtiling->gDataType = std::is_same_v<GateT, float> ? 2 : 1; // 1=bf16, 2=fp32
        gtiling->stateDataType = std::is_same_v<StateT, float> ? 2 : 1;
        gtiling->isVariedLen = args.tiling.isVariedLen;
        gtiling->shapeBatch = args.tiling.shapeBatch;
        gtiling->tokenBatch = args.tiling.tokenBatch;
        gtiling->useG = true;
        gtiling->useGk = false;
        gtiling->vWorkspaceOffset = args.tiling.vWorkspaceOffset;
        gtiling->vUpdateWorkspaceOffset = args.tiling.vUpdateWorkspaceOffset;
        gtiling->kDecayWorkspaceOffset = 0;
        gtiling->hWorkspaceOffset = args.tiling.hWorkspaceOffset;
        gtiling->numSeqWorkspaceOffset = args.tiling.gatedNumSeqOffset;
        gtiling->numChunksWorkspaceOffset = args.tiling.gatedNumChunksOffset;
    }
    GatedKernel gatedKernel;
    gatedKernel.Init(args.k, args.w, args.u, args.g, args.initialState,
                     args.cuSeqlens, args.h, args.vNew,
                     args.finalState,
                     reinterpret_cast<GM_ADDR>(gtiling),
                     user);
    gatedKernel.Process();
}

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline bool RunFwdHGatedDispatch(const FwdHKernelArgs &args)
{
    if constexpr (FwdHGatedReusable<GateT, CompilePolicy, STATE_V_FIRST>()) {
        using GatedStateT = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
        RunFwdHGatedReuse<GateT, GatedStateT, CompilePolicy::USE_EXP2, STATE_V_FIRST>(args);
        return true;
    }
    return false;
}
} // namespace GDN

#endif // ARCH22_CHUNK_FWD_H_UTILS_H
