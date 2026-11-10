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
 * gated 复用谓词/入口已归位回 chunk_fwd_h.cpp（优化前入口位置）；
 * 内核类已拆分为双引擎：ChunkFwdHGatedCubeArch22（arch22/chunk_fwd_h_cube.h）
 * 与 ChunkFwdHGatedVecArch22（arch22/chunk_fwd_h_vec.h），本文件保留
 * 镜像/policies/epilogue 特化/调度器基座。纯搬移，无代码语义变化。
 */

#ifndef ARCH22_CHUNK_FWD_H_UTILS_H
#define ARCH22_CHUNK_FWD_H_UTILS_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"

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

// ---- 原 gemm/kernel/gdn_fwd_h_kernel_preload.hpp 的 catlass 头（补齐主模板声明等前置）
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"

// ---- 内联自原 epilogue/gdn_fwd_h_epilogue_policies.hpp
namespace Catlass::Epilogue {

struct EpilogueAtlasGDNFwdHVnew {
    using ArchTag = Arch::AtlasA2;
};

struct EpilogueAtlasGDNFwdHUpdate {
    using ArchTag = Arch::AtlasA2;
};

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

#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

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
    using HUpdateElementInput = typename HUpdateInputType_::Element;
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
        gBrcbUbTensor_ping = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
        vNewOutputUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);
        vNewDecayUbTensor_ping = resource.ubBuf.template GetBufferByByte<VElementOutput>(PING_BUF_2_OFFSET);

        uUbTensor_pong = resource.ubBuf.template GetBufferByByte<UElementInput>(PONG_BUF_2_OFFSET);
        wsUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_BUF_0_OFFSET);
        gUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_BUF_OFFSET);
        gLastUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PONG_G_SUB_BUF_OFFSET);
        gInputUbTensor_pong = resource.ubBuf.template GetBufferByByte<GElementInput>(PONG_G_SUB_BUF_OFFSET);
        gBrcbUbTensor_pong = resource.ubBuf.template GetBufferByByte<float>(PING_BUF_0_OFFSET);
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
        // 当前场景内部一定连续
        // k [B, H, T, D]
        // g [B, H, T]
        // 在外部offset的基础上进一步offset
        // 当前asset kdim == vHeadDim
        int64_t offsetK = mOffset * nvActual + nOffset;
        int64_t offsetD = 0; // 因为要用最后一个数减去之前所有，所以全部读入

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
        AscendC::LocalTensor<float> gBrcbUbTensor = isPing ? gBrcbUbTensor_ping : gBrcbUbTensor_pong;
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
    AscendC::LocalTensor<float> gBrcbUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_ping;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_ping;

    AscendC::LocalTensor<UElementInput> uUbTensor_pong;
    AscendC::LocalTensor<float> wsUbTensor_pong;
    AscendC::LocalTensor<float> gUbTensor_pong;
    AscendC::LocalTensor<float> gLastUbTensor_pong;
    AscendC::LocalTensor<GElementInput> gInputUbTensor_pong;
    AscendC::LocalTensor<float> gBrcbUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewOutputUbTensor_pong;
    AscendC::LocalTensor<VElementOutput> vNewDecayUbTensor_pong;

    AscendC::LocalTensor<uint8_t> shareBuffer_;

};
}

#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_preload.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"

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
    // for debug
    uint32_t batchIdx{0};
    uint32_t headIdx{0};
    uint32_t chunkIdx{0};

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
    uint32_t shapeBatch{0};
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
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user, uint32_t coreIdx, uint32_t coreNum) {
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
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
        offset.batchIdx = stream.batchIdx; 
        offset.headIdx = stream.vHeadIdx; 
        offset.chunkIdx = stream.chunkIdx;
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
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdHPreload::Init(cu_seqlens, chunk_indices, tiling, user, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }

};

struct BlockSchedulerGdnFwdHPreloadVec : public BlockSchedulerGdnFwdHPreload {
    CATLASS_DEVICE
    BlockSchedulerGdnFwdHPreloadVec() {}

    CATLASS_DEVICE
    void Init(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR tiling, GM_ADDR user) {
        BlockSchedulerGdnFwdHPreload::Init(cu_seqlens, chunk_indices, tiling, user, AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(), AscendC::GetBlockNum());
    }

};

}  // namespace Catlass::Gemm::Block

#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"

#endif // ARCH22_CHUNK_FWD_H_UTILS_H
