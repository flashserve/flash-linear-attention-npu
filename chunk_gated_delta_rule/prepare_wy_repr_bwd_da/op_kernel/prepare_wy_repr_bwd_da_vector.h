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
 * \file prepare_wy_repr_bwd_da_vector.h
 * \brief
 */

#ifndef PREPARE_WY_REPR_BWD_DA_VECTOR_H
#define PREPARE_WY_REPR_BWD_DA_VECTOR_H

using namespace AscendC;

template <typename kType, typename betaType>
class PrepareWyReprBwdDAVectorProcess {
public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdDAVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR mask_, GM_ADDR dA_, GM_ADDR workspace_);
    __aicore__ inline void Init(const PrepareWyReprBwdDaTilingData& tiling, AscendC::TPipe *pipe_);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessVBeta();
    __aicore__ inline void ProcessKBetaG();
    __aicore__ inline void ProcessMDuDw();
    __aicore__ inline void ProcessG();
private:
    uint64_t B = 1;
    uint64_t T = 2048;
    uint64_t H = 4;
    uint64_t K = 128;
    uint64_t V = 128;
    uint64_t BT = 64;

    //这些会包在tiling data 里
    uint64_t dkRowNum = 32;
    uint64_t dkbRowNum = 32;
    uint64_t chunkSize = 64;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR mask;
    GM_ADDR dA;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;
private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<uint8_t> maskTensor;
    GlobalTensor<kType> dATensor;
    GlobalTensor<kType> dA1Tensor;
    GlobalTensor<kType> dA2Tensor;
    GlobalTensor<kType> dA4Tensor;
    GlobalTensor<kType> dA5Tensor;
    GlobalTensor<kType> dA6Tensor;
    GlobalTensor<kType> workSpaceTensor;
    GlobalTensor<kType> workSpace2Tensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> vInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> mdAInQue;
    TQue<AscendC::TPosition::VECIN, 1> mduInQue;
    TQue<AscendC::TPosition::VECIN, 1> mdwInQue;
    TQue<AscendC::TPosition::VECIN, 1> maskInQue;

    TQue<AscendC::TPosition::VECOUT, 1> vBetaOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> mduwOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> gOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> kBetaGOutQue;

    TBuf<AscendC::TPosition::VECCALC> vFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> kFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> mduFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> mdwFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> mduwCalFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> gFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> gFp32CalBuf;
    TBuf<AscendC::TPosition::VECCALC> gFp32BrcbRowBuf;
    TBuf<AscendC::TPosition::VECCALC> mdAFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> maskTBuf;
    TBuf<AscendC::TPosition::VECCALC> zeroFp32TBuf;
};

template <typename kType, typename betaType>
 __aicore__ inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::PrepareWyReprBwdDAVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR mask_, GM_ADDR dA_, GM_ADDR workspace_)
 :
    k(k_),
    v(v_),
    beta(beta_),
    A(A_),
    dw(dw_),
    du(du_),
    g(g_),
    mask(mask_),
    dA(dA_),
    workspace(workspace_)
    {};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::Init(const PrepareWyReprBwdDaTilingData& tiling, AscendC::TPipe *pipe_) {
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    workSpace2Tensor.SetGlobalBuffer((__gm__ kType *)workspace + B * H * T * BT);
    dA1Tensor.SetGlobalBuffer((__gm__ kType *)dA);
    dA2Tensor.SetGlobalBuffer((__gm__ kType *)workspace);
    dA4Tensor.SetGlobalBuffer((__gm__ kType *)dA);
    dA5Tensor.SetGlobalBuffer((__gm__ kType *)workspace);
    dA6Tensor.SetGlobalBuffer((__gm__ kType *)dA);
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::Process() {
    ProcessKBetaG();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // DumpTensor(dA1Tensor, 1, 4096);
    ProcessVBeta();
    pipe->Reset();
    AscendC::SyncAll<false>();
    // DumpTensor(dA2Tensor, 2, 4096);
    ProcessMDuDw();
    // DumpTensor(dA4Tensor, 4, 4096);
    pipe->Reset();
    AscendC::SyncAll<false>();
    // DumpTensor(dA5Tensor, 5, 4096);
    ProcessG();
    // DumpTensor(dA6Tensor, 6, 4096);
    return;
}


// k、beta和g 的计算
template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessKBetaG() {
    // todo 如果此处是浅融合， 此处应该为第二个CV，需要等待第一个MM的结果
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t rowNum = BT / 2;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    //init
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(gFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);

    //中间计算使用tmp
    pipe->InitBuffer(kBetaGOutQue, 2, rowNum * K * sizeof(kType));
    
    // 向外搬出的结果是workspace
    auto tensorKfp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetafp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbfp32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorGFp32 = gFp32Buf.Get<float32_t>();
    //
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for(uint32_t rowOffset = 0; rowOffset < chunkSize; rowOffset += rowNum) {
                // TODO 判断是否为与cude对应的vec核？
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                auto kOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * K;
                auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
                auto gOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
                //AscendC::printf("CrossCoreWaitFlag kOffset:%d, betaOffset:%d\n", kOffset, betaOffset);
                //copyin
                {
                    auto tensorKIn = kInQue.AllocTensor<kType>();
                    DataCopy(tensorKIn, kTensor[kOffset], K * rowNum);
                    auto tensorBetaIn = betaInQue.AllocTensor<betaType>();
                    DataCopy(tensorBetaIn, betaTensor[betaOffset], rowNum);
                    auto tensorGIn = gInQue.AllocTensor<betaType>();
                    DataCopy(tensorGIn, gTensor[gOffset], rowNum);

                    kInQue.EnQue(tensorKIn);
                    betaInQue.EnQue(tensorBetaIn);
                    gInQue.EnQue(tensorGIn);
                    //AscendC::printf("copyin\n");
                }
                //compute
                {
                    //AscendC::printf("150\n");
                    auto tensorKIn = kInQue.DeQue<kType>();
                    auto tensorBetaIn = betaInQue.DeQue<betaType>();
                    auto tensorGIn = gInQue.DeQue<betaType>();
                    auto tensorOut = kBetaGOutQue.AllocTensor<kType>();
                    // b_g处理
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorGFp32, tensorGIn, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorGFp32, tensorGIn, rowNum);
                    }
                    PipeBarrier<PIPE_V>();
                    Exp(tensorGFp32, tensorGFp32, rowNum);
                    
                    //AscendC::printf("153\n");
                    //cast fp32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetafp32, tensorBetaIn, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorBetafp32, tensorBetaIn, rowNum);
                    }
                    //AscendC::printf("205\n");
                    Cast(tensorKfp32, tensorKIn, RoundMode::CAST_NONE, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    Mul(tensorBetafp32, tensorBetafp32, tensorGFp32, rowNum);
                    PipeBarrier<PIPE_V>();
                    //brcb
                    // DumpTensor(tensorBetafp32, 0,  8 * rowNum);
                    Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum /8), {1, 8});
                    // DumpTensor(tensorBetaBrcbfp32, 0,  8 * rowNum);
                    PipeBarrier<PIPE_V>();
                    //AscendC::printf("213\n");
                    //mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorKfp32[perchannelResOffset], tensorKfp32[perchannelResOffset], tensorBetaBrcbfp32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    // DumpTensor(tensorKfp32, 1, K * rowNum);
                    PipeBarrier<PIPE_V>();

                    // 输出
                    Cast(tensorOut, tensorKfp32, RoundMode::CAST_RINT, K * rowNum);
                    kInQue.FreeTensor(tensorKIn);
                    betaInQue.FreeTensor(tensorBetaIn);
                    gInQue.FreeTensor(tensorGIn);
                    kBetaGOutQue.EnQue(tensorOut);
                    //AscendC::printf("compute\n");
                }
                //copyout
                {
                    auto tensorOut = kBetaGOutQue.DeQue<kType>();
                    DataCopy(workSpace2Tensor[kOffset], tensorOut, K * rowNum);
                    kBetaGOutQue.FreeTensor(tensorOut);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            //AscendC::printf("CrossCoreSetFlag\n");
        }
    }
    // DumpTensor(workSpace2Tensor, 111,  8192);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    return;
}

// v 和 beta 的计算
template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessVBeta() {
    // AscendC::printf("---hyh---274\n");
    uint32_t coreLoopsInB = CeilDiv(T, chunkSize);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t rowNum = BT / 2;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;

    //init
    pipe->InitBuffer(vInQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(vFp32Buf, rowNum * V * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(vBetaOutQue, 2, rowNum * V * sizeof(kType));
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    auto tensorVFp32 = vFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();

    // AscendC::printf("---yzq--loopIdx:%d\n", coreIdx);
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            // AscendC::printf("---hyh--CrossCoreWaitFlag-299 AIC finish\n");
            for(uint32_t rowOffset = 0; rowOffset < chunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                auto vOffset = ((bIdx * H + h) * T  + chunkIdx * chunkSize + rowOffset) * V;
                auto betaOffset = (bIdx * H + h) * T  + chunkIdx * chunkSize + rowOffset;
                // AscendC::printf("CrossCoreWaitFlag VOffset:%d, betaOffset:%d\n", vOffset, betaOffset);
                //copyin
                {
                    auto tensorVin = vInQue.AllocTensor<kType>();
                    DataCopy(tensorVin, vTensor[vOffset], V * rowNum);
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopy(tensorBetain, betaTensor[betaOffset], rowNum);
                    vInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                    //AscendC::printf("copyin\n");
                }
                //compute
                {
                    //AscendC::printf("150\n");
                    auto tensorVin = vInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorOut = vBetaOutQue.AllocTensor<kType>();
                    //AscendC::printf("153\n");
                    //cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    //AscendC::printf("159\n");
                    Cast(tensorVFp32, tensorVin, RoundMode::CAST_NONE, V * rowNum);
                    PipeBarrier<PIPE_V>();
                    //brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(rowNum /8), {1, 8});
                    // DumpTensor(tensorBetaBrcbFP32, 0,  8 * rowNum);
                    PipeBarrier<PIPE_V>();
                    //AscendC::printf("165\n");
                    //mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                    // 带着broadcast一起做了
                    while (perchannelResOffset < V) {
                        Mul(tensorVFp32[perchannelResOffset], tensorVFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    Cast(tensorOut, tensorVFp32, RoundMode::CAST_RINT, V * rowNum);
                    vInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);
                    vBetaOutQue.EnQue(tensorOut);
                    //AscendC::printf("compute\n");
                }
                //copyout
                {
                    auto tensorOut = vBetaOutQue.DeQue<kType>();
                    DataCopy(workSpace2Tensor[vOffset], tensorOut, V * rowNum);
                    vBetaOutQue.FreeTensor(tensorOut);
                    // AscendC::printf("kOffset:%ld, K * rowNum:%d\n", kOffset, K * rowNum);
                    // DumpTensor(workSpace2Tensor[vOffset], 222,  V * rowNum);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            // AscendC::printf("---hyh----CrossCoreSetFlag--363 AIV finish\n");
        }
    }
    // DumpTensor(workSpace2Tensor, 222, 8192);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    // AscendC::printf("---hyh----CrossCoreSetFlag--1\n");
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    // AscendC::printf("---hyh----CrossCoreSetFlag--2\n");
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    // AscendC::printf("---hyh----CrossCoreSetFlag--3\n");
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    // AscendC::printf("---hyh----CrossCoreSetFlag--4\n");
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessMDuDw() {
    uint32_t coreLoopsInB = CeilDiv(T, chunkSize);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t rowNum = BT / 2;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;

    pipe->InitBuffer(mduInQue, 2, rowNum * BT * sizeof(kType));
    pipe->InitBuffer(mdwInQue, 2, rowNum * BT * sizeof(kType));
    pipe->InitBuffer(mduFp32Buf, rowNum * BT * sizeof(float32_t));
    pipe->InitBuffer(mdwFp32Buf, rowNum * BT * sizeof(float32_t));
    pipe->InitBuffer(mduwCalFp32Buf, rowNum * BT * sizeof(float32_t));
    pipe->InitBuffer(maskTBuf, BT * BT / BIT_NUM_FOR_UINT8);
    pipe->InitBuffer(zeroFp32TBuf, BLOCK_SIZE);
    pipe->InitBuffer(mduwOutQue, 2, rowNum * BT * sizeof(kType));

    auto tensorMduFp32 = mduFp32Buf.Get<float32_t>();
    auto tensorMdwFp32 = mdwFp32Buf.Get<float32_t>();
    auto tensorDuwCalFP32 = mduwCalFp32Buf.Get<float32_t>();
    auto maskLocalTensor = maskTBuf.Get<uint8_t>();
    auto zeroFp32LocalTensor = zeroFp32TBuf.Get<float32_t>();

    maskTensor.SetGlobalBuffer((__gm__ uint8_t *)mask);
    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    copyParams.blockLen = BT * BT / BIT_NUM_FOR_UINT8;
    AscendC::DataCopyPad(maskLocalTensor, maskTensor, copyParams, {false, 0, 0, 0});
    // DumpTensor(maskLocalTensor, 1, BT * BT / BIT_NUM_FOR_UINT8);
    MTE2ToVSync();
    AscendC::Duplicate<float>(zeroFp32LocalTensor, float(0.0), BLOCK_SIZE / SIZE_FLOAT);

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for(uint32_t rowOffset = 0; rowOffset < chunkSize; rowOffset += rowNum) {
                // rowOffset/rowNum 代表是第几行
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                auto offset = ((bIdx * H + h) * T  + chunkIdx * chunkSize + rowOffset) * BT;
                //copyin
                {
                    auto tensorMduin = mduInQue.AllocTensor<kType>();
                    DataCopy(tensorMduin, dA2Tensor[offset], rowNum * BT);
                    auto tensorMdwin = mdwInQue.AllocTensor<kType>();;
                    DataCopy(tensorMdwin, dA1Tensor[offset], rowNum * BT);
                    mduInQue.EnQue(tensorMduin);
                    mdwInQue.EnQue(tensorMdwin);
                }
                //compute
                {
                    auto tensorMduin = mduInQue.DeQue<kType>();
                    auto tensorMdwin = mdwInQue.DeQue<kType>();
                    auto tensorMduwOut = mduwOutQue.AllocTensor<kType>();
                    //cast FP32
                    Cast(tensorMduFp32, tensorMduin, RoundMode::CAST_NONE, rowNum * BT);
                    Cast(tensorMdwFp32, tensorMdwin, RoundMode::CAST_NONE, rowNum * BT);
                    PipeBarrier<PIPE_V>();
                    // 相加：rowNum行 du + dw，元素个数为 rowNum * BT
                    AscendC::Add(tensorDuwCalFP32, tensorMduFp32, tensorMdwFp32, rowNum * BT);
                    PipeBarrier<PIPE_V>();
                    // 计算 dA4 = dA3 * mask 使用select
                    // dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride
                    AscendC::BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
                    AscendC::Select(tensorDuwCalFP32, maskLocalTensor[rowOffset * BT / BIT_NUM_FOR_UINT8],
                                    zeroFp32LocalTensor, tensorDuwCalFP32, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                                    CAL_NUM_FLOAT, rowNum * BT / CAL_NUM_FLOAT, repeatParams);
                    PipeBarrier<PIPE_V>();
                    AscendC::Cast(tensorMduwOut, tensorDuwCalFP32, AscendC::RoundMode::CAST_RINT, rowNum * BT);
                    mduInQue.FreeTensor(tensorMduin);
                    mdwInQue.FreeTensor(tensorMdwin);
                    mduwOutQue.EnQue(tensorMduwOut);

                }
                //copyout
                {
                    auto tensorMduwOut = mduwOutQue.DeQue<kType>();
                    DataCopy(dA4Tensor[offset], tensorMduwOut, rowNum * BT);
                    mduwOutQue.FreeTensor(tensorMduwOut);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    // DumpTensor(dA4Tensor, 4, 4096);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    return;
}


// g_sub_exp的处理
template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessG() {
    // 完成自我计算
    // 与matmul结果相加
    // 用A_mask完成计算
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t rowNum = BT / 2;
    uint32_t vecTaskIdx = 0;
    //init
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);
    // mdATensor.SetGlobalBuffer((__gm__ kType *)workspace);
    dATensor.SetGlobalBuffer((__gm__ kType *)dA);
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(mdAInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(gFp32CalBuf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(mdAFp32Buf, rowNum * sizeof(float32_t));
    
    //中间计算使用tmp
    pipe->InitBuffer(gOutQue, 2, rowNum * sizeof(kType));
    // 向外搬出最终的结果
    auto tensorGfp32 = gFp32Buf.Get<float32_t>();
    auto tensorGCalfp32 = gFp32CalBuf.Get<float32_t>();
    auto tensorMdAfp32 = mdAFp32Buf.Get<float32_t>();

    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            // ++vecTaskIdx;
            // if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
            //     // AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
            //     continue;
            // }
            // // auto gOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
            // auto gOffset = (bIdx * H + h) * T  + chunkIdx * BT;
            // // AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            // //copyin
            // {
            //     auto tensorGIn = gInQue.AllocTensor<betaType>();
            //     DataCopy(tensorGIn, gTensor[gOffset], rowNum);
                
            //     // DataCopy(tensormdAIn, mdATensor[gOffset], rowNum);
            //     gInQue.EnQue(tensorGIn);
            //     // mdAInQue.EnQue(tensormdAIn);
            // }
            // //compute && copyout
            // {
            //     auto tensorGIn = gInQue.DeQue<betaType>();
            //     if constexpr (!std::is_same<betaType, float32_t>()) {
            //         Cast(tensorGfp32, tensorGIn, RoundMode::CAST_NONE, rowNum);
            //     } else {
            //         DataCopy(tensorGfp32, tensorGIn, rowNum);
            //     }
            //     PipeBarrier<PIPE_V>();
            //     // todo 此处添加CUBE的同步等待
            //     for (int row = 1; row < rowNum; row++) {
            //         auto tensormdAIn = mdAInQue.AllocTensor<kType>();
            //         auto tensorGOut = gOutQue.AllocTensor<kType>();
            //         // 正三角，按照BT 一行一行算，WORKSAPCE搭配一行一行进，置为0所以是正三角方式
            //         AscendC::Duplicate<float>(tensorGCalfp32, float(0.0), rowNum);
            //         AscendC::Adds(tensorGCalfp32, tensorGfp32, -1 * tensorGfp32.GetValue(row), row);
            //         // AscendC::printf("[tensor 打印]  tensorGCalfp32 \n");
            //         // AscendC::DumpTensor(tensorGCalfp32, 5, 16);
            //         AscendC::Exp(tensorGCalfp32, tensorGCalfp32, row);
            //         // todo：使用外部搬入的workspace 直接参与运算
            //         AscendC::DataCopy(tensormdAIn, workSpaceTensor[gOffset], rowNum);
            //         mdAInQue.EnQue(tensormdAIn);
            //         tensormdAIn = mdAInQue.DeQue<kType>();
            //         // AscendC::printf("[tensor 打印]  tensormdAIn \n");
            //         // AscendC::DumpTensor(tensormdAIn, 5, 16);
            //         SetFlag<AscendC::HardEvent::MTE2_V>(0);
            //         WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            //         AscendC::Cast(tensorMdAfp32, tensormdAIn, AscendC::RoundMode::CAST_NONE, rowNum);
            //         // AscendC::printf("[tensor 打印]  tensorMdAfp32 \n");
            //         // AscendC::DumpTensor(tensorMdAfp32, 5, 16);
            //         AscendC::Muls(tensorMdAfp32, tensorMdAfp32, float(-1.0), row);
            //         AscendC::Mul(tensorGCalfp32, tensorMdAfp32, tensorGCalfp32, row);
            //         AscendC::Cast(tensorGOut, tensorGCalfp32, AscendC::RoundMode::CAST_NONE, row);
            //         mdAInQue.FreeTensor(tensormdAIn);
            //         gOutQue.EnQue(tensorGOut);
            //         tensorGOut = gOutQue.DeQue<kType>();
            //         // 直接搬出到外面了
            //         DataCopy(workSpace2Tensor[gOffset], tensorGOut, rowNum);
            //         gOutQue.FreeTensor(tensorGOut);
            //     }    
            //     gInQue.FreeTensor(tensorGIn);
            // }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    return;
}

#endif  // PREPARE_WY_REPR_BWD_DA_VECTOR_H

