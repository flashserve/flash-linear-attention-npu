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
 * \file prepare_wy_repr_bwd_full.h
 * \brief
 */


#ifndef PREPARE_WY_REPR_BWD_FULL_VECTOR_H
#define PREPARE_WY_REPR_BWD_FULL_VECTOR_H


using namespace AscendC;

template <typename kType, typename betaType>
class PrepareWyReprBwdFullVectorProcess {
 public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessKBeta();
    __aicore__ inline void ProcessDkb();
    __aicore__ inline void ProcessDkbg();
    __aicore__ inline void ProcessDvb();
    __aicore__ inline void Init(GM_ADDR tiling, AscendC::TPipe *pipe_);
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
    uint64_t dkbgRowNum = 16;
    uint64_t dvbRowNum = 32;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dA;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR dbeta;
    GM_ADDR dg;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;
private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<kType> dkTensor;
    GlobalTensor<kType> dvTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> dbetaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<betaType> dgTensor;
    GlobalTensor<kType> workSpaceTensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> vInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECIN, 1> dkInQue;
    TQue<AscendC::TPosition::VECIN, 1> dbetaInQue;
    TQue<AscendC::TPosition::VECIN, 1> dkbInQue;
    TQue<AscendC::TPosition::VECIN, 1> dkbgInQue;
    TQue<AscendC::TPosition::VECIN, 1> dvbInQue;

    TQue<AscendC::TPosition::VECOUT, 1> kBetaOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> dkOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> dBetaOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> dgOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> dvOutQue;

    TBuf<AscendC::TPosition::VECCALC> kFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> vFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dkFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dkbFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dkbgFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dvbFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> gFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> gFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> sharedTmpBuf;
    TBuf<AscendC::TPosition::VECCALC> dbetaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dbetaAddFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dgFp32Buf;
};

template <typename kType, typename betaType>
 __aicore__ inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::PrepareWyReprBwdFullVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,GM_ADDR workspace_)
 :
    k(k_),
    v(v_),
    beta(beta_),
    A(A_),
    dA(dA_),
    dw(dw_),
    du(du_),
    g(g_),
    dk(dk_),
    dv(dv_),
    dbeta(dbeta_),
    dg(dg_),
    workspace(workspace_)
    {};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Init(GM_ADDR tiling, AscendC::TPipe *pipe_) {
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    dkTensor.SetGlobalBuffer((__gm__ kType *)dk);
    dvTensor.SetGlobalBuffer((__gm__ kType *)dv);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    dbetaTensor.SetGlobalBuffer((__gm__ betaType *)dbeta);
    dgTensor.SetGlobalBuffer((__gm__ betaType *)dg);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Process() {
    //计算K * Beta[:None]
    ProcessKBeta();
    pipe->Reset();
    ProcessDkb();
    pipe->Reset();
    ProcessDkbg();
    pipe->Reset();
    ProcessDvb();
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDvb() {
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbgRowNum;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;

    //init
    pipe->InitBuffer(vInQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dbetaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dvbInQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(dvOutQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(dBetaOutQue, 2, rowNum * sizeof(betaType));

    pipe->InitBuffer(dvbFp32Buf, rowNum * V * sizeof(float32_t));
    pipe->InitBuffer(vFp32Buf, rowNum * V * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(dbetaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(dbetaAddFp32Buf, rowNum * sizeof(float32_t));

    pipe->InitBuffer(sharedTmpBuf, BT * sizeof(float32_t));

    auto tensorDvbFp32 = dvbFp32Buf.Get<float32_t>();
    auto tensorVFp32 = vFp32Buf.Get<float32_t>();
    auto tensorBetafp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbfp32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorDbetafp32 = dbetaFp32Buf.Get<float32_t>();
    auto tensorDbetaAddfp32 = dbetaAddFp32Buf.Get<float32_t>();
    auto tensorSharedTmp = sharedTmpBuf.Get<float32_t>();

    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    workSpaceTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for(uint32_t rowOffset = 0;rowOffset < BT; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                auto vOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * V;
                auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
                //copyin
                {
                    auto tensorDvbin = dvbInQue.AllocTensor<kType>();
                    auto tensorVin = vInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensordDbetaIn = dbetaInQue.AllocTensor<betaType>();

                    DataCopy(tensorDvbin, workSpaceTensor[vOffset], V * rowNum);
                    DataCopy(tensorVin, vTensor[vOffset], V * rowNum);     
                    DataCopy(tensorBetain, betaTensor[betaOffset], rowNum);
                    DataCopy(tensordDbetaIn, dbetaTensor[betaOffset], rowNum);

                    dvbInQue.EnQue(tensorDvbin);
                    vInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                    dbetaInQue.EnQue(tensordDbetaIn);
                }
                //compute
                {
                    auto tensorDvbin = dvbInQue.DeQue<kType>();
                    auto tensorVin = vInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorDbetain = dbetaInQue.DeQue<betaType>();

                    auto tensorDvOut = dvOutQue.AllocTensor<kType>();
                    auto tensorDbetaOut = dBetaOutQue.AllocTensor<betaType>();
                    //cast fp32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetafp32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                        Cast(tensorDbetafp32, tensorDbetain, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorBetafp32, tensorBetain, rowNum);
                        DataCopy(tensorDbetafp32, tensorDbetain, rowNum);
                    }
                    Cast(tensorDvbFp32, tensorDvbin, RoundMode::CAST_NONE, V * rowNum);

                    Cast(tensorVFp32, tensorVin, RoundMode::CAST_NONE, V * rowNum);
                    PipeBarrier<PIPE_V>();
                    //brcb(beta)  dvb * v -> v
                    Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum / 8), {1, 8});
                    Mul(tensorVFp32, tensorVFp32, tensorDvbFp32, V * rowNum);
                    PipeBarrier<PIPE_V>();
                    // dvb * beta -> dvb
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < V) {
                        Mul(tensorDvbFp32[perchannelResOffset], tensorDvbFp32[perchannelResOffset], tensorBetaBrcbfp32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    //reducesum
                    for(uint32_t row = 0;row < rowNum; row++) {
                        PipeBarrier<PIPE_V>();
                        ReduceSum(tensorVFp32, tensorVFp32[row * V], tensorSharedTmp,
                            FP32_PER_REPEAT_64, K / FP32_PER_REPEAT_64, 8);
                        SetFlag<AscendC::HardEvent::V_S>(0);
                        WaitFlag<AscendC::HardEvent::V_S>(0);
                        tensorDbetaAddfp32.SetValue(row, tensorVFp32.GetValue(0));
                    }

                    PipeBarrier<PIPE_V>();
                    // 累加处理原始dbeta
                    Add(tensorDbetaAddfp32, tensorDbetaAddfp32, tensorDbetafp32, rowNum);
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorDvOut, tensorDvbFp32, RoundMode::CAST_RINT, V * rowNum);
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorDbetaOut, tensorDbetaAddfp32, RoundMode::CAST_RINT, rowNum);
                    } else {
                        DataCopy(tensorDbetaOut, tensorDbetaAddfp32, rowNum);
                    }
                    dvbInQue.FreeTensor(tensorDvbin);
                    vInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);
                    dbetaInQue.FreeTensor(tensorDbetain);

                    dvOutQue.EnQue(tensorDvOut);
                    dBetaOutQue.EnQue(tensorDbetaOut);
                }
                //copyout
                {
                    auto tensorDvOut = dvOutQue.DeQue<kType>();
                    auto tensorDbetaOut = dBetaOutQue.DeQue<betaType>();
                    DataCopy(dvTensor[vOffset], tensorDvOut, V * rowNum);
                    DataCopy(dbetaTensor[betaOffset], tensorDbetaOut, rowNum);
                    dvOutQue.FreeTensor(tensorDvOut);
                    dBetaOutQue.FreeTensor(tensorDbetaOut);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbg() {
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbgRowNum;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;

    //init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dkbgInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dbetaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dgOutQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dBetaOutQue, 2, rowNum * sizeof(betaType));

    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(gFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(gFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(dkFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(dkbgFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(dbetaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(dbetaAddFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(dgFp32Buf, rowNum * sizeof(float32_t));

    pipe->InitBuffer(sharedTmpBuf, BT * sizeof(float32_t));

    auto tensorDkbgFp32 = dkbgFp32Buf.Get<float32_t>();
    auto tensorDkFp32 = dkFp32Buf.Get<float32_t>();
    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetafp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbfp32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorSharedTmp = sharedTmpBuf.Get<float32_t>();
    auto tensorGfp32 = gFp32Buf.Get<float32_t>();
    auto tensorDbetafp32 = dbetaFp32Buf.Get<float32_t>();
    auto tensorGbrcbfp32 = gFp32BrcbBuf.Get<float32_t>();
    auto tensorDbetaAddfp32 = dbetaAddFp32Buf.Get<float32_t>();
    auto tensorDgFp32 = dgFp32Buf.Get<float32_t>();

    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    workSpaceTensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for(uint32_t rowOffset = 0;rowOffset < BT; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                auto kOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * K;
                auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
                //copyin
                {
                    auto tensorDkbgin = dkbgInQue.AllocTensor<kType>();
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorGin = gInQue.AllocTensor<betaType>();
                    auto tensorDkin = dkInQue.AllocTensor<kType>();
                    auto tensordDbetaIn = dbetaInQue.AllocTensor<betaType>();

                    DataCopy(tensorDkbgin, workSpaceTensor[kOffset], K * rowNum);
                    DataCopy(tensorKin, kTensor[kOffset], K * rowNum);     
                    DataCopy(tensorBetain, betaTensor[betaOffset], rowNum);
                    DataCopy(tensorGin, gTensor[betaOffset], rowNum);
                    DataCopy(tensorDkin, dkTensor[kOffset], K * rowNum);
                    DataCopy(tensordDbetaIn, dbetaTensor[betaOffset], rowNum);

                    dkbgInQue.EnQue(tensorDkbgin);
                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    gInQue.EnQue(tensorGin);
                    dkInQue.EnQue(tensorDkin);
                    dbetaInQue.EnQue(tensordDbetaIn);
                }
                //compute
                {
                    auto tensorDkbgin = dkbgInQue.DeQue<kType>();
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorGin = gInQue.DeQue<betaType>();
                    auto tensorDkin = dkInQue.DeQue<kType>();
                    auto tensorDbetain = dbetaInQue.DeQue<betaType>();

                    auto tensorDkOut = dkOutQue.AllocTensor<kType>();
                    auto tensorDbetaOut = dBetaOutQue.AllocTensor<betaType>();
                    auto tensorDgOut = dgOutQue.AllocTensor<betaType>();
                    //cast fp32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetafp32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                        Cast(tensorGfp32, tensorGin, RoundMode::CAST_NONE, rowNum);
                        Cast(tensorDbetafp32, tensorDbetain, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorBetafp32, tensorBetain, rowNum);
                        DataCopy(tensorGfp32, tensorGin, rowNum);
                        DataCopy(tensorDbetafp32, tensorDbetain, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * rowNum);
                    Cast(tensorDkbgFp32, tensorDkbgin, RoundMode::CAST_NONE, K * rowNum);

                    PipeBarrier<PIPE_V>();
                    // exp(g) ->g
                    Exp(tensorGfp32, tensorGfp32, rowNum);
                    PipeBarrier<PIPE_V>();
                    //brcb(gexp)  brcb(beta)
                    Brcb(tensorGbrcbfp32, tensorGfp32, static_cast<uint8_t>(rowNum / 8), {1, 8});
                    Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum / 8), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // dkbg * gexp -> dkbg
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkbgFp32[perchannelResOffset], tensorDkbgFp32[perchannelResOffset], tensorGbrcbfp32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // dkbg(dkbg * gexp) * k ->k
                    
                    Mul(tensorKFp32, tensorKFp32, tensorDkbgFp32, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    perchannelResOffset = 0;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbfp32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    //reducesum
                    for(uint32_t row = 0;row < rowNum; row++) {
                        PipeBarrier<PIPE_V>();
                        ReduceSum(tensorKFp32, tensorKFp32[row * K], tensorSharedTmp,
                            FP32_PER_REPEAT_64, K / FP32_PER_REPEAT_64, 8);
                        ReduceSum(tensorDkFp32, tensorDkFp32[row * K], tensorSharedTmp,
                            FP32_PER_REPEAT_64, K / FP32_PER_REPEAT_64, 8);
                        SetFlag<AscendC::HardEvent::V_S>(0);
                        WaitFlag<AscendC::HardEvent::V_S>(0);
                        tensorDbetaAddfp32.SetValue(row, tensorKFp32.GetValue(0));
                        tensorDgFp32.SetValue(row, tensorDkFp32.GetValue(0));
                    }

                    PipeBarrier<PIPE_V>();
                    // cast dg
                    Cast(tensorDgOut, tensorDgFp32, RoundMode::CAST_RINT, rowNum);
                    // 累加第二部分产生的处理原始dbeta
                    Add(tensorDbetaAddfp32, tensorDbetaAddfp32, tensorDbetafp32, rowNum);
                    // 计算当前dk公式为 dkbg(dkbg * gexp) * beta -> dkbg
                    perchannelResOffset = 0;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkbgFp32[perchannelResOffset], tensorDkbgFp32[perchannelResOffset], tensorBetaBrcbfp32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    Cast(tensorDkFp32, tensorDkin, RoundMode::CAST_NONE, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    // 累加第二部分产生的处理原始dk
                    Add(tensorDkFp32, tensorDkFp32, tensorDkbgFp32, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, K * rowNum);
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorDbetaOut, tensorDbetaAddfp32, RoundMode::CAST_RINT, rowNum);
                    } else {
                        DataCopy(tensorDbetaOut, tensorDbetaAddfp32, rowNum);
                    }

                    dkbgInQue.FreeTensor(tensorDkbgin);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    gInQue.FreeTensor(tensorGin);
                    dkInQue.FreeTensor(tensorDkin);
                    dbetaInQue.FreeTensor(tensorDbetain);

                    dkOutQue.EnQue(tensorDkOut);
                    dBetaOutQue.EnQue(tensorDbetaOut);
                    dgOutQue.EnQue(tensorDgOut);
                }
                //copyout
                {
                    auto tensorDkOut = dkOutQue.DeQue<kType>();
                    auto tensorDbetaOut = dBetaOutQue.DeQue<betaType>();
                    auto tensorDgOut = dgOutQue.DeQue<betaType>();
                    DataCopy(dkTensor[kOffset], tensorDkOut, K * rowNum);
                    DataCopy(dbetaTensor[betaOffset], tensorDbetaOut, rowNum);
                    DataCopy(dgTensor[betaOffset], tensorDgOut, rowNum);
                    dkOutQue.FreeTensor(tensorDkOut);
                    dBetaOutQue.FreeTensor(tensorDbetaOut);
                    dgOutQue.FreeTensor(tensorDgOut);
                }
                
            }
            
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkb() {
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbRowNum;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;

    //init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dkbInQue, 2, rowNum * sizeof(kType));
    pipe->InitBuffer(dkOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dBetaOutQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkbFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(dkFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(sharedTmpBuf, BT * sizeof(float32_t));

    auto tensorDkbFp32 = dkbFp32Buf.Get<float32_t>();
    auto tensorDkFp32 = dkFp32Buf.Get<float32_t>();
    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetafp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbfp32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorSharedTmp = sharedTmpBuf.Get<float32_t>();
    //
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for(uint32_t rowOffset = 0;rowOffset < BT; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                auto kOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * K;
                auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
                //copyin
                {
                    auto tensorDkbin = dkbInQue.AllocTensor<kType>();
                    DataCopy(tensorDkbin, workSpaceTensor[kOffset], K * rowNum);
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    DataCopy(tensorKin, kTensor[kOffset], K * rowNum);
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopy(tensorBetain, betaTensor[betaOffset], rowNum);
                    auto tensorDkin = dkInQue.AllocTensor<kType>();
                    DataCopy(tensorDkin, dkTensor[kOffset], K * rowNum);
                    
                    dkbInQue.EnQue(tensorDkbin);
                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    dkInQue.EnQue(tensorDkin);
                    //AscendC::printf("copyin\n");
                }
                //compute
                {
                    auto tensorDkbin = dkbInQue.DeQue<kType>();
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorDkin = dkInQue.DeQue<kType>();

                    auto tensorDkOut = dkOutQue.AllocTensor<kType>();
                    auto tensorDbetaOut = dBetaOutQue.AllocTensor<betaType>();
                    //cast fp32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetafp32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorBetafp32, tensorBetain, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * rowNum);
                    Cast(tensorDkbFp32, tensorDkbin, RoundMode::CAST_NONE, K * rowNum);
                    Cast(tensorDkFp32, tensorDkin, RoundMode::CAST_NONE, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    //brcb
                    Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum /8), {1, 8});
                    PipeBarrier<PIPE_V>();
                    //mul
                    Mul(tensorKFp32, tensorKFp32, tensorDkbFp32, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkbFp32[perchannelResOffset], tensorDkbFp32[perchannelResOffset], tensorBetaBrcbfp32,
                            FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    // DumpTensor(tensorKFp32, 1,  K * rowNum);
                    //reducesum
                    for(uint32_t row = 0;row < rowNum; row++) {
                        PipeBarrier<PIPE_V>();
                        ReduceSum(tensorKFp32, tensorKFp32[row * K], tensorSharedTmp,
                            FP32_PER_REPEAT_64, K / FP32_PER_REPEAT_64, 8);
                        SetFlag<AscendC::HardEvent::V_S>(0);
                        WaitFlag<AscendC::HardEvent::V_S>(0);
                        tensorBetafp32.SetValue(row, tensorKFp32.GetValue(0));
                    }
                    // ADD
                    PipeBarrier<PIPE_V>();
                    Add(tensorDkFp32, tensorDkFp32, tensorDkbFp32, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, K * rowNum);
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorDbetaOut, tensorBetafp32, RoundMode::CAST_RINT, rowNum);
                    } else {
                        DataCopy(tensorDbetaOut, tensorBetafp32, rowNum);
                    }

                    dkbInQue.FreeTensor(tensorDkbin);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    dkInQue.FreeTensor(tensorDkin);
                    dkOutQue.EnQue(tensorDkOut);
                    dBetaOutQue.EnQue(tensorDbetaOut);
                }
                //copyout
                {
                    auto tensorDkOut = dkOutQue.DeQue<kType>();
                    auto tensorDbetaOut = dBetaOutQue.DeQue<betaType>();
                    DataCopy(dkTensor[kOffset], tensorDkOut, K * rowNum);
                    DataCopy(dbetaTensor[betaOffset], tensorDbetaOut, rowNum);
                    dkOutQue.FreeTensor(tensorDkOut);
                    dBetaOutQue.FreeTensor(tensorDbetaOut);
                }
                
            }
            
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            //AscendC::printf("CrossCoreSetFlag\n");
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBeta() {
    uint32_t coreLoopsInB = CeilDiv(T, BT);
    uint32_t coreLoops = B * coreLoopsInB;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = BT;
    uint32_t rowOffset = 0;
    if(GetSubBlockNum() > 1) {
        if(GetSubBlockIdx() == 0) {
            rowNum = rowNum / GetSubBlockNum();
        } else {
            rowOffset = rowNum / GetSubBlockNum();
            rowNum = rowNum - rowOffset;
        }
    }
    //init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(kBetaOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetafp32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbfp32 = betaFp32BrcbBuf.Get<float32_t>();
    //
    // printf("coreIdx:%d, coreNumAic:%d\n",coreIdx, coreNumAic );

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
        uint32_t bIdx = loopIdx / coreLoopsInB;
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        for (int h = 0; h < H; h++) {
            auto kOffset = ((bIdx * H + h) * T  + chunkIdx * BT + rowOffset) * K;
            auto betaOffset = (bIdx * H + h) * T  + chunkIdx * BT + rowOffset;
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            //AscendC::printf("CrossCoreWaitFlag kOffset:%d, betaOffset:%d\n", kOffset, betaOffset);
            //copyin
            {
                auto tensorKin = kInQue.AllocTensor<kType>();
                DataCopy(tensorKin, kTensor[kOffset], K * rowNum);
                auto tensorBetain = betaInQue.AllocTensor<betaType>();
                DataCopy(tensorBetain, betaTensor[betaOffset], rowNum);

                kInQue.EnQue(tensorKin);
                betaInQue.EnQue(tensorBetain);
                //AscendC::printf("copyin\n");
            }
            //compute
            {
                //AscendC::printf("150\n");
                auto tensorKin = kInQue.DeQue<kType>();
                auto tensorBetain = betaInQue.DeQue<betaType>();
                auto tensorOut = kBetaOutQue.AllocTensor<kType>();
                //AscendC::printf("153\n");
                //cast fp32
                if constexpr (!std::is_same<betaType, float32_t>()) {
                    Cast(tensorBetafp32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                } else {
                    DataCopy(tensorBetafp32, tensorBetain, rowNum);
                }
                //AscendC::printf("159\n");
                Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * rowNum);
                PipeBarrier<PIPE_V>();
                //brcb
                Brcb(tensorBetaBrcbfp32, tensorBetafp32, static_cast<uint8_t>(rowNum /8), {1, 8});
                // DumpTensor(tensorBetaBrcbfp32, 0,  8 * rowNum);
                PipeBarrier<PIPE_V>();
                //AscendC::printf("165\n");
                //mul
                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                while (perchannelResOffset < K) {
                    Mul(tensorKFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbfp32,
                        FP32_PER_REPEAT_64, rowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                // DumpTensor(tensorKFp32, 1,  K * rowNum);
                PipeBarrier<PIPE_V>();
                Cast(tensorOut, tensorKFp32, RoundMode::CAST_RINT, K * rowNum);

                kInQue.FreeTensor(tensorKin);
                betaInQue.FreeTensor(tensorBetain);
                kBetaOutQue.EnQue(tensorOut);
                //AscendC::printf("compute\n");
            }
            //copyout
            {
                auto tensorOut = kBetaOutQue.DeQue<kType>();
                DataCopy(workSpaceTensor[kOffset], tensorOut, K * rowNum);
                kBetaOutQue.FreeTensor(tensorOut);
                // AscendC::printf("kOffset:%ld, K * rowNum:%d\n", kOffset, K * rowNum);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
            //AscendC::printf("CrossCoreSetFlag\n");
        }
    }
    // DumpTensor(workSpaceTensor, 0,  8192);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    //AscendC::printf("CrossCoreWaitFlag\n");
    //AscendC::printf("CrossCoreWaitFlag\n");
    return;
}



#endif  // PREPARE_WY_REPR_BWD_FULL_VECTOR_H
