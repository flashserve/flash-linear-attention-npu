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
    __aicore__ inline PrepareWyReprBwdFullVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_,
                                                        GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_,
                                                        GM_ADDR dbeta_, GM_ADDR dg_, GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessKBeta();
    __aicore__ inline void ProcessDkb();
    __aicore__ inline void ProcessDkbg();
    __aicore__ inline void ProcessDvb();
    __aicore__ inline void ProcessKKT();
    __aicore__ inline void Init(const PrepareWyReprBwdFullTilingData &tiling, AscendC::TPipe *pipe_);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t H = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t kBeteVecRow = 0;
    uint64_t dkbVecRow = 0;
    uint64_t dkbgVecRow = 0;
    uint64_t dvbVecRow = 0;
    uint64_t kktVecRow = 0;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dA;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
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
    // GlobalTensor<uint64_t> cuSeqlensTensor;
    // GlobalTensor<uint64_t> chunkIndicesTensor;
    GlobalTensor<betaType> dgTensor;
    GlobalTensor<kType> dATensor;
    GlobalTensor<kType> workSpaceTensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> vInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECIN, 1> dkInQue;
    TQue<AscendC::TPosition::VECIN, 1> daInQue;
    TQue<AscendC::TPosition::VECIN, 1> dgInQue;
    TQue<AscendC::TPosition::VECIN, 1> dbetaInQue;
    TQue<AscendC::TPosition::VECIN, 1> dkbInQue;
    TQue<AscendC::TPosition::VECIN, 1> dkbgInQue;
    TQue<AscendC::TPosition::VECIN, 1> dvbInQue;
    TQue<AscendC::TPosition::VECIN, 1> kktInQue;

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
    TBuf<AscendC::TPosition::VECCALC> daFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> kktFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> daaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> gFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> gFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> reduceSumTmpBuf;
    TBuf<AscendC::TPosition::VECCALC> dbetaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dbetaAddFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> dgFp32Buf;
};

template <typename kType, typename betaType>
__aicore__ inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::PrepareWyReprBwdFullVectorProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), dA(dA_), dw(dw_), du(du_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), dk(dk_), dv(dv_), dbeta(dbeta_), dg(dg_), workspace(workspace_){};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Init(
    const PrepareWyReprBwdFullTilingData &tiling, AscendC::TPipe *pipe_)
{
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
    dATensor.SetGlobalBuffer((__gm__ kType *)dA);

    B = tiling.B;
    T = tiling.T;
    H = tiling.H;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    kBeteVecRow = tiling.kBeteVecRow;
    dkbVecRow = tiling.dkbVecRow;
    dkbgVecRow = tiling.dkbgVecRow;
    dvbVecRow = tiling.dvbVecRow;
    kktVecRow = tiling.kktVecRow;

    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Process()
{
    //计算K * Beta[:None]
    ProcessKBeta();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessDkb();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessDkbg();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessDvb();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessKKT();
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKKT()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = kktVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(betaInQue, 2, chunkSize * sizeof(betaType));
    pipe->InitBuffer(dgInQue, 2, chunkSize * sizeof(betaType));
    pipe->InitBuffer(daInQue, 2, rowNum * chunkSize * sizeof(kType));
    pipe->InitBuffer(kktInQue, 2, rowNum * chunkSize * sizeof(kType));

    pipe->InitBuffer(dgOutQue, 2, chunkSize * sizeof(betaType));

    pipe->InitBuffer(kktFp32Buf, rowNum * chunkSize * sizeof(float32_t));
    pipe->InitBuffer(daFp32Buf, rowNum * chunkSize * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, chunkSize * sizeof(float32_t));
    // daaFp32Buf，dg保存全量当前chunk数据，用于reducesum及累加
    pipe->InitBuffer(daaFp32Buf, chunkSize * chunkSize * sizeof(float32_t));
    pipe->InitBuffer(reduceSumTmpBuf, chunkSize * ONE_BLOCK_32);

    auto tensorKKTFp32 = kktFp32Buf.Get<float32_t>();
    auto tensorDaFp32 = daFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorDaaFP32 = daaFp32Buf.Get<float32_t>();

    auto tensorReduceSum = reduceSumTmpBuf.Get<float32_t>();
    //复用空间
    auto tensorDgFP32 = kktFp32Buf.Get<float32_t>();
    auto tensorDgFP32Add = betaFp32Buf.Get<float32_t>();
    auto tensorSum1DaaFP32 = daFp32Buf.Get<float32_t>();

    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, H, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        uint32_t wholeReduceSumCnt = CeilDiv(curChunkSize, FP32_PER_REPEAT_64);
        for (int h = 0; h < H; h++) {
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
                continue;
            }
            {// copyin
                auto tensorBeta = betaInQue.AllocTensor<betaType>();
                auto tensorDg = dgInQue.AllocTensor<betaType>();
                DataCopyPad(tensorBeta, betaTensor[h * T + bos], {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                DataCopyPad(tensorDg, dgTensor[h * T + bos], {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                betaInQue.EnQue(tensorBeta);
                dgInQue.EnQue(tensorDg);
            }
            {//compute
                auto tensorBeta =betaInQue.DeQue<betaType>();
                if constexpr (std::is_same<betaType, float32_t>()) {
                    DataCopy(tensorBetaFP32, tensorBeta, chunkSize);
                } else {
                    Cast(tensorBetaFP32, tensorBeta, RoundMode::CAST_NONE, curChunkSize);
                }
                betaInQue.FreeTensor(tensorBeta);
            }
            
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            //分批次处理计算daa
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                auto dAOffset = (h * T + bos + rowOffset) * chunkSize;
                auto betaOffset = h * T + bos + rowOffset;
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                // copyin
                {
                    auto tensordaIn = daInQue.AllocTensor<kType>();
                    auto tensorKKTin = kktInQue.AllocTensor<kType>();

                    DataCopy(tensordaIn, dATensor[dAOffset], chunkSize * curRowNum);
                    DataCopy(tensorKKTin, workSpaceTensor[dAOffset], chunkSize * curRowNum);

                    daInQue.EnQue(tensordaIn);
                    kktInQue.EnQue(tensorKKTin);
                }
                // compute
                {
                    auto tensordaIn = daInQue.DeQue<kType>();
                    auto tensorKKTin = kktInQue.DeQue<kType>();
                    // cast FP32
                    Cast(tensorKKTFp32, tensorKKTin, RoundMode::CAST_NONE, chunkSize * curRowNum);
                    Cast(tensorDaFp32, tensordaIn, RoundMode::CAST_NONE, chunkSize * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // KKT * beta -> KKT
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = chunkSize * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < curChunkSize) {
                        Mul(tensorKKTFp32[perchannelResOffset], tensorKKTFp32[perchannelResOffset], tensorBetaFP32[perchannelResOffset],
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 1, repeatStride, repeatStride, 0});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // da * KKT(KKT * beta) -> da_A
                    Mul(tensorDaaFP32[rowOffset * chunkSize], tensorDaFp32, tensorKKTFp32, chunkSize * curRowNum);
                    PipeBarrier<PIPE_V>();

                    daInQue.FreeTensor(tensordaIn);
                    kktInQue.FreeTensor(tensorKKTin);
                }
            }
            //最后处理daa的sum相减
            {
                uint32_t remainCnt = curChunkSize % FP32_PER_REPEAT_64;
                if(remainCnt > 0) {
                    uint32_t DuplicateOffset = wholeReduceSumCnt * FP32_PER_REPEAT_64 - FP32_PER_REPEAT_64;
                    uint64_t mask[1] = {0xffffffffffffffff};
                    mask[0] <<= remainCnt;
                    for (uint32_t row = 0; row < curChunkSize; row++) {
                        Duplicate(tensorDaaFP32[row * chunkSize + DuplicateOffset], 0.0f, mask, 1, 1, 8);
                    }
                    PipeBarrier<PIPE_V>();
                }
                // reducesum
                for (uint32_t row = 0; row < curChunkSize; row++) {
                    WholeReduceSum(tensorReduceSum[row * FP32_PER_BLOCK_8], tensorDaaFP32[row * chunkSize],
                                   FP32_PER_REPEAT_64, wholeReduceSumCnt, 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorSum1DaaFP32, tensorReduceSum, wholeReduceSumCnt, curChunkSize, 1, 1, 1);
                uint32_t remain_row = curChunkSize;
                uint32_t CalcCnt = 0;
                uint32_t Offset = 0;
                while (remain_row > 1) {
                    CalcCnt = (remain_row / 2) * chunkSize;
                    remain_row = CeilDiv(remain_row, 2);
                    Offset = remain_row * chunkSize;
                    Add(tensorDaaFP32, tensorDaaFP32, tensorDaaFP32[Offset], CalcCnt);
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_V>();
                Sub(tensorDgFP32Add, tensorDaaFP32, tensorSum1DaaFP32, curChunkSize);
                // DataCopy(tensorDgFP32Add, tensorDaaFP32, chunkSize);
                auto tensorDg = dgInQue.DeQue<betaType>();
                auto tensorOutDg = dgOutQue.AllocTensor<betaType>();
                if constexpr (!std::is_same<betaType, float32_t>()) {
                    Cast(tensorDgFP32, tensorDg, RoundMode::CAST_NONE, curChunkSize);
                } else {
                    DataCopy(tensorDgFP32, tensorDg, chunkSize);
                }
                dgInQue.FreeTensor(tensorDg);
                PipeBarrier<PIPE_V>();
                Add(tensorDgFP32Add, tensorDgFP32Add, tensorDgFP32, curChunkSize);
                PipeBarrier<PIPE_V>();

                if constexpr (!std::is_same<betaType, float32_t>()) {
                    Cast(tensorOutDg, tensorDgFP32Add, RoundMode::CAST_RINT, curChunkSize);
                } else {
                    DataCopy(tensorOutDg, tensorDgFP32Add, chunkSize);
                }
                dgOutQue.EnQue(tensorOutDg);
            }
            {//copyout
                auto tensorOutDg = dgOutQue.DeQue<betaType>();
                DataCopyPad(dgTensor[h * T + bos], tensorOutDg, {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                dgOutQue.FreeTensor(tensorOutDg);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    return;
}


template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDvb()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dvbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t wholeReduceSumCnt = CeilDiv(V, FP32_PER_REPEAT_64);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
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
    pipe->InitBuffer(reduceSumTmpBuf, rowNum * ONE_BLOCK_32);

    auto tensorDvbFp32 = dvbFp32Buf.Get<float32_t>();
    auto tensorVFp32 = vFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorDbetaFP32 = dbetaFp32Buf.Get<float32_t>();
    auto tensorDbetaAddFP32 = dbetaAddFp32Buf.Get<float32_t>();
    auto tensorReduceSum = reduceSumTmpBuf.Get<float32_t>();

    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, H, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * V;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorDvbin = dvbInQue.AllocTensor<kType>();
                    auto tensorVin = vInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensordDbetaIn = dbetaInQue.AllocTensor<betaType>();

                    DataCopy(tensorDvbin, workSpaceTensor[vOffset], V * curRowNum);
                    DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    DataCopyPad(tensordDbetaIn, dbetaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});

                    dvbInQue.EnQue(tensorDvbin);
                    vInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                    dbetaInQue.EnQue(tensordDbetaIn);
                }
                // compute
                {
                    auto tensorDvbin = dvbInQue.DeQue<kType>();
                    auto tensorVin = vInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorDbetain = dbetaInQue.DeQue<betaType>();

                    auto tensorDvOut = dvOutQue.AllocTensor<kType>();
                    auto tensorDbetaOut = dBetaOutQue.AllocTensor<betaType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorDbetaFP32, tensorDbetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                        DataCopy(tensorDbetaFP32, tensorDbetain, rowNum);
                    }
                    Cast(tensorDvbFp32, tensorDvbin, RoundMode::CAST_NONE, V * curRowNum);

                    Cast(tensorVFp32, tensorVin, RoundMode::CAST_NONE, V * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb(beta)  dvb * v -> v
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    Mul(tensorVFp32, tensorVFp32, tensorDvbFp32, V * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // dvb * beta -> dvb
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < V) {
                        Mul(tensorDvbFp32[perchannelResOffset], tensorDvbFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // reducesum
                    for (uint32_t row = 0; row < curRowNum; row++) {
                        WholeReduceSum(tensorReduceSum[row * FP32_PER_BLOCK_8], tensorVFp32[row * V],
                                       FP32_PER_REPEAT_64, wholeReduceSumCnt, 1, 1, 8);
                    }
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(tensorDbetaAddFP32, tensorReduceSum, wholeReduceSumCnt, curRowNum, 1, 1, 1);
                    PipeBarrier<PIPE_V>();
                    // 累加处理原始dbeta
                    Add(tensorDbetaAddFP32, tensorDbetaAddFP32, tensorDbetaFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorDvOut, tensorDvbFp32, RoundMode::CAST_RINT, V * curRowNum);
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorDbetaOut, tensorDbetaAddFP32, RoundMode::CAST_RINT, curRowNum);
                    } else {
                        DataCopy(tensorDbetaOut, tensorDbetaAddFP32, rowNum);
                    }
                    dvbInQue.FreeTensor(tensorDvbin);
                    vInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);
                    dbetaInQue.FreeTensor(tensorDbetain);

                    dvOutQue.EnQue(tensorDvOut);
                    dBetaOutQue.EnQue(tensorDbetaOut);
                }
                // copyout
                {
                    auto tensorDvOut = dvOutQue.DeQue<kType>();
                    auto tensorDbetaOut = dBetaOutQue.DeQue<betaType>();
                    DataCopy(dvTensor[vOffset], tensorDvOut, V * curRowNum);
                    DataCopyPad(dbetaTensor[betaOffset], tensorDbetaOut, {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    dvOutQue.FreeTensor(tensorDvOut);
                    dBetaOutQue.FreeTensor(tensorDbetaOut);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbg()
{

    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbgVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t wholeReduceSumCnt = CeilDiv(K, FP32_PER_REPEAT_64);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
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

    auto tensorDkbgFp32 = dkbgFp32Buf.Get<float32_t>();
    auto tensorDkFp32 = dkFp32Buf.Get<float32_t>();
    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorGFP32 = gFp32Buf.Get<float32_t>();
    auto tensorDbetaFP32 = dbetaFp32Buf.Get<float32_t>();
    auto tensorGbrcbFP32 = gFp32BrcbBuf.Get<float32_t>();
    auto tensorDbetaAddFP32 = dbetaAddFp32Buf.Get<float32_t>();
    auto tensorDgFp32 = dgFp32Buf.Get<float32_t>();

    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, H, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorDkbgin = dkbgInQue.AllocTensor<kType>();
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorGin = gInQue.AllocTensor<betaType>();
                    auto tensorDkin = dkInQue.AllocTensor<kType>();
                    auto tensordDbetaIn = dbetaInQue.AllocTensor<betaType>();

                    DataCopy(tensorDkbgin, workSpaceTensor[kOffset], K * curRowNum);
                    DataCopy(tensorKin, kTensor[kOffset], K * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    DataCopyPad(tensorGin, gTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    DataCopy(tensorDkin, dkTensor[kOffset], K * curRowNum);
                    DataCopyPad(tensordDbetaIn, dbetaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});

                    dkbgInQue.EnQue(tensorDkbgin);
                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    gInQue.EnQue(tensorGin);
                    dkInQue.EnQue(tensorDkin);
                    dbetaInQue.EnQue(tensordDbetaIn);
                }
                // compute
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
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorGFP32, tensorGin, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorDbetaFP32, tensorDbetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                        DataCopy(tensorGFP32, tensorGin, rowNum);
                        DataCopy(tensorDbetaFP32, tensorDbetain, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * curRowNum);
                    Cast(tensorDkbgFp32, tensorDkbgin, RoundMode::CAST_NONE, K * curRowNum);

                    PipeBarrier<PIPE_V>();
                    // exp(g) ->g
                    Exp(tensorGFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb(gexp)  brcb(beta)
                    Brcb(tensorGbrcbFP32, tensorGFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // dkbg * gexp -> dkbg
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkbgFp32[perchannelResOffset], tensorDkbgFp32[perchannelResOffset], tensorGbrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // dkbg(dkbg * gexp) * k ->k

                    Mul(tensorKFp32, tensorKFp32, tensorDkbgFp32, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    perchannelResOffset = 0;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }

                    // reducesum 
                    for (uint32_t row = 0; row < curRowNum; row++) {
                        WholeReduceSum(tensorKFp32[row * FP32_PER_BLOCK_8], tensorKFp32[row * K], FP32_PER_REPEAT_64,
                                       wholeReduceSumCnt, 1, 1, 8);
                        WholeReduceSum(tensorDkFp32[row * FP32_PER_BLOCK_8], tensorDkFp32[row * K], FP32_PER_REPEAT_64,
                                       wholeReduceSumCnt, 1, 1, 8);
                    }
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(tensorDbetaAddFP32, tensorKFp32, wholeReduceSumCnt, curRowNum, 1, 1, 1);
                    WholeReduceSum(tensorDgFp32, tensorDkFp32, wholeReduceSumCnt, curRowNum, 1, 1, 1);
                    PipeBarrier<PIPE_V>();
                    // cast dg
                    Cast(tensorDgOut, tensorDgFp32, RoundMode::CAST_RINT, curRowNum);
                    // 累加第二部分产生的处理原始dbeta
                    Add(tensorDbetaAddFP32, tensorDbetaAddFP32, tensorDbetaFP32, curRowNum);
                    // 计算当前dk公式为 dkbg(dkbg * gexp) * beta -> dkbg
                    perchannelResOffset = 0;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkbgFp32[perchannelResOffset], tensorDkbgFp32[perchannelResOffset],
                            tensorBetaBrcbFP32, FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    Cast(tensorDkFp32, tensorDkin, RoundMode::CAST_NONE, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // 累加第二部分产生的处理原始dk
                    Add(tensorDkFp32, tensorDkFp32, tensorDkbgFp32, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, K * curRowNum);
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorDbetaOut, tensorDbetaAddFP32, RoundMode::CAST_RINT, curRowNum);
                    } else {
                        DataCopy(tensorDbetaOut, tensorDbetaAddFP32, rowNum);
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
                // copyout
                {
                    auto tensorDkOut = dkOutQue.DeQue<kType>();
                    auto tensorDbetaOut = dBetaOutQue.DeQue<betaType>();
                    auto tensorDgOut = dgOutQue.DeQue<betaType>();
                    DataCopy(dkTensor[kOffset], tensorDkOut, K * curRowNum);
                    DataCopyPad(dbetaTensor[betaOffset], tensorDbetaOut, {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    DataCopyPad(dgTensor[betaOffset], tensorDgOut, {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    dkOutQue.FreeTensor(tensorDkOut);
                    dBetaOutQue.FreeTensor(tensorDbetaOut);
                    dgOutQue.FreeTensor(tensorDgOut);
                }
            }

            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkb()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t wholeReduceSumCnt = CeilDiv(K, FP32_PER_REPEAT_64);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dkbInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dkOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(dBetaOutQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dkbFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(dkFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(reduceSumTmpBuf, rowNum * ONE_BLOCK_32);

    auto tensorDkbFp32 = dkbFp32Buf.Get<float32_t>();
    auto tensorDkFp32 = dkFp32Buf.Get<float32_t>();
    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();
    auto tensorReduceSum = reduceSumTmpBuf.Get<float32_t>();

    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, H, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorDkbin = dkbInQue.AllocTensor<kType>();
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorDkin = dkInQue.AllocTensor<kType>();

                    DataCopy(tensorDkbin, workSpaceTensor[kOffset], K * curRowNum);
                    DataCopy(tensorKin, kTensor[kOffset], K * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    DataCopy(tensorDkin, dkTensor[kOffset], K * curRowNum);

                    dkbInQue.EnQue(tensorDkbin);
                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    dkInQue.EnQue(tensorDkin);
                }
                // compute
                {
                    auto tensorDkbin = dkbInQue.DeQue<kType>();
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorDkin = dkInQue.DeQue<kType>();

                    auto tensorDkOut = dkOutQue.AllocTensor<kType>();
                    auto tensorDbetaOut = dBetaOutQue.AllocTensor<betaType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * curRowNum);
                    Cast(tensorDkbFp32, tensorDkbin, RoundMode::CAST_NONE, K * curRowNum);
                    Cast(tensorDkFp32, tensorDkin, RoundMode::CAST_NONE, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    Mul(tensorKFp32, tensorKFp32, tensorDkbFp32, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorDkbFp32[perchannelResOffset], tensorDkbFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    // reducesum
                    for (uint32_t row = 0; row < curRowNum; row++) {
                        WholeReduceSum(tensorReduceSum[row * FP32_PER_BLOCK_8], tensorKFp32[row * K],
                                       FP32_PER_REPEAT_64, wholeReduceSumCnt, 1, 1, 8);
                    }
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(tensorBetaFP32, tensorReduceSum, wholeReduceSumCnt, curRowNum, 1, 1, 1);
                    // ADD
                    PipeBarrier<PIPE_V>();
                    Add(tensorDkFp32, tensorDkFp32, tensorDkbFp32, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, K * curRowNum);
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorDbetaOut, tensorBetaFP32, RoundMode::CAST_RINT, curRowNum);
                    } else {
                        DataCopy(tensorDbetaOut, tensorBetaFP32, rowNum);
                    }

                    dkbInQue.FreeTensor(tensorDkbin);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    dkInQue.FreeTensor(tensorDkin);
                    dkOutQue.EnQue(tensorDkOut);
                    dBetaOutQue.EnQue(tensorDbetaOut);
                }
                // copyout
                {
                    auto tensorDkOut = dkOutQue.DeQue<kType>();
                    auto tensorDbetaOut = dBetaOutQue.DeQue<betaType>();
                    DataCopy(dkTensor[kOffset], tensorDkOut, K * curRowNum);
                    DataCopyPad(dbetaTensor[betaOffset], tensorDbetaOut, {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    dkOutQue.FreeTensor(tensorDkOut);
                    dBetaOutQue.FreeTensor(tensorDbetaOut);
                }
            }

            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBeta()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = kBeteVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(kBetaOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);

    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, H, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < H; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    DataCopy(tensorKin, kTensor[kOffset], K * curRowNum);
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});

                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                }
                // compute
                {
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorOut = kBetaOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, rowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * rowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorKFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    Cast(tensorOut, tensorKFp32, RoundMode::CAST_RINT, K * curRowNum);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    kBetaOutQue.EnQue(tensorOut);
                }
                // copyout
                {
                    auto tensorOut = kBetaOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[kOffset], tensorOut, K * curRowNum);
                    kBetaOutQue.FreeTensor(tensorOut);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_3);
        }
    }
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_5);
    return;
}


#endif // PREPARE_WY_REPR_BWD_FULL_VECTOR_H
