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
 * \file chunk_gated_delta_rule_bwd_dhu_vec.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_VEC_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_VEC_H
#endif

#include <cmath>
#include "kernel_operator.h"
#include "chunk_gated_delta_rule_bwd_dhu_base.h"

using namespace AscendC;
namespace ChunkGDRBwdDhu {

template <typename DT, typename GT>
class GDRVec : public GDRBase<DT, GT>
{
public:
    __aicore__ inline GDRVec(){};
    __aicore__ inline void Process();
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace, const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData);
private:
    __aicore__ inline void TailChunkProcess(uint32_t tailChunkLen); 
    __aicore__ inline void InitUB();
    __aicore__ inline void InitGlobalTensor(GM_ADDR q, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                            GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace);
    __aicore__ inline void CaclOffset(const uint32_t taskIdx, uint64_t& tailChunkLen); 
    __aicore__ inline void CalcGatedQ(float& gLast, float& gLastExp, const bool isLastChunk); 
    __aicore__ inline void CalcDv2(const float gLast, uint64_t& curGmOffsetV, const bool isLastChunk); 
    __aicore__ inline void UpdateDh(const float gLastExp, uint64_t& curGmOffsetH, const bool isLastChunk); 


protected:
    uint64_t gmOffsetG_ = 0;
    uint64_t gmOffsetK_ = 0;
    uint64_t gmOffsetV_ = 0;
    uint64_t gmOffsetH_ = 0;

    uint64_t bdvOffset_ = 0;
    uint64_t gatedQOffset_ = 0;
    uint64_t qdoOffset_ = 0;
    uint64_t wV2Offset_ = 0;

    int32_t curChunkNum_ = 0;
    uint64_t dhBlockSize_ = 0;
    uint32_t cubeIdx_ = 0;
    uint32_t chunkIdx_ = 0;
    uint64_t bos_ = 0; // begin on seqence
}; // class GDRVec

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                        GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace, const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData)
{
    GDRBase<DT, GT>::InitTilingData(tilingData);
    InitUB();
    InitGlobalTensor(q, dv, g, cu_seqlens, dv2, dh, workspace);
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::InitUB() 
{   

    // | gCastLocal | gExpLocal | qLocal | q
    this->pipe.InitBuffer(this->vecTbuf, this->totalTbufByte);
    uint32_t offset = 0;
    // gLast and g
    // 申请chunkSize大小，subCore1只用前一半，sub0要用后一半拿gLast
    this->gCastLocal = this->vecTbuf.template Get<float>(this->chunkSize);
    offset += this->chunkSize * FLOAT_DTYPE_SIZE;
    uint64_t dv2Offset = offset;
    this->gExpLocal = this->vecTbuf.template GetWithOffset<float>(this->chunkSize, offset);
    offset += this->chunkSize * FLOAT_DTYPE_SIZE;
    uint32_t offsetQ = offset;
    this->gLocal = this->vecTbuf.template GetWithOffset<GT>(this->chunkSize, offset); // 32/64
    offset += this->chunkSize * (std::is_same<GT, float>::value ? FLOAT_DTYPE_SIZE : HALF_DTYPE_SIZE);
    
    // calc q_gated : q*gExp 
    this->qLocal = this->vecTbuf.template GetWithOffset<DT>(this->qBufSize, offsetQ); // 16k
    offsetQ += this->qBufSize * HALF_DTYPE_SIZE;
    this->qCastLocal = this->vecTbuf.template GetWithOffset<float>(this->qBufSize, offsetQ); // 32k
    offsetQ += this->qBufSize * FLOAT_DTYPE_SIZE;
    this->gBCLocal = this->vecTbuf.template GetWithOffset<float>(this->halfBT * FP32_PER_BLOCK, offsetQ); // 32k

    // cacl bdv * exp(bg_last-bg) + dv_ori
    // | gCastLocal | gBrcbLocal | dv/bdvLocal 32 | dvCast 64 | bdvCast 64 | 
    this->gBrcbLocal = this->vecTbuf.template GetWithOffset<float>(this->halfBT * FP32_PER_BLOCK, dv2Offset);
    dv2Offset += this->halfBT * BLOCK_SIZE;
    this->vInLocal = this->vecTbuf.template GetWithOffset<DT>(this->dvBufSize, dv2Offset); // 32k
    dv2Offset += this->dvBufSize * HALF_DTYPE_SIZE;
    this->dvCastLocal = this->vecTbuf.template GetWithOffset<float>(this->dvBufSize, dv2Offset); // 64k
    dv2Offset += this->dvBufSize * FLOAT_DTYPE_SIZE;
    this->bdvCastLocal = this->vecTbuf.template GetWithOffset<float>(this->dvBufSize, dv2Offset); // 64k
    // calc bdh = bdh + qdo*scale - wv2
    // | bdhCastLocal/wV2CastLocal 64 | qDoCastLocal 64 |
    // | 32 | bdhLocal/wV2Local       | 32 | qDoLocal   |
    uint64_t offsetDh = 0;
    uint32_t halfDhBufByte = this->dhBufSize * HALF_DTYPE_SIZE;
    this->bdhCastLocal = this->vecTbuf.template Get<float>(this->dhBufSize); 
    this->bdhLocal = this->vecTbuf.template GetWithOffset<DT>(this->dhBufSize, halfDhBufByte);
    // 复用
    this->wv2CastLocal = this->vecTbuf.template GetWithOffset<float>(this->dhBufSize, offsetDh);
    this->wv2Local = this->vecTbuf.template GetWithOffset<DT>(this->dhBufSize, offsetDh + halfDhBufByte);
    offsetDh += this->dhBufSize * FLOAT_DTYPE_SIZE;
    this->qdoCastLocal = this->vecTbuf.template GetWithOffset<float>(this->dhBufSize, offsetDh);
    this->qdoLocal = this->vecTbuf.template GetWithOffset<DT>(this->dhBufSize, offsetDh + halfDhBufByte);
    offsetDh += this->dhBufSize * FLOAT_DTYPE_SIZE;
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::InitGlobalTensor(GM_ADDR q, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace)
{
    this->gGm.SetGlobalBuffer((__gm__ GT *)g);
    this->qGm.SetGlobalBuffer((__gm__ DT *)q);
    this->dvGm.SetGlobalBuffer((__gm__ DT *)dv);
    this->dv2Gm.SetGlobalBuffer((__gm__ DT *)dv2);
    this->dhGm.SetGlobalBuffer((__gm__ DT *)dh);
    
    if (this->isVarLen) {
        this->cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
    }

    // workspace
    // | bdv | gatedQ | qDoWs | wDv2Ws |
    uint64_t wsOffset = 0;
    this->bdvGm.SetGlobalBuffer((__gm__ DT *)workspace + wsOffset);
    wsOffset += this->bdvWs; 
    this->gatedQGm.SetGlobalBuffer((__gm__ DT *)workspace + wsOffset);
    wsOffset += this->qWs;
    this->qdoGm.SetGlobalBuffer((__gm__ DT *)workspace + wsOffset);
    wsOffset += this->qDoWs;
    this->wv2Gm.SetGlobalBuffer((__gm__ DT *)workspace + wsOffset);
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::Process( )
{
    uint32_t totalTaskNum = this->B * this->H * this->seqNum;
    cubeIdx_ = this->coreIdx / 2; // 当前vec对应的cube核，两个vec核处理一个cube结果
    for (uint32_t i = cubeIdx_; i < totalTaskNum; i += this->usedCoreNum) {
        uint64_t tailChunkLen = 0;
        CaclOffset(i, tailChunkLen);
        float gLast = 0.0;
        float gLastExp = 0.0;
        uint64_t curGmOffsetV = 0;
        uint64_t curGmOffsetH = 0;
        bool isLastChunk = false;
        // last chunk process
        int32_t loopNum = curChunkNum_ - 1;
        if (tailChunkLen != 0) {
            this->curBT = tailChunkLen;
            TailChunkProcess(tailChunkLen);
            loopNum = curChunkNum_ - 2;
        }
        // 剩下的都是对齐的
        this->curBT = this->chunkSize;
        this->curCalcTK = this->qBufSize;
        this->curCalcTV = this->dvBufSize;
        this->curCalcBT = this->halfBT; 
        for (int32_t chunkIdx = loopNum; chunkIdx >= 0; chunkIdx --) {
            chunkIdx_ = chunkIdx;
            isLastChunk = chunkIdx_ == curChunkNum_ - 1 ? true : false;
            bos_ = chunkIdx_ * this->chunkSize;
            // gatedQ = q * gExp
            CalcGatedQ(gLast, gLastExp, isLastChunk);
            // 計算dv2 dv2 = bdv * exp(bg_last - bg) + dv[B,H,T,V]
            CalcDv2(gLast, curGmOffsetV, isLastChunk);
            if (chunkIdx_ > 0) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_DV2); // 计算完一个chunk的dv2,通知cube可以开始计算w @ dv2 
            }
            // updated dh
            if (chunkIdx_ == 0 && !isLastChunk) {
                // 每個chunk更新bdh給下個chunk用，chunkIdx=0作爲最後一個chunk，所有的chunk都已經更新完了，無需更新bdh。
                continue;
            }
            UpdateDh(gLastExp, curGmOffsetH, isLastChunk); 
        }
    }
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::TailChunkProcess(uint32_t tailChunkLen) 
{
    float gLast = 0.0;
    float gLastExp = 0.0;
    uint64_t curGmOffsetV = 0;
    uint64_t curGmOffsetH = 0;
    // 尾chunk大于halfBT时，0做halfBT， 1做剩下的； 只让0做
    if (tailChunkLen > this->halfBT) {
        this->curCalcBT = this->halfBT * (1 - this->subBlockIdx) +
                            (tailChunkLen - this->halfBT) * this->subBlockIdx;
    } else {
        this->curCalcBT =  tailChunkLen * (1 - this->subBlockIdx);
    }
    this->curCalcTK = this->curCalcBT * this->K;
    this->curCalcTV = this->curCalcBT * this->V;
    chunkIdx_ = curChunkNum_ - 1;
    bos_ = chunkIdx_ * this->chunkSize;
    // gatedQ = q * gExp
    CalcGatedQ(gLast, gLastExp, true);
    // 計算dv2 dv2 = bdv * exp(bg_last - bg) + dv[B,H,T,V]
    CalcDv2(gLast, curGmOffsetV, true);
    if (chunkIdx_ > 0) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_DV2); // 计算完一个chunk的dv2,通知cube可以开始计算w @ dv2 
    }
    // updated dh
    UpdateDh(gLastExp, curGmOffsetH, true); 
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::CaclOffset(const uint32_t taskIdx, uint64_t& tailChunkLen) 
{
    uint32_t BT = this->chunkSize;
    uint64_t b = 0;
    uint32_t h = taskIdx % this->H; // 当前任务在第几个h
    uint64_t preChunkNum = 0;
    uint64_t seqStartOffset = 0;
    uint64_t curSeqLen = 0;
    if (this->isVarLen) {
        uint32_t seqIdx = taskIdx / this->H; // 当前任务在第几个seq
        seqStartOffset = this->cuSeqlensGm.GetValue(seqIdx); // 当前seq在T中的起始索引
        uint64_t seqEndOffset = this->cuSeqlensGm.GetValue(seqIdx+1); // 当前seq在T中的结束索引
        curSeqLen = seqEndOffset - seqStartOffset;
        // 计算当前seq的起始chunkIdx
        uint64_t tmpStartOffset = 0;
        uint64_t tmpEndOffset = 0;
        uint64_t tmpChunkNum = 0;
        for (uint32_t seq = 0; seq < seqIdx; seq++) {
            tmpStartOffset = this->cuSeqlensGm.GetValue(seq); // 当前seq在T中的起始索引
            tmpEndOffset = this->cuSeqlensGm.GetValue(seq+1); // 当前seq在T中的结束索引
            auto tmpChunkNum = ((tmpEndOffset - tmpStartOffset) + BT - 1) / BT;
            preChunkNum += tmpChunkNum;
        }
        curChunkNum_ = (curSeqLen + BT - 1) / BT; // 当前seq的chunk数
    } else {
        curChunkNum_ = this->chunkNum;
        b = taskIdx / this->H;
        curSeqLen = this->T;
    }
    tailChunkLen = curSeqLen % BT; 
    
    dhBlockSize_ = this->K * this->V; // 16384

    bdvOffset_ = cubeIdx_ * BT * this->V;
    gatedQOffset_ = cubeIdx_ * BT * this->K;
    qdoOffset_ = cubeIdx_ * dhBlockSize_;
    wV2Offset_ = cubeIdx_ * dhBlockSize_;

    // calc offset
    gmOffsetK_ = (b * this->H + h) * this->T * this->K + seqStartOffset * this->K;
    gmOffsetV_ = (b * this->H + h) * this->T * this->V + seqStartOffset * this->V;
    gmOffsetH_ = (b * this->H + h) * this->chunkNum * dhBlockSize_ + preChunkNum * dhBlockSize_;
    gmOffsetG_ = (b * this->H + h) * this->T + seqStartOffset;

    if (this->subBlockIdx == 1) {
        gatedQOffset_ += this->halfBT * this->K;
        qdoOffset_ +=  this->halfK * this->V;
        wV2Offset_ +=  this->halfK * this->V;
        bdvOffset_ += this->halfBT * this->V;

        gmOffsetK_ += this->halfBT * this->K;
        gmOffsetV_ += this->halfBT * this->V;
        gmOffsetH_ += this->halfK * this->V;
        gmOffsetG_ += this->halfBT;
    }
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::CalcGatedQ(float& gLast, float& gLastExp, const bool isLastChunk) 
{
    if (this->curCalcBT == 0) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_GQ);
        return;
    }
    if constexpr (std::is_same<GT, float>::value) {
        if (this->subBlockIdx == 0) {
            CopyIn(this->gCastLocal, this->gCastLocal, this->gGm[gmOffsetG_ + bos_], this->curBT, false);
            Exp(this->gExpLocal, this->gCastLocal, this->curBT);
            gLast = this->gCastLocal.GetValue(static_cast<uint64_t>(this->curBT - 1));
            gLastExp = this->gExpLocal.GetValue(static_cast<uint64_t>(this->curBT - 1));
        } else {
            CopyIn(this->gCastLocal, this->gCastLocal, this->gGm[gmOffsetG_ + bos_], this->curCalcBT, false);
            Exp(this->gExpLocal, this->gCastLocal, this->curCalcBT);
            gLast = this->gCastLocal.GetValue(this->curCalcBT - 1);
            gLastExp = this->gExpLocal.GetValue(this->curCalcBT - 1);
        }
    } else {
        if (this->subBlockIdx == 0) {
            CopyIn(this->gCastLocal, this->gLocal, this->gGm[gmOffsetG_ + bos_], this->curBT);
            Exp(this->gExpLocal, this->gCastLocal, this->curBT);
            gLast = this->gCastLocal.GetValue(static_cast<uint64_t>(this->curBT - 1));
            gLastExp = this->gExpLocal.GetValue(static_cast<uint64_t>(this->curBT - 1));
        } else {
            CopyIn(this->gCastLocal, this->gLocal, this->gGm[gmOffsetG_ + bos_], this->curCalcBT);
            Exp(this->gExpLocal, this->gCastLocal, this->curCalcBT);
            gLast = this->gCastLocal.GetValue(this->curCalcBT - 1);
            gLastExp = this->gExpLocal.GetValue(this->curCalcBT - 1);
        }
    }
    if (chunkIdx_ == 0) {
        return; // chunkIdx==0时，不再需要更新dh，只需要拿到g和gLast用于计算dv2即可。
    }
    // COPY IN Q [B,H,T,K]
    CopyIn(this->qCastLocal, this->qLocal, this->qGm[gmOffsetK_ + bos_ * this->K], this->curCalcTK);
    // qCastLocal[halfBT, K] * gExp[halfBT, ] K=128,256 halfBT=32,64
    uint8_t repeatTimes = Ceil(this->halfBT, 8); // halfBT is 32 or 64
    Brcb(this->gBCLocal, this->gExpLocal, repeatTimes, {1,8});
    BlockMul(this->qCastLocal, this->gBCLocal, this->qCastLocal, 
                this->halfBT, static_cast<uint32_t>(this->K));
    CopyOut(this->qLocal, this->qCastLocal, this->gatedQGm[gatedQOffset_], this->curCalcTK);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_GQ); // 计算完一个chunk的gatedQ,通知cube可以开始计算gatedQ @ do
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::CalcDv2(const float gLast, uint64_t& curGmOffsetV, const bool isLastChunk) 
{
    curGmOffsetV = gmOffsetV_ + bos_ * this->V;
    if (isLastChunk) {
        // dv -> dv2 無需cast， 不依賴cube計算結果
        CopyIn(this->dvCastLocal, this->vInLocal, this->dvGm[curGmOffsetV], this->curCalcTV, false);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_MTE2_MTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_MTE2_MTE3);
        CopyOut(this->vInLocal, this->dvCastLocal, this->dv2Gm[curGmOffsetV], this->curCalcTV, false);
    } else {
        CopyIn(this->dvCastLocal, this->vInLocal, this->dvGm[curGmOffsetV], this->dvBufSize);
        Muls(this->gCastLocal, this->gCastLocal, static_cast<float>(-1.0), this->halfBT);
        Adds(this->gCastLocal, this->gCastLocal, gLast, this->halfBT);
        Exp(this->gCastLocal, this->gCastLocal, this->halfBT);
        uint8_t repeatTimes = Ceil(this->halfBT, FP32_PER_BLOCK); // halfBT is 32 or 64
        Brcb(this->gBrcbLocal, this->gCastLocal, repeatTimes, {1,FP32_PER_BLOCK});
        // halfBT * 32
        CrossCoreWaitFlag(CROSS_CORE_C2V_BDV); // cube计算完一个chunk的bdv,vec开始计算对应的dv2
        CopyIn(this->bdvCastLocal, this->vInLocal, this->bdvGm[bdvOffset_], this->dvBufSize);
        BlockMul(this->bdvCastLocal, this->gBrcbLocal, this->bdvCastLocal, this->halfBT, this->V);
        Add(this->bdvCastLocal, this->bdvCastLocal, this->dvCastLocal, this->dvBufSize);
        CopyOut(this->vInLocal, this->bdvCastLocal, this->dv2Gm[curGmOffsetV], this->dvBufSize);
    }
}

template <typename DT, typename GT>
__aicore__ inline void GDRVec<DT, GT>::UpdateDh(const float gLastExp, uint64_t& curGmOffsetH, const bool isLastChunk) 
{
    curGmOffsetH = gmOffsetH_ + chunkIdx_ * dhBlockSize_;
    if (isLastChunk) {
        // 初始化全零 dh_chunkIdx
        InitOutput<DT>(this->dhGm[curGmOffsetH], this->dhBufSize, 0); // 兩個vec核各初始化一半
    } else {
        CopyIn(this->bdhCastLocal, this->bdhLocal, this->dhGm[curGmOffsetH], this->dhBufSize);
        Muls(this->bdhCastLocal, this->bdhCastLocal, gLastExp, this->dhBufSize);
    }
    if (chunkIdx_ == 0) {
        return;
    }
    // dh_updated = dh_i-1 * exp(bg_last) + term1*scale - term2
    CrossCoreWaitFlag(CROSS_CORE_C2V_TERM1); 
    {
        CopyIn(this->qdoCastLocal, this->qdoLocal, this->qdoGm[qdoOffset_], this->dhBufSize);
        if (this->isScale) {
            Muls(this->qdoCastLocal, this->qdoCastLocal, this->scale, this->dhBufSize);
        }
        if (!isLastChunk) {
            Add(this->qdoCastLocal, this->bdhCastLocal, this->qdoCastLocal, this->dhBufSize);
        }
    }
    CrossCoreWaitFlag(CROSS_CORE_C2V_TERM2);
    {
        CopyIn(this->wv2CastLocal, this->wv2Local, this->wv2Gm[wV2Offset_], this->dhBufSize);
        Muls(this->wv2CastLocal, this->wv2CastLocal, static_cast<float>(-1.0), this->dhBufSize);
        Add(this->qdoCastLocal, this->qdoCastLocal, this->wv2CastLocal, this->dhBufSize);
    }
    CopyOut(this->bdhLocal, this->qdoCastLocal, this->dhGm[curGmOffsetH - dhBlockSize_], this->dhBufSize);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_BDH);
}

} // namespace ChunkGDRBwdDhu