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

template <typename DT>
class GDRVec : public GDRBase<DT>
{
public:
    __aicore__ inline GDRVec(){};
    __aicore__ inline void Process();
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace, const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData);
private:
    __aicore__ inline void InitUB();
    __aicore__ inline void InitGlobalTensor(GM_ADDR q, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                            GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace);
}; // class GDRVec

template <typename DT>
__aicore__ inline void GDRVec<DT>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, 
                                        GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace, const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData)
{
    GDRBase<DT>::InitTilingData(tilingData);
    InitUB();
    InitGlobalTensor(q, dv, g, cu_seqlens, dv2, dh, workspace);
}

template <typename DT>
__aicore__ inline void GDRVec<DT>::InitUB() 
{   

    // | gCastLocal | gExpLocal | qLocal | q
    this->pipe.InitBuffer(this->vecTbuf, this->totalTbufByte);
    uint32_t offset = 0;
    // gLast and g
    this->gCastLocal = this->vecTbuf.template Get<float>(this->halfBT); // 32/64
    offset += this->halfBT * FLOAT_DTYPE_SIZE;
    uint64_t dv2Offset = offset;
    this->gExpLocal = this->vecTbuf.template GetWithOffset<float>(this->halfBT, offset);
    offset += this->halfBT * FLOAT_DTYPE_SIZE;
    uint32_t offsetQ = offset;
    this->gLocal = this->vecTbuf.template GetWithOffset<DTYPE_G>(this->halfBT, offset); // 32/64
    offset += this->halfBT * HALF_DTYPE_SIZE;
    this->gLastLocal = this->vecTbuf.template GetWithOffset<DTYPE_G>(this->halfBT, offset); // bf16時，負責前半段的核要把後半段也搬進來拿last值
    offset += this->halfBT * HALF_DTYPE_SIZE;
    this->gLastCastLocal = this->vecTbuf.template GetWithOffset<float>(this->halfBT, offset); // bf16時，負責前半段的核要把後半段也搬進來拿last值
    offset += this->halfBT * FLOAT_DTYPE_SIZE;
    
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
    // | bdhCastLocal 64 | qDoCastLocal 64 | wV2CastLocal 64 |
    // | 32 | bdhLocal   | 32 | qDoLocal   | 32 | wV2Local   |
    uint64_t offsetDh = 0;
    uint32_t halfDhBufByte = this->dhBufSize * HALF_DTYPE_SIZE;
    this->bdhCastLocal = this->vecTbuf.template Get<float>(this->dhBufSize); 
    this->bdhLocal = this->vecTbuf.template GetWithOffset<DT>(this->dhBufSize, halfDhBufByte);
    offsetDh += this->dhBufSize * FLOAT_DTYPE_SIZE;
    this->qdoCastLocal = this->vecTbuf.template GetWithOffset<float>(this->dhBufSize, offsetDh);
    this->qdoLocal = this->vecTbuf.template GetWithOffset<DT>(this->dhBufSize, offsetDh + halfDhBufByte);
    offsetDh += this->dhBufSize * FLOAT_DTYPE_SIZE;
    this->wv2CastLocal = this->vecTbuf.template GetWithOffset<float>(this->dhBufSize, offsetDh);
    this->wv2Local = this->vecTbuf.template GetWithOffset<DT>(this->dhBufSize, offsetDh + halfDhBufByte);
    offsetDh += this->dhBufSize * FLOAT_DTYPE_SIZE;

}


template <typename DT>
__aicore__ inline void GDRVec<DT>::InitGlobalTensor(GM_ADDR q, GM_ADDR dv, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR dv2, GM_ADDR dh, GM_ADDR workspace)
{
    this->gGm.SetGlobalBuffer((__gm__ DTYPE_G *)g);
    this->qGm.SetGlobalBuffer((__gm__ DT *)q);
    this->dvGm.SetGlobalBuffer((__gm__ DT *)dv);
    this->dv2Gm.SetGlobalBuffer((__gm__ DT *)dv2);
    this->dhGm.SetGlobalBuffer((__gm__ DT *)dh);

    this->cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);

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

template <typename DT>
__aicore__ inline void GDRVec<DT>::Process( )
{
    uint32_t totalTaskNum = this->B * this->H * this->seqNum;
    uint32_t cubeIdx = this->coreIdx / 2; // 当前vec对应的cube核，两个vec核处理一个cube结果
    uint32_t BT = this->chunkSize;
    // sub_idx = 0处理前半段，1处理后半段
    for (uint32_t i = cubeIdx; i < totalTaskNum; i += this->usedCoreNum) {
        uint32_t seqIdx = i / this->H; // 当前任务在第几个seq
        uint32_t h = i % this->H; // 当前任务在第几个h
        // {0, 96, 224, 320} [0, 2, 4， 6]
        uint64_t seqStartOffset = this->cuSeqlensGm.GetValue(seqIdx); // 当前seq在T中的起始索引
        uint64_t seqEndOffset = this->cuSeqlensGm.GetValue(seqIdx+1); // 当前seq在T中的结束索引
        uint64_t curSeqLen = seqEndOffset - seqStartOffset;
        uint64_t tailChunkLen = curSeqLen % BT; 
        // 计算当前seq的起始chunkIdx
        uint64_t preChunkNum = 0;
        uint64_t tmpStartOffset = 0;
        uint64_t tmpEndOffset = 0;
        uint64_t tmpChunkNum = 0;
        for (uint32_t seq = 0; seq < seqIdx; seq++) {
            tmpStartOffset = this->cuSeqlensGm.GetValue(seq); // 当前seq在T中的起始索引
            tmpEndOffset = this->cuSeqlensGm.GetValue(seq+1); // 当前seq在T中的结束索引
            auto tmpChunkNum = ((tmpEndOffset - tmpStartOffset) + BT - 1) / BT;
            preChunkNum += tmpChunkNum;
        }
        int32_t curChunkNum = (curSeqLen + BT - 1) / BT; // 当前seq的chunk数
        
        uint64_t b = 0;
        uint64_t dhBlockSize = this->K * this->V;
        uint64_t gOffset = (b * this->H + h) * this->T;
        uint64_t gatedQOffset = cubeIdx * BT * this->K;
        uint64_t qdoOffset = cubeIdx * this->K * this->V;
        uint64_t wV2Offset = cubeIdx * this->K * this->V;
        uint64_t dhOffset = (b * this->H + h)* this->chunkNum * dhBlockSize + // [B,H,NT,K,V]定位到[NT,K,V] 
                            preChunkNum * dhBlockSize;   // 定位到當前seqence的起始chunk
        // calc offset
        uint64_t qGmOffset = 0;
        uint64_t dvGmOffset = 0;
        uint64_t tOffset = seqStartOffset;
        float gLast = 0.0;
        float gLastExp = 0.0;
        uint64_t bdvGmOffset = 0;
        uint64_t curDhOffset = 0;
        uint64_t nextDhOffset = 0;
        if (this->subBlockIdx == 1) {
            gatedQOffset += this->halfBT * this->K;
            qdoOffset +=  this->halfK * this->V;
            wV2Offset +=  this->halfK * this->V;
        }
        for (int32_t chunkIdx = curChunkNum - 1; chunkIdx >= 0; chunkIdx --) {
            if (this->subBlockIdx == 0) {
                bdvGmOffset = cubeIdx * BT * this->V;
                tOffset = seqStartOffset + chunkIdx * BT;

                CopyIn(this->gLastCastLocal, this->gLastLocal, this->gGm[gOffset + tOffset + this->halfBT], this->halfBT);
                // PipeBarrier<PIPE_ALL>();
                gLast = this->gLastCastLocal.GetValue(static_cast<uint64_t>(this->halfBT - 1));
                Exp(this->gLastCastLocal, this->gLastCastLocal, this->halfBT);
                gLastExp = this->gLastCastLocal.GetValue(static_cast<uint64_t>(this->halfBT - 1));
            } else {
                bdvGmOffset = cubeIdx * BT * this->V + this->halfBT * this->V;
                tOffset = seqStartOffset + chunkIdx * BT + this->halfBT;
            }
            CopyIn(this->gCastLocal, this->gLocal, this->gGm[gOffset + tOffset], this->halfBT);
            Exp(this->gExpLocal, this->gCastLocal, this->halfBT);
            if (this->subBlockIdx == 1) {
                gLast = this->gCastLocal.GetValue(this->halfBT - 1);
                gLastExp = this->gExpLocal.GetValue(this->halfBT - 1);
            }
            // COPY IN Q [B,H,T,K]
            uint64_t qGmOffset = (b * this->H + h) * this->T * this->K + tOffset * this->K;  
            CopyIn(this->qCastLocal, this->qLocal, this->qGm[qGmOffset], this->qBufSize);
            // qCastLocal[halfBT, K] * gExp[halfBT, ] K=128,256 halfBT=32,64
            uint8_t repeatTimes = Ceil(this->halfBT, 8); // halfBT is 32 or 64
            Brcb(this->gBCLocal, this->gExpLocal, repeatTimes, {1,8});
            BlockMul(this->qCastLocal, this->gBCLocal, this->qCastLocal, 
                     this->halfBT, static_cast<uint32_t>(this->K));
            CopyOut(this->qLocal, this->qCastLocal, this->gatedQGm[gatedQOffset], this->qBufSize);
            CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_GQ); // 计算完一个chunk的gatedQ,通知cube可以开始计算gatedQ @ do

            // 計算dv2 dv2 = bdv * exp(bg_last - bg) + dv[B,H,T,V]
            dvGmOffset = (b * this->H + h) * this->T * this->V + tOffset * this->V;
            if (chunkIdx == curChunkNum -1) {
                // dv -> dv2 無需cast， 不依賴cube計算結果
                CopyIn(this->dvCastLocal, this->vInLocal, this->dvGm[dvGmOffset], this->dvBufSize, false);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_MTE2_MTE3);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_MTE2_MTE3);
                CopyOut(this->vInLocal, this->dvCastLocal, this->dv2Gm[dvGmOffset], this->qBufSize, false);
            } else {
                CopyIn(this->dvCastLocal, this->vInLocal, this->dvGm[dvGmOffset], this->dvBufSize);
                // 64k
                Muls(this->gCastLocal, this->gCastLocal, static_cast<float>(-1.0), this->halfBT);
                Adds(this->gCastLocal, this->gCastLocal, gLast, this->halfBT);
                Exp(this->gCastLocal, this->gCastLocal, this->halfBT);
                uint8_t repeatTimes = Ceil(this->halfBT, 8); // halfBT is 32 or 64
                Brcb(this->gBrcbLocal, this->gCastLocal, repeatTimes, {1,8});
                
                // halfBT * 32
                CrossCoreWaitFlag(CROSS_CORE_C2V_BDV); // cube计算完一个chunk的bdv,vec开始计算对应的dv2
                CopyIn(this->bdvCastLocal, this->vInLocal, this->bdvGm[bdvGmOffset], this->dvBufSize);
                BlockMul(this->bdvCastLocal, this->gBrcbLocal, this->bdvCastLocal, this->halfBT, this->V);
                Add(this->bdvCastLocal, this->bdvCastLocal, this->dvCastLocal, this->qBufSize);
                CopyOut(this->vInLocal, this->bdvCastLocal, this->dv2Gm[dvGmOffset], this->qBufSize);
            }
            CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_DV2); // 计算完一个chunk的dv2,通知cube可以开始计算w @ dv2 
            
            if (chunkIdx == 0) {
                // 每個chunk更新bdh給下個chunk用，chunkIdx=0作爲最後一個chunk，所有的chunk都已經更新完了，無需更新bdh。
                continue;
            }
            curDhOffset = dhOffset + chunkIdx * dhBlockSize + this->subBlockIdx * this->dhBufSize;
            nextDhOffset = curDhOffset - dhBlockSize;
            if (chunkIdx == curChunkNum -1) {
                // 初始化全零 dh_chunkIdx
                InitOutput<DT>(this->dhGm[curDhOffset], this->dhBufSize, 0); // 兩個vec核各初始化一半
            } else {
                CopyIn(this->bdhCastLocal, this->bdhLocal, this->dhGm[curDhOffset], this->dhBufSize);
                Muls(this->bdhCastLocal, this->bdhCastLocal, gLastExp, this->dhBufSize);
            }
            // dh_updated = dh_i-1 * exp(bg_last) + term1*scale - term2
            CrossCoreWaitFlag(CROSS_CORE_C2V_TERM1); 
            {
                CopyIn(this->qdoCastLocal, this->qdoLocal, this->qdoGm[qdoOffset], this->dhBufSize);
                Muls(this->qdoCastLocal, this->qdoCastLocal, this->scale, this->dhBufSize);
                if (chunkIdx != curChunkNum -1) {
                    Add(this->qdoCastLocal, this->bdhCastLocal, this->qdoCastLocal, this->dhBufSize);
                }
            }
            CrossCoreWaitFlag(CROSS_CORE_C2V_TERM2);
            {
                CopyIn(this->wv2CastLocal, this->wv2Local, this->wv2Gm[wV2Offset], this->dhBufSize);
                Muls(this->wv2CastLocal, this->wv2CastLocal, static_cast<float>(-1.0), this->dhBufSize);
                Add(this->qdoCastLocal, this->qdoCastLocal, this->wv2CastLocal, this->dhBufSize);
            }
            CopyOut(this->bdhLocal, this->qdoCastLocal, this->dhGm[nextDhOffset], this->dhBufSize);
            CrossCoreSetFlag<0x2, PIPE_MTE3>(CROSS_CORE_V2C_BDH);
        }
    }
}

} // namespace ChunkGDRBwdDhu