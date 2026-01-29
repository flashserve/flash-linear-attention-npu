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
 * \file chunk_gated_delta_rule_bwd_dhu_base.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_BASE_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_BASE_H
#include "kernel_operator.h"

using namespace AscendC;
namespace ChunkGDRBwdDhu {
constexpr uint64_t SYNC_AIC_AIV_FLAG_0 = 0;
constexpr uint64_t SYNC_AIC_AIV_FLAG_1 = 1;
constexpr uint64_t SYNC_AIC_AIV_FLAG_2 = 2;
constexpr uint64_t SYNC_AIC_AIV_FLAG_3 = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_4 = 4;
constexpr uint64_t SYNC_AIC_AIV_FLAG_5 = 5;
constexpr uint64_t HALF_DTYPE_SIZE = 2;
constexpr uint64_t FLOAT_DTYPE_SIZE = 4;
constexpr uint64_t FP32_PER_BLOCK = 8;
constexpr uint64_t FP32_PER_REPEAT = 64;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t EVENT_V_MTE2 = 1;
constexpr uint32_t EVENT_MTE2_V = 1;
constexpr uint32_t EVENT_MTE3_V = 1;
constexpr uint32_t EVENT_V_MTE3 = 1;
constexpr uint32_t EVENT_MTE2_MTE3 = 1;

template <typename DT>
class GDRBase {
public:
    __aicore__ inline GDRBase(){};

protected:
    __aicore__ inline void Process();
    __aicore__ inline void InitTilingData(const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData);
    __aicore__ inline void BroadCastAndMul(const LocalTensor<float>& broadCastSrcLocal,  const LocalTensor<float>& broadCastDstLocal, 
                                           const LocalTensor<float>& mulSrcLocal, const uint32_t dim0, const uint32_t dim1);
    TPipe pipe;
    TBuf<TPosition::VECCALC> vecTbuf;
    
    // inputGm
    GlobalTensor<DT> qGm;
    GlobalTensor<DTYPE_G> gGm;
    GlobalTensor<DT> dvGm;
    GlobalTensor<int64_t> cuSeqlensGm;
    // output gm, also used as input
    GlobalTensor<DT> dv2Gm;
    GlobalTensor<DT> dhGm;
    // inprocess workspace gm
    GlobalTensor<DT> bdvGm;
    GlobalTensor<DT> wv2Gm;
    GlobalTensor<DT> qdoGm;
    GlobalTensor<DT> gatedQGm;
    
    // calc gated q
    LocalTensor<DT> qLocal; // [BT/2,K]
    LocalTensor<float> qCastLocal;
    LocalTensor<DTYPE_G> gLocal; // [BT/2,]
    LocalTensor<DTYPE_G> gLastLocal;
    LocalTensor<float> gCastLocal;
    LocalTensor<float> gLastCastLocal;
    LocalTensor<float> gBCLocal;
    LocalTensor<float> gBrcbLocal;

    LocalTensor<float> gExpLocal;
    LocalTensor<float> gFatorLocal;
    
    // update dv2
    // LocalTensor<DT> dvLocal; // [BT/2,V]
    // LocalTensor<DT> bdvLocal; // [BT/2,V]
    LocalTensor<DT> vInLocal; // [BT/2,V]

    LocalTensor<float> dvCastLocal;
    LocalTensor<float> bdvCastLocal;

    // updated dh
    LocalTensor<DT> bdhLocal; // [K/2,V]
    LocalTensor<float> bdhCastLocal;
    LocalTensor<DT> wv2Local; // [K/2,V]
    LocalTensor<float> wv2CastLocal;
    LocalTensor<DT> qdoLocal; // [K/2,V]
    LocalTensor<float> qdoCastLocal;
    
    // tiling data
    uint64_t B = 0;
    uint64_t H = 0;
    uint64_t T = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t seqNum = 0;
    uint64_t gBufSize = 0;
    uint64_t dvBufSize = 0;
    uint64_t qBufSize = 0;
    uint64_t dhBufSize = 0;
    uint64_t totalTbufByte = 0;
    uint64_t bdvWs = 0;
    uint64_t qWs = 0;
    uint64_t wDv2Ws = 0;
    uint64_t qDoWs = 0;
    uint64_t isVarLen = 0;
    uint64_t isScale = 0;
    uint32_t usedCoreNum = 0;
    float  scale = 0;

    // global params
    uint32_t coreIdx = 0;
    uint32_t subBlockIdx = 0;
    uint32_t halfBT = 0;


};

template <typename DT>
__aicore__ inline void GDRBase<DT>::InitTilingData(const ChunkGatedDeltaRuleBwdDhuTilingData& tilingData)
{
    this->B = tilingData.B;
    this->H = tilingData.H;
    this->T = tilingData.T;
    this->K = tilingData.K;
    this->V = tilingData.V;
    this->chunkSize = tilingData.chunkSize;
    this->chunkNum = tilingData.chunkNum;
    this->seqNum = tilingData.seqNum;
    this->gBufSize = tilingData.gBufSize;
    this->dvBufSize = tilingData.dvBufSize;
    this->qBufSize = tilingData.qBufSize;
    this->dhBufSize = tilingData.dhBufSize;
    this->totalTbufByte = tilingData.totalTbufByte;
    this->bdvWs = tilingData.bdvWs;
    this->qWs = tilingData.qWs;
    this->wDv2Ws = tilingData.wDv2Ws;
    this->qDoWs = tilingData.qDoWs;
    this->isVarLen = tilingData.isVarLen;
    this->isScale = tilingData.isScale;
    this->usedCoreNum = tilingData.usedCoreNum;
    this->scale = tilingData.scale;
    this->coreIdx = GetBlockIdx();
    this->subBlockIdx = GetSubBlockIdx();
    this->halfBT = this->chunkSize / 2 ;

}

template <typename DT>
__aicore__ inline void GDRBase<DT>::BroadCastAndMul(const LocalTensor<float>& broadCastSrcLocal,  const LocalTensor<float>& broadCastDstLocal, 
                                                    const LocalTensor<float>& mulSrcLocal, const uint32_t dim0, const uint32_t dim1) 
{
    const uint32_t dstShape[] = {dim0, dim1};
    const uint32_t srcShape[] = {dim0, 1};
    BroadCast<float, 2, 1, false>(broadCastDstLocal, broadCastSrcLocal, srcShape, dstShape);
    for (int32_t bt = 0; bt < dim0; bt++) {
        Mul(mulSrcLocal[bt * this->halfBT], mulSrcLocal[bt * dim0], broadCastDstLocal[bt * dim0], dim1);
        // 加了自動同步
    }
}

template <typename CAST_DT, typename SRC_DT>
__aicore__ inline void CopyIn(const LocalTensor<CAST_DT>& castLocal, const LocalTensor<SRC_DT>& srcLocal, 
                                                   const GlobalTensor<SRC_DT>& srcGM, const uint32_t len, bool isCast=true) 
{
    SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2);
    WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2);
    PipeBarrier<PIPE_MTE2>();
    if (len % BLOCK_SIZE == 0) {
        DataCopy(srcLocal, srcGM, len);
    }

    SetFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(EVENT_MTE2_V);

    if (isCast) {
        Cast(castLocal, srcLocal, RoundMode::CAST_NONE, len); // half -> fp32
    }
}

template <typename CAST_DT, typename SRC_DT>
__aicore__ inline void CopyOut(const LocalTensor<CAST_DT>& castLocal, const LocalTensor<SRC_DT>& srcLocal, 
                                                   const GlobalTensor<CAST_DT>& dstGM, const uint32_t len, bool isCast=true) 
{
    if (isCast) {
        Cast(castLocal, srcLocal, RoundMode::CAST_RINT, len);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3);

    if (len % BLOCK_SIZE == 0) {
        DataCopy(dstGM, castLocal, len);
    }

    SetFlag<HardEvent::MTE3_V>(EVENT_MTE3_V);
    WaitFlag<HardEvent::MTE3_V>(EVENT_MTE3_V);
}

template <typename DT>
__aicore__ inline void BlockMul(const LocalTensor<DT>& src0Local, const LocalTensor<DT>& src1Local,
                                const LocalTensor<DT>& dstLocal, const uint32_t dim0, const uint32_t dim1) 
{
    // src0Local[dim0, dim1], src1Local[dim0, oneBlockSize]
    uint32_t offset = 0;
    uint32_t tailNum = dim1 % FP32_PER_REPEAT;

    uint8_t repeatStride = dim1 * sizeof(DT) / BLOCK_SIZE; // 一行有幾個block
    uint8_t dstBlkStride = 1;
    uint8_t src0BlkStride = 1;
    uint8_t src1BlkStride = 0;
    uint8_t dstRepStride = repeatStride;
    uint8_t src0RepStride = repeatStride;
    uint8_t src1RepStride = 1;
    BinaryRepeatParams param(dstBlkStride, src0BlkStride, src1BlkStride, 
                             dstRepStride, src0RepStride, src1RepStride);
    while (offset < dim1) {
        Mul(dstLocal[offset], src0Local[offset], src1Local, FP32_PER_REPEAT, dim0, param);
        offset += FP32_PER_REPEAT;
    }
    if (tailNum != 0) {
        Mul(dstLocal[offset], src0Local[offset], src1Local, tailNum, dim0, param);
    }
}

}  // namespace ChunkGDRBwdDhu
#endif  // CHUNK_GATED_DELTA_RULE_BWD_DHU_BASE_H
