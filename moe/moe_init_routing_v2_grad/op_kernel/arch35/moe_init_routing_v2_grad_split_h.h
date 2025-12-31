/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_v2_grad_split_h.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V2_GRAD_SPLIT_H_COMPUTE_H
#define MOE_INIT_ROUTING_V2_GRAD_SPLIT_H_COMPUTE_H

#include "moe_init_routing_v2_grad_base.h"

namespace MoeInitRoutingV2Grad {
template <typename T, int64_t Mode = 0>
class MoeInitRoutingV2GradSplitHCompute
{
public:
    __aicore__ inline MoeInitRoutingV2GradSplitHCompute(const MoeInitRoutingV2GradRegbaseSplitHTilingData* tilingData)
    {
        this->n = tilingData->n;
        this->kUbFactor = tilingData->kUbFactor;
        this->k = tilingData->k;
        this->activeNum = tilingData->activeNum;
        this->h = tilingData->h;
        this->hBlockFactor = tilingData->hBlockFactor;
        this->hUbFactor = tilingData->hUbFactor;
        this->blockDim = tilingData->blockDim;
    }

    __aicore__ inline void Init(GM_ADDR gradExpandedX, GM_ADDR expandedRowIdx, GM_ADDR gradX, TPipe* tPipe)
    {
        this->blockIdx = GetBlockIdx();

        int64_t outputGmOffset = this->hBlockFactor * this->blockIdx; // 输出stride = H
        inputGm.SetGlobalBuffer((__gm__ T*)gradExpandedX);
        indexGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
        outputGm.SetGlobalBuffer((__gm__ T*)gradX + outputGmOffset);
        auto hUbAlignFactor = Ops::Base::CeilAlign(static_cast<int64_t>(this->hUbFactor * sizeof(T)), BLOCK_SIZE);
        tPipe->InitBuffer(inputQueue, DOUBLE_BUFFER, this->kUbFactor * this->k * hUbAlignFactor);
        tPipe->InitBuffer(outputQueue, 1, this->kUbFactor * hUbAlignFactor);
    }

    __aicore__ inline void Process()
    {
        int64_t currhBlockFactor = this->hBlockFactor;
        if (this->blockIdx == this->blockDim - 1) {
            currhBlockFactor = this->h % this->hBlockFactor == 0 ? this->hBlockFactor : this->h % this->hBlockFactor;
        }
        int64_t splitHLoopCnt = Ops::Base::CeilDiv(currhBlockFactor, this->hUbFactor);
        int64_t currSubhLen = this->hUbFactor;
        for (int64_t splitHIdx = 0; splitHIdx < splitHLoopCnt; ++splitHIdx) {
            if (splitHIdx == splitHLoopCnt - 1) {
                currSubhLen =
                    currhBlockFactor % this->hUbFactor == 0 ? this->hUbFactor : currhBlockFactor % this->hUbFactor;
            }
            int64_t currSubhAlign =
                Ops::Base::CeilAlign(static_cast<int64_t>(currSubhLen * sizeof(T)), BLOCK_SIZE) / sizeof(T);
            // 一次可以搬入this->kUbFactor个topK, UB需要循环 n / this->kUbFactor次
            uint32_t loopCnt = Ops::Base::CeilDiv(this->n, this->kUbFactor);
            int64_t currkUbFactor = this->kUbFactor;
            for (uint32_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
                if (loopIdx == loopCnt - 1) {
                    currkUbFactor = this->n % this->kUbFactor == 0 ? this->kUbFactor : this->n % this->kUbFactor;
                }
                int64_t outputOffset = splitHIdx * this->hUbFactor + loopIdx * this->h * this->kUbFactor;
                ProcessUB(splitHIdx, outputOffset, loopIdx, currkUbFactor, currSubhLen, currSubhAlign);
            }
        }
    }

private:
    __aicore__ inline void ProcessUB(
        uint32_t splitHIdx, int64_t outputOffset, uint32_t loopIdx, int64_t currkUbFactor, int64_t currSubhLen,
        int64_t currSubhAlign)
    {
        CopyIn(splitHIdx, loopIdx, currkUbFactor, currSubhLen, currSubhAlign);
        Compute(currkUbFactor, currSubhLen, currSubhAlign);
        CopyOut(outputOffset, currkUbFactor, currSubhLen, currSubhAlign);
    }

    __aicore__ inline void CopyIn(
        uint32_t splitHIdx, uint32_t loopIdx, int64_t currkUbFactor, int64_t currSubhLen, int64_t currSubhAlign)
    {
        LocalTensor<T> inputUb = inputQueue.AllocTensor<T>();
        for (int64_t nIdx = 0; nIdx < currkUbFactor; nIdx++) {
            int64_t inputOffset = this->blockIdx * this->hBlockFactor + splitHIdx * this->hUbFactor;
            int64_t nIndexOffset = (loopIdx * this->kUbFactor + nIdx) * this->k;
            int64_t nInputOffset = nIdx * this->k * currSubhAlign;
            for (int64_t kIdx = 0; kIdx < this->k; kIdx++) {
                int64_t curIndexOffset = nIndexOffset + kIdx;
                int32_t rowIdx = indexGm.GetValue(curIndexOffset);
                if constexpr (Mode == DROP_PAD_MODE) {
                    if (rowIdx == -1) {
                        Duplicate(inputUb[nInputOffset + kIdx * currSubhAlign], static_cast<T>(0.0f), currSubhAlign);
                        continue;
                    }
                }
                if constexpr (Mode == ACTIVE_MODE) {
                    if (rowIdx >= this->activeNum) {
                        Duplicate(inputUb[nInputOffset + kIdx * currSubhAlign], static_cast<T>(0.0f), currSubhAlign);
                        continue;
                    }
                }
                int64_t curInputOffset = rowIdx * this->h + inputOffset;
                DataCopyPadExtParams<T> dataCopyPadExtParams;
                dataCopyPadExtParams.isPad = false;
                dataCopyPadExtParams.leftPadding = 0;
                dataCopyPadExtParams.rightPadding = 0;
                dataCopyPadExtParams.paddingValue = 0;
                DataCopyExtParams copyInParams;
                copyInParams.blockCount = 1;
                copyInParams.blockLen = currSubhLen * sizeof(T);
                copyInParams.srcStride = 0;
                copyInParams.dstStride = 0;
                DataCopyPad(
                    inputUb[nInputOffset + kIdx * currSubhAlign], inputGm[curInputOffset], copyInParams,
                    dataCopyPadExtParams);
            }
        }
        inputQueue.EnQue(inputUb);
    }

    __aicore__ inline void Compute(int64_t currkUbFactor, int64_t currSubhLen, int64_t currSubhAlign)
    {
        LocalTensor<T> inputUb = inputQueue.template DeQue<T>();
        LocalTensor<T> outputUb = outputQueue.AllocTensor<T>();
        __local_mem__ T* inputUbAddr = (__local_mem__ T*)inputUb.GetPhyAddr();
        __local_mem__ T* outputUbAddr = (__local_mem__ T*)outputUb.GetPhyAddr();
        SequenceReduceSum<T, T, true>(
            inputUbAddr, outputUbAddr, currkUbFactor, this->k, currSubhLen, currSubhAlign, this->k);
        inputQueue.FreeTensor(inputUb);
        outputQueue.EnQue(outputUb);
    }

    __aicore__ inline void CopyOut(
        int64_t outputOffset, int64_t currkUbFactor, int64_t currSubhLen, int64_t currSubhAlign)
    {
        LocalTensor<T> outputUb = outputQueue.template DeQue<T>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currkUbFactor;
        copyInParams.blockLen = currSubhLen * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->h - currSubhLen) * sizeof(T);
        DataCopyPad(outputGm[outputOffset], outputUb, copyInParams);
        outputQueue.FreeTensor(outputUb);
    }

    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> outputGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;

    int64_t n;
    int64_t kUbFactor;
    int64_t k;
    int64_t activeNum;
    int64_t h;
    int64_t hBlockFactor;
    int64_t hUbFactor;
    int64_t blockDim;
    uint32_t blockIdx;
};
} // namespace MoeInitRoutingV2Grad
#endif // MOE_INIT_ROUTING_V2_GRAD_SPLIT_H_COMPUTE_H