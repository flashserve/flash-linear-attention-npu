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
 * \file moe_init_routing_v2_grad_full_load.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V2_GRAD_FULL_LOAD_COMPUTE_H
#define MOE_INIT_ROUTING_V2_GRAD_FULL_LOAD_COMPUTE_H

#include "moe_init_routing_v2_grad_base.h"

namespace MoeInitRoutingV2Grad {
template <typename T, int64_t Mode = 0>
class MoeInitRoutingV2GradFullLoadCompute
{
public:
    __aicore__ inline MoeInitRoutingV2GradFullLoadCompute(
        const MoeInitRoutingV2GradRegbaseFullLoadTilingData* tilingData)
    {
        td_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR gradExpandedX, GM_ADDR expandedRowIdx, GM_ADDR gradX, TPipe* tPipe)
    {
        auto blockIdx = GetBlockIdx();

        this->singleN =
            (blockIdx == td_->blockDim - 1) ? (td_->n - td_->nBlockFactor * (td_->blockDim - 1)) : td_->nBlockFactor;
        int64_t indexGmOffset = td_->nBlockFactor * blockIdx * td_->k;
        int64_t outputGmOffset = td_->nBlockFactor * blockIdx * td_->h;
        inputGm.SetGlobalBuffer((__gm__ T*)gradExpandedX);
        indexGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx + indexGmOffset);
        outputGm.SetGlobalBuffer((__gm__ T*)gradX + outputGmOffset);

        tPipe->InitBuffer(inputQueue, DOUBLE_BUFFER, td_->nUbFactor * td_->k * td_->hUbFactor * sizeof(T));
        tPipe->InitBuffer(outputQueue, 1, td_->nUbFactor * td_->hUbFactor * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int64_t nQuotient = (this->singleN + td_->nUbFactor - 1) / td_->nUbFactor;
        int64_t hQuotient = (td_->h + td_->hUbFactor - 1) / td_->hUbFactor;
        for (int64_t nUbLoopIdx = 0; nUbLoopIdx < nQuotient; nUbLoopIdx++) {
            int64_t currentN =
                (nUbLoopIdx == (nQuotient - 1)) ? (this->singleN - (nQuotient - 1) * td_->nUbFactor) : td_->nUbFactor;
            int64_t nIndexOffset = nUbLoopIdx * td_->nUbFactor * td_->k;
            int64_t nOutputOffset = nUbLoopIdx * td_->nUbFactor * td_->h;
            for (int64_t hUbLoopIdx = 0; hUbLoopIdx < hQuotient; hUbLoopIdx++) {
                int64_t currentH =
                    (hUbLoopIdx == (hQuotient - 1)) ? (td_->h - (hQuotient - 1) * td_->hUbFactor) : td_->hUbFactor;
                int64_t currentHAlign =
                    (((currentH * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T);
                int64_t inputOffset = hUbLoopIdx * td_->hUbFactor;
                int64_t outputOffset = nOutputOffset + hUbLoopIdx * td_->hUbFactor;
                ProcessUB(nIndexOffset, inputOffset, outputOffset, currentN, currentH, currentHAlign);
            }
        }
    }

private:
    __aicore__ inline void ProcessUB(
        int64_t indexOffset, int64_t inputOffset, int64_t outputOffset, int64_t currentN, int64_t currentH,
        int64_t currentHAlign)
    {
        CopyIn(indexOffset, inputOffset, currentN, currentH, currentHAlign);
        Compute(currentN, currentH, currentHAlign);
        CopyOut(outputOffset, currentN, currentH, currentHAlign);
    }

    __aicore__ inline void CopyIn(
        int64_t indexOffset, int64_t inputOffset, int64_t currentN, int64_t currentH, int64_t currentHAlign)
    {
        LocalTensor<T> inputUb = inputQueue.AllocTensor<T>();
        for (int64_t nIdx = 0; nIdx < currentN; nIdx++) {
            int64_t nIndexOffset = nIdx * td_->k;
            int64_t nInputOffset = nIndexOffset * currentHAlign;
            for (int64_t kIdx = 0; kIdx < td_->k; kIdx++) {
                int64_t curIndexOffset = indexOffset + nIndexOffset + kIdx;
                int32_t rowIdx = indexGm.GetValue(curIndexOffset);
                if constexpr (Mode == DROP_PAD_MODE) {
                    if (rowIdx == -1) {
                        Duplicate(inputUb[nInputOffset + kIdx * currentHAlign], static_cast<T>(0.0f), currentHAlign);
                        continue;
                    }
                }
                if constexpr (Mode == ACTIVE_MODE) {
                    if (rowIdx >= td_->activeNum) {
                        Duplicate(inputUb[nInputOffset + kIdx * currentHAlign], static_cast<T>(0.0f), currentHAlign);
                        continue;
                    }
                }
                int64_t curInputOffset = rowIdx * td_->h + inputOffset;
                DataCopyPadExtParams<T> dataCopyPadExtParams;
                dataCopyPadExtParams.isPad = false;
                dataCopyPadExtParams.leftPadding = 0;
                dataCopyPadExtParams.rightPadding = 0;
                dataCopyPadExtParams.paddingValue = 0;
                DataCopyExtParams copyInParams;
                copyInParams.blockCount = 1;
                copyInParams.blockLen = currentH * sizeof(T);
                copyInParams.srcStride = 0;
                copyInParams.dstStride = 0;
                DataCopyPad(
                    inputUb[nInputOffset + kIdx * currentHAlign], inputGm[curInputOffset], copyInParams,
                    dataCopyPadExtParams);
            }
        }
        inputQueue.EnQue(inputUb);
    }

    __aicore__ inline void Compute(int64_t currentN, int64_t currentH, int64_t currentHAlign)
    {
        LocalTensor<T> inputUb = inputQueue.template DeQue<T>();
        LocalTensor<T> outputUb = outputQueue.AllocTensor<T>();
        __local_mem__ T* inputUbAddr = (__local_mem__ T*)inputUb.GetPhyAddr();
        __local_mem__ T* outputUbAddr = (__local_mem__ T*)outputUb.GetPhyAddr();

        SequenceReduceSum<T, T, true>(inputUbAddr, outputUbAddr, currentN, td_->k, currentH, currentHAlign, td_->k);

        inputQueue.FreeTensor(inputUb);
        outputQueue.EnQue(outputUb);
    }

    __aicore__ inline void CopyOut(int64_t outputOffset, int64_t currentN, int64_t currentH, int64_t currentHAlign)
    {
        LocalTensor<T> outputUb = outputQueue.template DeQue<T>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentN;
        copyInParams.blockLen = currentH * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (td_->h - currentH) * sizeof(T);
        DataCopyPad(outputGm[outputOffset], outputUb, copyInParams);
        outputQueue.FreeTensor(outputUb);
    }

    const MoeInitRoutingV2GradRegbaseFullLoadTilingData* __restrict td_;
    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<T> outputGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;

    int64_t n;
    int64_t nBlockFactor;
    int64_t nUbFactor;
    int64_t k;
    int64_t activeNum;
    int64_t h;
    int64_t hUbFactor;
    int64_t blockDim;

    int64_t singleN;
};
} // namespace MoeInitRoutingV2Grad
#endif // MOE_INIT_ROUTING_V2_GRAD_FULL_LOAD_COMPUTE_H