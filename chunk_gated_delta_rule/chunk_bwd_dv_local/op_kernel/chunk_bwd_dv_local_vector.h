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
 * \file chunk_bwd_dv_local_base.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_VECTOR_H
#define CHUNK_BWD_DV_LOCAL_VECTOR_H

#include "kernel_operator.h"

namespace GDN {

template <typename QKVT, typename GT, typename Strategy>
class ChunkBwdDvLocalVector {
private:
    Strategy strategy;

public:
    __aicore__ inline ChunkBwdDvLocalVector(const Strategy &s) : strategy(s)
    {
    }
    __aicore__ inline void Init(GM_ADDR d_o, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v,
                                GM_ADDR workspace, const ChunkBwdDvLocalTilingData *__restrict tilingData,
                                AscendC::TPipe *pipe = nullptr);

    __aicore__ inline void Process();

    __aicore__ inline void ProcessChunk(const IndexResult &indexResult);


    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<QKVT> dOGm;
    AscendC::GlobalTensor<GT> gGm;
    AscendC::GlobalTensor<uint8_t> triMatrixGm;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm;
    AscendC::GlobalTensor<QKVT> dVGm;
    AscendC::GlobalTensor<QKVT> workspaceGm;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gTQueIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gHalfTQueIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> kqTQueIn;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFp32TBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFp32TBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFactorTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroFp32TBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kqFp32TBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> kqTQueOut;

    AscendC::LocalTensor<float> maskLocalTensor;
    AscendC::LocalTensor<float> zeroFp32LocalTensor;

    int64_t H;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t coreLoops;
    int64_t blockNum;
    int64_t subBlockNum;
    int64_t subBlockIdx;
    int64_t coreIdx;
    int64_t chunkSizeRepeatTime;
    uint8_t chunkSizeRepeatStride;
    float scale;

    int64_t vecTaskIdx;

    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<QKVT> qkvPadParams{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<GT> gPadParams{false, 0, 0, 0};
};

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Init(
    GM_ADDR d_o, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
    const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe)
{
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    gGm.SetGlobalBuffer((__gm__ GT *)g);
    cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
    chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunk_indices);
    dVGm.SetGlobalBuffer((__gm__ QKVT *)d_v);
    workspaceGm.SetGlobalBuffer((__gm__ QKVT *)workspace);

    H = tilingData->h;
    T = tilingData->t;
    K = tilingData->k;
    V = tilingData->v;
    scale = tilingData->scale;
    coreLoops = tilingData->b * strategy.chunkNumForT;
    blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    subBlockNum = AscendC::GetSubBlockNum();
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx() / subBlockNum);
    subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
    chunkSizeRepeatTime = CeilDiv(strategy.chunkSize, CAL_NUM_FLOAT);
    chunkSizeRepeatStride = static_cast<uint8_t>(chunkSizeRepeatTime * 8);
    vecTaskIdx = 0;

    pipe_ = pipe;
    // 181.5 KB
    pipe_->InitBuffer(gTQueIn, BUFFER_NUM, strategy.chunkSize * sizeof(GT));
    pipe_->InitBuffer(gHalfTQueIn, BUFFER_NUM, strategy.chunkSize / NUM_2 * sizeof(GT));
    pipe_->InitBuffer(kqTQueIn, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / 2);
    pipe_->InitBuffer(kqTQueOut, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / 2);
    pipe_->InitBuffer(gFp32TBuf, strategy.chunkSize * SIZE_FLOAT);
    pipe_->InitBuffer(gFp32TBuf2, strategy.chunkSize / NUM_2 * SIZE_FLOAT);
    pipe_->InitBuffer(brcbTBuf, strategy.chunkSize * BLOCK_SIZE);
    pipe_->InitBuffer(maskTBuf, MASK_LINE_SIZE * MASK_LINE_SIZE * SIZE_FLOAT);
    pipe_->InitBuffer(zeroFp32TBuf, BLOCK_SIZE);
    pipe_->InitBuffer(gFactorTBuf, strategy.chunkSize * strategy.chunkSize * SIZE_FLOAT / NUM_2);
    pipe_->InitBuffer(kqFp32TBuf, strategy.chunkSize * strategy.chunkSize * SIZE_FLOAT / NUM_2);

    maskLocalTensor = maskTBuf.template Get<float>();
    zeroFp32LocalTensor = zeroFp32TBuf.template Get<float>();
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Process()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);

    AscendC::Duplicate<float>(maskLocalTensor, float(1.0), MASK_LINE_SIZE * MASK_LINE_SIZE);
    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t index = 0; index < MASK_LINE_SIZE; index++) {
        AscendC::Duplicate<float>(maskLocalTensor[index * MASK_LINE_SIZE], float(0.0), index);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate<float>(zeroFp32LocalTensor, float(0.0), BLOCK_SIZE / SIZE_FLOAT);
    IndexResult indexResult;
    for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        strategy.calculate(loopIdx, indexResult);
        ProcessChunk(indexResult);
    }

    AscendC::SyncAll<false>();
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::ProcessChunk(const IndexResult &indexResult)
{
    int64_t padNum = strategy.chunkSize - indexResult.chunkLen;
    int64_t taskSplitLine = indexResult.chunkLen / NUM_2;
    int64_t taskStartLine = 0;
    int64_t taskEndLine = 0;
    int64_t taskLineNum = 0;
    int64_t taskOffset = 0;
    int64_t chunkLenRepeatTime = CeilDiv(indexResult.chunkLen, CAL_NUM_FLOAT);
    int64_t taskRepeatTime = 0;
    int64_t repeatOffset = 0;
    int64_t curSize = 0;
    int64_t startSplitMaskId = 0;
    int64_t endSplitMaskId = 0;
    // 清零fp32 chunkSize tensor
    AscendC::LocalTensor<float> gFp32LocalTensor = gFp32TBuf.template Get<float>();
    AscendC::Duplicate<float>(gFp32LocalTensor, float(0.0), strategy.chunkSize);
    AscendC::LocalTensor<float> gFactorLocalTensor = gFactorTBuf.template Get<float>();
    AscendC::LocalTensor<float> brcbLocalTensor = brcbTBuf.template Get<float>();
    AscendC::LocalTensor<float> kqFp32LocalTensor = kqFp32TBuf.template Get<float>();

    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t hIndex = 0; hIndex < H; hIndex++) {
        int64_t baseOffset = indexResult.curBatchId * H * T * strategy.chunkSize + hIndex * T * strategy.chunkSize +
                             indexResult.curTokenId * strategy.chunkSize;
        ++vecTaskIdx;
        if (vecTaskIdx % subBlockNum != subBlockIdx) { // 设置任务起止行（左闭右闭）
            taskStartLine = 0;
            taskEndLine = taskSplitLine - 1;
            taskOffset = baseOffset;
        } else {
            taskStartLine = taskSplitLine;
            taskEndLine = indexResult.chunkLen - 1;
            taskOffset = baseOffset + taskSplitLine * strategy.chunkSize;
        }
        taskLineNum = taskEndLine - taskStartLine + 1;
        if (taskLineNum == 0) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
            continue;
        }
        // 搬入chunkSize g
        int64_t baseGOffset = indexResult.curBatchId * H * T + hIndex * T + indexResult.curTokenId;
        {
            AscendC::LocalTensor<GT> gLocalTensor = gTQueIn.AllocTensor<GT>();
            copyParams.blockLen = indexResult.chunkLen * sizeof(GT);
            AscendC::DataCopyPad(gLocalTensor, gGm[baseGOffset], copyParams, gPadParams);
            gTQueIn.EnQue(gLocalTensor);
        }
        AscendC::LocalTensor<GT> gLocalTensor = gTQueIn.DeQue<GT>();
        // chunkLen g -> fp32,其余位置清零
        if constexpr (std::is_same<GT, float32_t>()) {
            for (int64_t index = 0; index < chunkLenRepeatTime; index++) {
                repeatOffset = index * CAL_NUM_FLOAT;
                curSize = index == chunkLenRepeatTime - 1 ? indexResult.chunkLen - repeatOffset : CAL_NUM_FLOAT;
                AscendC::Copy(gFp32LocalTensor[repeatOffset], gLocalTensor[repeatOffset], curSize, 1, {1, 1, 8, 8});
            }
        } else {
            AscendC::Cast(gFp32LocalTensor, gLocalTensor, AscendC::RoundMode::CAST_NONE, indexResult.chunkLen);
        }
        gTQueIn.FreeTensor(gLocalTensor);
        {
            AscendC::LocalTensor<GT> gHalfLocalTensor = gHalfTQueIn.AllocTensor<GT>();
            copyParams.blockLen = taskLineNum * sizeof(GT);
            AscendC::DataCopyPad(gHalfLocalTensor, gGm[baseGOffset + taskStartLine], copyParams, gPadParams);
            gHalfTQueIn.EnQue(gHalfLocalTensor);
        }
        AscendC::LocalTensor<GT> gHalfLocalTensor = gHalfTQueIn.DeQue<GT>();

        AscendC::LocalTensor<float> gFp32LocalTensor2 = gFp32TBuf2.template Get<float>();
        if constexpr (std::is_same<GT, float32_t>()) {
            AscendC::Copy(gFp32LocalTensor2, gHalfLocalTensor, taskLineNum, 1, {1, 1, 8, 8});
        } else {
            AscendC::Cast(gFp32LocalTensor2, gHalfLocalTensor, AscendC::RoundMode::CAST_NONE, taskLineNum);
        }
        gHalfTQueIn.FreeTensor(gHalfLocalTensor);
        AscendC::PipeBarrier<PIPE_V>();
        // copy gFp32LocalTensor  chunkLen / 2行
        for (int64_t index = 0; index < chunkSizeRepeatTime; index++) {
            repeatOffset = index * CAL_NUM_FLOAT;
            curSize = index == chunkSizeRepeatTime - 1 ? strategy.chunkSize - repeatOffset : CAL_NUM_FLOAT;
            AscendC::Copy(gFactorLocalTensor[repeatOffset], gFp32LocalTensor[repeatOffset], curSize, taskLineNum,
                          {1, 1, chunkSizeRepeatStride, 0});
        }
        // 计算 gFactor = gFactorLocalTensor - gFp32LocalTensor2
        Brcb(brcbLocalTensor, gFp32LocalTensor2, CeilDiv(taskLineNum, 8), {1, 8}); // Brcb处理数据个数需要8对齐
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t index = 0; index < chunkSizeRepeatTime; index++) {
            repeatOffset = index * CAL_NUM_FLOAT;
            curSize = index == chunkLenRepeatTime - 1 ? indexResult.chunkLen - repeatOffset : CAL_NUM_FLOAT;
            AscendC::Sub(gFactorLocalTensor[repeatOffset], gFactorLocalTensor[repeatOffset], brcbLocalTensor, curSize,
                         taskLineNum, {1, 1, 0, chunkSizeRepeatStride, chunkSizeRepeatStride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 大于0的位置全部清0
        AscendC::Mins(gFactorLocalTensor, gFactorLocalTensor,  float(0.0), taskLineNum * strategy.chunkSize);
        AscendC::PipeBarrier<PIPE_V>();
        // exp(gFactor)
        for (int64_t index = 0; index < chunkSizeRepeatTime; index++) {
            repeatOffset = index * CAL_NUM_FLOAT;
            curSize = index == chunkLenRepeatTime - 1 ? indexResult.chunkLen - repeatOffset : CAL_NUM_FLOAT;
            AscendC::Exp(gFactorLocalTensor[repeatOffset], gFactorLocalTensor[repeatOffset], curSize,
                         taskLineNum, {1, 1, chunkSizeRepeatStride, chunkSizeRepeatStride});
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 根据mask矩阵分割处理当前核分到的chunk行(taskLineNum)
        int64_t curTaskLine = taskStartLine;
        startSplitMaskId = taskStartLine / MASK_LINE_SIZE;
        endSplitMaskId = CeilDiv(taskEndLine, MASK_LINE_SIZE);
        for (int64_t index = startSplitMaskId; index < endSplitMaskId; index++) {
            repeatOffset = (curTaskLine - taskStartLine) * strategy.chunkSize;
            curSize = index == endSplitMaskId - 1 ? taskEndLine - curTaskLine + 1 :
                                                    (index + 1) * MASK_LINE_SIZE - curTaskLine + 1;
            //  前面补0
            for (int64_t colChunkIndex = 0; colChunkIndex < index; colChunkIndex++) {
                AscendC::Copy(gFactorLocalTensor[repeatOffset + colChunkIndex * MASK_LINE_SIZE], zeroFp32LocalTensor,
                              MASK_LINE_SIZE, curSize, {1, 0, chunkSizeRepeatStride, 0});
            }
            // 上三角处理
            repeatOffset = repeatOffset + index * MASK_LINE_SIZE;
            AscendC::Mul(gFactorLocalTensor[repeatOffset], gFactorLocalTensor[repeatOffset],
                         maskLocalTensor[(curTaskLine % MASK_LINE_SIZE) * MASK_LINE_SIZE], MASK_LINE_SIZE, curSize,
                         {1, 1, 1, chunkSizeRepeatStride, chunkSizeRepeatStride, MASK_LINE_SIZE * 4 / 32});
            curTaskLine = (index + 1) * MASK_LINE_SIZE;
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(gFactorLocalTensor, gFactorLocalTensor, scale, taskLineNum * strategy.chunkSize);
        AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);

        // 搬入 (k@q^T)
        {
            AscendC::LocalTensor<QKVT> kqLocalTensor = kqTQueIn.AllocTensor<QKVT>();
            copyParams.blockLen = taskLineNum * strategy.chunkSize * sizeof(QKVT);
            AscendC::DataCopyPad(kqLocalTensor, workspaceGm[taskOffset], copyParams, qkvPadParams);
            kqTQueIn.EnQue(kqLocalTensor);
        }
        AscendC::LocalTensor<QKVT> kqLocalTensor = kqTQueIn.DeQue<QKVT>();
        AscendC::Cast(kqFp32LocalTensor, kqLocalTensor, AscendC::RoundMode::CAST_NONE,
                      taskLineNum * strategy.chunkSize);
        kqTQueIn.FreeTensor(kqLocalTensor);
        AscendC::PipeBarrier<PIPE_V>();
        // 计算 out = gFactor * (k@q^T)
        AscendC::Mul(gFactorLocalTensor, gFactorLocalTensor, kqFp32LocalTensor, taskLineNum * strategy.chunkSize);
        AscendC::PipeBarrier<PIPE_V>();
        // 搬出
        {
            AscendC::LocalTensor<QKVT> kqOutLocalTensor = kqTQueOut.AllocTensor<QKVT>();
            AscendC::Cast(kqOutLocalTensor, gFactorLocalTensor, AscendC::RoundMode::CAST_RINT,
                          taskLineNum * strategy.chunkSize);
            kqTQueOut.EnQue(kqOutLocalTensor);
        }

        AscendC::LocalTensor<QKVT> kqOutLocalTensor = kqTQueOut.DeQue<QKVT>();
        AscendC::DataCopy(workspaceGm[taskOffset], kqOutLocalTensor, taskLineNum * strategy.chunkSize);

        kqTQueOut.FreeTensor(kqOutLocalTensor);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    }
}

} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_VECTOR_H