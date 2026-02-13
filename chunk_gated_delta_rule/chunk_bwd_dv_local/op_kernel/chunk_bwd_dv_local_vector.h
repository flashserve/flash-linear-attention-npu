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
    __aicore__ inline void Init(GM_ADDR d_o, GM_ADDR g, GM_ADDR upper_tri_matrix, GM_ADDR cu_seqlens,
                                GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
                                const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe = nullptr);

    __aicore__ inline void CopyIn(IndexResult indexResult, int64_t hIndex, int64_t taskStartLine, int64_t taskLineNum);

    __aicore__ inline void Process();

    __aicore__ inline void ProcessChunk(IndexResult indexResult);


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

    AscendC::LocalTensor<uint8_t> maskLocalTensor;
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
    float scale;

    int64_t vecTaskIdx;

    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<QKVT> qkvPadParams{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<GT> gPadParams{false, 0, 0, 0};
};

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Init(
    GM_ADDR d_o, GM_ADDR g, GM_ADDR upper_tri_matrix, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v,
    GM_ADDR workspace, const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe)
{
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    gGm.SetGlobalBuffer((__gm__ GT *)g);
    triMatrixGm.SetGlobalBuffer((__gm__ uint8_t *)upper_tri_matrix);
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
    vecTaskIdx = 0;

    pipe_ = pipe;
    // 181.5 KB
    pipe_->InitBuffer(gTQueIn, BUFFER_NUM, strategy.chunkSize * sizeof(GT));                              // 0.5
    pipe_->InitBuffer(gHalfTQueIn, BUFFER_NUM, strategy.chunkSize / NUM_2 * sizeof(GT));                  // 0.25
    pipe_->InitBuffer(kqTQueIn, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / 2);  // 64
    pipe_->InitBuffer(kqTQueOut, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / 2); // 32
    pipe_->InitBuffer(gFp32TBuf, strategy.chunkSize * SIZE_FLOAT);                                        // 0.5
    pipe_->InitBuffer(gFp32TBuf2, strategy.chunkSize / NUM_2 * SIZE_FLOAT);                               // 0.25
    pipe_->InitBuffer(brcbTBuf, strategy.chunkSize * BLOCK_SIZE);                                         // 4
    pipe_->InitBuffer(maskTBuf, strategy.chunkSize * strategy.chunkSize / BIT_NUM_FOR_UINT8);             // 16
    pipe_->InitBuffer(zeroFp32TBuf, BLOCK_SIZE);
    pipe_->InitBuffer(gFactorTBuf, strategy.chunkSize * strategy.chunkSize * SIZE_FLOAT / NUM_2); // 32
    pipe_->InitBuffer(kqFp32TBuf, strategy.chunkSize * strategy.chunkSize * SIZE_FLOAT / NUM_2);  // 32

    maskLocalTensor = maskTBuf.template Get<uint8_t>();
    zeroFp32LocalTensor = zeroFp32TBuf.template Get<float>();
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Process()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG_1);

    copyParams.blockLen = strategy.chunkSize * strategy.chunkSize / BIT_NUM_FOR_UINT8;
    AscendC::DataCopyPad(maskLocalTensor, triMatrixGm, copyParams, {false, 0, 0, 0});
    AscendC::Duplicate<float>(zeroFp32LocalTensor, float(0.0), BLOCK_SIZE / SIZE_FLOAT);
    MTE2ToVSync();
    for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        IndexResult indexResult = strategy.calculate(loopIdx);
        ProcessChunk(indexResult);
    }

    AscendC::SyncAll<false>();
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::ProcessChunk(IndexResult indexResult)
{
    int64_t padNum = strategy.chunkSize - indexResult.chunkLen;
    int64_t taskSplitLine = indexResult.chunkLen / MASK_ALIGN_LINE / NUM_2;
    taskSplitLine = MASK_ALIGN_LINE * taskSplitLine;
    int64_t taskStartLine = 0;
    int64_t taskEndLine = 0;
    int64_t taskLineNum = 0;
    int64_t taskOffset = 0;
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
            if (indexResult.chunkLen <= CHUNK_SIZE_64) {
                AscendC::Copy(gFp32LocalTensor, gLocalTensor, indexResult.chunkLen, 1, {1, 1, 8, 8});
            } else {
                AscendC::Copy(gFp32LocalTensor, gLocalTensor, CAL_NUM_FLOAT, 1, {1, 1, 8, 8});
                AscendC::Copy(gFp32LocalTensor[CAL_NUM_FLOAT], gLocalTensor[CAL_NUM_FLOAT],
                              indexResult.chunkLen - CAL_NUM_FLOAT, 1, {1, 1, 8, 8});
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

        // gFp32LocalTensor2 * -1
        AscendC::Muls(gFp32LocalTensor2, gFp32LocalTensor2, float(-1.0), taskLineNum);
        AscendC::PipeBarrier<PIPE_V>();
        // exp neg gLocalTensor
        AscendC::Exp(gFp32LocalTensor2, gFp32LocalTensor2, taskLineNum);

        // exp gFp32LocalTensor
        AscendC::Exp(gFp32LocalTensor, gFp32LocalTensor, indexResult.chunkLen);
        AscendC::PipeBarrier<PIPE_V>();

        // copy gFp32LocalTensor  chunkLen / 2行
        if (strategy.chunkSize == CHUNK_SIZE_64) {
            AscendC::Copy(gFactorLocalTensor, gFp32LocalTensor, CAL_NUM_FLOAT, taskLineNum, {1, 1, 8, 0});
        } else {
            AscendC::Copy(gFactorLocalTensor, gFp32LocalTensor, CAL_NUM_FLOAT, taskLineNum, {1, 1, 16, 0});

            AscendC::Copy(gFactorLocalTensor[CAL_NUM_FLOAT], gFp32LocalTensor[CAL_NUM_FLOAT], CAL_NUM_FLOAT,
                          taskLineNum, {1, 1, 16, 0});
        }

        // 计算 gFactor = gFp32LocalTensor *  exp neg gLocalTensor
        Brcb(brcbLocalTensor, gFp32LocalTensor2, CeilDiv(taskLineNum, 8), {1, 8}); // Brcb处理数据个数需要8对齐
        AscendC::PipeBarrier<PIPE_V>();
        if (strategy.chunkSize == CHUNK_SIZE_64) {
            AscendC::Mul(gFactorLocalTensor, gFactorLocalTensor, brcbLocalTensor, CAL_NUM_FLOAT, taskLineNum,
                         {1, 1, 0, 8, 8, 1});
        } else {
            AscendC::Mul(gFactorLocalTensor, gFactorLocalTensor, brcbLocalTensor, CAL_NUM_FLOAT, taskLineNum,
                         {1, 1, 0, 16, 16, 1});
            AscendC::Mul(gFactorLocalTensor[CAL_NUM_FLOAT], gFactorLocalTensor[CAL_NUM_FLOAT], brcbLocalTensor,
                         CAL_NUM_FLOAT, taskLineNum, {1, 1, 0, 16, 16, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 计算 gFactor = gFactor * mask 使用select
        // dstBlkStride, src0BlkStride, src1BlkStride, dstRepStride, src0RepStride, src1RepStride
        AscendC::BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
        AscendC::Select(gFactorLocalTensor, maskLocalTensor[taskStartLine * strategy.chunkSize / BIT_NUM_FOR_UINT8],
                        zeroFp32LocalTensor, gFactorLocalTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                        CAL_NUM_FLOAT, taskLineNum * strategy.chunkSize / CAL_NUM_FLOAT, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(gFactorLocalTensor, gFactorLocalTensor, scale, taskLineNum * strategy.chunkSize);
        AscendC::PipeBarrier<PIPE_V>();
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

        // 计算 out = gFactor * (k@q^T)
        AscendC::Mul(gFactorLocalTensor, gFactorLocalTensor, kqFp32LocalTensor, taskLineNum * strategy.chunkSize);
        AscendC::PipeBarrier<PIPE_V>();
        // 搬出
        {
            AscendC::LocalTensor<QKVT> kqOutLocalTensor = kqTQueOut.AllocTensor<QKVT>();
            AscendC::Cast(kqOutLocalTensor, gFactorLocalTensor, AscendC::RoundMode::CAST_NONE,
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