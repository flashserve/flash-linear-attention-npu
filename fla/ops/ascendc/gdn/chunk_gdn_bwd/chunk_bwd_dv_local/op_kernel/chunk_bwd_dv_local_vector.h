/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local_base.h
 * \brief
 */

#ifndef CHUNK_BWD_DV_LOCAL_VECTOR_H
#define CHUNK_BWD_DV_LOCAL_VECTOR_H

#include "kernel_operator.h"
#include "chunk_bwd_dv_local_common.h"

namespace GDN {
constexpr int32_t MASK_LINE_SIZE = 32;

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

    __aicore__ inline void ProcessChunk(const ChunkTaskIndex &chunkTask);

    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<QKVT> dOGm;
    AscendC::GlobalTensor<GT> gGm;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm;
    AscendC::GlobalTensor<QKVT> dVGm;
    AscendC::GlobalTensor<QKVT> workspaceGm;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gTQueIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> kqMatrixTQueIn;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFp32TBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gFactorTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskTBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroFp32TBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kqMatrixFp32TBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> kqMatrixTQueOut;

    AscendC::LocalTensor<float> maskLocalTensor;
    AscendC::LocalTensor<float> zeroFp32LocalTensor;

    int64_t headNum;
    int64_t seqLen;
    int64_t keyDim;
    int64_t valueDim;
    int64_t totalChunkTasks;
    int64_t aicCoreNum;
    int64_t subBlockNum;
    int64_t subBlockIdx;
    int64_t coreIdx;
    int64_t chunkSizeRepeatTime;
    uint8_t chunkSizeRepeatStride;
    float scale;
    int64_t vectorTaskIdx;         // vec 两个 subblock 交替分工时使用的任务计数
    int64_t workspaceHeadSlotNum;  // 每个 core 可同时保留的 head workspace 槽数

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

    headNum = tilingData->h;
    seqLen = tilingData->t;
    keyDim = tilingData->k;
    valueDim = tilingData->v;
    scale = tilingData->scale;
    totalChunkTasks = tilingData->b * strategy.chunkNumForT;
    aicCoreNum = static_cast<int64_t>(AscendC::GetBlockNum());
    subBlockNum = AscendC::GetSubBlockNum();
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx() / subBlockNum);
    subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
    chunkSizeRepeatTime = CeilDiv(strategy.chunkSize, CAL_NUM_FLOAT);
    chunkSizeRepeatStride = static_cast<uint8_t>(chunkSizeRepeatTime * 8);
    vectorTaskIdx = 0;
    workspaceHeadSlotNum = tilingData->headBufNum;

    pipe_ = pipe;
    // AIV 侧 UB 资源按“g 输入、Ws 输入、gating 中间矩阵、Ws 输出”拆分，避免 gating 和写回互相覆盖。
    pipe_->InitBuffer(gTQueIn, BUFFER_NUM, strategy.chunkSize * sizeof(GT));
    pipe_->InitBuffer(kqMatrixTQueIn, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / 2);
    pipe_->InitBuffer(kqMatrixTQueOut, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / 2);
    pipe_->InitBuffer(gFp32TBuf, strategy.chunkSize * SIZE_FLOAT);
    pipe_->InitBuffer(brcbTBuf, strategy.chunkSize * BLOCK_SIZE);
    pipe_->InitBuffer(maskTBuf, MASK_LINE_SIZE * MASK_LINE_SIZE * SIZE_FLOAT);
    pipe_->InitBuffer(zeroFp32TBuf, BLOCK_SIZE);
    pipe_->InitBuffer(gFactorTBuf, strategy.chunkSize * strategy.chunkSize * SIZE_FLOAT / NUM_2);
    pipe_->InitBuffer(kqMatrixFp32TBuf, strategy.chunkSize * strategy.chunkSize * SIZE_FLOAT / NUM_2);

    maskLocalTensor = maskTBuf.template Get<float>();
    zeroFp32LocalTensor = zeroFp32TBuf.template Get<float>();
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Process()
{
    // 预构造 32x32 下三角 mask 小块，后续按 row block 复用，减少每个 head 重复生成 mask 的开销。
    AscendC::Duplicate<float>(maskLocalTensor, float(1.0), MASK_LINE_SIZE * MASK_LINE_SIZE);
    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t index = 0; index < MASK_LINE_SIZE; index++) {
        AscendC::Duplicate<float>(maskLocalTensor[index * MASK_LINE_SIZE], float(0.0), index);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate<float>(zeroFp32LocalTensor, float(0.0), BLOCK_SIZE / SIZE_FLOAT);
    ChunkTaskIndex chunkTask;
    // AIV 与 AIC 使用相同的 chunk 轮询策略，确保两侧处理的 chunk/head 顺序严格一致。
    for (int64_t loopIdx = coreIdx; loopIdx < totalChunkTasks; loopIdx += aicCoreNum) {
        strategy.ResolveTask(loopIdx, chunkTask);
        ProcessChunk(chunkTask);
    }
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::ProcessChunk(const ChunkTaskIndex &chunkTask)
{
    // 切分点向上按 8 对齐，既保证 Brcb 源地址 32B 对齐，也保证两个 subblock 均不超过半块 UB 容量。
    int64_t alignedHalfRows = CeilDiv(chunkTask.chunkLen / NUM_2, 8) * 8;
    int64_t rowSplit = alignedHalfRows < chunkTask.chunkLen ? alignedHalfRows : chunkTask.chunkLen;
    int64_t rowStart = 0;
    int64_t rowEnd = 0;
    int64_t rowCount = 0;
    int64_t workspaceRowOffset = 0;
    int64_t chunkLenRepeatTime = CeilDiv(chunkTask.chunkLen, CAL_NUM_FLOAT);
    int64_t repeatOffset = 0;
    int64_t curSize = 0;

    AscendC::LocalTensor<float> gFp32LocalTensor = gFp32TBuf.template Get<float>();
    AscendC::Duplicate<float>(gFp32LocalTensor, float(0.0), strategy.chunkSize);
    AscendC::LocalTensor<float> gFactorLocalTensor = gFactorTBuf.template Get<float>();
    AscendC::LocalTensor<float> brcbLocalTensor = brcbTBuf.template Get<float>();
    AscendC::LocalTensor<float> kqMatrixFp32LocalTensor = kqMatrixFp32TBuf.template Get<float>();

    AscendC::PipeBarrier<PIPE_V>();
    for (int64_t hIndex = 0; hIndex < headNum; hIndex++) {
        int64_t wsOffset =
            GetCoreBufferSingleHWorkspaceOffset(coreIdx, hIndex, 0, strategy.chunkSize, workspaceHeadSlotNum);
        int64_t gOffset = chunkTask.batchId * headNum * seqLen + hIndex * seqLen + chunkTask.tokenStart;

        // 两个 subblock 之间交替分配任务
        ++vectorTaskIdx;
        if (vectorTaskIdx % subBlockNum != subBlockIdx) {
            rowStart = 0;
            rowEnd = rowSplit - 1;
            workspaceRowOffset = wsOffset;
        } else {
            rowStart = rowSplit;
            rowEnd = chunkTask.chunkLen - 1;
            workspaceRowOffset = wsOffset + rowSplit * strategy.chunkSize;
        }
        rowCount = rowEnd - rowStart + 1;

        if (rowCount <= 0) {
            // 即使当前 subblock 没有实际行，也必须参与 AIC/AIV 同步，避免另一侧等待 flag 死锁。
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_2);
            continue;
        }
        // 预先计算 gating，与 cube MMAD 重叠执行 ====
        // 先把当前 head/chunk 的 g 拉到 UB 并转成 fp32，后续所有 gating 运算都在 fp32 上完成。
        AscendC::LocalTensor<GT> gRawLocalTensor = gTQueIn.template AllocTensor<GT>();
        copyParams.blockLen = chunkTask.chunkLen * sizeof(GT);
        AscendC::DataCopyPad(gRawLocalTensor, gGm[gOffset], copyParams, gPadParams);
        gTQueIn.EnQue(gRawLocalTensor);
        gRawLocalTensor = gTQueIn.template DeQue<GT>();
        if constexpr (std::is_same<GT, float32_t>()) {
            if (chunkTask.chunkLen <= CHUNK_SIZE_64) {
                AscendC::Copy(gFp32LocalTensor, gRawLocalTensor, chunkTask.chunkLen, 1, {1, 1, 8, 8});
            } else {
                AscendC::Copy(gFp32LocalTensor, gRawLocalTensor, CAL_NUM_FLOAT, 1, {1, 1, 8, 8});
                AscendC::Copy(gFp32LocalTensor[CAL_NUM_FLOAT], gRawLocalTensor[CAL_NUM_FLOAT],
                              chunkTask.chunkLen - CAL_NUM_FLOAT, 1, {1, 1, 8, 8});
            }
        } else {
            AscendC::Cast(gFp32LocalTensor, gRawLocalTensor, AscendC::RoundMode::CAST_NONE, chunkTask.chunkLen);
        }
        AscendC::PipeBarrier<PIPE_V>();
        gTQueIn.FreeTensor(gRawLocalTensor);

        // 步骤 1：把 g_col 复制到 gFactorLocalTensor，并把 chunkLen 列广播成 rowCount 行
        // gFactor 初始保存每行相同的 g_col，矩阵形状为 rowCount x chunkSize。
        for (int64_t index = 0; index < chunkSizeRepeatTime; index++) {
            repeatOffset = index * CAL_NUM_FLOAT;
            curSize = index == chunkSizeRepeatTime - 1 ? strategy.chunkSize - repeatOffset : CAL_NUM_FLOAT;
            AscendC::Copy(gFactorLocalTensor[repeatOffset], gFp32LocalTensor[repeatOffset], curSize, rowCount,
                          {1, 1, chunkSizeRepeatStride, 0});
        }
        // 步骤 2：广播 g_row（rowSplit 按 8 对齐，因此 gFp32LocalTensor[rowStart] 也是 32B 对齐）
        // brcbLocalTensor 保存当前 subblock 负责行的 g_row，随后逐列执行 g_col - g_row。
        Brcb(brcbLocalTensor, gFp32LocalTensor[rowStart], CeilDiv(rowCount, 8), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t index = 0; index < chunkSizeRepeatTime; index++) {
            repeatOffset = index * CAL_NUM_FLOAT;
            curSize = index == chunkLenRepeatTime - 1 ? chunkTask.chunkLen - repeatOffset : CAL_NUM_FLOAT;
            AscendC::Sub(gFactorLocalTensor[repeatOffset], gFactorLocalTensor[repeatOffset], brcbLocalTensor, curSize,
                         rowCount, {1, 1, 0, chunkSizeRepeatStride, chunkSizeRepeatStride, 1});
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤 3：截断到 <= 0，防止 exp 溢出
        AscendC::Mins(gFactorLocalTensor, gFactorLocalTensor, float(0.0), rowCount * strategy.chunkSize);
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤 4：计算 exp(clamped g_col - g_row)，结果保证 <= 1.0
        for (int64_t index = 0; index < chunkSizeRepeatTime; index++) {
            repeatOffset = index * CAL_NUM_FLOAT;
            curSize = index == chunkLenRepeatTime - 1 ? chunkTask.chunkLen - repeatOffset : CAL_NUM_FLOAT;
            AscendC::Exp(gFactorLocalTensor[repeatOffset], gFactorLocalTensor[repeatOffset], curSize,
                         rowCount, {1, 1, chunkSizeRepeatStride, chunkSizeRepeatStride});
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 步骤 5：应用下三角 mask
        // mask 以 32 行为单位复用：完整的左侧列块直接清零，当前对角块乘 32x32 下三角模板。
        int64_t currentRow = rowStart;
        int64_t startSplitMaskId = rowStart / MASK_LINE_SIZE;
        int64_t endSplitMaskId = CeilDiv(rowEnd + 1, MASK_LINE_SIZE);
        for (int64_t index = startSplitMaskId; index < endSplitMaskId; index++) {
            repeatOffset = (currentRow - rowStart) * strategy.chunkSize;
            curSize = index == endSplitMaskId - 1 ? rowEnd - currentRow + 1 :
                                                    (index + 1) * MASK_LINE_SIZE - currentRow + 1;
            for (int64_t colChunkIndex = 0; colChunkIndex < index; colChunkIndex++) {
                AscendC::Copy(gFactorLocalTensor[repeatOffset + colChunkIndex * MASK_LINE_SIZE], zeroFp32LocalTensor,
                              MASK_LINE_SIZE, curSize, {1, 0, chunkSizeRepeatStride, 0});
            }
            repeatOffset = repeatOffset + index * MASK_LINE_SIZE;
            AscendC::Mul(gFactorLocalTensor[repeatOffset], gFactorLocalTensor[repeatOffset],
                         maskLocalTensor[(currentRow % MASK_LINE_SIZE) * MASK_LINE_SIZE], MASK_LINE_SIZE, curSize,
                         {1, 1, 1, chunkSizeRepeatStride, chunkSizeRepeatStride, MASK_LINE_SIZE * 4 / 32});
            currentRow = (index + 1) * MASK_LINE_SIZE;
        }
        AscendC::PipeBarrier<PIPE_V>();
        // 步骤 6：乘上缩放因子
        AscendC::Muls(gFactorLocalTensor, gFactorLocalTensor, scale, rowCount * strategy.chunkSize);

        // 等待 cube 第一阶段完成，此时 gating 已经预计算完成
        // 等到 AIC 把 K @ Q^T 写进 workspace 后，再把预计算好的 gating 因子乘到 Ws 上。
        AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);

        // 读取 workspace 中的 K @ Q^T，应用 gating 后再写回
        // 这里对 workspace 做 in-place 更新：原始 Ws -> gated Ws，供 AIC 的 P2 阶段直接消费。
        {
            AscendC::LocalTensor<QKVT> kqMatrixLocalTensor = kqMatrixTQueIn.AllocTensor<QKVT>();
            copyParams.blockLen = rowCount * strategy.chunkSize * sizeof(QKVT);
            AscendC::DataCopyPad(kqMatrixLocalTensor, workspaceGm[workspaceRowOffset], copyParams, qkvPadParams);
            kqMatrixTQueIn.EnQue(kqMatrixLocalTensor);
        }
        AscendC::LocalTensor<QKVT> kqMatrixLocalTensor = kqMatrixTQueIn.DeQue<QKVT>();

        AscendC::Cast(kqMatrixFp32LocalTensor, kqMatrixLocalTensor, AscendC::RoundMode::CAST_NONE,
                      rowCount * strategy.chunkSize);
        kqMatrixTQueIn.FreeTensor(kqMatrixLocalTensor);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(gFactorLocalTensor, gFactorLocalTensor, kqMatrixFp32LocalTensor, rowCount * strategy.chunkSize);
        AscendC::PipeBarrier<PIPE_V>();
        {
            AscendC::LocalTensor<QKVT> gatedKqMatrixLocalTensor = kqMatrixTQueOut.AllocTensor<QKVT>();
            AscendC::Cast(gatedKqMatrixLocalTensor, gFactorLocalTensor, AscendC::RoundMode::CAST_RINT,
                          rowCount * strategy.chunkSize);
            kqMatrixTQueOut.EnQue(gatedKqMatrixLocalTensor);
        }
        AscendC::LocalTensor<QKVT> gatedKqMatrixLocalTensor = kqMatrixTQueOut.DeQue<QKVT>();
        AscendC::DataCopy(workspaceGm[workspaceRowOffset], gatedKqMatrixLocalTensor, rowCount * strategy.chunkSize);
        kqMatrixTQueOut.FreeTensor(gatedKqMatrixLocalTensor);


        // 通知 AIC：当前 head 对应的 gated Ws 已经写回 workspace，可以开始 gated Ws @ dO。
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_2);

    }
}
} // namespace GDN
#endif // CHUNK_BWD_DV_LOCAL_VECTOR_H
