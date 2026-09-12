/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_common.h
 * \brief ChunkBwdDqkwg 通用常量和定义
 */

#ifndef CHUNK_BWD_DQKWG_COMMON_H
#define CHUNK_BWD_DQKWG_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t CAL_NUM_FLOAT = 64; // API一次能处理256B，能计算64个float元素
constexpr int32_t NUM_PER_REP_FP32 = 64;
constexpr int32_t MOD_64_MASK = 0x3f;
constexpr int32_t LOG2_64 = 6;

// ============================================================================
// CV 深融合同步信号 (chunk-interleaved A/B/C/D pipeline) —— raw 信用流水 (与基线同机制)
//
// 每个 mixed core 取自己的 chunk 子集; 任务流按 part-major / chunk-minor, 跨 4 stage 连续:
//   task = [A(c0..cM-1), B(c0..cM-1), C(c0..cM-1), D(c0..cM-1)]  (L = 4*M), 无 SyncAll。
//
// 为什么用 raw 信用 flag 而不是 CrossCoreFlagWithReverse:
//   反转 flag (prepare_wy 用) 本质是相位翻转的"近 lockstep 屏障", prepare_wy 的真正并发来自
//   workspace 双缓冲 (workspaceBufferCount=2) 让 AIC 写 task i+1 的 slot 同时 AIV 读 task i 的 slot。
//   本算子单次 per-chunk 反转握手 + 全量寻址没有这种错位, 反转 flag 会把 cube/vector 压成 lockstep
//   (窗口≈0) => 退化为串行 (cube_total + vector_total) => 性能劣化。
//   raw 信用 flag 是计数信号量 (与基线 new_dqkwg/main 完全相同的机制): vector 预置 N 个信用,
//   cube 每 task 先 WaitCredit (消耗 1 信用) 再算, 算完 SetReady; 这样 cube 可真正领先 vector N 个 task,
//   实现重叠。
//
// 信用流水 (连续, 单次预置/无 drain):
//   vector 启动时 (stage A 之前) 预置 N 个信用 (每个 AIV sub-block 各 SetCredit N 次, 0x2)。
//   cube  每 task : WaitCredit(节流到领先 <=N) -> 算该 chunk 全部 head -> SetCubeReady (PIPE_FIX)。
//   vector每 task : WaitCubeReady -> 处理全部 head -> SetVecCredit (PIPE_MTE3, 回 1 信用)。
//   计数: cube WaitCredit L 次; vector SetCredit (预置 N + L) 次 => 末尾余 N 信用 (单次 launch 无害,
//         无需 drain, 不会死锁)。cube SetCubeReady L 次 == 每 AIV sub-block WaitCubeReady L 次。
//
// 窗口 N 与正确性:
//   N <= M (每核 chunk 数) 保证 cube 领先不超过 M, 从而 C_cube(c)=task 2M+c 计算时 vector 已完成
//   >= 2M+c-N >= M+c = B_vector(c) => ds_temp(c) 已就绪; 同理 mm6 复用 wsDw、mm7 复用 wsMm5 安全。
//   In group mode N = min(groupSize, M), so C/D cannot outrun B_vector.
// ============================================================================
constexpr uint64_t SYNC_AIC_AIV_FLAG_0 = 5; // cube -> vector: 数据 ready (与基线一致)
constexpr uint64_t SYNC_AIV_AIC_FLAG_0 = 3; // vector -> cube: 信用 credit (与基线一致)
constexpr uint64_t SYNC_AIV_AIC_FLAG_1 = 4; // vector -> cube: 数据 clear

__aicore__ inline uint64_t DqkwgBtxKElemOffset(uint32_t coreIdx, uint32_t h, uint64_t H, uint64_t BT, uint64_t K)
{
    return (coreIdx * H + (uint64_t)h) * (BT * K);
}

__aicore__ inline uint64_t DqkwgBtbElemOffset(uint32_t coreIdx, uint32_t h, uint64_t H, uint64_t BT)
{
    return (coreIdx * H + (uint64_t)h) * (BT * BT);
}

__aicore__ inline uint64_t DqkwgScalarElemOffset(uint32_t coreIdx, uint32_t h, uint64_t H)
{
    return coreIdx * H + (uint64_t)h;
}

// cube 端: 产出 ready (FixPipe 写回 GM 后) / 取一个信用 (节流, 默认 wait 模式与基线一致)
__aicore__ inline void SetCubeReady()
{
    CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
}
__aicore__ inline void WaitCredit()
{
    CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
}
__aicore__ inline void WaitClear()
{
    CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_1);
}
// vector 端: 等待 cube ready (MTE2 读 GM 前) / 回一个信用 (MTE3 写 GM 后)
__aicore__ inline void WaitCubeReady()
{
    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
}
__aicore__ inline void SetVecCredit()
{
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
}
__aicore__ inline void SetVecClear()
{
    CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_1);
}

constexpr uint32_t FP32_PER_REPEAT = 64;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__aicore__ inline void GetChunkOffset(GlobalTensor<uint64_t> cuSeqlensTensor, GlobalTensor<uint64_t> chunkIndicesTensor, uint64_t B, uint64_t HV, uint64_t T,
                                      uint64_t chunkSize, uint32_t loopIdx, uint32_t &bos, uint32_t &eos, bool isVarLen)
{
    if (isVarLen) {
        uint32_t seqIdx = chunkIndicesTensor.GetValue(2 * loopIdx);
        uint32_t chunkIdx = chunkIndicesTensor.GetValue(2 * loopIdx + 1);
        uint32_t curSeqBegin = cuSeqlensTensor.GetValue(seqIdx);
        uint32_t curSeqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
        bos = curSeqBegin + chunkIdx * chunkSize;
        eos = bos + chunkSize > curSeqEnd ? curSeqEnd : bos + chunkSize;
    } else {
        uint32_t coreLoopsInB = CEIL_DIV(T, chunkSize);
        uint32_t chunkIdx = loopIdx % coreLoopsInB;
        uint32_t bIdx = loopIdx / coreLoopsInB;
        bos = chunkIdx * chunkSize;
        eos = bos + chunkSize > T ? T : bos + chunkSize;
        bos += (bIdx * HV * T);
        eos += (bIdx * HV * T);
    }
}

// count <= 64 * 256(16K)
__aicore__ inline void ReduceSumCustom(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
                                       uint32_t count)
{
    uint32_t repeatTimes = count >> LOG2_64;
    uint32_t tailCount = count & MOD_64_MASK;

    BinaryRepeatParams repeatParams = {1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0};
    if (likely(repeatTimes > 1)) {
        Add(srcLocal, srcLocal[NUM_PER_REP_FP32], srcLocal, NUM_PER_REP_FP32, repeatTimes - 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount > 0)) {
        Add(srcLocal, srcLocal[repeatTimes << LOG2_64], srcLocal, tailCount, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    WholeReduceSum(dstLocal, srcLocal, repeatTimes > 0 ? NUM_PER_REP_FP32 : count, 1, 0, 1, 0);
}

#endif // CHUNK_BWD_DQKWG_COMMON_H
