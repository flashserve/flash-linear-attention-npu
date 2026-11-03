// Finalize A2 private implementation; arithmetic/scheduling source: 07d1b942.
// Kept local so --ops=chunk_gated_delta_rule_bwd_finalize builds independently.
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
// Finalize 私有 DQ 的实际协议（每个 mixed core 独立）：
//
// task 按 chunk-major 执行 A -> B -> C -> D；每个 task 内遍历完整 HK 组。
// 每个 AIV 启动时向 clear(flag 4) 预置 3 次，B/C/D 各返还 1 次。
// AIC 每 task 在 A、B、C 末尾各 WaitClear，全部 task 后再 WaitClear 3 次。
// 因而 SetClear = 3 + 3*M，WaitClear = 3*M + 3，退出时没有 clear 余量。
// B_vector 在 ds_temp 写回后发 credit(flag 3)，C_cube 使用前恰好等待一次：
// SetCredit = WaitCredit = M，退出时没有 credit 余量。
// Cube 每 task 发 4 次 ready(flag 5)，每个 AIV 消费 4 次，退出时同样平衡。
// A_vector 的两个 AIV 使用 CrossCore 0x1/flag 8 对齐 dg_last 的复用；
// 该 CrossCore flag 与 HardEvent 的 0..7 本地事件属于不同命名空间。
//
// 不要按历史注释再加 drain：原 Cube Process 的末尾 3 次等待必须保留且仅执行一次。
// 所有阶段退出并 PipeBarrier<PIPE_ALL> 后，Finalize 的全参与 SyncAll 才允许
// 下一阶段 WY 重用 CrossCore flag 4 和同一 GM 临时区。
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
