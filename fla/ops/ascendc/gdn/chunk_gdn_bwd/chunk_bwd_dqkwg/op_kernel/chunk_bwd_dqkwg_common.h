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

 constexpr uint64_t CONST_B = 1;
 constexpr uint64_t CONST_HV = 4;
 constexpr uint64_t CONST_HK = 4;
 constexpr uint64_t CONST_T = 2816;
 constexpr uint64_t CONST_K = 128;
 constexpr uint64_t CONST_V = 128;
 constexpr uint64_t CONST_BT = 64;
 constexpr uint64_t CONST_NUM_CHUNKS = 44;//CONST_T / CONST_BT;  // 32
 constexpr int32_t CAL_NUM_FLOAT = 64; // API一次能处理256B，能计算64个float元素

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
 constexpr uint64_t SYNC_AIC_AIV_FLAG_0 = 5;  // cube -> vector: 数据 ready (与基线一致)
 constexpr uint64_t SYNC_AIV_AIC_FLAG_0 = 3;  // vector -> cube: 信用 credit (与基线一致)

 // Cross-stage temporaries use a per-core group-aligned ring. Short-lived
 // temporaries use an independent shallow ring. The host passes the cross ring
 // depth in wsBtxKSyncSlotsPerHead; short depth is fixed here.
 constexpr uint32_t DqkwgShortRingDepth = 8;

 // short 环深自适应: dw/mm6/mul1 的存活窗口只需 2G-1 个 slot (G=groupRingDepth/4)。固定 8 是按最大 G=4 配的,
 // 但大 H / 大 BT 的 memory-bound case 实际 G=1~2 时, 深度 8 严重过配 -> 环装不进 L2 -> FixPipe/MTE2 疯狂 miss。
 // 按 G 收缩到 2G-1 (地板 2, 保证 G=1 时仍有双缓冲) 可大幅减小环、贴近 L2。cube/vector/tiling 必须用同一公式。
 __aicore__ inline uint32_t DqkwgShortRingDepthFromGroup(uint32_t groupRingDepth) {
     uint32_t d = groupRingDepth / 2;               // = 2G (>= 2G-1 所需余量; G=4 时 =8 复现原值, G<4 时收缩)
     return d >= 2 ? d : 2;                         // 地板 2 (G=1 仍保双缓冲)
 }

 // ---- chunk-group-major: group each core's chunks and run A->B->C->D in-group ----
 // G = crossRingDepth / 4. The final group may merge a small tail, so short
 // rings must have at least 2G-1 slots for the largest supported G=4.
 __aicore__ inline uint32_t DqkwgGroupSizeFromRingDepth(uint32_t ringDepth) {
     uint32_t groupSize = ringDepth / 4;
     return groupSize == 0 ? 1 : groupSize;
 }

 // Given this core's current group start, return the group end (exclusive).
 // Cube/vector must use the same grouping so ready/credit order and ring slots
 // match exactly.
 __aicore__ inline uint32_t DqkwgGroupEnd(uint32_t loopBase, uint32_t coreLoops, uint32_t coreNum, uint32_t ringDepth) {
     if (loopBase >= coreLoops || coreNum == 0) { return coreLoops; }
     uint32_t groupSize = DqkwgGroupSizeFromRingDepth(ringDepth);
     uint32_t left = (coreLoops - loopBase + coreNum - 1) / coreNum;     // 本核从 loopBase 起还剩几个 chunk
     uint32_t take = (left <= 2 * groupSize - 1) ? left : groupSize;  // 尾巴合并: 剩 <=2G-1 整块取完
     uint32_t end = loopBase + take * coreNum;
     return (end < coreLoops) ? end : coreLoops;
 }

 __aicore__ inline uint64_t DqkwgGroupRingSlot(uint32_t coreIdx, uint32_t loopBase, uint32_t loopIdx,
                                               uint32_t coreNum, uint32_t ringDepth) {
     uint32_t groupSize = DqkwgGroupSizeFromRingDepth(ringDepth);
     uint32_t firstChunk = (coreNum != 0) ? ((loopBase - coreIdx) / coreNum) : 0;
     uint32_t parity = (firstChunk / groupSize) % 2;
     uint32_t pos = (coreNum != 0) ? ((loopIdx - loopBase) / coreNum) : 0;
     uint64_t slotInCore = (uint64_t)parity * (2 * groupSize) + (uint64_t)pos;
     return (uint64_t)coreIdx * ringDepth + slotInCore;
 }

 __aicore__ inline uint64_t DqkwgShortRingSlot(uint32_t coreIdx, uint32_t loopIdx, uint32_t coreNum,
                                               uint32_t shortRingDepth) {
     uint32_t j = (coreNum != 0) ? ((loopIdx - coreIdx) / coreNum) : 0;
     return (uint64_t)coreIdx * shortRingDepth + (uint64_t)(j % shortRingDepth);
 }

 __aicore__ inline uint64_t DqkwgBtxKRingElemOffset(uint32_t coreIdx, uint32_t loopBase, uint32_t loopIdx,
                                                    uint32_t coreNum, uint32_t h, uint64_t H, uint64_t BT, uint64_t K,
                                                    uint32_t ringDepth) {
     uint64_t slot = DqkwgGroupRingSlot(coreIdx, loopBase, loopIdx, coreNum, ringDepth);
     return (slot * H + (uint64_t)h) * (BT * K);
 }

 __aicore__ inline uint64_t DqkwgBtbRingElemOffset(uint32_t coreIdx, uint32_t loopBase, uint32_t loopIdx,
                                                   uint32_t coreNum, uint32_t h, uint64_t H, uint64_t BT,
                                                   uint32_t ringDepth) {
     uint64_t slot = DqkwgGroupRingSlot(coreIdx, loopBase, loopIdx, coreNum, ringDepth);
     return (slot * H + (uint64_t)h) * (BT * BT);
 }

 __aicore__ inline uint64_t DqkwgScalarRingElemOffset(uint32_t coreIdx, uint32_t loopBase, uint32_t loopIdx,
                                                      uint32_t coreNum, uint32_t h, uint64_t H,
                                                      uint32_t ringDepth) {
     uint64_t slot = DqkwgGroupRingSlot(coreIdx, loopBase, loopIdx, coreNum, ringDepth);
     return slot * H + (uint64_t)h;
 }

 __aicore__ inline uint64_t DqkwgShortBtxKRingElemOffset(uint32_t coreIdx, uint32_t loopIdx, uint32_t coreNum,
                                                         uint32_t h, uint64_t H, uint64_t BT, uint64_t K,
                                                         uint32_t shortRingDepth) {
     uint64_t slot = DqkwgShortRingSlot(coreIdx, loopIdx, coreNum, shortRingDepth);
     return (slot * H + (uint64_t)h) * (BT * K);
 }

 __aicore__ inline uint64_t DqkwgShortBtbRingElemOffset(uint32_t coreIdx, uint32_t loopIdx, uint32_t coreNum,
                                                        uint32_t h, uint64_t H, uint64_t BT,
                                                        uint32_t shortRingDepth) {
     uint64_t slot = DqkwgShortRingSlot(coreIdx, loopIdx, coreNum, shortRingDepth);
     return (slot * H + (uint64_t)h) * (BT * BT);
 }

 // cube 端: 产出 ready (FixPipe 写回 GM 后) / 取一个信用 (节流, 默认 wait 模式与基线一致)
 __aicore__ inline void SetCubeReady() {
     AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_AIC_AIV_FLAG_0);
 }
 __aicore__ inline void WaitCredit() {
     AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG_0);
 }
 // vector 端: 等待 cube ready (MTE2 读 GM 前) / 回一个信用 (MTE3 写 GM 后)
 __aicore__ inline void WaitCubeReady() {
     AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_0);
 }
 __aicore__ inline void SetVecCredit() {
     AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_0);
 }

 constexpr uint32_t UB_SIZE = 192 * 1024;  // 192KB
 constexpr uint32_t ONE_BLOCK_32 = 32;
 constexpr uint32_t FP32_PER_REPEAT = 64;
 constexpr uint32_t FP16_PER_BLOCK = 16;  // 32 bytes / 2 bytes per fp16

 constexpr uint32_t FP16_SIZE = 2;
 constexpr uint32_t FP32_SIZE = 4;
 constexpr uint32_t BF16_SIZE = 2;

 #define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
 #define ALIGN_UP(x, align) (((x) + (align) - 1) / (align) * (align))

 template<typename T>
 struct TypeTraits {
     using ComputeType = float;  // 默认计算类型为 fp32
     static constexpr bool needsCast = true;
 };

 template<>
 struct TypeTraits<half> {
     using ComputeType = float;
     static constexpr bool needsCast = true;
 };

 __aicore__ void inline GetChunkOffset(GM_ADDR cu_seqlens, GM_ADDR chunk_indices, uint64_t B, uint64_t HV, uint64_t T,
                                       uint64_t chunkSize, uint32_t loopIdx, uint32_t &bos, uint32_t &eos)
 {
     if (cu_seqlens == nullptr) {
         uint32_t coreLoopsInB = CEIL_DIV(T, chunkSize);
         uint32_t chunkIdx = loopIdx % coreLoopsInB;
         uint32_t bIdx = loopIdx / coreLoopsInB;
         bos = chunkIdx * chunkSize;
         eos = bos + chunkSize > T ? T : bos + chunkSize;
         bos += (bIdx * HV * T);
         eos += (bIdx * HV * T);
     } else {
         AscendC::GlobalTensor<uint64_t> cuSeqlensTensor;
         AscendC::GlobalTensor<uint64_t> chunkIndicesTensor;
         cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)cu_seqlens);
         chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)chunk_indices);

         uint32_t seqIdx = chunkIndicesTensor.GetValue(2 * loopIdx);
         uint32_t chunkIdx = chunkIndicesTensor.GetValue(2 * loopIdx + 1);
         uint32_t curSeqBegin = cuSeqlensTensor.GetValue(seqIdx);
         uint32_t curSeqEnd = cuSeqlensTensor.GetValue(seqIdx + 1);
         bos = curSeqBegin + chunkIdx * chunkSize;
         eos = bos + chunkSize > curSeqEnd ? curSeqEnd : bos + chunkSize;
     }

     return;
 }

 #endif  // CHUNK_BWD_DQKWG_COMMON_H
