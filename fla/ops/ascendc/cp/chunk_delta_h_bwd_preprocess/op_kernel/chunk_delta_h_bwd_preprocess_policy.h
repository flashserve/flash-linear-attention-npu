/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_delta_h_bwd_preprocess_policy.h
 * \brief
 * Stage / flag / buffer-ownership policy for chunk_delta_h_bwd_preprocess.
 *
 * 物理 Stage 划分（六 Stage，八 Stage 版本合并而来）：
 *
 *   共享        E 分支（E_r）                        P 分支（P_r）
 *   V0 ──▶ C1 ──▶ V2 ──▶ C3 ──▶ V4 ──────────────▶ C5
 *          └──────── 路 B（T1） ──┘
 *
 *   V0 · Vector  门控与衰减准备：Q̄s、K̄、decayK；[M,BT) 无效行写零；一次 VF
 *   C1 · Cube    路 A：dV_pre = K̄_c @ dH_old；路 B：T1 = W_c^T @ K̄_c（同一 Stage 两路，共用 K̄ 装载）
 *   V2 · Vector  dV̂' = -(dV_pre + dv_local)
 *   C3 · Cube    inc = Q̄s_c^T @ do_c + W_c^T @ dV̂'（两个 MMAD 累加进同一 L0C）
 *   V4 · Vector  路 A：dH_new = decayK_c ⊙ dH_old + inc；路 B：P_c = diag(decayK_c) - T1（共用 decayK）
 *   C5 · Cube    P_new = P_c @ P_old（K 方向归约分块，L0C 累加）
 *
 * 每个 Stage 只含一种计算引擎，且入口操作数必须在 Stage 入口前全部 ready；Cube 不消费本 Stage 新输出。
 */

#ifndef CHUNK_DELTA_H_BWD_PREPROCESS_POLICY_H
#define CHUNK_DELTA_H_BWD_PREPROCESS_POLICY_H

#include <cstdint>

namespace CP {

// Stage ID：顺序即依赖顺序
enum ChunkDeltaHBwdPreStage : uint32_t {
    STAGE_V0_GATE_PREP = 0,   // Vector：门控/衰减/操作数准备（Q̄s、K̄、decayK）
    STAGE_C1_FIRST_MMAD = 1,  // Cube：路 A dV_pre、路 B T1
    STAGE_V2_DV_HAT = 2,      // Vector：dV̂' = -(dV_pre + dv_local)
    STAGE_C3_INC = 3,         // Cube：inc = Q̄sᵀ @ do + Wᵀ @ dV̂'
    STAGE_V4_STATE = 4,       // Vector：路 A dH 更新、路 B P_c 对角注入
    STAGE_C5_CHAIN = 5,       // Cube：P_new = P_c @ P_old
    STAGE_NUM = 6,
};

// 同一个 Stage 内的两路输出：ready/free 必须各自独立，禁止合并成一条边
enum ChunkDeltaHBwdPrePath : uint32_t {
    PATH_A = 0,  // E 分支
    PATH_B = 1,  // P 分支
    PATH_NUM = 2,
};

// 每个 slot 里的模型 dtype 平面数：Q̄s 与 K̄（各 [chunkSize, K]），其后是 decayK[K] FP32
constexpr uint32_t NUM_SLOT_MODEL_PLANES = 2;

// 核间同步 flag id（AIC 与配对 AIV 之间）。
// v1 采用"同一 chunk 内 AIV/AIC 严格交替"的握手：每个 Stage 边界一个 flag id，
// 由于每次 set 都被对端 wait 一次，同一个 id 在 chunk 循环里复用是安全的。
// PROPOSED：pipeline 重叠版本需要改成带 reverse 的 credit 协议，届时要重新核对 flag 数量上限。
constexpr uint32_t CDHP_FLAG_V0_DONE = 0;  // AIV → AIC：slot（Q̄s/K̄/decayK）就绪
constexpr uint32_t CDHP_FLAG_C1_DONE = 1;  // AIC → AIV：dV_pre 与 T1 就绪
constexpr uint32_t CDHP_FLAG_V2_DONE = 2;  // AIV → AIC：dV̂' 就绪
constexpr uint32_t CDHP_FLAG_C3_DONE = 3;  // AIC → AIV：qterm 与 wterm 就绪
constexpr uint32_t CDHP_FLAG_V4_DONE = 4;  // AIV → AIC：dH 新值、P_c、P_bf16 就绪
constexpr uint32_t CDHP_FLAG_C5_DONE = 5;  // AIC → AIV：P 新值就绪（下一轮 V4 需要）

// set 之前先让本 pipe 排空（PipeBarrier 是"等待本 pipe 先前指令完成"的屏障）。
// 实测：不做排空时，cube 侧 C5 的 Fixpipe 落盘还没走完就把 flag 置起来，消费者在
// 非最后一个 task 上会读到"最后 16 行全 0"的半成品平面（E 不受影响、P 整链被污染）。
#define CDHP_AIV_SET(flag)                                  \
    do {                                                    \
        AscendC::PipeBarrier<PIPE_MTE3>();                   \
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag);     \
    } while (0)
// 注意：`CrossCoreWaitFlag(flag)` 的默认模板参数是 `pipe = PIPE_S`，在 A5（dav_3510）上
// WaitEventImpl 会按 pipe 只阻塞对应 pipe。若用默认值，等待只落在标量 pipe 上，
// 随后的 MTE2（GM→UB/L1）载入不会被拦在 flag 之后，会读到 cube 还没写完的中间平面
// （表现为 P/T1 相关的整块旧值，且随负载时好时坏）。这里显式绑定消费者 pipe = MTE2。
#define CDHP_AIV_WAIT(flag) AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag)
#define CDHP_AIC_SET(flag)                                 \
    do {                                                   \
        AscendC::PipeBarrier<PIPE_FIX>();                   \
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flag);     \
    } while (0)
#define CDHP_AIC_WAIT(flag) AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flag)

// 工作区平面在 user workspace 内的角色；具体偏移由 host tiling 给出
enum ChunkDeltaHBwdPreWs : uint32_t {
    WS_META = 0,   // segment / chunk 元数据
    WS_SLOT = 1,   // V0 的 chunk slot（Q̄s / K̄ / decayK），slotNum 份
    WS_T1 = 2,     // T1 平面 [K, K] FP32
    WS_PC = 3,     // P_c 平面 [K, K] FP32
    WS_DH = 4,     // dH ping-pong，2 * [K, V] FP32
    WS_P = 5,      // P ping-pong，2 * [K, K] FP32
    WS_NUM = 6,
};

// 反向链：chunk 按 i_t = NT-1 → 0 执行；dH 与 P 都是跨 chunk 状态
struct ChunkDeltaHBwdPreChainState {
    uint32_t chunkIdx;      // 当前 chunk 序号（从大到小）
    uint32_t slotIdx;       // chunkIdx % slotNum
    uint32_t dhtPingPong;   // 0 / 1
    uint32_t pPingPong;     // 0 / 1
    bool isLastChunk;       // i_t == 0：末 chunk 需要把 E_r / P_r 写 dhm
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_POLICY_H
