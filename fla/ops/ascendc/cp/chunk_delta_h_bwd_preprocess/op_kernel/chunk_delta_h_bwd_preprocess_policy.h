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
enum ChunkDeltaHBwdPreprocessStage : uint32_t {
    STAGE_V0_GATE_PREP = 0,   // Vector：门控/衰减/操作数准备（Q̄s、K̄、decayK）
    STAGE_C1_FIRST_MMAD = 1,  // Cube：路 A dV_pre、路 B T1
    STAGE_V2_DV_HAT = 2,      // Vector：dV̂' = -(dV_pre + dv_local)
    STAGE_C3_INC = 3,         // Cube：inc = Q̄sᵀ @ do + Wᵀ @ dV̂'
    STAGE_V4_STATE = 4,       // Vector：路 A dH 更新、路 B P_c 对角注入
    STAGE_C5_CHAIN = 5,       // Cube：P_new = P_c @ P_old
    STAGE_NUM = 6,
};

// 同一个 Stage 内的两路输出：ready/free 必须各自独立，禁止合并成一条边
enum ChunkDeltaHBwdPreprocessPath : uint32_t {
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

// v2.1：双 window 流水。每个 chunk 取 w = chunkIdx & 1，六个 ready 各自按 window 分 id，
// 于是 AIV 可以把 V0 提前两个 chunk（slot(w) 在 C1 消费完即释放），AIC 不再在 chunk 之间空等。
// 参考 chunk_gated_delta_rule_bwd_finalize 的「每方向一条 ready 链 + 双 window×4 head」做法。
constexpr uint32_t CDHP_WINDOW_COUNT = 2;

// 5 条 ready 协议（每条按 window 分 2 个 id）：
//   V0_READY    AIV→AIC：本 window 的 slot（Q̄s/W/-K̄/do/-dv/decay）就绪
//   T1_READY    AIC→AIV：T1(FP32) 就绪，请转成模型 dtype        （仅 arch22 使用）
//   T1BF_READY  AIV→AIC：T1 的模型 dtype 平面就绪（Z/ZP 的操作数）（仅 arch22 使用）
//   Z_READY     AIC→AIV：Z/ZP 就绪（链上的 MMAD 结果）
//   STATE_READY AIV→AIC：dH_bf / P_bf 就绪（下一 chunk 的 MMAD 操作数）
// arch35（A5）上 T1 的模型 dtype 平面改由**本核 fixpipe 直接写**（见 arch35 的 cube），
// 因此不再需要 T1_READY / T1BF_READY 这两条跨核 flag，AIV 的 StageT1Convert 整段消失：
//   * 少一次跨核往返，少 96 KiB/chunk 的 UB↔GM 搬运；
//   * "写落盘"先于"MTE2 读回"变成 AIC 核内的 FIX→MTE2 依赖（CDHP_AIC_EV_FIX_MTE2）。
// arch22（A2/A3）上核内 FIX→MTE2 自排空会导致 kernel 挂死（实测 AICore 100% 不复位），
// 因此该平台保持原来的三步握手。
constexpr uint32_t CDHP_FLAG_V0_READY = 0;
constexpr uint32_t CDHP_FLAG_T1_READY = 1;
constexpr uint32_t CDHP_FLAG_T1BF_READY = 2;
constexpr uint32_t CDHP_FLAG_Z_READY = 3;
constexpr uint32_t CDHP_FLAG_STATE_READY = 4;

// arch35（A5，mode 0x4"按 subblock 选配对 AIV"）：AIC 侧用 16 步长区分两个 AIV，AIV 侧用本地 id，
// 并且 id 按 window 分（base*2+win，最大 9）。
// arch22（A2/A3）当前仍是 1 AIC : 1 AIV，id 也按 window 分（base*2+win，最大 9）。
// A2 的 1:2 需要"每个 AIV 一段独立 id + 不按 window 分"，三次尝试都挂死（见 design.md §17/§21），
// 该方案与 op_host 的 aivPerBlock、kernel 的 KERNEL_TASK_TYPE 必须三处一起切，切之前不要单独改这里。
__aicore__ inline constexpr uint32_t CdhpFlag(uint32_t base, uint32_t window)
{
    return base * CDHP_WINDOW_COUNT + window;
}

// A5（dav-3510）的 1 AIC : 2 AIV 核型下，AIC 用 16 的 id 步长选择配对的 AIV（mode = 0x4 按 subblock
// 选配对 AIV）；AIV 侧只用自己的本地 id（就是上面的 CdhpFlag 结果，最大 9，落在 A5 允许的 0..10 内）。
// A2/A3 的 mode = 0x2 是 AIC:2*AIV 集合同步；1:1 下 aiv 只可能是 0，公式退化成原来的形式。
constexpr uint32_t CDHP_FLAG_SUBBLOCK_STRIDE = 16;
// mode 必须和核型配对：
//   arch35（A5）→ 1 AIC : 2 AIV，用 0x4（按 subblock 选配对 AIV，配 CdhpPeerFlag 的 16 步长）；
//   arch22（A2/A3）→ 1 AIC : 1 AIV，用 0x2（AIC 与本 block 的 AIV 集合同步）。
// kernel 是按 SoC 分别编译的，op_host 的 aivPerBlock 也按同一个 SoC 判定，两侧天然一致。
// 用错 mode 会在 launch 后直接报 synchronize failed（实测 507015），是运行期错误而不是精度问题。
// 注意历史：A2 三次接 1:2 都在 smoke 用例上挂死（详见 design.md §17、§21）：①两侧共用同一 id
// （集合同步语义下一 set 放行两个 AIV，计数纪律被破坏）；②per-AIV 段 + 不按 window 分，gva 通过、
// kda（Hv=64 两轮）挂住；③在 P 常驻 UB 之后重试，反而在第一个 gva 档就挂住。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CDHP_CROSS_CORE_MODE 0x4
#else
#define CDHP_CROSS_CORE_MODE 0x2
#endif

__aicore__ inline constexpr uint32_t CdhpPeerFlag(uint32_t aiv, uint32_t localId)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return localId + aiv * CDHP_FLAG_SUBBLOCK_STRIDE;
#else
    (void)aiv;
    return localId;
#endif
}

// set 之前先让本 pipe 排空（PipeBarrier 是"等待本 pipe 先前指令完成"的屏障）。
// 实测：不做排空时，cube 侧 C5 的 Fixpipe 落盘还没走完就把 flag 置起来，消费者在
// 非最后一个 task 上会读到"最后 16 行全 0"的半成品平面（E 不受影响、P 整链被污染）。
#define CDHP_AIV_SET(flag)                                  \
    do {                                                    \
        AscendC::PipeBarrier<PIPE_MTE3>();                   \
        AscendC::CrossCoreSetFlag<CDHP_CROSS_CORE_MODE, PIPE_MTE3>(flag); \
    } while (0)
// 注意：`CrossCoreWaitFlag(flag)` 的默认模板参数是 `pipe = PIPE_S`，在 A5（dav_3510）上
// WaitEventImpl 会按 pipe 只阻塞对应 pipe。若用默认值，等待只落在标量 pipe 上，
// 随后的 MTE2（GM→UB/L1）载入不会被拦在 flag 之后，会读到 cube 还没写完的中间平面
// （表现为 P/T1 相关的整块旧值，且随负载时好时坏）。这里显式绑定消费者 pipe = MTE2。
#define CDHP_AIV_WAIT(flag) AscendC::CrossCoreWaitFlag<CDHP_CROSS_CORE_MODE, PIPE_MTE2>(flag)
// AIC 侧：aiv 是配对 AIV 在该 block 内的序号（0/1）。A5 下同一 block 的两个 AIV 各有自己的一条
// ready 链，AIC 必须按 aiv 分别 wait/set；A2/A3 下 aiv 被忽略（集合同步覆盖两个 AIV）。
// 必须先排空 FIX pipe 再置 flag：本算子的 fixpipe 是落 GM（Z/ZP/T1 平面），置 flag 时若写还没
// 落盘，消费者会读到上一次留在该平面里的半成品（实测症状：E 面（先写的 Z）正常、P 面（后写的 ZP）
// 整链被污染，且哪个 head 出错随负载变化）。fwd_h / chunk_gdn_bwd_intra 的 fixpipe 是写 UB，
// UB 写在核内立即可见，那边才不需要这个排空，不能照抄。
#define CDHP_AIC_SET(aiv, flag)                                                          \
    do {                                                                                 \
        AscendC::PipeBarrier<PIPE_FIX>();                                                \
        AscendC::CrossCoreSetFlag<CDHP_CROSS_CORE_MODE, PIPE_FIX>(CdhpPeerFlag(aiv, flag)); \
    } while (0)
#define CDHP_AIC_WAIT(aiv, flag) \
    AscendC::CrossCoreWaitFlag<CDHP_CROSS_CORE_MODE, PIPE_MTE2>(CdhpPeerFlag(aiv, flag))

// 工作区平面在 user workspace 内的角色；具体偏移由 host tiling 给出
enum ChunkDeltaHBwdPreprocessWs : uint32_t {
    WS_META = 0,   // segment / chunk 元数据
    WS_SLOT = 1,   // V0 的 chunk slot（Q̄s / K̄ / decayK），slotNum 份
    WS_T1 = 2,     // T1 平面 [K, K] FP32
    WS_PC = 3,     // P_c 平面 [K, K] FP32
    WS_DH = 4,     // dH ping-pong，2 * [K, V] FP32
    WS_P = 5,      // P ping-pong，2 * [K, K] FP32
    WS_NUM = 6,
};

// 反向链：chunk 按 i_t = NT-1 → 0 执行；dH 与 P 都是跨 chunk 状态
struct ChunkDeltaHBwdPreprocessChainState {
    uint32_t chunkIdx;      // 当前 chunk 序号（从大到小）
    uint32_t slotIdx;       // chunkIdx % slotNum
    uint32_t dhtPingPong;   // 0 / 1
    uint32_t pPingPong;     // 0 / 1
    bool isLastChunk;       // i_t == 0：末 chunk 需要把 E_r / P_r 写 dhm
};

} // namespace CP

#endif // CHUNK_DELTA_H_BWD_PREPROCESS_POLICY_H
