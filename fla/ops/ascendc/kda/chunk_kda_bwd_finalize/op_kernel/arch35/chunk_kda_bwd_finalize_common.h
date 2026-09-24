/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "catlass/arch/cross_core_sync.hpp"
#include "../chunk_kda_bwd_finalize_struct.h"
#include "kernel_operator.h"

namespace KDA {

using FinalizeLocalType = bfloat16_t;

constexpr uint32_t KDA_FINALIZE_CHUNK = 64;
constexpr uint32_t KDA_FINALIZE_DIM = 128;
// 任务组织：一个 AIC 对应两个 AIV；每窗口两头，每 AIV 一头。
constexpr uint32_t KDA_FINALIZE_HEADS_PER_WINDOW = 2;
constexpr uint32_t KDA_FINALIZE_AIV_COUNT = 2;
// 交接 slot 随任务轮转；不代表所有阶段都有两份独立 UB 工作区。
constexpr uint32_t KDA_FINALIZE_AIV_SLOTS = 2;
constexpr uint32_t KDA_FINALIZE_WORKSPACE_SLOTS = 2 * KDA_FINALIZE_HEADS_PER_WINDOW;
constexpr uint32_t KDA_FINALIZE_SLOT_BYTES = 160 * 1024;

constexpr uint32_t KDA_FINALIZE_INTRA_ROWS = 32;

constexpr uint32_t KDA_FINALIZE_MATRIX_ELEMS = KDA_FINALIZE_CHUNK * KDA_FINALIZE_CHUNK;
constexpr uint32_t KDA_FINALIZE_VECTOR_ELEMS = KDA_FINALIZE_CHUNK * KDA_FINALIZE_DIM;
constexpr uint32_t KDA_FINALIZE_STATE_ELEMS = KDA_FINALIZE_DIM * KDA_FINALIZE_DIM;
constexpr uint32_t KDA_FINALIZE_MATRIX_BF16_BYTES = KDA_FINALIZE_MATRIX_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t KDA_FINALIZE_MATRIX_FP32_BYTES = KDA_FINALIZE_MATRIX_ELEMS * sizeof(float);
constexpr uint32_t KDA_FINALIZE_VECTOR_BF16_BYTES = KDA_FINALIZE_VECTOR_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t KDA_FINALIZE_VECTOR_FP32_BYTES = KDA_FINALIZE_VECTOR_ELEMS * sizeof(float);
constexpr uint32_t KDA_FINALIZE_STATE_BF16_BYTES = KDA_FINALIZE_STATE_ELEMS * sizeof(bfloat16_t);

// GM 槽内偏移保持固定；EXP2_GK 和 KE 仅保留地址，当前数据常驻片上。
// 三块 FP32 矩阵区在 StateAndBase 中原位转为 dk_base、dq_base、dg_base。
constexpr uint32_t KDA_FINALIZE_WS_DK_STATE_RAW = 0;
constexpr uint32_t KDA_FINALIZE_WS_DVB = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DKGB_RAW = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_EXP2_GK = 96 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_KE = 128 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_GK_LAST = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_RH = 144 * 1024 + 512;
constexpr uint32_t KDA_FINALIZE_WS_GATE_STATE = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DB_V = 145 * 1024;
constexpr uint32_t KDA_FINALIZE_WS_DK_BASE = KDA_FINALIZE_WS_DK_STATE_RAW;
constexpr uint32_t KDA_FINALIZE_WS_DQ_BASE = KDA_FINALIZE_WS_DVB;
constexpr uint32_t KDA_FINALIZE_WS_DG_BASE = KDA_FINALIZE_WS_DKGB_RAW;
constexpr uint32_t KDA_FINALIZE_WS_DB_BASE = KDA_FINALIZE_WS_DB_V;

// 每个 AIV 的 BuildZ 固定 UB 区；zV/zW 按交接 slot 选择地址。
constexpr uint32_t KDA_FINALIZE_UB_ZV = 0;
constexpr uint32_t KDA_FINALIZE_UB_ZW = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_ZB = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_WORK = 81 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BYTES = 248 * 1024;

// AIV 完成 StateAndBase 和 Tza 残差交接后，才允许 Cube 写入 [0,16) KiB 的 dAkk_raw。
constexpr uint32_t KDA_FINALIZE_UB_DAKK_RAW = 0;
constexpr uint32_t KDA_FINALIZE_UB_STAGE4_WORK = 16 * 1024;

// Stage5：dAkk_raw 占 [0,16) KiB，dAqk/dAkk 高低位占 [16,48) KiB，
// 三个向量高低位占 [48,144) KiB；驻留输入从 144 KiB 起，与本阶段输出分离。
constexpr uint32_t KDA_FINALIZE_UB_K_NEG = 48 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_Q_POS = 80 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BK_POS = 112 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_STAGE5_WORK = 144 * 1024;

// Stage6 的 [dq_local; left] FP32 结果复用原 zV 交接区；其余 UB 供分带结果处理。
constexpr uint32_t KDA_FINALIZE_UB_DQ_LOCAL_RAW = 0;
constexpr uint32_t KDA_FINALIZE_UB_STAGE7_DQ_BASE = 64 * 1024;
// 每个窗口中每 AIV 处理一个 head，q/k/beta 跨分带保留。
// E 在 StateAndBase 中使用；Stage5 将同址内容换成 gk，再生成当前带的平移指数。
// Stage11 在最后一带结束后复用这些输入区，载入 rawG 与 gate 参数。
constexpr uint32_t KDA_FINALIZE_UB_Q = 144 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_K = 160 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_EXP2_GK = 176 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_BETA = 208 * 1024;
// beta 增量在分带 Stage9 中生成，紧接着由 Stage10 消费。
constexpr uint32_t KDA_FINALIZE_UB_DB_DELTA = 209 * 1024;
constexpr uint32_t KDA_FINALIZE_UB_STAGE7_Q_RSTD = 212 * 1024;
// 完整 chunk 的 gate 扫描缓冲区；Stage11 使用前，该区的分带临时数据已释放。
constexpr uint32_t KDA_FINALIZE_UB_DG = 216 * 1024;

// 两个 owner 各有 128 KiB L1，合计占 [64,320) KiB。
// 与 Stage4 仍使用的 Akk [0,16)、Tza [416,448) KiB 分离，
// 本 head 的 dAkk 就绪后即可发布 Stage5 操作数，不必等另一 head。
// 五项操作数直接从 UB 写 L1；owner 槽一直保留到 Stage8 读完。
constexpr uint32_t KDA_FINALIZE_LOCAL_BASE = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_BYTES = 128 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_DAQK = 0;
constexpr uint32_t KDA_FINALIZE_LOCAL_DAKK = 16 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_K_NEG = 32 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_Q_POS = 64 * 1024;
constexpr uint32_t KDA_FINALIZE_LOCAL_BK_POS = 96 * 1024;

// 1C2V 定向通知：AIC 用 aiv * 16 选择目标 AIV 的 flag bank。
constexpr uint8_t KDA_FINALIZE_CROSS_MODE = 0x4;
constexpr uint64_t KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE = 16;
constexpr uint64_t KDA_FINALIZE_ZV_FREE_BASE = 0;
constexpr uint64_t KDA_FINALIZE_ZW_FREE_BASE = 2;
constexpr uint64_t KDA_FINALIZE_ZV_READY_BASE = 4;
constexpr uint64_t KDA_FINALIZE_ZW_READY_BASE = 6;
constexpr uint64_t KDA_FINALIZE_KE_READY_BASE = 8;
constexpr uint64_t KDA_FINALIZE_ZB_READY_BASE = 10;
constexpr uint64_t KDA_FINALIZE_ZB_FREE_BASE = 12;
// 同一编号按阶段复用：KE_READY 先通知 dW/kE 高低位就绪，
// 再通知 Tza 高低位就绪并允许覆盖 dAkk 的 UB；ZW_READY 依次承载 zW、Tza、dAkk。
// 上一载荷已消费后才能发布下一通知，不能把同编号视为同一数据。
constexpr uint64_t KDA_FINALIZE_DAKK_FREE_BASE = KDA_FINALIZE_KE_READY_BASE;
constexpr uint64_t KDA_FINALIZE_DAKK_READY_BASE = KDA_FINALIZE_ZW_READY_BASE;
// AIV→AIC 的 Stage5 操作数发布使用独立编号对；
// 不引入所有核共同参与的 barrier，避免不同核任务数不等时阻塞。
constexpr uint64_t KDA_FINALIZE_LOCAL_READY_BASE = 14;
// 此编号仅用于 AIC→AIV；LOCAL_READY 使用反方向。
// AIC→AIV 的 flag 14 留给最终 AIV-only SyncAll。
constexpr uint64_t KDA_FINALIZE_TASK_L1_FREE = 15;
// Stage2 已消费 zV 通知后，编号对复用于 Stage6 的 dq_local 交接。
// 分带结果处理在 UB→L1 完成、目标区可写后发布 FREE，Cube 写回后发布 READY。
constexpr uint64_t KDA_FINALIZE_DQ_LOCAL_FREE_BASE = KDA_FINALIZE_ZV_FREE_BASE;
constexpr uint64_t KDA_FINALIZE_DQ_LOCAL_READY_BASE = KDA_FINALIZE_ZV_READY_BASE;

struct FinalizeChunkInfo {
    int64_t b = 0;
    int64_t seq = 0;
    int64_t localChunk = 0;
    int64_t stateIndex = 0;
    int64_t tokenStart = 0;
    int64_t validRows = 0;
    bool valid = false;
};

__aicore__ inline int64_t FinalizeMin(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

// 将 chunk 任务解析为 token/state 偏移和有效行数；无效元数据不产生计算任务。
__aicore__ inline void ResolveFinalizeChunk(
    int64_t task, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    const ChunkKdaBwdFinalizeTilingData &tiling, FinalizeChunkInfo &info)
{
    // 1. 默认无效，先检查任务编号。
    info.valid = false;
    if (task < 0 || task >= tiling.chunkTaskNum) {
        return;
    }

    // 2. 定长：task 拆成 batch 与 batch 内 chunk。
    if (tiling.isVariable == 0) {
        info.b = task / tiling.denseChunkNum;
        info.seq = info.b;
        info.localChunk = task - info.b * tiling.denseChunkNum;
        info.stateIndex = info.localChunk;
        info.tokenStart = info.localChunk * tiling.chunkSize;
        info.validRows = FinalizeMin(tiling.chunkSize, tiling.T - info.tokenStart);
        info.valid = info.b >= 0 && info.b < tiling.B && info.validRows > 0;
        return;
    }

    // 3. 变长：从 chunk_indices 读取序列/chunk，再用 cu_seqlens 限定有效 token。
    if (cuSeqlens == nullptr || chunkIndices == nullptr) {
        return;
    }
    AscendC::GlobalTensor<int64_t> cu;
    AscendC::GlobalTensor<int64_t> indices;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
    indices.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(chunkIndices));
    info.seq = indices.GetValue(2 * task);
    info.localChunk = indices.GetValue(2 * task + 1);
    if (info.seq < 0 || info.seq >= tiling.seqNum || info.localChunk < 0) {
        return;
    }
    const int64_t seqBegin = cu.GetValue(info.seq);
    const int64_t seqEnd = cu.GetValue(info.seq + 1);
    info.b = 0;
    info.stateIndex = task;
    info.tokenStart = seqBegin + info.localChunk * tiling.chunkSize;
    info.validRows = FinalizeMin(tiling.chunkSize, seqEnd - info.tokenStart);
    info.valid = seqBegin >= 0 && seqEnd >= seqBegin && seqEnd <= tiling.T && info.validRows > 0;
}

// Token 按 [B,H,T,D] 或 [H,T,D] 连续存放；返回元素偏移。
__aicore__ inline int64_t FinalizeTokenOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head, int64_t width)
{
    if (tiling.isVariable != 0) {
        return (head * tiling.T + chunk.tokenStart) * width;
    }
    return ((chunk.b * tiling.NV + head) * tiling.T + chunk.tokenStart) * width;
}

// 前向缓存 h 为 chunk-major；Dhu 输出 dh 为 head-major，二者必须分别寻址。
__aicore__ inline int64_t FinalizeHOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head)
{
    if (tiling.isVariable != 0) {
        return (chunk.stateIndex * tiling.NV + head) * tiling.K * tiling.V;
    }
    return ((chunk.b * tiling.denseChunkNum + chunk.stateIndex) * tiling.NV + head) *
        tiling.K * tiling.V;
}

// dh 按 head-major 寻址；不要与前向 h 的 chunk-major 偏移共用。
__aicore__ inline int64_t FinalizeDhOffset(
    const ChunkKdaBwdFinalizeTilingData &tiling, const FinalizeChunkInfo &chunk,
    int64_t head)
{
    if (tiling.isVariable != 0) {
        return (chunk.stateIndex * tiling.NV + head) * tiling.K * tiling.V;
    }
    return ((chunk.b * tiling.denseChunkNum + chunk.stateIndex) * tiling.NV + head) *
        tiling.K * tiling.V;
}

// 每核四个 160 KiB GM 槽：两代窗口 × 每窗口两头；返回字节偏移。
__aicore__ inline uint64_t FinalizeWorkspaceSlotBase(
    int64_t coreIdx, uint64_t groupGeneration, uint32_t owner)
{
    // 1. 按窗口代次取 ping/pong，再选本窗口的 owner；复用由阶段同步保证。
    const uint64_t window = groupGeneration & 1U;
    const uint64_t slot = window * KDA_FINALIZE_HEADS_PER_WINDOW + owner;
    return (static_cast<uint64_t>(coreIdx) * KDA_FINALIZE_WORKSPACE_SLOTS + slot) *
        KDA_FINALIZE_SLOT_BYTES;
}

static_assert(KDA_FINALIZE_WS_DB_V + 512 <= KDA_FINALIZE_SLOT_BYTES,
              "Stage0--3 workspace slot exceeds 160 KiB.");
static_assert(KDA_FINALIZE_UB_WORK < KDA_FINALIZE_UB_BYTES,
              "BuildZ fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_UB_STAGE4_WORK < KDA_FINALIZE_UB_BYTES,
              "Stage4 fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_UB_STAGE5_WORK < KDA_FINALIZE_UB_BYTES,
              "Stage5 fixed UB handoff exceeds A5 UB.");
static_assert(KDA_FINALIZE_HEADS_PER_WINDOW == KDA_FINALIZE_AIV_COUNT,
              "Retained inputs require one head per AIV per window.");
static_assert(KDA_FINALIZE_UB_Q + KDA_FINALIZE_VECTOR_BF16_BYTES <= KDA_FINALIZE_UB_K &&
                  KDA_FINALIZE_UB_K + KDA_FINALIZE_VECTOR_BF16_BYTES <= KDA_FINALIZE_UB_EXP2_GK &&
                  KDA_FINALIZE_UB_EXP2_GK + KDA_FINALIZE_VECTOR_FP32_BYTES <= KDA_FINALIZE_UB_BETA,
              "Retained head inputs overlap.");
static_assert(KDA_FINALIZE_UB_DB_DELTA + KDA_FINALIZE_AIV_SLOTS * 256 <=
                  KDA_FINALIZE_UB_STAGE7_Q_RSTD,
              "Beta deltas overlap Q normalization scratch.");
static_assert(KDA_FINALIZE_UB_DG + KDA_FINALIZE_VECTOR_FP32_BYTES <= KDA_FINALIZE_UB_BYTES,
              "Retained dg exceeds A5 UB.");
static_assert(KDA_FINALIZE_UB_STAGE7_Q_RSTD + 256 <= KDA_FINALIZE_UB_BYTES,
              "Stage7 working set exceeds A5 UB.");
static_assert(KDA_FINALIZE_HEADS_PER_WINDOW * KDA_FINALIZE_LOCAL_BYTES <= 256 * 1024,
              "Stage5 paired LocalOperand window exceeds 256 KiB.");
static_assert(KDA_FINALIZE_LOCAL_BASE +
                  KDA_FINALIZE_HEADS_PER_WINDOW * KDA_FINALIZE_LOCAL_BYTES <=
              512 * 1024,
              "Stage5 LocalOperand window exceeds A5 L1.");

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_COMMON_H
