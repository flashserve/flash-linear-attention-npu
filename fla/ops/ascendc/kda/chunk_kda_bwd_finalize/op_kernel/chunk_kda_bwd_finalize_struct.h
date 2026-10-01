/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_STRUCT_H
#define CHUNK_KDA_BWD_FINALIZE_STRUCT_H

#include <cstdint>

namespace KDA {

// 完整 Finalize 的 host/device 共享 tiling；字段顺序与 op_host 保持一致。
// 当前实现包含 Stage0–12，定长/变长与可选 Q/K 归一化均使用本结构。
struct ChunkKdaBwdFinalizeTilingData {
    // 1. 输入维度与任务数量：workTask 是 head 窗口与 chunk 的组合。
    int64_t B;
    int64_t NQ;
    int64_t NV;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t denseChunkNum;
    int64_t totalChunkNum;
    int64_t chunkTaskNum;
    int64_t headWindowNum;
    int64_t workTaskNum;
    int64_t seqNum;
    int64_t chunkSize;

    // 2. host 已校验的模式标志；kernel 按原值选择布局与可选计算。
    uint32_t isVariable;
    uint32_t hasQkL2Norm;
    uint32_t safeGate;
    uint32_t useGateInKernel;
    uint32_t useExp2;
    uint32_t stateVFirst;

    // 3. 数学参数：Q 缩放与 safe-gate 下界。
    float scale;
    float lowerBound;

    // 4. 用户 workspace 内的字节偏移：循环槽、a_log 部分和、dt_bias 部分和。
    uint64_t slotWorkspaceOffset;
    uint64_t gatePartialOffset;
    uint64_t dtBiasPartialOffset;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_STRUCT_H
