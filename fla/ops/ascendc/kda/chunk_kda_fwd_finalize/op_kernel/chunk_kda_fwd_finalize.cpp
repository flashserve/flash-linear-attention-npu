/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "kernel_operator.h"
#include "chunk_kda_fwd_finalize_kernel.h"

__global__ __aicore__ void chunk_kda_fwd_finalize(
    GM_ADDR qg_scaled, GM_ADDR aqk, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // AIC 计算 Q@H 与 Aqk@V_new，AIV 完成两路结果相加、BF16 转换和 attn_out 回写，必须保留 1:2 混合核。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(KdaFinalize::ChunkKdaFwdFinalizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(KdaFinalize::ChunkKdaFwdFinalizeTilingData,
                                tilingData, tiling);

    KdaFinalize::FinalizeArgs args{};
    args.qgScaled = qg_scaled;
    args.aqk = aqk;
    args.vNew = v_new;
    args.h = h;
    args.cuSeqlens = cu_seqlens;
    args.chunkIndices = chunk_indices;
    args.attnOut = attn_out;
    args.workspace = AscendC::GetUserWorkspace(workspace);
    args.tiling.batch = static_cast<uint32_t>(tilingData.batch);
    args.tiling.seqNum = static_cast<uint32_t>(tilingData.seqNum);
    args.tiling.seqLen = static_cast<uint32_t>(tilingData.seqLen);
    args.tiling.valueHeadNum = static_cast<uint32_t>(tilingData.valueHeadNum);
    args.tiling.totalChunks = static_cast<uint32_t>(tilingData.totalChunks);
    args.tiling.usedCoreNum = static_cast<uint32_t>(tilingData.usedCoreNum);
    args.tiling.headsPerPartition = static_cast<uint32_t>(tilingData.headsPerPartition);
    args.tiling.isVarLen = tilingData.isVarLen;
    args.tiling.outputSequenceMajor = tilingData.outputSequenceMajor;
    args.tiling.stateVFirst = tilingData.stateVFirst;

    if (args.tiling.stateVFirst) {
        if (args.tiling.outputSequenceMajor) {
            KdaFinalize::RunFinalize<true, true>(args);
        } else {
            KdaFinalize::RunFinalize<true, false>(args);
        }
    } else if (args.tiling.outputSequenceMajor) {
        KdaFinalize::RunFinalize<false, true>(args);
    } else {
        KdaFinalize::RunFinalize<false, false>(args);
    }
}
