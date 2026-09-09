/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_prepare_policy.h"
#include "chunk_kda_fwd_prepare_struct.h"
#include "chunk_kda_fwd_prepare_tiling_key.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"
#else
#include "arch22/chunk_kda_fwd_prepare_cube.h"
#include "arch22/chunk_kda_fwd_prepare_vec.h"
#endif

namespace KdaPrepare {

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void RunPrepare(const PrepareKernelArgs &args)
{
    if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch35::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy>
            vec;
        vec.Init(args);
#else
        AscendC::TPipe pipe;
        Arch22::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy>
            vec;
        vec.Init(args, &pipe);
#endif
        vec.Process();
    }

    if ASCEND_IS_AIC {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch35::ChunkKdaFwdPrepareCube cube;
        cube.Init(args);
#else
        AscendC::TPipe pipe;
        Arch22::ChunkKdaFwdPrepareCube cube;
        cube.Init(args, &pipe);
#endif
        cube.Process();
    }
}

} // namespace KdaPrepare

// 本文件仍是公开接口布局尚未冻结的设计伪代码。入口形态直接使用真实
// Ascend C kernel API，便于后续把已验证的 Stage 逐个迁移到正式算子目录。
template <typename GateT, typename BetaT,
          KdaPrepare::QkNormMode NORM_MODE,
          KdaPrepare::BetaMode BETA_MODE,
          KdaPrepare::GateMode GATE_MODE,
          bool USE_EXP2, bool SAFE_GATE>
__global__ __aicore__ void chunk_kda_fwd_prepare_pseudocode(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR rawGate, GM_ADDR beta,
    GM_ADDR dtBias, GM_ADDR aLog, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
    GM_ADDR kg, GM_ADDR qgScaled, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    KdaPrepare::PrepareKernelArgs args{};
    args.q = q;
    args.k = k;
    args.v = v;
    args.rawGate = rawGate;
    args.beta = beta;
    args.dtBias = dtBias;
    args.aLog = aLog;
    args.cuSeqlens = cuSeqlens;
    args.chunkIndices = chunkIndices;
    args.gk = gk;
    args.aqk = aqk;
    args.akk = akk;
    args.w = w;
    args.u = u;
    args.qg = qg;
    args.kg = kg;
    args.qgScaled = qgScaled;
    args.workspace = AscendC::GetUserWorkspace(workspace);
    __gm__ const KdaPrepare::ChunkKdaFwdPrepareTilingData *tilingData =
        reinterpret_cast<__gm__ const KdaPrepare::ChunkKdaFwdPrepareTilingData *>(
            tiling);
    args.tiling.batch = tilingData->batch;
    args.tiling.seqNum = tilingData->seqNum;
    args.tiling.seqLen = tilingData->seqLen;
    args.tiling.qkHeadNum = tilingData->qkHeadNum;
    args.tiling.valueHeadNum = tilingData->valueHeadNum;
    args.tiling.totalChunks = tilingData->totalChunks;
    args.tiling.usedCoreNum = tilingData->usedCoreNum;
    args.tiling.headsPerPartition = tilingData->headsPerPartition;
    args.tiling.epsilon = tilingData->epsilon;
    args.tiling.lowerBound = tilingData->lowerBound;
    args.tiling.scale = tilingData->scale;
    args.tiling.isVarLen = tilingData->isVarLen;
    args.tiling.inputSequenceMajor = tilingData->inputSequenceMajor;
    args.tiling.hasDtBias = tilingData->hasDtBias;

    using Policy = KdaPrepare::PrepareCompilePolicy<
        NORM_MODE, BETA_MODE, GATE_MODE, USE_EXP2, SAFE_GATE>;
    KdaPrepare::RunPrepare<GateT, BetaT, Policy>(args);
}
