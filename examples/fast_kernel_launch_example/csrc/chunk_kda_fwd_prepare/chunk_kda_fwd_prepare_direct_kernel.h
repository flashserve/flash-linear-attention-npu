/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of the BSD 3-Clause License (the "License").
 */

#ifndef EXAMPLES_CHUNK_KDA_FWD_PREPARE_DIRECT_KERNEL_H
#define EXAMPLES_CHUNK_KDA_FWD_PREPARE_DIRECT_KERNEL_H

#include "acl/acl.h"
#include "kernel_operator.h"

// 直接复用正式实现中的 RunPrepare 和编译期策略，不引入正式算子的全局入口。
#include "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare_kernel.h"

namespace KdaPrepareDirect {

template <int GateType, int BetaType, uint32_t NormMode,
          uint32_t BetaMode, uint32_t GateMode, bool UseExp2, bool SafeGate,
          uint32_t OutputMode>
__global__ __aicore__ void DirectKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
    GM_ADDR kg, GM_ADDR qgScaled, GM_ADDR qHat, GM_ADDR kHat,
    GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR betaEff, GM_ADDR workspace,
    KdaPrepare::ChunkKdaFwdPrepareTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);

    KdaPrepare::PrepareKernelArgs args{};
    args.q = q;
    args.k = k;
    args.v = v;
    args.rawGate = g;
    args.beta = beta;
    args.aLog = aLog;
    args.dtBias = dtBias;
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
    args.qHat = qHat;
    args.kHat = kHat;
    args.qRstd = qRstd;
    args.kRstd = kRstd;
    args.betaEff = betaEff;
    args.workspace = AscendC::GetUserWorkspace(workspace);
    if (args.workspace == nullptr) {
        return;
    }

    args.tiling.batch = tiling.batch;
    args.tiling.seqNum = tiling.seqNum;
    args.tiling.seqLen = tiling.seqLen;
    args.tiling.qkHeadNum = tiling.qkHeadNum;
    args.tiling.valueHeadNum = tiling.valueHeadNum;
    args.tiling.totalChunks = tiling.totalChunks;
    args.tiling.usedCoreNum = tiling.usedCoreNum;
    args.tiling.headsPerPartition = tiling.headsPerPartition;
    args.tiling.epsilon = tiling.epsilon;
    args.tiling.lowerBound = tiling.lowerBound;
    args.tiling.scale = tiling.scale;
    args.tiling.isVarLen = tiling.isVarLen;
    args.tiling.inputSequenceMajor = tiling.inputSequenceMajor;
    args.tiling.hasDtBias = tiling.hasDtBias;

    using Policy = KdaPrepare::PrepareCompilePolicy<
        static_cast<KdaPrepare::QkNormMode>(NormMode),
        static_cast<KdaPrepare::BetaMode>(BetaMode),
        static_cast<KdaPrepare::GateMode>(GateMode), UseExp2, SafeGate,
        static_cast<KdaPrepare::OutputMode>(OutputMode)>;
    using GateT = typename KdaPrepare::PrepareStorageType<GateType>::type;
    using BetaT = typename KdaPrepare::PrepareStorageType<BetaType>::type;
    KdaPrepare::RunPrepare<GateT, BetaT, Policy>(args);
}

template <int GateType, int BetaType, uint32_t NormMode,
          uint32_t BetaMode, uint32_t GateMode, bool UseExp2, bool SafeGate,
          uint32_t OutputMode>
void Launch(
    uint32_t blockDim, aclrtStream stream,
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
    GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg,
    GM_ADDR kg, GM_ADDR qgScaled, GM_ADDR qHat, GM_ADDR kHat,
    GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR betaEff, GM_ADDR workspace,
    const KdaPrepare::ChunkKdaFwdPrepareTilingData &tiling)
{
    DirectKernel<GateType, BetaType, NormMode, BetaMode, GateMode, UseExp2,
                 SafeGate, OutputMode><<<blockDim, nullptr, stream>>>(
        q, k, v, g, beta, aLog, dtBias, cuSeqlens, chunkIndices,
        gk, aqk, akk, w, u, qg, kg, qgScaled, qHat, kHat,
        qRstd, kRstd, betaEff, workspace, tiling);
}

} // namespace KdaPrepareDirect

#endif // EXAMPLES_CHUNK_KDA_FWD_PREPARE_DIRECT_KERNEL_H
