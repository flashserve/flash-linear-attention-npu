/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of the BSD 3-Clause License (the "License").
 */

// 该文件只锁定 Ascend C 直调源码合同，尚未接入独立的编译和设备执行目标。
// 两个模板实例与 tests/op_cases/chunk_kda_fwd_prepare.json 中声明的代表 case 对应。
#include "acl/acl.h"
#include "kernel_operator.h"

#ifndef TORCH_MODE
#define TORCH_MODE
#define CHUNK_KDA_FWD_PREPARE_TEST_UNDEF_TORCH_MODE
#endif
#include "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/chunk_kda_fwd_prepare.cpp"
#ifdef CHUNK_KDA_FWD_PREPARE_TEST_UNDEF_TORCH_MODE
#undef TORCH_MODE
#undef CHUNK_KDA_FWD_PREPARE_TEST_UNDEF_TORCH_MODE
#endif

namespace ChunkKdaFwdPrepareDirectTest {

template <int GateType, int BetaType, uint32_t NormMode,
          uint32_t BetaMode, uint32_t GateMode, bool UseExp2, bool SafeGate>
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
        static_cast<KdaPrepare::GateMode>(GateMode), UseExp2, SafeGate>;
    using GateT = typename KdaPrepare::PrepareStorageType<GateType>::type;
    using BetaT = typename KdaPrepare::PrepareStorageType<BetaType>::type;
    KdaPrepare::RunPrepare<GateT, BetaT, Policy>(args);
}

template <int GateType, int BetaType, uint32_t NormMode,
          uint32_t BetaMode, uint32_t GateMode, bool UseExp2, bool SafeGate>
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
                 SafeGate><<<blockDim, nullptr, stream>>>(
        q, k, v, g, beta, aLog, dtBias, cuSeqlens, chunkIndices,
        gk, aqk, akk, w, u, qg, kg, qgScaled, qHat, kHat,
        qRstd, kRstd, betaEff, workspace, tiling);
}

// prepare_dense_bnsd_raw 对应的默认 FP32 gate/beta 路径。
template void Launch<
    CHUNK_KDA_FWD_PREPARE_TPL_FP32,
    CHUNK_KDA_FWD_PREPARE_TPL_FP32,
    CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
    CHUNK_KDA_FWD_PREPARE_BETA_RAW,
    CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
    false,
    false>(
    uint32_t, aclrtStream,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const KdaPrepare::ChunkKdaFwdPrepareTilingData &);

// prepare_dense_bsnd_fused 对应的 BF16、L2、SafeSigmoid、2*sigmoid、exp2 路径。
template void Launch<
    CHUNK_KDA_FWD_PREPARE_TPL_BF16,
    CHUNK_KDA_FWD_PREPARE_TPL_BF16,
    CHUNK_KDA_FWD_PREPARE_NORM_L2,
    CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID,
    CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID,
    true,
    true>(
    uint32_t, aclrtStream,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const KdaPrepare::ChunkKdaFwdPrepareTilingData &);

} // namespace ChunkKdaFwdPrepareDirectTest
