/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_fwd_h_tiling_key.h"
#include "chunk_fwd_h_policy.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_fwd_h_cube.h"
#include "arch35/chunk_fwd_h_vec.h"
#else
// gated 复用（仅 A2/arch22）：复用 chunk_gated_delta_rule_fwd_h 的 preload kernel。
// 基座/调度器/epilogue 特化在 arch22/chunk_fwd_h_utils.h，gated 双引擎类分别在
// arch22/chunk_fwd_h_cube.h 与 arch22/chunk_fwd_h_vec.h；复用谓词/入口在本文件。
#include "arch22/chunk_fwd_h_utils.h"
#include "arch22/chunk_fwd_h_cube.h"
#include "arch22/chunk_fwd_h_vec.h"
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace GDN {

template <int D_T_G>
struct FwdHGateDTypeTraits;

template <>
struct FwdHGateDTypeTraits<CHUNK_FWD_H_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct FwdHGateDTypeTraits<CHUNK_FWD_H_TPL_FP32> {
    using type = float;
};

template <int D_T_G, int V_DIM, bool USE_GK, bool USE_EXP2, bool STATE_FP32,
          bool STATE_V_FIRST, bool STORE_FINAL, bool DENSE_T16>
struct FwdHKernelTraits {
    static_assert(V_DIM == static_cast<int>(FWD_H_V), "ChunkFwdH only supports V=128.");
    using GateT = typename FwdHGateDTypeTraits<D_T_G>::type;
    using CompilePolicy = FwdHCompilePolicy<
        USE_GK ? FwdHGateMode::KEY_GK : FwdHGateMode::SCALAR_G, USE_EXP2, STATE_FP32,
        STORE_FINAL, DENSE_T16>;
    static constexpr bool STATE_V_FIRST_VALUE = STATE_V_FIRST;
};

// gated 复用谓词与入口（RunFwdHTyped 分流调用）——自 arch22/chunk_fwd_h_utils.h 文末归位回本文件（优化前入口位置）。

// gated 复用谓词：g 模式 + vfirst=0（gated 恒按 [K,V] 产出 state，
// vfirst=1 须经 aclnn 层转置翻译）；storeFinal 与 gate/state 的 fp32/bf16 均支持。
template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline constexpr bool FwdHGatedReusable()
{
    // 域不限（2026-11-10 exp 域复用释放）：log2（USE_EXP2=true）/ln（false）均
    // 可——域经模板维进 gated kernel（epilogue 内 LN2 换算 if constexpr 取舍）；
    // 与 host 侧 gatedEligible 两处同条件。
    return CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G &&
           !STATE_V_FIRST &&
           (std::is_same_v<GateT, float> || std::is_same_v<GateT, bfloat16_t>);
}

// gated 复用分支：构造 gated tiling 镜像（workspace 的 gatedTilingOffset 段，
// 各 AIC 幂等同值写）后调用 gated preload kernel；USE_EXP2 携带 g 输入域
// （true=log2，false=ln），gated epilogue 的 LN2 换算按此模板维取舍。
template <typename GateT, typename StateT, bool USE_EXP2>
__aicore__ inline void RunFwdHGatedReuse(const FwdHKernelArgs &args)
{
    using DataT = bfloat16_t;
    __gm__ uint8_t *user = args.workspace;
    __gm__ ChunkGatedDeltaRuleFwdHTilingData *gtiling = reinterpret_cast<
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *>(user + args.tiling.gatedTilingOffset);
    {
        gtiling->batch = args.tiling.batch;
        gtiling->seqlen = args.tiling.seqlen;
        gtiling->kNumHead = args.tiling.kNumHead;
        gtiling->vNumHead = args.tiling.vNumHead;
        gtiling->kHeadDim = args.tiling.kHeadDim;
        gtiling->vHeadDim = args.tiling.vHeadDim;
        gtiling->chunkSize = args.tiling.chunkSize;
        gtiling->useInitialState = args.tiling.useInitialState;
        gtiling->storeFinalState = args.tiling.storeFinalState;
        gtiling->dataType = 1; // bf16 输入
        gtiling->gDataType = std::is_same_v<GateT, float> ? 2 : 1; // 1=bf16, 2=fp32
        gtiling->stateDataType = std::is_same_v<StateT, float> ? 2 : 1;
        gtiling->isVariedLen = args.tiling.isVariedLen;
        gtiling->shapeBatch = args.tiling.shapeBatch;
        gtiling->tokenBatch = args.tiling.tokenBatch;
        gtiling->useG = true;
        gtiling->useGk = false;
        gtiling->vWorkspaceOffset = args.tiling.vWorkspaceOffset;
        gtiling->vUpdateWorkspaceOffset = args.tiling.vUpdateWorkspaceOffset;
        gtiling->kDecayWorkspaceOffset = 0;
        gtiling->hWorkspaceOffset = args.tiling.hWorkspaceOffset;
        gtiling->numSeqWorkspaceOffset = args.tiling.gatedNumSeqOffset;
        gtiling->numChunksWorkspaceOffset = args.tiling.gatedNumChunksOffset;
    }
    if ASCEND_IS_AIC {
        Catlass::Gemm::Kernel::ChunkFwdHGatedCubeArch22<DataT, GateT, StateT, float, USE_EXP2> gatedCube;
        gatedCube.Init(args.k, args.w, args.u, args.g, args.initialState,
                       args.cuSeqlens, args.chunkIndices, args.h, args.vNew,
                       args.finalState,
                       reinterpret_cast<GM_ADDR>(gtiling),
                       user);
        gatedCube.Process();
    }
    if ASCEND_IS_AIV {
        Catlass::Gemm::Kernel::ChunkFwdHGatedVecArch22<DataT, GateT, StateT, float, USE_EXP2> gatedVec;
        gatedVec.Init(args.k, args.w, args.u, args.g, args.initialState,
                      args.cuSeqlens, args.chunkIndices, args.h, args.vNew,
                      args.finalState,
                      reinterpret_cast<GM_ADDR>(gtiling),
                      user);
        gatedVec.Process();
    }
}

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline void RunFwdHTyped(const FwdHKernelArgs &args)
{
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
    // gated 复用分流：eligible 走 gated preload kernel，其余（gk/vfirst=1）回落
    // 原路径；log2/ln 两域均可进 gated（域随 CompilePolicy::USE_EXP2 作模板维
    // 传入，epilogue 内 LN2 换算按域取舍）。
    if constexpr (FwdHGatedReusable<GateT, CompilePolicy, STATE_V_FIRST>()) {
        using GatedStateT = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
        // varlen 且 head 组数为奇（kNumHead>1）时 gated kernel 会死锁，回落老路径。
        const int64_t group = args.tiling.kNumHead > 0
            ? static_cast<int64_t>(args.tiling.vNumHead) / static_cast<int64_t>(args.tiling.kNumHead)
            : 0;
        const bool gatedDeadlockShape = args.tiling.isVariedLen != 0 &&
            (group % 2) != 0 && args.tiling.kNumHead > 1;
        if (!gatedDeadlockShape) {
            RunFwdHGatedReuse<GateT, GatedStateT, CompilePolicy::USE_EXP2>(args);
            return;
        }
    }
#endif
    if ASCEND_IS_AIC {
        const bool hasCubeWork = args.tiling.useInitialState || CompilePolicy::STORE_FINAL ||
            args.tiling.seqlen > static_cast<uint32_t>(FWD_H_CHUNK);
        if (!hasCubeWork) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ChunkFwdHCubeArch35<CompilePolicy, STATE_V_FIRST> cube;
#else
        AscendC::TPipe pipe;
        ChunkFwdHCubeArch22<CompilePolicy, STATE_V_FIRST> cube;
#endif
        cube.Init(args);
        cube.Process();
    }
    if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ChunkFwdHVecArch35<GateT, CompilePolicy, STATE_V_FIRST> vec;
        vec.Init(args);
#else
        AscendC::TPipe pipe;
        ChunkFwdHVecArch22<GateT, CompilePolicy, STATE_V_FIRST> vec;
        vec.Init(args);
#endif
        vec.Process();
    }
}

} // namespace GDN

#ifndef TORCH_MODE
template <int D_T_G, int V_DIM, bool USE_GK, bool USE_EXP2, bool STATE_FP32,
          bool STATE_V_FIRST, bool STORE_FINAL, bool DENSE_T16>
__global__ __aicore__ void chunk_fwd_h(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
    GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GDN::FwdHKernelArgs args{};
    args.k = k;
    args.w = w;
    args.u = u;
    args.g = g;
    args.gk = gk;
    args.initialState = initial_state;
    args.cuSeqlens = cu_seqlens;
    args.chunkIndices = chunk_indices;
    args.h = h;
    args.vNew = v_new;
    args.finalState = final_state;
    args.workspace = AscendC::GetUserWorkspace(workspace);
    __gm__ const ChunkFwdHTilingData *tilingData =
        reinterpret_cast<__gm__ const ChunkFwdHTilingData *>(tiling);
    args.tiling.batch = static_cast<uint32_t>(tilingData->batch);
    args.tiling.seqlen = static_cast<uint32_t>(tilingData->seqlen);
    args.tiling.kNumHead = static_cast<uint32_t>(tilingData->kNumHead);
    args.tiling.vNumHead = static_cast<uint32_t>(tilingData->vNumHead);
    args.tiling.kHeadDim = static_cast<uint32_t>(tilingData->kHeadDim);
    args.tiling.vHeadDim = static_cast<uint32_t>(tilingData->vHeadDim);
    args.tiling.chunkSize = static_cast<uint32_t>(tilingData->chunkSize);
    args.tiling.useInitialState = tilingData->useInitialState;
    args.tiling.storeFinalState = tilingData->storeFinalState;
    args.tiling.isVariedLen = tilingData->isVariedLen;
    args.tiling.shapeBatch = static_cast<uint32_t>(tilingData->shapeBatch);
    args.tiling.tokenBatch = static_cast<uint32_t>(tilingData->tokenBatch);
    args.tiling.vWorkspaceOffset = static_cast<uint64_t>(tilingData->vWorkspaceOffset);
    args.tiling.vUpdateWorkspaceOffset = static_cast<uint64_t>(tilingData->vUpdateWorkspaceOffset);
    args.tiling.kDecayWorkspaceOffset = static_cast<uint64_t>(tilingData->kDecayWorkspaceOffset);
    args.tiling.hWorkspaceOffset = static_cast<uint64_t>(tilingData->hWorkspaceOffset);
    args.tiling.coreBoundCount = static_cast<uint32_t>(tilingData->coreBoundCount);
    if (args.tiling.coreBoundCount > GDN::FWD_H_MAX_CORE_BOUNDS) {
        args.tiling.coreBoundCount = 0; // malformed table — uniform partition fallback
    }
    args.tiling.coreBoundsGm = tilingData->coreTaskBounds;
    args.tiling.gatedTilingOffset = static_cast<uint64_t>(tilingData->gatedTilingOffset);
    args.tiling.gatedNumSeqOffset = static_cast<uint64_t>(tilingData->gatedNumSeqOffset);
    args.tiling.gatedNumChunksOffset = static_cast<uint64_t>(tilingData->gatedNumChunksOffset);
    using Traits = GDN::FwdHKernelTraits<
        D_T_G, V_DIM, USE_GK, USE_EXP2, STATE_FP32, STATE_V_FIRST, STORE_FINAL, DENSE_T16>;
    GDN::RunFwdHTyped<typename Traits::GateT, typename Traits::CompilePolicy,
                      Traits::STATE_V_FIRST_VALUE>(args);
}
#endif
