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
#include "arch22/chunk_fwd_h_cube.h"
#include "arch22/chunk_fwd_h_vec.h"
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace GDN {

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline void RunFwdHTyped(const FwdHKernelArgs &args)
{
    if ASCEND_IS_AIC {
        const bool hasCubeWork = args.tiling.useInitialState || args.tiling.storeFinalState ||
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

template <typename GateT, FwdHGateMode GATE_MODE, bool STATE_FP32>
__aicore__ inline void DispatchExpAndLayout(const FwdHKernelArgs &args)
{
    if (args.tiling.useExp2) {
        using Policy = FwdHCompilePolicy<GATE_MODE, true, STATE_FP32>;
        if (args.tiling.stateVFirst) {
            RunFwdHTyped<GateT, Policy, true>(args);
        } else {
            RunFwdHTyped<GateT, Policy, false>(args);
        }
    } else {
        using Policy = FwdHCompilePolicy<GATE_MODE, false, STATE_FP32>;
        if (args.tiling.stateVFirst) {
            RunFwdHTyped<GateT, Policy, true>(args);
        } else {
            RunFwdHTyped<GateT, Policy, false>(args);
        }
    }
}

template <typename GateT, FwdHGateMode GATE_MODE>
__aicore__ inline void DispatchState(const FwdHKernelArgs &args)
{
    if (args.tiling.stateDataType == FWD_H_DTYPE_FP32) {
        DispatchExpAndLayout<GateT, GATE_MODE, true>(args);
    } else {
        DispatchExpAndLayout<GateT, GATE_MODE, false>(args);
    }
}

template <typename GateT>
__aicore__ inline void DispatchGateMode(const FwdHKernelArgs &args)
{
    if (args.tiling.useGk) {
        DispatchState<GateT, FwdHGateMode::KEY_GK>(args);
    } else {
        DispatchState<GateT, FwdHGateMode::SCALAR_G>(args);
    }
}

template <int V_DIM>
__aicore__ inline void DispatchFwdH(const FwdHKernelArgs &args)
{
    static_assert(V_DIM == static_cast<int>(FWD_H_V), "ChunkFwdH only supports V=128.");
    if (args.tiling.gDataType == FWD_H_DTYPE_FP32) {
        DispatchGateMode<float>(args);
    } else {
        DispatchGateMode<bfloat16_t>(args);
    }
}

} // namespace GDN

#ifndef TORCH_MODE
template <int V_DIM>
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
    args.tiling.dataType = static_cast<uint8_t>(tilingData->dataType);
    args.tiling.gDataType = static_cast<uint8_t>(tilingData->gDataType);
    args.tiling.stateDataType = static_cast<uint8_t>(tilingData->stateDataType);
    args.tiling.isVariedLen = tilingData->isVariedLen;
    args.tiling.shapeBatch = static_cast<uint32_t>(tilingData->shapeBatch);
    args.tiling.tokenBatch = static_cast<uint32_t>(tilingData->tokenBatch);
    args.tiling.useG = tilingData->useG;
    args.tiling.useGk = tilingData->useGk;
    args.tiling.useExp2 = tilingData->useExp2;
    args.tiling.stateVFirst = tilingData->stateVFirst;
    args.tiling.vWorkspaceOffset = static_cast<uint64_t>(tilingData->vWorkspaceOffset);
    args.tiling.vUpdateWorkspaceOffset = static_cast<uint64_t>(tilingData->vUpdateWorkspaceOffset);
    args.tiling.kDecayWorkspaceOffset = static_cast<uint64_t>(tilingData->kDecayWorkspaceOffset);
    args.tiling.hWorkspaceOffset = static_cast<uint64_t>(tilingData->hWorkspaceOffset);
    GDN::DispatchFwdH<V_DIM>(args);
}
#endif
