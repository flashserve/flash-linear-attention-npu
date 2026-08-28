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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ChunkFwdHCubeArch35<CompilePolicy, STATE_V_FIRST> cube;
#else
        ChunkFwdHCubeArch22<CompilePolicy, STATE_V_FIRST> cube;
#endif
        cube.Init(args);
        cube.Process();
    }
    if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ChunkFwdHVecArch35<GateT, CompilePolicy, STATE_V_FIRST> vec;
#else
        ChunkFwdHVecArch22<GateT, CompilePolicy, STATE_V_FIRST> vec;
#endif
        vec.Init(args);
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

__aicore__ inline void DispatchFwdH(const FwdHKernelArgs &args)
{
    if (args.tiling.gDataType == FWD_H_DTYPE_FP32) {
        DispatchGateMode<float>(args);
    } else {
        DispatchGateMode<bfloat16_t>(args);
    }
}

} // namespace GDN

extern "C" __global__ __aicore__ void chunk_fwd_h(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v_new,
    GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(1)) {
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
        args.tiling.batch = tilingData->batch;
        args.tiling.seqlen = tilingData->seqlen;
        args.tiling.kNumHead = tilingData->kNumHead;
        args.tiling.vNumHead = tilingData->vNumHead;
        args.tiling.kHeadDim = tilingData->kHeadDim;
        args.tiling.vHeadDim = tilingData->vHeadDim;
        args.tiling.chunkSize = tilingData->chunkSize;
        args.tiling.useInitialState = tilingData->useInitialState;
        args.tiling.storeFinalState = tilingData->storeFinalState;
        args.tiling.dataType = tilingData->dataType;
        args.tiling.gDataType = tilingData->gDataType;
        args.tiling.stateDataType = tilingData->stateDataType;
        args.tiling.isVariedLen = tilingData->isVariedLen;
        args.tiling.shapeBatch = tilingData->shapeBatch;
        args.tiling.tokenBatch = tilingData->tokenBatch;
        args.tiling.useG = tilingData->useG;
        args.tiling.useGk = tilingData->useGk;
        args.tiling.useExp2 = tilingData->useExp2;
        args.tiling.stateVFirst = tilingData->stateVFirst;
        args.tiling.vWorkspaceOffset = tilingData->vWorkspaceOffset;
        args.tiling.vUpdateWorkspaceOffset = tilingData->vUpdateWorkspaceOffset;
        args.tiling.kDecayWorkspaceOffset = tilingData->kDecayWorkspaceOffset;
        args.tiling.hWorkspaceOffset = tilingData->hWorkspaceOffset;
        args.tiling.numSeqWorkspaceOffset = tilingData->numSeqWorkspaceOffset;
        args.tiling.numChunksWorkspaceOffset = tilingData->numChunksWorkspaceOffset;
        GDN::DispatchFwdH(args);
    }
}
