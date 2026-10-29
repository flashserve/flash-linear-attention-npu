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
// gated 复用路径（仅 A2/arch22）：chunk_gated_delta_rule_fwd_h 的 preload
// kernel 整体复用——kernel 入口按模板条件分支，eligible 时现场构造 gated
// tiling 镜像到 workspace 后原样调用（gated 代码零改动）。
#include "gated_reuse/chunk_gated_delta_rule_fwd_h_struct.h"
#include "gated_reuse/gemm/kernel/gdn_fwd_h_kernel_preload.hpp"
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
    // gated 复用路径的编译期 eligible 判定（与 FwdHGatedReusable 谓词同条件，
    // 唯一消费方是注释/审读——代码零引用）。gated preload kernel 恒按 [K,V]
    // 读写 state（update epilogue 行=kHeadDim、列=vHeadDim），仅匹配
    // state_v_first=False 契约；vfirst=1 须 aclnn 层 TransposeLastTwo 翻译后
    // 以 false 进 kernel（独立算子门已做），否则回落老路径 STATE_V_FIRST 模板。
    static constexpr bool GATED_REUSABLE =
        !USE_GK && V_DIM == 128 &&
        !STATE_V_FIRST && D_T_G == CHUNK_FWD_H_TPL_FP32;
};

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
// gated 复用谓词：g 模式 + gate fp32 + vfirst=0。
// storeFinal 两值均放行（2026-10-26 放宽轮①）：gated scheduler 的
// NeedProcessStage2 = storeFinalState || !isFinalState 与老路径跳末段同款语义
// （运行时读 tiling，镜像已透传该字段）；storeFinal=false 时 l0op 层的 {0} 占位
// finalState 不被写（epilogue 仅 isFinalState && storeFinalState 才写）。
// state fp32/bf16 均放行（2026-10-26 放宽轮②）：gated update epilogue 的非
// float 分支原生支持 bf16 final（CAST_RINT 量化，与老路径 BF16 state 模板
// 同款量化点），kernel 主循环事件预置按 FinalStateElement 类型自动切换；
// StateT 由分流点按 CompilePolicy::STATE_FP32 选型传入。
// vfirst 条件（2026-10-26 收口轮补回，fe54077c 曾删）：gated preload kernel 恒
// [K,V] 产出，vfirst=0 才语义匹配。对独立算子 aclnnChunkFwdH 是死条件——其
// aclnn 层已把 vfirst=1 翻译成 false+转置张量进 kernel（dense 形态；varlen
// 形态不翻译、由本谓词送老路径 STATE_V_FIRST 模板）。本条件真正拦截的是无
// 翻译层的 l0op 组合调用（gated 整链 chunk_gated_delta_rule_fwd 透传
// stateVFirst=1 等）：vfirst=1 直达 gated 会产出 [K,V] 与调用方 [V,K] 契约
// 错乱，须回落老路径。
// （bf16 gate 首版不接入——gated 的 bf16 gate 路径未在本战役验证过；
//   vfirst=1 若要接入须调用方 aclnn 层转置配合——先例见 b+ 路线图归档。）
template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline constexpr bool FwdHGatedReusable()
{
    // USE_EXP2 必须 true：gated preload kernel 固定按 log2 域 gate 计算
    // （上游 Python 层 RCP_LN2 缩放后传入）；exp 域 gate 用例回落原路径。
    // STATE_V_FIRST 必须 false：gated state 原生 [K,V] 布局（update epilogue
    // 行=kHeadDim、列=vHeadDim），仅匹配 state_v_first=False 契约——与 host
    // 侧 gatedEligible（tiling.cpp）及 Traits::GATED_REUSABLE 三处同条件。
    return CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G &&
           CompilePolicy::USE_EXP2 && !STATE_V_FIRST &&
           std::is_same_v<GateT, float>;
}

// gated 复用分支：从 ChunkFwdHTilingData + 模板参构造 gated tiling 镜像
// （写入 workspace 的 gatedTilingOffset 段），随后原样调用 gated preload
// kernel。镜像由每个 AIC 核独立构造（同值幂等写，skeleton varlen 建表先例
// 同款）；AIV 侧不构造——gated kernel 的 Init 在 AIC/AIV 分别读同一 GM 镜像。
// 输入侧 g 已是 log2 域（调用方 Python 层 RCP_LN2 缩放，与 gated 上游一致）。
// StateT 按实际 state dtype 实例化（放宽轮②）：fp32 state 走 FP32 直写链
// （update epilogue 的 float 分支），bf16 state 走 CAST_RINT 量化链（非 float
// 分支——量化点与老路径 BF16 state 模板同款，kernel 主循环的事件预置配置
// 亦按 FinalStateElement 类型自动切换，bf16 原生支持）。
template <typename GateT, typename StateT>
__aicore__ inline void RunFwdHGatedReuse(const FwdHKernelArgs &args)
{
    using DataT = bfloat16_t;
    using GatedKernel = Catlass::Gemm::Kernel::GDNFwdHKernelPreload<DataT, GateT, StateT, float>;
    __gm__ uint8_t *user = args.workspace;
    // [对照实验 gr2：临时回退共享镜像（昨晚构建成功形态）——判环境 vs 代码]
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
        gtiling->gDataType = 2; // fp32 gate（GATED_REUSABLE 限定）
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
    GatedKernel gatedKernel;
    gatedKernel.Init(args.k, args.w, args.u, args.g, args.initialState,
                     args.cuSeqlens, args.chunkIndices, args.h, args.vNew,
                     args.finalState,
                     reinterpret_cast<GM_ADDR>(gtiling),
                     user);
    gatedKernel.Process();
}
#endif

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
__aicore__ inline void RunFwdHTyped(const FwdHKernelArgs &args)
{
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
    // gated 复用分流（编译期折叠）：eligible 形态整体走 gated preload
    // kernel——其双流骨架/调度器/epilogue 为生产验证形态。非 eligible
    // （gk/gate bf16/vfirst=1）回落原路径；state fp32/bf16 均放行（放宽轮②，
    // StateT 按 CompilePolicy::STATE_FP32 选型，镜像 stateDataType 同步设置）。
    if constexpr (FwdHGatedReusable<GateT, CompilePolicy, STATE_V_FIRST>()) {
        using GatedStateT = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
        RunFwdHGatedReuse<GateT, GatedStateT>(args);
        return;
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
    args.tiling.gatedGateDataType = static_cast<uint32_t>(tilingData->gatedGateDataType);
    args.tiling.gatedStateDataType = static_cast<uint32_t>(tilingData->gatedStateDataType);
    using Traits = GDN::FwdHKernelTraits<
        D_T_G, V_DIM, USE_GK, USE_EXP2, STATE_FP32, STATE_V_FIRST, STORE_FINAL, DENSE_T16>;
    GDN::RunFwdHTyped<typename Traits::GateT, typename Traits::CompilePolicy,
                      Traits::STATE_V_FIRST_VALUE>(args);
}
#endif
