/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_KERNEL_H
#define CHUNK_KDA_FWD_PREPARE_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_prepare_policy.h"
#include "chunk_kda_fwd_prepare_runtime_policy.h"
#include "chunk_kda_fwd_prepare_struct.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"
#else
#include "arch22/chunk_kda_fwd_prepare_cube.h"
#include "arch22/chunk_kda_fwd_prepare_vec.h"
#include "arch22/chunk_kda_fwd_prepare_cube_target.h"
#include "arch22/chunk_kda_fwd_prepare_vec_target.h"
#include "arch22/chunk_kda_fwd_prepare_relay_config.h"
#endif

namespace KdaPrepare {

template <int DTYPE>
struct PrepareStorageType;

template <>
struct PrepareStorageType<CHUNK_KDA_FWD_PREPARE_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct PrepareStorageType<CHUNK_KDA_FWD_PREPARE_TPL_FP32> {
    using type = float;
};

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
template <typename GateT, typename BetaT, typename CompilePolicy, bool BF16_RELAY>
__aicore__ inline void RunArch22Generic(const PrepareKernelArgs &args, AscendC::TPipe *pipe)
{
    if ASCEND_IS_AIV {
        Arch22::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy, BF16_RELAY> vec;
        vec.Init(args, pipe);
        vec.Process();
    }
    if ASCEND_IS_AIC {
        Arch22::ChunkKdaFwdPrepareCube<CompilePolicy, BF16_RELAY> cube;
        cube.Init(args, pipe);
        cube.Process();
    }
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void RunArch22Selected(const PrepareKernelArgs &args, AscendC::TPipe *pipe)
{
    constexpr bool kBf16RelayEligible =
        std::is_same_v<GateT, float> && std::is_same_v<BetaT, bfloat16_t>;
    if constexpr (kBf16RelayEligible) {
        const auto &t = args.tiling;
        if (t.normMode == static_cast<uint32_t>(QkNormMode::L2) &&
            t.betaMode == static_cast<uint32_t>(BetaMode::Sigmoid) &&
            t.gateMode == static_cast<uint32_t>(GateMode::SafeSigmoid) &&
            t.useExp2 && t.outputMode == static_cast<uint32_t>(OutputMode::Save) &&
            Arch22RelayConfig::Matches({t.qkHeadNum, t.valueHeadNum,
                                       t.inputSequenceMajor, t.epsilon,
                                       t.lowerBound, t.scale})) {
            // Both engines make this decision from the same immutable tiling.
            RunArch22Generic<GateT, BetaT, PrepareCommonMathPolicy, true>(args, pipe);
            return;
        }
    }
    if constexpr (std::is_same_v<BetaT, bfloat16_t>) {
        const auto &t = args.tiling;
        if (t.normMode == static_cast<uint32_t>(QkNormMode::L2) &&
            t.betaMode == static_cast<uint32_t>(BetaMode::Sigmoid) &&
            t.useExp2 && t.outputMode == static_cast<uint32_t>(OutputMode::Save)) {
            RunArch22Generic<GateT, BetaT, PrepareCommonMathPolicy, false>(args, pipe);
            return;
        }
    }
    RunArch22Generic<GateT, BetaT, CompilePolicy, false>(args, pipe);
}
#endif

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void RunPrepare(const PrepareKernelArgs &args)
{
    if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch35::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
        vec.Init(args);
        vec.Process();
#else
        AscendC::TPipe pipe;
        if constexpr (CompilePolicy::arch22FastLeaf) {
            Arch22Target::TargetVectorContext<GateT, BetaT, CompilePolicy> vec;
            Arch22Target::InitTargetVec(vec, args, &pipe);
            Arch22Target::ProcessTargetVec(vec);
        } else {
            RunArch22Selected<GateT, BetaT, CompilePolicy>(args, &pipe);
        }
#endif
    }

    if ASCEND_IS_AIC {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch35::ChunkKdaFwdPrepareCube<CompilePolicy> cube;
        cube.Init(args);
        cube.Process();
#else
        AscendC::TPipe pipe;
        if constexpr (CompilePolicy::arch22FastLeaf) {
            Arch22Target::TargetCubeContext<CompilePolicy> cube;
            Arch22Target::InitTargetCube(cube, args, &pipe);
            Arch22Target::ProcessTargetCube(cube);
        } else {
            RunArch22Selected<GateT, BetaT, CompilePolicy>(args, &pipe);
        }
#endif
    }
}

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_KERNEL_H
