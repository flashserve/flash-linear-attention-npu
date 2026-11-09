/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_KERNEL_H
#define CHUNK_KDA_FWD_FINALIZE_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_kda_fwd_finalize_cube.h"
// 各架构的 AIV mover 是不同头文件、不同命名空间（Arch92/Arch35），无法用 trait 表驱动，
// 只能用预处理分支选头；Arch22 无独立 AIV mover，不包含。C++ 侧差异已收敛到 platform.h。
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
#include "arch92/chunk_kda_fwd_finalize_vec.h"
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_finalize_vec.h"
#endif

namespace KdaFinalize {

template <bool UseAivInputMover, bool StateVFirst, bool OutputSequenceMajor>
__aicore__ inline void RunFinalize(const FinalizeArgs &args)
{
    if ASCEND_IS_AIC {
        FinalizeCube<StateVFirst, OutputSequenceMajor, UseAivInputMover> cube;
        cube.Init(args);
        cube.Process();
    }
// AIV 分支实例化的是具体架构命名空间下的 mover（Arch92/Arch35），命名空间/模板实例
// 无法用 trait 表驱动，保留预处理分支。
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
    if constexpr (UseAivInputMover) {
        if ASCEND_IS_AIV {
            Arch92::FinalizeInputMover<StateVFirst> inputMover;
            inputMover.Init(args);
            inputMover.Process();
        }
    }
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if constexpr (UseAivInputMover) {
        if ASCEND_IS_AIV {
            Arch35::FinalizeInputMover<StateVFirst> inputMover;
            inputMover.Init(args);
            inputMover.Process();
        }
    }
#endif
}

template <bool UseAivInputMover>
__aicore__ inline void DispatchFinalize(const FinalizeArgs &args)
{
    if (args.tiling.stateVFirst) {
        if (args.tiling.outputSequenceMajor) {
            RunFinalize<UseAivInputMover, true, true>(args);
        } else {
            RunFinalize<UseAivInputMover, true, false>(args);
        }
    } else if (args.tiling.outputSequenceMajor) {
        RunFinalize<UseAivInputMover, false, true>(args);
    } else {
        RunFinalize<UseAivInputMover, false, false>(args);
    }
}

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_KERNEL_H
