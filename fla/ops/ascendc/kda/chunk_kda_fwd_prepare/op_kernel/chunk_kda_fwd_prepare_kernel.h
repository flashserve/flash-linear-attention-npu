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
#include "chunk_kda_fwd_prepare_struct.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
#include "arch92/chunk_kda_fwd_prepare_cube.h"
#include "arch92/chunk_kda_fwd_prepare_vec.h"
#else
#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"
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

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void RunPrepare(const PrepareKernelArgs &args)
{
    if ASCEND_IS_AIV {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
        Arch92::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
#else
        Arch35::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
#endif
        vec.Init(args);
        vec.Process();
    }
    if ASCEND_IS_AIC {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
        Arch92::ChunkKdaFwdPrepareCube<CompilePolicy> cube;
#else
        Arch35::ChunkKdaFwdPrepareCube<CompilePolicy> cube;
#endif
        cube.Init(args);
        cube.Process();
    }
}

} // namespace KdaPrepare
#endif // CHUNK_KDA_FWD_PREPARE_KERNEL_H
