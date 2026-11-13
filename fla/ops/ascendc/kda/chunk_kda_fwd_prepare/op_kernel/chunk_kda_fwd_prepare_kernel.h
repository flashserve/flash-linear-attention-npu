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

// A2 is dispatched directly through its six-key entry in the .cpp file.
#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"

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
        Arch35::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
        vec.Init(args);
        vec.Process();
    }
    if ASCEND_IS_AIC {
        Arch35::ChunkKdaFwdPrepareCube<CompilePolicy> cube;
        cube.Init(args);
        cube.Process();
    }
}

} // namespace KdaPrepare
#endif // CHUNK_KDA_FWD_PREPARE_KERNEL_H
