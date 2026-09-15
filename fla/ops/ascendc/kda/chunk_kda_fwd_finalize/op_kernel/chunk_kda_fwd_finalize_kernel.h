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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_finalize_vec.h"
#else
#include "arch22/chunk_kda_fwd_finalize_vec.h"
#endif

namespace KdaFinalize {

template <bool StateVFirst, bool OutputSequenceMajor>
__aicore__ inline void RunFinalize(const FinalizeArgs &args)
{
    if ASCEND_IS_AIC {
        FinalizeCube<StateVFirst> cube;
        cube.Init(args);
        cube.Process();
    }
    if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch35::FinalizeVec<OutputSequenceMajor> vec;
        vec.Init(args);
#else
        AscendC::TPipe pipe;
        Arch22::FinalizeVec<OutputSequenceMajor> vec;
        vec.Init(args, &pipe);
#endif
        vec.Process();
    }
}

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_KERNEL_H
