/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of the BSD 3-Clause License (the "License").
 */

// 本文件只声明源码合同，不作为独立编译证据。真实模板编译和设备比对
// 由 fast_kernel_launch_example 完成，两处共用同一份 Launch。
#include "examples/fast_kernel_launch_example/csrc/chunk_kda_fwd_prepare/chunk_kda_fwd_prepare_direct_kernel.h"

namespace KdaPrepareDirect {

// prepare_dense_bnsd_raw 对应的默认 FP32 gate/beta 路径。
template void Launch<
    CHUNK_KDA_FWD_PREPARE_TPL_FP32,
    CHUNK_KDA_FWD_PREPARE_TPL_FP32,
    CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
    CHUNK_KDA_FWD_PREPARE_BETA_RAW,
    CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
    false,
    false,
    CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE>(
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
    true,
    CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE>(
    uint32_t, aclrtStream,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const KdaPrepare::ChunkKdaFwdPrepareTilingData &);

// prepare_dense_bsnd_fused_save 使用相同数学配置，并搬出全部 13 个输出。
template void Launch<
    CHUNK_KDA_FWD_PREPARE_TPL_BF16,
    CHUNK_KDA_FWD_PREPARE_TPL_BF16,
    CHUNK_KDA_FWD_PREPARE_NORM_L2,
    CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID,
    CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID,
    true,
    true,
    CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE>(
    uint32_t, aclrtStream,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    const KdaPrepare::ChunkKdaFwdPrepareTilingData &);

} // namespace KdaPrepareDirect
