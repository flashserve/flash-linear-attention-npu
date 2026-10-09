/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_PRE_PROCESS_FWD_KERNEL_MERGED_H
#define OP_API_INC_ACLNN_PRE_PROCESS_FWD_KERNEL_MERGED_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnPreProcessFwdKernelMergedGetWorkspaceSize
 * parameters (order aligned with the ctypes wrapper
 * torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py):
 * k : required,  [1, HK, T, K] BF16（gk 路径下为预 gate 的 kg，HK == HV）
 * w : required,  GDN/KDA [1, HV, T, K]；DPLR [1, HK, T, K]
 * u : required,  [1, HV, T, V] BF16
 * gOptional : optional, [1, HV, T] 标量 gate（FP32/BF16）；与 gk 二选一
 * gkOptional : optional, [1, HV, T, K] 逐 K gate（FP32/BF16）；与 g 二选一
 * bgOptional : **必须传 nullptr**。它是 DPLR 的专用输入，本版本不支持 DPLR，
 *              传非空会被 host 直接拒绝（这一位仅为保持 ABI 参数顺序而保留）
 * vOptional : **必须传 nullptr**。同上：DPLR 专用；GDN/KDA 的取值来自 u
 * cuSeqlensOptional : required, host int 数组（严格递增，0 <= cu[0] < cu[-1] <= T）
 * chunkSize : required（固定 64）
 * hmOut : required, [Nseq, HV, K, V+K] FP32（Nseq = len(cuSeqlens)-1）
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPreProcessFwdKernelMergedGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *bgOptional,
    const aclTensor *vOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    const aclTensor *hmOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnPreProcessFwdKernelMerged
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPreProcessFwdKernelMerged(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
