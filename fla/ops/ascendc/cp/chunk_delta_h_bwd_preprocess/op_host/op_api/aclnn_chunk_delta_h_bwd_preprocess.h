/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ACLNN_CHUNK_DELTA_H_BWD_PREPROCESS_H_
#define ACLNN_CHUNK_DELTA_H_BWD_PREPROCESS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnChunkDeltaHBwdPreprocessGetWorkspaceSize
 * parameters :
 * q : required, [B, Hk, T, K], GDN passes raw q, KDA/GDN2 pass qg
 * k : required, [B, Hk, T, K], GDN passes raw k, KDA/GDN2 pass kg
 * w : required, [B, Hv, T, K], WY auxiliary w_wy (not the public w_gate)
 * dO : required, [B, Hv, T, V], output gradient
 * dv : required, [B, Hv, T, V], local dV that does not contain the future-state term
 * gOptional : optional, [B, Hv, T], scalar chunk-local cumulative log2 gate (GDN)
 * gkOptional : optional, [B, Hv, T, K], per-K chunk-local cumulative log2 gate (KDA/GDN2)
 * cuSeqlensOptional : optional, INT64 [N+1], the [bos, eos) of the single packed segment
 * scale : optional attribute, multiplies Q^T dO only
 * chunkSize : optional attribute, must match the chunk size used by gate cumsum / WY / backward
 * dhmOut : required output, [Hv, K, V+K] FP32; [..., 0:V] = E_r, [..., V:V+K] = P_r
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkDeltaHBwdPreprocessGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *dO,
    const aclTensor *dv,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclIntArray *cuSeqlensOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *dhmOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnChunkDeltaHBwdPreprocess
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkDeltaHBwdPreprocess(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_CHUNK_DELTA_H_BWD_PREPROCESS_H_
