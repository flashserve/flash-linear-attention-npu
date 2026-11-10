/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file kda_fwd_h_struct.h
 * \brief Private tiling data struct for ChunkKdaFwd's embedded FwdH.
 *
 * The aclnn/ascendc framework auto-generates a kernel-side tiling struct named
 * KdaFwdHPrivateTilingData from the fused ChunkKdaFwd tiling layout. This private
 * copy decouples ChunkKdaFwd from the standalone GDN FwdH operator. The fast launch extension
 * compiles the kernel standalone (without that auto-generated header), so it provides
 * the same plain struct here. The field order/types mirror the macro definition so the
 * kernel/scheduler can read it the same way, and so the struct can be passed by value
 * to the kernel via the <<<>>> launch (its address is a valid GM_ADDR on Atlas A2).
 */

#ifndef KDA_FWD_H_PRIVATE_STRUCT_H
#define KDA_FWD_H_PRIVATE_STRUCT_H

#include <cstdint>

struct KdaFwdHPrivateTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t dataType;
    int64_t gDataType;
    int64_t stateDataType;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    bool useG;
    bool useGk;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
    int64_t numSeqWorkspaceOffset;
    int64_t numChunksWorkspaceOffset;
};

#endif // KDA_FWD_H_PRIVATE_STRUCT_H
