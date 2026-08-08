/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_bwd_dhu_struct.h
 * \brief Shared tiling data for chunk_gated_delta_rule_bwd_dhu.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H
#define CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

namespace GDN {

#define TPL_BF16 10
#define TPL_FP16 20
#define TPL_FP32 30

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(ChunkGatedDeltaRuleBwdDhu,
    ASCENDC_TPL_DTYPE_DECL(D_T_Q, TPL_BF16,
        TPL_FP16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, TPL_BF16,
        TPL_FP16,
        TPL_FP32),
    ASCENDC_TPL_UINT_DECL(V, 1, ASCENDC_TPL_UI_LIST, 128, 256),
    ASCENDC_TPL_UINT_DECL(USE_GK, 1, ASCENDC_TPL_UI_LIST, 0, 1),
);

#define TPL_SEL_ONE(Q_TYPE, G_TYPE, V_VALUE, USE_GK_VALUE) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DTYPE_SEL(D_T_Q, Q_TYPE), \
        ASCENDC_TPL_DTYPE_SEL(D_T_G, G_TYPE), \
        ASCENDC_TPL_UINT_SEL(V, ASCENDC_TPL_UI_LIST, V_VALUE), \
        ASCENDC_TPL_UINT_SEL(USE_GK, ASCENDC_TPL_UI_LIST, USE_GK_VALUE), \
    )

#define TPL_SEL_FOR_PAIR(Q_TYPE, G_TYPE, USE_GK_VALUE) \
    TPL_SEL_ONE(Q_TYPE, G_TYPE, 128, USE_GK_VALUE), \
    TPL_SEL_ONE(Q_TYPE, G_TYPE, 256, USE_GK_VALUE)

#define TPL_SEL_FOR_GATE(USE_GK_VALUE) \
    TPL_SEL_FOR_PAIR(TPL_BF16, TPL_FP32, USE_GK_VALUE), \
    TPL_SEL_FOR_PAIR(TPL_FP16, TPL_FP32, USE_GK_VALUE), \
    TPL_SEL_FOR_PAIR(TPL_BF16, TPL_BF16, USE_GK_VALUE), \
    TPL_SEL_FOR_PAIR(TPL_FP16, TPL_FP16, USE_GK_VALUE)

ASCENDC_TPL_SEL(
    TPL_SEL_FOR_GATE(0),
    TPL_SEL_FOR_GATE(1),
);
#undef TPL_SEL_FOR_GATE
#undef TPL_SEL_FOR_PAIR
#undef TPL_SEL_ONE
#endif

struct ChunkGatedDeltaRuleBwdDhuTilingData {
    int64_t B;
    int64_t HK;
    int64_t HV;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t HRatio;
    int64_t chunkSize;
    int64_t chunkNumForT;
    int64_t totalChunkNum;
    int64_t chunkTaskNum;
    int64_t seqNum;
    int64_t headWindowNum;
    int64_t taskNum;
    int64_t isVariable;
    int64_t hasDh0;
    int64_t dh0ClearCoreNum;
    int64_t dh0ClearElemsPerCore;
    int64_t dh0ClearTailElems;
    int64_t hasGk;
    int64_t workspaceElemsPerSubBlock;
    int64_t qgWorkspaceOffset;
    int64_t stateWorkspaceOffset;
    int64_t dvStateWorkspaceOffset;
    int64_t termQWorkspaceOffset;
    int64_t dv2WorkspaceOffset;
    int64_t termWWorkspaceOffset;
    int64_t qgWorkspaceElems;
    int64_t stateWorkspaceElems;
    int64_t dvStateWorkspaceElems;
    int64_t termQWorkspaceElems;
    int64_t dv2WorkspaceElems;
    int64_t termWWorkspaceElems;
    int64_t vecRow;
    float scale;
};

} // namespace GDN

#endif // CHUNK_GATED_DELTA_RULE_BWD_DHU_STRUCT_H
