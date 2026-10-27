/** Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause */
#ifndef CHUNK_KDA_FWD_PREPARE_RUNTIME_KEY_H
#define CHUNK_KDA_FWD_PREPARE_RUNTIME_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"
#include "chunk_kda_fwd_prepare_runtime_policy.h"

namespace KdaPrepare::RuntimeKey {
ASCENDC_TPL_ARGS_DECL(
    ChunkKdaFwdPrepare,
    ASCENDC_TPL_DTYPE_DECL(D_T_GATE, 10, 30),
    ASCENDC_TPL_DTYPE_DECL(D_T_BETA, 10, 30),
    ASCENDC_TPL_UINT_DECL(PIPELINE_VARIANT, 1, ASCENDC_TPL_UI_LIST, 0, 1));

#define PREPARE_RUNTIME_SEL(G, B, P) \
    ASCENDC_TPL_ARGS_SEL( \
        ASCENDC_TPL_DTYPE_SEL(D_T_GATE, G), \
        ASCENDC_TPL_DTYPE_SEL(D_T_BETA, B), \
        ASCENDC_TPL_UINT_SEL(PIPELINE_VARIANT, ASCENDC_TPL_UI_LIST, P))
ASCENDC_TPL_SEL(
    PREPARE_RUNTIME_SEL(10, 10, 0),
    PREPARE_RUNTIME_SEL(10, 30, 0),
    PREPARE_RUNTIME_SEL(30, 10, 0),
    PREPARE_RUNTIME_SEL(30, 30, 0),
    PREPARE_RUNTIME_SEL(10, 10, 1),
    PREPARE_RUNTIME_SEL(30, 10, 1));
#undef PREPARE_RUNTIME_SEL

#if !defined(ASCENDC_TPL_KERNEL) && !defined(ASCENDC_TPL_PRE)
inline uint64_t Encode(uint64_t gate, uint64_t beta, PreparePipeline pipeline)
{
    return GET_TPL_TILING_KEY(gate, beta, static_cast<uint64_t>(pipeline));
}
#endif
} // namespace KdaPrepare::RuntimeKey
#endif
