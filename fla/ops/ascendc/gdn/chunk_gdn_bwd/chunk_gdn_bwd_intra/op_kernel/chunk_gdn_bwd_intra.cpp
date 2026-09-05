/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"
#include "chunk_gdn_bwd_intra_struct.h"
#include "chunk_gdn_bwd_intra_tiling_key.h"
#include "chunk_gdn_bwd_intra_stage0.h"

namespace GDN {

template <uint32_t KEY>
struct IntraMainType;
template <>
struct IntraMainType<CHUNK_GDN_BWD_INTRA_MAIN_BF16> {
    using type = bfloat16_t;
};
template <>
struct IntraMainType<CHUNK_GDN_BWD_INTRA_MAIN_FP16> {
    using type = half;
};

template <uint32_t KEY>
struct IntraAuxType;
template <>
struct IntraAuxType<CHUNK_GDN_BWD_INTRA_GATE_BF16> {
    using type = bfloat16_t;
};
template <>
struct IntraAuxType<CHUNK_GDN_BWD_INTRA_GATE_FP32> {
    using type = float;
};

template <uint32_t MAIN_KEY, uint32_t GATE_KEY, uint32_t BETA_KEY>
__aicore__ inline void DispatchStage0(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR a, GM_ADDR dO,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u,
    GM_ADDR dvLocal, GM_ADDR workspace,
    const ChunkGdnBwdIntraTilingData *__restrict tiling)
{
    using MainT = typename IntraMainType<MAIN_KEY>::type;
    using GateT = typename IntraAuxType<GATE_KEY>::type;
    using BetaT = typename IntraAuxType<BETA_KEY>::type;
    if ASCEND_IS_AIC {
        ChunkGdnBwdIntraStage0Cube<MainT> stage0;
        stage0.Init(q, k, v, dO, w, u, dvLocal, workspace,
                    cuSeqlens, chunkIndices, tiling);
        stage0.Process();
    }
    if ASCEND_IS_AIV {
        ChunkGdnBwdIntraStage0Vector<MainT, GateT, BetaT> stage0;
        stage0.Init(a, g, beta, w, u, dvLocal, workspace,
                    cuSeqlens, chunkIndices, tiling);
        stage0.Process();
    }
}

} // namespace GDN

#ifndef TORCH_MODE
template <uint32_t STRATEGY_KEY, uint32_t MAIN_KEY, uint32_t GATE_KEY, uint32_t BETA_KEY>
__global__ __aicore__ void chunk_gdn_bwd_intra(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR a,
    GM_ADDR dO, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u,
    GM_ADDR dvLocal, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)STRATEGY_KEY;
    REGISTER_TILING_DEFAULT(GDN::ChunkGdnBwdIntraTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkGdnBwdIntraTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = tilingData.stage == 0 ? workspace : AscendC::GetUserWorkspace(workspace);
    GDN::DispatchStage0<MAIN_KEY, GATE_KEY, BETA_KEY>(
        q, k, v, g, beta, a, dO, cuSeqlens, chunkIndices, w, u, dvLocal,
        userWorkspace, &tilingData);
}
#endif
