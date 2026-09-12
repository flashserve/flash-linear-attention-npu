/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"
#include "chunk_gdn_bwd_intra_struct.h"
#include "chunk_gdn_bwd_intra_tiling_key.h"
#include "chunk_gdn_bwd_intra_cube.h"
#include "chunk_gdn_bwd_intra_vector.h"

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

template <uint32_t G_KEY, uint32_t MAIN_KEY, uint32_t GATE_KEY, uint32_t BETA_KEY>
__aicore__ inline void DispatchKernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR a, GM_ADDR dO,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u,
    GM_ADDR dvLocal, GM_ADDR workspace,
    const ChunkGdnBwdIntraTilingData *__restrict tiling)
{
    static_assert(G_KEY >= 1 && G_KEY <= 4, "G must be in [1, 4].");
    using MainT = typename IntraMainType<MAIN_KEY>::type;
    using GateT = typename IntraAuxType<GATE_KEY>::type;
    using BetaT = typename IntraAuxType<BETA_KEY>::type;
    // 一个 MixBlock 包含 1 个 AIC 和 2 个 AIV：AIC 顺序执行 Stage 0/2，
    // 两个 AIV 并行执行 Stage 1 的前、后半行，二者通过 CrossCore flag 交接数据。
    if ASCEND_IS_AIC {
        ChunkGdnBwdIntraCube<MainT, G_KEY> kernel;
        kernel.Init(q, k, v, dO, w, u, dvLocal, workspace,
                    cuSeqlens, chunkIndices, tiling);
        kernel.Process();
    }
    if ASCEND_IS_AIV {
        ChunkGdnBwdIntraVector<MainT, GateT, BetaT, G_KEY> kernel;
        kernel.Init(v, dO, a, g, beta, dvLocal, workspace,
                    cuSeqlens, chunkIndices, tiling);
        kernel.Process();
    }
}

} // namespace GDN

#ifndef TORCH_MODE
template <uint32_t STRATEGY_KEY, uint32_t G_KEY,
          uint32_t MAIN_KEY, uint32_t GATE_KEY, uint32_t BETA_KEY>
__global__ __aicore__ void chunk_gdn_bwd_intra(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR a,
    GM_ADDR dO, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u,
    GM_ADDR dvLocal, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)STRATEGY_KEY;
    REGISTER_TILING_DEFAULT(GDN::ChunkGdnBwdIntraTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkGdnBwdIntraTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GDN::DispatchKernel<G_KEY, MAIN_KEY, GATE_KEY, BETA_KEY>(
        q, k, v, g, beta, a, dO, cuSeqlens, chunkIndices, w, u, dvLocal,
        userWorkspace, &tilingData);
}
#endif
