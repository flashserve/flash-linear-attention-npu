#include "kernel_operator.h"
#include "arch35/chunk_gated_delta_rule_bwd_finalize_struct.h"
#include "arch35/chunk_gated_delta_rule_bwd_finalize_cube.h"
#include "arch35/chunk_gated_delta_rule_bwd_finalize_vector.h"

namespace GDN {

template <int DTYPE>
struct DTypeTraits;

template <>
struct DTypeTraits<TPL_BF16> { using type = bfloat16_t; };
template <>
struct DTypeTraits<TPL_FP32> { using type = float; };

} // namespace GDN

#ifndef TORCH_MODE
template <int D_T_Q, int D_T_G, bool USE_QK_L2NORM, bool USE_BETA_SIGMOID, bool ALLOW_NEG_EIGVAL>
__global__ __aicore__ void chunk_gated_delta_rule_bwd_finalize(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR v_new, GM_ADDR dO, GM_ADDR du,
    GM_ADDR g, GM_ADDR beta, GM_ADDR h, GM_ADDR dh, GM_ADDR A,
    GM_ADDR q_rstd, GM_ADDR k_rstd, GM_ADDR beta_raw,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta,
    GM_ADDR dg,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // 执行完整 Stage 0--15。非最终输出的跨核中间量按生命周期
    // 复用 workspace 区域，最终只返回 dq/dk/dv/dbeta/dg。
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

    REGISTER_TILING_DEFAULT(GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkGatedDeltaRuleBwdFinalizeTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    using QType = typename GDN::DTypeTraits<D_T_Q>::type;
    const int64_t workspaceRegionBytes = static_cast<int64_t>(AscendC::GetBlockNum()) *
        GDN::WORKSPACE_BUFFER_COUNT_8 * GDN::WORKSPACE_VECTOR_ELEMS * sizeof(QType);
    // Workspace aliases follow the actual stage lifetimes. Some regions are
    // intentionally reused after the earlier producer has been consumed.
    GM_ADDR kbgDoGWorkspace = userWorkspace;             // S0 kbg -> S12 doG
    GM_ADDR vbDkbWorkspace = userWorkspace + workspaceRegionBytes; // S0 vb -> S12 dkb
    GM_ADDR dqStage12Workspace = userWorkspace + 2 * workspaceRegionBytes;
    GM_ADDR dvbDkbTWorkspace = userWorkspace + 3 * workspaceRegionBytes;
    GM_ADDR dA0Workspace = userWorkspace + 4 * workspaceRegionBytes;
    GM_ADDR dk0Workspace = userWorkspace + 5 * workspaceRegionBytes;
    GM_ADDR dsWorkspace = userWorkspace + 6 * workspaceRegionBytes;
    if ASCEND_IS_AIC {
        GDN::ChunkGatedDeltaRuleBwdFinalizeCube<QType> cube;
        // dkb/dkb_t 分别复用 vb/dvb 区；Stage 9 写入时旧语义已完成消费。
        cube.Init(q, k, v_new, dO, du, h, dh, A,
                  kbgDoGWorkspace, vbDkbWorkspace, dA0Workspace,
                  dqStage12Workspace, vbDkbWorkspace, dvbDkbTWorkspace,
                  dsWorkspace, kbgDoGWorkspace, dk0Workspace, dk0Workspace,
                  cu_seqlens, chunk_indices, &tilingData);
        cube.Process();
    } else {
        using GType = typename GDN::DTypeTraits<D_T_G>::type;
        using BetaType = GType;
        AscendC::TPipe pipe;
        GDN::ChunkGatedDeltaRuleBwdFinalizeVector<
            QType, GType, BetaType, USE_QK_L2NORM, USE_BETA_SIGMOID, ALLOW_NEG_EIGVAL> vec;
        // 跨 stage 的矩阵/向量中间量使用独立 workspace 区域。
        vec.Init(q, k, v, v_new, dO, g, beta, h, dh,
                 q_rstd, k_rstd, dq, dk, dg, cu_seqlens, chunk_indices,
                 kbgDoGWorkspace, vbDkbWorkspace,
                 dvbDkbTWorkspace, dA0Workspace,
                 vbDkbWorkspace, dvbDkbTWorkspace,
                 kbgDoGWorkspace,
                 beta_raw, dbeta, dsWorkspace, dqStage12Workspace, dv,
                 dk0Workspace, dk0Workspace, &tilingData, &pipe);
        vec.Process();
    }
}
#endif
