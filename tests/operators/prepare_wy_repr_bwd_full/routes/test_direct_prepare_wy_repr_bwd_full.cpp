// Canonical Ascend C direct-launch contract for PrepareWyReprBwdFull.
// blockDim/workspace/tiling are produced from tests/op_cases/prepare_wy_repr_bwd_full.json.
#include "acl/acl.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void prepare_wy_repr_bwd_full(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dA, GM_ADDR dw, GM_ADDR du, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling);

void LaunchPrepareWyReprBwdFull(uint32_t blockDim, aclrtStream stream, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dA, GM_ADDR dw, GM_ADDR du, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling)
{
    prepare_wy_repr_bwd_full<<<blockDim, nullptr, stream>>>(k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, workspace, tiling);
}
