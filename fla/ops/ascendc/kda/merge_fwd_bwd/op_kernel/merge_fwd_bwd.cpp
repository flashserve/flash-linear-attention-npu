/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "merge_fwd_bwd_struct.h"
#include "merge_fwd_bwd_tiling_key.h"
#include "merge_fwd_bwd_common.h"
#include "merge_fwd_bwd_cube.h"
#include "merge_fwd_bwd_vector.h"

namespace MergeFwBwd {

template <typename T>
__aicore__ inline void MergeFwdBwdKernelImpl(
    GM_ADDR agHm, GM_ADDR h, GM_ADDR workspace, const MergeFwdBwdTilingData *tiling)
{
    if ASCEND_IS_AIC {
        MergeFwdBwdCubeProcess<T> cube(agHm, h, workspace);
        cube.Init(tiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        MergeFwdBwdVectorProcess<T> vector(agHm, h, workspace, tiling);
        vector.Init(&pipe);
        vector.Process();
    }
}

} // namespace MergeFwBwd

#ifndef TORCH_MODE
using namespace AscendC;

__aicore__ inline void CopyTilingFromGm(
    const __gm__ MergeFwBwd::MergeFwdBwdTilingData *src, MergeFwBwd::MergeFwdBwdTilingData &dst)
{
    dst.S = src->S;
    dst.Hv = src->Hv;
    dst.K = src->K;
    dst.V = src->V;
    dst.rank = src->rank;
    dst.N = src->N;
    dst.forward = src->forward;
    dst.usedAic = src->usedAic;
    dst.sysWorkspaceSize = src->sysWorkspaceSize;
}

template <int D_T>
__global__ __aicore__ void merge_fwd_bwd(GM_ADDR ag_hm, GM_ADDR h, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MergeFwBwd::MergeFwdBwdTilingData);
    MergeFwBwd::MergeFwdBwdTilingData tilingData;
    CopyTilingFromGm(reinterpret_cast<__gm__ MergeFwBwd::MergeFwdBwdTilingData *>(tiling), tilingData);
    // 3510 GetUserWorkspace() uses __get_kfc_workspace_addr() and is not shared
    // AIC/AIV. Adding sysWorkspaceSize here OOBs because `workspace` is already
    // the launch user pointer. Use the kernel argument as-is.
    GM_ADDR userWorkspace = workspace;

    if constexpr (D_T == MERGE_FWD_BWD_TPL_FP32) {
        MergeFwBwd::MergeFwdBwdKernelImpl<float>(ag_hm, h, userWorkspace, &tilingData);
    } else {
        MergeFwBwd::MergeFwdBwdKernelImpl<bfloat16_t>(ag_hm, h, userWorkspace, &tilingData);
    }
}
#endif
