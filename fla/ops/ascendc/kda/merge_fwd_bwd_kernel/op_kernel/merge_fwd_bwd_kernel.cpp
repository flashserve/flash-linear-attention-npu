/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif

#include "merge_fwd_bwd_kernel_struct.h"
#include "merge_fwd_bwd_kernel_tiling_key.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/merge_fwd_bwd_kernel_cube.h"
#include "arch35/merge_fwd_bwd_kernel_vector.h"
#else
#include "merge_fwd_bwd_kernel_cube.h"
#include "merge_fwd_bwd_kernel_vector.h"
#endif

namespace MergeFwBwd {

template <typename T>
__aicore__ inline void MergeFwdBwdKernelImpl(
    GM_ADDR agHm, GM_ADDR h, GM_ADDR workspace, const MergeFwdBwdKernelTilingData *tiling)
{
    // FP32 scratch [3,HV,128,128] lives at the start of user workspace.
    GM_ADDR scratch = workspace;
    if ASCEND_IS_AIC {
        MergeFwdBwdKernelCubeProcess<T> cube(agHm, scratch, h, workspace);
        cube.Init(tiling);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        TPipe pipe;
        MergeFwdBwdKernelVectorProcess<T> vector(agHm, scratch, h, workspace, tiling);
        vector.Init(&pipe);
        vector.Process();
    }
}

} // namespace MergeFwBwd

#ifndef TORCH_MODE
using namespace AscendC;

__aicore__ inline void CopyTilingFromGm(
    const __gm__ MergeFwBwd::MergeFwdBwdKernelTilingData *src, MergeFwBwd::MergeFwdBwdKernelTilingData &dst)
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
__global__ __aicore__ void merge_fwd_bwd_kernel(GM_ADDR ag_hm, GM_ADDR h, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MergeFwBwd::MergeFwdBwdKernelTilingData);
    MergeFwBwd::MergeFwdBwdKernelTilingData tilingData;
    CopyTilingFromGm(reinterpret_cast<__gm__ MergeFwBwd::MergeFwdBwdKernelTilingData *>(tiling), tilingData);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // 3510 GetUserWorkspace() uses __get_kfc_workspace_addr() and is not shared
    // AIC/AIV. Adding sysWorkspaceSize here OOBs because `workspace` is already
    // the launch user pointer. Use the kernel argument as-is.
    GM_ADDR userWorkspace = workspace;
#else
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        userWorkspace = workspace + static_cast<uint64_t>(tilingData.sysWorkspaceSize);
    }
#endif

    if constexpr (D_T == MERGE_FWD_BWD_KERNEL_TPL_FP32) {
        MergeFwBwd::MergeFwdBwdKernelImpl<float>(ag_hm, h, userWorkspace, &tilingData);
    } else {
        MergeFwBwd::MergeFwdBwdKernelImpl<bfloat16_t>(ag_hm, h, userWorkspace, &tilingData);
    }
}
#endif
