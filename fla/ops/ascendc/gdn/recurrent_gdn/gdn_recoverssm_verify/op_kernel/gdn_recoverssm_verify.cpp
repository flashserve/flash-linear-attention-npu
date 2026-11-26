/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_verify_tiling_data.h"
#include "gdn_recoverssm_verify_tiling_key.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/gdn_recoverssm_verify.h"
#else
#error "GdnRecoverssmVerify currently supports Ascend 950 only"
#endif

using namespace AscendC;

namespace GDN {

template <int D_T_STATE>
struct GrsvStateType;

template <>
struct GrsvStateType<GRSV_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct GrsvStateType<GRSV_TPL_FP32> {
    using type = float;
};

} // namespace GDN

#ifndef TORCH_MODE
template <int D_T_STATE, bool USE_QK_L2NORM>
__global__ __aicore__ void gdn_recoverssm_verify(GM_ADDR A_log, GM_ADDR a, GM_ADDR b, GM_ADDR dt_bias, GM_ADDR q,
                                                 GM_ADDR k, GM_ADDR v, GM_ADDR checkpoint_state, GM_ADDR replay_cache,
                                                 GM_ADDR query_start_loc, GM_ADDR state_indices, GM_ADDR out,
                                                 GM_ADDR replay_cache_out, GM_ADDR workspace, GM_ADDR tiling_gm)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(GDN::GdnRecoverssmVerifyTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using StateType = typename GDN::GrsvStateType<D_T_STATE>::type;
    GDN::GdnRecoverssmVerifyVector<bfloat16_t, StateType, USE_QK_L2NORM> op(&tilingData);
    op.Init(A_log, a, b, dt_bias, q, k, v, checkpoint_state, replay_cache, query_start_loc, state_indices, out);
    (void)replay_cache_out;
    op.Process();
}
#endif
