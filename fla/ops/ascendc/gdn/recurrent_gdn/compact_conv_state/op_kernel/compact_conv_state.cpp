/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "compact_conv_state_tiling_data.h"
#include "compact_conv_state_tiling_key.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/compact_conv_state.h"
#else
#error "CompactConvState currently supports Ascend 950 only"
#endif

using namespace AscendC;

namespace GDN {
template <int D_T_STATE>
struct CcsStateType;
template <>
struct CcsStateType<CCS_TPL_BF16> {
    using type = bfloat16_t;
};
template <>
struct CcsStateType<CCS_TPL_FP32> {
    using type = float;
};

template <int D_T_META>
struct CcsMetaType;
template <>
struct CcsMetaType<CCS_TPL_INT32> {
    using type = int32_t;
};
template <>
struct CcsMetaType<CCS_TPL_INT64> {
    using type = int64_t;
};
} // namespace GDN

template <int D_T_STATE, bool ALIGN_MODE, int D_T_PLAN, int D_T_STATE_INDEX, bool SHARED_DIM_STRIDE>
__global__ __aicore__ void
compact_conv_state(GM_ADDR convState, GM_ADDR baseAddrs, GM_ADDR blockStrides, GM_ADDR dimStrides, GM_ADDR tokenStrides,
                   GM_ADDR stateIndices, GM_ADDR commitLens, GM_ADDR finalStateIndices, GM_ADDR boundaryStateIndices,
                   GM_ADDR boundaryRecoveryLens, GM_ADDR convStateOut, GM_ADDR workspace, GM_ADDR tiling_gm)
{
    (void)convStateOut;
    (void)workspace;
    REGISTER_TILING_DEFAULT(GDN::CompactConvStateTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using StateType = typename GDN::CcsStateType<D_T_STATE>::type;
    using PlanType = typename GDN::CcsMetaType<D_T_PLAN>::type;
    using StateIndexType = typename GDN::CcsMetaType<D_T_STATE_INDEX>::type;
    GDN::CompactConvStateVector<StateType, PlanType, StateIndexType, ALIGN_MODE, SHARED_DIM_STRIDE> op;
    op.Init(&tilingData, convState, baseAddrs, blockStrides, dimStrides, tokenStrides, stateIndices, commitLens,
            finalStateIndices, boundaryStateIndices, boundaryRecoveryLens);
    op.Process();
}
