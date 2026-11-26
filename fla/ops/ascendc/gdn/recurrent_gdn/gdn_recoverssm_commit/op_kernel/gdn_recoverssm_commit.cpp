/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "gdn_recoverssm_commit_tiling_data.h"
#include "gdn_recoverssm_commit_tiling_key.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/gdn_recoverssm_commit.h"
#else
#error "GdnRecoverssmCommit currently supports Ascend 950 only"
#endif

using namespace AscendC;

namespace GDN {

template <int D_T_STATE>
struct GrscStateType;

template <>
struct GrscStateType<GRSC_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct GrscStateType<GRSC_TPL_FP32> {
    using type = float;
};

template <int D_T_META>
struct GrscMetaType;

template <>
struct GrscMetaType<GRSC_TPL_INT32> {
    using type = int32_t;
};

template <>
struct GrscMetaType<GRSC_TPL_INT64> {
    using type = int64_t;
};

} // namespace GDN

template <int D_T_STATE, bool ALIGN_MODE, int D_T_PLAN, int D_T_STATE_INDEX>
__global__ __aicore__ void
gdn_recoverssm_commit(GM_ADDR state, GM_ADDR replay, GM_ADDR stateCommitCacheAddressStrideTable, GM_ADDR stateIndices,
                      GM_ADDR commitLens, GM_ADDR finalStateIndices, GM_ADDR boundaryStateIndices,
                      GM_ADDR boundaryRecoveryLens, GM_ADDR stateOut, GM_ADDR workspace, GM_ADDR tiling_gm)
{
    (void)stateOut;
    (void)workspace;
    REGISTER_TILING_DEFAULT(GDN::GdnRecoverssmCommitTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using StateType = typename GDN::GrscStateType<D_T_STATE>::type;
    using PlanType = typename GDN::GrscMetaType<D_T_PLAN>::type;
    using StateIndexType = typename GDN::GrscMetaType<D_T_STATE_INDEX>::type;
    GDN::GdnRecoverssmCommitVector<StateType, PlanType, StateIndexType, ALIGN_MODE> op;
    op.Init(&tilingData, state, replay, stateCommitCacheAddressStrideTable, stateIndices, commitLens, finalStateIndices,
            boundaryStateIndices, boundaryRecoveryLens);
    op.Process();
}
