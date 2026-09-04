/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "kernel_operator.h"
#include "w_mmad_demo.h"

using namespace AscendC;

// Stage4/5 MBH demo: (I+L)^{-1} from strictly-lower L. a=L, c=inv.
extern "C" __global__ __aicore__ void w_mmad_demo(
    GM_ADDR a, GM_ADDR b, GM_ADDR pre, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    (void)tilingData;
    WMmadDemoOp::Kernel op;
    op.Init(a, b, pre, c, workspace);
    op.Process();
}
