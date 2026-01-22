/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prepare_wy_repr_bwd_full.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "prepare_wy_repr_bwd_full.h"
#include "lib/matmul_intf.h"
// #include "kernel_basic_intf.h"
using namespace AscendC;
__global__ __aicore__ void prepare_wy_repr_bwd_full(GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR A, GM_ADDR dA, GM_ADDR dw, GM_ADDR du, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
                                                    GM_ADDR dk, GM_ADDR dv, GM_ADDR dbeta, GM_ADDR dg, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    AscendC::AscendCUtils::SetOverflow(1);
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_1);
        if ASCEND_IS_AIC{
            PrepareWyReprBwdFullProcess<DTYPE_K> prepareWyReprBwdFullProcess(k, v, beta, A, dA, dw, du, g, dk, dv, dbeta, dg, workspace);
            prepareWyReprBwdFullProcess.Init(tiling);
            prepareWyReprBwdFullProcess.Process();
        }
        if ASCEND_IS_AIV{
            uint32_t coreIdx = AscendC::GetBlockIdx();
            TQue<AscendC::TPosition::VECIN, 1> quein;
            TQue<AscendC::TPosition::VECOUT, 1> queout;
            GlobalTensor<half> dkTensor;
            tPipe.InitBuffer(quein, 2, 128 * sizeof(half));
            tPipe.InitBuffer(queout, 2,128 * sizeof(half));
            dkTensor.SetGlobalBuffer((__gm__ half*)dk);


            printf("AIV hello world coreIdx:%u, AscendC::GetBlockNum():%u\n", coreIdx, AscendC::GetBlockNum());
            for (uint32_t loopIdx = coreIdx; loopIdx < 32; loopIdx += AscendC::GetBlockNum()) {
                // uint32_t bIdx = loopIdx / coreLoopsInB;
                // GemmCoord blockCoord = matmulBlockSchedulerDkb.GetBlockCoord(loopIdx);
                // GemmCoord actualBlockShape = matmulBlockSchedulerDkb.GetActualBlockShape(blockCoord);
                for (int h = 0; h < 4; h++) {
                    printf("offset %d\n", h * 2048 * 128 + 64 * loopIdx);
                    AscendC::CrossCoreWaitFlag(0x8);
                    AscendC::LocalTensor<half> in = quein.AllocTensor<half>();
                    // DataCopyParam param{1, }
                    DataCopy(in, dkTensor[h * 2048 * 128 + 64 * loopIdx], 128);
                    quein.EnQue(in);
                    auto calc = quein.DeQue<half>();
                    // AscendC::DumpTensor(calc, 0, 128);
                    auto out = queout.AllocTensor<half>();
                    AscendC::Adds(out, calc, (half)10.0, 128);
                    // AscendC::DumpTensor(out, 1, 128);
                    quein.FreeTensor(calc);
                    queout.EnQue(out);
                    auto outtensor = queout.DeQue<half>();
                    DataCopy(dkTensor[h * 2048 * 128 + 64 * loopIdx], outtensor, 128);
                    // AscendC::DumpTensor(dkTensor[h * 2048 * 128 + 64 * loopIdx], 2, 128);
                    queout.FreeTensor(outtensor);
                }
            }

        }
    }
    return;
}
