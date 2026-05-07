/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_kernel.h
 * \brief
 */
#ifndef ASCEND_OPS_GROUPED_MATMUL_KERNEL_H
#define ASCEND_OPS_GROUPED_MATMUL_KERNEL_H


#include "grouped_matmul.h"


using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

namespace {
constexpr CubeFormat wFormat = CubeFormat::ND;
}

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;


#define GMM_CUBE_IMP(processClass, transA, transB, sync, cfg)                                                          \
    do {                                                                                                               \
        if ASCEND_IS_AIV {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        using matmulType = MMImplType<xType<transA>, weightType<transB>, yType, biasType, cfg>;                        \
        matmulType::MT mm;                                                                                             \
        mm.SetSubBlockIdx(0);                                                                                          \
        mm.Init(&tilingData->mmTilingData, &tPipe);                                                                     \
        GMMCompute<matmulType, sync> computeOp(mm);                                                                    \
        computeOp.Init(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y,   \
                       user1, &tilingData->gmmBaseParams, &tilingData->mmTilingData, &tPipe);                            \
        processClass<decltype(computeOp)> op(computeOp);                                                               \
        op.Init(&tilingData->gmmBaseParams, &tilingData->mmTilingData, (int32_t *)&tilingData->gmmArray.mList[0],           \
                groupList);                                                                                            \
        op.Process();                                                                                                  \
    } while (0)

template <int D_T_A, int D_T_B, int D_T_Y, int TRANS_A, int TRANS_B, int GROUP_LIST_TYPE, int IS_STATIC_TILING_API,
          int A8W4_KERNEL_TEMPLATE, int A16W8_KERNEL_TEMPLATE, int AIV_AIC_RATIO, bool IS_ENABLE_FIXED_AXIS>
__aicore__ inline void GroupedMatmulKernelImpl(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                               GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
                                               GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                               GroupedMatmulTilingData *tilingData)
{
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workspace);
    GMM_CUBE_IMP(GMMProcess, false, false, false, matmulCFGUnitFlag);
}

#endif