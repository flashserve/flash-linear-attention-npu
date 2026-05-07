/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tiling_common.h
 * \brief
 */
#ifndef ASCEND_OPS_GROUPED_MATMUL_TILING_COMMON_H
#define ASCEND_OPS_GROUPED_MATMUL_TILING_COMMON_H

#include <cstdio>
#include "ascendc/host_api/tiling/template_argument.h"
#include "lib/matmul/bmm_tiling.h"
#include "platform/platform_ascendc.h"

#include "op_kernel/grouped_matmul_tiling.h"
#include "grouped_matmul_torch.h"

namespace GroupedMatmulNs {
constexpr int32_t BEST_BASEN = 256;
constexpr int64_t SPLIT_K = 2L;

static int64_t maxM_ = 0L;
static int64_t maxN_ = 0L;
static int64_t maxK_ = 0L;
static int32_t minK_ = INT32_MAX;
static int32_t baseM_ = 0;
static int32_t baseN_ = 0;
static int32_t baseK_ = 0;
static uint32_t usedCoreNum_ = 0;
static uint32_t groupNum_ = 0;
class GroupedMatmulTiling {
public:
    template <typename TensorListType, typename OptionalTensorListType, typename OptionalTensorType>
    static void
    GroupedMatmulCommonTiling(const TensorListType &x, const TensorListType &weight, const OptionalTensorListType &bias,
                              const OptionalTensorListType &scale, const OptionalTensorListType &offset,
                              const OptionalTensorListType &antiquantScale,
                              const OptionalTensorListType &antiquantOffset, const OptionalTensorType &groupList,
                              const OptionalTensorListType &perTokenScale, GroupedMatmulTilingData &tilingData,
                              uint32_t coreNum, uint64_t ubSize)
    {
        auto shapeGroupList = getGroupListShape(groupList);
        groupNum_ = shapeGroupList[0];

        auto shapeX = getTensorListFirstShape(x);
        maxM_ = shapeX[0];
        maxK_ = shapeX[1];

        auto shapeWeight = getTensorListFirstShape(weight);
        maxN_ = shapeWeight[1];

        Init(tilingData);
        RunFusionKernelTiling(tilingData, coreNum, ubSize);
    }

    static void Init(GroupedMatmulTilingData &tilingData)
    {
        // 当前只支持以下组合
        tilingData.gmmBaseParams.set_singleWeight(1);
        tilingData.gmmBaseParams.set_singleX(1);
        tilingData.gmmBaseParams.set_singleY(1);

        tilingData.gmmBaseParams.set_groupNum(groupNum_); // groupList shape
        tilingData.gmmBaseParams.set_groupType(0);        // 当前只支持0
        tilingData.gmmBaseParams.set_groupListType(1);    // 当前只支持1
        tilingData.gmmBaseParams.set_m(maxM_);
        tilingData.gmmBaseParams.set_k(maxK_);
        tilingData.gmmBaseParams.set_n(maxN_);


        tilingData.gmmArray.kList[0] = maxK_;
        tilingData.gmmArray.nList[0] = maxN_;
    }

    static void InitPlatformInfo(matmul_tiling::PlatformInfo &platformInfo)
    {
        platformInfo.socVersion = platform_ascendc::SocVersion::ASCEND910B;
        platformInfo.l1Size = 512 * 1024;
        platformInfo.l0CSize = 128 * 1024;
        platformInfo.ubSize = 192 * 1024;
        platformInfo.l0ASize = 64 * 1024;
        platformInfo.l0BSize = 64 * 1024;
    }


    static bool GMMSetMMTiling(GroupedMatmulTilingData &tilingData, uint64_t ubSize_)
    {
        matmul_tiling::DataType matmulDtype = matmul_tiling::DataType::DT_BFLOAT16;
        matmul_tiling::PlatformInfo platformInfo;
        InitPlatformInfo(platformInfo);
        matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);
        int64_t mInMM = maxM_; // if msd, m in matmul should mul steps
        mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
        mm.SetOrgShape(mInMM, maxN_, maxK_);
        mm.SetShape(mInMM, baseN_, maxK_);
        mm.SetFixSplit(baseM_, baseN_, baseK_);
        mm.SetBufferSpace(platformInfo.l1Size, platformInfo.l0CSize, ubSize_);
        if (mm.GetTiling(tilingData.mmTilingData) == -1) {
            std::cout << "GetTiling failed" << std::endl;
            return false;
        }
        uint32_t mmStepKa = 4;
        uint32_t mmStepKb = 4;

        constexpr uint32_t stepM = 1; // 1: stepM set fixed value 1
        constexpr uint32_t stepN = 1; // 1: stepN set fixed value 1
        uint32_t mmDepthA1 = 8;
        uint32_t mmDepthB1 = 8;
        tilingData.mmTilingData.shareMode = 0;
        tilingData.mmTilingData.dbL0C = 1;           // disable double buffer for LOC
        tilingData.mmTilingData.baseM = baseM_;      // set precomputed baseM
        tilingData.mmTilingData.baseN = baseN_;      // set precomputed baseN
        tilingData.mmTilingData.baseK = baseK_;      // set precomputed baseK
        tilingData.mmTilingData.stepKa = mmStepKa;   // set precomputed mmStepKa
        tilingData.mmTilingData.depthA1 = mmDepthA1; // set precomputed mmDepthA1
        tilingData.mmTilingData.stepKb = mmStepKb;   // set precomputed mmStepKb
        tilingData.mmTilingData.depthB1 = mmDepthB1; // set precomputed mmDepthB1
        tilingData.mmTilingData.stepM = stepM;       // set precomputed stepM
        tilingData.mmTilingData.stepN = stepN;       // set precomputed stepN
        return true;
    }

    static void RunFusionKernelTiling(GroupedMatmulTilingData &tilingData, uint32_t coreNum, uint64_t ubSize)
    {
        usedCoreNum_ = coreNum;
        if (!CalMMTiling()) {
            std::cout << "GMM CalMMTiling failed" << std::endl;
            return;
        }

        if (!GMMSetMMTiling(tilingData, ubSize)) {
            std::cout << "GMM GMMSetMMTiling failed" << std::endl;
            return;
        }
        tilingData.gmmBaseParams.singleN = 0; // 0 is the default value
        tilingData.gmmBaseParams.workspaceSize = 16 * 1024 * 1024;
        tilingData.mmTilingData.usedCoreNum = usedCoreNum_; // usedCoreNum is ai_core num
        tilingData.gmmBaseParams.coreNum = usedCoreNum_;    // ai cube number
    }

    static bool CalMMTiling()
    {
        baseM_ = 128;
        baseM_ = baseM_ > maxM_ ? SixteenAlign(maxM_, true) : SixteenAlign(static_cast<uint32_t>(baseM_));
        baseN_ = 256;
        baseK_ = 64;
        return true;
    }
};
} // namespace GroupedMatmulNs
#endif
