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
 * \file moe_distribute_dispatch_v2_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
#include "mc2_hcom_topo_info.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;
namespace {
    constexpr uint32_t X_INDEX = 0U;
    constexpr uint32_t EXPERT_IDS_INDEX = 1U;
    constexpr uint32_t SCALES_INDEX = 2U;
    constexpr uint32_t X_ACTIVE_MASK_INDEX = 3U;
    constexpr uint32_t EXPERT_SCALES_INDEX = 4U;
    constexpr uint32_t ELASTIC_INFO_INDEX = 5U;
    constexpr uint32_t PERFORMANCE_INFO_INDEX = 6U;
    constexpr uint32_t OUTPUT_EXPAND_X_INDEX = 0U;
    constexpr uint32_t OUTPUT_DYNAMIC_SCALES_INDEX = 1U;
    constexpr uint32_t OUTPUT_ASSIST_INFO_INDEX = 2U;
    constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3U;
    constexpr uint32_t OUTPUT_EP_RECV_COUNTS_INDEX = 4U;
    constexpr uint32_t OUTPUT_TP_RECV_COUNTS_INDEX = 5U;
    constexpr uint32_t OUTPUT_EXPAND_SCALES_INDEX = 6U;

    constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
    constexpr uint32_t ATTR_GROUP_TP_INDEX = 4;
    constexpr uint32_t ATTR_TP_WORLD_SIZE_INDEX = 5;
    constexpr uint32_t ATTR_TP_RANK_ID_INDEX = 6;
    constexpr uint32_t ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
    constexpr uint32_t ATTR_SHARED_EXPERT_NUM_INDEX = 8;
    constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
    constexpr uint32_t ATTR_QUANT_MODE_INDEX = 10;
    constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 11;
    constexpr uint32_t ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 12;
    constexpr uint32_t ATTR_COMM_ALG_INDEX = 13;
    constexpr uint32_t ATTR_ZERO_EXPERT_NUM_INDEX = 14;
    constexpr uint32_t ATTR_COPY_EXPERT_NUM_INDEX = 15;
    constexpr uint32_t ATTR_CONST_EXPERT_NUM_INDEX = 16;

    constexpr uint32_t TWO_DIMS = 2;
    constexpr uint32_t ONE_DIM = 1;
    constexpr uint32_t DYN_SCALE_DIMS = 1;
    constexpr uint32_t ASSIST_INFO_DIMS = 1;
    constexpr uint32_t DYNAMIC_SCALE_DIM_NUM = 1;
    constexpr uint64_t INIT_TILINGKEY = 10000;
    constexpr uint32_t ARR_LENGTH = 128;
    constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
    constexpr uint32_t NO_SCALES = 0;
    constexpr uint32_t STATIC_SCALES = 1;
    constexpr uint32_t DYNAMIC_SCALES = 2;
    constexpr uint32_t OP_TYPE_ALL_GATHER = 6;

    constexpr uint32_t UNQUANT_MODE = 0;
    constexpr uint32_t STATIC_QUANT_MODE = 1;
    constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
    constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;
    constexpr int64_t MAX_SHARED_EXPERT_NUM = 4;
    constexpr int64_t MAX_EP_WORLD_SIZE = 768L; // 384 * 2
    constexpr int64_t MIN_EP_WORLD_SIZE = 2;
    constexpr int64_t EP_RESTRICT_8 = 8;
    constexpr int64_t MAX_TP_WORLD_SIZE = 2;
    constexpr int64_t BS_UPPER_BOUND = 512;
    constexpr int64_t FULLMESH_BS_UPPER_BOUND = 256;

    constexpr uint64_t NUM_10 = 10ULL;
    constexpr uint32_t TILINGKEY_SCALES = 10;
    constexpr uint32_t TILINGKEY_TP_WORLD_SIZE = 100;
    constexpr uint32_t TILINGKEY_COMM_ALG = 1000;
    constexpr uint32_t TP_WORLD_SIZE_TWO = 2;
    constexpr uint32_t VERSION_2 = 2;
    constexpr uint32_t HCOMMCNT_2 = 2;
    constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
    constexpr int64_t LOCAL_EXPERT_MAX_SIZE = 2048;
    constexpr int64_t K_MAX = 16;
    constexpr int64_t FULLMESH_K_MAX = 12;
    constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
    constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512;
    constexpr uint32_t RANK_LIST_NUM = 2;
    constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024; // Bytes
    constexpr int64_t H_MIN = 1024;
    constexpr int64_t H_MAX = 8192;
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr uint64_t TRIPLE = 3;
    constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
    constexpr uint64_t FULL_MESH_DATA_ALIGN = 480UL;
    constexpr uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL; // scale32B + 3*4expandIdx
    constexpr uint64_t DOUBLE_DATA_BUFFER = 2UL;
    constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
    constexpr uint64_t UB_ALIGN = 32UL;
    constexpr int64_t ELASTIC_METAINFO_OFFSET = 4;

    // A2定义
    const char *K_INNER_DEBUG = "MoeDistributeDispatchV2 Tiling Debug";
    constexpr uint32_t RANK_NUM_PER_NODE_A2 = 8;
    constexpr uint32_t BLOCK_SIZE_A2 = 32;
    constexpr uint32_t MAX_K_VALUE_A2 = 16;
    constexpr uint32_t MAX_HIDDEN_SIZE_A2 = 7168;
    constexpr uint32_t LAYERED_MAX_HIDDEN_SIZE_A2 = 10240;
    constexpr int32_t MAX_EP_WORLD_SIZE_A2 = 256;
    constexpr int32_t MAX_EP_WORLD_SIZE_A2_LAYERED = 64;
    constexpr int32_t MAX_MOE_EXPERT_NUMS_A2 = 512;
    constexpr int32_t UNLAYERED_EXP_NUM_PER_RANK_A2 = 24;
    constexpr uint32_t MAX_BATCH_SIZE_A2 = 256;
    constexpr size_t USER_WORKSPACE_A2 = 1UL * 1024UL * 1024UL; // moeExpertNum_ * sizeof(uint32_t) + epWorldSize_ * 2 * 32
    constexpr uint64_t TILING_KEY_BASE_A2 = 2000000000;
    constexpr uint64_t TILING_KEY_LAYERED_COMM_A2 = 100000000;
    constexpr uint64_t INIT_TILINGKEY_A2 = 1000;
}

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", tilingData.moeDistributeDispatchV2Info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", tilingData.moeDistributeDispatchV2Info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", tilingData.moeDistributeDispatchV2Info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", tilingData.moeDistributeDispatchV2Info.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", tilingData.moeDistributeDispatchV2Info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", tilingData.moeDistributeDispatchV2Info.bs);
    OP_LOGD(nodeName, "k is %u.", tilingData.moeDistributeDispatchV2Info.k);
    OP_LOGD(nodeName, "h is %u.", tilingData.moeDistributeDispatchV2Info.h);
    OP_LOGD(nodeName, "aivNum is %u.", tilingData.moeDistributeDispatchV2Info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", tilingData.moeDistributeDispatchV2Info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeEp);
    OP_LOGD(nodeName, "totalWinSizeTP is %lu.", tilingData.moeDistributeDispatchV2Info.totalWinSizeTp);
    OP_LOGD(nodeName, "hasElastic is %d.", tilingData.moeDistributeDispatchV2Info.hasElasticInfo);
    OP_LOGD(nodeName, "zeroComputeExpertNum is %d", tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);
    OP_LOGD(nodeName, "cumSumUBMinValue is %d", tilingData.moeDistributeDispatchV2Info.cumSumUBMinValue);
}

static bool CheckTensorDim(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(nodeName, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
        xStorageShape->GetStorageShape().GetDimNum()), return false);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(nodeName, "expertIdShape is null."), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expertIdShape dims must be 2, but current dim num is %lu.",
        expertIdStorageShape->GetStorageShape().GetDimNum()), return false);
    const int64_t expertIdDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdDim0);
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdDim1);

    // 如果scales不为空进行shape维度检查
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(nodeName, "scalesShape is null."), return false);
        OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "scalesShape dims must be 2, but current dim num is %lu.",
            scalesStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "scales dim0 = %ld", scalesStorageShape->GetStorageShape().GetDim(0));
        OP_LOGD(nodeName, "scales dim1 = %ld", scalesStorageShape->GetStorageShape().GetDim(1));
    }

    if (isActiveMask) {
        const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskStorageShape == nullptr, OP_LOGE(nodeName, "xActiveMask shape is null."),
            return false);
        const int64_t xActiveMaskDimNum = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNum != ONE_DIM) && (xActiveMaskDimNum != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask shape dim must be 1 or 2, but current dim num is %ld.",
            xActiveMaskDimNum), return false);
        OP_TILING_CHECK((xActiveMaskStorageShape->GetStorageShape().GetDim(0) != xDim0), OP_LOGE(nodeName,
            "The input of xActiveMask dim0 = %ld is not equal to x dim0 = %ld.",
            xActiveMaskStorageShape->GetStorageShape().GetDim(0), xDim0), return false);
        OP_TILING_CHECK(((xActiveMaskDimNum == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != expertIdDim1)), OP_LOGE(nodeName,
            "The input of xActiveMask dim1 = %ld is not equal to expertId dim1 = %ld.",
            xActiveMaskStorageShape->GetStorageShape().GetDim(1), expertIdDim1),
            return false);
    }
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoStorageShape == nullptr, OP_LOGE(nodeName, "elasticInfo is null."), return false);
        OP_TILING_CHECK(elasticInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
            OP_LOGE(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
            elasticInfoStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "elasticInfo dim0 = %ld", elasticInfoStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXStorageShape == nullptr, OP_LOGE(nodeName, "expandXShape is null."), return false);
    OP_TILING_CHECK(expandXStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "expandXShape dims must be 2, but current dim num is %lu.",
        expandXStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expandX dim0 = %ld", expandXStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expandX dim1 = %ld", expandXStorageShape->GetStorageShape().GetDim(1));

    if (quantMode == DYNAMIC_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesStorageShape == nullptr,
            OP_LOGE(nodeName, "dynamicScalesShape is null."), return false);
        OP_TILING_CHECK(dynamicScalesStorageShape->GetStorageShape().GetDimNum() != DYNAMIC_SCALE_DIM_NUM,
            OP_LOGE(nodeName, "dynamicScalesShape dims must be %u, but current dim num is %lu.",
            DYNAMIC_SCALE_DIM_NUM, dynamicScalesStorageShape->GetStorageShape().GetDimNum()), return false);
        OP_LOGD(nodeName, "dynamicScales dim0 = %ld", dynamicScalesStorageShape->GetStorageShape().GetDim(0));
    }

    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoStorageShape == nullptr, OP_LOGE(nodeName, "assistInfoShape is null."), return false);
    OP_TILING_CHECK(assistInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "assistInfoShape dims must be 1, but current dim num is %lu.",
        assistInfoStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "assistInfoForCombine dim0 = %ld", assistInfoStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr,
        OP_LOGE(nodeName, "expertTokenNumsShape is null."), return false);
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "expertTokenNumsShape dims must be 1, but current dim num is %lu.",
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountStorageShape == nullptr, OP_LOGE(nodeName, "epRecvCountShape is null."), return false);
    OP_TILING_CHECK(epRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "epRecvCountShape dims must be 1, but current dim num is %lu.",
        epRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "epRecvCount dim0 = %ld", epRecvCountStorageShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountStorageShape == nullptr,
        OP_LOGE(nodeName, "tpRecvCountShape is null."), return false);
    OP_TILING_CHECK(tpRecvCountStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "tpRecvCountShape dims must be 1, but current dim num is %lu.",
        tpRecvCountStorageShape->GetStorageShape().GetDimNum()), return false);
    OP_LOGD(nodeName, "tpRecvCount dim0 = %ld", tpRecvCountStorageShape->GetStorageShape().GetDim(0));

    return true;
}

static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "expertId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(expertIdDesc->GetDataType()).c_str()), return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "scales dataType is invalid, dataType should be float, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
    }

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(nodeName,
            "xActiveMask dataType is invalid, dataType should be bool, but is %s.",
            Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str()), return false);
    }

    if (hasElasticInfo) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(elasticInfoDesc->GetDataType() != ge::DT_INT32, OP_LOGE(nodeName,
            "elasticInfoDesc dataType is invalid, dataType should be int32, but is %s.",
            Ops::Base::ToString(elasticInfoDesc->GetDataType()).c_str()), return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    if (quantMode != NO_SCALES) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, but is %s.",
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    } else {
        OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %s, but is %s.",
            Ops::Base::ToString(xDesc->GetDataType()).c_str(), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
            return false);
    }

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is %s.",
            Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str()), return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(assistInfoDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "assistInfoForCombine dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(assistInfoDesc->GetDataType()).c_str()), return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."),
        return false);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(nodeName, "expertTokenNums dataType is invalid, dataType should be int64, but is %s.",
        Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str()), return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(epRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "epRecvCounts dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(epRecvCountsDesc->GetDataType()).c_str()), return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(tpRecvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "tpRecvCounts dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(tpRecvCountsDesc->GetDataType()).c_str()), return false);

    return true;
}

static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const uint32_t hasElasticInfo)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(nodeName, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "x format is invalid."), return false);

    auto expertIdDesc = context->GetInputDesc(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(nodeName, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertId format is invalid."), return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(nodeName, "scalesDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "scales format is invalid."), return false);
    }

    if (isActiveMask) {
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr, OP_LOGE(nodeName, "xActiveMaskDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "xActiveMask format is invalid."), return false);
    }

    if (static_cast<bool>(hasElasticInfo)) {
        auto elasticInfoDesc = context->GetOptionalInputDesc(ELASTIC_INFO_INDEX);
        OP_TILING_CHECK(elasticInfoDesc == nullptr, OP_LOGE(nodeName, "elasticInfoDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(elasticInfoDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "elasticInfo format is invalid."), return false);
    }

    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    OP_TILING_CHECK(expandXDesc == nullptr, OP_LOGE(nodeName, "expandXDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expandXDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expandX format is invalid."), return false);

    if (quantMode == DYNAMIC_SCALES) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(dynamicScalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "dynamicScales format is invalid."), return false);
    }

    auto assistInfoDesc = context->GetOutputDesc(OUTPUT_ASSIST_INFO_INDEX);
    OP_TILING_CHECK(assistInfoDesc == nullptr, OP_LOGE(nodeName, "assistInfoDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(assistInfoDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "assistInfoForCombine format is invalid."), return false);

    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, OP_LOGE(nodeName, "expertTokenNumsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "expertTokenNums format is invalid."), return false);

    auto epRecvCountsDesc = context->GetOutputDesc(OUTPUT_EP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(epRecvCountsDesc == nullptr, OP_LOGE(nodeName, "epRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(epRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "epRecvCounts format is invalid."), return false);

    auto tpRecvCountsDesc = context->GetOutputDesc(OUTPUT_TP_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(tpRecvCountsDesc == nullptr, OP_LOGE(nodeName, "tpRecvCountsDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(tpRecvCountsDesc->GetStorageFormat())) ==
        ge::FORMAT_FRACTAL_NZ, OP_LOGE(nodeName, "tpRecvCounts format is invalid."), return false);

    return true;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, std::string &groupEp, std::string &groupTp, bool &isSetCommAlg)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto groupTpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_TP_RANK_ID_INDEX);
    auto expertShardPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_SHARED_EXPERT_NUM_INDEX));
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    // 判空
    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName, "groupEpPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE(nodeName, "commAlgPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr, OP_LOGE(nodeName, "tpWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr, OP_LOGE(nodeName, "tpRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertShardPtr == nullptr, OP_LOGE(nodeName, "expertShardPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr, OP_LOGE(nodeName, "sharedExpertRankNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(nodeName, "moeExpertNumPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(nodeName, "quantModePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr, OP_LOGE(nodeName, "expertTokenNumsTypePtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(nodeName, "zeroExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(nodeName, "copyExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr, OP_LOGE(nodeName, "constExpertNumPtr is null."),
        return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = *moeExpertNumPtr;
    int64_t epWorldSize = *epWorldSizePtr;
    int64_t sharedExpertRankNum = *sharedExpertRankNumPtr;
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    OP_TILING_CHECK((zeroExpertNum < 0), OP_LOGE(nodeName,
        "zeroExpertNum less than 0, zeroExpertNum is %ld.", zeroExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((copyExpertNum < 0), OP_LOGE(nodeName,
        "copyExpertNum less than 0, copyExpertNum is %ld.", copyExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((constExpertNum < 0), OP_LOGE(nodeName,
        "constExpertNum less than 0, constExpertNum is %ld.", constExpertNum), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
        constExpertNum);
    OP_TILING_CHECK(zeroComputeExpertNum + moeExpertNum > INT32_MAX,
        OP_LOGE(nodeName,
        "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
         zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
        OP_LOGE(nodeName, "epWorldSize is invalid, only support [%ld, %ld], but got epWorldSize=%ld.",
        MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*tpWorldSizePtr < 0) || (*tpWorldSizePtr > MAX_TP_WORLD_SIZE),
        OP_LOGE(nodeName, "tpWorldSize is invalid, only support [0, %ld], but got tpWorldSize=%ld.",
        MAX_TP_WORLD_SIZE, *tpWorldSizePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %ld), but got epRankId=%ld.",
        epWorldSize, *epRankIdPtr), return ge::GRAPH_FAILED);
    if (*tpWorldSizePtr > 1) {
        OP_TILING_CHECK((*tpRankIdPtr < 0) || (*tpRankIdPtr >= *tpWorldSizePtr),
            OP_LOGE(nodeName, "tpRankId is invalid, only support [0, %ld), but got tpRankId=%ld.",
            *tpWorldSizePtr, *tpRankIdPtr), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((groupTpPtr == nullptr) || (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
            (strnlen(groupTpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
            OP_LOGE(nodeName, "groupTpPtr is null."), return ge::GRAPH_FAILED);
        groupTp = std::string(groupTpPtr);
    } else {
        OP_TILING_CHECK(*tpRankIdPtr != 0,
            OP_LOGE(nodeName, "tpRankId is invalid, NoTp mode only support 0, but got tpRankId=%ld.", *tpRankIdPtr),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(*expertShardPtr != 0,
        OP_LOGE(nodeName, "expertShardType is invalid, only support 0, but got expertShardType=%ld.",
        *expertShardPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sharedExpertNumPtr < 0) || (*sharedExpertNumPtr > MAX_SHARED_EXPERT_NUM),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support [0, %ld], but got sharedExpertNum=%ld.",
        MAX_SHARED_EXPERT_NUM, *sharedExpertNumPtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((sharedExpertRankNum < 0) || (sharedExpertRankNum >= epWorldSize),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support [0, %ld), but got sharedExpertRankNum=%ld.",
        epWorldSize, sharedExpertRankNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
        OP_LOGE(nodeName, "moeExpertNum is invalid, only support (0, %ld], but got moeExpertNum=%ld.",
        MOE_EXPERT_MAX_NUM, moeExpertNum), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(NO_SCALES)) ||
        (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OP_LOGE(nodeName, "quantMode is invalid, only support [0, %u], but got quantMode=%ld.",
        DYNAMIC_SCALES, *quantModePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*expertTokenNumsTypePtr != 0) && (*expertTokenNumsTypePtr != 1),
        OP_LOGE(nodeName, "expertTokenNumsType only support 0 or 1, but got expertTokenNumsType=%ld.",
        *expertTokenNumsTypePtr), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((strlen(commAlgPtr) != 0) && (strcmp(commAlgPtr, "fullmesh_v1") != 0) && (strcmp(commAlgPtr, "fullmesh_v2") != 0),
        OP_LOGE(nodeName, "Attr commAlg is invalid, current only support fullmesh_v1 and fullmesh_v2, but got commAlg = %s.", commAlgPtr), 
        return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    isSetCommAlg = ((strcmp(commAlgPtr, "fullmesh_v2") == 0) ? true : false);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 isSetCommAlg = %d\n", isSetCommAlg);
    tilingData.moeDistributeDispatchV2Info.epWorldSize = static_cast<uint32_t>(epWorldSize);
    tilingData.moeDistributeDispatchV2Info.tpWorldSize = static_cast<uint32_t>(*tpWorldSizePtr);
    tilingData.moeDistributeDispatchV2Info.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.tpRankId = static_cast<uint32_t>(*tpRankIdPtr);
    tilingData.moeDistributeDispatchV2Info.expertShardType = static_cast<uint32_t>(*expertShardPtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum = static_cast<uint32_t>(sharedExpertRankNum);
    if (tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum == 0U) {
        if (tilingData.moeDistributeDispatchV2Info.sharedExpertNum == 1U) {
            tilingData.moeDistributeDispatchV2Info.sharedExpertNum = 0U;
        }
    }
    tilingData.moeDistributeDispatchV2Info.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.moeDistributeDispatchV2Info.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.moeDistributeDispatchV2Info.expertTokenNumsType = static_cast<uint32_t>(*expertTokenNumsTypePtr);
    tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 zeroComputeExpertNum = %d\n",
        tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum);
    uint32_t localMoeExpertNum = static_cast<uint32_t>(moeExpertNum) / (static_cast<uint32_t>(epWorldSize) - static_cast<uint32_t>(sharedExpertRankNum));
    uint32_t lastDim = localMoeExpertNum * static_cast<uint32_t>(epWorldSize);
    std::vector<int64_t> srcShapeDim = {1, lastDim};	
    auto srcShape = ge::Shape(srcShapeDim);
    uint32_t cumSumUBMaxValue = 0;
    uint32_t cumSumUBMinValue = 0;
    AscendC::GetCumSumMaxMinTmpSize(srcShape, sizeof(float), true, true, cumSumUBMaxValue, cumSumUBMinValue);
    tilingData.moeDistributeDispatchV2Info.cumSumUBMinValue = static_cast<uint32_t>(cumSumUBMinValue);    
    OP_LOGD(nodeName, "lastDim = %d, MoeDistributeDispatchV2 cumSumUBMinValue = %d\n", lastDim,
        tilingData.moeDistributeDispatchV2Info.cumSumUBMinValue);
    return ge::GRAPH_SUCCESS;
}

static bool CheckSharedAttrs(const char *nodeName,
    const MoeDistributeDispatchV2TilingData &tilingData)
{
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验共享专家卡数和共享专家数是否只有一个为0
    OP_TILING_CHECK((sharedExpertNum == 0U) && (sharedExpertRankNum > 0U),
        OP_LOGE(nodeName, "sharedExpertRankNum is invalid, only support 0 when sharedExpertNum is 0, but got %u.",
        sharedExpertRankNum), return false);
    OP_TILING_CHECK((sharedExpertNum > 0U) && (sharedExpertRankNum == 0U),
        OP_LOGE(nodeName, "sharedExpertNum is invalid, only support 0 when sharedExpertRankNum is 0, but got %u.",
        sharedExpertNum), return false);

    if ((sharedExpertNum > 0U) && (sharedExpertRankNum > 0U)) {
        // 校验共享专家卡数能否整除共享专家数
        OP_TILING_CHECK(((sharedExpertRankNum % sharedExpertNum) != 0U),
            OP_LOGE(nodeName, "sharedExpertRankNum should be divisible by sharedExpertNum, but sharedExpertRankNum=%u, "
            "sharedExpertNum=%u.", sharedExpertRankNum, sharedExpertNum), return false);
    }

    return true;
}

static bool CheckCommAlgAttrs(const gert::TilingContext *context, const char *nodeName,
    const MoeDistributeDispatchV2TilingData &tilingData, bool isSetCommAlg)
{
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    bool hasElasticInfo = tilingData.moeDistributeDispatchV2Info.hasElasticInfo;
    int32_t zeroComputeExpertNum = tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum;
    bool isExpertMask = tilingData.moeDistributeDispatchV2Info.isExpertMask;
    // 获取bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    uint32_t bs = static_cast<uint32_t>(xDim0);

    // 获取topk
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    uint32_t k = static_cast<uint32_t>(expertIdsDim1);
    OP_TILING_CHECK((isSetCommAlg && hasElasticInfo), OP_LOGE(nodeName, "Cannot support elasticInfo when comm_alg = fullmesh_v2"), 
        return false);
    // 校验特殊专家和FullMesh_v2不能同时启用
    OP_TILING_CHECK((isSetCommAlg && (zeroComputeExpertNum > 0)), OP_LOGE(nodeName, "Cannot support zeroComputeExpert when comm_alg = fullmesh_v2"), 
        return false);
    // 校验二维isExpertMask和FullMesh_v2不能同时启用
    OP_TILING_CHECK((isSetCommAlg && isExpertMask), OP_LOGE(nodeName, "Cannot support ExpertMask when comm_alg = fullmesh_v2"), 
        return false);
    // 检查comm_alg和tpWorldSize是否冲突
    OP_TILING_CHECK(isSetCommAlg && (tpWorldSize == TP_WORLD_SIZE_TWO), OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, tp_world_size cannot be 2."),
        return false);
    // 检查comm_alg和bs是否冲突
    OP_TILING_CHECK(isSetCommAlg && (bs > FULLMESH_BS_UPPER_BOUND), OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, bs should be between [1, %ld], but got %ld.", FULLMESH_BS_UPPER_BOUND, bs),
        return false);
    // 检查comm_alg和topK是否冲突
    OP_TILING_CHECK(isSetCommAlg && (k > FULLMESH_K_MAX), OP_LOGE(nodeName, "When comm_alg is fullmesh_v2, topK should be between [1, %ld], but got %ld.", FULLMESH_K_MAX, k),
        return false);
    return true;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context, const char *nodeName, 
    MoeDistributeDispatchV2TilingData &tilingData, uint32_t &localMoeExpertNum, bool isActiveMask, bool isSetCommAlg)
{
    uint32_t epWorldSize = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t tpWorldSize = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    uint32_t moeExpertNum = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    OP_TILING_CHECK(!CheckSharedAttrs(nodeName, tilingData),
        OP_LOGE(nodeName, "Check shared expert related attributes failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckCommAlgAttrs(context, nodeName, tilingData, isSetCommAlg),
        OP_LOGE(nodeName, "Check comm_alg related attributes failed."), return ge::GRAPH_FAILED);
    // 校验moe专家数量能否均分给多机
    localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
    OP_TILING_CHECK(moeExpertNum % (epWorldSize - sharedExpertRankNum) != 0,
        OP_LOGE(nodeName, "moeExpertNum should be divisible by (epWorldSize - sharedExpertRankNum), "
        "but moeExpertNum=%u, epWorldSize=%u, sharedExpertRankNum=%u.", moeExpertNum, epWorldSize, sharedExpertRankNum),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localMoeExpertNum <= 0) || (localMoeExpertNum * epWorldSize > LOCAL_EXPERT_MAX_SIZE),OP_LOGE(nodeName, "localMoeExpertNum is invalid, "
        "localMoeExpertNum * epWorldSize must be less than or equal to 2048, and localMoeExpertNum must be greater than 0, "
        "but got localMoeExpertNum * epWorldSize = %u, localMoeExpertNum = %u", localMoeExpertNum * epWorldSize, localMoeExpertNum), return ge::GRAPH_FAILED);
    // 校验tp=2时单个moe卡上专家数是否等于1
    OP_TILING_CHECK((tpWorldSize > 1) && (localMoeExpertNum > 1), OP_LOGE(nodeName, "Cannot support multi-moeExpert %u "
        "in a rank when tpWorldSize = %u > 1", localMoeExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((tpWorldSize > 1) && (tilingData.moeDistributeDispatchV2Info.hasElasticInfo), OP_LOGE(nodeName, "Cannot support elasticInfo"
        " when tpWorldSize = %u > 1", tpWorldSize), return ge::GRAPH_FAILED);
    // 校验输入x的dim 0并设bs
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK((xDim0 > BS_UPPER_BOUND) || (xDim0 <= 0),
        OP_LOGE(nodeName, "xDim0(BS) is invalid. Should be between [1, %ld], but got xDim0=%ld.", BS_UPPER_BOUND,
                xDim0), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.bs = static_cast<uint32_t>(xDim0);

    // 校验globalBS
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);
    OP_TILING_CHECK(globalBsPtr == nullptr, OP_LOGE(nodeName, "globalBsPtr is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 *globalBsPtr = %ld, bs = %ld, epWorldSize = %u\n",
        *globalBsPtr, xDim0, epWorldSize);
    OP_TILING_CHECK((*globalBsPtr != 0) && ((*globalBsPtr < xDim0 * static_cast<int64_t>(epWorldSize)) ||
        ((*globalBsPtr) % (static_cast<int64_t>(epWorldSize)) != 0)), OP_LOGE(nodeName, "globalBS is invalid, only "
        "support 0 or maxBs(maxBs is the largest bs on all ranks) * epWorldSize, but got globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isActiveMask),
        OP_LOGE(nodeName, "Different bs on different rank cannot work when isActiveMask=true, globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(((*globalBsPtr > (xDim0 * static_cast<int64_t>(epWorldSize))) && isSetCommAlg),
        OP_LOGE(nodeName, "Different bs on different rank cannot work when comm_alg = fullmesh_v2, globalBS=%ld, "
        "bs=%ld, epWorldSize=%u.", *globalBsPtr, xDim0, epWorldSize), return ge::GRAPH_FAILED);
    if (*globalBsPtr == 0) {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(xDim0) * epWorldSize;
    } else {
        tilingData.moeDistributeDispatchV2Info.globalBs = static_cast<uint32_t>(*globalBsPtr);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(const gert::TilingContext *context, const char *nodeName,
    MoeDistributeDispatchV2TilingData &tilingData, const uint32_t quantMode, const bool isScales,
    const bool isSharedExpert,const bool hasElasticInfo, const int64_t localMoeExpertNum)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = *constExpertNumPtr;

    uint32_t A = 0U;
    uint32_t globalBs = tilingData.moeDistributeDispatchV2Info.globalBs;
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;

    // 校验输入x的维度1并设h, bs已校验过
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK((xDim1 < H_MIN) || (xDim1 > H_MAX), OP_LOGE(nodeName,
        "xShape dims1(H) should be in [%ld, %ld], but got %ld.",
        H_MIN, H_MAX, xDim1), return ge::GRAPH_FAILED); // 32字节对齐
    tilingData.moeDistributeDispatchV2Info.h = static_cast<uint32_t>(xDim1);

    // 校验expert_id的维度并设k
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.moeExpertNum);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const int64_t expertIdsDim0 = expertIdStorageShape->GetStorageShape().GetDim(0);
    const int64_t expertIdsDim1 = expertIdStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(xDim0 != expertIdsDim0, OP_LOGE(nodeName, "xShape's dim0 not equal to expertIdShape's dim0, "
        "xShape's dim0 is %ld, expertIdShape's dim0 is %ld.", xDim0, expertIdsDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((expertIdsDim1 <= 0) || (expertIdsDim1 > K_MAX) || (expertIdsDim1 > moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum),
        OP_LOGE(nodeName, "expertIdShape's dim1(k) should be in (0, min(%ld, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum = %ld)], "
        "but got expertIdShape's dim1=%ld.", K_MAX, moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum, expertIdsDim1), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.k = static_cast<uint32_t>(expertIdsDim1);

    // 校验scales的维度
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
        const int64_t scalesDim0 = scalesStorageShape->GetStorageShape().GetDim(0);
        const int64_t scalesDim1 = scalesStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(scalesDim0 != (static_cast<int64_t>(sharedExpertNum) + moeExpertNum),
            OP_LOGE(nodeName, "scales's dim0 not equal to sharedExpertNum + moeExpertNum, "
            "scales's dim0 is %ld, sharedExpertNum is %ld, moeExpertNum is %ld.",
            scalesDim0, static_cast<int64_t>(sharedExpertNum), moeExpertNum), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(xDim1 != scalesDim1, OP_LOGE(nodeName, "scales's dim1 not equal to xShape's dim1, "
            "xShape's dim1 is %ld, scales's dim1 is %ld.", xDim1, scalesDim1), return ge::GRAPH_FAILED);
    }
    uint32_t rankNumPerSharedExpert = 0;
    uint32_t epWorldSizeU32 = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    uint32_t maxBs = globalBs / epWorldSizeU32;
    uint32_t maxSharedGroupNum = 0;
    if ((sharedExpertNum != 0U) && (sharedExpertRankNum != 0U)) { // 除零保护
        rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
        maxSharedGroupNum = (epWorldSizeU32 + rankNumPerSharedExpert - 1U) / rankNumPerSharedExpert;
    }
    if (isSharedExpert) { // 本卡为共享专家
        A = maxBs * maxSharedGroupNum;
    } else {     // 本卡为moe专家
        A = globalBs * std::min(localMoeExpertNum, expertIdsDim1);
    }

    // 校验elasticInfo的维度，并更新一下最大输出的值
    if (hasElasticInfo) {
        const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
        const int64_t elasticInfoDim0 = elasticInfoStorageShape->GetStorageShape().GetDim(0);
        const int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);

        OP_TILING_CHECK(elasticInfoDim0 != (ELASTIC_METAINFO_OFFSET + RANK_LIST_NUM * epWorldSize),
            OP_LOGE(nodeName, "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
            "elasticInfo's dim0 is %ld, epWorldSize is %ld.",
            elasticInfoDim0, epWorldSize), return ge::GRAPH_FAILED);
        A = std::max( static_cast<int64_t>(maxBs * maxSharedGroupNum) , globalBs * std::min(localMoeExpertNum, expertIdsDim1));
    }
    tilingData.moeDistributeDispatchV2Info.a = A;

    // 校验expandX的维度
    int64_t tpWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    const gert::StorageShape *expandXStorageShape = context->GetOutputShape(OUTPUT_EXPAND_X_INDEX);
    const int64_t expandXDim0 = expandXStorageShape->GetStorageShape().GetDim(0);
    const int64_t expandXDim1 = expandXStorageShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(expandXDim0 < tpWorldSize * static_cast<int64_t>(A), OP_LOGE(nodeName,
        "expandX's dim0 not greater than or equal to A*tpWorldSize, "
        "expandX's dim0 is %ld, A*tpWorldSize is %ld.", expandXDim0, tpWorldSize * A), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 != expandXDim1, OP_LOGE(nodeName, "expandX's dim1 not equal to xShape's dim1, "
        "xShape's dim1 is %ld, expandX's dim1 is %ld.", xDim1, expandXDim1), return ge::GRAPH_FAILED);

    // 校验dynamicScales的维度
    if (quantMode != NO_SCALES) {
        const gert::StorageShape *dynamicScalesStorageShape = context->GetOutputShape(OUTPUT_DYNAMIC_SCALES_INDEX);
        const int64_t dynamicScalesDim0 = dynamicScalesStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(dynamicScalesDim0 < static_cast<int64_t>(A) * tpWorldSize, OP_LOGE(nodeName,
            "dynamicScales's dim0 should be equal to or greater than A*tpWorldSize, dynamicScales's dim0 is %ld, "
            "A*tpWorldSize is %ld.", dynamicScalesDim0, A * tpWorldSize), return ge::GRAPH_FAILED);
    }

    // 校验assistInfo的维度
    const gert::StorageShape *assistInfoStorageShape = context->GetOutputShape(OUTPUT_ASSIST_INFO_INDEX);
    const int64_t assistInfoDim0 = assistInfoStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(assistInfoDim0 < static_cast<int64_t>(A * TRIPLE), OP_LOGE(nodeName, "assistInfoDim0 < A * 3,"
        " assistInfoDim0 is %ld, A * 3 is %ld.", assistInfoDim0, static_cast<int64_t>(A * TRIPLE)), return ge::GRAPH_FAILED);

    // 校验expertTokenNums的维度
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    if (hasElasticInfo) {
        OP_TILING_CHECK(expertTokenNumsDim0 != (localMoeExpertNum > 1 ? localMoeExpertNum : 1),  OP_LOGE(nodeName,
            "elastic scaling expertTokenNums's Dim0 not equal to max(localMoeExpertNum,1), expertTokenNumsDim0 is %ld, "
            "localMoeExpertNum is %ld.", expertTokenNumsDim0, localMoeExpertNum), return ge::GRAPH_FAILED);
    } else if (isSharedExpert) {
        OP_TILING_CHECK(expertTokenNumsDim0 != 1, OP_LOGE(nodeName, "shared expertTokenNums's dim0 %ld not equal to 1.",
            expertTokenNumsDim0), return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(expertTokenNumsDim0 != localMoeExpertNum, OP_LOGE(nodeName,
            "moe expertTokenNums's Dim0 not equal to localMoeExpertNum, expertTokenNumsDim0 is %ld, "
            "localMoeExpertNum is %ld.", expertTokenNumsDim0, localMoeExpertNum), return ge::GRAPH_FAILED);
    }

    // 校验epRecvCount和tpRecvCount的维度
    int64_t epWorldSize = static_cast<int64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    const gert::StorageShape *epRecvCountStorageShape = context->GetOutputShape(OUTPUT_EP_RECV_COUNTS_INDEX);
    const gert::StorageShape *tpRecvCountStorageShape = context->GetOutputShape(OUTPUT_TP_RECV_COUNTS_INDEX);
    const int64_t epRecvCountDim0 = epRecvCountStorageShape->GetStorageShape().GetDim(0);
    const int64_t tpRecvCountDim0 = tpRecvCountStorageShape->GetStorageShape().GetDim(0);
    int64_t epRecvCount = ((isSharedExpert) ? epWorldSize : epWorldSize * localMoeExpertNum);
    if (hasElasticInfo){
        epRecvCount = std::max(epWorldSize, epWorldSize * localMoeExpertNum);
    }
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        epRecvCount *= tpWorldSize;
    }
    OP_TILING_CHECK(epRecvCountDim0 < epRecvCount, OP_LOGE(nodeName,
        "dimension 0 of epRecvCount should be greater than or equal to epWorldSize * localMoeExpertNum * tpWorldSize, "
        "but dimension 0 of epRecvCount is %ld, epWorldSize is %ld, localMoeExpertNum is %ld, tpWorldSize is %ld.",
        epRecvCountDim0, epWorldSize, localMoeExpertNum, tpWorldSize), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tpRecvCountDim0 != tpWorldSize, OP_LOGE(nodeName,
        "dimension 0 of tpRecvCount should be equal to tpWorldSize, but dimension 0 of tpRecvCount is %ld, "
        "tpWorldSize is %ld.", tpRecvCountDim0, tpWorldSize), return ge::GRAPH_FAILED);
    
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(PERFORMANCE_INFO_INDEX);
    OP_TILING_CHECK(performanceInfoStorageShape != nullptr,
        OP_LOGE(K_INNER_DEBUG, "This input is not support, performanceInfo should be nullptr."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext *context, const char *nodeName,
    const bool isActiveMask, const bool isScales, const bool hasElasticInfo, const uint32_t quantMode)
{
    OP_TILING_CHECK(!CheckTensorDim(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo),
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo),
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckTensorFormat(context, nodeName, isScales, quantMode, isActiveMask, hasElasticInfo),
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const bool isScales, const uint32_t quantMode,
    const uint32_t tpWorldSize, const bool isSetCommAlg)
{
    uint32_t fullMesh = TILINGKEY_NO_FULLMESH;
    bool tp = false;
    uint32_t tilingKeyQuantMode = TILINGKEY_NO_QUANT;
    bool scaleMode = false;
    uint32_t layeredMode = TILINGKEY_TPL_MTE;
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        tp = true;
    }
    if (quantMode == STATIC_QUANT_MODE) {
        tilingKeyQuantMode = TILINGKEY_STATIC_QUANT;
    } else if (quantMode == DYNAMIC_QUANT_MODE) {
        tilingKeyQuantMode = TILINGKEY_DYNAMIC_QUANT;
    }
    if (isScales) {
        scaleMode = true;
    }
    if (isSetCommAlg) {
        fullMesh = TILINGKEY_ENABLE_FULLMESH;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode, 
                                            fullMesh, layeredMode, TILINGKEY_TPL_A3);
    return tilingKey;
}

static ge::graphStatus GetCclBufferSize(const char* groupStr, uint64_t* cclBufferSize, const char* nodeName)
{
    HcclComm hcclComm;
    OP_TILING_CHECK(Mc2Hcom::MC2HcomTopology::CommGetCclBufferSizeByGroup(groupStr, cclBufferSize, &hcclComm)
        != HCCL_SUCCESS, OP_LOGE(nodeName, "CommGetCclBufferSizeByGroup failed"), return ge::GRAPH_FAILED);
    if (hcclComm == nullptr) {
        OP_TILING_CHECK(Mc2Hcom::MC2HcomTopology::CommGetGroupLocalWindowSize(groupStr, cclBufferSize) != HCCL_SUCCESS,
            OP_LOGE(nodeName, "GetGroupLocalWindowSize failed"), return ge::GRAPH_FAILED);
        OP_LOGD(nodeName, "Successfully get cclBufferSize from topoInfo");
    } else {
        OP_LOGD(nodeName, "Successfully get cclBufferSize from HCCL");
    }
    OP_TILING_CHECK(*cclBufferSize == 0,
            OP_LOGE(nodeName, "Get cclBufferSize failed, cclBufferSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData *tiling,
    const std::string groupEp, const std::string groupTp, const uint32_t tpWorldSize)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupEp = %s", groupEp.c_str());
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    uint32_t opType2 = OP_TYPE_ALL_GATHER;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType1, algConfigAllToAllStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);   // 通过不拉起AICPU，提高算子退出性能
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling1 GetTiling failed"), return ge::GRAPH_FAILED);

    if (tpWorldSize > 1) {
        OP_LOGD(nodeName, "MoeDistributeDispatchV2 groupTp = %s", groupTp.c_str());
        mc2CcTilingConfig.SetGroupName(groupTp);
        mc2CcTilingConfig.SetOpType(opType2);
        mc2CcTilingConfig.SetAlgConfig(algConfigAllGatherStr);
        OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling2) != 0,
            OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling2 GetTiling failed"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeDistributeDispatchV2TilingData &tilingData,
    const char *nodeName, const bool isSetCommAlg, uint32_t &localMoeExpertNum)
{
    auto attrs = context->GetAttrs();
    uint64_t maxWindowSizeEp = 0;
    auto groupEpHccl = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    OP_TILING_CHECK(GetCclBufferSize(groupEpHccl, &maxWindowSizeEp, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get Ep HcclBufferSizeEP failed, HcclBufferSizeEP is %lu", maxWindowSizeEp),
        return ge::GRAPH_FAILED);
    uint32_t sharedExpertNum = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    uint64_t h = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.h);
    uint64_t k = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.k);
    uint64_t epWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.epWorldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.globalBs) / epWorldSize;
    // combine数据区 token首地址对齐512
    uint64_t tokenNeedSizeCombine = ((h * MAX_OUT_DTYPE_SIZE  + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // dispatch数据区 token首对齐512，有效token长度h_align_32b + scale(32b) + 三元组(3*4b)
    uint64_t tokenActualLen = ((h * MAX_OUT_DTYPE_SIZE  + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER;
    uint64_t tokenNeedSizeDispatch = 0;
    if (isSetCommAlg) {
        tokenNeedSizeDispatch = ((tokenActualLen + FULL_MESH_DATA_ALIGN - 1UL) / FULL_MESH_DATA_ALIGN) * WIN_ADDR_ALIGN;
    } else {
        tokenNeedSizeDispatch = ((tokenActualLen + WIN_ADDR_ALIGN - 1UL) / WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    }
    uint64_t actualSize = ((maxBs * tokenNeedSizeDispatch * epWorldSize * static_cast<uint64_t>(localMoeExpertNum))
        + (maxBs * tokenNeedSizeCombine * (k + static_cast<uint64_t>(sharedExpertNum)))) * DOUBLE_DATA_BUFFER;
    OP_TILING_CHECK((actualSize > maxWindowSizeEp),
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, maxBs = %lu, h = %lu, epWorldSize = %lu,"
            " localMoeExpertNum = %u, sharedExpertNum = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            " k = %lu, NEEDED_HCCL_BUFFSIZE(((maxBs * tokenNeedSizeDispatch * ep_worldsize * localMoeExpertNum) +"
            " (maxBs * tokenNeedSizeCombine * (k + sharedExpertNum))) * 2) = %luMB, HCCL_BUFFSIZE=%luMB.",
            maxBs, h, epWorldSize, localMoeExpertNum, sharedExpertNum, tokenNeedSizeDispatch, tokenNeedSizeCombine, k,
            actualSize / MB_SIZE + 1UL, maxWindowSizeEp / MB_SIZE), return ge::GRAPH_FAILED);
    tilingData.moeDistributeDispatchV2Info.totalWinSizeEp = maxWindowSizeEp;
    OP_LOGD(nodeName, "windowSize = %lu", maxWindowSizeEp);

    uint64_t tpWorldSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.tpWorldSize);
    if (tpWorldSize == TP_WORLD_SIZE_TWO) {
        uint64_t maxWindowSizeTp = 0;
        auto groupTpHccl = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_TP_INDEX));
        OP_TILING_CHECK(GetCclBufferSize(groupTpHccl, &maxWindowSizeTp, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Get Ep HcclBufferSizeTP failed, HcclBufferSizeTP is %lu", maxWindowSizeTp),
            return ge::GRAPH_FAILED);
        actualSize = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.a) * (tokenNeedSizeDispatch +
        tokenNeedSizeCombine) * DOUBLE_DATA_BUFFER;
        OP_TILING_CHECK((actualSize > maxWindowSizeTp),
        OP_LOGE(nodeName, "TP HCCL_BUFFSIZE is too SMALL, A = %u, tokenNeedSizeDispatch = %lu, tokenNeedSizeCombine = %lu,"
            "NEEDED_HCCL_BUFFSIZE(A * (tokenNeedSizeDispatch + tokenNeedSizeCombine) * 2UL) = %luMB, TP HCCL_BUFFSIZE=%luMB.",
            tilingData.moeDistributeDispatchV2Info.a,  tokenNeedSizeDispatch, tokenNeedSizeCombine, actualSize / MB_SIZE + 1UL,
            maxWindowSizeTp / MB_SIZE), return ge::GRAPH_FAILED);
        tilingData.moeDistributeDispatchV2Info.totalWinSizeTp = maxWindowSizeTp;
        OP_LOGD(nodeName, "TpwindowSize = %lu", maxWindowSizeTp);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA3TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeDispatchV2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t quantMode = NO_SCALES;
    bool isScales = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isSetCommAlg = false;
    uint32_t localMoeExpertNum = 1;
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchV2 tiling check func.");

    // 获取入参属性
    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp, groupTp, isSetCommAlg) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    // 获取scales
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    isScales = (scalesStorageShape != nullptr);

    // 获取xActiveMask
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    isActiveMask = (xActiveMaskStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.isTokenMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    tilingData->moeDistributeDispatchV2Info.isExpertMask = ((isActiveMask) &&
        (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));

    // 获取elasticInfo
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
    hasElasticInfo = (elasticInfoStorageShape != nullptr);
    tilingData->moeDistributeDispatchV2Info.hasElasticInfo = hasElasticInfo;

    quantMode = tilingData->moeDistributeDispatchV2Info.quantMode;

    // 检查quantMode和scales是否匹配
    OP_TILING_CHECK(quantMode == STATIC_SCALES, OP_LOGE(nodeName, "cannot support static quant now."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) || ((!isScales) && (quantMode == STATIC_SCALES)),
        OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
        static_cast<int32_t>(isScales), quantMode), return ge::GRAPH_FAILED);

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        TilingCheckMoeDistributeDispatch(context, nodeName, isActiveMask, isScales, hasElasticInfo, quantMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(CheckAttrs(context, nodeName, *tilingData, localMoeExpertNum, isActiveMask, isSetCommAlg) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr failed."), return ge::GRAPH_FAILED);

    uint32_t epRankId = tilingData->moeDistributeDispatchV2Info.epRankId;
    uint32_t sharedExpertNum = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    uint32_t sharedExpertRankNum = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    bool isSharedExpert = (epRankId < sharedExpertRankNum);

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData, quantMode, isScales,
        isSharedExpert, hasElasticInfo, static_cast<int64_t>(localMoeExpertNum)) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    OP_TILING_CHECK(CheckWinSize(context, *tilingData, nodeName, isSetCommAlg, localMoeExpertNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, groupEp, groupTp, tpWorldSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set hcomm cfg failed."), return ge::GRAPH_FAILED);
    uint64_t tilingKey = CalTilingKey(isScales, quantMode, tpWorldSize, isSetCommAlg);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1); // 设置为batch mode模式, 所有核同时启动
    tilingData->moeDistributeDispatchV2Info.totalUbSize = ubSize;
    tilingData->moeDistributeDispatchV2Info.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

// a2函数
static ge::graphStatus MoeDistributeDispatchA2CheckAttrAndSetTiling(const gert::TilingContext *context, MoeDistributeDispatchA2Info& info, const bool isLayered)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto tpWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_TP_WORLD_SIZE_INDEX);
    auto tpRankIdPtr = attrs->GetAttrPointer<int>(ATTR_TP_RANK_ID_INDEX);
    auto expertSharedTypePtr = attrs->GetAttrPointer<int>(ATTR_EXPERT_SHARD_TYPE_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int>(ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int>(ATTR_GLOBAL_BS_INDEX);
    auto expertTokenNumsTypePtr = attrs->GetAttrPointer<int>(ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    auto constExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_CONST_EXPERT_NUM_INDEX));

    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);
    int32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);

    OP_TILING_CHECK((groupEpPtr == nullptr) || (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupEpPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(K_INNER_DEBUG, "groupEp is invalid."), return ge::GRAPH_FAILED);
    int32_t maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2;
    if (isLayered) {
        maxEpWorldSizeA2 = MAX_EP_WORLD_SIZE_A2_LAYERED;
    }
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0 || *epWorldSizePtr > maxEpWorldSizeA2 ||
        *epWorldSizePtr % RANK_NUM_PER_NODE_A2 != 0,
        OP_LOGE(K_INNER_DEBUG, "epWorldSize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr || *epRankIdPtr < 0 || *epRankIdPtr >= *epWorldSizePtr,
        OP_LOGE(K_INNER_DEBUG, "epRankId is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr % *epWorldSizePtr != 0 ||
        *moeExpertNumPtr <= 0 || *moeExpertNumPtr > MAX_MOE_EXPERT_NUMS_A2,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(!isLayered && *moeExpertNumPtr / *epWorldSizePtr > UNLAYERED_EXP_NUM_PER_RANK_A2,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum is %d, in case of unlayered, it must no more than %d.", 
            *moeExpertNumPtr / *epWorldSizePtr, UNLAYERED_EXP_NUM_PER_RANK_A2), return GRAPH_FAILED);
    OP_TILING_CHECK(tpWorldSizePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "tpWorldSize is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(tpRankIdPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "tpRankId is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertSharedTypePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertSharedType is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(sharedExpertRankNumPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "sharedExpertRankNum is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(quantModePtr == nullptr || (*quantModePtr != UNQUANT_MODE && *quantModePtr != DYNAMIC_QUANT_MODE),
        OP_LOGE(K_INNER_DEBUG, "quantMode is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(globalBsPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "globalBs is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsTypePtr == nullptr || *expertTokenNumsTypePtr < 0 || *expertTokenNumsTypePtr > 1,
        OP_LOGE(K_INNER_DEBUG, "expertTokenNumsType is invalid. Must be 0 or 1. "), return GRAPH_FAILED);
    OP_TILING_CHECK(zeroExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "zeroExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(copyExpertNumPtr == nullptr, OP_LOGE(K_INNER_DEBUG, "copyExpertNumPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(constExpertNumPtr == nullptr || *constExpertNumPtr != 0,
        OP_LOGE(K_INNER_DEBUG, "constExpertNum is invalid. Must be 0."), return ge::GRAPH_FAILED);

    // 判断是否满足uint32_t及其他限制
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t zeroExpertNum = *zeroExpertNumPtr;
    int64_t copyExpertNum = *copyExpertNumPtr;
    int64_t constExpertNum = 0LL;
    int64_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;

    OP_LOGD(K_INNER_DEBUG, "zeroExpertNum=%ld,copyExpertNum= %ld, constExpertNum=%ld", zeroExpertNum, copyExpertNum,
        constExpertNum);
    OP_TILING_CHECK(zeroComputeExpertNum + moeExpertNum > INT32_MAX,
        OP_LOGE(K_INNER_DEBUG,
        "zeroExpertNum[%ld] + copyExpertNum[%ld] + constExpertNum[%ld] + moeExpertNum[%ld] exceed INT32_MAX.",
         zeroExpertNum, copyExpertNum, constExpertNum, moeExpertNum), return ge::GRAPH_FAILED);

    info.epWorldSize = *epWorldSizePtr;
    info.tpWorldSize = static_cast<uint32_t>(0);
    info.epRankId = *epRankIdPtr;
    info.tpRankId = static_cast<uint32_t>(0);
    info.expertSharedType = static_cast<uint32_t>(0);
    info.sharedExpertRankNum = static_cast<uint32_t>(0);
    info.moeExpertNum = *moeExpertNumPtr;
    info.quantMode = *quantModePtr;
    if (*globalBsPtr == 0) {
        info.globalBs = *epWorldSizePtr * bs;
    } else {
        info.globalBs = *globalBsPtr;
    }
    info.expertTokenNumsType = *expertTokenNumsTypePtr;
    info.zeroComputeExpertNum = static_cast<int32_t>(zeroComputeExpertNum);

    OP_LOGD(K_INNER_DEBUG, "quantMode=%d", info.quantMode);
    OP_LOGD(K_INNER_DEBUG, "globalBs=%d", info.globalBs);
    OP_LOGD(K_INNER_DEBUG, "expertTokenNumsType=%d", info.expertTokenNumsType);
    OP_LOGD(K_INNER_DEBUG, "expertSharedType=%d", info.expertSharedType);
    OP_LOGD(K_INNER_DEBUG, "sharedExpertRankNum=%d", info.sharedExpertRankNum);
    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%d", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "epWorldSize=%d", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "tpWorldSize=%d", info.tpWorldSize);
    OP_LOGD(K_INNER_DEBUG, "epRankId=%d", info.epRankId);
    OP_LOGD(K_INNER_DEBUG, "tpRankId=%d", info.tpRankId);
    OP_LOGD(K_INNER_DEBUG, "zeroComputeExpertNum=%d", info.zeroComputeExpertNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2CheckShapeAndSetTiling(const gert::TilingContext *context,
                                                                     MoeDistributeDispatchA2Info &info,
                                                                     bool isLayered)
{
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(EXPERT_IDS_INDEX);
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    const gert::StorageShape *expertScalesStorageShape = context->GetOptionalInputShape(EXPERT_SCALES_INDEX);
    const gert::StorageShape *elasticInfoStorageShape = context->GetOptionalInputShape(ELASTIC_INFO_INDEX);
    const gert::StorageShape *performanceInfoStorageShape = context->GetOptionalInputShape(PERFORMANCE_INFO_INDEX);
    const gert::StorageShape *expandScalesStorageShape = context->GetOutputShape(OUTPUT_EXPAND_SCALES_INDEX);

    OP_TILING_CHECK(xStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "xShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertIdShape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expertScalesStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertScales is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(isLayered && expandScalesStorageShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expandScales is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(elasticInfoStorageShape != nullptr,
        OP_LOGE(K_INNER_DEBUG, "current does not support elasticInfo as input"), return GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "x dims is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "expertId dims is invalid."), return GRAPH_FAILED);
    OP_LOGD(nodeName, "X dim0 = %ld", xStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "X dim1 = %ld", xStorageShape->GetStorageShape().GetDim(1));
    OP_LOGD(nodeName, "expertId dim0 = %ld", expertIdStorageShape->GetStorageShape().GetDim(0));
    OP_LOGD(nodeName, "expertId dim1 = %ld", expertIdStorageShape->GetStorageShape().GetDim(1));

    uint32_t h = xStorageShape->GetStorageShape().GetDim(1);
    uint32_t bs = expertIdStorageShape->GetStorageShape().GetDim(0);
    uint32_t k = expertIdStorageShape->GetStorageShape().GetDim(1);
    bool isScales = (scalesStorageShape != nullptr);
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    uint32_t maxHiddenSizeA2 = isLayered ? LAYERED_MAX_HIDDEN_SIZE_A2 : MAX_HIDDEN_SIZE_A2;
    OP_TILING_CHECK(h % BLOCK_SIZE_A2 != 0 || h == 0 || h > maxHiddenSizeA2,
        OP_LOGE(K_INNER_DEBUG, "hiddensize is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(bs == 0 || bs > MAX_BATCH_SIZE_A2,
        OP_LOGE(K_INNER_DEBUG, "batchsize is invalid."), return GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto zeroExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_ZERO_EXPERT_NUM_INDEX));
    auto copyExpertNumPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_COPY_EXPERT_NUM_INDEX));
    // 判断是否满足uint32_t及其他限制
    int32_t moeExpertNum = *moeExpertNumPtr;
    int32_t zeroExpertNum = static_cast<int32_t>(*zeroExpertNumPtr);
    int32_t copyExpertNum = static_cast<int32_t>(*copyExpertNumPtr);
    int32_t constExpertNum = 0;
    int32_t zeroComputeExpertNum = zeroExpertNum + copyExpertNum + constExpertNum;
    OP_TILING_CHECK(k == 0 || k > MAX_K_VALUE_A2 || k > moeExpertNum + zeroComputeExpertNum,
        OP_LOGE(K_INNER_DEBUG, "k is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(*quantModePtr == UNQUANT_MODE && isScales,
        OP_LOGE(K_INNER_DEBUG, "scales should be null when quantMode is unQuant."), return GRAPH_FAILED);

    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        const int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(((xActiveMaskDimNums != ONE_DIM) && (xActiveMaskDimNums != TWO_DIMS)),
            OP_LOGE(nodeName, "xActiveMask must be 1-dimension or 2-dimension, but got %lu dim",
            xActiveMaskDimNums), return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != static_cast<int64_t>(bs),
            OP_LOGE(nodeName, "xActiveMask's dim0 not equal to expertIds's dim0, xActiveMask's dim0 is %ld, "
            "expertIds's dim0 is %ld", xActiveMaskDim0, bs), return GRAPH_FAILED);

        OP_TILING_CHECK(((xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS) &&
            (xActiveMaskStorageShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(k))),
            OP_LOGE(nodeName, "xActiveMask's dim1 not equal to expertIds's dim1, xActiveMask's dim1 is %ld, "
            "expertIds's dim1 is %ld", xActiveMaskStorageShape->GetStorageShape().GetDim(1), k), return GRAPH_FAILED);
    }

    OP_TILING_CHECK(performanceInfoStorageShape != nullptr && performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(K_INNER_DEBUG, "When performanceInfo is not null, it needs to be one-dimensional."), return GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(performanceInfoStorageShape != nullptr && performanceInfoStorageShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(*epWorldSizePtr),
        OP_LOGE(
            K_INNER_DEBUG,
            "The Size of performanceInfo should be equal to epWorldSize when performanceInfo is not null,"
            "but performanceInfo Size is %ld and epWorldSize is %d.",
            performanceInfoStorageShape->GetStorageShape().GetDim(0), *epWorldSizePtr),
        return GRAPH_FAILED);

    info.isTokenMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == ONE_DIM));
    info.isExpertMask = ((isActiveMask) && (xActiveMaskStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS));
    info.bs = bs;
    info.k = k;
    info.h = h;

    OP_LOGD(K_INNER_DEBUG, "isTokenMask is %d", static_cast<int32_t>(info.isTokenMask));
    OP_LOGD(K_INNER_DEBUG, "isExpertMask is %d", static_cast<int32_t>(info.isExpertMask));
    OP_LOGD(K_INNER_DEBUG, "batchSize is %u", info.bs);
    OP_LOGD(K_INNER_DEBUG, "k is %u", info.k);
    OP_LOGD(K_INNER_DEBUG, "hiddenSize is %u", info.h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(const gert::TilingContext *context, MoeDistributeDispatchA2Info& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

// 为了兼容老版本，在未配置commAlg参数时，读取环境变量；
// commAlg参数当前支持"fullmesh"和"hierarchy"两种，其余使用默认fullmesh不分层方案。
static ge::graphStatus MoeDistributeDispatchA2CheckCommAlg(const gert::TilingContext *context, bool &isLayered)
{
    isLayered = false;
    auto attrs = context->GetAttrs();
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    if (commAlg == nullptr || strlen(commAlg) == 0 || strcmp(commAlg, "0") == 0) {
        OP_LOGW(K_INNER_DEBUG, "Attr commAlg is invalid, please configure fullmesh or hierarchy.");

        const char* hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
        const char* hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
        if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr &&
            strcmp(hcclIntraPcieEnable, "1") == 0 && strcmp(hcclIntraRoceEnable, "0") == 0) {
            OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE = 1 and HCCL_INTRA_ROCE_ENABLE = 0, use hierarchy algorithm.");
            isLayered = true;
        } else {
            OP_LOGD(K_INNER_DEBUG, "ENV HCCL_INTRA_PCIE_ENABLE != 1 or HCCL_INTRA_ROCE_ENABLE != 0, use default fullmesh algorithm.");
        }
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGI(K_INNER_DEBUG, "commAlg is %s", commAlg);

    if (strcmp(commAlg, "fullmesh") == 0) {
        return ge::GRAPH_SUCCESS;
    } else if (strcmp(commAlg, "hierarchy") == 0) {
        isLayered = true;
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(K_INNER_DEBUG, "commAlg is not support");
        return GRAPH_FAILED;
    }
}

static uint64_t MoeDistributeDispatchA2CalcTilingKey(const gert::TilingContext *context, const bool isLayered)
{
    bool tp = false;
    bool scaleMode = false;
    uint32_t layeredMode = TILINGKEY_TPL_MTE;

    if (isLayered) {
        layeredMode = TILINGKEY_TPL_AICPU;
    }
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    bool isScales = (scalesStorageShape != nullptr);
    if (isScales) {
        scaleMode = true;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(tp, TILINGKEY_NO_QUANT, scaleMode, 
                                            TILINGKEY_NO_FULLMESH, layeredMode, TILINGKEY_TPL_A2);
    OP_LOGD(K_INNER_DEBUG, "tilingKey=%lu", tilingKey);

    return tilingKey;
}

static ge::graphStatus MoeDistributeDispatchA2TilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchA2 tiling func.");

    // 1. tilingData
    MoeDistributeDispatchA2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    MoeDistributeDispatchA2Info& info = tilingData->moeDistributeDispatchInfo;

    bool isLayered = false;
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckCommAlg(context, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckShapeAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckAttrAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    context->SetAicpuBlockDim(mc2tiling::AICPU_BLOCK_DIM_A2);

    uint64_t tilingKey = MoeDistributeDispatchA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + USER_WORKSPACE_A2;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    std::string algConfig = isLayered ? "BatchWrite=level1:hierarchy" : "BatchWrite=level1:fullmesh";
    uint32_t opType = 18; // BatchWrite

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Leave MoeDistributeDispatchA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeDispatchV2TilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeDispatchA2TilingFuncImpl(context);
    } else {
        ret = MoeDistributeDispatchA3TilingFuncImpl(context);
    }
    return ret;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);
} // namespace optiling