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
 * \file grouped_matmul_tiling.cpp
 * \brief
 */
#include "chunk_gated_delta_rule_bwd_dhu_tiling.h"

using namespace Ops::Transformer::OpTiling;
using namespace ge;
using namespace AscendC;

namespace optiling {
namespace {
  constexpr uint32_t INPUT_Q_IDX = 0;
  constexpr uint32_t INPUT_K_IDX = 1;
  constexpr uint32_t INPUT_W_IDX = 2;
  constexpr uint32_t INPUT_DO_IDX = 3;
  constexpr uint32_t INPUT_DV_IDX = 4;
  constexpr uint32_t INPUT_G_IDX = 5;
  constexpr uint32_t INPUT_GK_IDX = 6;
  constexpr uint32_t INPUT_H0_IDX = 7;
  constexpr uint32_t INPUT_DHT_IDX = 8;
  constexpr uint32_t INPUT_CU_SEQLENS_IDX = 9;
  constexpr uint32_t INPUT_CHUNK_INDICES_IDX = 10;

  constexpr uint32_t OUTPUT_DH_IDX = 0;
  constexpr uint32_t OUTPUT_DH0_IDX = 1;
  constexpr uint32_t OUTPUT_DV2_IDX = 2;
  
  constexpr uint32_t ATTR_SCALE_IDX = 0;
  constexpr uint32_t ATTR_CHUNK_SIZE_IDX = 1;

  constexpr uint32_t DIM_0 = 0;
  constexpr uint32_t DIM_1 = 1;
  constexpr uint32_t DIM_2 = 2;
  constexpr uint32_t DIM_3 = 3;
  constexpr uint32_t NUM_64 = 64;
  constexpr uint32_t NUM_128 = 128;
  constexpr uint32_t NUM_2 = 2;
  constexpr uint32_t NUM_3 = 3;

  constexpr uint32_t HALF_DTYPE_SIZE = 2;
  constexpr uint32_t FP32_DTYPE_SIZE = 4;
  constexpr uint32_t INT8_DTYPE_SIZE = 1;

  template <typename T> 
  static T CeilDiv(T a, T b) {
    if (b == 0) {
      return a;
    }
    return (a + b - 1) / b;
  }
}

bool ChunkGatedDeltaRuleBwdDhuTiling::Init(gert::TilingContext* context) {

  const gert::Shape qShape = context->GetInputShape(INPUT_Q_IDX)->GetStorageShape();
  const gert::Shape doShape = context->GetInputShape(INPUT_DO_IDX)->GetStorageShape();
  B = qShape.GetDim(DIM_0);
  H = qShape.GetDim(DIM_1);
  T = qShape.GetDim(DIM_2);
  K = qShape.GetDim(DIM_3);
  V = doShape.GetDim(DIM_3);
  
  auto attrs = context->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "attrs is nullptr."), return false);
  const float *scalePtr = attrs->GetAttrPointer<float>(ATTR_SCALE_IDX);
  IS_SCALE = scalePtr == nullptr ? false : true;
  float scale = IS_SCALE ? *scalePtr : 1.0;
  const uint32_t *chunkSizePtr = attrs->GetAttrPointer<uint32_t>(ATTR_CHUNK_SIZE_IDX);
  uint32_t chunkSize = chunkSizePtr == nullptr ? NUM_64 : *chunkSizePtr;
  OP_CHECK_IF(!(chunkSize == NUM_64 || chunkSize == NUM_128), 
              OP_LOGE(context->GetNodeName(), "chunk_size should be 64 or 128, but got %d.", chunkSize), 
              return false);

  tilingData.set_B(B);
  tilingData.set_H(H);
  tilingData.set_T(T);
  tilingData.set_K(K);
  tilingData.set_V(V);
  tilingData.set_isScale(IS_SCALE);
  tilingData.set_scale(scale);
  tilingData.set_chunkSize(chunkSize);
  return true;
}

bool ChunkGatedDeltaRuleBwdDhuTiling::VarLenSetting(gert::TilingContext* context) {
  const auto cuSeqlens = context->GetInputTensor(INPUT_CU_SEQLENS_IDX);

  const auto chunkIndices = context->GetInputTensor(INPUT_CHUNK_INDICES_IDX);

  if (cuSeqlens != nullptr && chunkIndices != nullptr) {
    IS_VARIABLE_LEN = true;
  } else if (!(cuSeqlens == nullptr && chunkIndices == nullptr)) {
    OP_LOGE(context->GetNodeName(), 
    "cu_seqlens and chunkIndices must both be provided or both be omitted.");
    return false;
  }
  tilingData.set_isVarLen(IS_VARIABLE_LEN);

  if (!IS_VARIABLE_LEN) {
    int64_t chunkNum = static_cast<int64_t>(CeilDiv(T, chunkSize)); 
    tilingData.set_chunkNum(chunkNum);
    tilingData.set_seqNum(1);
  } else {
    auto seqNum = cuSeqlens->GetShapeSize() - 1;
    auto chunkNum = chunkIndices->GetShapeSize() / 2;
    tilingData.set_seqNum(seqNum);
    tilingData.set_chunkNum(chunkNum);
  }
  return true;
}


bool ChunkGatedDeltaRuleBwdDhuTiling::CheckInputShape(gert::TilingContext* context) {
  OP_CHECK_IF(IS_VARIABLE_LEN && B != 1, 
              OP_LOGE(context->GetNodeName(), 
              "B must be 1 when seqence is variable len, but got %u.", B), return false);
  return true;  
}

bool ChunkGatedDeltaRuleBwdDhuTiling::CalcUb(gert::TilingContext *context) {
  // AIC_AIV_1_2, 每个VEC处理BT/2行
  uint32_t halfBT = CeilDiv(static_cast<uint32_t>(chunkSize), NUM_2);
  uint32_t halfK = CeilDiv(static_cast<uint32_t>(K), NUM_2);

  uint32_t gBufByte = halfBT * HALF_DTYPE_SIZE;
  uint32_t gCastBufByte = halfBT * FP32_DTYPE_SIZE;
  uint32_t dvoBufByte = halfBT * V * HALF_DTYPE_SIZE;
  uint32_t dvoCastBufByte = halfBT * V * FP32_DTYPE_SIZE;
  uint32_t dqkBufByte = halfBT * K * HALF_DTYPE_SIZE;
  uint32_t dqkCastBufByte = halfBT * K * FP32_DTYPE_SIZE;
  uint32_t dhBufByte = halfK * V * HALF_DTYPE_SIZE;
  uint32_t dhCastBufByte = halfK * V * FP32_DTYPE_SIZE;

  uint32_t tBufByte = NUM_3 * halfK * V * FP32_DTYPE_SIZE;// 96/192k
  auto platformInfoPtr = context->GetPlatformInfo();
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  uint64_t maxUbSize = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);

  OP_CHECK_IF(tBufByte > maxUbSize, OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
              "K/V is too large, K should less than 128 and V should less than 256."), 
              return false);
  tilingData.set_gBufSize(halfBT);
  tilingData.set_dvBufSize(halfBT * V);
  tilingData.set_qBufSize(halfBT * K);
  tilingData.set_dhBufSize(halfK * V);
  tilingData.set_totalTbufByte(tBufByte);
  return true;
}

void ChunkGatedDeltaRuleBwdDhuTiling::SetWorkspaceSize(gert::TilingContext* context) {
  auto platformInfoPtr = context->GetPlatformInfo();
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAic();
  uint32_t taskNum = B * H * tilingData.get_seqNum(); // 每个核处理一个batch的一个头的完整sequence
  uint32_t usedCoreNum = taskNum > totalCoreNum ? totalCoreNum : taskNum;  
  tilingData.set_usedCoreNum(usedCoreNum);
  context->SetBlockDim(usedCoreNum);

  uint64_t bdvWs = chunkSize * V * usedCoreNum * HALF_DTYPE_SIZE;
  uint64_t qWs = K * V * usedCoreNum * HALF_DTYPE_SIZE;
  uint64_t wDv2Ws = K * V * usedCoreNum * HALF_DTYPE_SIZE;
  uint64_t qDoWs = K * V * usedCoreNum * HALF_DTYPE_SIZE;
  uint64_t usrWsSize = bdvWs + qWs + wDv2Ws + qDoWs;

  uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t* workspace = context->GetWorkspaceSizes(1);
  workspace[0] = usrWsSize + sysWorkspaceSize;
  tilingData.set_bdvWs(bdvWs);
  tilingData.set_qWs(qWs);
  tilingData.set_wDv2Ws(wDv2Ws);
  tilingData.set_qDoWs(qDoWs);
}

void ChunkGatedDeltaRuleBwdDhuTiling::PrintTilingData(gert::TilingContext *context) {
  OP_LOGD(context->GetNodeName(), "End Run ChunkGatedDeltaRuleBwdDhu Tiling");
  OP_LOGD(context->GetNodeName(), "B is %lu.", tilingData.get_B());
  OP_LOGD(context->GetNodeName(), "H is %lu.", tilingData.get_H());
  OP_LOGD(context->GetNodeName(), "T is %lu.", tilingData.get_T());
  OP_LOGD(context->GetNodeName(), "K is %lu.", tilingData.get_K());
  OP_LOGD(context->GetNodeName(), "V is %lu.", tilingData.get_V());
  OP_LOGD(context->GetNodeName(), "chunkSize is %lu.", tilingData.get_chunkSize());
  OP_LOGD(context->GetNodeName(), "chunkNum is %lu.", tilingData.get_chunkNum());
  OP_LOGD(context->GetNodeName(), "seqNum is %lu.", tilingData.get_seqNum());
  OP_LOGD(context->GetNodeName(), "gBufSize is %lu.", tilingData.get_gBufSize());
  OP_LOGD(context->GetNodeName(), "dvBufSize is %lu.", tilingData.get_dvBufSize());
  OP_LOGD(context->GetNodeName(), "qBufSize is %lu.", tilingData.get_qBufSize());
  OP_LOGD(context->GetNodeName(), "dhBufSize is %lu.", tilingData.get_dhBufSize());
  OP_LOGD(context->GetNodeName(), "totalTbufByte is %lu.", tilingData.get_totalTbufByte());
  OP_LOGD(context->GetNodeName(), "bdvWs is %lu.", tilingData.get_bdvWs());
  OP_LOGD(context->GetNodeName(), "qWs is %lu.", tilingData.get_qWs());
  OP_LOGD(context->GetNodeName(), "wDv2Ws is %lu.", tilingData.get_wDv2Ws());
  OP_LOGD(context->GetNodeName(), "qDoWs is %lu.", tilingData.get_qDoWs());
  OP_LOGD(context->GetNodeName(), "isVarLen is %lu.", tilingData.get_isVarLen());
  OP_LOGD(context->GetNodeName(), "isScale is %lu.", tilingData.get_isScale());
  OP_LOGD(context->GetNodeName(), "usedCoreNum is %u.", tilingData.get_usedCoreNum());
  OP_LOGD(context->GetNodeName(), "scale is %f.", tilingData.get_scale());
}


ASCENDC_EXTERN_C ge::graphStatus Tiling4ChunkGDRBwdDhu(gert::TilingContext* context) {
  ChunkGatedDeltaRuleBwdDhuTiling tiling;
  OP_CHECK_IF(!tiling.Init(context), 
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling init failed"), 
              return ge::GRAPH_FAILED);
  
  OP_CHECK_IF(!tiling.VarLenSetting(context), 
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling init failed"), 
            return ge::GRAPH_FAILED);
  
  OP_CHECK_IF(!tiling.CheckInputShape(context), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
              "CheckInputShape Failed"), return ge::GRAPH_FAILED);

  OP_CHECK_IF(!tiling.CalcUb(context), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Set Ub Failed"), return ge::GRAPH_FAILED);
  context->SetTilingKey(1);
  auto platformInfoPtr = context->GetPlatformInfo();
  OP_CHECK_IF(platformInfoPtr == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
              return ge::GRAPH_FAILED);
  tiling.SetWorkspaceSize(context);
  tiling.tilingData.SaveToBuffer(
    context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.tilingData.GetDataSize());
  tiling.PrintTilingData(context);
  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4ChunkGDRBwdDhu(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleBwdDhu)
.Tiling(Tiling4ChunkGDRBwdDhu)
.TilingParse<ChunkGatedDeltaRuleBwdDhuCompileInfo>(TilingPrepare4ChunkGDRBwdDhu);  // regist into the framework
}  // namespace optiling
