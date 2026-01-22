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
  ChunkGatedDeltaRuleBwdDhuTilingData chunkGatedDeltaRuleBwdDhuTiling;
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
  constexpr uint32_t INPUT_TRI_MASK_IDX = 11;

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

  const auto dO = context->GetInputDesc(INPUT_DO_IDX);
  
  const auto cuSeqlens = context->GetInputDesc(INPUT_CU_SEQLENS_IDX);
  const auto chunkIndices = context->GetInputDesc(INPUT_CHUNK_INDICES_IDX);

  if (cuSeqlens != nullptr && chunkIndices != nullptr) {
    IS_VARIABLE_LEN = true;
  } else if (!(cuSeqlens == nullptr && chunkIndices == nullptr)) {
    OP_LOGE(context->GetNodeName(), 
    "cu_seqlens and chunkIndices must both be provided or both be omitted.");
    return false;
  }

  auto attrs = context->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "attrs is nullptr."), return false);
  const float *scalePtr = attrs->GetAttrPointer<float>(ATTR_SCALE_IDX);
  IS_SCALE = scalePtr == nullptr ? false : true;
  const uint32_t *chunkSizePtr = attrs->GetAttrPointer<uint32_t>(ATTR_CHUNK_SIZE_IDX);
  uint32_t chunkSize = chunkSizePtr == nullptr ? NUM_64 : *chunkSizePtr;
  OP_CHECK_IF(!(chunkSize == NUM_64 || chunkSize == NUM_128), 
              OP_LOGE(context->GetNodeName(), "chunk_size should be 64 or 128, but got %d.", chunkSize), 
              return false);
  if (!IS_VARIABLE_LEN) {
    chunkNum = CeilDiv(T, chunkSize); 
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
  return true;
}

ASCENDC_EXTERN_C ge::graphStatus Tiling4ChunkGDRBwdDhu(gert::TilingContext* context) {
  ChunkGatedDeltaRuleBwdDhuTiling tiling;
  OP_CHECK_IF(!tiling.Init(context), 
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "tiling init failed"), 
              return ge::GRAPH_FAILED);
  
  OP_CHECK_IF(!tiling.CheckInputShape(context), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
              "CheckInputShape Failed"), return ge::GRAPH_FAILED);

  
  chunkGatedDeltaRuleBwdDhuTiling.set_totalCoreNum(40);
  context->SetTilingKey(1);
  context->SetBlockDim(40);
  auto platformInfoPtr = context->GetPlatformInfo();
  OP_CHECK_IF(platformInfoPtr == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null!"),
              return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t* workspace = context->GetWorkspaceSizes(1);
  uint64_t usrWorkspaceSize = 0;
  workspace[0] = usrWorkspaceSize + sysWorkspaceSize;
  chunkGatedDeltaRuleBwdDhuTiling.SaveToBuffer(
    context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(chunkGatedDeltaRuleBwdDhuTiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepare4ChunkGDRBwdDhu(gert::TilingParseContext* context) {
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleBwdDhu)
.Tiling(Tiling4ChunkGDRBwdDhu)
.TilingParse<ChunkGatedDeltaRuleBwdDhuCompileInfo>(TilingPrepare4ChunkGDRBwdDhu);  // regist into the framework
}  // namespace optiling
