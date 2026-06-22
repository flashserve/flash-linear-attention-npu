/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
  constexpr uint32_t BLOCK_SIZE = 32;

  constexpr uint32_t HALF_DTYPE_SIZE = 2;
  constexpr uint32_t FP32_DTYPE_SIZE = 4;
  constexpr uint32_t INT8_DTYPE_SIZE = 1;

  constexpr uint32_t TILING_KEY = 0;
  constexpr uint32_t TILING_KEY_G_FP32 = 1;

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
  const gert::Shape kShape = context->GetInputShape(INPUT_K_IDX)->GetStorageShape();
  const gert::Shape wShape = context->GetInputShape(INPUT_W_IDX)->GetStorageShape();
  const gert::Shape doShape = context->GetInputShape(INPUT_DO_IDX)->GetStorageShape();
  const gert::Shape dvShape = context->GetInputShape(INPUT_DV_IDX)->GetStorageShape();
  B = qShape.GetDim(DIM_0);
  Hk = qShape.GetDim(DIM_1);
  T = qShape.GetDim(DIM_2);
  K = qShape.GetDim(DIM_3);
  Hv = doShape.GetDim(DIM_1);
  V = doShape.GetDim(DIM_3);

  OP_CHECK_IF(
      kShape.GetDim(DIM_0) != static_cast<int64_t>(B) || kShape.GetDim(DIM_1) != static_cast<int64_t>(Hk) ||
          kShape.GetDim(DIM_2) != static_cast<int64_t>(T) || kShape.GetDim(DIM_3) != static_cast<int64_t>(K),
      OP_LOGE(context->GetNodeName(),
              "k must match q as [B,Hk,T,K]; q [%lu,%lu,%lu,%lu], k [%ld,%ld,%ld,%ld].", B, Hk, T, K,
              kShape.GetDim(DIM_0), kShape.GetDim(DIM_1), kShape.GetDim(DIM_2), kShape.GetDim(DIM_3)),
      return false);
  OP_CHECK_IF(
      wShape.GetDim(DIM_0) != static_cast<int64_t>(B) || wShape.GetDim(DIM_1) != static_cast<int64_t>(Hv) ||
          wShape.GetDim(DIM_2) != static_cast<int64_t>(T) || wShape.GetDim(DIM_3) != static_cast<int64_t>(K),
      OP_LOGE(context->GetNodeName(),
              "w must be [B,Hv,T,K] with Hv=dO.dim1; expect [%lu,%lu,%lu,%lu], got [%ld,%ld,%ld,%ld].", B, Hv, T, K,
              wShape.GetDim(DIM_0), wShape.GetDim(DIM_1), wShape.GetDim(DIM_2), wShape.GetDim(DIM_3)),
      return false);
  OP_CHECK_IF(doShape.GetDim(DIM_0) != static_cast<int64_t>(B) || doShape.GetDim(DIM_2) != static_cast<int64_t>(T),
              OP_LOGE(context->GetNodeName(), "dO batch/time must match q."), return false);
  OP_CHECK_IF(
      dvShape.GetDim(DIM_0) != static_cast<int64_t>(B) || dvShape.GetDim(DIM_1) != static_cast<int64_t>(Hv) ||
          dvShape.GetDim(DIM_2) != static_cast<int64_t>(T) || dvShape.GetDim(DIM_3) != static_cast<int64_t>(V),
      OP_LOGE(context->GetNodeName(), "dv must be [B,Hv,T,V] aligned with dO."), return false);
  // GVA（Grouped Value Attention）：Hk 为 q/k 头数，Hv 为 value 头数（与 do/dv/w对齐）；须 Hv 是 Hk 的整数倍（见 FLA num_v_heads % num_heads）。
  OP_CHECK_IF(Hv == 0 || Hk == 0 || (Hv % Hk) != 0,
              OP_LOGE(context->GetNodeName(),
                      "GVA: Hv (value heads) must be an integer multiple of Hk (q/k heads); require Hv mod Hk == 0; got Hk=%lu Hv=%lu.",
                      Hk, Hv),
              return false);
  
  auto attrs = context->GetAttrs();
  OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "attrs is nullptr."), return false);
  const double *scalePtr = attrs->GetAttrPointer<double>(ATTR_SCALE_IDX);
  IS_SCALE = scalePtr == nullptr ? false : true;
  float scale = IS_SCALE ? *scalePtr : 1.0;
  const uint32_t *chunkSizePtr = attrs->GetAttrPointer<uint32_t>(ATTR_CHUNK_SIZE_IDX);
  chunkSize = chunkSizePtr == nullptr ? NUM_64 : *chunkSizePtr;
  OP_CHECK_IF(!(chunkSize == NUM_64 || chunkSize == NUM_128), 
              OP_LOGE(context->GetNodeName(), "chunk_size should be 64 or 128, but got %d.", chunkSize), 
              return false);
  
  tilingData.set_B(B);
  tilingData.set_Hv(Hv);
  tilingData.set_Hk(Hk);
  tilingData.set_T(T);
  tilingData.set_K(K);
  tilingData.set_V(V);
  tilingData.set_isScale(IS_SCALE);
  tilingData.set_scale(scale);
  tilingData.set_chunkSize(chunkSize);
  return true;
}

bool ChunkGatedDeltaRuleBwdDhuTiling::VarLenSetting(gert::TilingContext* context) {
  const auto cuSeqlens = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
  const auto chunkIndices = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
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
    auto chunkNum = chunkIndices->GetShapeSize() / NUM_2;
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

bool ChunkGatedDeltaRuleBwdDhuTiling::CheckInputDtype(gert::TilingContext* context) {
  const auto gDtype = context->GetOptionalInputDesc(INPUT_G_IDX)->GetDataType();
  const auto qDtype = context->GetOptionalInputDesc(INPUT_Q_IDX)->GetDataType();
  if (gDtype != qDtype && gDtype != ge::DT_FLOAT) {
    OP_LOGE(context->GetNodeName(), "gDtype must be DT_FLOAT or as same as qDtype");
    return false;
  }
  if (gDtype == ge::DT_FLOAT) {
    tilingKey = TILING_KEY_G_FP32;
  } else {
    tilingKey = TILING_KEY;
  }
  return true;  
}

bool ChunkGatedDeltaRuleBwdDhuTiling::CalcUb(gert::TilingContext *context) {
  // AIC_AIV_1_2, 每个VEC处理BT/2行
  uint32_t halfBT = CeilDiv(static_cast<uint32_t>(chunkSize), NUM_2);
  uint32_t halfK = CeilDiv(static_cast<uint32_t>(K), NUM_2);
  uint32_t gBrcbBufByte = halfBT * BLOCK_SIZE;
  uint32_t dvBufByte = halfBT * V * HALF_DTYPE_SIZE;
  uint32_t dvCastBufByte = halfBT * V * FP32_DTYPE_SIZE;
  uint32_t dqkBufByte = halfBT * K * HALF_DTYPE_SIZE;
  uint32_t dqkCastBufByte = halfBT * K * FP32_DTYPE_SIZE;
  uint32_t dhCastBufByte = halfK * V * FP32_DTYPE_SIZE;

  // 与 chunk_gated_delta_rule_bwd_dhu_vec.h InitUB 对齐（同一 vecTbuf 内绝对偏移尖峰）：
  // - dv2 链：gCast 后 dv2Offset = chunkSize*4，再接 gBrcb、vIn(fp16)、dvCast、bdvCast(fp32)
  // - gatedQ 链：gCast+gExp 共 2*chunkSize*4，offsetQ 起 qLocal(fp16)+qCast(fp32)+gBCLocal
  // - UpdateDh：分时复用 [0, 2*dhCastBufByte)，与上两链取 max
  const uint32_t dvPeak =
      chunkSize * FP32_DTYPE_SIZE + gBrcbBufByte + dvBufByte + NUM_2 * dvCastBufByte;
  const uint32_t gatedQPeak =
      NUM_2 * chunkSize * FP32_DTYPE_SIZE + dqkBufByte + dqkCastBufByte + gBrcbBufByte;
  const uint32_t dhPeak = NUM_2 * dhCastBufByte;
  uint32_t tBufByte = std::max(dhPeak, std::max(dvPeak, gatedQPeak));
  
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
  uint32_t taskNum = B * Hk * tilingData.get_seqNum();
  uint32_t usedCoreNum = taskNum > totalCoreNum ? totalCoreNum : taskNum;  
  tilingData.set_usedCoreNum(usedCoreNum);
  context->SetBlockDim(usedCoreNum);

  // GVA 方案 A：粗粒度 task 为 (b,hq,seq)，组内各 value 头 h 串行（见 cube/vec 中 for h 循环）。
  // Workspace 按物理核 coreIdx / vec 侧 cubeIdx 分片；每片仍对应「单 chunk、单条流水线」的 bdv/gatedQ/dh term，
  // 完成该 h 的一步后再处理下一 h，同一片复用，故 per-core 步长仍为 chunkSize*V、BT*K、K*V，无需再乘 (Hv/Hk)。
  // 若将来改为组内多头并行或双缓冲，需同时改 kernel 内偏移与下列 * usedCoreNum 的核算。
  uint64_t bdvWs = chunkSize * V * usedCoreNum;
  uint64_t qWs = K * chunkSize * usedCoreNum;
  uint64_t wDv2Ws = K * V * usedCoreNum;
  uint64_t qDoWs = K * V * usedCoreNum;
  size_t usrWsSize = static_cast<size_t>((bdvWs + qWs + wDv2Ws + qDoWs) * HALF_DTYPE_SIZE);

  size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
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
  OP_LOGD(context->GetNodeName(), "Hv is %lu.", tilingData.get_Hv());
  OP_LOGD(context->GetNodeName(), "Hk is %lu.", tilingData.get_Hk());
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

  OP_CHECK_IF(!tiling.CheckInputDtype(context), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
              "CheckInputDtype Failed"), return ge::GRAPH_FAILED);
  
  OP_CHECK_IF(!tiling.CalcUb(context), OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
            "Set Ub Failed"), return ge::GRAPH_FAILED);
  context->SetTilingKey(tiling.tilingKey);
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
