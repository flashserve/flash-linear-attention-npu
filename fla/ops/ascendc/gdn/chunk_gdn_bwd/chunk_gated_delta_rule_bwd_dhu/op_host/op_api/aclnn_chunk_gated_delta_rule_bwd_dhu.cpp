/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
#include "aclnn_chunk_gated_delta_rule_bwd_dhu.h"
#include "chunk_gated_delta_rule_bwd_dhu.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include <string>

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkGatedDeltaRuleBwdDhu";

static constexpr size_t CHUNK_BWD_DHU_QKV_DIM_NUM = 4;
static constexpr size_t CHUNK_BWD_DHU_G_DIM_NUM = 3;
static constexpr size_t CHUNK_BWD_DHU_STATE_DIM_NUM = 4;
static constexpr size_t CHUNK_BWD_DHU_DH_DIM_NUM = 5;
static constexpr size_t CHUNK_BWD_DHU_DIM_HEAD_DIM = 3;
static constexpr size_t CHUNK_BWD_DHU_MIN_CU_SEQLENS_SIZE = 2;
static constexpr int64_t CHUNK_BWD_DHU_K_HEAD_DIM = 128;
static constexpr int64_t CHUNK_BWD_DHU_V_HEAD_DIM_128 = 128;
static constexpr int64_t CHUNK_BWD_DHU_V_HEAD_DIM_256 = 256;
static constexpr int64_t CHUNK_BWD_DHU_CHUNK_SIZE_64 = 64;
static constexpr int64_t CHUNK_BWD_DHU_CHUNK_SIZE_128 = 128;
static constexpr int64_t CHUNK_BWD_DHU_VARLEN_BATCH = 1;
static constexpr int64_t CHUNK_BWD_DHU_CHUNK_INDICES_PAIR = 2;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkGatedDeltaRuleBwdDhuParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *dO = nullptr;
    const aclTensor *dv = nullptr;
    const aclTensor *gOptional = nullptr;
    const aclTensor *gkOptional = nullptr;
    const aclTensor *h0Optional = nullptr;
    const aclTensor *dhtOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    bool useExp2 = false;
    bool stateVFirst = false;
    const aclTensor *dhOut = nullptr;
    const aclTensor *dh0Out = nullptr;
    const aclTensor *dv2Out = nullptr;
};

static aclnnStatus CheckNotNull(ChunkGatedDeltaRuleBwdDhuParams params)
{
    if (params.q == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.k == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.w == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "w");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dO == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dO");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.dv == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dv");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if ((params.gOptional != nullptr) == (params.gkOptional != nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "g,gk", "exactly one of g and gk must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gkOptional != nullptr && !params.useExp2) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "gk,use_exp2", "gk present, use_exp2=false", "use_exp2 must be true when gk is provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dhOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dhOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.h0Optional != nullptr) {
        if (params.dh0Out == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dh0Out");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    }
    if (params.dv2Out == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dv2Out");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkGatedDeltaRuleBwdDhuParams params)
{
    (void)params;
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkGatedDeltaRuleBwdDhuParams params)
{
    const auto &qShape = params.q->GetViewShape();
    const auto &kShape = params.k->GetViewShape();
    const auto &wShape = params.w->GetViewShape();
    const auto &dOShape = params.dO->GetViewShape();
    const auto &dvShape = params.dv->GetViewShape();
    const auto &dhOutShape = params.dhOut->GetViewShape();
    const auto &dv2OutShape = params.dv2Out->GetViewShape();

    // q/k/w/dO/dv/dv2Out 4D、g 3D、gk 4D、dhOut 5D
    if (qShape.GetDimNum() != CHUNK_BWD_DHU_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "q", std::to_string(qShape.GetDimNum()), "q should be 4D [B, HK, T, K].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kShape.GetDimNum() != CHUNK_BWD_DHU_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "k", std::to_string(kShape.GetDimNum()), "k should be 4D [B, HK, T, K].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (wShape.GetDimNum() != CHUNK_BWD_DHU_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "w", std::to_string(wShape.GetDimNum()), "w should be 4D [B, HV, T, K].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dOShape.GetDimNum() != CHUNK_BWD_DHU_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "dO", std::to_string(dOShape.GetDimNum()), "dO should be 4D [B, HV, T, V].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dvShape.GetDimNum() != CHUNK_BWD_DHU_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "dv", std::to_string(dvShape.GetDimNum()), "dv should be 4D [B, HV, T, V].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dv2OutShape.GetDimNum() != CHUNK_BWD_DHU_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "dv2Out", std::to_string(dv2OutShape.GetDimNum()), "dv2Out should be 4D [B, HV, T, V].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dhOutShape.GetDimNum() != CHUNK_BWD_DHU_DH_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "dhOut", std::to_string(dhOutShape.GetDimNum()),
            "dhOut should be [B, NT, HV, K, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gOptional != nullptr) {
        if (params.gOptional->GetViewShape().GetDimNum() != CHUNK_BWD_DHU_G_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "g", std::to_string(params.gOptional->GetViewShape().GetDimNum()),
                "g should be 3D [B, HV, T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (params.gkOptional->GetViewShape().GetDimNum() != CHUNK_BWD_DHU_QKV_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "gk", std::to_string(params.gkOptional->GetViewShape().GetDimNum()),
                "gk should be 4D [B, HV, T, K]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // cuSeqlens 与 chunkIndices 必须成对提供，
    // cuSeqlens 至少含 [0, T] 两个边界，chunkIndices 按 [seqIdx, chunkIdx] 成对扁平存储
    const bool isVarlen = params.cuSeqlensOptional != nullptr || params.chunkIndicesOptional != nullptr;
    if ((params.cuSeqlensOptional != nullptr) != (params.chunkIndicesOptional != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cuSeqlens,chunkIndices",
            std::string(params.cuSeqlensOptional != nullptr ? "cuSeqlens=present" : "cuSeqlens=absent") + "," +
                (params.chunkIndicesOptional != nullptr ? "chunkIndices=present" : "chunkIndices=absent"),
            "cu_seqlens and chunk_indices must be provided together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isVarlen) {
        if (params.cuSeqlensOptional->Size() < CHUNK_BWD_DHU_MIN_CU_SEQLENS_SIZE) {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                OP_NAME, "cuSeqlens", std::to_string(params.cuSeqlensOptional->Size()),
                ">= " + std::to_string(CHUNK_BWD_DHU_MIN_CU_SEQLENS_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(params.chunkIndicesOptional->Size() > 0 &&
              params.chunkIndicesOptional->Size() % CHUNK_BWD_DHU_CHUNK_INDICES_PAIR == 0)) {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                OP_NAME, "chunkIndices", std::to_string(params.chunkIndicesOptional->Size()),
                "positive multiple of " + std::to_string(CHUNK_BWD_DHU_CHUNK_INDICES_PAIR));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const int64_t B = qShape.GetDim(0);
    const int64_t HK = qShape.GetDim(1);
    const int64_t T = qShape.GetDim(2);
    const int64_t K = qShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM);
    const int64_t HV = dvShape.GetDim(1);
    const int64_t V = dvShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM);

    // q≡k 同形；dO≡dv 同形；
    // w 为 [B, HV, T, K]；q 与 dO 共享 B、T
    if (!(qShape.GetDim(0) == kShape.GetDim(0) && qShape.GetDim(1) == kShape.GetDim(1) &&
          qShape.GetDim(2) == kShape.GetDim(2) &&
          qShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == kShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q,k", "shape mismatch", "q and k should have the same shape [B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(dOShape.GetDim(0) == dvShape.GetDim(0) && dOShape.GetDim(1) == dvShape.GetDim(1) &&
          dOShape.GetDim(2) == dvShape.GetDim(2) &&
          dOShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == dvShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "dO,dv", "shape mismatch", "dO and dv should have the same shape [B, HV, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(wShape.GetDim(0) == B && wShape.GetDim(1) == HV && wShape.GetDim(2) == T &&
          wShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == K)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "w", std::string(op::ToString(wShape).GetString()),
            "w should be [B, HV, T, K] aligned with q and dv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(dOShape.GetDim(0) == B && dOShape.GetDim(2) == T)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "dO.B,dO.T", std::to_string(dOShape.GetDim(0)) + "," + std::to_string(dOShape.GetDim(2)),
            "q and dO should have the same batch B=" + std::to_string(B) + " and seqlen T=" + std::to_string(T));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // HV 可被 HK 整除
    if (!(HK > 0 && HV > 0 && HV % HK == 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "HK,HV", std::to_string(HK) + "," + std::to_string(HV),
            "GVA requires HV divisible by HK");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // K = 128，V ∈ {128, 256}，chunkSize ∈ {64, 128}
    if (K != CHUNK_BWD_DHU_K_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "K", std::to_string(K), "K should be " + std::to_string(CHUNK_BWD_DHU_K_HEAD_DIM));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(V == CHUNK_BWD_DHU_V_HEAD_DIM_128 || V == CHUNK_BWD_DHU_V_HEAD_DIM_256)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "V", std::to_string(V),
            "V should be " + std::to_string(CHUNK_BWD_DHU_V_HEAD_DIM_128) + " or " +
                std::to_string(CHUNK_BWD_DHU_V_HEAD_DIM_256));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.chunkSize == CHUNK_BWD_DHU_CHUNK_SIZE_64 ||
          params.chunkSize == CHUNK_BWD_DHU_CHUNK_SIZE_128)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunkSize", std::to_string(params.chunkSize),
            "chunkSize should be " + std::to_string(CHUNK_BWD_DHU_CHUNK_SIZE_64) + " or " +
                std::to_string(CHUNK_BWD_DHU_CHUNK_SIZE_128));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // g 为 [B, HV, T]，gk 为 [B, HV, T, K]
    if (params.gOptional != nullptr) {
        const auto &gShape = params.gOptional->GetViewShape();
        if (!(gShape.GetDim(0) == B && gShape.GetDim(1) == HV && gShape.GetDim(2) == T)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "g", std::string(op::ToString(gShape).GetString()),
                "g should be [B, HV, T] aligned with dv");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        const auto &gkShape = params.gkOptional->GetViewShape();
        if (!(gkShape.GetDim(0) == B && gkShape.GetDim(1) == HV && gkShape.GetDim(2) == T &&
              gkShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == K)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "gk", std::string(op::ToString(gkShape).GetString()),
                "gk should be [B, HV, T, K] with K aligned to q");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // 变长模式仅支持 B = 1（README §4.3）
    if (isVarlen && B != CHUNK_BWD_DHU_VARLEN_BATCH) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "B", std::to_string(B),
            "varlen mode requires B=" + std::to_string(CHUNK_BWD_DHU_VARLEN_BATCH));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 输出形状（README §3.3）：NT 定长为 ceil(T/chunkSize)、变长为 len(chunkIndices)/2；dv2Out 与 dv 同形
    const int64_t numChunks =
        isVarlen ? static_cast<int64_t>(params.chunkIndicesOptional->Size() /
                                        CHUNK_BWD_DHU_CHUNK_INDICES_PAIR)
                 : (T + params.chunkSize - 1) / params.chunkSize;
    if (!(dhOutShape.GetDim(0) == B && dhOutShape.GetDim(1) == numChunks &&
          dhOutShape.GetDim(2) == HV && dhOutShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == K &&
          dhOutShape.GetDim(4) == V)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "dhOut", std::string(op::ToString(dhOutShape).GetString()),
            "[B, NT, HV, K, V] with NT=" + std::to_string(numChunks));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(dv2OutShape.GetDim(0) == B && dv2OutShape.GetDim(1) == HV && dv2OutShape.GetDim(2) == T &&
          dv2OutShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == V)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "dv2Out", std::string(op::ToString(dv2OutShape).GetString()),
            "dv2Out should be [B, HV, T, V] same as dv");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 状态张量（README §3.1/§3.3/§3.4）：h0/dht/dh0Out 的状态布局由 stateVFirst 决定，
    // 为 [N, HV, K, V]（false）或 [N, HV, V, K]（true）；N 定长为 B、变长为 len(cu_seqlens)-1
    const int64_t seqNum =
        isVarlen ? static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1 : B;
    const aclTensor *states[] = {params.h0Optional, params.dhtOptional, params.dh0Out};
    const char *stateNames[] = {"h0", "dht", "dh0Out"};
    for (size_t idx = 0; idx < sizeof(states) / sizeof(states[0]); ++idx) {
        if (states[idx] == nullptr) {
            continue;
        }
        const auto &stateShape = states[idx]->GetViewShape();
        if (stateShape.GetDimNum() != CHUNK_BWD_DHU_STATE_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, stateNames[idx], std::to_string(stateShape.GetDimNum()),
                std::string(stateNames[idx]) + " should be 4D [N, HV, K, V] or [N, HV, V, K]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t stateK = params.stateVFirst ? stateShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM)
                                                  : stateShape.GetDim(2);
        const int64_t stateV = params.stateVFirst ? stateShape.GetDim(2)
                                                  : stateShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM);
        if (!(stateShape.GetDim(0) == seqNum && stateShape.GetDim(1) == HV && stateK == K && stateV == V)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, stateNames[idx], std::string(op::ToString(stateShape).GetString()),
                std::string(stateNames[idx]) +
                    " should be [N, HV, K, V] when stateVFirst=false and [N, HV, V, K] otherwise, with N=" +
                    std::to_string(seqNum));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkGatedDeltaRuleBwdDhuParams params)
{
    // 与 def.cpp 动态 dtype 组一致）：
    // q/k/w/dO/dv/h0/dht/dhOut/dh0Out/dv2Out 与 q 同 dtype ∈ {BF16, FP16}；g/gk ∈ {FP32, 与 q 同}
    const ge::DataType inputDtype = params.q->GetDataType();
    if (!(inputDtype == ge::DT_BF16 || inputDtype == ge::DT_FLOAT16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "q", ge::TypeUtils::DataTypeToSerialString(inputDtype),
            "q dtype should be bfloat16 or float16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.k->GetDataType() != inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "k", ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.w->GetDataType() != inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "w", ge::TypeUtils::DataTypeToSerialString(params.w->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dO->GetDataType() != inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "dO", ge::TypeUtils::DataTypeToSerialString(params.dO->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dv->GetDataType() != inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "dv", ge::TypeUtils::DataTypeToSerialString(params.dv->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dhOut->GetDataType() != inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "dhOut", ge::TypeUtils::DataTypeToSerialString(params.dhOut->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dv2Out->GetDataType() != inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "dv2Out", ge::TypeUtils::DataTypeToSerialString(params.dv2Out->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // g/gk 可为 FP32（提精度）或与 q 同 dtype
    const ge::DataType gateDtype = params.gOptional != nullptr
                                       ? params.gOptional->GetDataType()
                                       : params.gkOptional->GetDataType();
    if (!(gateDtype == ge::DT_FLOAT || gateDtype == inputDtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "g/gk", ge::TypeUtils::DataTypeToSerialString(gateDtype),
            "g/gk dtype should be float32 or the same as q: " +
                ge::TypeUtils::DataTypeToSerialString(inputDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.h0Optional != nullptr) {
        if (params.h0Optional->GetDataType() != inputDtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "h0", ge::TypeUtils::DataTypeToSerialString(params.h0Optional->GetDataType()),
                "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.dh0Out != nullptr) {
            if (params.dh0Out->GetDataType() != inputDtype) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    OP_NAME, "dh0Out", ge::TypeUtils::DataTypeToSerialString(params.dh0Out->GetDataType()),
                    "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    } else {
        // dh0 仅在提供 h0 时写出
        if (params.dh0Out != nullptr) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "dh0Out", "present", "dh0Out requires h0 to be provided");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dhtOptional != nullptr) {
        if (params.dhtOptional->GetDataType() != inputDtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "dht", ge::TypeUtils::DataTypeToSerialString(params.dhtOptional->GetDataType()),
                "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(inputDtype));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus OptionalDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    return DataContiguous(tensor, executor);
}

static aclnnStatus ParamsDataContiguous(ChunkGatedDeltaRuleBwdDhuParams &params, aclOpExecutor *executorPtr)
{
    if (!(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous q failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous k failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.w, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous w failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.dO, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous dO failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.dv, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous dv failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(OptionalDataContiguous(params.gOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous g failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(OptionalDataContiguous(params.gkOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous gk failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(OptionalDataContiguous(params.h0Optional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous h0 failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(OptionalDataContiguous(params.dhtOptional, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous dht failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleBwdDhuParams params)
{
    // 空指针码透传：CheckNotNull 返回 161001（ACLNN_ERR_PARAM_NULLPTR），
    // 不可经 CHECK_RET 统一改写为 161002，否则丢失 NULLPTR 语义
    aclnnStatus ret = CheckNotNull(params);
    if (!(ret == ACLNN_SUCCESS)) {
        return ret;
    }
    {
        const aclnnStatus checkStatus = CheckFormat(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckDtype(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckShape(params);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *dO,
    const aclTensor *dv,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *h0Optional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *dhOut,
    const aclTensor *dh0Out,
    const aclTensor *dv2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkGatedDeltaRuleBwdDhuParams params{
        q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional,
        cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, useExp2, stateVFirst,
        dhOut, dh0Out, dv2Out};
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleBwdDhu,
                   DFX_IN(q, k, w, dO, dv, gOptional, gkOptional, h0Optional, dhtOptional,
                          cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, useExp2, stateVFirst),
                   DFX_OUT(dhOut, dh0Out, dv2Out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();

    // 固定写法，参数检查
    auto ret = CheckParams(params);
    if (!(ret == ACLNN_SUCCESS)) {
        return ret;
    }
    if (!(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto result = l0op::ChunkGatedDeltaRuleBwdDhu(
        params.q, params.k, params.w, params.dO, params.dv, params.gOptional, params.gkOptional,
        params.h0Optional, params.dhtOptional, params.cuSeqlensOptional, params.chunkIndicesOptional,
        params.scale, params.chunkSize, params.useExp2, params.stateVFirst,
        params.dhOut, params.dh0Out, params.dv2Out, executorPtr);
    if (!(result[0] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(result[1] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(result[2] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dhOut, executorPtr);
    if (!(viewCopyResult != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (params.dh0Out != nullptr) {
        viewCopyResult = l0op::ViewCopy(result[1], params.dh0Out, executorPtr);
        if (!(viewCopyResult != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    viewCopyResult = l0op::ViewCopy(result[2], params.dv2Out, executorPtr);
    if (!(viewCopyResult != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwdDhu);
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "This is an error in ChunkGatedDeltaRuleBwdDhu launch aicore.", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
