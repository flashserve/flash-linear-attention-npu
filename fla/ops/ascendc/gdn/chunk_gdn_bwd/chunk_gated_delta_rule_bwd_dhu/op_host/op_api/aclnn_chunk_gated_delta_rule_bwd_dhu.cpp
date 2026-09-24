/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

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

using namespace op;

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
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.dO != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dO must not be nullptr.");
    CHECK_COND(params.dv != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dv must not be nullptr.");
    CHECK_COND((params.gOptional != nullptr) != (params.gkOptional != nullptr), ACLNN_ERR_PARAM_NULLPTR,
               "Exactly one of g and gk must be provided.");
    CHECK_COND(params.gkOptional == nullptr || params.useExp2, ACLNN_ERR_PARAM_INVALID,
               "use_exp2 must be true when gk is provided.");
    CHECK_COND(params.dhOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dhOut must not be nullptr.");
    if (params.h0Optional != nullptr) {
        CHECK_COND(params.dh0Out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dh0Out must not be nullptr when h0Optional is not nullptr.");
    }
    CHECK_COND(params.dv2Out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dv2Out must not be nullptr.");
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
    CHECK_COND(qShape.GetDimNum() == CHUNK_BWD_DHU_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "q should be 4D [B, HK, T, K].");
    CHECK_COND(kShape.GetDimNum() == CHUNK_BWD_DHU_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "k should be 4D [B, HK, T, K].");
    CHECK_COND(wShape.GetDimNum() == CHUNK_BWD_DHU_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "w should be 4D [B, HV, T, K].");
    CHECK_COND(dOShape.GetDimNum() == CHUNK_BWD_DHU_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "dO should be 4D [B, HV, T, V].");
    CHECK_COND(dvShape.GetDimNum() == CHUNK_BWD_DHU_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "dv should be 4D [B, HV, T, V].");
    CHECK_COND(dv2OutShape.GetDimNum() == CHUNK_BWD_DHU_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "dv2Out should be 4D [B, HV, T, V].");
    CHECK_COND(dhOutShape.GetDimNum() == CHUNK_BWD_DHU_DH_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "dhOut should be [B, NT, HV, K, V].");
    if (params.gOptional != nullptr) {
        CHECK_COND(params.gOptional->GetViewShape().GetDimNum() == CHUNK_BWD_DHU_G_DIM_NUM,
                   ACLNN_ERR_PARAM_INVALID, "g should be 3D [B, HV, T].");
    } else {
        CHECK_COND(params.gkOptional->GetViewShape().GetDimNum() == CHUNK_BWD_DHU_QKV_DIM_NUM,
                   ACLNN_ERR_PARAM_INVALID, "gk should be 4D [B, HV, T, K].");
    }

    // cuSeqlens 与 chunkIndices 必须成对提供，
    // cuSeqlens 至少含 [0, T] 两个边界，chunkIndices 按 [seqIdx, chunkIdx] 成对扁平存储
    const bool isVarlen = params.cuSeqlensOptional != nullptr || params.chunkIndicesOptional != nullptr;
    CHECK_COND((params.cuSeqlensOptional != nullptr) == (params.chunkIndicesOptional != nullptr),
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens and chunk_indices must be provided together.");
    if (isVarlen) {
        CHECK_COND(params.cuSeqlensOptional->Size() >= CHUNK_BWD_DHU_MIN_CU_SEQLENS_SIZE,
                   ACLNN_ERR_PARAM_INVALID, "cu_seqlens should contain at least %zu elements.",
                   CHUNK_BWD_DHU_MIN_CU_SEQLENS_SIZE);
        CHECK_COND(params.chunkIndicesOptional->Size() > 0 &&
                       params.chunkIndicesOptional->Size() % CHUNK_BWD_DHU_CHUNK_INDICES_PAIR == 0,
                   ACLNN_ERR_PARAM_INVALID,
                   "chunk_indices size should be a positive multiple of %ld.",
                   CHUNK_BWD_DHU_CHUNK_INDICES_PAIR);
    }

    const int64_t B = qShape.GetDim(0);
    const int64_t HK = qShape.GetDim(1);
    const int64_t T = qShape.GetDim(2);
    const int64_t K = qShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM);
    const int64_t HV = dvShape.GetDim(1);
    const int64_t V = dvShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM);

    // q≡k 同形；dO≡dv 同形；
    // w 为 [B, HV, T, K]；q 与 dO 共享 B、T
    CHECK_COND(qShape.GetDim(0) == kShape.GetDim(0) && qShape.GetDim(1) == kShape.GetDim(1) &&
                   qShape.GetDim(2) == kShape.GetDim(2) &&
                   qShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) ==
                       kShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM),
               ACLNN_ERR_PARAM_INVALID, "q and k should have the same shape [B, HK, T, K].");
    CHECK_COND(dOShape.GetDim(0) == dvShape.GetDim(0) && dOShape.GetDim(1) == dvShape.GetDim(1) &&
                   dOShape.GetDim(2) == dvShape.GetDim(2) &&
                   dOShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) ==
                       dvShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM),
               ACLNN_ERR_PARAM_INVALID, "dO and dv should have the same shape [B, HV, T, V].");
    CHECK_COND(wShape.GetDim(0) == B && wShape.GetDim(1) == HV && wShape.GetDim(2) == T &&
                   wShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == K,
               ACLNN_ERR_PARAM_INVALID, "w should be [B, HV, T, K] aligned with q and dv.");
    CHECK_COND(dOShape.GetDim(0) == B && dOShape.GetDim(2) == T, ACLNN_ERR_PARAM_INVALID,
               "q and dO should have the same batch and seqlen.");

    // HV 可被 HK 整除
    CHECK_COND(HK > 0 && HV > 0 && HV % HK == 0, ACLNN_ERR_PARAM_INVALID,
               "GVA requires HV divisible by HK, but got HK=%ld, HV=%ld.", HK, HV);

    // K = 128，V ∈ {128, 256}，chunkSize ∈ {64, 128}
    CHECK_COND(K == CHUNK_BWD_DHU_K_HEAD_DIM, ACLNN_ERR_PARAM_INVALID,
               "K should be %ld, but got %ld.", CHUNK_BWD_DHU_K_HEAD_DIM, K);
    CHECK_COND(V == CHUNK_BWD_DHU_V_HEAD_DIM_128 || V == CHUNK_BWD_DHU_V_HEAD_DIM_256,
               ACLNN_ERR_PARAM_INVALID, "V should be %ld or %ld, but got %ld.",
               CHUNK_BWD_DHU_V_HEAD_DIM_128, CHUNK_BWD_DHU_V_HEAD_DIM_256, V);
    CHECK_COND(params.chunkSize == CHUNK_BWD_DHU_CHUNK_SIZE_64 ||
                   params.chunkSize == CHUNK_BWD_DHU_CHUNK_SIZE_128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize should be %ld or %ld, but got %ld.",
               CHUNK_BWD_DHU_CHUNK_SIZE_64, CHUNK_BWD_DHU_CHUNK_SIZE_128, params.chunkSize);

    // g 为 [B, HV, T]，gk 为 [B, HV, T, K]
    if (params.gOptional != nullptr) {
        const auto &gShape = params.gOptional->GetViewShape();
        CHECK_COND(gShape.GetDim(0) == B && gShape.GetDim(1) == HV && gShape.GetDim(2) == T,
                   ACLNN_ERR_PARAM_INVALID, "g should be [B, HV, T] aligned with dv.");
    } else {
        const auto &gkShape = params.gkOptional->GetViewShape();
        CHECK_COND(gkShape.GetDim(0) == B && gkShape.GetDim(1) == HV && gkShape.GetDim(2) == T &&
                       gkShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == K,
                   ACLNN_ERR_PARAM_INVALID, "gk should be [B, HV, T, K] with K aligned to q.");
    }

    // 变长模式仅支持 B = 1（README §4.3）
    CHECK_COND(!isVarlen || B == CHUNK_BWD_DHU_VARLEN_BATCH, ACLNN_ERR_PARAM_INVALID,
               "varlen mode requires B=%ld, but got B=%ld.", CHUNK_BWD_DHU_VARLEN_BATCH, B);

    // 输出形状（README §3.3）：NT 定长为 ceil(T/chunkSize)、变长为 len(chunkIndices)/2；dv2Out 与 dv 同形
    const int64_t numChunks =
        isVarlen ? static_cast<int64_t>(params.chunkIndicesOptional->Size() /
                                        CHUNK_BWD_DHU_CHUNK_INDICES_PAIR)
                 : (T + params.chunkSize - 1) / params.chunkSize;
    CHECK_COND(dhOutShape.GetDim(0) == B &&
                   dhOutShape.GetDim(1) == numChunks && dhOutShape.GetDim(2) == HV &&
                   dhOutShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == K && dhOutShape.GetDim(4) == V,
               ACLNN_ERR_PARAM_INVALID, "dhOut should be [B, NT, HV, K, V] with NT=%ld.", numChunks);
    CHECK_COND(dv2OutShape.GetDim(0) == B && dv2OutShape.GetDim(1) == HV &&
                   dv2OutShape.GetDim(2) == T &&
                   dv2OutShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM) == V,
               ACLNN_ERR_PARAM_INVALID, "dv2Out should be [B, HV, T, V] same as dv.");

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
        CHECK_COND(stateShape.GetDimNum() == CHUNK_BWD_DHU_STATE_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
                   "%s should be 4D [N, HV, K, V] or [N, HV, V, K].", stateNames[idx]);
        const int64_t stateK = params.stateVFirst ? stateShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM)
                                                  : stateShape.GetDim(2);
        const int64_t stateV = params.stateVFirst ? stateShape.GetDim(2)
                                                  : stateShape.GetDim(CHUNK_BWD_DHU_DIM_HEAD_DIM);
        CHECK_COND(stateShape.GetDim(0) == seqNum && stateShape.GetDim(1) == HV && stateK == K &&
                       stateV == V,
                   ACLNN_ERR_PARAM_INVALID,
                   "%s should be [N, HV, K, V] when stateVFirst=false and [N, HV, V, K] otherwise, "
                   "with N=%ld.",
                   stateNames[idx], seqNum);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkGatedDeltaRuleBwdDhuParams params)
{
    // 与 def.cpp 动态 dtype 组一致）：
    // q/k/w/dO/dv/h0/dht/dhOut/dh0Out/dv2Out 与 q 同 dtype ∈ {BF16, FP16}；g/gk ∈ {FP32, 与 q 同}
    const ge::DataType inputDtype = params.q->GetDataType();
    CHECK_COND(inputDtype == ge::DT_BF16 || inputDtype == ge::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
               "q dtype should be bfloat16 or float16.");
    CHECK_COND(params.k->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "k dtype should be same as q.");
    CHECK_COND(params.w->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "w dtype should be same as q.");
    CHECK_COND(params.dO->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "dO dtype should be same as q.");
    CHECK_COND(params.dv->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "dv dtype should be same as q.");
    CHECK_COND(params.dhOut->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "dhOut dtype should be same as q.");
    CHECK_COND(params.dv2Out->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "dv2Out dtype should be same as q.");
    // g/gk 可为 FP32（提精度）或与 q 同 dtype
    const ge::DataType gateDtype = params.gOptional != nullptr
                                       ? params.gOptional->GetDataType()
                                       : params.gkOptional->GetDataType();
    CHECK_COND(gateDtype == ge::DT_FLOAT || gateDtype == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "g/gk dtype should be float32 or same as q.");
    if (params.h0Optional != nullptr) {
        CHECK_COND(params.h0Optional->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
                   "h0 dtype should be same as q.");
        if (params.dh0Out != nullptr) {
            CHECK_COND(params.dh0Out->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
                       "dh0Out dtype should be same as q.");
        }
    } else {
        // dh0 仅在提供 h0 时写出
        CHECK_COND(params.dh0Out == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "dh0Out requires h0 to be provided.");
    }
    if (params.dhtOptional != nullptr) {
        CHECK_COND(params.dhtOptional->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
                   "dht dtype should be same as q.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.w, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous w failed.");
    CHECK_COND(DataContiguous(params.dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dO failed.");
    CHECK_COND(DataContiguous(params.dv, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dv failed.");
    CHECK_COND(OptionalDataContiguous(params.gOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");
    CHECK_COND(OptionalDataContiguous(params.gkOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous gk failed.");
    CHECK_COND(OptionalDataContiguous(params.h0Optional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous h0 failed.");
    CHECK_COND(OptionalDataContiguous(params.dhtOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous dht failed.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleBwdDhuParams params)
{
    // 空指针码透传：CheckNotNull 返回 161001（ACLNN_ERR_PARAM_NULLPTR），
    // 不可经 CHECK_RET 统一改写为 161002，否则丢失 NULLPTR 语义
    aclnnStatus ret = CheckNotNull(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
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
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");

    auto result = l0op::ChunkGatedDeltaRuleBwdDhu(
        params.q, params.k, params.w, params.dO, params.dv, params.gOptional, params.gkOptional,
        params.h0Optional, params.dhtOptional, params.cuSeqlensOptional, params.chunkIndicesOptional,
        params.scale, params.chunkSize, params.useExp2, params.stateVFirst,
        params.dhOut, params.dh0Out, params.dv2Out, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(result[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result[0], params.dhOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (params.dh0Out != nullptr) {
        viewCopyResult = l0op::ViewCopy(result[1], params.dh0Out, executorPtr);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    viewCopyResult = l0op::ViewCopy(result[2], params.dv2Out, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwdDhu);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkGatedDeltaRuleBwdDhu launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
