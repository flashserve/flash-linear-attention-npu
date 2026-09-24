/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chunk_fwd_o.h"
#include "chunk_fwd_o.h"
#include <dlfcn.h>
#include <cstring>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"


using namespace op;

static constexpr size_t CHUNK_FWD_O_QKV_DIM_NUM = 4;
static constexpr size_t CHUNK_FWD_O_H_DIM_NUM = 5;
static constexpr size_t CHUNK_FWD_O_G_DIM_NUM = 3;
static constexpr size_t CHUNK_FWD_O_DIM_HEAD_DIM = 3;
static constexpr int64_t CHUNK_FWD_O_K_HEAD_DIM = 128;
static constexpr int64_t CHUNK_FWD_O_MAX_V_HEAD_DIM = 256;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkFwdOParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *h = nullptr;
    const aclTensor *g = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkOffsetsOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    bool useExp2 = false;
    bool stateVFirst = false;
    const char *outputLayout = nullptr;
    const aclTensor *oOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkFwdOParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.h != nullptr, ACLNN_ERR_PARAM_NULLPTR, "h must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.oOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "oOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkFwdOParams params)
{
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkFwdOParams params)
{
    const auto &qShape = params.q->GetViewShape();
    const auto &kShape = params.k->GetViewShape();
    const auto &vShape = params.v->GetViewShape();
    const auto &hShape = params.h->GetViewShape();
    const auto &gShape = params.g->GetViewShape();
    const auto &oShape = params.oOut->GetViewShape();

    // 维度数：q/k/v 4D、h 5D、g 3D（README §3.1；oOut 维度数按 outputLayout 在下方判定）
    CHECK_COND(qShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "q should be 4D [B, HK, T, K].");
    CHECK_COND(kShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "k should be 4D [B, HK, T, K].");
    CHECK_COND(vShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "v should be 4D [B, HV, T, V].");
    CHECK_COND(hShape.GetDimNum() == CHUNK_FWD_O_H_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "h should be 5D [B, numChunks, HV, K, V].");
    CHECK_COND(gShape.GetDimNum() == CHUNK_FWD_O_G_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "g should be 3D [B, HV, T].");

    // 特征维契约（README §4.2 额外限制）：K = 128，V ∈ {128, 256}
    CHECK_COND(qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) == CHUNK_FWD_O_K_HEAD_DIM, ACLNN_ERR_PARAM_INVALID,
               "K should be %ld, but got %ld.", CHUNK_FWD_O_K_HEAD_DIM,
               qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM));
    CHECK_COND(kShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) == CHUNK_FWD_O_K_HEAD_DIM, ACLNN_ERR_PARAM_INVALID,
               "K should be %ld, but got %ld.", CHUNK_FWD_O_K_HEAD_DIM,
               kShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM));
    const int64_t vDim = vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM);
    CHECK_COND(vDim == 128 || vDim == CHUNK_FWD_O_MAX_V_HEAD_DIM, ACLNN_ERR_PARAM_INVALID,
               "V should be 128 or %ld, but got %ld.", CHUNK_FWD_O_MAX_V_HEAD_DIM, vDim);

    // 跨张量一致性（README §3.4/§4.2）：q≡k 同形；q/v 的 B、T 一致；g 与 v 的 B/HV/T 对齐
    CHECK_COND(qShape.GetDim(0) == kShape.GetDim(0) && qShape.GetDim(1) == kShape.GetDim(1) &&
                   qShape.GetDim(2) == kShape.GetDim(2) && qShape.GetDim(3) == kShape.GetDim(3),
               ACLNN_ERR_PARAM_INVALID, "q and k should have the same shape [B, HK, T, K].");
    CHECK_COND(qShape.GetDim(0) == vShape.GetDim(0) && qShape.GetDim(2) == vShape.GetDim(2),
               ACLNN_ERR_PARAM_INVALID, "q and v should have the same batch and seqlen.");
    CHECK_COND(vShape.GetDim(0) == gShape.GetDim(0) && vShape.GetDim(1) == gShape.GetDim(1) &&
                   vShape.GetDim(2) == gShape.GetDim(2),
               ACLNN_ERR_PARAM_INVALID, "g should be [B, HV, T] aligned with v.");

    // h 对齐：B/HV 与 v 一致；K/V 末两维按 stateVFirst 交换（stateVFirst=true 时为 [V, K]，
    // 见 README §3.2 与 tiling ShapeCheck 一致）；numChunks 维不校验（varlen 下由 chunkOffsets 决定）
    const int64_t hK = hShape.GetDim(params.stateVFirst ? 4 : 3);
    const int64_t hV = hShape.GetDim(params.stateVFirst ? 3 : 4);
    CHECK_COND(hShape.GetDim(0) == vShape.GetDim(0) && hShape.GetDim(2) == vShape.GetDim(1) &&
                   hK == qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) && hV == vDim,
               ACLNN_ERR_PARAM_INVALID, "Check h shape failed for state_v_first=%d.", params.stateVFirst);

    // oOut 按 outputLayout 逐维校验（README §3.3/§4.2；tiling 层无 oOut 信息）
    const char *layout = params.outputLayout == nullptr ? "BNSD" : params.outputLayout;
    if (std::strcmp(layout, "BNSD") == 0) {
        CHECK_COND(oShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
                   "oOut should be 4D [B, HV, T, V] under BNSD.");
        CHECK_COND(oShape.GetDim(0) == vShape.GetDim(0) && oShape.GetDim(1) == vShape.GetDim(1) &&
                       oShape.GetDim(2) == vShape.GetDim(2) && oShape.GetDim(3) == vDim,
                   ACLNN_ERR_PARAM_INVALID, "oOut should be [B, HV, T, V] same as v under BNSD.");
    } else if (std::strcmp(layout, "BSND") == 0) {
        CHECK_COND(oShape.GetDimNum() == CHUNK_FWD_O_QKV_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
                   "oOut should be 4D [B, T, HV, V] under BSND.");
        CHECK_COND(oShape.GetDim(0) == vShape.GetDim(0) && oShape.GetDim(1) == vShape.GetDim(2) &&
                       oShape.GetDim(2) == vShape.GetDim(1) && oShape.GetDim(3) == vDim,
                   ACLNN_ERR_PARAM_INVALID, "oOut should be [B, T, HV, V] under BSND.");
    } else if (std::strcmp(layout, "TND") == 0) {
        CHECK_COND(oShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID,
                   "oOut should be 3D [T, HV, V] under TND.");
        CHECK_COND(oShape.GetDim(0) == vShape.GetDim(2) && oShape.GetDim(1) == vShape.GetDim(1) &&
                       oShape.GetDim(2) == vDim,
                   ACLNN_ERR_PARAM_INVALID, "oOut should be [T, HV, V] under TND.");
    } else if (std::strcmp(layout, "NTD") == 0) {
        CHECK_COND(oShape.GetDimNum() == 3, ACLNN_ERR_PARAM_INVALID,
                   "oOut should be 3D [HV, T, V] under NTD.");
        CHECK_COND(oShape.GetDim(0) == vShape.GetDim(1) && oShape.GetDim(1) == vShape.GetDim(2) &&
                       oShape.GetDim(2) == vDim,
                   ACLNN_ERR_PARAM_INVALID, "oOut should be [HV, T, V] under NTD.");
    }

    // chunkSize 契约（README §3.2/§4.2：仅支持 64/128）
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize should be 64 or 128, but got %ld.", params.chunkSize);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkFwdOParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous q failed.");
    CHECK_COND(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous k failed.");
    CHECK_COND(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous v failed.");
    CHECK_COND(DataContiguous(params.h, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous h failed.");
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkFwdOParams params)
{
    // dtype 契约（README §3.1/§3.3，与 def.cpp 动态 dtype 组一致）：
    // q/k/v/h/oOut ∈ {BF16, FP16} 且与 q 同 dtype；g ∈ {FP32, BF16, FP16} 独立白名单
    const ge::DataType qDtype = params.q->GetDataType();
    CHECK_COND(qDtype == ge::DT_BF16 || qDtype == ge::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID,
               "q dtype should be BF16 or FP16.");
    CHECK_COND(params.k->GetDataType() == qDtype, ACLNN_ERR_PARAM_INVALID, "k dtype should be same as q.");
    CHECK_COND(params.v->GetDataType() == qDtype, ACLNN_ERR_PARAM_INVALID, "v dtype should be same as q.");
    CHECK_COND(params.h->GetDataType() == qDtype, ACLNN_ERR_PARAM_INVALID, "h dtype should be same as q.");
    CHECK_COND(params.oOut->GetDataType() == qDtype, ACLNN_ERR_PARAM_INVALID,
               "oOut dtype should be same as q.");
    // g 组合约束（与 def.cpp 动态 dtype 组一致）：g 可为 FP32（提精度）或与 q 同 dtype；
    // q=BF16 → g∈{FP32, BF16}，q=FP16 → g∈{FP32, FP16}
    const ge::DataType gDtype = params.g->GetDataType();
    CHECK_COND(gDtype == ge::DT_FLOAT || gDtype == qDtype, ACLNN_ERR_PARAM_INVALID,
               "g dtype should be FP32 or same as q.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkFwdOParams params)
{
    // 空指针码透传：CheckNotNull 返回 161001（ACLNN_ERR_PARAM_NULLPTR），
    // 不可经 CHECK_RET 统一改写为 161002，否则丢失 NULLPTR 语义
    aclnnStatus ret = CheckNotNull(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool stateVFirst,
    const char *outputLayout,
    const aclTensor *oOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkFwdOParams params{q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional, scale, chunkSize,
                           useExp2, stateVFirst, outputLayout, oOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkFwdO,
                   DFX_IN(q, k, v, h, g, cuSeqlensOptional, chunkOffsetsOptional, scale, chunkSize,
                          useExp2, stateVFirst, outputLayout),
                   DFX_OUT(oOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查（透传 CheckParams 的错误码，保留 NULLPTR 161001 语义）
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::ChunkFwdO(params.q, params.k, params.v, params.h, params.g, params.cuSeqlensOptional,
                                 params.chunkOffsetsOptional, params.scale, params.chunkSize, params.useExp2,
                                 params.stateVFirst, params.outputLayout, params.oOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.oOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkFwdO(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkFwdO);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkFwdO launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
