/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
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
#include <string>


using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkFwdO";

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
    if (params.q == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.k == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "k");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.v == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "v");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.h == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "h");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.g == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "g");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.oOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "oOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
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
    if (qShape.GetDimNum() != CHUNK_FWD_O_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "q", std::to_string(qShape.GetDimNum()),
            "q should be 4D [B, HK, T, K].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kShape.GetDimNum() != CHUNK_FWD_O_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "k", std::to_string(kShape.GetDimNum()),
            "k should be 4D [B, HK, T, K].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (vShape.GetDimNum() != CHUNK_FWD_O_QKV_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "v", std::to_string(vShape.GetDimNum()),
            "v should be 4D [B, HV, T, V].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hShape.GetDimNum() != CHUNK_FWD_O_H_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "h", std::to_string(hShape.GetDimNum()),
            "h should be 5D [B, numChunks, HV, K, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gShape.GetDimNum() != CHUNK_FWD_O_G_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "g", std::to_string(gShape.GetDimNum()), "g should be 3D [B, HV, T]");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 特征维契约（README §4.2 额外限制）：K = 128，V ∈ {128, 256}
    if (qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) != CHUNK_FWD_O_K_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "q.K", std::to_string(qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM)),
            "K should be " + std::to_string(CHUNK_FWD_O_K_HEAD_DIM));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (kShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) != CHUNK_FWD_O_K_HEAD_DIM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "k.K", std::to_string(kShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM)),
            "K should be " + std::to_string(CHUNK_FWD_O_K_HEAD_DIM));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t vDim = vShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM);
    if (!(vDim == 128 || vDim == CHUNK_FWD_O_MAX_V_HEAD_DIM)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "v.V", std::to_string(vDim),
            "V should be 128 or " + std::to_string(CHUNK_FWD_O_MAX_V_HEAD_DIM));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 跨张量一致性（README §3.4/§4.2）：q≡k 同形；q/v 的 B、T 一致；g 与 v 的 B/HV/T 对齐
    if (!(qShape.GetDim(0) == kShape.GetDim(0) && qShape.GetDim(1) == kShape.GetDim(1) &&
          qShape.GetDim(2) == kShape.GetDim(2) && qShape.GetDim(3) == kShape.GetDim(3))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q,k",
            "q=[" + std::to_string(qShape.GetDim(0)) + "," + std::to_string(qShape.GetDim(1)) + "," +
                std::to_string(qShape.GetDim(2)) + "," + std::to_string(qShape.GetDim(3)) + "]",
            "q and k should have the same shape [B, HK, T, K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(qShape.GetDim(0) == vShape.GetDim(0) && qShape.GetDim(2) == vShape.GetDim(2))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "q.B,q.T,v.B,v.T",
            std::to_string(qShape.GetDim(0)) + "," + std::to_string(qShape.GetDim(2)) + "," +
                std::to_string(vShape.GetDim(0)) + "," + std::to_string(vShape.GetDim(2)),
            "q and v should have the same batch and seqlen");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(vShape.GetDim(0) == gShape.GetDim(0) && vShape.GetDim(1) == gShape.GetDim(1) &&
          vShape.GetDim(2) == gShape.GetDim(2))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "g", "[" + std::to_string(gShape.GetDim(0)) + "," + std::to_string(gShape.GetDim(1)) +
                "," + std::to_string(gShape.GetDim(2)) + "]",
            "g should be [B, HV, T] aligned with v");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // h 对齐：B/HV 与 v 一致；K/V 末两维按 stateVFirst 交换（stateVFirst=true 时为 [V, K]，
    // 见 README §3.2 与 tiling ShapeCheck 一致）；numChunks 维不校验（varlen 下由 chunkOffsets 决定）
    const int64_t hK = hShape.GetDim(params.stateVFirst ? 4 : 3);
    const int64_t hV = hShape.GetDim(params.stateVFirst ? 3 : 4);
    if (!(hShape.GetDim(0) == vShape.GetDim(0) && hShape.GetDim(2) == vShape.GetDim(1) &&
          hK == qShape.GetDim(CHUNK_FWD_O_DIM_HEAD_DIM) && hV == vDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "h", "K=" + std::to_string(hK) + ",V=" + std::to_string(hV),
            "h must align B/HV with v and K/V with q/v for stateVFirst=" +
                std::to_string(params.stateVFirst ? 1 : 0));
        return ACLNN_ERR_PARAM_INVALID;
    }

    // oOut 按 outputLayout 逐维校验（README §3.3/§4.2；tiling 层无 oOut 信息）
    const char *layout = params.outputLayout == nullptr ? "BNSD" : params.outputLayout;
    if (std::strcmp(layout, "BNSD") == 0) {
        if (oShape.GetDimNum() != CHUNK_FWD_O_QKV_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "oOut", std::to_string(oShape.GetDimNum()),
                "oOut should be 4D [B, HV, T, V] under BNSD");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(oShape.GetDim(0) == vShape.GetDim(0) && oShape.GetDim(1) == vShape.GetDim(1) &&
              oShape.GetDim(2) == vShape.GetDim(2) && oShape.GetDim(3) == vDim)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "oOut", "rank=" + std::to_string(oShape.GetDimNum()),
                "oOut should be [B, HV, T, V] same as v under BNSD");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (std::strcmp(layout, "BSND") == 0) {
        if (oShape.GetDimNum() != CHUNK_FWD_O_QKV_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "oOut", std::to_string(oShape.GetDimNum()),
                "oOut should be 4D [B, T, HV, V] under BSND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(oShape.GetDim(0) == vShape.GetDim(0) && oShape.GetDim(1) == vShape.GetDim(2) &&
              oShape.GetDim(2) == vShape.GetDim(1) && oShape.GetDim(3) == vDim)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "oOut", "rank=" + std::to_string(oShape.GetDimNum()),
                "oOut should be [B, T, HV, V] under BSND");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (std::strcmp(layout, "TND") == 0) {
        if (oShape.GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "oOut", std::to_string(oShape.GetDimNum()),
                "oOut should be 3D [T, HV, V] under TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(oShape.GetDim(0) == vShape.GetDim(2) && oShape.GetDim(1) == vShape.GetDim(1) &&
              oShape.GetDim(2) == vDim)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "oOut", "rank=" + std::to_string(oShape.GetDimNum()),
                "oOut should be [T, HV, V] under TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (std::strcmp(layout, "NTD") == 0) {
        if (oShape.GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, "oOut", std::to_string(oShape.GetDimNum()),
                "oOut should be 3D [HV, T, V] under NTD");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(oShape.GetDim(0) == vShape.GetDim(1) && oShape.GetDim(1) == vShape.GetDim(2) &&
              oShape.GetDim(2) == vDim)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "oOut", "rank=" + std::to_string(oShape.GetDimNum()),
                "oOut should be [HV, T, V] under NTD");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    // chunkSize 契约（README §3.2/§4.2：仅支持 64/128）
    if (!(params.chunkSize == 64 || params.chunkSize == 128)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunkSize", std::to_string(params.chunkSize), "chunkSize should be 64 or 128");
        return ACLNN_ERR_PARAM_INVALID;
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

static aclnnStatus ParamsDataContiguous(ChunkFwdOParams &params, aclOpExecutor *executorPtr)
{
    if (!(DataContiguous(params.q, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous q failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.k, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous k failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.v, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous v failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.h, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous h failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: Contiguous g failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkFwdOParams params)
{
    // dtype 契约（README §3.1/§3.3，与 def.cpp 动态 dtype 组一致）：
    // q/k/v/h/oOut ∈ {BF16, FP16} 且与 q 同 dtype；g ∈ {FP32, BF16, FP16} 独立白名单
    const ge::DataType qDtype = params.q->GetDataType();
    if (!(qDtype == ge::DT_BF16 || qDtype == ge::DT_FLOAT16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "q", ge::TypeUtils::DataTypeToSerialString(qDtype),
            "q dtype should be BF16 or FP16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.k->GetDataType() != qDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "k", ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(qDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.v->GetDataType() != qDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "v", ge::TypeUtils::DataTypeToSerialString(params.v->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(qDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.h->GetDataType() != qDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "h", ge::TypeUtils::DataTypeToSerialString(params.h->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(qDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.oOut->GetDataType() != qDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "oOut", ge::TypeUtils::DataTypeToSerialString(params.oOut->GetDataType()),
            "dtype should be the same as q: " + ge::TypeUtils::DataTypeToSerialString(qDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // g 组合约束（与 def.cpp 动态 dtype 组一致）：g 可为 FP32（提精度）或与 q 同 dtype；
    // q=BF16 → g∈{FP32, BF16}，q=FP16 → g∈{FP32, FP16}
    const ge::DataType gDtype = params.g->GetDataType();
    if (!(gDtype == ge::DT_FLOAT || gDtype == qDtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "g", ge::TypeUtils::DataTypeToSerialString(gDtype),
            "g dtype should be FP32 or the same as q: " + ge::TypeUtils::DataTypeToSerialString(qDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkFwdOParams params)
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
        const aclnnStatus checkStatus = CheckShape(params);
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
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查（透传 CheckParams 的错误码，保留 NULLPTR 161001 语义）
    auto ret = CheckParams(params);
    if (!(ret == ACLNN_SUCCESS)) {
        return ret;
    }
    if (!(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto result = l0op::ChunkFwdO(params.q, params.k, params.v, params.h, params.g, params.cuSeqlensOptional,
                                 params.chunkOffsetsOptional, params.scale, params.chunkSize, params.useExp2,
                                 params.stateVFirst, params.outputLayout, params.oOut, executorPtr);
    if (!(result[0] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(
            OP_NAME, "%s: operator returned no output; see the preceding tiling/impl error for the root cause",
            __func__);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result[0], params.oOut, executorPtr);
    if (!(viewCopyResult != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: viewCopyResult != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkFwdO(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkFwdO);
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "This is an error in ChunkFwdO launch aicore.", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
