/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Load structured reporters before opdev's legacy logging macros. The opdev
// headers must retain their logging convention for CHECK_RET and DFX helpers.
#include "op_common/log/log.h"
#include "aclnn_chunk_fwd_h.h"
#include "chunk_fwd_h.h"
#include <dlfcn.h>
#include <new>
#include <vector>
#include <algorithm>

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
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

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkFwdHParams {
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *u = nullptr;
    const aclTensor *gOptional = nullptr;
    const aclTensor *gkOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    bool outputFinalState = false;
    int64_t chunkSize = 64;
    bool saveNewValue = true;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    bool useExp2 = false;
    bool stateVFirst = false;
    const aclTensor *hOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkFwdHParams params)
{
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.u != nullptr, ACLNN_ERR_PARAM_NULLPTR, "u must not be nullptr.");

    CHECK_COND(params.hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hOut must not be nullptr.");
    CHECK_COND(params.vNewOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "vNewOut must not be nullptr.");
    CHECK_COND(params.outputFinalState == (params.finalStateOut != nullptr), ACLNN_ERR_PARAM_INVALID,
               "outputFinalState must equal the physical presence of finalStateOut.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkFwdHParams params)
{
    const aclTensor *tensors[] = {
        params.k,
        params.w,
        params.u,
        params.gOptional,
        params.gkOptional,
        params.initialStateOptional,
        params.hOut,
        params.vNewOut,
        params.finalStateOut,
    };
    const char *tensorNames[] = {
        "k",
        "w",
        "u",
        "gOptional",
        "gkOptional",
        "initialStateOptional",
        "hOut",
        "vNewOut",
        "finalStateOut",
    };
    for (size_t idx = 0; idx < sizeof(tensors) / sizeof(tensors[0]); ++idx) {
        if (tensors[idx] == nullptr) {
            continue;
        }
        const auto storageFormat = tensors[idx]->GetStorageFormat();
        const auto viewFormat = tensors[idx]->GetViewFormat();
        CHECK_COND(storageFormat == Format::FORMAT_ND && viewFormat == Format::FORMAT_ND &&
                       !IsPrivateFormat(storageFormat),
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must use non-private ND storage/view format, but got storage=%d and view=%d.",
                   tensorNames[idx], static_cast<int>(storageFormat),
                   static_cast<int>(viewFormat));
    }
    const aclTensor *outputs[] = {params.hOut, params.vNewOut, params.finalStateOut};
    const char *outputNames[] = {"hOut", "vNewOut", "finalStateOut"};
    for (size_t idx = 0; idx < sizeof(outputs) / sizeof(outputs[0]); ++idx) {
        if (outputs[idx] == nullptr) {
            continue;
        }
        CHECK_COND(IsContiguous(outputs[idx]), ACLNN_ERR_PARAM_INVALID,
                   "%s must be contiguous because ChunkFwdH writes it directly.",
                   outputNames[idx]);
    }
    return ACLNN_SUCCESS;
}

static int64_t CountChunks(const aclIntArray *cuSeqlens, int64_t seqlen, int64_t chunkSize)
{
    if (cuSeqlens == nullptr) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        totalChunks += (length + chunkSize - 1) / chunkSize;
    }
    return totalChunks;
}

static aclnnStatus CheckVariableLengthInputs(const ChunkFwdHParams &params)
{
    const auto kShape = params.k->GetViewShape();
    CHECK_COND(kShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
               "k must be rank 4 before validating variable-length metadata.");
    if (params.cuSeqlensOptional == nullptr) {
        CHECK_COND(params.chunkIndicesOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional requires cuSeqlensOptional.");
        return ACLNN_SUCCESS;
    }

    CHECK_COND(kShape.GetDim(0) == 1, ACLNN_ERR_PARAM_INVALID,
               "Variable-length BNSD input requires B=1, but got B=%ld.", kShape.GetDim(0));
    CHECK_COND(params.cuSeqlensOptional->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND((*params.cuSeqlensOptional)[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional[0] must be 0.");
    CHECK_COND((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] == kShape.GetDim(2),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional last element must equal T=%ld.", kShape.GetDim(2));
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        CHECK_COND((*params.cuSeqlensOptional)[seq] <= (*params.cuSeqlensOptional)[seq + 1],
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be non-decreasing; equal neighbours denote empty sequences.");
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks = CountChunks(params.cuSeqlensOptional, kShape.GetDim(2), params.chunkSize);
    CHECK_COND(params.chunkIndicesOptional->Size() == static_cast<size_t>(totalChunks) * 2,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain exactly one (seq_id, chunk_id) pair per chunk.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        const int64_t length = (*params.cuSeqlensOptional)[seq + 1] - (*params.cuSeqlensOptional)[seq];
        const int64_t chunks = (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND((*params.chunkIndicesOptional)[offset] == static_cast<int64_t>(seq) &&
                           (*params.chunkIndicesOptional)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional must use canonical sequence-major chunk order.");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkFwdHParams params)
{
    auto kShape = params.k->GetViewShape();
    auto wShape = params.w->GetViewShape();
    auto uShape = params.u->GetViewShape();
    CHECK_COND(kShape.GetDimNum() == 4 && wShape.GetDimNum() == 4 && uShape.GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "k, w and u must be rank-4 BNSD tensors.");
    CHECK_COND(kShape.GetDim(0) > 0 && kShape.GetDim(1) > 0 && kShape.GetDim(2) > 0 &&
                   uShape.GetDim(1) > 0,
               ACLNN_ERR_PARAM_INVALID,
               "B, HK, T and HV must all be positive, but got B=%ld, HK=%ld, T=%ld, HV=%ld.",
               kShape.GetDim(0), kShape.GetDim(1), kShape.GetDim(2), uShape.GetDim(1));
    CHECK_COND(kShape.GetDim(0) == wShape.GetDim(0) && kShape.GetDim(0) == uShape.GetDim(0) &&
                   wShape.GetDim(1) == uShape.GetDim(1) && kShape.GetDim(2) == wShape.GetDim(2) &&
                   kShape.GetDim(2) == uShape.GetDim(2) && kShape.GetDim(3) == wShape.GetDim(3),
               ACLNN_ERR_PARAM_INVALID,
               "k, w and u must match in B/T, w and u must match in HV, and k and w must match in K.");
    CHECK_COND(kShape.GetDim(3) == 128 && uShape.GetDim(3) == 128, ACLNN_ERR_PARAM_INVALID,
               "FwdH requires K=128 and V=128, but got K=%ld and V=%ld.",
               kShape.GetDim(3), uShape.GetDim(3));
    if (params.gOptional != nullptr) {
        CHECK_COND(uShape.GetDim(1) >= kShape.GetDim(1) && uShape.GetDim(1) % kShape.GetDim(1) == 0,
                   ACLNN_ERR_PARAM_INVALID,
                   "g-only requires HV >= HK and HV divisible by HK, but got HK=%ld and HV=%ld.",
                   kShape.GetDim(1), uShape.GetDim(1));
    } else {
        CHECK_COND(kShape.GetDim(1) == uShape.GetDim(1), ACLNN_ERR_PARAM_INVALID,
                   "gk-only requires the prepared kg head count to equal HV, but got kg heads=%ld and HV=%ld.",
                   kShape.GetDim(1), uShape.GetDim(1));
    }
    if (params.gOptional != nullptr) {
        auto gShape = params.gOptional->GetViewShape();
        CHECK_COND(gShape.GetDimNum() == 3 && gShape.GetDim(0) == uShape.GetDim(0) &&
                       gShape.GetDim(1) == uShape.GetDim(1) && gShape.GetDim(2) == uShape.GetDim(2),
                   ACLNN_ERR_PARAM_INVALID, "g must have shape [B, HV, T].");
    }
    const int64_t batch = kShape.GetDim(0);
    const int64_t hv = uShape.GetDim(1);
    const int64_t kDim = kShape.GetDim(3);
    const int64_t vDim = uShape.GetDim(3);
    const int64_t seqNum = params.cuSeqlensOptional == nullptr
                               ? batch
                               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    const int64_t totalChunks = CountChunks(params.cuSeqlensOptional, kShape.GetDim(2), params.chunkSize);
    auto hShape = params.hOut->GetViewShape();
    CHECK_COND(hShape.GetDimNum() == 5 && hShape.GetDim(0) == batch && hShape.GetDim(1) == totalChunks &&
                   hShape.GetDim(2) == hv,
               ACLNN_ERR_PARAM_INVALID,
               "hOut must be [B, num_chunks, HV, K, V] (or [B, num_chunks, HV, V, K]), "
               "where num_chunks=%ld.", totalChunks);
    const int64_t hK = params.stateVFirst ? hShape.GetDim(4) : hShape.GetDim(3);
    const int64_t hV = params.stateVFirst ? hShape.GetDim(3) : hShape.GetDim(4);
    CHECK_COND(hK == kDim && hV == vDim, ACLNN_ERR_PARAM_INVALID,
               "hOut state dimensions must be [K, V] when stateVFirst=false and [V, K] otherwise.");
    auto vNewShape = params.vNewOut->GetViewShape();
    CHECK_COND(vNewShape.GetDimNum() == 4 && vNewShape.GetDim(0) == batch &&
                   vNewShape.GetDim(1) == hv && vNewShape.GetDim(2) == kShape.GetDim(2) &&
                   vNewShape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "vNewOut must have shape [B, HV, T, V].");
    const aclTensor *states[] = {params.initialStateOptional, params.finalStateOut};
    const char *stateNames[] = {"initialStateOptional", "finalStateOut"};
    for (size_t idx = 0; idx < 2; ++idx) {
        if (states[idx] == nullptr) {
            continue;
        }
        auto stateShape = states[idx]->GetViewShape();
        CHECK_COND(stateShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "%s must be rank 4.", stateNames[idx]);
        const int64_t stateK = params.stateVFirst ? stateShape.GetDim(3) : stateShape.GetDim(2);
        const int64_t stateV = params.stateVFirst ? stateShape.GetDim(2) : stateShape.GetDim(3);
        CHECK_COND(stateShape.GetDim(0) == seqNum &&
                       stateShape.GetDim(1) == hv && stateK == kDim && stateV == vDim,
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must be [N, HV, K, V] when stateVFirst=false and [N, HV, V, K] otherwise.",
                   stateNames[idx]);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkFwdHParams params)
{
    auto inputDtype = params.k->GetDataType();
    CHECK_COND(inputDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "k, w and u only support bfloat16.");
    CHECK_COND(params.w->GetDataType() == inputDtype && params.u->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "k, w and u must have the same dtype.");
    CHECK_COND(params.hOut->GetDataType() == inputDtype && params.vNewOut->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "hOut and vNewOut dtype must match k, w and u.");
    auto gateDtype = params.gOptional != nullptr ? params.gOptional->GetDataType() : params.gkOptional->GetDataType();
    CHECK_COND(gateDtype == DataType::DT_FLOAT || gateDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "g/gk dtype must be bfloat16 or float32.");
    if (params.initialStateOptional != nullptr) {
        auto stateDtype = params.initialStateOptional->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_BF16 || stateDtype == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "initialStateOptional dtype must be bfloat16 or float32.");
        if (params.finalStateOut != nullptr) {
            CHECK_COND(params.finalStateOut->GetDataType() == stateDtype, ACLNN_ERR_PARAM_INVALID,
                       "finalStateOut dtype must match initialStateOptional dtype.");
        }
    } else if (params.finalStateOut != nullptr) {
        auto stateDtype = params.finalStateOut->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_BF16 || stateDtype == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "finalStateOut dtype must be bfloat16 or float32.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// state_v_first=true 的末两维转置辅助（照抄 chunk_gated_delta_rule_fwd_h 同名实现，
// 第七轮方案③：gated 路径 kernel 恒 [K,V] 产出，aclnn 层做 [K,V]<->[V,K] 适配）
// typed DFX 上报：内部翻译原语（l0op::Cast / TransposeLastTwo / AllocTensor）失败时
// 携带参数名、实际值与期望值，替代裸 CHECK_RET 的无上下文状态码返回（报错宏惯例
// 对齐 chunk_kda_bwd_prepare，返回码保持 ACLNN_ERR_INNER_NULLPTR 不变）。
static constexpr const char *OP_NAME = "aclnnChunkFwdH";

static aclnnStatus TranslationNullptr(const char *name, const char *expected)
{
    OP_LOGE_FOR_INVALID_VALUE(OP_NAME, name, "nullptr", expected);
    return ACLNN_ERR_INNER_NULLPTR;
}

static op::Shape SwapLastTwo(const op::Shape &input)
{
    op::Shape output = input;
    const int64_t dimA = input.GetDim(input.GetDimNum() - 2);
    const int64_t dimB = input.GetDim(input.GetDimNum() - 1);
    output.SetDim(input.GetDimNum() - 2, dimB);
    output.SetDim(input.GetDimNum() - 1, dimA);
    return output;
}

static const aclTensor *TransposeLastTwo(const aclTensor *input, aclOpExecutor *executor)
{
    const size_t rank = input->GetViewShape().GetDimNum();
    std::vector<int64_t> perm(rank);
    for (size_t idx = 0; idx < rank; ++idx) {
        perm[idx] = static_cast<int64_t>(idx);
    }
    std::swap(perm[rank - 2], perm[rank - 1]);
    const aclIntArray *permArray = executor->AllocIntArray(perm.data(), perm.size());
    if (permArray == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "perm", "nullptr perm array",
                                  "int64 perm swapping the last two dims");
        return nullptr;
    }
    return l0op::Transpose(input, permArray, executor);
}

static aclnnStatus ParamsDataContiguous(ChunkFwdHParams &params, aclOpExecutor *executorPtr)
{
    auto status = DataContiguous(params.k, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.w, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.u, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (params.gOptional != nullptr) {
        status = DataContiguous(params.gOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
        // bf16 gate cast 翻译（放宽轮④，先例 gated 本尊整链对 beta/g 的同款处理）：
        // gated 谓词要求 fp32 gate——bf16 g 在此 cast 到 fp32（无损），kernel 侧
        // 恒见 float gate；数值与老路径"先 cast 再算"完全等价（gated 原生 bf16
        // gate 会在 bf16 域算 decay，精度低，故必须选入口 cast 而非原生放行）。
        if (params.gOptional->GetDataType() == DataType::DT_BF16) {
            const aclTensor *gFp32 = l0op::Cast(params.gOptional, DataType::DT_FLOAT, executorPtr);
            if (gFp32 == nullptr) {
                return TranslationNullptr("g", "fp32 gate tensor after the bf16 -> fp32 entry cast");
            }
            params.gOptional = gFp32;
        }
    }
    if (params.gkOptional != nullptr) {
        status = DataContiguous(params.gkOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    if (params.initialStateOptional != nullptr) {
        status = DataContiguous(params.initialStateOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGateOptionalNonNull(const ChunkFwdHParams &params)
{
    CHECK_COND((params.gOptional != nullptr) != (params.gkOptional != nullptr), ACLNN_ERR_PARAM_INVALID,
               "Exactly one of g and gk must be provided.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGkParams(const ChunkFwdHParams &params)
{
    if (params.gkOptional != nullptr) {
        auto gkShape = params.gkOptional->GetViewShape();
        CHECK_COND(gkShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "gk must have rank 4 when provided, got rank %ld.", gkShape.GetDimNum());
        CHECK_COND(gkShape.GetDim(3) == params.k->GetViewShape().GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[3] (K) must match k.shape[3] (K).");
        CHECK_COND(gkShape.GetDim(2) == params.k->GetViewShape().GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[2] (T) must match k.shape[2] (T).");
        CHECK_COND(gkShape.GetDim(1) == params.u->GetViewShape().GetDim(1), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[1] (HV) must match u.shape[1] (HV).");
        CHECK_COND(gkShape.GetDim(0) == params.k->GetViewShape().GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[0] (B) must match k.shape[0] (B).");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkFwdHParams params)
{
    auto status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGateOptionalNonNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_COND(params.chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 64, but got %ld.", params.chunkSize);
    CHECK_COND(params.saveNewValue, ACLNN_ERR_PARAM_INVALID, "saveNewValue must be true.");
    status = CheckVariableLengthInputs(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckShape(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGkParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckDtype(params);
}

aclnnStatus aclnnChunkFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "executor must not be nullptr.");
    ChunkFwdHParams params{k,
                                         w,
                                         u,
                                         gOptional,
                                         gkOptional,
                                         initialStateOptional,
                                         outputFinalState,
                                         chunkSize,
                                         saveNewValue,
                                         cuSeqlensOptional,
                                         chunkIndicesOptional,
                                         useExp2,
                                         stateVFirst,
                                         hOut,
                                         vNewOut,
                                         finalStateOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkFwdH,
                   DFX_IN(k, w, u, gOptional, gkOptional, initialStateOptional, outputFinalState, chunkSize,
                          saveNewValue, cuSeqlensOptional, chunkIndicesOptional, useExp2, stateVFirst),
                   DFX_OUT(hOut, vNewOut, finalStateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = ParamsDataContiguous(params, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    // stateVFirst=true 的 aclnn 层翻译（第七轮方案③，照抄 gated）——varlen 感知
    // （2026-10-26 分流路由收口轮）：dense 维持翻译（转 [K,V] 进 kernel + 传
    // false 进 gated，含税端到端优于老路径）；varlen 不翻译（vfirst 原样直达，
    // kernel 谓词送老路径 STATE_V_FIRST 模板——A/B 实测裁定：老路径 ~5939 优于
    // gated 含税 ~7292，h 1GB 级 Transpose 税吞掉 gated 全部优势）。
    const bool translateVFirst = params.stateVFirst && params.cuSeqlensOptional == nullptr;
    const aclTensor *initialStateCompute = params.initialStateOptional;
    const aclTensor *hCompute = params.hOut;
    const aclTensor *finalStateCompute = params.finalStateOut;
    if (translateVFirst && initialStateCompute != nullptr) {
        initialStateCompute = TransposeLastTwo(initialStateCompute, executorPtr);
        if (initialStateCompute == nullptr) {
            return TranslationNullptr("initial_state", "transposed [K,V] tensor from TransposeLastTwo");
        }
    }
    if (translateVFirst) {
        hCompute = executorPtr->AllocTensor(
            SwapLastTwo(params.hOut->GetViewShape()), params.hOut->GetDataType(), Format::FORMAT_ND);
        if (hCompute == nullptr) {
            return TranslationNullptr("h", "swapped-layout staging tensor from AllocTensor");
        }
        if (params.finalStateOut != nullptr) {
            finalStateCompute = executorPtr->AllocTensor(
                SwapLastTwo(params.finalStateOut->GetViewShape()), params.finalStateOut->GetDataType(),
                Format::FORMAT_ND);
            if (finalStateCompute == nullptr) {
                return TranslationNullptr("final_state", "swapped-layout staging tensor from AllocTensor");
            }
        }
    }
    auto result = l0op::ChunkFwdH(
        params.k, params.w, params.u, params.gOptional, params.gkOptional, initialStateCompute,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.outputFinalState, params.chunkSize,
        params.saveNewValue, params.useExp2, params.stateVFirst && !translateVFirst,
        hCompute, params.vNewOut, finalStateCompute, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr &&
                  (!outputFinalState || result[2] != nullptr),
              ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *hResult = result[0];
    if (translateVFirst) {
        hResult = TransposeLastTwo(hResult, executorPtr);
        if (hResult == nullptr) {
            return TranslationNullptr("h", "transposed output tensor from TransposeLastTwo");
        }
    }
    auto viewCopyResult0 = l0op::ViewCopy(hResult, params.hOut, executorPtr);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(result[1], params.vNewOut, executorPtr);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (outputFinalState && params.finalStateOut != nullptr) {
        const aclTensor *finalStateResult = result[2];
        if (translateVFirst) {
            finalStateResult = TransposeLastTwo(finalStateResult, executorPtr);
            if (finalStateResult == nullptr) {
                return TranslationNullptr("final_state", "transposed output tensor from TransposeLastTwo");
            }
        }
        auto viewCopyResult2 = l0op::ViewCopy(finalStateResult, params.finalStateOut, executorPtr);
        CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkFwdH(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkFwdH);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkFwdH launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
