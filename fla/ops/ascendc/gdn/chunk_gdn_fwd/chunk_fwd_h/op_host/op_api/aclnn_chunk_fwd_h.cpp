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
#include "graph/utils/type_utils.h"
#include "aclnn_chunk_fwd_h.h"
#include "chunk_fwd_h.h"
#include <dlfcn.h>
#include <new>
#include <vector>
#include <algorithm>

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
#include <string>


using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkFwdH";

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
    if (params.k == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "k", "k must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.w == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "w", "w must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.u == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "u", "u must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (params.hOut == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "hOut", "hOut must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.vNewOut == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "vNewOut", "vNewOut must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.outputFinalState != (params.finalStateOut != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "outputFinalState,finalStateOut",
            std::string("outputFinalState=") + (params.outputFinalState ? "true" : "false") +
                ",finalStateOut=" + (params.finalStateOut != nullptr ? "present" : "absent"),
            "outputFinalState must equal the physical presence of finalStateOut");
        return ACLNN_ERR_PARAM_INVALID;
    }
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
        if (!(storageFormat == Format::FORMAT_ND && viewFormat == Format::FORMAT_ND &&
              !IsPrivateFormat(storageFormat))) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                OP_NAME, tensorNames[idx],
                std::to_string(static_cast<int>(storageFormat)) + "/" +
                    std::to_string(static_cast<int>(viewFormat)),
                "non-private ND storage/view format only");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const aclTensor *outputs[] = {params.hOut, params.vNewOut, params.finalStateOut};
    const char *outputNames[] = {"hOut", "vNewOut", "finalStateOut"};
    for (size_t idx = 0; idx < sizeof(outputs) / sizeof(outputs[0]); ++idx) {
        if (outputs[idx] == nullptr) {
            continue;
        }
        if (!IsContiguous(outputs[idx])) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, outputNames[idx], "output must be contiguous because ChunkFwdH writes it directly");
            return ACLNN_ERR_PARAM_INVALID;
        }
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
    if (kShape.GetDimNum() != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "k", std::to_string(kShape.GetDimNum()),
            "k must be rank 4 before validating variable-length metadata");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional == nullptr) {
        if (params.chunkIndicesOptional != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, "chunkIndicesOptional", "chunkIndicesOptional requires cuSeqlensOptional");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (kShape.GetDim(0) != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "B", std::to_string(kShape.GetDim(0)), "variable-length BNSD input requires B=1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional->Size() < 2) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "cuSeqlensOptional", std::to_string(params.cuSeqlensOptional->Size()), ">= 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[0] != 0) {
        OP_LOGE_FOR_INVALID_VALUE(
            OP_NAME, "cuSeqlensOptional[0]", std::to_string((*params.cuSeqlensOptional)[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] != kShape.GetDim(2)) {
        OP_LOGE_FOR_INVALID_VALUE(
            OP_NAME, "cuSeqlensOptional[last]",
            std::to_string((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1]),
            std::to_string(kShape.GetDim(2)));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        if ((*params.cuSeqlensOptional)[seq] > (*params.cuSeqlensOptional)[seq + 1]) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "cuSeqlensOptional[seq],cuSeqlensOptional[seq+1]",
                std::to_string((*params.cuSeqlensOptional)[seq]) + "," +
                    std::to_string((*params.cuSeqlensOptional)[seq + 1]),
                "cuSeqlensOptional must be non-decreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks = CountChunks(params.cuSeqlensOptional, kShape.GetDim(2), params.chunkSize);
    if (params.chunkIndicesOptional->Size() != static_cast<size_t>(totalChunks) * 2) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "chunkIndicesOptional", std::to_string(params.chunkIndicesOptional->Size()),
            std::to_string(static_cast<size_t>(totalChunks) * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        const int64_t length = (*params.cuSeqlensOptional)[seq + 1] - (*params.cuSeqlensOptional)[seq];
        const int64_t chunks = (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if ((*params.chunkIndicesOptional)[offset] != static_cast<int64_t>(seq) ||
                (*params.chunkIndicesOptional)[offset + 1] != chunk) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    OP_NAME, "chunkIndicesOptional[offset],chunkIndicesOptional[offset+1]",
                    std::to_string((*params.chunkIndicesOptional)[offset]) + "," +
                        std::to_string((*params.chunkIndicesOptional)[offset + 1]),
                    "chunkIndicesOptional must use canonical sequence-major chunk order");
                return ACLNN_ERR_PARAM_INVALID;
            }
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
    if (!(kShape.GetDimNum() == 4 && wShape.GetDimNum() == 4 && uShape.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            OP_NAME, "k,w,u",
            std::to_string(kShape.GetDimNum()) + "," + std::to_string(wShape.GetDimNum()) + "," +
                std::to_string(uShape.GetDimNum()),
            "k, w and u must be rank-4 BNSD tensors");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(kShape.GetDim(0) > 0 && kShape.GetDim(1) > 0 && kShape.GetDim(2) > 0 && uShape.GetDim(1) > 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "B,HK,T,HV",
            std::to_string(kShape.GetDim(0)) + "," + std::to_string(kShape.GetDim(1)) + "," +
                std::to_string(kShape.GetDim(2)) + "," + std::to_string(uShape.GetDim(1)),
            "B, HK, T and HV must all be positive");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(kShape.GetDim(0) == wShape.GetDim(0) && kShape.GetDim(0) == uShape.GetDim(0) &&
          wShape.GetDim(1) == uShape.GetDim(1) && kShape.GetDim(2) == wShape.GetDim(2) &&
          kShape.GetDim(2) == uShape.GetDim(2) && kShape.GetDim(3) == wShape.GetDim(3))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "k,w,u",
            "k=[" + std::to_string(kShape.GetDim(0)) + "," + std::to_string(kShape.GetDim(1)) + "," +
                std::to_string(kShape.GetDim(2)) + "," + std::to_string(kShape.GetDim(3)) + "]",
            "k/w/u must agree on B and T, w/u on HV, and k/w on K");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(kShape.GetDim(3) == 128 && uShape.GetDim(3) == 128)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "K,V", std::to_string(kShape.GetDim(3)) + "," + std::to_string(uShape.GetDim(3)),
            "FwdH requires K=128 and V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gOptional != nullptr) {
        if (!(uShape.GetDim(1) >= kShape.GetDim(1) && uShape.GetDim(1) % kShape.GetDim(1) == 0)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "HK,HV",
                std::to_string(kShape.GetDim(1)) + "," + std::to_string(uShape.GetDim(1)),
                "g-only requires HV >= HK and HV divisible by HK");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (kShape.GetDim(1) != uShape.GetDim(1)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "kg.H,HV",
                std::to_string(kShape.GetDim(1)) + "," + std::to_string(uShape.GetDim(1)),
                "gk-only requires the prepared kg head count to equal HV");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.gOptional != nullptr) {
        auto gShape = params.gOptional->GetViewShape();
        if (!(gShape.GetDimNum() == 3 && gShape.GetDim(0) == uShape.GetDim(0) &&
              gShape.GetDim(1) == uShape.GetDim(1) && gShape.GetDim(2) == uShape.GetDim(2))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "g",
                "rank=" + std::to_string(gShape.GetDimNum()) + ",[" + std::to_string(gShape.GetDim(0)) + "," +
                    std::to_string(gShape.GetDim(1)) + "," + std::to_string(gShape.GetDim(2)) + "]",
                "g must have shape [B, HV, T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
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
    if (!(hShape.GetDimNum() == 5 && hShape.GetDim(0) == batch && hShape.GetDim(1) == totalChunks &&
          hShape.GetDim(2) == hv)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "hOut",
            "rank=" + std::to_string(hShape.GetDimNum()) + ",[" + std::to_string(hShape.GetDim(0)) + "," +
                std::to_string(hShape.GetDim(1)) + "," + std::to_string(hShape.GetDim(2)) + ",...]",
            "[B,num_chunks,HV,K,V] with num_chunks=" + std::to_string(totalChunks));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t hK = params.stateVFirst ? hShape.GetDim(4) : hShape.GetDim(3);
    const int64_t hV = params.stateVFirst ? hShape.GetDim(3) : hShape.GetDim(4);
    if (!(hK == kDim && hV == vDim)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "hOut.K,hOut.V", std::to_string(hK) + "," + std::to_string(hV),
            "hOut state dims must be [K,V] (" + std::to_string(kDim) + "," + std::to_string(vDim) +
                ") when stateVFirst=false, [V,K] otherwise");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto vNewShape = params.vNewOut->GetViewShape();
    if (!(vNewShape.GetDimNum() == 4 && vNewShape.GetDim(0) == batch &&
          vNewShape.GetDim(1) == hv && vNewShape.GetDim(2) == kShape.GetDim(2) &&
          vNewShape.GetDim(3) == vDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "vNewOut",
            "rank=" + std::to_string(vNewShape.GetDimNum()) + ",[" + std::to_string(vNewShape.GetDim(0)) + "," +
                std::to_string(vNewShape.GetDim(1)) + "," + std::to_string(vNewShape.GetDim(2)) + "," +
                std::to_string(vNewShape.GetDim(3)) + "]",
            "vNewOut must have shape [B, HV, T, V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *states[] = {params.initialStateOptional, params.finalStateOut};
    const char *stateNames[] = {"initialStateOptional", "finalStateOut"};
    for (size_t idx = 0; idx < 2; ++idx) {
        if (states[idx] == nullptr) {
            continue;
        }
        auto stateShape = states[idx]->GetViewShape();
        if (stateShape.GetDimNum() != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                OP_NAME, stateNames[idx], std::to_string(stateShape.GetDimNum()),
                std::string(stateNames[idx]) + " must be rank 4");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t stateK = params.stateVFirst ? stateShape.GetDim(3) : stateShape.GetDim(2);
        const int64_t stateV = params.stateVFirst ? stateShape.GetDim(2) : stateShape.GetDim(3);
        if (!(stateShape.GetDim(0) == seqNum && stateShape.GetDim(1) == hv &&
              stateK == kDim && stateV == vDim)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, stateNames[idx],
                "[" + std::to_string(stateShape.GetDim(0)) + "," + std::to_string(stateShape.GetDim(1)) + "," +
                    std::to_string(stateK) + "," + std::to_string(stateV) + "]",
                std::string(stateNames[idx]) + " must be [N,HV,K,V] when stateVFirst=false, [N,HV,V,K] otherwise");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkFwdHParams params)
{
    auto inputDtype = params.k->GetDataType();
    if (inputDtype != DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "k,w,u", ge::TypeUtils::DataTypeToSerialString(inputDtype),
            "k, w and u only support bfloat16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.w->GetDataType() == inputDtype && params.u->GetDataType() == inputDtype)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "k,w,u",
            std::string("k=") + ge::TypeUtils::DataTypeToSerialString(inputDtype) +
                ",w=" + ge::TypeUtils::DataTypeToSerialString(params.w->GetDataType()) +
                ",u=" + ge::TypeUtils::DataTypeToSerialString(params.u->GetDataType()),
            "k, w and u must have the same dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.hOut->GetDataType() == inputDtype && params.vNewOut->GetDataType() == inputDtype)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "hOut,vNewOut",
            std::string("hOut=") + ge::TypeUtils::DataTypeToSerialString(params.hOut->GetDataType()) +
                ",vNewOut=" + ge::TypeUtils::DataTypeToSerialString(params.vNewOut->GetDataType()),
            "hOut and vNewOut dtype must match k, w and u");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto gateDtype = params.gOptional != nullptr ? params.gOptional->GetDataType() : params.gkOptional->GetDataType();
    if (!(gateDtype == DataType::DT_FLOAT || gateDtype == DataType::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "g/gk", ge::TypeUtils::DataTypeToSerialString(gateDtype),
            "g/gk dtype must be bfloat16 or float32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.initialStateOptional != nullptr) {
        auto stateDtype = params.initialStateOptional->GetDataType();
        if (!(stateDtype == DataType::DT_BF16 || stateDtype == DataType::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "initialStateOptional", ge::TypeUtils::DataTypeToSerialString(stateDtype),
                "initialStateOptional dtype must be bfloat16 or float32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.finalStateOut != nullptr) {
            if (params.finalStateOut->GetDataType() != stateDtype) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    OP_NAME, "finalStateOut",
                    ge::TypeUtils::DataTypeToSerialString(params.finalStateOut->GetDataType()),
                    std::string("must match initialStateOptional: ") +
                        ge::TypeUtils::DataTypeToSerialString(stateDtype));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    } else if (params.finalStateOut != nullptr) {
        auto stateDtype = params.finalStateOut->GetDataType();
        if (!(stateDtype == DataType::DT_BF16 || stateDtype == DataType::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "finalStateOut", ge::TypeUtils::DataTypeToSerialString(stateDtype),
                "finalStateOut dtype must be bfloat16 or float32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// state_v_first=true 的末两维转置辅助（gated kernel 恒 [K,V] 产出，aclnn 层做
// [K,V]<->[V,K] 适配）。翻译原语失败时 typed 上报参数名/实际值/期望值，返回码不变。

// gated 引擎与配套翻译仅 DAV_2201（A2/A3）生效，其余芯片保持基线行为。
static bool SocSupportsGatedPath()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201;
}

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
        // bf16 g 不在入口 cast，由 kernel 内 Cast 转 fp32（数值等价）。
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
    if (!((params.gOptional != nullptr) != (params.gkOptional != nullptr))) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "g", "Exactly one of g and gk must be provided.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGkParams(const ChunkFwdHParams &params)
{
    if (params.gkOptional != nullptr) {
        auto gkShape = params.gkOptional->GetViewShape();
        if (!(gkShape.GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(OP_NAME, "rank", std::to_string(gkShape.GetDimNum()), "gk must have rank 4 when provided, got rank %ld.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        if (!(gkShape.GetDim(3) == params.k->GetViewShape().GetDim(3))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "k", "invalid", "gk.shape[3] (K) must match k.shape[3] (K).");
        return ACLNN_ERR_PARAM_INVALID;
    }
        if (!(gkShape.GetDim(2) == params.k->GetViewShape().GetDim(2))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "k", "invalid", "gk.shape[2] (T) must match k.shape[2] (T).");
        return ACLNN_ERR_PARAM_INVALID;
    }
        if (!(gkShape.GetDim(1) == params.u->GetViewShape().GetDim(1))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "HV", "invalid", "gk.shape[1] (HV) must match u.shape[1] (HV).");
        return ACLNN_ERR_PARAM_INVALID;
    }
        if (!(gkShape.GetDim(0) == params.k->GetViewShape().GetDim(0))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "k", "invalid", "gk.shape[0] (B) must match k.shape[0] (B).");
        return ACLNN_ERR_PARAM_INVALID;
    }
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
    if (!(params.chunkSize == 64)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "chunkSize", std::to_string(params.chunkSize), "chunkSize must be 64, but got %ld.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.saveNewValue)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "saveNewValue", std::to_string(params.saveNewValue), "saveNewValue must be true.");
        return ACLNN_ERR_PARAM_INVALID;
    }
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
    if (!(workspaceSize != nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "workspaceSize", "workspaceSize must not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(executor != nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, "executor", "executor must not be nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
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
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_CREATE_EXECUTOR,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = ParamsDataContiguous(params, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    // vfirst=1 dense 翻译（转 [K,V] 进 kernel + 传 false 进 gated）；varlen 不
    // 翻译（vfirst 原样直达，由 kernel 谓词送老路径）。
    const bool translateVFirst = params.stateVFirst && params.cuSeqlensOptional == nullptr &&
        SocSupportsGatedPath();
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
