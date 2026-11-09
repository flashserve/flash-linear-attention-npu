/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "aclnn_chunk_kda_fwd_prepare.h"
#include "chunk_kda_fwd_prepare.h"
#include "../chunk_kda_fwd_prepare_output_mask.h"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "../../../../common/host_utils/log.h"

using namespace op;

namespace {

constexpr const char *PREPARE_ENTITY = "aclnnChunkKdaFwdPrepare";
constexpr int64_t PREPARE_CHUNK_SIZE = 64;
constexpr int64_t PREPARE_K_DIM = 128;
constexpr int64_t PREPARE_V_DIM = 128;

enum class PrepareLayout {
    BNSD,
    BSND,
    NTD,
    TND,
};

struct ChunkKdaFwdPrepareParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = "BNSD";
    double scale = 1.0;
    int64_t chunkSize = PREPARE_CHUNK_SIZE;
    double epsilon = 1.0e-6;
    bool useQkL2normInKernel = false;
    bool useGateInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool useExp2 = false;
    const aclTensor *gkOut = nullptr;
    const aclTensor *aqkOut = nullptr;
    const aclTensor *akkOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *qgScaledOut = nullptr;
    const aclTensor *qHatOut = nullptr;
    const aclTensor *kHatOut = nullptr;
    const aclTensor *qRstdOut = nullptr;
    const aclTensor *kRstdOut = nullptr;
    const aclTensor *betaEffOut = nullptr;
};

struct PrepareShapeInfo {
    bool packed = false;
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t qkHeadNum = 0;
    int64_t valueHeadNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
};

size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs == nullptr || rhs == nullptr || Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t dim = 0; dim < Rank(lhs); ++dim) {
        if (Dim(lhs, dim) != Dim(rhs, dim)) {
            return false;
        }
    }
    return true;
}

bool HasShape(const aclTensor *tensor,
              std::initializer_list<int64_t> expected)
{
    if (tensor == nullptr || Rank(tensor) != expected.size()) {
        return false;
    }
    size_t dim = 0;
    for (int64_t value : expected) {
        if (Dim(tensor, dim++) != value) {
            return false;
        }
    }
    return true;
}

aclnnStatus ParseLayout(const char *layout, PrepareLayout &parsed)
{
    if (layout == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_ENTITY, "layout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (std::strcmp(layout, "BNSD") == 0) {
        parsed = PrepareLayout::BNSD;
    } else if (std::strcmp(layout, "BSND") == 0) {
        parsed = PrepareLayout::BSND;
    } else if (std::strcmp(layout, "NTD") == 0) {
        parsed = PrepareLayout::NTD;
    } else if (std::strcmp(layout, "TND") == 0) {
        parsed = PrepareLayout::TND;
    } else {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "layout", layout, "BNSD, BSND, NTD or TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNotNull(const ChunkKdaFwdPrepareParams &params)
{
    const aclTensor *inputs[] = {params.q, params.k, params.v, params.g,
                                 params.beta};
    const char *inputNames[] = {"q", "k", "v", "g", "beta"};
    for (size_t index = 0; index < sizeof(inputs) / sizeof(inputs[0]); ++index) {
        if (inputs[index] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_ENTITY, inputNames[index]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }

    const aclTensor *outputs[] = {
        params.gkOut, params.aqkOut, params.wOut,
        params.uOut, params.kgOut, params.qgScaledOut};
    const char *outputNames[] = {
        "gkOut", "aqkOut", "wOut", "uOut", "kgOut", "qgScaledOut"};
    for (size_t index = 0; index < sizeof(outputs) / sizeof(outputs[0]); ++index) {
        if (outputs[index] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_ENTITY, outputNames[index]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

uint32_t GetOutputMask(const ChunkKdaFwdPrepareParams &params)
{
    const aclTensor *outputs[] = {
        params.gkOut,       params.aqkOut,  params.akkOut,
        params.wOut,        params.uOut,    params.qgOut,
        params.kgOut,       params.qgScaledOut,
        params.qHatOut,     params.kHatOut, params.qRstdOut,
        params.kRstdOut,    params.betaEffOut};
    uint32_t outputMask = 0;
    for (size_t index = 0; index < sizeof(outputs) / sizeof(outputs[0]); ++index) {
        if (outputs[index] != nullptr) {
            outputMask |= 1U << index;
        }
    }
    return outputMask;
}

int64_t GetOutputMode(const ChunkKdaFwdPrepareParams &params)
{
    const uint32_t outputMask = GetOutputMask(params);
    if (outputMask == optiling::PREPARE_REQUIRED_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_NONE;
    }
    if (outputMask == optiling::PREPARE_FORWARD_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_FORWARD;
    }
    if (outputMask == optiling::PREPARE_RECOMPUTE_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_RECOMPUTE;
    }
    if (outputMask == optiling::PREPARE_SAVE_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_SAVE;
    }
    return -1;
}

aclnnStatus CheckOutputMode(const ChunkKdaFwdPrepareParams &params)
{
    if (GetOutputMode(params) < 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "outputMask", "invalid", "none/forward/recompute/save");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFormat(const ChunkKdaFwdPrepareParams &params)
{
    const aclTensor *tensors[] = {
        params.q,          params.k,        params.v,
        params.g,          params.beta,     params.aLogOptional,
        params.dtBiasOptional,
        params.gkOut,      params.aqkOut,   params.akkOut,
        params.wOut,       params.uOut,     params.qgOut,
        params.kgOut,      params.qgScaledOut,
        params.qHatOut,    params.kHatOut,  params.qRstdOut,
        params.kRstdOut,   params.betaEffOut};
    const char *names[] = {
        "q",          "k",        "v",          "g",          "beta",
        "aLogOptional", "dtBiasOptional",
        "gkOut",      "aqkOut",   "akkOut",     "wOut",       "uOut",
        "qgOut",      "kgOut",    "qgScaledOut", "qHatOut",   "kHatOut",
        "qRstdOut",   "kRstdOut", "betaEffOut"};
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        if (tensors[index] == nullptr) {
            continue;
        }
        const auto storageFormat = tensors[index]->GetStorageFormat();
        const auto viewFormat = tensors[index]->GetViewFormat();
        // 只拦私有（分形）格式：ND/NCHW/NCL/NHWC 等非私有拼写在逻辑维度上等价，
        // 张量本身已由 CheckContiguous 保证连续，不能因为拼写不同就拒绝调用方。
        if (IsPrivateFormat(storageFormat) || IsPrivateFormat(viewFormat)) {
            OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(PREPARE_ENTITY, names[index],
                "storage=" + std::to_string(static_cast<int>(storageFormat)) + ",view=" + std::to_string(static_cast<int>(viewFormat)),
                "non-private format");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const aclTensor *outputs[] = {
        params.gkOut,       params.aqkOut,  params.akkOut,
        params.wOut,        params.uOut,    params.qgOut,
        params.kgOut,       params.qgScaledOut,
        params.qHatOut,     params.kHatOut, params.qRstdOut,
        params.kRstdOut,    params.betaEffOut};
    const char *outputNames[] = {
        "gkOut",       "aqkOut",  "akkOut",      "wOut",      "uOut",
        "qgOut",       "kgOut",   "qgScaledOut", "qHatOut",   "kHatOut",
        "qRstdOut",    "kRstdOut", "betaEffOut"};
    for (size_t index = 0; index < sizeof(outputs) / sizeof(outputs[0]); ++index) {
        if (outputs[index] == nullptr) {
            continue;
        }
        if (!IsContiguous(outputs[index])) {
            OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(PREPARE_ENTITY, outputNames[index], "non-contiguous", "contiguous");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const ChunkKdaFwdPrepareParams &params)
{
    if (!(params.q->GetDataType() == DataType::DT_BF16 && params.k->GetDataType() == DataType::DT_BF16 &&
          params.v->GetDataType() == DataType::DT_BF16)) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PREPARE_ENTITY, "q,k,v",
            "q=" + fla_npu::host_utils::DfxDtypeString(params.q->GetDataType()) + ",k=" + fla_npu::host_utils::DfxDtypeString(params.k->GetDataType()) + ",v=" + fla_npu::host_utils::DfxDtypeString(params.v->GetDataType()), "BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto gateDtype = params.g->GetDataType();
    const auto betaDtype = params.beta->GetDataType();
    if (!(gateDtype == DataType::DT_BF16 || gateDtype == DataType::DT_FLOAT)) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PREPARE_ENTITY, "g", fla_npu::host_utils::DfxDtypeString(gateDtype), "BF16,FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(betaDtype == DataType::DT_BF16 || betaDtype == DataType::DT_FLOAT)) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PREPARE_ENTITY, "beta", fla_npu::host_utils::DfxDtypeString(betaDtype), "BF16,FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.aLogOptional != nullptr) {
        if (params.aLogOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PREPARE_ENTITY, "aLogOptional",
                fla_npu::host_utils::DfxDtypeString(params.aLogOptional->GetDataType()), "FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dtBiasOptional != nullptr) {
        if (params.dtBiasOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PREPARE_ENTITY, "dtBiasOptional",
                fla_npu::host_utils::DfxDtypeString(params.dtBiasOptional->GetDataType()), "FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const aclTensor *bf16Outputs[] = {
        params.aqkOut, params.akkOut, params.wOut, params.uOut,
        params.qgOut, params.kgOut, params.qgScaledOut,
        params.qHatOut, params.kHatOut};
    for (const aclTensor *output : bf16Outputs) {
        if (output == nullptr) {
            continue;
        }
        if (output->GetDataType() != DataType::DT_BF16) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PREPARE_ENTITY, "output",
                fla_npu::host_utils::DfxDtypeString(output->GetDataType()), "BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const aclTensor *fp32Outputs[] = {
        params.gkOut, params.qRstdOut, params.kRstdOut, params.betaEffOut};
    for (const aclTensor *output : fp32Outputs) {
        if (output == nullptr) {
            continue;
        }
        if (output->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(PREPARE_ENTITY, "output",
                fla_npu::host_utils::DfxDtypeString(output->GetDataType()), "FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

bool FitsUint32(int64_t value)
{
    return value > 0 &&
        static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

bool FitsDmaSourceStride(int64_t headNum, uint64_t rowBytes)
{
    return headNum > 0 && rowBytes > 0 &&
        static_cast<uint64_t>(headNum - 1) <=
            std::numeric_limits<uint32_t>::max() / rowBytes;
}

aclnnStatus CheckSequenceMajorDmaStride(
    const ChunkKdaFwdPrepareParams &params, PrepareLayout layout,
    const PrepareShapeInfo &info)
{
    if (layout != PrepareLayout::BSND && layout != PrepareLayout::TND) {
        return ACLNN_SUCCESS;
    }

    const uint64_t gateElementBytes =
        params.g->GetDataType() == DataType::DT_FLOAT ? sizeof(float) : 2U;
    const uint64_t betaElementBytes =
        params.beta->GetDataType() == DataType::DT_FLOAT ? sizeof(float) : 2U;
    const bool strideValid =
        FitsDmaSourceStride(info.qkHeadNum, PREPARE_K_DIM * 2U) &&
        FitsDmaSourceStride(info.valueHeadNum, PREPARE_V_DIM * 2U) &&
        FitsDmaSourceStride(info.valueHeadNum,
                            PREPARE_K_DIM * gateElementBytes) &&
        FitsDmaSourceStride(info.valueHeadNum, betaElementBytes);
    if (!strideValid) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "HK,HV",
            "HK=" + std::to_string(info.qkHeadNum) + ",HV=" + std::to_string(info.valueHeadNum), "DMA stride in uint32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShape(const ChunkKdaFwdPrepareParams &params,
                         PrepareLayout layout, PrepareShapeInfo &info)
{
    if (!SameShape(params.q, params.k)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "q,k", "mismatch", "identical");
        return ACLNN_ERR_PARAM_INVALID;
    }
    info.packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
    const size_t matrixRank = info.packed ? 3 : 4;
    const size_t scalarRank = info.packed ? 2 : 3;
    if (!(Rank(params.q) == matrixRank && Rank(params.v) == matrixRank && Rank(params.g) == matrixRank && Rank(params.beta) == scalarRank)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(PREPARE_ENTITY, "q,k,v,g,beta", "rank mismatch", "layout-dependent rank");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (layout == PrepareLayout::TND) {
        info.batch = 1;
        info.seqLen = Dim(params.q, 0);
        info.qkHeadNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.valueHeadNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 2);
        if (
            !(HasShape(params.v, {info.seqLen, info.valueHeadNum, info.vDim}) &&
              HasShape(params.g, {info.seqLen, info.valueHeadNum, info.kDim}) &&
              HasShape(params.beta, {info.seqLen, info.valueHeadNum}))) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "v,g,beta", "mismatch", "[T,HV,V],[T,HV,K],[T,HV]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (layout == PrepareLayout::NTD) {
        info.batch = 1;
        info.qkHeadNum = Dim(params.q, 0);
        info.seqLen = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.valueHeadNum = Dim(params.v, 0);
        info.vDim = Dim(params.v, 2);
        if (
            !(HasShape(params.v, {info.valueHeadNum, info.seqLen, info.vDim}) &&
              HasShape(params.g, {info.valueHeadNum, info.seqLen, info.kDim}) &&
              HasShape(params.beta, {info.valueHeadNum, info.seqLen}))) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "v,g,beta", "mismatch", "[HV,T,V],[HV,T,K],[HV,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (layout == PrepareLayout::BSND) {
        info.batch = Dim(params.q, 0);
        info.seqLen = Dim(params.q, 1);
        info.qkHeadNum = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.valueHeadNum = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        if (
            !(HasShape(params.v, {info.batch, info.seqLen, info.valueHeadNum, info.vDim}) &&
              HasShape(params.g, {info.batch, info.seqLen, info.valueHeadNum, info.kDim}) &&
              HasShape(params.beta, {info.batch, info.seqLen, info.valueHeadNum}))) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "v,g,beta", "mismatch", "[B,T,HV,V],[B,T,HV,K],[B,T,HV]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        info.batch = Dim(params.q, 0);
        info.qkHeadNum = Dim(params.q, 1);
        info.seqLen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.valueHeadNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        if (
            !(HasShape(params.v, {info.batch, info.valueHeadNum, info.seqLen, info.vDim}) &&
              HasShape(params.g, {info.batch, info.valueHeadNum, info.seqLen, info.kDim}) &&
              HasShape(params.beta, {info.batch, info.valueHeadNum, info.seqLen}))) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "v,g,beta", "mismatch", "[B,HV,T,V],[B,HV,T,K],[B,HV,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (!(FitsUint32(info.batch) && FitsUint32(info.seqLen) && FitsUint32(info.qkHeadNum) && FitsUint32(info.valueHeadNum))) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "B,T,HK,HV",
            "B=" + std::to_string(info.batch) + ",T=" + std::to_string(info.seqLen) + ",HK=" + std::to_string(info.qkHeadNum) + ",HV=" + std::to_string(info.valueHeadNum), "positive uint32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.kDim == PREPARE_K_DIM && info.vDim == PREPARE_V_DIM)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(PREPARE_ENTITY, "K,V",
            "K=" + std::to_string(info.kDim) + ",V=" + std::to_string(info.vDim), "K=V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.valueHeadNum >= info.qkHeadNum && info.valueHeadNum % info.qkHeadNum == 0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(PREPARE_ENTITY, "HK,HV",
            "HK=" + std::to_string(info.qkHeadNum) + ",HV=" + std::to_string(info.valueHeadNum), "HV>=HK and HV%HK==0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckModesAndOptionalInputs(
    const ChunkKdaFwdPrepareParams &params, const PrepareShapeInfo &info)
{
    if (params.chunkSize != PREPARE_CHUNK_SIZE) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "chunkSize", std::to_string(params.chunkSize), "64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Tiling 和 kernel 均以 FP32 保存这些属性，必须按实际落盘精度校验。
    const float scaleFp32 = static_cast<float>(params.scale);
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    const float lowerBoundFp32 = static_cast<float>(params.lowerBound);
    if (!(std::isfinite(scaleFp32) && std::isfinite(epsilonFp32) && epsilonFp32 > 0.0F && std::isfinite(lowerBoundFp32))) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "scale/epsilon/lowerBound", "non-finite", "finite, epsilon>0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.allowNegEigval && !params.useBetaSigmoidInKernel) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "allowNegEigval", "true", "requires useBetaSigmoidInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!params.useGateInKernel) {
        if (!(params.aLogOptional == nullptr && params.dtBiasOptional == nullptr && !params.safeGate)) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "aLogOptional/dtBiasOptional/safeGate", "not empty",
                "must be empty when useGateInKernel=false");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (params.aLogOptional == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_ENTITY, "aLogOptional");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!HasShape(params.aLogOptional, {info.valueHeadNum})) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "aLogOptional", "mismatch", "[HV=" + std::to_string(info.valueHeadNum) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dtBiasOptional != nullptr) {
        if (!HasShape(params.dtBiasOptional, {info.valueHeadNum * info.kDim})) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "dtBiasOptional", "mismatch",
                "[HV*K=" + std::to_string(info.valueHeadNum * info.kDim) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.safeGate) {
        if (!(lowerBoundFp32 >= -5.0F && lowerBoundFp32 < 0.0F)) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "lowerBound", std::to_string(params.lowerBound), "[-5,0)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

int64_t CountChunks(const aclIntArray *cuSeqlens, int64_t seqLen,
                    int64_t chunkSize)
{
    if (cuSeqlens == nullptr) {
        return (seqLen + chunkSize - 1) / chunkSize;
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        totalChunks += (length + chunkSize - 1) / chunkSize;
    }
    return totalChunks;
}

aclnnStatus CheckVariableLengthInputs(
    const ChunkKdaFwdPrepareParams &params, const PrepareShapeInfo &info)
{
    if (params.cuSeqlensOptional == nullptr) {
        if (params.chunkIndicesOptional != nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "chunkIndicesOptional", "present", "requires cuSeqlensOptional");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (!(info.packed || info.batch == 1)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "B", std::to_string(info.batch), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional->Size() < 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(PREPARE_ENTITY, "cuSeqlensOptional",
            std::to_string(params.cuSeqlensOptional->Size()), ">=2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[0] != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "cuSeqlensOptional[0]", std::to_string((*params.cuSeqlensOptional)[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] != info.seqLen) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "cuSeqlensOptional[last]",
            std::to_string((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1]), std::to_string(info.seqLen));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        if ((*params.cuSeqlensOptional)[seq] > (*params.cuSeqlensOptional)[seq + 1]) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "cuSeqlensOptional", "non-monotonic", "nondecreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks = CountChunks(
        params.cuSeqlensOptional, info.seqLen, params.chunkSize);
    if (params.chunkIndicesOptional->Size() != static_cast<size_t>(totalChunks) * 2) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(PREPARE_ENTITY, "chunkIndicesOptional",
            std::to_string(params.chunkIndicesOptional->Size()), std::to_string(totalChunks * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        const int64_t length = (*params.cuSeqlensOptional)[seq + 1] -
            (*params.cuSeqlensOptional)[seq];
        const int64_t chunks =
            (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if (!((*params.chunkIndicesOptional)[offset] == static_cast<int64_t>(seq) &&
                  (*params.chunkIndicesOptional)[offset + 1] == chunk)) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(PREPARE_ENTITY, "chunkIndicesOptional", "non-canonical", "sequence-major (seq_id,local_chunk_id)");
                return ACLNN_ERR_PARAM_INVALID;
            }
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const ChunkKdaFwdPrepareParams &params,
                              const PrepareShapeInfo &info)
{
    const auto valueMatrixValid = [&](const aclTensor *tensor,
                                      int64_t dimension) {
        return info.packed
            ? HasShape(tensor,
                       {info.valueHeadNum, info.seqLen, dimension})
            : HasShape(tensor, {info.batch, info.valueHeadNum, info.seqLen,
                                dimension});
    };
    const auto qkMatrixValid = [&](const aclTensor *tensor) {
        return info.packed
            ? HasShape(tensor, {info.qkHeadNum, info.seqLen, info.kDim})
            : HasShape(tensor, {info.batch, info.qkHeadNum, info.seqLen,
                                info.kDim});
    };
    const auto valueScalarValid = [&](const aclTensor *tensor) {
        return info.packed
            ? HasShape(tensor, {info.valueHeadNum, info.seqLen})
            : HasShape(tensor,
                       {info.batch, info.valueHeadNum, info.seqLen});
    };
    const auto qkScalarValid = [&](const aclTensor *tensor) {
        return info.packed
            ? HasShape(tensor, {info.qkHeadNum, info.seqLen})
            : HasShape(tensor,
                       {info.batch, info.qkHeadNum, info.seqLen});
    };

    if (!valueMatrixValid(params.gkOut, info.kDim)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "gkOut", "mismatch", "head-major [B,HV,T,K]/[HV,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!valueMatrixValid(params.aqkOut, params.chunkSize)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "aqkOut", "mismatch", "head-major [B,HV,T,64]/[HV,T,64]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.akkOut != nullptr && !valueMatrixValid(params.akkOut, params.chunkSize)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "akkOut", "mismatch", "head-major [B,HV,T,64]/[HV,T,64]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(valueMatrixValid(params.wOut, info.kDim) && valueMatrixValid(params.kgOut, info.kDim) &&
          valueMatrixValid(params.qgScaledOut, info.kDim))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "wOut,kgOut,qgScaledOut", "mismatch", "head-major [B,HV,T,K]/[HV,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.qgOut != nullptr && !valueMatrixValid(params.qgOut, info.kDim)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "qgOut", "mismatch", "head-major [B,HV,T,K]/[HV,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!valueMatrixValid(params.uOut, info.vDim)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "uOut", "mismatch", "head-major [B,HV,T,V]/[HV,T,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((params.qHatOut == nullptr || qkMatrixValid(params.qHatOut)) &&
          (params.kHatOut == nullptr || qkMatrixValid(params.kHatOut)))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "qHatOut,kHatOut", "mismatch", "head-major [B,HK,T,K]/[HK,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((params.qRstdOut == nullptr || qkScalarValid(params.qRstdOut)) &&
          (params.kRstdOut == nullptr || qkScalarValid(params.kRstdOut)))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "qRstdOut,kRstdOut", "mismatch", "head-major [B,HK,T]/[HK,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.betaEffOut != nullptr && !valueScalarValid(params.betaEffOut)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(PREPARE_ENTITY, "betaEffOut", "mismatch", "head-major [B,HV,T]/[HV,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkKdaFwdPrepareParams &params)
{
    aclnnStatus status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckOutputMode(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    PrepareLayout layout;
    status = ParseLayout(params.layout, layout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    PrepareShapeInfo info;
    status = ResolveShape(params, layout, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckModesAndOptionalInputs(params, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckVariableLengthInputs(params, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckDtype(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSequenceMajorDmaStride(params, layout, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckOutputShapes(params, info);
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus MakeInputsContiguous(ChunkKdaFwdPrepareParams &params,
                                 aclOpExecutor *executor)
{
    const aclTensor **inputs[] = {
        &params.q, &params.k, &params.v, &params.g, &params.beta,
        &params.aLogOptional, &params.dtBiasOptional};
    for (const aclTensor **input : inputs) {
        const aclnnStatus status = MakeContiguous(*input, executor);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKdaFwdPrepareGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool useExp2,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *qgScaledOut,
    const aclTensor *qHatOut,
    const aclTensor *kHatOut,
    const aclTensor *qRstdOut,
    const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_COMPAT(PREPARE_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    ChunkKdaFwdPrepareParams params{
        q,          k,         v,          g,          beta,
        aLogOptional, dtBiasOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        layout,     scale,     chunkSize,  epsilon,
        useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel,
        allowNegEigval, safeGate, lowerBound, useExp2,
        gkOut,      aqkOut,    akkOut,     wOut,       uOut,
        qgOut,      kgOut,     qgScaledOut,
        qHatOut,    kHatOut,   qRstdOut,   kRstdOut,   betaEffOut};

    L2_DFX_PHASE_1(
        aclnnChunkKdaFwdPrepare,
        DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional,
               cuSeqlensOptional, chunkIndicesOptional, layout, scale,
               chunkSize, epsilon, useQkL2normInKernel, useGateInKernel,
               useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound,
               useExp2),
        DFX_OUT(gkOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut,
                qgScaledOut, qHatOut, kHatOut, qRstdOut, kRstdOut,
                betaEffOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    aclnnStatus status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeInputsContiguous(params, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const auto result = l0op::ChunkKdaFwdPrepare(
        params.q, params.k, params.v, params.g, params.beta,
        params.aLogOptional, params.dtBiasOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.layout,
        params.scale, params.chunkSize, params.epsilon,
        params.useQkL2normInKernel, params.useGateInKernel,
        params.useBetaSigmoidInKernel, params.allowNegEigval,
        params.safeGate, params.lowerBound, params.useExp2,
        params.gkOut, params.aqkOut, params.akkOut, params.wOut, params.uOut,
        params.qgOut, params.kgOut, params.qgScaledOut, params.qHatOut,
        params.kHatOut, params.qRstdOut, params.kRstdOut, params.betaEffOut,
        GetOutputMode(params), executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr &&
                  result[3] != nullptr && result[4] != nullptr &&
                  result[6] != nullptr && result[7] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwdPrepare);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaFwdPrepare AI Core 启动失败。");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
