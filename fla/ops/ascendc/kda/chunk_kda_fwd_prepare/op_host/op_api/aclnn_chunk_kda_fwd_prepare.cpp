/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
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

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkKdaFwdPrepare";

namespace {

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
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "layout");
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "layout", layout, "layout must be BNSD/BSND/NTD/TND");
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
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, inputNames[index], "required input must not be nullptr");
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
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, outputNames[index], "required output must not be nullptr");
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "outputMask", std::to_string(GetOutputMask(params)),
            "nullptr output combination must be one of none/forward/recompute/save");
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
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                OP_NAME, names[index],
                std::to_string(static_cast<int>(storageFormat)) + "/" +
                    std::to_string(static_cast<int>(viewFormat)),
                "non-private ND/NCHW/NCL/NHWC format only");
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
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, outputNames[index], "output is written by kernel and must be contiguous");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const ChunkKdaFwdPrepareParams &params)
{
    if (!(params.q->GetDataType() == DataType::DT_BF16 &&
          params.k->GetDataType() == DataType::DT_BF16 &&
          params.v->GetDataType() == DataType::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "q,k,v",
            "q=" + ge::TypeUtils::DataTypeToSerialString(params.q->GetDataType()) +
                ",k=" + ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()) +
                ",v=" + ge::TypeUtils::DataTypeToSerialString(params.v->GetDataType()),
            "q/k/v support BF16 only");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto gateDtype = params.g->GetDataType();
    const auto betaDtype = params.beta->GetDataType();
    if (gateDtype != DataType::DT_BF16 && gateDtype != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "g", ge::TypeUtils::DataTypeToSerialString(gateDtype),
            "g supports BF16 or FP32 only");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (betaDtype != DataType::DT_BF16 && betaDtype != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "beta", ge::TypeUtils::DataTypeToSerialString(betaDtype),
            "beta supports BF16 or FP32 only");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.aLogOptional != nullptr) {
        if (params.aLogOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "aLogOptional",
                ge::TypeUtils::DataTypeToSerialString(params.aLogOptional->GetDataType()),
                "aLogOptional supports FP32 only");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dtBiasOptional != nullptr) {
        if (params.dtBiasOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "dtBiasOptional",
                ge::TypeUtils::DataTypeToSerialString(params.dtBiasOptional->GetDataType()),
                "dtBiasOptional supports FP32 only");
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
        if (!(output->GetDataType() == DataType::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "output", ge::TypeUtils::DataTypeToSerialString(output->GetDataType()),
            "Aqk/Akk/w/u/qg/kg/qgScaled/qHat/kHat 输出必须为 BF16");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    }
    const aclTensor *fp32Outputs[] = {
        params.gkOut, params.qRstdOut, params.kRstdOut, params.betaEffOut};
    for (const aclTensor *output : fp32Outputs) {
        if (output == nullptr) {
            continue;
        }
        if (!(output->GetDataType() == DataType::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "output", ge::TypeUtils::DataTypeToSerialString(output->GetDataType()),
            "gk/qRstd/kRstd/betaEff 输出必须为 FP32");
        return ACLNN_ERR_PARAM_NULLPTR;
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
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "q,k,v,g,beta",
            "cross-head DMA stride exceeds uint32: HK=" + std::to_string(info.qkHeadNum) +
                ",HV=" + std::to_string(info.valueHeadNum));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShape(const ChunkKdaFwdPrepareParams &params,
                         PrepareLayout layout, PrepareShapeInfo &info)
{
    if (!SameShape(params.q, params.k)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q,k",
            "q=" + std::string(op::ToString(params.q->GetViewShape()).GetString()) +
                ",k=" + std::string(op::ToString(params.k->GetViewShape()).GetString()),
            "q and k must have identical shape");
        return ACLNN_ERR_PARAM_INVALID;
    }
    info.packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
    const size_t matrixRank = info.packed ? 3 : 4;
    const size_t scalarRank = info.packed ? 2 : 3;
    if (!(Rank(params.q) == matrixRank && Rank(params.v) == matrixRank &&
          Rank(params.g) == matrixRank && Rank(params.beta) == scalarRank)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q,k,v,g,beta",
            "q=" + std::to_string(Rank(params.q)) + ",k=" + std::to_string(Rank(params.k)) +
                ",v=" + std::to_string(Rank(params.v)) + ",g=" + std::to_string(Rank(params.g)) +
                ",beta=" + std::to_string(Rank(params.beta)),
            "ranks must be 4/4/4/4/3 for BNSD/BSND or 3/3/3/3/2 for NTD/TND");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (layout == PrepareLayout::TND) {
        info.batch = 1;
        info.seqLen = Dim(params.q, 0);
        info.qkHeadNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.valueHeadNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 2);
        if (!(HasShape(params.v, {info.seqLen, info.valueHeadNum, info.vDim}) &&
          HasShape(params.g, {info.seqLen, info.valueHeadNum, info.kDim}) &&
          HasShape(params.beta, {info.seqLen, info.valueHeadNum}))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "v,g,beta", "shape mismatch",
            "TND 要求 v/g/beta 分别为 [T,HV,V]、[T,HV,K]、[T,HV]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    } else if (layout == PrepareLayout::NTD) {
        info.batch = 1;
        info.qkHeadNum = Dim(params.q, 0);
        info.seqLen = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.valueHeadNum = Dim(params.v, 0);
        info.vDim = Dim(params.v, 2);
        if (!(HasShape(params.v, {info.valueHeadNum, info.seqLen, info.vDim}) &&
          HasShape(params.g, {info.valueHeadNum, info.seqLen, info.kDim}) &&
          HasShape(params.beta, {info.valueHeadNum, info.seqLen}))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "v,g,beta", "shape mismatch",
            "NTD 要求 v/g/beta 分别为 [HV,T,V]、[HV,T,K]、[HV,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    } else if (layout == PrepareLayout::BSND) {
        info.batch = Dim(params.q, 0);
        info.seqLen = Dim(params.q, 1);
        info.qkHeadNum = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.valueHeadNum = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        if (!(HasShape(params.v, {info.batch, info.seqLen, info.valueHeadNum, info.vDim}) &&
          HasShape(params.g, {info.batch, info.seqLen, info.valueHeadNum, info.kDim}) &&
          HasShape(params.beta, {info.batch, info.seqLen, info.valueHeadNum}))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "v,g,beta", "shape mismatch",
            "BSND 要求 v/g/beta 分别为 [B,T,HV,V]、[B,T,HV,K]、[B,T,HV]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    } else {
        info.batch = Dim(params.q, 0);
        info.qkHeadNum = Dim(params.q, 1);
        info.seqLen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.valueHeadNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        if (!(HasShape(params.v, {info.batch, info.valueHeadNum, info.seqLen, info.vDim}) &&
          HasShape(params.g, {info.batch, info.valueHeadNum, info.seqLen, info.kDim}) &&
          HasShape(params.beta, {info.batch, info.valueHeadNum, info.seqLen}))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "v,g,beta", "shape mismatch",
            "BNSD 要求 v/g/beta 分别为 [B,HV,T,V]、[B,HV,T,K]、[B,HV,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    }

    if (!(FitsUint32(info.batch) && FitsUint32(info.seqLen) &&
          FitsUint32(info.qkHeadNum) && FitsUint32(info.valueHeadNum))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "B,T,HK,HV",
            std::to_string(info.batch) + "," + std::to_string(info.seqLen) + "," +
                std::to_string(info.qkHeadNum) + "," + std::to_string(info.valueHeadNum),
            "B/T/HK/HV must be positive and representable by uint32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.kDim != PREPARE_K_DIM || info.vDim != PREPARE_V_DIM) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "q.K,v.V",
            std::to_string(info.kDim) + "," + std::to_string(info.vDim),
            "K=V=128 is required by the current implementation");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (info.valueHeadNum < info.qkHeadNum || info.valueHeadNum % info.qkHeadNum != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "q.HK,v.HV",
            std::to_string(info.qkHeadNum) + "," + std::to_string(info.valueHeadNum),
            "GVA requires HV>=HK and HV divisible by HK");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckModesAndOptionalInputs(
    const ChunkKdaFwdPrepareParams &params, const PrepareShapeInfo &info)
{
    if (params.chunkSize != PREPARE_CHUNK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunkSize", std::to_string(params.chunkSize), "only 64 is supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Tiling 和 kernel 均以 FP32 保存这些属性，必须按实际落盘精度校验。
    const float scaleFp32 = static_cast<float>(params.scale);
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    const float lowerBoundFp32 = static_cast<float>(params.lowerBound);
    if (!(std::isfinite(scaleFp32) && std::isfinite(epsilonFp32) &&
          epsilonFp32 > 0.0F && std::isfinite(lowerBoundFp32))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "scale,epsilon,lowerBound",
            std::to_string(scaleFp32) + "," + std::to_string(epsilonFp32) + "," +
                std::to_string(lowerBoundFp32),
            "scale and lowerBound must be finite in FP32; epsilon must be positive and finite");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.allowNegEigval && !params.useBetaSigmoidInKernel) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "allowNegEigval", "true",
            "allowNegEigval=true requires useBetaSigmoidInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!params.useGateInKernel) {
        if (!(params.aLogOptional == nullptr && params.dtBiasOptional == nullptr &&
              !params.safeGate)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "aLogOptional,dtBiasOptional,safeGate",
                std::string(params.aLogOptional == nullptr ? "aLog=absent" : "aLog=present") +
                    "," + (params.dtBiasOptional == nullptr ? "dtBias=absent" : "dtBias=present") +
                    "," + (params.safeGate ? "safeGate=true" : "safeGate=false"),
                "aLogOptional/dtBiasOptional must be absent and safeGate false when useGateInKernel=false");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (params.aLogOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "aLogOptional");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!HasShape(params.aLogOptional, {info.valueHeadNum})) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "aLogOptional",
            std::string(op::ToString(params.aLogOptional->GetViewShape()).GetString()),
            "[HV] with HV=" + std::to_string(info.valueHeadNum));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dtBiasOptional != nullptr) {
        if (!HasShape(params.dtBiasOptional, {info.valueHeadNum * info.kDim})) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "dtBiasOptional",
                std::string(op::ToString(params.dtBiasOptional->GetViewShape()).GetString()),
                "flattened [HV*K] with HV=" + std::to_string(info.valueHeadNum) +
                    ",K=" + std::to_string(info.kDim));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.safeGate) {
        if (!(lowerBoundFp32 >= -5.0F && lowerBoundFp32 < 0.0F)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "lowerBound", std::to_string(lowerBoundFp32),
                "lowerBound must be in [-5,0) when safeGate=true");
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
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "chunkIndicesOptional");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        return ACLNN_SUCCESS;
    }
    if (!(info.packed || info.batch == 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "B", std::to_string(info.batch), "rank-4 varlen input requires B=1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional->Size() < 2) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "cuSeqlensOptional",
            std::to_string(params.cuSeqlensOptional->Size()), ">= 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[0] != 0) {
        OP_LOGE_FOR_INVALID_VALUE(
            OP_NAME, "cuSeqlensOptional[0]",
            std::to_string((*params.cuSeqlensOptional)[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] != info.seqLen) {
        OP_LOGE_FOR_INVALID_VALUE(
            OP_NAME, "cuSeqlensOptional[last]",
            std::to_string((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1]),
            std::to_string(info.seqLen));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        if ((*params.cuSeqlensOptional)[seq] > (*params.cuSeqlensOptional)[seq + 1]) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "cuSeqlensOptional[seq],cuSeqlensOptional[seq+1]",
                std::to_string((*params.cuSeqlensOptional)[seq]) + "," +
                    std::to_string((*params.cuSeqlensOptional)[seq + 1]),
                "cuSeqlensOptional must be nondecreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks = CountChunks(
        params.cuSeqlensOptional, info.seqLen, params.chunkSize);
    if (params.chunkIndicesOptional->Size() != static_cast<size_t>(totalChunks) * 2) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "chunkIndicesOptional",
            std::to_string(params.chunkIndicesOptional->Size()),
            std::to_string(static_cast<size_t>(totalChunks) * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        const int64_t length = (*params.cuSeqlensOptional)[seq + 1] -
            (*params.cuSeqlensOptional)[seq];
        const int64_t chunks =
            (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if ((*params.chunkIndicesOptional)[offset] != static_cast<int64_t>(seq) ||
                (*params.chunkIndicesOptional)[offset + 1] != chunk) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    OP_NAME, "chunkIndicesOptional[offset],chunkIndicesOptional[offset+1]",
                    std::to_string((*params.chunkIndicesOptional)[offset]) + "," +
                        std::to_string((*params.chunkIndicesOptional)[offset + 1]),
                    "chunkIndicesOptional must store canonical sequence-major (seq_id,local_chunk_id)");
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
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "gkOut",
            std::string(op::ToString(params.gkOut->GetViewShape()).GetString()),
            "fixed head-major [B,HV,T,K]/[HV,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!valueMatrixValid(params.aqkOut, params.chunkSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "aqkOut",
            std::string(op::ToString(params.aqkOut->GetViewShape()).GetString()),
            "fixed head-major [B,HV,T,64]/[HV,T,64]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.akkOut != nullptr && !valueMatrixValid(params.akkOut, params.chunkSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "akkOut",
            std::string(op::ToString(params.akkOut->GetViewShape()).GetString()),
            "fixed head-major [B,HV,T,64]/[HV,T,64]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(valueMatrixValid(params.wOut, info.kDim) &&
          valueMatrixValid(params.kgOut, info.kDim) &&
          valueMatrixValid(params.qgScaledOut, info.kDim))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "wOut,kgOut,qgScaledOut",
            "wOut=" + std::string(op::ToString(params.wOut->GetViewShape()).GetString()) +
                ",kgOut=" + std::string(op::ToString(params.kgOut->GetViewShape()).GetString()) +
                ",qgScaledOut=" + std::string(op::ToString(params.qgScaledOut->GetViewShape()).GetString()),
            "fixed head-major [B,HV,T,K]/[HV,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.qgOut != nullptr && !valueMatrixValid(params.qgOut, info.kDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "qgOut",
            std::string(op::ToString(params.qgOut->GetViewShape()).GetString()),
            "fixed head-major [B,HV,T,K]/[HV,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!valueMatrixValid(params.uOut, info.vDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "uOut",
            std::string(op::ToString(params.uOut->GetViewShape()).GetString()),
            "fixed head-major [B,HV,T,V]/[HV,T,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((params.qHatOut != nullptr && !qkMatrixValid(params.qHatOut)) ||
        (params.kHatOut != nullptr && !qkMatrixValid(params.kHatOut))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "qHatOut,kHatOut",
            std::string(params.qHatOut == nullptr ? "qHat=absent" :
                        "qHat=" + std::string(op::ToString(params.qHatOut->GetViewShape()).GetString())) +
                "," + (params.kHatOut == nullptr ? "kHat=absent" :
                       "kHat=" + std::string(op::ToString(params.kHatOut->GetViewShape()).GetString())),
            "fixed head-major [B,HK,T,K]/[HK,T,K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((params.qRstdOut != nullptr && !qkScalarValid(params.qRstdOut)) ||
        (params.kRstdOut != nullptr && !qkScalarValid(params.kRstdOut))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "qRstdOut,kRstdOut",
            std::string(params.qRstdOut == nullptr ? "qRstd=absent" :
                        "qRstd=" + std::string(op::ToString(params.qRstdOut->GetViewShape()).GetString())) +
                "," + (params.kRstdOut == nullptr ? "kRstd=absent" :
                       "kRstd=" + std::string(op::ToString(params.kRstdOut->GetViewShape()).GetString())),
            "fixed head-major [B,HK,T]/[HK,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.betaEffOut != nullptr && !valueScalarValid(params.betaEffOut)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "betaEffOut",
            std::string(op::ToString(params.betaEffOut->GetViewShape()).GetString()),
            "fixed head-major [B,HV,T]/[HV,T]");
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
    if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
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
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
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
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
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
    if (!(result[0] != nullptr && result[1] != nullptr && result[3] != nullptr && result[4] != nullptr && result[6] != nullptr && result[7] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: result[0] != nullptr && result[1] != nullptr && result[3] != nullptr && result[4] != nullptr && result[6] != nullptr && result[7] != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

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
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkKdaFwdPrepare AI Core 启动失败。", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
