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
#include <sstream>
#include <string>
#include "graph/utils/type_utils.h"

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

#include "../chunk_kda_fwd_prepare_error.h"

using namespace op;

namespace {

constexpr const char *OP_NAME = "aclnnChunkKdaFwdPrepare";

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

template <typename T>
std::string ToText(T value)
{
    std::ostringstream text;
    text << value;
    return text.str();
}

std::string DimsText(std::initializer_list<int64_t> dims)
{
    std::string text = "[";
    for (int64_t dim : dims) {
        if (text.size() > 1)
            text += ',';
        text += ToText(dim);
    }
    return text + ']';
}

std::string ShapeText(const aclTensor *tensor)
{
    if (tensor == nullptr)
        return "null";
    const auto &shape = tensor->GetViewShape();
    std::string text = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0)
            text += ',';
        text += ToText(shape.GetDim(i));
    }
    return text + ']';
}

std::string TensorShapes(std::initializer_list<const aclTensor *> tensors)
{
    std::string text;
    for (const auto *tensor : tensors) {
        if (!text.empty())
            text += ", ";
        text += ShapeText(tensor);
    }
    return text;
}

std::string DtypeText(const aclTensor *tensor)
{
    return ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType());
}

std::string LogicalText(const PrepareShapeInfo &info)
{
    return "B=" + ToText(info.batch) + ",T=" + ToText(info.seqLen) + ",HK=" + ToText(info.qkHeadNum) +
           ",HV=" + ToText(info.valueHeadNum) + ",K=" + ToText(info.kDim) + ",V=" + ToText(info.vDim);
}

std::string ValueMatrixText(const PrepareShapeInfo &info, int64_t dimension)
{
    return info.packed ? DimsText({info.valueHeadNum, info.seqLen, dimension}) :
                         DimsText({info.batch, info.valueHeadNum, info.seqLen, dimension});
}

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
    if (!(layout != nullptr)) {
        KdaPrepareDfx::Missing(OP_NAME, "layout",
                                                 "required pointer is null; pass BNSD, BSND, NTD or TND");
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
        if (!(false)) {
            KdaPrepareDfx::Format(OP_NAME, "layout", layout, "BNSD, BSND, NTD or TND (case-sensitive)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNotNull(const ChunkKdaFwdPrepareParams &params)
{
    const aclTensor *inputs[] = {params.q, params.k, params.v, params.g,
                                 params.beta};
    const char *inputNames[] = {"q", "k", "v", "g", "beta"};
    for (size_t index = 0; index < sizeof(inputs) / sizeof(inputs[0]); ++index) {
        if (!(inputs[index] != nullptr)) {
            KdaPrepareDfx::Missing(OP_NAME, inputNames[index],
                                                     "required input tensor is null; provide a valid aclTensor");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }

    const aclTensor *outputs[] = {
        params.gkOut, params.aqkOut, params.wOut,
        params.uOut, params.kgOut, params.qgScaledOut};
    const char *outputNames[] = {
        "gkOut", "aqkOut", "wOut", "uOut", "kgOut", "qgScaledOut"};
    for (size_t index = 0; index < sizeof(outputs) / sizeof(outputs[0]); ++index) {
        if (!(outputs[index] != nullptr)) {
            KdaPrepareDfx::Missing(
                OP_NAME, outputNames[index],
                "required output tensor is null; allocate this output in all four output modes");
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
    if (!(GetOutputMode(params) >= 0)) {
        KdaPrepareDfx::ValuesReason(
            OP_NAME, "output pointers", ToText(GetOutputMask(params)),
            "outputMask must match none=219, forward=223, recompute=8159 or save=8191; keep the 13 ABI slots");
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
        if (!(!IsPrivateFormat(storageFormat) && !IsPrivateFormat(viewFormat))) {
            KdaPrepareDfx::FormatsReason(OP_NAME, names[index],
                                                    "storage=" + ToText(static_cast<int>(storageFormat)) +
                                                        ",view=" + ToText(static_cast<int>(viewFormat)),
                                                    "use non-private storage/view formats such as ND/NCHW/NCL/NHWC; "
                                                    "private fractal formats are unsupported");
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
        if (!(IsContiguous(outputs[index]))) {
            KdaPrepareDfx::ArgumentReason(
                OP_NAME, outputNames[index],
                "output shape=" + ShapeText(outputs[index]) +
                    " is not contiguous; allocate contiguous output storage because the kernel writes it directly");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const ChunkKdaFwdPrepareParams &params)
{
    struct TensorDtype {
        const aclTensor *tensor;
        const char *name;
        bool bf16;
        bool fp32;
    };
    const TensorDtype tensors[] = {
        {params.q, "q", true, false},
        {params.k, "k", true, false},
        {params.v, "v", true, false},
        {params.g, "g", true, true},
        {params.beta, "beta", true, true},
        {params.aLogOptional, "a_log", false, true},
        {params.dtBiasOptional, "dt_bias", false, true},
        {params.aqkOut, "aqkOut", true, false},
        {params.akkOut, "akkOut", true, false},
        {params.wOut, "wOut", true, false},
        {params.uOut, "uOut", true, false},
        {params.qgOut, "qgOut", true, false},
        {params.kgOut, "kgOut", true, false},
        {params.qgScaledOut, "qgScaledOut", true, false},
        {params.qHatOut, "qHatOut", true, false},
        {params.kHatOut, "kHatOut", true, false},
        {params.gkOut, "gkOut", false, true},
        {params.qRstdOut, "qRstdOut", false, true},
        {params.kRstdOut, "kRstdOut", false, true},
        {params.betaEffOut, "betaEffOut", false, true},
    };
    for (const auto &item : tensors) {
        if (item.tensor == nullptr)
            continue;
        const auto dtype = item.tensor->GetDataType();
        if (!((item.bf16 && dtype == DataType::DT_BF16) || (item.fp32 && dtype == DataType::DT_FLOAT))) {
            KdaPrepareDfx::Dtype(OP_NAME, item.name, DtypeText(item.tensor),
                                      item.bf16 ? (item.fp32 ? "BF16 or FLOAT (FP32)" : "BF16") : "FLOAT (FP32)");
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
    if (!(strideValid)) {
        KdaPrepareDfx::ValuesReason(
            OP_NAME, "HK,HV", ToText(info.qkHeadNum) + "," + ToText(info.valueHeadNum),
            "layout=" + std::string(params.layout) +
                "; each (heads-1)*rowBytes DMA source stride must fit uint32; reduce the head counts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShape(const ChunkKdaFwdPrepareParams &params,
                         PrepareLayout layout, PrepareShapeInfo &info)
{
    if (!(SameShape(params.q, params.k))) {
        KdaPrepareDfx::ShapesReason(OP_NAME, "q,k", TensorShapes({params.q, params.k}),
                                               "q and k must have identical shapes in layout=" +
                                                   std::string(params.layout));
        return ACLNN_ERR_PARAM_INVALID;
    }
    info.packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
    const size_t matrixRank = info.packed ? 3 : 4;
    const size_t scalarRank = info.packed ? 2 : 3;
    if (!(Rank(params.q) == matrixRank && Rank(params.v) == matrixRank && Rank(params.g) == matrixRank &&
          Rank(params.beta) == scalarRank)) {
        KdaPrepareDfx::RanksReason(OP_NAME, "q,v,g,beta",
                                                  ToText(Rank(params.q)) + "," + ToText(Rank(params.v)) + "," +
                                                      ToText(Rank(params.g)) + "," + ToText(Rank(params.beta)),
                                                  "layout=" + std::string(params.layout) + "; q/k/v/g rank=" +
                                                      ToText(matrixRank) + ", beta rank=" + ToText(scalarRank));
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
            KdaPrepareDfx::ShapesReason(OP_NAME, "v,g,beta", TensorShapes({params.v, params.g, params.beta}),
                                                   "layout=TND; expected [T,HV,V], [T,HV,K], [T,HV]; " +
                                                       LogicalText(info));
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
            KdaPrepareDfx::ShapesReason(OP_NAME, "v,g,beta", TensorShapes({params.v, params.g, params.beta}),
                                                   "layout=NTD; expected [HV,T,V], [HV,T,K], [HV,T]; " +
                                                       LogicalText(info));
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
            KdaPrepareDfx::ShapesReason(OP_NAME, "v,g,beta", TensorShapes({params.v, params.g, params.beta}),
                                                   "layout=BSND; expected [B,T,HV,V], [B,T,HV,K], [B,T,HV]; " +
                                                       LogicalText(info));
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
            KdaPrepareDfx::ShapesReason(OP_NAME, "v,g,beta", TensorShapes({params.v, params.g, params.beta}),
                                                   "layout=BNSD; expected [B,HV,T,V], [B,HV,T,K], [B,HV,T]; " +
                                                       LogicalText(info));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const int64_t dimensions[] = {info.batch, info.seqLen, info.qkHeadNum, info.valueHeadNum};
    const char *dimensionNames[] = {"B", "T", "HK", "HV"};
    for (size_t i = 0; i < 4; ++i) {
        if (!FitsUint32(dimensions[i])) {
            KdaPrepareDfx::ValueReason(OP_NAME, dimensionNames[i], ToText(dimensions[i]),
                                                  "must be in [1,4294967295]; layout=" + std::string(params.layout));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!(info.kDim == PREPARE_K_DIM && info.vDim == PREPARE_V_DIM)) {
        KdaPrepareDfx::Value(OP_NAME, info.kDim != PREPARE_K_DIM ? "K" : "V",
                                  ToText(info.kDim != PREPARE_K_DIM ? info.kDim : info.vDim), "128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.valueHeadNum >= info.qkHeadNum && info.valueHeadNum % info.qkHeadNum == 0)) {
        KdaPrepareDfx::ValuesReason(OP_NAME, "HK,HV",
                                               ToText(info.qkHeadNum) + "," + ToText(info.valueHeadNum),
                                               "GVA requires HV>=HK and HV%HK=0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckModesAndOptionalInputs(
    const ChunkKdaFwdPrepareParams &params, const PrepareShapeInfo &info)
{
    if (!(params.chunkSize == PREPARE_CHUNK_SIZE)) {
        KdaPrepareDfx::Value(OP_NAME, "chunk_size", ToText(params.chunkSize), "64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Tiling 和 kernel 均以 FP32 保存这些属性，必须按实际落盘精度校验。
    const float scaleFp32 = static_cast<float>(params.scale);
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    const float lowerBoundFp32 = static_cast<float>(params.lowerBound);
    if (!std::isfinite(scaleFp32)) {
        KdaPrepareDfx::ValueReason(OP_NAME, "scale", ToText(scaleFp32),
                                              "must remain finite after conversion to FP32; supplied=" +
                                                  ToText(params.scale));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(epsilonFp32) || epsilonFp32 <= 0.0F) {
        KdaPrepareDfx::ValueReason(OP_NAME, "epsilon", ToText(epsilonFp32),
                                              "must be positive and finite after conversion to FP32; supplied=" +
                                                  ToText(params.epsilon));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(lowerBoundFp32)) {
        KdaPrepareDfx::ValueReason(OP_NAME, "lower_bound", ToText(lowerBoundFp32),
                                              "must remain finite after conversion to FP32; supplied=" +
                                                  ToText(params.lowerBound));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(!params.allowNegEigval || params.useBetaSigmoidInKernel)) {
        KdaPrepareDfx::ValuesReason(OP_NAME, "allow_neg_eigval,use_beta_sigmoid_in_kernel",
                                               ToText(params.allowNegEigval) + "," +
                                                   ToText(params.useBetaSigmoidInKernel),
                                               "allow_neg_eigval=true requires use_beta_sigmoid_in_kernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!params.useGateInKernel) {
        if (!(params.aLogOptional == nullptr && params.dtBiasOptional == nullptr && !params.safeGate)) {
            KdaPrepareDfx::ValuesReason(OP_NAME, "use_gate_in_kernel,a_log,dt_bias,safe_gate",
                                                   "false," + ToText(params.aLogOptional != nullptr) + "," +
                                                       ToText(params.dtBiasOptional != nullptr) + "," +
                                                       ToText(params.safeGate),
                                                   "with gate disabled, a_log/dt_bias must be absent and safe_gate "
                                                   "must be false; pointer values describe presence");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (!(params.aLogOptional != nullptr)) {
        KdaPrepareDfx::Missing(
            OP_NAME, "a_log", "required when use_gate_in_kernel=true; provide a FP32 tensor with shape [HV]");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(HasShape(params.aLogOptional, {info.valueHeadNum}))) {
        KdaPrepareDfx::Shape(OP_NAME, "a_log", ShapeText(params.aLogOptional), DimsText({info.valueHeadNum}));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dtBiasOptional != nullptr) {
        if (!(HasShape(params.dtBiasOptional, {info.valueHeadNum * info.kDim}))) {
            KdaPrepareDfx::Shape(OP_NAME, "dt_bias", ShapeText(params.dtBiasOptional),
                                      DimsText({info.valueHeadNum * info.kDim}));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.safeGate) {
        if (!(lowerBoundFp32 >= -5.0F && lowerBoundFp32 < 0.0F)) {
            KdaPrepareDfx::ValueReason(OP_NAME, "lower_bound", ToText(lowerBoundFp32),
                                                  "safe_gate=true requires the FP32 value in [-5,0)");
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
        if (!(params.chunkIndicesOptional == nullptr)) {
            KdaPrepareDfx::ArgumentReason(
                OP_NAME, "chunk_indices",
                "provided while cu_seqlens is absent; provide cu_seqlens together with canonical chunk_indices");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (!(info.packed || info.batch == 1)) {
        KdaPrepareDfx::ValueReason(OP_NAME, "B", ToText(info.batch),
                                              "rank-4 variable-length input requires B=1; layout=" +
                                                  std::string(params.layout));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.cuSeqlensOptional->Size() >= 2)) {
        KdaPrepareDfx::ListSize(OP_NAME, "cu_seqlens", ToText(params.cuSeqlensOptional->Size()),
                                     ">=2 ([0,total_tokens])");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((*params.cuSeqlensOptional)[0] == 0)) {
        KdaPrepareDfx::Value(OP_NAME, "cu_seqlens[0]", ToText((*params.cuSeqlensOptional)[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] == info.seqLen)) {
        KdaPrepareDfx::Value(OP_NAME, "cu_seqlens[-1]",
                                  ToText((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1]),
                                  ToText(info.seqLen));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        if (!((*params.cuSeqlensOptional)[seq] <= (*params.cuSeqlensOptional)[seq + 1])) {
            KdaPrepareDfx::ValuesReason(
                OP_NAME, "cu_seqlens adjacent elements",
                ToText((*params.cuSeqlensOptional)[seq]) + "," + ToText((*params.cuSeqlensOptional)[seq + 1]),
                "must be nondecreasing; invalid pair starts at index=" + ToText(seq));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks = CountChunks(
        params.cuSeqlensOptional, info.seqLen, params.chunkSize);
    if (!(params.chunkIndicesOptional->Size() == static_cast<size_t>(totalChunks) * 2)) {
        KdaPrepareDfx::ListSize(OP_NAME, "chunk_indices", ToText(params.chunkIndicesOptional->Size()),
                                     ToText(totalChunks * 2));
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
                KdaPrepareDfx::ValuesReason(OP_NAME, "chunk_indices pair",
                                                       ToText((*params.chunkIndicesOptional)[offset]) + "," +
                                                           ToText((*params.chunkIndicesOptional)[offset + 1]),
                                                       "at offset=" + ToText(offset) +
                                                           ", expected canonical sequence-major pair=" + ToText(seq) +
                                                           "," + ToText(chunk));
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

    if (!(valueMatrixValid(params.gkOut, info.kDim))) {
        KdaPrepareDfx::Shape(OP_NAME, "gkOut", ShapeText(params.gkOut), ValueMatrixText(info, info.kDim));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(valueMatrixValid(params.aqkOut, params.chunkSize))) {
        KdaPrepareDfx::Shape(OP_NAME, "aqkOut", ShapeText(params.aqkOut), ValueMatrixText(info, params.chunkSize));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.akkOut == nullptr || valueMatrixValid(params.akkOut, params.chunkSize))) {
        KdaPrepareDfx::Shape(OP_NAME, "akkOut", ShapeText(params.akkOut), ValueMatrixText(info, params.chunkSize));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(valueMatrixValid(params.wOut, info.kDim) && valueMatrixValid(params.kgOut, info.kDim) &&
          valueMatrixValid(params.qgScaledOut, info.kDim))) {
        KdaPrepareDfx::ShapesReason(
            OP_NAME, "wOut,kgOut,qgScaledOut", TensorShapes({params.wOut, params.kgOut, params.qgScaledOut}),
            "each output must use fixed head-major shape=" + ValueMatrixText(info, info.kDim));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.qgOut == nullptr || valueMatrixValid(params.qgOut, info.kDim))) {
        KdaPrepareDfx::Shape(OP_NAME, "qgOut", ShapeText(params.qgOut), ValueMatrixText(info, info.kDim));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(valueMatrixValid(params.uOut, info.vDim))) {
        KdaPrepareDfx::Shape(OP_NAME, "uOut", ShapeText(params.uOut), ValueMatrixText(info, info.vDim));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((params.qHatOut == nullptr || qkMatrixValid(params.qHatOut)) &&
          (params.kHatOut == nullptr || qkMatrixValid(params.kHatOut)))) {
        KdaPrepareDfx::ShapesReason(
            OP_NAME, "qHatOut,kHatOut", TensorShapes({params.qHatOut, params.kHatOut}),
            "each non-null output must use fixed head-major shape=" +
                (info.packed ? DimsText({info.qkHeadNum, info.seqLen, info.kDim}) :
                               DimsText({info.batch, info.qkHeadNum, info.seqLen, info.kDim})));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((params.qRstdOut == nullptr || qkScalarValid(params.qRstdOut)) &&
          (params.kRstdOut == nullptr || qkScalarValid(params.kRstdOut)))) {
        KdaPrepareDfx::ShapesReason(OP_NAME, "qRstdOut,kRstdOut",
                                               TensorShapes({params.qRstdOut, params.kRstdOut}),
                                               "each non-null output must use fixed head-major shape=" +
                                                   (info.packed ? DimsText({info.qkHeadNum, info.seqLen}) :
                                                                  DimsText({info.batch, info.qkHeadNum, info.seqLen})));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.betaEffOut == nullptr || valueScalarValid(params.betaEffOut))) {
        KdaPrepareDfx::Shape(OP_NAME, "betaEffOut", ShapeText(params.betaEffOut),
                                  info.packed ? DimsText({info.valueHeadNum, info.seqLen}) :
                                                DimsText({info.batch, info.valueHeadNum, info.seqLen}));
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
    if (tensor == nullptr) {
        // Contiguous reports its own conversion/allocation failure.
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "Contiguous returned null.");
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
    if (!(workspaceSize != nullptr)) {
        KdaPrepareDfx::Missing(
            OP_NAME, "workspaceSize",
            "required output pointer is null; pass the address of a writable uint64_t variable");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(executor != nullptr)) {
        KdaPrepareDfx::Missing(
            OP_NAME, "executor",
            "required output pointer is null; pass the address of a writable aclOpExecutor* variable");
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
        // Genuine internal executor initialization failure; not a user argument error.
        const char *reason = "aclnnChunkKdaFwdPrepare: CREATE_EXECUTOR returned null; preserve executor allocation "
                             "and initialization diagnostics.";
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s", reason);
        REPORT_INNER_ERR_MSG("EZ9999", "%s", reason);
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
    if (result[0] == nullptr || result[1] == nullptr || result[3] == nullptr || result[4] == nullptr ||
        result[6] == nullptr || result[7] == nullptr) {
        // L0/launcher owns the root cause; preserve its first structured report.
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "ChunkKdaFwdPrepare L0 construction failed.");
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
    const aclnnStatus status = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (status != ACLNN_SUCCESS) {
        // CommonOpExecutorRun owns the first error; do not replace it with EZ9999.
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "ChunkKdaFwdPrepare execution failed, status=%d.", status);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
