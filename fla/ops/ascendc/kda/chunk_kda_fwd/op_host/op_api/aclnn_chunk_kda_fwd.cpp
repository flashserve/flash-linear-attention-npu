/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "aclnn_chunk_kda_fwd.h"
#if defined(__has_include)
#if __has_include("op_common/log/log.h")
#include "op_common/log/log.h"
#undef OP_LOGE
#undef OP_LOGE_WITHOUT_REPORT
#endif
#endif
#include "chunk_kda_fwd.h"
#include "aclnn_chunk_kda_fwd_v2.h"
#include "chunk_kda_fwd_v2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
// K/V 只交付两档且必须同档：K=V=64 或 K=V=128；混合档与其它取值都不支持。
constexpr int64_t KDA_FWD_KV_DIM_SMALL = 64;
constexpr int64_t KDA_FWD_KV_DIM_LARGE = 128;
constexpr int64_t MAX_KDA_HEAD_NUM = 128;
constexpr int64_t KDA_STAGE_FULL = -1;
constexpr int64_t KDA_STAGE_GATE_PREPARE = 0;
constexpr int64_t KDA_STAGE_COUNT = 4;

constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;
constexpr const char *KDA_FWD_ENTITY = "aclnnChunkKdaFwd";

enum class KdaFwdLayout {
    BSND,
    BNSD,
    TND,
    NTD,
};

struct ChunkKdaFwdParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = "BSND";
    double scale = 1.0;
    int64_t chunkSize = 64;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool useGateInKernel = false;
    bool stateVFirst = false;
    // 归一化 / 指数开关：v1 入口固定使用与历史实现一致的默认值，
    // 只有 aclnnChunkKdaFwdV2 才会按调用方传参设置。
    double epsilon = 1.0e-6;
    bool useQkL2normInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool useExp2 = true;
    const aclTensor *attnOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
    const aclTensor *gkOut = nullptr;
    const aclTensor *aqkOut = nullptr;
    const aclTensor *akkOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *hOut = nullptr;
    // 反向 L2 norm 保存值；nullptr 表示本次不导出（可选性只在 L2 层表达）。
    const aclTensor *qHatOut = nullptr;
    const aclTensor *kHatOut = nullptr;
    const aclTensor *qRstdOut = nullptr;
    const aclTensor *kRstdOut = nullptr;
    const aclTensor *betaEffOut = nullptr;
};

struct KdaShapeInfo {
    bool isRank3 = false;
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t hNum = 0;
    int64_t hvNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    int64_t seqNum = 0;
    int64_t totalChunks = 0;
};

std::string DfxMakeShapeString(std::initializer_list<int64_t> dims)
{
    std::ostringstream stream;
    stream << "[";
    bool first = true;
    for (int64_t dim : dims) {
        if (!first) {
            stream << ",";
        }
        stream << dim;
        first = false;
    }
    stream << "]";
    return stream.str();
}

std::string DfxTensorShapeString(const aclTensor *tensor)
{
    const auto &shape = tensor->GetViewShape();
    std::ostringstream stream;
    stream << "[";
    for (size_t idx = 0; idx < shape.GetDimNum(); ++idx) {
        if (idx != 0) {
            stream << ",";
        }
        stream << shape.GetDim(idx);
    }
    stream << "]";
    return stream.str();
}

std::string DfxDtypeString(DataType dtype)
{
    switch (dtype) {
        case DataType::DT_FLOAT16:
            return "FLOAT16";
        case DataType::DT_BF16:
            return "BF16";
        case DataType::DT_FLOAT:
            return "FLOAT";
        default:
            return std::to_string(static_cast<int32_t>(dtype));
    }
}

void DfxReportInvalidInput(const char *paramName)
{
#if defined(OP_LOGE_LIBOPAPI_REPORT) && defined(OP_LOGE_WITH_INVALID_INPUT)
    OP_LOGE_WITH_INVALID_INPUT(KDA_FWD_ENTITY, paramName);
#else
    OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Parameter %s of %s is required, but it is empty.", paramName,
            KDA_FWD_ENTITY);
#endif
}

void DfxReportInvalidShape(const char *paramName, const std::string &actual, const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_SHAPE)
    OP_LOGE_FOR_INVALID_SHAPE(KDA_FWD_ENTITY, paramName, actual.c_str(), expected.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameter %s of %s has incorrect shape %s. It should be %s.",
            paramName, KDA_FWD_ENTITY, actual.c_str(), expected.c_str());
#endif
}

void DfxReportInvalidShapeWithReason(const char *paramName, const std::string &actual,
                                     const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON)
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(KDA_FWD_ENTITY, paramName, actual.c_str(), reason.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameter %s of %s has incorrect shape %s. Reason: %s.",
            paramName, KDA_FWD_ENTITY, actual.c_str(), reason.c_str());
#endif
}

void DfxReportInvalidShapes(const char *paramNames, const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON)
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(KDA_FWD_ENTITY, paramNames, actual.c_str(), reason.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameters %s of %s have incorrect shapes %s. Reason: %s.",
            paramNames, KDA_FWD_ENTITY, actual.c_str(), reason.c_str());
#endif
}

void DfxReportInvalidDtype(const char *paramName, const std::string &actual, const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_DTYPE)
    OP_LOGE_FOR_INVALID_DTYPE(KDA_FWD_ENTITY, paramName, actual.c_str(), expected.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameter %s of %s has incorrect dtype %s. It should be %s.",
            paramName, KDA_FWD_ENTITY, actual.c_str(), expected.c_str());
#endif
}

void DfxReportInvalidDtypes(const char *paramNames, const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON)
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(KDA_FWD_ENTITY, paramNames, actual.c_str(), reason.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameters %s of %s have incorrect dtypes %s. Reason: %s.",
            paramNames, KDA_FWD_ENTITY, actual.c_str(), reason.c_str());
#endif
}

void DfxReportInvalidValue(const char *paramName, const std::string &actual, const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_VALUE)
    OP_LOGE_FOR_INVALID_VALUE(KDA_FWD_ENTITY, paramName, actual.c_str(), expected.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameter %s of %s has incorrect value %s. It should be %s.",
            paramName, KDA_FWD_ENTITY, actual.c_str(), expected.c_str());
#endif
}

void DfxReportInvalidValueWithReason(const char *paramName, const std::string &actual,
                                     const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_VALUE_WITH_REASON)
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(KDA_FWD_ENTITY, paramName, actual.c_str(), reason.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameter %s of %s has incorrect value %s. Reason: %s.",
            paramName, KDA_FWD_ENTITY, actual.c_str(), reason.c_str());
#endif
}

void DfxReportInvalidValues(const char *paramNames, const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_VALUES_WITH_REASON)
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(KDA_FWD_ENTITY, paramNames, actual.c_str(), reason.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameters %s of %s have incorrect values %s. Reason: %s.",
            paramNames, KDA_FWD_ENTITY, actual.c_str(), reason.c_str());
#endif
}

void DfxReportInvalidListSize(const char *paramName, const std::string &actual, const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_LISTSIZE)
    OP_LOGE_FOR_INVALID_LISTSIZE(KDA_FWD_ENTITY, paramName, actual.c_str(), expected.c_str());
#else
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Parameter %s of %s has incorrect element nums %s. It should be %s.",
            paramName, KDA_FWD_ENTITY, actual.c_str(), expected.c_str());
#endif
}

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

const aclTensor *Transpose(const aclTensor *input, const std::vector<int64_t> &perm, aclOpExecutor *executor)
{
    const aclIntArray *permArray = executor->AllocIntArray(perm.data(), perm.size());
    if (permArray == nullptr) {
        return nullptr;
    }
    const aclTensor *transposed = l0op::Transpose(input, permArray, executor);
    if (transposed == nullptr) {
        return nullptr;
    }
    const aclTensor *materialized = l0op::Contiguous(transposed, executor);
    if (materialized == nullptr) {
        return nullptr;
    }
    const aclTensor *reshaped =
        l0op::Reshape(materialized, transposed->GetViewShape(), executor);
    if (reshaped == nullptr) {
        return nullptr;
    }
    reshaped->SetStorageShape(reshaped->GetViewShape());
    reshaped->SetOriginalShape(reshaped->GetViewShape());
    return reshaped;
}

const aclTensor *TransposeLastTwo(const aclTensor *input, aclOpExecutor *executor)
{
    const size_t rank = input->GetViewShape().GetDimNum();
    std::vector<int64_t> perm(rank);
    for (size_t idx = 0; idx < rank; ++idx) {
        perm[idx] = static_cast<int64_t>(idx);
    }
    std::swap(perm[rank - 2], perm[rank - 1]);
    return Transpose(input, perm, executor);
}

static int64_t Dim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

static size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

static bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs == nullptr || rhs == nullptr || Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t idx = 0; idx < Rank(lhs); ++idx) {
        if (Dim(lhs, idx) != Dim(rhs, idx)) {
            return false;
        }
    }
    return true;
}

bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> expected)
{
    if (tensor == nullptr || Rank(tensor) != expected.size()) {
        return false;
    }
    size_t idx = 0;
    for (int64_t dim : expected) {
        if (Dim(tensor, idx++) != dim) {
            return false;
        }
    }
    return true;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseLayout(const char *layout, KdaFwdLayout &parsed)
{
    if (layout == nullptr) {
        DfxReportInvalidInput("layout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = KdaFwdLayout::BSND;
    } else if (std::strcmp(layout, "BNSD") == 0) {
        parsed = KdaFwdLayout::BNSD;
    } else if (std::strcmp(layout, "TND") == 0) {
        parsed = KdaFwdLayout::TND;
    } else if (std::strcmp(layout, "NTD") == 0) {
        parsed = KdaFwdLayout::NTD;
    } else {
        DfxReportInvalidValue("layout", layout, "BSND,BNSD,TND,NTD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckCuSeqlens(const aclIntArray *cuSeqlens, int64_t seqlen)
{
    if (cuSeqlens == nullptr) {
        return ACLNN_SUCCESS;
    }
    if (cuSeqlens->Size() < 2) {
        DfxReportInvalidListSize("cuSeqlensOptional", std::to_string(cuSeqlens->Size()), ">=2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*cuSeqlens)[0] != 0) {
        DfxReportInvalidValue("cuSeqlensOptional[0]", std::to_string((*cuSeqlens)[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*cuSeqlens)[cuSeqlens->Size() - 1] != seqlen) {
        DfxReportInvalidValue("cuSeqlensOptional[last]",
                              std::to_string((*cuSeqlens)[cuSeqlens->Size() - 1]),
                              std::to_string(seqlen));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t idx = 0; idx + 1 < cuSeqlens->Size(); ++idx) {
        if ((*cuSeqlens)[idx] > (*cuSeqlens)[idx + 1]) {
            DfxReportInvalidValues("cuSeqlensOptional[idx],cuSeqlensOptional[idx+1]",
                                   std::to_string((*cuSeqlens)[idx]) + "," +
                                       std::to_string((*cuSeqlens)[idx + 1]),
                                   "cu_seqlens must be nondecreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

int64_t CountChunks(const aclIntArray *cuSeqlens, int64_t seqlen, int64_t chunkSize)
{
    if (cuSeqlens == nullptr) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    int64_t chunks = 0;
    for (size_t idx = 0; idx + 1 < cuSeqlens->Size(); ++idx) {
        chunks += ((*cuSeqlens)[idx + 1] - (*cuSeqlens)[idx] + chunkSize - 1) / chunkSize;
    }
    return chunks;
}

aclnnStatus CheckChunkIndices(const aclIntArray *chunkIndices, const aclIntArray *cuSeqlens,
                              int64_t totalChunks, int64_t chunkSize)
{
    if (chunkIndices == nullptr) {
        return ACLNN_SUCCESS;
    }
    if (cuSeqlens == nullptr) {
        DfxReportInvalidValues("chunkIndicesOptional,cuSeqlensOptional", "present,missing",
                               "chunk_indices requires cu_seqlens");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (chunkIndices->Size() != static_cast<size_t>(totalChunks) * 2) {
        DfxReportInvalidListSize("chunkIndicesOptional", std::to_string(chunkIndices->Size()),
                                 std::to_string(static_cast<size_t>(totalChunks) * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        const int64_t chunks = (length + chunkSize - 1) / chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if ((*chunkIndices)[offset] != static_cast<int64_t>(seq) ||
                (*chunkIndices)[offset + 1] != chunk) {
                DfxReportInvalidValues(
                    "chunkIndicesOptional[offset],chunkIndicesOptional[offset+1]",
                    std::to_string((*chunkIndices)[offset]) + "," +
                        std::to_string((*chunkIndices)[offset + 1]),
                    "expected canonical sequence-major (" + std::to_string(seq) + "," +
                        std::to_string(chunk) + ")");
                return ACLNN_ERR_PARAM_INVALID;
            }
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShapeInfo(const ChunkKdaFwdParams &params, KdaFwdLayout layout, KdaShapeInfo &info)
{
    info.isRank3 = layout == KdaFwdLayout::TND || layout == KdaFwdLayout::NTD;
    const size_t tensorRank = info.isRank3 ? 3 : 4;
    const size_t betaRank = info.isRank3 ? 2 : 3;
    const aclTensor *rankedInputs[] = {params.q, params.k, params.v, params.g};
    const char *rankedNames[] = {"q", "k", "v", "g"};
    for (size_t idx = 0; idx < sizeof(rankedInputs) / sizeof(rankedInputs[0]); ++idx) {
        if (Rank(rankedInputs[idx]) != tensorRank) {
            DfxReportInvalidShapeWithReason(
                rankedNames[idx], std::to_string(Rank(rankedInputs[idx])) + "D",
                "rank must be " + std::to_string(tensorRank) + " for the selected layout");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (Rank(params.beta) != betaRank) {
        DfxReportInvalidShapeWithReason(
            "beta", std::to_string(Rank(params.beta)) + "D",
            "rank must be " + std::to_string(betaRank) + " for the selected layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!SameShape(params.q, params.k)) {
        DfxReportInvalidShapes("q,k",
                               "q=" + DfxTensorShapeString(params.q) + ",k=" +
                                   DfxTensorShapeString(params.k),
                               "q and k must have identical shape");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (layout == KdaFwdLayout::TND) {
        info.batch = 1;
        info.seqlen = Dim(params.q, 0);
        info.hNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 2);
        const std::string expectedV = DfxMakeShapeString({info.seqlen, info.hvNum, info.vDim});
        const std::string expectedG = DfxMakeShapeString({info.seqlen, info.hvNum, info.kDim});
        const std::string expectedBeta = DfxMakeShapeString({info.seqlen, info.hvNum});
        if (!HasShape(params.v, {info.seqlen, info.hvNum, info.vDim})) {
            DfxReportInvalidShape("v", DfxTensorShapeString(params.v), expectedV);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.seqlen, info.hvNum, info.kDim})) {
            DfxReportInvalidShape("g", DfxTensorShapeString(params.g), expectedG);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.seqlen, info.hvNum})) {
            DfxReportInvalidShape("beta", DfxTensorShapeString(params.beta), expectedBeta);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (layout == KdaFwdLayout::NTD) {
        info.batch = 1;
        info.hNum = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 0);
        info.vDim = Dim(params.v, 2);
        const std::string expectedV = DfxMakeShapeString({info.hvNum, info.seqlen, info.vDim});
        const std::string expectedG = DfxMakeShapeString({info.hvNum, info.seqlen, info.kDim});
        const std::string expectedBeta = DfxMakeShapeString({info.hvNum, info.seqlen});
        if (!HasShape(params.v, {info.hvNum, info.seqlen, info.vDim})) {
            DfxReportInvalidShape("v", DfxTensorShapeString(params.v), expectedV);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.hvNum, info.seqlen, info.kDim})) {
            DfxReportInvalidShape("g", DfxTensorShapeString(params.g), expectedG);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.hvNum, info.seqlen})) {
            DfxReportInvalidShape("beta", DfxTensorShapeString(params.beta), expectedBeta);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (layout == KdaFwdLayout::BSND) {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.hNum = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        const std::string expectedV =
            DfxMakeShapeString({info.batch, info.seqlen, info.hvNum, info.vDim});
        const std::string expectedG =
            DfxMakeShapeString({info.batch, info.seqlen, info.hvNum, info.kDim});
        const std::string expectedBeta =
            DfxMakeShapeString({info.batch, info.seqlen, info.hvNum});
        if (!HasShape(params.v, {info.batch, info.seqlen, info.hvNum, info.vDim})) {
            DfxReportInvalidShape("v", DfxTensorShapeString(params.v), expectedV);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.batch, info.seqlen, info.hvNum, info.kDim})) {
            DfxReportInvalidShape("g", DfxTensorShapeString(params.g), expectedG);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.batch, info.seqlen, info.hvNum})) {
            DfxReportInvalidShape("beta", DfxTensorShapeString(params.beta), expectedBeta);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        info.batch = Dim(params.q, 0);
        info.hNum = Dim(params.q, 1);
        info.seqlen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        const std::string expectedV =
            DfxMakeShapeString({info.batch, info.hvNum, info.seqlen, info.vDim});
        const std::string expectedG =
            DfxMakeShapeString({info.batch, info.hvNum, info.seqlen, info.kDim});
        const std::string expectedBeta =
            DfxMakeShapeString({info.batch, info.hvNum, info.seqlen});
        if (!HasShape(params.v, {info.batch, info.hvNum, info.seqlen, info.vDim})) {
            DfxReportInvalidShape("v", DfxTensorShapeString(params.v), expectedV);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.batch, info.hvNum, info.seqlen, info.kDim})) {
            DfxReportInvalidShape("g", DfxTensorShapeString(params.g), expectedG);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.batch, info.hvNum, info.seqlen})) {
            DfxReportInvalidShape("beta", DfxTensorShapeString(params.beta), expectedBeta);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    info.seqNum = params.cuSeqlensOptional == nullptr
                      ? info.batch
                      : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    info.totalChunks = CountChunks(params.cuSeqlensOptional, info.seqlen, params.chunkSize);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypes(const ChunkKdaFwdParams &params)
{
    const DataType dataType = params.q->GetDataType();
    if (dataType != DataType::DT_FLOAT16 && dataType != DataType::DT_BF16) {
        DfxReportInvalidDtype("q", DfxDtypeString(dataType), "FLOAT16,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.k->GetDataType() != dataType || params.v->GetDataType() != dataType) {
        DfxReportInvalidDtypes("k,v",
                               "q=" + DfxDtypeString(dataType) + ",k=" +
                                   DfxDtypeString(params.k->GetDataType()) + ",v=" +
                                   DfxDtypeString(params.v->GetDataType()),
                               "q, k and v must use the same dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const DataType gateType = params.g->GetDataType();
    if (gateType != DataType::DT_FLOAT && gateType != DataType::DT_BF16) {
        DfxReportInvalidDtype("g", DfxDtypeString(gateType), "FLOAT,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const DataType betaType = params.beta->GetDataType();
    if (betaType != DataType::DT_FLOAT && betaType != DataType::DT_BF16) {
        DfxReportInvalidDtype("beta", DfxDtypeString(betaType), "FLOAT,BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.aLogOptional != nullptr) {
        if (params.aLogOptional->GetDataType() != DataType::DT_FLOAT) {
            DfxReportInvalidDtype("aLogOptional",
                                  DfxDtypeString(params.aLogOptional->GetDataType()), "FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dtBiasOptional != nullptr) {
        if (params.dtBiasOptional->GetDataType() != DataType::DT_FLOAT) {
            DfxReportInvalidDtype("dtBiasOptional",
                                  DfxDtypeString(params.dtBiasOptional->GetDataType()), "FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.initialStateOptional != nullptr) {
        if (params.initialStateOptional->GetDataType() != DataType::DT_FLOAT) {
            DfxReportInvalidDtype("initialStateOptional",
                                  DfxDtypeString(params.initialStateOptional->GetDataType()), "FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckStateShape(const aclTensor *state, const char *name, const KdaShapeInfo &info, bool stateVFirst)
{
    if (state == nullptr) {
        return ACLNN_SUCCESS;
    }
    const bool valid = stateVFirst
                           ? HasShape(state, {info.seqNum, info.hvNum, info.vDim, info.kDim})
                           : HasShape(state, {info.seqNum, info.hvNum, info.kDim, info.vDim});
    if (!valid) {
        const std::string expected =
            stateVFirst ? DfxMakeShapeString({info.seqNum, info.hvNum, info.vDim, info.kDim})
                        : DfxMakeShapeString({info.seqNum, info.hvNum, info.kDim, info.vDim});
        DfxReportInvalidShape(name, DfxTensorShapeString(state), expected);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOptionalOutput(const aclTensor *output, const char *name, DataType expectedDtype,
                                std::initializer_list<int64_t> expectedDims)
{
    if (output == nullptr) {
        return ACLNN_SUCCESS;
    }
    if (!HasShape(output, expectedDims)) {
        DfxReportInvalidShape(name, DfxTensorShapeString(output), DfxMakeShapeString(expectedDims));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (output->GetDataType() != expectedDtype) {
        DfxReportInvalidDtype(name, DfxDtypeString(output->GetDataType()), DfxDtypeString(expectedDtype));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const ChunkKdaFwdParams &params, const KdaShapeInfo &info)
{
    const DataType dataType = params.q->GetDataType();
    auto status = CheckOptionalOutput(
        params.attnOut, "attnOut", dataType,
        info.isRank3 ? std::initializer_list<int64_t>{info.seqlen, info.hvNum, info.vDim}
                     : std::initializer_list<int64_t>{info.batch, info.seqlen, info.hvNum, info.vDim});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckOptionalOutput(
        params.gkOut, "gkOut", DataType::DT_FLOAT,
        info.isRank3 ? std::initializer_list<int64_t>{info.hvNum, info.seqlen, info.kDim}
                     : std::initializer_list<int64_t>{info.batch, info.hvNum, info.seqlen, info.kDim});
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    for (const auto &matrixOutput : {std::make_pair(params.aqkOut, "aqkOut"),
                                     std::make_pair(params.akkOut, "akkOut")}) {
        status = CheckOptionalOutput(
            matrixOutput.first, matrixOutput.second, dataType,
            info.isRank3
                ? std::initializer_list<int64_t>{info.hvNum, info.seqlen, params.chunkSize}
                : std::initializer_list<int64_t>{info.batch, info.hvNum, info.seqlen, params.chunkSize});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    for (const auto &kOutput : {std::make_pair(params.wOut, "wOut"),
                                std::make_pair(params.qgOut, "qgOut"),
                                std::make_pair(params.kgOut, "kgOut")}) {
        status = CheckOptionalOutput(
            kOutput.first, kOutput.second, dataType,
            info.isRank3
                ? std::initializer_list<int64_t>{info.hvNum, info.seqlen, info.kDim}
                : std::initializer_list<int64_t>{info.batch, info.hvNum, info.seqlen, info.kDim});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    for (const auto &vOutput : {std::make_pair(params.uOut, "uOut"),
                                std::make_pair(params.vNewOut, "vNewOut")}) {
        status = CheckOptionalOutput(
            vOutput.first, vOutput.second, dataType,
            info.isRank3
                ? std::initializer_list<int64_t>{info.hvNum, info.seqlen, info.vDim}
                : std::initializer_list<int64_t>{info.batch, info.hvNum, info.seqlen, info.vDim});
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    status = CheckOptionalOutput(
        params.hOut, "hOut", dataType,
        params.stateVFirst
            ? (info.isRank3
                   ? std::initializer_list<int64_t>{info.totalChunks, info.hvNum, info.vDim, info.kDim}
                   : std::initializer_list<int64_t>{info.batch, info.totalChunks, info.hvNum, info.vDim,
                                                    info.kDim})
            : (info.isRank3
                   ? std::initializer_list<int64_t>{info.totalChunks, info.hvNum, info.kDim, info.vDim}
                   : std::initializer_list<int64_t>{info.batch, info.totalChunks, info.hvNum, info.kDim,
                                                    info.vDim}));
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_RET(CheckStateShape(params.finalStateOut, "finalStateOut", info, params.stateVFirst) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (params.finalStateOut != nullptr) {
        if (params.finalStateOut->GetDataType() != DataType::DT_FLOAT) {
            DfxReportInvalidDtype("finalStateOut",
                                  DfxDtypeString(params.finalStateOut->GetDataType()), "FLOAT");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkKdaFwdParams &params, KdaFwdLayout &layout, KdaShapeInfo &info)
{
    const aclTensor *requiredInputs[] = {params.q, params.k, params.v, params.g, params.beta};
    const char *requiredInputNames[] = {"q", "k", "v", "g", "beta"};
    for (size_t idx = 0; idx < sizeof(requiredInputs) / sizeof(requiredInputs[0]); ++idx) {
        if (requiredInputs[idx] == nullptr) {
            DfxReportInvalidInput(requiredInputNames[idx]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (params.attnOut == nullptr) {
        DfxReportInvalidInput("attnOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    // aqkOut 对应 op def 的 REQUIRED 输出；akkOut 对应 OPTIONAL，省略时算子内部
    // 自建不导出的占位张量，V2 组合入口随之使用 Prepare 的 none 档。
    if (params.aqkOut == nullptr) {
        DfxReportInvalidInput("aqkOut");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.chunkSize != 64) {
        DfxReportInvalidValue("chunkSize", std::to_string(params.chunkSize), "64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto status = ParseLayout(params.layout, layout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = ResolveShapeInfo(params, layout, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    // 空 tensor（batch 或序列长度为 0）无法构成可执行的计算规模。host 侧在进入 tiling
    // 之前拦截并说明是哪个逻辑维为 0，避免只向调用方回传无上下文的 561103。
    if (info.batch <= 0) {
        DfxReportInvalidValue("q.shape[B]", std::to_string(info.batch), ">0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.seqlen <= 0) {
        DfxReportInvalidValue("q.shape[T]", std::to_string(info.seqlen), ">0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.hNum <= 0 || info.hvNum < info.hNum || info.hvNum % info.hNum != 0) {
        DfxReportInvalidValues("H,HV",
                               std::to_string(info.hNum) + "," + std::to_string(info.hvNum),
                               "H and HV must be positive, HV >= H, and HV must be divisible by H");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.hNum > MAX_KDA_HEAD_NUM || info.hvNum > MAX_KDA_HEAD_NUM) {
        DfxReportInvalidValues("H,HV",
                               std::to_string(info.hNum) + "," + std::to_string(info.hvNum),
                               "H and HV must be <= 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // K/V 仅支持 K=V=64 与 K=V=128 两档，不支持混合档（如 K=64/V=128）或
    // 其它取值；报错打印实际 K/V，便于调用方直接定位 shape。
    const bool kvDimSupported =
        (info.kDim == KDA_FWD_KV_DIM_SMALL && info.vDim == KDA_FWD_KV_DIM_SMALL) ||
        (info.kDim == KDA_FWD_KV_DIM_LARGE && info.vDim == KDA_FWD_KV_DIM_LARGE);
    if (!kvDimSupported) {
        DfxReportInvalidValues("q.shape[K],v.shape[V]",
                               "K=" + std::to_string(info.kDim) + ",V=" + std::to_string(info.vDim),
                               "K and V must both be 64 or both be 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    status = CheckDtypes(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckCuSeqlens(params.cuSeqlensOptional, info.seqlen);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (params.cuSeqlensOptional != nullptr && !info.isRank3 && info.batch != 1) {
        DfxReportInvalidValueWithReason("q.shape[0]", std::to_string(info.batch),
                                        "rank4 input with cu_seqlens requires B=1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional != nullptr && info.seqNum > MAX_KDA_VARLEN_SEQUENCES) {
        DfxReportInvalidValue("cuSeqlensOptional.size", std::to_string(info.seqNum), "<=1024");
        return ACLNN_ERR_PARAM_INVALID;
    }
    status = CheckChunkIndices(params.chunkIndicesOptional, params.cuSeqlensOptional,
                               info.totalChunks, params.chunkSize);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckStateShape(params.initialStateOptional, "initialStateOptional", info,
                             params.stateVFirst);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    // 归一化 / gate 开关约束：epsilon 只在 in-kernel L2 norm 时参与计算，
    // allowNegEigval 依赖 in-kernel beta sigmoid。
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    if (!std::isfinite(epsilonFp32) || epsilonFp32 <= 0.0F) {
        DfxReportInvalidValue("epsilon", std::to_string(params.epsilon), "positive finite number");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.allowNegEigval && !params.useBetaSigmoidInKernel) {
        DfxReportInvalidValueWithReason("allowNegEigval", "true",
                                        "allowNegEigval=true requires useBetaSigmoidInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.useGateInKernel) {
        if (params.aLogOptional == nullptr) {
            DfxReportInvalidInput("aLogOptional");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (!HasShape(params.aLogOptional, {info.hvNum})) {
            DfxReportInvalidShape("aLogOptional", DfxTensorShapeString(params.aLogOptional),
                                  DfxMakeShapeString({info.hvNum}));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.dtBiasOptional != nullptr) {
            if (!HasShape(params.dtBiasOptional, {info.hvNum * info.kDim})) {
                DfxReportInvalidShape("dtBiasOptional", DfxTensorShapeString(params.dtBiasOptional),
                                      DfxMakeShapeString({info.hvNum * info.kDim}));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (params.safeGate) {
            if (params.lowerBound < -5.0 || params.lowerBound >= 0.0) {
                DfxReportInvalidValue("lowerBound", std::to_string(params.lowerBound),
                                      "[-5,0) when safeGate is true");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return CheckOutputShapes(params, info);
}

aclnnStatus ContiguousInputs(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    const aclTensor *inputs[] = {params.q, params.k, params.v, params.g, params.beta,
                                 params.aLogOptional, params.dtBiasOptional,
                                 params.initialStateOptional};
    for (const aclTensor *&input : inputs) {
        auto status = MakeContiguous(input, executor);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

const aclTensor *AllocTensor(aclOpExecutor *executor, const op::Shape &shape, DataType dtype)
{
    return executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
}

const aclTensor *AsRank4(const aclTensor *tensor, const op::Shape &shape, aclOpExecutor *executor)
{
    return l0op::Reshape(tensor, shape, executor);
}

bool AllOutputsContiguous(const ChunkKdaFwdParams &params)
{
    const aclTensor *outputs[] = {
        params.attnOut,     params.finalStateOut, params.gkOut, params.aqkOut,
        params.akkOut,      params.wOut,          params.uOut,  params.qgOut,
        params.kgOut,       params.vNewOut,       params.hOut};
    for (const aclTensor *output : outputs) {
        if (output != nullptr && !IsContiguous(output)) {
            return false;
        }
    }
    return true;
}

// V2 场景判据：三个独立算子的公共约束（BF16、K=V=128、chunk=64、输出可直接写回、
// cu_seqlens 严格递增）全部满足时才允许走组合实现。
bool CanUseKdaFwdV2Path(const ChunkKdaFwdParams &params, const KdaShapeInfo &info)
{
    if (params.q->GetDataType() != DataType::DT_BF16) {
        return false;
    }
    if (info.kDim != 128 || info.vDim != 128) {
        return false;
    }
    if (params.chunkSize != 64) {
        return false;
    }
    // 本算子的 L2 允许 cuSeqlens 非递减（即空序列），三个独立算子要求严格递增。
    if (params.cuSeqlensOptional != nullptr) {
        const aclIntArray *cuSeqlens = params.cuSeqlensOptional;
        for (size_t idx = 0; idx + 1 < cuSeqlens->Size(); ++idx) {
            if ((*cuSeqlens)[idx] >= (*cuSeqlens)[idx + 1]) {
                return false;
            }
        }
    }
    // 状态的 shape 与 FP32 约束已由 CheckParams 校验；三算子链沿用同一契约。
    return AllOutputsContiguous(params);
}

l0op::KdaFwdV2Args MakeKdaFwdV2Args(const ChunkKdaFwdParams &params,
                                              const KdaShapeInfo &info)
{
    l0op::KdaFwdV2Args args;
    args.q = params.q;
    args.k = params.k;
    args.v = params.v;
    args.g = params.g;
    args.beta = params.beta;
    args.aLog = params.aLogOptional;
    args.dtBias = params.dtBiasOptional;
    args.initialState = params.initialStateOptional;
    args.cuSeqlens = params.cuSeqlensOptional;
    args.chunkIndices = params.chunkIndicesOptional;
    args.layout = params.layout;
    // 公开 attnOut 固定 sequence-major：rank-4 为 BSND，rank-3 为 TND。
    args.attnLayout = info.isRank3 ? "TND" : "BSND";
    args.scale = params.scale;
    args.epsilon = params.epsilon;
    args.lowerBound = params.lowerBound;
    args.chunkSize = params.chunkSize;
    args.safeGate = params.safeGate;
    args.useGateInKernel = params.useGateInKernel;
    args.stateVFirst = params.stateVFirst;
    args.useQkL2normInKernel = params.useQkL2normInKernel;
    args.useBetaSigmoidInKernel = params.useBetaSigmoidInKernel;
    args.allowNegEigval = params.allowNegEigval;
    args.useExp2 = params.useExp2;
    args.batch = info.batch;
    args.qkHeads = info.hNum;
    args.valueHeads = info.hvNum;
    args.seqLen = info.seqlen;
    args.kDim = info.kDim;
    args.vDim = info.vDim;
    args.seqNum = info.seqNum;
    args.totalChunks = info.totalChunks;
    args.packed = info.isRank3;
    args.attnOut = params.attnOut;
    args.finalStateOut = params.finalStateOut;
    args.gkOut = params.gkOut;
    args.aqkOut = params.aqkOut;
    args.akkOut = params.akkOut;
    args.wOut = params.wOut;
    args.uOut = params.uOut;
    args.qgOut = params.qgOut;
    args.kgOut = params.kgOut;
    args.vNewOut = params.vNewOut;
    args.hOut = params.hOut;
    args.qHatOut = params.qHatOut;
    args.kHatOut = params.kHatOut;
    args.qRstdOut = params.qRstdOut;
    args.kRstdOut = params.kRstdOut;
    args.betaEffOut = params.betaEffOut;
    return args;
}

bool IsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}
} // namespace

aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkKdaFwdParams params{
        q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
        cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize,
        safeGate, lowerBound, useGateInKernel, stateVFirst, 1.0e-6, false, false, false, true,
        attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut};
    if (workspaceSize == nullptr) {
        DfxReportInvalidInput("workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        DfxReportInvalidInput("executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(
        aclnnChunkKdaFwd,
        DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
               cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize,
               safeGate, lowerBound, useGateInKernel, stateVFirst),
        DFX_OUT(attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut,
                qgOut, kgOut, vNewOut, hOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    KdaFwdLayout parsedLayout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    auto status = CheckParams(params, parsedLayout, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = ContiguousInputs(params, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const aclTensor *qHead = params.q;
    const aclTensor *kHead = params.k;
    const aclTensor *vHead = params.v;
    const aclTensor *gHead = params.g;
    const aclTensor *betaHead = params.beta;
    if (parsedLayout == KdaFwdLayout::BSND) {
        // beta is small and its scalar-per-token rows are not DMA friendly in
        // sequence-major form. Keep only this lightweight transpose.
        betaHead = Transpose(params.beta, {0, 2, 1}, executorPtr);
    } else if (parsedLayout == KdaFwdLayout::TND) {
        qHead = Transpose(params.q, {1, 0, 2}, executorPtr);
        kHead = Transpose(params.k, {1, 0, 2}, executorPtr);
        vHead = Transpose(params.v, {1, 0, 2}, executorPtr);
        gHead = Transpose(params.g, {1, 0, 2}, executorPtr);
        betaHead = Transpose(params.beta, {1, 0}, executorPtr);
    }
    CHECK_RET(qHead != nullptr && kHead != nullptr && vHead != nullptr &&
                  gHead != nullptr && betaHead != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    if (info.isRank3) {
        qHead = AsRank4(qHead, MakeShape({1, info.hNum, info.seqlen, info.kDim}), executorPtr);
        kHead = AsRank4(kHead, MakeShape({1, info.hNum, info.seqlen, info.kDim}), executorPtr);
        vHead = AsRank4(vHead, MakeShape({1, info.hvNum, info.seqlen, info.vDim}), executorPtr);
        gHead = AsRank4(gHead, MakeShape({1, info.hvNum, info.seqlen, info.kDim}), executorPtr);
        betaHead = AsRank4(betaHead, MakeShape({1, info.hvNum, info.seqlen}), executorPtr);
        CHECK_RET(qHead != nullptr && kHead != nullptr && vHead != nullptr &&
                      gHead != nullptr && betaHead != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    const op::Shape gkShape4 = MakeShape({info.batch, info.hvNum, info.seqlen, info.kDim});
    const op::Shape matrixShape4 =
        MakeShape({info.batch, info.hvNum, info.seqlen, params.chunkSize});
    const op::Shape kShape4 = MakeShape({info.batch, info.hvNum, info.seqlen, info.kDim});
    const op::Shape vShape4 = MakeShape({info.batch, info.hvNum, info.seqlen, info.vDim});
    const op::Shape hShape5 =
        MakeShape({info.batch, info.totalChunks, info.hvNum, info.kDim, info.vDim});
    const op::Shape hExportShape5 =
        params.stateVFirst
            ? MakeShape({info.batch, info.totalChunks, info.hvNum, info.vDim, info.kDim})
            : MakeShape({info.batch, info.totalChunks, info.hvNum, info.kDim, info.vDim});
    const op::Shape stateShape4 =
        MakeShape({info.seqNum, info.hvNum, info.kDim, info.vDim});
    const op::Shape placeholderShape = MakeShape({1});
    const bool useDenseA5FastPath =
        params.cuSeqlensOptional == nullptr && params.q->GetDataType() == DataType::DT_BF16 &&
        params.chunkSize == 64 && info.kDim == 128 && info.vDim == 128 &&
        info.seqlen % params.chunkSize == 0;
    const bool splitStages =
        IsAscend950() && info.totalChunks > 1 && !useDenseA5FastPath;

    const aclTensor *gkCompute = params.gkOut;
    if (gkCompute != nullptr && info.isRank3) {
        gkCompute = AsRank4(gkCompute, gkShape4, executorPtr);
    }
    if (gkCompute == nullptr) {
        gkCompute = AllocTensor(
            executorPtr, splitStages ? gkShape4 : placeholderShape,
            DataType::DT_FLOAT);
    }
    CHECK_RET(gkCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *aqkCompute = params.aqkOut;
    const aclTensor *akkCompute = params.akkOut;
    const aclTensor *wExport = params.wOut;
    const aclTensor *uExport = params.uOut;
    const aclTensor *qgExport = params.qgOut;
    const aclTensor *kgExport = params.kgOut;
    const aclTensor *vNewExport = params.vNewOut;
    const aclTensor *hExport = params.hOut;
    if (info.isRank3) {
        aqkCompute = aqkCompute == nullptr ? nullptr : AsRank4(aqkCompute, matrixShape4, executorPtr);
        akkCompute = akkCompute == nullptr ? nullptr : AsRank4(akkCompute, matrixShape4, executorPtr);
        wExport = wExport == nullptr ? nullptr : AsRank4(wExport, kShape4, executorPtr);
        uExport = uExport == nullptr ? nullptr : AsRank4(uExport, vShape4, executorPtr);
        qgExport = qgExport == nullptr ? nullptr : AsRank4(qgExport, kShape4, executorPtr);
        kgExport = kgExport == nullptr ? nullptr : AsRank4(kgExport, kShape4, executorPtr);
        vNewExport = vNewExport == nullptr ? nullptr : AsRank4(vNewExport, vShape4, executorPtr);
        if (hExport != nullptr) {
            hExport = AsRank4(hExport, hExportShape5, executorPtr);
        }
    }
    CHECK_RET((params.wOut == nullptr || wExport != nullptr) &&
                  (params.uOut == nullptr || uExport != nullptr) &&
                  (params.qgOut == nullptr || qgExport != nullptr) &&
                  (params.kgOut == nullptr || kgExport != nullptr) &&
                  (params.vNewOut == nullptr || vNewExport != nullptr) &&
                  (params.hOut == nullptr || hExport != nullptr),
              ACLNN_ERR_INNER_NULLPTR);
    if (aqkCompute == nullptr) {
        aqkCompute = AllocTensor(executorPtr, matrixShape4, params.q->GetDataType());
    }
    if (akkCompute == nullptr) {
        akkCompute = AllocTensor(executorPtr, matrixShape4, params.q->GetDataType());
    }
    const aclTensor *wCompute = wExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? kShape4 : placeholderShape,
                      params.q->GetDataType())
        : wExport;
    const aclTensor *uCompute = uExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? vShape4 : placeholderShape,
                      params.q->GetDataType())
        : uExport;
    const aclTensor *qgCompute = qgExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? kShape4 : placeholderShape,
                      params.q->GetDataType())
        : qgExport;
    const aclTensor *kgCompute = kgExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? kShape4 : placeholderShape,
                      params.q->GetDataType())
        : kgExport;
    const aclTensor *vNewCompute = vNewExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? vShape4 : placeholderShape,
                      params.q->GetDataType())
        : vNewExport;
    const aclTensor *hCompute = AllocTensor(
        executorPtr, hExport == nullptr && !splitStages ? placeholderShape : hShape5,
        params.q->GetDataType());
    CHECK_RET(aqkCompute != nullptr && akkCompute != nullptr && wCompute != nullptr &&
                  uCompute != nullptr && qgCompute != nullptr && kgCompute != nullptr &&
                  vNewCompute != nullptr && hCompute != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *initialStateCompute = params.initialStateOptional;
    if (params.stateVFirst && initialStateCompute != nullptr) {
        initialStateCompute = TransposeLastTwo(initialStateCompute, executorPtr);
        CHECK_RET(initialStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const bool outputFinalState = params.finalStateOut != nullptr;
    const aclTensor *finalStateCompute = AllocTensor(
        executorPtr, outputFinalState || splitStages ? stateShape4 : placeholderShape,
        DataType::DT_FLOAT);
    CHECK_RET(finalStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *attnCompute = params.attnOut;
    if (info.isRank3) {
        attnCompute = AsRank4(
            params.attnOut, MakeShape({1, info.seqlen, info.hvNum, info.vDim}), executorPtr);
        CHECK_RET(attnCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *qgScaledCompute = AllocTensor(
        executorPtr, splitStages ? kShape4 : placeholderShape,
        params.q->GetDataType());
    const aclTensor *uSeedCompute = AllocTensor(
        executorPtr, splitStages ? vShape4 : placeholderShape,
        params.q->GetDataType());
    CHECK_RET(qgScaledCompute != nullptr && uSeedCompute != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto launchStage = [&](int64_t stage) {
        return l0op::KdaChunkForward(
            qHead, kHead, vHead, gHead, betaHead, params.aLogOptional,
            params.dtBiasOptional, initialStateCompute, params.cuSeqlensOptional,
            params.chunkIndicesOptional, params.scale, params.chunkSize,
            params.safeGate, parsedLayout == KdaFwdLayout::BSND,
            params.useGateInKernel, params.lowerBound, attnCompute,
            finalStateCompute, gkCompute, aqkCompute, akkCompute, wCompute,
            uCompute, qgCompute, kgCompute, vNewCompute, hCompute,
            qgScaledCompute, uSeedCompute, stage, executorPtr);
    };
    l0op::KdaCoreOutputs result{};
    if (splitStages) {
        // Physical launch boundaries reset the A5 event state between the
        // prepare, post-WU, recurrent, and output pipelines.
        for (int64_t stage = KDA_STAGE_GATE_PREPARE; stage < KDA_STAGE_COUNT;
             ++stage) {
            result = launchStage(stage);
            for (const aclTensor *tensor : result) {
                CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
    } else {
        result = launchStage(KDA_STAGE_FULL);
        for (const aclTensor *tensor : result) {
            CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    if (outputFinalState) {
        const aclTensor *finalStateResult = result[1];
        if (params.stateVFirst) {
            finalStateResult = TransposeLastTwo(finalStateResult, executorPtr);
            CHECK_RET(finalStateResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(l0op::ViewCopy(finalStateResult, params.finalStateOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    if (hExport != nullptr) {
        const aclTensor *hResult = params.stateVFirst
            ? TransposeLastTwo(result[10], executorPtr) : result[10];
        CHECK_RET(hResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(hResult, hExport, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwd(void *workspace, uint64_t workspaceSize,
                             aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaFwd launch failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdV2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    double epsilon,
    bool useQkL2normInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool useExp2,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    const aclTensor *qHatOut,
    const aclTensor *kHatOut,
    const aclTensor *qRstdOut,
    const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkKdaFwdParams params{
        q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
        cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize,
        safeGate, lowerBound, useGateInKernel, stateVFirst, epsilon, useQkL2normInKernel,
        useBetaSigmoidInKernel, allowNegEigval, useExp2, attnOut, finalStateOut, gkOut,
        aqkOut, akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut,
        qHatOut, kHatOut, qRstdOut, kRstdOut, betaEffOut};
    if (workspaceSize == nullptr) {
        DfxReportInvalidInput("workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        DfxReportInvalidInput("executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(
        aclnnChunkKdaFwdV2,
        DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
               cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize,
               safeGate, lowerBound, useGateInKernel, stateVFirst, epsilon,
               useQkL2normInKernel, useBetaSigmoidInKernel, allowNegEigval, useExp2),
        DFX_OUT(attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut,
                qgOut, kgOut, vNewOut, hOut, qHatOut, kHatOut, qRstdOut,
                kRstdOut, betaEffOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    KdaFwdLayout parsedLayout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    auto status = CheckParams(params, parsedLayout, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = ContiguousInputs(params, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    // V2 只实现三个独立算子的组合路径；场景不满足时由 Python 入口回落到
    // aclnnChunkKdaFwd，因此这里直接返回参数错误并说明要求。
    if (!CanUseKdaFwdV2Path(params, info)) {
        DfxReportInvalidValueWithReason(
            "entry", "aclnnChunkKdaFwdV2",
            "requires BF16 q/k/v, K=V=128, chunk_size=64, contiguous outputs, and strictly "
            "increasing cu_seqlens; use aclnnChunkKdaFwd instead");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclnnStatus kdaFwdV2Status =
        l0op::KdaFwdV2(MakeKdaFwdV2Args(params, info), executorPtr);
    CHECK_RET(kdaFwdV2Status == ACLNN_SUCCESS, kdaFwdV2Status);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdV2(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwdV2);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaFwdV2 launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
