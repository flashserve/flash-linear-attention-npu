/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "aclnn_chunk_kda_fwd.h"
#include "chunk_kda_fwd.h"
#include "aclnn_chunk_kda_fwd_v2.h"
#include "chunk_kda_fwd_v2.h"
#include "../../arch35/chunk32/op_host/op_api/chunk_kda_fwd_fused.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
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
    // 归一化 / 指数开关:v1 入口固定使用默认值,
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
    CHECK_COND(layout != nullptr, ACLNN_ERR_PARAM_INVALID,
               "layout must be uppercase and one of BSND, BNSD, TND or NTD.");
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = KdaFwdLayout::BSND;
    } else if (std::strcmp(layout, "BNSD") == 0) {
        parsed = KdaFwdLayout::BNSD;
    } else if (std::strcmp(layout, "TND") == 0) {
        parsed = KdaFwdLayout::TND;
    } else if (std::strcmp(layout, "NTD") == 0) {
        parsed = KdaFwdLayout::NTD;
    } else {
        CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                   "layout must be uppercase and one of BSND, BNSD, TND or NTD.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckCuSeqlens(const aclIntArray *cuSeqlens, int64_t seqlen)
{
    if (cuSeqlens == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(cuSeqlens->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND((*cuSeqlens)[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional[0] must be 0.");
    CHECK_COND((*cuSeqlens)[cuSeqlens->Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional last element must equal the sequence length.");
    for (size_t idx = 0; idx + 1 < cuSeqlens->Size(); ++idx) {
        CHECK_COND((*cuSeqlens)[idx] <= (*cuSeqlens)[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be nondecreasing.");
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
    CHECK_COND(cuSeqlens != nullptr, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional requires cuSeqlensOptional.");
    CHECK_COND(chunkIndices->Size() == static_cast<size_t>(totalChunks) * 2,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain exactly one (seq_id, chunk_id) pair per chunk.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        const int64_t chunks = (length + chunkSize - 1) / chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND((*chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                           (*chunkIndices)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional must use canonical sequence-major chunk order.");
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
    CHECK_COND(Rank(params.q) == tensorRank && Rank(params.k) == tensorRank &&
                   Rank(params.v) == tensorRank && Rank(params.g) == tensorRank &&
                   Rank(params.beta) == betaRank,
               ACLNN_ERR_PARAM_INVALID,
               "q/k/v/g and beta ranks must match layout: rank3/rank2 for TND/NTD, rank4/rank3 for BSND/BNSD.");
    CHECK_COND(SameShape(params.q, params.k), ACLNN_ERR_PARAM_INVALID,
               "q and k must have identical shape.");

    if (layout == KdaFwdLayout::TND) {
        info.batch = 1;
        info.seqlen = Dim(params.q, 0);
        info.hNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 2);
        CHECK_COND(HasShape(params.v, {info.seqlen, info.hvNum, info.vDim}) &&
                       HasShape(params.g, {info.seqlen, info.hvNum, info.kDim}) &&
                       HasShape(params.beta, {info.seqlen, info.hvNum}),
                   ACLNN_ERR_PARAM_INVALID, "TND expects v/g/beta as [T,HV,V], [T,HV,K], [T,HV].");
    } else if (layout == KdaFwdLayout::NTD) {
        info.batch = 1;
        info.hNum = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 0);
        info.vDim = Dim(params.v, 2);
        CHECK_COND(HasShape(params.v, {info.hvNum, info.seqlen, info.vDim}) &&
                       HasShape(params.g, {info.hvNum, info.seqlen, info.kDim}) &&
                       HasShape(params.beta, {info.hvNum, info.seqlen}),
                   ACLNN_ERR_PARAM_INVALID, "NTD expects v/g/beta as [HV,T,V], [HV,T,K], [HV,T].");
    } else if (layout == KdaFwdLayout::BSND) {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.hNum = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.seqlen, info.hvNum, info.vDim}) &&
                       HasShape(params.g, {info.batch, info.seqlen, info.hvNum, info.kDim}) &&
                       HasShape(params.beta, {info.batch, info.seqlen, info.hvNum}),
                   ACLNN_ERR_PARAM_INVALID,
                   "BSND expects v/g/beta as [B,T,HV,V], [B,T,HV,K], [B,T,HV].");
    } else {
        info.batch = Dim(params.q, 0);
        info.hNum = Dim(params.q, 1);
        info.seqlen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.hvNum, info.seqlen, info.vDim}) &&
                       HasShape(params.g, {info.batch, info.hvNum, info.seqlen, info.kDim}) &&
                       HasShape(params.beta, {info.batch, info.hvNum, info.seqlen}),
                   ACLNN_ERR_PARAM_INVALID,
                   "BNSD expects v/g/beta as [B,HV,T,V], [B,HV,T,K], [B,HV,T].");
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
    CHECK_COND((dataType == DataType::DT_FLOAT16 || dataType == DataType::DT_BF16) &&
                   params.k->GetDataType() == dataType && params.v->GetDataType() == dataType,
               ACLNN_ERR_PARAM_INVALID, "q, k and v must use the same float16 or bfloat16 dtype.");
    const DataType gateType = params.g->GetDataType();
    CHECK_COND(gateType == DataType::DT_FLOAT || gateType == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "g must be float32 or bfloat16.");
    const DataType betaType = params.beta->GetDataType();
    CHECK_COND(betaType == DataType::DT_FLOAT || betaType == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "beta must be float32 or bfloat16.");
    if (params.aLogOptional != nullptr) {
        CHECK_COND(params.aLogOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "aLogOptional must be float32.");
    }
    if (params.dtBiasOptional != nullptr) {
        CHECK_COND(params.dtBiasOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "dtBiasOptional must be float32.");
    }
    if (params.initialStateOptional != nullptr) {
        CHECK_COND(params.initialStateOptional->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "initialStateOptional must be float32.");
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
    CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID,
               "%s must be [N,HV,K,V] when stateVFirst=false and [N,HV,V,K] otherwise.", name);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const ChunkKdaFwdParams &params, const KdaShapeInfo &info)
{
    const DataType dataType = params.q->GetDataType();
    const bool attnShapeValid = info.isRank3
                                    ? HasShape(params.attnOut, {info.seqlen, info.hvNum, info.vDim})
                                    : HasShape(params.attnOut,
                                               {info.batch, info.seqlen, info.hvNum, info.vDim});
    CHECK_COND(attnShapeValid && params.attnOut->GetDataType() == dataType,
               ACLNN_ERR_PARAM_INVALID,
               "attnOut must match q dtype and use fixed sequence-major TND/BSND layout.");
    if (params.gkOut != nullptr) {
        const bool valid = info.isRank3
                               ? HasShape(params.gkOut, {info.hvNum, info.seqlen, info.kDim})
                               : HasShape(params.gkOut,
                                          {info.batch, info.hvNum, info.seqlen, info.kDim});
        CHECK_COND(valid && params.gkOut->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "gkOut must be float32 in fixed head-major NTD/BNSD layout.");
    }
    const aclTensor *matrixOutputs[] = {params.aqkOut, params.akkOut};
    for (const aclTensor *output : matrixOutputs) {
        if (output == nullptr) {
            // Akk 为 OPTIONAL：省略时不导出，V2 组合入口退到 Prepare 的 none 档。
            continue;
        }
        const bool valid = info.isRank3
                               ? HasShape(output, {info.hvNum, info.seqlen, params.chunkSize})
                               : HasShape(output,
                                          {info.batch, info.hvNum, info.seqlen, params.chunkSize});
        CHECK_COND(valid && output->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "Aqk/Akk must match q dtype and use fixed head-major NTD/BNSD layout when provided.");
    }
    const aclTensor *kOutputs[] = {params.wOut, params.qgOut, params.kgOut};
    for (const aclTensor *output : kOutputs) {
        if (output == nullptr) {
            continue;
        }
        const bool valid = info.isRank3
                               ? HasShape(output, {info.hvNum, info.seqlen, info.kDim})
                               : HasShape(output,
                                          {info.batch, info.hvNum, info.seqlen, info.kDim});
        CHECK_COND(valid && output->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "w/qg/kg must match q dtype and use fixed head-major NTD/BNSD layout.");
    }
    const aclTensor *vOutputs[] = {params.uOut, params.vNewOut};
    for (const aclTensor *output : vOutputs) {
        if (output == nullptr) {
            continue;
        }
        const bool valid = info.isRank3
                               ? HasShape(output, {info.hvNum, info.seqlen, info.vDim})
                               : HasShape(output,
                                          {info.batch, info.hvNum, info.seqlen, info.vDim});
        CHECK_COND(valid && output->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "u/vNew must match q dtype and use fixed head-major NTD/BNSD layout.");
    }
    if (params.hOut != nullptr) {
        const bool valid = params.stateVFirst
                               ? (info.isRank3
                                      ? HasShape(params.hOut,
                                                 {info.totalChunks, info.hvNum, info.vDim, info.kDim})
                                      : HasShape(params.hOut,
                                                 {info.batch, info.totalChunks, info.hvNum,
                                                  info.vDim, info.kDim}))
                               : (info.isRank3
                                      ? HasShape(params.hOut,
                                                 {info.totalChunks, info.hvNum, info.kDim, info.vDim})
                                      : HasShape(params.hOut,
                                                 {info.batch, info.totalChunks, info.hvNum,
                                                  info.kDim, info.vDim}));
        CHECK_COND(valid && params.hOut->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "hOut must match q dtype, use fixed sequence-major layout, and follow stateVFirst.");
    }
    CHECK_RET(CheckStateShape(params.finalStateOut, "finalStateOut", info, params.stateVFirst) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (params.finalStateOut != nullptr) {
        CHECK_COND(params.finalStateOut->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "finalStateOut must be float32.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkKdaFwdParams &params, KdaFwdLayout &layout, KdaShapeInfo &info)
{
    CHECK_COND(params.q != nullptr && params.k != nullptr && params.v != nullptr &&
                   params.g != nullptr && params.beta != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q, k, v, g and beta must not be nullptr.");
    CHECK_COND(params.attnOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "attnOut must not be nullptr.");
    // aqkOut 对应 op def 的 REQUIRED 输出；akkOut 对应 OPTIONAL，省略时算子内部
    // 自建不导出的占位张量，V2 组合入口随之使用 Prepare 的 none 档。
    CHECK_COND(params.aqkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "aqkOut must not be nullptr.");
    CHECK_COND(params.chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "chunkSize only supports 64 on the three-stage path "
               "(32 is handled by the chunk_kda_fwd_fused kernel).");
    CHECK_RET(ParseLayout(params.layout, layout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ResolveShapeInfo(params, layout, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    // 空 tensor（batch 或序列长度为 0）无法构成可执行的计算规模。host 侧在进入 tiling
    // 之前拦截并说明是哪个逻辑维为 0，避免只向调用方回传无上下文的 561103。
    CHECK_COND(info.batch > 0, ACLNN_ERR_PARAM_INVALID,
               "batch dimension must be greater than 0, but got B=%ld.", info.batch);
    CHECK_COND(info.seqlen > 0, ACLNN_ERR_PARAM_INVALID,
               "sequence length must be greater than 0, but got T=%ld.", info.seqlen);
    CHECK_COND(info.hNum > 0 && info.hvNum >= info.hNum && info.hvNum % info.hNum == 0,
               ACLNN_ERR_PARAM_INVALID,
               "H and HV must be positive, HV must be greater than or equal to H, and HV must be divisible by H.");
    CHECK_COND(info.hNum <= MAX_KDA_HEAD_NUM && info.hvNum <= MAX_KDA_HEAD_NUM,
               ACLNN_ERR_PARAM_INVALID, "H and HV must be less than or equal to 128.");
    // K/V 仅支持 K=V=64 与 K=V=128 两档，不支持混合档（如 K=64/V=128）或
    // 其它取值；报错打印实际 K/V，便于调用方直接定位 shape。
    const bool kvDimSupported =
        (info.kDim == KDA_FWD_KV_DIM_SMALL && info.vDim == KDA_FWD_KV_DIM_SMALL) ||
        (info.kDim == KDA_FWD_KV_DIM_LARGE && info.vDim == KDA_FWD_KV_DIM_LARGE);
    CHECK_COND(kvDimSupported, ACLNN_ERR_PARAM_INVALID,
               "K/V must both be 64 or both be 128; mixed K/V and other dims are not "
               "supported, but got Kdim=%ld, Vdim=%ld.",
               info.kDim, info.vDim);
    CHECK_RET(CheckDtypes(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckCuSeqlens(params.cuSeqlensOptional, info.seqlen) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.isRank3 || info.batch == 1,
               ACLNN_ERR_PARAM_INVALID,
               "rank4 varlen input with cuSeqlensOptional requires B=1.");
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.seqNum <= MAX_KDA_VARLEN_SEQUENCES,
               ACLNN_ERR_PARAM_INVALID, "varlen input supports at most 1024 sequences.");
    CHECK_RET(CheckChunkIndices(params.chunkIndicesOptional, params.cuSeqlensOptional,
                                info.totalChunks, params.chunkSize) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckStateShape(params.initialStateOptional, "initialStateOptional", info,
                              params.stateVFirst) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    // 归一化 / gate 开关约束：epsilon 只在 in-kernel L2 norm 时参与计算，
    // allowNegEigval 依赖 in-kernel beta sigmoid。
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    CHECK_COND(std::isfinite(epsilonFp32) && epsilonFp32 > 0.0F, ACLNN_ERR_PARAM_INVALID,
               "epsilon must be a positive finite number, but got %f.", params.epsilon);
    CHECK_COND(!params.allowNegEigval || params.useBetaSigmoidInKernel, ACLNN_ERR_PARAM_INVALID,
               "allowNegEigval=true requires useBetaSigmoidInKernel=true.");
    if (params.useGateInKernel) {
        CHECK_COND(params.aLogOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "aLogOptional is required when useGateInKernel is true.");
        CHECK_COND(HasShape(params.aLogOptional, {info.hvNum}), ACLNN_ERR_PARAM_INVALID,
                   "aLogOptional must have shape [HV].");
        if (params.dtBiasOptional != nullptr) {
            CHECK_COND(HasShape(params.dtBiasOptional, {info.hvNum * info.kDim}),
                       ACLNN_ERR_PARAM_INVALID, "dtBiasOptional must have shape [HV*K].");
        }
        if (params.safeGate) {
            CHECK_COND(params.lowerBound >= -5.0 && params.lowerBound < 0.0,
                       ACLNN_ERR_PARAM_INVALID,
                       "lowerBound must be in [-5, 0) when safeGate is true.");
        }
    }
    CHECK_RET(CheckOutputShapes(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus ContiguousInputs(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    CHECK_RET(MakeContiguous(params.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.initialStateOptional, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
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
// ==================== chunk32 大融合路由(issue #829)====================
// chunkSize==32 → 单 kernel 大融合(chunk_kda_fwd_fused)。
// chunk32 没有第二条实现路径,不满足融合 kernel 契约即报错。
// kernel 契约:[B*H,T,K] 头连续 bf16(batch 拍平进头维);K=V=128;任意 T>0
// (尾块核内钳制);HV % H == 0;dense batch 使用单启动任务循环;
// 原始 g + a_log(fp32 [H]) + dt_bias(fp32 [H*K]);beta 核内 sigmoid、q/k 核内 L2 归一;
// 产出 attn_out + 可选 final_state;可选 initial_state(fp32 [1,H,128,128],state_v_first=false);
// varlen 支持 TND/NTD;不支持中间态导出。
aclnnStatus RunChunkKdaFwd32FusedVarlen(ChunkKdaFwdParams &params, aclOpExecutor *executor,
                                        const KdaShapeInfo &info);

aclnnStatus CheckChunkKdaFwd32Params(const ChunkKdaFwdParams &params, KdaFwdLayout &layout,
                                     KdaShapeInfo &info)
{
    CHECK_COND(params.q != nullptr && params.k != nullptr && params.v != nullptr &&
                   params.g != nullptr && params.beta != nullptr && params.attnOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q, k, v, g, beta and attnOut must not be nullptr.");
    CHECK_COND(params.safeGate, ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) currently requires safeGate=true.");
    CHECK_RET(ParseLayout(params.layout, layout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ResolveShapeInfo(params, layout, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckCuSeqlens(params.cuSeqlensOptional, info.seqlen) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.isRank3 || info.batch == 1,
               ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) rank4 varlen input requires B=1.");
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.seqNum <= MAX_KDA_VARLEN_SEQUENCES,
               ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) varlen supports at most 1024 sequences.");
    CHECK_RET(CheckChunkIndices(params.chunkIndicesOptional, params.cuSeqlensOptional,
                                info.totalChunks, params.chunkSize) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.gkOut == nullptr &&
                   params.wOut == nullptr && params.uOut == nullptr &&
                   params.qgOut == nullptr && params.kgOut == nullptr &&
                   params.vNewOut == nullptr && params.hOut == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) only exports attn_out/final_state; gk/w/u/qg/kg/v_new/h "
               "are not produced.");
    if (params.initialStateOptional != nullptr || params.finalStateOut != nullptr) {
        CHECK_COND(!params.stateVFirst, ACLNN_ERR_PARAM_INVALID,
                   "chunkSize=32 (fused kernel) state IO requires state_v_first=false.");
    }
    CHECK_COND(params.useGateInKernel && params.aLogOptional != nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) requires useGateInKernel=true with aLog.");
    CHECK_COND(params.dtBiasOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) requires dtBiasOptional to be non-null.");
    CHECK_COND(HasShape(params.aLogOptional, {info.hvNum}), ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) aLogOptional must have shape [HV].");
    CHECK_COND(HasShape(params.dtBiasOptional, {info.hvNum * info.kDim}), ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) dtBiasOptional must have shape [HV*K].");
    CHECK_RET(CheckDtypes(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    // q/k/v 必须 bf16;g/beta 到达侧只可能
    // fp32/bf16,规则见下方 gateDt:任一 fp32 → 两者无损上转 fp32(组合 1)。
    CHECK_COND(params.q->GetDataType() == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) requires q/k/v all bf16.");
    // dense batch:kernel 单启动内按任务(task = batch*HV + value_head)循环,
    // 拍平后 HV 维携带 batch;dt_bias/a_log 由 tiling 传 hvNumReal 保持按真实头寻址。
    CHECK_COND(info.batch >= 1, ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) requires batch >= 1, but got B=%ld.", info.batch);
    // GVA(issue #829):HV 是 H 的整数倍即可(H==HV 为特例),kernel 按组映射 q/k 头
    CHECK_COND(info.hNum > 0 && info.hvNum >= info.hNum && info.hvNum % info.hNum == 0,
               ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) requires HV to be a positive multiple of H, "
               "but got H=%ld HV=%ld.", info.hNum, info.hvNum);
    CHECK_COND(info.kDim == KDA_FWD_KV_DIM_LARGE && info.vDim == KDA_FWD_KV_DIM_LARGE,
               ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) requires K == V == 128, but got K=%ld V=%ld.",
               info.kDim, info.vDim);
    CHECK_COND(info.seqlen > 0, ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) requires T > 0, got T=%ld.",
               info.seqlen);
    CHECK_RET(CheckStateShape(params.initialStateOptional, "initialStateOptional", info,
                              params.stateVFirst) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputShapes(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus RunChunkKdaFwd32FusedDense(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    KdaFwdLayout layout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    CHECK_RET(CheckChunkKdaFwd32Params(params, layout, info) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInputs(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // dense BSND:等长 batch 可表示为整齐 varlen,合成
    // cu=[0,T,2T,...],[B,S,H,*] 连续视图零成本压成 packed [B*S,H,*],走 varlen
    // 核内直读路径,省掉 q/k/v/g 的 host 大转置(实测占端到端 ~28%)与回拷。
    // B=1 同样成立(cu=[0,T]);BNSD 本就头连续零转置,均不受此影响。
    if (!info.isRank3 && layout == KdaFwdLayout::BSND) {
        std::vector<int64_t> cuVals(info.batch + 1);
        for (int64_t b = 0; b <= info.batch; ++b) {
            cuVals[b] = b * info.seqlen;
        }
        aclIntArray *cuArr = aclCreateIntArray(cuVals.data(), cuVals.size());
        CHECK_RET(cuArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
        ChunkKdaFwdParams ps = params;
        const int64_t totalTok = info.batch * info.seqlen;
        ps.q = l0op::Reshape(params.q, MakeShape({totalTok, info.hNum, info.kDim}), executor);
        ps.k = l0op::Reshape(params.k, MakeShape({totalTok, info.hNum, info.kDim}), executor);
        ps.v = l0op::Reshape(params.v, MakeShape({totalTok, info.hvNum, info.vDim}), executor);
        ps.g = l0op::Reshape(params.g, MakeShape({totalTok, info.hvNum, info.kDim}), executor);
        ps.beta = l0op::Reshape(params.beta, MakeShape({totalTok, info.hvNum}), executor);
        ps.attnOut = l0op::Reshape(params.attnOut, MakeShape({totalTok, info.hvNum, info.vDim}), executor);
        ps.cuSeqlensOptional = cuArr;
        aclnnStatus ret = ACLNN_ERR_INNER;
        if (ps.q != nullptr && ps.k != nullptr && ps.v != nullptr && ps.g != nullptr &&
            ps.beta != nullptr && ps.attnOut != nullptr) {
            KdaShapeInfo vinfo;   // packed 形态:TND 同构(batch=1, seqlen=B*T)
            vinfo.isRank3 = true;
            vinfo.batch = 1;
            vinfo.seqlen = totalTok;
            vinfo.hNum = info.hNum;
            vinfo.hvNum = info.hvNum;
            vinfo.kDim = info.kDim;
            vinfo.vDim = info.vDim;
            vinfo.seqNum = info.batch;
            ret = RunChunkKdaFwd32FusedVarlen(ps, executor, vinfo);
        }
        aclDestroyIntArray(cuArr);
        return ret;
    }

    // 布局规整到 [H,T,K] 头连续:NTD rank3 直达,其余 host 侧转置
    const aclTensor *qNtd = params.q;
    const aclTensor *kNtd = params.k;
    const aclTensor *vNtd = params.v;
    const aclTensor *gNtd = params.g;
    const aclTensor *betaNtd = params.beta;
    const bool rank3Ntd = info.isRank3 && layout == KdaFwdLayout::NTD;
    if (!rank3Ntd) {
        if (info.isRank3 && layout == KdaFwdLayout::TND) {
            qNtd = Transpose(params.q, {1, 0, 2}, executor);
            kNtd = Transpose(params.k, {1, 0, 2}, executor);
            vNtd = Transpose(params.v, {1, 0, 2}, executor);
            gNtd = Transpose(params.g, {1, 0, 2}, executor);
            betaNtd = Transpose(params.beta, {1, 0}, executor);
        } else if (!info.isRank3 && layout == KdaFwdLayout::BSND) {
            qNtd = Transpose(params.q, {0, 2, 1, 3}, executor);
            kNtd = Transpose(params.k, {0, 2, 1, 3}, executor);
            vNtd = Transpose(params.v, {0, 2, 1, 3}, executor);
            gNtd = Transpose(params.g, {0, 2, 1, 3}, executor);
            betaNtd = Transpose(params.beta, {0, 2, 1}, executor);
        } else if (!info.isRank3 && layout == KdaFwdLayout::BNSD) {
            // [B,H,S,*] 已是头连续,去 batch 维即可
        } else {
            CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                       "chunkSize=32 (fused kernel) got unsupported layout/rank combination.");
        }
        CHECK_RET(qNtd != nullptr && kNtd != nullptr && vNtd != nullptr && gNtd != nullptr &&
                      betaNtd != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        if (!info.isRank3) {
            // [B,H,T,*] → [B*H,T,*](batch 拍平进头维;转置后物理布局 batch-outer 头连续,
            // flatten 合法;kernel 任务号 fh = batch*HV + head 与此布局一一对应)
            qNtd = l0op::Reshape(qNtd, MakeShape({info.batch * info.hNum, info.seqlen, info.kDim}), executor);
            kNtd = l0op::Reshape(kNtd, MakeShape({info.batch * info.hNum, info.seqlen, info.kDim}), executor);
            vNtd = l0op::Reshape(vNtd, MakeShape({info.batch * info.hvNum, info.seqlen, info.vDim}), executor);
            gNtd = l0op::Reshape(gNtd, MakeShape({info.batch * info.hvNum, info.seqlen, info.kDim}), executor);
            betaNtd = l0op::Reshape(betaNtd, MakeShape({info.batch * info.hvNum, info.seqlen}), executor);
            CHECK_RET(qNtd != nullptr && kNtd != nullptr && vNtd != nullptr && gNtd != nullptr &&
                          betaNtd != nullptr,
                      ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // g/beta dtype 规则:任一为 fp32 → 两者都无损上转 fp32
    // (bf16→fp32 逐位保真),走 def 组合 1(kernel 核内 fp32 原生,全程无量化);
    // 全 bf16 → 组合 0。q 恒 bf16,不再以 q 的 dtype 为锚。
    const bool gateFp32 = (gNtd->GetDataType() == DataType::DT_FLOAT) ||
                          (betaNtd->GetDataType() == DataType::DT_FLOAT);
    const DataType gateDt = gateFp32 ? DataType::DT_FLOAT : DataType::DT_BF16;
    if (gNtd->GetDataType() != gateDt) {
        gNtd = l0op::Cast(gNtd, gateDt, executor);
        CHECK_RET(gNtd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (betaNtd->GetDataType() != gateDt) {
        betaNtd = l0op::Cast(betaNtd, gateDt, executor);
        CHECK_RET(betaNtd != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 输出:attnOut 恒定序列主序(TND/BSND,见 CheckOutputShapes),kernel 产出
    // [H,T,V] 头连续,统一写临时张量再转置回写。
    const aclTensor *oTmp = AllocTensor(executor, MakeShape({info.batch * info.hvNum, info.seqlen, info.vDim}),
                                        params.q->GetDataType());
    CHECK_RET(oTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // initial_state:[N_seq,H,K,V] 拍平成 [N_seq*H,K,V](kernel 按 (seq*H+head) 寻址);
    // final_state 先写 [N_seq*H,K,V] 临时张量,结束后 ViewCopy 回 [N_seq,H,K,V] 输出槽。
    const int64_t stateHeads = info.seqNum * info.hvNum;
    const aclTensor *initialStateHead = nullptr;
    if (params.initialStateOptional != nullptr) {
        initialStateHead = l0op::Reshape(params.initialStateOptional,
                                         MakeShape({stateHeads, info.kDim, info.vDim}), executor);
        CHECK_RET(initialStateHead != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor *finalStateTmp = nullptr;
    if (params.finalStateOut != nullptr) {
        finalStateTmp = AllocTensor(executor, MakeShape({stateHeads, info.kDim, info.vDim}),
                                    DataType::DT_FLOAT);
        CHECK_RET(finalStateTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // optional 输出缺省时本框架不生成 placeholder,kernel 参数
        // 序列少一个指针,tiling 指针错位 → AICore 异常(507015)。始终给一个真实张量:
        // [1,1,1] 在 tiling 侧按"未请求"处理(ShapeSize==1),kernel 不会写它。
        finalStateTmp = AllocTensor(executor, MakeShape({1, 1, 1}), DataType::DT_FLOAT);
        CHECK_RET(finalStateTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *fused = l0op::KdaChunkForwardFused(
        qNtd, kNtd, vNtd, gNtd, betaNtd, params.aLogOptional, params.dtBiasOptional,
        initialStateHead, params.scale, params.lowerBound, oTmp, finalStateTmp,
        nullptr /* dense 无 cu_seqlens */, params.epsilon, executor);
    CHECK_RET(fused != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *oBack = nullptr;
    if (info.isRank3) {  // TND/NTD:[H,T,V] → [T,H,V]
        oBack = Transpose(oTmp, {1, 0, 2}, executor);
    } else {             // BSND/BNSD:[B*HV,S,V] → [B,HV,S,V] → [B,S,HV,V]
        const aclTensor *oRank4 =
            l0op::Reshape(oTmp, MakeShape({info.batch, info.hvNum, info.seqlen, info.vDim}), executor);
        CHECK_RET(oRank4 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        oBack = Transpose(oRank4, {0, 2, 1, 3}, executor);
    }
    CHECK_RET(oBack != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto copied = l0op::ViewCopy(oBack, params.attnOut, executor);
    CHECK_RET(copied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (params.finalStateOut != nullptr) {
        const aclTensor *fsBack = l0op::Reshape(
            finalStateTmp, MakeShape({info.seqNum, info.hvNum, info.kDim, info.vDim}), executor);
        CHECK_RET(fsBack != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto fsCopied = l0op::ViewCopy(fsBack, params.finalStateOut, executor);
        CHECK_RET(fsCopied != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

// ==================== chunk32 varlen:单 kernel 启动====================
// 任务 = seq*HV + value_head 拍平,一次融合 kernel 启动处理全部序列:序列边界经
// cu_seqlens attr 打包进 tiling,kernel 逐任务解析;packed token-major 布局核内
// 跨步直读(不再逐序列转置/启动);空序列 kernel 内写 final=initial(或 0)。
aclnnStatus RunChunkKdaFwd32FusedVarlen(ChunkKdaFwdParams &params, aclOpExecutor *executor,
                                        const KdaShapeInfo &info)
{
    const int64_t nSeq = static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;

    CHECK_RET(ContiguousInputs(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // token-major 视图:BSND [1,T,*] 去 B 维;TND 直通
    const aclTensor *qTok = params.q;
    const aclTensor *kTok = params.k;
    const aclTensor *vTok = params.v;
    const aclTensor *gTok = params.g;
    const aclTensor *betaTok = params.beta;
    const aclTensor *oTok = params.attnOut;
    if (!info.isRank3) {   // BSND:[1,T,HV,*] → [T,HV,*](B==1 由外层契约卡住)
        qTok = l0op::Reshape(params.q, MakeShape({info.seqlen, info.hNum, info.kDim}), executor);
        kTok = l0op::Reshape(params.k, MakeShape({info.seqlen, info.hNum, info.kDim}), executor);
        vTok = l0op::Reshape(params.v, MakeShape({info.seqlen, info.hvNum, info.vDim}), executor);
        gTok = l0op::Reshape(params.g, MakeShape({info.seqlen, info.hvNum, info.kDim}), executor);
        betaTok = l0op::Reshape(params.beta, MakeShape({info.seqlen, info.hvNum}), executor);
        oTok = l0op::Reshape(params.attnOut, MakeShape({info.seqlen, info.hvNum, info.vDim}), executor);
        CHECK_RET(qTok != nullptr && kTok != nullptr && vTok != nullptr && gTok != nullptr &&
                      betaTok != nullptr && oTok != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    // g/beta dtype 规则同 dense 路径:任一 fp32 → 两者无损上转 fp32(组合 1),否则 bf16
    const bool gateFp32 = (gTok->GetDataType() == DataType::DT_FLOAT) ||
                          (betaTok->GetDataType() == DataType::DT_FLOAT);
    const DataType gateDt = gateFp32 ? DataType::DT_FLOAT : DataType::DT_BF16;
    if (gTok->GetDataType() != gateDt) {
        gTok = l0op::Cast(gTok, gateDt, executor);
        CHECK_RET(gTok != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (betaTok->GetDataType() != gateDt) {
        betaTok = l0op::Cast(betaTok, gateDt, executor);
        CHECK_RET(betaTok != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // beta 仅 token-major 需要转置为头连续 [HV,T]；NTD 输入已是该布局。
    if (std::strcmp(params.layout, "NTD") != 0) {
        betaTok = Transpose(betaTok, {1, 0}, executor);
        CHECK_RET(betaTok != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // state:[nSeq,HV,K,V] 拍平 [nSeq*HV,K,V](kernel 按任务号 fh = seq*HV+head 寻址)
    const int64_t stateHeads = nSeq * info.hvNum;
    const aclTensor *initialStateHead = nullptr;
    if (params.initialStateOptional != nullptr) {
        initialStateHead = l0op::Reshape(params.initialStateOptional,
                                         MakeShape({stateHeads, info.kDim, info.vDim}), executor);
        CHECK_RET(initialStateHead != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // final_state:直写调用方输出的拍平视图(少一次回拷);空序列契约 final=initial
    // 由整体预填兜住(kernel 只写非空序列;无 initial 时空序列由 kernel 清零)。
    const aclTensor *finalStateTmp = nullptr;
    if (params.finalStateOut != nullptr) {
        finalStateTmp = l0op::Reshape(params.finalStateOut,
                                      MakeShape({stateHeads, info.kDim, info.vDim}), executor);
        CHECK_RET(finalStateTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (initialStateHead != nullptr) {
            // 核内 istate 直通实测不稳定(见 kernel WriteEmptyFinalState 注释);
            // ViewCopy 的 dst 不能是 workspace 张量,故预填直接落在调用方输出上
            auto pre = l0op::ViewCopy(params.initialStateOptional, params.finalStateOut, executor);
            CHECK_RET(pre != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        finalStateTmp = AllocTensor(executor, MakeShape({1, 1, 1}), DataType::DT_FLOAT);
        CHECK_RET(finalStateTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *fused = l0op::KdaChunkForwardFused(
        qTok, kTok, vTok, gTok, betaTok, params.aLogOptional, params.dtBiasOptional,
        initialStateHead, params.scale, params.lowerBound, oTok, finalStateTmp,
        params.cuSeqlensOptional, params.epsilon, executor,
        std::strcmp(params.layout, "NTD") == 0 ? "NTD" : "TND");
    CHECK_RET(fused != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus RunChunkKdaFwd32Fused(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    // epsilon 用于 q/k 核内 L2 归一化并传递到 kernel;chunk32 路径不经过
    // CheckParams,校验在此兜底(与通用路径同一语义:转 fp32 后有限且大于 0)
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    CHECK_COND(std::isfinite(epsilonFp32) && epsilonFp32 > 0.0F, ACLNN_ERR_PARAM_INVALID,
               "epsilon must be a positive finite number, but got %f.", params.epsilon);
    if (params.cuSeqlensOptional == nullptr) {
        return RunChunkKdaFwd32FusedDense(params, executor);
    }
    KdaFwdLayout layout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    CHECK_RET(CheckChunkKdaFwd32Params(params, layout, info) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    // varlen 支持 BSND/TND 的 token-major 布局和 NTD 的 head-major 布局;BNSD 使用 chunk64 路径
    CHECK_COND(layout != KdaFwdLayout::BNSD, ACLNN_ERR_PARAM_INVALID,
               "chunkSize=32 (fused kernel) varlen requires sequence-major layout (BSND/TND) or NTD; BNSD must use chunk64.");
    return RunChunkKdaFwd32FusedVarlen(params, executor, info);
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
    if (params.chunkSize == 32) {
        // chunk32 → 单 kernel 大融合(issue #829),独立校验与执行路径;
        // 产出 attn_out + 可选 final_state,不产出中间量,契约不满足时由路由内部报错。
        CHECK_RET(RunChunkKdaFwd32Fused(params, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    KdaFwdLayout parsedLayout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    CHECK_RET(CheckParams(params, parsedLayout, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInputs(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

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
    if (params.chunkSize == 32) {
        // issue #829:公开入口 aclnnChunkKdaFwdV2 同样接受 chunk_size=32,
        // 走与 V1 相同的单 kernel 大融合路径。V2 特有的反向保存值出口在 chunk32
        // 无意义,必须全空;kernel 语义固定(核内 L2norm/sigmoid、exp2),
        // 显式请求与之矛盾的开关直接拒绝,不静默误算。
        CHECK_COND(params.qHatOut == nullptr && params.kHatOut == nullptr &&
                   params.qRstdOut == nullptr && params.kRstdOut == nullptr &&
                   params.betaEffOut == nullptr,
                   ACLNN_ERR_PARAM_INVALID,
                   "chunkSize=32 (fused kernel) does not export backward saved values "
                   "(q_hat/k_hat/q_rstd/k_rstd/beta_eff).");
        CHECK_COND(params.useQkL2normInKernel && params.useBetaSigmoidInKernel &&
                   !params.allowNegEigval && params.useExp2,
                   ACLNN_ERR_PARAM_INVALID,
                   "chunkSize=32 (fused kernel) fixes use_qk_l2norm_in_kernel=true, "
                   "use_beta_sigmoid_in_kernel=true, allow_neg_eigval=false, use_exp2=true.");
        CHECK_RET(RunChunkKdaFwd32Fused(params, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    KdaFwdLayout parsedLayout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    CHECK_RET(CheckParams(params, parsedLayout, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInputs(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // V2 只实现三个独立算子的组合路径；场景不满足时由 Python 入口回落到
    // aclnnChunkKdaFwd，因此这里直接返回参数错误并说明要求。
    CHECK_COND(CanUseKdaFwdV2Path(params, info), ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaFwdV2 只支持三算子组合场景：q/k/v 为 BF16、K=V=128、chunk_size=64、"
               "公开输出连续且 cu_seqlens 严格递增，请改用 aclnnChunkKdaFwd。");
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
