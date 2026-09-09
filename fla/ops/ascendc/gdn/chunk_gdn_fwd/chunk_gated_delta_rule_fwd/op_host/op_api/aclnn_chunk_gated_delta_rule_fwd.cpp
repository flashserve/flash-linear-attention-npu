/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_gated_delta_rule_fwd.h"

#include "chunk_gated_delta_rule_fwd.h"
#include "../../../chunk_fwd_h/op_host/op_api/chunk_fwd_h.h"
#include "../../../chunk_fwd_o/op_host/op_api/chunk_fwd_o.h"
#include "../../../chunk_gated_delta_rule_fwd_prepare/op_host/op_api/chunk_gated_delta_rule_fwd_prepare.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <cstring>
#include <initializer_list>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_DIM = 128;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_V256 = 256;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 = 64;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128 = 128;

struct ChunkGatedDeltaRuleFwdParams {
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
    const char *layout = nullptr;
    double scale = 1.0;
    int64_t chunkSize = CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64;
    bool useExp2 = false;
    bool useQkL2norm = false;
    bool allowNegEigval = false;
    bool stateVFirst = false;
    const aclTensor *oOut = nullptr;
    const aclTensor *finalStateOutOptional = nullptr;
    const aclTensor *qHatOutOptional = nullptr;
    const aclTensor *kHatOutOptional = nullptr;
    const aclTensor *qRstdOutOptional = nullptr;
    const aclTensor *kRstdOutOptional = nullptr;
    const aclTensor *betaEffOutOptional = nullptr;
    const aclTensor *gCumsumOutOptional = nullptr;
    const aclTensor *aOutOptional = nullptr;
    const aclTensor *hOutOptional = nullptr;
};

enum class GdnLayout { BNSD, BSND, NTD, TND };

struct GdnShapeInfo {
    GdnLayout layout = GdnLayout::BNSD;
    bool isSequenceMajor = false;
    int64_t batch = 0;
    int64_t hq = 0;
    int64_t hv = 0;
    int64_t seqlen = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
};

static bool UsePreparePath(const ChunkGatedDeltaRuleFwdParams &params)
{
    return params.useExp2 && params.useQkL2norm;
}

static op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

static const aclIntArray *MakePerm(std::initializer_list<int64_t> dims, aclOpExecutor *executor)
{
    return executor->AllocIntArray(dims.begin(), dims.size());
}

static const aclTensor *TransposeContiguous(const aclTensor *tensor, std::initializer_list<int64_t> dims,
                                            aclOpExecutor *executor)
{
    const aclIntArray *perm = MakePerm(dims, executor);
    if (perm == nullptr) {
        return nullptr;
    }
    const aclTensor *permuted = l0op::Transpose(tensor, perm, executor);
    if (permuted == nullptr) {
        return nullptr;
    }
    const aclTensor *materialized = l0op::Contiguous(permuted, executor);
    if (materialized == nullptr) {
        return nullptr;
    }

    // Contiguous materializes the storage, but the resulting tensor can still
    // carry the transpose view metadata. Re-declare the logical shape so the
    // following custom ops see a dense BHT/BHTC tensor instead of a stale view.
    const aclTensor *reshaped = l0op::Reshape(materialized, permuted->GetViewShape(), executor);
    if (reshaped == nullptr) {
        return nullptr;
    }
    reshaped->SetStorageShape(reshaped->GetViewShape());
    reshaped->SetOriginalShape(reshaped->GetViewShape());
    return reshaped;
}

static int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

static size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

static bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> dims)
{
    if (tensor == nullptr || Rank(tensor) != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (Dim(tensor, index++) != dim) {
            return false;
        }
    }
    return true;
}

static bool ShapeEqual(const op::Shape &lhs, const op::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t index = 0; index < lhs.GetDimNum(); ++index) {
        if (lhs.GetDim(index) != rhs.GetDim(index)) {
            return false;
        }
    }
    return true;
}

static int64_t SeqNum(const ChunkGatedDeltaRuleFwdParams &params, int64_t batch)
{
    return params.cuSeqlensOptional == nullptr
               ? batch
               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
}

static int64_t ExpectedChunks(const ChunkGatedDeltaRuleFwdParams &params, int64_t seqlen)
{
    if (params.cuSeqlensOptional == nullptr) {
        return (seqlen + params.chunkSize - 1) / params.chunkSize;
    }

    int64_t total = 0;
    const aclIntArray &cu = *params.cuSeqlensOptional;
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        const int64_t length = cu[idx + 1] - cu[idx];
        total += (length + params.chunkSize - 1) / params.chunkSize;
    }
    return total;
}

static aclnnStatus CheckStateShape(const aclTensor *state, const char *name, int64_t seqNum, int64_t hv,
                                   int64_t kDim, int64_t vDim)
{
    if (state == nullptr) {
        return ACLNN_SUCCESS;
    }
    const auto shape = state->GetViewShape();
    CHECK_COND(shape.GetDimNum() == 4 && shape.GetDim(0) == seqNum && shape.GetDim(1) == hv &&
                   shape.GetDim(2) == kDim && shape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "%s must have shape [seqNum,Hv,K,V].", name);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMetadata(const ChunkGatedDeltaRuleFwdParams &params, int64_t seqlen)
{
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const aclIntArray &cu = *params.cuSeqlensOptional;
    CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlens must contain at least [0,totalTokens].");
    CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlens must start at 0 and end at T.");
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        CHECK_COND(cu[idx] <= cu[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlens must be nondecreasing.");
    }

    const aclIntArray &indices = *params.chunkIndicesOptional;
    const int64_t expectedChunks = ExpectedChunks(params, seqlen);
    CHECK_COND(indices.Size() % 2 == 0 && static_cast<int64_t>(indices.Size() / 2) == expectedChunks,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndices must contain one (seqId,localChunkId) pair per chunk.");
    size_t index = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        const int64_t seqChunks = (cu[seq + 1] - cu[seq] + params.chunkSize - 1) / params.chunkSize;
        for (int64_t localChunk = 0; localChunk < seqChunks; ++localChunk) {
            CHECK_COND(indices[index] == static_cast<int64_t>(seq) && indices[index + 1] == localChunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndices must use canonical sequence-major order.");
            index += 2;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ViewCopyIfPresent(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    if (dst == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(src != nullptr && l0op::ViewCopy(src, dst, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

#define GDN_STAGE_CHECK(condition, code) \
    do {                                  \
        if (!(condition)) {               \
            return static_cast<aclnnStatus>(code); \
        }                                 \
    } while (false)

static aclnnStatus CheckRank(const aclTensor *tensor, size_t rank, const char *name)
{
    CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "%s must not be nullptr.", name);
    CHECK_COND(tensor->GetViewShape().GetDimNum() == rank, ACLNN_ERR_PARAM_INVALID,
               "%s must be rank %zu.", name, rank);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOptionalRank(const aclTensor *tensor, size_t rank, const char *name)
{
    return tensor == nullptr ? ACLNN_SUCCESS : CheckRank(tensor, rank, name);
}

static bool IsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}

static aclnnStatus ResolveShapeInfo(const ChunkGatedDeltaRuleFwdParams &params, GdnShapeInfo &info)
{
    if (std::strcmp(params.layout, "BNSD") == 0) {
        info.layout = GdnLayout::BNSD;
    } else if (std::strcmp(params.layout, "BSND") == 0) {
        info.layout = GdnLayout::BSND;
    } else if (std::strcmp(params.layout, "NTD") == 0) {
        info.layout = GdnLayout::NTD;
    } else if (std::strcmp(params.layout, "TND") == 0) {
        info.layout = GdnLayout::TND;
    } else {
        CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                   "layout must be uppercase and one of BNSD, BSND, NTD or TND.");
    }
    info.isSequenceMajor = info.layout == GdnLayout::BSND || info.layout == GdnLayout::TND;
    CHECK_COND(Rank(params.q) == 4 && Rank(params.k) == 4 && Rank(params.v) == 4 &&
                   Rank(params.oOut) == 4 && Rank(params.g) == 3 && Rank(params.beta) == 3,
               ACLNN_ERR_PARAM_INVALID,
               "q/k/v/o must be rank 4 and g/beta must be rank 3.");
    CHECK_COND(ShapeEqual(params.q->GetViewShape(), params.k->GetViewShape()),
               ACLNN_ERR_PARAM_INVALID, "q and k must have identical shapes.");

    if (!info.isSequenceMajor) {
        info.batch = Dim(params.q, 0);
        info.hq = Dim(params.q, 1);
        info.seqlen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hv = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.hv, info.seqlen, info.vDim}),
                   ACLNN_ERR_PARAM_INVALID, "BNSD/NTD expects v as [B,Hv,T,V].");
    } else {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.hq = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hv = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.seqlen, info.hv, info.vDim}),
                   ACLNN_ERR_PARAM_INVALID, "BSND/TND expects v as [B,T,Hv,V].");
    }
    CHECK_COND(HasShape(params.oOut, {info.batch, info.seqlen, info.hv, info.vDim}),
               ACLNN_ERR_PARAM_INVALID, "oOut must use BSND shape [B,T,Hv,V].");
    CHECK_COND(HasShape(params.g, {info.batch, info.seqlen, info.hv}) &&
                   HasShape(params.beta, {info.batch, info.seqlen, info.hv}),
               ACLNN_ERR_PARAM_INVALID, "g and beta must have shape [B,T,Hv].");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSupportedL2Contract(const ChunkGatedDeltaRuleFwdParams &params)
{
    CHECK_COND(params.layout != nullptr, ACLNN_ERR_PARAM_NULLPTR, "layout must not be nullptr.");
    CHECK_COND(params.aLogOptional == nullptr && params.dtBiasOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "aLogOptional and dtBiasOptional are reserved by the stable ABI but are not supported yet.");
    if (UsePreparePath(params)) {
        CHECK_COND(IsAscend950(), ACLNN_ERR_PARAM_INVALID,
                   "useExp2=true is supported on Ascend950 only.");
        CHECK_COND(std::strcmp(params.layout, "BNSD") == 0 || std::strcmp(params.layout, "BSND") == 0 ||
                       std::strcmp(params.layout, "NTD") == 0 || std::strcmp(params.layout, "TND") == 0,
                   ACLNN_ERR_PARAM_INVALID,
                   "Ascend950 useExp2 path supports BNSD, BSND, NTD and TND.");
        CHECK_COND(!params.allowNegEigval || params.betaEffOutOptional != nullptr,
                   ACLNN_ERR_PARAM_INVALID,
                   "allowNegEigval=true requires betaEffOutOptional to enable beta sigmoid.");
        return ACLNN_SUCCESS;
    }
    CHECK_COND(std::strcmp(params.layout, "BNSD") == 0, ACLNN_ERR_PARAM_INVALID,
               "The current Phase 6 implementation supports layout=BNSD only.");
    CHECK_COND(!params.stateVFirst, ACLNN_ERR_PARAM_INVALID,
               "stateVFirst=true is supported by the Ascend950 useExp2 path only.");
    CHECK_COND(!params.allowNegEigval, ACLNN_ERR_PARAM_INVALID,
               "allowNegEigval=true is supported by the Ascend950 useExp2 path only.");
    CHECK_COND(params.qHatOutOptional == nullptr && params.kHatOutOptional == nullptr &&
                   params.qRstdOutOptional == nullptr && params.kRstdOutOptional == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "Q/K L2Norm intermediate outputs are reserved by the stable ABI but are not supported yet.");
    CHECK_COND(params.betaEffOutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "betaEffOutOptional is reserved by the stable ABI but is not supported yet.");
    CHECK_COND(params.hOutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "hOutOptional is reserved by the stable ABI but is not supported yet.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleFwdParams &params)
{
    const aclnnStatus supportedContractStatus = CheckSupportedL2Contract(params);
    if (supportedContractStatus != ACLNN_SUCCESS) {
        return supportedContractStatus;
    }
    CHECK_COND(params.q != nullptr && params.k != nullptr && params.v != nullptr && params.g != nullptr &&
                   params.beta != nullptr && params.oOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q/k/v/g/beta/oOut must not be nullptr.");
    GdnShapeInfo info;
    CHECK_RET(ResolveShapeInfo(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const size_t qkvRank = 4;
    const size_t scalarRank = 3;
    CHECK_RET(CheckOptionalRank(params.gCumsumOutOptional, scalarRank, "gCumsumOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.aOutOptional, qkvRank, "aOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.qHatOutOptional, qkvRank, "qHatOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.kHatOutOptional, qkvRank, "kHatOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.qRstdOutOptional, scalarRank, "qRstdOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.kRstdOutOptional, scalarRank, "kRstdOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.betaEffOutOptional, scalarRank, "betaEffOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOptionalRank(params.hOutOptional, 5, "hOutOptional") == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    const int64_t chunks = ExpectedChunks(params, info.seqlen);
    CHECK_COND(info.batch > 0 && info.hq > 0 && info.hv > 0 && info.seqlen > 0,
               ACLNN_ERR_PARAM_INVALID, "B/H/T dimensions must be positive.");
    CHECK_COND(info.hv % info.hq == 0, ACLNN_ERR_PARAM_INVALID,
               "Phase 6 GVA requires Hv divisible by Hk.");
    CHECK_COND(info.kDim == CHUNK_GATED_DELTA_RULE_FWD_DIM &&
                   (info.vDim == CHUNK_GATED_DELTA_RULE_FWD_DIM ||
                    info.vDim == CHUNK_GATED_DELTA_RULE_FWD_V256),
               ACLNN_ERR_PARAM_INVALID,
               "The Phase 6 composite GDN core supports K=128 and V=128/256.");
    if (params.gCumsumOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.gCumsumOutOptional->GetViewShape(), params.g->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID, "gCumsumOutOptional shape must match g.");
    }
    if (params.aOutOptional != nullptr) {
        const bool valid = HasShape(params.aOutOptional,
                                    {info.batch, info.hv, info.seqlen, params.chunkSize});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID,
                   "aOutOptional must use head-major [B,Hv,T,chunkSize].");
    }
    if (params.qHatOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.qHatOutOptional->GetViewShape(), params.q->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID,
                   "qHatOutOptional shape must match q.");
    }
    if (params.kHatOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.kHatOutOptional->GetViewShape(), params.k->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID,
                   "kHatOutOptional shape must match k.");
    }
    if (params.qRstdOutOptional != nullptr) {
        const bool valid = info.isSequenceMajor
                               ? HasShape(params.qRstdOutOptional, {info.batch, info.seqlen, info.hq})
                               : HasShape(params.qRstdOutOptional, {info.batch, info.hq, info.seqlen});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID, "qRstdOutOptional shape must follow q layout.");
    }
    if (params.kRstdOutOptional != nullptr) {
        const bool valid = info.isSequenceMajor
                               ? HasShape(params.kRstdOutOptional, {info.batch, info.seqlen, info.hq})
                               : HasShape(params.kRstdOutOptional, {info.batch, info.hq, info.seqlen});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID, "kRstdOutOptional shape must follow k layout.");
    }
    if (params.betaEffOutOptional != nullptr) {
        CHECK_COND(ShapeEqual(params.betaEffOutOptional->GetViewShape(), params.beta->GetViewShape()),
                   ACLNN_ERR_PARAM_INVALID, "betaEffOutOptional shape must match beta.");
    }
    if (params.hOutOptional != nullptr) {
        const int64_t stateDim0 = params.stateVFirst ? info.vDim : info.kDim;
        const int64_t stateDim1 = params.stateVFirst ? info.kDim : info.vDim;
        const bool valid = HasShape(params.hOutOptional,
                                    {info.batch, info.hv, chunks, stateDim0, stateDim1});
        CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID,
                   "hOutOptional shape must match stateVFirst.");
    }
    CHECK_COND(params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 ||
                   params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128.");
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "cuSeqlens and chunkIndices must be both present or both absent.");
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.batch == 1, ACLNN_ERR_PARAM_INVALID,
               "varlen rank-4 input requires physical B=1.");
    CHECK_RET(CheckMetadata(params, info.seqlen) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const int64_t seqNum = SeqNum(params, info.batch);
    const int64_t stateDim0 = params.stateVFirst ? info.vDim : info.kDim;
    const int64_t stateDim1 = params.stateVFirst ? info.kDim : info.vDim;
    CHECK_RET(CheckStateShape(params.initialStateOptional, "initialState", seqNum, info.hv, stateDim0, stateDim1) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (params.finalStateOutOptional != nullptr) {
        CHECK_RET(CheckStateShape(params.finalStateOutOptional, "finalStateOut", seqNum, info.hv,
                                  stateDim0, stateDim1) ==
                      ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    const DataType dtype = params.q->GetDataType();
    CHECK_COND(dtype == DataType::DT_FLOAT16 || dtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q/k/v must be float16 or bfloat16.");
    CHECK_COND(params.k->GetDataType() == dtype && params.v->GetDataType() == dtype &&
                   params.oOut->GetDataType() == dtype &&
                   (params.aOutOptional == nullptr || params.aOutOptional->GetDataType() == dtype),
               ACLNN_ERR_PARAM_INVALID, "q/k/v/oOut/aOutOptional must have the same dtype.");
    CHECK_COND((params.beta->GetDataType() == DataType::DT_FLOAT || params.beta->GetDataType() == dtype) &&
                   params.g->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "beta must be float32 or match q/k/v, and g must be float32.");
    CHECK_COND(params.gCumsumOutOptional == nullptr || params.gCumsumOutOptional->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "gCumsumOutOptional must be float32.");
    CHECK_COND((params.qHatOutOptional == nullptr || params.qHatOutOptional->GetDataType() == dtype) &&
                   (params.kHatOutOptional == nullptr || params.kHatOutOptional->GetDataType() == dtype) &&
                   (params.hOutOptional == nullptr || params.hOutOptional->GetDataType() == dtype),
               ACLNN_ERR_PARAM_INVALID, "qHatOutOptional/kHatOutOptional/hOutOptional must match q dtype.");
    CHECK_COND((params.qRstdOutOptional == nullptr ||
                    params.qRstdOutOptional->GetDataType() == DataType::DT_FLOAT) &&
                   (params.kRstdOutOptional == nullptr ||
                    params.kRstdOutOptional->GetDataType() == DataType::DT_FLOAT) &&
                   (params.betaEffOutOptional == nullptr ||
                    params.betaEffOutOptional->GetDataType() == DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "qRstdOutOptional/kRstdOutOptional/betaEffOutOptional must be float32.");
    if (UsePreparePath(params)) {
        CHECK_COND(dtype == DataType::DT_BF16 && info.vDim == CHUNK_GATED_DELTA_RULE_FWD_DIM &&
                       params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 && info.hv / info.hq <= 4,
                   ACLNN_ERR_PARAM_INVALID,
                   "Ascend950 useExp2 path requires BF16, K=V=128, chunkSize=64 and Hv/Hk in {1,2,3,4}.");
    }
    if (params.initialStateOptional != nullptr) {
        const DataType stateDtype = params.initialStateOptional->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_FLOAT || stateDtype == dtype, ACLNN_ERR_PARAM_INVALID,
                   "initialState dtype must be float32 or match q/k/v.");
    }
    if (params.finalStateOutOptional != nullptr) {
        const DataType expectedStateDtype = params.initialStateOptional == nullptr
                                                ? DataType::DT_FLOAT
                                                : params.initialStateOptional->GetDataType();
        CHECK_COND(params.finalStateOutOptional->GetDataType() == expectedStateDtype,
                   ACLNN_ERR_PARAM_INVALID,
                   "finalStateOut dtype must match initialState, or be float32 when initialState is absent.");
    }
    return ACLNN_SUCCESS;
}

} // namespace

static aclnnStatus ChunkGatedDeltaRuleFwdGetWorkspaceSizeImpl(
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
    bool useExp2,
    bool useQkL2norm,
    bool allowNegEigval,
    bool stateVFirst,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *qHatOutOptional,
    const aclTensor *kHatOutOptional,
    const aclTensor *qRstdOutOptional,
    const aclTensor *kRstdOutOptional,
    const aclTensor *betaEffOutOptional,
    const aclTensor *gCumsumOutOptional,
    const aclTensor *aOutOptional,
    const aclTensor *hOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkGatedDeltaRuleFwdParams params{
        q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
        cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize, useExp2,
        useQkL2norm, allowNegEigval, stateVFirst, oOut, finalStateOutOptional, qHatOutOptional,
        kHatOutOptional, qRstdOutOptional, kRstdOutOptional, betaEffOutOptional,
        gCumsumOutOptional, aOutOptional, hOutOptional};
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(MakeContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.initialStateOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    GdnShapeInfo info;
    CHECK_RET(ResolveShapeInfo(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const int64_t batch = info.batch;
    const int64_t hq = info.hq;
    const int64_t hv = info.hv;
    const int64_t seqlen = info.seqlen;
    const int64_t kDim = info.kDim;
    const int64_t vDim = info.vDim;
    const int64_t seqNum = SeqNum(params, batch);
    const bool outputFinalState = params.finalStateOutOptional != nullptr;
    const aclTensor *gSequence = params.g;
    const aclTensor *gBht = gSequence == nullptr
                                ? nullptr
                                : TransposeContiguous(gSequence, {0, 2, 1}, executorPtr);
    const aclTensor *betaFloat = params.beta->GetDataType() == DataType::DT_FLOAT
                                     ? params.beta
                                     : l0op::Cast(params.beta, DataType::DT_FLOAT, executorPtr);
    const aclTensor *betaSequence = betaFloat;
    const aclTensor *betaBht = betaSequence == nullptr
                                   ? nullptr
                                   : TransposeContiguous(betaSequence, {0, 2, 1}, executorPtr);
    GDN_STAGE_CHECK(gBht != nullptr && betaBht != nullptr, 169102);

    if (UsePreparePath(params)) {
        const op::Shape qkShape = MakeShape({batch, hq, seqlen, kDim});
        const op::Shape scalarQkShape = MakeShape({batch, hq, seqlen});
        const op::Shape scalarHvShape = MakeShape({batch, hv, seqlen});
        const op::Shape wShape = MakeShape({batch, hv, seqlen, kDim});
        const op::Shape vShape = MakeShape({batch, hv, seqlen, vDim});
        const op::Shape aShape = MakeShape({batch, hv, seqlen, params.chunkSize});
        const int64_t stateDim0 = params.stateVFirst ? vDim : kDim;
        const int64_t stateDim1 = params.stateVFirst ? kDim : vDim;
        const op::Shape hShape = MakeShape({batch, hv, ExpectedChunks(params, seqlen), stateDim0, stateDim1});
        const op::Shape stateShape = MakeShape({seqNum, hv, stateDim0, stateDim1});
        const DataType dtype = params.q->GetDataType();
        const DataType stateDtype = params.initialStateOptional == nullptr
                                        ? DataType::DT_FLOAT
                                        : params.initialStateOptional->GetDataType();

        const aclTensor *gCumsumBht = executorPtr->AllocTensor(scalarHvShape, DataType::DT_FLOAT, Format::FORMAT_ND);
        const aclTensor *w = executorPtr->AllocTensor(wShape, dtype, Format::FORMAT_ND);
        const aclTensor *u = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
        const aclTensor *a = executorPtr->AllocTensor(aShape, dtype, Format::FORMAT_ND);
        const aclTensor *qHat = executorPtr->AllocTensor(qkShape, dtype, Format::FORMAT_ND);
        const aclTensor *kHat = executorPtr->AllocTensor(qkShape, dtype, Format::FORMAT_ND);
        const aclTensor *qRstd = executorPtr->AllocTensor(scalarQkShape, DataType::DT_FLOAT, Format::FORMAT_ND);
        const aclTensor *kRstd = executorPtr->AllocTensor(scalarQkShape, DataType::DT_FLOAT, Format::FORMAT_ND);
        const aclTensor *betaEffBht = params.betaEffOutOptional == nullptr
                                          ? nullptr
                                          : executorPtr->AllocTensor(scalarHvShape, DataType::DT_FLOAT,
                                                                     Format::FORMAT_ND);
        const aclTensor *h = executorPtr->AllocTensor(hShape, dtype, Format::FORMAT_ND);
        const aclTensor *vNew = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
        const aclTensor *finalState = params.finalStateOutOptional == nullptr
                                          ? executorPtr->AllocTensor(stateShape, stateDtype, Format::FORMAT_ND)
                                          : params.finalStateOutOptional;
        GDN_STAGE_CHECK(gCumsumBht != nullptr && w != nullptr && u != nullptr && a != nullptr &&
                            qHat != nullptr && kHat != nullptr && qRstd != nullptr && kRstd != nullptr &&
                            h != nullptr && vNew != nullptr && finalState != nullptr &&
                            (params.betaEffOutOptional == nullptr || betaEffBht != nullptr),
                        169103);

        const aclTensor *qHead = params.q;
        const aclTensor *kHead = params.k;
        const aclTensor *vHead = params.v;
        if (info.isSequenceMajor) {
            qHead = TransposeContiguous(params.q, {0, 2, 1, 3}, executorPtr);
            kHead = TransposeContiguous(params.k, {0, 2, 1, 3}, executorPtr);
            vHead = TransposeContiguous(params.v, {0, 2, 1, 3}, executorPtr);
        }
        GDN_STAGE_CHECK(qHead != nullptr && kHead != nullptr && vHead != nullptr, 169108);

        auto prepareResult = l0op::ChunkGatedDeltaRuleFwdPrepare(
            qHead, kHead, vHead, gBht, betaBht, nullptr, nullptr,
            params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
            params.allowNegEigval, true, true, false, betaEffBht != nullptr,
            params.aOutOptional != nullptr, gCumsumBht, w, u, a, qHat, kHat, qRstd,
            kRstd, betaEffBht, executorPtr);
        GDN_STAGE_CHECK(prepareResult[0] != nullptr && prepareResult[1] != nullptr &&
                            prepareResult[2] != nullptr && prepareResult[4] != nullptr &&
                            prepareResult[5] != nullptr,
                        169104);

        auto hResult = l0op::ChunkFwdH(
            kHat, w, u, gCumsumBht, nullptr, params.initialStateOptional,
            params.cuSeqlensOptional, params.chunkIndicesOptional, outputFinalState,
            params.chunkSize, true, true, params.stateVFirst, h, vNew, finalState, executorPtr);
        GDN_STAGE_CHECK(hResult[0] != nullptr && hResult[1] != nullptr, 169105);

        auto oResult = l0op::ChunkFwdO(
            qHat, kHat, vNew, h, gCumsumBht, params.cuSeqlensOptional,
            params.chunkIndicesOptional, params.scale, params.chunkSize, true, params.stateVFirst,
            "BSND", params.oOut, executorPtr);
        GDN_STAGE_CHECK(oResult[0] != nullptr, 169106);

        if (params.qHatOutOptional != nullptr) {
            const aclTensor *qHatExport = qHat;
            if (info.isSequenceMajor) {
                qHatExport = TransposeContiguous(qHatExport, {0, 2, 1, 3}, executorPtr);
            }
            CHECK_RET(ViewCopyIfPresent(qHatExport, params.qHatOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.kHatOutOptional != nullptr) {
            const aclTensor *kHatExport = kHat;
            if (info.isSequenceMajor) {
                kHatExport = TransposeContiguous(kHatExport, {0, 2, 1, 3}, executorPtr);
            }
            CHECK_RET(ViewCopyIfPresent(kHatExport, params.kHatOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.qRstdOutOptional != nullptr) {
            const aclTensor *qRstdExport = qRstd;
            if (info.isSequenceMajor) {
                qRstdExport = TransposeContiguous(qRstdExport, {0, 2, 1}, executorPtr);
            }
            CHECK_RET(ViewCopyIfPresent(qRstdExport, params.qRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.kRstdOutOptional != nullptr) {
            const aclTensor *kRstdExport = kRstd;
            if (info.isSequenceMajor) {
                kRstdExport = TransposeContiguous(kRstdExport, {0, 2, 1}, executorPtr);
            }
            CHECK_RET(ViewCopyIfPresent(kRstdExport, params.kRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        const aclTensor *aExport = a;
        const aclTensor *hExport = h;
        CHECK_RET(ViewCopyIfPresent(aExport, params.aOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopyIfPresent(hExport, params.hOutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        if (params.gCumsumOutOptional != nullptr) {
            const aclTensor *gCumsumBth = TransposeContiguous(gCumsumBht, {0, 2, 1}, executorPtr);
            const aclTensor *gCumsumExport = gCumsumBth;
            CHECK_RET(ViewCopyIfPresent(gCumsumExport, params.gCumsumOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.betaEffOutOptional != nullptr) {
            const aclTensor *betaEffBth = TransposeContiguous(betaEffBht, {0, 2, 1}, executorPtr);
            const aclTensor *betaEffExport = betaEffBth;
            CHECK_RET(ViewCopyIfPresent(betaEffExport, params.betaEffOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto aStorageBhtc = executorPtr->AllocTensor(MakeShape({batch, hv, seqlen, params.chunkSize}),
                                                  params.k->GetDataType(), Format::FORMAT_ND);
    const aclTensor *finalState = params.finalStateOutOptional;
    if (!outputFinalState) {
        finalState = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
    }
    const aclTensor *gCumsumCompute = params.gCumsumOutOptional;
    if (gCumsumCompute == nullptr) {
        gCumsumCompute =
            executorPtr->AllocTensor(MakeShape({batch, seqlen, hv}), DataType::DT_FLOAT, Format::FORMAT_ND);
    }
    const aclTensor *aCompute = params.aOutOptional;
    if (aCompute == nullptr) {
        aCompute = executorPtr->AllocTensor(MakeShape({batch, hv, seqlen, params.chunkSize}), params.q->GetDataType(),
                                            Format::FORMAT_ND);
    }
    GDN_STAGE_CHECK(aStorageBhtc != nullptr && finalState != nullptr &&
                        gCumsumCompute != nullptr && aCompute != nullptr,
                    169101);

    const aclTensor *oHead = executorPtr->AllocTensor(
        MakeShape({batch, hv, seqlen, vDim}), params.q->GetDataType(), Format::FORMAT_ND);
    GDN_STAGE_CHECK(oHead != nullptr, 169109);
    auto phase6Result = l0op::ChunkGatedDeltaRuleFwd(
        params.q, params.k, params.v, betaBht, aStorageBhtc, gBht, nullptr,
        params.initialStateOptional, params.cuSeqlensOptional, params.chunkIndicesOptional,
        outputFinalState, params.chunkSize, params.scale, oHead, finalState,
        gCumsumCompute, aCompute, executorPtr);
    GDN_STAGE_CHECK(phase6Result[0] != nullptr && phase6Result[2] != nullptr &&
                        phase6Result[3] != nullptr,
                        169112);
    const aclTensor *oSequence = TransposeContiguous(oHead, {0, 2, 1, 3}, executorPtr);
    GDN_STAGE_CHECK(oSequence != nullptr && l0op::ViewCopy(oSequence, params.oOut, executorPtr) != nullptr,
                    169107);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize(
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
    bool useExp2,
    bool useQkL2norm,
    bool allowNegEigval,
    bool stateVFirst,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *qHatOutOptional,
    const aclTensor *kHatOutOptional,
    const aclTensor *qRstdOutOptional,
    const aclTensor *kRstdOutOptional,
    const aclTensor *betaEffOutOptional,
    const aclTensor *gCumsumOutOptional,
    const aclTensor *aOutOptional,
    const aclTensor *hOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwd,
                   DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional, cuSeqlensOptional,
                          chunkIndicesOptional, layout, scale, chunkSize, useExp2, useQkL2norm,
                          allowNegEigval, stateVFirst),
                   DFX_OUT(oOut, finalStateOutOptional, qHatOutOptional, kHatOutOptional, qRstdOutOptional,
                           kRstdOutOptional, betaEffOutOptional, gCumsumOutOptional, aOutOptional, hOutOptional));
    return ChunkGatedDeltaRuleFwdGetWorkspaceSizeImpl(
        q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional,
        layout, scale, chunkSize, useExp2, useQkL2norm, allowNegEigval, stateVFirst, oOut,
        finalStateOutOptional, qHatOutOptional,
        kHatOutOptional, qRstdOutOptional, kRstdOutOptional, betaEffOutOptional, gCumsumOutOptional, aOutOptional,
        hOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnChunkGatedDeltaRuleFwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleFwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkGatedDeltaRuleFwd launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
