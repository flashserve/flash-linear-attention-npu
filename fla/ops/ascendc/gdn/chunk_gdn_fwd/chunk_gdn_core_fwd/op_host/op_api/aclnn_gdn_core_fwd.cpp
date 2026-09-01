/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_gdn_core_fwd.h"

#include "chunk_gdn_core_fwd.h"
#include "../../op_kernel/chunk_gdn_core_output_mask.h"

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
#include <initializer_list>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t GDN_CORE_DIM = 128;
constexpr int64_t GDN_CORE_V256 = 256;
constexpr int64_t GDN_CORE_CHUNK_64 = 64;
constexpr int64_t GDN_CORE_CHUNK_128 = 128;

struct GdnCoreFwdParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    bool outputFinalState = false;
    int64_t chunkSize = GDN_CORE_CHUNK_64;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    const aclTensor *oOut = nullptr;
    const aclTensor *finalStateOutOptional = nullptr;
    const aclTensor *gCumsumOut = nullptr;
    const aclTensor *aOut = nullptr;
};

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

static int64_t SeqNum(const GdnCoreFwdParams &params)
{
    return params.cuSeqlensOptional == nullptr
               ? Dim(params.q, 0)
               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
}

static int64_t ExpectedChunks(const GdnCoreFwdParams &params)
{
    if (params.cuSeqlensOptional == nullptr) {
        return (Dim(params.q, 2) + params.chunkSize - 1) / params.chunkSize;
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

static aclnnStatus CheckMetadata(const GdnCoreFwdParams &params)
{
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const int64_t seqlen = Dim(params.q, 2);
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
    const int64_t expectedChunks = ExpectedChunks(params);
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

static aclnnStatus CheckParams(const GdnCoreFwdParams &params)
{
    CHECK_RET(CheckRank(params.q, 4, "q") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRank(params.k, 4, "k") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRank(params.v, 4, "v") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRank(params.g, 3, "g") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRank(params.beta, 3, "beta") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRank(params.oOut, 4, "oOut") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (params.gCumsumOut != nullptr) {
        CHECK_RET(CheckRank(params.gCumsumOut, 3, "gCumsumOut") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    if (params.aOut != nullptr) {
        CHECK_RET(CheckRank(params.aOut, 4, "aOut") == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    const int64_t batch = Dim(params.q, 0);
    const int64_t hq = Dim(params.q, 1);
    const int64_t seqlen = Dim(params.q, 2);
    const int64_t kDim = Dim(params.q, 3);
    const int64_t hk = Dim(params.k, 1);
    const int64_t hv = Dim(params.v, 1);
    const int64_t vDim = Dim(params.v, 3);
    CHECK_COND(batch > 0 && hq > 0 && hk > 0 && hv > 0 && seqlen > 0,
               ACLNN_ERR_PARAM_INVALID, "B/H/T dimensions must be positive.");
    CHECK_COND(hq == hk, ACLNN_ERR_PARAM_INVALID, "q and k must have the same head count.");
    CHECK_COND(hv % hq == 0, ACLNN_ERR_PARAM_INVALID,
               "Phase 6 GVA requires Hv divisible by Hk.");
    CHECK_COND(kDim == GDN_CORE_DIM && (vDim == GDN_CORE_DIM || vDim == GDN_CORE_V256),
               ACLNN_ERR_PARAM_INVALID,
               "The Phase 6 composite GDN core supports K=128 and V=128/256.");
    CHECK_COND(Dim(params.k, 0) == batch && Dim(params.k, 2) == seqlen && Dim(params.k, 3) == kDim,
               ACLNN_ERR_PARAM_INVALID, "k shape must match q in B/T/K.");
    CHECK_COND(Dim(params.v, 0) == batch && Dim(params.v, 2) == seqlen,
               ACLNN_ERR_PARAM_INVALID, "v shape must match q in B/T.");
    CHECK_COND(Dim(params.beta, 0) == batch && Dim(params.beta, 1) == seqlen && Dim(params.beta, 2) == hv,
               ACLNN_ERR_PARAM_INVALID, "beta must have shape [B,T,Hv].");
    CHECK_COND(Dim(params.g, 0) == batch && Dim(params.g, 1) == seqlen && Dim(params.g, 2) == hv,
               ACLNN_ERR_PARAM_INVALID, "g must have shape [B,T,Hv].");
    CHECK_COND(Dim(params.oOut, 0) == batch && Dim(params.oOut, 1) == hv &&
                   Dim(params.oOut, 2) == seqlen && Dim(params.oOut, 3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "oOut must have shape [B,Hv,T,V].");
    if (params.gCumsumOut != nullptr) {
        CHECK_COND(Dim(params.gCumsumOut, 0) == batch && Dim(params.gCumsumOut, 1) == seqlen &&
                       Dim(params.gCumsumOut, 2) == hv,
                   ACLNN_ERR_PARAM_INVALID, "gCumsumOut must have shape [B,T,Hv].");
    }
    if (params.aOut != nullptr) {
        CHECK_COND(Dim(params.aOut, 0) == batch && Dim(params.aOut, 1) == hv &&
                       Dim(params.aOut, 2) == seqlen && Dim(params.aOut, 3) == params.chunkSize,
                   ACLNN_ERR_PARAM_INVALID, "aOut must have shape [B,Hv,T,chunkSize].");
    }
    CHECK_COND(params.chunkSize == GDN_CORE_CHUNK_64 || params.chunkSize == GDN_CORE_CHUNK_128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128.");
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "cuSeqlens and chunkIndices must be both present or both absent.");
    CHECK_COND(params.cuSeqlensOptional == nullptr || batch == 1, ACLNN_ERR_PARAM_INVALID,
               "varlen rank-4 input requires physical B=1.");
    CHECK_RET(CheckMetadata(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const int64_t seqNum = SeqNum(params);
    CHECK_RET(CheckStateShape(params.initialStateOptional, "initialState", seqNum, hv, kDim, vDim) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (params.outputFinalState) {
        CHECK_RET(CheckStateShape(params.finalStateOutOptional, "finalStateOut", seqNum, hv, kDim, vDim) ==
                      ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }

    const DataType dtype = params.q->GetDataType();
    CHECK_COND(dtype == DataType::DT_FLOAT16 || dtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q/k/v must be float16 or bfloat16.");
    CHECK_COND(params.k->GetDataType() == dtype && params.v->GetDataType() == dtype &&
                   params.oOut->GetDataType() == dtype &&
                   (params.aOut == nullptr || params.aOut->GetDataType() == dtype),
               ACLNN_ERR_PARAM_INVALID, "q/k/v/oOut and aOut, when present, must have the same dtype.");
    CHECK_COND((params.beta->GetDataType() == DataType::DT_FLOAT || params.beta->GetDataType() == dtype) &&
                   params.g->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "beta must be float32 or match q/k/v, and g must be float32.");
    CHECK_COND(params.gCumsumOut == nullptr ||
                   params.gCumsumOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "gCumsumOut must be float32 when present.");
    CHECK_COND(!params.outputFinalState || params.finalStateOutOptional != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "finalStateOut is required when outputFinalState is true.");
    if (params.initialStateOptional != nullptr) {
        const DataType stateDtype = params.initialStateOptional->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_FLOAT || stateDtype == dtype, ACLNN_ERR_PARAM_INVALID,
                   "initialState dtype must be float32 or match q/k/v.");
    }
    if (params.outputFinalState) {
        const DataType expectedStateDtype = params.initialStateOptional == nullptr
                                                ? DataType::DT_FLOAT
                                                : params.initialStateOptional->GetDataType();
        CHECK_COND(params.finalStateOutOptional->GetDataType() == expectedStateDtype,
                   ACLNN_ERR_PARAM_INVALID,
                   "finalStateOut dtype must match initialState, or be float32 when initialState is absent.");
    }
    return ACLNN_SUCCESS;
}

}  // namespace

static aclnnStatus GdnCoreFwdGetWorkspaceSizeImpl(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *gCumsumOut,
    const aclTensor *aOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    GdnCoreFwdParams params{q, k, v, g, beta, initialStateOptional, outputFinalState, chunkSize,
                            cuSeqlensOptional, chunkIndicesOptional, scale, oOut, finalStateOutOptional,
                            gCumsumOut, aOut};
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

    const int64_t batch = Dim(params.q, 0);
    const int64_t hv = Dim(params.v, 1);
    const int64_t seqlen = Dim(params.v, 2);
    auto aStorageBhtc = executorPtr->AllocTensor(MakeShape({batch, hv, seqlen, params.chunkSize}),
                                                  params.k->GetDataType(), Format::FORMAT_ND);
    auto finalState = params.finalStateOutOptional;
    if (!params.outputFinalState) {
        finalState = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
    }
    auto gCumsumOutput = params.gCumsumOut;
    if (gCumsumOutput == nullptr) {
        gCumsumOutput = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT,
                                                 Format::FORMAT_ND);
    }
    auto aOutput = params.aOut;
    if (aOutput == nullptr) {
        aOutput = executorPtr->AllocTensor(MakeShape({1}), params.k->GetDataType(),
                                           Format::FORMAT_ND);
    }
    GDN_STAGE_CHECK(aStorageBhtc != nullptr && finalState != nullptr &&
                        gCumsumOutput != nullptr && aOutput != nullptr,
                    169101);

    const aclTensor *gBht = TransposeContiguous(params.g, {0, 2, 1}, executorPtr);
    const aclTensor *betaFloat = params.beta->GetDataType() == DataType::DT_FLOAT
                                     ? params.beta
                                     : l0op::Cast(params.beta, DataType::DT_FLOAT, executorPtr);
    const aclTensor *betaBht = betaFloat == nullptr
                                   ? nullptr
                                   : TransposeContiguous(betaFloat, {0, 2, 1}, executorPtr);
    GDN_STAGE_CHECK(gBht != nullptr && betaBht != nullptr, 169102);
    const int64_t outputMask =
        (params.gCumsumOut != nullptr
             ? static_cast<int64_t>(GDN::GDN_CORE_OUTPUT_G_CUMSUM)
             : 0) |
        (params.aOut != nullptr ? static_cast<int64_t>(GDN::GDN_CORE_OUTPUT_A) : 0);

    auto phase6Result = l0op::ChunkGdnCoreFwd(
        params.q, params.k, params.v, betaBht, aStorageBhtc, gBht, nullptr,
        params.initialStateOptional, params.cuSeqlensOptional, params.chunkIndicesOptional,
        params.outputFinalState, params.chunkSize, params.scale, outputMask, params.oOut, finalState,
        gCumsumOutput, aOutput, executorPtr);
    GDN_STAGE_CHECK(phase6Result[0] != nullptr && phase6Result[2] != nullptr &&
                        phase6Result[3] != nullptr,
                    169112);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

#define GDN_CORE_GET_WORKSPACE_ARGS \
    q, k, v, g, beta, initialStateOptional, outputFinalState, chunkSize, cuSeqlensOptional, \
        chunkIndicesOptional, scale, oOut, finalStateOutOptional, gCumsumOut, aOut, workspaceSize, executor

aclnnStatus aclnnGdnCoreFwdPhase6GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *initialStateOptional, bool outputFinalState, int64_t chunkSize,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, double scale,
    const aclTensor *oOut, const aclTensor *finalStateOutOptional, const aclTensor *gCumsumOut,
    const aclTensor *aOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnGdnCoreFwdPhase6,
                   DFX_IN(q, k, v, g, beta, initialStateOptional, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(oOut, finalStateOutOptional, gCumsumOut, aOut));
    return GdnCoreFwdGetWorkspaceSizeImpl(
        GDN_CORE_GET_WORKSPACE_ARGS);
}

#undef GDN_CORE_GET_WORKSPACE_ARGS

aclnnStatus aclnnGdnCoreFwdPhase6(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGdnCoreFwdPhase6);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "GdnCoreFwdPhase6 launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
