/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details.
 */

#include "aclnn_chunk_kda_fwd_intra_sub_chunk.h"
#include "chunk_kda_fwd_intra_sub_chunk.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
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
constexpr int64_t SUB_CHUNK_SIZE = 16;
constexpr int64_t MAX_K_DIM = 256;
constexpr int64_t MAX_VARLEN_SEQUENCES = 1024;

struct IntraSubChunkParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    double scale = 1.0;
    int64_t chunkSize = 64;
    const aclTensor *aqkOut = nullptr;
    const aclTensor *akkdOut = nullptr;
};

aclnnStatus ContiguousInPlace(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

int64_t Dim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t idx = 0; idx < Rank(lhs); ++idx) {
        if (Dim(lhs, idx) != Dim(rhs, idx)) {
            return false;
        }
    }
    return true;
}

aclnnStatus CheckCuSeqlens(const aclIntArray *cuSeqlensOptional, int64_t seqlen)
{
    if (cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const aclIntArray &cu = *cuSeqlensOptional;
    CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND(cu[0] == 0, ACLNN_ERR_PARAM_INVALID, "cuSeqlensOptional[0] must be 0.");
    CHECK_COND(cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional last element must equal the sequence length.");
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        CHECK_COND(cu[idx] <= cu[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be nondecreasing.");
    }
    return ACLNN_SUCCESS;
}

int64_t ExpectedChunks(const aclIntArray *cuSeqlensOptional, int64_t seqlen, int64_t chunkSize)
{
    if (cuSeqlensOptional == nullptr) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    int64_t total = 0;
    const aclIntArray &cu = *cuSeqlensOptional;
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        int64_t length = cu[idx + 1] - cu[idx];
        total += (length + chunkSize - 1) / chunkSize;
    }
    return total;
}

aclnnStatus CheckChunkIndices(const aclIntArray *chunkIndicesOptional, const aclIntArray *cuSeqlensOptional,
                              int64_t expectedChunks, int64_t chunkSize)
{
    if (chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(cuSeqlensOptional != nullptr, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional is only valid when cuSeqlensOptional is provided.");
    CHECK_COND(chunkIndicesOptional->Size() % 2 == 0, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain (seq_id, chunk_id) pairs.");
    CHECK_COND(static_cast<int64_t>(chunkIndicesOptional->Size() / 2) == expectedChunks, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain exactly expectedChunks (seq_id, chunk_id) pairs.");
    const aclIntArray &indices = *chunkIndicesOptional;
    const aclIntArray &cu = *cuSeqlensOptional;
    int64_t seqNum = static_cast<int64_t>(cu.Size()) - 1;
    for (size_t idx = 0; idx < indices.Size(); idx += 2) {
        int64_t seq = indices[idx];
        int64_t localChunk = indices[idx + 1];
        CHECK_COND(seq >= 0 && seq < seqNum, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional seq_id must be in [0, seq_num).");
        int64_t seqLength = cu[seq + 1] - cu[seq];
        int64_t seqChunks = (seqLength + chunkSize - 1) / chunkSize;
        CHECK_COND(localChunk >= 0 && localChunk < seqChunks, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional chunk_id is outside the selected sequence.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const IntraSubChunkParams &params)
{
    CHECK_COND(params.q != nullptr && params.k != nullptr && params.g != nullptr && params.beta != nullptr,
               ACLNN_ERR_PARAM_INVALID, "q/k/g/beta are required.");
    CHECK_COND(params.aqkOut != nullptr && params.akkdOut != nullptr, ACLNN_ERR_PARAM_INVALID,
               "aqkOut/akkdOut are required.");
    CHECK_COND(params.chunkSize == 32 || params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 32, 64 or 128.");
    CHECK_COND(Rank(params.q) == 4 && Rank(params.k) == 4 && Rank(params.g) == 4 && Rank(params.beta) == 3,
               ACLNN_ERR_PARAM_INVALID,
               "BNSD layout required: q/k [B,H,T,K], g [B,HV,T,K], beta [B,HV,T].");
    CHECK_COND(SameShape(params.q, params.k), ACLNN_ERR_PARAM_INVALID, "q/k must share shape [B,H,T,K].");

    const int64_t B = Dim(params.q, 0);
    const int64_t H = Dim(params.q, 1);
    const int64_t T = Dim(params.q, 2);
    const int64_t K = Dim(params.q, 3);
    const int64_t HV = Dim(params.g, 1);
    CHECK_COND(Dim(params.g, 0) == B && Dim(params.g, 2) == T && Dim(params.g, 3) == K, ACLNN_ERR_PARAM_INVALID,
               "g must be [B,HV,T,K] matching q on B/T/K.");
    CHECK_COND(Dim(params.beta, 0) == B && Dim(params.beta, 1) == HV && Dim(params.beta, 2) == T,
               ACLNN_ERR_PARAM_INVALID, "beta must be [B,HV,T] matching g.");
    CHECK_COND(H > 0 && HV > 0 && HV >= H && (HV % H) == 0 && H <= 128 && HV <= 128, ACLNN_ERR_PARAM_INVALID,
               "H/HV must be positive, HV>=H, HV%H==0, and both <= 128.");

    CHECK_COND(K > 0 && K <= MAX_K_DIM && (K % 16) == 0, ACLNN_ERR_PARAM_INVALID,
               "K must be a positive multiple of 16 and <= 256.");

    CHECK_COND(Rank(params.aqkOut) == 4 && Dim(params.aqkOut, 0) == B && Dim(params.aqkOut, 1) == HV &&
                   Dim(params.aqkOut, 2) == T && Dim(params.aqkOut, 3) == params.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "aqkOut must be [B,HV,T,chunkSize].");
    CHECK_COND(Rank(params.akkdOut) == 4 && Dim(params.akkdOut, 0) == B && Dim(params.akkdOut, 1) == HV &&
                   Dim(params.akkdOut, 2) == T && Dim(params.akkdOut, 3) == SUB_CHUNK_SIZE,
               ACLNN_ERR_PARAM_INVALID, "akkdOut must be [B,HV,T,16] float32.");
    CHECK_COND(params.akkdOut->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "akkdOut dtype must be float32.");
    CHECK_COND(params.aqkOut->GetDataType() == params.q->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "aqkOut dtype must match q.");

    const bool hasCu = params.cuSeqlensOptional != nullptr;
    const bool hasIdx = params.chunkIndicesOptional != nullptr;
    CHECK_COND(hasCu == hasIdx, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must both be provided or both be omitted.");
    if (hasCu) {
        CHECK_COND(B == 1, ACLNN_ERR_PARAM_INVALID, "varlen mode currently requires B=1.");
        CHECK_RET(CheckCuSeqlens(params.cuSeqlensOptional, T) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        const int64_t seqNum = static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
        CHECK_COND(seqNum <= MAX_VARLEN_SEQUENCES, ACLNN_ERR_PARAM_INVALID,
                   "varlen input supports at most 1024 sequences in one call.");
        const int64_t expected = ExpectedChunks(params.cuSeqlensOptional, T, params.chunkSize);
        CHECK_RET(CheckChunkIndices(params.chunkIndicesOptional, params.cuSeqlensOptional, expected,
                                    params.chunkSize) == ACLNN_SUCCESS,
                  ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkKdaFwdIntraSubChunkGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *aqkOut,
    const aclTensor *akkdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaFwdIntraSubChunk,
                   DFX_IN(q, k, g, beta, cuSeqlensOptional, chunkIndicesOptional),
                   DFX_OUT(aqkOut, akkdOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    IntraSubChunkParams params{q, k, g, beta, cuSeqlensOptional, chunkIndicesOptional, scale, chunkSize, aqkOut,
                               akkdOut};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.aqkOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInPlace(params.akkdOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::ChunkKdaFwdIntraSubChunk(
        params.q, params.k, params.g, params.beta, params.cuSeqlensOptional, params.chunkIndicesOptional,
        static_cast<float>(params.scale), params.chunkSize, params.aqkOut, params.akkdOut, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdIntraSubChunk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwdIntraSubChunk);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
