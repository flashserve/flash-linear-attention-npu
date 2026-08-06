#include "aclnn_chunk_kda_bwd_intra.h"
#include "chunk_kda_bwd_intra.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

struct Params {
    const aclTensor *q;
    const aclTensor *k;
    const aclTensor *gk;
    const aclTensor *beta;
    const aclTensor *dAqk;
    const aclTensor *dAkk;
    const aclTensor *dq;
    const aclTensor *dk;
    const aclTensor *db;
    const aclTensor *dg;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    int64_t chunkSize;
    bool safeGate;
    const char *layout;
    const aclTensor *dqOut;
    const aclTensor *dkOut;
    const aclTensor *dbOut;
    const aclTensor *dgOut;
};

enum class Layout {
    BSND,
    BNSD,
    TND,
};

const aclTensor *TransposeSwap12(const aclTensor *input, aclOpExecutor *executor)
{
    const size_t rank = input->GetViewShape().GetDimNum();
    CHECK_RET(rank >= 3, nullptr);
    std::vector<int64_t> perm(rank);
    for (size_t idx = 0; idx < rank; ++idx) {
        perm[idx] = static_cast<int64_t>(idx);
    }
    std::swap(perm[1], perm[2]);
    const aclIntArray *permArray = executor->AllocIntArray(perm.data(), perm.size());
    CHECK_RET(permArray != nullptr, nullptr);
    return l0op::Transpose(input, permArray, executor);
}

const aclTensor *TransposeSwap12Contiguous(const aclTensor *input, aclOpExecutor *executor)
{
    const aclTensor *transposed = TransposeSwap12(input, executor);
    CHECK_RET(transposed != nullptr, nullptr);
    const aclTensor *materialized = l0op::Contiguous(transposed, executor);
    CHECK_RET(materialized != nullptr, nullptr);
    const aclTensor *reshaped =
        l0op::Reshape(materialized, transposed->GetViewShape(), executor);
    CHECK_RET(reshaped != nullptr, nullptr);
    reshaped->SetStorageShape(reshaped->GetViewShape());
    reshaped->SetOriginalShape(reshaped->GetViewShape());
    return reshaped;
}

const aclTensor *TransposeSwap12ToOutput(
    const aclTensor *input, const aclTensor *output, aclOpExecutor *executor)
{
    const aclTensor *transposed = TransposeSwap12(input, executor);
    CHECK_RET(transposed != nullptr, nullptr);
    return l0op::ViewCopy(transposed, output, executor);
}

static bool SameShape(const aclTensor *a, const aclTensor *b)
{
    const auto lhs = a->GetViewShape();
    const auto rhs = b->GetViewShape();
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static aclnnStatus ParseLayout(const char *text, Layout &layout)
{
    CHECK_COND(text != nullptr, ACLNN_ERR_PARAM_INVALID,
               "layout must not be nullptr and must be BSND, BNSD or TND.");
    if (std::strcmp(text, "BSND") == 0) {
        layout = Layout::BSND;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(text, "BNSD") == 0) {
        layout = Layout::BNSD;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(text, "TND") == 0) {
        layout = Layout::TND;
        return ACLNN_SUCCESS;
    }
    CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwdIntra supports dense BSND/BNSD or varlen TND.");
}

aclnnStatus ValidateVarLenMetadata(
    const Params &p, int64_t totalTokens, int64_t &totalChunks)
{
    CHECK_COND(p.cuSeqlens != nullptr, ACLNN_ERR_PARAM_INVALID,
               "varlen input requires cu_seqlens.");
    CHECK_COND(p.cuSeqlens->Size() >= 2 && p.cuSeqlens->Size() <= 65,
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens must contain 2..65 entries.");
    const aclIntArray &cu = *p.cuSeqlens;
    CHECK_COND(cu[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens[0] must be zero.");
    CHECK_COND(cu[cu.Size() - 1] == totalTokens, ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens[-1] must equal total T.");

    totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        CHECK_COND(cu[seq] >= 0 && cu[seq + 1] >= cu[seq],
                   ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be nondecreasing.");
        const int64_t length = cu[seq + 1] - cu[seq];
        totalChunks += (length + p.chunkSize - 1) / p.chunkSize;
    }
    CHECK_COND(totalChunks > 0, ACLNN_ERR_PARAM_INVALID,
               "varlen input must contain at least one non-empty sequence.");

    if (p.chunkIndices != nullptr) {
        CHECK_COND(
            p.chunkIndices->Size() == static_cast<size_t>(totalChunks) * 2,
            ACLNN_ERR_PARAM_INVALID,
            "chunk_indices must contain exactly two values per chunk.");
        size_t index = 0;
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            const int64_t length = cu[seq + 1] - cu[seq];
            const int64_t count =
                (length + p.chunkSize - 1) / p.chunkSize;
            for (int64_t localChunk = 0; localChunk < count; ++localChunk) {
                CHECK_COND(
                    (*p.chunkIndices)[index] == static_cast<int64_t>(seq) &&
                        (*p.chunkIndices)[index + 1] == localChunk,
                    ACLNN_ERR_PARAM_INVALID,
                    "chunk_indices must use canonical sequence-major order.");
                index += 2;
            }
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus Check(const Params &p, Layout &layout, int64_t &totalChunks)
{
    const aclTensor *required[] = {
        p.q, p.k, p.gk, p.beta, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg,
        p.dqOut, p.dkOut, p.dbOut, p.dgOut
    };
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwdIntra tensor arguments must not be nullptr.");
        CHECK_COND(IsContiguous(tensor), ACLNN_ERR_PARAM_INVALID,
                   "ChunkKdaBwdIntra only supports contiguous tensors.");
    }
    const aclTensor *bf16[] = {p.q, p.k};
    for (const aclTensor *tensor : bf16) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID, "q/k must be BF16.");
    }
    CHECK_COND(
        p.beta->GetDataType() == DataType::DT_BF16 ||
            p.beta->GetDataType() == DataType::DT_FLOAT,
        ACLNN_ERR_PARAM_INVALID, "beta must be BF16 or FP32.");
    const aclTensor *fp32[] = {
        p.gk, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg,
        p.dqOut, p.dkOut, p.dbOut, p.dgOut
    };
    for (const aclTensor *tensor : fp32) {
        CHECK_COND(tensor->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID,
                   "gk/dA/gradient inputs and outputs must be FP32.");
    }
    CHECK_RET(ParseLayout(p.layout, layout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(p.safeGate, ACLNN_ERR_PARAM_INVALID,
               "safe_gate=false is reserved but not supported in v1.");
    CHECK_COND(p.chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "chunk_size must be 64.");
    const bool isVarLen = p.cuSeqlens != nullptr;
    CHECK_COND(isVarLen || p.chunkIndices == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "chunk_indices requires cu_seqlens.");
    CHECK_COND((isVarLen && layout != Layout::BNSD) ||
                   (!isVarLen && layout != Layout::TND),
               ACLNN_ERR_PARAM_INVALID,
               "varlen supports TND/BSND; dense supports BSND/BNSD.");
    const auto q = p.q->GetViewShape();
    const size_t expectedRank = layout == Layout::TND ? 3 : 4;
    CHECK_COND(q.GetDimNum() == expectedRank, ACLNN_ERR_PARAM_INVALID,
               "q rank does not match layout.");
    const int64_t b = layout == Layout::TND ? 1 : q.GetDim(0);
    const int64_t t = layout == Layout::TND ? q.GetDim(0) :
                      q.GetDim(layout == Layout::BSND ? 1 : 2);
    const int64_t h = layout == Layout::TND ? q.GetDim(1) :
                      q.GetDim(layout == Layout::BSND ? 2 : 1);
    const int64_t k = layout == Layout::TND ? q.GetDim(2) : q.GetDim(3);
    CHECK_COND(b > 0 && h > 0 && t > 0, ACLNN_ERR_PARAM_INVALID,
               "B/H/T must be positive.");
    CHECK_COND((isVarLen && k == 128) ||
                   (!isVarLen && (k == 64 || k == 128 || k == 256)),
               ACLNN_ERR_PARAM_INVALID,
               "varlen supports K=128; dense supports K=64/128/256.");
    CHECK_COND(!isVarLen || layout != Layout::BSND || b == 1,
               ACLNN_ERR_PARAM_INVALID,
               "varlen BSND compatibility requires B=1.");
    CHECK_COND(SameShape(p.q, p.k) && SameShape(p.q, p.gk) &&
                   SameShape(p.q, p.dq) && SameShape(p.q, p.dk) &&
                   SameShape(p.q, p.dg) && SameShape(p.q, p.dqOut) &&
                   SameShape(p.q, p.dkOut) && SameShape(p.q, p.dgOut),
               ACLNN_ERR_PARAM_INVALID,
               "q/k/gk/dq/dk/dg and vector outputs must have identical shape.");
    const auto beta = p.beta->GetViewShape();
    const size_t scalarRank = layout == Layout::TND ? 2 : 3;
    CHECK_COND(beta.GetDimNum() == scalarRank, ACLNN_ERR_PARAM_INVALID,
               "beta/db/dbOut rank does not match layout.");
    const bool scalarDimsMatch = layout == Layout::TND ?
        (beta.GetDim(0) == t && beta.GetDim(1) == h) :
        (beta.GetDim(0) == b &&
         beta.GetDim(layout == Layout::BSND ? 1 : 2) == t &&
         beta.GetDim(layout == Layout::BSND ? 2 : 1) == h);
    CHECK_COND(scalarDimsMatch && SameShape(p.beta, p.db) &&
                   SameShape(p.beta, p.dbOut),
               ACLNN_ERR_PARAM_INVALID,
               "beta/db/dbOut shape does not match layout.");
    const auto da = p.dAqk->GetViewShape();
    const size_t matrixRank = layout == Layout::TND ? 3 : 4;
    CHECK_COND(da.GetDimNum() == matrixRank, ACLNN_ERR_PARAM_INVALID,
               "dAqk/dAkk rank does not match layout.");
    const bool matrixDimsMatch = layout == Layout::TND ?
        (da.GetDim(0) == t && da.GetDim(1) == h &&
         da.GetDim(2) == p.chunkSize) :
        (da.GetDim(0) == b &&
         da.GetDim(layout == Layout::BSND ? 1 : 2) == t &&
         da.GetDim(layout == Layout::BSND ? 2 : 1) == h &&
         da.GetDim(3) == p.chunkSize);
    CHECK_COND(matrixDimsMatch && SameShape(p.dAqk, p.dAkk),
               ACLNN_ERR_PARAM_INVALID,
               "dAqk/dAkk shape does not match layout.");
    totalChunks = 0;
    if (isVarLen) {
        CHECK_RET(
            ValidateVarLenMetadata(p, t, totalChunks) == ACLNN_SUCCESS,
            ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdIntraGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *gk, const aclTensor *beta,
    const aclTensor *dAqk, const aclTensor *dAkk, const aclTensor *dq, const aclTensor *dk,
    const aclTensor *db, const aclTensor *dg, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, int64_t chunkSize, bool safeGate,
    const char *layout, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dbOut, const aclTensor *dgOut, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdIntra,
                   DFX_IN(q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
                          cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate, layout),
                   DFX_OUT(dqOut, dkOut, dbOut, dgOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "workspaceSize and executor must not be nullptr.");
    Params params{q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
                  cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate, layout,
                  dqOut, dkOut, dbOut, dgOut};
    Layout parsedLayout;
    int64_t totalChunks = 0;
    CHECK_RET(Check(params, parsedLayout, totalChunks) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    const bool isVarLen = cuSeqlensOptional != nullptr;
    const bool isInternalLayout = parsedLayout == Layout::BNSD || isVarLen;

    const aclTensor *qBnsd = q;
    const aclTensor *kBnsd = k;
    const aclTensor *gkBnsd = gk;
    const aclTensor *betaBns = beta;
    const aclTensor *dAqkBnst = dAqk;
    const aclTensor *dAkkBnst = dAkk;
    const aclTensor *dqBnsd = dq;
    const aclTensor *dkBnsd = dk;
    const aclTensor *dbBns = db;
    const aclTensor *dgBnsd = dg;
    const aclTensor *dqOutBnsd = dqOut;
    const aclTensor *dkOutBnsd = dkOut;
    const aclTensor *dbOutBns = dbOut;
    const aclTensor *dgOutBnsd = dgOut;

    if (!isInternalLayout) {
        qBnsd = TransposeSwap12Contiguous(q, executorPtr);
        kBnsd = TransposeSwap12Contiguous(k, executorPtr);
        gkBnsd = TransposeSwap12Contiguous(gk, executorPtr);
        betaBns = TransposeSwap12Contiguous(beta, executorPtr);
        dAqkBnst = TransposeSwap12Contiguous(dAqk, executorPtr);
        dAkkBnst = TransposeSwap12Contiguous(dAkk, executorPtr);
        dqBnsd = TransposeSwap12Contiguous(dq, executorPtr);
        dkBnsd = TransposeSwap12Contiguous(dk, executorPtr);
        dbBns = TransposeSwap12Contiguous(db, executorPtr);
        dgBnsd = TransposeSwap12Contiguous(dg, executorPtr);
        // FinishHead reads each gradient tile before overwriting the same disjoint
        // tile. Reuse the transposed gradient inputs as the internal outputs so
        // dense BSND does not allocate another four full-size BNSD tensors.
        dqOutBnsd = dqBnsd;
        dkOutBnsd = dkBnsd;
        dbOutBns = dbBns;
        dgOutBnsd = dgBnsd;
        CHECK_RET(qBnsd != nullptr && kBnsd != nullptr && gkBnsd != nullptr &&
                      betaBns != nullptr && dAqkBnst != nullptr && dAkkBnst != nullptr &&
                      dqBnsd != nullptr && dkBnsd != nullptr && dbBns != nullptr &&
                      dgBnsd != nullptr && dqOutBnsd != nullptr && dkOutBnsd != nullptr &&
                      dbOutBns != nullptr && dgOutBnsd != nullptr,
                   ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::ChunkKdaBwdIntra(
        qBnsd, kBnsd, gkBnsd, betaBns, dAqkBnst, dAkkBnst,
        dqBnsd, dkBnsd, dbBns, dgBnsd,
        cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate,
        isVarLen ? 1 : 0, totalChunks,
        dqOutBnsd, dkOutBnsd, dbOutBns, dgOutBnsd, executorPtr);
    for (const aclTensor *tensor : result) {
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!isInternalLayout) {
        CHECK_RET(TransposeSwap12ToOutput(result[0], dqOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(TransposeSwap12ToOutput(result[1], dkOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(TransposeSwap12ToOutput(result[2], dbOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(TransposeSwap12ToOutput(result[3], dgOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdIntra(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdIntra);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaBwdIntra launch failed.");
    return ACLNN_SUCCESS;
}
