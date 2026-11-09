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

#include "../../../../common/host_utils/log.h"

using namespace op;

namespace {

constexpr const char *CHUNK_KDA_BWD_INTRA_ENTITY = "aclnnChunkKdaBwdIntra";

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
    if (text == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
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
    OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "layout", text, "BSND, BNSD or TND");
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus ValidateVarLenMetadata(
    const Params &p, int64_t totalTokens, int64_t &totalChunks)
{
    if (p.cuSeqlens == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "cu_seqlens");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(p.cuSeqlens->Size() >= 2 && p.cuSeqlens->Size() <= 65)) {
        OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "cu_seqlens",
                                                      std::to_string(p.cuSeqlens->Size()), "2..65");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclIntArray &cu = *p.cuSeqlens;
    if (cu[0] != 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "cu_seqlens[0]",
                                                   std::to_string(cu[0]), "0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cu[cu.Size() - 1] != totalTokens) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "cu_seqlens[last]",
                                                   std::to_string(cu[cu.Size() - 1]), std::to_string(totalTokens));
        return ACLNN_ERR_PARAM_INVALID;
    }

    totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        if (!(cu[seq] >= 0 && cu[seq + 1] >= cu[seq])) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "cu_seqlens",
                                                       "non-monotonic", "nondecreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t length = cu[seq + 1] - cu[seq];
        totalChunks += (length + p.chunkSize - 1) / p.chunkSize;
    }
    if (totalChunks <= 0) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "cu_seqlens", "empty",
                                                   "at least one non-empty sequence");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (p.chunkIndices != nullptr) {
        if (p.chunkIndices->Size() != static_cast<size_t>(totalChunks) * 2) {
            OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "chunk_indices",
                                                          std::to_string(p.chunkIndices->Size()),
                                                          std::to_string(totalChunks * 2));
            return ACLNN_ERR_PARAM_INVALID;
        }
        size_t index = 0;
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            const int64_t length = cu[seq + 1] - cu[seq];
            const int64_t count =
                (length + p.chunkSize - 1) / p.chunkSize;
            for (int64_t localChunk = 0; localChunk < count; ++localChunk) {
                if (!((*p.chunkIndices)[index] == static_cast<int64_t>(seq) &&
                      (*p.chunkIndices)[index + 1] == localChunk)) {
                    OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "chunk_indices",
                                                               "non-canonical", "sequence-major order");
                    return ACLNN_ERR_PARAM_INVALID;
                }
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
        if (tensor == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "tensor");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (!IsContiguous(tensor)) {
            OP_LOGE_LIBOPAPI_REPORT_FORMAT_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "tensor", "non-contiguous", "contiguous");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const aclTensor *bf16[] = {p.q, p.k};
    for (const aclTensor *tensor : bf16) {
        if (tensor->GetDataType() != DataType::DT_BF16) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "q/k",
                                                       fla_npu::host_utils::DfxDtypeString(tensor->GetDataType()), "BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!(p.beta->GetDataType() == DataType::DT_BF16 || p.beta->GetDataType() == DataType::DT_FLOAT)) {
        OP_LOGE_LIBOPAPI_REPORT_DTYPE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "beta",
                                                   fla_npu::host_utils::DfxDtypeString(p.beta->GetDataType()), "BF16,FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *fp32[] = {
        p.gk, p.dAqk, p.dAkk, p.dq, p.dk, p.db, p.dg,
        p.dqOut, p.dkOut, p.dbOut, p.dgOut
    };
    for (const aclTensor *tensor : fp32) {
        if (tensor->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_LIBOPAPI_REPORT_DTYPE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "gk/dA/gradient",
                                                       fla_npu::host_utils::DfxDtypeString(tensor->GetDataType()), "FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    auto parseRet = ParseLayout(p.layout, layout);
    if (parseRet != ACLNN_SUCCESS) return parseRet;
    if (!p.safeGate) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "safeGate", "false",
                                                   "reserved but not supported in v1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (p.chunkSize != 64) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "chunkSize",
                                                   std::to_string(p.chunkSize), "64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool isVarLen = p.cuSeqlens != nullptr;
    if (!isVarLen && p.chunkIndices != nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "chunk_indices", "present",
                                                   "requires cu_seqlens");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((isVarLen && layout == Layout::BNSD) || (!isVarLen && layout == Layout::TND)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "layout", "mismatch",
                                                   "varlen supports TND/BSND; dense supports BSND/BNSD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto q = p.q->GetViewShape();
    const size_t expectedRank = layout == Layout::TND ? 3 : 4;
    if (q.GetDimNum() != expectedRank) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "q",
                                                             "rank=" + std::to_string(q.GetDimNum()),
                                                             "rank=" + std::to_string(expectedRank));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t b = layout == Layout::TND ? 1 : q.GetDim(0);
    const int64_t t = layout == Layout::TND ? q.GetDim(0) :
                      q.GetDim(layout == Layout::BSND ? 1 : 2);
    const int64_t h = layout == Layout::TND ? q.GetDim(1) :
                      q.GetDim(layout == Layout::BSND ? 2 : 1);
    const int64_t k = layout == Layout::TND ? q.GetDim(2) : q.GetDim(3);
    if (!(b > 0 && h > 0 && t > 0)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "q.shape[B/H/T]",
                                                   "non-positive", "positive");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((isVarLen && k == 128) || (!isVarLen && (k == 64 || k == 128 || k == 256)))) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "q.shape[K]", std::to_string(k),
                                                   "varlen 128; dense 64/128/256");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isVarLen && layout == Layout::BSND && b != 1) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "q.shape[B]", std::to_string(b), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(SameShape(p.q, p.k) && SameShape(p.q, p.gk) && SameShape(p.q, p.dq) && SameShape(p.q, p.dk) &&
          SameShape(p.q, p.dg) && SameShape(p.q, p.dqOut) && SameShape(p.q, p.dkOut) && SameShape(p.q, p.dgOut))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "q,k,gk,dq,dk,dg,dqOut,dkOut,dgOut",
                                                   "mismatch", "identical");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto beta = p.beta->GetViewShape();
    const size_t scalarRank = layout == Layout::TND ? 2 : 3;
    if (beta.GetDimNum() != scalarRank) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "beta",
                                                             "rank=" + std::to_string(beta.GetDimNum()),
                                                             "rank=" + std::to_string(scalarRank));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool scalarDimsMatch = layout == Layout::TND ?
        (beta.GetDim(0) == t && beta.GetDim(1) == h) :
        (beta.GetDim(0) == b &&
         beta.GetDim(layout == Layout::BSND ? 1 : 2) == t &&
         beta.GetDim(layout == Layout::BSND ? 2 : 1) == h);
    if (!(scalarDimsMatch && SameShape(p.beta, p.db) && SameShape(p.beta, p.dbOut))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "beta,db,dbOut", "mismatch", "match layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto da = p.dAqk->GetViewShape();
    const size_t matrixRank = layout == Layout::TND ? 3 : 4;
    if (da.GetDimNum() != matrixRank) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "dAqk/dAkk",
                                                             "rank=" + std::to_string(da.GetDimNum()),
                                                             "rank=" + std::to_string(matrixRank));
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool matrixDimsMatch = layout == Layout::TND ?
        (da.GetDim(0) == t && da.GetDim(1) == h &&
         da.GetDim(2) == p.chunkSize) :
        (da.GetDim(0) == b &&
         da.GetDim(layout == Layout::BSND ? 1 : 2) == t &&
         da.GetDim(layout == Layout::BSND ? 2 : 1) == h &&
         da.GetDim(3) == p.chunkSize);
    if (!(matrixDimsMatch && SameShape(p.dAqk, p.dAkk))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "dAqk,dAkk", "mismatch", "match layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    totalChunks = 0;
    if (isVarLen) {
        auto varlenRet = ValidateVarLenMetadata(p, t, totalChunks);
        if (varlenRet != ACLNN_SUCCESS) return varlenRet;
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
    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_IMPAT(CHUNK_KDA_BWD_INTRA_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    Params params{q, k, gk, beta, dAqk, dAkk, dq, dk, db, dg,
                  cuSeqlensOptional, chunkIndicesOptional, chunkSize, safeGate, layout,
                  dqOut, dkOut, dbOut, dgOut};
    Layout parsedLayout;
    int64_t totalChunks = 0;
    auto checkRet = Check(params, parsedLayout, totalChunks);
    if (checkRet != ACLNN_SUCCESS) {
        return checkRet;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    OP_LOGW("npu_chunk_kda_bwd_intra",
            "npu_chunk_kda_bwd_intra is deprecated and will be removed on "
            "January 15, 2027 (v26.12.0); use the fused npu_chunk_kda_bwd instead.");

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
