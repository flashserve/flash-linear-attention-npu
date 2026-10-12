/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_gated_delta_rule_fwd.h"

#include "chunk_gated_delta_rule_fwd.h"
#include "chunk_gated_delta_rule_fwd_error.h"
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
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include <cmath>
#include <limits>
#include <utility>
#include <cstring>
#include <initializer_list>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t HEAD_DIM_128 = 128;
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

static bool IsAscend950()
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    using Ops::Transformer::AclnnUtil::IsRegbase;
    return IsRegbase(npuArch);
}

static bool UsePreparePath(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (IsAscend950()) {
        // On Ascend950 the supported new path is selected by its shape contract.
        // use_exp2 and layout are independent features of that path.
        if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.layout == nullptr ||
            params.q->GetViewShape().GetDimNum() != 4 || params.k->GetViewShape().GetDimNum() != 4 ||
            params.v->GetViewShape().GetDimNum() != 4) {
            return false;
        }
        const bool sequenceMajor = std::strcmp(params.layout, "BSND") == 0 ||
                                   std::strcmp(params.layout, "TND") == 0;
        const size_t headDim = sequenceMajor ? 2 : 1;
        const int64_t hq = params.q->GetViewShape().GetDim(headDim);
        const int64_t hv = params.v->GetViewShape().GetDim(headDim);
        return params.q->GetDataType() == DataType::DT_BF16 &&
               params.k->GetDataType() == DataType::DT_BF16 && params.v->GetDataType() == DataType::DT_BF16 &&
               params.q->GetViewShape().GetDim(3) == HEAD_DIM_128 &&
               params.v->GetViewShape().GetDim(3) == HEAD_DIM_128 &&
               params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 && hq > 0 && hv / hq <= 4;
    }
    const bool legacyLayout = std::strcmp(params.layout, "BNSD") == 0 ||
                              std::strcmp(params.layout, "NTD") == 0;
    // Keep the non-Ascend950 legacy selection unchanged.
    return params.useExp2 || params.useQkL2norm ||
           params.aLogOptional != nullptr || params.dtBiasOptional != nullptr ||
           params.betaEffOutOptional != nullptr || params.allowNegEigval ||
           params.hOutOptional != nullptr || params.stateVFirst || !legacyLayout;
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
        return seqlen / params.chunkSize + (seqlen % params.chunkSize != 0);
    }

    int64_t total = 0;
    const aclIntArray &cu = *params.cuSeqlensOptional;
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        const int64_t length = cu[idx + 1] - cu[idx];
        total += length / params.chunkSize + (length % params.chunkSize != 0);
    }
    return total;
}

static aclnnStatus CheckShape(const aclTensor *tensor, const char *name, const op::Shape &expected)
{
    if (tensor != nullptr && !ShapeEqual(tensor->GetViewShape(), expected)) {
        gdn_error::Shape(name, op::ToString(tensor->GetViewShape()).GetString(), op::ToString(expected).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckStateShape(const aclTensor *state, const char *name, int64_t seqNum, int64_t hv,
                                   int64_t stateDim0, int64_t stateDim1)
{
    return CheckShape(state, name, MakeShape({seqNum, hv, stateDim0, stateDim1}));
}

static aclnnStatus CheckMetadata(const ChunkGatedDeltaRuleFwdParams &params, int64_t seqlen)
{
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const aclIntArray &cu = *params.cuSeqlensOptional;
    if (cu.Size() < 2) {
        gdn_error::ListSize("cuSeqlensOptional", cu.Size(), ">= 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cu[0] != 0 || cu[cu.Size() - 1] != seqlen) {
        gdn_error::Argument("cuSeqlensOptional", "must start at 0 and end at T=" + std::to_string(seqlen) +
                            "; got first=" + std::to_string(cu[0]) + ", last=" + std::to_string(cu[cu.Size() - 1]));
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t expectedChunks = 0;
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        if (cu[idx] < 0 || cu[idx + 1] > seqlen || cu[idx] > cu[idx + 1]) {
            gdn_error::Argument("cuSeqlensOptional", "must be nondecreasing within [0,T]; pair " +
                                std::to_string(idx) + " is [" + std::to_string(cu[idx]) + "," +
                                std::to_string(cu[idx + 1]) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t length = cu[idx + 1] - cu[idx];
        const int64_t chunks = length / params.chunkSize + (length % params.chunkSize != 0);
        if (expectedChunks > std::numeric_limits<int64_t>::max() - chunks) {
            gdn_error::Argument("cuSeqlensOptional", "total chunk count exceeds int64 range");
            return ACLNN_ERR_PARAM_INVALID;
        }
        expectedChunks += chunks;
    }
    const aclIntArray &indices = *params.chunkIndicesOptional;
    if (indices.Size() % 2 != 0 || indices.Size() / 2 != static_cast<uint64_t>(expectedChunks)) {
        gdn_error::ListSize("chunkIndicesOptional", indices.Size(),
                            "2 * chunk count (" + std::to_string(expectedChunks) + ")");
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t index = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        const int64_t length = cu[seq + 1] - cu[seq];
        const int64_t seqChunks = length / params.chunkSize + (length % params.chunkSize != 0);
        for (int64_t localChunk = 0; localChunk < seqChunks; ++localChunk) {
            if (indices[index] != static_cast<int64_t>(seq) || indices[index + 1] != localChunk) {
                gdn_error::Argument("chunkIndicesOptional", "canonical pair " + std::to_string(index / 2) +
                                    " must be [" + std::to_string(seq) + "," + std::to_string(localChunk) +
                                    "]; got [" + std::to_string(indices[index]) + "," +
                                    std::to_string(indices[index + 1]) + "]");
                return ACLNN_ERR_PARAM_INVALID;
            }
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

static aclnnStatus CheckRequiredInputs(const ChunkGatedDeltaRuleFwdParams &params)
{
    const aclTensor *tensors[] = {params.q, params.k, params.v, params.g, params.beta, params.oOut};
    const char *names[] = {"q", "k", "v", "g", "beta", "oOut"};
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        if (tensors[index] == nullptr) {
            gdn_error::Required(names[index]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckZeroShape(const ChunkGatedDeltaRuleFwdParams &params, uint64_t *workspaceSize)
{
    if (params.q->IsEmpty() || params.k->IsEmpty() || params.v->IsEmpty() || params.g->IsEmpty() ||
        params.beta->IsEmpty()) {
        *workspaceSize = 0UL;
        return ACLNN_ERR_PARAM_INVALID;
    }
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
    if (tensor == nullptr) {
        gdn_error::Required(name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (Rank(tensor) != rank) {
        gdn_error::Rank(name, Rank(tensor), rank);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const aclTensor *tensor, const char *name,
                               std::initializer_list<DataType> allowed, bool output = false)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    std::string expected;
    for (const auto dtype : allowed) {
        if (tensor->GetDataType() == dtype) {
            return ACLNN_SUCCESS;
        }
        if (!expected.empty()) {
            expected += " or ";
        }
        expected += op::ToString(dtype).GetString();
    }
    gdn_error::Dtype(name, op::ToString(tensor->GetDataType()).GetString(), expected, output);
    return ACLNN_ERR_PARAM_INVALID;
}

static aclnnStatus ResolveShapeInfo(const ChunkGatedDeltaRuleFwdParams &params, GdnShapeInfo &info)
{
    if (params.layout == nullptr) {
        gdn_error::Required("layout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (std::strcmp(params.layout, "BNSD") == 0) {
        info.layout = GdnLayout::BNSD;
    } else if (std::strcmp(params.layout, "BSND") == 0) {
        info.layout = GdnLayout::BSND;
    } else if (std::strcmp(params.layout, "NTD") == 0) {
        info.layout = GdnLayout::NTD;
    } else if (std::strcmp(params.layout, "TND") == 0) {
        info.layout = GdnLayout::TND;
    } else {
        gdn_error::Attr("layout", params.layout, "BNSD, BSND, NTD or TND (uppercase)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto &entry : {std::make_pair(params.q, "q"), std::make_pair(params.k, "k"),
                              std::make_pair(params.v, "v"), std::make_pair(params.oOut, "oOut")}) {
        CHECK_RET(CheckRank(entry.first, 4, entry.second) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckRank(params.g, 3, "g") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckRank(params.beta, 3, "beta") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.k, "k", params.q->GetViewShape()) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    info.isSequenceMajor = info.layout == GdnLayout::BSND || info.layout == GdnLayout::TND;
    info.batch = Dim(params.q, 0);
    info.hq = Dim(params.q, info.isSequenceMajor ? 2 : 1);
    info.seqlen = Dim(params.q, info.isSequenceMajor ? 1 : 2);
    info.kDim = Dim(params.q, 3);
    info.hv = Dim(params.v, info.isSequenceMajor ? 2 : 1);
    info.vDim = Dim(params.v, 3);
    const auto vShape = info.isSequenceMajor
                            ? MakeShape({info.batch, info.seqlen, info.hv, info.vDim})
                            : MakeShape({info.batch, info.hv, info.seqlen, info.vDim});
    CHECK_RET(CheckShape(params.v, "v", vShape) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.oOut, "oOut", MakeShape({info.batch, info.seqlen, info.hv, info.vDim})) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const auto scalarShape = MakeShape({info.batch, info.seqlen, info.hv});
    CHECK_RET(CheckShape(params.g, "g", scalarShape) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return CheckShape(params.beta, "beta", scalarShape);
}

static aclnnStatus CheckSupportedL2Contract(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (UsePreparePath(params)) {
        if (!IsAscend950()) {
            gdn_error::Argument("feature combination", "the prepare path is supported on Ascend950 only");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!params.useQkL2norm && (params.qHatOutOptional != nullptr || params.kHatOutOptional != nullptr ||
                                  params.qRstdOutOptional != nullptr || params.kRstdOutOptional != nullptr)) {
            gdn_error::Argument("Q/K L2Norm outputs", "must be omitted when useQkL2norm is false");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(params.layout, "BNSD") != 0 && std::strcmp(params.layout, "NTD") != 0) {
        gdn_error::Attr("layout", params.layout, "BNSD or NTD on the Phase 6 path");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto &entry : {std::make_pair(params.qHatOutOptional, "qHatOutOptional"),
                              std::make_pair(params.kHatOutOptional, "kHatOutOptional"),
                              std::make_pair(params.qRstdOutOptional, "qRstdOutOptional"),
                              std::make_pair(params.kRstdOutOptional, "kRstdOutOptional"),
                              std::make_pair(params.betaEffOutOptional, "betaEffOutOptional"),
                              std::make_pair(params.hOutOptional, "hOutOptional")}) {
        if (entry.first != nullptr) {
            gdn_error::Argument(entry.second, "reserved output is unsupported on this path; must be omitted");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleFwdParams &params)
{
    GdnShapeInfo info;
    const auto shapeStatus = ResolveShapeInfo(params, info);
    if (shapeStatus != ACLNN_SUCCESS) {
        return shapeStatus;
    }
    if (info.batch <= 0 || info.hq <= 0 || info.hv <= 0 || info.seqlen <= 0) {
        gdn_error::Argument("q/v", "B/H/T dimensions must be positive; q=" +
                            std::string(op::ToString(params.q->GetViewShape()).GetString()) + ", v=" +
                            op::ToString(params.v->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.hv % info.hq != 0) {
        gdn_error::Argument("v", "Hv=" + std::to_string(info.hv) + " must be divisible by Hk=" +
                            std::to_string(info.hq));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.kDim != HEAD_DIM_128 ||
        (info.vDim != HEAD_DIM_128 && info.vDim != CHUNK_GATED_DELTA_RULE_FWD_V256)) {
        gdn_error::Argument("q/v", "requires K=128 and V=128/256; got K=" + std::to_string(info.kDim) +
                            ", V=" + std::to_string(info.vDim));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Validate divisors and metadata before any chunk arithmetic or index access.
    if (params.chunkSize != CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 &&
        params.chunkSize != CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128) {
        gdn_error::Attr("chunkSize", std::to_string(params.chunkSize), "64 or 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(params.scale) || std::abs(params.scale) > std::numeric_limits<float>::max()) {
        gdn_error::Attr("scale", std::to_string(params.scale), "finite and within the finite FP32 range");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((params.cuSeqlensOptional == nullptr) != (params.chunkIndicesOptional == nullptr)) {
        gdn_error::Argument("cuSeqlensOptional/chunkIndicesOptional", "must be both present or both absent");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional != nullptr && info.batch != 1) {
        gdn_error::Argument("q/v", "varlen requires physical B=1; got B=" + std::to_string(info.batch));
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckMetadata(params, info.seqlen) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const int64_t chunks = ExpectedChunks(params, info.seqlen);
    const int64_t seqNum = SeqNum(params, info.batch);
    const int64_t stateDim0 = params.stateVFirst ? info.vDim : info.kDim;
    const int64_t stateDim1 = params.stateVFirst ? info.kDim : info.vDim;
    CHECK_RET(CheckStateShape(params.initialStateOptional, "initialStateOptional", seqNum, info.hv,
                              stateDim0, stateDim1) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckStateShape(params.finalStateOutOptional, "finalStateOutOptional", seqNum, info.hv,
                              stateDim0, stateDim1) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.gCumsumOutOptional, "gCumsumOutOptional", params.g->GetViewShape()) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.aOutOptional, "aOutOptional",
                         MakeShape({info.batch, info.hv, info.seqlen, params.chunkSize})) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.qHatOutOptional, "qHatOutOptional", params.q->GetViewShape()) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.kHatOutOptional, "kHatOutOptional", params.k->GetViewShape()) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const auto rstdShape = MakeShape({info.batch, info.hq, info.seqlen});
    CHECK_RET(CheckShape(params.qRstdOutOptional, "qRstdOutOptional", rstdShape) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.kRstdOutOptional, "kRstdOutOptional", rstdShape) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.betaEffOutOptional, "betaEffOutOptional", params.beta->GetViewShape()) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params.hOutOptional, "hOutOptional",
                         MakeShape({info.batch, chunks, info.hv, stateDim0, stateDim1})) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const DataType dtype = params.q->GetDataType();
    CHECK_RET(CheckDtype(params.q, "q", {DataType::DT_FLOAT16, DataType::DT_BF16}) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    for (const auto &entry : {std::make_pair(params.k, "k"), std::make_pair(params.v, "v")}) {
        CHECK_RET(CheckDtype(entry.first, entry.second, {dtype}) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    for (const auto &entry : {std::make_pair(params.g, "g"), std::make_pair(params.beta, "beta"),
                              std::make_pair(params.initialStateOptional, "initialStateOptional")}) {
        CHECK_RET(CheckDtype(entry.first, entry.second, {DataType::DT_FLOAT, dtype}) ==
                      ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    for (const auto &entry : {std::make_pair(params.oOut, "oOut"),
                              std::make_pair(params.aOutOptional, "aOutOptional"),
                              std::make_pair(params.qHatOutOptional, "qHatOutOptional"),
                              std::make_pair(params.kHatOutOptional, "kHatOutOptional"),
                              std::make_pair(params.hOutOptional, "hOutOptional")}) {
        CHECK_RET(CheckDtype(entry.first, entry.second, {dtype}, true) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    for (const auto &entry : {std::make_pair(params.gCumsumOutOptional, "gCumsumOutOptional"),
                              std::make_pair(params.qRstdOutOptional, "qRstdOutOptional"),
                              std::make_pair(params.kRstdOutOptional, "kRstdOutOptional"),
                              std::make_pair(params.betaEffOutOptional, "betaEffOutOptional")}) {
        CHECK_RET(CheckDtype(entry.first, entry.second, {DataType::DT_FLOAT}, true) ==
                      ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    const auto stateDtype = params.initialStateOptional == nullptr
                                ? DataType::DT_FLOAT : params.initialStateOptional->GetDataType();
    CHECK_RET(CheckDtype(params.finalStateOutOptional, "finalStateOutOptional", {stateDtype}, true) ==
                  ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSupportedL2Contract(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (UsePreparePath(params) &&
        (dtype != DataType::DT_BF16 || info.vDim != HEAD_DIM_128 ||
         params.chunkSize != CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 || info.hv / info.hq > 4)) {
        gdn_error::Argument("prepare path", "requires BF16, K=V=128, chunkSize=64 and Hv/Hk in {1,2,3,4}");
        return ACLNN_ERR_PARAM_INVALID;
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
    if (workspaceSize == nullptr || executor == nullptr) {
        gdn_error::Required(workspaceSize == nullptr ? "workspaceSize" : "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    CHECK_RET(CheckRequiredInputs(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    if (CheckZeroShape(params, workspaceSize) != ACLNN_SUCCESS) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    const auto paramsStatus = CheckParams(params);
    if (paramsStatus != ACLNN_SUCCESS) {
        return paramsStatus;
    }

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
    const aclTensor *betaUsed = params.beta;
    const aclTensor *gUsed = params.g;
    if(params.beta->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == DataType::DT_FLOAT) {
        betaUsed = params.beta->GetDataType() == DataType::DT_FLOAT
                                        ? params.beta
                                        : l0op::Cast(params.beta, DataType::DT_FLOAT, executorPtr);
        gUsed = params.g->GetDataType() == DataType::DT_FLOAT
                                        ? params.g
                                        : l0op::Cast(params.g, DataType::DT_FLOAT, executorPtr);
    }
    const aclTensor *gBht = gUsed == nullptr
                                ? nullptr
                                : TransposeContiguous(gUsed, {0, 2, 1}, executorPtr);
    const aclTensor *betaBht = betaUsed == nullptr
                                   ? nullptr
                                   : TransposeContiguous(betaUsed, {0, 2, 1}, executorPtr);
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
        const op::Shape hShape = MakeShape({batch, ExpectedChunks(params, seqlen), hv, stateDim0, stateDim1});
        const op::Shape stateShape = MakeShape({seqNum, hv, stateDim0, stateDim1});
        const DataType dtype = params.q->GetDataType();
        const DataType stateDtype = params.initialStateOptional == nullptr
                                        ? DataType::DT_FLOAT
                                        : params.initialStateOptional->GetDataType();

        const aclTensor *gCumsumBht = executorPtr->AllocTensor(scalarHvShape, DataType::DT_FLOAT, Format::FORMAT_ND);
        const aclTensor *w = executorPtr->AllocTensor(wShape, dtype, Format::FORMAT_ND);
        const aclTensor *u = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
        const aclTensor *a = executorPtr->AllocTensor(aShape, dtype, Format::FORMAT_ND);
        const aclTensor *qHat = params.useQkL2norm
                                    ? executorPtr->AllocTensor(qkShape, dtype, Format::FORMAT_ND)
                                    : nullptr;
        const aclTensor *kHat = params.useQkL2norm
                                    ? executorPtr->AllocTensor(qkShape, dtype, Format::FORMAT_ND)
                                    : nullptr;
        const aclTensor *qRstd = params.useQkL2norm
                                     ? executorPtr->AllocTensor(scalarQkShape, DataType::DT_FLOAT, Format::FORMAT_ND)
                                     : nullptr;
        const aclTensor *kRstd = params.useQkL2norm
                                     ? executorPtr->AllocTensor(scalarQkShape, DataType::DT_FLOAT, Format::FORMAT_ND)
                                     : nullptr;
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
                            (!params.useQkL2norm ||
                             (qHat != nullptr && kHat != nullptr && qRstd != nullptr && kRstd != nullptr)) &&
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
        const aclTensor *qCompute = params.useQkL2norm ? qHat : qHead;
        const aclTensor *kCompute = params.useQkL2norm ? kHat : kHead;

        auto prepareResult = l0op::ChunkGatedDeltaRuleFwdPrepare(
            qHead, kHead, vHead, gBht, betaBht, params.aLogOptional, params.dtBiasOptional,
            params.cuSeqlensOptional, params.chunkIndicesOptional, params.chunkSize,
            params.allowNegEigval, params.useExp2, params.useQkL2norm,
            params.aLogOptional != nullptr || params.dtBiasOptional != nullptr, betaEffBht != nullptr,
            params.aOutOptional != nullptr, gCumsumBht, w, u, a, qHat, kHat, qRstd,
            kRstd, betaEffBht, executorPtr);
        GDN_STAGE_CHECK(prepareResult[0] != nullptr && prepareResult[1] != nullptr &&
                            prepareResult[2] != nullptr &&
                            (!params.useQkL2norm ||
                             (prepareResult[4] != nullptr && prepareResult[5] != nullptr)),
                        169104);

        auto hResult = l0op::ChunkFwdH(
            kCompute, w, u, gCumsumBht, nullptr, params.initialStateOptional,
            params.cuSeqlensOptional, params.chunkIndicesOptional, outputFinalState,
            params.chunkSize, true, params.useExp2, params.stateVFirst, h, vNew, finalState, executorPtr);
        GDN_STAGE_CHECK(hResult[0] != nullptr && hResult[1] != nullptr, 169105);

        auto oResult = l0op::ChunkFwdO(
            qCompute, kCompute, vNew, h, gCumsumBht, params.cuSeqlensOptional,
            params.chunkIndicesOptional, params.scale, params.chunkSize, params.useExp2, params.stateVFirst,
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
            CHECK_RET(ViewCopyIfPresent(qRstd, params.qRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_INNER_NULLPTR);
        }
        if (params.kRstdOutOptional != nullptr) {
            CHECK_RET(ViewCopyIfPresent(kRstd, params.kRstdOutOptional, executorPtr) == ACLNN_SUCCESS,
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
        // A5 Phase6 recognizes this required-output placeholder and skips only
        // the unused public BTH export; its internal BHT cumsum remains intact.
        const auto gCumsumShape = IsAscend950() ? MakeShape({1}) : MakeShape({batch, seqlen, hv});
        gCumsumCompute = executorPtr->AllocTensor(gCumsumShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    } else if (IsAscend950()) {
        // Public descriptors may expose a flat storage shape. Give tiling an
        // executor-owned dense BTH view without changing the caller's descriptor
        // or allocating/copying output data. Preserve the caller's data offset.
        const auto gCumsumShape = MakeShape({batch, seqlen, hv});
        auto *gCumsumView = executorPtr->CreateView(
            gCumsumCompute, gCumsumShape, gCumsumCompute->GetViewOffset());
        if (gCumsumView != nullptr) {
            gCumsumView->SetStorageShape(gCumsumShape);
            gCumsumView->SetOriginalShape(gCumsumShape);
        }
        gCumsumCompute = gCumsumView;
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
