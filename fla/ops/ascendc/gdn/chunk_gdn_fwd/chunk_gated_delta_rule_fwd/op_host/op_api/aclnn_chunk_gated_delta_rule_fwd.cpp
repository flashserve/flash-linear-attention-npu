/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
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
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <cmath>
#include <cstring>
#include <initializer_list>

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkGatedDeltaRuleFwd";

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t HEAD_DIM_128 = 128;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_V256 = 256;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_32 = 32;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 = 64;
constexpr int64_t CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128 = 128;
// Stage P encodes the GM row gap in DataCopyExtParams as uint32 bytes.
// This is a transport-field bound, so larger Hv stays on the BHT route.
constexpr int64_t CUMSUM_PREPARE_MAX_HEADS =
    static_cast<int64_t>(0xffffffffULL / sizeof(float) + 1);

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

static bool IsDav2201Device();

static bool IsNativeQkvLayout(const ChunkGatedDeltaRuleFwdParams &params)
{
    return IsDav2201Device() && params.layout != nullptr &&
           (std::strcmp(params.layout, "BSND") == 0 || std::strcmp(params.layout, "TND") == 0);
}

static bool UseA2PreparePath(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (!IsDav2201Device() || params.q == nullptr || params.k == nullptr || params.v == nullptr ||
        params.layout == nullptr || params.q->GetViewShape().GetDimNum() != 4 ||
        params.k->GetViewShape().GetDimNum() != 4 || params.v->GetViewShape().GetDimNum() != 4) {
        return false;
    }
    const bool sequenceMajor = std::strcmp(params.layout, "BSND") == 0 ||
                               std::strcmp(params.layout, "TND") == 0;
    const bool headMajor = std::strcmp(params.layout, "BNSD") == 0 ||
                           std::strcmp(params.layout, "NTD") == 0;
    if (!sequenceMajor && !headMajor) {
        return false;
    }
    if (params.cuSeqlensOptional != nullptr) {
        for (size_t i = 1; i < params.cuSeqlensOptional->Size(); ++i) {
            if ((*params.cuSeqlensOptional)[i - 1] >= (*params.cuSeqlensOptional)[i]) {
                // ChunkFwdH does not publish a state for an empty sequence.
                return false;
            }
        }
    }
    const size_t headDim = sequenceMajor ? 2 : 1;
    const int64_t hk = params.k->GetViewShape().GetDim(headDim);
    const int64_t hv = params.v->GetViewShape().GetDim(headDim);
    // The complete Prepare/H/O route is bounded by ChunkFwdH's contract.
    // Standalone Prepare supports more dtypes and dimensions than this route.
    return params.q->GetDataType() == DataType::DT_BF16 &&
           params.k->GetDataType() == DataType::DT_BF16 && params.v->GetDataType() == DataType::DT_BF16 &&
           params.q->GetViewShape().GetDim(3) == HEAD_DIM_128 &&
           params.k->GetViewShape().GetDim(3) == HEAD_DIM_128 &&
           params.v->GetViewShape().GetDim(3) == HEAD_DIM_128 &&
           params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 && hk > 0 && hv > 0 && hv % hk == 0;
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
    if (UseA2PreparePath(params)) {
        return true;
    }
    const bool legacyLayout = std::strcmp(params.layout, "BNSD") == 0 ||
                              std::strcmp(params.layout, "NTD") == 0;
    // DAV_2201 A2/A3 devices read token-major QKV directly; feature selection is
    // otherwise unchanged across the two products.
    return params.useExp2 || params.useQkL2norm ||
           params.aLogOptional != nullptr || params.dtBiasOptional != nullptr ||
           params.betaEffOutOptional != nullptr || params.allowNegEigval ||
           params.hOutOptional != nullptr || params.stateVFirst ||
           (!legacyLayout && !IsNativeQkvLayout(params));
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

static const aclTensor *DenseQkvView(const aclTensor *tensor, aclOpExecutor *executor)
{
    // MakeContiguous has already materialized noncontiguous inputs. This is metadata only.
    // Preserve a contiguous caller view's storage offset without mutating its descriptor.
    auto *view = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageShape(tensor->GetViewShape());
        view->SetOriginalShape(tensor->GetViewShape());
    }
    return view;
}

static const aclTensor *A2PrepareInputView(const aclTensor *tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsPrivateFormat(tensor->GetStorageFormat()) ||
        IsPrivateFormat(tensor->GetViewFormat())) {
        return nullptr;
    }
    auto elements = [](const op::Shape &shape, uint64_t &count) {
        count = 1;
        if (shape.GetDimNum() == 0) {
            return false;
        }
        for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
            const int64_t dim = shape.GetDim(axis);
            if (dim <= 0 || count > static_cast<uint64_t>(INT64_MAX) / static_cast<uint64_t>(dim)) {
                return false;
            }
            count *= static_cast<uint64_t>(dim);
        }
        return true;
    };
    uint64_t count = 0;
    uint64_t storageElements = 0;
    const int64_t offset = tensor->GetViewOffset();
    if (!elements(tensor->GetViewShape(), count) || !elements(tensor->GetStorageShape(), storageElements) ||
        offset < 0 || static_cast<uint64_t>(offset) > static_cast<uint64_t>(INT64_MAX) - count) {
        return nullptr;
    }
    // Storage shape may describe the logical view, not the backing allocation extent.
    // Preserve the offset when sharing storage; a contiguous slice needs no data copy.
    auto *view = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
    if (view != nullptr) {
        view->SetStorageShape(tensor->GetViewShape());
        view->SetOriginalShape(tensor->GetViewShape());
        view->SetStorageFormat(Format::FORMAT_ND);
        view->SetViewFormat(Format::FORMAT_ND);
        view->SetOriginalFormat(Format::FORMAT_ND);
    }
    return view;
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
    if (!(shape.GetDimNum() == 4 && shape.GetDim(0) == seqNum && shape.GetDim(1) == hv &&
          shape.GetDim(2) == kDim && shape.GetDim(3) == vDim)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, name, std::string(op::ToString(shape).GetString()),
            "[seqNum,Hv,K,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMetadata(const ChunkGatedDeltaRuleFwdParams &params, int64_t seqlen)
{
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    const aclIntArray &cu = *params.cuSeqlensOptional;
    if (cu.Size() < 2) {
        OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "cuSeqlens", std::to_string(cu.Size()), ">= 2");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(cu[0] == 0 && cu[cu.Size() - 1] == seqlen)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cuSeqlens[0],cuSeqlens[last]",
            std::to_string(cu[0]) + "," + std::to_string(cu[cu.Size() - 1]),
            "cuSeqlens must start at 0 and end at T=" + std::to_string(seqlen));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        if (cu[idx] > cu[idx + 1]) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "cuSeqlens[idx],cuSeqlens[idx+1]",
                std::to_string(cu[idx]) + "," + std::to_string(cu[idx + 1]),
                "cuSeqlens must be nondecreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    const aclIntArray &indices = *params.chunkIndicesOptional;
    const int64_t expectedChunks = ExpectedChunks(params, seqlen);
    if (!(indices.Size() % 2 == 0 && static_cast<int64_t>(indices.Size() / 2) == expectedChunks)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "chunkIndices", std::to_string(indices.Size()),
            std::to_string(expectedChunks * 2));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t index = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        const int64_t seqChunks = (cu[seq + 1] - cu[seq] + params.chunkSize - 1) / params.chunkSize;
        for (int64_t localChunk = 0; localChunk < seqChunks; ++localChunk) {
            if (indices[index] != static_cast<int64_t>(seq) || indices[index + 1] != localChunk) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    OP_NAME, "chunkIndices[index],chunkIndices[index+1]",
                    std::to_string(indices[index]) + "," + std::to_string(indices[index + 1]),
                    "chunkIndices must use canonical sequence-major order");
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
    if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: tensor != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckRequiredInputs(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.g == nullptr ||
        params.beta == nullptr || params.oOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q,k,v,g,beta,oOut");
        return ACLNN_ERR_PARAM_NULLPTR;
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
    if (!(src != nullptr && l0op::ViewCopy(src, dst, executor) != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: src != nullptr && l0op::ViewCopy(src, dst, executor) != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
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
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(OP_NAME, name, "tensor must not be nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (tensor->GetViewShape().GetDimNum() != rank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, name, std::to_string(tensor->GetViewShape().GetDimNum()),
            "rank must be " + std::to_string(rank));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOptionalRank(const aclTensor *tensor, size_t rank, const char *name)
{
    return tensor == nullptr ? ACLNN_SUCCESS : CheckRank(tensor, rank, name);
}

static bool IsDav2201Device()
{
    // A2 and A3 share the DAV_2201 instruction/tiling contract. Select the
    // common Phase 6 route from the runtime architecture instead of
    // hard-coding the A2 910B SoC name.
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201;
}

static bool UsePreparedCumsum(const ChunkGatedDeltaRuleFwdParams &params,
                              const GdnShapeInfo &info)
{
    return IsDav2201Device() && !UsePreparePath(params) &&
           info.hv <= CUMSUM_PREPARE_MAX_HEADS &&
           info.kDim == HEAD_DIM_128 &&
           (info.vDim == HEAD_DIM_128 ||
            info.vDim == CHUNK_GATED_DELTA_RULE_FWD_V256) &&
           (params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 ||
            params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128);
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "layout", params.layout,
            "layout must be uppercase and one of BNSD, BSND, NTD or TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    info.isSequenceMajor = info.layout == GdnLayout::BSND || info.layout == GdnLayout::TND;
    if (!(Rank(params.q) == 4 && Rank(params.k) == 4 && Rank(params.v) == 4 &&
          Rank(params.oOut) == 4 && Rank(params.g) == 3 && Rank(params.beta) == 3)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            OP_NAME, "q,k,v,oOut,g,beta",
            std::to_string(Rank(params.q)) + "," + std::to_string(Rank(params.k)) + "," +
                std::to_string(Rank(params.v)) + "," + std::to_string(Rank(params.oOut)) + "," +
                std::to_string(Rank(params.g)) + "," + std::to_string(Rank(params.beta)),
            "q/k/v/oOut must be rank 4 and g/beta must be rank 3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!ShapeEqual(params.q->GetViewShape(), params.k->GetViewShape())) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q,k",
            "q=" + std::string(op::ToString(params.q->GetViewShape()).GetString()) +
                ",k=" + std::string(op::ToString(params.k->GetViewShape()).GetString()),
            "q and k must have identical shapes");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (!info.isSequenceMajor) {
        info.batch = Dim(params.q, 0);
        info.hq = Dim(params.q, 1);
        info.seqlen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hv = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        if (!HasShape(params.v, {info.batch, info.hv, info.seqlen, info.vDim})) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "v", std::string(op::ToString(params.v->GetViewShape()).GetString()),
                "BNSD/NTD expects v as [B,Hv,T,V]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.hq = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hv = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        if (!HasShape(params.v, {info.batch, info.seqlen, info.hv, info.vDim})) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "v", std::string(op::ToString(params.v->GetViewShape()).GetString()),
                "BSND/TND expects v as [B,T,Hv,V]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!HasShape(params.oOut, {info.batch, info.seqlen, info.hv, info.vDim})) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "oOut", std::string(op::ToString(params.oOut->GetViewShape()).GetString()),
            "oOut must use BSND shape [B,T,Hv,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HasShape(params.g, {info.batch, info.seqlen, info.hv}) &&
          HasShape(params.beta, {info.batch, info.seqlen, info.hv}))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "g,beta",
            "g=" + std::string(op::ToString(params.g->GetViewShape()).GetString()) +
                ",beta=" + std::string(op::ToString(params.beta->GetViewShape()).GetString()),
            "g and beta must have shape [B,T,Hv]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSupportedL2Contract(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (params.layout == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "layout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (IsDav2201Device()) {
        if (!(params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 ||
              params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "chunkSize", std::to_string(params.chunkSize), "chunkSize must be 64 or 128");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (UsePreparePath(params)) {
        if (!(IsAscend950() || IsDav2201Device())) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "soc", "unsupported", "the prepare path requires Ascend950 or DAV_2201");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(std::strcmp(params.layout, "BNSD") == 0 || std::strcmp(params.layout, "BSND") == 0 ||
              std::strcmp(params.layout, "NTD") == 0 || std::strcmp(params.layout, "TND") == 0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "layout", params.layout,
                "the prepare path supports BNSD, BSND, NTD and TND");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(params.useQkL2norm ||
              (params.qHatOutOptional == nullptr && params.kHatOutOptional == nullptr &&
               params.qRstdOutOptional == nullptr && params.kRstdOutOptional == nullptr))) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "useQkL2norm,qHatOut,kHatOut,qRstdOut,kRstdOut",
                std::string("useQkL2norm=") + (params.useQkL2norm ? "true" : "false"),
                "Q/K L2Norm outputs must be omitted when useQkL2norm is false");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (!(std::strcmp(params.layout, "BNSD") == 0 || std::strcmp(params.layout, "NTD") == 0 ||
          IsNativeQkvLayout(params))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "layout", params.layout,
            "this device does not support the requested QKV layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.qHatOutOptional != nullptr || params.kHatOutOptional != nullptr ||
        params.qRstdOutOptional != nullptr || params.kRstdOutOptional != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "qHatOut,kHatOut,qRstdOut,kRstdOut", "present",
            "Q/K L2Norm intermediate outputs are reserved and not supported yet");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.betaEffOutOptional != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "betaEffOut", "present", "betaEffOutOptional is reserved and not supported yet");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.hOutOptional != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "hOut", "present", "hOutOptional is reserved and not supported yet");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckChunk32Params(const ChunkGatedDeltaRuleFwdParams &params)
{
    if (!IsAscend950()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "soc", "unsupported", "chunkSize=32 is supported on Ascend950 only");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.layout != nullptr &&
          (std::strcmp(params.layout, "BNSD") == 0 || std::strcmp(params.layout, "BSND") == 0 ||
           std::strcmp(params.layout, "NTD") == 0 || std::strcmp(params.layout, "TND") == 0))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "layout", params.layout == nullptr ? "nullptr" : params.layout,
            "chunkSize=32 supports BNSD, BSND, NTD and TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(!params.useExp2 && params.useQkL2norm)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "useExp2,useQkL2norm",
            std::string("useExp2=") + (params.useExp2 ? "true" : "false") +
                ",useQkL2norm=" + (params.useQkL2norm ? "true" : "false"),
            "chunkSize=32 requires useExp2=false and useQkL2norm=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.aLogOptional != nullptr || params.dtBiasOptional != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "aLog,dtBias",
            std::string(params.aLogOptional != nullptr ? "aLog=present" : "aLog=absent") + "," +
                (params.dtBiasOptional != nullptr ? "dtBias=present" : "dtBias=absent"),
            "chunkSize=32 does not support aLog or dtBias");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((params.gCumsumOutOptional == nullptr) != (params.aOutOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "gCumsumOut,A",
            std::string(params.gCumsumOutOptional == nullptr ? "g_cumsum=absent" : "g_cumsum=present") + "," +
                (params.aOutOptional == nullptr ? "A=absent" : "A=present"),
            "chunkSize=32 g_cumsum and A must be provided together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool anyL2 = params.qHatOutOptional != nullptr || params.kHatOutOptional != nullptr ||
                       params.qRstdOutOptional != nullptr || params.kRstdOutOptional != nullptr;
    const bool allL2 = params.qHatOutOptional != nullptr && params.kHatOutOptional != nullptr &&
                       params.qRstdOutOptional != nullptr && params.kRstdOutOptional != nullptr;
    if (anyL2 && !allL2) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "qHatOut,kHatOut,qRstdOut,kRstdOut", "partial set",
            "chunkSize=32 L2 outputs must be provided together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(Rank(params.q) == 4 && Rank(params.k) == 4 && Rank(params.v) == 4 &&
          Rank(params.oOut) == 4 && Rank(params.g) == 3 && Rank(params.beta) == 3)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            OP_NAME, "q,k,v,oOut,g,beta",
            std::to_string(Rank(params.q)) + "," + std::to_string(Rank(params.k)) + "," +
                std::to_string(Rank(params.v)) + "," + std::to_string(Rank(params.oOut)) + "," +
                std::to_string(Rank(params.g)) + "," + std::to_string(Rank(params.beta)),
            "chunkSize=32 requires rank-4 q/k/v/o and rank-3 g/beta");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const bool sequenceMajor = std::strcmp(params.layout, "BSND") == 0 ||
                               std::strcmp(params.layout, "TND") == 0;
    const int64_t batch = Dim(params.q, 0);
    const int64_t tokens = sequenceMajor ? Dim(params.q, 1) : Dim(params.q, 2);
    const int64_t keyHeads = sequenceMajor ? Dim(params.q, 2) : Dim(params.q, 1);
    const int64_t keyDim = Dim(params.q, 3);
    const int64_t valueHeads = sequenceMajor ? Dim(params.v, 2) : Dim(params.v, 1);
    const int64_t valueDim = Dim(params.v, 3);
    if (!(batch > 0 && keyHeads > 0 && valueHeads > 0 && tokens > 0 &&
          valueHeads % keyHeads == 0 && keyDim == HEAD_DIM_128 && valueDim == HEAD_DIM_128)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "B,Hk,Hv,T,K,V",
            std::to_string(batch) + "," + std::to_string(keyHeads) + "," + std::to_string(valueHeads) +
                "," + std::to_string(tokens) + "," + std::to_string(keyDim) + "," + std::to_string(valueDim),
            "chunkSize=32 requires positive B/H/T, Hv divisible by Hk, and K=V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((sequenceMajor
               ? (HasShape(params.k, {batch, tokens, keyHeads, keyDim}) &&
                  HasShape(params.v, {batch, tokens, valueHeads, valueDim}))
               : (HasShape(params.k, {batch, keyHeads, tokens, keyDim}) &&
                  HasShape(params.v, {batch, valueHeads, tokens, valueDim}))) &&
          HasShape(params.g, {batch, tokens, valueHeads}) &&
          HasShape(params.beta, {batch, tokens, valueHeads}) &&
          HasShape(params.oOut, {batch, tokens, valueHeads, valueDim}))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "k,v,g,beta,oOut",
            "k=" + std::string(op::ToString(params.k->GetViewShape()).GetString()) +
                ",v=" + std::string(op::ToString(params.v->GetViewShape()).GetString()),
            "chunkSize=32 expects q/k/v in the requested layout, g/beta as [B,T,Hv], and o as BSND [B,T,Hv,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.qHatOutOptional != nullptr) {
        const bool hatShape = sequenceMajor
                                  ? HasShape(params.qHatOutOptional, {batch, tokens, keyHeads, keyDim}) &&
                                        HasShape(params.kHatOutOptional, {batch, tokens, keyHeads, keyDim})
                                  : HasShape(params.qHatOutOptional, {batch, keyHeads, tokens, keyDim}) &&
                                        HasShape(params.kHatOutOptional, {batch, keyHeads, tokens, keyDim});
        if (!(hatShape &&
              HasShape(params.qRstdOutOptional, {batch, keyHeads, tokens}) &&
              HasShape(params.kRstdOutOptional, {batch, keyHeads, tokens}) &&
              params.qRstdOutOptional->GetDataType() == DataType::DT_FLOAT &&
              params.kRstdOutOptional->GetDataType() == DataType::DT_FLOAT &&
              params.qHatOutOptional->GetDataType() == DataType::DT_BF16 &&
              params.kHatOutOptional->GetDataType() == DataType::DT_BF16)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "qHatOut,kHatOut,qRstdOut,kRstdOut",
                "qHat=" + std::string(op::ToString(params.qHatOutOptional->GetViewShape()).GetString()) +
                    ",qRstd=" + std::string(op::ToString(params.qRstdOutOptional->GetViewShape()).GetString()),
                "chunkSize=32 q_hat/k_hat match q/k, and rstd is FP32 [B,Hk,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.betaEffOutOptional != nullptr) {
        if (!(HasShape(params.betaEffOutOptional, {batch, tokens, valueHeads}) &&
              params.betaEffOutOptional->GetDataType() == DataType::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "betaEffOut",
                std::string(op::ToString(params.betaEffOutOptional->GetViewShape()).GetString()),
                "chunkSize=32 beta_eff is FP32 [B,T,Hv]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.hOutOptional != nullptr) {
        const op::Shape &hShape = params.hOutOptional->GetViewShape();
        if (!(hShape.GetDimNum() == 5 && hShape.GetDim(0) == batch &&
              hShape.GetDim(1) == ExpectedChunks(params, tokens) &&
              hShape.GetDim(2) == valueHeads && hShape.GetDim(3) == HEAD_DIM_128 &&
              hShape.GetDim(4) == HEAD_DIM_128 &&
              params.hOutOptional->GetDataType() == DataType::DT_BF16)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "hOut", std::string(op::ToString(hShape).GetString()),
                "chunkSize=32 h is BF16 [B,NT,Hv,128,128]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.gCumsumOutOptional != nullptr) {
        if (!(HasShape(params.gCumsumOutOptional, {batch, tokens, valueHeads}) &&
              params.gCumsumOutOptional->GetDataType() == DataType::DT_FLOAT &&
              HasShape(params.aOutOptional, {batch, valueHeads, tokens, CHUNK_GATED_DELTA_RULE_FWD_CHUNK_32}) &&
              params.aOutOptional->GetDataType() == DataType::DT_BF16)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "gCumsumOut,A",
                "g_cumsum=" + std::string(op::ToString(params.gCumsumOutOptional->GetViewShape()).GetString()) +
                    ",A=" + std::string(op::ToString(params.aOutOptional->GetViewShape()).GetString()),
                "chunkSize=32 g_cumsum is FP32 [B,T,Hv] and A is BF16 [B,Hv,T,32]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (!(params.q->GetDataType() == DataType::DT_BF16 &&
          params.k->GetDataType() == DataType::DT_BF16 &&
          params.v->GetDataType() == DataType::DT_BF16 &&
          params.beta->GetDataType() == DataType::DT_BF16 &&
          params.g->GetDataType() == DataType::DT_FLOAT &&
          params.oOut->GetDataType() == DataType::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "q,k,v,beta,g,oOut",
            std::string("q=") + ge::TypeUtils::DataTypeToSerialString(params.q->GetDataType()) +
                ",g=" + ge::TypeUtils::DataTypeToSerialString(params.g->GetDataType()),
            "chunkSize=32 requires BF16 q/k/v/beta/oOut and FP32 g");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(params.scale)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "scale", std::to_string(params.scale), "chunkSize=32 requires a finite scale");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((params.cuSeqlensOptional == nullptr) != (params.chunkIndicesOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cuSeqlens,chunkIndices",
            std::string(params.cuSeqlensOptional == nullptr ? "cuSeqlens=absent" : "cuSeqlens=present") + "," +
                (params.chunkIndicesOptional == nullptr ? "chunkIndices=absent" : "chunkIndices=present"),
            "cuSeqlens and chunkIndices must be both present or both absent");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional != nullptr && batch != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "B", std::to_string(batch), "chunkSize=32 varlen input requires physical B=1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    {
        const aclnnStatus checkStatus = CheckMetadata(params, tokens);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };

    const int64_t seqNum = SeqNum(params, batch);
    const int64_t stateDim0 = params.stateVFirst ? valueDim : keyDim;
    const int64_t stateDim1 = params.stateVFirst ? keyDim : valueDim;
    {
        const aclnnStatus checkStatus = CheckStateShape(params.initialStateOptional, "initialState", seqNum, valueHeads, stateDim0, stateDim1);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckStateShape(params.finalStateOutOptional, "finalStateOut", seqNum, valueHeads, stateDim0, stateDim1);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    if (params.initialStateOptional != nullptr) {
        const DataType stateDtype = params.initialStateOptional->GetDataType();
        if (!(stateDtype == DataType::DT_BF16 || stateDtype == DataType::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "initialState", ge::TypeUtils::DataTypeToSerialString(stateDtype),
                "chunkSize=32 initialState must be BF16 or FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.finalStateOutOptional != nullptr) {
        const bool finalFp32 = params.initialStateOptional == nullptr ||
                               params.initialStateOptional->GetDataType() == DataType::DT_FLOAT;
        const DataType expect = finalFp32 ? DataType::DT_FLOAT : DataType::DT_BF16;
        if (params.finalStateOutOptional->GetDataType() != expect) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "finalStateOut",
                ge::TypeUtils::DataTypeToSerialString(params.finalStateOutOptional->GetDataType()),
                "chunkSize=32 finalStateOut is FP32 when initialState is absent/FP32, BF16 when initialState is BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleFwdParams &params)
{
    const aclnnStatus supportedContractStatus = CheckSupportedL2Contract(params);
    if (supportedContractStatus != ACLNN_SUCCESS) {
        return supportedContractStatus;
    }
    GdnShapeInfo info;
    if (!(ResolveShapeInfo(params, info) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const size_t qkvRank = 4;
    const size_t scalarRank = 3;
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.gCumsumOutOptional, scalarRank, "gCumsumOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.aOutOptional, qkvRank, "aOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.qHatOutOptional, qkvRank, "qHatOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.kHatOutOptional, qkvRank, "kHatOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.qRstdOutOptional, scalarRank, "qRstdOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.kRstdOutOptional, scalarRank, "kRstdOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.betaEffOutOptional, scalarRank, "betaEffOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckOptionalRank(params.hOutOptional, 5, "hOutOptional");
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };

    const int64_t chunks = ExpectedChunks(params, info.seqlen);
    if (!(info.batch > 0 && info.hq > 0 && info.hv > 0 && info.seqlen > 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "B,Hk,Hv,T",
            std::to_string(info.batch) + "," + std::to_string(info.hq) + "," +
                std::to_string(info.hv) + "," + std::to_string(info.seqlen),
            "B/Hk/Hv/T dimensions must be positive");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.hv % info.hq != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "Hk,Hv", std::to_string(info.hq) + "," + std::to_string(info.hv),
            "GVA requires Hv divisible by Hk");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.kDim == HEAD_DIM_128 &&
          (info.vDim == HEAD_DIM_128 || info.vDim == CHUNK_GATED_DELTA_RULE_FWD_V256))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "K,V", std::to_string(info.kDim) + "," + std::to_string(info.vDim),
            "composite GDN core supports K=128 and V=128/256");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gCumsumOutOptional != nullptr) {
        if (!ShapeEqual(params.gCumsumOutOptional->GetViewShape(), params.g->GetViewShape())) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "gCumsumOut",
                std::string(op::ToString(params.gCumsumOutOptional->GetViewShape()).GetString()),
                "gCumsumOutOptional shape must match g");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.aOutOptional != nullptr) {
        const bool valid = HasShape(params.aOutOptional,
                                    {info.batch, info.hv, info.seqlen, params.chunkSize});
        if (!valid) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "aOut", std::string(op::ToString(params.aOutOptional->GetViewShape()).GetString()),
                "aOutOptional must use head-major [B,Hv,T,chunkSize]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.qHatOutOptional != nullptr) {
        if (!ShapeEqual(params.qHatOutOptional->GetViewShape(), params.q->GetViewShape())) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "qHatOut",
                std::string(op::ToString(params.qHatOutOptional->GetViewShape()).GetString()),
                "qHatOutOptional shape must match q");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.kHatOutOptional != nullptr) {
        if (!ShapeEqual(params.kHatOutOptional->GetViewShape(), params.k->GetViewShape())) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "kHatOut",
                std::string(op::ToString(params.kHatOutOptional->GetViewShape()).GetString()),
                "kHatOutOptional shape must match k");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.qRstdOutOptional != nullptr) {
        const bool valid = HasShape(params.qRstdOutOptional, {info.batch, info.hq, info.seqlen});
        if (!valid) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "qRstdOut", std::string(op::ToString(params.qRstdOutOptional->GetViewShape()).GetString()),
                "[B,Hk,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.kRstdOutOptional != nullptr) {
        const bool valid = HasShape(params.kRstdOutOptional, {info.batch, info.hq, info.seqlen});
        if (!valid) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "kRstdOut", std::string(op::ToString(params.kRstdOutOptional->GetViewShape()).GetString()),
                "[B,Hk,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.betaEffOutOptional != nullptr) {
        if (!ShapeEqual(params.betaEffOutOptional->GetViewShape(), params.beta->GetViewShape())) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "betaEffOut",
                std::string(op::ToString(params.betaEffOutOptional->GetViewShape()).GetString()),
                "betaEffOutOptional shape must match beta");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.hOutOptional != nullptr) {
        const int64_t stateDim0 = params.stateVFirst ? info.vDim : info.kDim;
        const int64_t stateDim1 = params.stateVFirst ? info.kDim : info.vDim;
        const bool valid = HasShape(params.hOutOptional,
                                    {info.batch, chunks, info.hv, stateDim0, stateDim1});
        if (!valid) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "hOut", std::string(op::ToString(params.hOutOptional->GetViewShape()).GetString()),
                "hOutOptional shape must match stateVFirst");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!(params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 ||
          params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_128)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunkSize", std::to_string(params.chunkSize), "chunkSize must be 64 or 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((params.cuSeqlensOptional == nullptr) != (params.chunkIndicesOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cuSeqlens,chunkIndices",
            std::string(params.cuSeqlensOptional == nullptr ? "cuSeqlens=absent" : "cuSeqlens=present") + "," +
                (params.chunkIndicesOptional == nullptr ? "chunkIndices=absent" : "chunkIndices=present"),
            "cuSeqlens and chunkIndices must be both present or both absent");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional != nullptr && info.batch != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "B", std::to_string(info.batch), "varlen rank-4 input requires physical B=1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    {
        const aclnnStatus checkStatus = CheckMetadata(params, info.seqlen);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };

    const int64_t seqNum = SeqNum(params, info.batch);
    const int64_t stateDim0 = params.stateVFirst ? info.vDim : info.kDim;
    const int64_t stateDim1 = params.stateVFirst ? info.kDim : info.vDim;
    if (!(CheckStateShape(params.initialStateOptional, "initialState", seqNum, info.hv, stateDim0, stateDim1) ==
                  ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.finalStateOutOptional != nullptr) {
        if (!(CheckStateShape(params.finalStateOutOptional, "finalStateOut", seqNum, info.hv,
                                  stateDim0, stateDim1) ==
                      ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    }

    const DataType dtype = params.q->GetDataType();
    if (!(dtype == DataType::DT_FLOAT16 || dtype == DataType::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "q", ge::TypeUtils::DataTypeToSerialString(dtype),
            "q/k/v must be float16 or bfloat16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.k->GetDataType() == dtype && params.v->GetDataType() == dtype &&
          params.oOut->GetDataType() == dtype &&
          (params.aOutOptional == nullptr || params.aOutOptional->GetDataType() == dtype))) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "q,k,v,oOut,aOut",
            std::string("q=") + ge::TypeUtils::DataTypeToSerialString(dtype) +
                ",k=" + ge::TypeUtils::DataTypeToSerialString(params.k->GetDataType()) +
                ",v=" + ge::TypeUtils::DataTypeToSerialString(params.v->GetDataType()),
            "q/k/v/oOut/aOutOptional must have the same dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((params.beta->GetDataType() == DataType::DT_FLOAT || params.beta->GetDataType() == dtype) &&
          (params.g->GetDataType() == DataType::DT_FLOAT || params.g->GetDataType() == dtype))) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "g,beta",
            std::string("g=") + ge::TypeUtils::DataTypeToSerialString(params.g->GetDataType()) +
                ",beta=" + ge::TypeUtils::DataTypeToSerialString(params.beta->GetDataType()),
            "g and beta must be float32 or match q/k/v");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.gCumsumOutOptional != nullptr &&
        params.gCumsumOutOptional->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "gCumsumOut",
            ge::TypeUtils::DataTypeToSerialString(params.gCumsumOutOptional->GetDataType()),
            "gCumsumOutOptional must be float32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((params.qHatOutOptional == nullptr || params.qHatOutOptional->GetDataType() == dtype) &&
          (params.kHatOutOptional == nullptr || params.kHatOutOptional->GetDataType() == dtype) &&
          (params.hOutOptional == nullptr || params.hOutOptional->GetDataType() == dtype))) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "qHatOut,kHatOut,hOut", "mismatch",
            "qHatOutOptional/kHatOutOptional/hOutOptional must match q dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((params.qRstdOutOptional == nullptr || params.qRstdOutOptional->GetDataType() == DataType::DT_FLOAT) &&
          (params.kRstdOutOptional == nullptr || params.kRstdOutOptional->GetDataType() == DataType::DT_FLOAT) &&
          (params.betaEffOutOptional == nullptr || params.betaEffOutOptional->GetDataType() == DataType::DT_FLOAT))) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "qRstdOut,kRstdOut,betaEffOut", "mismatch",
            "qRstdOutOptional/kRstdOutOptional/betaEffOutOptional must be float32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (UsePreparePath(params) && IsDav2201Device()) {
        if (!UseA2PreparePath(params)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, "prepare_path",
                "the DAV_2201 Prepare/H/O route requires BF16, K=V=128, chunkSize=64 and nonempty sequences");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.useExp2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "useExp2", "true",
                "DAV_2201 ChunkFwdO does not support useExp2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.dtBiasOptional != nullptr && params.aLogOptional == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, "aLog", "dtBias requires aLog for gate-in-kernel");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (params.allowNegEigval && params.betaEffOutOptional == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                OP_NAME, "betaEffOut", "allowNegEigval requires beta sigmoid");
            return ACLNN_ERR_PARAM_INVALID;
        }
        for (const aclTensor *gateParam : {params.aLogOptional, params.dtBiasOptional}) {
            if (!(gateParam == nullptr ||
                  (HasShape(gateParam, {info.hv}) &&
                   (gateParam->GetDataType() == DataType::DT_FLOAT ||
                    gateParam->GetDataType() == DataType::DT_FLOAT16 ||
                    gateParam->GetDataType() == DataType::DT_BF16)))) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    OP_NAME, "aLog,dtBias",
                    gateParam == nullptr ? "nullptr" : std::string(op::ToString(gateParam->GetViewShape()).GetString()),
                    "aLog/dtBias must be floating-point vectors of shape [Hv]");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    } else if (UsePreparePath(params)) {
        if (!(dtype == DataType::DT_BF16 && info.vDim == HEAD_DIM_128 &&
              params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_64 && info.hv / info.hq <= 4)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "dtype,V,chunkSize,Hv/Hk",
                ge::TypeUtils::DataTypeToSerialString(dtype) + "," + std::to_string(info.vDim) + "," +
                    std::to_string(params.chunkSize) + "," + std::to_string(info.hv / info.hq),
                "Ascend950 prepare path requires BF16, K=V=128, chunkSize=64 and Hv/Hk in {1,2,3,4}");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.initialStateOptional != nullptr) {
        const DataType stateDtype = params.initialStateOptional->GetDataType();
        if (!(stateDtype == DataType::DT_FLOAT || stateDtype == dtype)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "initialState", ge::TypeUtils::DataTypeToSerialString(stateDtype),
                "initialState dtype must be float32 or match q/k/v");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.finalStateOutOptional != nullptr) {
        const DataType expectedStateDtype = params.initialStateOptional == nullptr
                                                ? DataType::DT_FLOAT
                                                : params.initialStateOptional->GetDataType();
        if (params.finalStateOutOptional->GetDataType() != expectedStateDtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "finalStateOut",
                ge::TypeUtils::DataTypeToSerialString(params.finalStateOutOptional->GetDataType()),
                "finalStateOut dtype must match initialState, or be float32 when initialState is absent");
            return ACLNN_ERR_PARAM_INVALID;
        }
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
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize,executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(CheckRequiredInputs(params) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto executorPtr = uniqueExecutor.get();
    if (CheckZeroShape(params, workspaceSize) != ACLNN_SUCCESS) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    const bool useChunk32 =
        params.chunkSize == CHUNK_GATED_DELTA_RULE_FWD_CHUNK_32;
    const aclnnStatus paramStatus =
        useChunk32 ? CheckChunk32Params(params) : CheckParams(params);
    if (!(paramStatus == ACLNN_SUCCESS)) {
        return paramStatus;
    }

    if (!(MakeContiguous(params.q, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.k, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.v, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.g, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.beta, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.initialStateOptional, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (useChunk32) {
        // The kernel consumes head-major q/k/v and [B,Hv,T] g/beta, and writes
        // BSND o directly. Only the inputs are transposed here.
        const bool sequenceMajor = std::strcmp(params.layout, "BSND") == 0 ||
                                   std::strcmp(params.layout, "TND") == 0;
        const int64_t batch = Dim(params.q, 0);
        const int64_t hv = sequenceMajor ? Dim(params.v, 2) : Dim(params.v, 1);
        const int64_t kDim = Dim(params.q, 3);
        const int64_t vDim = Dim(params.v, 3);
        const aclTensor *qHead = sequenceMajor ? TransposeContiguous(params.q, {0, 2, 1, 3}, executorPtr)
                                               : params.q;
        const aclTensor *kHead = sequenceMajor ? TransposeContiguous(params.k, {0, 2, 1, 3}, executorPtr)
                                               : params.k;
        const aclTensor *vHead = sequenceMajor ? TransposeContiguous(params.v, {0, 2, 1, 3}, executorPtr)
                                               : params.v;
        const aclTensor *gBht = TransposeContiguous(params.g, {0, 2, 1}, executorPtr);
        const aclTensor *betaBht = TransposeContiguous(params.beta, {0, 2, 1}, executorPtr);
        const int64_t seqNum = SeqNum(params, batch);
        const DataType stateDtype =
            params.initialStateOptional == nullptr
                ? DataType::DT_FLOAT
                : params.initialStateOptional->GetDataType();
        const int64_t stateDim0 = params.stateVFirst ? vDim : kDim;
        const int64_t stateDim1 = params.stateVFirst ? kDim : vDim;
        const aclTensor *finalState =
            params.finalStateOutOptional == nullptr
                ? executorPtr->AllocTensor(
                      MakeShape({seqNum, hv, stateDim0, stateDim1}),
                      stateDtype, Format::FORMAT_ND)
                : params.finalStateOutOptional;
        const int64_t seqlen = sequenceMajor ? Dim(params.q, 1) : Dim(params.q, 2);
        const bool exportGA = params.gCumsumOutOptional != nullptr;
        const bool exportBeta = params.betaEffOutOptional != nullptr;
        const bool exportH = params.hOutOptional != nullptr;
        const aclTensor *betaArg = exportBeta
            ? executorPtr->AllocTensor(MakeShape({batch, hv, seqlen}), DataType::DT_FLOAT, Format::FORMAT_ND)
            : executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
        const aclTensor *gCumsumArg = exportGA
            ? executorPtr->AllocTensor(MakeShape({batch, hv, seqlen}), DataType::DT_FLOAT, Format::FORMAT_ND)
            : executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
        const aclTensor *aArg = exportGA
            ? params.aOutOptional
            : executorPtr->AllocTensor(MakeShape({1}), DataType::DT_BF16, Format::FORMAT_ND);
        GDN_STAGE_CHECK(qHead != nullptr && kHead != nullptr && vHead != nullptr &&
                            gBht != nullptr && betaBht != nullptr &&
                            params.oOut != nullptr && finalState != nullptr &&
                            gCumsumArg != nullptr &&
                            aArg != nullptr,
                        169114);

        // The public beta slot is FP32. Chunk32 keeps its BF16 beta in the
        // unused a_storage slot and its FP32 g in raw_g. oOut stays BSND.
        // Hats are head-major inside the kernel. Sequence-major callers get a
        // transpose back into the public tensor after the kernel returns.
        const int64_t hq = sequenceMajor ? Dim(params.q, 2) : Dim(params.q, 1);
        const bool exportL2 = params.qHatOutOptional != nullptr;
        const aclTensor *qHatHead = nullptr;
        const aclTensor *kHatHead = nullptr;
        const aclTensor *qRstdArg = params.qRstdOutOptional;
        const aclTensor *kRstdArg = params.kRstdOutOptional;
        if (exportL2 && sequenceMajor) {
            qHatHead = executorPtr->AllocTensor(MakeShape({batch, hq, seqlen, kDim}),
                                                DataType::DT_BF16, Format::FORMAT_ND);
            kHatHead = executorPtr->AllocTensor(MakeShape({batch, hq, seqlen, kDim}),
                                                DataType::DT_BF16, Format::FORMAT_ND);
        } else if (exportL2) {
            qHatHead = params.qHatOutOptional;
            kHatHead = params.kHatOutOptional;
        } else {
            // Keep the kernel argument list fixed. A missing optional output is
            // dropped by the launcher and shifts workspace/tiling.
            qHatHead = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_BF16, Format::FORMAT_ND);
            kHatHead = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_BF16, Format::FORMAT_ND);
            qRstdArg = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
            kRstdArg = executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND);
        }
        auto chunk32Result = l0op::ChunkGatedDeltaRuleFwd(
            qHead, kHead, vHead, gBht, betaBht, gBht, nullptr,
            params.initialStateOptional, params.cuSeqlensOptional,
            params.chunkIndicesOptional, true,
            CHUNK_GATED_DELTA_RULE_FWD_CHUNK_32, params.scale, exportGA,
            params.oOut, finalState, gCumsumArg, aArg,
            0, 0, 0, params.stateVFirst, params.allowNegEigval,
            qHatHead, kHatHead, qRstdArg, kRstdArg, exportL2,
            exportH ? params.hOutOptional
                    : executorPtr->AllocTensor(MakeShape({1}), DataType::DT_BF16, Format::FORMAT_ND),
            exportH, betaArg, exportBeta, executorPtr);
        if (exportL2 && sequenceMajor) {
            const aclTensor *qHatBsnd = TransposeContiguous(qHatHead, {0, 2, 1, 3}, executorPtr);
            const aclTensor *kHatBsnd = TransposeContiguous(kHatHead, {0, 2, 1, 3}, executorPtr);
            if (!(ViewCopyIfPresent(qHatBsnd, params.qHatOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(qHatBsnd, params.qHatOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
            if (!(ViewCopyIfPresent(kHatBsnd, params.kHatOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(kHatBsnd, params.kHatOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        if (exportGA) {
            const aclTensor *gPublic = TransposeContiguous(gCumsumArg, {0, 2, 1}, executorPtr);
            if (!(ViewCopyIfPresent(gPublic, params.gCumsumOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(gPublic, params.gCumsumOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        if (exportBeta) {
            const aclTensor *betaPublic = TransposeContiguous(betaArg, {0, 2, 1}, executorPtr);
            if (!(ViewCopyIfPresent(betaPublic, params.betaEffOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(betaPublic, params.betaEffOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        GDN_STAGE_CHECK(chunk32Result[0] != nullptr &&
                            chunk32Result[1] != nullptr,
                        169115);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    GdnShapeInfo info;
    if (!(ResolveShapeInfo(params, info) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t batch = info.batch;
    const int64_t hq = info.hq;
    const int64_t hv = info.hv;
    const int64_t seqlen = info.seqlen;
    const int64_t kDim = info.kDim;
    const int64_t vDim = info.vDim;
    const int64_t seqNum = SeqNum(params, batch);
    const bool outputFinalState = params.finalStateOutOptional != nullptr;
    const bool usePreparedCumsum = UsePreparedCumsum(params, info);

    const aclTensor *betaUsed = params.beta;
    const aclTensor *gUsed = params.g;
    // The legacy Phase6 kernels consume FP32 gate/beta. Keep A5's native
    // low-precision prepare contract while normalizing the legacy inputs.
    if (!IsAscend950() || params.beta->GetDataType() == DataType::DT_FLOAT ||
        params.g->GetDataType() == DataType::DT_FLOAT) {
        betaUsed = params.beta->GetDataType() == DataType::DT_FLOAT
                       ? params.beta : l0op::Cast(params.beta, DataType::DT_FLOAT, executorPtr);
        gUsed = params.g->GetDataType() == DataType::DT_FLOAT
                    ? params.g : l0op::Cast(params.g, DataType::DT_FLOAT, executorPtr);
    }
    const aclTensor *gRaw = usePreparedCumsum
                               ? gUsed
                               : (gUsed == nullptr ? nullptr
                                  : TransposeContiguous(gUsed, {0, 2, 1}, executorPtr));
    const aclTensor *betaBht = betaUsed == nullptr ? nullptr
                                 : TransposeContiguous(betaUsed, {0, 2, 1}, executorPtr);
    GDN_STAGE_CHECK(gRaw != nullptr && betaBht != nullptr, 169102);

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
        const aclTensor *finalState = params.finalStateOutOptional == nullptr || IsDav2201Device()
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
        if (IsDav2201Device()) {
            qHead = A2PrepareInputView(qHead, executorPtr);
            kHead = A2PrepareInputView(kHead, executorPtr);
            vHead = A2PrepareInputView(vHead, executorPtr);
        }
        GDN_STAGE_CHECK(qHead != nullptr && kHead != nullptr && vHead != nullptr, 169108);
        if (info.isSequenceMajor) {
            qHead = TransposeContiguous(qHead, {0, 2, 1, 3}, executorPtr);
            kHead = TransposeContiguous(kHead, {0, 2, 1, 3}, executorPtr);
            vHead = TransposeContiguous(vHead, {0, 2, 1, 3}, executorPtr);
        }
        GDN_STAGE_CHECK(qHead != nullptr && kHead != nullptr && vHead != nullptr, 169108);
        const aclTensor *qCompute = params.useQkL2norm ? qHat : qHead;
        const aclTensor *kCompute = params.useQkL2norm ? kHat : kHead;
        const aclTensor *aLog = params.aLogOptional;
        const aclTensor *dtBias = params.dtBiasOptional;
        const aclTensor *initialState = params.initialStateOptional;
        if (IsDav2201Device()) {
            if (!(MakeContiguous(aLog, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
            if (!(MakeContiguous(dtBias, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
            if (aLog != nullptr && aLog->GetDataType() != DataType::DT_FLOAT) {
                aLog = l0op::Cast(aLog, DataType::DT_FLOAT, executorPtr);
            }
            if (dtBias != nullptr && dtBias->GetDataType() != DataType::DT_FLOAT) {
                dtBias = l0op::Cast(dtBias, DataType::DT_FLOAT, executorPtr);
            }
            aLog = A2PrepareInputView(aLog, executorPtr);
            dtBias = A2PrepareInputView(dtBias, executorPtr);
            initialState = A2PrepareInputView(initialState, executorPtr);
            GDN_STAGE_CHECK((params.aLogOptional == nullptr || aLog != nullptr) &&
                                (params.dtBiasOptional == nullptr || dtBias != nullptr) &&
                                (params.initialStateOptional == nullptr || initialState != nullptr),
                            169109);
        }

        auto prepareResult = l0op::ChunkGatedDeltaRuleFwdPrepare(
            qHead, kHead, vHead, gRaw, betaBht, aLog, dtBias,
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
            kCompute, w, u, gCumsumBht, nullptr, initialState,
            params.cuSeqlensOptional, params.chunkIndicesOptional, outputFinalState,
            params.chunkSize, true, params.useExp2, params.stateVFirst, h, vNew, finalState, executorPtr);
        GDN_STAGE_CHECK(hResult[0] != nullptr && hResult[1] != nullptr, 169105);
        if (IsDav2201Device()) {
            if (!(ViewCopyIfPresent(finalState, params.finalStateOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(finalState, params.finalStateOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }

        const bool a2Prepare = IsDav2201Device();
        const aclTensor *hForO = h;
        const aclTensor *oCompute = params.oOut;
        if (a2Prepare) {
            // A2 ChunkFwdO consumes K,V state and writes head-major output.
            if (params.stateVFirst) {
                hForO = TransposeContiguous(h, {0, 1, 2, 4, 3}, executorPtr);
            }
            oCompute = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
            GDN_STAGE_CHECK(hForO != nullptr && oCompute != nullptr, 169110);
        }
        auto oResult = l0op::ChunkFwdO(
            qCompute, kCompute, vNew, hForO, gCumsumBht, params.cuSeqlensOptional,
            params.chunkIndicesOptional, params.scale, params.chunkSize, params.useExp2,
            a2Prepare ? false : params.stateVFirst, a2Prepare ? "BNSD" : "BSND", oCompute, executorPtr);
        GDN_STAGE_CHECK(oResult[0] != nullptr, 169106);
        if (a2Prepare) {
            const aclTensor *oExport = TransposeContiguous(oCompute, {0, 2, 1, 3}, executorPtr);
            GDN_STAGE_CHECK(oExport != nullptr, 169111);
            if (!(ViewCopyIfPresent(oExport, params.oOut, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(oExport, params.oOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }

        if (params.qHatOutOptional != nullptr) {
            const aclTensor *qHatExport = qHat;
            if (info.isSequenceMajor) {
                qHatExport = TransposeContiguous(qHatExport, {0, 2, 1, 3}, executorPtr);
            }
            if (!(ViewCopyIfPresent(qHatExport, params.qHatOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(qHatExport, params.qHatOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        if (params.kHatOutOptional != nullptr) {
            const aclTensor *kHatExport = kHat;
            if (info.isSequenceMajor) {
                kHatExport = TransposeContiguous(kHatExport, {0, 2, 1, 3}, executorPtr);
            }
            if (!(ViewCopyIfPresent(kHatExport, params.kHatOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(kHatExport, params.kHatOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        if (params.qRstdOutOptional != nullptr) {
            if (!(ViewCopyIfPresent(qRstd, params.qRstdOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(qRstd, params.qRstdOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        if (params.kRstdOutOptional != nullptr) {
            if (!(ViewCopyIfPresent(kRstd, params.kRstdOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(kRstd, params.kRstdOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        const aclTensor *aExport = a;
        const aclTensor *hExport = h;
        if (!(ViewCopyIfPresent(aExport, params.aOutOptional, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(aExport, params.aOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (!(ViewCopyIfPresent(hExport, params.hOutOptional, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(hExport, params.hOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (params.gCumsumOutOptional != nullptr) {
            const aclTensor *gCumsumBth = TransposeContiguous(gCumsumBht, {0, 2, 1}, executorPtr);
            const aclTensor *gCumsumExport = gCumsumBth;
            if (!(ViewCopyIfPresent(gCumsumExport, params.gCumsumOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(gCumsumExport, params.gCumsumOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        if (params.betaEffOutOptional != nullptr) {
            const aclTensor *betaEffBth = TransposeContiguous(betaEffBht, {0, 2, 1}, executorPtr);
            const aclTensor *betaEffExport = betaEffBth;
            if (!(ViewCopyIfPresent(betaEffExport, params.betaEffOutOptional, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopyIfPresent(betaEffExport, params.betaEffOutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
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

    const bool sequenceMajorOutput = IsDav2201Device();
    const aclTensor *oCompute = executorPtr->AllocTensor(
        sequenceMajorOutput ? MakeShape({batch, seqlen, hv, vDim}) : MakeShape({batch, hv, seqlen, vDim}),
        params.q->GetDataType(), Format::FORMAT_ND);
    GDN_STAGE_CHECK(oCompute != nullptr, 169109);
    const bool nativeQkv = IsNativeQkvLayout(params);
    const aclTensor *qInput = nativeQkv ? DenseQkvView(params.q, executorPtr) : params.q;
    const aclTensor *kInput = nativeQkv ? DenseQkvView(params.k, executorPtr) : params.k;
    const aclTensor *vInput = nativeQkv ? DenseQkvView(params.v, executorPtr) : params.v;
    GDN_STAGE_CHECK(qInput != nullptr && kInput != nullptr && vInput != nullptr, 169113);
    auto phase6Result = l0op::ChunkGatedDeltaRuleFwd(
        qInput, kInput, vInput, betaBht, aStorageBhtc, gRaw, nullptr,
        params.initialStateOptional, params.cuSeqlensOptional, params.chunkIndicesOptional,
        outputFinalState, params.chunkSize, params.scale, params.gCumsumOutOptional != nullptr,
        oCompute, finalState,
        gCumsumCompute, aCompute, usePreparedCumsum ? 1 : 0, nativeQkv ? 1 : 0,
        sequenceMajorOutput ? 1 : 0, false, false,
        executorPtr->AllocTensor(MakeShape({1}), DataType::DT_BF16, Format::FORMAT_ND),
        executorPtr->AllocTensor(MakeShape({1}), DataType::DT_BF16, Format::FORMAT_ND),
        executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND),
        executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND),
        false,
        executorPtr->AllocTensor(MakeShape({1}), DataType::DT_BF16, Format::FORMAT_ND),
        false,
        executorPtr->AllocTensor(MakeShape({1}), DataType::DT_FLOAT, Format::FORMAT_ND),
        false, executorPtr);
    GDN_STAGE_CHECK(phase6Result[0] != nullptr && phase6Result[2] != nullptr &&
                        phase6Result[3] != nullptr,
                        169112);
    // ViewCopy binds a dense output or materializes the caller's strided view.
    const aclTensor *oSequence = sequenceMajorOutput ? oCompute :
        TransposeContiguous(oCompute, {0, 2, 1, 3}, executorPtr);
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
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkGatedDeltaRuleFwd launch failed.", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
