/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
// Keep opdev's logging convention for CHECK_COND and CHECK_RET.
#include "op_common/log/log.h"
#include "aclnn_chunk_gated_delta_rule_bwd.h"

#ifndef GDN_BWD_ENABLE_ARCH22
#define GDN_BWD_ENABLE_ARCH22 0
#endif

#include "../../../chunk_gdn_bwd_intra/op_host/op_api/chunk_gdn_bwd_intra.h"
#include "../../../chunk_gated_delta_rule_bwd_dhu/op_host/op_api/chunk_gated_delta_rule_bwd_dhu.h"
#include "../../../chunk_gated_delta_rule_bwd_finalize/op_host/op_api/chunk_gated_delta_rule_bwd_finalize.h"
#include "../../../../chunk_gdn_fwd/chunk_fwd_h/op_host/op_api/chunk_fwd_h.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"

#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>

#if GDN_BWD_ENABLE_ARCH22
#include "external/aclnn_kernels/aclnn_platform.h"
#endif

using namespace op;

namespace {

constexpr const char *OP_NAME = "aclnnChunkGatedDeltaRuleBwd";
constexpr int64_t DIM_128 = 128;
constexpr int64_t CHUNK_64 = 64;

enum class Layout { BNSD, BSND, NTD, TND };

struct ShapeInfo {
    Layout layout = Layout::BNSD;
    bool sequenceMajor = false;
    int64_t batch = 0;
    int64_t hk = 0;
    int64_t hv = 0;
    int64_t tokens = 0;
    int64_t keyDim = 0;
    int64_t valueDim = 0;
};

struct Params {
    const aclTensor *q;
    const aclTensor *k;
    const aclTensor *v;
    const aclTensor *g;
    const aclTensor *beta;
    const aclTensor *a;
    const aclTensor *dO;
    const aclTensor *initialState;
    const aclTensor *dht;
    const aclTensor *qRstd;
    const aclTensor *kRstd;
    const aclTensor *betaRaw;
    const aclTensor *aLog;
    const aclTensor *dtBias;
    const aclIntArray *cuSeqlens;
    const aclIntArray *chunkIndices;
    const char *layout;
    double scale;
    int64_t chunkSize;
    bool useExp2;
    bool useGateInKernel;
    bool useQkL2normInKernel;
    bool useBetaSigmoidInKernel;
    bool stateVFirst;
    const aclTensor *dqOut;
    const aclTensor *dkOut;
    const aclTensor *dvOut;
    const aclTensor *dBetaOut;
    const aclTensor *dGOut;
    const aclTensor *dh0Out;
    const aclTensor *dALogOut;
    const aclTensor *dDtBiasOut;
};

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> dims)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != dims.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : dims) {
        if (tensor->GetViewShape().GetDim(index++) != dim) {
            return false;
        }
    }
    return true;
}

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    const auto &lhsShape = lhs->GetViewShape();
    const auto &rhsShape = rhs->GetViewShape();
    if (lhsShape.GetDimNum() != rhsShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhsShape.GetDimNum(); ++i) {
        if (lhsShape.GetDim(i) != rhsShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool IsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}

#if GDN_BWD_ENABLE_ARCH22
bool IsDav2201Device()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201;
}
#endif

const aclIntArray *MakePerm(std::initializer_list<int64_t> dims, aclOpExecutor *executor)
{
    return executor->AllocIntArray(dims.begin(), dims.size());
}

const aclTensor *TransposeContiguous(
    const aclTensor *tensor, std::initializer_list<int64_t> dims, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    const aclIntArray *perm = MakePerm(dims, executor);
    if (perm == nullptr) {
        return nullptr;
    }
    const aclTensor *transposed = l0op::Transpose(tensor, perm, executor);
    if (transposed == nullptr) {
        return nullptr;
    }
    const aclTensor *contiguous = l0op::Contiguous(transposed, executor);
    if (contiguous == nullptr) {
        return nullptr;
    }
    const aclTensor *reshaped = l0op::Reshape(contiguous, transposed->GetViewShape(), executor);
    if (reshaped != nullptr) {
        reshaped->SetStorageShape(reshaped->GetViewShape());
        reshaped->SetOriginalShape(reshaped->GetViewShape());
        auto *mutableTensor = const_cast<aclTensor *>(reshaped);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    }
    return reshaped;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_COND(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckRequiredInputs(const Params &params)
{
    const std::array<const aclTensor *, 12> required = {
        params.q, params.k, params.v, params.g, params.beta, params.a,
        params.dO, params.dqOut, params.dkOut, params.dvOut,
        params.dBetaOut, params.dGOut};
    const char *names[] = {"q", "k", "v", "g", "beta", "A", "d_o", "dq", "dk", "dv", "d_beta", "d_g"};
    for (size_t index = 0; index < required.size(); ++index) {
        if (required[index] == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, names[index]);
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckZeroShape(const Params &params, uint64_t *workspaceSize)
{
    if (params.q->IsEmpty() || params.k->IsEmpty() || params.v->IsEmpty() ||
        params.g->IsEmpty() || params.beta->IsEmpty() || params.a->IsEmpty() ||
        params.dO->IsEmpty()) {
        *workspaceSize = 0UL;
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ViewCopy(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    CHECK_COND(src != nullptr && dst != nullptr, ACLNN_ERR_INNER_NULLPTR,
               "ViewCopy source and destination must not be nullptr.");
    CHECK_COND(l0op::ViewCopy(src, dst, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR,
               "ViewCopy failed.");
    return ACLNN_SUCCESS;
}

int64_t ChunkCount(const Params &params, int64_t tokens);
int64_t SequenceCount(const Params &params, int64_t batch);

int64_t ChunkCount(const Params &params, int64_t tokens)
{
    if (params.chunkIndices != nullptr) {
        return static_cast<int64_t>(params.chunkIndices->Size() / 2);
    }
    return (tokens + params.chunkSize - 1) / params.chunkSize;
}

int64_t SequenceCount(const Params &params, int64_t batch)
{
    return params.cuSeqlens == nullptr
               ? batch
               : static_cast<int64_t>(params.cuSeqlens->Size() - 1);
}

aclnnStatus CheckMetadata(const Params &params, int64_t tokens)
{
    const bool varied = params.cuSeqlens != nullptr;
    if (!(varied == (params.chunkIndices != nullptr))) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "cu_seqlens/chunk_indices",
                                  std::string("cu_seqlens=") + (varied ? "present" : "absent") +
                                      ", chunk_indices=" + (params.chunkIndices != nullptr ? "present" : "absent"),
                                  "both present or both absent");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!varied) {
        return ACLNN_SUCCESS;
    }
    if (!(params.cuSeqlens->Size() >= 2 && (*params.cuSeqlens)[0] == 0)) {
        OP_LOGE_FOR_INVALID_VALUE(
            OP_NAME, "cu_seqlens",
            std::string("size=") + std::to_string(params.cuSeqlens->Size()) +
                ", first=" + (params.cuSeqlens->Size() == 0 ? "absent" : std::to_string((*params.cuSeqlens)[0])),
            "at least two entries, starting at 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!((*params.cuSeqlens)[params.cuSeqlens->Size() - 1] == tokens)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "cu_seqlens[-1]",
                                  std::to_string((*params.cuSeqlens)[params.cuSeqlens->Size() - 1]),
                                  std::to_string(tokens));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
        const int64_t begin = (*params.cuSeqlens)[seq];
        const int64_t end = (*params.cuSeqlens)[seq + 1];
        if (end < begin) {
            OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "cu_seqlens adjacent entries",
                                      std::string("index=") + std::to_string(seq) + ", begin=" + std::to_string(begin) +
                                          ", end=" + std::to_string(end),
                                      "end >= begin");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t chunks = (end - begin + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if (!(offset + 1 < params.chunkIndices->Size() &&
                  (*params.chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                  (*params.chunkIndices)[offset + 1] == chunk)) {
                OP_LOGE_FOR_INVALID_VALUE(
                    OP_NAME, "chunk_indices",
                    std::string("size=") + std::to_string(params.chunkIndices->Size()) + ", pair=" +
                        (offset + 1 < params.chunkIndices->Size() ?
                             "[" + std::to_string((*params.chunkIndices)[offset]) + "," +
                                 std::to_string((*params.chunkIndices)[offset + 1]) + "]" :
                             "absent or incomplete"),
                    std::string("canonical sequence-major pairs; pair at offset ") + std::to_string(offset) +
                        " must be [" + std::to_string(seq) + "," + std::to_string(chunk) + "]");
                return ACLNN_ERR_PARAM_INVALID;
            }
            offset += 2;
        }
    }
    if (offset != params.chunkIndices->Size()) {
        OP_LOGE_FOR_INVALID_LISTSIZE(OP_NAME, "chunk_indices", std::to_string(params.chunkIndices->Size()),
                                     std::to_string(offset));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShapeInfo(const Params &params, ShapeInfo &info)
{
    if (params.layout == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "layout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (std::strcmp(params.layout, "BNSD") == 0) {
        info.layout = Layout::BNSD;
    } else if (std::strcmp(params.layout, "BSND") == 0) {
        info.layout = Layout::BSND;
    } else if (std::strcmp(params.layout, "NTD") == 0) {
        info.layout = Layout::NTD;
    } else if (std::strcmp(params.layout, "TND") == 0) {
        info.layout = Layout::TND;
    } else {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "layout", params.layout, "BNSD, BSND, NTD or TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    info.sequenceMajor = info.layout == Layout::BSND || info.layout == Layout::TND;
    if (!(params.q->GetViewShape().GetDimNum() == 4 && params.v->GetViewShape().GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            OP_NAME, "q/v",
            std::string("q.rank=") + std::to_string(params.q->GetViewShape().GetDimNum()) +
                ", v.rank=" + std::to_string(params.v->GetViewShape().GetDimNum()),
            "q and v must both have rank 4 for BNSD/BSND/NTD/TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.sequenceMajor) {
        info.batch = params.q->GetViewShape().GetDim(0);
        info.tokens = params.q->GetViewShape().GetDim(1);
        info.hk = params.q->GetViewShape().GetDim(2);
        info.keyDim = params.q->GetViewShape().GetDim(3);
        info.hv = params.v->GetViewShape().GetDim(2);
        info.valueDim = params.v->GetViewShape().GetDim(3);
    } else {
        info.batch = params.q->GetViewShape().GetDim(0);
        info.hk = params.q->GetViewShape().GetDim(1);
        info.tokens = params.q->GetViewShape().GetDim(2);
        info.keyDim = params.q->GetViewShape().GetDim(3);
        info.hv = params.v->GetViewShape().GetDim(1);
        info.valueDim = params.v->GetViewShape().GetDim(3);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckLayoutShapes(const Params &params, const ShapeInfo &info)
{
    if (!(SameShape(params.q, params.k) && SameShape(params.q, params.dqOut) && SameShape(params.k, params.dkOut) &&
          SameShape(params.v, params.dvOut))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(OP_NAME, "q/k/v/dq/dk/dv",
                                               std::string("q=") + op::ToString(params.q->GetViewShape()).GetString() +
                                                   ", k=" + op::ToString(params.k->GetViewShape()).GetString() +
                                                   ", v=" + op::ToString(params.v->GetViewShape()).GetString() +
                                                   ", dq=" + op::ToString(params.dqOut->GetViewShape()).GetString() +
                                                   ", dk=" + op::ToString(params.dkOut->GetViewShape()).GetString() +
                                                   ", dv=" + op::ToString(params.dvOut->GetViewShape()).GetString(),
                                               "k and dq must match q; dk must match k; dv must match v");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool validVShape = info.sequenceMajor
                                 ? HasShape(params.v, {info.batch, info.tokens, info.hv, info.valueDim})
                                 : HasShape(params.v, {info.batch, info.hv, info.tokens, info.valueDim});
    if (!(validVShape)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            OP_NAME, "v", params.v == nullptr ? "nullptr" : op::ToString(params.v->GetViewShape()).GetString(),
            op::ToString(info.sequenceMajor ? MakeShape({info.batch, info.tokens, info.hv, info.valueDim}) :
                                              MakeShape({info.batch, info.hv, info.tokens, info.valueDim}))
                .GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HasShape(params.g, {info.batch, info.tokens, info.hv}) &&
          HasShape(params.beta, {info.batch, info.tokens, info.hv}))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "g/beta",
            std::string("g=") + op::ToString(params.g->GetViewShape()).GetString() +
                ", beta=" + op::ToString(params.beta->GetViewShape()).GetString(),
            std::string("both must equal BSN ") +
                op::ToString(MakeShape({info.batch, info.tokens, info.hv})).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HasShape(params.dO, {info.batch, info.tokens, info.hv, info.valueDim}))) {
        OP_LOGE_FOR_INVALID_SHAPE(
            OP_NAME, "d_o", params.dO == nullptr ? "nullptr" : op::ToString(params.dO->GetViewShape()).GetString(),
            op::ToString(MakeShape({info.batch, info.tokens, info.hv, info.valueDim})).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HasShape(params.a, {info.batch, info.hv, info.tokens, params.chunkSize}))) {
        OP_LOGE_FOR_INVALID_SHAPE(
            OP_NAME, "A", params.a == nullptr ? "nullptr" : op::ToString(params.a->GetViewShape()).GetString(),
            op::ToString(MakeShape({info.batch, info.hv, info.tokens, params.chunkSize})).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HasShape(params.dBetaOut, {info.batch, info.tokens, info.hv}) &&
          HasShape(params.dGOut, {info.batch, info.tokens, info.hv}))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "d_beta/d_g",
            std::string("d_beta=") + op::ToString(params.dBetaOut->GetViewShape()).GetString() +
                ", d_g=" + op::ToString(params.dGOut->GetViewShape()).GetString(),
            std::string("both must equal BSN ") +
                op::ToString(MakeShape({info.batch, info.tokens, info.hv})).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.qRstd != nullptr || params.kRstd != nullptr) {
        if (!(HasShape(params.qRstd, {info.batch, info.hk, info.tokens}) &&
              HasShape(params.kRstd, {info.batch, info.hk, info.tokens}))) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "q_rstd/k_rstd",
                std::string("q_rstd=") +
                    (params.qRstd == nullptr ? "nullptr" : op::ToString(params.qRstd->GetViewShape()).GetString()) +
                    ", k_rstd=" +
                    (params.kRstd == nullptr ? "nullptr" : op::ToString(params.kRstd->GetViewShape()).GetString()),
                std::string("both must equal BNS ") +
                    op::ToString(MakeShape({info.batch, info.hk, info.tokens})).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.betaRaw != nullptr) {
        if (!(HasShape(params.betaRaw, {info.batch, info.tokens, info.hv}))) {
            OP_LOGE_FOR_INVALID_SHAPE(
                OP_NAME, "beta_raw",
                params.betaRaw == nullptr ? "nullptr" : op::ToString(params.betaRaw->GetViewShape()).GetString(),
                op::ToString(MakeShape({info.batch, info.tokens, info.hv})).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t sequences = SequenceCount(params, info.batch);
    const int64_t stateDim0 = params.stateVFirst ? info.valueDim : info.keyDim;
    const int64_t stateDim1 = params.stateVFirst ? info.keyDim : info.valueDim;
    if (params.initialState != nullptr) {
        if (!(HasShape(params.initialState, {sequences, info.hv, stateDim0, stateDim1}) &&
              HasShape(params.dh0Out, {sequences, info.hv, stateDim0, stateDim1}))) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "initial_state/dh0",
                std::string("initial_state=") + op::ToString(params.initialState->GetViewShape()).GetString() +
                    ", dh0=" +
                    (params.dh0Out == nullptr ? "nullptr" : op::ToString(params.dh0Out->GetViewShape()).GetString()),
                std::string("both must match state_v_first: ") +
                    op::ToString(MakeShape({sequences, info.hv, stateDim0, stateDim1})).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dht != nullptr) {
        if (!(HasShape(params.dht, {sequences, info.hv, stateDim0, stateDim1}))) {
            OP_LOGE_FOR_INVALID_SHAPE(OP_NAME, "dht",
                                      params.dht == nullptr ? "nullptr" :
                                                              op::ToString(params.dht->GetViewShape()).GetString(),
                                      op::ToString(MakeShape({sequences, info.hv, stateDim0, stateDim1})).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const Params &params, ShapeInfo &info)
{
#if GDN_BWD_ENABLE_ARCH22
    if (!IsDav2201Device()) {
#endif
        if (!(IsAscend950())) {
            OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "SoC", aclrtGetSocName() == nullptr ? "unknown" : aclrtGetSocName(),
                                      "Ascend950 on this build");
            return ACLNN_ERR_PARAM_INVALID;
        }
#if GDN_BWD_ENABLE_ARCH22
    }
#endif
    if (params.useGateInKernel) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "use_gate_in_kernel", params.useGateInKernel ? "true" : "false", "false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.dALogOut == nullptr && params.dDtBiasOut == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "d_a_log/d_dt_bias",
                                  std::string("d_a_log=") + (params.dALogOut == nullptr ? "nullptr" : "present") +
                                      ", d_dt_bias=" + (params.dDtBiasOut == nullptr ? "nullptr" : "present"),
                                  "both nullptr (reserved outputs)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.chunkSize != CHUNK_64) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "chunk_size", std::to_string(params.chunkSize), "64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(params.scale)) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "scale", std::to_string(params.scale), "a finite number");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(ResolveShapeInfo(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    if (!(info.batch > 0 && info.hk > 0 && info.hv > 0 && info.tokens > 0)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "B/HK/HV/T",
                                  std::string("B=") + std::to_string(info.batch) + ", HK=" + std::to_string(info.hk) +
                                      ", HV=" + std::to_string(info.hv) + ", T=" + std::to_string(info.tokens),
                                  "all > 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.keyDim == DIM_128 && info.valueDim == DIM_128)) {
        OP_LOGE_FOR_INVALID_VALUE(
            OP_NAME, "K/V", std::string("K=") + std::to_string(info.keyDim) + ", V=" + std::to_string(info.valueDim),
            "K=128 and V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.hv % info.hk == 0 && info.hv / info.hk <= 4)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "HV/HK",
                                  std::string("HV=") + std::to_string(info.hv) + ", HK=" + std::to_string(info.hk),
                                  "HV % HK == 0 and HV/HK in [1,4]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.sequenceMajor) {
        // DataCopyExtParams expresses the GM row gap in bytes. The same
        // bound protects public input gathers and gradient output scatters.
        constexpr uint64_t ROW_BYTES = DIM_128 * sizeof(uint16_t);
        constexpr uint64_t MAX_HEADS =
            std::numeric_limits<uint32_t>::max() / ROW_BYTES + 1;
        const std::string expectedHeads = "<= " + std::to_string(MAX_HEADS) +
            " for the uint32_t GM row gap in layout=" + params.layout;
        if (static_cast<uint64_t>(info.hk) > MAX_HEADS) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkGatedDeltaRuleBwd", "q/k/dq/dk.HK",
                                      std::to_string(info.hk), expectedHeads);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (static_cast<uint64_t>(info.hv) > MAX_HEADS) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkGatedDeltaRuleBwd", "v/dv.HV",
                                      std::to_string(info.hv), expectedHeads);
            return ACLNN_ERR_PARAM_INVALID;
        }
        uint64_t remaining = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        for (uint64_t factor : {static_cast<uint64_t>(info.batch),
                                static_cast<uint64_t>(info.tokens),
                                static_cast<uint64_t>(info.hv), ROW_BYTES}) {
            if (factor > remaining) {
                const std::string actual = "B=" + std::to_string(info.batch) +
                    ", T=" + std::to_string(info.tokens) + ", HV=" + std::to_string(info.hv);
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnChunkGatedDeltaRuleBwd",
                    "v/dv byte address", actual,
                    "layout=" + std::string(params.layout) + " requires B*T*HV*128*2 <= INT64_MAX");
                return ACLNN_ERR_PARAM_INVALID;
            }
            remaining /= factor;
        }
    }
    CHECK_RET(CheckLayoutShapes(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckMetadata(params, info.tokens) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (!(params.cuSeqlens == nullptr || info.batch == 1)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "q.B", std::to_string(info.batch),
                                  "1 when cu_seqlens/chunk_indices are present");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const DataType dtype = params.q->GetDataType();
    if (dtype != DataType::DT_BF16) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, "q", op::ToString(params.q->GetDataType()).GetString(), "BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.k->GetDataType() == dtype && params.v->GetDataType() == dtype && params.a->GetDataType() == dtype &&
          params.dO->GetDataType() == dtype && params.dqOut->GetDataType() == dtype &&
          params.dkOut->GetDataType() == dtype && params.dvOut->GetDataType() == dtype)) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(
            OP_NAME, "k/v/A/d_o/dq/dk/dv",
            std::string("k=") + op::ToString(params.k->GetDataType()).GetString() + ", " + std::string("v=") +
                op::ToString(params.v->GetDataType()).GetString() + ", " + std::string("A=") +
                op::ToString(params.a->GetDataType()).GetString() + ", " + std::string("d_o=") +
                op::ToString(params.dO->GetDataType()).GetString() + ", " + std::string("dq=") +
                op::ToString(params.dqOut->GetDataType()).GetString() + ", " + std::string("dk=") +
                op::ToString(params.dkOut->GetDataType()).GetString() + ", " + std::string("dv=") +
                op::ToString(params.dvOut->GetDataType()).GetString(),
            "BF16 for each tensor");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool validGType = params.g->GetDataType() == DataType::DT_BF16 ||
                            params.g->GetDataType() == DataType::DT_FLOAT;
    const bool validBetaType = params.beta->GetDataType() == DataType::DT_BF16 ||
                               params.beta->GetDataType() == DataType::DT_FLOAT;
    if (!(validGType && validBetaType && params.dGOut->GetDataType() == params.g->GetDataType() &&
          params.dBetaOut->GetDataType() == params.beta->GetDataType())) {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(
            OP_NAME, "g/beta/d_g/d_beta",
            std::string("g=") + op::ToString(params.g->GetDataType()).GetString() + ", " + std::string("beta=") +
                op::ToString(params.beta->GetDataType()).GetString() + ", " + std::string("d_g=") +
                op::ToString(params.dGOut->GetDataType()).GetString() + ", " + std::string("d_beta=") +
                op::ToString(params.dBetaOut->GetDataType()).GetString(),
            "g and beta: BF16 or FP32; d_g=g.dtype; d_beta=beta.dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(!params.useQkL2normInKernel || (params.qRstd != nullptr && params.kRstd != nullptr))) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "q_rstd/k_rstd",
                                  std::string("q_rstd=") + (params.qRstd == nullptr ? "nullptr" : "present") +
                                      ", k_rstd=" + (params.kRstd == nullptr ? "nullptr" : "present"),
                                  "both present when use_qk_l2norm_in_kernel=true");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(params.useQkL2normInKernel || (params.qRstd == nullptr && params.kRstd == nullptr))) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "use_qk_l2norm_in_kernel", "false", "true when q_rstd or k_rstd is present");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(!params.useBetaSigmoidInKernel || params.betaRaw != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "beta_raw", "nullptr", "present when use_beta_sigmoid_in_kernel=true");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!(params.useBetaSigmoidInKernel || params.betaRaw == nullptr)) {
        OP_LOGE_WITH_INVALID_ATTR(OP_NAME, "use_beta_sigmoid_in_kernel", "false", "true when beta_raw is present");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (params.qRstd != nullptr) {
        if (!(params.qRstd->GetDataType() == DataType::DT_FLOAT && params.kRstd->GetDataType() == DataType::DT_FLOAT)) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(
                OP_NAME, "q_rstd/k_rstd",
                std::string("q_rstd=") + op::ToString(params.qRstd->GetDataType()).GetString() + ", " +
                    std::string("k_rstd=") + op::ToString(params.kRstd->GetDataType()).GetString(),
                "FP32 for each tensor");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.betaRaw != nullptr) {
        if (params.betaRaw->GetDataType() != params.beta->GetDataType()) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, "beta_raw",
                                             op::ToString(params.betaRaw->GetDataType()).GetString(),
                                             op::ToString(params.beta->GetDataType()).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (params.initialState != nullptr) {
        if (!(params.initialState->GetDataType() == dtype && params.dh0Out->GetDataType() == dtype)) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(
                OP_NAME, "initial_state/dh0",
                std::string("initial_state=") + op::ToString(params.initialState->GetDataType()).GetString() + ", " +
                    std::string("dh0=") + op::ToString(params.dh0Out->GetDataType()).GetString(),
                "BF16 for each tensor");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (params.dh0Out != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "dh0/initial_state", "dh0=present, initial_state=nullptr",
                                      "dh0=nullptr when initial_state is absent");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dht != nullptr) {
        if (params.dht->GetDataType() != dtype) {
            OP_LOGE_WITH_INVALID_INPUT_DTYPE(OP_NAME, "dht", op::ToString(params.dht->GetDataType()).GetString(),
                                             "BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

#if GDN_BWD_ENABLE_ARCH22
aclnnStatus MakeOutputContiguousArch22(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor != nullptr && !IsContiguous(tensor)) {
        tensor = executor->AllocTensor(tensor->GetViewShape(), tensor->GetDataType(), Format::FORMAT_ND);
        CHECK_COND(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR,
                   "allocating A2/A3 contiguous output failed.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CopyOutputArch22(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    if (dst != nullptr && src != dst) {
        return ViewCopy(src, dst, executor);
    }
    return ACLNN_SUCCESS;
}
#endif

} // namespace

extern "C" aclnnStatus aclnnChunkGatedDeltaRuleBwdGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *dO, const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional, const aclTensor *qRstdOptional,
    const aclTensor *kRstdOptional, const aclTensor *betaRawOptional,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    const char *layout, double scale, int64_t chunkSize, bool useExp2,
    bool useGateInKernel, bool useQkL2normInKernel,
    bool useBetaSigmoidInKernel, bool stateVFirst,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dh0OutOptional, const aclTensor *dALogOutOptional,
    const aclTensor *dDtBiasOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnChunkGatedDeltaRuleBwd,
        DFX_IN(q, k, v, g, beta, a, dO, initialStateOptional, dhtOptional,
               qRstdOptional, kRstdOptional, betaRawOptional, aLogOptional,
               dtBiasOptional, cuSeqlensOptional, chunkIndicesOptional,
               layout, scale, chunkSize, useExp2,
               useGateInKernel, useQkL2normInKernel,
               useBetaSigmoidInKernel, stateVFirst),
        DFX_OUT(dqOut, dkOut, dvOut, dBetaOut, dGOut, dh0OutOptional,
                dALogOutOptional, dDtBiasOutOptional));
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    Params params{q, k, v, g, beta, a, dO, initialStateOptional, dhtOptional,
                  qRstdOptional, kRstdOptional, betaRawOptional,
                  aLogOptional, dtBiasOptional,
                  cuSeqlensOptional, chunkIndicesOptional, layout, scale,
                  chunkSize, useExp2, useGateInKernel,
                  useQkL2normInKernel, useBetaSigmoidInKernel, stateVFirst,
                  dqOut, dkOut, dvOut, dBetaOut, dGOut, dh0OutOptional,
                  dALogOutOptional, dDtBiasOutOptional};
    CHECK_RET(CheckRequiredInputs(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();
    if (CheckZeroShape(params, workspaceSize) != ACLNN_SUCCESS) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ShapeInfo info;
    CHECK_RET(CheckParams(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(MakeContiguous(params.q, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.k, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.v, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.beta, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.a, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.dO, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.initialState, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.dht, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.qRstd, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.kRstd, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.betaRaw, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const aclTensor *qHead = params.q;
    const aclTensor *kHead = params.k;
    const aclTensor *vHead = params.v;
    const bool useFp32Gate = params.g->GetDataType() == DataType::DT_FLOAT ||
                             params.beta->GetDataType() == DataType::DT_FLOAT;
    const DataType gateType = useFp32Gate ? DataType::DT_FLOAT : DataType::DT_BF16;
    const aclTensor *initialStateKv = params.initialState;
    const aclTensor *dOHead = TransposeContiguous(params.dO, {0, 2, 1, 3}, executorPtr);
    // Scalar inputs enter as BSN. Compute all gate paths at their common higher precision in BNS.
    const aclTensor *gCompute = params.g->GetDataType() == gateType
                                    ? params.g
                                    : l0op::Cast(params.g, gateType, executorPtr);
    const aclTensor *betaCompute = params.beta->GetDataType() == gateType
                                       ? params.beta
                                       : l0op::Cast(params.beta, gateType, executorPtr);
    const aclTensor *betaRawCompute = params.betaRaw;
    if (betaRawCompute != nullptr && betaRawCompute->GetDataType() != gateType) {
        betaRawCompute = l0op::Cast(betaRawCompute, gateType, executorPtr);
    }

    const aclTensor *gHead = TransposeContiguous(gCompute, {0, 2, 1}, executorPtr);
    const aclTensor *betaHead = TransposeContiguous(betaCompute, {0, 2, 1}, executorPtr);
    const aclTensor *betaRawHead = TransposeContiguous(betaRawCompute, {0, 2, 1}, executorPtr);

    CHECK_COND(qHead != nullptr && kHead != nullptr && vHead != nullptr && dOHead != nullptr &&
                   gHead != nullptr && betaHead != nullptr &&
                   (!params.useBetaSigmoidInKernel || betaRawHead != nullptr),
               ACLNN_ERR_INNER_NULLPTR, "input layout conversion failed.");
    if (params.stateVFirst) {
        initialStateKv = TransposeContiguous(params.initialState, {0, 1, 3, 2}, executorPtr);
        CHECK_COND(params.initialState == nullptr || initialStateKv != nullptr,
                   ACLNN_ERR_INNER_NULLPTR, "state layout conversion failed.");
    }
    const int64_t chunks = ChunkCount(params, info.tokens);
    const DataType dtype = qHead->GetDataType();
    const op::Shape vShape = MakeShape({info.batch, info.hv, info.tokens, info.valueDim});
    const op::Shape gateShape = MakeShape({info.batch, info.hv, info.tokens});
    const op::Shape wShape = MakeShape({info.batch, info.hv, info.tokens, info.keyDim});
    const op::Shape hShape = MakeShape({info.batch, chunks, info.hv, info.keyDim, info.valueDim});
    const op::Shape hChunkShape = MakeShape({info.batch, chunks, info.hv, info.keyDim, info.valueDim});

    const aclTensor *w = executorPtr->AllocTensor(wShape, dtype, Format::FORMAT_ND);
    const aclTensor *u = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
    const aclTensor *dvLocal = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
    const aclTensor *h = executorPtr->AllocTensor(hChunkShape, dtype, Format::FORMAT_ND);
    const aclTensor *vNew = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
    const aclTensor *dh = executorPtr->AllocTensor(hShape, dtype, Format::FORMAT_ND);
    const aclTensor *dv2 = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);

    const aclTensor *dqHead = params.dqOut;
    const aclTensor *dkHead = params.dkOut;
    const aclTensor *dvHead = params.dvOut;
    const aclTensor *dBetaHead = executorPtr->AllocTensor(gateShape, gateType, Format::FORMAT_ND);
    const aclTensor *dGHead = executorPtr->AllocTensor(gateShape, gateType, Format::FORMAT_ND);
    CHECK_COND(w != nullptr && u != nullptr && dvLocal != nullptr && h != nullptr &&
                   vNew != nullptr && dh != nullptr && dv2 != nullptr && dqHead != nullptr &&
                   dkHead != nullptr && dvHead != nullptr && dBetaHead != nullptr && dGHead != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "allocating composite intermediate tensors failed.");

#if GDN_BWD_ENABLE_ARCH22
    if (IsDav2201Device()) {
        CHECK_RET(MakeOutputContiguousArch22(dqHead, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(MakeOutputContiguousArch22(dkHead, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(MakeOutputContiguousArch22(dvHead, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        // params 是本次调用的私有副本；仅 A2/A3 用连续 dh0 暂存。
        CHECK_RET(MakeOutputContiguousArch22(params.dh0Out, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }
#endif
    // A2/A3 直接复用融合算子 ChunkGdnBwdIntra：BF16/FP32 gate、use_exp2 与
    // varlen（空序列无 chunk 记录，天然跳过）均由内核原生处理。
    const auto intraResult = l0op::ChunkGdnBwdIntra(
        qHead, kHead, vHead, gHead, betaHead, params.a, dOHead,
        params.cuSeqlens, params.chunkIndices, params.scale, params.chunkSize,
        params.useExp2, w, u, dvLocal, executorPtr, params.layout);
    CHECK_COND(intraResult[0] != nullptr && intraResult[1] != nullptr &&
                   intraResult[2] != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkGdnBwdIntra composition failed.");

    const auto fwdHResult = l0op::ChunkFwdH(
        kHead, w, u, gHead, nullptr, initialStateKv,
        params.cuSeqlens, params.chunkIndices, false, params.chunkSize, true,
        params.useExp2, false, h, vNew, nullptr, executorPtr, params.layout);
    CHECK_COND(fwdHResult[0] != nullptr && fwdHResult[1] != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkFwdH composition failed.");

    const auto dhuResult = l0op::ChunkGatedDeltaRuleBwdDhu(
        qHead, kHead, w, dOHead, dvLocal, gHead, nullptr,
        params.initialState, params.dht, params.cuSeqlens, params.chunkIndices,
        params.scale, params.chunkSize, params.useExp2, params.stateVFirst, dh, params.dh0Out,
        dv2, executorPtr, params.layout);
    CHECK_COND(dhuResult[0] != nullptr && dhuResult[1] != nullptr &&
                   dhuResult[2] != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkGatedDeltaRuleBwdDhu composition failed.");

    const auto finalizeResult = l0op::ChunkGatedDeltaRuleBwdFinalize(
        qHead, kHead, vHead, vNew, dOHead, dv2, gHead, betaHead, h, dh,
        params.a, params.qRstd, params.kRstd, betaRawHead, params.cuSeqlens,
        params.chunkIndices, params.scale, params.chunkSize,
        params.useQkL2normInKernel, params.useBetaSigmoidInKernel,
        params.useGateInKernel, false, params.useExp2,
        dqHead, dkHead, dvHead, dBetaHead, dGHead, executorPtr, params.layout);
    for (const aclTensor *tensor : finalizeResult) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR,
                   "ChunkGatedDeltaRuleBwdFinalize composition failed.");
    }

#if GDN_BWD_ENABLE_ARCH22
    if (IsDav2201Device()) {
        CHECK_RET(CopyOutputArch22(dqHead, params.dqOut, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CopyOutputArch22(dkHead, params.dkOut, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CopyOutputArch22(dvHead, params.dvOut, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CopyOutputArch22(params.dh0Out, dh0OutOptional, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }
#endif
    const aclTensor *dBetaSequence = TransposeContiguous(dBetaHead, {0, 2, 1}, executorPtr);
    const aclTensor *dGSequence = TransposeContiguous(dGHead, {0, 2, 1}, executorPtr);
    if (dBetaSequence != nullptr && dBetaSequence->GetDataType() != params.dBetaOut->GetDataType()) {
        dBetaSequence = l0op::Cast(dBetaSequence, params.dBetaOut->GetDataType(), executorPtr);
    }
    if (dGSequence != nullptr && dGSequence->GetDataType() != params.dGOut->GetDataType()) {
        dGSequence = l0op::Cast(dGSequence, params.dGOut->GetDataType(), executorPtr);
    }
    CHECK_RET(ViewCopy(dBetaSequence, params.dBetaOut, executorPtr) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(ViewCopy(dGSequence, params.dGOut, executorPtr) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkGatedDeltaRuleBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkGatedDeltaRuleBwd launch failed.");
    return ACLNN_SUCCESS;
}
