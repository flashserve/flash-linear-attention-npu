/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
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

#include <array>
#include <cmath>
#include <cstring>
#include <initializer_list>

#if GDN_BWD_ENABLE_ARCH22
#include "external/aclnn_kernels/aclnn_platform.h"
#endif

#include <string>

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkGatedDeltaRuleBwd";

namespace {

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
    if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Contiguous failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckRequiredInputs(const Params &params)
{
    const std::array<const aclTensor *, 12> required = {
        params.q, params.k, params.v, params.g, params.beta, params.a,
        params.dO, params.dqOut, params.dkOut, params.dvOut,
        params.dBetaOut, params.dGOut};
    for (const aclTensor *tensor : required) {
        if (tensor == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "inputs");
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
    if (!(src != nullptr && dst != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ViewCopy source and destination must not be nullptr.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(l0op::ViewCopy(src, dst, executor) != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ViewCopy failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
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
    if (varied != (params.chunkIndices != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cu_seqlens,chunk_indices",
            std::string(varied ? "cu_seqlens=present" : "cu_seqlens=absent") + "," +
                (params.chunkIndices != nullptr ? "chunk_indices=present" : "chunk_indices=absent"),
            "cu_seqlens and chunk_indices must be provided together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!varied) {
        return ACLNN_SUCCESS;
    }
    if (!(params.cuSeqlens->Size() >= 2 && (*params.cuSeqlens)[0] == 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cu_seqlens[0]", std::to_string(params.cuSeqlens->Size()) + " entries, first=" +
                std::to_string(params.cuSeqlens->Size() > 0 ? (*params.cuSeqlens)[0] : 0),
            "cu_seqlens must have at least 2 entries and start at zero");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*params.cuSeqlens)[params.cuSeqlens->Size() - 1] != tokens) {
        OP_LOGE_FOR_INVALID_VALUE(
            OP_NAME, "cu_seqlens[last]",
            std::to_string((*params.cuSeqlens)[params.cuSeqlens->Size() - 1]),
            std::to_string(tokens));
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
        const int64_t begin = (*params.cuSeqlens)[seq];
        const int64_t end = (*params.cuSeqlens)[seq + 1];
        if (end < begin) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "cu_seqlens[seq],cu_seqlens[seq+1]",
                std::to_string(begin) + "," + std::to_string(end),
                "cu_seqlens must be nondecreasing");
            return ACLNN_ERR_PARAM_INVALID;
        }
        const int64_t chunks = (end - begin + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            if (!(offset + 1 < params.chunkIndices->Size() &&
                  (*params.chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                  (*params.chunkIndices)[offset + 1] == chunk)) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    OP_NAME, "chunk_indices[offset],chunk_indices[offset+1]",
                    std::to_string((*params.chunkIndices)[offset]) + "," +
                        std::to_string((*params.chunkIndices)[offset + 1]),
                    "chunk_indices must use canonical sequence-major order");
                return ACLNN_ERR_PARAM_INVALID;
            }
            offset += 2;
        }
    }
    if (offset != params.chunkIndices->Size()) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            OP_NAME, "chunk_indices", std::to_string(params.chunkIndices->Size()),
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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "layout", params.layout, "layout must be BNSD, BSND, NTD or TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    info.sequenceMajor = info.layout == Layout::BSND || info.layout == Layout::TND;
    if (!(params.q->GetViewShape().GetDimNum() == 4 && params.v->GetViewShape().GetDimNum() == 4)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            OP_NAME, "q,v",
            std::to_string(params.q->GetViewShape().GetDimNum()) + "," +
                std::to_string(params.v->GetViewShape().GetDimNum()),
            "q and v must be rank 4");
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
    if (!(SameShape(params.q, params.k) && SameShape(params.q, params.dqOut) &&
          SameShape(params.k, params.dkOut) && SameShape(params.v, params.dvOut))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q,k,dq,dk,v,dv", "shape mismatch",
            "dq/dk/dv must match q/k/v respectively");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool validVShape = info.sequenceMajor
                                 ? HasShape(params.v, {info.batch, info.tokens, info.hv, info.valueDim})
                                 : HasShape(params.v, {info.batch, info.hv, info.tokens, info.valueDim});
    if (!validVShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "v", std::string(op::ToString(params.v->GetViewShape()).GetString()),
            "v shape must follow layout (BNSD/NTD or BSND/TND)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HasShape(params.g, {info.batch, info.tokens, info.hv}) &&
          HasShape(params.beta, {info.batch, info.tokens, info.hv}))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "g,beta",
            "g=" + std::string(op::ToString(params.g->GetViewShape()).GetString()) +
                ",beta=" + std::string(op::ToString(params.beta->GetViewShape()).GetString()),
            "g and beta must be BSN [B,T,HV]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!HasShape(params.dO, {info.batch, info.tokens, info.hv, info.valueDim})) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "dO", std::string(op::ToString(params.dO->GetViewShape()).GetString()),
            "d_o must be BSND [B,T,HV,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!HasShape(params.a, {info.batch, info.hv, info.tokens, params.chunkSize})) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "A", std::string(op::ToString(params.a->GetViewShape()).GetString()),
            "[B,HV,T,chunkSize]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(HasShape(params.dBetaOut, {info.batch, info.tokens, info.hv}) &&
          HasShape(params.dGOut, {info.batch, info.tokens, info.hv}))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "dBeta,dG",
            "dBeta=" + std::string(op::ToString(params.dBetaOut->GetViewShape()).GetString()) +
                ",dG=" + std::string(op::ToString(params.dGOut->GetViewShape()).GetString()),
            "dBeta and dG must be [B,T,HV]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.qRstd != nullptr || params.kRstd != nullptr) {
        if (!(HasShape(params.qRstd, {info.batch, info.hk, info.tokens}) &&
              HasShape(params.kRstd, {info.batch, info.hk, info.tokens}))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "qRstd,kRstd",
                "qRstd=" + std::string(op::ToString(params.qRstd->GetViewShape()).GetString()) +
                    ",kRstd=" + std::string(op::ToString(params.kRstd->GetViewShape()).GetString()),
                "q_rstd and k_rstd must be BNS [B,HK,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.betaRaw != nullptr) {
        if (!HasShape(params.betaRaw, {info.batch, info.tokens, info.hv})) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "betaRaw", std::string(op::ToString(params.betaRaw->GetViewShape()).GetString()),
                "beta_raw must be BSN [B,T,HV]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const int64_t sequences = SequenceCount(params, info.batch);
    const int64_t stateDim0 = params.stateVFirst ? info.valueDim : info.keyDim;
    const int64_t stateDim1 = params.stateVFirst ? info.keyDim : info.valueDim;
    if (params.initialState != nullptr) {
        if (!(HasShape(params.initialState, {sequences, info.hv, stateDim0, stateDim1}) &&
              HasShape(params.dh0Out, {sequences, info.hv, stateDim0, stateDim1}))) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "initialState,dh0", "shape mismatch",
                "initial_state and dh0 shapes must match state_v_first");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dht != nullptr) {
        if (!HasShape(params.dht, {sequences, info.hv, stateDim0, stateDim1})) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                OP_NAME, "dht", std::string(op::ToString(params.dht->GetViewShape()).GetString()),
                "dht shape must match state_v_first");
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
        if (!IsAscend950()) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                OP_NAME, "soc", "unsupported", "ChunkGatedDeltaRuleBwd is supported on Ascend950 only");
            return ACLNN_ERR_PARAM_INVALID;
        }
#if GDN_BWD_ENABLE_ARCH22
    }
#endif
    if (params.useGateInKernel) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "use_gate_in_kernel", "true", "use_gate_in_kernel=true is not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dALogOut != nullptr || params.dDtBiasOut != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "dALogOut,dDtBiasOut", "present",
            "d_a_log and d_dt_bias are reserved and must be nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.chunkSize != CHUNK_64) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunk_size", std::to_string(params.chunkSize), "chunk_size must be 64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!std::isfinite(params.scale)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "scale", std::to_string(params.scale), "scale must be finite");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(ResolveShapeInfo(params, info) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.batch > 0 && info.hk > 0 && info.hv > 0 && info.tokens > 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "B,HK,HV,T",
            std::to_string(info.batch) + "," + std::to_string(info.hk) + "," +
                std::to_string(info.hv) + "," + std::to_string(info.tokens),
            "B/HK/HV/T must be positive");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.keyDim == DIM_128 && info.valueDim == DIM_128)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "K,V", std::to_string(info.keyDim) + "," + std::to_string(info.valueDim),
            "K and V must both be 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(info.hv % info.hk == 0 && info.hv / info.hk <= 4)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "HK,HV", std::to_string(info.hk) + "," + std::to_string(info.hv),
            "HV/HK must be an integer in [1, 4]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    {
        const aclnnStatus checkStatus = CheckLayoutShapes(params, info);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    {
        const aclnnStatus checkStatus = CheckMetadata(params, info.tokens);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };
    if (params.cuSeqlens != nullptr && info.batch != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "B", std::to_string(info.batch), "varlen rank-4 input requires B=1");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const DataType dtype = params.q->GetDataType();
    if (dtype != DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            OP_NAME, "q", ge::TypeUtils::DataTypeToSerialString(dtype),
            "the composite currently supports BF16 q/k/v/A/d_o only");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.k->GetDataType() == dtype && params.v->GetDataType() == dtype &&
          params.a->GetDataType() == dtype && params.dO->GetDataType() == dtype &&
          params.dqOut->GetDataType() == dtype && params.dkOut->GetDataType() == dtype &&
          params.dvOut->GetDataType() == dtype)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "q,k,v,A,dO,dq,dk,dv",
            ge::TypeUtils::DataTypeToSerialString(dtype),
            "main tensors must all use BF16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool validGType = params.g->GetDataType() == DataType::DT_BF16 ||
                            params.g->GetDataType() == DataType::DT_FLOAT;
    const bool validBetaType = params.beta->GetDataType() == DataType::DT_BF16 ||
                               params.beta->GetDataType() == DataType::DT_FLOAT;
    if (!(validGType && validBetaType &&
          params.dGOut->GetDataType() == params.g->GetDataType() &&
          params.dBetaOut->GetDataType() == params.beta->GetDataType())) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "g,beta,dG,dBeta",
            std::string("g=") + ge::TypeUtils::DataTypeToSerialString(params.g->GetDataType()) +
                ",beta=" + ge::TypeUtils::DataTypeToSerialString(params.beta->GetDataType()),
            "g/beta must use BF16 or FP32, and each gradient must match its input dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.useQkL2normInKernel && (params.qRstd == nullptr || params.kRstd == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "qRstd,kRstd");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!params.useQkL2normInKernel && (params.qRstd != nullptr || params.kRstd != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "qRstd,kRstd", "present",
            "q_rstd and k_rstd require Q/K L2Norm backward");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.useBetaSigmoidInKernel && params.betaRaw == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "betaRaw");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!params.useBetaSigmoidInKernel && params.betaRaw != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "betaRaw", "present", "beta_raw requires beta sigmoid backward");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (params.qRstd != nullptr) {
        if (!(params.qRstd->GetDataType() == DataType::DT_FLOAT &&
              params.kRstd->GetDataType() == DataType::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                OP_NAME, "qRstd,kRstd",
                std::string("qRstd=") + ge::TypeUtils::DataTypeToSerialString(params.qRstd->GetDataType()) +
                    ",kRstd=" + ge::TypeUtils::DataTypeToSerialString(params.kRstd->GetDataType()),
                "q_rstd and k_rstd must use FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.betaRaw != nullptr) {
        if (params.betaRaw->GetDataType() != params.beta->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "betaRaw", ge::TypeUtils::DataTypeToSerialString(params.betaRaw->GetDataType()),
                "beta_raw must use the same BF16 or FP32 dtype as beta");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (params.initialState != nullptr) {
        if (!(params.initialState->GetDataType() == dtype && params.dh0Out->GetDataType() == dtype)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                OP_NAME, "initialState,dh0",
                ge::TypeUtils::DataTypeToSerialString(params.initialState->GetDataType()),
                "initial_state and dh0 must use the main BF16 dtype");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (params.dh0Out != nullptr) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "dh0Out", "present", "dh0 output requires initial_state");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.dht != nullptr) {
        if (params.dht->GetDataType() != dtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "dht", ge::TypeUtils::DataTypeToSerialString(params.dht->GetDataType()),
                "dht must use the main BF16 dtype");
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
        if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "allocating A2/A3 contiguous output failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
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
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize,executor");
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
    if (!(CheckRequiredInputs(params) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    aclOpExecutor *executorPtr = uniqueExecutor.get();
    if (CheckZeroShape(params, workspaceSize) != ACLNN_SUCCESS) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ShapeInfo info;
    {
        const aclnnStatus checkStatus = CheckParams(params, info);
        if (checkStatus != ACLNN_SUCCESS) {
            return checkStatus;
        }
    };

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
    if (!(MakeContiguous(params.a, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.dO, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.initialState, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.dht, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.qRstd, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.kRstd, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MakeContiguous(params.betaRaw, executorPtr) == ACLNN_SUCCESS)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    const aclTensor *qHead = params.q;
    const aclTensor *kHead = params.k;
    const aclTensor *vHead = params.v;
    const bool useFp32Gate = params.g->GetDataType() == DataType::DT_FLOAT ||
                             params.beta->GetDataType() == DataType::DT_FLOAT;
    const DataType gateType = useFp32Gate ? DataType::DT_FLOAT : DataType::DT_BF16;
    const aclTensor *initialStateKv = params.initialState;
    if (info.sequenceMajor) {
        qHead = TransposeContiguous(params.q, {0, 2, 1, 3}, executorPtr);
        kHead = TransposeContiguous(params.k, {0, 2, 1, 3}, executorPtr);
        vHead = TransposeContiguous(params.v, {0, 2, 1, 3}, executorPtr);
    }
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

    if (!(qHead != nullptr && kHead != nullptr && vHead != nullptr && dOHead != nullptr &&
                   gHead != nullptr && betaHead != nullptr &&
                   (!params.useBetaSigmoidInKernel || betaRawHead != nullptr))) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "input layout conversion failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (params.stateVFirst) {
        initialStateKv = TransposeContiguous(params.initialState, {0, 1, 3, 2}, executorPtr);
        if (!(params.initialState == nullptr || initialStateKv != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "state layout conversion failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    }
    const int64_t chunks = ChunkCount(params, info.tokens);
    const DataType dtype = qHead->GetDataType();
    const op::Shape qShape = MakeShape({info.batch, info.hk, info.tokens, info.keyDim});
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

    const aclTensor *dqHead = info.sequenceMajor
                                  ? executorPtr->AllocTensor(qShape, dtype, Format::FORMAT_ND)
                                  : params.dqOut;
    const aclTensor *dkHead = info.sequenceMajor
                                  ? executorPtr->AllocTensor(qShape, dtype, Format::FORMAT_ND)
                                  : params.dkOut;
    const aclTensor *dvHead = info.sequenceMajor
                                  ? executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND)
                                  : params.dvOut;
    const aclTensor *dBetaHead = executorPtr->AllocTensor(gateShape, gateType, Format::FORMAT_ND);
    const aclTensor *dGHead = executorPtr->AllocTensor(gateShape, gateType, Format::FORMAT_ND);
    if (!(w != nullptr && u != nullptr && dvLocal != nullptr && h != nullptr &&
                   vNew != nullptr && dh != nullptr && dv2 != nullptr && dqHead != nullptr &&
                   dkHead != nullptr && dvHead != nullptr && dBetaHead != nullptr && dGHead != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "allocating composite intermediate tensors failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

#if GDN_BWD_ENABLE_ARCH22
    if (IsDav2201Device()) {
        if (!(MakeOutputContiguousArch22(dqHead, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: MakeOutputContiguousArch22(dqHead, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (!(MakeOutputContiguousArch22(dkHead, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: MakeOutputContiguousArch22(dkHead, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (!(MakeOutputContiguousArch22(dvHead, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: MakeOutputContiguousArch22(dvHead, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        // params 是本次调用的私有副本；仅 A2/A3 用连续 dh0 暂存。
        if (!(MakeOutputContiguousArch22(params.dh0Out, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: MakeOutputContiguousArch22(params.dh0Out, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
#endif
    // A2/A3 直接复用融合算子 ChunkGdnBwdIntra：BF16/FP32 gate、use_exp2 与
    // varlen（空序列无 chunk 记录，天然跳过）均由内核原生处理。
    const auto intraResult = l0op::ChunkGdnBwdIntra(
        qHead, kHead, vHead, gHead, betaHead, params.a, dOHead,
        params.cuSeqlens, params.chunkIndices, params.scale, params.chunkSize,
        params.useExp2, w, u, dvLocal, executorPtr);
    if (!(intraResult[0] != nullptr && intraResult[1] != nullptr &&
                   intraResult[2] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkGdnBwdIntra composition failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const auto fwdHResult = l0op::ChunkFwdH(
        kHead, w, u, gHead, nullptr, initialStateKv,
        params.cuSeqlens, params.chunkIndices, false, params.chunkSize, true,
        params.useExp2, false, h, vNew, nullptr, executorPtr);
    if (!(fwdHResult[0] != nullptr && fwdHResult[1] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkFwdH composition failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const auto dhuResult = l0op::ChunkGatedDeltaRuleBwdDhu(
        qHead, kHead, w, dOHead, dvLocal, gHead, nullptr,
        params.initialState, params.dht, params.cuSeqlens, params.chunkIndices,
        params.scale, params.chunkSize, params.useExp2, params.stateVFirst, dh, params.dh0Out,
        dv2, executorPtr);
    if (!(dhuResult[0] != nullptr && dhuResult[1] != nullptr &&
                   dhuResult[2] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkGatedDeltaRuleBwdDhu composition failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const auto finalizeResult = l0op::ChunkGatedDeltaRuleBwdFinalize(
        qHead, kHead, vHead, vNew, dOHead, dv2, gHead, betaHead, h, dh,
        params.a, params.qRstd, params.kRstd, betaRawHead, params.cuSeqlens,
        params.chunkIndices, params.scale, params.chunkSize,
        params.useQkL2normInKernel, params.useBetaSigmoidInKernel,
        params.useGateInKernel, false, params.useExp2,
        dqHead, dkHead, dvHead, dBetaHead, dGHead, executorPtr);
    for (const aclTensor *tensor : finalizeResult) {
        if (!(tensor != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkGatedDeltaRuleBwdFinalize composition failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    }

    if (info.sequenceMajor) {
        const aclTensor *dqSequence = TransposeContiguous(dqHead, {0, 2, 1, 3}, executorPtr);
        const aclTensor *dkSequence = TransposeContiguous(dkHead, {0, 2, 1, 3}, executorPtr);
        const aclTensor *dvSequence = TransposeContiguous(dvHead, {0, 2, 1, 3}, executorPtr);
        if (!(ViewCopy(dqSequence, params.dqOut, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopy(dqSequence, params.dqOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (!(ViewCopy(dkSequence, params.dkOut, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopy(dkSequence, params.dkOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (!(ViewCopy(dvSequence, params.dvOut, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopy(dvSequence, params.dvOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
#if GDN_BWD_ENABLE_ARCH22
    if (IsDav2201Device()) {
        if (!info.sequenceMajor) {
            if (!(CopyOutputArch22(dqHead, params.dqOut, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: CopyOutputArch22(dqHead, params.dqOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
            if (!(CopyOutputArch22(dkHead, params.dkOut, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: CopyOutputArch22(dkHead, params.dkOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
            if (!(CopyOutputArch22(dvHead, params.dvOut, executorPtr) == ACLNN_SUCCESS)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: CopyOutputArch22(dvHead, params.dvOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        if (!(CopyOutputArch22(params.dh0Out, dh0OutOptional, executorPtr) == ACLNN_SUCCESS)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: CopyOutputArch22(params.dh0Out, dh0OutOptional, executorPtr) == ACLNN_SUCCESS failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
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
    if (!(ViewCopy(dBetaSequence, params.dBetaOut, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopy(dBetaSequence, params.dBetaOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(ViewCopy(dGSequence, params.dGOut, executorPtr) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: ViewCopy(dGSequence, params.dGOut, executorPtr) == ACLNN_SUCCESS failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkGatedDeltaRuleBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleBwd);
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkGatedDeltaRuleBwd launch failed.", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
