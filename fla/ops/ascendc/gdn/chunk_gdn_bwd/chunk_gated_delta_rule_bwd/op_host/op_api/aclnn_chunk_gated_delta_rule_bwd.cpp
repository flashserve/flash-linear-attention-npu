/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_gated_delta_rule_bwd.h"

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

using namespace op;

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
    CHECK_COND(varied == (params.chunkIndices != nullptr), ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be provided together.");
    if (!varied) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(params.cuSeqlens->Size() >= 2 && (*params.cuSeqlens)[0] == 0,
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens must start at zero.");
    CHECK_COND((*params.cuSeqlens)[params.cuSeqlens->Size() - 1] == tokens,
               ACLNN_ERR_PARAM_INVALID, "cu_seqlens must end at T.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlens->Size(); ++seq) {
        const int64_t begin = (*params.cuSeqlens)[seq];
        const int64_t end = (*params.cuSeqlens)[seq + 1];
        CHECK_COND(end >= begin, ACLNN_ERR_PARAM_INVALID,
                   "cu_seqlens must be nondecreasing.");
        const int64_t chunks = (end - begin + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND(offset + 1 < params.chunkIndices->Size() &&
                           (*params.chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                           (*params.chunkIndices)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunk_indices must use canonical sequence-major order.");
            offset += 2;
        }
    }
    CHECK_COND(offset == params.chunkIndices->Size(), ACLNN_ERR_PARAM_INVALID,
               "chunk_indices size does not match cu_seqlens.");
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShapeInfo(const Params &params, ShapeInfo &info)
{
    CHECK_COND(params.layout != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "layout must not be nullptr.");
    if (std::strcmp(params.layout, "BNSD") == 0) {
        info.layout = Layout::BNSD;
    } else if (std::strcmp(params.layout, "BSND") == 0) {
        info.layout = Layout::BSND;
    } else if (std::strcmp(params.layout, "NTD") == 0) {
        info.layout = Layout::NTD;
    } else if (std::strcmp(params.layout, "TND") == 0) {
        info.layout = Layout::TND;
    } else {
        CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                   "layout must be BNSD, BSND, NTD or TND.");
    }
    info.sequenceMajor = info.layout == Layout::BSND || info.layout == Layout::TND;
    CHECK_COND(params.q->GetViewShape().GetDimNum() == 4 &&
                   params.v->GetViewShape().GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "q and v must be rank 4.");
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
    CHECK_COND(SameShape(params.q, params.k) && SameShape(params.q, params.dqOut) &&
                   SameShape(params.k, params.dkOut) && SameShape(params.v, params.dvOut),
               ACLNN_ERR_PARAM_INVALID, "dq/dk/dv must match q/k/v respectively.");
    const bool validVShape = info.sequenceMajor
                                 ? HasShape(params.v, {info.batch, info.tokens, info.hv, info.valueDim})
                                 : HasShape(params.v, {info.batch, info.hv, info.tokens, info.valueDim});
    CHECK_COND(validVShape, ACLNN_ERR_PARAM_INVALID,
               "v shape must follow layout (BNSD/NTD or BSND/TND).");
    CHECK_COND(HasShape(params.g, {info.batch, info.tokens, info.hv}) &&
                   HasShape(params.beta, {info.batch, info.tokens, info.hv}),
               ACLNN_ERR_PARAM_INVALID, "g and beta must be BSN [B,T,HV].");
    CHECK_COND(HasShape(params.dO, {info.batch, info.tokens, info.hv, info.valueDim}),
               ACLNN_ERR_PARAM_INVALID, "d_o must be BSND [B,T,HV,V].");
    CHECK_COND(HasShape(params.a, {info.batch, info.hv, info.tokens, params.chunkSize}),
               ACLNN_ERR_PARAM_INVALID, "A must be [B,HV,T,64].");
    CHECK_COND(HasShape(params.dBetaOut, {info.batch, info.tokens, info.hv}) &&
                   HasShape(params.dGOut, {info.batch, info.tokens, info.hv}),
               ACLNN_ERR_PARAM_INVALID, "dBeta and dG must be [B,T,HV].");
    if (params.qRstd != nullptr || params.kRstd != nullptr) {
        CHECK_COND(HasShape(params.qRstd, {info.batch, info.hk, info.tokens}) &&
                       HasShape(params.kRstd, {info.batch, info.hk, info.tokens}),
                   ACLNN_ERR_PARAM_INVALID, "q_rstd and k_rstd must be BNS [B,HK,T].");
    }
    if (params.betaRaw != nullptr) {
        CHECK_COND(HasShape(params.betaRaw, {info.batch, info.tokens, info.hv}),
                   ACLNN_ERR_PARAM_INVALID, "beta_raw must be BSN [B,T,HV].");
    }
    const int64_t sequences = SequenceCount(params, info.batch);
    const int64_t stateDim0 = params.stateVFirst ? info.valueDim : info.keyDim;
    const int64_t stateDim1 = params.stateVFirst ? info.keyDim : info.valueDim;
    if (params.initialState != nullptr) {
        CHECK_COND(HasShape(params.initialState, {sequences, info.hv, stateDim0, stateDim1}) &&
                       HasShape(params.dh0Out, {sequences, info.hv, stateDim0, stateDim1}),
                   ACLNN_ERR_PARAM_INVALID, "initial_state and dh0 shapes must match state_v_first.");
    }
    if (params.dht != nullptr) {
        CHECK_COND(HasShape(params.dht, {sequences, info.hv, stateDim0, stateDim1}),
                   ACLNN_ERR_PARAM_INVALID, "dht shape must match state_v_first.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const Params &params, ShapeInfo &info)
{
    CHECK_COND(IsAscend950(), ACLNN_ERR_PARAM_INVALID,
               "ChunkGatedDeltaRuleBwd is supported on Ascend950 only.");
    const std::array<const aclTensor *, 12> required = {
        params.q, params.k, params.v, params.g, params.beta, params.a,
        params.dO, params.dqOut, params.dkOut, params.dvOut,
        params.dBetaOut, params.dGOut};
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "required tensor must not be nullptr.");
    }
    CHECK_COND(params.useExp2, ACLNN_ERR_PARAM_INVALID,
               "use_exp2=false is not supported.");
    CHECK_COND(!params.useGateInKernel, ACLNN_ERR_PARAM_INVALID,
               "use_gate_in_kernel=true is not supported.");
    CHECK_COND(params.dALogOut == nullptr && params.dDtBiasOut == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "d_a_log and d_dt_bias are reserved and must be nullptr.");
    CHECK_COND(params.chunkSize == CHUNK_64, ACLNN_ERR_PARAM_INVALID,
               "chunk_size must be 64.");
    CHECK_COND(std::isfinite(params.scale), ACLNN_ERR_PARAM_INVALID,
               "scale must be finite.");
    CHECK_RET(ResolveShapeInfo(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(info.batch > 0 && info.hk > 0 && info.hv > 0 && info.tokens > 0,
               ACLNN_ERR_PARAM_INVALID, "B/HK/HV/T must be positive.");
    CHECK_COND(info.keyDim == DIM_128 && info.valueDim == DIM_128,
               ACLNN_ERR_PARAM_INVALID, "K and V must both be 128.");
    CHECK_COND(info.hv % info.hk == 0 && info.hv / info.hk <= 4,
               ACLNN_ERR_PARAM_INVALID, "HV/HK must be an integer in [1, 4].");
    CHECK_RET(CheckLayoutShapes(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckMetadata(params, info.tokens) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.cuSeqlens == nullptr || info.batch == 1,
               ACLNN_ERR_PARAM_INVALID, "varlen rank-4 input requires B=1.");

    const DataType dtype = params.q->GetDataType();
    CHECK_COND(dtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "the composite currently supports BF16 q/k/v/A/d_o only.");
    CHECK_COND(params.k->GetDataType() == dtype && params.v->GetDataType() == dtype &&
                   params.a->GetDataType() == dtype && params.dO->GetDataType() == dtype &&
                   params.dqOut->GetDataType() == dtype && params.dkOut->GetDataType() == dtype &&
                   params.dvOut->GetDataType() == dtype,
               ACLNN_ERR_PARAM_INVALID, "main tensors must all use BF16.");
    const bool validGType = params.g->GetDataType() == DataType::DT_BF16 ||
                            params.g->GetDataType() == DataType::DT_FLOAT;
    const bool validBetaType = params.beta->GetDataType() == DataType::DT_BF16 ||
                               params.beta->GetDataType() == DataType::DT_FLOAT;
    CHECK_COND(validGType && validBetaType &&
                   params.dGOut->GetDataType() == params.g->GetDataType() &&
                   params.dBetaOut->GetDataType() == params.beta->GetDataType(),
               ACLNN_ERR_PARAM_INVALID,
               "g/beta must use BF16 or FP32, and each gradient must match its input dtype.");
    CHECK_COND(!params.useQkL2normInKernel ||
                   (params.qRstd != nullptr && params.kRstd != nullptr),
               ACLNN_ERR_PARAM_NULLPTR,
               "q_rstd and k_rstd are required for Q/K L2Norm backward.");
    CHECK_COND(params.useQkL2normInKernel ||
                   (params.qRstd == nullptr && params.kRstd == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "q_rstd and k_rstd require Q/K L2Norm backward.");
    CHECK_COND(!params.useBetaSigmoidInKernel || params.betaRaw != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "beta_raw is required for beta sigmoid backward.");
    CHECK_COND(params.useBetaSigmoidInKernel || params.betaRaw == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "beta_raw requires beta sigmoid backward.");

    if (params.qRstd != nullptr) {
        CHECK_COND(params.qRstd->GetDataType() == DataType::DT_FLOAT &&
                       params.kRstd->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "q_rstd and k_rstd must use FP32.");
    }
    if (params.betaRaw != nullptr) {
        CHECK_COND(params.betaRaw->GetDataType() == params.beta->GetDataType(),
                   ACLNN_ERR_PARAM_INVALID,
                   "beta_raw must use the same BF16 or FP32 dtype as beta.");
    }

    if (params.initialState != nullptr) {
        CHECK_COND(params.initialState->GetDataType() == dtype &&
                       params.dh0Out->GetDataType() == dtype,
                   ACLNN_ERR_PARAM_INVALID,
                   "initial_state and dh0 must use the main BF16 dtype.");
    } else {
        CHECK_COND(params.dh0Out == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "dh0 output requires initial_state.");
    }
    if (params.dht != nullptr) {
        CHECK_COND(params.dht->GetDataType() == dtype, ACLNN_ERR_PARAM_INVALID,
                   "dht must use the main BF16 dtype.");
    }
    return ACLNN_SUCCESS;
}

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
    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");

    Params params{q, k, v, g, beta, a, dO, initialStateOptional, dhtOptional,
                  qRstdOptional, kRstdOptional, betaRawOptional,
                  aLogOptional, dtBiasOptional,
                  cuSeqlensOptional, chunkIndicesOptional, layout, scale,
                  chunkSize, useExp2, useGateInKernel,
                  useQkL2normInKernel, useBetaSigmoidInKernel, stateVFirst,
                  dqOut, dkOut, dvOut, dBetaOut, dGOut, dh0OutOptional,
                  dALogOutOptional, dDtBiasOutOptional};
    ShapeInfo info;
    CHECK_RET(CheckParams(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

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
    const op::Shape qShape = MakeShape({info.batch, info.hk, info.tokens, info.keyDim});
    const op::Shape vShape = MakeShape({info.batch, info.hv, info.tokens, info.valueDim});
    const op::Shape gateShape = MakeShape({info.batch, info.hv, info.tokens});
    const op::Shape wShape = MakeShape({info.batch, info.hv, info.tokens, info.keyDim});
    const op::Shape hShape = MakeShape({info.batch, info.hv, chunks, info.keyDim, info.valueDim});

    const aclTensor *w = executorPtr->AllocTensor(wShape, dtype, Format::FORMAT_ND);
    const aclTensor *u = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
    const aclTensor *dvLocal = executorPtr->AllocTensor(vShape, dtype, Format::FORMAT_ND);
    const aclTensor *h = executorPtr->AllocTensor(hShape, dtype, Format::FORMAT_ND);
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
    CHECK_COND(w != nullptr && u != nullptr && dvLocal != nullptr && h != nullptr &&
                   vNew != nullptr && dh != nullptr && dv2 != nullptr && dqHead != nullptr &&
                   dkHead != nullptr && dvHead != nullptr && dBetaHead != nullptr && dGHead != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "allocating composite intermediate tensors failed.");

    const auto intraResult = l0op::ChunkGdnBwdIntra(
        qHead, kHead, vHead, gHead, betaHead, params.a, dOHead,
        params.cuSeqlens, params.chunkIndices, params.scale, params.chunkSize,
        params.useExp2, w, u, dvLocal, executorPtr);
    CHECK_COND(intraResult[0] != nullptr && intraResult[1] != nullptr &&
                   intraResult[2] != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkGdnBwdIntra composition failed.");

    const auto fwdHResult = l0op::ChunkFwdH(
        kHead, w, u, gHead, nullptr, initialStateKv,
        params.cuSeqlens, params.chunkIndices, false, params.chunkSize, true,
        params.useExp2, false, h, vNew, nullptr, executorPtr);
    CHECK_COND(fwdHResult[0] != nullptr && fwdHResult[1] != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkFwdH composition failed.");

    const auto dhuResult = l0op::ChunkGatedDeltaRuleBwdDhu(
        qHead, kHead, w, dOHead, dvLocal, gHead, nullptr,
        params.initialState, params.dht, params.cuSeqlens, params.chunkIndices,
        params.scale, params.chunkSize, params.useExp2, params.stateVFirst, dh, params.dh0Out,
        dv2, executorPtr);
    CHECK_COND(dhuResult[0] != nullptr && dhuResult[1] != nullptr &&
                   dhuResult[2] != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkGatedDeltaRuleBwdDhu composition failed.");

    const auto finalizeResult = l0op::ChunkGatedDeltaRuleBwdFinalize(
        qHead, kHead, vHead, vNew, dOHead, dv2, gHead, betaHead, h, dh,
        params.a, params.qRstd, params.kRstd, betaRawHead, params.cuSeqlens,
        params.chunkIndices, params.scale, params.chunkSize,
        params.useQkL2normInKernel, params.useBetaSigmoidInKernel,
        params.useGateInKernel, false, params.useExp2,
        dqHead, dkHead, dvHead, dBetaHead, dGHead, executorPtr);
    for (const aclTensor *tensor : finalizeResult) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR,
                   "ChunkGatedDeltaRuleBwdFinalize composition failed.");
    }

    if (info.sequenceMajor) {
        const aclTensor *dqSequence = TransposeContiguous(dqHead, {0, 2, 1, 3}, executorPtr);
        const aclTensor *dkSequence = TransposeContiguous(dkHead, {0, 2, 1, 3}, executorPtr);
        const aclTensor *dvSequence = TransposeContiguous(dvHead, {0, 2, 1, 3}, executorPtr);
        CHECK_RET(ViewCopy(dqSequence, params.dqOut, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopy(dkSequence, params.dkOut, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(ViewCopy(dvSequence, params.dvOut, executorPtr) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }
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
