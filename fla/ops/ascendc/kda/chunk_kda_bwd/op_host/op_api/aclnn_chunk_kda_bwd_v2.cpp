// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
#include "aclnn_chunk_kda_bwd_v2.h"
#ifdef KDA_BWD_OPTIMIZED_A5
#include "../../../chunk_kda_bwd_prepare/op_host/op_api/chunk_kda_bwd_prepare.h"
#include "../../../chunk_kda_bwd_finalize/op_host/op_api/chunk_kda_bwd_finalize.h"
#include "../../../chunk_kda_bwd_recompute/op_host/op_api/chunk_kda_bwd_recompute.h"
#include "../../../../gdn/chunk_gdn_fwd/chunk_fwd_h/op_host/op_api/chunk_fwd_h.h"
#include "../../../../gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_api/chunk_gated_delta_rule_bwd_dhu.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include <utility>
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <string>

using namespace op;

static constexpr const char *OP_NAME = "aclnnChunkKdaBwd";
namespace {
void NormalizeOriginalShapes(
    std::initializer_list<const aclTensor *> tensors)
{
    for (const aclTensor *tensor : tensors) {
        if (tensor != nullptr) {
            tensor->SetOriginalShape(tensor->GetViewShape());
        }
    }
}

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (auto dim : dims) shape.AppendDim(dim);
    return shape;
}

bool MatchesTensor(const aclTensor *tensor, const op::Shape &shape, DataType dtype)
{
    return tensor && tensor->GetViewShape() == shape &&
        tensor->GetDataType() == dtype && IsContiguous(tensor);
}

// Head/token exchange for both Token tensors and their scalar gradients.
const aclTensor *SwapHeadToken(const aclTensor *tensor, bool packed, aclOpExecutor *executor)
{
    if (!tensor) return nullptr;
    const auto rank = tensor->GetViewShape().GetDimNum();
    std::vector<int64_t> perm(rank);
    for (size_t i = 0; i < rank; ++i) perm[i] = static_cast<int64_t>(i);
    const size_t first = packed ? 0 : 1;
    if (rank <= first + 1) return nullptr;
    std::swap(perm[first], perm[first + 1]);
    const auto *order = executor->AllocIntArray(perm.data(), perm.size());
    const auto *result = order ? l0op::Transpose(tensor, order, executor) : nullptr;
    return result ? l0op::Contiguous(result, executor) : nullptr;
}

const aclTensor *BatchView(const aclTensor *tensor, bool packed, aclOpExecutor *executor)
{
    if (!tensor || !packed) return tensor;
    op::Shape shape;
    shape.AppendDim(1);
    for (size_t i = 0; i < tensor->GetViewShape().GetDimNum(); ++i) {
        shape.AppendDim(tensor->GetViewShape().GetDim(i));
    }
    return l0op::Reshape(tensor, shape, executor);
}

bool CopyGradient(const aclTensor *source, const aclTensor *target, aclOpExecutor *executor)
{
    if (!source || !target) return false;
    if (source == target) return true;
    if (source->GetDataType() != target->GetDataType()) {
        source = l0op::Cast(source, target->GetDataType(), executor);
        if (!source) return false;
    }
    if (source->GetViewShape() != target->GetViewShape()) {
        source = l0op::Reshape(source, target->GetViewShape(), executor);
        if (!source) return false;
    }
    return l0op::ViewCopy(source, target, executor) != nullptr;
}

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdV2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *wOptional, const aclTensor *qgOptional,
    const aclTensor *kgOptional, const aclTensor *vNewOptional,
    const aclTensor *hOptional, const aclTensor *dO,
    const aclTensor *rawGOptional, const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional, const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize, bool safeGate,
    bool useGateInKernel, double lowerBound,
    bool disableRecompute, bool useExp2, bool stateVFirst,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaEffOptional, const char *layout,
    bool useBetaSigmoidInKernel, bool allowNegEigval,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbOut,
    const aclTensor *dgOut, const aclTensor *dh0OutOptional,
    const aclTensor *dAOutOptional, const aclTensor *dBiasOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnChunkKdaBwdV2,
        DFX_IN(q, k, v, beta, gk, aqk, akk, wOptional, qgOptional,
               kgOptional, vNewOptional, hOptional, dO,
               rawGOptional, aLogOptional, dtBiasOptional,
               initialStateOptional, dhtOptional, cuSeqlensOptional,
               chunkIndicesOptional, scale, chunkSize, safeGate,
               useGateInKernel, lowerBound, disableRecompute, useExp2,
               stateVFirst, qRstdOptional, kRstdOptional, betaEffOptional, layout, useBetaSigmoidInKernel, allowNegEigval),
        DFX_OUT(dqOut, dkOut, dvOut, dbOut, dgOut, dh0OutOptional,
                dAOutOptional, dBiasOutOptional));

    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "workspaceSize,executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!useGateInKernel) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "useGateInKernel", "false", "use_gate_in_kernel must be true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "npuArch", std::to_string(static_cast<int32_t>(npuArch)),
            "V2 requires NpuArch::DAV_3510");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (betaEffOptional != nullptr || useBetaSigmoidInKernel || allowNegEigval) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "beta_eff,useBetaSigmoidInKernel,allowNegEigval",
            std::string(betaEffOptional != nullptr ? "beta_eff=present" : "beta_eff=absent") + "," +
                (useBetaSigmoidInKernel ? "useBetaSigmoid=true" : "useBetaSigmoid=false") + "," +
                (allowNegEigval ? "allowNegEigval=true" : "allowNegEigval=false"),
            "A5 beta sigmoid backward is reserved: beta_eff must be null and both flags false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(lowerBound >= -5.0 && lowerBound < 0.0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "lowerBound", std::to_string(lowerBound), "A5 V2 requires -5 <= lower_bound < 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto *tensor : {q, k, v, beta, aqk, akk, dO, dqOut, dkOut, dvOut, dbOut, dgOut}) {
        if (tensor == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q,k,v,beta,aqk,akk,dO,dq,dk,dv,d_beta,d_g");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (!(chunkSize == 64 && safeGate && useExp2 && !stateVFirst)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "K", "invalid", "A5 V2 requires C=64, safe_gate=true, use_exp2=true and K-first state.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (initialStateOptional != nullptr || dhtOptional != nullptr || dh0OutOptional != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "initialState,dht,dh0",
            std::string(initialStateOptional != nullptr ? "initialState=present" : "initialState=absent") + "," +
                (dhtOptional != nullptr ? "dht=present" : "dht=absent") + "," +
                (dh0OutOptional != nullptr ? "dh0=present" : "dh0=absent"),
            "initial_state/dht/dh0 are not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((cuSeqlensOptional == nullptr) != (chunkIndicesOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cuSeqlens,chunkIndices",
            std::string(cuSeqlensOptional == nullptr ? "cuSeqlens=absent" : "cuSeqlens=present") + "," +
                (chunkIndicesOptional == nullptr ? "chunkIndices=absent" : "chunkIndices=present"),
            "cuSeqlens and chunkIndices must be supplied together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((qRstdOptional == nullptr) != (kRstdOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "q_rstd,k_rstd",
            std::string(qRstdOptional == nullptr ? "q_rstd=absent" : "q_rstd=present") + "," +
                (kRstdOptional == nullptr ? "k_rstd=absent" : "k_rstd=present"),
            "q_rstd and k_rstd must appear together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (allowNegEigval && !useBetaSigmoidInKernel) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "allowNegEigval", "true", "allow_neg_eigval requires beta sigmoid backward");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(std::isfinite(static_cast<float>(scale)) && std::isfinite(static_cast<float>(lowerBound)))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "scale,lowerBound",
            std::to_string(scale) + "," + std::to_string(lowerBound),
            "scale and lower_bound must be finite");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool variable = cuSeqlensOptional != nullptr;
    const bool packed = layout && (std::strcmp(layout, "NTD") == 0 || std::strcmp(layout, "TND") == 0);
    const bool inputTokenMajor = layout && (std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "TND") == 0);
    if (!(layout && (packed || std::strcmp(layout, "BNSD") == 0 || std::strcmp(layout, "BSND") == 0))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "layout", layout == nullptr ? "nullptr" : layout,
            "layout must be BNSD, BSND, NTD or TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto &vs = v->GetViewShape();
    if (!(vs.GetDimNum() == (packed ? 3U : 4U) && (!packed || variable))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "v", std::string(op::ToString(vs).GetString()),
            packed ? "3D layout requires cuSeqlens" : "4D layout requires v as [B,H,T,V] or [B,T,H,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!( !variable || packed || vs.GetDim(0) == 1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "B", std::to_string(vs.GetDim(0)), "4D varlen calls require B=1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const size_t base = packed ? 0 : 1;
    const int64_t B = packed ? 1 : vs.GetDim(0);
    const int64_t H = vs.GetDim(base + (inputTokenMajor ? 1 : 0));
    const int64_t T = vs.GetDim(base + (inputTokenMajor ? 0 : 1));
    if (!(B > 0 && H > 0 && T > 0 && vs.GetDim(base + 2) == 128)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "B,H,T,V",
            std::to_string(B) + "," + std::to_string(H) + "," + std::to_string(T) + "," +
                std::to_string(vs.GetDim(base + 2)),
            "B/H/T must be positive and K=V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(T <= (std::numeric_limits<int64_t>::max() - 63) / 128 / 4 / H / B)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "T", std::to_string(T), "token tensor byte size must not overflow int64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Keep the existing A5 indexing and recompute buffer limits.
    if (!(H <= std::numeric_limits<uint32_t>::max() / T &&
          B <= std::numeric_limits<uint32_t>::max() / (H * T))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "H,B", std::to_string(H) + "," + std::to_string(B),
            "V2 token indexing must fit uint32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(disableRecompute || (H <= 256 && H % 8 == 0))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "H", std::to_string(H),
            "recompute requires H<=256 and H divisible by 8 when disableRecompute=false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto token = packed ? MakeShape({H,T,128}) : MakeShape({B,H,T,128});
    const auto scalar = packed ? MakeShape({H,T}) : MakeShape({B,H,T});
    const auto matrix = packed ? MakeShape({H,T,64}) : MakeShape({B,H,T,64});
    const auto externalToken = inputTokenMajor ?
        (packed ? MakeShape({T,H,128}) : MakeShape({B,T,H,128})) : token;
    const auto outputToken = inputTokenMajor ?
        (packed ? MakeShape({T,H,128}) : MakeShape({B,T,H,128})) : token;
    const auto gateToken = packed ? MakeShape({T,H,128}) : MakeShape({B,T,H,128});
    const auto gateScalar = packed ? MakeShape({T,H}) : MakeShape({B,T,H});
    const auto betaShape = gateScalar;
    if (!(MatchesTensor(q, qRstdOptional ? token : externalToken, DataType::DT_BF16) &&
          MatchesTensor(k, kRstdOptional ? token : externalToken, DataType::DT_BF16) &&
          MatchesTensor(v, externalToken, DataType::DT_BF16) &&
          MatchesTensor(dO, gateToken, DataType::DT_BF16) &&
          MatchesTensor(aqk, matrix, DataType::DT_BF16) &&
          MatchesTensor(akk, matrix, DataType::DT_BF16))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q,k,v,dO,aqk,akk",
            "q=" + std::string(op::ToString(q->GetViewShape()).GetString()) +
                ",v=" + std::string(op::ToString(v->GetViewShape()).GetString()) +
                ",aqk=" + std::string(op::ToString(aqk->GetViewShape()).GetString()),
            "BF16 inputs and fixed head-major caches must match the layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MatchesTensor(beta, betaShape, DataType::DT_BF16) ||
          MatchesTensor(beta, betaShape, DataType::DT_FLOAT))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "beta", std::string(op::ToString(beta->GetViewShape()).GetString()),
            "beta must be the original token-major BF16/FP32 input");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(useBetaSigmoidInKernel
              ? MatchesTensor(betaEffOptional, scalar, DataType::DT_FLOAT)
              : betaEffOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            OP_NAME, "betaEff",
            "fused sigmoid requires FP32 head-major beta_eff; otherwise beta_eff must be null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qRstdOptional != nullptr &&
        !(MatchesTensor(qRstdOptional, scalar, DataType::DT_FLOAT) &&
          MatchesTensor(kRstdOptional, scalar, DataType::DT_FLOAT))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "q_rstd,k_rstd",
            "q_rstd=" + std::string(op::ToString(qRstdOptional->GetViewShape()).GetString()) +
                ",k_rstd=" + std::string(op::ToString(kRstdOptional->GetViewShape()).GetString()),
            "rstd must be a head-major FP32 pair");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MatchesTensor(dqOut, outputToken, q->GetDataType()) &&
          MatchesTensor(dkOut, outputToken, k->GetDataType()))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "dq,dk",
            "dq=" + std::string(op::ToString(dqOut->GetViewShape()).GetString()) +
                ",dk=" + std::string(op::ToString(dkOut->GetViewShape()).GetString()),
            "dq/dk must match the layout and q/k dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MatchesTensor(dvOut, outputToken, DataType::DT_BF16) &&
          MatchesTensor(dbOut, gateScalar, beta->GetDataType()) &&
          MatchesTensor(dgOut, gateToken, rawGOptional ? rawGOptional->GetDataType() : DataType::DT_FLOAT))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "dv,db,dg",
            "dv=" + std::string(op::ToString(dvOut->GetViewShape()).GetString()) +
                ",db=" + std::string(op::ToString(dbOut->GetViewShape()).GetString()) +
                ",dg=" + std::string(op::ToString(dgOut->GetViewShape()).GetString()),
            "dv/db/dg must match the layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useGateInKernel &&
        !((MatchesTensor(rawGOptional, gateToken, DataType::DT_FLOAT) ||
           MatchesTensor(rawGOptional, gateToken, DataType::DT_BF16)) &&
          MatchesTensor(aLogOptional, MakeShape({H}), DataType::DT_FLOAT))) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            OP_NAME, "raw_g,a_log",
            "raw_g=" + std::string(op::ToString(rawGOptional->GetViewShape()).GetString()) +
                ",a_log=" + std::string(op::ToString(aLogOptional->GetViewShape()).GetString()),
            "raw_g must be BF16/FP32 and a_log must be FP32 for gate backward");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dtBiasOptional != nullptr &&
        !(dtBiasOptional->GetDataType() == DataType::DT_FLOAT && IsContiguous(dtBiasOptional) &&
          dtBiasOptional->GetViewShape().GetShapeSize() == H * 128)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "dt_bias", std::string(op::ToString(dtBiasOptional->GetViewShape()).GetString()),
            "dt_bias must be contiguous FP32 with H*K elements");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dAOutOptional != nullptr &&
        !(aLogOptional != nullptr && MatchesTensor(dAOutOptional, MakeShape({H}), DataType::DT_FLOAT))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "dA", std::string(op::ToString(dAOutOptional->GetViewShape()).GetString()),
            "dA must be FP32 [H] and requires a_log");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dBiasOutOptional != nullptr &&
        !(dtBiasOptional != nullptr && IsContiguous(dBiasOutOptional) &&
          dBiasOutOptional->GetDataType() == DataType::DT_FLOAT &&
          dBiasOutOptional->GetViewShape().GetShapeSize() == H * 128)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            OP_NAME, "dbias", std::string(op::ToString(dBiasOutOptional->GetViewShape()).GetString()),
            "dbias must be contiguous FP32 with H*K elements and requires dt_bias");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t nc = (T + 63) / 64;
    if (variable) {
        const auto *cu = cuSeqlensOptional;
        const auto *indices = chunkIndicesOptional;
        if (!(cu->Size() >= 2 &&
                       (*cu)[0] == 0 && (*cu)[cu->Size()-1] == T)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens", "invalid", "Invalid cu_seqlens.");
        return ACLNN_ERR_PARAM_INVALID;
    }
        nc = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t begin = (*cu)[seq], end = (*cu)[seq+1];
            if (!(begin >= 0 && end > begin && end <= T)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "cu_seqlens", "invalid", "sequence boundaries are invalid");
        return ACLNN_ERR_PARAM_INVALID;
    }
            for (int64_t c = 0; c < (end - begin + 63) / 64; ++c, ++nc) {
                if (!(static_cast<size_t>(2*nc+1) < indices->Size() &&
                      (*indices)[2*nc] == static_cast<int64_t>(seq) && (*indices)[2*nc+1] == c)) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        OP_NAME, "chunkIndices[2*nc],chunkIndices[2*nc+1]",
                        std::to_string((*indices)[2*nc]) + "," + std::to_string((*indices)[2*nc+1]),
                        "chunkIndices must be canonical sequence-major (seq,chunk) pairs");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
        if (indices->Size() != static_cast<size_t>(2*nc)) {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                OP_NAME, "chunkIndices", std::to_string(indices->Size()), std::to_string(2 * nc));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!(nc > 0 && nc <= std::numeric_limits<int64_t>::max() / 128 / 128 / 2 / H / B)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunkCount", std::to_string(nc), "state tensor byte size must not overflow int64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto state = packed ? MakeShape({nc,H,128,128}) : MakeShape({B,nc,H,128,128});
    if (disableRecompute) {
        for (const auto *cache : {wOptional, qgOptional, kgOptional, vNewOptional}) {
            if (!MatchesTensor(cache, token, DataType::DT_BF16)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    OP_NAME, "w,qg,kg,v_new",
                    cache == nullptr ? "nullptr" : std::string(op::ToString(cache->GetViewShape()).GetString()),
                    "saved Token cache must be BF16 with the head-major token shape");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (!(MatchesTensor(gk, token, DataType::DT_FLOAT) &&
              MatchesTensor(hOptional, state, DataType::DT_BF16))) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                OP_NAME, "gk,h",
                "gk=" + std::string(op::ToString(gk->GetViewShape()).GetString()) +
                    ",h=" + std::string(op::ToString(hOptional->GetViewShape()).GetString()),
                "saved gk must be FP32 token-major and h must be BF16 chunk-major");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (gk != nullptr || wOptional != nullptr || qgOptional != nullptr || kgOptional != nullptr ||
            vNewOptional != nullptr || hOptional != nullptr) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                OP_NAME, "gk,w,qg,kg,v_new,h",
                std::string(gk != nullptr ? "gk=present" : "gk=absent") + "," +
                    (wOptional != nullptr ? "w=present" : "w=absent") + "," +
                    (hOptional != nullptr ? "h=present" : "h=absent"),
                "recompute requires the six saved caches to be absent");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (!(uniqueExecutor.get())) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: uniqueExecutor.get() failed", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto *ex = uniqueExecutor.get();
    NormalizeOriginalShapes({q, k, v, beta, betaEffOptional, gk, aqk, akk, wOptional, qgOptional,
        kgOptional, vNewOptional, hOptional, dO, rawGOptional, aLogOptional,
        dtBiasOptional, qRstdOptional, kRstdOptional, dqOut, dkOut, dvOut, dbOut,
        dgOut, dAOutOptional, dBiasOutOptional});
    auto alloc = [&](const op::Shape &shape, DataType type) {
        return ex->AllocTensor(shape, type, Format::FORMAT_ND);
    };
    // Recompute and Finalize share one head-major beta input.
    const auto *betaHead = useBetaSigmoidInKernel ? betaEffOptional : SwapHeadToken(beta, packed, ex);
    if (!(betaHead)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: betaHead failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor *aLog = aLogOptional;
    const aclTensor *dtBias = dtBiasOptional;
    if (!dtBias) {
        const std::vector<float> zeros(static_cast<size_t>(H * 128), 0.0f);
        const auto *flat = ex->ConvertToTensor(zeros.data(), zeros.size(), DataType::DT_FLOAT);
        if (!(flat)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: flat failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        dtBias = l0op::Reshape(flat, MakeShape({H,128}), ex);
    } else {
        dtBias = l0op::Reshape(dtBias, MakeShape({H,128}), ex);
    }
    if (!(dtBias)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: dtBias failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor *w = wOptional, *qg = qgOptional, *kg = kgOptional;
    const aclTensor *vNew = vNewOptional, *h = hOptional;
    if (!disableRecompute) {
        // Recompute promotes BF16 raw gates to FP32 on-chip.
        const auto *rawRecompute = rawGOptional;
        const auto *logRecompute = aLog;
        const auto *biasRecompute = dtBias;
        if (!(rawRecompute && logRecompute && biasRecompute)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: rawRecompute && logRecompute && biasRecompute failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        w = alloc(token, DataType::DT_BF16);
        qg = alloc(token, DataType::DT_BF16);
        kg = alloc(token, DataType::DT_BF16);
        vNew = alloc(token, DataType::DT_BF16);
        h = alloc(state, DataType::DT_BF16);
        gk = alloc(token, DataType::DT_FLOAT);
        const auto *u = alloc(token, DataType::DT_BF16);
        if (!(w && qg && kg && vNew && h && gk && u)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: w && qg && kg && vNew && h && gk && u failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        const auto result = l0op::ChunkKdaBwdRecompute(
            BatchView(q, packed, ex), BatchView(k, packed, ex), BatchView(v, packed, ex),
            BatchView(rawRecompute, packed, ex), BatchView(betaHead, packed, ex), BatchView(akk, packed, ex),
            logRecompute, biasRecompute, cuSeqlensOptional, chunkIndicesOptional, chunkSize, useGateInKernel,
            useExp2, lowerBound, packed ? "NTD" : "BNSD", BatchView(w, packed, ex), BatchView(u, packed, ex),
            BatchView(qg, packed, ex), BatchView(kg, packed, ex), BatchView(gk, packed, ex), ex, layout, qRstdOptional != nullptr);
        for (const auto *value : result) {
            if (!(value)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: value failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
        const auto forward = l0op::ChunkFwdH(
            BatchView(kg, packed, ex), BatchView(w, packed, ex), BatchView(u, packed, ex),
            nullptr, BatchView(gk, packed, ex), nullptr, cuSeqlensOptional, chunkIndicesOptional,
            false, chunkSize, true, useExp2, stateVFirst,
            BatchView(h, packed, ex), BatchView(vNew, packed, ex), nullptr, ex);
        if (!(forward[0] && forward[1])) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: forward[0] && forward[1] failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const auto *dAqk = alloc(matrix, DataType::DT_FLOAT);
    const auto *dvLocal = alloc(token, DataType::DT_BF16);
    const auto *dqRaw = alloc(token, DataType::DT_FLOAT);
    const auto *dh = alloc(state, DataType::DT_BF16);
    const auto *dvScan = alloc(token, DataType::DT_BF16);
    if (!(dAqk && dvLocal && dqRaw && dh && dvScan)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: dAqk && dvLocal && dqRaw && dh && dvScan failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    // A5 Prepare's existing varlen kernel uses rank-3 NTD. Drop only the
    // leading B=1 view dimension for 4D varlen callers; no data is moved.
    auto prepareView = [&](const aclTensor *tensor) -> const aclTensor * {
        if (!variable || packed) return tensor;
        op::Shape shape;
        for (size_t i = 1; i < tensor->GetViewShape().GetDimNum(); ++i) {
            shape.AppendDim(tensor->GetViewShape().GetDim(i));
        }
        return l0op::Reshape(tensor, shape, ex);
    };
    const aclTensor *dOInput = dO;
    if (T == H) {
        // Square shapes cannot distinguish token-major from head-major.
        dOInput = SwapHeadToken(dO, packed, ex);
        if (!(dOInput != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: dOInput != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const auto prepare = l0op::ChunkKdaBwdPrepare(
        prepareView(aqk), prepareView(vNew), prepareView(dOInput), prepareView(h),
        cuSeqlensOptional, chunkIndicesOptional, variable ? "NTD" : "BNSD", scale, chunkSize, stateVFirst,
        prepareView(dAqk), prepareView(dvLocal), prepareView(dqRaw), ex);
    for (const auto *value : prepare) {
        if (!(value)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: value failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const auto scan = l0op::ChunkGatedDeltaRuleBwdDhu(
        BatchView(qg, packed, ex), BatchView(kg, packed, ex), BatchView(w, packed, ex),
        BatchView(dOInput, packed, ex), BatchView(dvLocal, packed, ex), nullptr,
        BatchView(gk, packed, ex), nullptr, nullptr, cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useExp2, false, BatchView(dh, packed, ex), nullptr,
        BatchView(dvScan, packed, ex), ex);
    if (!(scan[0] && scan[2])) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: scan[0] && scan[2] failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    // Finalize converts original gate parameters on-chip.
    const auto *betaFinal = betaHead;
    if (!(betaFinal && (!rawGOptional || rawGOptional) && (!aLogOptional || aLog))) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: betaFinal && (!rawGOptional || rawGOptional) && (!aLogOptional || aLog) failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    // Finalize writes q/k/v gradients in layout and gate gradients token-major.
    const auto *dbFinal = dbOut;
    const auto *dA = dAOutOptional ? dAOutOptional : alloc(MakeShape({H}), DataType::DT_FLOAT);
    const auto *dBias = alloc(MakeShape({H,128}), DataType::DT_FLOAT);
    if (!(dbFinal && dA && dBias)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: dbFinal && dA && dBias failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    NormalizeOriginalShapes({q, k, v, gk, rawGOptional, betaFinal, aLog, dtBias,
        akk, vNew, h, dh, dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
        dqOut, dkOut, dvOut, dbFinal, dgOut, dA, dBias});
    const auto finalize = l0op::ChunkKdaBwdFinalize(
        q, k, v, gk, rawGOptional, betaFinal, aLog, dtBias,
        akk, vNew, h, dh, dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
        cuSeqlensOptional, chunkIndicesOptional, scale, lowerBound, chunkSize,
        safeGate, useGateInKernel, useExp2, stateVFirst,
        useBetaSigmoidInKernel, allowNegEigval, layout,
        dqOut, dkOut, dvOut, dbFinal, dgOut, dA, dBias, ex, layout);
    for (const auto *value : finalize) {
        if (!(value)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: value failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (!(!dBiasOutOptional || CopyGradient(dBias, dBiasOutOptional, ex))) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "Unable to adapt KDA backward outputs.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdV2(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdV2);
    return CommonOpExecutorRun(workspace,workspaceSize,executor,stream);
}
#endif
