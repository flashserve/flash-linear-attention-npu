#include "op_common/log/log.h"
#include "base/err_msg.h"
#include "graph/utils/type_utils.h"
#include "aclnn_chunk_kda_bwd.h"

#include "chunk_kda_bwd.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#ifdef KDA_BWD_STAGES_ARCH22
#include "../../../chunk_kda_bwd_prepare/op_host/op_api/chunk_kda_bwd_prepare.h"
#include "../../../chunk_kda_bwd_finalize/op_host/op_api/chunk_kda_bwd_finalize.h"
#include "../../../chunk_kda_bwd_recompute/op_host/op_api/chunk_kda_bwd_recompute.h"
#include "../../../../gdn/chunk_gdn_fwd/chunk_fwd_h/op_host/op_api/chunk_fwd_h.h"
#include "../../../../gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_api/chunk_gated_delta_rule_bwd_dhu.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#endif

#include <initializer_list>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <utility>

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

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

#ifndef KDA_BWD_STAGES_ARCH22
const aclTensor *ConvertIntArrayToTensor(
    const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor =
        executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}
#endif

#ifdef KDA_BWD_STAGES_ARCH22
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
#endif

} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdGetWorkspaceSize(
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
        aclnnChunkKdaBwd,
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
#ifdef KDA_BWD_STAGES_ARCH22
    for (const auto *tensor : {q, k, v, beta, aqk, akk, dO, dqOut, dkOut, dvOut, dbOut, dgOut}) {
        if (tensor == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q,k,v,beta,aqk,akk,dO,dq,dk,dv,d_beta,d_g");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (!safeGate) {
        const std::vector<const char *> keys = {"param_name", "op_name", "incorrect_value", "correct_value"};
        const std::vector<const char *> values = {"safe_gate", "aclnnChunkKdaBwd", "false", "true on A2/A3"};
        REPORT_PREDEFINED_ERR_MSG("EZ0024", keys, values);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(chunkSize == 64 && useExp2 && !stateVFirst)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "chunkSize,useExp2,stateVFirst",
            std::to_string(chunkSize) + "," + (useExp2 ? "true" : "false") + "," +
                (stateVFirst ? "true" : "false"),
            "A2/A3 requires chunkSize=64, useExp2=true and stateVFirst=false");
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
        if (!(cu->Size() >= 2 && cu->Size() <= 1025 && (*cu)[0] == 0 && (*cu)[cu->Size()-1] == T)) {
            OP_LOGE_FOR_INVALID_LISTSIZE(
                OP_NAME, "cuSeqlens", std::to_string(cu->Size()), "2..1025 entries from 0 to T=" + std::to_string(T));
            return ACLNN_ERR_PARAM_INVALID;
        }
        nc = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t begin = (*cu)[seq], end = (*cu)[seq+1];
            if (!(begin >= 0 && end >= begin && end <= T)) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    OP_NAME, "cuSeqlens[seq],cuSeqlens[seq+1]",
                    std::to_string(begin) + "," + std::to_string(end),
                    "sequence boundaries must satisfy 0 <= begin <= end <= T=" + std::to_string(T));
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
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME,
                           "%s: CREATE_EXECUTOR returned nullptr; propagate the resource first error.", __func__);
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
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
    // Public do is token-major; Prepare and Dhu share one head-major copy.
    const auto *dOHead = SwapHeadToken(dO, packed, ex);
    if (!(dOHead)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: dOHead failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const auto prepare = l0op::ChunkKdaBwdPrepare(
        aqk, vNew, dOHead, h,
        cuSeqlensOptional, chunkIndicesOptional, packed ? "NTD" : "BNSD", scale, chunkSize, stateVFirst,
        dAqk, dvLocal, dqRaw, ex);
    for (const auto *value : prepare) {
        if (!(value)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: value failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const auto scan = l0op::ChunkGatedDeltaRuleBwdDhu(
        BatchView(qg, packed, ex), BatchView(kg, packed, ex), BatchView(w, packed, ex),
        BatchView(dOHead, packed, ex), BatchView(dvLocal, packed, ex), nullptr,
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
#else
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
        OP_NAME, "layout", "single-layout backward is currently implemented on A2/A3 only");
    return ACLNN_ERR_PARAM_INVALID;
    const aclTensor *required[] = {
        q, k, v, beta, gk, aqk, akk, dO,
        dqOut, dkOut, dvOut, dbOut, dgOut};
    for (const aclTensor *tensor : required) {
        if (tensor == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "q,k,v,beta,gk,aqk,akk,dO,dq,dk,dv,d_beta,d_g");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (chunkSize != 64) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "chunkSize", std::to_string(chunkSize), "ChunkKdaBwd requires chunk_size=64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!disableRecompute) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "disableRecompute", "false", "disable_recompute=false is reserved but not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!useExp2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "useExp2", "false", "use_exp2=false is reserved but not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (stateVFirst) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            OP_NAME, "stateVFirst", "true", "state_v_first=true is reserved but not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (wOptional == nullptr || qgOptional == nullptr || kgOptional == nullptr ||
        vNewOptional == nullptr || hOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "w,qg,kg,v_new,h");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if ((cuSeqlensOptional == nullptr) != (chunkIndicesOptional == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "cu_seqlens,chunk_indices",
            std::string(cuSeqlensOptional == nullptr ? "cu_seqlens=absent" : "cu_seqlens=present") + "," +
                (chunkIndicesOptional == nullptr ? "chunk_indices=absent" : "chunk_indices=present"),
            "cu_seqlens and chunk_indices must be supplied together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // PR291 has not completed the reviewed dht/dh0 semantics. Keep the
    // public composition honest until that implementation is repaired.
    if (initialStateOptional != nullptr || dhtOptional != nullptr || dh0OutOptional != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "initial_state,dht,dh0",
            std::string(initialStateOptional != nullptr ? "initial_state=present" : "initial_state=absent") + "," +
                (dhtOptional != nullptr ? "dht=present" : "dht=absent") + "," +
                (dh0OutOptional != nullptr ? "dh0=present" : "dh0=absent"),
            "initial_state, dht and dh0 are not supported by this integration");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useGateInKernel && (rawGOptional == nullptr || aLogOptional == nullptr || dAOutOptional == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "raw_g,a_log,dA");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!useGateInKernel &&
        (rawGOptional != nullptr || aLogOptional != nullptr || dtBiasOptional != nullptr ||
         dAOutOptional != nullptr || dBiasOutOptional != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            OP_NAME, "raw_g,a_log,dt_bias,dA,dbias", "present while use_gate_in_kernel=false",
            "raw_g, a_log, dt_bias, dA and dbias require use_gate_in_kernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useGateInKernel && dtBiasOptional == nullptr && dBiasOutOptional != nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dt_bias");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (useGateInKernel && dtBiasOptional != nullptr && dBiasOutOptional == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dbias");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    for (const auto *param : {aLogOptional, dtBiasOptional, dAOutOptional, dBiasOutOptional}) {
        if (param != nullptr && param->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                OP_NAME, "a_log,dt_bias,dA,dbias",
                ge::TypeUtils::DataTypeToSerialString(param->GetDataType()),
                "a_log, dt_bias and their gradients must be FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const op::Shape &qShape = q->GetViewShape();

    const op::Shape &hShape = hOptional->GetViewShape();
    const bool isVarLen = cuSeqlensOptional != nullptr;
    if (!(qShape.GetDimNum() == (isVarLen ? 3U : 4U) && hShape.GetDimNum() == (isVarLen ? 4U : 5U))) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            OP_NAME, "q,h",
            std::to_string(qShape.GetDimNum()) + "," + std::to_string(hShape.GetDimNum()),
            "canonical dense q/h ranks are 4/5 and varlen ranks are 3/4");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto *executorPtr = uniqueExecutor.get();

    const aclTensor *cuTensor =
        ConvertIntArrayToTensor(cuSeqlensOptional, executorPtr);
    const aclTensor *chunkTensor =
        ConvertIntArrayToTensor(chunkIndicesOptional, executorPtr);
    if (!(!isVarLen || (cuTensor != nullptr && chunkTensor != nullptr))) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "convert varlen metadata to tensors failed.", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // Keep the device ABI stable when optional outputs are absent.  The
    // current CANN optional-output launcher does not materialize a nullptr
    // placeholder reliably, which would shift workspace/tiling arguments.
    // These aliases are never written unless the corresponding feature is
    // enabled, in which case the public validation above requires real
    // dA/dbias outputs.
    // Legacy A5 keeps its FP32 device ABI. Adapt at L2; V2 writes the
    // requested dtype directly in Finalize and does not take this fallback.
    if (!(dbOut->GetDataType() == beta->GetDataType() &&
          (!useGateInKernel || (dgOut->GetDataType() == rawGOptional->GetDataType() &&
                                dAOutOptional->GetDataType() == aLogOptional->GetDataType())))) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            OP_NAME, "db,dg,dA",
            std::string("db=") + ge::TypeUtils::DataTypeToSerialString(dbOut->GetDataType()) +
                ",beta=" + ge::TypeUtils::DataTypeToSerialString(beta->GetDataType()),
            "gate gradients must match their input dtypes");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto fp32Output = [&](const aclTensor *out) -> const aclTensor * {
        if (!out || out->GetDataType() == DataType::DT_FLOAT) return out;
        return executorPtr->AllocTensor(out->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);
    };
    const auto *dbKernel = fp32Output(dbOut);
    const auto *dgKernel = fp32Output(dgOut);
    const auto *dAKernel = dAOutOptional;
    if (!(dbKernel && dgKernel && (!dAOutOptional || dAKernel))) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: dbKernel && dgKernel && (!dAOutOptional || dAKernel) failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    const aclTensor *dAForKernel = dAKernel ? dAKernel : dgKernel;
    const aclTensor *dBiasForKernel =
        dBiasOutOptional != nullptr ? dBiasOutOptional : dgOut;
    // 一维存储描述可能使 OriginalShape 展平；下发前按逻辑视图统一形状，不改变数据布局。
    NormalizeOriginalShapes({
        q, k, v, beta, gk, aqk, akk, wOptional, qgOptional, kgOptional,
        vNewOptional, hOptional, dO, rawGOptional, aLogOptional,
        dtBiasOptional, cuTensor, chunkTensor, dqOut, dkOut, dvOut, dbKernel,
        dgKernel, dAForKernel, dBiasForKernel});

    const auto result = l0op::ChunkKdaBwd(
        q, k, v, beta, gk, aqk, akk, wOptional, qgOptional, kgOptional,
        vNewOptional, hOptional, dO,
        rawGOptional, aLogOptional, dtBiasOptional, cuTensor, chunkTensor,
        static_cast<float>(scale), chunkSize, safeGate, useGateInKernel,
        static_cast<float>(lowerBound), disableRecompute, useExp2,
        stateVFirst,
        dqOut, dkOut, dvOut, dbKernel, dgKernel,
        dAForKernel, dBiasForKernel, executorPtr);
    for (size_t i = 0; i < 5; ++i) {
        if (!(result[i] != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: result[i] != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (!(!useGateInKernel || result[5] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: !useGateInKernel || result[5] != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (!(dtBiasOptional == nullptr || result[6] != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: dtBiasOptional == nullptr || result[6] != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    for (const auto &pair : {std::pair<const aclTensor *, const aclTensor *>{dbKernel, dbOut},
                            {dgKernel, dgOut}, {dAKernel, dAOutOptional}}) {
        if (!pair.second || pair.first == pair.second) continue;
        const auto *converted = l0op::Cast(pair.first, pair.second->GetDataType(), executorPtr);
        if (!(converted && l0op::ViewCopy(converted, pair.second, executorPtr))) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: converted && l0op::ViewCopy(converted, pair.second, executorPtr) failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

#endif
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwd);
    if (!(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: " "ChunkKdaBwd executor launch failed.", __func__);
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
