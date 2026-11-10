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
#include "host_utils/log.h"

using namespace op;

namespace {

constexpr const char *CHUNK_KDA_BWD_ENTITY = "aclnnChunkKdaBwd";

std::string DfxTensorShapeOf(const aclTensor *t)
{
    return t == nullptr ? std::string("null") : fla_npu::host_utils::DfxShapeString(t->GetViewShape());
}

std::string DfxTensorDtypeOf(const aclTensor *t)
{
    return t == nullptr ? std::string("null") : fla_npu::host_utils::DfxDtypeString(t->GetDataType());
}

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
    const char *outputLayout = layout;

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

    if (workspaceSize == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_OUTPUT_COMPAT(CHUNK_KDA_BWD_ENTITY, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_OUTPUT_COMPAT(CHUNK_KDA_BWD_ENTITY, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!useGateInKernel) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "useGateInKernel", "false", "true");
        return ACLNN_ERR_PARAM_INVALID;
    }
#ifdef KDA_BWD_STAGES_ARCH22
    for (const auto *tensor : {q, k, v, beta, aqk, akk, dO, dqOut, dkOut, dvOut, dbOut, dgOut}) {

        if (tensor == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT_COMPAT(CHUNK_KDA_BWD_ENTITY, "required tensor");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (!(chunkSize == 64 && useExp2 && !stateVFirst)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "chunkSize/useExp2/stateVFirst",
            "not matching", "C=64, use_exp2=true, K-first state");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (initialStateOptional || dhtOptional || dh0OutOptional) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "initialState/dht/dh0", "present", "not supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((cuSeqlensOptional == nullptr) != (chunkIndicesOptional == nullptr)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "cuSeqlens/chunkIndices", "not together", "supply both");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((qRstdOptional == nullptr) != (kRstdOptional == nullptr)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(CHUNK_KDA_BWD_ENTITY, "qRstdOptional,kRstdOptional", "not together", "must appear together");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (allowNegEigval && !useBetaSigmoidInKernel) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "allowNegEigval", "true", "requires useBetaSigmoidInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(std::isfinite(static_cast<float>(scale)) && std::isfinite(static_cast<float>(lowerBound)))) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "scale/lowerBound", "non-finite", "finite");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool variable = cuSeqlensOptional != nullptr;
    const bool packed = layout && (std::strcmp(layout, "NTD") == 0 || std::strcmp(layout, "TND") == 0);
    const bool inputTokenMajor = layout && (std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "TND") == 0);
    if (!(layout && (packed || std::strcmp(layout, "BNSD") == 0 || std::strcmp(layout, "BSND") == 0))) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "layout",
            layout ? layout : "null", "BNSD, BSND, NTD or TND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto &vs = v->GetViewShape();
    if (!(vs.GetDimNum() == (packed ? 3U : 4U) && (!packed || variable))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(CHUNK_KDA_BWD_ENTITY, "v",
            "rank=" + std::to_string(vs.GetDimNum()), "3D calls require cu_seqlens");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (variable && !packed && vs.GetDim(0) != 1) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "v.shape[0]", std::to_string(vs.GetDim(0)),
            "1 (4D varlen calls require B=1)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const size_t base = packed ? 0 : 1;
    const int64_t B = packed ? 1 : vs.GetDim(0);
    const int64_t H = vs.GetDim(base + (inputTokenMajor ? 1 : 0));
    const int64_t T = vs.GetDim(base + (inputTokenMajor ? 0 : 1));
    if (!(B > 0 && H > 0 && T > 0 && vs.GetDim(base + 2) == 128)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "B/H/T/K", "non-positive or K!=128", "positive B/H/T and K=V=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (T > (std::numeric_limits<int64_t>::max() - 63) / 128 / 4 / H / B) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "T", std::to_string(T), "in int64 range");
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
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "q,k,v,dO,aqk,akk", "q=" + DfxTensorShapeOf(q) + "/" + DfxTensorDtypeOf(q) + ",k=" + DfxTensorShapeOf(k) + "/" + DfxTensorDtypeOf(k) + ",v=" + DfxTensorShapeOf(v) + "/" + DfxTensorDtypeOf(v) + ",dO=" + DfxTensorShapeOf(dO) + "/" + DfxTensorDtypeOf(dO) + ",aqk=" + DfxTensorShapeOf(aqk) + "/" + DfxTensorDtypeOf(aqk) + ",akk=" + DfxTensorShapeOf(akk) + "/" + DfxTensorDtypeOf(akk),
            "BF16 fixed head-major caches");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MatchesTensor(beta, betaShape, DataType::DT_BF16) || MatchesTensor(beta, betaShape, DataType::DT_FLOAT))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "beta", DfxTensorShapeOf(beta) + "/" + DfxTensorDtypeOf(beta), "token-major BF16/FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useBetaSigmoidInKernel
            ? !MatchesTensor(betaEffOptional, scalar, DataType::DT_FLOAT)
            : betaEffOptional != nullptr) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "betaEffOptional", DfxTensorShapeOf(betaEffOptional) + "/" + DfxTensorDtypeOf(betaEffOptional),
            "FP32 head-major when useBetaSigmoidInKernel, else null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qRstdOptional && !(MatchesTensor(qRstdOptional, scalar, DataType::DT_FLOAT) &&
                            MatchesTensor(kRstdOptional, scalar, DataType::DT_FLOAT))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "qRstd,kRstd", "qRstd=" + DfxTensorShapeOf(qRstdOptional) + ",kRstd=" + DfxTensorShapeOf(kRstdOptional) + "/" + DfxTensorDtypeOf(kRstdOptional), "head-major FP32 pair");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MatchesTensor(dqOut, outputToken, q->GetDataType()) &&
          MatchesTensor(dkOut, outputToken, k->GetDataType()))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "dq/dk", "dq=" + DfxTensorShapeOf(dqOut) + "/" + DfxTensorDtypeOf(dqOut) + ",dk=" + DfxTensorShapeOf(dkOut) + "/" + DfxTensorDtypeOf(dkOut), "match layout and q/k dtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(MatchesTensor(dvOut, outputToken, DataType::DT_BF16) &&
          MatchesTensor(dbOut, gateScalar, beta->GetDataType()) &&
          MatchesTensor(dgOut, gateToken, rawGOptional ? rawGOptional->GetDataType() : DataType::DT_FLOAT))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "dvOut,dbOut,dgOut", "dv=" + DfxTensorShapeOf(dvOut) + "/" + DfxTensorDtypeOf(dvOut) + ",db=" + DfxTensorShapeOf(dbOut) + "/" + DfxTensorDtypeOf(dbOut) + ",dg=" + DfxTensorShapeOf(dgOut) + "/" + DfxTensorDtypeOf(dgOut), "match layout");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (useGateInKernel &&
        !(((MatchesTensor(rawGOptional, gateToken, DataType::DT_FLOAT) || MatchesTensor(rawGOptional, gateToken, DataType::DT_BF16)) &&
           MatchesTensor(aLogOptional, MakeShape({H}), DataType::DT_FLOAT)))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "rawGOptional,aLogOptional", "rawG=" + DfxTensorShapeOf(rawGOptional) + "/" + DfxTensorDtypeOf(rawGOptional) + ",aLog=" + DfxTensorShapeOf(aLogOptional) + "/" + DfxTensorDtypeOf(aLogOptional),
            "raw_g BF16/FP32 token-major and a_log FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dtBiasOptional && !(dtBiasOptional->GetDataType() == DataType::DT_FLOAT && IsContiguous(dtBiasOptional) &&
                             dtBiasOptional->GetViewShape().GetShapeSize() == H * 128)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "dtBiasOptional", DfxTensorShapeOf(dtBiasOptional) + "/" + DfxTensorDtypeOf(dtBiasOptional), "contiguous FP32 H*K");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dAOutOptional && !(aLogOptional && MatchesTensor(dAOutOptional, MakeShape({H}), DataType::DT_FLOAT))) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "dAOutOptional", DfxTensorShapeOf(dAOutOptional) + "/" + DfxTensorDtypeOf(dAOutOptional), "[H] FP32 with aLog");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (dBiasOutOptional && !(dtBiasOptional && IsContiguous(dBiasOutOptional) &&
                               dBiasOutOptional->GetDataType() == DataType::DT_FLOAT &&
                               dBiasOutOptional->GetViewShape().GetShapeSize() == H * 128)) {
        OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "dBiasOutOptional", DfxTensorShapeOf(dBiasOutOptional) + "/" + DfxTensorDtypeOf(dBiasOutOptional), "contiguous FP32 H*K");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t nc = (T + 63) / 64;
    if (variable) {
        const auto *cu = cuSeqlensOptional;
        const auto *indices = chunkIndicesOptional;
        if (!(cu->Size() >= 2 && cu->Size() <= 1025 && (*cu)[0] == 0 && (*cu)[cu->Size()-1] == T)) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "cuSeqlensOptional", "invalid", "2..1025, [0]=0, [last]=T");
            return ACLNN_ERR_PARAM_INVALID;
        }
        nc = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t begin = (*cu)[seq], end = (*cu)[seq+1];
            if (!(begin >= 0 && end >= begin && end <= T)) {
                OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "cuSeqlensOptional", "invalid bounds", "0<=begin<=end<=T");
                return ACLNN_ERR_PARAM_INVALID;
            }
            for (int64_t c = 0; c < (end - begin + 63) / 64; ++c, ++nc) {
                if (!(static_cast<size_t>(2*nc+1) < indices->Size() &&
                      (*indices)[2*nc] == static_cast<int64_t>(seq) && (*indices)[2*nc+1] == c)) {
                    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "chunkIndicesOptional", "non-canonical", "sequence-major");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
        if (indices->Size() != static_cast<size_t>(2*nc)) {
            OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(CHUNK_KDA_BWD_ENTITY, "chunkIndicesOptional",
                std::to_string(indices->Size()), std::to_string(2*nc));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (!(nc > 0 && nc <= std::numeric_limits<int64_t>::max() / 128 / 128 / 2 / H / B)) {
        OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "nc", std::to_string(nc), "positive, in int64 range");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto state = packed ? MakeShape({nc,H,128,128}) : MakeShape({B,nc,H,128,128});
    if (disableRecompute) {
        for (const auto *cache : {wOptional, qgOptional, kgOptional, vNewOptional}) {
            if (!MatchesTensor(cache, token, DataType::DT_BF16)) {
                OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "saved cache", "shape=" + DfxTensorShapeOf(cache) + "/" + DfxTensorDtypeOf(cache), "BF16 token cache");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (!(MatchesTensor(gk, token, DataType::DT_FLOAT) && MatchesTensor(hOptional, state, DataType::DT_BF16))) {
            OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(CHUNK_KDA_BWD_ENTITY, "gk,hOptional", "gk=" + DfxTensorShapeOf(gk) + "/" + DfxTensorDtypeOf(gk) + ",h=" + DfxTensorShapeOf(hOptional) + "/" + DfxTensorDtypeOf(hOptional), "saved gk FP32 and chunk-major h BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (gk || wOptional || qgOptional || kgOptional || vNewOptional || hOptional) {
            OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "saved caches", "present", "must be absent for recompute");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get(), ACLNN_ERR_INNER_CREATE_EXECUTOR);
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
    CHECK_RET(betaHead, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *aLog = aLogOptional;
    const aclTensor *dtBias = dtBiasOptional;
    if (!dtBias) {
        const std::vector<float> zeros(static_cast<size_t>(H * 128), 0.0f);
        const auto *flat = ex->ConvertToTensor(zeros.data(), zeros.size(), DataType::DT_FLOAT);
        CHECK_RET(flat, ACLNN_ERR_INNER_NULLPTR);
        dtBias = l0op::Reshape(flat, MakeShape({H,128}), ex);
    } else {
        dtBias = l0op::Reshape(dtBias, MakeShape({H,128}), ex);
    }
    CHECK_RET(dtBias, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *w = wOptional, *qg = qgOptional, *kg = kgOptional;
    const aclTensor *vNew = vNewOptional, *h = hOptional;
    if (!disableRecompute) {
        // Recompute promotes BF16 raw gates to FP32 on-chip.
        const auto *rawRecompute = rawGOptional;
        const auto *logRecompute = aLog;
        const auto *biasRecompute = dtBias;
        CHECK_RET(rawRecompute && logRecompute && biasRecompute, ACLNN_ERR_INNER_NULLPTR);
        w = alloc(token, DataType::DT_BF16);
        qg = alloc(token, DataType::DT_BF16);
        kg = alloc(token, DataType::DT_BF16);
        vNew = alloc(token, DataType::DT_BF16);
        h = alloc(state, DataType::DT_BF16);
        gk = alloc(token, DataType::DT_FLOAT);
        const auto *u = alloc(token, DataType::DT_BF16);
        CHECK_RET(w && qg && kg && vNew && h && gk && u, ACLNN_ERR_INNER_NULLPTR);
        const auto result = l0op::ChunkKdaBwdRecompute(
            BatchView(q, packed, ex), BatchView(k, packed, ex), BatchView(v, packed, ex),
            BatchView(rawRecompute, packed, ex), BatchView(betaHead, packed, ex), BatchView(akk, packed, ex),
            logRecompute, biasRecompute, cuSeqlensOptional, chunkIndicesOptional, chunkSize, useGateInKernel,
            useExp2, lowerBound, packed ? "NTD" : "BNSD", BatchView(w, packed, ex), BatchView(u, packed, ex),
            BatchView(qg, packed, ex), BatchView(kg, packed, ex), BatchView(gk, packed, ex), ex, layout, qRstdOptional != nullptr);
        for (const auto *value : result) {
            CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
        }
        const auto forward = l0op::ChunkFwdH(
            BatchView(kg, packed, ex), BatchView(w, packed, ex), BatchView(u, packed, ex),
            nullptr, BatchView(gk, packed, ex), nullptr, cuSeqlensOptional, chunkIndicesOptional,
            false, chunkSize, true, useExp2, stateVFirst,
            BatchView(h, packed, ex), BatchView(vNew, packed, ex), nullptr, ex);
        CHECK_RET(forward[0] && forward[1], ACLNN_ERR_INNER_NULLPTR);
    }
    const auto *dAqk = alloc(matrix, DataType::DT_FLOAT);
    const auto *dvLocal = alloc(token, DataType::DT_BF16);
    const auto *dqRaw = alloc(token, DataType::DT_FLOAT);
    const auto *dh = alloc(state, DataType::DT_BF16);
    const auto *dvScan = alloc(token, DataType::DT_BF16);
    CHECK_RET(dAqk && dvLocal && dqRaw && dh && dvScan, ACLNN_ERR_INNER_NULLPTR);
    // Public do is token-major; Prepare and Dhu share one head-major copy.
    const auto *dOHead = SwapHeadToken(dO, packed, ex);
    CHECK_RET(dOHead, ACLNN_ERR_INNER_NULLPTR);
    const auto prepare = l0op::ChunkKdaBwdPrepare(
        aqk, vNew, dOHead, h,
        cuSeqlensOptional, chunkIndicesOptional, packed ? "NTD" : "BNSD", scale, chunkSize, stateVFirst,
        dAqk, dvLocal, dqRaw, ex);
    for (const auto *value : prepare) {
        CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
    }
    const auto scan = l0op::ChunkGatedDeltaRuleBwdDhu(
        BatchView(qg, packed, ex), BatchView(kg, packed, ex), BatchView(w, packed, ex),
        BatchView(dOHead, packed, ex), BatchView(dvLocal, packed, ex), nullptr,
        BatchView(gk, packed, ex), nullptr, nullptr, cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useExp2, false, BatchView(dh, packed, ex), nullptr,
        BatchView(dvScan, packed, ex), ex);
    CHECK_RET(scan[0] && scan[2], ACLNN_ERR_INNER_NULLPTR);
    // Finalize converts original gate parameters on-chip.
    const auto *betaFinal = betaHead;
    CHECK_RET(betaFinal && (!rawGOptional || rawGOptional) && (!aLogOptional || aLog),
              ACLNN_ERR_INNER_NULLPTR);
    // Finalize writes q/k/v gradients in layout and gate gradients token-major.
    const auto *dbFinal = dbOut;
    const auto *dA = dAOutOptional ? dAOutOptional : alloc(MakeShape({H}), DataType::DT_FLOAT);
    const auto *dBias = alloc(MakeShape({H,128}), DataType::DT_FLOAT);
    CHECK_RET(dbFinal && dA && dBias, ACLNN_ERR_INNER_NULLPTR);
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
        CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_COND(!dBiasOutOptional || CopyGradient(dBias, dBiasOutOptional, ex),
               ACLNN_ERR_INNER_NULLPTR, "Unable to adapt KDA backward outputs.");
#else

    // A5 (non-arch22) backward is not implemented. Keep failing fast regardless
    // of the capability combination: issue #991 requires a pure DFX remediation
    // not to enable previously-failing kernel paths.
    OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(CHUNK_KDA_BWD_ENTITY, "interface capabilities",
        "A5 backward not supported", "implemented on A2/A3 (arch22) only");
    return ACLNN_ERR_PARAM_INVALID;

    // A5 (non-arch22) backward is not implemented; the legacy unreachable
    // validation/launch path that followed this early return is removed.

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
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwd executor launch failed.");
    return ACLNN_SUCCESS;
}
