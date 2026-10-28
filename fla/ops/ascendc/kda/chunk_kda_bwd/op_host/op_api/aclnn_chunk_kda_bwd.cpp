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

const aclTensor *ToFloat32(const aclTensor *tensor, aclOpExecutor *executor)
{
    if (!tensor || tensor->GetDataType() == DataType::DT_FLOAT) return tensor;
    return l0op::Cast(tensor, DataType::DT_FLOAT, executor);
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
    const char *inputLayout, const char *outputLayout,
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
               stateVFirst, qRstdOptional, kRstdOptional, inputLayout,
               outputLayout, useBetaSigmoidInKernel, allowNegEigval),
        DFX_OUT(dqOut, dkOut, dvOut, dbOut, dgOut, dh0OutOptional,
                dAOutOptional, dBiasOutOptional));

    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    CHECK_COND(useGateInKernel, ACLNN_ERR_PARAM_INVALID,
               "chunk_kda_bwd: use_gate_in_kernel must be true.");
#ifdef KDA_BWD_STAGES_ARCH22
    for (const auto *tensor : {q, k, v, beta, aqk, akk, dO, dqOut, dkOut, dvOut, dbOut, dgOut}) {
        CHECK_COND(tensor, ACLNN_ERR_PARAM_NULLPTR, "KDA backward required tensor is null.");
    }
    CHECK_COND(chunkSize == 64 && useExp2 && !stateVFirst,
               ACLNN_ERR_PARAM_INVALID, "A2/A3 requires C=64, use_exp2=true and K-first state.");
    CHECK_COND(!initialStateOptional && !dhtOptional && !dh0OutOptional,
               ACLNN_ERR_PARAM_INVALID, "initial_state/dht/dh0 are not supported.");
    CHECK_COND((cuSeqlensOptional == nullptr) == (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "Supply both metadata arrays.");
    CHECK_COND((qRstdOptional == nullptr) == (kRstdOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "q_rstd and k_rstd must appear together.");
    CHECK_COND(!allowNegEigval || useBetaSigmoidInKernel,
               ACLNN_ERR_PARAM_INVALID, "allow_neg_eigval requires beta sigmoid backward.");
    CHECK_COND(std::isfinite(static_cast<float>(scale)) &&
                   std::isfinite(static_cast<float>(lowerBound)),
               ACLNN_ERR_PARAM_INVALID, "scale/lower_bound must be finite.");
    const bool packed = cuSeqlensOptional != nullptr;
    const bool inputTokenMajor = inputLayout &&
        std::strcmp(inputLayout, packed ? "TND" : "BSND") == 0;
    const bool outputTokenMajor = outputLayout &&
        std::strcmp(outputLayout, packed ? "TND" : "BSND") == 0;
    CHECK_COND(inputLayout && (inputTokenMajor ||
                   std::strcmp(inputLayout, packed ? "NTD" : "BNSD") == 0) &&
                   outputLayout && (outputTokenMajor ||
                   std::strcmp(outputLayout, packed ? "NTD" : "BNSD") == 0),
               ACLNN_ERR_PARAM_INVALID, "layout does not match dense/packed input.");
    CHECK_COND(disableRecompute || !inputTokenMajor, ACLNN_ERR_PARAM_INVALID,
               "Recompute input_layout must be BNSD or NTD.");
    const auto &vs = v->GetViewShape();
    CHECK_COND(vs.GetDimNum() == (packed ? 3U : 4U),
               ACLNN_ERR_PARAM_INVALID, "Invalid Token rank.");
    const size_t base = packed ? 0 : 1;
    const int64_t B = packed ? 1 : vs.GetDim(0);
    const int64_t H = vs.GetDim(base + (inputTokenMajor ? 1 : 0));
    const int64_t T = vs.GetDim(base + (inputTokenMajor ? 0 : 1));
    CHECK_COND(B > 0 && H > 0 && T > 0 && vs.GetDim(base + 2) == 128,
               ACLNN_ERR_PARAM_INVALID, "Requires positive B/H/T and K=V=128.");
    CHECK_COND(T <= (std::numeric_limits<int64_t>::max() - 63) / 128 / 4 / H / B,
               ACLNN_ERR_PARAM_INVALID, "Token tensor byte size overflows.");
    const auto token = packed ? MakeShape({H,T,128}) : MakeShape({B,H,T,128});
    const auto scalar = packed ? MakeShape({H,T}) : MakeShape({B,H,T});
    const auto matrix = packed ? MakeShape({H,T,64}) : MakeShape({B,H,T,64});
    const auto externalToken = inputTokenMajor ?
        (packed ? MakeShape({T,H,128}) : MakeShape({B,T,H,128})) : token;
    const auto outputToken = outputTokenMajor ?
        (packed ? MakeShape({T,H,128}) : MakeShape({B,T,H,128})) : token;
    const auto gateToken = outputToken;
    const auto gateScalar = outputTokenMajor ?
        (packed ? MakeShape({T,H}) : MakeShape({B,T,H})) : scalar;
    CHECK_COND(MatchesTensor(q, qRstdOptional ? token : externalToken, DataType::DT_BF16) &&
                   MatchesTensor(k, kRstdOptional ? token : externalToken, DataType::DT_BF16) &&
                   MatchesTensor(v, externalToken, DataType::DT_BF16) &&
                   MatchesTensor(dO, externalToken, DataType::DT_BF16) &&
                   MatchesTensor(aqk, matrix, DataType::DT_BF16) &&
                   MatchesTensor(akk, matrix, DataType::DT_BF16),
               ACLNN_ERR_PARAM_INVALID, "Invalid BF16 inputs or fixed head-major caches.");
    CHECK_COND(MatchesTensor(beta, scalar, DataType::DT_BF16) ||
                   MatchesTensor(beta, scalar, DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "beta must be head-major BF16/FP32 effective beta.");
    CHECK_COND(!qRstdOptional ||
                   (MatchesTensor(qRstdOptional, scalar, DataType::DT_FLOAT) &&
                    MatchesTensor(kRstdOptional, scalar, DataType::DT_FLOAT)),
               ACLNN_ERR_PARAM_INVALID, "rstd must be a head-major FP32 pair.");
    CHECK_COND(MatchesTensor(dqOut, outputToken, q->GetDataType()) &&
                   MatchesTensor(dkOut, outputToken, k->GetDataType()),
               ACLNN_ERR_PARAM_INVALID, "dq/dk must match output_layout and q/k dtype.");
    CHECK_COND(MatchesTensor(dvOut, outputToken, DataType::DT_BF16) &&
                   MatchesTensor(dbOut, gateScalar, beta->GetDataType()) &&
                   MatchesTensor(dgOut, gateToken, rawGOptional ? rawGOptional->GetDataType() : DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "dv/db/dg must match output_layout.");
    CHECK_COND(!useGateInKernel ||
                   ((MatchesTensor(rawGOptional, externalToken, DataType::DT_FLOAT) ||
                     MatchesTensor(rawGOptional, externalToken, DataType::DT_BF16)) &&
                    (MatchesTensor(aLogOptional, MakeShape({H}), DataType::DT_FLOAT) ||
                     MatchesTensor(aLogOptional, MakeShape({H}), DataType::DT_BF16))),
               ACLNN_ERR_PARAM_INVALID, "raw_g and a_log are required for gate backward.");
    CHECK_COND(!dtBiasOptional ||
                   ((dtBiasOptional->GetDataType() == DataType::DT_FLOAT || dtBiasOptional->GetDataType() == DataType::DT_BF16) &&
                    IsContiguous(dtBiasOptional) && dtBiasOptional->GetViewShape().GetShapeSize() == H * 128),
               ACLNN_ERR_PARAM_INVALID, "dt_bias must be contiguous BF16/FP32 with H*K elements.");
    CHECK_COND(!dAOutOptional || (aLogOptional && MatchesTensor(dAOutOptional, MakeShape({H}), aLogOptional->GetDataType())),
               ACLNN_ERR_PARAM_INVALID, "Invalid dA output.");
    CHECK_COND(!dBiasOutOptional || (dtBiasOptional && IsContiguous(dBiasOutOptional) &&
                   dBiasOutOptional->GetDataType() == dtBiasOptional->GetDataType() &&
                   dBiasOutOptional->GetViewShape().GetShapeSize() == H * 128),
               ACLNN_ERR_PARAM_INVALID, "Invalid dbias output.");
    int64_t nc = (T + 63) / 64;
    if (packed) {
        const auto *cu = cuSeqlensOptional;
        const auto *indices = chunkIndicesOptional;
        CHECK_COND(cu->Size() >= 2 && cu->Size() <= 1025 &&
                       (*cu)[0] == 0 && (*cu)[cu->Size()-1] == T,
                   ACLNN_ERR_PARAM_INVALID, "Invalid cu_seqlens.");
        nc = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t begin = (*cu)[seq], end = (*cu)[seq+1];
            CHECK_COND(begin >= 0 && end >= begin && end <= T,
                       ACLNN_ERR_PARAM_INVALID, "Invalid sequence boundaries.");
            for (int64_t c = 0; c < (end - begin + 63) / 64; ++c, ++nc) {
                CHECK_COND(static_cast<size_t>(2*nc+1) < indices->Size() &&
                               (*indices)[2*nc] == static_cast<int64_t>(seq) && (*indices)[2*nc+1] == c,
                           ACLNN_ERR_PARAM_INVALID, "Expected sequence-major chunk_indices.");
            }
        }
        CHECK_COND(indices->Size() == static_cast<size_t>(2*nc),
                   ACLNN_ERR_PARAM_INVALID, "Extra chunk indices.");
    }
    CHECK_COND(nc > 0 && nc <= std::numeric_limits<int64_t>::max() / 128 / 128 / 2 / H / B,
               ACLNN_ERR_PARAM_INVALID, "State tensor byte size overflows.");
    const auto state = packed ? MakeShape({nc,H,128,128}) : MakeShape({B,nc,H,128,128});
    if (disableRecompute) {
        for (const auto *cache : {wOptional, qgOptional, kgOptional, vNewOptional}) {
            CHECK_COND(MatchesTensor(cache, token, DataType::DT_BF16),
                       ACLNN_ERR_PARAM_INVALID, "Missing or invalid saved Token cache.");
        }
        CHECK_COND(MatchesTensor(gk, token, DataType::DT_FLOAT) &&
                       MatchesTensor(hOptional, state, DataType::DT_BF16),
                   ACLNN_ERR_PARAM_INVALID, "Expected saved gk and chunk-major h.");
    } else {
        CHECK_COND(!gk && !wOptional && !qgOptional && !kgOptional && !vNewOptional && !hOptional,
                   ACLNN_ERR_PARAM_INVALID, "Recompute requires the six saved caches to be absent.");
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get(), ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *ex = uniqueExecutor.get();
    NormalizeOriginalShapes({q, k, v, beta, gk, aqk, akk, wOptional, qgOptional,
        kgOptional, vNewOptional, hOptional, dO, rawGOptional, aLogOptional,
        dtBiasOptional, qRstdOptional, kRstdOptional, dqOut, dkOut, dvOut, dbOut,
        dgOut, dAOutOptional, dBiasOutOptional});
    auto alloc = [&](const op::Shape &shape, DataType type) {
        return ex->AllocTensor(shape, type, Format::FORMAT_ND);
    };
    const auto *qHead = inputTokenMajor && !qRstdOptional ? SwapHeadToken(q, packed, ex) : q;
    const auto *kHead = inputTokenMajor && !kRstdOptional ? SwapHeadToken(k, packed, ex) : k;
    const auto *vHead = inputTokenMajor ? SwapHeadToken(v, packed, ex) : v;
    const auto *dOHead = inputTokenMajor ? SwapHeadToken(dO, packed, ex) : dO;
    const auto *rawHead = inputTokenMajor && rawGOptional ? SwapHeadToken(rawGOptional, packed, ex) : rawGOptional;
    CHECK_RET(qHead && kHead && vHead && dOHead && (!rawGOptional || rawHead),
              ACLNN_ERR_INNER_NULLPTR);
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
        const auto *rawRecompute = ToFloat32(rawHead, ex);
        const auto *logRecompute = ToFloat32(aLog, ex);
        const auto *biasRecompute = ToFloat32(dtBias, ex);
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
            BatchView(qHead, packed, ex), BatchView(kHead, packed, ex), BatchView(vHead, packed, ex),
            BatchView(rawRecompute, packed, ex), BatchView(beta, packed, ex), BatchView(akk, packed, ex),
            logRecompute, biasRecompute, cuSeqlensOptional, chunkIndicesOptional, chunkSize, useGateInKernel,
            useExp2, lowerBound, inputLayout, BatchView(w, packed, ex), BatchView(u, packed, ex),
            BatchView(qg, packed, ex), BatchView(kg, packed, ex), BatchView(gk, packed, ex), ex);
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
    // Saved caches stay head-major. Prepare's three inputs must share its layout.
    const auto *aqkPrepare = disableRecompute && inputTokenMajor ? SwapHeadToken(aqk, packed, ex) : aqk;
    const auto *vNewPrepare = disableRecompute && inputTokenMajor ? SwapHeadToken(vNew, packed, ex) : vNew;
    CHECK_RET(aqkPrepare && vNewPrepare, ACLNN_ERR_INNER_NULLPTR);
    const char *prepareLayout = disableRecompute ? inputLayout : (packed ? "NTD" : "BNSD");
    const auto prepare = l0op::ChunkKdaBwdPrepare(
        aqkPrepare, vNewPrepare, disableRecompute ? dO : dOHead, h,
        cuSeqlensOptional, chunkIndicesOptional, prepareLayout, scale, chunkSize, stateVFirst,
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
    const auto *betaFinal = beta;
    CHECK_RET(betaFinal && (!rawGOptional || rawHead) && (!aLogOptional || aLog),
              ACLNN_ERR_INNER_NULLPTR);
    // Finalize writes all five gradients directly in outputLayout.
    const auto *dbFinal = dbOut;
    const auto *dA = dAOutOptional ? dAOutOptional : alloc(MakeShape({H}), aLogOptional->GetDataType());
    const auto *dBias = alloc(MakeShape({H,128}), dtBias->GetDataType());
    CHECK_RET(dbFinal && dA && dBias, ACLNN_ERR_INNER_NULLPTR);
    NormalizeOriginalShapes({qHead, kHead, vHead, gk, rawHead, betaFinal, aLog, dtBias,
        akk, vNew, h, dh, dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
        dqOut, dkOut, dvOut, dbFinal, dgOut, dA, dBias});
    const auto finalize = l0op::ChunkKdaBwdFinalize(
        qHead, kHead, vHead, gk, rawHead, betaFinal, aLog, dtBias,
        akk, vNew, h, dh, dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
        cuSeqlensOptional, chunkIndicesOptional, scale, lowerBound, chunkSize,
        safeGate, useGateInKernel, useExp2, stateVFirst,
        useBetaSigmoidInKernel, allowNegEigval, outputLayout,
        dqOut, dkOut, dvOut, dbFinal, dgOut, dA, dBias, ex);
    for (const auto *value : finalize) {
        CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_COND(!dBiasOutOptional || CopyGradient(dBias, dBiasOutOptional, ex),
               ACLNN_ERR_INNER_NULLPTR, "Unable to adapt KDA backward outputs.");
#else
    const bool packedLayout = cuSeqlensOptional != nullptr;
    CHECK_COND(inputLayout && outputLayout &&
                   std::strcmp(inputLayout, packedLayout ? "NTD" : "BNSD") == 0 &&
                   std::strcmp(outputLayout, packedLayout ? "NTD" : "BNSD") == 0 &&
                   !qRstdOptional && !kRstdOptional &&
                   !useBetaSigmoidInKernel && !allowNegEigval,
               ACLNN_ERR_PARAM_INVALID, "A5 original backward keeps its existing interface capabilities.");
    const aclTensor *required[] = {
        q, k, v, beta, gk, aqk, akk, dO,
        dqOut, dkOut, dvOut, dbOut, dgOut};
    for (const aclTensor *tensor : required) {
        CHECK_COND(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "ChunkKdaBwd required tensor is nullptr.");
    }
    CHECK_COND(chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "ChunkKdaBwd requires chunk_size=64.");
    CHECK_COND(disableRecompute, ACLNN_ERR_PARAM_INVALID,
               "disable_recompute=false is reserved but not supported.");
    CHECK_COND(useExp2, ACLNN_ERR_PARAM_INVALID,
               "use_exp2=false is reserved but not supported.");
    CHECK_COND(!stateVFirst, ACLNN_ERR_PARAM_INVALID,
               "state_v_first=true is reserved but not supported.");
    CHECK_COND(wOptional != nullptr && qgOptional != nullptr &&
                   kgOptional != nullptr && vNewOptional != nullptr &&
                   hOptional != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "w, qg, kg, v_new and h are required when disable_recompute=true.");
    CHECK_COND((cuSeqlensOptional == nullptr) ==
                   (chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "cu_seqlens and chunk_indices must be supplied together.");
    // PR291 has not completed the reviewed dht/dh0 semantics. Keep the
    // public composition honest until that implementation is repaired.
    CHECK_COND(initialStateOptional == nullptr && dhtOptional == nullptr &&
                   dh0OutOptional == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "initial_state, dht and dh0 are not supported by this PR291 integration.");
    CHECK_COND(!useGateInKernel ||
                   (rawGOptional != nullptr && aLogOptional != nullptr &&
                    dAOutOptional != nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "raw_g, a_log and dA are required for raw-gate backward.");
    CHECK_COND(useGateInKernel ||
                   (rawGOptional == nullptr && aLogOptional == nullptr &&
                    dtBiasOptional == nullptr && dAOutOptional == nullptr &&
                    dBiasOutOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID,
               "raw_g, a_log, dt_bias, dA and dbias require use_gate_in_kernel=true.");
    CHECK_COND(!useGateInKernel || dtBiasOptional != nullptr ||
                   dBiasOutOptional == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "dt_bias is required when dbias output is requested.");
    CHECK_COND(!useGateInKernel || dtBiasOptional == nullptr ||
                   dBiasOutOptional != nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "dbias output is required when dt_bias is present.");
    const op::Shape &qShape = q->GetViewShape();

    const op::Shape &hShape = hOptional->GetViewShape();
    const bool isVarLen = cuSeqlensOptional != nullptr;
    CHECK_COND(qShape.GetDimNum() == (isVarLen ? 3U : 4U) &&
                   hShape.GetDimNum() == (isVarLen ? 4U : 5U),
               ACLNN_ERR_PARAM_INVALID,
               "canonical dense q/h ranks are 4/5 and varlen ranks are 3/4.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *executorPtr = uniqueExecutor.get();

    const aclTensor *cuTensor =
        ConvertIntArrayToTensor(cuSeqlensOptional, executorPtr);
    const aclTensor *chunkTensor =
        ConvertIntArrayToTensor(chunkIndicesOptional, executorPtr);
    CHECK_COND(!isVarLen || (cuTensor != nullptr && chunkTensor != nullptr),
               ACLNN_ERR_INNER_NULLPTR,
               "convert varlen metadata to tensors failed.");

    // Keep the device ABI stable when optional outputs are absent.  The
    // current CANN optional-output launcher does not materialize a nullptr
    // placeholder reliably, which would shift workspace/tiling arguments.
    // These aliases are never written unless the corresponding feature is
    // enabled, in which case the public validation above requires real
    // dA/dbias outputs.
    // Legacy A5 keeps its FP32 device ABI. Adapt at L2; V2 writes the
    // requested dtype directly in Finalize and does not take this fallback.
    CHECK_COND(dbOut->GetDataType() == beta->GetDataType() &&
                   (!useGateInKernel || (dgOut->GetDataType() == rawGOptional->GetDataType() &&
                    dAOutOptional->GetDataType() == aLogOptional->GetDataType())),
               ACLNN_ERR_PARAM_INVALID, "gate gradients must match their input dtypes.");
    auto fp32Output = [&](const aclTensor *out) -> const aclTensor * {
        if (!out || out->GetDataType() == DataType::DT_FLOAT) return out;
        return executorPtr->AllocTensor(out->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);
    };
    const auto *dbKernel = fp32Output(dbOut);
    const auto *dgKernel = fp32Output(dgOut);
    const auto *dAKernel = fp32Output(dAOutOptional);
    CHECK_RET(dbKernel && dgKernel && (!dAOutOptional || dAKernel), ACLNN_ERR_INNER_NULLPTR);
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
        CHECK_RET(result[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(!useGateInKernel || result[5] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dtBiasOptional == nullptr || result[6] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    for (const auto &pair : {std::pair<const aclTensor *, const aclTensor *>{dbKernel, dbOut},
                            {dgKernel, dgOut}, {dAKernel, dAOutOptional}}) {
        if (!pair.second || pair.first == pair.second) continue;
        const auto *converted = l0op::Cast(pair.first, pair.second->GetDataType(), executorPtr);
        CHECK_RET(converted && l0op::ViewCopy(converted, pair.second, executorPtr), ACLNN_ERR_INNER_NULLPTR);
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
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaBwd executor launch failed.");
    return ACLNN_SUCCESS;
}
