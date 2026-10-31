// Copyright (c) 2026 Tianjin University, Ltd. BSD-3-Clause.
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
#include "opdev/tensor_view_utils.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

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

    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    CHECK_COND(useGateInKernel, ACLNN_ERR_PARAM_INVALID,
               "chunk_kda_bwd: use_gate_in_kernel must be true.");
    const char *soc = aclrtGetSocName();
    CHECK_COND(soc && std::strstr(soc, "Ascend950"), ACLNN_ERR_PARAM_INVALID,
               "V2 requires Ascend950/A5.");
    CHECK_COND(!betaEffOptional && !useBetaSigmoidInKernel && !allowNegEigval,
               ACLNN_ERR_PARAM_INVALID,
               "A5 beta sigmoid backward is reserved: beta_eff must be null and both flags false.");
    CHECK_COND(lowerBound >= -5.0 && lowerBound < 0.0,
               ACLNN_ERR_PARAM_INVALID, "A5 V2 requires -5<=lower_bound<0.");
    for (const auto *tensor : {q, k, v, beta, aqk, akk, dO, dqOut, dkOut, dvOut, dbOut, dgOut}) {
        CHECK_COND(tensor, ACLNN_ERR_PARAM_NULLPTR, "KDA backward required tensor is null.");
    }
    CHECK_COND(chunkSize == 64 && safeGate && useExp2 && !stateVFirst,
               ACLNN_ERR_PARAM_INVALID, "A5 V2 requires C=64, safe_gate=true, use_exp2=true and K-first state.");
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
    const bool variable = cuSeqlensOptional != nullptr;
    const bool packed = layout && (std::strcmp(layout, "NTD") == 0 || std::strcmp(layout, "TND") == 0);
    const bool inputTokenMajor = layout && (std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "TND") == 0);
    CHECK_COND(layout && (packed || std::strcmp(layout, "BNSD") == 0 || std::strcmp(layout, "BSND") == 0),
               ACLNN_ERR_PARAM_INVALID, "layout must be BNSD, BSND, NTD or TND.");
    const auto &vs = v->GetViewShape();
    CHECK_COND(vs.GetDimNum() == (packed ? 3U : 4U) && (!packed || variable),
               ACLNN_ERR_PARAM_INVALID, "Layout rank mismatch; 3D calls require cu_seqlens.");
    CHECK_COND(!variable || packed || vs.GetDim(0) == 1,
               ACLNN_ERR_PARAM_INVALID, "4D varlen calls require B=1.");
    const size_t base = packed ? 0 : 1;
    const int64_t B = packed ? 1 : vs.GetDim(0);
    const int64_t H = vs.GetDim(base + (inputTokenMajor ? 1 : 0));
    const int64_t T = vs.GetDim(base + (inputTokenMajor ? 0 : 1));
    CHECK_COND(B > 0 && H > 0 && T > 0 && vs.GetDim(base + 2) == 128,
               ACLNN_ERR_PARAM_INVALID, "Requires positive B/H/T and K=V=128.");
    CHECK_COND(T <= (std::numeric_limits<int64_t>::max() - 63) / 128 / 4 / H / B,
               ACLNN_ERR_PARAM_INVALID, "Token tensor byte size overflows.");
    // Keep the existing A5 indexing and recompute buffer limits.
    CHECK_COND(H <= std::numeric_limits<uint32_t>::max() / T &&
                   B <= std::numeric_limits<uint32_t>::max() / (H * T),
               ACLNN_ERR_PARAM_INVALID, "V2 token indexing exceeds uint32 range.");
    CHECK_COND(disableRecompute || (H <= 256 && H % 8 == 0),
               ACLNN_ERR_PARAM_INVALID, "Recompute requires H<=256 and H divisible by 8.");
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
    CHECK_COND(MatchesTensor(q, qRstdOptional ? token : externalToken, DataType::DT_BF16) &&
                   MatchesTensor(k, kRstdOptional ? token : externalToken, DataType::DT_BF16) &&
                   MatchesTensor(v, externalToken, DataType::DT_BF16) &&
                   MatchesTensor(dO, gateToken, DataType::DT_BF16) &&
                   MatchesTensor(aqk, matrix, DataType::DT_BF16) &&
                   MatchesTensor(akk, matrix, DataType::DT_BF16),
               ACLNN_ERR_PARAM_INVALID, "Invalid BF16 inputs or fixed head-major caches.");
    CHECK_COND(MatchesTensor(beta, betaShape, DataType::DT_BF16) ||
                   MatchesTensor(beta, betaShape, DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "beta must be the original token-major BF16/FP32 input.");
    CHECK_COND(useBetaSigmoidInKernel
                   ? MatchesTensor(betaEffOptional, scalar, DataType::DT_FLOAT)
                   : betaEffOptional == nullptr,
               ACLNN_ERR_PARAM_INVALID,
               "Fused sigmoid requires FP32 head-major beta_eff; otherwise beta_eff must be null.");
    CHECK_COND(!qRstdOptional ||
                   (MatchesTensor(qRstdOptional, scalar, DataType::DT_FLOAT) &&
                    MatchesTensor(kRstdOptional, scalar, DataType::DT_FLOAT)),
               ACLNN_ERR_PARAM_INVALID, "rstd must be a head-major FP32 pair.");
    CHECK_COND(MatchesTensor(dqOut, outputToken, q->GetDataType()) &&
                   MatchesTensor(dkOut, outputToken, k->GetDataType()),
               ACLNN_ERR_PARAM_INVALID, "dq/dk must match layout and q/k dtype.");
    CHECK_COND(MatchesTensor(dvOut, outputToken, DataType::DT_BF16) &&
                   MatchesTensor(dbOut, gateScalar, beta->GetDataType()) &&
                   MatchesTensor(dgOut, gateToken, rawGOptional ? rawGOptional->GetDataType() : DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "dv/db/dg must match layout.");
    CHECK_COND(!useGateInKernel ||
                   ((MatchesTensor(rawGOptional, gateToken, DataType::DT_FLOAT) ||
                     MatchesTensor(rawGOptional, gateToken, DataType::DT_BF16)) &&
                    MatchesTensor(aLogOptional, MakeShape({H}), DataType::DT_FLOAT)),
               ACLNN_ERR_PARAM_INVALID, "raw_g must be BF16/FP32 and a_log must be FP32 for gate backward.");
    CHECK_COND(!dtBiasOptional ||
                   (dtBiasOptional->GetDataType() == DataType::DT_FLOAT &&
                    IsContiguous(dtBiasOptional) && dtBiasOptional->GetViewShape().GetShapeSize() == H * 128),
               ACLNN_ERR_PARAM_INVALID, "dt_bias must be contiguous FP32 with H*K elements.");
    CHECK_COND(!dAOutOptional || (aLogOptional && MatchesTensor(dAOutOptional, MakeShape({H}), DataType::DT_FLOAT)),
               ACLNN_ERR_PARAM_INVALID, "Invalid dA output.");
    CHECK_COND(!dBiasOutOptional || (dtBiasOptional && IsContiguous(dBiasOutOptional) &&
                   dBiasOutOptional->GetDataType() == DataType::DT_FLOAT &&
                   dBiasOutOptional->GetViewShape().GetShapeSize() == H * 128),
               ACLNN_ERR_PARAM_INVALID, "Invalid dbias output.");
    int64_t nc = (T + 63) / 64;
    if (variable) {
        const auto *cu = cuSeqlensOptional;
        const auto *indices = chunkIndicesOptional;
        CHECK_COND(cu->Size() >= 2 &&
                       (*cu)[0] == 0 && (*cu)[cu->Size()-1] == T,
                   ACLNN_ERR_PARAM_INVALID, "Invalid cu_seqlens.");
        nc = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t begin = (*cu)[seq], end = (*cu)[seq+1];
            CHECK_COND(begin >= 0 && end > begin && end <= T,
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
    const auto prepare = l0op::ChunkKdaBwdPrepare(
        prepareView(aqk), prepareView(vNew), prepareView(dO), prepareView(h),
        cuSeqlensOptional, chunkIndicesOptional, variable ? "NTD" : "BNSD", scale, chunkSize, stateVFirst,
        prepareView(dAqk), prepareView(dvLocal), prepareView(dqRaw), ex, true);
    for (const auto *value : prepare) {
        CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
    }
    const auto scan = l0op::ChunkGatedDeltaRuleBwdDhu(
        BatchView(qg, packed, ex), BatchView(kg, packed, ex), BatchView(w, packed, ex),
        BatchView(dO, packed, ex), BatchView(dvLocal, packed, ex), nullptr,
        BatchView(gk, packed, ex), nullptr, nullptr, cuSeqlensOptional, chunkIndicesOptional,
        scale, chunkSize, useExp2, false, BatchView(dh, packed, ex), nullptr,
        BatchView(dvScan, packed, ex), ex, true);
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
