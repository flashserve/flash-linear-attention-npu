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
#include "aclnn_kernels/reshape.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace op;
namespace {
op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape result;
    for (auto dim : dims) result.AppendDim(dim);
    return result;
}

bool MatchesTensor(const aclTensor *tensor, const op::Shape &shape, DataType dtype)
{
    return tensor != nullptr && tensor->GetDataType() == dtype &&
        tensor->GetViewShape() == shape && IsContiguous(tensor);
}

const aclTensor *AllocTensor(aclOpExecutor *executor, const op::Shape &shape, DataType dtype)
{
    return executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
}

const aclTensor *ToFloat32(const aclTensor *tensor, aclOpExecutor *executor)
{
    return tensor->GetDataType() == DataType::DT_FLOAT ? tensor :
        l0op::Cast(tensor, DataType::DT_FLOAT, executor);
}

const aclTensor *BatchView(const aclTensor *tensor, bool packed, aclOpExecutor *executor)
{
    if (!packed) return tensor;
    op::Shape shape;
    shape.AppendDim(1);
    const auto &original = tensor->GetViewShape();
    for (size_t i = 0; i < original.GetDimNum(); ++i) shape.AppendDim(original.GetDim(i));
    return l0op::Reshape(tensor, shape, executor);
}
} // namespace

extern "C" aclnnStatus aclnnChunkKdaBwdV2GetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *beta, const aclTensor *gk,
    const aclTensor *aqk, const aclTensor *akk,
    const aclTensor *w, const aclTensor *qg, const aclTensor *kg,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dO,
    const aclTensor *rawG, const aclTensor *aLog, const aclTensor *dtBias,
    const aclTensor *initialState, const aclTensor *dht,
    const aclIntArray *cu, const aclIntArray *indices,
    double scale, int64_t chunkSize, bool safeGate, bool useGateInKernel,
    double lowerBound, bool disableRecompute, bool useExp2, bool stateVFirst,
    const aclTensor *qRstd, const aclTensor *kRstd,
    const aclTensor *dq, const aclTensor *dk, const aclTensor *dv,
    const aclTensor *db, const aclTensor *dg, const aclTensor *dh0,
    const aclTensor *dA, const aclTensor *dBias,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKdaBwdV2,
        DFX_IN(q, k, v, beta, gk, aqk, akk, w, qg, kg, vNew, h, dO,
               rawG, aLog, dtBias, initialState, dht, cu, indices, scale,
               chunkSize, safeGate, useGateInKernel, lowerBound,
               disableRecompute, useExp2, stateVFirst, qRstd, kRstd),
        DFX_OUT(dq, dk, dv, db, dg, dh0, dA, dBias));
    CHECK_COND(workspaceSize && executor && q, ACLNN_ERR_PARAM_NULLPTR, "V2 required argument is null.");
    const char *soc = aclrtGetSocName();
    CHECK_COND(soc && std::strstr(soc, "Ascend950"), ACLNN_ERR_PARAM_INVALID, "V2 requires Ascend950.");
    CHECK_COND(chunkSize == 64 && safeGate && useGateInKernel && useExp2 && !stateVFirst,
        ACLNN_ERR_PARAM_INVALID, "V2 requires C=64, safe gate, gate-in-kernel, exp2, K-first state.");
    CHECK_COND(std::isfinite(static_cast<float>(scale)) && std::isfinite(lowerBound) && lowerBound < 0 && lowerBound >= -5,
        ACLNN_ERR_PARAM_INVALID, "Invalid scale or safe-gate lower_bound.");
    CHECK_COND(!initialState && !dht && !dh0, ACLNN_ERR_PARAM_INVALID, "V2 state inputs/outputs are not supported.");
    CHECK_COND((cu == nullptr) == (indices == nullptr), ACLNN_ERR_PARAM_INVALID, "Supply both metadata arrays.");
    const bool packed = cu != nullptr;
    const auto &qs = q->GetViewShape();
    CHECK_COND(qs.GetDimNum() == (packed ? 3U : 4U), ACLNN_ERR_PARAM_INVALID, "V2 token rank mismatch.");
    const int64_t B = packed ? 1 : qs.GetDim(0);
    const int64_t H = qs.GetDim(packed ? 0 : 1);
    const int64_t T = qs.GetDim(packed ? 1 : 2);
    CHECK_COND(B > 0 && H > 0 && T > 0 && qs.GetDim(packed ? 2 : 3) == 128,
        ACLNN_ERR_PARAM_INVALID, "V2 requires positive B/H/T and K=V=128.");
    CHECK_COND(H <= std::numeric_limits<uint32_t>::max() / T &&
        B <= std::numeric_limits<uint32_t>::max() / (H * T),
        ACLNN_ERR_PARAM_INVALID, "V2 token indexing exceeds uint32 range.");
    // Recompute copies A_log in groups of eight into a 256-float buffer.
    CHECK_COND(disableRecompute || (H <= 256 && H % 8 == 0), ACLNN_ERR_PARAM_INVALID,
        "Recompute currently requires H<=256 and H divisible by 8.");
    const auto token = packed ? MakeShape({H,T,128}) : MakeShape({B,H,T,128});
    const auto scalar = packed ? MakeShape({H,T}) : MakeShape({B,H,T});
    const auto matrix = packed ? MakeShape({H,T,64}) : MakeShape({B,H,T,64});
    for (const auto *x : {q,k,v,dO,dq,dk,dv}) {
        CHECK_COND(MatchesTensor(x, token, DataType::DT_BF16), ACLNN_ERR_PARAM_INVALID, "V2 token shape/dtype/stride mismatch.");
    }
    CHECK_COND(MatchesTensor(aqk,matrix,DataType::DT_BF16) && MatchesTensor(akk,matrix,DataType::DT_BF16),
        ACLNN_ERR_PARAM_INVALID, "Aqk/Akk must be BF16 token-row matrices.");
    CHECK_COND(beta && (MatchesTensor(beta,scalar,DataType::DT_BF16) || MatchesTensor(beta,scalar,DataType::DT_FLOAT)) &&
        MatchesTensor(db,scalar,beta->GetDataType()), ACLNN_ERR_PARAM_INVALID, "dbeta must match beta dtype/shape.");
    CHECK_COND(MatchesTensor(rawG,token,DataType::DT_FLOAT) || MatchesTensor(rawG,token,DataType::DT_BF16),
        ACLNN_ERR_PARAM_INVALID, "raw_g must match token shape, BF16/FP32.");
    CHECK_COND(MatchesTensor(dg,token,DataType::DT_FLOAT), ACLNN_ERR_PARAM_INVALID, "dg must be FP32.");
    CHECK_COND(MatchesTensor(aLog,MakeShape({H}),DataType::DT_FLOAT) || MatchesTensor(aLog,MakeShape({H}),DataType::DT_BF16),
        ACLNN_ERR_PARAM_INVALID, "A_log must be [H], BF16/FP32.");
    CHECK_COND(!dtBias || MatchesTensor(dtBias,MakeShape({H,128}),DataType::DT_FLOAT), ACLNN_ERR_PARAM_INVALID,
        "V2 dt_bias must be FP32 [H,128] when supplied.");
    CHECK_COND((!qRstd && !kRstd) || (MatchesTensor(qRstd,scalar,DataType::DT_FLOAT) && MatchesTensor(kRstd,scalar,DataType::DT_FLOAT)),
        ACLNN_ERR_PARAM_INVALID, "rstd must be a matching FP32 pair.");
    CHECK_COND((!dA || MatchesTensor(dA,MakeShape({H}),DataType::DT_FLOAT)) &&
        (!dBias || MatchesTensor(dBias,MakeShape({H,128}),DataType::DT_FLOAT)), ACLNN_ERR_PARAM_INVALID, "Parameter output shape/dtype mismatch.");
    int64_t nc = (T + 63) / 64;
    if (packed) {
        CHECK_COND(cu->Size() >= 2 && (*cu)[0] == 0 && (*cu)[cu->Size()-1] == T,
            ACLNN_ERR_PARAM_INVALID, "Invalid cu_seqlens endpoints.");
        nc = 0;
        for (size_t s = 0; s + 1 < cu->Size(); ++s) {
            CHECK_COND((*cu)[s] >= 0 && (*cu)[s] < (*cu)[s+1] && (*cu)[s+1] <= T,
                ACLNN_ERR_PARAM_INVALID, "V2 expects nonempty sequences; wrapper compacts empty entries.");
            const int64_t count = ((*cu)[s+1]-(*cu)[s]+63)/64;
            for (int64_t c = 0; c < count; ++c, ++nc) {
                CHECK_COND(static_cast<size_t>(2*nc+1) < indices->Size() &&
                    (*indices)[2*nc] == static_cast<int64_t>(s) && (*indices)[2*nc+1] == c,
                    ACLNN_ERR_PARAM_INVALID, "Expected canonical sequence-major chunk_indices.");
            }
        }
        CHECK_COND(indices->Size() == static_cast<size_t>(2*nc), ACLNN_ERR_PARAM_INVALID, "Extra chunk indices.");
    }
    const auto state = packed ? MakeShape({H,nc,128,128}) : MakeShape({B,H,nc,128,128});
    const auto savedHShape = packed ? MakeShape({nc,H,128,128}) : MakeShape({B,nc,H,128,128});
    if (disableRecompute) {
        for (const auto *x : {w,qg,kg,vNew}) {
            CHECK_COND(MatchesTensor(x,token,DataType::DT_BF16), ACLNN_ERR_PARAM_INVALID, "Saved token cache is invalid.");
        }
        CHECK_COND(MatchesTensor(h,savedHShape,DataType::DT_BF16) && MatchesTensor(gk,token,DataType::DT_FLOAT),
            ACLNN_ERR_PARAM_INVALID, "Expected forward chunk-major h and FP32 gk caches.");
    } else {
        CHECK_COND(!w && !qg && !kg && !vNew && !h && !gk, ACLNN_ERR_PARAM_INVALID,
            "Recompute mode requires saved caches to be absent.");
    }
    auto owner = CREATE_EXECUTOR();
    CHECK_RET(owner.get(), ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto *ex = owner.get();
    if (!dtBias) {
        const std::vector<float> zeros(static_cast<size_t>(H * 128), 0.0f);
        const auto *flat = ex->ConvertToTensor(zeros.data(), zeros.size(), DataType::DT_FLOAT);
        CHECK_RET(flat, ACLNN_ERR_INNER_NULLPTR);
        dtBias = l0op::Reshape(flat, MakeShape({H,128}), ex);
        CHECK_RET(dtBias, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!dA) {
        dA = AllocTensor(ex, MakeShape({H}), DataType::DT_FLOAT);
    }
    if (!dBias) {
        dBias = AllocTensor(ex, MakeShape({H,128}), DataType::DT_FLOAT);
    }
    CHECK_RET(dA && dBias, ACLNN_ERR_INNER_NULLPTR);
    if (!disableRecompute) {
        rawG = ToFloat32(rawG,ex);
        aLog = ToFloat32(aLog,ex);
        CHECK_RET(rawG && aLog, ACLNN_ERR_INNER_NULLPTR);
        w = AllocTensor(ex, token, DataType::DT_BF16);
        qg = AllocTensor(ex, token, DataType::DT_BF16);
        kg = AllocTensor(ex, token, DataType::DT_BF16);
        vNew = AllocTensor(ex, token, DataType::DT_BF16);
        h = AllocTensor(ex, savedHShape, DataType::DT_BF16);
        gk = AllocTensor(ex, token, DataType::DT_FLOAT);
        const auto *u = AllocTensor(ex, token, DataType::DT_BF16);
        CHECK_RET(w && qg && kg && vNew && h && gk && u, ACLNN_ERR_INNER_NULLPTR);
        const auto *q4 = BatchView(q, packed, ex);
        const auto *k4 = BatchView(k, packed, ex);
        const auto *v4 = BatchView(v, packed, ex);
        const auto *rawG4 = BatchView(rawG, packed, ex);
        const auto *beta3 = BatchView(beta, packed, ex);
        const auto *akk4 = BatchView(akk, packed, ex);
        const auto *w4 = BatchView(w, packed, ex);
        const auto *u4 = BatchView(u, packed, ex);
        const auto *qg4 = BatchView(qg, packed, ex);
        const auto *kg4 = BatchView(kg, packed, ex);
        const auto *gk4 = BatchView(gk, packed, ex);
        const auto *h5 = BatchView(h, packed, ex);
        const auto *vNew4 = BatchView(vNew, packed, ex);
        CHECK_RET(q4 && k4 && v4 && rawG4 && beta3 && akk4 && w4 && u4 && qg4 && kg4 && gk4 && h5 && vNew4,
            ACLNN_ERR_INNER_NULLPTR);
        const auto recomputeResult = l0op::ChunkKdaBwdRecompute(
            q4, k4, v4, rawG4, beta3, akk4, aLog, dtBias, cu, indices,
            64, true, true, lowerBound, w4, u4, qg4, kg4, gk4, ex);
        for (const auto *value : recomputeResult) {
            CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
        }
        const auto forwardResult = l0op::ChunkFwdH(
            kg4, w4, u4, nullptr, gk4, nullptr, cu, indices,
            false, 64, true, true, false, true, h5, vNew4, nullptr, ex);
        // finalStateOut is intentionally absent; only h and v_new are required.
        CHECK_RET(forwardResult[0] && forwardResult[1], ACLNN_ERR_INNER_NULLPTR);
        // h_chunk_major=true：FwdH 直接产出 chunk-major h（与 saved 模式供给的布局一致），
        // Prepare/Finalize 按 chunk-major 契约读取，链上不再需要 host Transpose。
    }
    const auto *dAqk=AllocTensor(ex,matrix,DataType::DT_FLOAT);
    const auto *dv0=AllocTensor(ex,token,DataType::DT_BF16);
    const auto *dqRaw=AllocTensor(ex,token,DataType::DT_FLOAT);
    const auto *dh=AllocTensor(ex,state,DataType::DT_BF16);
    const auto *dvScan=AllocTensor(ex,token,DataType::DT_BF16);
    CHECK_RET(dAqk && dv0 && dqRaw && dh && dvScan,ACLNN_ERR_INNER_NULLPTR);
    const auto prepareResult = l0op::ChunkKdaBwdPrepare(
        aqk, vNew, dO, h, cu, indices, scale, 64, false, dAqk, dv0, dqRaw, ex);
    for (const auto *value : prepareResult) {
        CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
    }
    const auto *qg4 = BatchView(qg, packed, ex);
    const auto *kg4 = BatchView(kg, packed, ex);
    const auto *w4 = BatchView(w, packed, ex);
    const auto *dO4 = BatchView(dO, packed, ex);
    const auto *dv04 = BatchView(dv0, packed, ex);
    const auto *gk4 = BatchView(gk, packed, ex);
    const auto *dh5 = BatchView(dh, packed, ex);
    const auto *dvScan4 = BatchView(dvScan, packed, ex);
    CHECK_RET(qg4 && kg4 && w4 && dO4 && dv04 && gk4 && dh5 && dvScan4, ACLNN_ERR_INNER_NULLPTR);
    const auto dhuResult = l0op::ChunkGatedDeltaRuleBwdDhu(
        qg4, kg4, w4, dO4, dv04, nullptr, gk4, nullptr, nullptr,
        cu, indices, scale, 64, true, false, dh5, nullptr, dvScan4, ex);
    CHECK_RET(dhuResult[0] && dhuResult[2], ACLNN_ERR_INNER_NULLPTR);
    // Saved mode needs these conversions only in finalize. Keep them at that
    // consumer, after the custom-op platform tiling has been initialized.
    // CANN 9.1 Cast otherwise reports coreNum=0 on a cold BF16-gate call.
    rawG = ToFloat32(rawG,ex);
    aLog = ToFloat32(aLog,ex);
    CHECK_RET(rawG && aLog, ACLNN_ERR_INNER_NULLPTR);
    const auto finalizeResult = l0op::ChunkKdaBwdFinalize(
        q, k, v, gk, rawG, beta, aLog, dtBias, akk, vNew, h, dh, dvScan, dAqk, dqRaw,
        qRstd, kRstd, cu, indices, scale, lowerBound, 64, true, true, true, false,
        dq, dk, dv, db, dg, dA, dBias, ex);
    for (const auto *value : finalizeResult) {
        CHECK_RET(value, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize=owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnChunkKdaBwdV2(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaBwdV2);
    return CommonOpExecutorRun(workspace,workspaceSize,executor,stream);
}
#endif
