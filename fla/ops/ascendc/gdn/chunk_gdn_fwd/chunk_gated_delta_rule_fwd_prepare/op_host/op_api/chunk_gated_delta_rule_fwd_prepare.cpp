/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#include "chunk_gated_delta_rule_fwd_prepare.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwdPrepare);

namespace {

const aclTensor *ConvertIntArrayToTensor(const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    const_cast<aclTensor *>(tensor)->SetStorageFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetViewFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}

} // namespace

const std::array<const aclTensor *, 9> ChunkGatedDeltaRuleFwdPrepare(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    bool allowNegEigval,
    bool useExp2,
    bool useQkL2norm,
    bool useGateInKernel,
    bool useBetaSigmoid,
    bool outputA,
    const aclTensor *gOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *aOut,
    const aclTensor *qHatOptional,
    const aclTensor *kHatOptional,
    const aclTensor *qRstdOptional,
    const aclTensor *kRstdOptional,
    const aclTensor *betaEffOptional,
    aclOpExecutor *executor)
{
    const bool hasHats = qHatOptional != nullptr && kHatOptional != nullptr &&
                         qRstdOptional != nullptr && kRstdOptional != nullptr;
    const bool noHats = qHatOptional == nullptr && kHatOptional == nullptr &&
                        qRstdOptional == nullptr && kRstdOptional == nullptr;
    if (!hasHats && !noHats) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "ChunkGatedDeltaRuleFwdPrepare qHat/kHat/qRstd/kRstd must all be set or all nullptr.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    if (useQkL2norm && !hasHats) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "ChunkGatedDeltaRuleFwdPrepare useQkL2norm=true requires qHat, kHat, qRstd and kRstd.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    if (!useQkL2norm && !noHats) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ChunkGatedDeltaRuleFwdPrepare useQkL2norm=false requires omitting qHat/kHat/qRstd/kRstd.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    L0_DFX(ChunkGatedDeltaRuleFwdPrepare, q, k, v, g, beta, aLogOptional, dtBiasOptional, cuSeqlensOptional,
           chunkIndicesOptional, chunkSize, allowNegEigval, useExp2, useQkL2norm, useGateInKernel,
           useBetaSigmoid, outputA, gOut, wOut, uOut, aOut, qHatOptional, kHatOptional, qRstdOptional,
           kRstdOptional, betaEffOptional);

    const aclTensor *actualCuSeqlens = ConvertIntArrayToTensor(cuSeqlensOptional, executor);
    const aclTensor *actualChunkIndices = ConvertIntArrayToTensor(chunkIndicesOptional, executor);
    const aclTensor *actualQHat = qHatOptional;
    const aclTensor *actualKHat = kHatOptional;
    const aclTensor *actualQRstd = qRstdOptional;
    const aclTensor *actualKRstd = kRstdOptional;
    if (!useQkL2norm) {
        // Keep optional output slots occupied so GE does not pack beta_eff into
        // q_hat and leave later GM (tiling) as 0x80000000. Kernel aliases k̂ to k
        // and never reads/writes these dummies.
        actualQHat = executor->AllocTensor(q->GetViewShape(), q->GetDataType(), Format::FORMAT_ND);
        actualKHat = executor->AllocTensor(k->GetViewShape(), k->GetDataType(), Format::FORMAT_ND);
        op::Shape rstdShape;
        rstdShape.SetDimNum(3);
        rstdShape.SetDim(0, q->GetViewShape().GetDim(0));
        rstdShape.SetDim(1, q->GetViewShape().GetDim(1));
        rstdShape.SetDim(2, q->GetViewShape().GetDim(2));
        actualQRstd = executor->AllocTensor(rstdShape, DataType::DT_FLOAT, Format::FORMAT_ND);
        actualKRstd = executor->AllocTensor(rstdShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    }
    const aclTensor *actualBetaEff = betaEffOptional;
    if (actualBetaEff == nullptr) {
        actualBetaEff = executor->AllocTensor(g->GetViewShape(), DataType::DT_FLOAT, Format::FORMAT_ND);
    }
    if ((cuSeqlensOptional != nullptr && actualCuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && actualChunkIndices == nullptr) ||
        actualQHat == nullptr || actualKHat == nullptr || actualQRstd == nullptr ||
        actualKRstd == nullptr || actualBetaEff == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Create internal tensor failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwdPrepare,
        OP_INPUT(q, k, v, g, beta, aLogOptional, dtBiasOptional, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(gOut, wOut, uOut, aOut, actualQHat, actualKHat, actualQRstd, actualKRstd, actualBetaEff),
        OP_ATTR(chunkSize, allowNegEigval, useExp2, useQkL2norm, useGateInKernel, useBetaSigmoid, outputA));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkGatedDeltaRuleFwdPrepare failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {gOut, wOut, uOut, aOut, qHatOptional, kHatOptional, qRstdOptional, kRstdOptional, actualBetaEff};
}

} // namespace l0op
