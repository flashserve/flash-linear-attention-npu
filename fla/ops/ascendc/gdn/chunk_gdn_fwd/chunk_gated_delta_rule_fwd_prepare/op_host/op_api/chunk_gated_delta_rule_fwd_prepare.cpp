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
    if (qHatOptional == nullptr || kHatOptional == nullptr ||
        qRstdOptional == nullptr || kRstdOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "ChunkGatedDeltaRuleFwdPrepare requires qHat, kHat, qRstd and kRstd "
                "(this version always runs Q/K L2Norm).");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    L0_DFX(ChunkGatedDeltaRuleFwdPrepare, q, k, v, g, beta, aLogOptional, dtBiasOptional, cuSeqlensOptional,
           chunkIndicesOptional, chunkSize, allowNegEigval, useExp2, useQkL2norm, useGateInKernel,
           useBetaSigmoid, gOut, wOut, uOut, aOut, qHatOptional, kHatOptional, qRstdOptional,
           kRstdOptional, betaEffOptional);

    const aclTensor *actualCuSeqlens = ConvertIntArrayToTensor(cuSeqlensOptional, executor);
    const aclTensor *actualChunkIndices = ConvertIntArrayToTensor(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && actualCuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && actualChunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert optional int array to tensor failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwdPrepare,
        OP_INPUT(q, k, v, g, beta, aLogOptional, dtBiasOptional, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(gOut, wOut, uOut, aOut, qHatOptional, kHatOptional, qRstdOptional, kRstdOptional, betaEffOptional),
        OP_ATTR(chunkSize, allowNegEigval, useExp2, useQkL2norm, useGateInKernel, useBetaSigmoid));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkGatedDeltaRuleFwdPrepare failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {gOut, wOut, uOut, aOut, qHatOptional, kHatOptional, qRstdOptional, kRstdOptional, betaEffOptional};
}

} // namespace l0op
