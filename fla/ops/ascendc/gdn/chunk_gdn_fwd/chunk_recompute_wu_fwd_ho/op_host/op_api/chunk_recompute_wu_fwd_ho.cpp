/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_recompute_wu_fwd_ho.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkRecomputeWUFwdHO);

namespace {

const aclTensor *ConvertMetadata(const aclIntArray *values, aclOpExecutor *executor)
{
    if (values == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(values, DataType::DT_INT64);
    if (tensor != nullptr) {
        auto *mutableTensor = const_cast<aclTensor *>(tensor);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    }
    return tensor;
}

} // namespace

const std::array<const aclTensor *, 2> ChunkRecomputeWUFwdHO(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *g,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    double scale,
    const aclTensor *oOut,
    const aclTensor *finalStateOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkRecomputeWUFwdHO, q, k, v, beta, a, g, gkOptional, initialStateOptional,
           cuSeqlensOptional, chunkIndicesOptional, outputFinalState, chunkSize, scale,
           oOut, finalStateOut);
    const aclTensor *cuSeqlens = ConvertMetadata(cuSeqlensOptional, executor);
    const aclTensor *chunkIndices = ConvertMetadata(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cuSeqlens == nullptr) ||
        (chunkIndicesOptional != nullptr && chunkIndices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Convert fused D+HO metadata failed.");
        return {nullptr, nullptr};
    }

    const aclError ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkRecomputeWUFwdHO,
        OP_INPUT(q, k, v, beta, a, g, gkOptional, initialStateOptional, cuSeqlens, chunkIndices),
        OP_OUTPUT(oOut, finalStateOut),
        OP_ATTR(outputFinalState, chunkSize, scale));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkRecomputeWUFwdHO failed.");
        return {nullptr, nullptr};
    }
    return {oOut, finalStateOut};
}

} // namespace l0op
