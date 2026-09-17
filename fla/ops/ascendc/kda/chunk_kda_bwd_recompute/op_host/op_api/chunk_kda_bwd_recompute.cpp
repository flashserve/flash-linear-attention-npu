#include "chunk_kda_bwd_recompute.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdRecompute);

namespace {
void SetNdFormat(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
}
} // namespace

const std::array<const aclTensor *, 5> ChunkKdaBwdRecompute(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *g, const aclTensor *beta,
    const aclTensor *a, const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional, int64_t chunkSize,
    bool useGateInKernel, bool useExp2, double lowerBound, const aclTensor *wOut, const aclTensor *uOut,
    const aclTensor *qgOut, const aclTensor *kgOut, const aclTensor *gkOut, aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdRecompute, q, k, v, g, beta, a, aLogOptional, dtBiasOptional, cuSeqlensOptional,
           chunkIndicesOptional, chunkSize, useGateInKernel, useExp2, lowerBound, wOut, uOut, qgOut, kgOut, gkOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        SetNdFormat(actualCuSeqlens);
    }
    const aclTensor *actualChunkIndices = nullptr;
    if (chunkIndicesOptional != nullptr) {
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        SetNdFormat(actualChunkIndices);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdRecompute,
        OP_INPUT(q, k, v, g, beta, a, aLogOptional, dtBiasOptional, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(wOut, uOut, qgOut, kgOut, gkOut),
        OP_ATTR(chunkSize, useGateInKernel, useExp2, static_cast<float>(lowerBound)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdRecompute failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {wOut, uOut, qgOut, kgOut, gkOut};
}

} // namespace l0op
