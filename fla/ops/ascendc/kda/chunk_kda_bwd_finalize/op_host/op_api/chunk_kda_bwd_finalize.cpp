#include "chunk_kda_bwd_finalize.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkKdaBwdFinalize);

const std::array<const aclTensor *, 7> ChunkKdaBwdFinalize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *gk, const aclTensor *rawG, const aclTensor *beta,
    const aclTensor *aLog, const aclTensor *dtBias, const aclTensor *akk,
    const aclTensor *vNew, const aclTensor *h, const aclTensor *dh,
    const aclTensor *dvScan, const aclTensor *dAqk, const aclTensor *dqRaw,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    double scale, double lowerBound, int64_t chunkSize,
    bool safeGate, bool useGateInKernel, bool useExp2, bool stateVFirst,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dALogOut, const aclTensor *dDtBiasOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkKdaBwdFinalize,
           q, k, v, gk, rawG, beta, aLog, dtBias, akk,
           vNew, h, dh, dvScan, dAqk, dqRaw,
           qRstdOptional, kRstdOptional,
           cuSeqlensOptional, chunkIndicesOptional,
           scale, lowerBound, chunkSize,
           safeGate, useGateInKernel, useExp2, stateVFirst,
           dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut);
    const aclTensor *cuTensor = nullptr;
    const aclTensor *indicesTensor = nullptr;
    if (cuSeqlensOptional != nullptr) {
        cuTensor = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        indicesTensor = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (cuTensor == nullptr || indicesTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert KernelC metadata");
            return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        }
        for (const aclTensor *tensor : {cuTensor, indicesTensor}) {
            auto *mutableTensor = const_cast<aclTensor *>(tensor);
            mutableTensor->SetStorageFormat(Format::FORMAT_ND);
            mutableTensor->SetViewFormat(Format::FORMAT_ND);
            mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
        }
    }
    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkKdaBwdFinalize,
        OP_INPUT(q, k, v, gk, rawG, beta, aLog, dtBias, akk, vNew, h, dh,
                 dvScan, dAqk, dqRaw, qRstdOptional, kRstdOptional,
                 cuTensor, indicesTensor),
        OP_OUTPUT(dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut),
        OP_ATTR(scale, lowerBound, chunkSize, safeGate,
                useGateInKernel, useExp2, stateVFirst));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdFinalize failed");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut};
}

} // namespace l0op
