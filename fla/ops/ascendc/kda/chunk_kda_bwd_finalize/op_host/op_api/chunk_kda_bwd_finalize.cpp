#include "chunk_kda_bwd_finalize.h"
#include <cstring>
#include <vector>

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
    bool useBetaSigmoidInKernel, bool allowNegEigval, const char *outputLayout,
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
           useBetaSigmoidInKernel, allowNegEigval, outputLayout,
           dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut);
    const std::array<const aclTensor *, 7> empty{};
    if (allowNegEigval && !useBetaSigmoidInKernel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "allowNegEigval=true requires useBetaSigmoidInKernel=true");
        return empty;
    }
#ifndef KDA_BWD_STAGES_ARCH22
    if (useBetaSigmoidInKernel) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "A5 finalize: use_beta_sigmoid_in_kernel=true is reserved but not supported");
        return empty;
    }
#endif
    if (!q || !k || !v || !gk || !beta || !h || !outputLayout ||
        (!cuSeqlensOptional && chunkIndicesOptional)) return empty;
    const size_t rank = q->GetViewShape().GetDimNum();
    if (rank != 3 && rank != 4) return empty;
    const bool rank3 = rank == 3;
    const bool tokenMajor = std::strcmp(outputLayout, rank3 ? "TND" : "BSND") == 0;
    if (!tokenMajor && std::strcmp(outputLayout, rank3 ? "NTD" : "BNSD") != 0) return empty;
    const std::array<const aclTensor *, 7> requested{dqOut, dkOut, dvOut, dBetaOut, dGOut, dALogOut, dDtBiasOut};
    for (auto *output : requested) if (!output) return empty;
    // Generate metadata once on Host; both direct L0 and ACLNN use this path.
    if (cuSeqlensOptional) {
        const int64_t tokens = q->GetViewShape().GetDim(rank - 2);
        const size_t size = cuSeqlensOptional->Size();
        if (tokens <= 0 || chunkSize != 64 || size < 2 ||
            (!rank3 && q->GetViewShape().GetDim(0) != 1) ||
            (*cuSeqlensOptional)[0] != 0 || (*cuSeqlensOptional)[size - 1] != tokens) return empty;
        std::vector<int64_t> generated;
        size_t task = 0;
        for (size_t seq = 0; seq + 1 < size; ++seq) {
            const int64_t start = (*cuSeqlensOptional)[seq];
            const int64_t end = (*cuSeqlensOptional)[seq + 1];
            if (start < 0 || end < start || end > tokens) return empty;
            const int64_t chunks = (end - start) / 64 + ((end - start) % 64 != 0);
            for (int64_t chunk = 0; chunk < chunks; ++chunk, ++task) {
                if (chunkIndicesOptional) {
                    if (task >= chunkIndicesOptional->Size() / 2 ||
                        (*chunkIndicesOptional)[2 * task] != static_cast<int64_t>(seq) ||
                        (*chunkIndicesOptional)[2 * task + 1] != chunk) return empty;
                } else {
                    generated.push_back(static_cast<int64_t>(seq));
                    generated.push_back(chunk);
                }
            }
        }
        if (chunkIndicesOptional && chunkIndicesOptional->Size() != 2 * task) return empty;
        if (!chunkIndicesOptional) {
            chunkIndicesOptional = executor->AllocIntArray(generated.data(), generated.size());
            if (!chunkIndicesOptional) return empty;
        }
    }
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
                useGateInKernel, useExp2, stateVFirst,
                useBetaSigmoidInKernel, allowNegEigval, outputLayout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaBwdFinalize failed");
        return {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return requested;
}

} // namespace l0op
