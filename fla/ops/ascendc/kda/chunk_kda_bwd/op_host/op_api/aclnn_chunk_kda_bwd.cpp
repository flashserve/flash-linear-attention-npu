#include "aclnn_chunk_kda_bwd.h"

#include "chunk_kda_bwd.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

namespace {

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
               stateVFirst),
        DFX_OUT(dqOut, dkOut, dvOut, dbOut, dgOut, dh0OutOptional,
                dAOutOptional, dBiasOutOptional));

    CHECK_COND(workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
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
    const aclTensor *dAForKernel =
        dAOutOptional != nullptr ? dAOutOptional : dgOut;
    const aclTensor *dBiasForKernel =
        dBiasOutOptional != nullptr ? dBiasOutOptional : dgOut;
    const auto result = l0op::KdaChunkBackward(
        q, k, v, beta, gk, aqk, akk, wOptional, qgOptional, kgOptional,
        vNewOptional, hOptional, dO,
        rawGOptional, aLogOptional, dtBiasOptional, cuTensor, chunkTensor,
        static_cast<float>(scale), chunkSize, safeGate, useGateInKernel,
        static_cast<float>(lowerBound), disableRecompute, useExp2,
        stateVFirst,
        dqOut, dkOut, dvOut, dbOut, dgOut,
        dAForKernel, dBiasForKernel, executorPtr);
    for (size_t i = 0; i < 5; ++i) {
        CHECK_RET(result[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(!useGateInKernel || result[5] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dtBiasOptional == nullptr || result[6] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

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
