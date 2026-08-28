#include "aclnn_chunk_kkt_solve_tri.h"
#include "chunk_kkt_solve_tri.h"

#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKktSolveTriGetWorkspaceSize(
    const aclTensor *k, const aclTensor *g, const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, const aclTensor *aOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkKktSolveTri,
                   DFX_IN(k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize), DFX_OUT(aOut));
    CHECK_COND(k != nullptr && g != nullptr && beta != nullptr && aOut != nullptr &&
                   workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "Required inputs and outputs must not be null.");
    CHECK_COND(k->GetViewShape().GetDimNum() == 4 && g->GetViewShape().GetDimNum() == 3 &&
                   beta->GetViewShape().GetDimNum() == 3 && aOut->GetViewShape().GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "Expected k/A rank 4 and g/beta rank 3.");
    CHECK_COND((k->GetDataType() == DataType::DT_FLOAT16 || k->GetDataType() == DataType::DT_BF16) &&
                   g->GetDataType() == DataType::DT_FLOAT && beta->GetDataType() == DataType::DT_FLOAT &&
                   aOut->GetDataType() == k->GetDataType(),
               ACLNN_ERR_PARAM_INVALID, "Expected fp16/bf16 k/A and fp32 g/beta.");
    CHECK_COND(chunkSize == 64 || chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 64 or 128.");
    CHECK_COND((cuSeqlensOptional == nullptr) == (chunkIndicesOptional == nullptr), ACLNN_ERR_PARAM_INVALID,
               "Variable-length metadata must be provided together.");
    const int64_t batch = Dim(k, 0);
    const int64_t hk = Dim(k, 1);
    const int64_t tokens = Dim(k, 2);
    const int64_t hv = Dim(g, 1);
    CHECK_COND(Dim(k, 3) == 128 && batch > 0 && hk > 0 && tokens > 0 && hv == hk &&
                   Dim(g, 0) == batch && Dim(g, 2) == tokens &&
                   Dim(beta, 0) == batch && Dim(beta, 1) == hv && Dim(beta, 2) == tokens &&
                   Dim(aOut, 0) == batch && Dim(aOut, 1) == hk && Dim(aOut, 2) == tokens &&
                   Dim(aOut, 3) == chunkSize,
               ACLNN_ERR_PARAM_INVALID,
               "Input and output shapes are incompatible; Phase 2 expects q/k expanded to Hv before entry.");
    if (cuSeqlensOptional != nullptr) {
        CHECK_COND(batch == 1 && cuSeqlensOptional->Size() >= 2 && chunkIndicesOptional->Size() % 2 == 0,
                   ACLNN_ERR_PARAM_INVALID, "Invalid variable-length metadata.");
        const aclIntArray &cu = *cuSeqlensOptional;
        const aclIntArray &indices = *chunkIndicesOptional;
        CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == tokens,
                   ACLNN_ERR_PARAM_INVALID, "cuSeqlens must start at 0 and end at T.");
        size_t chunkOffset = 0;
        for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
            const int64_t begin = cu[seq];
            const int64_t end = cu[seq + 1];
            CHECK_COND(begin <= end, ACLNN_ERR_PARAM_INVALID, "cuSeqlens must be nondecreasing.");
            const int64_t count = (end - begin + chunkSize - 1) / chunkSize;
            for (int64_t localChunk = 0; localChunk < count; ++localChunk) {
                CHECK_COND(chunkOffset + 1 < indices.Size() &&
                               indices[chunkOffset] == static_cast<int64_t>(seq) &&
                               indices[chunkOffset + 1] == localChunk,
                           ACLNN_ERR_PARAM_INVALID,
                           "chunkIndices must use canonical sequence-major order.");
                chunkOffset += 2;
            }
        }
        CHECK_COND(chunkOffset == indices.Size(), ACLNN_ERR_PARAM_INVALID,
                   "chunkIndices must contain one pair per chunk.");
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(MakeContiguous(k, executorPtr) == ACLNN_SUCCESS &&
                  MakeContiguous(g, executorPtr) == ACLNN_SUCCESS &&
                  MakeContiguous(beta, executorPtr) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    const aclTensor *result = l0op::ChunkKktSolveTri(
        k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize, aOut, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result, aOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKktSolveTri(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKktSolveTri);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKktSolveTri launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
