#include "aclnn_chunk_cumsum_kkt.h"

#include "chunk_cumsum_kkt.h"

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
constexpr int64_t K_DIM = 128;

struct ChunkCumsumKktParams {
    const aclTensor *k = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *gCumsumOut = nullptr;
    const aclTensor *aOut = nullptr;
};

int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

aclnnStatus CheckVarlenMetadata(const ChunkCumsumKktParams &params, int64_t batch, int64_t tokens)
{
    CHECK_COND((params.cuSeqlensOptional == nullptr) == (params.chunkIndicesOptional == nullptr),
               ACLNN_ERR_PARAM_INVALID, "Variable-length metadata must be provided together.");
    if (params.cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }

    CHECK_COND(batch == 1 && params.cuSeqlensOptional->Size() >= 2 &&
                   params.chunkIndicesOptional->Size() % 2 == 0,
               ACLNN_ERR_PARAM_INVALID, "Invalid variable-length metadata.");
    const aclIntArray &cu = *params.cuSeqlensOptional;
    const aclIntArray &indices = *params.chunkIndicesOptional;
    CHECK_COND(cu[0] == 0 && cu[cu.Size() - 1] == tokens,
               ACLNN_ERR_PARAM_INVALID, "cuSeqlens must start at 0 and end at T.");

    size_t chunkOffset = 0;
    for (size_t seq = 0; seq + 1 < cu.Size(); ++seq) {
        const int64_t begin = cu[seq];
        const int64_t end = cu[seq + 1];
        CHECK_COND(begin <= end, ACLNN_ERR_PARAM_INVALID, "cuSeqlens must be nondecreasing.");
        const int64_t count = (end - begin + params.chunkSize - 1) / params.chunkSize;
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
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkCumsumKktParams &params)
{
    CHECK_COND(params.k != nullptr && params.g != nullptr && params.beta != nullptr &&
                   params.gCumsumOut != nullptr && params.aOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "Required inputs and outputs must not be nullptr.");
    CHECK_COND(params.k->GetViewShape().GetDimNum() == 4 &&
                   params.g->GetViewShape().GetDimNum() == 3 &&
                   params.beta->GetViewShape().GetDimNum() == 3 &&
                   params.gCumsumOut->GetViewShape().GetDimNum() == 3 &&
                   params.aOut->GetViewShape().GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "Expected k/A rank 4 and g/beta/gCumsum rank 3.");
    CHECK_COND((params.k->GetDataType() == DataType::DT_FLOAT16 ||
                    params.k->GetDataType() == DataType::DT_BF16) &&
                   params.g->GetDataType() == DataType::DT_FLOAT &&
                   params.beta->GetDataType() == DataType::DT_FLOAT &&
                   params.gCumsumOut->GetDataType() == DataType::DT_FLOAT &&
                   params.aOut->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "Expected fp16/bf16 k and fp32 g/beta/gCumsum/A.");
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be 64 or 128.");

    const int64_t batch = Dim(params.k, 0);
    const int64_t heads = Dim(params.k, 1);
    const int64_t tokens = Dim(params.k, 2);
    CHECK_COND(batch > 0 && heads > 0 && tokens > 0 && Dim(params.k, 3) == K_DIM &&
                   Dim(params.g, 0) == batch && Dim(params.g, 1) == heads && Dim(params.g, 2) == tokens &&
                   Dim(params.beta, 0) == batch && Dim(params.beta, 1) == heads &&
                   Dim(params.beta, 2) == tokens &&
                   Dim(params.gCumsumOut, 0) == batch && Dim(params.gCumsumOut, 1) == heads &&
                   Dim(params.gCumsumOut, 2) == tokens &&
                   Dim(params.aOut, 0) == batch && Dim(params.aOut, 1) == heads &&
                   Dim(params.aOut, 2) == tokens && Dim(params.aOut, 3) == params.chunkSize,
               ACLNN_ERR_PARAM_INVALID,
               "Expected k=[B,H,T,128], g/beta/gCumsum=[B,H,T] and A=[B,H,T,chunkSize].");
    return CheckVarlenMetadata(params, batch, tokens);
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

aclnnStatus aclnnChunkCumsumKktGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *gCumsumOut,
    const aclTensor *aOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkCumsumKktParams params{k, g, beta, cuSeqlensOptional, chunkIndicesOptional,
                                chunkSize, gCumsumOut, aOut};
    L2_DFX_PHASE_1(aclnnChunkCumsumKkt,
                   DFX_IN(k, g, beta, cuSeqlensOptional, chunkIndicesOptional, chunkSize),
                   DFX_OUT(gCumsumOut, aOut));
    CHECK_COND(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize and executor must not be nullptr.");
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(MakeContiguous(params.k, executorPtr) == ACLNN_SUCCESS &&
                  MakeContiguous(params.g, executorPtr) == ACLNN_SUCCESS &&
                  MakeContiguous(params.beta, executorPtr) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::ChunkCumsumKkt(
        params.k, params.g, params.beta, params.cuSeqlensOptional, params.chunkIndicesOptional,
        params.chunkSize, params.gCumsumOut, params.aOut, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[0], params.gCumsumOut, executorPtr) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[1], params.aOut, executorPtr) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkCumsumKkt(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkCumsumKkt);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkCumsumKkt launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
