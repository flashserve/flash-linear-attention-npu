/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_local_cumsum.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkLocalCumsumParams {
    const aclTensor *g = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOutOptional = nullptr;
    int64_t chunkSize = 0;
    bool reverse = false;
    double scale = 1.0;
    bool headFirst = true;
    const char *outputDtypeOptional = "float32";
    const aclTensor *out = nullptr;
};

static int64_t NextPowerOfTwo(int64_t value)
{
    int64_t result = 1;
    while (result < std::max<int64_t>(value, 1)) {
        result <<= 1;
    }
    return result;
}

static aclnnStatus CheckNotNull(ChunkLocalCumsumParams params)
{
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "out must not be nullptr.");
    CHECK_COND(params.outputDtypeOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "outputDtypeOptional must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkLocalCumsumParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    const auto gShape = params.g->GetViewShape();
    const auto outShape = params.out->GetViewShape();
    CHECK_COND(gShape.GetDimNum() >= 3, ACLNN_ERR_PARAM_INVALID,
               "g must have rank at least 3 for [B,H,T,...].");
    CHECK_COND(outShape.GetDimNum() == gShape.GetDimNum(), ACLNN_ERR_PARAM_INVALID,
               "out rank must equal g rank.");
    int64_t tail = 1;
    for (size_t dim = 0; dim < gShape.GetDimNum(); ++dim) {
        CHECK_COND(gShape.GetDim(dim) > 0, ACLNN_ERR_PARAM_INVALID,
                   "g dimensions must be positive; dimension %zu is %ld.", dim, gShape.GetDim(dim));
        CHECK_COND(outShape.GetDim(dim) == gShape.GetDim(dim), ACLNN_ERR_PARAM_INVALID,
                   "out shape must equal g shape; dimension %zu differs.", dim);
        if (dim >= 3) {
            CHECK_COND(tail <= std::numeric_limits<int64_t>::max() / gShape.GetDim(dim),
                       ACLNN_ERR_PARAM_INVALID, "the product of g tail dimensions overflows int64.");
            tail *= gShape.GetDim(dim);
        }
    }
    CHECK_COND(params.g->GetDataType() == DataType::DT_FLOAT &&
                   params.out->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "g and out must use float32.");
    CHECK_COND(params.chunkSize > 0 && (params.chunkSize & (params.chunkSize - 1)) == 0,
               ACLNN_ERR_PARAM_INVALID, "chunkSize must be a positive power of two.");
    CHECK_COND(params.headFirst, ACLNN_ERR_PARAM_INVALID,
               "headFirst=false is not supported; g must use [B, H, T, ...] layout.");
    CHECK_COND(std::strcmp(params.outputDtypeOptional, "float32") == 0 ||
                   std::strcmp(params.outputDtypeOptional, "torch.float") == 0 ||
                   std::strcmp(params.outputDtypeOptional, "torch.float32") == 0,
               ACLNN_ERR_PARAM_INVALID, "outputDtypeOptional must select float32.");
    const bool hasCuSeqlens = params.cuSeqlensOptional != nullptr;
    const bool hasChunkIndices = params.chunkIndicesOutOptional != nullptr;
    CHECK_COND(hasCuSeqlens == hasChunkIndices, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional and chunkIndicesOutOptional must be provided together.");
    CHECK_COND(tail <= std::numeric_limits<int64_t>::max() / params.chunkSize,
               ACLNN_ERR_PARAM_INVALID, "tail size multiplied by chunkSize overflows int64.");
    const int64_t tileBase = (static_cast<int64_t>(1) << 17) / (tail * params.chunkSize);
    const int64_t blockT = NextPowerOfTwo(tileBase);
    CHECK_COND(blockT >= params.chunkSize, ACLNN_ERR_PARAM_INVALID,
               "the derived processing block length must be at least chunkSize.");
    if (!hasCuSeqlens) {
        return ACLNN_SUCCESS;
    }

    CHECK_COND(gShape.GetDim(0) == 1, ACLNN_ERR_PARAM_INVALID,
               "B must be 1 when varlen metadata is provided.");
    const aclIntArray &cuSeqlens = *params.cuSeqlensOptional;
    const aclIntArray &chunkIndices = *params.chunkIndicesOutOptional;
    CHECK_COND(cuSeqlens.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0,T].");
    CHECK_COND(chunkIndices.Size() > 0 && chunkIndices.Size() % 2 == 0,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOutOptional must contain flattened (seq_id, block_id) pairs.");
    const int64_t time = gShape.GetDim(2);
    CHECK_COND(cuSeqlens[0] == 0 && cuSeqlens[cuSeqlens.Size() - 1] == time,
               ACLNN_ERR_PARAM_INVALID, "cuSeqlensOptional must start at 0 and end at T=%ld.", time);
    size_t expectedIndex = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens.Size(); ++seq) {
        const int64_t seqStart = cuSeqlens[seq];
        const int64_t seqEnd = cuSeqlens[seq + 1];
        CHECK_COND(seqStart >= 0 && seqStart <= seqEnd && seqEnd <= time,
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be non-decreasing and within [0,T].");
        const int64_t blockCount = (seqEnd - seqStart + blockT - 1) / blockT;
        for (int64_t localBlock = 0; localBlock < blockCount; ++localBlock) {
            CHECK_COND(expectedIndex + 1 < chunkIndices.Size(), ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOutOptional contains fewer pairs than required.");
            CHECK_COND(chunkIndices[expectedIndex] == static_cast<int64_t>(seq) &&
                           chunkIndices[expectedIndex + 1] == localBlock,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOutOptional must use canonical sequence-major block order.");
            expectedIndex += 2;
        }
    }
    CHECK_COND(expectedIndex == chunkIndices.Size(), ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOutOptional pair count does not match cuSeqlensOptional and processing block length.");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkLocalCumsumParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous g failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkLocalCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOutOptional,
    int64_t chunkSize,
    bool reverse,
    double scale,
    bool headFirst,
    char *outputDtypeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    const char *outputDtype = outputDtypeOptional == nullptr ? "float32" : outputDtypeOptional;
    ChunkLocalCumsumParams params{
        g, cuSeqlensOptional, chunkIndicesOutOptional, chunkSize, reverse, scale, headFirst, outputDtype, out};

    L2_DFX_PHASE_1(aclnnChunkLocalCumsum,
                   DFX_IN(g, cuSeqlensOptional, chunkIndicesOutOptional, chunkSize, reverse, scale, headFirst,
                          outputDtypeOptional),
                   DFX_OUT(out));

    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");

    auto result = l0op::ChunkLocalCumsum(params.g, params.cuSeqlensOptional, params.chunkIndicesOutOptional,
                                        params.chunkSize, params.reverse, params.scale, params.headFirst,
                                        params.outputDtypeOptional, params.out, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, params.out, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkLocalCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkLocalCumsum);
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ChunkLocalCumsum launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
