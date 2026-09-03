/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_chunk_local_cumsum.h"

#include <string>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkLocalCumsum);

const aclTensor *ChunkLocalCumsum(
    const aclTensor *g,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOutOptional,
    int64_t chunkSize,
    bool reverse,
    double scale,
    bool headFirst,
    const char *outputDtypeOptional,
    const aclTensor *out,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkLocalCumsum, g, cuSeqlensOptional, chunkIndicesOutOptional, chunkSize, reverse, scale, headFirst,
           outputDtypeOptional, out);

    std::string outputDtypeStr(outputDtypeOptional ? outputDtypeOptional : "float32");
    float scaleFloat = static_cast<float>(scale);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkLocalCumsum,
        OP_INPUT(g, cuSeqlensOptional, chunkIndicesOutOptional),
        OP_OUTPUT(out),
        OP_ATTR(chunkSize, reverse, scaleFloat, headFirst, outputDtypeStr));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}

}  // namespace l0op
