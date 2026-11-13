/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "op_common/log/log.h"
#include <cstring>

#include "chunk_gdn_bwd_intra.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGdnBwdIntra);

namespace {
void SetNdFormat(const aclTensor *tensor)
{
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
}
} // namespace

const std::array<const aclTensor *, 3> ChunkGdnBwdIntra(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *dO, const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional, double scale, int64_t chunkSize,
    bool useExp2, const aclTensor *wOut, const aclTensor *uOut,
    const aclTensor *dvLocalOut, aclOpExecutor *executor,
    const char *layout)
{
    if (layout == nullptr ||
        (std::strcmp(layout, "BNSD") != 0 && std::strcmp(layout, "BSND") != 0 &&
         std::strcmp(layout, "NTD") != 0 && std::strcmp(layout, "TND") != 0)) {
        OP_LOGE_WITH_INVALID_ATTR("ChunkGdnBwdIntra", "layout", layout == nullptr ? "nullptr" : layout,
                                  "BNSD, BSND, NTD or TND");
        return {nullptr, nullptr, nullptr};
    }
    L0_DFX(ChunkGdnBwdIntra, q, k, v, g, beta, a, dO, cuSeqlensOptional,
           chunkIndicesOptional, scale, chunkSize, useExp2, layout,
           wOut, uOut, dvLocalOut);

    const aclTensor *actualCuSeqlens = nullptr;
    const aclTensor *actualChunkIndices = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        actualChunkIndices = executor->ConvertToTensor(chunkIndicesOptional, DataType::DT_INT64);
        if (actualCuSeqlens == nullptr || actualChunkIndices == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to convert varlen metadata tensors.");
            return {nullptr, nullptr, nullptr};
        }
        SetNdFormat(actualCuSeqlens);
        SetNdFormat(actualChunkIndices);
    }

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGdnBwdIntra,
        OP_INPUT(q, k, v, g, beta, a, dO, actualCuSeqlens, actualChunkIndices),
        OP_OUTPUT(wOut, uOut, dvLocalOut),
        OP_ATTR(scale, chunkSize, useExp2, layout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE ChunkGdnBwdIntra failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {wOut, uOut, dvLocalOut};
}

} // namespace l0op
