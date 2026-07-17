/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
#include "solve_tri_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <string>

namespace optiling {

constexpr uint32_t INPUT_X_IDX = 0;
constexpr uint32_t INPUT_CU_SEQLENS_IDX = 1;
constexpr uint32_t INPUT_CHUNK_INDICES_IDX = 2;
constexpr uint32_t OUTPUT_X_OUT_IDX = 0;
constexpr uint32_t ATTR_LAYOUT_IDX = 0;

static ge::graphStatus SolveTriTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = ascendcPlatform.GetCoreNumAic();
    if (coreNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    auto inputShape = context->GetInputShape(INPUT_X_IDX);
    auto outputShape = context->GetOutputShape(OUTPUT_X_OUT_IDX);
    auto inputDesc = context->GetInputDesc(INPUT_X_IDX);
    auto outputDesc = context->GetOutputDesc(OUTPUT_X_OUT_IDX);
    if (inputShape == nullptr || outputShape == nullptr || inputDesc == nullptr || outputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto shape = inputShape->GetStorageShape();
    auto outShape = outputShape->GetStorageShape();
    int64_t ndim = shape.GetDimNum();
    if (ndim != 3 && ndim != 4) {
        return ge::GRAPH_FAILED;
    }
    if (outShape.GetDimNum() != ndim) {
        return ge::GRAPH_FAILED;
    }
    for (int64_t dim = 0; dim < ndim; ++dim) {
        if (shape.GetDim(dim) <= 0 || outShape.GetDim(dim) != shape.GetDim(dim)) {
            OP_LOGE(context->GetNodeName(), "x dimensions must be positive and x_out shape must match x.");
            return ge::GRAPH_FAILED;
        }
    }
    auto inputDtype = inputDesc->GetDataType();
    if ((inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16) ||
        outputDesc->GetDataType() != inputDtype) {
        OP_LOGE(context->GetNodeName(), "x/x_out must use matching float16 or bfloat16 dtype.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char *layoutStr = attrs->GetStr(ATTR_LAYOUT_IDX);
    std::string layout = layoutStr ? layoutStr : "bsnd";

    int64_t layoutMode = 1;
    if (layout == "bhtd") {
        layoutMode = 0;
    } else if (layout == "bsnd") {
        layoutMode = 1;
    } else if (layout == "tnd") {
        layoutMode = 2;
    } else {
        OP_LOGE(context->GetNodeName(), "layout must be lowercase bhtd, bsnd or tnd, but got %s.", layout.c_str());
        return ge::GRAPH_FAILED;
    }

    if ((layoutMode == 2 && ndim != 3) || (layoutMode != 2 && ndim != 4)) {
        OP_LOGE(context->GetNodeName(), "layout %s does not match input rank %ld.", layout.c_str(), ndim);
        return ge::GRAPH_FAILED;
    }

    int64_t B = 1;
    int64_t H = 0;
    int64_t T = 0;
    int64_t BT = 0;
    if (ndim == 4) {
        if (layoutMode == 0) {
            B = shape.GetDim(0);
            H = shape.GetDim(1);
            T = shape.GetDim(2);
            BT = shape.GetDim(3);
        } else {
            B = shape.GetDim(0);
            T = shape.GetDim(1);
            H = shape.GetDim(2);
            BT = shape.GetDim(3);
        }
    } else {
        T = shape.GetDim(0);
        H = shape.GetDim(1);
        BT = shape.GetDim(2);
    }

    int64_t chunkSize = BT;
    if (chunkSize != 16 && chunkSize != 32 && chunkSize != 64 && chunkSize != 128) {
        OP_LOGE(context->GetNodeName(), "the last dimension C must be 16, 32, 64 or 128, but got %ld.",
                chunkSize);
        return ge::GRAPH_FAILED;
    }

    int64_t isVarlen = (layoutMode == 2) ? 1 : 0;

    int64_t totalChunks = 0;
    int64_t numChunks = 0;
    int64_t totalTiles = 0;
    int64_t lastChunkValidSize = 0;

    const bool hasCuSeqlens = context->GetOptionalInputDesc(INPUT_CU_SEQLENS_IDX) != nullptr;
    const bool hasChunkIndices = context->GetOptionalInputDesc(INPUT_CHUNK_INDICES_IDX) != nullptr;
    if (hasCuSeqlens != hasChunkIndices || (isVarlen != static_cast<int64_t>(hasCuSeqlens))) {
        OP_LOGE(context->GetNodeName(),
                "cu_seqlens and chunk_indices must be provided together and only for tnd layout.");
        return ge::GRAPH_FAILED;
    }

    if (isVarlen) {
        auto cuSeqlensShape = context->GetOptionalInputShape(INPUT_CU_SEQLENS_IDX);
        auto chunkIndicesShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
        if (cuSeqlensShape == nullptr || chunkIndicesShape == nullptr) {
            OP_LOGE(context->GetNodeName(), "tnd layout requires cu_seqlens and chunk_indices.");
            return ge::GRAPH_FAILED;
        }
        if (cuSeqlensShape->GetStorageShape().GetDimNum() != 1 ||
            chunkIndicesShape->GetStorageShape().GetDimNum() != 1) {
            OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must be rank-1.");
            return ge::GRAPH_FAILED;
        }
        if (context->GetOptionalInputDesc(INPUT_CU_SEQLENS_IDX)->GetDataType() != ge::DT_INT64 ||
            context->GetOptionalInputDesc(INPUT_CHUNK_INDICES_IDX)->GetDataType() != ge::DT_INT64) {
            OP_LOGE(context->GetNodeName(), "cu_seqlens and chunk_indices must use int64.");
            return ge::GRAPH_FAILED;
        }
        int64_t chunkIndicesLen = chunkIndicesShape->GetStorageShape().GetDim(0);
        if (cuSeqlensShape->GetStorageShape().GetDim(0) < 2 || chunkIndicesLen <= 0 || chunkIndicesLen % 2 != 0) {
            OP_LOGE(context->GetNodeName(), "invalid cu_seqlens/chunk_indices shape for tnd layout.");
            return ge::GRAPH_FAILED;
        }
        const gert::Tensor *cuTensor = context->GetOptionalInputTensor(INPUT_CU_SEQLENS_IDX);
        const gert::Tensor *indicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        const int64_t *cuData = cuTensor == nullptr ? nullptr : cuTensor->GetData<int64_t>();
        const int64_t *indicesData = indicesTensor == nullptr ? nullptr : indicesTensor->GetData<int64_t>();
        if (cuData == nullptr || indicesData == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int64_t sequenceCount = cuSeqlensShape->GetStorageShape().GetDim(0) - 1;
        if (cuData[0] != 0 || cuData[sequenceCount] != T) {
            OP_LOGE(context->GetNodeName(), "cu_seqlens must start at 0 and end at T=%ld.", T);
            return ge::GRAPH_FAILED;
        }
        int64_t expectedChunks = 0;
        for (int64_t seq = 0; seq < sequenceCount; ++seq) {
            if (cuData[seq] < 0 || cuData[seq] > cuData[seq + 1] || cuData[seq + 1] > T) {
                OP_LOGE(context->GetNodeName(), "cu_seqlens must be nondecreasing and within [0,T].");
                return ge::GRAPH_FAILED;
            }
            int64_t localChunkCount = (cuData[seq + 1] - cuData[seq] + chunkSize - 1) / chunkSize;
            for (int64_t localChunk = 0; localChunk < localChunkCount; ++localChunk) {
                if (expectedChunks * 2 + 1 >= chunkIndicesLen || indicesData[expectedChunks * 2] != seq ||
                    indicesData[expectedChunks * 2 + 1] != localChunk) {
                    OP_LOGE(context->GetNodeName(),
                            "chunk_indices must list every chunk in canonical sequence-major order.");
                    return ge::GRAPH_FAILED;
                }
                ++expectedChunks;
            }
        }
        if (chunkIndicesLen != expectedChunks * 2) {
            OP_LOGE(context->GetNodeName(), "chunk_indices pair count does not match cu_seqlens and C.");
            return ge::GRAPH_FAILED;
        }
        totalChunks = expectedChunks;
        totalTiles = totalChunks * H;
        numChunks = 0;
        lastChunkValidSize = 0;
    } else {
        totalChunks = 0;
        numChunks = (T + chunkSize - 1) / chunkSize;
        totalTiles = B * numChunks * H;
        int64_t remainder = T % chunkSize;
        lastChunkValidSize = (remainder == 0) ? chunkSize : remainder;
    }

    int64_t tilesPerCore = (totalTiles + coreNum - 1) / coreNum;
    if (totalTiles <= 0 || tilesPerCore <= 0) return ge::GRAPH_FAILED;

    int64_t dtypeMode = 0;
    if (inputDtype == ge::DT_BF16) {
        dtypeMode = 1;
    }

    // Set tiling data
    SolveTriTilingData tiling;
    tiling.set_totalTiles(totalTiles);
    tiling.set_matrixSize(chunkSize);
    tiling.set_numHeads(H);
    tiling.set_seqLen(T);
    tiling.set_batchSize(B);
    tiling.set_isLower(1);
    tiling.set_hasCuSeqlens(static_cast<int64_t>(hasCuSeqlens));
    tiling.set_tilesPerCore(tilesPerCore);
    tiling.set_chunkSize(chunkSize);
    tiling.set_numChunks(numChunks);
    tiling.set_lastChunkValidSize(lastChunkValidSize);
    tiling.set_isVarlen(isVarlen);
    tiling.set_totalChunks(totalChunks);
    tiling.set_layoutMode(layoutMode);
    tiling.set_dtypeMode(dtypeMode);

    auto rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
        rawTilingData->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(1);
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());

    int64_t usedCoreNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    context->SetBlockDim(usedCoreNum);

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    ws[0] = sysWorkspaceSize;
#else
    size_t sharedSize = 3 * chunkSize * chunkSize * sizeof(uint16_t);
    size_t perCoreSize = 2 * chunkSize * chunkSize * sizeof(uint16_t);
    size_t userWorkspaceSize = sharedSize + usedCoreNum * perCoreSize;
    userWorkspaceSize = ((userWorkspaceSize + 511) / 512) * 512;
    ws[0] = userWorkspaceSize + sysWorkspaceSize;
#endif
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SolveTriTilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct SolveTriCompileInfo {};

IMPL_OP_OPTILING(SolveTri)
    .Tiling(SolveTriTilingFunc)
    .TilingParse<SolveTriCompileInfo>(SolveTriTilingParse);

}  // namespace optiling
