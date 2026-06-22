/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#include "solve_tril_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <string>

namespace optiling {

constexpr uint32_t INPUT_X_IDX = 0;
constexpr uint32_t INPUT_CU_SEQLENS_IDX = 1;
constexpr uint32_t INPUT_CHUNK_INDICES_IDX = 2;
constexpr uint32_t OUTPUT_X_OUT_IDX = 0;
constexpr uint32_t ATTR_CHUNK_SIZE_IDX = 0;
constexpr uint32_t ATTR_LAYOUT_IDX = 1;

static ge::graphStatus SolveTrilTilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = ascendcPlatform.GetCoreNumAic();
    if (coreNum == 0) return ge::GRAPH_FAILED;

    auto inputShape = context->GetInputShape(INPUT_X_IDX);
    if (inputShape == nullptr) return ge::GRAPH_FAILED;
    auto shape = inputShape->GetStorageShape();
    int64_t ndim = shape.GetDimNum();
    if (ndim != 3 && ndim != 4) return ge::GRAPH_FAILED;

    auto attrs = context->GetAttrs();
    int64_t chunkSize = *attrs->GetInt(ATTR_CHUNK_SIZE_IDX);
    const char *layoutStr = attrs->GetStr(ATTR_LAYOUT_IDX);
    std::string layout = layoutStr ? layoutStr : "bsnd";

    int64_t layoutMode = 1;
    if (layout == "bhtd") {
        layoutMode = 0;
    } else if (layout == "bsnd") {
        layoutMode = 1;
    } else if (layout == "tnd") {
        layoutMode = 2;
    }

    int64_t B, H, T, BT;
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
        B = 1;
        T = shape.GetDim(0);
        H = shape.GetDim(1);
        BT = shape.GetDim(2);
    }

    if (chunkSize <= 0) chunkSize = BT;

    int64_t isVarlen = (layoutMode == 2) ? 1 : 0;
    int64_t hasCuSeqlens = isVarlen;

    int64_t totalChunks = 0;
    int64_t numChunks = 0;
    int64_t totalTiles = 0;
    int64_t lastChunkValidSize = 0;

    if (isVarlen) {
        auto chunkIndicesShape = context->GetInputShape(INPUT_CHUNK_INDICES_IDX);
        int64_t chunkIndicesLen = chunkIndicesShape->GetStorageShape().GetDim(0);
        totalChunks = chunkIndicesLen / 2;
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

    SolveTrilTilingData tiling;
    tiling.set_totalTiles(totalTiles);
    tiling.set_matrixSize(chunkSize);
    tiling.set_numHeads(H);
    tiling.set_seqLen(T);
    tiling.set_batchSize(B);
    tiling.set_isLower(1);
    tiling.set_hasCuSeqlens(hasCuSeqlens);
    tiling.set_tilesPerCore(tilesPerCore);
    tiling.set_chunkSize(chunkSize);
    tiling.set_numChunks(numChunks);
    tiling.set_lastChunkValidSize(lastChunkValidSize);
    tiling.set_isVarlen(isVarlen);
    tiling.set_totalChunks(totalChunks);
    tiling.set_layoutMode(layoutMode);

    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    int64_t usedCoreNum = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    if (usedCoreNum > coreNum) usedCoreNum = coreNum;
    context->SetBlockDim(usedCoreNum);

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t sharedSize = 3 * chunkSize * chunkSize * sizeof(uint16_t);
    size_t perCoreSize = chunkSize * chunkSize * sizeof(uint16_t);
    size_t userWorkspaceSize = sharedSize + usedCoreNum * perCoreSize;
    userWorkspaceSize = ((userWorkspaceSize + 511) / 512) * 512;
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = userWorkspaceSize + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SolveTrilTilingParse(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SolveTrilCompileInfo {};

IMPL_OP_OPTILING(SolveTril)
    .Tiling(SolveTrilTilingFunc)
    .TilingParse<SolveTrilCompileInfo>(SolveTrilTilingParse);

}  // namespace optiling
