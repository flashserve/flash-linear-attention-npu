/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file solve_tril_tiling.cpp
 * \brief Tiling for SolveTril.
 *
 * Layout modes:
 *   - BSND (mode=0): input [B, S, H, BT], fixed-length sequences
 *   - TND  (mode=1): input [T, H, BT] (B=1, S=T), varlen via cu_seqlens + chunk_indices_out
 */

#include "solve_tril_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/solve_tril_common.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int32_t MODE_BSND = 0;
constexpr int32_t MODE_TND  = 1;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SolveTrilTilingFunc(gert::TilingContext* context)
{
    // 1. Platform info
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Read layout attr (0=BSND, 1=TND)
    int64_t layoutVal = 0;
    auto layoutPtr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (layoutPtr != nullptr) {
        layoutVal = *layoutPtr;
    }
    OP_CHECK_IF(layoutVal != MODE_BSND && layoutVal != MODE_TND,
                OP_LOGE(context, "SolveTril: invalid layout=%ld, expected 0(BSND) or 1(TND)", layoutVal),
                return ge::GRAPH_FAILED);

    // 3. Parse input shape
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto shape = inputShape->GetStorageShape();
    int64_t dimNum = shape.GetDimNum();

    int64_t B, S, H, BT;
    if (layoutVal == MODE_BSND) {
        // BSND: [B, S, H, BT]
        OP_CHECK_IF(dimNum != 4, OP_LOGE(context,
            "SolveTril BSND mode expects 4D input [B,S,H,BT], got %ldD", dimNum),
            return ge::GRAPH_FAILED);
        B  = shape.GetDim(0);
        S  = shape.GetDim(1);
        H  = shape.GetDim(2);
        BT = shape.GetDim(3);
    } else {
        // TND: [T, H, BT] (B=1, S=T)
        OP_CHECK_IF(dimNum != 3,
            OP_LOGE(context, "SolveTril TND mode expects 3D input [T,H,BT], got %ldD", dimNum),
            return ge::GRAPH_FAILED);
        B  = 1;
        S  = shape.GetDim(0);
        H  = shape.GetDim(1);
        BT = shape.GetDim(2);
    }

    // 4. Validate BT
    OP_CHECK_IF(BT != 16 && BT != 32 && BT != 64 && BT != 128,
                OP_LOGE(context, "SolveTril: BT=%ld not in {16,32,64,128}", BT),
                return ge::GRAPH_FAILED);

    // 5. Derived values
    int64_t NT = CeilDiv(S, BT);
    int64_t mbhLevels = 0;
    if (BT == 32)      mbhLevels = 1;
    else if (BT == 64) mbhLevels = 2;
    else if (BT == 128)mbhLevels = 3;

    // 6. Compute tiling fields based on mode
    int64_t chunkNumInSeq;
    int64_t chunkNumTotal;

    if (layoutVal == MODE_BSND) {
        chunkNumInSeq = NT;
        chunkNumTotal = B * H * chunkNumInSeq;
    } else {
        // TND mode: chunkNumTotal = chunk_indices.shape(0) * numHead
        chunkNumInSeq = NT;  // invalid/meaningless in TND
        auto chunkIndicesShape = context->GetOptionalInputShape(2);
        OP_CHECK_IF(chunkIndicesShape == nullptr,
            OP_LOGE(context, "SolveTril TND mode requires chunk_indices_out"),
            return ge::GRAPH_FAILED);
        int64_t numChunkIndices = chunkIndicesShape->GetStorageShape().GetDim(0);
        chunkNumTotal = numChunkIndices * H;
    }

    // 7. Multi-core split
    int64_t usedCoreNum = chunkNumTotal < coreNum ? chunkNumTotal : coreNum;
    if (usedCoreNum == 0) usedCoreNum = 1;
    int64_t taskPerCore = CeilDiv(chunkNumTotal, usedCoreNum);

    // 8. Workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    size_t wsSize = WS_SYS_SIZE;
    if (mbhLevels >= 2) {
        wsSize = static_cast<size_t>(chunkNumTotal) * static_cast<size_t>(BT) * BT * sizeof(float);
    }
    currentWorkspace[0] = wsSize;

    // 9. Fill TilingData
    SolveTrilTilingData* tiling = context->GetTilingData<SolveTrilTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->batchSize     = B;
    tiling->seqLength     = S;
    tiling->numHead       = H;
    tiling->chunkSize     = BT;
    tiling->chunkNumInSeq = chunkNumInSeq;
    tiling->chunkNumTotal = chunkNumTotal;
    tiling->mode          = static_cast<int32_t>(layoutVal);
    tiling->blockDim      = usedCoreNum;
    tiling->taskPerCore   = taskPerCore;
    tiling->rowStride     = H * BT;
    tiling->mbhLevels     = mbhLevels;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 10. TilingKey
    auto inputDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dType = static_cast<uint32_t>(inputDtype);
    ASCENDC_TPL_SEL_PARAM(context, dType, static_cast<uint32_t>(mbhLevels));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSolveTril([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SolveTrilCompileInfo {};

IMPL_OP_OPTILING(SolveTril)
    .Tiling(SolveTrilTilingFunc)
    .TilingParse<SolveTrilCompileInfo>(TilingParseForSolveTril);

} // namespace optiling
