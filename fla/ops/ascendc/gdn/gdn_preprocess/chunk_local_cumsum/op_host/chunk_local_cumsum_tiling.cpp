/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_local_cumsum_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cstring>
#include <cstdint>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "../op_kernel/chunk_local_cumsum_tiling_data.h"

using namespace ge;

namespace optiling {
namespace {
constexpr size_t G_INDEX = 0;
constexpr size_t CU_SEQLENS_INDEX = 1;
constexpr size_t CHUNK_INDICES_INDEX = 2;
constexpr size_t OUT_INDEX = 0;
constexpr size_t ATTR_CHUNK_SIZE_INDEX = 0;
constexpr size_t ATTR_REVERSE_INDEX = 1;
constexpr size_t ATTR_SCALE_INDEX = 2;
constexpr size_t ATTR_HEAD_FIRST_INDEX = 3;
constexpr size_t ATTR_OUTPUT_DTYPE_INDEX = 4;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr int64_t H_TILE_SIZE = 512;
constexpr int64_t INPUT_DTYPE_FP32 = 0;
constexpr int64_t INPUT_DTYPE_FP16 = 1;
constexpr int64_t INPUT_DTYPE_BF16 = 2;

struct ChunkLocalCumsumCompileInfo {
    int64_t aivNum = 0;
};

static int64_t NextPowerOfTwo(int64_t value)
{
    int64_t v = std::max<int64_t>(value, 1);
    int64_t result = 1;
    while (result < v) {
        result <<= 1;
    }
    return result;
}

static bool IsPowerOfTwo(int64_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

static int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}
} // namespace

static ge::graphStatus TilingPrepareForChunkLocalCumsum(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<ChunkLocalCumsumCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfoPtr->aivNum <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "aivNum is invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingChunkLocalCumsum(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(G_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(G_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputShape(OUT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(OUT_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetAttrs());

    const auto &gShape = context->GetInputShape(G_INDEX)->GetStorageShape();
    OP_CHECK_IF(gShape.GetDimNum() != 3,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "g must be rank 3 for [B, H, T], but rank is %zu.",
                                            gShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    auto inDtype = context->GetInputDesc(G_INDEX)->GetDataType();
    auto outDtype = context->GetOutputDesc(OUT_INDEX)->GetDataType();
    OP_CHECK_IF(inDtype != ge::DT_FLOAT && inDtype != ge::DT_FLOAT16 && inDtype != ge::DT_BF16,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "g dtype must be float32, float16, or bfloat16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outDtype != inDtype,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "out dtype must be the same as g dtype."),
                return ge::GRAPH_FAILED);

    auto chunkSizePtr = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_INDEX);
    auto reversePtr = context->GetAttrs()->GetAttrPointer<bool>(ATTR_REVERSE_INDEX);
    auto scalePtr = context->GetAttrs()->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    auto headFirstPtr = context->GetAttrs()->GetAttrPointer<bool>(ATTR_HEAD_FIRST_INDEX);
    const char *outputDtype = context->GetAttrs()->GetAttrPointer<char>(ATTR_OUTPUT_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, chunkSizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, reversePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, headFirstPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDtype);

    OP_CHECK_IF(!*headFirstPtr,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "head_first=false is not supported; ChunkLocalCumsum currently supports "
                                            "only [B, H, T] layout."),
                return ge::GRAPH_FAILED);

    bool outputDtypeSame = std::strcmp(outputDtype, "same") == 0 ||
                           std::strcmp(outputDtype, "same_as_input") == 0 ||
                           std::strcmp(outputDtype, "input") == 0;
    bool outputDtypeFloat32 = std::strcmp(outputDtype, "float32") == 0 ||
                              std::strcmp(outputDtype, "torch.float") == 0 ||
                              std::strcmp(outputDtype, "torch.float32") == 0;
    OP_CHECK_IF(!outputDtypeSame && !(outputDtypeFloat32 && inDtype == ge::DT_FLOAT),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "output_dtype must preserve g dtype; got %s.", outputDtype),
                return ge::GRAPH_FAILED);

    int64_t chunkSize = *chunkSizePtr;
    OP_CHECK_IF(!IsPowerOfTwo(chunkSize),
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "chunk_size must be a power of two, but got %ld.",
                                            chunkSize),
                return ge::GRAPH_FAILED);

    int64_t batch = gShape.GetDim(0);
    int64_t head = gShape.GetDim(1);
    int64_t t = gShape.GetDim(2);
    int64_t tail = 1;
    OP_CHECK_IF(batch <= 0 || head <= 0 || t <= 0 || tail <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "g shape must be positive [B, H, T], got B=%ld, H=%ld, T=%ld.",
                                            batch, head, t),
                return ge::GRAPH_FAILED);
    int64_t outer = batch * head;

    int64_t cuSeqlensElements = 0;
    auto cuSeqlensShapePtr = context->GetInputShape(CU_SEQLENS_INDEX);
    if (context->GetOptionalInputDesc(CU_SEQLENS_INDEX) != nullptr && cuSeqlensShapePtr != nullptr) {
        cuSeqlensElements = cuSeqlensShapePtr->GetStorageShape().GetShapeSize();
    }
    bool isVarlen = cuSeqlensElements > 0;
    if (isVarlen) {
        auto chunkIndicesShapePtr = context->GetInputShape(CHUNK_INDICES_INDEX);
        OP_CHECK_IF(context->GetOptionalInputDesc(CHUNK_INDICES_INDEX) == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                "chunk_indices_out is required when cu_seqlens is provided."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesShapePtr);
        OP_CHECK_IF(chunkIndicesShapePtr->GetStorageShape().GetShapeSize() == 0,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                "chunk_indices_out is required when cu_seqlens is not empty."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(batch != 1,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                                "B must be 1 when cu_seqlens is provided, but got %ld.", batch),
                    return ge::GRAPH_FAILED);
    }

    int64_t btBase = (static_cast<int64_t>(1) << 17) / (tail * chunkSize);
    int64_t blockT = NextPowerOfTwo(btBase);
    OP_CHECK_IF(blockT < chunkSize,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "BLOCK_T=%ld is smaller than chunk_size=%ld.", blockT, chunkSize),
                return ge::GRAPH_FAILED);

    int64_t nt = (t + blockT - 1) / blockT;
    if (isVarlen) {
        const auto &chunkIndicesShape = context->GetInputShape(CHUNK_INDICES_INDEX)->GetStorageShape();
        OP_CHECK_IF(chunkIndicesShape.GetShapeSize() % 2 != 0,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "chunk_indices_out element count must be even."),
                    return ge::GRAPH_FAILED);
        nt = chunkIndicesShape.GetShapeSize() / 2;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "aivNum is invalid."),
                return ge::GRAPH_FAILED);
    int64_t hTileNum = CeilDiv(tail, H_TILE_SIZE);
    int64_t fixedTaskNum = outer * CeilDiv(t, chunkSize) * hTileNum;
    int64_t varlenTaskNum = outer * nt * hTileNum;
    int64_t taskNum = isVarlen ? varlenTaskNum : fixedTaskNum;
    int64_t blockDim = std::max<int64_t>(1, std::min<int64_t>(aivNum, taskNum));

    auto tiling = context->GetTilingData<ChunkLocalCumsumTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->b = outer;
    tiling->t = t;
    tiling->h = tail;
    tiling->chunkSize = chunkSize;
    tiling->blockT = blockT;
    tiling->numBlocks = nt;
    tiling->totalElements = gShape.GetShapeSize();
    tiling->isVarlen = isVarlen ? 1 : 0;
    tiling->reverse = *reversePtr ? 1 : 0;
    tiling->headFirst = *headFirstPtr ? 1 : 0;
    if (inDtype == ge::DT_FLOAT16) {
        tiling->inputDtype = INPUT_DTYPE_FP16;
    } else if (inDtype == ge::DT_BF16) {
        tiling->inputDtype = INPUT_DTYPE_BF16;
    } else {
        tiling->inputDtype = INPUT_DTYPE_FP32;
    }
    tiling->scale = *scalePtr;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkLocalCumsum)
    .Tiling(TilingChunkLocalCumsum)
    .TilingParse<ChunkLocalCumsumCompileInfo>(TilingPrepareForChunkLocalCumsum);
} // namespace optiling
