/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dqkwg_tiling.cpp
 * \brief ChunkBwdDqkwg Tiling 实现
 */

#include "chunk_bwd_dqkwg_tiling.h"
#include <register/op_impl_registry.h>
#include "tiling_base/data_copy_transpose_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_type.h"
#include <cmath>

namespace optiling {

constexpr int64_t CONST_B = 1;
constexpr int64_t CONST_H = 4;
constexpr int64_t CONST_T = 2816;
constexpr int64_t CONST_K = 128;
constexpr int64_t CONST_V = 128;
constexpr int64_t CONST_BT = 64;

constexpr size_t FP16_SIZE = 2;
constexpr size_t FP32_SIZE = 4;

int64_t CeilDiv(int64_t a, int64_t b)
{
    if (unlikely(b == 0)) {
        return 0;
    }
    return (a + b - 1) / b;
}

ASCENDC_EXTERN_C ge::graphStatus TilingChunkBwdDqkwg(gert::TilingContext* context) {
    const gert::Shape qStorageShape = context->GetRequiredInputShape(0)->GetStorageShape();
    const gert::Shape kStorageShape = context->GetRequiredInputShape(1)->GetStorageShape();
    const gert::Shape vStorageShape = context->GetRequiredInputShape(2)->GetStorageShape();
    const gert::Shape gStorageShape = context->GetRequiredInputShape(3)->GetStorageShape();
    const gert::Shape hStorageShape = context->GetRequiredInputShape(4)->GetStorageShape();
    const gert::Shape doxStorageShape = context->GetRequiredInputShape(5)->GetStorageShape();
    const gert::Shape dhStorageShape = context->GetRequiredInputShape(6)->GetStorageShape();
    const gert::Shape dvStorageShape = context->GetRequiredInputShape(7)->GetStorageShape();

    int64_t B = vStorageShape.GetDim(0);
    int64_t HV = vStorageShape.GetDim(1);
    int64_t T = vStorageShape.GetDim(2);
    int64_t HK = kStorageShape.GetDim(1);
    int64_t K = kStorageShape.GetDim(3);
    int64_t V = vStorageShape.GetDim(3);
    int64_t BT = CONST_BT;
    if (HK == 0 || HV % HK != 0) {
        OP_LOGE(context, "HV must be a multiple of HK, but HV = %ld, HK = %ld.", HV, HK);
        return ge::GRAPH_FAILED;
    }
    int64_t n_ratio = HV / HK;
    auto attr = context->GetAttrs();
    const int ATTR_CHUNK_SIZE_ITEM = 1;
    const int32_t* chunkSizePtr = attr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_ITEM);
    if (chunkSizePtr != nullptr) {
        BT = *chunkSizePtr;
        if (BT != 64 && BT != 128) {
            OP_LOGE(context, "BT should be 64 or 128.");
            return ge::GRAPH_FAILED;
        }
    }

    if (context->GetOptionalInputTensor(10) != nullptr || context->GetOptionalInputTensor(11) != nullptr) {
        OP_LOGE(context, "w and g_gamma should be set at nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto cuSeqlensTensor = context->GetOptionalInputTensor(8);
    int64_t numChunks = CeilDiv(T, BT);
    int isVarLen = 0;
    if (cuSeqlensTensor != nullptr) {
        auto cuChunkIndicesTensor = context->GetOptionalInputTensor(9);
        const gert::StorageShape *chunkIndicesShape = context->GetOptionalInputShape(9);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesShape);
        const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();
        numChunks = chunkIndicesStorageShape.GetDim(0);
        if (numChunks % 2 != 0) {
            OP_LOGE(context, "numChunks should be even, but now is %ld.", chunkIndicesStorageShape.GetDim(1));
            return ge::GRAPH_FAILED;
        }
        numChunks /= 2;
        isVarLen = 1;
    }
    if (isVarLen == 1 && B != 1) {
        OP_LOGE(context, "varlen mode only support B = 1, but now B = %ld.", B);
        return ge::GRAPH_FAILED;
    }
    {
        // 检查输入维度是否符合预期
        // q, k: [B, HK, T, K]
        // v, dox, dv: [B, HV, T, V]
        // g: [B, HV, T]
        // h, dh: [B, HV, numChunks, K, V]
        if (qStorageShape.GetDim(0) != B || qStorageShape.GetDim(1) != HK || qStorageShape.GetDim(2) != T || qStorageShape.GetDim(3) != K ||
            kStorageShape.GetDim(0) != B || kStorageShape.GetDim(1) != HK || kStorageShape.GetDim(2) != T || kStorageShape.GetDim(3) != K ||
            vStorageShape.GetDim(0) != B || vStorageShape.GetDim(1) != HV || vStorageShape.GetDim(2) != T || vStorageShape.GetDim(3) != V ||
            gStorageShape.GetDim(0) != B || gStorageShape.GetDim(1) != HV || gStorageShape.GetDim(2) != T ||
            hStorageShape.GetDim(0) != B || hStorageShape.GetDim(1) != HV || hStorageShape.GetDim(2) != numChunks || hStorageShape.GetDim(3) != K || hStorageShape.GetDim(4) != V ||
            doxStorageShape.GetDim(0) != B || doxStorageShape.GetDim(1) != HV || doxStorageShape.GetDim(2) != T || doxStorageShape.GetDim(3) != V ||
            dhStorageShape.GetDim(0) != B || dhStorageShape.GetDim(1) != HV || dhStorageShape.GetDim(2) != numChunks || dhStorageShape.GetDim(3) != K || dhStorageShape.GetDim(4) != V ||
            dvStorageShape.GetDim(0) != B || dvStorageShape.GetDim(1) != HV || dvStorageShape.GetDim(2) != T || dvStorageShape.GetDim(3) != V) {
            OP_LOGE(context, "Input tensor shapes do not match expected dimensions: q,k [B,HK,T,K], v,dox,dv [B,HV,T,V], g [B,HV,T], h,dh [B,HV,NC,K,V].");
            return ge::GRAPH_FAILED;
        }
        if (K != 128) {
            OP_LOGE(context, "K should be 128, but now K = %ld.", K);
            return ge::GRAPH_FAILED;
        }
        if (V != 128 && V != 256) {
            OP_LOGE(context, "V should be 128 or 256, but now V = %ld.", V);
            return ge::GRAPH_FAILED;
        }
    }

    const int ATTR_SCALE_ITEM = 0;
    const float* scalePtr = attr->GetAttrPointer<float>(ATTR_SCALE_ITEM);
    if (scalePtr == nullptr) {
        OP_LOGE(context, "scale should not be nullptr.");
        return ge::GRAPH_FAILED;
    }
    float scale = *scalePtr;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();

    context->SetTilingKey(1);

    size_t dgLastSize = B * HV * numChunks * 1 * FP32_SIZE;
    dgLastSize = ((dgLastSize + 31) / 32) * 32;

    size_t mm5Size = B * HV * T * K * FP16_SIZE;
    size_t dsTempSize = B * HV * T * BT * FP16_SIZE;

    size_t offset = 0;

    size_t wsDgLastOffset = offset;
    offset += dgLastSize;

    size_t wsMm5Offset = offset;
    offset += mm5Size;

    size_t wsDsTempOffset = offset;
    offset += dsTempSize;

    size_t totalUserWorkspace = offset;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(sysWorkspaceSize + totalUserWorkspace);

    context->SetBlockDim(aicNum);
    context->SetScheduleMode(1);

    ChunkBwdDqkwgTilingData tilingData;
    tilingData.set_B(B);
    tilingData.set_HV(HV);
    tilingData.set_HK(HK);
    tilingData.set_T(T);
    tilingData.set_K(K);
    tilingData.set_V(V);
    tilingData.set_BT(BT);
    tilingData.set_numChunks(numChunks);
    tilingData.set_scale(scale);
    tilingData.set_mul0RowNum(V == 256 ? 16 : 32);

    tilingData.set_wsDgLastOffset(wsDgLastOffset);
    tilingData.set_dgLastSize(dgLastSize);
    tilingData.set_wsMm5Offset(wsMm5Offset);
    tilingData.set_wsDsTempOffset(wsDsTempOffset);
    tilingData.set_totalWorkspaceSize(totalUserWorkspace);

    tilingData.set_isVarLen(isVarLen);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

struct ChunkBwdDqkwgCompileInfo {};
ASCENDC_EXTERN_C ge::graphStatus TilingParseChunkBwdDqkwg(gert::TilingParseContext* context) {
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkBwdDqkwg)
    .Tiling(TilingChunkBwdDqkwg)
    .TilingParse<ChunkBwdDqkwgCompileInfo>(TilingParseChunkBwdDqkwg);

}  // namespace optiling
