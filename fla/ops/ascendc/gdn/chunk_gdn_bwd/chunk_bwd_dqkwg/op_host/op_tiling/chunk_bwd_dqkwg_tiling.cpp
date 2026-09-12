/**
 * Copyright (c) 2026 Tianjin University, Ltd.
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
#include <algorithm>

namespace optiling {

constexpr int64_t CONST_BT = 64;

// 数据类型大小
constexpr size_t FP16_SIZE = 2;
constexpr size_t FP32_SIZE = 4;

constexpr size_t INPUT_Q_IDX = 0;
constexpr size_t INPUT_K_IDX = 1;
constexpr size_t INPUT_V_IDX = 2;
constexpr size_t INPUT_G_IDX = 3;
constexpr size_t INPUT_H_IDX = 4;
constexpr size_t INPUT_DOX_IDX = 5;
constexpr size_t INPUT_DH_IDX = 6;
constexpr size_t INPUT_DV_IDX = 7;
constexpr size_t INPUT_CUSEQLENS_IDX = 8;
constexpr size_t INPUT_CHUNK_INDICES_IDX = 9;
constexpr size_t INPUT_W_IDX = 10;
constexpr size_t INPUT_G_GAMMA_IDX = 11;
constexpr int ATTR_SCALE_ITEM = 0;
constexpr int ATTR_CHUNK_SIZE_ITEM = 1;

int64_t CeilDiv(int64_t a, int64_t b)
{
    if (unlikely(b == 0)) {
        return 0;
    }
    return (a + b - 1) / b;
}

ASCENDC_EXTERN_C ge::graphStatus TilingChunkBwdDqkwg(gert::TilingContext* context) {
    const gert::Shape qStorageShape = context->GetRequiredInputShape(INPUT_Q_IDX)->GetStorageShape();
    const gert::Shape kStorageShape = context->GetRequiredInputShape(INPUT_K_IDX)->GetStorageShape();
    const gert::Shape vStorageShape = context->GetRequiredInputShape(INPUT_V_IDX)->GetStorageShape();
    const gert::Shape gStorageShape = context->GetRequiredInputShape(INPUT_G_IDX)->GetStorageShape();
    const gert::Shape hStorageShape = context->GetRequiredInputShape(INPUT_H_IDX)->GetStorageShape();
    const gert::Shape doxStorageShape = context->GetRequiredInputShape(INPUT_DOX_IDX)->GetStorageShape();
    const gert::Shape dhStorageShape = context->GetRequiredInputShape(INPUT_DH_IDX)->GetStorageShape();
    const gert::Shape dvStorageShape = context->GetRequiredInputShape(INPUT_DV_IDX)->GetStorageShape();

    int64_t B = vStorageShape.GetDim(0);
    int64_t HV = vStorageShape.GetDim(1);   // value 侧 head 数 (v/g/h/do/dh/dv 及全部输出)
    int64_t T = vStorageShape.GetDim(2);
    int64_t HK = kStorageShape.GetDim(1);   // key/query 侧 head 数 (q/k)
    int64_t K = kStorageShape.GetDim(3);
    int64_t V = vStorageShape.GetDim(3);
    int64_t BT = CONST_BT;
    // GVA: HV = n_ratio * HK, n_ratio 由 q/v shape 自动推导
    if (HK == 0 || HV % HK != 0) {
        OP_LOGE(context->GetNodeName(), "HV must be a multiple of HK, but HV = %ld, HK = %ld.", HV, HK);
        return ge::GRAPH_FAILED;
    }
    auto attr = context->GetAttrs();
    const int32_t* chunkSizePtr = attr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_ITEM);
    if (chunkSizePtr != nullptr) {
        BT = *chunkSizePtr;
        if (BT != 64 && BT != 128) {
            OP_LOGE(context->GetNodeName(), "BT should be 64 or 128, but got %ld.", BT);
            return ge::GRAPH_FAILED;
        }

    }


    if (context->GetOptionalInputTensor(INPUT_W_IDX) != nullptr ||
        context->GetOptionalInputTensor(INPUT_G_GAMMA_IDX) != nullptr) {
        OP_LOGE(context->GetNodeName(), "w and g_gamma should be set at nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto cuSeqlensTensor = context->GetOptionalInputTensor(INPUT_CUSEQLENS_IDX);
    int64_t numChunks = CeilDiv(T, BT);  // = 32
    int isVarLen = 0;
    if (cuSeqlensTensor != nullptr) {
        auto cuChunkIndicesTensor = context->GetOptionalInputTensor(INPUT_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, cuChunkIndicesTensor);
        const gert::StorageShape *chunkIndicesShape = context->GetOptionalInputShape(INPUT_CHUNK_INDICES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesShape);
        const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();
        numChunks = chunkIndicesStorageShape.GetDim(0);
        if (numChunks % 2 != 0) {
            OP_LOGE(context->GetNodeName(), "numChunks should be even, but now is %ld.", numChunks);
            return ge::GRAPH_FAILED;
        }
        numChunks /= 2;
        isVarLen = 1;
    }
    if (isVarLen == 1 && B != 1) {
        OP_LOGE(context->GetNodeName(), "varlen mode only support B = 1, but now B = %ld.", B);
        return ge::GRAPH_FAILED;
    }
    {
        // 检查输入维度是否符合预期
        // q, k: [B, HK, T, K]; v, dox, dv: [B, HV, T, V]; g: [B, HV, T]; h, dh: [B, HV, numChunks, K, V]
        if (qStorageShape.GetDim(0) != B || qStorageShape.GetDim(1) != HK || qStorageShape.GetDim(2) != T || qStorageShape.GetDim(3) != K ||
            kStorageShape.GetDim(0) != B || kStorageShape.GetDim(1) != HK || kStorageShape.GetDim(2) != T || kStorageShape.GetDim(3) != K ||
            vStorageShape.GetDim(0) != B || vStorageShape.GetDim(1) != HV || vStorageShape.GetDim(2) != T || vStorageShape.GetDim(3) != V ||
            gStorageShape.GetDim(0) != B || gStorageShape.GetDim(1) != HV || gStorageShape.GetDim(2) != T ||
            hStorageShape.GetDim(0) != B || hStorageShape.GetDim(1) != HV || hStorageShape.GetDim(2) != numChunks || hStorageShape.GetDim(3) != K || hStorageShape.GetDim(4) != V ||
            doxStorageShape.GetDim(0) != B || doxStorageShape.GetDim(1) != HV || doxStorageShape.GetDim(2) != T || doxStorageShape.GetDim(3) != V ||
            dhStorageShape.GetDim(0) != B || dhStorageShape.GetDim(1) != HV || dhStorageShape.GetDim(2) != numChunks || dhStorageShape.GetDim(3) != K || dhStorageShape.GetDim(4) != V ||
            dvStorageShape.GetDim(0) != B || dvStorageShape.GetDim(1) != HV || dvStorageShape.GetDim(2) != T || dvStorageShape.GetDim(3) != V) {
            OP_LOGE(context->GetNodeName(),
                "Input tensor shapes do not match expected dimensions. Expected: q,k [B,HK,T,K], "
                "v,dox,dv [B,HV,T,V], g [B,HV,T], h,dh [B,HV,NC,K,V].");
            return ge::GRAPH_FAILED;
        }
        if (K != 128) {
            OP_LOGE(context->GetNodeName(), "K should be 128, but now K = %ld.", K);
            return ge::GRAPH_FAILED;
        }
        if (V != 128 && V != 256) {
            OP_LOGE(context->GetNodeName(), "V should be 128 or 256, but now V = %ld.", V);
            return ge::GRAPH_FAILED;
        }

    }



    // 计算 scale = 1.0 / sqrt(K)
    // float scale = 1.0f / std::sqrt(static_cast<float>(K));
    const float* scalePtr = attr->GetAttrPointer<float>(ATTR_SCALE_ITEM);
    if (scalePtr == nullptr) {
        OP_LOGE(context->GetNodeName(), "scale should not be nullptr.");
        return ge::GRAPH_FAILED;
    }
    float scale = *scalePtr;

    // 获取平台信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    const int64_t physicalAicNum = ascendcPlatform.GetCoreNumAic();

    // 设置 TilingKey
    context->SetTilingKey(1);

    auto align32 = [](size_t value) -> size_t {
        return ((value + 31) / 32) * 32;
    };

    const int64_t coreLoops = B * numChunks;
    int64_t aicNum = physicalAicNum;
    if (aicNum < 1) {
        aicNum = 1;
    }
    int64_t ringCoreSlots = std::min(aicNum, coreLoops);
    if (ringCoreSlots < 1) {
        ringCoreSlots = 1;
    }

    const size_t sharedBtxKSize = align32(static_cast<size_t>(ringCoreSlots) * HV * BT * K * FP16_SIZE);
    const size_t sharedBtbSize = align32(static_cast<size_t>(ringCoreSlots) * HV * BT * BT * FP16_SIZE);
    size_t dgLastSize = align32(static_cast<size_t>(ringCoreSlots) * HV * FP32_SIZE);

    size_t offset = 0;
    size_t wsMm3Offset = offset;
    offset += sharedBtbSize;
    size_t wsMm4Offset = offset;
    offset += sharedBtxKSize;
    size_t wsMm6Offset = offset;
    offset += sharedBtxKSize;
    size_t wsMm5Offset = offset;
    offset += sharedBtxKSize;
    size_t wsMm7Offset = offset;
    offset += sharedBtxKSize;
    size_t wsDsTempOffset = offset;
    offset += sharedBtbSize;
    size_t wsDgLastOffset = offset;
    offset += dgLastSize;
    size_t totalUserWorkspace = offset;

    // 设置 workspace 大小
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(sysWorkspaceSize + totalUserWorkspace);

    // 设置 block 数量
    context->SetBlockDim(ringCoreSlots);
    context->SetScheduleMode(1); // mixed AIC/AIV schedule

    // 填充 TilingData
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
    tilingData.set_aicCoreNum(static_cast<uint32_t>(aicNum));

    tilingData.set_wsMm3Offset(wsMm3Offset);
    tilingData.set_wsMm4Offset(wsMm4Offset);
    tilingData.set_wsMm6Offset(wsMm6Offset);
    tilingData.set_wsMm5Offset(wsMm5Offset);
    tilingData.set_wsMm7Offset(wsMm7Offset);
    tilingData.set_wsDsTempOffset(wsDsTempOffset);
    tilingData.set_wsDgLastOffset(wsDgLastOffset);

    // 检查是否有 cu_seqlens 输入来判断 IS_VARLEN
    tilingData.set_isVarLen(isVarLen);

    // 保存 tiling data
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
