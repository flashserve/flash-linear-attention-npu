/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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

// 数据类型大小
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

    int64_t B = vStorageShape.GetDim(0);//CONST_B;
    int64_t H = vStorageShape.GetDim(1);//CONST_H;
    int64_t T = vStorageShape.GetDim(2);//CONST_T;
    int64_t K = kStorageShape.GetDim(3);//CONST_K;
    int64_t V = vStorageShape.GetDim(3);//CONST_V;
    int64_t BT = CONST_BT;

    auto cuSeqlensTensor = context->GetOptionalInputTensor(8);
    int64_t numChunks = CeilDiv(T, BT);  // = 32
    int isVarLen = 0;
    if (cuSeqlensTensor != nullptr) {
        auto cuChunkIndicesTensor = context->GetOptionalInputTensor(9);
        const gert::StorageShape *chunkIndicesShape = context->GetOptionalInputShape(9);
        OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesShape);
        const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();
        numChunks = chunkIndicesStorageShape.GetDim(0);
        if (chunkIndicesStorageShape.GetDim(1) != 2) {
            std::cout << "the 2nd dim of chunkIndicesStorageShape should be 2, but now is  " << chunkIndicesStorageShape.GetDim(1) << "!" << std::endl;
            return ge::GRAPH_FAILED;
        }
        isVarLen = 1;
    }

    
    // std::cout << "[tiling] B: " << B << ", H: " << H << ", T: " << T << ", K: " << K << ", V: " << V << ", BT: " << BT << ", numChunks: " << numChunks << std::endl;
    
    
    // 计算 scale = 1.0 / sqrt(K)
    // float scale = 1.0f / std::sqrt(static_cast<float>(K));
    const int ATTR_SCALE_ITEM = 0;
    auto attr = context->GetAttrs();
    const float* scalePtr = attr->GetAttrPointer<float>(ATTR_SCALE_ITEM);
    if (scalePtr == nullptr) {
        std::cout << "scale is nullptr!" << std::endl;
        return ge::GRAPH_FAILED;
    }
    float scale = *scalePtr;
    // std::cout << "[tiling] scale is " << scale << std::endl;
    
    // 获取平台信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    
    // 设置 TilingKey
    context->SetTilingKey(1);
    
    size_t dgLastSize = B * H * numChunks * 1 * FP32_SIZE;         // b_dg_last (fp32)
    // // 对齐到 32 字节
    dgLastSize = ((dgLastSize + 31) / 32) * 32;
    
    size_t mm5Size = B * H * T * BT * FP16_SIZE;                   // mm5 (q @ k^T)
    size_t dsTempSize = B * H * T * BT * FP16_SIZE;                // b_ds_temp
    size_t mm6Size = B * H * T * K * FP16_SIZE;                   // mm6
    size_t mm7Size = B * H * T * K * FP16_SIZE;                   // mm6

    
    // Workspace 偏移计算
    size_t offset = 0;
    
    // size_t wsDwOffset = offset;
    // offset += dwSize;
    
    size_t wsDgLastOffset = offset;
    offset += dgLastSize;
    
    size_t wsMm5Offset = offset;
    offset += mm5Size;
    
    size_t wsDsTempOffset = offset;
    offset += dsTempSize;

    size_t wsMm6Offset = offset;
    offset += mm6Size;

    size_t wsMm7Offset = offset;
    offset += mm7Size;
    
    size_t totalUserWorkspace = offset;
    // std::cout << "[tiling] totalUserWorkspace: " << totalUserWorkspace << ", sysWorkspaceSize: " << sysWorkspaceSize << "\n";
    
    // 设置 workspace 大小
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(sysWorkspaceSize + totalUserWorkspace);
    
    // 设置 block 数量
    context->SetBlockDim(aicNum);
    
    // 填充 TilingData
    ChunkBwdDqkwgTilingData tilingData;
    tilingData.set_B(B);
    tilingData.set_H(H);
    tilingData.set_T(T);
    tilingData.set_K(K);
    tilingData.set_V(V);
    tilingData.set_BT(BT);
    tilingData.set_numChunks(numChunks);
    tilingData.set_scale(scale);
    
    // tilingData.set_wsDwOffset(wsDwOffset);
    tilingData.set_wsDgLastOffset(wsDgLastOffset);
    tilingData.set_dgLastSize(dgLastSize);
    tilingData.set_wsMm5Offset(wsMm5Offset);
    tilingData.set_wsDsTempOffset(wsDsTempOffset);
    tilingData.set_totalWorkspaceSize(totalUserWorkspace);
    tilingData.set_wsMm6Offset(wsMm6Offset);
    tilingData.set_wsMm7Offset(wsMm7Offset);
    
    // 检查是否有 cu_seqlens 输入来判断 IS_VARLEN
    tilingData.set_isVarLen(isVarLen);
    
    // 保存 tiling data
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), 
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingParseChunkBwdDqkwg(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkBwdDqkwg)
    .Tiling(TilingChunkBwdDqkwg);

}  // namespace optiling
