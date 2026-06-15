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
 #include <cstdlib>
 
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
     auto attr = context->GetAttrs();
     const int ATTR_CHUNK_SIZE_ITEM = 1;
     const int32_t* chunkSizePtr = attr->GetAttrPointer<int32_t>(ATTR_CHUNK_SIZE_ITEM);
     if (chunkSizePtr != nullptr) {
         BT = *chunkSizePtr;
         // std::cout << "[tiling] BT: " << BT << "\n";
         if (BT != 64 && BT != 128) {
             std::cout << "BT should be 64 or 128 ." << std::endl;
             return ge::GRAPH_FAILED;
         }
         
     }
 
 
     if (context->GetOptionalInputTensor(10) != nullptr || context->GetOptionalInputTensor(11) != nullptr) {
         std::cout << "w and g_gamma should be set at nullptr." << std::endl;
         return ge::GRAPH_FAILED;
     }
 
     auto cuSeqlensTensor = context->GetOptionalInputTensor(8);
     int64_t numChunks = CeilDiv(T, BT);  // = 32
     int isVarLen = 0;
     if (cuSeqlensTensor != nullptr) {
         auto cuChunkIndicesTensor = context->GetOptionalInputTensor(9);
         const gert::StorageShape *chunkIndicesShape = context->GetOptionalInputShape(9);
         OP_CHECK_NULL_WITH_CONTEXT(context, chunkIndicesShape);
         const gert::Shape chunkIndicesStorageShape = chunkIndicesShape->GetStorageShape();
         numChunks = chunkIndicesStorageShape.GetDim(0);
         if (numChunks % 2 != 0) {
             std::cout << "numChunks should be even, but now is " << chunkIndicesStorageShape.GetDim(1) << "!" << std::endl;
             return ge::GRAPH_FAILED;
         }
         numChunks /= 2;
         isVarLen = 1;
     }
     if (isVarLen == 1 && B != 1) {
         std::cout << "varlen mode only support B = 1, but now B = " << B << "." << std::endl;
         return ge::GRAPH_FAILED;
     }
         // std::cout << "[tiling] isVarLen: " << isVarLen << "\n";
     {
         // 检查输入维度是否符合预期
         if (qStorageShape.GetDim(0) != B || qStorageShape.GetDim(1) != H || qStorageShape.GetDim(2) != T || qStorageShape.GetDim(3) != K ||
             kStorageShape.GetDim(0) != B || kStorageShape.GetDim(1) != H || kStorageShape.GetDim(2) != T || kStorageShape.GetDim(3) != K ||
             vStorageShape.GetDim(0) != B || vStorageShape.GetDim(1) != H || vStorageShape.GetDim(2) != T || vStorageShape.GetDim(3) != V ||
             gStorageShape.GetDim(0) != B || gStorageShape.GetDim(1) != H || gStorageShape.GetDim(2) != T ||
             hStorageShape.GetDim(0) != B || hStorageShape.GetDim(1) != H || hStorageShape.GetDim(2) != numChunks || hStorageShape.GetDim(3) != K || hStorageShape.GetDim(4) != V ||
             doxStorageShape.GetDim(0) != B || doxStorageShape.GetDim(1) != H || doxStorageShape.GetDim(2) != T || doxStorageShape.GetDim(3) != V ||
             dhStorageShape.GetDim(0) != B || dhStorageShape.GetDim(1) != H || dhStorageShape.GetDim(2) != numChunks || dhStorageShape.GetDim(3) != K || dhStorageShape.GetDim(4) != V ||
             dvStorageShape.GetDim(0) != B || dvStorageShape.GetDim(1) != H || dvStorageShape.GetDim(2) != T || dvStorageShape.GetDim(3) != V) {
             std::cout << "Input tensor shapes do not match expected dimensions!" << std::endl;
             return ge::GRAPH_FAILED;
         }
         if (K != 128) {
             std::cout << "K should be 128, but now K = " << K << "." << std::endl;
             return ge::GRAPH_FAILED;
         }
         if (V != 128 && V != 256) {
             std::cout << "V should be 128 or 256, but now V = " << V << "." << std::endl;
             return ge::GRAPH_FAILED;
         }
 
     }
     
     // std::cout << "[tiling] B: " << B << ", H: " << H << ", T: " << T << ", K: " << K << ", V: " << V << ", BT: " << BT << ", numChunks: " << numChunks << std::endl;
     
     
     // 计算 scale = 1.0 / sqrt(K)
     // float scale = 1.0f / std::sqrt(static_cast<float>(K));
     const int ATTR_SCALE_ITEM = 0;
     
     const float* scalePtr = attr->GetAttrPointer<float>(ATTR_SCALE_ITEM);
     if (scalePtr == nullptr) {
         std::cout << "scale should not be nullptr!" << std::endl;
         return ge::GRAPH_FAILED;
     }
     float scale = *scalePtr;
     // std::cout << "[tiling] scale is " << scale << std::endl;
     
     // 获取平台信息
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
     auto sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
     const int64_t aicNum = ascendcPlatform.GetCoreNumAic();
     
     uint64_t tilingKey = 2;
     const char *implEnv = std::getenv("DQKWG_IMPL");
     if (implEnv != nullptr && (implEnv[0] == '1' || implEnv[0] == 'o' || implEnv[0] == 'O')) {
         tilingKey = 1;
     }
     context->SetTilingKey(tilingKey);

     // size_t mm5Size = B * H * T * BT * FP16_SIZE;                   // mm5 (q @ k^T)
     size_t mm5Size = B * H * T * K * FP16_SIZE;// mm5 (q @ k^T): B * H * T * BT * FP16_SIZE, mm6/mm7 需要复用 mm5 的空间，BT 可能是 64 或 128，这里直接按 128 来计算，保证空间足够
     size_t dsTempSize = B * H * T * BT * FP16_SIZE;                // b_ds_temp

     size_t offset = 0;
     size_t wsDwOffset = 0;
     size_t wsDgLastOffset = 0;
     size_t wsMm5Offset = 0;
     size_t wsDsTempOffset = 0;
     size_t wsMm6Offset = 0;
     size_t wsMm7Offset = 0;
     size_t wsMul1Offset = 0;
     size_t dgLastSize = 0;

     if (tilingKey == 1) {
         // Legacy path: Part 1 writes dw directly to output, so only dg_last/mm5/ds_temp need workspace.
         dgLastSize = B * H * numChunks * FP32_SIZE;
         dgLastSize = ((dgLastSize + 31) / 32) * 32;

         wsDgLastOffset = offset;
         offset += dgLastSize;
         wsMm5Offset = offset;
         offset += mm5Size;
         wsDsTempOffset = offset;
         offset += dsTempSize;
         wsMm6Offset = wsMm5Offset;
         wsMm7Offset = wsMm5Offset;
         wsMul1Offset = 0;
     } else {
         // Current path: Cube stores dw in dtype workspace, Vector finalizes/sign-fixes it.
         size_t dwSize = B * H * T * K * FP16_SIZE;
         dwSize = ((dwSize + 31) / 32) * 32;
         dgLastSize = B * H * numChunks * 8 * FP32_SIZE;         // one 32B slot per scalar
         dgLastSize = ((dgLastSize + 31) / 32) * 32;

         wsDwOffset = offset;
         offset += dwSize;
         wsDgLastOffset = offset;
         offset += dgLastSize;
         wsMm5Offset = offset;
         offset += mm5Size;
         wsDsTempOffset = offset;
         offset += dsTempSize;
         wsMm6Offset = wsMm5Offset;
         wsMm7Offset = wsMm5Offset;
         wsMul1Offset = 0;
     }

     size_t totalUserWorkspace = offset;
     // std::cout << "[tiling] totalUserWorkspace: " << totalUserWorkspace << ", sysWorkspaceSize: " << sysWorkspaceSize << "\n";
     
     // 设置 workspace 大小
     size_t* workspaces = context->GetWorkspaceSizes(1);
     workspaces[0] = static_cast<size_t>(sysWorkspaceSize + totalUserWorkspace);
     
     // 设置 block 数量
     context->SetBlockDim(aicNum);
     context->SetScheduleMode(1); // set as batchmod for template using SyncAll
     
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
     tilingData.set_mul0RowNum(V == 256 ? 16 : 32);
     
     tilingData.set_wsDwOffset(wsDwOffset);
     tilingData.set_wsDgLastOffset(wsDgLastOffset);
     tilingData.set_dgLastSize(dgLastSize);
     tilingData.set_wsMm5Offset(wsMm5Offset);
     tilingData.set_wsDsTempOffset(wsDsTempOffset);
     tilingData.set_totalWorkspaceSize(totalUserWorkspace);
     tilingData.set_wsMm6Offset(wsMm6Offset);
     tilingData.set_wsMm7Offset(wsMm7Offset);
     tilingData.set_wsMul1Offset(wsMul1Offset);
     
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

