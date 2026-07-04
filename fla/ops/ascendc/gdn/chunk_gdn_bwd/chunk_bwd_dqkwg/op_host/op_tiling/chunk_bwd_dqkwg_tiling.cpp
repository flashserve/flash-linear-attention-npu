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
#include <cmath>
#include <algorithm>

namespace optiling {

constexpr int64_t CONST_B = 1;
constexpr int64_t CONST_HV = 4;
constexpr int64_t CONST_HK = 4;
constexpr int64_t CONST_T = 2816;
constexpr int64_t CONST_K = 128;
constexpr int64_t CONST_V = 128;
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
    int64_t n_ratio = HV / HK;
    (void)n_ratio;
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

    const size_t mainDgLastSize = align32(static_cast<size_t>(B) * HV *numChunks * FP32_SIZE);
    const size_t mainMm5Size = static_cast<size_t>(B) * HV *T * K * FP16_SIZE;
    const size_t mainDsTempSize = static_cast<size_t>(B) * HV *T * BT * FP16_SIZE;
    const size_t mainWorkspaceSize = mainDgLastSize + mainMm5Size + mainDsTempSize;

    // actualWorkspaceForDepth: 与下面真实分配完全一致 (short 环深自适应 = 2G, mm7 寄生 mm5 group 区, 不再 max)。
    auto actualWorkspaceForDepth = [&](int64_t g) -> size_t {
        int64_t shortD = (g / 2 >= 2) ? (g / 2) : 2;                 // 自适应 short = 2G, 地板 2
        size_t shortBtxK = align32(static_cast<size_t>(ringCoreSlots) * shortD * HV *BT * K * FP16_SIZE);
        size_t sharedBtxK = align32(static_cast<size_t>(ringCoreSlots) * g * HV *BT * K * FP16_SIZE);  // mm5+mm7 group
        size_t groupBtb = align32(static_cast<size_t>(ringCoreSlots) * g * HV *BT * BT * FP16_SIZE);
        size_t shortBtb = align32(static_cast<size_t>(ringCoreSlots) * shortD * HV *BT * BT * FP16_SIZE);
        size_t dgLast = align32(static_cast<size_t>(ringCoreSlots) * g * HV *FP32_SIZE);
        return shortBtxK + sharedBtxK + groupBtb + shortBtb + dgLast;
    };

    // 选 G: 取 <= min(main, L2 驻留预算) 的最大 G。**memory-bound 关键 (msprof 实锤)**: 大 H/大 BT 的 case
    // 是 L2 受限 (cube FixPipe 0.997 / MTE2 0.869 被 L2 miss 打满, L2 hit 仅 0.13), group 环 (mm5/ds_temp,
    // 深度 4G) 装不进 L2 -> 环越大越慢。用 L2 预算把它们压到小 G (group 环减半、贴 L2); 小 case 环本来就小,
    // 仍能拿到 G=4。L2_RING_BUDGET 按 910B 实测取 (case_03 432MB 不劣 / case_11 604MB 劣之间), 跨设备可调。
    const size_t L2_RING_BUDGET = static_cast<size_t>(512) * 1024 * 1024;
    const size_t ringBudget = std::min(mainWorkspaceSize, L2_RING_BUDGET);
    int64_t groupRingDepth = 4;
    for (int64_t candidate : {16, 8, 4}) {
        if (actualWorkspaceForDepth(candidate) <= ringBudget) {
            groupRingDepth = candidate;
            break;
        }
    }

    // ---- 仅针对劣化形状的 overlap 深度修复 (2026-06-28) ----
    // 判别证据: case_10 (H=16/BT=128, G=2, ring~288MB) 不劣化; case_11/12 (H=32/BT=128) 被 512MB 预算压到
    // G=1 (ring 也~288MB) 却 +6% 劣化。两者 ring 大小几乎相同 => 差异不在 ring 体积, 而在 overlap 深度:
    // G=1 => 信用窗口 N=min(G,M)=1 => cube 只能领先 vector 1 个 task (近 lockstep); G=2 => N=2 真正重叠。
    // (旁证: 历史实测 G=2 = +4.92% < 当前 G=1 = +6.29%。) 故把被压到 depth<8 的形状抬到 depth=8 (G=2), 对齐 case_10。
    //
    // **只改坏形状, 好 case 字节不变**: 此 if 仅命中"当前 groupRingDepth<8 且 main 容得下 depth=8"的形状 =>
    //   - case_10 已是 depth=8 (8<8 false) -> 不变;
    //   - case_01/03 等在 G=4=depth16 (16<8 false) -> 不变;
    //   - tiny case footprint(8)>main -> 守卫挡掉 -> 不变;
    //   - 唯独 case_11/12/24 (H=32/BT=128, 被预算压到 depth=4) 抬到 8。
    const int64_t MIN_DEPTH_FOR_OVERLAP = 8;  // G=2 => N>=2, 对齐不劣化的 case_10
    if (groupRingDepth < MIN_DEPTH_FOR_OVERLAP &&
        actualWorkspaceForDepth(MIN_DEPTH_FOR_OVERLAP) <= mainWorkspaceSize) {
        groupRingDepth = MIN_DEPTH_FOR_OVERLAP;
    }

    // short 环深自适应 (= 内核 DqkwgShortRingDepthFromGroup 同公式): dw/mm6/mul1 只需 2G-1 个 slot, 固定 8 严重过配。
    // 大 H/大 BT 的 memory-bound case (G=1~2) 把 short 环砍掉一大半 -> 环贴近 L2 -> 缓解 FixPipe/MTE2 的 L2 miss。
    const int64_t adaptiveShortDepth = (groupRingDepth / 2 >= 2) ? (groupRingDepth / 2) : 2;  // 2G (G=4→8 复现原值; G<4 收缩), 地板 2
    const size_t shortBtxKSize = align32(static_cast<size_t>(ringCoreSlots) * adaptiveShortDepth * HV *BT * K * FP16_SIZE);
    // mm7 改用 group 环 (与 mm5 同槽, 单写, stage B/D 时序错开), 故 mm5/mm7 共享区 = group 深度 (不再 max(group,short))。
    const size_t sharedBtxKSize = align32(static_cast<size_t>(ringCoreSlots) * groupRingDepth * HV *BT * K * FP16_SIZE);
    const size_t groupBtbSize = align32(static_cast<size_t>(ringCoreSlots) * groupRingDepth * HV *BT * BT * FP16_SIZE);
    const size_t shortBtbSize = align32(static_cast<size_t>(ringCoreSlots) * adaptiveShortDepth * HV *BT * BT * FP16_SIZE);
    size_t dgLastSize = align32(static_cast<size_t>(ringCoreSlots) * groupRingDepth * HV *FP32_SIZE);

    size_t offset = 0;
    size_t wsDwOffset = offset;
    offset += shortBtxKSize;

    size_t wsMm5Offset = offset;
    offset += sharedBtxKSize;

    size_t wsDsTempOffset = offset;
    offset += groupBtbSize;

    size_t wsMul1Offset = offset;
    offset += shortBtbSize;

    size_t wsDgLastOffset = offset;
    offset += dgLastSize;

    size_t wsMm6Offset = wsDwOffset;
    size_t wsMm7Offset = wsMm5Offset;
    size_t totalUserWorkspace = offset;

    // 设置 workspace 大小
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(sysWorkspaceSize + totalUserWorkspace);

    // 设置 block 数量
    context->SetBlockDim(aicNum);
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

    tilingData.set_wsDwOffset(wsDwOffset);
    tilingData.set_wsBtxKSyncSlotsPerHead(static_cast<uint64_t>(groupRingDepth));
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
