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
 * \file moe_finalize_routing_v2_h_partial_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_H_PARTIAL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_H_PARTIAL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"
namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2HPartialLoad
{
public:
    __aicore__ inline MoeFinalizeRoutingV2HPartialLoad()
    {}

    __aicore__ inline void Init(
        GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scales,
        GM_ADDR expertIdx, GM_ADDR y, GM_ADDR workspace, const MoeFinalizeRoutingV2RegbaseTilingData* tilingDataPtr,
        TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        hasX1 = x1 != nullptr;
        hasX2 = x2 != nullptr;
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasScales = scales != nullptr;
        expandedXGm.SetGlobalBuffer((__gm__ T*)expandedX);
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        biasGm.SetGlobalBuffer((__gm__ T*)bias);
        scalesGm.SetGlobalBuffer((__gm__ S*)scales);
        expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        int32_t hFactorAlignedT = RoundUp<T>(tilingData->hFactor);
        int32_t hFactorAlignedFloat = RoundUp<float>(tilingData->hFactor);
        pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, hFactorAlignedFloat * sizeof(float));
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(biasQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            for (int64_t hIdx = 0; hIdx < tilingData->hLoop; hIdx += 1) {
                int64_t hFactor = (hIdx == tilingData->hLoop - 1) ? tilingData->tailHFactor : tilingData->hFactor;
                if (hasX1) {
                    x1Local = x1Que.AllocTensor<T>();
                    int64_t x1GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                         rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                    CopyIn(x1Gm[x1GmOffset], x1Local, 1, hFactor);
                    x1Que.EnQue(x1Local);
                    x1Local = x1Que.DeQue<T>();
                }
                if (hasX2) {
                    x2Local = x2Que.AllocTensor<T>();
                    int64_t x2GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                         rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                    CopyIn(x2Gm[x2GmOffset], x2Local, 1, hFactor);
                    x2Que.EnQue(x2Local);
                    x2Local = x2Que.DeQue<T>();
                }

                yLocal = yQue.AllocTensor<float>();
                ProcessX1AndX2(yLocal, x1Local, x2Local, hFactor, hasX1, hasX2);
                ProcessYWithInput(rowOuterIdx, hIdx, hFactor);
                yQue.EnQue(yLocal);

                if (hasX1) {
                    x1Que.FreeTensor(x1Local);
                }
                if (hasX2) {
                    x2Que.FreeTensor(x2Local);
                }
                yLocal = yQue.DeQue<float>();
                int64_t yGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                    rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, hFactor);
                yQue.FreeTensor(yLocal);
            }
        }
    }

private:
    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t hIdx, int64_t hFactor)
    {
        for (int64_t kIdx = 0; kIdx < tilingData->k; kIdx += 1) {
            SetExpandedRowIdxOffset(rowOuterIdx, kIdx);
            int64_t expandedRowIdxGmValue = expandedRowIdxGm.GetValue(expandedRowIdxOffset);
            if constexpr (dropPadMode == DROP_PAD_COLUMN || dropPadMode == DROP_PAD_ROW) {
                if (expandedRowIdxGmValue == INVALID_IDX) {
                    continue;
                }
            } else {
                if (expandedRowIdxGmValue >= tilingData->activeNum) {
                    continue;
                }
            }
            expandedXLocal = expandedXQue.AllocTensor<T>();
            CopyIn(
                expandedXGm[expandedRowIdxGmValue * tilingData->h + hIdx * tilingData->hFactor], expandedXLocal, 1,
                hFactor);
            expandedXQue.EnQue(expandedXLocal);
            if (hasBiasAndExpertIdx) {
                SetExpertIdxOffset(rowOuterIdx, kIdx);
                int64_t biasGmOffset =
                    expertIdxGm.GetValue(expertIdxOffset) * tilingData->h + hIdx * tilingData->hFactor;
                biasLocal = biasQue.AllocTensor<T>();
                CopyIn(biasGm[biasGmOffset], biasLocal, 1, hFactor);
                biasQue.EnQue(biasLocal);
            }
            if (hasScales) {
                SetScaleOffset(rowOuterIdx, kIdx);
                scale = scalesGm.GetValue(scaleOffset);
            }
            expandedXLocal = expandedXQue.DeQue<T>();
            if (hasBiasAndExpertIdx) {
                biasLocal = biasQue.DeQue<T>();
            }
            ProcessExpandXBiasScale<T, S>(
                yLocal, expandedXLocal, biasLocal, scale, hFactor, hasBiasAndExpertIdx, hasScales);
            expandedXQue.FreeTensor(expandedXLocal);
            if (hasBiasAndExpertIdx) {
                biasQue.FreeTensor(biasLocal);
            }
        }
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, hFactor);
        }
    }

    __aicore__ inline void SetExpandedRowIdxOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx + kIdx * tilingData->row;
        } else {
            expandedRowIdxOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        }
    }

    __aicore__ inline void SetExpertIdxOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expertIdxOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
        }
    }

    __aicore__ inline void SetScaleOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            scaleOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        } else {
            scaleOffset = expandedRowIdxOffset;
        }
    }

private:
    TPipe* pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData* tilingData;
    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> biasGm;
    GlobalTensor<S> scalesGm;
    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<T> yGm;

    LocalTensor<T> expandedXLocal;
    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;
    LocalTensor<T> biasLocal;
    LocalTensor<float> yLocal;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;

    S scale;
    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasScales{false};

    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
    int64_t scaleOffset{0};
};
} // namespace MoeFinalizeRoutingV2Regbase

#endif