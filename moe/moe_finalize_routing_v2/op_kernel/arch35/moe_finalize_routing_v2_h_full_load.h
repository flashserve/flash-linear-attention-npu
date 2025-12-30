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
 * \file moe_finalize_routing_v2_h_full_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_H_FULL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_H_FULL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"

namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2HFullLoad
{
public:
    __aicore__ inline MoeFinalizeRoutingV2HFullLoad()
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

        int32_t rowFactorHAlignedT = RoundUp<T>(tilingData->rowFactor * tilingData->h);
        int32_t rowFactorHAlignedFloat = RoundUp<float>(tilingData->rowFactor * tilingData->h);
        int32_t kFactorAlignedT = RoundUp<T>(tilingData->kFactor);
        pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, tilingData->kFactor * tilingData->hAligned * sizeof(T));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, rowFactorHAlignedFloat * sizeof(float));
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(biasQue, DOUBLE_BUFFER, tilingData->kFactor * tilingData->hAligned * sizeof(T));
        }
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasScales) {
            pipe->InitBuffer(scalesQue, DOUBLE_BUFFER, kFactorAlignedT * sizeof(S));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                       tilingData->tailRowFactorOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            if (hasX1) {
                x1Local = x1Que.AllocTensor<T>();
                int64_t x1GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                     rowOuterIdx * tilingData->rowFactor * tilingData->h;
                CopyIn(x1Gm[x1GmOffset], x1Local, 1, rowInnerLoop * tilingData->h);
                x1Que.EnQue(x1Local);
                x1Local = x1Que.DeQue<T>();
            }
            if (hasX2) {
                x2Local = x2Que.AllocTensor<T>();
                int64_t x2GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                     rowOuterIdx * tilingData->rowFactor * tilingData->h;
                CopyIn(x2Gm[x2GmOffset], x2Local, 1, rowInnerLoop * tilingData->h);
                x2Que.EnQue(x2Local);
                x2Local = x2Que.DeQue<T>();
            }

            yLocal = yQue.AllocTensor<float>();
            ProcessX1AndX2(yLocal, x1Local, x2Local, rowInnerLoop * tilingData->h, hasX1, hasX2);
            ProcessYWithInput(rowOuterIdx, rowInnerLoop);
            yQue.EnQue(yLocal);

            if (hasX1) {
                x1Que.FreeTensor(x1Local);
            }
            if (hasX2) {
                x2Que.FreeTensor(x2Local);
            }
            yLocal = yQue.DeQue<float>();
            int64_t yGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                rowOuterIdx * tilingData->rowFactor * tilingData->h;
            CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, rowInnerLoop * tilingData->h);
            yQue.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            for (int64_t kOuterIdx = 0; kOuterIdx < tilingData->kLoop; kOuterIdx += 1) {
                int64_t curKFactor = kOuterIdx == tilingData->kLoop - 1 ? tilingData->tailKFactor : tilingData->kFactor;
                if (hasScales) {
                    scalesLocal = scalesQue.AllocTensor<S>();
                    int64_t scalesGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                             rowOuterIdx * tilingData->rowFactor * tilingData->k +
                                             rowInnerIdx * tilingData->k + kOuterIdx * tilingData->kFactor;
                    CopyIn(scalesGm[scalesGmOffset], scalesLocal, 1, curKFactor);
                    scalesQue.EnQue(scalesLocal);
                    scalesLocal = scalesQue.DeQue<S>();
                }
                expandedXLocal = expandedXQue.AllocTensor<T>();
                if (hasBiasAndExpertIdx) {
                    biasLocal = biasQue.AllocTensor<T>();
                }
                CopyExpanedXAndBiasScales(curKFactor, rowOuterIdx, rowInnerIdx, kOuterIdx);
                expandedXQue.EnQue(expandedXLocal);
                expandedXLocal = expandedXQue.DeQue<T>();
                if (hasBiasAndExpertIdx) {
                    biasQue.EnQue(biasLocal);
                    biasLocal = biasQue.DeQue<T>();
                }
                if (hasScales) {
                    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                    SetFlag<HardEvent::S_V>(eventId);
                    WaitFlag<HardEvent::S_V>(eventId);
                }
                ProcessExpandXBiasScaleOptimized<T>(
                    yLocal[rowInnerIdx * tilingData->h], expandedXLocal, biasLocal, scalesLocal, validK, tilingData->h,
                    hasBiasAndExpertIdx, hasScales);
                expandedXQue.FreeTensor(expandedXLocal);
                if (hasBiasAndExpertIdx) {
                    biasQue.FreeTensor(biasLocal);
                }
                if (hasScales) {
                    scalesQue.FreeTensor(scalesLocal);
                }
            }
        }
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline void CopyExpanedXAndBiasScales(
        int64_t curKFactor, int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOuterIdx)
    {
        validK = 0;
        for (int64_t kInnerIdx = 0; kInnerIdx < curKFactor; kInnerIdx += 1) {
            SetExpandedRowIdxOffset(rowOuterIdx, rowInnerIdx, kOuterIdx, kInnerIdx);
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
            CopyIn(
                expandedXGm[expandedRowIdxGmValue * tilingData->h], expandedXLocal[validK * tilingData->hAligned], 1,
                tilingData->h);
            if (hasBiasAndExpertIdx) {
                SetExpertIdxOffset(rowOuterIdx, rowInnerIdx, kOuterIdx, kInnerIdx);
                int64_t biasGmOffset = expertIdxGm.GetValue(expertIdxOffset) * tilingData->h;
                CopyIn(biasGm[biasGmOffset], biasLocal[validK * tilingData->hAligned], 1, tilingData->h);
            }
            if (hasScales) {
                S scaleValue = scalesLocal.GetValue(kInnerIdx);
                scalesLocal.SetValue(validK, scaleValue);
            }
            validK += 1;
        }
    }

    __aicore__ inline void SetExpandedRowIdxOffset(
        int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOuterIdx, int64_t kInnerIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor +
                                   rowInnerIdx + (kOuterIdx * tilingData->kFactor + kInnerIdx) * tilingData->row;
        } else {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                   rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                                   kOuterIdx * tilingData->kFactor + kInnerIdx;
        }
    }

    __aicore__ inline void SetExpertIdxOffset(
        int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOuterIdx, int64_t kInnerIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expertIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                              rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                              kOuterIdx * tilingData->kFactor + kInnerIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
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
    LocalTensor<S> scalesLocal;
    LocalTensor<float> yLocal;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scalesQue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;

    S scale;
    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasScales{false};

    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
    int64_t scaleOffset{0};
    int16_t validK{0};
};
} // namespace MoeFinalizeRoutingV2Regbase

#endif