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
 * \file moe_finalize_routing_fp_cuth_k2_optimized.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_FP_CUTH_K_TWO_OPTIMIZED
#define MOE_FINALIZE_ROUTING_V2_FP_CUTH_K_TWO_OPTIMIZED

#include "moe_finalize_routing_v2_common.h"
#include "kernel_tiling/kernel_tiling.h"

constexpr int64_t BUFFER_NUM_OPTIMIZED_V2 = 2;
namespace MoeFinalizeRoutingV2 {

using namespace AscendC;
template <typename T>
class MoeFinalizeRoutingV2FpCuthK2Optimized
{
public:
    __aicore__ inline MoeFinalizeRoutingV2FpCuthK2Optimized(
        const MoeFinalizeRoutingV2TilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(
        GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
        GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CkechColAlignment();
    __aicore__ inline void CopyIn(
        int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen, bool isPadH,
        int64_t rightPaddingH);
    __aicore__ inline void Compute(
        int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen, bool isPadH,
        int64_t rightPaddingH);
    __aicore__ inline void CopyOut(int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen);
    __aicore__ inline int64_t AlignmentProcess(int64_t param);
    __aicore__ inline int64_t PadProcessT(int64_t param);
    __aicore__ inline void PrepareData();

private:
    TPipe& pipe_;
    const MoeFinalizeRoutingV2TilingData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM_OPTIMIZED_V2> skip1Skip2ScalesQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_OPTIMIZED_V2> expertForSourceRowQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_OPTIMIZED_V2> outQueue_;

    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb0_;
    TBuf<QuePosition::VECCALC> biasBufDb0_;
    TBuf<QuePosition::VECCALC> expandedPermutedRowsBufDb1_;
    TBuf<QuePosition::VECCALC> biasBufDb1_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> expandedSrcToDstRowBuff_;

    GlobalTensor<T> gmExpandedPermutedRows_;
    GlobalTensor<T> gmSkip1_;
    GlobalTensor<T> gmSkip2_;
    GlobalTensor<T> gmBias_;
    GlobalTensor<T> gmScales_;
    GlobalTensor<int32_t> gmExpandedSrcToDstRow_;
    GlobalTensor<int32_t> gmExpertForSourceRow_;
    GlobalTensor<T> gmOut_;

    LocalTensor<int32_t> expandedSrcToDstRow_;

    int64_t workspace_{0};

    // input number
    int64_t inputSkipIdx_{0};
    int64_t inputScalesAndExpertIdx_{0};
    int64_t outputIdx_{0};
    int64_t curCoreHandleNumPerLoop_{0};

    const int64_t onceAlgnNum_{ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T))};

    bool isPadNormalH_ = false;
    bool isPadUnnormalH_ = false;
    bool isPadK_ = false;
    bool isPadKInt32_ = false;
    uint16_t rightPaddingNormalH_ = 0;
    uint16_t rightPaddingUnnormalH_ = 0;
    uint16_t rightPaddingK_ = 0;
    uint16_t rightPaddingKInt32_ = 0;

    int64_t curHandleNumTailLoop_{0};
    int64_t curNormLoopCount_{0};
    int64_t curTotalLoopCount_{0};

    bool isPadSourceToDstRow_ = false;
    uint16_t rightPaddingSourceToDstRow_ = 0;
    int64_t curCoreHandleNum_ = 0;
};

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpCuthK2Optimized<T>::AlignmentProcess(int64_t param)
{
    return (param + onceAlgnNum_ - 1) / onceAlgnNum_ * onceAlgnNum_;
}

template <typename T>
__aicore__ inline int64_t MoeFinalizeRoutingV2FpCuthK2Optimized<T>::PadProcessT(int64_t param)
{
    return onceAlgnNum_ - param % onceAlgnNum_;
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK2Optimized<T>::CkechColAlignment()
{
    if (tilingData_.normalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadNormalH_ = true;
        rightPaddingNormalH_ = PadProcessT(tilingData_.normalH);
    }

    if (tilingData_.unnormalH * sizeof(T) % ONE_BLK_SIZE) {
        isPadUnnormalH_ = true;
        rightPaddingUnnormalH_ = PadProcessT(tilingData_.unnormalH);
    }

    if (tilingData_.K * sizeof(T) % ONE_BLK_SIZE) {
        isPadK_ = true;
        rightPaddingK_ = PadProcessT(tilingData_.K);
    }

    if (tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
        isPadKInt32_ = true;
        rightPaddingKInt32_ = PadProcessInt32(tilingData_.K);
    }
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        if (curCoreHandleNum_ * INT32_BYTES % ONE_BLK_SIZE) {
            isPadSourceToDstRow_ = true;
            rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_);
        }
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        if (curCoreHandleNum_ * tilingData_.K * INT32_BYTES % ONE_BLK_SIZE) {
            isPadSourceToDstRow_ = true;
            rightPaddingSourceToDstRow_ = PadProcessInt32(curCoreHandleNum_ * tilingData_.K);
        }
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK2Optimized<T>::Init(
    GM_ADDR expandedPermutedRows, GM_ADDR expandedSrcToDstRow, GM_ADDR skip1, GM_ADDR skip2, GM_ADDR bias,
    GM_ADDR scales, GM_ADDR expertForSourceRow, GM_ADDR out, GM_ADDR workspace)
{
    if (GetBlockIdx() + 1 == tilingData_.usedCoreNum) {
        curCoreHandleNumPerLoop_ = tilingData_.tailCoreHandleNumPerLoop;
        curHandleNumTailLoop_ = tilingData_.tailCoreHandleNumTailLoop;
        curNormLoopCount_ = (tilingData_.tailCoreLoopNum - 1) * (tilingData_.hSliceNum + 1);
        curTotalLoopCount_ = tilingData_.tailCoreLoopNum * (tilingData_.hSliceNum + 1);
        curCoreHandleNum_ = tilingData_.tailCoreHandleNum;
    } else {
        curCoreHandleNumPerLoop_ = tilingData_.normalCoreHandleNumPerLoop;
        curHandleNumTailLoop_ = tilingData_.normalCoreHandleNumTailLoop;
        curNormLoopCount_ = (tilingData_.normalCoreLoopNum - 1) * (tilingData_.hSliceNum + 1);
        curTotalLoopCount_ = tilingData_.normalCoreLoopNum * (tilingData_.hSliceNum + 1);
        curCoreHandleNum_ = tilingData_.normalCoreHandleNum;
    }

    // 检查要处理的列数是否对齐以及应该如何对齐
    CkechColAlignment();

    // gmInput分核 && 输入偏移量初始化
    inputSkipIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    gmSkip1_.SetGlobalBuffer((__gm__ T*)skip1 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    if (tilingData_.skip2IsNull == 0) {
        gmSkip2_.SetGlobalBuffer((__gm__ T*)skip2 + inputSkipIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);
        pipe_.InitBuffer(
            skip1Skip2ScalesQueue_, BUFFER_NUM_OPTIMIZED_V2,
            tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T) * 2 +
                tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.K) * sizeof(T));
    } else {
        pipe_.InitBuffer(
            skip1Skip2ScalesQueue_, BUFFER_NUM_OPTIMIZED_V2,
            tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T) +
                tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.K) * sizeof(T));
    }

    inputScalesAndExpertIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K;
    gmScales_.SetGlobalBuffer(
        (__gm__ T*)scales + inputScalesAndExpertIdx_, tilingData_.normalCoreHandleNum * tilingData_.K);
    gmExpertForSourceRow_.SetGlobalBuffer(
        (__gm__ int32_t*)expertForSourceRow + inputScalesAndExpertIdx_,
        tilingData_.normalCoreHandleNum * tilingData_.K);

    gmExpandedSrcToDstRow_.SetGlobalBuffer(
        (__gm__ int32_t*)expandedSrcToDstRow, tilingData_.totalRowNum * tilingData_.K);

    outputIdx_ = GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.H;
    gmOut_.SetGlobalBuffer((__gm__ T*)out + outputIdx_, tilingData_.normalCoreHandleNum * tilingData_.H);

    gmExpandedPermutedRows_.SetGlobalBuffer(
        (__gm__ T*)expandedPermutedRows, tilingData_.totalRowNum * tilingData_.K * tilingData_.H);
    gmBias_.SetGlobalBuffer((__gm__ T*)bias, tilingData_.biasRowNum * tilingData_.H);

    // 申请 buffer 空间
    pipe_.InitBuffer(
        expertForSourceRowQueue_, BUFFER_NUM_OPTIMIZED_V2,
        tilingData_.normalCoreHandleNumPerLoop * Int32AlignmentProcess(tilingData_.K) * sizeof(int32_t));
    pipe_.InitBuffer(
        outQueue_, BUFFER_NUM_OPTIMIZED_V2,
        tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
    pipe_.InitBuffer(biasBufDb0_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(expandedPermutedRowsBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
    pipe_.InitBuffer(biasBufDb1_, AlignmentProcess(tilingData_.normalH) * sizeof(T));
    pipe_.InitBuffer(tmpBuf_, AlignmentProcess(tilingData_.normalH) * sizeof(T));

    pipe_.InitBuffer(
        expandedSrcToDstRowBuff_, Int32AlignmentProcess(curCoreHandleNum_) * tilingData_.K * sizeof(int32_t));
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK2Optimized<T>::PrepareData()
{
    expandedSrcToDstRow_ = expandedSrcToDstRowBuff_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> padParamsSrcToDstRow{
        isPadSourceToDstRow_, 0, static_cast<uint8_t>(rightPaddingSourceToDstRow_), 0};
    if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
        DataCopyExtParams copyParamsSrcToDstRow{
            static_cast<uint16_t>(tilingData_.K), static_cast<uint32_t>(curCoreHandleNum_ * sizeof(int32_t)),
            static_cast<uint32_t>((tilingData_.totalRowNum - curCoreHandleNum_) * sizeof(int32_t)), 0, 0};
        DataCopyPadCustom<int32_t, DataCopyExtParams, DataCopyPadExtParams<int32_t>>(
            expandedSrcToDstRow_, gmExpandedSrcToDstRow_[GetBlockIdx() * tilingData_.normalCoreHandleNum],
            copyParamsSrcToDstRow, padParamsSrcToDstRow);
    } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
        DataCopyExtParams copyParamsSrcToDstRow{
            static_cast<uint16_t>(1), static_cast<uint32_t>(curCoreHandleNum_ * tilingData_.K * sizeof(int32_t)), 0, 0,
            0};
        DataCopyPadCustom<int32_t, DataCopyExtParams, DataCopyPadExtParams<int32_t>>(
            expandedSrcToDstRow_,
            gmExpandedSrcToDstRow_[GetBlockIdx() * tilingData_.normalCoreHandleNum * tilingData_.K],
            copyParamsSrcToDstRow, padParamsSrcToDstRow);
    }
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK2Optimized<T>::CopyIn(
    int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.AllocTensor<int32_t>();

    LocalTensor<T> skip1Skip2ScalesLocal = skip1Skip2ScalesQueue_.AllocTensor<T>();

    // ---------------------------- [Expert] -------------------------------
    DataCopyParams copyParamsExpert{
        static_cast<uint16_t>(lineNumInCurrentLoop), static_cast<uint16_t>(tilingData_.K * sizeof(int32_t)), 0, 0};
    DataCopyPadParams padParamsExpert{isPadKInt32_, 0, static_cast<uint8_t>(rightPaddingKInt32_), 0};
    DataCopyPadCustom<int32_t, DataCopyParams, DataCopyPadParams>(
        expertForSourceRowLocal,
        gmExpertForSourceRow_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.K],
        copyParamsExpert, padParamsExpert);

    // ---------------------------- [Scales] -------------------------------
    DataCopyParams copyParamsScales{
        static_cast<uint16_t>(lineNumInCurrentLoop), static_cast<uint16_t>(tilingData_.K * sizeof(T)), 0, 0};
    DataCopyPadParams padParamsScales{isPadK_, 0, static_cast<uint8_t>(rightPaddingK_), 0};
    DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams>(
        skip1Skip2ScalesLocal[tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH)],
        gmScales_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.K], copyParamsScales,
        padParamsScales);

    expertForSourceRowQueue_.EnQue(expertForSourceRowLocal);

    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

    // ---------------------------- [Skip] -------------------------------
    DataCopyParams copyParamsSkip{
        static_cast<uint16_t>(lineNumInCurrentLoop), static_cast<uint16_t>(dataLen * sizeof(T)),
        static_cast<uint16_t>((tilingData_.H - dataLen) * sizeof(T)), 0};
    DataCopyPadParams padParamsSkip{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams>(
        skip1Skip2ScalesLocal[0],
        gmSkip1_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias],
        copyParamsSkip, padParamsSkip);

    if (tilingData_.skip2IsNull == 0) {
        DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams>(
            skip1Skip2ScalesLocal
                [tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) +
                 tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.K)],
            gmSkip2_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias],
            copyParamsSkip, padParamsSkip);
    }

    skip1Skip2ScalesQueue_.EnQue(skip1Skip2ScalesLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK2Optimized<T>::Compute(
    int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen, bool isPadH, int64_t rightPaddingH)
{
    LocalTensor<int32_t> expertForSourceRowLocal = expertForSourceRowQueue_.DeQue<int32_t>();
    LocalTensor<T> skip1Skip2ScalesLocal = skip1Skip2ScalesQueue_.DeQue<T>();
    LocalTensor<T> outLocal = outQueue_.AllocTensor<T>();

    LocalTensor<T> expandedPermutedTmpUbDb0 = expandedPermutedRowsBufDb0_.Get<T>();
    LocalTensor<T> biasTmpUbDb0 = biasBufDb0_.Get<T>();

    LocalTensor<T> expandedPermutedTmpUbDb1 = expandedPermutedRowsBufDb1_.Get<T>();
    LocalTensor<T> biasTmpUbDb1 = biasBufDb1_.Get<T>();

    LocalTensor<T> tmpLocal = tmpBuf_.Get<T>();
    int64_t baseRowLine = nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_;

    DataCopyParams copyParams{1, static_cast<uint16_t>(dataLen * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{isPadH, 0, static_cast<uint8_t>(rightPaddingH), 0};

    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

    for (int64_t i = 0; i < lineNumInCurrentLoop; i++) {
        int64_t outRowIndex = i * AlignmentProcess(tilingData_.normalH);
        for (int64_t j = 0; j < tilingData_.K / PARALLEL_NUM; j++) {
            /*******************************乒***********************************************/
            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
            int64_t expandedSrcToDstRowIndexDb0 = 0;
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb0 =
                    baseRowLine + i + PARALLEL_NUM * j * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb0 = baseRowLine * tilingData_.K + i * tilingData_.K + j * PARALLEL_NUM;
            }
            int64_t expandedPermutedRowsIndexDb0 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb0);

            int64_t biasIndexDb0 =
                expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams>(
                expandedPermutedTmpUbDb0, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb0 * tilingData_.H + bias],
                copyParams, padParams);
            DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams>(biasTmpUbDb0, gmBias_[biasIndexDb0 * tilingData_.H + bias], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            /*******************************乓***********************************************/
            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID1);
            }
            int64_t expandedSrcToDstRowIndexDb1 = 0;
            if (tilingData_.dropPadMode == MODE_VALUE_0 || tilingData_.dropPadMode == MODE_VALUE_1) {
                expandedSrcToDstRowIndexDb1 =
                    baseRowLine + i + (PARALLEL_NUM * j + 1) * Int32AlignmentProcess(curCoreHandleNum_);
            } else if (tilingData_.dropPadMode == MODE_VALUE_2 || tilingData_.dropPadMode == MODE_VALUE_3) {
                expandedSrcToDstRowIndexDb1 = baseRowLine * tilingData_.K + i * tilingData_.K + j * PARALLEL_NUM + 1;
            }
            int64_t expandedPermutedRowsIndexDb1 = expandedSrcToDstRow_.GetValue(expandedSrcToDstRowIndexDb1);
            int64_t biasIndexDb1 =
                expertForSourceRowLocal.GetValue(i * Int32AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j + 1);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID1);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams>(
                expandedPermutedTmpUbDb1, gmExpandedPermutedRows_[expandedPermutedRowsIndexDb1 * tilingData_.H + bias],
                copyParams, padParams);
            DataCopyPadCustom<T, DataCopyParams, DataCopyPadParams>(
                biasTmpUbDb1, gmBias_[biasIndexDb1 * tilingData_.H + bias], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
            }
            T scalesValDb0 =
                skip1Skip2ScalesLocal[tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH)]
                    .GetValue(i * AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j);
            SetFlag<HardEvent::S_V>(EVENT_ID2);

            if (!((i == 0) && (j == 0))) {
                WaitFlag<HardEvent::V_S>(EVENT_ID3);
            }
            T scalesValDb1 =
                skip1Skip2ScalesLocal[tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH)]
                    .GetValue(i * AlignmentProcess(tilingData_.K) + PARALLEL_NUM * j + 1);
            SetFlag<HardEvent::S_V>(EVENT_ID3);

            if ((i == 0) && (j == 0)) {
                if (tilingData_.skip2IsNull == 0) {
                    Add(outLocal, skip1Skip2ScalesLocal,
                        skip1Skip2ScalesLocal
                            [tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.normalH) +
                             tilingData_.normalCoreHandleNumPerLoop * AlignmentProcess(tilingData_.K)],
                        lineNumInCurrentLoop * AlignmentProcess(dataLen));
                } else {
                    Adds(outLocal, skip1Skip2ScalesLocal, (T)0, lineNumInCurrentLoop * AlignmentProcess(dataLen));
                }
            }

            /*******************************乒***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add(skip1Skip2ScalesLocal, expandedPermutedTmpUbDb0, biasTmpUbDb0, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (tilingData_.K / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
            }
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
            Muls(skip1Skip2ScalesLocal, skip1Skip2ScalesLocal, scalesValDb0, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (tilingData_.K / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID2);
            }
            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], skip1Skip2ScalesLocal, dataLen);

            /*******************************乓***********************************************/
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Add(tmpLocal, expandedPermutedTmpUbDb1, biasTmpUbDb1, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (tilingData_.K / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID1);
            }
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::S_V>(EVENT_ID3);
            Muls(tmpLocal, tmpLocal, scalesValDb1, dataLen);
            if (!((i == (lineNumInCurrentLoop - 1)) && (j == (tilingData_.K / PARALLEL_NUM - 1)))) {
                SetFlag<HardEvent::V_S>(EVENT_ID3);
            }

            PipeBarrier<PIPE_V>();
            Add(outLocal[outRowIndex], outLocal[outRowIndex], tmpLocal, dataLen);
        }
    }

    expertForSourceRowQueue_.FreeTensor(expertForSourceRowLocal);
    skip1Skip2ScalesQueue_.FreeTensor(skip1Skip2ScalesLocal);
    outQueue_.EnQue(outLocal);
}
template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK2Optimized<T>::CopyOut(
    int64_t nLoopIdx, int64_t lineNumInCurrentLoop, int64_t bias, int64_t dataLen)
{
    LocalTensor<T> outLocal = outQueue_.DeQue<T>();
    DataCopyParams copyParams{
        static_cast<uint16_t>(lineNumInCurrentLoop), static_cast<uint16_t>(dataLen * sizeof(T)), 0,
        static_cast<uint16_t>((tilingData_.H - dataLen) * sizeof(T))};
    DataCopyPadCustom<T>(
        gmOut_[nLoopIdx / (tilingData_.hSliceNum + 1) * curCoreHandleNumPerLoop_ * tilingData_.H + bias], outLocal,
        copyParams);
    outQueue_.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MoeFinalizeRoutingV2FpCuthK2Optimized<T>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }
    PrepareData();
    for (int64_t n = 0; n < curNormLoopCount_; n++) {
        bool isNormalH = (n + 1) % (tilingData_.hSliceNum + 1) != 0;
        int64_t bias = isNormalH ? (n % (tilingData_.hSliceNum + 1)) * tilingData_.normalH :
                                   tilingData_.hSliceNum * tilingData_.normalH;
        int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
        int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
        bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;
        CopyIn(n, curCoreHandleNumPerLoop_, bias, dataLen, isPadH, rightPaddingH);
        Compute(n, curCoreHandleNumPerLoop_, bias, dataLen, isPadH, rightPaddingH);
        CopyOut(n, curCoreHandleNumPerLoop_, bias, dataLen);
    }
    for (int64_t n = curNormLoopCount_; n < curTotalLoopCount_; n++) {
        bool isNormalH = (n + 1) % (tilingData_.hSliceNum + 1) != 0;
        int64_t bias = isNormalH ? (n % (tilingData_.hSliceNum + 1)) * tilingData_.normalH :
                                   tilingData_.hSliceNum * tilingData_.normalH;
        int64_t dataLen = isNormalH ? tilingData_.normalH : tilingData_.unnormalH;
        int64_t rightPaddingH = isNormalH ? rightPaddingNormalH_ : rightPaddingUnnormalH_;
        bool isPadH = isNormalH ? isPadNormalH_ : isPadUnnormalH_;

        CopyIn(n, curHandleNumTailLoop_, bias, dataLen, isPadH, rightPaddingH);
        Compute(n, curHandleNumTailLoop_, bias, dataLen, isPadH, rightPaddingH);
        CopyOut(n, curHandleNumTailLoop_, bias, dataLen);
    }
}
} // namespace MoeFinalizeRoutingV2
#endif // MOE_FINALIZE_ROUTING_V2_FP_CUTH_K_TWO_OPTIMIZED
