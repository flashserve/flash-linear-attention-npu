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
 * \file masked_select.h
 * \brief
 */
#ifndef MASKED_SELECT_H_
#define MASKED_SELECT_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "moe_sort_base_with_routing_map.h"
using namespace AscendC;
#define IS_1_BYTES_TYPE is_same<T, int8_t>::value || is_same<T, uint8_t>::value
#define IS_2_BYTES_TYPE is_same<T, int16_t>::value || is_same<T, uint16_t>::value || is_same<T, half>::value || is_same<T, bfloat16_t>::value
#define IS_4_BYTES_TYPE is_same<T, int32_t>::value || is_same<T, uint32_t>::value || is_same<T, float>::value
#define IS_8_BYTES_TYPE is_same<T, int64_t>::value || is_same<T, uint64_t>::value || is_same<T, double>::value

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int64_t BUFFER_32B_CALNUM = 32 / sizeof(float);

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename, typename>
struct is_same : public false_type {};

template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};


__aicore__ inline int32_t AlignUp(int32_t a, int32_t b) {
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b) {
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline void VToMTE3Sync() {
  event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
  WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

// 二分累加到2 * offset以内，再累加成offset大小
__aicore__ inline void BinaryAddFunc(LocalTensor<float> tmpBuffer, int32_t hiddensizeLen,
                                     int32_t threshold, int32_t offset) {
    int32_t totalLen = hiddensizeLen;
    int32_t halfLen =  AlignUp(CeilDiv(totalLen, 2), BUFFER_32B_CALNUM);
    while (totalLen > threshold) {
        PipeBarrier<PIPE_V>();
        Add(tmpBuffer, tmpBuffer, tmpBuffer[halfLen], totalLen - halfLen);
        PipeBarrier<PIPE_V>();
        totalLen = halfLen;
        halfLen =  AlignUp(CeilDiv(totalLen, 2), BUFFER_32B_CALNUM);
    }
    PipeBarrier<PIPE_V>();
    Add(tmpBuffer, tmpBuffer, tmpBuffer[offset], totalLen - offset);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumFunc(LocalTensor<float> dstBuffer, LocalTensor<float> tmpBuffer, int32_t hiddensizeLen)
{
    // 定义阈值常量
    const int32_t chunkLimit4096 = 4096;
    const int32_t chunkLimit512 = 512;
    const int32_t chunkLimit64 = 64;
    // 定义BlockReduceSum参数常量
    const int32_t repeatTime64 = 64;
    const int32_t repeatTime8 = 8;
    const int32_t repeatTime1 = 1;
    const int32_t maskValue64 = 64;
    const int32_t dstStride1 = 1;
    const int32_t srcBlockStride1 = 1;
    const int32_t srcRepeatStride8 = 8;
    // 定义中间reduce目标大小
    const int32_t targetSize8192 = 8192;
    const int32_t targetSize1024 = 1024;
    const int32_t targetSize128 = 128;
    // 定义WholeReduceSum参数
    const int32_t wholeReduceInputSize8 = 8;
    if (hiddensizeLen >= chunkLimit4096) {
// 二分累加到8192以内，再累加成4096大小，用blockreducesum
#ifndef __CCE_KT_TEST__
        BinaryAddFunc(tmpBuffer, hiddensizeLen, targetSize8192, chunkLimit4096);
#endif
        // 用BlockReduceSum+WholeReduceSum 把4096 reduce成1个
        PipeBarrier<PIPE_V>();
        BlockReduceSum(tmpBuffer, tmpBuffer, repeatTime64, maskValue64, dstStride1, srcBlockStride1,
                       srcRepeatStride8); // 输出512
        PipeBarrier<PIPE_V>();
        BlockReduceSum(tmpBuffer, tmpBuffer, repeatTime8, maskValue64, dstStride1, srcBlockStride1,
                       srcRepeatStride8); // 输出64
        PipeBarrier<PIPE_V>();
        BlockReduceSum(tmpBuffer, tmpBuffer, repeatTime1, maskValue64, dstStride1, srcBlockStride1,
                       srcRepeatStride8); // 输出8
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstBuffer, tmpBuffer, wholeReduceInputSize8, dstStride1, dstStride1, dstStride1,
                       srcRepeatStride8); // 输出1
        PipeBarrier<PIPE_V>();
    } else if (hiddensizeLen >= chunkLimit512) {
        // 二分累加到1024以内，再累加成512大小，用blockreducesum
        PipeBarrier<PIPE_V>();
        BinaryAddFunc(tmpBuffer, hiddensizeLen, targetSize1024, chunkLimit512); // 累加成512大小
        PipeBarrier<PIPE_V>();
        // 用BlockReduceSum+WholeReduceSum 把512 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, repeatTime8, maskValue64, dstStride1, srcBlockStride1,
                       srcRepeatStride8); // 输出64
        PipeBarrier<PIPE_V>();
        BlockReduceSum(tmpBuffer, tmpBuffer, repeatTime1, maskValue64, dstStride1, srcBlockStride1,
                       srcRepeatStride8); // 输出8
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstBuffer, tmpBuffer, wholeReduceInputSize8, dstStride1, dstStride1, dstStride1,
                       srcRepeatStride8); // 输出1
        PipeBarrier<PIPE_V>();
    } else if (hiddensizeLen >= chunkLimit64) {
        // 二分累加到128以内，再累加成64大小，用blockreducesum
        PipeBarrier<PIPE_V>();
        BinaryAddFunc(tmpBuffer, hiddensizeLen, targetSize128, chunkLimit64); // 累加成64大小
        PipeBarrier<PIPE_V>();
        // 用BlockReduceSum+WholeReduceSum 把64 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, repeatTime1, maskValue64, dstStride1, srcBlockStride1,
                       srcRepeatStride8); // 输出8
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstBuffer, tmpBuffer, wholeReduceInputSize8, dstStride1, dstStride1, dstStride1,
                       srcRepeatStride8); // 输出1
        PipeBarrier<PIPE_V>();
    } else {
        // 64个以内的元素直接WholeReduceSum成1个
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dstBuffer, tmpBuffer, hiddensizeLen, dstStride1, dstStride1, dstStride1, srcRepeatStride8);
        PipeBarrier<PIPE_V>();
    }
}

namespace AscendC {

constexpr uint32_t SHAPEOUT_SIZE = 2;
constexpr uint32_t BIT_NUM_PER_BYTE = 8;
constexpr uint32_t HEAD_BLOCK_SIZE = 64;
constexpr uint32_t OFFSET_SHIFT_BITS = 3; // offset偏移量移位输，<<3 等价于 *8
constexpr uint32_t INT64_LENGTH_IN_INT32 = 2; // INT64 相当于 2个int32长
constexpr uint32_t GATHER_RESULT_STRIDE = 8;
constexpr uint32_t DATA_ALIGN = 256;

template <typename T>
class KernelMaskedSelectV3 {
public:
    __aicore__ inline KernelMaskedSelectV3 () {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace, 
                                const MaskedSelectMTUTilingData* maskedSelectTilingData, TPipe* tPipe)
    {   
        this->pipe = tPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        __gm__ T* globalWorkTensor = (__gm__ T*)((__gm__ uint64_t*)workspace);
        blockIdx = GetBlockIdx();
        this->formerNum = maskedSelectTilingData->formerNum;
        this->formerLength = maskedSelectTilingData->formerLength;
        this->formertileNum = maskedSelectTilingData->formertileNum;
        this->formertileLength = maskedSelectTilingData->formertileLength;
        this->formerlasttileLength = maskedSelectTilingData->formerlasttileLength;
        this->formerlasttileLength = maskedSelectTilingData->formerlasttileLength;

        this->needCoreNum = maskedSelectTilingData->needCoreNum; // 修改
        blockDim = this->needCoreNum;
        this->tailNum = maskedSelectTilingData->tailNum;
        this->tailLength = maskedSelectTilingData->tailLength;
        this->tailtileNum = maskedSelectTilingData->tailtileNum;
        this->tailtileLength = maskedSelectTilingData->tailtileLength;
        this->taillasttileLength = maskedSelectTilingData->taillasttileLength;
        this->tokenNum = maskedSelectTilingData->tokenNum;
        if (blockIdx < this->formerNum) {  //分到大块核的处理
            this->tileLength = this->formertileLength / BUFFER_NUM;
            this->lasttileLength = this->formerlasttileLength / BUFFER_NUM;
            this->tileNum = this->formertileNum * BUFFER_NUM;
            xGlobal.SetGlobalBuffer((__gm__ T*)x + this->formerLength * blockIdx, this->formerLength);
            maskGlobal.SetGlobalBuffer((__gm__ uint8_t*)mask + this->formerLength * blockIdx, this->formerLength);
            workGlobal.SetGlobalBuffer(globalWorkTensor + this->formerLength * blockIdx, this->formerLength);
        } else {  //分到小块核的处理，需要处理的数据量比大核少alignNum个
            this->tileLength = this->tailtileLength / BUFFER_NUM;
            this->lasttileLength = this->taillasttileLength / BUFFER_NUM;
            this->tileNum = this->tailtileNum * BUFFER_NUM;

            xGlobal.SetGlobalBuffer(
                (__gm__ T*)x + this->formerLength * this->formerNum +
                    this->tailLength * (blockIdx - this->formerNum),
                this->tailLength);
            maskGlobal.SetGlobalBuffer(
                (__gm__ uint8_t*)mask + this->formerLength * this->formerNum +
                    this->tailLength * (blockIdx - this->formerNum),
                this->tailLength);
            workGlobal.SetGlobalBuffer(
                globalWorkTensor + this->formerLength * this->formerNum +
                    this->tailLength * (blockIdx - this->formerNum),
                this->tailLength);
        }
        uint64_t alignNum = DATA_ALIGN / sizeof(half);       // 256/<8>=32
        tileLengthAlign = (tileLength + alignNum - 1) / alignNum * alignNum;
        offsetGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace, blockDim);
        pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLengthAlign * sizeof(T));
        pipe->InitBuffer(inQueueMask, BUFFER_NUM, this->tileLengthAlign * sizeof(uint8_t));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, this->tileLengthAlign * sizeof(T));

        pipe->InitBuffer(moveQue, BUFFER_NUM, this->tileLengthAlign * sizeof(T));
        pipe->InitBuffer(offsetBuf, BLOCK_SIZE);

        pipe->InitBuffer(sumBuf, BLOCK_SIZE);

        pipe->InitBuffer(maskCastBuf, this->tileLengthAlign * sizeof(float));
        pipe->InitBuffer(bitMaskBuf, this->tileLengthAlign);

        if constexpr (IS_1_BYTES_TYPE) {
            pipe->InitBuffer(xCastBuf, this->tileLengthAlign * sizeof(half));
            pipe->InitBuffer(yCastBuf, this->tileLengthAlign * sizeof(half));
        }
        offsetLocal = offsetBuf.Get<int32_t>();
    }

    __aicore__ inline void Process(GM_ADDR y)
    {
        if(GetBlockIdx() < this->needCoreNum){
            int32_t loopCount = this->tileNum;
            PipeBarrier<PIPE_V>();
            Duplicate(offsetLocal, 0, 8);
            
            PipeBarrier<PIPE_V>();
    
            //GYW 先处理可以整分的。
            maskHalfLocal = inQueueX.AllocTensor<half>();
            for (int32_t i = 0; i < loopCount; ++i) {
                CopyInMask(i);
                ComputeOffset(i);
            }
            inQueueX.FreeTensor(maskHalfLocal);
            VToMTE3Sync();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(offsetGlobal[blockIdx], offsetLocal, copyParams);        //workspace 写入 offset
        }

        SyncAll();

        if(GetBlockIdx() < this->needCoreNum){
            int32_t loopCount = this->tileNum;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
            PipeBarrier<PIPE_ALL>();

            uint64_t ind = 0;
            copyParams.blockLen = static_cast<uint32_t>(blockDim * sizeof(int32_t));
            DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            DataCopyPad(offsetLocal, offsetGlobal, copyParams, padParams);        //workspace 写入 offset
            PipeBarrier<PIPE_ALL>();
    
            for (int32_t i = 0; i < blockIdx; i++) { 
                ind += offsetLocal.GetValue(i);
            }
            this->outOffset = 0;
            
            yGlobal.SetGlobalBuffer((__gm__ T*)y + ind);
            PipeBarrier<PIPE_ALL>();
    
            for (int32_t i = 0; i < loopCount; ++i) {
                CopyIn(i);
                Compute(i);
                CopyOut2GM();
                outOffset += rsvdCnt;
            }
        }
        //搬运至GM
}
private:
    __aicore__ inline void CopyInMove(int32_t progress, int32_t length)
    {
        LocalTensor<T> xLocal = moveQue.AllocTensor<T>();
        
        if constexpr (IS_8_BYTES_TYPE) {//int64 uint64 double
            DataCopyPadDoubleWord(xLocal, workGlobal[progress * (this->tileLength)], length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, workGlobal[progress * (this->tileLength)], copyParams, padParams);
        }
        moveQue.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutMove(int32_t progress,int32_t length)
    {
        LocalTensor<T> yLocal = moveQue.DeQue<T>();
        
        if constexpr (IS_8_BYTES_TYPE) {
            DataCopyPadDoubleWord(yGlobal[progress * (this->tileLength)], yLocal, length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGlobal[progress * (this->tileLength)], yLocal, copyParams);
        }
        moveQue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
        uint64_t ind = progress * this->tileLength;
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            // 最后一个block，最后一个tile
            length = this->lasttileLength;
        } 

        if constexpr (IS_8_BYTES_TYPE) {//int64 uint64 double
            DataCopyPadDoubleWord(xLocal, xGlobal[ind], length);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, xGlobal[ind], copyParams, padParams);
        }

        {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(length), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
            DataCopyPad(maskLocal, maskGlobal[ind], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
        inQueueMask.EnQue(maskLocal);
    }

    __aicore__ inline void CopyInMask(int32_t progress)
    {
        LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
        uint64_t ind = progress * this->tileLength;
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            // 最后一个block，最后一个tile
            length = this->lasttileLength;
        } 

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(length), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(maskLocal, maskGlobal[ind], copyParams, padParams);

        inQueueMask.EnQue(maskLocal);
    }

__aicore__ inline void GenerateMask(const LocalTensor<uint8_t>& mask, LocalTensor<uint8_t>& bitMask,uint32_t count)
    {
        LocalTensor<half> maskCastLocal = maskCastBuf.Get<half>();
        Cast(maskCastLocal, mask, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        if constexpr (IS_8_BYTES_TYPE) {
            LocalTensor<int16_t> maskCastInt16 = maskCastLocal.template ReinterpretCast<int16_t>();
            LocalTensor<int16_t> maskCastInt16Shift = maskCastLocal[this->tileLength].template ReinterpretCast<int16_t>();
            Cast(maskCastInt16, maskCastLocal, RoundMode::CAST_ROUND, this->tileLength);

            ShiftLeft(maskCastInt16Shift, maskCastInt16, static_cast<int16_t>(BIT_NUM_PER_BYTE), this->tileLengthAlign);
            Add(maskCastInt16Shift, maskCastInt16, maskCastInt16Shift, this->tileLength);

            Cast(maskCastLocal, maskCastInt16Shift.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, this->tileLengthAlign * INT64_LENGTH_IN_INT32);
            CompareScalar(bitMask, maskCastLocal, static_cast<half>(1.0), CMPMODE::EQ, this->tileLength * INT64_LENGTH_IN_INT32);
        } else {
            CompareScalar(bitMask, maskCastLocal, static_cast<half>(1.0), CMPMODE::EQ, this->tileLengthAlign);
        }
        PipeBarrier<PIPE_V>();
    }

__aicore__ inline void GatherResult(LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
                                        const LocalTensor<uint8_t>& bitMaskLocal, int32_t count)
    {
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = 1;
        params.src0RepeatStride = GATHER_RESULT_STRIDE;
        params.src1RepeatStride = 1;

        if constexpr (IS_8_BYTES_TYPE) {
            uint32_t mask = count * INT64_LENGTH_IN_INT32;
            LocalTensor<uint32_t> bitMask = bitMaskLocal.ReinterpretCast<uint32_t>();
            LocalTensor<int32_t> dstCastLocal = dstLocal.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> srcCastLocal = srcLocal.template ReinterpretCast<int32_t>();
            GatherMask(dstCastLocal, srcCastLocal, bitMask, true, mask, params, rsvdCnt);
        } else if constexpr (IS_4_BYTES_TYPE) {
            uint32_t mask = count;
            LocalTensor<uint32_t> bitMask = bitMaskLocal.ReinterpretCast<uint32_t>();
#ifndef __CCE_KT_TEST__
            GatherMask(dstLocal, srcLocal, bitMask, true, mask, params, rsvdCnt);
#endif
        } else if constexpr (IS_2_BYTES_TYPE) {
            uint32_t mask = count;
            LocalTensor<uint16_t> bitMask = bitMaskLocal.ReinterpretCast<uint16_t>();
#ifndef __CCE_KT_TEST__
            GatherMask(dstLocal, srcLocal, bitMask, true, mask, params, rsvdCnt);//rsvdCnt 最终有效元素个数
#endif
        } else {
            uint32_t mask = count;
            LocalTensor<half> xCastLocal = xCastBuf.Get<half>();
            LocalTensor<half> yCastLocal = yCastBuf.Get<half>();
            Duplicate(xCastLocal, static_cast<half>(0), static_cast<int32_t>(this->tileLength));
            Cast(xCastLocal, srcLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint16_t> bitMask = bitMaskLocal.ReinterpretCast<uint16_t>();
#ifndef __CCE_KT_TEST__
            GatherMask(yCastLocal, xCastLocal, bitMask, true, mask, params, rsvdCnt);
#endif
            Cast(dstLocal, yCastLocal, RoundMode::CAST_NONE, rsvdCnt);
        }
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();
        LocalTensor<uint8_t> bitMaskLocal = bitMaskBuf.Get<uint8_t>();// GYW  DeQue 和 GET区别？
        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }
        GenerateMask(maskLocal, bitMaskLocal, length);
        inQueueMask.FreeTensor(maskLocal);
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        GatherResult(yLocal, xLocal, bitMaskLocal, length);
        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeOffset(int32_t progress)
    {
        LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        LocalTensor<int32_t> sumInt32 = sumLocal.template ReinterpretCast<int32_t>();
        LocalTensor<float> maskCastLocal = maskCastBuf.Get<float>();

        uint32_t length = this->tileLength;
        if (progress == this->tileNum - 1) {
            length = this->lasttileLength;
        }
        PipeBarrier<PIPE_V>();
        Cast(maskHalfLocal, maskLocal, RoundMode::CAST_NONE, length);
        inQueueMask.FreeTensor(maskLocal);
        PipeBarrier<PIPE_V>();
        Cast(maskCastLocal, maskHalfLocal, RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
        ReduceSumFunc(sumLocal, maskCastLocal, length);
        PipeBarrier<PIPE_V>();

        Cast(sumInt32, sumLocal, RoundMode::CAST_ROUND, 1);
        PipeBarrier<PIPE_V>();
        Add(offsetLocal, offsetLocal, sumInt32, 1);
    }
    __aicore__ inline void DataCopyPadDoubleWord(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
                                                int64_t count)
    {
        GlobalTensor<int32_t> srcCastGlobal;
        srcCastGlobal.SetGlobalBuffer((__gm__ int32_t*)srcGlobal.GetPhyAddr(), count * INT64_LENGTH_IN_INT32);//将GM 中 64 转成 32 * 2

        LocalTensor<int32_t> dstCastLocal = dstLocal.template ReinterpretCast<int32_t>();//将 ue转 int32

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * INT64_LENGTH_IN_INT32 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(dstCastLocal, srcCastGlobal, copyParams, padParams); 
    }

    __aicore__ inline void DataCopyPadDoubleWord(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
                                                int64_t count)
    {
        GlobalTensor<int32_t> dstCastGlobal;
        dstCastGlobal.SetGlobalBuffer((__gm__ int32_t*)dstGlobal.GetPhyAddr(), count * INT64_LENGTH_IN_INT32);

        LocalTensor<int32_t> srcCastLocal = srcLocal.template ReinterpretCast<int32_t>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * INT64_LENGTH_IN_INT32 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(dstCastGlobal, srcCastLocal, copyParams);
    }

    __aicore__ inline void CopyOut2GM()
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rsvdCnt * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGlobal[outOffset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe* pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueMask;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moveQue;
    TBuf<TPosition::VECCALC> maskCastBuf;
    TBuf<TPosition::VECCALC> bitMaskBuf;
    TBuf<TPosition::VECCALC> xCastBuf;
    TBuf<TPosition::VECCALC> yCastBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> sumBuf;

    GlobalTensor<T> xGlobal;
    GlobalTensor<T> yGlobal;
    GlobalTensor<uint64_t> shapeoutGlobal;
    GlobalTensor<uint8_t> maskGlobal;
    GlobalTensor<T> workGlobal;
    GlobalTensor<int32_t> offsetGlobal;
    LocalTensor<int32_t> offsetLocal;
    LocalTensor<half> maskHalfLocal;
    // 输入
    uint64_t blockDim;
    uint64_t formerNum;
    uint64_t formerLength;
    uint64_t formertileNum;
    uint64_t formertileLength;
    uint64_t formerlasttileLength;
    uint64_t tailNum;
    uint64_t tailLength;
    uint64_t tailtileNum;
    uint64_t tailtileLength;
    uint64_t taillasttileLength;
    uint64_t tokenNum;
    uint64_t tileLengthAlign;
    // 本block/核的
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t lasttileLength;
    uint64_t needCoreNum;
    uint64_t rsvdCnt = 0;
    uint64_t outOffset = 0;
    uint32_t blockIdx = 0;
};
} // namespace AscendC

#endif // MASKED_SELECT_H_