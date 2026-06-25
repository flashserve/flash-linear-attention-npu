/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_bwd_dv_local_vector.h
 * \brief A5 regbase vector implementation for chunk_bwd_dv_local.
 */

#ifndef CHUNK_BWD_DV_LOCAL_ARCH35_VECTOR_H
#define CHUNK_BWD_DV_LOCAL_ARCH35_VECTOR_H

#include <type_traits>
#include "chunk_bwd_dv_local_struct.h"
#include "chunk_bwd_dv_local_common.h"
#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

namespace GDN {

template <typename QKVT, typename GT, typename Strategy>
class ChunkBwdDvLocalVector {
private:
    Strategy strategy;

public:
    __aicore__ inline ChunkBwdDvLocalVector(const Strategy &s) : strategy(s)
    {
    }

    __aicore__ inline void Init(GM_ADDR d_o, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v,
                                GM_ADDR workspace, const ChunkBwdDvLocalTilingData *__restrict tilingData,
                                AscendC::TPipe *pipe = nullptr);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessChunk(const IndexResult &indexResult);
    template <uint16_t N_SIZE>
    __simd_vf__ inline void ProcessGatedKqComputerVF(__ubuf__ QKVT *kqOut, __ubuf__ QKVT *kqIn,
                                                     __ubuf__ GT *gAllIn, __ubuf__ GT *gRowIn,
                                                     uint16_t mSize, uint16_t validColSize,
                                                     uint16_t startRow, float scale);

    AscendC::TPipe *pipe_;
    AscendC::GlobalTensor<QKVT> dOGm;
    AscendC::GlobalTensor<GT> gGm;
    AscendC::GlobalTensor<int64_t> cuSeqlensGm;
    AscendC::GlobalTensor<int64_t> chunkIndicesGm;
    AscendC::GlobalTensor<QKVT> dVGm;
    AscendC::GlobalTensor<QKVT> workspaceGm;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gTQueIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gHalfTQueIn;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> kqTQueIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> kqTQueOut;

    int64_t H_qk;
    int64_t H_do;
    int64_t hRatio;
    int64_t headBufNum;
    int64_t T;
    int64_t K;
    int64_t V;
    int64_t coreLoops;
    int64_t blockNum;
    int64_t subBlockNum;
    int64_t subBlockIdx;
    int64_t coreIdx;
    float scale;
    int64_t vecTaskIdx;

    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<QKVT> qkvPadParams{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<GT> gPadParams{false, 0, 0, 0};
};

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Init(
    GM_ADDR d_o, GM_ADDR g, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR d_v, GM_ADDR workspace,
    const ChunkBwdDvLocalTilingData *__restrict tilingData, AscendC::TPipe *pipe)
{
    dOGm.SetGlobalBuffer((__gm__ QKVT *)d_o);
    gGm.SetGlobalBuffer((__gm__ GT *)g);
    cuSeqlensGm.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
    chunkIndicesGm.SetGlobalBuffer((__gm__ int64_t *)chunk_indices);
    dVGm.SetGlobalBuffer((__gm__ QKVT *)d_v);
    workspaceGm.SetGlobalBuffer((__gm__ QKVT *)workspace);

    H_qk = tilingData->hQk;
    H_do = tilingData->hDo;
    hRatio = tilingData->hRatio;
    headBufNum = tilingData->headBufNum;
    T = tilingData->t;
    K = tilingData->k;
    V = tilingData->v;
    scale = tilingData->scale;
    coreLoops = tilingData->b * strategy.chunkNumForT;
    blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    subBlockNum = AscendC::GetSubBlockNum();
    coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx() / subBlockNum);
    subBlockIdx = static_cast<int64_t>(AscendC::GetSubBlockIdx());
    vecTaskIdx = 0;

    pipe_ = pipe;
    pipe_->InitBuffer(gTQueIn, BUFFER_NUM, strategy.chunkSize * sizeof(GT));
    pipe_->InitBuffer(gHalfTQueIn, BUFFER_NUM, strategy.chunkSize / NUM_2 * sizeof(GT));
    pipe_->InitBuffer(kqTQueIn, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / NUM_2);
    pipe_->InitBuffer(kqTQueOut, BUFFER_NUM, strategy.chunkSize * strategy.chunkSize * sizeof(QKVT) / NUM_2);
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::Process()
{
    IndexResult indexResult;
    for (int64_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        strategy.calculate(loopIdx, indexResult);
        ProcessChunk(indexResult);
    }
}

template <typename QKVT, typename GT, typename Strategy>
__aicore__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::ProcessChunk(const IndexResult &indexResult)
{
    int64_t taskSplitLine = indexResult.chunkLen / NUM_2;
    int64_t taskStartLine = 0;
    int64_t taskEndLine = 0;
    int64_t taskLineNum = 0;
    int64_t taskOffset = 0;
    int64_t coreBaseOffset = coreIdx * headBufNum * strategy.chunkSize * strategy.chunkSize;
    int64_t p1SlotNum = headBufNum / (hRatio + 1);
    if (p1SlotNum <= 0) {
        p1SlotNum = NUM_2;
    }

    for (int64_t doHead = 0; doHead < H_do; doHead++) {
        int64_t qkHead = doHead / hRatio;
        int64_t doGroup = doHead % hRatio;
        int64_t p1Slot = qkHead % p1SlotNum;
        int64_t gatedSlot = p1SlotNum + (qkHead % p1SlotNum) * hRatio + doGroup;
        int64_t baseReadOffset = coreBaseOffset + p1Slot * strategy.chunkSize * strategy.chunkSize;
        int64_t baseWriteOffset = coreBaseOffset + gatedSlot * strategy.chunkSize * strategy.chunkSize;

        ++vecTaskIdx;
        if (vecTaskIdx % subBlockNum != subBlockIdx) {
            taskStartLine = 0;
            taskEndLine = taskSplitLine - 1;
            taskOffset = baseWriteOffset;
        } else {
            taskStartLine = taskSplitLine;
            taskEndLine = indexResult.chunkLen - 1;
            taskOffset = baseWriteOffset + taskSplitLine * strategy.chunkSize;
        }

        taskLineNum = taskEndLine - taskStartLine + 1;
        if (taskLineNum == 0) {
            if (doHead % hRatio == 0) {
                AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_1);
            continue;
        }

        int64_t baseGOffset = indexResult.curBatchId * H_do * T + doHead * T + indexResult.curTokenId;
        int64_t taskReadOffset = baseReadOffset + taskStartLine * strategy.chunkSize;

        {
            AscendC::LocalTensor<GT> gLocalTensor = gTQueIn.AllocTensor<GT>();
            copyParams.blockLen = indexResult.chunkLen * sizeof(GT);
            AscendC::DataCopyPad(gLocalTensor, gGm[baseGOffset], copyParams, gPadParams);
            gTQueIn.EnQue(gLocalTensor);
        }
        {
            AscendC::LocalTensor<GT> gHalfLocalTensor = gHalfTQueIn.AllocTensor<GT>();
            copyParams.blockLen = taskLineNum * sizeof(GT);
            AscendC::DataCopyPad(gHalfLocalTensor, gGm[baseGOffset + taskStartLine], copyParams, gPadParams);
            gHalfTQueIn.EnQue(gHalfLocalTensor);
        }

        if (doHead % hRatio == 0) {
            AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG_3);
        }

        {
            AscendC::LocalTensor<QKVT> kqLocalTensor = kqTQueIn.AllocTensor<QKVT>();
            copyParams.blockLen = taskLineNum * strategy.chunkSize * sizeof(QKVT);
            AscendC::DataCopyPad(kqLocalTensor, workspaceGm[taskReadOffset], copyParams, qkvPadParams);
            kqTQueIn.EnQue(kqLocalTensor);
        }

        {
            AscendC::LocalTensor<GT> gLocalTensor = gTQueIn.DeQue<GT>();
            AscendC::LocalTensor<GT> gHalfLocalTensor = gHalfTQueIn.DeQue<GT>();
            AscendC::LocalTensor<QKVT> kqLocalTensor = kqTQueIn.DeQue<QKVT>();
            AscendC::LocalTensor<QKVT> kqOutLocalTensor = kqTQueOut.AllocTensor<QKVT>();

            auto gAllInAddr = reinterpret_cast<uint64_t>(gLocalTensor.GetPhyAddr());
            auto gRowInAddr = reinterpret_cast<uint64_t>(gHalfLocalTensor.GetPhyAddr());
            auto kqInAddr = reinterpret_cast<uint64_t>(kqLocalTensor.GetPhyAddr());
            auto kqOutAddr = reinterpret_cast<uint64_t>(kqOutLocalTensor.GetPhyAddr());

            if (strategy.chunkSize == 64) {
                ProcessGatedKqComputerVF<64>((__ubuf__ QKVT *)kqOutAddr, (__ubuf__ QKVT *)kqInAddr,
                                             (__ubuf__ GT *)gAllInAddr, (__ubuf__ GT *)gRowInAddr,
                                             static_cast<uint16_t>(taskLineNum),
                                             static_cast<uint16_t>(indexResult.chunkLen),
                                             static_cast<uint16_t>(taskStartLine), scale);
            } else {
                ProcessGatedKqComputerVF<128>((__ubuf__ QKVT *)kqOutAddr, (__ubuf__ QKVT *)kqInAddr,
                                              (__ubuf__ GT *)gAllInAddr, (__ubuf__ GT *)gRowInAddr,
                                              static_cast<uint16_t>(taskLineNum),
                                              static_cast<uint16_t>(indexResult.chunkLen),
                                              static_cast<uint16_t>(taskStartLine), scale);
            }

            gTQueIn.FreeTensor(gLocalTensor);
            gHalfTQueIn.FreeTensor(gHalfLocalTensor);
            kqTQueIn.FreeTensor(kqLocalTensor);
            kqTQueOut.EnQue(kqOutLocalTensor);
        }

        {
            AscendC::LocalTensor<QKVT> kqOutLocalTensor = kqTQueOut.DeQue<QKVT>();
            AscendC::DataCopy(workspaceGm[taskOffset], kqOutLocalTensor, taskLineNum * strategy.chunkSize);
            kqTQueOut.FreeTensor(kqOutLocalTensor);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG_1);
    }
}

template <typename QKVT, typename GT, typename Strategy>
template <uint16_t N_SIZE>
__simd_vf__ inline void ChunkBwdDvLocalVector<QKVT, GT, Strategy>::ProcessGatedKqComputerVF(
    __ubuf__ QKVT *kqOut, __ubuf__ QKVT *kqIn, __ubuf__ GT *gAllIn, __ubuf__ GT *gRowIn,
    uint16_t mSize, uint16_t validColSize, uint16_t startRow, float scale)
{
    constexpr uint32_t ONE_ELE_NUM = N_SIZE;
    uint32_t qkvMaskLen = N_SIZE;
    uint32_t gMaskLen = N_SIZE;
    uint32_t storeMaskLen = N_SIZE;
    RegTensor<GT> gAllReg;
    RegTensor<GT> gRowReg;
    RegTensor<QKVT> kqInReg;
    RegTensor<QKVT> kqOutReg;
    RegTensor<float> gAllFP32ZeroReg, gAllFP32OneReg;
    RegTensor<float> gRowFP32Reg;
    RegTensor<float> gFactorFP32ZeroReg, gFactorFP32OneReg;
    RegTensor<float> kqFP32ZeroReg, kqFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg;
    RegTensor<float> zeroReg, rowIdxReg, scaleReg;
    RegTensor<float> validColReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskQkv = UpdateMask<QKVT>(qkvMaskLen);
    MaskReg maskG = UpdateMask<GT>(gMaskLen);
    MaskReg maskStore = UpdateMask<QKVT>(storeMaskLen);
    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskInvalidZeroSelect, maskInvalidOneSelect;

    Duplicate(zeroReg, static_cast<float>(0.0), maskFull32);
    Duplicate(scaleReg, scale, maskFull32);
    Duplicate(validColReg, static_cast<float>(validColSize), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    if constexpr (!std::is_same<GT, float>()) {
        LoadIn<GT, false>(gAllReg, gAllIn);
        Cast<float, GT, ctHalf2Fp32Zero>(gAllFP32ZeroReg, gAllReg, maskG);
        Cast<float, GT, ctHalf2Fp32One>(gAllFP32OneReg, gAllReg, maskG);
    } else {
        LoadAlign<GT, LoadDist::DIST_DINTLV_B32>(gAllFP32ZeroReg, gAllFP32OneReg, gAllIn);
    }

    Duplicate(rowIdxReg, static_cast<float>(startRow));
    LoadIn<QKVT, false>(kqInReg, kqIn);
    LoadIn<GT, true>(gRowReg, gRowIn);
    for (uint16_t rowIdx = 0; rowIdx < mSize - 1; ++rowIdx) {
        CastHalf2Float<QKVT>(kqFP32ZeroReg, kqFP32OneReg, kqInReg, maskQkv);
        HalfOrFloat2Float<GT>(gRowFP32Reg, gRowReg, maskFull16, maskFull32);

        SubFloatTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, gAllFP32ZeroReg, gAllFP32OneReg,
                       gRowFP32Reg, gRowFP32Reg, maskFull32);
        MinsFloatTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, gFactorFP32ZeroReg, gFactorFP32OneReg,
                        static_cast<float>(0.0), maskFull32);
        ExpFloatTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, gFactorFP32ZeroReg, gFactorFP32OneReg, maskFull32);
        Mul(gFactorFP32ZeroReg, gFactorFP32ZeroReg, scaleReg, maskFull32);
        Mul(gFactorFP32OneReg, gFactorFP32OneReg, scaleReg, maskFull32);

        CompareTwoReg<float, CMPMODE::LT>(maskZeroSelect, maskOneSelect,
                                          colIdxFP32ZeroReg, colIdxFP32OneReg,
                                          rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, zeroReg, zeroReg,
                     gFactorFP32ZeroReg, gFactorFP32OneReg, maskZeroSelect, maskOneSelect);
        CompareTwoReg<float, CMPMODE::GE>(maskInvalidZeroSelect, maskInvalidOneSelect,
                                          colIdxFP32ZeroReg, colIdxFP32OneReg,
                                          validColReg, validColReg, maskFull32);
        SelectTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, zeroReg, zeroReg,
                     gFactorFP32ZeroReg, gFactorFP32OneReg,
                     maskInvalidZeroSelect, maskInvalidOneSelect);
        MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, kqFP32ZeroReg, kqFP32OneReg,
                       gFactorFP32ZeroReg, gFactorFP32OneReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg,
                     resultFP32ZeroReg, resultFP32OneReg,
                     maskInvalidZeroSelect, maskInvalidOneSelect);
        CastFloat2Half<QKVT>(kqOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        LoadIn<QKVT, false>(kqInReg, kqIn + ONE_ELE_NUM * (rowIdx + 1));
        LoadIn<GT, true>(gRowReg, gRowIn + rowIdx + 1);
        StoreAlign((__ubuf__ QKVT *&)kqOut + ONE_ELE_NUM * rowIdx, kqOutReg, maskStore);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);
    }

    uint16_t rowIdx = mSize - 1;
    CastHalf2Float<QKVT>(kqFP32ZeroReg, kqFP32OneReg, kqInReg, maskQkv);
    HalfOrFloat2Float<GT>(gRowFP32Reg, gRowReg, maskFull16, maskFull32);

    SubFloatTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, gAllFP32ZeroReg, gAllFP32OneReg,
                   gRowFP32Reg, gRowFP32Reg, maskFull32);
    MinsFloatTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, gFactorFP32ZeroReg, gFactorFP32OneReg,
                    static_cast<float>(0.0), maskFull32);
    ExpFloatTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, gFactorFP32ZeroReg, gFactorFP32OneReg, maskFull32);
    Mul(gFactorFP32ZeroReg, gFactorFP32ZeroReg, scaleReg, maskFull32);
    Mul(gFactorFP32OneReg, gFactorFP32OneReg, scaleReg, maskFull32);

    CompareTwoReg<float, CMPMODE::LT>(maskZeroSelect, maskOneSelect,
                                      colIdxFP32ZeroReg, colIdxFP32OneReg,
                                      rowIdxReg, rowIdxReg, maskFull32);
    SelectTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, zeroReg, zeroReg,
                 gFactorFP32ZeroReg, gFactorFP32OneReg, maskZeroSelect, maskOneSelect);
    CompareTwoReg<float, CMPMODE::GE>(maskInvalidZeroSelect, maskInvalidOneSelect,
                                      colIdxFP32ZeroReg, colIdxFP32OneReg,
                                      validColReg, validColReg, maskFull32);
    SelectTwoReg(gFactorFP32ZeroReg, gFactorFP32OneReg, zeroReg, zeroReg,
                 gFactorFP32ZeroReg, gFactorFP32OneReg,
                 maskInvalidZeroSelect, maskInvalidOneSelect);
    MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, kqFP32ZeroReg, kqFP32OneReg,
                   gFactorFP32ZeroReg, gFactorFP32OneReg, maskFull32);
    SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg,
                 resultFP32ZeroReg, resultFP32OneReg,
                 maskInvalidZeroSelect, maskInvalidOneSelect);
    CastFloat2Half<QKVT>(kqOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
    StoreAlign((__ubuf__ QKVT *&)kqOut + ONE_ELE_NUM * rowIdx, kqOutReg, maskStore);
}

} // namespace GDN

#endif // CHUNK_BWD_DV_LOCAL_ARCH35_VECTOR_H
