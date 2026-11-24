/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_BWD_DQKWG_VECTOR_H
#define CHUNK_BWD_DQKWG_VECTOR_H

#include "chunk_bwd_dqkwg_common.h"
#include "kernel_operator.h"
#include "chunk_bwd_dqkwg_regbase.h"

using namespace AscendC;

template <typename DataType, typename GType>
class ChunkBwdDqkwgVectorProcess {
public:
    __aicore__ inline ChunkBwdDqkwgVectorProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_,
                                                 GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen, GM_ADDR chunk_indices,
                                                 GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace);

    __aicore__ inline void Init(const ChunkBwdDqkwgTilingData &tiling, TPipe *pipe_);
    __aicore__ inline void Process();

private:

    __aicore__ inline void ProcessAVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT);
    __aicore__ inline void ProcessBVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT);
    __aicore__ inline void ProcessCVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT);
    __aicore__ inline void ProcessCVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT);
    __aicore__ inline void ProcessDVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT);
    __aicore__ inline void ProcessDVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT);

    __aicore__ inline void ComputeMul1HalfFp32(const LocalTensor<float> &outFp32, const LocalTensor<float> &tensorMaskA,
                                               const LocalTensor<float> &tensorGTmpFp32, const LocalTensor<float> &tensorGFp32Left,
                                               const LocalTensor<float> &tensorGFp32Right, uint32_t BT_sub_start, uint32_t real_BT);

    __aicore__ inline void CopyGateWithPad(const LocalTensor<GType> &dst, const GlobalTensor<GType> &src, uint64_t offset,
                                           uint32_t validLen, uint32_t totalLen);
    __aicore__ inline void ComputeDqState(const LocalTensor<float> &tensorDqInFp32, const LocalTensor<float> &tensorGFp32, const LocalTensor<float> &tensorGTmpFp32, uint32_t real_BT);

private:

    GM_ADDR ptrQ;
    GM_ADDR ptrK;
    GM_ADDR ptrV;
    GM_ADDR ptrG;
    GM_ADDR ptrH;
    GM_ADDR ptrDo;
    GM_ADDR ptrDh;
    GM_ADDR ptrDv;
    GM_ADDR ptrCuSeqLen;
    GM_ADDR ptrChunkIndices;
    GM_ADDR ptrDq;
    GM_ADDR ptrDk;
    GM_ADDR ptrDw;
    GM_ADDR ptrDg;
    GM_ADDR ptrWorkspace;

    uint64_t B;
    uint64_t HV;
    uint64_t HK;
    uint64_t T;
    uint64_t K;
    uint64_t V;
    uint64_t BT;
    uint64_t half_BT;
    uint64_t numChunks;
    float scale;
    bool isVarLen;
    bool gvaMode;
    uint32_t mul0RowNum = 0;
    uint64_t n_ratio = 1;
    uint32_t coreNum = 0;
    uint32_t coreIdx = 0;
    uint32_t subBlockIdx = 0;
    uint32_t subBlockNum = 1;

    uint64_t wsMm3Offset;
    uint64_t wsMm4Offset;
    uint64_t wsMm6Offset;
    uint64_t wsMm5Offset;
    uint64_t wsMm7Offset;
    uint64_t wsDsTempOffset;
    uint64_t wsDgLastOffset;
    int BUFFER_NUM = 1;

    TPipe *pipe = nullptr;

    GlobalTensor<DataType> gmQ, gmK, gmV, gmDo, gmH, gmDh, gmDv;
    GlobalTensor<DataType> gmDq, gmDk, gmDw;
    GlobalTensor<GType> gmG, gmDg;
    GlobalTensor<float> gmDgLast;
    GlobalTensor<DataType> gmMm3, gmMm4, gmMm6, gmMm5, gmMm7, gmDsTemp;
    GlobalTensor<uint64_t> cuSeqlensTensor, chunkIndicesTensor;

    TQue<TPosition::VECIN, 2> inQue1;
    TQue<TPosition::VECIN, 2> inQue2;
    TQue<TPosition::VECIN, 2> inQue3;
    TQue<TPosition::VECIN, 2> inQue4;
    TQue<TPosition::VECOUT, 2> outQue1;
    TQue<TPosition::VECOUT, 2> outQue2;

    TBuf<TPosition::VECCALC> calcBuf1;
    TBuf<TPosition::VECCALC> calcBuf2;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> dgBuf;

    LocalTensor<float> tensorGTmpFp32;
    LocalTensor<float> tensorDgTmpFp32;
    LocalTensor<float> tensorDgFinal;
    LocalTensor<float> tensorMaskA;

    static constexpr uint32_t UB_BLOCK_SIZE = 32;
};

template <typename DataType, typename GType>
__aicore__ inline ChunkBwdDqkwgVectorProcess<DataType, GType>::ChunkBwdDqkwgVectorProcess(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR h, GM_ADDR do_, GM_ADDR dh, GM_ADDR dv, GM_ADDR cu_seqlen,
    GM_ADDR chunk_indices, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR dg, GM_ADDR workspace)
    : ptrQ(q), ptrK(k), ptrV(v), ptrG(g), ptrH(h), ptrDo(do_), ptrDh(dh), ptrDv(dv), ptrCuSeqLen(cu_seqlen),
      ptrChunkIndices(chunk_indices), ptrDq(dq), ptrDk(dk), ptrDw(dw), ptrDg(dg), ptrWorkspace(workspace)
{
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Init(const ChunkBwdDqkwgTilingData &tiling,
                                                                         TPipe *pipe_)
{
    pipe = pipe_;

    scale = tiling.scale;
    B = tiling.B;
    HV = tiling.HV;
    HK = tiling.HK;
    n_ratio = (HK > 0) ? (HV / HK) : 1;
    T = tiling.T;
    K = tiling.K;
    V = tiling.V;
    BT = tiling.BT;
    half_BT = BT / 2;
    numChunks = tiling.numChunks;
    coreNum = tiling.aicCoreNum;
    coreIdx = GetBlockIdx() / GetSubBlockNum();
    subBlockIdx = GetSubBlockIdx();
    subBlockNum = GetSubBlockNum();
    wsMm3Offset = tiling.wsMm3Offset;
    wsMm4Offset = tiling.wsMm4Offset;
    wsMm6Offset = tiling.wsMm6Offset;
    wsMm5Offset = tiling.wsMm5Offset;
    wsMm7Offset = tiling.wsMm7Offset;
    wsDsTempOffset = tiling.wsDsTempOffset;
    wsDgLastOffset = tiling.wsDgLastOffset;
    isVarLen = (tiling.isVarLen == 1);
    gvaMode = (HV != HK);
    mul0RowNum = tiling.mul0RowNum;

    if (BT == 64) {
        BUFFER_NUM = 2;
    } else {
        BUFFER_NUM = 1;
    }

    gmQ.SetGlobalBuffer((__gm__ DataType *)ptrQ);
    gmK.SetGlobalBuffer((__gm__ DataType *)ptrK);
    gmV.SetGlobalBuffer((__gm__ DataType *)ptrV);
    gmG.SetGlobalBuffer((__gm__ GType *)ptrG);
    gmH.SetGlobalBuffer((__gm__ DataType *)ptrH);
    gmDo.SetGlobalBuffer((__gm__ DataType *)ptrDo);
    gmDh.SetGlobalBuffer((__gm__ DataType *)ptrDh);
    gmDv.SetGlobalBuffer((__gm__ DataType *)ptrDv);

    gmDq.SetGlobalBuffer((__gm__ DataType *)ptrDq);
    gmDk.SetGlobalBuffer((__gm__ DataType *)ptrDk);
    gmDw.SetGlobalBuffer((__gm__ DataType *)ptrDw);
    gmDg.SetGlobalBuffer((__gm__ GType *)ptrDg);

    gmMm3.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm3Offset));
    gmMm4.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm4Offset));
    gmMm6.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm6Offset));
    gmMm5.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm5Offset));
    gmMm7.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsMm7Offset));
    gmDsTemp.SetGlobalBuffer((__gm__ DataType *)((__gm__ uint8_t *)ptrWorkspace + wsDsTempOffset));
    gmDgLast.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)ptrWorkspace + wsDgLastOffset));

    if (isVarLen) {
        cuSeqlensTensor.SetGlobalBuffer((__gm__ uint64_t *)ptrCuSeqLen);
        chunkIndicesTensor.SetGlobalBuffer((__gm__ uint64_t *)ptrChunkIndices);
    }

    pipe->InitBuffer(inQue1, BUFFER_NUM, BT * K * sizeof(float));
    pipe->InitBuffer(inQue2, BUFFER_NUM, BT * K * sizeof(float));
    pipe->InitBuffer(inQue3, 2, BT * sizeof(float));
    pipe->InitBuffer(inQue4, 2, BT * sizeof(float));
    pipe->InitBuffer(outQue1, BUFFER_NUM, BT * K * sizeof(DataType));
    pipe->InitBuffer(outQue2, BUFFER_NUM, BT * sizeof(float));

    pipe->InitBuffer(calcBuf1, BT * 8 * sizeof(float));
    pipe->InitBuffer(calcBuf2, BT * 8 * sizeof(float));
    pipe->InitBuffer(maskBuf, 64 * 64 * sizeof(float));
    pipe->InitBuffer(dgBuf, BT * 8 * sizeof(float));

    tensorGTmpFp32 = calcBuf1.Get<float>();
    tensorDgTmpFp32 = calcBuf2.Get<float>();
    tensorDgFinal = dgBuf.Get<float>();
    tensorMaskA = maskBuf.Get<float>();
    Duplicate(tensorMaskA, 0.0f, 64 * 64);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < 64; i++) {
        Duplicate(tensorMaskA[i * 64], scale, i + 1);
    }
    PipeBarrier<PIPE_V>();
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::Process()
{

    SetVecClear();
    SetVecClear();
    SetVecClear();
    uint32_t coreLoops = B * numChunks;
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {

        uint32_t bos = 0;
        uint32_t eos = 0;
        GetChunkOffset(cuSeqlensTensor, chunkIndicesTensor, B, HV, T, BT, loopIdx, bos, eos, isVarLen);
        uint32_t real_BT = eos - bos;
        ProcessAVector(coreIdx, loopIdx, bos, real_BT);
        ProcessBVector(coreIdx, loopIdx, bos, real_BT);
        if (gvaMode) {
            ProcessCVectorForGva(coreIdx, loopIdx, bos, real_BT);
            ProcessDVectorForGva(coreIdx, loopIdx, bos, real_BT);
        } else {
            ProcessCVector(coreIdx, loopIdx, bos, real_BT);
            ProcessDVector(coreIdx, loopIdx, bos, real_BT);
        }
    }
}

template <typename DataType, typename GType>
__aicore__ inline void
ChunkBwdDqkwgVectorProcess<DataType, GType>::CopyGateWithPad(const LocalTensor<GType> &dst, const GlobalTensor<GType> &src,
                                                             uint64_t offset, uint32_t validLen, uint32_t totalLen)
{

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(validLen * sizeof(GType)), 0, 0, 0};

    DataCopyPadExtParams<GType> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src[offset], copyParams, padParams);
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ComputeMul1HalfFp32(
    const LocalTensor<float> &outFp32, const LocalTensor<float> &tensorMaskA,
    const LocalTensor<float> &tensorGTmpFp32, const LocalTensor<float> &tensorGFp32Left,
    const LocalTensor<float> &tensorGFp32Right, uint32_t BT_sub_start, uint32_t real_BT)
{
    if (BT == 64) {
        Mul1Half<64, 0>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                        (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
    } else {
        if (BT_sub_start == 0) {
            Mul1Half<128, 0>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                             (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        } else {
            Mul1Half<128, 64>((__ubuf__ float *)outFp32.GetPhyAddr(), (__ubuf__ float *)tensorGFp32Left.GetPhyAddr(),
                              (__ubuf__ float *)tensorMaskA.GetPhyAddr(), static_cast<uint16_t>(real_BT), scale);
        }
    }
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ComputeDqState(
    const LocalTensor<float> &tensorDqInFp32, const LocalTensor<float> &tensorGFp32, const LocalTensor<float> &tensorGTmpFp32, uint32_t real_BT)
{
    DqState((__ubuf__ float *)tensorDqInFp32.GetPhyAddr(), (__ubuf__ float *)tensorDqInFp32.GetPhyAddr(),
            (__ubuf__ float *)tensorGFp32.GetPhyAddr(), (__ubuf__ float *)tensorGTmpFp32.GetPhyAddr(), real_BT, scale);
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessAVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    const uint32_t hDhSize = mul0RowNum * V;

    uint32_t bIdx = loopIdx / numChunks;
    uint32_t chunkIdx = loopIdx % numChunks;
    uint32_t actual_dwSize = real_BT * K;
    WaitCubeReady();

    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(0x8);
    AscendC::CrossCoreWaitFlag(0x8);

    for (uint32_t h = 0; h < HV; h++) {
        if ((h % subBlockNum) != subBlockIdx) {
            continue;
        }
        uint64_t hOffset = ((bIdx * numChunks + chunkIdx) * HV + h) * K * V;
        uint64_t dwOffset = (h * T + bos) * K;
        uint64_t dgLastOffset = DqkwgScalarElemOffset(coreIdx, h, HV);

        auto tensorSumFp32 = inQue1.AllocTensor<float>();
        for (uint32_t row = 0; row < K; row += mul0RowNum) {
            {
                auto tensorHFp32 = inQue2.AllocTensor<float>();

                auto tensorHIn = tensorHFp32[hDhSize].template ReinterpretCast<DataType>();
                DataCopy(tensorHIn, gmH[hOffset + row * V], hDhSize);
                DataCopy(tensorHIn[hDhSize], gmDh[hOffset + row * V], hDhSize);
                inQue2.EnQue(tensorHFp32);
            }
            {
                auto tensorHFp32 = inQue2.DeQue<float>();
                auto tensorDhFp32 = tensorHFp32[hDhSize];
                auto tensorHIn = tensorDhFp32.template ReinterpretCast<DataType>();
                if (row == 0) {
                    DgLastMulAccum<DataType, false>(
                        (__ubuf__ float *)tensorSumFp32.GetPhyAddr(), (__ubuf__ DataType *)tensorHIn.GetPhyAddr(),
                        (__ubuf__ DataType *)tensorHIn[hDhSize].GetPhyAddr(), hDhSize);
                } else {
                    DgLastMulAccum<DataType, true>(
                        (__ubuf__ float *)tensorSumFp32.GetPhyAddr(), (__ubuf__ DataType *)tensorHIn.GetPhyAddr(),
                        (__ubuf__ DataType *)tensorHIn[hDhSize].GetPhyAddr(), hDhSize);
                }
                inQue2.FreeTensor(tensorHFp32);
            }
        }
        {
            auto tensorDgLastOut = outQue2.AllocTensor<float>();
            ReduceSumCustom(tensorDgLastOut, tensorSumFp32, hDhSize);
            PipeBarrier<PIPE_V>();
            outQue2.EnQue(tensorDgLastOut);
            inQue1.FreeTensor(tensorSumFp32);
        }
        {
            auto tensorDgLastOut = outQue2.DeQue<float>();
            DataCopyPad(gmDgLast[dgLastOffset], tensorDgLastOut, {1, sizeof(float), 0, 0});
            outQue2.FreeTensor(tensorDgLastOut);
        }

        {
            auto tensorDwIn = inQue1.AllocTensor<DataType>();
            DataCopy(tensorDwIn[actual_dwSize], gmDw[dwOffset], actual_dwSize);
            inQue1.EnQue(tensorDwIn);
        }
        {
            auto tensorDwIn = inQue1.DeQue<DataType>();
            auto tensorDwOut = outQue1.AllocTensor<DataType>();
            DwNegate<DataType>((__ubuf__ DataType *)tensorDwOut.GetPhyAddr(),
                                (__ubuf__ DataType *)tensorDwIn[actual_dwSize].GetPhyAddr(),
                                actual_dwSize);
            outQue1.EnQue(tensorDwOut);
            inQue1.FreeTensor(tensorDwIn);
        }
        {
            auto tensorDwOut = outQue1.DeQue<DataType>();
            DataCopy(gmDw[dwOffset], tensorDwOut, actual_dwSize);
            outQue1.FreeTensor(tensorDwOut);
        }
    }
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessBVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t actual_dsSize = real_BT * BT;
    WaitCubeReady();

    for (uint32_t h = 0; h < HV; h++) {
        if ((h % subBlockNum) != subBlockIdx) {
            continue;
        }
        uint64_t gOffset = (h * T + bos);
        uint64_t dsOffset = DqkwgBtbElemOffset(coreIdx, h, HV, BT);
        uint64_t mm3Offset = dsOffset;
        uint64_t dgOffset = gOffset;

        {
            auto tensorDsIn = inQue1.AllocTensor<DataType>();
            auto tensorGIn = inQue3.AllocTensor<GType>();
            DataCopy(tensorDsIn[BT * BT], gmDsTemp[dsOffset], actual_dsSize);
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
            }
            inQue1.EnQue(tensorDsIn);
            inQue3.EnQue(tensorGIn);
        }
        {
            auto tensorDsInFp16 = inQue1.DeQue<DataType>();
            auto tensorDsInFp32 = tensorDsInFp16.template ReinterpretCast<float>();
            auto tensorGIn = inQue3.DeQue<GType>();
            auto tensorGFp32Left = tensorGIn.template ReinterpretCast<float>();
            auto tensorDsTempOut = outQue1.AllocTensor<DataType>();
            auto tensorDgOut = outQue2.AllocTensor<float>();

            Cast(tensorDsInFp32, tensorDsInFp16[BT * BT], RoundMode::CAST_NONE, actual_dsSize);
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorGFp32Left, tensorGIn[BT], RoundMode::CAST_NONE, BT);
            }
            PipeBarrier<PIPE_V>();

            auto tensorMul1InFp32 = inQue2.AllocTensor<float>();
            uint32_t vec0 = (real_BT >= 64) ? 64 : real_BT;
            uint32_t vec1 = (real_BT > 64) ? (real_BT - 64) : 0;
            ComputeMul1HalfFp32(tensorMul1InFp32, tensorMaskA, tensorGTmpFp32, tensorGFp32Left, tensorDgTmpFp32, 0, vec0);
            if (vec1 > 0) {
                ComputeMul1HalfFp32(tensorMul1InFp32[vec0 * BT], tensorMaskA, tensorGTmpFp32, tensorGFp32Left, tensorDgTmpFp32, vec0, vec1);
            }
            PipeBarrier<PIPE_V>();

            Mul(tensorDsInFp32, tensorDsInFp32, tensorMul1InFp32, actual_dsSize);
            PipeBarrier<PIPE_V>();
            inQue3.FreeTensor(tensorGIn);
            inQue2.FreeTensor(tensorMul1InFp32);

            auto tensorMm3InFp16Tmp = inQue2.AllocTensor<DataType>();
            DataCopy(tensorMm3InFp16Tmp[BT * BT], gmMm3[mm3Offset], actual_dsSize);
            inQue2.EnQue(tensorMm3InFp16Tmp);
            auto tensorMm3InFp16 = inQue2.DeQue<DataType>();
            auto tensorMm3InFp32 = tensorMm3InFp16.template ReinterpretCast<float>();
            Cast(tensorMm3InFp32, tensorMm3InFp16[BT * BT], RoundMode::CAST_NONE, actual_dsSize);
            PipeBarrier<PIPE_V>();

            Mul(tensorMm3InFp32, tensorDsInFp32, tensorMm3InFp32, actual_dsSize);
            Cast(tensorDsTempOut, tensorDsInFp32, RoundMode::CAST_RINT, actual_dsSize);
            PipeBarrier<PIPE_V>();

            if (real_BT > FP32_PER_REPEAT) {
                DataCopy(tensorDsInFp32, tensorMm3InFp32, {static_cast<uint16_t>(real_BT), 8, 8, 0});
                PipeBarrier<PIPE_V>();
                Add(tensorDsInFp32, tensorDsInFp32, tensorMm3InFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT, real_BT, {1, 1, 1, 8, 8, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut, tensorDsInFp32, FP32_PER_REPEAT, real_BT, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                Add(tensorMm3InFp32, tensorMm3InFp32[BT], tensorMm3InFp32, FP32_PER_REPEAT, real_BT - 1, {1, 1, 1, 0, static_cast<uint8_t>(BT / 8), 0});
                PipeBarrier<PIPE_V>();
                Add(tensorMm3InFp32[FP32_PER_REPEAT], tensorMm3InFp32[BT + FP32_PER_REPEAT], tensorMm3InFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT, real_BT - 1, {1, 1, 1, 0, static_cast<uint8_t>(BT / 8), 0});
                PipeBarrier<PIPE_V>();
            } else {
                WholeReduceSum(tensorDgOut, tensorMm3InFp32, real_BT, real_BT, 1, 1, BT / 8);
                PipeBarrier<PIPE_V>();
                if (real_BT > 1) {
                    Add(tensorMm3InFp32, tensorMm3InFp32[BT], tensorMm3InFp32, real_BT, real_BT - 1, {1, 1, 1, 0, static_cast<uint8_t>(BT / 8), 0});
                    PipeBarrier<PIPE_V>();
                }
            }

            Sub(tensorDgOut, tensorDgOut, tensorMm3InFp32, real_BT);
            PipeBarrier<PIPE_V>();
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, real_BT);
            }

            outQue1.EnQue(tensorDsTempOut);
            outQue2.EnQue(tensorDgOut);
            inQue1.FreeTensor(tensorDsInFp16);
            inQue2.FreeTensor(tensorMm3InFp16);
        }
        {
            auto tensorDsTempOut = outQue1.DeQue<DataType>();
            auto tensorDgOut = outQue2.DeQue<GType>();
            DataCopy(gmDsTemp[dsOffset], tensorDsTempOut, actual_dsSize);
            DataCopyPad(gmDg[dgOffset], tensorDgOut, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
            outQue1.FreeTensor(tensorDsTempOut);
            outQue2.FreeTensor(tensorDgOut);
        }
    }
    SetVecCredit();
    SetVecClear();
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessCVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t actual_dqSize = real_BT * K;
    WaitCubeReady();

    for (uint32_t h = 0; h < HV; h++) {
        if ((h % subBlockNum) != subBlockIdx) {
            continue;
        }
        uint64_t qkOffset = (h * T + bos) * K;
        uint64_t dqOutOffset = (h * T + bos) * K;
        uint64_t gOffset = (h * T + bos);
        uint64_t mm4Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        {
            auto tensorDqIn = inQue1.AllocTensor<DataType>();
            auto tensorQIn = inQue2.AllocTensor<DataType>();
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            DataCopy(tensorDqIn[actual_dqSize], gmMm4[mm4Offset], actual_dqSize);
            DataCopy(tensorQIn[actual_dqSize], gmQ[qkOffset], actual_dqSize);
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue1.EnQue(tensorDqIn);
            inQue2.EnQue(tensorQIn);
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        {
            auto tensorDqInFp16 = inQue1.DeQue<DataType>();
            auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
            auto tensorQInFp16 = inQue2.DeQue<DataType>();
            auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();
            auto tensorGIn = inQue3.DeQue<GType>();
            auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
            auto tensorDgIn = inQue4.DeQue<GType>();
            auto tensorDgAdd = tensorDgIn.template ReinterpretCast<float>();
            auto tensorDgOut = outQue2.AllocTensor<float>();

            Cast(tensorDqInFp32, tensorDqInFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
            Cast(tensorQInFp32, tensorQInFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, BT);
                Cast(tensorDgAdd, tensorDgIn[BT], RoundMode::CAST_NONE, BT);
            }
            PipeBarrier<PIPE_V>();

            ComputeDqState(tensorDqInFp32, tensorGFp32, tensorGTmpFp32, real_BT);

            Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, actual_dqSize);
            PipeBarrier<PIPE_V>();
            Add(tensorQInFp32, tensorQInFp32, tensorQInFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, real_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorDgOut, tensorQInFp32, FP32_PER_REPEAT, real_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();
            Add(tensorDgOut, tensorDgAdd, tensorDgOut, real_BT);
            PipeBarrier<PIPE_V>();
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, BT);
            }

            outQue2.EnQue(tensorDgOut);
            inQue2.FreeTensor(tensorQInFp16);
            inQue3.FreeTensor(tensorGIn);
            inQue4.FreeTensor(tensorDgIn);

            {
                auto tensorDgOutDeq = outQue2.DeQue<GType>();
                DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
                outQue2.FreeTensor(tensorDgOutDeq);
            }

            {
                auto tensorMm6In = inQue2.AllocTensor<DataType>();
                DataCopy(tensorMm6In[actual_dqSize], gmMm6[mm4Offset], actual_dqSize);
                inQue2.EnQue(tensorMm6In);
            }
            {
                auto tensorMm6InFp16 = inQue2.DeQue<DataType>();
                auto tensorMm6Fp32 = tensorMm6InFp16.template ReinterpretCast<float>();
                auto tensorDqOut = outQue1.AllocTensor<DataType>();
                Cast(tensorMm6Fp32, tensorMm6InFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
                PipeBarrier<PIPE_V>();
                Add(tensorDqInFp32, tensorDqInFp32, tensorMm6Fp32, actual_dqSize);
                PipeBarrier<PIPE_V>();
                Cast(tensorDqOut, tensorDqInFp32, RoundMode::CAST_RINT, actual_dqSize);
                outQue1.EnQue(tensorDqOut);
                inQue1.FreeTensor(tensorDqInFp16);
                inQue2.FreeTensor(tensorMm6InFp16);
            }
            {
                auto tensorDqOut = outQue1.DeQue<DataType>();
                DataCopy(gmDq[dqOutOffset], tensorDqOut, actual_dqSize);
                outQue1.FreeTensor(tensorDqOut);
            }
        }
    }
    SetVecClear();
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessCVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t half_dqSize = half_BT * K;
    uint32_t actual_dqSize = real_BT * K;
    uint32_t first_BT = real_BT > half_BT ? half_BT : real_BT;
    uint32_t second_BT = real_BT - first_BT;
    uint32_t first_dqSize = first_BT * K;
    uint32_t second_dqSize = second_BT * K;
    uint32_t bIdx = loopIdx / numChunks;
    uint64_t bos_hk = bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(HV - HK) * T;
    WaitCubeReady();

    LocalTensor<float> tensorDqSum;
    for (uint32_t h = 0; h < HV; h++) {
        uint32_t hk_idx = h / n_ratio;
        if ((hk_idx % subBlockNum) != subBlockIdx) {
            continue;
        }
        uint64_t qkOffset = (hk_idx * T + bos_hk) * K;
        uint64_t dqOutOffset = (hk_idx * T + bos_hk) * K;
        uint64_t gOffset = (h * T + bos);
        uint64_t mm4Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        {
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        auto tensorGIn = inQue3.DeQue<GType>();
        auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
        auto tensorDgIn = inQue4.DeQue<GType>();
        auto tensorDgAdd = tensorDgIn.template ReinterpretCast<float>();
        auto tensorDgOut = outQue2.AllocTensor<float>();
        if constexpr (!std::is_same<GType, float>::value) {
            Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, BT);
            Cast(tensorDgAdd, tensorDgIn[BT], RoundMode::CAST_NONE, BT);
        }

        {
            auto tensorDqIn = inQue2.AllocTensor<DataType>();
            auto tensorQIn = tensorDqIn[half_dqSize * 2];
            DataCopy(tensorDqIn[half_dqSize], gmMm4[mm4Offset], first_dqSize);
            DataCopy(tensorQIn[half_dqSize], gmQ[qkOffset], first_dqSize);
            inQue2.EnQue(tensorDqIn);
        }
        {
            auto tensorDqInFp16 = inQue2.DeQue<DataType>();
            auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
            auto tensorQInFp16 = tensorDqInFp16[half_dqSize * 2];
            auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();

            Cast(tensorDqInFp32, tensorDqInFp16[half_dqSize], RoundMode::CAST_NONE, first_dqSize);
            Cast(tensorQInFp32, tensorQInFp16[half_dqSize], RoundMode::CAST_NONE, first_dqSize);
            PipeBarrier<PIPE_V>();

            ComputeDqState(tensorDqInFp32, tensorGFp32, tensorGTmpFp32, first_BT);

            Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, first_dqSize);
            PipeBarrier<PIPE_V>();
            Add(tensorQInFp32, tensorQInFp32, tensorQInFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, first_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorDgOut, tensorQInFp32, FP32_PER_REPEAT, first_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();

            if (h % n_ratio == 0) {
                tensorDqSum = inQue1.AllocTensor<float>();
                DataCopy(tensorDqSum, tensorDqInFp32, first_dqSize);
                PipeBarrier<PIPE_V>();
            } else {
                Add(tensorDqSum, tensorDqSum, tensorDqInFp32, first_dqSize);
                PipeBarrier<PIPE_V>();
            }
            inQue2.FreeTensor(tensorDqInFp16);
        }
        if (second_BT > 0) {

            {
                auto tensorDqIn = inQue2.AllocTensor<DataType>();
                auto tensorQIn = tensorDqIn[half_dqSize * 2];
                DataCopy(tensorDqIn[half_dqSize], gmMm4[mm4Offset + half_dqSize], second_dqSize);
                DataCopy(tensorQIn[half_dqSize], gmQ[qkOffset + half_dqSize], second_dqSize);
                inQue2.EnQue(tensorDqIn);
            }
            {
                auto tensorDqInFp16 = inQue2.DeQue<DataType>();
                auto tensorDqInFp32 = tensorDqInFp16.template ReinterpretCast<float>();
                auto tensorQInFp16 = tensorDqInFp16[half_dqSize * 2];
                auto tensorQInFp32 = tensorQInFp16.template ReinterpretCast<float>();

                Cast(tensorDqInFp32, tensorDqInFp16[half_dqSize], RoundMode::CAST_NONE, second_dqSize);
                Cast(tensorQInFp32, tensorQInFp16[half_dqSize], RoundMode::CAST_NONE, second_dqSize);
                PipeBarrier<PIPE_V>();

                ComputeDqState(tensorDqInFp32, tensorGFp32[half_BT], tensorGTmpFp32, second_BT);

                Mul(tensorQInFp32, tensorDqInFp32, tensorQInFp32, second_dqSize);
                PipeBarrier<PIPE_V>();
                Add(tensorQInFp32, tensorQInFp32, tensorQInFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, second_BT, {1, 1, 1, 16, 16, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgOut[half_BT], tensorQInFp32, FP32_PER_REPEAT, second_BT, 1, 1, 16);
                PipeBarrier<PIPE_V>();

                if (h % n_ratio == 0) {
                    DataCopy(tensorDqSum[half_dqSize], tensorDqInFp32, second_dqSize);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(tensorDqSum[half_dqSize], tensorDqSum[half_dqSize], tensorDqInFp32, second_dqSize);
                    PipeBarrier<PIPE_V>();
                }
                inQue2.FreeTensor(tensorDqInFp16);
            }
        }

        Add(tensorDgOut, tensorDgAdd, tensorDgOut, real_BT);
        PipeBarrier<PIPE_V>();
        if constexpr (!std::is_same<GType, float>::value) {
            Cast(tensorDgOut.template ReinterpretCast<GType>(), tensorDgOut, RoundMode::CAST_RINT, BT);
        }
        outQue2.EnQue(tensorDgOut);
        inQue3.FreeTensor(tensorGIn);
        inQue4.FreeTensor(tensorDgIn);

        {
            auto tensorDgOutDeq = outQue2.DeQue<GType>();
            DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
            outQue2.FreeTensor(tensorDgOutDeq);
        }

        {
            auto tensorMm6In = inQue2.AllocTensor<DataType>();
            DataCopy(tensorMm6In[actual_dqSize], gmMm6[mm4Offset], actual_dqSize);
            inQue2.EnQue(tensorMm6In);
        }
        {
            auto tensorMm6InFp16 = inQue2.DeQue<DataType>();
            auto tensorMm6Fp32 = tensorMm6InFp16.template ReinterpretCast<float>();
            Cast(tensorMm6Fp32, tensorMm6InFp16[actual_dqSize], RoundMode::CAST_NONE, actual_dqSize);
            PipeBarrier<PIPE_V>();
            Add(tensorDqSum, tensorDqSum, tensorMm6Fp32, actual_dqSize);
            PipeBarrier<PIPE_V>();
            inQue2.FreeTensor(tensorMm6InFp16);
        }
        if (h % n_ratio == n_ratio - 1) {
            auto tensorDqOut = outQue1.AllocTensor<DataType>();
            Cast(tensorDqOut, tensorDqSum, RoundMode::CAST_RINT, actual_dqSize);
            outQue1.EnQue(tensorDqOut);
            inQue1.FreeTensor(tensorDqSum);

            tensorDqOut = outQue1.DeQue<DataType>();
            DataCopy(gmDq[dqOutOffset], tensorDqOut, actual_dqSize);
            outQue1.FreeTensor(tensorDqOut);
        }
    }
    SetVecClear();
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessDVector(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t real_BT_aligned = (real_BT + 15) / 16 * 16;
    uint32_t actual_dkSize = real_BT * K;
    WaitCubeReady();

    TEventID eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    for (uint32_t h = 0; h < HV; h++) {
        if ((h % subBlockNum) != subBlockIdx) {
            continue;
        }
        uint64_t kOffset = (h * T + bos) * K;
        uint64_t dkOutOffset = (h * T + bos) * K;
        uint64_t gOffset = (h * T + bos);
        uint64_t mm5Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        {
            uint64_t dgLastOffset = DqkwgScalarElemOffset(coreIdx, h, HV);
            DataCopyExtParams copyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 7, 0};
            TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
            DataCopyPad(tensorDgTmpFp32[8], gmDgLast[dgLastOffset], copyParams, padParams);
            TEventID eDg = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eDg);
            WaitFlag<HardEvent::MTE2_V>(eDg);
            Brcb(tensorGTmpFp32[16], tensorDgTmpFp32[8], 1, {1, 8});
            PipeBarrier<PIPE_V>();
        }

        {
            auto tensorDkIn = inQue1.AllocTensor<DataType>();
            auto tensorKIn = inQue2.AllocTensor<DataType>();
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            DataCopy(tensorDkIn[actual_dkSize], gmMm5[mm5Offset], actual_dkSize);
            DataCopy(tensorKIn[actual_dkSize], gmK[kOffset], actual_dkSize);
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue1.EnQue(tensorDkIn);
            inQue2.EnQue(tensorKIn);
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        {
            auto tensorDkIn = inQue1.DeQue<DataType>();
            auto tensorDkFp32 = tensorDkIn.template ReinterpretCast<float>();
            auto tensorKIn = inQue2.DeQue<DataType>();
            auto tensorKFp32 = tensorKIn.template ReinterpretCast<float>();
            auto tensorGIn = inQue3.DeQue<GType>();
            auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
            auto tensorDgIn = inQue4.DeQue<GType>();
            auto tensorDgTmp = tensorDgIn.template ReinterpretCast<float>();
            auto tensorDgOut = outQue2.AllocTensor<GType>();

            Cast(tensorDkFp32, tensorDkIn[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
            Cast(tensorKFp32, tensorKIn[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
            if constexpr (!std::is_same<GType, float>::value) {
                Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
                Cast(tensorDgTmp, tensorDgIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
            }
            PipeBarrier<PIPE_V>();

            uint32_t last_line_no = (real_BT - 1) / 8 * 8;
            uint32_t last_line_idx = real_BT - 1 - last_line_no;
            Brcb(tensorDgFinal, tensorGFp32[last_line_no], 1, {1, 8});
            PipeBarrier<PIPE_V>();
            Muls(tensorGFp32, tensorGFp32, -1.0f, real_BT_aligned);
            DataCopy(tensorGTmpFp32, tensorDgFinal[last_line_idx * 8], 8);
            PipeBarrier<PIPE_V>();
            AscendC::Add(tensorGFp32, tensorGFp32, tensorDgFinal[last_line_idx * 8], CAL_NUM_FLOAT,
                            BT / CAL_NUM_FLOAT, {1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
            Exp(tensorGFp32, tensorGFp32, real_BT_aligned);
            PipeBarrier<PIPE_V>();

            Brcb(tensorDgFinal, tensorGFp32, CEIL_DIV(real_BT, 8), {1, 8});
            PipeBarrier<PIPE_V>();
            AscendC::Mul(tensorDkFp32, tensorDkFp32, tensorDgFinal, CAL_NUM_FLOAT, real_BT, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(tensorDkFp32[CAL_NUM_FLOAT], tensorDkFp32[CAL_NUM_FLOAT], tensorDgFinal, CAL_NUM_FLOAT,
                            real_BT, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();

            Mul(tensorKFp32, tensorKFp32, tensorDkFp32, actual_dkSize);
            PipeBarrier<PIPE_V>();

            Add(tensorKFp32, tensorKFp32, tensorKFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, real_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorGFp32, tensorKFp32, FP32_PER_REPEAT, real_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();

            Sub(tensorDgTmp, tensorDgTmp, tensorGFp32, BT);
            PipeBarrier<PIPE_V>();

            if (real_BT > FP32_PER_REPEAT) {
                Add(tensorGFp32, tensorGFp32, tensorGFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT);
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorDgFinal, tensorGFp32, FP32_PER_REPEAT, 1, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                WholeReduceSum(tensorDgFinal, tensorGFp32, real_BT, 1, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            }
            Brcb(tensorDgTmpFp32, tensorDgFinal, 1, {1, 8});
            PipeBarrier<PIPE_V>();
            Exp(tensorGTmpFp32, tensorGTmpFp32, 8);
            PipeBarrier<PIPE_V>();
            Mul(tensorDgTmpFp32[16], tensorGTmpFp32[16], tensorGTmpFp32, 8);
            PipeBarrier<PIPE_V>();
            Add(tensorDgTmpFp32[16], tensorDgTmpFp32[16], tensorDgTmpFp32, 8);
            PipeBarrier<PIPE_V>();
            Brcb(tensorDgFinal, tensorDgTmpFp32[16], 1, {1, 8});
            PipeBarrier<PIPE_V>();
            uint64_t offset = (real_BT - 1) / 8 * 8;
            uint64_t mask[1] = {0};
            mask[0] = 1ULL << (real_BT - 1 - offset);
            Add(tensorDgTmp[offset], tensorDgTmp[offset], tensorDgFinal, mask, 1, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<GType, float>::value) {
                DataCopy(tensorDgOut, tensorDgTmp, BT);
            } else {
                Cast(tensorDgOut, tensorDgTmp, RoundMode::CAST_RINT, BT);
            }
            PipeBarrier<PIPE_V>();

            outQue2.EnQue(tensorDgOut);
            inQue2.FreeTensor(tensorKIn);
            inQue3.FreeTensor(tensorGIn);
            inQue4.FreeTensor(tensorDgIn);
            {
                auto tensorDgOutDeq = outQue2.DeQue<GType>();
                DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
                outQue2.FreeTensor(tensorDgOutDeq);
            }

            {
                auto tensorMm7In = inQue2.AllocTensor<DataType>();
                DataCopy(tensorMm7In[actual_dkSize], gmMm7[mm5Offset], actual_dkSize);
                inQue2.EnQue(tensorMm7In);
            }
            {
                auto tensorMm7In = inQue2.DeQue<DataType>();
                auto tensorMm7Fp32 = tensorMm7In.template ReinterpretCast<float>();
                auto tensorDkOut = outQue1.AllocTensor<DataType>();
                Cast(tensorMm7Fp32, tensorMm7In[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
                PipeBarrier<PIPE_V>();
                Add(tensorDkFp32, tensorDkFp32, tensorMm7Fp32, actual_dkSize);
                PipeBarrier<PIPE_V>();
                Cast(tensorDkOut, tensorDkFp32, RoundMode::CAST_RINT, actual_dkSize);
                outQue1.EnQue(tensorDkOut);
                inQue1.FreeTensor(tensorDkIn);
                inQue2.FreeTensor(tensorMm7In);
            }
            {
                auto tensorDkOut = outQue1.DeQue<DataType>();
                DataCopy(gmDk[dkOutOffset], tensorDkOut, actual_dkSize);
                outQue1.FreeTensor(tensorDkOut);
            }
        }
    }
    SetVecClear();
}

template <typename DataType, typename GType>
__aicore__ inline void ChunkBwdDqkwgVectorProcess<DataType, GType>::ProcessDVectorForGva(uint32_t coreIdx, uint32_t loopIdx, uint32_t bos, uint32_t real_BT)
{
    uint32_t half_dkSize = half_BT * K;
    uint32_t real_BT_aligned = (real_BT + 15) / 16 * 16;
    uint32_t actual_dkSize = real_BT * K;
    uint32_t first_BT = real_BT > half_BT ? half_BT : real_BT;
    uint32_t second_BT = real_BT - first_BT;
    uint32_t first_dkSize = first_BT * K;
    uint32_t second_dkSize = second_BT * K;
    uint32_t bIdx = loopIdx / numChunks;
    uint64_t bos_hk = bos - static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(HV - HK) * T;
    WaitCubeReady();

    TEventID eventIdMte3ToMte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    LocalTensor<float> tensorDkSum;
    for (uint32_t h = 0; h < HV; h++) {
        uint32_t hk_idx = h / n_ratio;
        if ((hk_idx % subBlockNum) != subBlockIdx) {
            continue;
        }
        uint64_t kOffset = (hk_idx * T + bos_hk) * K;
        uint64_t dkOutOffset = (hk_idx * T + bos_hk) * K;
        uint64_t gOffset = (h * T + bos);
        uint64_t mm5Offset = DqkwgBtxKElemOffset(coreIdx, h, HV, BT, K);

        {
            uint64_t dgLastOffset = DqkwgScalarElemOffset(coreIdx, h, HV);
            TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventId);
            WaitFlag<HardEvent::V_MTE2>(eventId);
            DataCopyExtParams copyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 7, 0};
            DataCopyPad(tensorDgTmpFp32[8], gmDgLast[dgLastOffset], copyParams, padParams);
            TEventID eDg = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eDg);
            WaitFlag<HardEvent::MTE2_V>(eDg);
            Brcb(tensorGTmpFp32[16], tensorDgTmpFp32[8], 1, {1, 8});
            PipeBarrier<PIPE_V>();
        }

        {
            auto tensorGIn = inQue3.AllocTensor<GType>();
            auto tensorDgIn = inQue4.AllocTensor<GType>();
            if constexpr (std::is_same<GType, float>::value) {
                CopyGateWithPad(tensorGIn, gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn, gmDg, gOffset, real_BT, BT);
            } else {
                CopyGateWithPad(tensorGIn[BT], gmG, gOffset, real_BT, BT);
                CopyGateWithPad(tensorDgIn[BT], gmDg, gOffset, real_BT, BT);
            }
            inQue3.EnQue(tensorGIn);
            inQue4.EnQue(tensorDgIn);
        }
        auto tensorGIn = inQue3.DeQue<GType>();
        auto tensorGFp32 = tensorGIn.template ReinterpretCast<float>();
        auto tensorDgIn = inQue4.DeQue<GType>();
        auto tensorDgTmp = tensorDgIn.template ReinterpretCast<float>();
        auto tensorDgOut = outQue2.AllocTensor<GType>();
        if constexpr (!std::is_same<GType, float>::value) {
            Cast(tensorGFp32, tensorGIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
            Cast(tensorDgTmp, tensorDgIn[BT], RoundMode::CAST_NONE, real_BT_aligned);
        }
        PipeBarrier<PIPE_V>();

        uint32_t last_line_no = (real_BT - 1) / 8 * 8;
        uint32_t last_line_idx = real_BT - 1 - last_line_no;
        Brcb(tensorDgFinal, tensorGFp32[last_line_no], 1, {1, 8});
        PipeBarrier<PIPE_V>();
        Muls(tensorGFp32, tensorGFp32, -1.0f, real_BT_aligned);
        DataCopy(tensorGTmpFp32, tensorDgFinal[last_line_idx * 8], 8);
        PipeBarrier<PIPE_V>();
        AscendC::Add(tensorGFp32, tensorGFp32, tensorDgFinal[last_line_idx * 8], CAL_NUM_FLOAT,
                        BT / CAL_NUM_FLOAT, {1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Exp(tensorGFp32, tensorGFp32, real_BT_aligned);
        PipeBarrier<PIPE_V>();

        {
            auto tensorDkIn = inQue2.AllocTensor<DataType>();
            auto tensorKIn = tensorDkIn[half_dkSize * 2];
            DataCopy(tensorDkIn[half_dkSize], gmMm5[mm5Offset], first_dkSize);
            DataCopy(tensorKIn[half_dkSize], gmK[kOffset], first_dkSize);
            inQue2.EnQue(tensorDkIn);
        }
        {
            auto tensorDkInFp16 = inQue2.DeQue<DataType>();
            auto tensorDkFp32 = tensorDkInFp16.template ReinterpretCast<float>();
            auto tensorKInFp16 = tensorDkInFp16[half_dkSize * 2];
            auto tensorKFp32 = tensorKInFp16.template ReinterpretCast<float>();

            Cast(tensorDkFp32, tensorDkInFp16[half_dkSize], RoundMode::CAST_NONE, first_dkSize);
            Cast(tensorKFp32, tensorKInFp16[half_dkSize], RoundMode::CAST_NONE, first_dkSize);
            PipeBarrier<PIPE_V>();

            Brcb(tensorDgFinal, tensorGFp32, CEIL_DIV(first_BT, 8), {1, 8});
            PipeBarrier<PIPE_V>();
            AscendC::Mul(tensorDkFp32, tensorDkFp32, tensorDgFinal, CAL_NUM_FLOAT, first_BT, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(tensorDkFp32[CAL_NUM_FLOAT], tensorDkFp32[CAL_NUM_FLOAT], tensorDgFinal, CAL_NUM_FLOAT,
                            first_BT, {1, 1, 0, 16, 16, 1});
            PipeBarrier<PIPE_V>();

            if (h % n_ratio == 0) {
                tensorDkSum = inQue1.AllocTensor<float>();
                DataCopy(tensorDkSum, tensorDkFp32, first_dkSize);
                PipeBarrier<PIPE_V>();
            } else {
                Add(tensorDkSum, tensorDkSum, tensorDkFp32, first_dkSize);
                PipeBarrier<PIPE_V>();
            }

            Mul(tensorKFp32, tensorKFp32, tensorDkFp32, first_dkSize);
            PipeBarrier<PIPE_V>();

            Add(tensorKFp32, tensorKFp32, tensorKFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, first_BT, {1, 1, 1, 16, 16, 16});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorGFp32, tensorKFp32, FP32_PER_REPEAT, first_BT, 1, 1, 16);
            PipeBarrier<PIPE_V>();

            inQue2.FreeTensor(tensorDkInFp16);
        }
        if (second_BT > 0) {

            {
                auto tensorDkIn = inQue2.AllocTensor<DataType>();
                auto tensorKIn = tensorDkIn[half_dkSize * 2];
                DataCopy(tensorDkIn[half_dkSize], gmMm5[mm5Offset + half_dkSize], second_dkSize);
                DataCopy(tensorKIn[half_dkSize], gmK[kOffset + half_dkSize], second_dkSize);
                inQue2.EnQue(tensorDkIn);
            }
            {
                auto tensorDkInFp16 = inQue2.DeQue<DataType>();
                auto tensorDkFp32 = tensorDkInFp16.template ReinterpretCast<float>();
                auto tensorKInFp16 = tensorDkInFp16[half_dkSize * 2];
                auto tensorKFp32 = tensorKInFp16.template ReinterpretCast<float>();

                Cast(tensorDkFp32, tensorDkInFp16[half_dkSize], RoundMode::CAST_NONE, second_dkSize);
                Cast(tensorKFp32, tensorKInFp16[half_dkSize], RoundMode::CAST_NONE, second_dkSize);
                PipeBarrier<PIPE_V>();

                Brcb(tensorDgFinal, tensorGFp32[half_BT], CEIL_DIV(second_BT, 8), {1, 8});
                PipeBarrier<PIPE_V>();
                AscendC::Mul(tensorDkFp32, tensorDkFp32, tensorDgFinal, CAL_NUM_FLOAT, second_BT, {1, 1, 0, 16, 16, 1});
                AscendC::Mul(tensorDkFp32[CAL_NUM_FLOAT], tensorDkFp32[CAL_NUM_FLOAT], tensorDgFinal, CAL_NUM_FLOAT,
                                second_BT, {1, 1, 0, 16, 16, 1});
                PipeBarrier<PIPE_V>();

                if (h % n_ratio == 0) {
                    DataCopy(tensorDkSum[half_dkSize], tensorDkFp32, second_dkSize);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(tensorDkSum[half_dkSize], tensorDkSum[half_dkSize], tensorDkFp32, second_dkSize);
                    PipeBarrier<PIPE_V>();
                }

                Mul(tensorKFp32, tensorKFp32, tensorDkFp32, second_dkSize);
                PipeBarrier<PIPE_V>();

                Add(tensorKFp32, tensorKFp32, tensorKFp32[FP32_PER_REPEAT], FP32_PER_REPEAT, second_BT, {1, 1, 1, 16, 16, 16});
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorGFp32[half_BT], tensorKFp32, FP32_PER_REPEAT, second_BT, 1, 1, 16);
                PipeBarrier<PIPE_V>();

                inQue2.FreeTensor(tensorDkInFp16);
            }
        }

        Sub(tensorDgTmp, tensorDgTmp, tensorGFp32, BT);
        PipeBarrier<PIPE_V>();

        if (real_BT > FP32_PER_REPEAT) {
            Add(tensorGFp32, tensorGFp32, tensorGFp32[FP32_PER_REPEAT], real_BT - FP32_PER_REPEAT);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tensorDgFinal, tensorGFp32, FP32_PER_REPEAT, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        } else {
            WholeReduceSum(tensorDgFinal, tensorGFp32, real_BT, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
        Brcb(tensorDgTmpFp32, tensorDgFinal, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        Exp(tensorGTmpFp32, tensorGTmpFp32, 8);
        PipeBarrier<PIPE_V>();
        Mul(tensorDgTmpFp32[16], tensorGTmpFp32[16], tensorGTmpFp32, 8);
        PipeBarrier<PIPE_V>();
        Add(tensorDgTmpFp32[16], tensorDgTmpFp32[16], tensorDgTmpFp32, 8);
        PipeBarrier<PIPE_V>();
        Brcb(tensorDgFinal, tensorDgTmpFp32[16], 1, {1, 8});
        PipeBarrier<PIPE_V>();
        uint64_t offset = (real_BT - 1) / 8 * 8;
        uint64_t mask[1] = {0};
        mask[0] = 1ULL << (real_BT - 1 - offset);
        Add(tensorDgTmp[offset], tensorDgTmp[offset], tensorDgFinal, mask, 1, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<GType, float>::value) {
            DataCopy(tensorDgOut, tensorDgTmp, BT);
        } else {
            Cast(tensorDgOut, tensorDgTmp, RoundMode::CAST_RINT, BT);
        }
        PipeBarrier<PIPE_V>();

        outQue2.EnQue(tensorDgOut);
        inQue3.FreeTensor(tensorGIn);
        inQue4.FreeTensor(tensorDgIn);
        {
            auto tensorDgOutDeq = outQue2.DeQue<GType>();
            DataCopyPad(gmDg[gOffset], tensorDgOutDeq, {1, static_cast<uint16_t>(real_BT * sizeof(GType)), 0, 0});
            outQue2.FreeTensor(tensorDgOutDeq);
        }

        {
            auto tensorMm7In = inQue2.AllocTensor<DataType>();
            DataCopy(tensorMm7In[actual_dkSize], gmMm7[mm5Offset], actual_dkSize);
            inQue2.EnQue(tensorMm7In);
        }
        {
            auto tensorMm7In = inQue2.DeQue<DataType>();
            auto tensorMm7Fp32 = tensorMm7In.template ReinterpretCast<float>();
            Cast(tensorMm7Fp32, tensorMm7In[actual_dkSize], RoundMode::CAST_NONE, actual_dkSize);
            PipeBarrier<PIPE_V>();
            Add(tensorDkSum, tensorDkSum, tensorMm7Fp32, actual_dkSize);
            PipeBarrier<PIPE_V>();
            inQue2.FreeTensor(tensorMm7In);
        }
        if (h % n_ratio == n_ratio - 1) {
            auto tensorDkOut = outQue1.AllocTensor<DataType>();
            Cast(tensorDkOut, tensorDkSum, RoundMode::CAST_RINT, actual_dkSize);
            outQue1.EnQue(tensorDkOut);
            inQue1.FreeTensor(tensorDkSum);

            tensorDkOut = outQue1.DeQue<DataType>();
            DataCopy(gmDk[dkOutOffset], tensorDkOut, actual_dkSize);
            outQue1.FreeTensor(tensorDkOut);
        }
    }
    SetVecClear();
}

#endif
