/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef RECOMPUTE_W_U_FWD_VECTOR_H
#define RECOMPUTE_W_U_FWD_VECTOR_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 9201
#endif

#include "recompute_w_u_fwd_struct.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_utils/vector/regbase.hpp"
using namespace AscendC;
using namespace AscendC::MicroAPI;

using GDN::GdnMegaArch35RecomputeWUTilingData;

template <typename kType, typename betaType, bool kFlattenHeadTasks = false,
          bool kCoefficientGenerationTaskOrder = false>
class RecomputeWUFwdVectorProcess {
public:

    __aicore__ inline RecomputeWUFwdVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
                                                        GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessVbAndKbgExpInterleaved();
    __aicore__ inline void Init(const GdnMegaArch35RecomputeWUTilingData &tiling, AscendC::TPipe *pipe_);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t Hv = 1;
    uint64_t Hk = 1;
    uint64_t hvPerHk = 1;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t vbVecRow = 0;
    uint64_t kbgExpVecRow = 0;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR w;
    GM_ADDR u;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;

private:
    static constexpr uint32_t GM_RING_DEPTH = 8;
    static_assert(GM_RING_DEPTH <= Arch::MAX_REVERSE_DEPTH,
                  "Ring window must keep every cross-core flag below the hardware credit limit.");
    Arch::CrossCoreFlag flagAivVbReady{SYNC_AIV_AIC_FLAG_3};
    Arch::CrossCoreFlag flagAivKbgExpReady{SYNC_AIV_AIC_FLAG_4};
    Arch::CrossCoreFlag flagAicSlotFree{SYNC_AIC_AIV_FLAG_5};
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<kType> workSpaceTensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> vInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECOUT, 1> vbOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> kBetagExpOutQue;

    TBuf<AscendC::TPosition::VECCALC> kFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> vFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32Buf;
    TBuf<AscendC::TPosition::VECCALC> betaFp32BrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> gFp32Buf;

    constexpr static CastTrait ctFp322KTypeRintZero = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, AscendC::RoundMode::CAST_RINT};
    constexpr static CastTrait ctFp322KTypeRintOne = {
        RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    __aicore__ inline void NotifyVbReady();
    __aicore__ inline void NotifyKbgExpReady();
    __aicore__ inline void WaitSlotFree();
    __simd_callee__ inline void CastFloat2KTypeRint(
        RegTensor<kType> &dstReg, RegTensor<float> &srcZeroReg,
        RegTensor<float> &srcOneReg, MaskReg &mask);
    __simd_vf__ inline void ProcessVbComputerVF(
        __ubuf__ kType *vbOut, __ubuf__ kType *vIn, __ubuf__ betaType *betaIn,
        uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessKbgExpComputerVF(
        __ubuf__ kType *kbgOut, __ubuf__ kType *kIn, __ubuf__ betaType *betaIn,
        __ubuf__ betaType *gIn, uint16_t mSize, uint16_t nSize);

};

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ inline RecomputeWUFwdVectorProcess<kType, betaType,
                                               kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::RecomputeWUFwdVectorProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType, kFlattenHeadTasks,
                                                   kCoefficientGenerationTaskOrder>::Init(
    const GdnMegaArch35RecomputeWUTilingData &tiling, AscendC::TPipe *pipe_)
{
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    B = tiling.B;
    T = tiling.T;
    Hv = tiling.Hv;
    Hk = tiling.Hk;
    hvPerHk = tiling.hvPerHk;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    vbVecRow = tiling.vbVecRow;
    kbgExpVecRow = tiling.kbgExpVecRow;
    return;
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType, kFlattenHeadTasks,
                                                   kCoefficientGenerationTaskOrder>::Process()
{
    ProcessVbAndKbgExpInterleaved();
    return;
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::NotifyVbReady()
{
    Arch::CrossCoreSetFlag<0x4, PIPE_MTE3>(flagAivVbReady);
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::NotifyKbgExpReady()
{
    Arch::CrossCoreSetFlag<0x4, PIPE_MTE3>(flagAivKbgExpReady);
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::WaitSlotFree()
{

    Arch::CrossCoreWaitFlag(flagAicSlotFree);
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__simd_callee__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::CastFloat2KTypeRint(
    RegTensor<kType> &dstReg, RegTensor<float> &srcZeroReg,
    RegTensor<float> &srcOneReg, MaskReg &mask)
{
    Cast<kType, float, ctFp322KTypeRintOne>(dstReg, srcOneReg, mask);
    Cast<kType, float, ctFp322KTypeRintZero>(dstReg, srcZeroReg, mask);
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__simd_vf__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::ProcessVbComputerVF(
    __ubuf__ kType *vbOut, __ubuf__ kType *vIn, __ubuf__ betaType *betaIn,
    uint16_t mSize, uint16_t nSize)
{
    const uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    const uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        RegTensor<betaType> betaInReg;
        RegTensor<float> betaBrcbFP32Reg;
        LoadIn<betaType, true>(betaInReg, betaIn + mIdx);
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);

        uint16_t nIdx = 0;
        for (; nIdx + 1 < nLoopCnt; nIdx += 2) {
            RegTensor<kType> vInReg0;
            RegTensor<kType> vInReg1;
            RegTensor<float> vFP32ZeroReg0;
            RegTensor<float> vFP32OneReg0;
            RegTensor<float> vFP32ZeroReg1;
            RegTensor<float> vFP32OneReg1;
            RegTensor<float> vBetaFP32ZeroReg0;
            RegTensor<float> vBetaFP32OneReg0;
            RegTensor<float> vBetaFP32ZeroReg1;
            RegTensor<float> vBetaFP32OneReg1;
            RegTensor<kType> vbOutReg0;
            RegTensor<kType> vbOutReg1;

            LoadIn<kType, false>(vInReg0, vIn + mIdx * nSize + nIdx * eleNumPerVf);
            LoadIn<kType, false>(vInReg1, vIn + mIdx * nSize + (nIdx + 1) * eleNumPerVf);
            CastHalf2Float<kType>(vFP32ZeroReg0, vFP32OneReg0, vInReg0, maskFull16);
            CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
            MulFloatTwoReg(vBetaFP32ZeroReg0, vBetaFP32OneReg0,
                           vFP32ZeroReg0, vFP32OneReg0,
                           betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1,
                           vFP32ZeroReg1, vFP32OneReg1,
                           betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            CastFloat2KTypeRint(vbOutReg0, vBetaFP32ZeroReg0, vBetaFP32OneReg0, maskFull32);
            CastFloat2KTypeRint(vbOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
            StoreAlign(vbOut + mIdx * nSize + nIdx * eleNumPerVf, vbOutReg0, maskFull16);
            StoreAlign(vbOut + mIdx * nSize + (nIdx + 1) * eleNumPerVf, vbOutReg1, maskFull16);
        }
        for (; nIdx < nLoopCnt; ++nIdx) {
            RegTensor<kType> vInReg;
            RegTensor<float> vFP32ZeroReg;
            RegTensor<float> vFP32OneReg;
            RegTensor<float> vBetaFP32ZeroReg;
            RegTensor<float> vBetaFP32OneReg;
            RegTensor<kType> vbOutReg;
            LoadIn<kType, false>(vInReg, vIn + mIdx * nSize + nIdx * eleNumPerVf);
            CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
            MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg,
                           vFP32ZeroReg, vFP32OneReg,
                           betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            CastFloat2KTypeRint(vbOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
            StoreAlign(vbOut + mIdx * nSize + nIdx * eleNumPerVf, vbOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__simd_vf__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::ProcessKbgExpComputerVF(
    __ubuf__ kType *kbgOut, __ubuf__ kType *kIn, __ubuf__ betaType *betaIn,
    __ubuf__ betaType *gIn, uint16_t mSize, uint16_t nSize)
{
    const uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    const uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        RegTensor<betaType> betaInReg;
        RegTensor<betaType> gInReg;
        RegTensor<float> betaFP32Reg;
        RegTensor<float> gFP32Reg;
        RegTensor<float> betaGFP32Reg;
        LoadIn<betaType, true>(betaInReg, betaIn + mIdx);
        LoadIn<betaType, true>(gInReg, gIn + mIdx);
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

        for (uint16_t nIdx = 0; nIdx < nLoopCnt; ++nIdx) {
            RegTensor<kType> kInReg;
            RegTensor<float> kFP32ZeroReg;
            RegTensor<float> kFP32OneReg;
            RegTensor<float> kBetaGFP32ZeroReg;
            RegTensor<float> kBetaGFP32OneReg;
            RegTensor<kType> kbgOutReg;
            LoadIn<kType, false>(kInReg, kIn + mIdx * nSize + nIdx * eleNumPerVf);
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg,
                           kFP32ZeroReg, kFP32OneReg,
                           betaGFP32Reg, betaGFP32Reg, maskFull32);
            CastFloat2KTypeRint(kbgOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
            StoreAlign(kbgOut + mIdx * nSize + nIdx * eleNumPerVf, kbgOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks,
    kCoefficientGenerationTaskOrder>::ProcessVbAndKbgExpInterleaved()
{

    const uint32_t coreLoops = kFlattenHeadTasks ? chunkNum * Hv : chunkNum;
    const uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    const uint32_t coreNumAic = GetBlockNum();
    const uint32_t rowNum = vbVecRow;
    const uint32_t maxDataDim = K > V ? K : V;
    uint32_t loopBegin = coreIdx;
    uint32_t loopEnd = coreLoops;
    uint32_t loopStep = coreNumAic;
    if constexpr (kCoefficientGenerationTaskOrder) {
        const uint32_t tasksPerCore = (coreLoops + coreNumAic - 1) / coreNumAic;
        loopBegin = coreIdx * tasksPerCore;
        loopEnd = (loopBegin + tasksPerCore) < coreLoops ? loopBegin + tasksPerCore : coreLoops;
        loopStep = 1;
    }

    pipe->InitBuffer(vInQue, 2, rowNum * maxDataDim * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(vbOutQue, 2, rowNum * maxDataDim * sizeof(kType));

    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; loopIdx += loopStep) {
        uint32_t chunkIdx = 0;
        uint32_t hBegin = 0;
        uint32_t hEnd = 0;
        DecodeRecomputeTask<kFlattenHeadTasks, kCoefficientGenerationTaskOrder>(
            loopIdx, cu_seqlens, Hv, T, chunkSize, chunkNum, chunkIdx, hBegin, hEnd);
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize,
                       chunkIdx, bos, eos);
        const uint32_t curChunkSize = eos - bos;

        for (uint32_t h = hBegin; h < hEnd; ++h) {
            if (vecTaskIdx >= GM_RING_DEPTH) {

                WaitSlotFree();
            }
            const uint32_t slotId = vecTaskIdx % GM_RING_DEPTH;
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {

                NotifyVbReady();
                NotifyKbgExpReady();
                continue;
            }

            const uint64_t ringTask =
                static_cast<uint64_t>(coreIdx) * GM_RING_DEPTH + slotId;
            const uint64_t vbDstBase = ringTask * chunkSize * V;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                const uint32_t curRowNum =
                    (rowOffset + rowNum > curChunkSize) ? curChunkSize - rowOffset : rowNum;
                const uint64_t vOffset = (h * T + bos + rowOffset) * V;
                const uint64_t betaOffset = h * T + bos + rowOffset;
                auto tensorVinCopy = vInQue.AllocTensor<kType>();
                auto tensorBetaCopy = betaInQue.AllocTensor<betaType>();
                DataCopy(tensorVinCopy, vTensor[vOffset], V * curRowNum);
                DataCopyPad(tensorBetaCopy, betaTensor[betaOffset],
                            {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                            {false, 0, 0, 0});
                vInQue.EnQue(tensorVinCopy);
                betaInQue.EnQue(tensorBetaCopy);

                auto tensorVinCompute = vInQue.DeQue<kType>();
                auto tensorBetaCompute = betaInQue.DeQue<betaType>();
                auto tensorOutCompute = vbOutQue.AllocTensor<kType>();
                ProcessVbComputerVF(
                    (__ubuf__ kType *)tensorOutCompute.GetPhyAddr(),
                    (__ubuf__ kType *)tensorVinCompute.GetPhyAddr(),
                    (__ubuf__ betaType *)tensorBetaCompute.GetPhyAddr(),
                    static_cast<uint16_t>(curRowNum), static_cast<uint16_t>(V));
                vInQue.FreeTensor(tensorVinCompute);
                betaInQue.FreeTensor(tensorBetaCompute);
                vbOutQue.EnQue(tensorOutCompute);

                auto tensorOutCopy = vbOutQue.DeQue<kType>();
                DataCopy(workSpaceTensor[vbDstBase + rowOffset * V], tensorOutCopy,
                         V * curRowNum);
                vbOutQue.FreeTensor(tensorOutCopy);
            }
            NotifyVbReady();

            const uint64_t hk = h / hvPerHk;
            const uint64_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
            const uint64_t bIdx = cu_seqlens ? 0 : (chunkIdx / coreLoopsInB);
            const uint64_t bosK = cu_seqlens ? bos : (bos - bIdx * (Hv - Hk) * T);
            const uint64_t kbgRingBase = static_cast<uint64_t>(coreNumAic) *
                GM_RING_DEPTH * chunkSize * V;
            const uint64_t kbgDstBase = kbgRingBase + ringTask * chunkSize * K;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                const uint32_t curRowNum =
                    (rowOffset + rowNum > curChunkSize) ? curChunkSize - rowOffset : rowNum;
                const uint64_t kSrcOffset = (hk * T + bosK + rowOffset) * K;
                const uint64_t betaOffset = h * T + bos + rowOffset;
                auto tensorKCopy = vInQue.AllocTensor<kType>();
                auto tensorBetaCopy = betaInQue.AllocTensor<betaType>();
                auto tensorGCopy = gInQue.AllocTensor<betaType>();
                DataCopy(tensorKCopy, kTensor[kSrcOffset], K * curRowNum);
                DataCopyPad(tensorBetaCopy, betaTensor[betaOffset],
                            {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                            {false, 0, 0, 0});
                DataCopyPad(tensorGCopy, gTensor[betaOffset],
                            {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                            {false, 0, 0, 0});
                vInQue.EnQue(tensorKCopy);
                betaInQue.EnQue(tensorBetaCopy);
                gInQue.EnQue(tensorGCopy);

                auto tensorKCompute = vInQue.DeQue<kType>();
                auto tensorBetaCompute = betaInQue.DeQue<betaType>();
                auto tensorGCompute = gInQue.DeQue<betaType>();
                auto tensorOutCompute = vbOutQue.AllocTensor<kType>();
                ProcessKbgExpComputerVF(
                    (__ubuf__ kType *)tensorOutCompute.GetPhyAddr(),
                    (__ubuf__ kType *)tensorKCompute.GetPhyAddr(),
                    (__ubuf__ betaType *)tensorBetaCompute.GetPhyAddr(),
                    (__ubuf__ betaType *)tensorGCompute.GetPhyAddr(),
                    static_cast<uint16_t>(curRowNum), static_cast<uint16_t>(K));
                vInQue.FreeTensor(tensorKCompute);
                betaInQue.FreeTensor(tensorBetaCompute);
                gInQue.FreeTensor(tensorGCompute);
                vbOutQue.EnQue(tensorOutCompute);

                auto tensorOutCopy = vbOutQue.DeQue<kType>();
                DataCopy(workSpaceTensor[kbgDstBase + rowOffset * K], tensorOutCopy,
                         K * curRowNum);
                vbOutQue.FreeTensor(tensorOutCopy);
            }
            NotifyKbgExpReady();
        }
    }

    const uint32_t trailingCredits =
        vecTaskIdx < GM_RING_DEPTH ? vecTaskIdx : GM_RING_DEPTH;
    for (uint32_t i = 0; i < trailingCredits; ++i) {
        WaitSlotFree();
    }
}

#endif
