/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd.h
 * \brief
 */


#ifndef RECOMPUTE_W_U_FWD_VECTOR_H
#define RECOMPUTE_W_U_FWD_VECTOR_H

#include "recompute_w_u_fwd_struct.h"
#include "catlass/arch/cross_core_sync.hpp"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_utils/vector/regbase.hpp"
#endif
using namespace AscendC;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using namespace AscendC::MicroAPI;
#endif

using GDN::RecomputeWUFwdTilingData;

template <typename kType, typename betaType, bool kFlattenHeadTasks = false,
          bool kCoefficientGenerationTaskOrder = false>
class RecomputeWUFwdVectorProcess {
public:
    /** @brief constructor */
    __aicore__ inline RecomputeWUFwdVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
                                                        GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessVb();
    __aicore__ inline void ProcessKbgExp();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessVbAndKbgExpInterleaved();
#endif
    __aicore__ inline void Init(const RecomputeWUFwdTilingData &tiling, AscendC::TPipe *pipe_);

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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr uint32_t GM_RING_DEPTH = 8;
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAivVbReady{
        SYNC_AIC_AIV_FLAG_5, SYNC_AIV_AIC_FLAG_3};
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAivKbgExpReady{
        SYNC_AIC_AIV_FLAG_6, SYNC_AIV_AIC_FLAG_4};
#endif
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_AIC_AIV_FLAG_5, SYNC_AIV_AIC_FLAG_3};
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
    // TBuf<AscendC::TPosition::VECCALC> gFp32BrcbBuf;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    constexpr static CastTrait ctFp322KTypeRintZero = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, AscendC::RoundMode::CAST_RINT};
    constexpr static CastTrait ctFp322KTypeRintOne = {
        RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    __aicore__ inline void NotifyVbReady();
    __aicore__ inline void NotifyKbgExpReady();
    __simd_callee__ inline void CastFloat2KTypeRint(
        RegTensor<kType> &dstReg, RegTensor<float> &srcZeroReg,
        RegTensor<float> &srcOneReg, MaskReg &mask);
    __simd_vf__ inline void ProcessVbComputerVF(
        __ubuf__ kType *vbOut, __ubuf__ kType *vIn, __ubuf__ betaType *betaIn,
        uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessKbgExpComputerVF(
        __ubuf__ kType *kbgOut, __ubuf__ kType *kIn, __ubuf__ betaType *betaIn,
        __ubuf__ betaType *gIn, uint16_t mSize, uint16_t nSize);
#endif

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
    const RecomputeWUFwdTilingData &tiling, AscendC::TPipe *pipe_)
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    ProcessVbAndKbgExpInterleaved();
#else
    //计算K * Beta[:None]
    ProcessVb();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessKbgExp();
#endif
    return;
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::NotifyVbReady()
{
    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivVbReady);
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ inline void RecomputeWUFwdVectorProcess<
    kType, betaType, kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::NotifyKbgExpReady()
{
    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivKbgExpReady);
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

    // Vb and KbgExp are sequential within one logical task, so four queues can
    // be shared and sized for max(K, V). This is the compact A5 UB layout.
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
            const uint32_t slotId = vecTaskIdx % GM_RING_DEPTH;
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                // CrossCoreWait consumes one credit from each AIV subcore. The
                // non-owning subcore therefore publishes a matching empty credit.
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
}
#endif


template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType,
                                                    kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::ProcessVb()
{
    uint32_t coreLoops = kFlattenHeadTasks ? chunkNum * Hv : chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = vbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t wholeReduceSumCnt = CeilDiv(V, FP32_PER_REPEAT_64);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // // init
    pipe->InitBuffer(vInQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(vbOutQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(vFp32Buf, rowNum * V * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);


    auto tensorVFp32 = vFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();

    uint32_t loopBegin = coreIdx;
    uint32_t loopEnd = coreLoops;
    uint32_t loopStep = coreNumAic;
    if constexpr (kCoefficientGenerationTaskOrder) {
        // Phase 6 coefficient generation assigns a contiguous task range to each AIC. Keep the
        // recompute consumer on the same core so A is locally produced first.
        const uint32_t tasksPerCore = (coreLoops + coreNumAic - 1) / coreNumAic;
        loopBegin = coreIdx * tasksPerCore;
        loopEnd = (loopBegin + tasksPerCore) < coreLoops ? loopBegin + tasksPerCore : coreLoops;
        loopStep = 1;
    }
    for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; loopIdx += loopStep) {
        uint32_t chunkIdx = 0;
        uint32_t hBegin = 0;
        uint32_t hEnd = 0;
        DecodeRecomputeTask<kFlattenHeadTasks, kCoefficientGenerationTaskOrder>(
            loopIdx, cu_seqlens, Hv, T, chunkSize, chunkNum, chunkIdx, hBegin, hEnd);
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize, chunkIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (uint32_t h = hBegin; h < hEnd; ++h) {
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
                continue;
            }
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * V;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorVin = vInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();

                    DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});

                    vInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                }
                // compute
                {
                    auto tensorVin = vInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();

                    auto tensorVbOut = vbOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                    }
                    Cast(tensorVFp32, tensorVin, RoundMode::CAST_NONE, V * curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < V) {
                        Mul(tensorVFp32[perchannelResOffset], tensorVFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    // cast
                    Cast(tensorVbOut, tensorVFp32, RoundMode::CAST_RINT, V * curRowNum);

                    vInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);

                    vbOutQue.EnQue(tensorVbOut);
                }
                // copyout
                {
                    auto tensorVbOut = vbOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[vOffset], tensorVbOut, V * curRowNum);
                    vbOutQue.FreeTensor(tensorVbOut);
                }
            }

            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
        }
    }
    return;
}

template <typename kType, typename betaType, bool kFlattenHeadTasks, bool kCoefficientGenerationTaskOrder>
__aicore__ void inline RecomputeWUFwdVectorProcess<kType, betaType,
                                                    kFlattenHeadTasks, kCoefficientGenerationTaskOrder>::ProcessKbgExp()
{
    uint32_t coreLoops = kFlattenHeadTasks ? chunkNum * Hv : chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = vbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(kBetagExpOutQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(kFp32Buf, rowNum * K * sizeof(float32_t));
    pipe->InitBuffer(betaFp32Buf, rowNum * sizeof(float32_t));
    pipe->InitBuffer(betaFp32BrcbBuf, rowNum * ONE_BLOCK_32);
    pipe->InitBuffer(gFp32Buf, rowNum * sizeof(float32_t));

    auto tensorKFp32 = kFp32Buf.Get<float32_t>();
    auto tensorGFP32 = gFp32Buf.Get<float32_t>();
    auto tensorBetaFP32 = betaFp32Buf.Get<float32_t>();
    auto tensorBetaBrcbFP32 = betaFp32BrcbBuf.Get<float32_t>();

    uint32_t loopBegin = coreIdx;
    uint32_t loopEnd = coreLoops;
    uint32_t loopStep = coreNumAic;
    if constexpr (kCoefficientGenerationTaskOrder) {
        const uint32_t tasksPerCore = (coreLoops + coreNumAic - 1) / coreNumAic;
        loopBegin = coreIdx * tasksPerCore;
        loopEnd = (loopBegin + tasksPerCore) < coreLoops ? loopBegin + tasksPerCore : coreLoops;
        loopStep = 1;
    }
    for (uint32_t loopIdx = loopBegin; loopIdx < loopEnd; loopIdx += loopStep) {
        uint32_t chunkIdx = 0;
        uint32_t hBegin = 0;
        uint32_t hEnd = 0;
        DecodeRecomputeTask<kFlattenHeadTasks, kCoefficientGenerationTaskOrder>(
            loopIdx, cu_seqlens, Hv, T, chunkSize, chunkNum, chunkIdx, hBegin, hEnd);
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize, chunkIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (uint32_t h = hBegin; h < hEnd; ++h) {
            ++vecTaskIdx;
            uint64_t hk = h / hvPerHk;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
                continue;
            }
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                // 注意：定长场景下 GetChunkOffset 返回的 bos 已含按 Hv 计的批次偏移 bIdx*Hv*T
                // （见 recompute_w_u_fwd_common.h GetChunkOffset 的 bos += bIdx*H*T，此处 H 传入的是 Hv）。
                // k 只有 Hk 个 head，需把批次偏移换算成 bIdx*Hk*T，即 bos - bIdx*(Hv-Hk)*T。
                // 此换算强耦合于 GetChunkOffset 的批次偏移实现，若后者修改需同步更新此处。
                // coreLoopsInB 必须与 GetChunkOffset 内保持一致的算法。
                uint64_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
                uint64_t bIdx = cu_seqlens ? 0 : (chunkIdx / coreLoopsInB);
                uint64_t bosK = cu_seqlens ? bos : (bos - bIdx * (Hv - Hk) * T);
                auto kSrcOffset = (hk * T + bosK + rowOffset) * K;
                auto kDstOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
                // copyin
                {
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorGin = gInQue.AllocTensor<betaType>();
                    DataCopy(tensorKin, kTensor[kSrcOffset], K * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    DataCopyPad(tensorGin, gTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    gInQue.EnQue(tensorGin);
                }
                // compute
                {
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorGin = gInQue.DeQue<betaType>();
                    auto tensorOut = kBetagExpOutQue.AllocTensor<kType>();
                    // cast FP32
                    if constexpr (!std::is_same<betaType, float32_t>()) {
                        Cast(tensorBetaFP32, tensorBetain, RoundMode::CAST_NONE, curRowNum);
                        Cast(tensorGFP32, tensorGin, RoundMode::CAST_NONE, curRowNum);
                    } else {
                        DataCopy(tensorBetaFP32, tensorBetain, rowNum);
                        DataCopy(tensorGFP32, tensorGin, rowNum);
                    }
                    Cast(tensorKFp32, tensorKin, RoundMode::CAST_NONE, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                    Exp(tensorGFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    Mul(tensorBetaFP32, tensorBetaFP32, tensorGFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                    // brcb
                    Brcb(tensorBetaBrcbFP32, tensorBetaFP32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                    PipeBarrier<PIPE_V>();
                    // mul
                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < K) {
                        Mul(tensorKFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbFP32,
                            FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    Cast(tensorOut, tensorKFp32, RoundMode::CAST_RINT, K * curRowNum);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    gInQue.FreeTensor(tensorGin);
                    kBetagExpOutQue.EnQue(tensorOut);
                }
                // copyout
                {
                    auto tensorOut = kBetagExpOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[kDstOffset], tensorOut, K * curRowNum);
                    kBetagExpOutQue.FreeTensor(tensorOut);
                }
            }
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
        }
    }
    return;
}


#endif // RECOMPUTE_W_U_FWD_VECTOR_H
