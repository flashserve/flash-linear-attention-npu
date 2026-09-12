/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recompute_w_u_fwd_vector_regbase.h
 * \brief A5 Regbase Vector 实现。
 *        相对 Membase 版本的改动：
 *          - ProcessVb / ProcessKbgExp 计算内核用 Regbase API 替代 Membase
 *          - 消除 Brcb、PipeBarrier，用 RegTensor/MulFloatTwoReg 替代
 *          - 双行预取隐藏延迟
 */

#ifndef RECOMPUTE_W_U_FWD_VECTOR_REGBASE_H
#define RECOMPUTE_W_U_FWD_VECTOR_REGBASE_H

#include "../recompute_w_u_fwd_struct.h"
#include "../recompute_w_u_fwd_common.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

using GDN::RecomputeWUFwdTilingData;

template <typename kType, typename betaType, int VDim>
class RecomputeWUFwdVectorProcessRegbase {
public:
    constexpr static CastTrait ctHalf2Fp32Zero = {
        RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
    constexpr static CastTrait ctFp322KTypeRintZero = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING, AscendC::RoundMode::CAST_RINT};
    constexpr static CastTrait ctFp322KTypeRintOne = {
        RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    __aicore__ inline RecomputeWUFwdVectorProcessRegbase(
        GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_, GM_ADDR cu_seqlens_,
        GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_, GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessVbAndKbgExpInterleaved();
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
    uint64_t interleavedVecRow = 0;

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

    // Keep the two stages on independent channels so a later KbgExp event
    // cannot satisfy the preceding Vb wait when many tasks are in flight.
    static constexpr uint32_t GM_RING_DEPTH = GDN::RECOMPUTE_W_U_FWD_GM_RING_DEPTH;
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAivVbReady{
        SYNC_AIC_AIV_FLAG_5, SYNC_AIV_AIC_FLAG_3};
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAivKbgExpReady{
        SYNC_AIC_AIV_FLAG_6, SYNC_AIV_AIC_FLAG_4};
    Arch::CrossCoreFlagWithReverse<GM_RING_DEPTH> flagAivRingSlotFree{
        SYNC_AIC_AIV_RING_SLOT_FREE_FLAG, SYNC_AIV_AIC_RING_SLOT_FREE_REVERSE_FLAG};
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<kType> workSpaceTensor;

    TQue<AscendC::TPosition::VECIN, 1> dataInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECOUT, 1> dataOutQue;

    __aicore__ inline void NotifyVbReady();
    __aicore__ inline void NotifyKbgExpReady();
    __aicore__ inline void WaitRingSlotFree();
    __simd_callee__ inline void CastFloat2KTypeRint(
        RegTensor<kType>& dstReg, RegTensor<float>& srcZeroReg,
        RegTensor<float>& srcOneReg, MaskReg& mask);

    // Regbase VB compute: vb = v * beta
    __simd_vf__ inline void ProcessVbComputerVF(
        __ubuf__ kType* vbOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
        uint16_t mSize, uint16_t nSize);

    // Regbase KbgExp compute: kbg_exp = k * beta * exp(g)
    __simd_vf__ inline void ProcessKbgExpComputerVF(
        __ubuf__ kType* kbgOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
        __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize);
};

template <typename kType, typename betaType, int VDim>
__aicore__ inline RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::RecomputeWUFwdVectorProcessRegbase(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR w_, GM_ADDR u_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), w(w_), u(u_), workspace(workspace_){};

template <typename kType, typename betaType, int VDim>
__aicore__ void inline RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::Init(
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
    interleavedVecRow = tiling.interleavedVecRow;
    return;
}

template <typename kType, typename betaType, int VDim>
__aicore__ void inline RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::Process()
{
    ProcessVbAndKbgExpInterleaved();
    return;
}

template <typename kType, typename betaType, int VDim>
__aicore__ inline void RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::NotifyVbReady()
{
    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivVbReady);
}

template <typename kType, typename betaType, int VDim>
__aicore__ inline void RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::NotifyKbgExpReady()
{
    Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivKbgExpReady);
}

template <typename kType, typename betaType, int VDim>
__aicore__ inline void RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::WaitRingSlotFree()
{
    // Block the next MTE3 store until AIC has retired the slot's MTE2 read.
    Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAivRingSlotFree);
}

template <typename kType, typename betaType, int VDim>
__simd_callee__ inline void RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::CastFloat2KTypeRint(
    RegTensor<kType>& dstReg, RegTensor<float>& srcZeroReg,
    RegTensor<float>& srcOneReg, MaskReg& mask)
{
    Cast<kType, float, ctFp322KTypeRintOne>(dstReg, srcOneReg, mask);
    Cast<kType, float, ctFp322KTypeRintZero>(dstReg, srcZeroReg, mask);
}

// =================== Regbase VB compute ===================
// vb = v * beta
// Regbase: LoadIn beta (broadcast), LoadIn v, Cast to FP32, Mul, Cast back, StoreAlign
template <typename kType, typename betaType, int VDim>
__simd_vf__ inline void RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::ProcessVbComputerVF(
    __ubuf__ kType* vbOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;
    uint32_t oneEleNum = min(eleNumPerVf, (uint32_t)nSize);

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        RegTensor<betaType> betaInReg;
        RegTensor<kType> vInReg;
        RegTensor<float> vFP32ZeroReg, vFP32OneReg;
        RegTensor<float> vBetaFP32ZeroReg, vBetaFP32OneReg;
        RegTensor<float> betaBrcbFP32Reg;
        RegTensor<kType> vbOutReg;

        LoadIn<betaType, true>(betaInReg, betaIn + mIdx);
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);

        uint16_t nIdx = 0;
        for (nIdx = 0; nIdx + 1 < nLoopCnt; nIdx += 2) {
            // 双列预取
            RegTensor<kType> vInReg1;
            RegTensor<float> vFP32ZeroReg1, vFP32OneReg1;
            RegTensor<float> vBetaFP32ZeroReg1, vBetaFP32OneReg1;
            RegTensor<kType> vbOutReg1;

            LoadIn<kType, false>(vInReg, vIn + mIdx * nSize + nIdx * eleNumPerVf);
            LoadIn<kType, false>(vInReg1, vIn + mIdx * nSize + (nIdx + 1) * eleNumPerVf);
            CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
            CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
            MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg,
                           betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1,
                           betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            CastFloat2KTypeRint(vbOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
            CastFloat2KTypeRint(vbOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
            StoreAlign(vbOut + mIdx * nSize + nIdx * eleNumPerVf, vbOutReg, maskFull16);
            StoreAlign(vbOut + mIdx * nSize + (nIdx + 1) * eleNumPerVf, vbOutReg1, maskFull16);
        }
        for (; nIdx < nLoopCnt; nIdx++) {
            LoadIn<kType, false>(vInReg, vIn + mIdx * nSize + nIdx * eleNumPerVf);
            CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
            MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg,
                           betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            CastFloat2KTypeRint(vbOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
            StoreAlign(vbOut + mIdx * nSize + nIdx * eleNumPerVf, vbOutReg, maskFull16);
        }
    }
}

// =================== Regbase KbgExp compute ===================
// kbg_exp = k * beta * exp(g)
template <typename kType, typename betaType, int VDim>
__simd_vf__ inline void RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::ProcessKbgExpComputerVF(
    __ubuf__ kType* kbgOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleNumPerVf - 1) / eleNumPerVf;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        RegTensor<betaType> betaInReg;
        RegTensor<betaType> gInReg;
        RegTensor<float> betaFP32Reg, gFP32Reg, betaGFP32Reg;
        RegTensor<kType> kInReg;
        RegTensor<float> kFP32ZeroReg, kFP32OneReg;
        RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg;
        RegTensor<kType> kbgOutReg;

        LoadIn<betaType, true>(betaInReg, betaIn + mIdx);
        LoadIn<betaType, true>(gInReg, gIn + mIdx);
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

        for (uint16_t nIdx = 0; nIdx < nLoopCnt; nIdx++) {
            LoadIn<kType, false>(kInReg, kIn + mIdx * nSize + nIdx * eleNumPerVf);
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg,
                           betaGFP32Reg, betaGFP32Reg, maskFull32);
            CastFloat2KTypeRint(kbgOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
            StoreAlign(kbgOut + mIdx * nSize + nIdx * eleNumPerVf, kbgOutReg, maskFull16);
        }
    }
}

// =================== ProcessVbAndKbgExpInterleaved ===================
// 交替模式：每个 chunk+head 内先做 Vb 再做 KbgExp，使 Cube 端能复用 L1 中的 A 矩阵
// 优化：一次性初始化所有 6 个 buffer，避免循环内 pipe->Reset() 调用
template <typename kType, typename betaType, int VDim>
__aicore__ void inline RecomputeWUFwdVectorProcessRegbase<kType, betaType, VDim>::ProcessVbAndKbgExpInterleaved()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = interleavedVecRow;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // Vb and KbgExp execute sequentially, so reuse data queues sized for the
    // larger K/V dimension. This keeps the A5 double-buffer queue count at four.
    uint32_t maxDataDim = K > V ? K : V;
    pipe->InitBuffer(dataInQue, 2, rowNum * maxDataDim * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(dataOutQue, 2, rowNum * maxDataDim * sizeof(kType));

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, Hv, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < Hv; h++) {
            ++vecTaskIdx;
            // The first ring window is initially empty.  Thereafter every
            // task reuses the slot released by the task GM_RING_DEPTH steps
            // earlier.  All AIV subblocks consume the release token so the
            // reverse-depth accounting remains aligned with AIC.
            if (vecTaskIdx > GM_RING_DEPTH) {
                WaitRingSlotFree();
            }
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                // Both AIV subcores must contribute one credit per stage.
                NotifyVbReady();
                NotifyKbgExpReady();
                continue;
            }

            // === 第一部分：Vb = v * beta ===
            const uint64_t ringTask = static_cast<uint64_t>(coreIdx) *
                GM_RING_DEPTH + (vecTaskIdx - 1) % GM_RING_DEPTH;
            const uint64_t vbDstBase = ringTask * chunkSize * V;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * V;
                auto betaOffset = h * T + bos + rowOffset;
                {
                    auto tensorVin = dataInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    dataInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                }
                {
                    auto tensorVin = dataInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorVbOut = dataOutQue.AllocTensor<kType>();
                    __ubuf__ kType* vInAddr = (__ubuf__ kType*)tensorVin.GetPhyAddr();
                    __ubuf__ betaType* betaInAddr = (__ubuf__ betaType*)tensorBetain.GetPhyAddr();
                    __ubuf__ kType* vbOutAddr = (__ubuf__ kType*)tensorVbOut.GetPhyAddr();
                    ProcessVbComputerVF(vbOutAddr, vInAddr, betaInAddr,
                                        static_cast<uint16_t>(curRowNum), static_cast<uint16_t>(V));
                    dataInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);
                    dataOutQue.EnQue(tensorVbOut);
                }
                {
                    auto tensorVbOut = dataOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[vbDstBase + rowOffset * V], tensorVbOut, V * curRowNum);
                    dataOutQue.FreeTensor(tensorVbOut);
                }
            }
            // Vb 完成通知 AIC 可以做 U matmul
            NotifyVbReady();

            // === 第二部分：KbgExp = k * beta * exp(g) ===
            uint64_t hk = h / hvPerHk;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                uint64_t coreLoopsInB = (T + chunkSize - 1) / chunkSize;
                uint64_t bIdx = cu_seqlens ? 0 : (loopIdx / coreLoopsInB);
                uint64_t bosK = cu_seqlens ? bos : (bos - bIdx * (Hv - Hk) * T);
                auto kSrcOffset = (hk * T + bosK + rowOffset) * K;
                const uint64_t kbgRingBase = static_cast<uint64_t>(coreNumAic) *
                    GM_RING_DEPTH * chunkSize * V;
                const uint64_t kbgDstBase = kbgRingBase + ringTask * chunkSize * K;
                auto betaOffset = h * T + bos + rowOffset;
                {
                    auto tensorKin = dataInQue.AllocTensor<kType>();
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    auto tensorGin = gInQue.AllocTensor<betaType>();
                    DataCopy(tensorKin, kTensor[kSrcOffset], K * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopyPad(tensorGin, gTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    dataInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                    gInQue.EnQue(tensorGin);
                }
                {
                    auto tensorKin = dataInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorGin = gInQue.DeQue<betaType>();
                    auto tensorOut = dataOutQue.AllocTensor<kType>();
                    __ubuf__ kType* kInAddr = (__ubuf__ kType*)tensorKin.GetPhyAddr();
                    __ubuf__ betaType* betaInAddr = (__ubuf__ betaType*)tensorBetain.GetPhyAddr();
                    __ubuf__ betaType* gInAddr = (__ubuf__ betaType*)tensorGin.GetPhyAddr();
                    __ubuf__ kType* kbgOutAddr = (__ubuf__ kType*)tensorOut.GetPhyAddr();
                    ProcessKbgExpComputerVF(kbgOutAddr, kInAddr, betaInAddr, gInAddr,
                                             static_cast<uint16_t>(curRowNum), static_cast<uint16_t>(K));
                    dataInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    gInQue.FreeTensor(tensorGin);
                    dataOutQue.EnQue(tensorOut);
                }
                {
                    auto tensorOut = dataOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[kbgDstBase + rowOffset * K], tensorOut, K * curRowNum);
                    dataOutQue.FreeTensor(tensorOut);
                }
            }
            // KbgExp 完成通知 AIC 可以做 W matmul
            NotifyKbgExpReady();
        }
    }

    // AIC publishes one release for every task, while the first ring window
    // does not consume a release token. Drain the outstanding tail so the
    // AIC/AIV Set/Wait and reverse-credit counts stay balanced on Ascend 950.
    uint32_t pendingRingSlots = vecTaskIdx < GM_RING_DEPTH ? vecTaskIdx : GM_RING_DEPTH;
    for (uint32_t i = 0; i < pendingRingSlots; ++i) {
        WaitRingSlotFree();
    }
}

#endif // RECOMPUTE_W_U_FWD_VECTOR_REGBASE_H
