/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full.h
 * \brief
 */


#ifndef PREPARE_WY_REPR_BWD_FULL_VECTOR_H
#define PREPARE_WY_REPR_BWD_FULL_VECTOR_H
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_utils/vector/regbase.hpp"
constexpr uint32_t UB_ALIGN_SIZE = 32;

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename kType, typename betaType>
class PrepareWyReprBwdFullVectorProcess {
public:
    constexpr static CastTrait ctHalf2Fp32Zero = {
        RegLayout::ZERO,
        SatMode::SAT,
        MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_NONE,
    };
    constexpr static CastTrait ctHalf2Fp32One = {
        RegLayout::ONE,
        SatMode::SAT,
        MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_NONE,
    };
    constexpr static CastTrait ctFp322HalfZero = {
        RegLayout::ZERO,
        SatMode::NO_SAT,
        MaskMergeMode::MERGING,
        AscendC::RoundMode::CAST_ROUND
    };
    constexpr static CastTrait ctFp322HalfOne = {
        RegLayout::ONE,
        SatMode::NO_SAT,
        MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND
    };

    /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_,
                                                        GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_,
                                                        GM_ADDR dbeta_, GM_ADDR dg_, GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessKBetaA5();
    __aicore__ inline void ProcessDkbA5();
    __aicore__ inline void ProcessDkbgA5();
    __aicore__ inline void ProcessDkGatherA5();
    __simd_vf__ inline void ProcessDkGatherComputerVFOneLineOneCol(__ubuf__ kType* dkOut,
                                                  __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
                                                  uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkGatherComputerVFMutiLineOneCol(__ubuf__ kType* dkOut,
                                                  __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
                                                  uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkGatherComputerVFTwoCol(__ubuf__ kType* dkOut,
                                                  __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
                                                  uint16_t mSize, uint16_t nSize);
    __aicore__ inline void ProcessDvbA5();
    __aicore__ inline void ProcessKKTA5();
    __aicore__ inline void Init(const PrepareWyReprBwdFullTilingDataA5 &tiling, AscendC::TPipe *pipe_);
    __simd_vf__ inline void ProcessKBetaComputerVFMutiLineOneCol(__ubuf__ kType* kBetaOut,
                                                   __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                   uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt);
    __simd_vf__ inline void ProcessKBetaComputerVFOneLineOneCol(__ubuf__ kType* kBetaOut,
                                                   __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                   uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessKBetaComputerVFTwoCol(__ubuf__ kType* kBetaOut,
                                                   __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                   uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkbComputerVFOneLineOneCol(__ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut,
                                                 __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                 __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkbComputerVFMutiLineOneCol(__ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut,
                                                 __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                 __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkbComputerVFTwoCol(__ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut,
                                                 __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                 __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkbgComputerGatherDkVF(__ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, 
                                                  __ubuf__ betaType* dgOut, __ubuf__ kType* kIn, 
                                                  __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
                                                  __ubuf__ kType* dkIn, __ubuf__ kType* dkGatherIn, __ubuf__ betaType* dBetaIn,
                                                  __ubuf__ kType* dkbgIn, uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkbgComputerVF(__ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, 
                                                  __ubuf__ betaType* dgOut, __ubuf__ kType* kIn, 
                                                  __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
                                                  __ubuf__ kType* dkIn, __ubuf__ betaType* dBetaIn,
                                                  __ubuf__ kType* dkbgIn, uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDvbComputerVFOneLineOneCol(__ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut,
                                                 __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
                                                 __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDvbComputerVFMutiLineOneCol(__ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut,
                                                 __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
                                                 __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDvbComputerVFTwoCol(__ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut,
                                                 __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
                                                 __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessKKTComputerVF(__ubuf__ kType* kktIn,
                                                 __ubuf__ betaType* betaIn, __ubuf__ kType* daIn,
                                                 __ubuf__ float* reducesum1, __ubuf__ float* reducesum0,
                                                 uint32_t calcColSize,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessKKTComputerVFSub(__ubuf__ betaType* dgOut, __ubuf__ betaType* dgIn,
                                                    __ubuf__ float* reducesum1, __ubuf__ float* reducesum0,
                                                    uint16_t calcSize);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t HV = 0;
    uint64_t HK = 0;
    uint64_t groupSize = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t kBeteVecRow = 0;
    uint64_t dkbVecRow = 0;
    uint64_t dkbgVecRow = 0;
    uint64_t dkGatherVecRow = 0;
    uint64_t dvbVecRow = 0;
    uint64_t kktVecRow = 0;
    uint64_t kBetaCVNum = 0;
    uint64_t dkbCVNum = 0;
    uint64_t dkbgCVNum = 0;
    uint64_t dvbCVNum = 0;
    uint64_t kktCVNum = 0;

    uint64_t dkbVecOffset = 0;

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dA;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR dk;
    GM_ADDR dv;
    GM_ADDR dbeta;
    GM_ADDR dg;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;

private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<kType> dkTensor;
    GlobalTensor<kType> dvTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<betaType> dbetaTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<betaType> dgTensor;
    GlobalTensor<kType> dATensor;
    GlobalTensor<kType> workSpaceTensor;
    GlobalTensor<kType> workSpaceDkTensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;

    TQue<AscendC::TPosition::VECOUT, 1> kBetaOutQue;
};

template <typename kType, typename betaType>
__aicore__ inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::PrepareWyReprBwdFullVectorProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_,
    GM_ADDR cu_seqlens_, GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dbeta_, GM_ADDR dg_,
    GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), dA(dA_), dw(dw_), du(du_), g(g_), cu_seqlens(cu_seqlens_),
      chunk_indices(chunk_indices_), dk(dk_), dv(dv_), dbeta(dbeta_), dg(dg_), workspace(workspace_){};

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Init(
    const PrepareWyReprBwdFullTilingDataA5 &tiling, AscendC::TPipe *pipe_)
{
    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    dkTensor.SetGlobalBuffer((__gm__ kType *)dk);
    dvTensor.SetGlobalBuffer((__gm__ kType *)dv);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    dbetaTensor.SetGlobalBuffer((__gm__ betaType *)dbeta);
    dgTensor.SetGlobalBuffer((__gm__ betaType *)dg);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);
    dATensor.SetGlobalBuffer((__gm__ kType *)dA);

    B = tiling.B;
    T = tiling.T;
    HV = tiling.HV;
    HK = tiling.HK;
    groupSize = HV / HK;
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    kBeteVecRow = tiling.kBeteVecRow;
    dkbVecRow = tiling.dkbVecRow;
    dkbgVecRow = tiling.dkbgVecRow;
    dkGatherVecRow = tiling.dkGatherVecRow;
    dvbVecRow = tiling.dvbVecRow;
    kktVecRow = tiling.kktVecRow;
    kBetaCVNum = tiling.kBetaCVNum;
    dkbCVNum = tiling.dkbCVNum;
    dkbgCVNum = tiling.dkbgCVNum;
    dvbCVNum = tiling.dvbCVNum;
    kktCVNum = tiling.kktCVNum;

    workSpaceDkTensor.SetGlobalBuffer((__gm__ kType *)workspace + B * HV * T * V);

    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Process()
{
    //计算K * Beta[:None]
    ProcessKBetaA5();
    AscendC::SyncAll<false>();
    ProcessDkbA5();
    AscendC::SyncAll<false>();
    ProcessDkbgA5();
    AscendC::SyncAll<false>();
    if (groupSize > 1) {
        ProcessDkGatherA5();
    }
    ProcessDvbA5();
    AscendC::SyncAll<false>();
    ProcessKKTA5();

    return;
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKKTComputerVF(
    __ubuf__ kType* kktIn, __ubuf__ betaType* betaIn, __ubuf__ kType* daIn,
    __ubuf__ float* reducesum1, __ubuf__ float* reducesum0,
    uint32_t calcColSize, uint16_t mSize, uint16_t nSize)
{
    RegTensor<betaType> betaInReg;
    RegTensor<kType> kktInReg;
    RegTensor<kType> daInReg;
    RegTensor<float> betaFP32ZeroReg, betaFP32OneReg, kktFP32ZeroReg, kktFP32OneReg, daFP32ZeroReg, daFP32OneReg;
    RegTensor<float> daaFP32ZeroReg, daaFP32OneReg, daaFP32AddReg, reduceSumReg, Sum1DaaLineFP32Reg;
    RegTensor<float> dgFP32AddZeroReg, dgFP32AddOneReg;
    RegTensor<float> reducesum0ZeroReg, reducesum0OneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    uint32_t calcNNum = calcColSize;
    UnalignRegForStore uStoreReducesum;

    if constexpr (!std::is_same<betaType, float>()) {
        LoadAlign(betaInReg, betaIn);
        Cast<float, betaType, ctHalf2Fp32Zero>(betaFP32ZeroReg, betaInReg, maskFull16);
        Cast<float, betaType, ctHalf2Fp32One>(betaFP32OneReg, betaInReg, maskFull16);
    } else {
        LoadAlign<betaType, LoadDist::DIST_DINTLV_B32>(betaFP32ZeroReg, betaFP32OneReg, betaIn);
    }
    Duplicate(dgFP32AddZeroReg, static_cast<float>(0), maskFull32);
    Duplicate(dgFP32AddOneReg, static_cast<float>(0), maskFull32);
    uint32_t nextEleOffset = 0;
    // 先copy第一块数据
    LoadAlign(kktInReg, kktIn);
    LoadAlign(daInReg, daIn);
    // 做前mSize-1行的exe并copy下一个loop需要的数据
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t castKktCountZero = calcNNum;
        uint32_t castKktCountOne = calcNNum;
        MaskReg castKktMaskCountZero16, castKktMaskCountOne16, castDaMaskCountZero16, castDaMaskCountOne16;
        castKktMaskCountZero16 = UpdateMask<half>(castKktCountZero);
        castKktMaskCountOne16 = UpdateMask<half>(castKktCountOne);
        uint32_t castDaCountZero = calcNNum;
        uint32_t castDaCountOne = calcNNum;
        castDaMaskCountZero16 = UpdateMask<half>(castDaCountZero);
        castDaMaskCountOne16 = UpdateMask<half>(castDaCountOne);
        Cast<float, kType, ctHalf2Fp32Zero>(kktFP32ZeroReg, kktInReg, castKktMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(kktFP32OneReg, kktInReg, castKktMaskCountOne16);
        Cast<float, kType, ctHalf2Fp32Zero>(daFP32ZeroReg, daInReg, castDaMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(daFP32OneReg, daInReg, castDaMaskCountOne16);
        uint32_t nextEleOffset = (mIdx + 1) * nSize;
        LoadAlign(kktInReg, kktIn + nextEleOffset);
        LoadAlign(daInReg, daIn + nextEleOffset);
        Mul(kktFP32ZeroReg, kktFP32ZeroReg, betaFP32ZeroReg, maskFull32);
        Mul(kktFP32OneReg, kktFP32OneReg, betaFP32OneReg, maskFull32);
        Mul(daaFP32ZeroReg, daFP32ZeroReg, kktFP32ZeroReg, maskFull32);
        Mul(daaFP32OneReg, daFP32OneReg, kktFP32OneReg, maskFull32);
        // 每行相加
        Add(daaFP32AddReg, daaFP32ZeroReg, daaFP32OneReg, maskFull32);
        Add(reduceSumReg, reduceSumReg, daaFP32AddReg, maskFull32);
        // 每列相加
        Add(dgFP32AddZeroReg, dgFP32AddZeroReg, daaFP32ZeroReg, maskFull32);
        Add(dgFP32AddOneReg, dgFP32AddOneReg, daaFP32OneReg, maskFull32);
        ReduceSum(Sum1DaaLineFP32Reg, reduceSumReg, maskFull32);
        auto actReducesum1 = reducesum1 + mIdx;
        StoreUnAlign(actReducesum1, Sum1DaaLineFP32Reg, uStoreReducesum, 1);
        StoreUnAlignPost(actReducesum1, uStoreReducesum, 0);
    }
    // 最后一行数据exe
    uint16_t mIdx = mSize - 1;
    {
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t castKktCountZero = calcNNum;
        uint32_t castKktCountOne = calcNNum;
        MaskReg castKktMaskCountZero16, castKktMaskCountOne16, castDaMaskCountZero16, castDaMaskCountOne16;
        castKktMaskCountZero16 = UpdateMask<half>(castKktCountZero);
        castKktMaskCountOne16 = UpdateMask<half>(castKktCountOne);
        uint32_t castDaCountZero = calcNNum;
        uint32_t castDaCountOne = calcNNum;
        castDaMaskCountZero16 = UpdateMask<half>(castDaCountZero);
        castDaMaskCountOne16 = UpdateMask<half>(castDaCountOne);
        Cast<float, kType, ctHalf2Fp32Zero>(kktFP32ZeroReg, kktInReg, castKktMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(kktFP32OneReg, kktInReg, castKktMaskCountOne16);
        Cast<float, kType, ctHalf2Fp32Zero>(daFP32ZeroReg, daInReg, castDaMaskCountZero16);
        Cast<float, kType, ctHalf2Fp32One>(daFP32OneReg, daInReg, castDaMaskCountOne16);
        Mul(kktFP32ZeroReg, kktFP32ZeroReg, betaFP32ZeroReg, maskFull32);
        Mul(kktFP32OneReg, kktFP32OneReg, betaFP32OneReg, maskFull32);
        Mul(daaFP32ZeroReg, daFP32ZeroReg, kktFP32ZeroReg, maskFull32);
        Mul(daaFP32OneReg, daFP32OneReg, kktFP32OneReg, maskFull32);
        // 每行相加
        Add(daaFP32AddReg, daaFP32ZeroReg, daaFP32OneReg, maskFull32);
        Add(reduceSumReg, reduceSumReg, daaFP32AddReg, maskFull32);
        // 每列相加
        Add(dgFP32AddZeroReg, dgFP32AddZeroReg, daaFP32ZeroReg, maskFull32);
        Add(dgFP32AddOneReg, dgFP32AddOneReg, daaFP32OneReg, maskFull32);
        ReduceSum(Sum1DaaLineFP32Reg, reduceSumReg, maskFull32);
        auto actReducesum1 = reducesum1 + mIdx;
        StoreUnAlign(actReducesum1, Sum1DaaLineFP32Reg, uStoreReducesum, 1);
        StoreUnAlignPost(actReducesum1, uStoreReducesum, 0);
    }
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(reducesum0ZeroReg, reducesum0OneReg, reducesum0);
    Add(reducesum0ZeroReg, reducesum0ZeroReg, dgFP32AddZeroReg, maskFull32);
    Add(reducesum0OneReg, reducesum0OneReg, dgFP32AddOneReg, maskFull32);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(reducesum0, reducesum0ZeroReg, reducesum0OneReg, maskFull32);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKKTComputerVFSub(
    __ubuf__ betaType* dgOut, __ubuf__ betaType* dgIn, __ubuf__ float* reducesum1, __ubuf__ float* reducesum0,
    uint16_t calcSize)
{
    uint32_t eleNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(betaType);
    uint16_t nLoopCnt = (calcSize + eleNumPerVf - 1) / eleNumPerVf;

    RegTensor<betaType> dgInReg;
    RegTensor<float> sum1DaaLineZeroFP32Reg, sum1DaaLineOneFP32Reg, dgFP32AddZeroReg, dgFP32AddOneReg;
    RegTensor<float> dgFP32ZeroReg, dgFP32OneReg;
    RegTensor<betaType> dgOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
        uint32_t eleOffset = vfBlockIdx * eleNumPerVf;
        if constexpr (!std::is_same<betaType, float>()) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(sum1DaaLineZeroFP32Reg, sum1DaaLineOneFP32Reg, reducesum1 + eleOffset);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(dgFP32AddZeroReg, dgFP32AddOneReg, reducesum0 + eleOffset);
            LoadAlign(dgInReg, dgIn + eleOffset);
            Sub(dgFP32AddZeroReg, dgFP32AddZeroReg, sum1DaaLineZeroFP32Reg, maskFull32);
            Sub(dgFP32AddOneReg, dgFP32AddOneReg, sum1DaaLineOneFP32Reg, maskFull32);

            Cast<float, betaType, ctHalf2Fp32Zero>(dgFP32ZeroReg, dgInReg, maskFull16);
            Cast<float, betaType, ctHalf2Fp32One>(dgFP32OneReg, dgInReg, maskFull16);
            Add(dgFP32AddZeroReg, dgFP32AddZeroReg, dgFP32ZeroReg, maskFull32);
            Add(dgFP32AddOneReg, dgFP32AddOneReg, dgFP32OneReg, maskFull32);

            Cast<betaType, float, ctFp322HalfOne>(dgOutReg, dgFP32AddOneReg, maskFull32);
            Cast<betaType, float, ctFp322HalfZero>(dgOutReg, dgFP32AddZeroReg, maskFull32);
            UnalignRegForStore uStoreDg;
            auto actDgOut = dgOut + eleOffset;
            uint32_t copyNum = min(eleNumPerVf, calcSize - eleOffset);
            StoreUnAlign(actDgOut, dgOutReg, uStoreDg, copyNum);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        } else {
            LoadAlign(sum1DaaLineZeroFP32Reg, reducesum1 + eleOffset);
            LoadAlign(dgFP32AddZeroReg, reducesum0 + eleOffset);
            LoadAlign(dgFP32ZeroReg, dgIn + eleOffset);
            Sub(dgFP32AddZeroReg, dgFP32AddZeroReg, sum1DaaLineZeroFP32Reg, maskFull32);
            Add(dgFP32AddZeroReg, dgFP32AddZeroReg, dgFP32ZeroReg, maskFull32);

            UnalignRegForStore uStoreDg;
            auto actDgOut = dgOut + eleOffset;
            uint32_t copyNum = min(eleNumPerVf, calcSize - eleOffset);
            StoreUnAlign(actDgOut, dgFP32AddZeroReg, uStoreDg, copyNum);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        }
    }
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKKTA5()
{
    Arch::Resource<Arch::Ascend950> resource;
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = kktVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    uint32_t ubOffset = 0;
    uint32_t ubListId = 0;
    uint32_t cvListId = 0;
    uint32_t ubBetaDgListId = 0;
    int32_t eventVMTE2 = 0;
    int32_t eventMTE2V = 0;
    int32_t eventMTE3V = 0;
    int32_t eventVMTE3 = 0;

    AscendC::LocalTensor<kType> tensorKKTInList[MAX_CUBE_VEC_SYNC_NUM];
    AscendC::LocalTensor<betaType> tensorBetaInList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDgInList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorDaInList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDgOutList[UB_STAGES];

    int32_t eventUbBetaInVMTE2List[UB_STAGES];
    int32_t eventUbBetaInMTE2VList[UB_STAGES];
    int32_t eventUbDgInVMTE2List[UB_STAGES];
    int32_t eventUbDgInMTE2VList[UB_STAGES];
    int32_t eventUbDaInVMTE2List[UB_STAGES];
    int32_t eventUbDaInMTE2VList[UB_STAGES];
    int32_t eventUbOutMTE3VList[UB_STAGES];
    int32_t eventUbOutVMTE3List[UB_STAGES];

    // CV buffers: cube core writes kktIn
    for (uint32_t i = 0; i < kktCVNum; ++i) {
        tensorKKTInList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * chunkSize * sizeof(kType);
    }

    // UB stage buffers
    for (uint32_t i = 0; i < UB_STAGES; i++) {
        tensorBetaInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += chunkSize * sizeof(betaType);
        tensorDgInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += chunkSize * sizeof(betaType);
        tensorDaInList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * chunkSize * sizeof(kType);
        tensorDgOutList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += chunkSize * sizeof(betaType);

        eventUbBetaInVMTE2List[i] = eventVMTE2++;
        eventUbBetaInMTE2VList[i] = eventMTE2V++;
        eventUbDgInVMTE2List[i] = eventVMTE2++;
        eventUbDgInMTE2VList[i] = eventMTE2V++;
        eventUbDaInVMTE2List[i] = eventVMTE2++;
        eventUbDaInMTE2VList[i] = eventMTE2V++;
        eventUbOutMTE3VList[i] = eventMTE3V++;
        eventUbOutVMTE3List[i] = eventVMTE3++;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBetaInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbDgInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbDaInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }
    auto tensorReducesum1 = resource.ubBuf.template GetBufferByByte<float32_t>(ubOffset);
    ubOffset += chunkSize * sizeof(float32_t);
    auto tensorReducesum0 = resource.ubBuf.template GetBufferByByte<float32_t>(ubOffset);
    ubOffset += chunkSize * sizeof(float32_t);

    for (uint32_t i = 0; i < kktCVNum; i++) {
        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + i);
    }

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (uint64_t h_k = 0; h_k < HK; h_k++) {
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                continue;
            }
            uint32_t tmpCvListId = cvListId;
            for (uint64_t i = 0; i < groupSize; i++) {
                uint64_t h_v = h_k * groupSize + i;
                auto& tensorBeta = tensorBetaInList[ubBetaDgListId];
                auto& tensorDg = tensorDgInList[ubBetaDgListId];
                {// copyin beta/dg
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBetaInVMTE2List[ubBetaDgListId]);
                    DataCopyPad(tensorBeta, betaTensor[h_v * T + bos], {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbBetaInMTE2VList[ubBetaDgListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbDgInVMTE2List[ubBetaDgListId]);
                    DataCopyPad(tensorDg, dgTensor[h_v * T + bos], {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbDgInMTE2VList[ubBetaDgListId]);
                }
                Duplicate(tensorReducesum0, 0.0f, chunkSize);
                tmpCvListId = cvListId;
                for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                    auto& tensordaIn = tensorDaInList[ubListId];
                    auto& tensorKKTin = tensorKKTInList[tmpCvListId];
                    // copyin da
                    {
                        auto dAOffset = (h_v * T + bos + rowOffset) * chunkSize;
                        curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbDaInVMTE2List[ubListId]);
                        DataCopy(tensordaIn, dATensor[dAOffset], chunkSize * curRowNum);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbDaInMTE2VList[ubListId]);
                    }
                    // compute
                    {
                        auto betaInAddr = reinterpret_cast<uint64_t>(tensorBeta.GetPhyAddr());
                        auto daInAddr = reinterpret_cast<uint64_t>(tensordaIn.GetPhyAddr());
                        auto kKTinAddr = reinterpret_cast<uint64_t>(tensorKKTin.GetPhyAddr());
                        auto reducesum1Addr = reinterpret_cast<uint64_t>(tensorReducesum1[rowOffset].GetPhyAddr());
                        auto reducesum0Addr = reinterpret_cast<uint64_t>(tensorReducesum0.GetPhyAddr());

                        if (rowOffset == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbBetaInMTE2VList[ubBetaDgListId]);
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbDaInMTE2VList[ubListId]);
                        if (i == 0) {
                            AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + tmpCvListId);
                        }
                        ProcessKKTComputerVF((__ubuf__ kType*)kKTinAddr, (__ubuf__ betaType*)betaInAddr,
                            (__ubuf__ kType*)daInAddr, (__ubuf__ float*)reducesum1Addr, (__ubuf__ float*)reducesum0Addr,
                            curChunkSize, curRowNum, chunkSize);
                        if (i == groupSize - 1) {
                            AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + tmpCvListId);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbDaInVMTE2List[ubListId]);
                        if (rowOffset + rowNum >= curChunkSize) {
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBetaInVMTE2List[ubBetaDgListId]);
                        }
                    }
                    ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
                    tmpCvListId = (tmpCvListId + 1 < kktCVNum) ? (tmpCvListId + 1) : 0;
                }
                auto& tensorOutDg = tensorDgOutList[ubBetaDgListId];
                auto dgInAddr = reinterpret_cast<uint64_t>(tensorDg.GetPhyAddr());
                auto dgOutAddr = reinterpret_cast<uint64_t>(tensorOutDg.GetPhyAddr());
                auto reducesum1Addr = reinterpret_cast<uint64_t>(tensorReducesum1[rowOffset].GetPhyAddr());
                auto reducesum0Addr = reinterpret_cast<uint64_t>(tensorReducesum0.GetPhyAddr());
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbDgInMTE2VList[ubBetaDgListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubBetaDgListId]);
                ProcessKKTComputerVFSub((__ubuf__ betaType*)dgOutAddr, (__ubuf__ betaType*)dgInAddr,
                            (__ubuf__ float*)reducesum1Addr, (__ubuf__ float*)reducesum0Addr,
                            curChunkSize);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubBetaDgListId]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbDgInVMTE2List[ubBetaDgListId]);
                {//copyout
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubBetaDgListId]);
                    DataCopyPad(dgTensor[h_v * T + bos], tensorOutDg, {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubBetaDgListId]);
                }
                ubBetaDgListId = (ubBetaDgListId + 1 < UB_STAGES) ? (ubBetaDgListId + 1) : 0;
            }
            cvListId = tmpCvListId;
        }
    }
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbDaInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbDgInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBetaInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }

    return;
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDvbComputerVFMutiLineOneCol(
    __ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<kType> dvbInReg, dvbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg1, dvbFP32OneReg1;
    RegTensor<betaType> dbetaInReg;
    RegTensor<float> dbetaFP32Reg;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<kType> dvOutReg, dvOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);
    LoadIn<kType, false>(dvbInReg, dvbIn);
    LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);
        LoadIn<kType, false>(dvbInReg, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        // Row 0: dBeta = reduceSum(v * dvb) + dbeta
        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        // Row 1
        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg1, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM + 1);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg1, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM + 1);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(vInReg, vIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dvbInReg, dvbIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
            CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
            MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
            MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);

            Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
            ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
            LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx * PRELOAD_NUM);
            HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
            Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
            StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDvbComputerVFOneLineOneCol(
    __ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn, uint16_t mSize, uint16_t nSize)
{
    RegTensor<betaType> betaInReg;
    RegTensor<kType> vInReg;
    RegTensor<kType> dvbInReg;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg;
    RegTensor<float> dvbFP32ZeroReg, dvbFP32OneReg;
    RegTensor<betaType> dbetaInReg;
    RegTensor<float> dbetaFP32Reg;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<kType> dvOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(dvbInReg, dvbIn);
    HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
    CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
    MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
    MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
    CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
    StoreAlign((__ubuf__ kType*&)dvOut, dvOutReg, maskFull16);

    Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
    ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
    LoadIn<betaType, true>(dbetaInReg, dbetaIn);
    HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);
    Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
    StoreUnAlignOut<betaType>(dBetaOut, dBetaFP32Reg, maskFull32, uStore, 1);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDvbComputerVFTwoCol(
    __ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<betaType> betaInReg;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<kType> dvbInReg, dvbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg1, dvbFP32OneReg1;
    RegTensor<betaType> dbetaInReg;
    RegTensor<float> dbetaFP32Reg;
    RegTensor<float> dBetaFP32Reg, dBetaFP32Reg1;
    RegTensor<kType> dvOutReg, dvOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);
    LoadIn<kType, false>(dvbInReg, dvbIn);
    LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + mIdx + 1);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);
        LoadIn<kType, false>(dvbInReg, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dvbInReg1, dvbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);

        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, vFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg, dvbFP32OneReg, dvbInReg, maskFull16);
        CastHalf2Float<kType>(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbInReg1, maskFull16);

        MulFloatTwoReg(vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg, vFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg, dvbFP32OneReg, dvbFP32ZeroReg, dvbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(vFP32ZeroReg1, vFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dvbFP32ZeroReg1, dvbFP32OneReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

        CastFloat2Half<kType>(dvOutReg, dvbFP32ZeroReg, dvbFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dvOutReg1, dvbFP32ZeroReg1, dvbFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM), dvOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dvOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dvOutReg1, maskFull16);

        Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, vFP32ZeroReg, maskFull32);
        LoadIn<betaType, true>(dbetaInReg, dbetaIn + mIdx);
        HalfOrFloat2Float<betaType>(dbetaFP32Reg, dbetaInReg, maskFull16, maskFull32);

        Add(vFP32ZeroReg1, vFP32ZeroReg1, vFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, vFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dbetaFP32Reg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDvbA5()
{
    Arch::Resource<Arch::Ascend950> resource;
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dvbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t wholeReduceSumCnt = CeilDiv(V, FP32_PER_REPEAT_64);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    uint32_t ubOffset = 0;
    uint32_t ubListId = 0;
    uint32_t cvListId = 0;
    int32_t eventVMTE2 = 0;
    int32_t eventMTE2V = 0;
    int32_t eventMTE3V = 0;
    int32_t eventVMTE3 = 0;
    AscendC::LocalTensor<kType> tensorDvbinList[MAX_CUBE_VEC_SYNC_NUM];
    AscendC::LocalTensor<kType> tensorVinList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorBetainList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDbetaInList[UB_STAGES];

    AscendC::LocalTensor<kType> tensorDvOutList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDbetaOutList[UB_STAGES];

    int32_t eventUbInVMTE2List[UB_STAGES];
    int32_t eventUbInMTE2VList[UB_STAGES];
    int32_t eventUbOutMTE3VList[UB_STAGES];
    int32_t eventUbOutVMTE3List[UB_STAGES];

    // init
    for (uint32_t i = 0; i < dvbCVNum; ++i) {
        tensorDvbinList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * V * sizeof(kType);
    }
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        tensorVinList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * V * sizeof(kType);
        tensorBetainList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);
        tensorDbetaInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);

        tensorDvOutList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * V * sizeof(kType);
        tensorDbetaOutList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);

        eventUbInVMTE2List[i] = eventVMTE2++;
        eventUbInMTE2VList[i] = eventMTE2V++;
        eventUbOutMTE3VList[i] = eventMTE3V++;
        eventUbOutVMTE3List[i] = eventVMTE3++;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }

    for (uint32_t i = 0; i < dvbCVNum; i++) {
        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + i);
    }
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (uint64_t h_v = 0; h_v < HV; h_v++) {
            uint32_t taskNum = CeilDiv(curChunkSize, rowNum);
            taskNum = CeilDiv(taskNum, GetSubBlockNum());
            uint32_t dealTaskNum = 0;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h_v * T + bos + rowOffset) * V;
                auto betaOffset = h_v * T + bos + rowOffset;
                
                auto& tensorDvbin = tensorDvbinList[cvListId];
                auto& tensorVin = tensorVinList[ubListId];
                auto& tensorBetain = tensorBetainList[ubListId];
                auto& tensorDbetaIn = tensorDbetaInList[ubListId];
                auto& tensorDvOut = tensorDvOutList[ubListId];
                auto& tensorDbetaOut = tensorDbetaOutList[ubListId];
                // copyin
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                    DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopyPad(tensorDbetaIn, dbetaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                }
                // compute
                {
                    auto dvbInAddr = reinterpret_cast<uint64_t>(tensorDvbin.GetPhyAddr());
                    auto vInAddr = reinterpret_cast<uint64_t>(tensorVin.GetPhyAddr());
                    auto betaInAddr = reinterpret_cast<uint64_t>(tensorBetain.GetPhyAddr());
                    auto dbetaInAddr = reinterpret_cast<uint64_t>(tensorDbetaIn.GetPhyAddr());
                    auto dBetaOutAddr = reinterpret_cast<uint64_t>(tensorDbetaOut.GetPhyAddr());
                    auto dvOutAddr = reinterpret_cast<uint64_t>(tensorDvOut.GetPhyAddr());
                    
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
                    if (V <= eleKNumPerVf) {
                        if (curRowNum == 1) {
                            ProcessDvbComputerVFOneLineOneCol(
                                    (__ubuf__ kType*)dvOutAddr, (__ubuf__ betaType*)dBetaOutAddr,
                                    (__ubuf__ kType*)dvbInAddr, (__ubuf__ kType*)vInAddr,
                                    (__ubuf__ betaType*)betaInAddr, (__ubuf__ betaType*)dbetaInAddr,
                                    curRowNum, V);
                        } else {
                            ProcessDvbComputerVFMutiLineOneCol(
                                    (__ubuf__ kType*)dvOutAddr, (__ubuf__ betaType*)dBetaOutAddr,
                                    (__ubuf__ kType*)dvbInAddr, (__ubuf__ kType*)vInAddr,
                                    (__ubuf__ betaType*)betaInAddr, (__ubuf__ betaType*)dbetaInAddr,
                                    curRowNum, V);
                        }
                    } else {
                        ProcessDvbComputerVFTwoCol(
                                    (__ubuf__ kType*)dvOutAddr, (__ubuf__ betaType*)dBetaOutAddr,
                                    (__ubuf__ kType*)dvbInAddr, (__ubuf__ kType*)vInAddr,
                                    (__ubuf__ betaType*)betaInAddr, (__ubuf__ betaType*)dbetaInAddr,
                                    curRowNum, V);
                    }
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                }
                // copyout
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    DataCopy(dvTensor[vOffset], tensorDvOut, V * curRowNum);
                    DataCopyPad(dbetaTensor[betaOffset], tensorDbetaOut, {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                }
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
                cvListId = (cvListId + 1 < dvbCVNum) ? (cvListId + 1) : 0;
                dealTaskNum++;
            }
            for (uint32_t taskId = dealTaskNum; taskId < taskNum; taskId++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                cvListId = (cvListId + 1 < dvbCVNum) ? (cvListId + 1) : 0;
            }
        }
    }
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }
    return;
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbgComputerGatherDkVF(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ betaType* dgOut,
    __ubuf__ kType* kIn, __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkGatherIn, __ubuf__ betaType* dBetaIn, __ubuf__ kType* dkbgIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<kType> dkInReg, dkGatherInReg;
    RegTensor<betaType> dbetaInReg;
    RegTensor<kType> dkbgInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, dgFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, dBetaAddZeroReg, dkbgFP32ZeroReg, dkbgFP32OneReg;
    RegTensor<float> gBrcbFP32ZeroReg, dbetaFP32ZeroReg, dbetaFP32OneReg;
    RegTensor<float> kReduceSumReg, dkReduceSumReg;
    RegTensor<kType> dkOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStoreDBeta;
    UnalignRegForStore uStoreDg;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(dbetaInReg, dBetaIn);
    LoadIn<kType>(kInReg, kIn);
    LoadIn<kType>(dkbgInReg, dkbgIn);
    LoadIn<kType>(dkInReg, dkIn);
    LoadIn<kType>(dkGatherInReg, dkGatherIn);
    // 前mSize - 1行
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1));
        LoadIn<betaType, true>(dbetaInReg, dBetaIn + (mIdx + 1));

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            CastHalf2Float<kType>(dkGatherFP32ZeroReg, dkGatherFP32OneReg, dkGatherInReg, maskFull16);
            LoadIn<kType>(dkGatherInReg, dkGatherIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
    // 最后一行
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        // 最后一行的前nLoopCnt - 1
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            CastHalf2Float<kType>(dkGatherFP32ZeroReg, dkGatherFP32OneReg, dkGatherInReg, maskFull16);
            LoadIn<kType>(dkGatherInReg, dkGatherIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        // 最后一行最后一个loop
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            CastHalf2Float<kType>(dkGatherFP32ZeroReg, dkGatherFP32OneReg, dkGatherInReg, maskFull16);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkGatherFP32ZeroReg, dkGatherFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbgComputerVF(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ betaType* dgOut,
    __ubuf__ kType* kIn, __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
    __ubuf__ kType* dkIn, __ubuf__ betaType* dBetaIn, __ubuf__ kType* dkbgIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<kType> dkInReg;
    RegTensor<betaType> dbetaInReg;
    RegTensor<kType> dkbgInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, dgFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, dBetaAddZeroReg, dkbgFP32ZeroReg, dkbgFP32OneReg;
    RegTensor<float> gBrcbFP32ZeroReg, dbetaFP32ZeroReg, dbetaFP32OneReg;
    RegTensor<float> kReduceSumReg, dkReduceSumReg;
    RegTensor<kType> dkOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStoreDBeta;
    UnalignRegForStore uStoreDg;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(dbetaInReg, dBetaIn);
    LoadIn<kType>(kInReg, kIn);
    LoadIn<kType>(dkbgInReg, dkbgIn);
    LoadIn<kType>(dkInReg, dkIn);
    // 前mSize - 1行
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1));
        LoadIn<betaType, true>(dbetaInReg, dBetaIn + (mIdx + 1));

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
    // 最后一行
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gBrcbFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(dbetaFP32ZeroReg, dbetaInReg, maskFull16, maskFull32);

        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        // 最后一行的前nLoopCnt - 1
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);
            LoadIn<kType>(kInReg, kIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            LoadIn<kType>(dkbgInReg, dkbgIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            LoadIn<kType>(dkInReg, dkIn + mIdx * nSize + (vfBlockIdx + 1) * eleKNumPerVf);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        // 最后一行最后一个loop
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgInReg, maskFull16);

            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            MulFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            MulFloatTwoReg(dkbgFP32ZeroReg, dkbgFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);

            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbgFP32ZeroReg, dkbgFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + mIdx * nSize + vfBlockIdx * eleKNumPerVf, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaAddZeroReg, maskFull32, uStoreDBeta, 1);
        StoreUnAlignOut<betaType>(dgOut + mIdx, dgFP32ZeroReg, maskFull32, uStoreDg, 1);
    }
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbgA5()
{
    Arch::Resource<Arch::Ascend950> resource;
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbgVecRow;
    uint32_t vecTaskIdx = 0;
    uint32_t wholeReduceSumCnt = CeilDiv(K, FP32_PER_REPEAT_64);
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint64_t kBos = 0;
    uint32_t curRowNum = rowNum;

    uint32_t ubOffset = 0;
    uint32_t ubListId = 0;
    uint32_t cvListId = 0;
    AscendC::LocalTensor<kType> tensorDkbgInList[MAX_CUBE_VEC_SYNC_NUM];
    AscendC::LocalTensor<kType> tensorKinList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorBetaInList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorGInList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorWorkspaceDkInList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDbetaInList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorDkOutList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDbetaOutList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDgOutList[UB_STAGES];

    int32_t eventVMTE2 = 0;
    int32_t eventMTE2V = 0;
    int32_t eventMTE3V = 0;
    int32_t eventVMTE3 = 0;
    int32_t eventUbInVMTE2List[UB_STAGES];
    int32_t eventUbInMTE2VList[UB_STAGES];
    int32_t eventUbOutMTE3VList[UB_STAGES];
    int32_t eventUbOutVMTE3List[UB_STAGES];

    for (uint32_t i = 0; i < dkbgCVNum; ++i) {
        tensorDkbgInList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
    }
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        tensorKinList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
        tensorBetaInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);
        tensorGInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);
        tensorWorkspaceDkInList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
        tensorDbetaInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);

        tensorDkOutList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
        tensorDbetaOutList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);
        tensorDgOutList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);

        eventUbInVMTE2List[i] = eventVMTE2++;
        eventUbInMTE2VList[i] = eventMTE2V++;
        eventUbOutMTE3VList[i] = eventMTE3V++;
        eventUbOutVMTE3List[i] = eventVMTE3++;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }

    for (uint32_t i = 0; i < dkbgCVNum; i++) {
        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + i);
    }

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        kBos = bos;
        if (cu_seqlens == nullptr && HV != HK) {
            GetKBosByVBos(bos, T, HV, HK, kBos);
        }
        uint32_t curChunkSize = eos - bos;
        for (uint64_t h_v = 0; h_v < HV; h_v++) {
            uint32_t taskNum = CeilDiv(curChunkSize, rowNum);
            taskNum = CeilDiv(taskNum, GetSubBlockNum());
            uint32_t dealTaskNum = 0;

            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                uint64_t h_k = h_v / groupSize;
                uint64_t needGatherDk = h_v % groupSize;
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h_k * T + kBos + rowOffset) * K;
                auto betaOffset = h_v * T + bos + rowOffset;
                auto dkInOffset = (h_v * T + bos + rowOffset) * K;
                auto dkOutOffset = (h_k * T + kBos + rowOffset) * K;

                auto& tensorDkbgIn = tensorDkbgInList[cvListId];
                auto& tensorKin = tensorKinList[ubListId];
                auto& tensorBetaIn = tensorBetaInList[ubListId];
                auto& tensorGIn = tensorGInList[ubListId];
                auto& tensorWorkspaceDkIn = tensorWorkspaceDkInList[ubListId];
                auto& tensorDbetaIn = tensorDbetaInList[ubListId];
                auto& tensorDkOut = tensorDkOutList[ubListId];
                auto& tensorDbetaOut = tensorDbetaOutList[ubListId];
                auto& tensorDgOut = tensorDgOutList[ubListId];

                // copyin
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                    DataCopy(tensorKin, kTensor[kOffset], K * curRowNum);
                    DataCopyPad(tensorBetaIn, betaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopyPad(tensorGIn, gTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopy(tensorWorkspaceDkIn, workSpaceDkTensor[dkInOffset], K * curRowNum);
                    DataCopyPad(tensorDbetaIn, dbetaTensor[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                                {false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                }
                // compute
                {
                    auto dkbgInAddr = reinterpret_cast<uint64_t>(tensorDkbgIn.GetPhyAddr());
                    auto kInAddr = reinterpret_cast<uint64_t>(tensorKin.GetPhyAddr());
                    auto betaInAddr = reinterpret_cast<uint64_t>(tensorBetaIn.GetPhyAddr());
                    auto gInAddr = reinterpret_cast<uint64_t>(tensorGIn.GetPhyAddr());
                    auto dkWorkspaceInAddr = reinterpret_cast<uint64_t>(tensorWorkspaceDkIn.GetPhyAddr());
                    auto dBetaInAddr = reinterpret_cast<uint64_t>(tensorDbetaIn.GetPhyAddr());
                    auto dkOutAddr = reinterpret_cast<uint64_t>(tensorDkOut.GetPhyAddr());
                    auto dBetaOutAddr = reinterpret_cast<uint64_t>(tensorDbetaOut.GetPhyAddr());
                    auto dgOutAddr = reinterpret_cast<uint64_t>(tensorDgOut.GetPhyAddr());

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                    ProcessDkbgComputerVF(
                        (__ubuf__ kType*)dkOutAddr,
                        (__ubuf__ betaType*)dBetaOutAddr,
                        (__ubuf__ betaType*)dgOutAddr,
                        (__ubuf__ kType*)kInAddr,
                        (__ubuf__ betaType*)betaInAddr,
                        (__ubuf__ betaType*)gInAddr,
                        (__ubuf__ kType*)dkWorkspaceInAddr,
                        (__ubuf__ betaType*)dBetaInAddr,
                        (__ubuf__ kType*)dkbgInAddr,
                        curRowNum, K);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                }
                // copyout
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    if (needGatherDk) {
                        DataCopy(workSpaceDkTensor[dkInOffset], tensorDkOut, K * curRowNum);
                    } else {
                        DataCopy(dkTensor[dkOutOffset], tensorDkOut, K * curRowNum);
                    }
                    DataCopyPad(dbetaTensor[betaOffset], tensorDbetaOut,
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    DataCopyPad(dgTensor[betaOffset], tensorDgOut,
                                {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                }
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
                cvListId = (cvListId + 1 < dkbgCVNum) ? (cvListId + 1) : 0;
                dealTaskNum++;
            }

            for (uint32_t taskId = dealTaskNum; taskId < taskNum; taskId++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                cvListId = (cvListId + 1 < dkbgCVNum) ? (cvListId + 1) : 0;
            }
        }
    }
    
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkGatherA5()
{
    Arch::Resource<Arch::Ascend950> resource;
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkGatherVecRow;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint64_t kBos = 0;
    uint32_t curRowNum = rowNum;

    uint32_t ubOffset = 0;
    uint32_t ubListId = 0;
    uint32_t dkOutUbListId = 0;
    AscendC::LocalTensor<kType> tensorWsDkInList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorDkGatherInList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorDkOutList[UB_STAGES];

    int32_t eventVMTE2 = 0;
    int32_t eventMTE2V = 0;
    int32_t eventMTE3V = 0;
    int32_t eventVMTE3 = 0;
    int32_t eventUbInVMTE2List[UB_STAGES];
    int32_t eventUbInMTE2VList[UB_STAGES];
    int32_t eventUbOutMTE3VList[UB_STAGES];
    int32_t eventUbOutVMTE3List[UB_STAGES];

    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        tensorWsDkInList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
        tensorDkGatherInList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
        tensorDkOutList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);

        eventUbInVMTE2List[i] = eventVMTE2++;
        eventUbInMTE2VList[i] = eventMTE2V++;
        eventUbOutMTE3VList[i] = eventMTE3V++;
        eventUbOutVMTE3List[i] = eventVMTE3++;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        kBos = bos;
        if (cu_seqlens == nullptr && HV != HK) {
            GetKBosByVBos(bos, T, HV, HK, kBos);
        }
        uint32_t curChunkSize = eos - bos;
        for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                continue;
            }
            for (uint64_t h_v = 0; h_v < HV; h_v++) {
                uint64_t h_k = h_v / groupSize;
                uint64_t needGatherDk = h_v % groupSize;
                if (!needGatherDk) {
                    continue;
                }

                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto dkInOffset = (h_v * T + bos + rowOffset) * K;
                auto dkOutOffset = (h_k * T + kBos + rowOffset) * K;

                auto& tensorWsDkIn = tensorWsDkInList[ubListId];
                auto& tensorDkGatherIn = tensorDkGatherInList[dkOutUbListId];
                auto& tensorDkOut = tensorDkOutList[dkOutUbListId];

                // copyin
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                    DataCopy(tensorWsDkIn, workSpaceDkTensor[dkInOffset], K * curRowNum);
                    if (needGatherDk == 1) {
                        DataCopy(tensorDkGatherIn, dkTensor[dkOutOffset], K * curRowNum);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                }
                // compute
                {
                    auto dkOutAddr = reinterpret_cast<uint64_t>(tensorDkOut.GetPhyAddr());
                    auto wsDkInAddr = reinterpret_cast<uint64_t>(tensorWsDkIn.GetPhyAddr());
                    auto dkGatherInAddr = reinterpret_cast<uint64_t>(tensorDkGatherIn.GetPhyAddr());
                    if (needGatherDk != 1) {
                        dkGatherInAddr = dkOutAddr;
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                    if (needGatherDk == 1) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[dkOutUbListId]);
                    }
                    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
                    if (K <= eleKNumPerVf) {
                        if (curRowNum == 1) {
                            ProcessDkGatherComputerVFOneLineOneCol(
                                (__ubuf__ kType*)dkOutAddr,
                                (__ubuf__ kType*)wsDkInAddr, (__ubuf__ kType*)dkGatherInAddr,
                                curRowNum, K);
                        } else {
                            ProcessDkGatherComputerVFMutiLineOneCol(
                                (__ubuf__ kType*)dkOutAddr,
                                (__ubuf__ kType*)wsDkInAddr, (__ubuf__ kType*)dkGatherInAddr,
                                curRowNum, K);
                        }
                    } else {
                        ProcessDkGatherComputerVFTwoCol(
                            (__ubuf__ kType*)dkOutAddr,
                            (__ubuf__ kType*)wsDkInAddr, (__ubuf__ kType*)dkGatherInAddr,
                            curRowNum, K);
                    }
                    if (needGatherDk == groupSize - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[dkOutUbListId]);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                }
                // copyout
                {
                    if (needGatherDk == groupSize - 1) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[dkOutUbListId]);
                        DataCopy(dkTensor[dkOutOffset], tensorDkOut, K * curRowNum);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[dkOutUbListId]);
                    }
                }
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
                if (needGatherDk == groupSize - 1) {
                    dkOutUbListId = (dkOutUbListId + 1 < UB_STAGES) ? (dkOutUbListId + 1) : 0;
                }
            }
        }
    }

    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }
    return;
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkGatherComputerVFOneLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
    uint16_t mSize, uint16_t nSize)
{
    RegTensor<kType> dk1Reg, dk2Reg;
    RegTensor<float> dk1FP32ZeroReg, dk1FP32OneReg;
    RegTensor<float> dk2FP32ZeroReg, dk2FP32OneReg;
    RegTensor<kType> dkOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<kType, false>(dk1Reg, dkIn1);
    LoadIn<kType, false>(dk2Reg, dkIn2);
    CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
    CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
    AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
    CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
    StoreAlign((__ubuf__ kType*&)dkOut, dkOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkGatherComputerVFMutiLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<kType> dk1Reg, dk1Reg1;
    RegTensor<kType> dk2Reg, dk2Reg1;
    RegTensor<float> dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg1, dk1FP32OneReg1;
    RegTensor<float> dk2FP32ZeroReg, dk2FP32OneReg, dk2FP32ZeroReg1, dk2FP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<kType, false>(dk1Reg, dkIn1);
    LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum);
    LoadIn<kType, false>(dk2Reg, dkIn2);
    LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);
        LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * (mIdx * PRELOAD_NUM));
            CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
            CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
            AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkGatherComputerVFTwoCol(
    __ubuf__ kType* dkOut, __ubuf__ kType* dkIn1, __ubuf__ kType* dkIn2,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<kType> dk1Reg, dk1Reg1;
    RegTensor<kType> dk2Reg, dk2Reg1;
    RegTensor<float> dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg1, dk1FP32OneReg1;
    RegTensor<float> dk2FP32ZeroReg, dk2FP32OneReg, dk2FP32ZeroReg1, dk2FP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<kType, false>(dk1Reg, dkIn1);
    LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum);
    LoadIn<kType, false>(dk2Reg, dkIn2);
    LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);
        LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
        CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
        CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);

        AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
        AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(dk1Reg, dkIn1 + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dk1Reg1, dkIn1 + oneEleNum * (mIdx * PRELOAD_NUM + 1));
            LoadIn<kType, false>(dk2Reg, dkIn2 + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dk2Reg1, dkIn2 + oneEleNum * (mIdx * PRELOAD_NUM + 1));
            CastHalf2Float<kType>(dk1FP32ZeroReg, dk1FP32OneReg, dk1Reg, maskFull16);
            CastHalf2Float<kType>(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1Reg1, maskFull16);
            CastHalf2Float<kType>(dk2FP32ZeroReg, dk2FP32OneReg, dk2Reg, maskFull16);
            CastHalf2Float<kType>(dk2FP32ZeroReg1, dk2FP32OneReg1, dk2Reg1, maskFull16);
            AddFloatTwoReg(dk1FP32ZeroReg, dk1FP32OneReg, dk1FP32ZeroReg, dk1FP32OneReg, dk2FP32ZeroReg, dk2FP32OneReg, maskFull32);
            AddFloatTwoReg(dk1FP32ZeroReg1, dk1FP32OneReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, dk2FP32ZeroReg1, dk2FP32OneReg1, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dk1FP32ZeroReg, dk1FP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg1, dk1FP32ZeroReg1, dk1FP32OneReg1, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbComputerVFMutiLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<kType> dkInReg, dkInReg1;
    RegTensor<kType> dkbInReg, dkbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg1, dkFP32OneReg1;
    RegTensor<float> dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg1, dkbFP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<betaType> dBetaOutZeroReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    LoadIn<kType, false>(dkInReg, dkIn);
    LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum);
    LoadIn<kType, false>(dkbInReg, dkbIn);
    LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        LoadIn<kType, false>(dkInReg, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);
        LoadIn<kType, false>(dkbInReg, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM + 1, dBetaFP32Reg, maskFull32, uStore, 1);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dkInReg, dkIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(dkbInReg, dkbIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
            CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
            MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
            MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
            CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);

            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
            StoreUnAlignOut<betaType>(dBetaOut + mIdx * PRELOAD_NUM, dBetaFP32Reg, maskFull32, uStore, 1);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbComputerVFOneLineOneCol(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, uint16_t mSize, uint16_t nSize)
{
    RegTensor<betaType> betaInReg;
    RegTensor<kType> kInReg;
    RegTensor<kType> dkInReg;
    RegTensor<kType> dkbInReg;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg;
    RegTensor<float> dkFP32ZeroReg, dkFP32OneReg;
    RegTensor<float> dkbFP32ZeroReg, dkbFP32OneReg;
    RegTensor<kType> dkOutReg;
    RegTensor<float> dBetaFP32Reg;
    RegTensor<betaType> dBetaOutZeroReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(dkInReg, dkIn);
    LoadIn<kType, false>(dkbInReg, dkbIn);
    HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
    CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
    CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
    MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
    MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
    AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
    CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
    StoreAlign((__ubuf__ kType*&)dkOut, dkOutReg, maskFull16);

    Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
    ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);
    StoreUnAlignOut<betaType>(dBetaOut, dBetaFP32Reg, maskFull32, uStore, 1);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbComputerVFTwoCol(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize - 1;
    uint16_t lastLoopCnt = mSize % PRELOAD_NUM;
    RegTensor<betaType> betaInReg;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<kType> dkInReg, dkInReg1;
    RegTensor<kType> dkbInReg, dkbInReg1;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg1, dkFP32OneReg1;
    RegTensor<float> dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg1, dkbFP32OneReg1;
    RegTensor<kType> dkOutReg, dkOutReg1;
    RegTensor<float> dBetaFP32Reg, dBetaFP32Reg1;
    RegTensor<betaType> dBetaOutZeroReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForStore uStore;

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    LoadIn<kType, false>(dkInReg, dkIn);
    LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum);
    LoadIn<kType, false>(dkbInReg, dkbIn);
    LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + mIdx + 1);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        LoadIn<kType, false>(dkInReg, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkInReg1, dkIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);
        LoadIn<kType, false>(dkbInReg, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dkbInReg1, dkbIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, kFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg, dkFP32OneReg, dkInReg, maskFull16);
        CastHalf2Float<kType>(dkFP32ZeroReg1, dkFP32OneReg1, dkInReg1, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg, dkbFP32OneReg, dkbInReg, maskFull16);
        CastHalf2Float<kType>(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbInReg1, maskFull16);

        MulFloatTwoReg(kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg, kFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg, dkbFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg, dkFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, maskFull32);

        MulFloatTwoReg(kFP32ZeroReg1, kFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);
        MulFloatTwoReg(dkbFP32ZeroReg1, dkbFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        AddFloatTwoReg(dkFP32ZeroReg1, dkFP32OneReg1, dkFP32ZeroReg1, dkFP32OneReg1, dkbFP32ZeroReg1, dkbFP32OneReg1, maskFull32);

        CastFloat2Half<kType>(dkOutReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dkOutReg1, dkFP32ZeroReg1, dkFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM), dkOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)dkOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dkOutReg1, maskFull16);

        Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
        ReduceSum(dBetaFP32Reg, kFP32ZeroReg, maskFull32);

        Add(kFP32ZeroReg1, kFP32ZeroReg1, kFP32OneReg1, maskFull32);
        ReduceSum(dBetaFP32Reg1, kFP32ZeroReg1, maskFull32);
        Add(dBetaFP32Reg, dBetaFP32Reg, dBetaFP32Reg1, maskFull32);
        StoreUnAlignOut<betaType>(dBetaOut + mIdx, dBetaFP32Reg, maskFull32, uStore, 1);
    }
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbA5()
{
    Arch::Resource<Arch::Ascend950> resource;
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = dkbVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint64_t kBos = 0;
    uint32_t curRowNum = rowNum;

    uint32_t ubOffset = 0;
    uint32_t ubListId = 0;
    uint32_t cvListId = 0;
    int32_t eventVMTE2 = 0;
    int32_t eventMTE2V = 0;
    int32_t eventMTE3V = 0;
    int32_t eventVMTE3 = 0;
    AscendC::LocalTensor<kType> tensorDkbinList[MAX_CUBE_VEC_SYNC_NUM];
    AscendC::LocalTensor<kType> tensorKinList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorBetainList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorDkinList[UB_STAGES];

    AscendC::LocalTensor<kType> tensorDkOutList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorDbetaOutList[UB_STAGES];

    int32_t eventUbInVMTE2List[UB_STAGES];
    int32_t eventUbInMTE2VList[UB_STAGES];
    int32_t eventUbOutMTE3VList[UB_STAGES];
    int32_t eventUbOutVMTE3List[UB_STAGES];

    // init
    for (uint32_t i = 0; i < dkbCVNum; ++i) {
        tensorDkbinList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
    }
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        tensorKinList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
        tensorBetainList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);
        tensorDkinList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);

        tensorDkOutList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * K * sizeof(kType);
        tensorDbetaOutList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);

        eventUbInVMTE2List[i] = eventVMTE2++;
        eventUbInMTE2VList[i] = eventMTE2V++;
        eventUbOutMTE3VList[i] = eventMTE3V++;
        eventUbOutVMTE3List[i] = eventVMTE3++;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }

    for (uint32_t i = 0; i < dkbCVNum; i++) {
        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + i);
    }
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        kBos = bos;
        if (cu_seqlens == nullptr && HV != HK) {
            GetKBosByVBos(bos, T, HV, HK, kBos);
        }
        uint32_t curChunkSize = eos - bos;
        for (uint64_t h_v = 0; h_v < HV; h_v++) {
            uint32_t taskNum = CeilDiv(curChunkSize, rowNum);
            taskNum = CeilDiv(taskNum, GetSubBlockNum());
            uint32_t dealTaskNum = 0;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                uint64_t h_k = h_v / groupSize;
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h_k * T + kBos + rowOffset) * K;
                auto betaOffset = h_v * T + bos + rowOffset;
                auto dkOffset = (h_v * T + bos + rowOffset) * K;

                auto& tensorDkbin = tensorDkbinList[cvListId];
                auto& tensorKin = tensorKinList[ubListId];
                auto& tensorBetain = tensorBetainList[ubListId];
                auto& tensorDkin = tensorDkinList[ubListId];
                auto& tensorDkOut = tensorDkOutList[ubListId];
                auto& tensorDbetaOut = tensorDbetaOutList[ubListId];
                // copyin
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                    DataCopy(tensorKin, kTensor[kOffset], K * curRowNum);
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                    DataCopy(tensorDkin, workSpaceDkTensor[dkOffset], K * curRowNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                }
                // compute
                {
                    auto dkOutAddr = reinterpret_cast<uint64_t>(tensorDkOut.GetPhyAddr());
                    auto dBetaOutAddr = reinterpret_cast<uint64_t>(tensorDbetaOut.GetPhyAddr());
                    auto kInAddr = reinterpret_cast<uint64_t>(tensorKin.GetPhyAddr());
                    auto betaInAddr = reinterpret_cast<uint64_t>(tensorBetain.GetPhyAddr());
                    auto dkInAddr = reinterpret_cast<uint64_t>(tensorDkin.GetPhyAddr());
                    auto dkbInAddr = reinterpret_cast<uint64_t>(tensorDkbin.GetPhyAddr());
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
                    if (K <= eleKNumPerVf) {
                        if (curRowNum == 1) {
                            ProcessDkbComputerVFOneLineOneCol(
                                    (__ubuf__ kType*)dkOutAddr, (__ubuf__ betaType*)dBetaOutAddr, (__ubuf__ kType*)kInAddr,
                                    (__ubuf__ betaType*)betaInAddr,(__ubuf__ kType*)dkInAddr, (__ubuf__ kType*)dkbInAddr,
                                    curRowNum, K);
                        } else {
                            ProcessDkbComputerVFMutiLineOneCol(
                                    (__ubuf__ kType*)dkOutAddr, (__ubuf__ betaType*)dBetaOutAddr, (__ubuf__ kType*)kInAddr,
                                    (__ubuf__ betaType*)betaInAddr,(__ubuf__ kType*)dkInAddr, (__ubuf__ kType*)dkbInAddr,
                                    curRowNum, K);
                        }
                    } else {
                        ProcessDkbComputerVFTwoCol(
                                    (__ubuf__ kType*)dkOutAddr, (__ubuf__ betaType*)dBetaOutAddr, (__ubuf__ kType*)kInAddr,
                                    (__ubuf__ betaType*)betaInAddr,(__ubuf__ kType*)dkInAddr, (__ubuf__ kType*)dkbInAddr,
                                    curRowNum, K);
                    }
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                }
                // copyout
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    DataCopy(workSpaceDkTensor[dkOffset], tensorDkOut, K * curRowNum);
                    DataCopyPad(dbetaTensor[betaOffset], tensorDbetaOut, {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                }
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
                cvListId = (cvListId + 1 < dkbCVNum) ? (cvListId + 1) : 0;
                dealTaskNum++;
            }
            for (uint32_t taskId = dealTaskNum; taskId < taskNum; taskId++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                cvListId = (cvListId + 1 < dkbCVNum) ? (cvListId + 1) : 0;
            }
        }
    }
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
    }
    return;
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBetaComputerVFMutiLineOneCol(
    __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / 2 - 1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kBetaFP32ZeroReg, kBetaFP32OneReg;
    RegTensor<float> kFP32ZeroReg1, kFP32OneReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<kType> kBetaOutReg, kBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    // 前mSize - 3行
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * 2);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * 2 + 1);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * 2));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * 2 + 1));
        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    // 最后三行
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg1, betaInReg1, maskFull16, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);

        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg1, betaBrcbFP32ZeroReg1, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);

        mIdx += 1;
        // 最后一行
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * 2);
            LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx * 2));
            HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBetaComputerVFOneLineOneCol(
    __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kBetaFP32ZeroReg, kBetaFP32OneReg;
    RegTensor<float> betaBrcbFP32ZeroReg;
    RegTensor<kType> kBetaOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);
    
    HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
    MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
    CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
    StoreAlign(kBetaOut, kBetaOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBetaComputerVFTwoCol(
    __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kBetaFP32ZeroReg, kBetaFP32OneReg;
    RegTensor<float> kFP32ZeroReg1, kFP32OneReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg1;
    RegTensor<kType> kBetaOutReg, kBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx + 1) * 2);
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * (mIdx + 1) * 2 + 1);
        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mSize - 1;
    // 最后一行
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32ZeroReg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        MulFloatTwoReg(kBetaFP32ZeroReg, kBetaFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        MulFloatTwoReg(kBetaFP32ZeroReg1, kBetaFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg, kBetaFP32ZeroReg, kBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaOutReg1, kBetaFP32ZeroReg1, kBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2), kBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaOut + oneEleNum * (mIdx * 2 + 1), kBetaOutReg1, maskFull16);
    }
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBetaA5()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = kBeteVecRow;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint64_t kBos = 0;
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(kBetaOutQue, 2, rowNum * K * sizeof(kType));
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        kBos = bos;
        if (cu_seqlens == nullptr && HV != HK) {
            GetKBosByVBos(bos, T, HV, HK, kBos);
        }
        uint32_t curChunkSize = eos - bos;
        for (uint64_t h_v = 0; h_v < HV; h_v++) {
            AscendC::CrossCoreWaitFlag(SYNC_FLAG_5);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                uint64_t h_k = h_v / groupSize;
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h_k * T + kBos + rowOffset) * K;
                auto betaOffset = h_v * T + bos + rowOffset;
                auto kBetaOffset = (h_v * T + bos + rowOffset) * K;
                // copyin
                {
                    auto tensorKin = kInQue.AllocTensor<kType>();
                    DataCopy(tensorKin, kTensor[kOffset], K * curRowNum);
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopyPad(tensorBetain, betaTensor[betaOffset], {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});

                    kInQue.EnQue(tensorKin);
                    betaInQue.EnQue(tensorBetain);
                }
                // compute
                {
                    auto tensorKin = kInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorOut = kBetaOutQue.AllocTensor<kType>();
                    auto kBetaOutAddr = reinterpret_cast<uint64_t>(tensorOut.GetPhyAddr());
                    auto kInAddr = reinterpret_cast<uint64_t>(tensorKin.GetPhyAddr());
                    auto betaInAddr = reinterpret_cast<uint64_t>(tensorBetain.GetPhyAddr());
                    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
                    if (K <= eleKNumPerVf) {
                        uint16_t lastLoopCnt = curRowNum % PRELOAD_NUM;
                        if (curRowNum == 1) {
                            ProcessKBetaComputerVFOneLineOneCol(
                                    (__ubuf__ kType*)kBetaOutAddr, (__ubuf__ kType*)kInAddr, (__ubuf__ betaType*)betaInAddr,
                                    curRowNum, K);
                        } else {
                            ProcessKBetaComputerVFMutiLineOneCol(
                                    (__ubuf__ kType*)kBetaOutAddr, (__ubuf__ kType*)kInAddr, (__ubuf__ betaType*)betaInAddr,
                                    curRowNum, K, lastLoopCnt);
                        }
                    } else {
                        ProcessKBetaComputerVFTwoCol(
                                    (__ubuf__ kType*)kBetaOutAddr, (__ubuf__ kType*)kInAddr, (__ubuf__ betaType*)betaInAddr,
                                    curRowNum, K);
                    }
                    
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    kBetaOutQue.EnQue(tensorOut);
                }
                // copyout
                {
                    auto tensorOut = kBetaOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[kBetaOffset], tensorOut, K * curRowNum);
                    kBetaOutQue.FreeTensor(tensorOut);
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_FLAG_3);
        }
    }
    AscendC::CrossCoreWaitFlag(SYNC_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_FLAG_5);
    AscendC::CrossCoreWaitFlag(SYNC_FLAG_5);
    return;
}


#endif // PREPARE_WY_REPR_BWD_FULL_VECTOR_H
