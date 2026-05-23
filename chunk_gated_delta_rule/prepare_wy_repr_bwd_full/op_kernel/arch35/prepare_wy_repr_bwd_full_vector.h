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
    __aicore__ inline void ProcessDvbA5();
    __aicore__ inline void ProcessKKTA5();
    __aicore__ inline void Init(const PrepareWyReprBwdFullTilingDataA5 &tiling, AscendC::TPipe *pipe_);
    __simd_vf__ inline void ProcessKBetaComputerVF(__ubuf__ kType* kBetaOut,
                                                   __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                   uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkbComputerVF(__ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut,
                                                 __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                 __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn,
                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDkbgComputerVF(__ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, 
                                                  __ubuf__ betaType* dgOut, __ubuf__ kType* kIn, 
                                                  __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
                                                  __ubuf__ kType* dkIn, __ubuf__ betaType* dBetaIn,
                                                  __ubuf__ kType* dkbgIn, uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessDvbComputerVF(__ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut,
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
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t kBeteVecRow = 0;
    uint64_t dkbVecRow = 0;
    uint64_t dkbgVecRow = 0;
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
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    kBeteVecRow = tiling.kBeteVecRow;
    dkbVecRow = tiling.dkbVecRow;
    dkbgVecRow = tiling.dkbgVecRow;
    dvbVecRow = tiling.dvbVecRow;
    kktVecRow = tiling.kktVecRow;
    kBetaCVNum = tiling.kBetaCVNum;
    dkbCVNum = tiling.dkbCVNum;
    dkbgCVNum = tiling.dkbgCVNum;
    dvbCVNum = tiling.dvbCVNum;
    kktCVNum = tiling.kktCVNum;

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
        for (int h = 0; h < HV; h++) {
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                continue;
            }
            auto& tensorBeta = tensorBetaInList[ubBetaDgListId];
            auto& tensorDg = tensorDgInList[ubBetaDgListId];
            {// copyin beta/dg
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbBetaInVMTE2List[ubBetaDgListId]);
                DataCopyPad(tensorBeta, betaTensor[h * T + bos], {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbBetaInMTE2VList[ubBetaDgListId]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbDgInVMTE2List[ubBetaDgListId]);
                DataCopyPad(tensorDg, dgTensor[h * T + bos], {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},{false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbDgInMTE2VList[ubBetaDgListId]);
            }
            Duplicate(tensorReducesum0, 0.0f, chunkSize);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                auto& tensordaIn = tensorDaInList[ubListId];
                auto& tensorKKTin = tensorKKTInList[cvListId];
                // copyin da
                {
                    auto dAOffset = (h * T + bos + rowOffset) * chunkSize;
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
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                    ProcessKKTComputerVF((__ubuf__ kType*)kKTinAddr, (__ubuf__ betaType*)betaInAddr,
                        (__ubuf__ kType*)daInAddr, (__ubuf__ float*)reducesum1Addr, (__ubuf__ float*)reducesum0Addr,
                        curChunkSize, curRowNum, chunkSize);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbDaInVMTE2List[ubListId]);
                    if (rowOffset + rowNum >= curChunkSize) {
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbBetaInVMTE2List[ubBetaDgListId]);
                    }
                }
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
                cvListId = (cvListId + 1 < kktCVNum) ? (cvListId + 1) : 0;
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
                DataCopyPad(dgTensor[h * T + bos], tensorOutDg, {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubBetaDgListId]);
            }
            ubBetaDgListId = (ubBetaDgListId + 1 < UB_STAGES) ? (ubBetaDgListId + 1) : 0;
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
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDvbComputerVF(
    __ubuf__ kType* dvOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* dvbIn, __ubuf__ kType* vIn,
    __ubuf__ betaType* betaIn, __ubuf__ betaType* dbetaIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t nKUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(kType)));
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    RegTensor<betaType> betaInReg;
    RegTensor<float> betaBrcbFP32Reg, dbetaFP32Reg;
    RegTensor<kType> vInReg;
    RegTensor<kType> dbetaInReg;
    RegTensor<kType> dvbInReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, dbetaAddFP32Reg, dvbFP32ZeroReg, dvbFP32OneReg, reduceSumReg;
    RegTensor<kType> dvOutReg;
    RegTensor<betaType> dbetaOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForLoad uLoadBeta;
    UnalignRegForLoad uLoadDbeta;
    UnalignRegForStore uStore;

    LoadAlign(vInReg, vIn);
    LoadAlign(dvbInReg, dvbIn);
    uint32_t nextEleOffset = 0;
    // 前mSize-1行
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaInReg, uLoadBeta, betaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32Reg, betaInReg, maskFull16);
            LoadUnAlignPre(uLoadDbeta, dbetaIn + mIdx);
            LoadUnAlign(dbetaInReg, uLoadDbeta, dbetaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(dbetaFP32Reg, dbetaInReg, maskFull16);
        } else {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaBrcbFP32Reg, uLoadBeta, betaIn + mIdx);
            LoadUnAlignPre(uLoadDbeta, dbetaIn + mIdx);
            LoadUnAlign(dbetaFP32Reg, uLoadDbeta, dbetaIn + mIdx);
        }
        Duplicate(betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t mOffset = mIdx * nKUbAligned;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            // uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            // LoadAlign(vInReg, vIn + eleOffset);
            // LoadAlign(dvbInReg, dvbIn + eleOffset);
            Cast<float, kType, ctHalf2Fp32Zero>(vFP32ZeroReg, vInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(vFP32OneReg, vInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dvbFP32ZeroReg, dvbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dvbFP32OneReg, dvbInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(vInReg, vIn + nextEleOffset);
            LoadAlign(dvbInReg, dvbIn + nextEleOffset);
            Mul(vFP32ZeroReg, vFP32ZeroReg, dvbFP32ZeroReg, maskFull32);
            Mul(vFP32OneReg, vFP32OneReg, dvbFP32OneReg, maskFull32);
            Mul(dvbFP32ZeroReg, dvbFP32ZeroReg, betaBrcbFP32Reg, maskFull32);
            Mul(dvbFP32OneReg, dvbFP32OneReg, betaBrcbFP32Reg, maskFull32);

            Cast<kType, float, ctFp322HalfOne>(dvOutReg, dvbFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dvOutReg, dvbFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dvOut + dstUbOffset, dvOutReg, maskFull16);

            Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
            Add(reduceSumReg, reduceSumReg, vFP32ZeroReg, maskFull32);
        }
        ReduceSum(dbetaAddFP32Reg, reduceSumReg, maskFull32);
        Add(dbetaAddFP32Reg, dbetaAddFP32Reg, dbetaFP32Reg, maskFull32);

        auto actDbetaOut = dBetaOut + mIdx;
        if constexpr (!std::is_same<betaType, float32_t>()) {
            Cast<betaType, float, ctFp322HalfZero>(dbetaOutReg, dbetaAddFP32Reg, maskFull32);
            StoreUnAlign(actDbetaOut, dbetaOutReg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        } else {
            StoreUnAlign(actDbetaOut, dbetaAddFP32Reg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        }
    }
    uint16_t mIdx = mSize - 1;
    {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaInReg, uLoadBeta, betaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32Reg, betaInReg, maskFull16);
            LoadUnAlignPre(uLoadDbeta, dbetaIn + mIdx);
            LoadUnAlign(dbetaInReg, uLoadDbeta, dbetaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(dbetaFP32Reg, dbetaInReg, maskFull16);
        } else {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaBrcbFP32Reg, uLoadBeta, betaIn + mIdx);
            LoadUnAlignPre(uLoadDbeta, dbetaIn + mIdx);
            LoadUnAlign(dbetaFP32Reg, uLoadDbeta, dbetaIn + mIdx);
        }
        Duplicate(betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t mOffset = mIdx * nKUbAligned;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            // uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            // LoadAlign(vInReg, vIn + eleOffset);
            // LoadAlign(dvbInReg, dvbIn + eleOffset);
            Cast<float, kType, ctHalf2Fp32Zero>(vFP32ZeroReg, vInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(vFP32OneReg, vInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dvbFP32ZeroReg, dvbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dvbFP32OneReg, dvbInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(vInReg, vIn + nextEleOffset);
            LoadAlign(dvbInReg, dvbIn + nextEleOffset);
            Mul(vFP32ZeroReg, vFP32ZeroReg, dvbFP32ZeroReg, maskFull32);
            Mul(vFP32OneReg, vFP32OneReg, dvbFP32OneReg, maskFull32);
            Mul(dvbFP32ZeroReg, dvbFP32ZeroReg, betaBrcbFP32Reg, maskFull32);
            Mul(dvbFP32OneReg, dvbFP32OneReg, betaBrcbFP32Reg, maskFull32);

            Cast<kType, float, ctFp322HalfOne>(dvOutReg, dvbFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dvOutReg, dvbFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dvOut + dstUbOffset, dvOutReg, maskFull16);

            Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
            Add(reduceSumReg, reduceSumReg, vFP32ZeroReg, maskFull32);
        }
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            // uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            // LoadAlign(vInReg, vIn + eleOffset);
            // LoadAlign(dvbInReg, dvbIn + eleOffset);
            Cast<float, kType, ctHalf2Fp32Zero>(vFP32ZeroReg, vInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(vFP32OneReg, vInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dvbFP32ZeroReg, dvbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dvbFP32OneReg, dvbInReg, maskFull16);
            Mul(vFP32ZeroReg, vFP32ZeroReg, dvbFP32ZeroReg, maskFull32);
            Mul(vFP32OneReg, vFP32OneReg, dvbFP32OneReg, maskFull32);
            Mul(dvbFP32ZeroReg, dvbFP32ZeroReg, betaBrcbFP32Reg, maskFull32);
            Mul(dvbFP32OneReg, dvbFP32OneReg, betaBrcbFP32Reg, maskFull32);

            Cast<kType, float, ctFp322HalfOne>(dvOutReg, dvbFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dvOutReg, dvbFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dvOut + dstUbOffset, dvOutReg, maskFull16);

            Add(vFP32ZeroReg, vFP32ZeroReg, vFP32OneReg, maskFull32);
            Add(reduceSumReg, reduceSumReg, vFP32ZeroReg, maskFull32);
        }
        ReduceSum(dbetaAddFP32Reg, reduceSumReg, maskFull32);
        Add(dbetaAddFP32Reg, dbetaAddFP32Reg, dbetaFP32Reg, maskFull32);

        auto actDbetaOut = dBetaOut + mIdx;
        if constexpr (!std::is_same<betaType, float32_t>()) {
            Cast<betaType, float, ctFp322HalfZero>(dbetaOutReg, dbetaAddFP32Reg, maskFull32);
            StoreUnAlign(actDbetaOut, dbetaOutReg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        } else {
            StoreUnAlign(actDbetaOut, dbetaAddFP32Reg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        }
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
        for (int h = 0; h < HV; h++) {
            uint32_t taskNum = CeilDiv(curChunkSize, rowNum);
            taskNum = CeilDiv(taskNum, GetSubBlockNum());
            uint32_t dealTaskNum = 0;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h * T + bos + rowOffset) * V;
                auto betaOffset = h * T + bos + rowOffset;
                
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
                    ProcessDvbComputerVF((__ubuf__ kType*)dvOutAddr, (__ubuf__ betaType*)dBetaOutAddr,
                                         (__ubuf__ kType*)dvbInAddr, (__ubuf__ kType*)vInAddr,
                                         (__ubuf__ betaType*)betaInAddr, (__ubuf__ betaType*)dbetaInAddr,
                                         curRowNum, V);
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
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbgComputerVF(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ betaType* dgOut,
    __ubuf__ kType* kIn, __ubuf__ betaType* betaIn, __ubuf__ betaType* gIn,
    __ubuf__ kType* dkIn, __ubuf__ betaType* dBetaIn, __ubuf__ kType* dkbgIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t nKUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(kType)));
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<kType> dkInReg;
    RegTensor<betaType> dbetaInReg;
    RegTensor<kType> dkbgInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, dgFP32ZeroReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, dBetaAddZeroReg, dBetaAddOneReg, dkbgFP32ZeroReg, dkbgFP32OneReg;
    RegTensor<float> gBrcbFP32ZeroReg, gBrcbFP32OneReg, dbetaFP32ZeroReg, dbetaFP32OneReg;
    RegTensor<float> kReduceSumReg, dkReduceSumReg;
    RegTensor<kType> dkOutReg;
    RegTensor<betaType> dBetaOutReg;
    RegTensor<betaType> dgOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForLoad uLoadBeta;
    UnalignRegForLoad uLoadG;
    UnalignRegForLoad uLoadDBeta;
    UnalignRegForStore uStoreDBeta;
    UnalignRegForStore uStoreDg;

    if constexpr (!std::is_same<betaType, float>()) {
        LoadAlign(dbetaInReg, dBetaIn);
        Cast<float, betaType, ctHalf2Fp32Zero>(dbetaFP32ZeroReg, dbetaInReg, maskFull16);
        Cast<float, betaType, ctHalf2Fp32One>(dbetaFP32OneReg, dbetaInReg, maskFull16);
    } else {
        LoadAlign<betaType, LoadDist::DIST_DINTLV_B32>(dbetaFP32ZeroReg, dbetaFP32OneReg, dBetaIn);
    }
    LoadAlign(kInReg, kIn);
    LoadAlign(dkbgInReg, dkbgIn);
    LoadAlign(dkInReg, dkIn);
    uint32_t nextEleOffset = 0;
    // 前mSize - 1行
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaInReg, uLoadBeta, betaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32ZeroReg, betaInReg, maskFull16);
            LoadUnAlignPre(uLoadG, gIn + mIdx);
            LoadUnAlign(gInReg, uLoadG, gIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(gBrcbFP32ZeroReg, gInReg, maskFull16);
            LoadUnAlignPre(uLoadDBeta, dBetaIn + mIdx);
            LoadUnAlign(dbetaInReg, uLoadDBeta, dBetaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(dbetaFP32ZeroReg, dbetaInReg, maskFull16);
        } else {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaBrcbFP32ZeroReg, uLoadBeta, betaIn + mIdx);
            LoadUnAlignPre(uLoadG, gIn + mIdx);
            LoadUnAlign(gBrcbFP32ZeroReg, uLoadG, gIn + mIdx);
            LoadUnAlignPre(uLoadDBeta, dBetaIn + mIdx);
            LoadUnAlign(dbetaFP32ZeroReg, uLoadDBeta, dBetaIn + mIdx);
        }
        MaskReg maskgExpReg;
        uint32_t expNum = 1;
        maskgExpReg = UpdateMask<betaType>(expNum);
        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskgExpReg);
        Duplicate(betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        Duplicate(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t mOffset = mIdx * nKUbAligned;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkbgFP32ZeroReg, dkbgInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkbgFP32OneReg, dkbgInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(kInReg, kIn + nextEleOffset);
            LoadAlign(dkbgInReg, dkbgIn + nextEleOffset);
            
            Mul(dkbgFP32ZeroReg, dkbgFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            Mul(dkbgFP32OneReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, maskFull32);
            Mul(kFP32ZeroReg, kFP32ZeroReg, dkbgFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, dkbgFP32OneReg, maskFull32);
            Mul(dkFP32ZeroReg, kFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkFP32OneReg, kFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            Mul(dkbgFP32ZeroReg, dkbgFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkbgFP32OneReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);

            Cast<float, kType, ctHalf2Fp32Zero>(dkFP32ZeroReg, dkInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkFP32OneReg, dkInReg, maskFull16);
            LoadAlign(dkInReg, dkIn + nextEleOffset);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkbgFP32ZeroReg, maskFull32);
            Add(dkFP32OneReg, dkFP32OneReg, dkbgFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(dkOutReg, dkFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dkOutReg, dkFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dkOut + dstUbOffset, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        auto actDbetaOut = dBetaOut + mIdx;
        auto actDgOut = dgOut + mIdx;
        if constexpr (!std::is_same<betaType, float32_t>()) {
            Cast<betaType, float, ctFp322HalfZero>(dBetaOutReg, dBetaAddZeroReg, maskFull32);
            StoreUnAlign(actDbetaOut, dBetaOutReg, uStoreDBeta, 1);
            StoreUnAlignPost(actDbetaOut, uStoreDBeta, 0);
            Cast<betaType, float, ctFp322HalfZero>(dgOutReg, dgFP32ZeroReg, maskFull32);
            StoreUnAlign(actDgOut, dgOutReg, uStoreDg, 1);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        } else {
            StoreUnAlign(actDbetaOut, dBetaAddZeroReg, uStoreDBeta, 1);
            StoreUnAlignPost(actDbetaOut, uStoreDBeta, 0);
            StoreUnAlign(actDgOut, dgFP32ZeroReg, uStoreDg, 1);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        }
    }
    // 最后一行
    uint16_t mIdx =  mSize - 1;
    {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaInReg, uLoadBeta, betaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32ZeroReg, betaInReg, maskFull16);
            LoadUnAlignPre(uLoadG, gIn + mIdx);
            LoadUnAlign(gInReg, uLoadG, gIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(gBrcbFP32ZeroReg, gInReg, maskFull16);
            LoadUnAlignPre(uLoadDBeta, dBetaIn + mIdx);
            LoadUnAlign(dbetaInReg, uLoadDBeta, dBetaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(dbetaFP32ZeroReg, dbetaInReg, maskFull16);
        } else {
            LoadUnAlignPre(uLoadBeta, betaIn + mIdx);
            LoadUnAlign(betaBrcbFP32ZeroReg, uLoadBeta, betaIn + mIdx);
            LoadUnAlignPre(uLoadG, gIn + mIdx);
            LoadUnAlign(gBrcbFP32ZeroReg, uLoadG, gIn + mIdx);
            LoadUnAlignPre(uLoadDBeta, dBetaIn + mIdx);
            LoadUnAlign(dbetaFP32ZeroReg, uLoadDBeta, dBetaIn + mIdx);
        }
        MaskReg maskgExpReg;
        uint32_t expNum = 1;
        maskgExpReg = UpdateMask<betaType>(expNum);
        Exp(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskgExpReg);
        Duplicate(betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        Duplicate(gBrcbFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
        Duplicate(kReduceSumReg, static_cast<float>(0), maskFull32);
        Duplicate(dkReduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t mOffset = mIdx * nKUbAligned;
        // 最后一行的前nLoopCnt - 1
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkbgFP32ZeroReg, dkbgInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkbgFP32OneReg, dkbgInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(kInReg, kIn + nextEleOffset);
            LoadAlign(dkbgInReg, dkbgIn + nextEleOffset);
            
            Mul(dkbgFP32ZeroReg, dkbgFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            Mul(dkbgFP32OneReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, maskFull32);
            Mul(kFP32ZeroReg, kFP32ZeroReg, dkbgFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, dkbgFP32OneReg, maskFull32);
            Mul(dkFP32ZeroReg, kFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkFP32OneReg, kFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            Mul(dkbgFP32ZeroReg, dkbgFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkbgFP32OneReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);

            Cast<float, kType, ctHalf2Fp32Zero>(dkFP32ZeroReg, dkInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkFP32OneReg, dkInReg, maskFull16);
            LoadAlign(dkInReg, dkIn + nextEleOffset);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkbgFP32ZeroReg, maskFull32);
            Add(dkFP32OneReg, dkFP32OneReg, dkbgFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(dkOutReg, dkFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dkOutReg, dkFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dkOut + dstUbOffset, dkOutReg, maskFull16);
        }
        // 最后一行最后一个loop
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkbgFP32ZeroReg, dkbgInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkbgFP32OneReg, dkbgInReg, maskFull16);
            
            Mul(dkbgFP32ZeroReg, dkbgFP32ZeroReg, gBrcbFP32ZeroReg, maskFull32);
            Mul(dkbgFP32OneReg, dkbgFP32OneReg, gBrcbFP32ZeroReg, maskFull32);
            Mul(kFP32ZeroReg, kFP32ZeroReg, dkbgFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, dkbgFP32OneReg, maskFull32);
            Mul(dkFP32ZeroReg, kFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkFP32OneReg, kFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(kReduceSumReg, kReduceSumReg, kFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkFP32OneReg, maskFull32);
            Add(dkReduceSumReg, dkReduceSumReg, dkFP32ZeroReg, maskFull32);
            Mul(dkbgFP32ZeroReg, dkbgFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkbgFP32OneReg, dkbgFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);

            Cast<float, kType, ctHalf2Fp32Zero>(dkFP32ZeroReg, dkInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkFP32OneReg, dkInReg, maskFull16);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkbgFP32ZeroReg, maskFull32);
            Add(dkFP32OneReg, dkFP32OneReg, dkbgFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(dkOutReg, dkFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dkOutReg, dkFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dkOut + dstUbOffset, dkOutReg, maskFull16);
        }
        ReduceSum(dBetaAddZeroReg, kReduceSumReg, maskFull32);
        ReduceSum(dgFP32ZeroReg, dkReduceSumReg, maskFull32);
        Add(dBetaAddZeroReg, dBetaAddZeroReg, dbetaFP32ZeroReg, maskFull32);
        auto actDbetaOut = dBetaOut + mIdx;
        auto actDgOut = dgOut + mIdx;
        if constexpr (!std::is_same<betaType, float32_t>()) {
            Cast<betaType, float, ctFp322HalfZero>(dBetaOutReg, dBetaAddZeroReg, maskFull32);
            StoreUnAlign(actDbetaOut, dBetaOutReg, uStoreDBeta, 1);
            StoreUnAlignPost(actDbetaOut, uStoreDBeta, 0);
            Cast<betaType, float, ctFp322HalfZero>(dgOutReg, dgFP32ZeroReg, maskFull32);
            StoreUnAlign(actDgOut, dgOutReg, uStoreDg, 1);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        } else {
            StoreUnAlign(actDbetaOut, dBetaAddZeroReg, uStoreDBeta, 1);
            StoreUnAlignPost(actDbetaOut, uStoreDBeta, 0);
            StoreUnAlign(actDgOut, dgFP32ZeroReg, uStoreDg, 1);
            StoreUnAlignPost(actDgOut, uStoreDg, 0);
        }
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
    uint32_t curRowNum = rowNum;

    uint32_t ubOffset = 0;
    uint32_t ubListId = 0;
    uint32_t cvListId = 0;
    AscendC::LocalTensor<kType> tensorDkbgInList[MAX_CUBE_VEC_SYNC_NUM];
    AscendC::LocalTensor<kType> tensorKinList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorBetaInList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorGInList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorDkInList[UB_STAGES];
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
        tensorDkInList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
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
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < HV; h++) {
            uint32_t taskNum = CeilDiv(curChunkSize, rowNum);
            taskNum = CeilDiv(taskNum, GetSubBlockNum());
            uint32_t dealTaskNum = 0;

            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;

                auto& tensorDkbgIn = tensorDkbgInList[cvListId];
                auto& tensorKin = tensorKinList[ubListId];
                auto& tensorBetaIn = tensorBetaInList[ubListId];
                auto& tensorGIn = tensorGInList[ubListId];
                auto& tensorDkIn = tensorDkInList[ubListId];
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
                    DataCopy(tensorDkIn, dkTensor[kOffset], K * curRowNum);
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
                    auto dkInAddr = reinterpret_cast<uint64_t>(tensorDkIn.GetPhyAddr());
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
                        (__ubuf__ kType*)dkInAddr,
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
                    DataCopy(dkTensor[kOffset], tensorDkOut, K * curRowNum);
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
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessDkbComputerVF(
    __ubuf__ kType* dkOut, __ubuf__ betaType* dBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ kType* dkIn, __ubuf__ kType* dkbIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t nKUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(kType)));
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    RegTensor<betaType> betaInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32OneReg, dBetaFP32Reg;
    RegTensor<kType> kInReg;
    RegTensor<kType> dkInReg;
    RegTensor<kType> dkbInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, dkFP32ZeroReg, dkFP32OneReg, dkbFP32ZeroReg, dkbFP32OneReg, reduceSumReg;
    RegTensor<kType> dkOutReg;
    RegTensor<betaType> dBetaOutZeroReg, dBetaOutOneReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForLoad uLoad;
    UnalignRegForStore uStore;

    LoadAlign(kInReg, kIn);
    LoadAlign(dkbInReg, dkbIn);
    LoadAlign(dkInReg, dkIn);
    uint32_t nextEleOffset = 0;
    // 前mSize-1行
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            LoadUnAlignPre(uLoad, betaIn + mIdx);
            LoadUnAlign(betaInReg, uLoad, betaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32ZeroReg, betaInReg, maskFull16);
        } else {
            LoadUnAlignPre(uLoad, betaIn + mIdx);
            LoadUnAlign(betaBrcbFP32ZeroReg, uLoad, betaIn + mIdx);
        }
        Duplicate(betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t mOffset = mIdx * nKUbAligned;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkbFP32ZeroReg, dkbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkbFP32OneReg, dkbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkFP32ZeroReg, dkInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkFP32OneReg, dkInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(kInReg, kIn + nextEleOffset);
            LoadAlign(dkbInReg, dkbIn + nextEleOffset);
            LoadAlign(dkInReg, dkIn + nextEleOffset);
            Mul(kFP32ZeroReg, kFP32ZeroReg, dkbFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, dkbFP32OneReg, maskFull32);
            Mul(dkbFP32ZeroReg, dkbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkbFP32OneReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkbFP32ZeroReg, maskFull32);
            Add(dkFP32OneReg, dkFP32OneReg, dkbFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(dkOutReg, dkFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dkOutReg, dkFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dkOut + dstUbOffset, dkOutReg, maskFull16);

            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(reduceSumReg, reduceSumReg, kFP32ZeroReg, maskFull32);
        }
        ReduceSum(dBetaFP32Reg, reduceSumReg, maskFull32);

        auto actDbetaOut = dBetaOut + mIdx;
        if constexpr (!std::is_same<betaType, float32_t>()) {
            Cast<betaType, float, ctFp322HalfZero>(dBetaOutZeroReg, dBetaFP32Reg, maskFull32);
            StoreUnAlign(actDbetaOut, dBetaOutZeroReg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        } else {
            StoreUnAlign(actDbetaOut, dBetaFP32Reg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        }
    }
    // 最后一行
    uint16_t mIdx = mSize - 1;
    {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            LoadUnAlignPre(uLoad, betaIn + mIdx);
            LoadUnAlign(betaInReg, uLoad, betaIn + mIdx);
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32ZeroReg, betaInReg, maskFull16);
        } else {
            LoadUnAlignPre(uLoad, betaIn + mIdx);
            LoadUnAlign(betaBrcbFP32ZeroReg, uLoad, betaIn + mIdx);
        }
        Duplicate(betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        Duplicate(reduceSumReg, static_cast<float>(0), maskFull32);
        uint32_t mOffset = mIdx * nKUbAligned;
        uint32_t nextEleOffset = 0;
        // 最后一行的前nLoopCnt - 1
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkbFP32ZeroReg, dkbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkbFP32OneReg, dkbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkFP32ZeroReg, dkInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkFP32OneReg, dkInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(kInReg, kIn + nextEleOffset);
            LoadAlign(dkbInReg, dkbIn + nextEleOffset);
            LoadAlign(dkInReg, dkIn + nextEleOffset);
            Mul(kFP32ZeroReg, kFP32ZeroReg, dkbFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, dkbFP32OneReg, maskFull32);
            Mul(dkbFP32ZeroReg, dkbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkbFP32OneReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkbFP32ZeroReg, maskFull32);
            Add(dkFP32OneReg, dkFP32OneReg, dkbFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(dkOutReg, dkFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dkOutReg, dkFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dkOut + dstUbOffset, dkOutReg, maskFull16);

            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(reduceSumReg, reduceSumReg, kFP32ZeroReg, maskFull32);
        }
        // 最后一行最后一个loop
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkbFP32ZeroReg, dkbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkbFP32OneReg, dkbInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32Zero>(dkFP32ZeroReg, dkInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(dkFP32OneReg, dkInReg, maskFull16);
            Mul(kFP32ZeroReg, kFP32ZeroReg, dkbFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, dkbFP32OneReg, maskFull32);
            Mul(dkbFP32ZeroReg, dkbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(dkbFP32OneReg, dkbFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Add(dkFP32ZeroReg, dkFP32ZeroReg, dkbFP32ZeroReg, maskFull32);
            Add(dkFP32OneReg, dkFP32OneReg, dkbFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(dkOutReg, dkFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(dkOutReg, dkFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)dkOut + dstUbOffset, dkOutReg, maskFull16);

            Add(kFP32ZeroReg, kFP32ZeroReg, kFP32OneReg, maskFull32);
            Add(reduceSumReg, reduceSumReg, kFP32ZeroReg, maskFull32);
        }
        ReduceSum(dBetaFP32Reg, reduceSumReg, maskFull32);

        auto actDbetaOut = dBetaOut + mIdx;
        if constexpr (!std::is_same<betaType, float32_t>()) {
            Cast<betaType, float, ctFp322HalfZero>(dBetaOutZeroReg, dBetaFP32Reg, maskFull32);
            StoreUnAlign(actDbetaOut, dBetaOutZeroReg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        } else {
            StoreUnAlign(actDbetaOut, dBetaFP32Reg, uStore, 1);
            StoreUnAlignPost(actDbetaOut, uStore, 0);
        }
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
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < HV; h++) {
            uint32_t taskNum = CeilDiv(curChunkSize, rowNum);
            taskNum = CeilDiv(taskNum, GetSubBlockNum());
            uint32_t dealTaskNum = 0;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;

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
                    DataCopy(tensorDkin, dkTensor[kOffset], K * curRowNum);
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
                    ProcessDkbComputerVF(
                        (__ubuf__ kType*)dkOutAddr, (__ubuf__ betaType*)dBetaOutAddr, (__ubuf__ kType*)kInAddr, 
                        (__ubuf__ betaType*)betaInAddr,(__ubuf__ kType*)dkInAddr, (__ubuf__ kType*)dkbInAddr, 
                        curRowNum, K);
                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                }
                // copyout
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    DataCopy(dkTensor[kOffset], tensorDkOut, K * curRowNum);
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
__simd_vf__ inline void PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessKBetaComputerVF(
    __ubuf__ kType* kBetaOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t nKUbAligned = Align(nSize, static_cast<uint16_t>(UB_ALIGN_SIZE / sizeof(kType)));
    uint16_t nLoopCnt = (nSize + eleKNumPerVf - 1) / eleKNumPerVf;
    RegTensor<kType> kInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg;
    RegTensor<betaType> betaInReg;
    RegTensor<float> betaBrcbFP32ZeroReg, betaBrcbFP32OneReg;
    RegTensor<kType> kBetaOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    UnalignRegForLoad uLoad;
    UnalignRegForStore uStore;

    if constexpr (!std::is_same<betaType, float>()) {
        LoadUnAlignPre(uLoad, betaIn);
        LoadUnAlign(betaInReg, uLoad, betaIn);
    } else {
        LoadUnAlignPre(uLoad, betaIn);
        LoadUnAlign(betaInReg, uLoad, betaIn);
    }
    LoadAlign(kInReg, kIn);
    uint32_t nextEleOffset = 0;
    for (uint16_t mIdx = 0; mIdx < mSize - 1; mIdx++) {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32ZeroReg, betaInReg, maskFull16);
            Duplicate(betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            LoadUnAlignPre(uLoad, betaIn + (mIdx + 1));
            LoadUnAlign(betaInReg, uLoad, betaIn + (mIdx + 1));
        } else {
            Duplicate(betaBrcbFP32ZeroReg, betaInReg, maskFull32);
            LoadUnAlignPre(uLoad, betaIn + (mIdx + 1));
            LoadUnAlign(betaInReg, uLoad, betaIn + (mIdx + 1));
        }
        uint32_t mOffset = mIdx * nKUbAligned;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; vfBlockIdx++) {
            uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(kInReg, kIn + nextEleOffset);
            Mul(kFP32ZeroReg, kFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(kBetaOutReg, kFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(kBetaOutReg, kFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)kBetaOut + dstUbOffset, kBetaOutReg, maskFull16);
        }
    }
    uint16_t mIdx = mSize - 1;
    {
        // cast from bf16/fp16 to FP32
        if constexpr (!std::is_same<betaType, float>()) {
            Cast<float, betaType, ctHalf2Fp32Zero>(betaBrcbFP32ZeroReg, betaInReg, maskFull16);
            Duplicate(betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        } else {
            Duplicate(betaBrcbFP32ZeroReg, betaInReg, maskFull32);
        }
        Duplicate(betaBrcbFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
        uint32_t mOffset = mIdx * nKUbAligned;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt - 1; vfBlockIdx++) {
            uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            uint32_t thisEleNum = min(eleKNumPerVf, nSize - vfBlockIdx * eleKNumPerVf);
            nextEleOffset += thisEleNum;
            LoadAlign(kInReg, kIn + nextEleOffset);
            Mul(kFP32ZeroReg, kFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(kBetaOutReg, kFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(kBetaOutReg, kFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)kBetaOut + dstUbOffset, kBetaOutReg, maskFull16);
        }
        uint16_t vfBlockIdx = nLoopCnt - 1;
        {
            uint32_t eleOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            Cast<float, kType, ctHalf2Fp32Zero>(kFP32ZeroReg, kInReg, maskFull16);
            Cast<float, kType, ctHalf2Fp32One>(kFP32OneReg, kInReg, maskFull16);
            Mul(kFP32ZeroReg, kFP32ZeroReg, betaBrcbFP32ZeroReg, maskFull32);
            Mul(kFP32OneReg, kFP32OneReg, betaBrcbFP32ZeroReg, maskFull32);
            Cast<kType, float, ctFp322HalfOne>(kBetaOutReg, kFP32OneReg, maskFull32);
            Cast<kType, float, ctFp322HalfZero>(kBetaOutReg, kFP32ZeroReg, maskFull32);
            uint32_t dstUbOffset = mOffset + vfBlockIdx * eleKNumPerVf;
            StoreAlign((__ubuf__ kType*&)kBetaOut + dstUbOffset, kBetaOutReg, maskFull16);
        }
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
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(kBetaOutQue, 2, rowNum * K * sizeof(kType));
    
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h = 0; h < HV; h++) {
            AscendC::CrossCoreWaitFlag(SYNC_FLAG_5);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h * T + bos + rowOffset) * K;
                auto betaOffset = h * T + bos + rowOffset;
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
                    ProcessKBetaComputerVF(
                        (__ubuf__ kType*)kBetaOutAddr, (__ubuf__ kType*)kInAddr, (__ubuf__ betaType*)betaInAddr,
                        curRowNum, K);
                    kInQue.FreeTensor(tensorKin);
                    betaInQue.FreeTensor(tensorBetain);
                    kBetaOutQue.EnQue(tensorOut);
                }
                // copyout
                {
                    auto tensorOut = kBetaOutQue.DeQue<kType>();
                    DataCopy(workSpaceTensor[kOffset], tensorOut, K * curRowNum);
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
