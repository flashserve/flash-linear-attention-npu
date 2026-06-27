/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_da_vector.h
 * \brief
 */

#ifndef PREPARE_WY_REPR_BWD_DA_VECTOR_H
#define PREPARE_WY_REPR_BWD_DA_VECTOR_H
#include "prepare_wy_repr_bwd_da_tiling_data_apt.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_utils/vector/regbase.hpp"

using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename kType, typename betaType>
class PrepareWyReprBwdDAVectorProcess {
public:
     /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdDAVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dw_,
                                                      GM_ADDR du_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                      GM_ADDR chunk_indices_, GM_ADDR dA_, GM_ADDR workspace_);
    __aicore__ inline void Init(const PrepareWyReprBwdDaTilingDataA5& tiling, AscendC::TPipe *pipe_);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessVBeta();
    __simd_vf__ inline void ProcessVBetaComputerVFOneLineOneCol(__ubuf__ kType* vBetaOut,
                                                                __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
                                                                uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessVBetaComputerVFMutiLineOneCol(__ubuf__ kType* vBetaOut,
                                                                 __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
                                                                 uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt);
    __simd_vf__ inline void ProcessVBetaComputerVFTwoCol(__ubuf__ kType* vBetaOut,
                                                         __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
                                                         uint16_t mSize, uint16_t nSize);
    __aicore__ inline void ProcessKBetaG();
    __aicore__ inline void ProcessMDuDw();
    __simd_vf__ inline void ProcessMDuDwComputerVFOneLineOneCol(__ubuf__ kType* mduwOut,
                                                                __ubuf__ kType* mduIn, __ubuf__ kType* mdwIn,
                                                                uint16_t mSize, uint16_t nSize, uint32_t startRow);
    __simd_vf__ inline void ProcessMDuDwComputerVFMutiLineOneCol(__ubuf__ kType* mduwOut,
                                                                 __ubuf__ kType* mduIn, __ubuf__ kType* mdwIn,
                                                                 uint16_t mSize, uint16_t nSize, uint32_t startRow,
                                                                 uint16_t lastLoopCnt);
    __simd_vf__ inline void ProcessKBetaGComputerVFOneLineOneCol(__ubuf__ kType* kBetaGOut,
                                                                 __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                                 __ubuf__ betaType* gIn,
                                                                 uint16_t mSize, uint16_t nSize);
    __simd_vf__ inline void ProcessKBetaGComputerVFMutiLineOneCol(__ubuf__ kType* kBetaGOut,
                                                                  __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                                  __ubuf__ betaType* gIn,
                                                                  uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt);
    __simd_vf__ inline void ProcessKBetaGComputerVFTwoCol(__ubuf__ kType* kBetaGOut,
                                                          __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
                                                          __ubuf__ betaType* gIn,
                                                          uint16_t mSize, uint16_t nSize);
    __aicore__ inline void ProcessG();
    __simd_vf__ inline void ProcessGComputerVFOneLineOneCol(__ubuf__ kType* dAOut, __ubuf__ kType* dA6In,
                                                            __ubuf__ betaType* gIn, __ubuf__ betaType* gAllIn,
                                                            uint16_t mSize, uint16_t nSize,
                                                            uint32_t startRow, uint32_t calcColSize);
    __simd_vf__ inline void ProcessGComputerVFMutiLineOneCol(__ubuf__ kType* dAOut, __ubuf__ kType* dA6In,
                                                             __ubuf__ betaType* gIn, __ubuf__ betaType* gAllIn,
                                                             uint16_t mSize, uint16_t nSize, uint32_t startRow,
                                                             uint16_t lastLoopCnt, uint32_t calcColSize);
private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t HV = 0;
    uint64_t HK = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t BT = 0;
    uint64_t chunkNum = 0;
    uint64_t rowNumKBetaG = 0;
    uint64_t rowNumVBeta = 0;
    uint64_t rowNumMDuDw = 0;
    uint64_t rowNumG = 0;
    uint64_t gCVNum = 0;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{SYNC_FLAG_2, SYNC_FLAG_3};
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_FLAG_4, SYNC_FLAG_5};

    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR beta;
    GM_ADDR A;
    GM_ADDR dw;
    GM_ADDR du;
    GM_ADDR g;
    GM_ADDR cu_seqlens;
    GM_ADDR chunk_indices;
    GM_ADDR dA;
    GM_ADDR workspace;
    AscendC::TPipe *pipe = nullptr;
private:
    GlobalTensor<kType> kTensor;
    GlobalTensor<kType> vTensor;
    GlobalTensor<betaType> gTensor;
    GlobalTensor<betaType> betaTensor;
    GlobalTensor<kType> dATensor;
    GlobalTensor<kType> dA1Tensor;
    GlobalTensor<kType> dA2Tensor;
    GlobalTensor<kType> dA4Tensor;
    GlobalTensor<kType> dA5Tensor;
    GlobalTensor<kType> dA6Tensor;
    GlobalTensor<kType> workSpaceTensor;
    GlobalTensor<kType> workSpace2Tensor;

    TQue<AscendC::TPosition::VECIN, 1> kInQue;
    TQue<AscendC::TPosition::VECIN, 1> vInQue;
    TQue<AscendC::TPosition::VECIN, 1> gInQue;
    TQue<AscendC::TPosition::VECIN, 1> betaInQue;
    TQue<AscendC::TPosition::VECIN, 1> mduInQue;
    TQue<AscendC::TPosition::VECIN, 1> mdwInQue;

    TQue<AscendC::TPosition::VECOUT, 1> kBetaGOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> vBetaOutQue;
    TQue<AscendC::TPosition::VECOUT, 1> mduwOutQue;
};

template <typename kType, typename betaType>
__aicore__ inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::PrepareWyReprBwdDAVectorProcess(
    GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR cu_seqlens_,
    GM_ADDR chunk_indices_, GM_ADDR dA_, GM_ADDR workspace_)
    : k(k_), v(v_), beta(beta_), A(A_), dw(dw_), du(du_), g(g_),
      cu_seqlens(cu_seqlens_), chunk_indices(chunk_indices_), dA(dA_), workspace(workspace_) {}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::Init(
    const PrepareWyReprBwdDaTilingDataA5& tiling, AscendC::TPipe *pipe_) {
    B = tiling.B;
    T = tiling.T;
    HV = tiling.HV;
    HK = tiling.HK;
    K = tiling.K;
    V = tiling.V;
    BT = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    rowNumKBetaG = tiling.rowNumKBetaG;
    rowNumVBeta = tiling.rowNumVBeta;
    rowNumMDuDw = tiling.rowNumMDuDw;
    rowNumG = tiling.rowNumG;
    gCVNum = tiling.gCVNum;

    pipe = pipe_;
    workSpaceTensor.SetGlobalBuffer((__gm__ kType *)workspace);
    workSpace2Tensor.SetGlobalBuffer((__gm__ kType *)workspace + B * HV * T * BT);
    dA1Tensor.SetGlobalBuffer((__gm__ kType *)dA);
    dA2Tensor.SetGlobalBuffer((__gm__ kType *)workspace);
    dA4Tensor.SetGlobalBuffer((__gm__ kType *)dA);
    dA5Tensor.SetGlobalBuffer((__gm__ kType *)workspace);
    dA6Tensor.SetGlobalBuffer((__gm__ kType *)dA);
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::Process() {
    ProcessKBetaG();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessVBeta();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessMDuDw();
    pipe->Reset();
    AscendC::SyncAll<false>();
    ProcessG();
    return;
}

// k、beta和g 的计算
template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessKBetaG() {
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = rowNumKBetaG;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;
    // init
    kTensor.SetGlobalBuffer((__gm__ kType *)k);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);
    gTensor.SetGlobalBuffer((__gm__ betaType *)g);

    // 搬入
    pipe->InitBuffer(kInQue, 2, rowNum * K * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(gInQue, 2, rowNum * sizeof(betaType));
    // 搬出
    pipe->InitBuffer(kBetaGOutQue, 2, rowNum * K * sizeof(kType));

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, BT, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        uint64_t keyBos = bos;
        if (cu_seqlens == nullptr && HV != HK) {
            uint64_t batchIdx = bos / (HV * T);
            uint64_t timeBos = bos - batchIdx * HV * T;
            keyBos = batchIdx * HK * T + timeBos;
        }
        uint64_t groupSize = HV / HK;
        for (int h_v = 0; h_v < HV; h_v++) {
            uint64_t h_k = h_v / groupSize;
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto kOffset = (h_k * T + keyBos + rowOffset) * K;
                auto betaOffset = h_v * T + bos + rowOffset;
                auto gOffset = h_v * T + bos + rowOffset;
                // copyin
                {
                    auto tensorKIn = kInQue.AllocTensor<kType>();
                    DataCopy(tensorKIn, kTensor[kOffset], K * curRowNum);
                    auto tensorBetaIn = betaInQue.AllocTensor<betaType>();
                    DataCopyPad(tensorBetaIn, betaTensor[betaOffset],
                        {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                        {false, 0, 0, 0});
                    auto tensorGIn = gInQue.AllocTensor<betaType>();
                    DataCopyPad(tensorGIn, gTensor[gOffset],
                        {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                        {false, 0, 0, 0});

                    kInQue.EnQue(tensorKIn);
                    betaInQue.EnQue(tensorBetaIn);
                    gInQue.EnQue(tensorGIn);
                }
                // compute
                {
                    auto tensorKIn = kInQue.DeQue<kType>();
                    auto tensorBetaIn = betaInQue.DeQue<betaType>();
                    auto tensorGIn = gInQue.DeQue<betaType>();
                    auto tensorOut = kBetaGOutQue.AllocTensor<kType>();
                    auto kBetaGOutAddr = reinterpret_cast<uint64_t>(tensorOut.GetPhyAddr());
                    auto kInAddr = reinterpret_cast<uint64_t>(tensorKIn.GetPhyAddr());
                    auto betaInAddr = reinterpret_cast<uint64_t>(tensorBetaIn.GetPhyAddr());
                    auto gInAddr = reinterpret_cast<uint64_t>(tensorGIn.GetPhyAddr());
                    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
                    if (K <= eleKNumPerVf) {
                        if (curRowNum == 1) {
                            ProcessKBetaGComputerVFOneLineOneCol(
                                    (__ubuf__ kType*)kBetaGOutAddr, (__ubuf__ kType*)kInAddr,
                                    (__ubuf__ betaType*)betaInAddr, (__ubuf__ betaType*)gInAddr,
                                    curRowNum, K);
                        } else {
                            uint16_t lastLoopCnt = curRowNum % PRELOAD_NUM;
                            ProcessKBetaGComputerVFMutiLineOneCol(
                                    (__ubuf__ kType*)kBetaGOutAddr, (__ubuf__ kType*)kInAddr,
                                    (__ubuf__ betaType*)betaInAddr, (__ubuf__ betaType*)gInAddr,
                                    curRowNum, K, lastLoopCnt);
                        }
                    } else {
                        ProcessKBetaGComputerVFTwoCol(
                                    (__ubuf__ kType*)kBetaGOutAddr, (__ubuf__ kType*)kInAddr,
                                    (__ubuf__ betaType*)betaInAddr, (__ubuf__ betaType*)gInAddr,
                                    curRowNum, K);
                    }
                    kInQue.FreeTensor(tensorKIn);
                    betaInQue.FreeTensor(tensorBetaIn);
                    gInQue.FreeTensor(tensorGIn);
                    kBetaGOutQue.EnQue(tensorOut);
                }
                //copyout
                {
                    auto tensorOut = kBetaGOutQue.DeQue<kType>();
                    auto kbgOffset = (h_v * T + bos + rowOffset) * K;
                    DataCopy(workSpace2Tensor[kbgOffset], tensorOut, K * curRowNum);
                    kBetaGOutQue.FreeTensor(tensorOut);
                }
            }
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
        }
    }
    return;
}

// v 和 beta 的计算
template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessVBeta() {
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = rowNumVBeta;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    // init
    pipe->InitBuffer(vInQue, 2, rowNum * V * sizeof(kType));
    pipe->InitBuffer(betaInQue, 2, rowNum * sizeof(betaType));
    pipe->InitBuffer(vBetaOutQue, 2, rowNum * V * sizeof(kType));

    vTensor.SetGlobalBuffer((__gm__ kType *)v);
    betaTensor.SetGlobalBuffer((__gm__ betaType *)beta);

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, BT, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h_v = 0; h_v < HV; h_v++) {
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto vOffset = (h_v * T + bos + rowOffset) * V;
                auto betaOffset = h_v * T + bos + rowOffset;
                // copyin
                {
                    auto tensorVin = vInQue.AllocTensor<kType>();
                    DataCopy(tensorVin, vTensor[vOffset], V * curRowNum);
                    auto tensorBetain = betaInQue.AllocTensor<betaType>();
                    DataCopyPad(tensorBetain, betaTensor[betaOffset],
                        {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                        {false, 0, 0, 0});
                    vInQue.EnQue(tensorVin);
                    betaInQue.EnQue(tensorBetain);
                }
                // compute
                {
                    auto tensorVin = vInQue.DeQue<kType>();
                    auto tensorBetain = betaInQue.DeQue<betaType>();
                    auto tensorOut = vBetaOutQue.AllocTensor<kType>();
                    auto vBetaOutAddr = reinterpret_cast<uint64_t>(tensorOut.GetPhyAddr());
                    auto vInAddr = reinterpret_cast<uint64_t>(tensorVin.GetPhyAddr());
                    auto betaInAddr = reinterpret_cast<uint64_t>(tensorBetain.GetPhyAddr());
                    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
                    if (V <= eleKNumPerVf) {
                        if (curRowNum == 1) {
                            ProcessVBetaComputerVFOneLineOneCol(
                                    (__ubuf__ kType*)vBetaOutAddr, (__ubuf__ kType*)vInAddr,
                                    (__ubuf__ betaType*)betaInAddr,
                                    curRowNum, V);
                        } else {
                            uint16_t lastLoopCnt = curRowNum % PRELOAD_NUM;
                            ProcessVBetaComputerVFMutiLineOneCol(
                                    (__ubuf__ kType*)vBetaOutAddr, (__ubuf__ kType*)vInAddr,
                                    (__ubuf__ betaType*)betaInAddr,
                                    curRowNum, V, lastLoopCnt);
                        }
                    } else {
                        ProcessVBetaComputerVFTwoCol(
                                    (__ubuf__ kType*)vBetaOutAddr, (__ubuf__ kType*)vInAddr,
                                    (__ubuf__ betaType*)betaInAddr,
                                    curRowNum, V);
                    }
                    vInQue.FreeTensor(tensorVin);
                    betaInQue.FreeTensor(tensorBetain);
                    vBetaOutQue.EnQue(tensorOut);
                }
                // copyout
                {
                    auto tensorOut = vBetaOutQue.DeQue<kType>();
                    DataCopy(workSpace2Tensor[vOffset], tensorOut, V * curRowNum);
                    vBetaOutQue.FreeTensor(tensorOut);
                }
            }
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
        }
    }
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessMDuDw() {
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = rowNumMDuDw;
    uint32_t rowOffset = 0;
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t curRowNum = rowNum;

    pipe->InitBuffer(mduInQue, 2, rowNum * BT * sizeof(kType));
    pipe->InitBuffer(mdwInQue, 2, rowNum * BT * sizeof(kType));
    pipe->InitBuffer(mduwOutQue, 2, rowNum * BT * sizeof(kType));

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, BT, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h_v = 0; h_v < HV; h_v++) {
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto offset = (h_v * T + bos + rowOffset) * BT;
                // copyin
                {
                    auto tensorMduin = mduInQue.AllocTensor<kType>();
                    DataCopy(tensorMduin, dA2Tensor[offset], curRowNum * BT);
                    auto tensorMdwin = mdwInQue.AllocTensor<kType>();
                    DataCopy(tensorMdwin, dA1Tensor[offset], curRowNum * BT);
                    mduInQue.EnQue(tensorMduin);
                    mdwInQue.EnQue(tensorMdwin);
                }
                // compute
                {
                    auto tensorMduin = mduInQue.DeQue<kType>();
                    auto tensorMdwin = mdwInQue.DeQue<kType>();
                    auto tensorMduwOut = mduwOutQue.AllocTensor<kType>();
                    auto mduwOutAddr = reinterpret_cast<uint64_t>(tensorMduwOut.GetPhyAddr());
                    auto mduInAddr = reinterpret_cast<uint64_t>(tensorMduin.GetPhyAddr());
                    auto mdwInAddr = reinterpret_cast<uint64_t>(tensorMdwin.GetPhyAddr());
                    if (curRowNum == 1) {
                        ProcessMDuDwComputerVFOneLineOneCol(
                                (__ubuf__ kType*)mduwOutAddr, (__ubuf__ kType*)mduInAddr,
                                (__ubuf__ kType*)mdwInAddr,
                                curRowNum, BT, rowOffset);
                    } else {
                        uint16_t lastLoopCnt = curRowNum % PRELOAD_NUM;
                        ProcessMDuDwComputerVFMutiLineOneCol(
                                (__ubuf__ kType*)mduwOutAddr, (__ubuf__ kType*)mduInAddr,
                                (__ubuf__ kType*)mdwInAddr,
                                curRowNum, BT, rowOffset, lastLoopCnt);
                    }
                    mduInQue.FreeTensor(tensorMduin);
                    mdwInQue.FreeTensor(tensorMdwin);
                    mduwOutQue.EnQue(tensorMduwOut);
                }
                //copyout
                {
                    auto tensorMduwOut = mduwOutQue.DeQue<kType>();
                    DataCopy(dA4Tensor[offset], tensorMduwOut, curRowNum * BT);
                    mduwOutQue.FreeTensor(tensorMduwOut);
                }
            }
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
        }
    }
    return;
}

// g_sub_exp的处理
template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessG() {
    Arch::Resource<Arch::Ascend950> resource;
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t rowNum = rowNumG;
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

    AscendC::LocalTensor<kType> tensorDA6InList[MAX_CUBE_VEC_SYNC_NUM];
    AscendC::LocalTensor<betaType> tensorGInList[UB_STAGES];
    AscendC::LocalTensor<betaType> tensorGAllInList[UB_STAGES];
    AscendC::LocalTensor<kType> tensorDAOutList[UB_STAGES];

    int32_t eventUbInVMTE2List[UB_STAGES];
    int32_t eventUbInMTE2VList[UB_STAGES];
    int32_t eventUbOutMTE3VList[UB_STAGES];
    int32_t eventUbOutVMTE3List[UB_STAGES];
    int32_t eventUbGAllInVMTE2List[UB_STAGES];
    int32_t eventUbGAllInMTE2VList[UB_STAGES];

    // CV buffers: dA6 from cube
    for (uint32_t i = 0; i < gCVNum; ++i) {
        tensorDA6InList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * BT * sizeof(kType);
    }

    // UB stage buffers
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        tensorGInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += rowNum * sizeof(betaType);
        tensorGAllInList[i] = resource.ubBuf.template GetBufferByByte<betaType>(ubOffset);
        ubOffset += BT * sizeof(betaType);
        tensorDAOutList[i] = resource.ubBuf.template GetBufferByByte<kType>(ubOffset);
        ubOffset += rowNum * BT * sizeof(kType);

        eventUbInVMTE2List[i] = eventVMTE2++;
        eventUbInMTE2VList[i] = eventMTE2V++;
        eventUbOutMTE3VList[i] = eventMTE3V++;
        eventUbOutVMTE3List[i] = eventVMTE3++;
        eventUbGAllInVMTE2List[i] = eventVMTE2++;
        eventUbGAllInMTE2VList[i] = eventMTE2V++;

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbGAllInVMTE2List[i]);
    }

    for (uint32_t i = 0; i < gCVNum; i++) {
        AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + i);
    }

    uint32_t ubGAllListId = 0;

    gTensor.SetGlobalBuffer((__gm__ betaType *)g);
    dATensor.SetGlobalBuffer((__gm__ kType *)dA);

    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, BT, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        for (int h_v = 0; h_v < HV; h_v++) {
            uint32_t taskNum = CeilDiv(curChunkSize, rowNum);
            taskNum = CeilDiv(taskNum, GetSubBlockNum());
            uint32_t dealTaskNum = 0;

            // copyin gAll [1, BT] once per head
            auto& tensorGAllIn = tensorGAllInList[ubGAllListId];
            {
                auto gAllOffset = h_v * T + bos;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbGAllInVMTE2List[ubGAllListId]);
                DataCopyPad(tensorGAllIn, gTensor[gAllOffset],
                    {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                    {false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbGAllInMTE2VList[ubGAllListId]);
            }
            int gAllInMTE2VNum = 1;

            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum) {
                ++vecTaskIdx;
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    continue;
                }
                curRowNum = (rowOffset + rowNum) > curChunkSize ? curChunkSize - rowOffset : rowNum;
                auto gOffset = h_v * T + bos + rowOffset;
                auto offset = (h_v * T + bos + rowOffset) * BT;

                auto& tensorDA6In = tensorDA6InList[cvListId];
                auto& tensorGIn = tensorGInList[ubListId];
                auto& tensorDAOut = tensorDAOutList[ubListId];
                // copyin g
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                    DataCopyPad(tensorGIn, gTensor[gOffset],
                        {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                        {false, 0, 0, 0});
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                }
                // compute
                {
                    auto dA6InAddr = reinterpret_cast<uint64_t>(tensorDA6In.GetPhyAddr());
                    auto gInAddr = reinterpret_cast<uint64_t>(tensorGIn.GetPhyAddr());
                    auto gAllInAddr = reinterpret_cast<uint64_t>(tensorGAllIn.GetPhyAddr());
                    auto dAOutAddr = reinterpret_cast<uint64_t>(tensorDAOut.GetPhyAddr());

                    if (rowOffset == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbGAllInMTE2VList[ubGAllListId]);
                        gAllInMTE2VNum--;
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbInMTE2VList[ubListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);

                    if (curRowNum == 1) {
                        ProcessGComputerVFOneLineOneCol(
                                (__ubuf__ kType*)dAOutAddr, (__ubuf__ kType*)dA6InAddr,
                                (__ubuf__ betaType*)gInAddr, (__ubuf__ betaType*)gAllInAddr,
                                curRowNum, BT, rowOffset, curChunkSize);
                    } else {
                        uint16_t lastLoopCnt = curRowNum % PRELOAD_NUM;
                        ProcessGComputerVFMutiLineOneCol(
                                (__ubuf__ kType*)dAOutAddr, (__ubuf__ kType*)dA6InAddr,
                                (__ubuf__ betaType*)gInAddr, (__ubuf__ betaType*)gAllInAddr,
                                curRowNum, BT, rowOffset, lastLoopCnt, curChunkSize);
                    }

                    AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[ubListId]);
                }
                // copyout
                {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbOutVMTE3List[ubListId]);
                    DataCopy(dATensor[offset], tensorDAOut, BT * curRowNum);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[ubListId]);
                }
                ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
                cvListId = (cvListId + 1 < gCVNum) ? (cvListId + 1) : 0;
                dealTaskNum++;
            }

            for (uint32_t taskId = dealTaskNum; taskId < taskNum; taskId++) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(SYNC_AIC_AIV_FLAG_BEGIN + cvListId);
                AscendC::CrossCoreSetFlag<0x4, PIPE_V>(SYNC_AIV_AIC_FLAG_BEGIN + cvListId);
                cvListId = (cvListId + 1 < gCVNum) ? (cvListId + 1) : 0;
            }

            if (gAllInMTE2VNum > 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbGAllInMTE2VList[ubGAllListId]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbGAllInVMTE2List[ubGAllListId]);
            ubGAllListId = (ubGAllListId + 1 < UB_STAGES) ? (ubGAllListId + 1) : 0;
        }
    }
    for (uint32_t i = 0; i < UB_STAGES; ++i) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbInVMTE2List[i]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbOutMTE3VList[i]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbGAllInVMTE2List[i]);
    }
    return;
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessKBetaGComputerVFOneLineOneCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    RegTensor<kType> kInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg;
    RegTensor<float> betaFP32Reg, gFP32Reg, betaGFP32Reg;
    RegTensor<kType> kBetaGOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<kType, false>(kInReg, kIn);

    HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
    HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
    Exp(gFP32Reg, gFP32Reg, maskFull32);
    Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

    CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
    MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
    CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
    StoreAlign(kBetaGOut, kBetaGOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessKBetaGComputerVFMutiLineOneCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<betaType> gInReg, gInReg1;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1;
    RegTensor<float> betaFP32Reg, betaFP32Reg1;
    RegTensor<float> gFP32Reg, gFP32Reg1;
    RegTensor<float> betaGFP32Reg, betaGFP32Reg1;
    RegTensor<kType> kBetaGOutReg, kBetaGOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(gInReg1, gIn + 1);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaFP32Reg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg1, gInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(gInReg1, gIn + (mIdx + 1) * PRELOAD_NUM + 1);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Exp(gFP32Reg1, gFP32Reg1, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg1, betaFP32Reg1, gFP32Reg1, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg1, betaGFP32Reg1, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaGOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaFP32Reg1, betaInReg1, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg1, gInReg1, maskFull16, maskFull32);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Exp(gFP32Reg1, gFP32Reg1, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg1, betaFP32Reg1, gFP32Reg1, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);

        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg1, betaGFP32Reg1, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), kBetaGOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<betaType, true>(gInReg, gIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(kInReg, kIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
            HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
            Exp(gFP32Reg, gFP32Reg, maskFull32);
            Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

            CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
            MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
            CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * PRELOAD_NUM), kBetaGOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessKBetaGComputerVFTwoCol(
    __ubuf__ kType* kBetaGOut, __ubuf__ kType* kIn, __ubuf__ betaType* betaIn,
    __ubuf__ betaType* gIn, uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> kInReg, kInReg1;
    RegTensor<betaType> betaInReg;
    RegTensor<betaType> gInReg;
    RegTensor<float> kFP32ZeroReg, kFP32OneReg, kFP32ZeroReg1, kFP32OneReg1;
    RegTensor<float> kBetaGFP32ZeroReg, kBetaGFP32OneReg, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1;
    RegTensor<float> betaFP32Reg, gFP32Reg, betaGFP32Reg;
    RegTensor<kType> kBetaGOutReg, kBetaGOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<kType, false>(kInReg, kIn);
    LoadIn<kType, false>(kInReg1, kIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1));
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        LoadIn<kType, false>(kInReg, kIn + oneEleNum * ((mIdx + 1) * 2));
        LoadIn<kType, false>(kInReg1, kIn + oneEleNum * ((mIdx + 1) * 2 + 1));

        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg, betaGFP32Reg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2 + 1), kBetaGOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32Reg, gInReg, maskFull16, maskFull32);
        Exp(gFP32Reg, gFP32Reg, maskFull32);
        Mul(betaGFP32Reg, betaFP32Reg, gFP32Reg, maskFull32);

        CastHalf2Float<kType>(kFP32ZeroReg, kFP32OneReg, kInReg, maskFull16);
        CastHalf2Float<kType>(kFP32ZeroReg1, kFP32OneReg1, kInReg1, maskFull16);
        MulFloatTwoReg(kBetaGFP32ZeroReg, kBetaGFP32OneReg, kFP32ZeroReg, kFP32OneReg, betaGFP32Reg, betaGFP32Reg, maskFull32);
        MulFloatTwoReg(kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, kFP32ZeroReg1, kFP32OneReg1, betaGFP32Reg, betaGFP32Reg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg, kBetaGFP32ZeroReg, kBetaGFP32OneReg, maskFull32);
        CastFloat2Half<kType>(kBetaGOutReg1, kBetaGFP32ZeroReg1, kBetaGFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2), kBetaGOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)kBetaGOut + oneEleNum * (mIdx * 2 + 1), kBetaGOutReg1, maskFull16);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessVBetaComputerVFOneLineOneCol(
    __ubuf__ kType* vBetaOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    RegTensor<kType> vInReg;
    RegTensor<betaType> betaInReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg;
    RegTensor<float> vBetaFP32ZeroReg, vBetaFP32OneReg;
    RegTensor<float> betaBrcbFP32Reg;
    RegTensor<kType> vBetaOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);

    HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
    CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
    MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
    CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
    StoreAlign(vBetaOut, vBetaOutReg, maskFull16);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessVBetaComputerVFMutiLineOneCol(
    __ubuf__ kType* vBetaOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<betaType> betaInReg, betaInReg1;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> vBetaFP32ZeroReg, vBetaFP32OneReg, vBetaFP32ZeroReg1, vBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32Reg, betaBrcbFP32Reg1;
    RegTensor<kType> vBetaOutReg, vBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<betaType, true>(betaInReg1, betaIn + 1);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg1, betaInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(betaInReg1, betaIn + (mIdx + 1) * PRELOAD_NUM + 1);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg1, betaBrcbFP32Reg1, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), vBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg1, betaInReg1, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg1, betaBrcbFP32Reg1, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), vBetaOutReg1, maskFull16);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<betaType, true>(betaInReg, betaIn + mIdx * PRELOAD_NUM);
            LoadIn<kType, false>(vInReg, vIn + oneEleNum * (mIdx * PRELOAD_NUM));
            HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
            CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
            MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
            CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
            StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * PRELOAD_NUM), vBetaOutReg, maskFull16);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessVBetaComputerVFTwoCol(
    __ubuf__ kType* vBetaOut, __ubuf__ kType* vIn, __ubuf__ betaType* betaIn,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize - 1;
    RegTensor<kType> vInReg, vInReg1;
    RegTensor<betaType> betaInReg;
    RegTensor<float> vFP32ZeroReg, vFP32OneReg, vFP32ZeroReg1, vFP32OneReg1;
    RegTensor<float> vBetaFP32ZeroReg, vBetaFP32OneReg, vBetaFP32ZeroReg1, vBetaFP32OneReg1;
    RegTensor<float> betaBrcbFP32Reg;
    RegTensor<kType> vBetaOutReg, vBetaOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    LoadIn<betaType, true>(betaInReg, betaIn);
    LoadIn<kType, false>(vInReg, vIn);
    LoadIn<kType, false>(vInReg1, vIn + oneEleNum);

    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        LoadIn<betaType, true>(betaInReg, betaIn + (mIdx + 1));
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        LoadIn<kType, false>(vInReg, vIn + oneEleNum * ((mIdx + 1) * 2));
        LoadIn<kType, false>(vInReg1, vIn + oneEleNum * ((mIdx + 1) * 2 + 1));
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2 + 1), vBetaOutReg1, maskFull16);
    }
    uint16_t mIdx = mLoopCnt;
    {
        HalfOrFloat2Float<betaType>(betaBrcbFP32Reg, betaInReg, maskFull16, maskFull32);
        CastHalf2Float<kType>(vFP32ZeroReg, vFP32OneReg, vInReg, maskFull16);
        CastHalf2Float<kType>(vFP32ZeroReg1, vFP32OneReg1, vInReg1, maskFull16);
        MulFloatTwoReg(vBetaFP32ZeroReg, vBetaFP32OneReg, vFP32ZeroReg, vFP32OneReg, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        MulFloatTwoReg(vBetaFP32ZeroReg1, vBetaFP32OneReg1, vFP32ZeroReg1, vFP32OneReg1, betaBrcbFP32Reg, betaBrcbFP32Reg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg, vBetaFP32ZeroReg, vBetaFP32OneReg, maskFull32);
        CastFloat2Half<kType>(vBetaOutReg1, vBetaFP32ZeroReg1, vBetaFP32OneReg1, maskFull32);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2), vBetaOutReg, maskFull16);
        StoreAlign((__ubuf__ kType*&)vBetaOut + oneEleNum * (mIdx * 2 + 1), vBetaOutReg1, maskFull16);
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessMDuDwComputerVFOneLineOneCol(
    __ubuf__ kType* mduwOut, __ubuf__ kType* mduIn, __ubuf__ kType* mdwIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    RegTensor<kType> mduInReg, mdwInReg;
    RegTensor<float> mduFP32ZeroReg, mduFP32OneReg;
    RegTensor<float> mdwFP32ZeroReg, mdwFP32OneReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg, zeroReg, rowIdxReg;
    RegTensor<kType> mduwOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();

    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskStore;
    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    LoadIn<kType, false>(mduInReg, mduIn);
    LoadIn<kType, false>(mdwInReg, mdwIn);
    Duplicate(rowIdxReg, static_cast<float>(startRow));
    CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
    CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
    AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);

    CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
    SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);

    CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
    uint32_t storeLen = oneEleNum;
    maskStore = UpdateMask<kType>(storeLen);
    StoreAlign(mduwOut, mduwOutReg, maskStore);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessMDuDwComputerVFMutiLineOneCol(
    __ubuf__ kType* mduwOut, __ubuf__ kType* mduIn, __ubuf__ kType* mdwIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow, uint16_t lastLoopCnt)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> mduInReg, mduInReg1;
    RegTensor<kType> mdwInReg, mdwInReg1;
    RegTensor<float> mduFP32ZeroReg, mduFP32OneReg, mduFP32ZeroReg1, mduFP32OneReg1;
    RegTensor<float> mdwFP32ZeroReg, mdwFP32OneReg, mdwFP32ZeroReg1, mdwFP32OneReg1;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg, resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, rowIdxReg;
    RegTensor<kType> mduwOutReg, mduwOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskZeroSelect1, maskOneSelect1;
    MaskReg maskStore;
    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    LoadIn<kType, false>(mduInReg, mduIn);
    LoadIn<kType, false>(mduInReg1, mduIn + oneEleNum);
    LoadIn<kType, false>(mdwInReg, mdwIn);
    LoadIn<kType, false>(mdwInReg1, mdwIn + oneEleNum);

    Duplicate(rowIdxReg, static_cast<float>(startRow));
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
        CastHalf2Float<kType>(mduFP32ZeroReg1, mduFP32OneReg1, mduInReg1, maskFull16);
        LoadIn<kType, false>(mduInReg, mduIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(mduInReg1, mduIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
        CastHalf2Float<kType>(mdwFP32ZeroReg1, mdwFP32OneReg1, mdwInReg1, maskFull16);
        LoadIn<kType, false>(mdwInReg, mdwIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(mdwInReg1, mdwIn + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));

        AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);
        AddFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, mduFP32ZeroReg1, mduFP32OneReg1, mdwFP32ZeroReg1, mdwFP32OneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(mduwOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);
        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM), mduwOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), mduwOutReg1, maskStore);
    }
    uint16_t mIdx = mLoopCnt;
    {
        CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
        CastHalf2Float<kType>(mduFP32ZeroReg1, mduFP32OneReg1, mduInReg1, maskFull16);
        CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
        CastHalf2Float<kType>(mdwFP32ZeroReg1, mdwFP32OneReg1, mdwInReg1, maskFull16);

        AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);
        AddFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, mduFP32ZeroReg1, mduFP32OneReg1, mdwFP32ZeroReg1, mdwFP32OneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(mduwOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);
        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM), mduwOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), mduwOutReg1, maskStore);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(mduInReg, mduIn + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<kType, false>(mdwInReg, mdwIn + oneEleNum * (mIdx * PRELOAD_NUM));
            CastHalf2Float<kType>(mduFP32ZeroReg, mduFP32OneReg, mduInReg, maskFull16);
            CastHalf2Float<kType>(mdwFP32ZeroReg, mdwFP32OneReg, mdwInReg, maskFull16);
            AddFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, mduFP32ZeroReg, mduFP32OneReg, mdwFP32ZeroReg, mdwFP32OneReg, maskFull32);

            CompareTwoReg<float, CMPMODE::GE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
            SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);

            CastFloat2Half<kType>(mduwOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
            uint32_t storeLen = oneEleNum;
            maskStore = UpdateMask<kType>(storeLen);
            StoreAlign((__ubuf__ kType*&)mduwOut + oneEleNum * (mIdx * PRELOAD_NUM), mduwOutReg, maskStore);
        }
    }
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessGComputerVFOneLineOneCol(
    __ubuf__ kType* dAOut, __ubuf__ kType* dA6In, __ubuf__ betaType* gIn, __ubuf__ betaType* gAllIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow, uint32_t calcColSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> dA6InReg;
    RegTensor<betaType> gInReg, gAllInReg;
    RegTensor<float> dA6FP32ZeroReg, dA6FP32OneReg;
    RegTensor<float> gFP32ZeroReg;
    RegTensor<float> gFactorZeroReg, gFactorOneReg;
    RegTensor<float> gAllFP32ZeroReg, gAllFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> zeroReg, negOneReg, rowIdxReg;
    RegTensor<kType> dAOutReg;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskStore;
    MaskReg castDA6MaskCount16;
    uint32_t castDA6Count = calcColSize;
    castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);

    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Duplicate(negOneReg, static_cast<float>(-1), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    if constexpr (!std::is_same<betaType, float>()) {
        LoadAlign(gAllInReg, gAllIn);
        Cast<float, betaType, ctHalf2Fp32Zero>(gAllFP32ZeroReg, gAllInReg, maskFull16);
        Cast<float, betaType, ctHalf2Fp32One>(gAllFP32OneReg, gAllInReg, maskFull16);
    } else {
        LoadAlign<betaType, LoadDist::DIST_DINTLV_B32>(gAllFP32ZeroReg, gAllFP32OneReg, gAllIn);
    }
    LoadIn<kType, false>(dA6InReg, dA6In);
    LoadIn<betaType, true>(gInReg, gIn);

    Duplicate(rowIdxReg, static_cast<float>(startRow));
    CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
    HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);

    SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
    MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
    ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

    // result = -dA6 * gAll
    MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
    MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

    CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
    SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
    Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

    CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
    uint32_t storeLen = oneEleNum;
    maskStore = UpdateMask<kType>(storeLen);
    StoreAlign((__ubuf__ kType*&)dAOut, dAOutReg, maskStore);
}

template <typename kType, typename betaType>
__simd_vf__ inline void PrepareWyReprBwdDAVectorProcess<kType, betaType>::ProcessGComputerVFMutiLineOneCol(
    __ubuf__ kType* dAOut, __ubuf__ kType* dA6In, __ubuf__ betaType* gIn, __ubuf__ betaType* gAllIn,
    uint16_t mSize, uint16_t nSize, uint32_t startRow, uint16_t lastLoopCnt, uint32_t calcColSize)
{
    uint32_t eleKNumPerVf = AscendC::VECTOR_REG_WIDTH / sizeof(kType);
    uint32_t oneEleNum = min(eleKNumPerVf, nSize);
    uint16_t mLoopCnt = mSize / PRELOAD_NUM - 1;
    RegTensor<kType> dA6InReg, dA6InReg1;
    RegTensor<betaType> gInReg, gInReg1, gAllInReg;
    RegTensor<float> dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg1, dA6FP32OneReg1;
    RegTensor<float> gFP32ZeroReg, gFP32ZeroReg1;
    RegTensor<float> gFactorZeroReg, gFactorOneReg, gFactorZeroReg1, gFactorOneReg1;
    RegTensor<float> gAllFP32ZeroReg, gAllFP32OneReg;
    RegTensor<float> resultFP32ZeroReg, resultFP32OneReg, resultFP32ZeroReg1, resultFP32OneReg1;
    RegTensor<half> colIdxReg;
    RegTensor<float> colIdxFP32ZeroReg, colIdxFP32OneReg;
    RegTensor<float> zeroReg, negOneReg, rowIdxReg;
    RegTensor<kType> dAOutReg, dAOutReg1;

    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskFull16 = CreateMask<half, MaskPattern::ALL>();
    MaskReg maskZeroSelect, maskOneSelect;
    MaskReg maskZeroSelect1, maskOneSelect1;
    MaskReg maskStore;
    MaskReg castDA6MaskCount16;

    Duplicate(zeroReg, static_cast<float>(0), maskFull32);
    Duplicate(negOneReg, static_cast<float>(-1), maskFull32);
    Arange(colIdxReg, 0);
    CastHalf2Float<half>(colIdxFP32ZeroReg, colIdxFP32OneReg, colIdxReg, maskFull16);

    if constexpr (!std::is_same<betaType, float>()) {
        LoadAlign(gAllInReg, gAllIn);
        Cast<float, betaType, ctHalf2Fp32Zero>(gAllFP32ZeroReg, gAllInReg, maskFull16);
        Cast<float, betaType, ctHalf2Fp32One>(gAllFP32OneReg, gAllInReg, maskFull16);
    } else {
        LoadAlign<betaType, LoadDist::DIST_DINTLV_B32>(gAllFP32ZeroReg, gAllFP32OneReg, gAllIn);
    }
    LoadIn<kType, false>(dA6InReg, dA6In);
    LoadIn<kType, false>(dA6InReg1, dA6In + oneEleNum);
    LoadIn<betaType, true>(gInReg, gIn);
    LoadIn<betaType, true>(gInReg1, gIn + 1);

    Duplicate(rowIdxReg, static_cast<float>(startRow));
    for (uint16_t mIdx = 0; mIdx < mLoopCnt; mIdx++) {
        uint32_t castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
        castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6InReg1, castDA6MaskCount16);
        LoadIn<kType, false>(dA6InReg, dA6In + oneEleNum * ((mIdx + 1) * PRELOAD_NUM));
        LoadIn<kType, false>(dA6InReg1, dA6In + oneEleNum * ((mIdx + 1) * PRELOAD_NUM + 1));
        HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32ZeroReg1, gInReg1, maskFull16, maskFull32);
        LoadIn<betaType, true>(gInReg, gIn + (mIdx + 1) * PRELOAD_NUM);
        LoadIn<betaType, true>(gInReg1, gIn + (mIdx + 1) * PRELOAD_NUM + 1);

        SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
        SubFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg1, gFP32ZeroReg1, maskFull32);
        MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
        MinsFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, float(0.0), maskFull32);
        ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        ExpFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        // result = -dA6 * gAll
        MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        MulFloatTwoReg(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dAOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);
        
        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM), dAOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dAOutReg1, maskStore);
    }
    uint16_t mIdx = mLoopCnt;
    {
        uint32_t castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
        castDA6Count = calcColSize;
        castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
        CastHalf2Float<kType>(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6InReg1, castDA6MaskCount16);
        HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);
        HalfOrFloat2Float<betaType>(gFP32ZeroReg1, gInReg1, maskFull16, maskFull32);

        SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
        SubFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg1, gFP32ZeroReg1, maskFull32);
        MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
        MinsFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, float(0.0), maskFull32);
        ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        ExpFloatTwoReg(gFactorZeroReg1, gFactorOneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        // result = -dA6 * gAll
        MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);
        MulFloatTwoReg(dA6FP32ZeroReg1, dA6FP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, negOneReg, negOneReg, maskFull32);
        MulFloatTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, dA6FP32ZeroReg1, dA6FP32OneReg1, gFactorZeroReg1, gFactorOneReg1, maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect1, maskOneSelect1, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
        SelectTwoReg(resultFP32ZeroReg1, resultFP32OneReg1, zeroReg, zeroReg, resultFP32ZeroReg1, resultFP32OneReg1, maskZeroSelect1, maskOneSelect1);
        Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

        CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
        CastFloat2Half<kType>(dAOutReg1, resultFP32ZeroReg1, resultFP32OneReg1, maskFull32);
        uint32_t storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM), dAOutReg, maskStore);
        storeLen = oneEleNum;
        maskStore = UpdateMask<kType>(storeLen);
        StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM + 1), dAOutReg1, maskStore);

        mIdx += 1;
        for (uint16_t i = 0; i < lastLoopCnt; i++) {
            LoadIn<kType, false>(dA6InReg, dA6In + oneEleNum * (mIdx * PRELOAD_NUM));
            LoadIn<betaType, true>(gInReg, gIn + mIdx * PRELOAD_NUM);

            uint32_t castDA6Count = calcColSize;
            castDA6MaskCount16 = UpdateMask<kType>(castDA6Count);
            CastHalf2Float<kType>(dA6FP32ZeroReg, dA6FP32OneReg, dA6InReg, castDA6MaskCount16);
            HalfOrFloat2Float<betaType>(gFP32ZeroReg, gInReg, maskFull16, maskFull32);

            SubFloatTwoReg(gFactorZeroReg, gFactorOneReg, gAllFP32ZeroReg, gAllFP32OneReg, gFP32ZeroReg, gFP32ZeroReg, maskFull32);
            MinsFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, float(0.0), maskFull32);
            ExpFloatTwoReg(gFactorZeroReg, gFactorOneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

            // result = -dA6 * gAll
            MulFloatTwoReg(dA6FP32ZeroReg, dA6FP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, negOneReg, negOneReg, maskFull32);
            MulFloatTwoReg(resultFP32ZeroReg, resultFP32OneReg, dA6FP32ZeroReg, dA6FP32OneReg, gFactorZeroReg, gFactorOneReg, maskFull32);

            CompareTwoReg<float, CMPMODE::LE>(maskZeroSelect, maskOneSelect, colIdxFP32ZeroReg, colIdxFP32OneReg, rowIdxReg, rowIdxReg, maskFull32);
            SelectTwoReg(resultFP32ZeroReg, resultFP32OneReg, zeroReg, zeroReg, resultFP32ZeroReg, resultFP32OneReg, maskZeroSelect, maskOneSelect);
            Adds(rowIdxReg, rowIdxReg, static_cast<float>(1), maskFull32);

            CastFloat2Half<kType>(dAOutReg, resultFP32ZeroReg, resultFP32OneReg, maskFull32);
            uint32_t storeLen = oneEleNum;
            maskStore = UpdateMask<kType>(storeLen);
            StoreAlign((__ubuf__ kType*&)dAOut + oneEleNum * (mIdx * PRELOAD_NUM), dAOutReg, maskStore);
        }
    }
}

#endif  // PREPARE_WY_REPR_BWD_DA_VECTOR_H
