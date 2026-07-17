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

using namespace AscendC;

template <typename kType, typename betaType>
class PrepareWyReprBwdFullVectorProcess {
public:
    /** @brief constructor */
    __aicore__ inline PrepareWyReprBwdFullVectorProcess(GM_ADDR k_, GM_ADDR v_, GM_ADDR beta_, GM_ADDR A_, GM_ADDR dA_,
                                                        GM_ADDR dw_, GM_ADDR du_, GM_ADDR g_, GM_ADDR cu_seqlens_,
                                                        GM_ADDR chunk_indices_, GM_ADDR dk_, GM_ADDR dv_,
                                                        GM_ADDR dbeta_, GM_ADDR dg_, GM_ADDR workspace_);

    __aicore__ inline void Process();
    __aicore__ inline void ProcessFused();
    __aicore__ inline void Init(const PrepareWyReprBwdFullTilingData &tiling, AscendC::TPipe *pipe_);

private:
    uint64_t B = 0;
    uint64_t T = 0;
    uint64_t HV = 0;
    uint64_t HK = 0;
    uint64_t K = 0;
    uint64_t V = 0;
    uint64_t chunkSize = 0;
    uint64_t chunkNum = 0;
    uint64_t fusedKVecRow = 0;
    uint64_t fusedVVecRow = 0;
    uint64_t fusedKktVecRow = 0;
    uint64_t workspaceSlotSize = 0;
    uint64_t workspaceBufferCount = 1;
    uint64_t usedCoreNum = 0;

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
    // GlobalTensor<uint64_t> cuSeqlensTensor;
    // GlobalTensor<uint64_t> chunkIndicesTensor;
    GlobalTensor<betaType> dgTensor;
    GlobalTensor<kType> dATensor;
    GlobalTensor<kType> workSpaceTensor;
    Arch::CrossCoreFlag flagAicFinishStore{SYNC_FLAG_2};
    Arch::CrossCoreFlagWithReverse<> flagAivFinishStore{SYNC_FLAG_4, SYNC_FLAG_5};
    Arch::CrossCoreFlag flagWorkspaceFree[2] = {6, 7};

    TQue<AscendC::TPosition::VECIN, 1> oneInQue;
    TQue<AscendC::TPosition::VECOUT, 1> oneOutQue;

    TBuf<AscendC::TPosition::VECCALC> persistentArena;
    TBuf<AscendC::TPosition::VECCALC> tempArena;
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
    const PrepareWyReprBwdFullTilingData &tiling, AscendC::TPipe *pipe_)
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
    K = tiling.K;
    V = tiling.V;
    chunkSize = tiling.chunkSize;
    chunkNum = tiling.chunkNum;
    fusedKVecRow = tiling.fusedKVecRow;
    fusedVVecRow = tiling.fusedVVecRow;
    fusedKktVecRow = tiling.fusedKktVecRow;
    workspaceSlotSize = tiling.workspaceSlotSize;
    workspaceBufferCount = tiling.workspaceBufferCount;
    usedCoreNum = tiling.usedCoreNum;

    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::Process()
{
    ProcessFused();
    return;
}

template <typename kType, typename betaType>
__aicore__ void inline PrepareWyReprBwdFullVectorProcess<kType, betaType>::ProcessFused()
{
    uint32_t coreLoops = chunkNum;
    uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
    uint32_t coreNumAic = GetBlockNum();
    uint32_t subBlockIdx = GetSubBlockIdx();
    uint32_t subBlockNum = GetSubBlockNum();
    uint32_t vecTaskIdx = 0;
    uint32_t bos = 0;
    uint32_t eos = 0;
    uint32_t rowK = fusedKVecRow;
    uint32_t rowV = fusedVVecRow;
    uint32_t rowKkt = fusedKktVecRow;
    uint32_t rowOwned = rowK < rowV ? rowK : rowV;
    uint64_t maxDim = K > V ? K : V;
    maxDim = maxDim > chunkSize ? maxDim : chunkSize;
    uint64_t maxRow = rowK > rowV ? rowK : rowV;
    maxRow = maxRow > rowKkt ? maxRow : rowKkt;
    uint64_t align = ONE_BLOCK_32;

    // Workspace slot offsets must match tiling.cpp.  kBytes/vBytes are
    // rounded to 32B because AIC and AIV exchange all intermediate matrices
    // through GM workspace and both sides use aligned GM copies.
    uint64_t kBytes = ((chunkSize * K * sizeof(kType) + align - 1) / align) * align;
    uint64_t vBytes = ((chunkSize * V * sizeof(kType) + align - 1) / align) * align;
    uint64_t maxKVBytes = kBytes > vBytes ? kBytes : vBytes;

    // One input queue and one output queue are intentionally reused by all
    // logical tensors.  The size is the largest tile that can be copied by
    // this vector kernel: K/V tiles or a beta/g chunk vector.
    uint64_t queueBytes = maxRow * maxDim * sizeof(kType);
    uint64_t betaChunkBytes = chunkSize * sizeof(betaType);
    queueBytes = queueBytes > betaChunkBytes ? queueBytes : betaChunkBytes;
    pipe->InitBuffer(oneInQue, 1, queueBytes);
    pipe->InitBuffer(oneOutQue, 1, queueBytes);

    // persistentArena stores values with chunk lifetime:
    //   beta fp32       : beta cast/copy to FP32
    //   exp(g) fp32     : g cast/copy to FP32, then Exp in place
    //   dbeta accumulator: dbeta partial sum across all formula terms
    //   dg accumulator   : dg partial sum across all formula terms
    // tempArena is reused by the current stage only.
    uint64_t persistentStride = ((chunkSize * sizeof(float32_t) + align - 1) / align) * align / sizeof(float32_t);
    uint64_t persistentBytes = 4 * persistentStride * sizeof(float32_t);
    uint64_t group1Bytes = 5 * rowK * K * sizeof(float32_t) + 3 * rowK * ONE_BLOCK_32;
    uint64_t group2VBytes = 2 * rowV * V * sizeof(float32_t) + 2 * rowV * ONE_BLOCK_32;
    uint64_t group2KktBytes = chunkSize * chunkSize * sizeof(float32_t) +
                              2 * rowKkt * chunkSize * sizeof(float32_t) + chunkSize * ONE_BLOCK_32;
    uint64_t tempBytes = group1Bytes > group2VBytes ? group1Bytes : group2VBytes;
    tempBytes = tempBytes > group2KktBytes ? tempBytes : group2KktBytes;
    pipe->InitBuffer(persistentArena, persistentBytes);
    pipe->InitBuffer(tempArena, tempBytes + ONE_BLOCK_32);

    auto persistent = persistentArena.Get<float32_t>();
    auto temp = tempArena.Get<float32_t>();
    auto tensorBetaFP32 = persistent[0];
    auto tensorGExpFP32 = persistent[persistentStride];
    auto tensorDbetaAccFP32 = persistent[2 * persistentStride];
    auto tensorDgAccFP32 = persistent[3 * persistentStride];

    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagWorkspaceFree[0]);
    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagWorkspaceFree[1]);

    // Outer loop distributes chunks across AIC/AIV core groups; inner loop
    // iterates heads for the same chunk.  For each (chunk, head), AIV performs
    // all vector work and consumes AIC results before moving to the next head.
    for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNumAic) {
        GetChunkOffset(cu_seqlens, chunk_indices, B, HV, T, chunkSize, loopIdx, bos, eos);
        uint32_t curChunkSize = eos - bos;
        uint32_t wholeReduceKCnt = CeilDiv(K, FP32_PER_REPEAT_64);
        uint32_t wholeReduceVCnt = CeilDiv(V, FP32_PER_REPEAT_64);
        uint32_t wholeReduceChunkCnt = CeilDiv(curChunkSize, FP32_PER_REPEAT_64);
        // GQA uses kh = h / (HV / HK) for k/dk, while the rest of the tensors
        // are still indexed by value head h.
        uint64_t keyBos = bos;
        if (cu_seqlens == nullptr && HV != HK) {
            uint64_t batchIdx = bos / (HV * T);
            uint64_t timeBos = bos - batchIdx * HV * T;
            keyBos = batchIdx * HK * T + timeBos;
        }
        uint64_t headRatio = HV / HK;
        for (uint32_t h = 0; h < HV; h++) {
            // Ping-pong slot selected by logical task order.  AIC and AIV use
            // the same formula, so both sides agree on which slot stores DK,
            // Dkb, Dkbg, Kbeta/Dvb and KKT for this (chunk, head).
            uint64_t workspaceBufferIdx = vecTaskIdx % workspaceBufferCount;
            uint64_t slotBaseBytes = (coreIdx * workspaceBufferCount + workspaceBufferIdx) * workspaceSlotSize;
            uint64_t slotDK = slotBaseBytes / sizeof(kType);
            uint64_t slotDkb = (slotBaseBytes + kBytes) / sizeof(kType);
            uint64_t slotDkbg = (slotBaseBytes + 2 * kBytes) / sizeof(kType);
            uint64_t slotKbetaDvb = (slotBaseBytes + 3 * kBytes) / sizeof(kType);
            uint64_t slotKKT = (slotBaseBytes + 3 * kBytes + maxKVBytes) / sizeof(kType);
            ++vecTaskIdx;

            uint64_t kh = h / headRatio;
            uint64_t valueBase = h * T + bos;
            uint64_t keyBase = kh * T + keyBos;
            bool isFirstValueHeadInGroup = (h % headRatio) == 0;
            uint64_t betaOffset = valueBase;
            uint64_t kBaseOffset = keyBase * K;
            uint64_t vBaseOffset = valueBase * V;
            uint64_t daBaseOffset = valueBase * chunkSize;

            {
                // Load beta[chunk] from GM to UB and normalize it to FP32.
                // beta is reused by Kbeta = K * beta, dk, dv, dbeta and dg.
                auto tensorIn = oneInQue.AllocTensor<betaType>();
                DataCopyPad(tensorIn, betaTensor[betaOffset],
                            {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                            {false, 0, 0, 0});
                oneInQue.EnQue(tensorIn);
                auto tensorBeta = oneInQue.DeQue<betaType>();
                if constexpr (std::is_same<betaType, float32_t>()) {
                    // Use Adds(x, 0) instead of UB->UB DataCopy so varlen tail
                    // chunks with non-32B element counts are still legal.
                    Adds(tensorBetaFP32, tensorBeta, 0.0f, curChunkSize);
                } else {
                    Cast(tensorBetaFP32, tensorBeta, RoundMode::CAST_NONE, curChunkSize);
                }
                oneInQue.FreeTensor(tensorBeta);
            }
            {
                // Load g[chunk], cast/copy to FP32, then compute exp(g) in UB.
                // exp(g) participates in Dkbg-related dbeta/dg/dk terms.
                auto tensorIn = oneInQue.AllocTensor<betaType>();
                DataCopyPad(tensorIn, gTensor[betaOffset],
                            {1, curChunkSize * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0},
                            {false, 0, 0, 0});
                oneInQue.EnQue(tensorIn);
                auto tensorG = oneInQue.DeQue<betaType>();
                if constexpr (std::is_same<betaType, float32_t>()) {
                    // Same non-aligned varlen tail consideration as beta.
                    Adds(tensorGExpFP32, tensorG, 0.0f, curChunkSize);
                } else {
                    Cast(tensorGExpFP32, tensorG, RoundMode::CAST_NONE, curChunkSize);
                }
                oneInQue.FreeTensor(tensorG);
            }
            PipeBarrier<PIPE_V>();
            Exp(tensorGExpFP32, tensorGExpFP32, curChunkSize);
            // Accumulators are initialized once for the whole (chunk, head)
            // and updated by later row tiles.
            Duplicate(tensorDbetaAccFP32, 0.0f, curChunkSize);
            Duplicate(tensorDgAccFP32, 0.0f, curChunkSize);
            PipeBarrier<PIPE_V>();

            // Stage V0: generate Kbeta = K * beta[:, None].
            // Each AIV sub-block owns interleaved row tiles:
            //   subBlock 0: rows [0, rowOwned), [2*rowOwned, 3*rowOwned), ...
            //   subBlock 1: rows [rowOwned, 2*rowOwned), ...
            // beta is Brcb-broadcast from [row] scalars to row-wise 8-lane
            // blocks so one vector Mul can apply beta to K columns.
            for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                 rowOffset += rowOwned * subBlockNum) {
                uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                auto tensorKFp32 = temp;
                auto tensorBetaBrcbFP32 = temp[rowK * K];
                auto tensorIn = oneInQue.AllocTensor<kType>();
                DataCopy(tensorIn, kTensor[kBaseOffset + rowOffset * K], K * curRowNum);
                oneInQue.EnQue(tensorIn);
                auto tensorKIn = oneInQue.DeQue<kType>();
                Cast(tensorKFp32, tensorKIn, RoundMode::CAST_NONE, K * curRowNum);
                oneInQue.FreeTensor(tensorKIn);
                PipeBarrier<PIPE_V>();
                // Broadcast beta[rowOffset:rowOffset+curRowNum] to FP32
                // blocks; it is used as beta[:, None] in K * beta.
                Brcb(tensorBetaBrcbFP32, tensorBetaFP32[rowOffset], static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                PipeBarrier<PIPE_V>();
                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                while (perchannelResOffset < K) {
                    Mul(tensorKFp32[perchannelResOffset], tensorKFp32[perchannelResOffset], tensorBetaBrcbFP32,
                        FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                PipeBarrier<PIPE_V>();
                auto tensorOut = oneOutQue.AllocTensor<kType>();
                Cast(tensorOut, tensorKFp32, RoundMode::CAST_RINT, K * curRowNum);
                oneOutQue.EnQue(tensorOut);
                auto tensorKbeta = oneOutQue.DeQue<kType>();
                DataCopy(workSpaceTensor[slotKbetaDvb + rowOffset * K], tensorKbeta, K * curRowNum);
                oneOutQue.FreeTensor(tensorKbeta);
            }

            // Tell AIC that Kbeta is ready in workspace; AIC can now compute
            // DK = dA @ Kbeta.  Then wait for AIC's first ready event:
            // Dkb = dA^T @ K.
            Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore);
            Arch::CrossCoreWaitFlag(flagAicFinishStore);

            bool waitedDkbgReady = false;
            bool waitedDKReady = false;

            // Stage V1: consume Dkb/Dkbg/DK in row tiles and produce dk plus
            // dbeta/dg partials:
            //   dbeta += reduce(Dkb * K)
            //   Dkb    = Dkb * beta[:, None]
            //   Dkbg   = (A^T @ dw) * exp(g)[:, None]
            //   dbeta += reduce(Dkbg * K)
            //   dg    += reduce(Dkbg * K) * beta
            //   Dkbg   = Dkbg * beta[:, None]
            //   dk     = DK + Dkb + Dkbg
            for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                 rowOffset += rowOwned * subBlockNum) {
                uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                uint64_t rowElem = rowK * K;
                auto tensorDkbFp32 = temp;
                auto tensorDkbgFp32 = temp[rowElem];
                auto tensorKFp32 = temp[2 * rowElem];
                auto tensorProductFp32 = temp[3 * rowElem];
                auto tensorDKFp32 = temp[4 * rowElem];
                auto tensorBetaBrcbFP32 = temp[5 * rowElem];
                auto tensorGbrcbFP32 = tensorBetaBrcbFP32[rowK * FP32_PER_BLOCK_8];
                auto tensorReduceSum = tensorGbrcbFP32[rowK * FP32_PER_BLOCK_8];

                auto tensorIn = oneInQue.AllocTensor<kType>();
                DataCopy(tensorIn, workSpaceTensor[slotDkb + rowOffset * K], K * curRowNum);
                oneInQue.EnQue(tensorIn);
                auto tensorDkbIn = oneInQue.DeQue<kType>();
                Cast(tensorDkbFp32, tensorDkbIn, RoundMode::CAST_NONE, K * curRowNum);
                oneInQue.FreeTensor(tensorDkbIn);

                tensorIn = oneInQue.AllocTensor<kType>();
                DataCopy(tensorIn, kTensor[kBaseOffset + rowOffset * K], K * curRowNum);
                oneInQue.EnQue(tensorIn);
                auto tensorKIn = oneInQue.DeQue<kType>();
                Cast(tensorKFp32, tensorKIn, RoundMode::CAST_NONE, K * curRowNum);
                oneInQue.FreeTensor(tensorKIn);
                PipeBarrier<PIPE_V>();

                Brcb(tensorBetaBrcbFP32, tensorBetaFP32[rowOffset], static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                Brcb(tensorGbrcbFP32, tensorGExpFP32[rowOffset], static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                PipeBarrier<PIPE_V>();

                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = K * sizeof(float32_t) / ONE_BLOCK_32;
                // tensorProductFp32 = Dkb * K, then row-reduce over K to
                // obtain the Dkb contribution to dbeta.
                Mul(tensorProductFp32, tensorDkbFp32, tensorKFp32, K * curRowNum);
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < curRowNum; row++) {
                    WholeReduceSum(tensorReduceSum[row * FP32_PER_BLOCK_8], tensorProductFp32[row * K],
                                   FP32_PER_REPEAT_64, wholeReduceKCnt, 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorProductFp32, tensorReduceSum, wholeReduceKCnt, curRowNum, 1, 1, 1);
                PipeBarrier<PIPE_V>();
                Add(tensorDbetaAccFP32[rowOffset], tensorDbetaAccFP32[rowOffset], tensorProductFp32, curRowNum);

                perchannelResOffset = 0;
                while (perchannelResOffset < K) {
                    // Dkb becomes the second dk term:
                    //   (dA^T @ K) * beta[:, None]
                    Mul(tensorDkbFp32[perchannelResOffset], tensorDkbFp32[perchannelResOffset], tensorBetaBrcbFP32,
                        FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                PipeBarrier<PIPE_V>();

                if (!waitedDkbgReady) {
                    // Second AIC ready event: Dkbg = A^T @ dw is available.
                    Arch::CrossCoreWaitFlag(flagAicFinishStore);
                    waitedDkbgReady = true;
                }
                tensorIn = oneInQue.AllocTensor<kType>();
                DataCopy(tensorIn, workSpaceTensor[slotDkbg + rowOffset * K], K * curRowNum);
                oneInQue.EnQue(tensorIn);
                auto tensorDkbgIn = oneInQue.DeQue<kType>();
                Cast(tensorDkbgFp32, tensorDkbgIn, RoundMode::CAST_NONE, K * curRowNum);
                oneInQue.FreeTensor(tensorDkbgIn);
                PipeBarrier<PIPE_V>();

                perchannelResOffset = 0;
                while (perchannelResOffset < K) {
                    // Apply exp(g) first:
                    //   Dkbg = (A^T @ dw) * exp(g)[:, None]
                    Mul(tensorDkbgFp32[perchannelResOffset], tensorDkbgFp32[perchannelResOffset], tensorGbrcbFP32,
                        FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                PipeBarrier<PIPE_V>();

                // tensorProductFp32 = Dkbg * K.  Row reduce gives:
                //   dbeta2 = reduce((A^T@dw) * exp(g) * K)
                // The same row-reduced value times beta is the Dkbg
                // contribution to dg.
                Mul(tensorProductFp32, tensorDkbgFp32, tensorKFp32, K * curRowNum);
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < curRowNum; row++) {
                    WholeReduceSum(tensorReduceSum[row * FP32_PER_BLOCK_8], tensorProductFp32[row * K],
                                   FP32_PER_REPEAT_64, wholeReduceKCnt, 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorKFp32, tensorReduceSum, wholeReduceKCnt, curRowNum, 1, 1, 1);
                PipeBarrier<PIPE_V>();
                Add(tensorDbetaAccFP32[rowOffset], tensorDbetaAccFP32[rowOffset], tensorKFp32, curRowNum);
                Mul(tensorKFp32, tensorKFp32, tensorBetaFP32[rowOffset], curRowNum);
                PipeBarrier<PIPE_V>();
                Add(tensorDgAccFP32[rowOffset], tensorDgAccFP32[rowOffset], tensorKFp32, curRowNum);

                perchannelResOffset = 0;
                while (perchannelResOffset < K) {
                    // Apply beta to form the third dk term:
                    //   (A^T @ dw) * exp(g)[:, None] * beta[:, None]
                    Mul(tensorDkbgFp32[perchannelResOffset], tensorDkbgFp32[perchannelResOffset], tensorBetaBrcbFP32,
                        FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                PipeBarrier<PIPE_V>();
                Add(tensorDkbFp32, tensorDkbFp32, tensorDkbgFp32, K * curRowNum);
                PipeBarrier<PIPE_V>();
                if (!waitedDKReady) {
                    // Third AIC ready event: DK = dA @ Kbeta is available.
                    Arch::CrossCoreWaitFlag(flagAicFinishStore);
                    waitedDKReady = true;
                }

                tensorIn = oneInQue.AllocTensor<kType>();
                DataCopy(tensorIn, workSpaceTensor[slotDK + rowOffset * K], K * curRowNum);
                oneInQue.EnQue(tensorIn);
                auto tensorDKIn = oneInQue.DeQue<kType>();
                Cast(tensorDKFp32, tensorDKIn, RoundMode::CAST_NONE, K * curRowNum);
                oneInQue.FreeTensor(tensorDKIn);
                PipeBarrier<PIPE_V>();

                // Final dk row tile:
                //   dk = DK + Dkb*beta + Dkbg*exp(g)*beta
                Add(tensorDKFp32, tensorDKFp32, tensorDkbFp32, K * curRowNum);
                PipeBarrier<PIPE_V>();
                // Multiple value heads in one GQA group contribute to the same
                // dk[kh]; the first head writes, later heads accumulate.
                if (!isFirstValueHeadInGroup) {
                    tensorIn = oneInQue.AllocTensor<kType>();
                    DataCopy(tensorIn, dkTensor[kBaseOffset + rowOffset * K], K * curRowNum);
                    oneInQue.EnQue(tensorIn);
                    auto tensorDkPrevIn = oneInQue.DeQue<kType>();
                    Cast(tensorProductFp32, tensorDkPrevIn, RoundMode::CAST_NONE, K * curRowNum);
                    oneInQue.FreeTensor(tensorDkPrevIn);
                    PipeBarrier<PIPE_V>();
                    Add(tensorDKFp32, tensorDKFp32, tensorProductFp32, K * curRowNum);
                    PipeBarrier<PIPE_V>();
                }
                auto tensorOut = oneOutQue.AllocTensor<kType>();
                Cast(tensorOut, tensorDKFp32, RoundMode::CAST_RINT, K * curRowNum);
                oneOutQue.EnQue(tensorOut);
                auto tensorDkOut = oneOutQue.DeQue<kType>();
                DataCopy(dkTensor[kBaseOffset + rowOffset * K], tensorDkOut, K * curRowNum);
                oneOutQue.FreeTensor(tensorDkOut);
            }
            if (!waitedDkbgReady) {
                Arch::CrossCoreWaitFlag(flagAicFinishStore);
            }
            if (!waitedDKReady) {
                Arch::CrossCoreWaitFlag(flagAicFinishStore);
            }

            // Fourth AIC ready event: Dvb = A^T @ du is available.
            Arch::CrossCoreWaitFlag(flagAicFinishStore);

            // Stage V2: consume Dvb and V to compute:
            //   dbeta += reduce(Dvb * V)
            //   dv     = Dvb * beta[:, None]
            for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                 rowOffset += rowOwned * subBlockNum) {
                uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                uint64_t rowElem = rowV * V;
                auto tensorDvbFp32 = temp;
                auto tensorVFp32 = temp[rowElem];
                auto tensorBetaBrcbFP32 = temp[2 * rowElem];
                auto tensorReduceSum = tensorBetaBrcbFP32[rowV * FP32_PER_BLOCK_8];

                auto tensorIn = oneInQue.AllocTensor<kType>();
                DataCopy(tensorIn, workSpaceTensor[slotKbetaDvb + rowOffset * V], V * curRowNum);
                oneInQue.EnQue(tensorIn);
                auto tensorDvbIn = oneInQue.DeQue<kType>();
                Cast(tensorDvbFp32, tensorDvbIn, RoundMode::CAST_NONE, V * curRowNum);
                oneInQue.FreeTensor(tensorDvbIn);

                tensorIn = oneInQue.AllocTensor<kType>();
                DataCopy(tensorIn, vTensor[vBaseOffset + rowOffset * V], V * curRowNum);
                oneInQue.EnQue(tensorIn);
                auto tensorVIn = oneInQue.DeQue<kType>();
                Cast(tensorVFp32, tensorVIn, RoundMode::CAST_NONE, V * curRowNum);
                oneInQue.FreeTensor(tensorVIn);
                PipeBarrier<PIPE_V>();

                // Broadcast beta for dv = Dvb * beta[:, None].
                Brcb(tensorBetaBrcbFP32, tensorBetaFP32[rowOffset], static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                // Reuse tensorVFp32 as product Dvb * V before reducing over V.
                Mul(tensorVFp32, tensorVFp32, tensorDvbFp32, V * curRowNum);
                PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < curRowNum; row++) {
                    WholeReduceSum(tensorReduceSum[row * FP32_PER_BLOCK_8], tensorVFp32[row * V],
                                   FP32_PER_REPEAT_64, wholeReduceVCnt, 1, 1, 8);
                }
                PipeBarrier<PIPE_V>();
                WholeReduceSum(tensorVFp32, tensorReduceSum, wholeReduceVCnt, curRowNum, 1, 1, 1);
                PipeBarrier<PIPE_V>();
                Add(tensorDbetaAccFP32[rowOffset], tensorDbetaAccFP32[rowOffset], tensorVFp32, curRowNum);

                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = V * sizeof(float32_t) / ONE_BLOCK_32;
                while (perchannelResOffset < V) {
                    // Form dv row tile: (A^T @ du) * beta[:, None].
                    Mul(tensorDvbFp32[perchannelResOffset], tensorDvbFp32[perchannelResOffset], tensorBetaBrcbFP32,
                        FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += FP32_PER_REPEAT_64;
                }
                PipeBarrier<PIPE_V>();
                auto tensorOut = oneOutQue.AllocTensor<kType>();
                Cast(tensorOut, tensorDvbFp32, RoundMode::CAST_RINT, V * curRowNum);
                oneOutQue.EnQue(tensorOut);
                auto tensorDvOut = oneOutQue.DeQue<kType>();
                DataCopy(dvTensor[vBaseOffset + rowOffset * V], tensorDvOut, V * curRowNum);
                oneOutQue.FreeTensor(tensorDvOut);
            }

            // Fifth AIC ready event: KKT = K @ K^T is available.
            Arch::CrossCoreWaitFlag(flagAicFinishStore);

            {
                uint64_t kktOffset = 0;
                uint64_t daOffset = chunkSize * chunkSize;
                uint64_t rowSumOffset = daOffset + rowKkt * chunkSize;
                uint64_t reduceOffset = rowSumOffset + rowKkt;
                auto tensorKKTFp32 = temp[kktOffset];
                auto tensorDaFp32 = temp[daOffset];
                auto tensorRowSumFP32 = temp[rowSumOffset];
                auto tensorReduceSum = temp[reduceOffset];

                // Stage V3: compute the KKT/dA contribution to dg.
                // For every KKT row block:
                //   1) load dA rows and KKT rows
                //   2) multiply KKT columns by beta[col]
                //   3) DAA[row, col] = dA^T[row, col] * KKT[row, col] * beta[col]
                // After all rows are materialized, add:
                //   dg += col_reduce(DAA) - row_reduce(DAA)
                for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowKkt) {
                    uint32_t curRowNum = rowOffset + rowKkt > curChunkSize ? curChunkSize - rowOffset : rowKkt;
                    auto tensorIn = oneInQue.AllocTensor<kType>();
                    DataCopy(tensorIn, dATensor[daBaseOffset + rowOffset * chunkSize], chunkSize * curRowNum);
                    oneInQue.EnQue(tensorIn);
                    auto tensorDaIn = oneInQue.DeQue<kType>();
                    Cast(tensorDaFp32, tensorDaIn, RoundMode::CAST_NONE, chunkSize * curRowNum);
                    oneInQue.FreeTensor(tensorDaIn);

                    tensorIn = oneInQue.AllocTensor<kType>();
                    DataCopy(tensorIn, workSpaceTensor[slotKKT + rowOffset * chunkSize], chunkSize * curRowNum);
                    oneInQue.EnQue(tensorIn);
                    auto tensorKKTIn = oneInQue.DeQue<kType>();
                    Cast(tensorKKTFp32[rowOffset * chunkSize], tensorKKTIn, RoundMode::CAST_NONE,
                         chunkSize * curRowNum);
                    oneInQue.FreeTensor(tensorKKTIn);
                    PipeBarrier<PIPE_V>();

                    uint64_t perchannelResOffset = 0;
                    uint8_t repeatStride = chunkSize * sizeof(float32_t) / ONE_BLOCK_32;
                    while (perchannelResOffset < curChunkSize) {
                        // Broadcast beta along KKT columns:
                        //   KKT[:, col] *= beta[col]
                        Mul(tensorKKTFp32[rowOffset * chunkSize + perchannelResOffset],
                            tensorKKTFp32[rowOffset * chunkSize + perchannelResOffset],
                            tensorBetaFP32[perchannelResOffset], FP32_PER_REPEAT_64, curRowNum,
                            {1, 1, 1, repeatStride, repeatStride, 0});
                        perchannelResOffset += FP32_PER_REPEAT_64;
                    }
                    PipeBarrier<PIPE_V>();
                    Mul(tensorKKTFp32[rowOffset * chunkSize], tensorDaFp32, tensorKKTFp32[rowOffset * chunkSize],
                        chunkSize * curRowNum);
                    PipeBarrier<PIPE_V>();
                    uint32_t remainCnt = curChunkSize % FP32_PER_REPEAT_64;
                    if (remainCnt > 0) {
                        uint32_t duplicateOffset = wholeReduceChunkCnt * FP32_PER_REPEAT_64 - FP32_PER_REPEAT_64;
                        uint64_t mask[1] = {0xffffffffffffffff};
                        mask[0] <<= remainCnt;
                        for (uint32_t row = 0; row < curRowNum; row++) {
                            // Zero padded columns beyond curChunkSize so the
                            // later WholeReduceSum over 64-wide repeats does
                            // not include stale values in the varlen tail.
                            Duplicate(tensorKKTFp32[(rowOffset + row) * chunkSize + duplicateOffset], 0.0f, mask, 1,
                                      1, 8);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }

                // All GM reads from this workspace slot are complete.  The
                // remaining reductions use UB only, so AIC can safely reuse
                // the slot after both AIV sub-blocks reach this point.
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE2>(flagWorkspaceFree[workspaceBufferIdx]);

                // row_reduce(DAA): only reduce rows owned by this AIV sub-block.
                // Each sub-block writes disjoint dg rows later, so reducing all
                // rows here duplicated work without changing the final output.
                for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                     rowOffset += rowOwned * subBlockNum) {
                    uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                    for (uint32_t row = 0; row < curRowNum; row++) {
                        WholeReduceSum(tensorReduceSum[(rowOffset + row) * FP32_PER_BLOCK_8],
                                       tensorKKTFp32[(rowOffset + row) * chunkSize], FP32_PER_REPEAT_64,
                                       wholeReduceChunkCnt, 1, 1, 8);
                    }
                }
                PipeBarrier<PIPE_V>();
                for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                     rowOffset += rowOwned * subBlockNum) {
                    uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                    WholeReduceSum(tensorRowSumFP32, tensorReduceSum[rowOffset * FP32_PER_BLOCK_8],
                                   wholeReduceChunkCnt, curRowNum, 1, 1, 1);
                    PipeBarrier<PIPE_V>();
                    Muls(tensorRowSumFP32, tensorRowSumFP32, -1.0f, curRowNum);
                    PipeBarrier<PIPE_V>();
                    Add(tensorDgAccFP32[rowOffset], tensorDgAccFP32[rowOffset], tensorRowSumFP32, curRowNum);
                    PipeBarrier<PIPE_V>();
                }

                // col_reduce(DAA): tree-add rows until the first row contains
                // the sum for every column; add it to dg.
                uint32_t remainRow = curChunkSize;
                while (remainRow > 1) {
                    uint32_t calcCnt = (remainRow / 2) * chunkSize;
                    remainRow = CeilDiv(remainRow, 2);
                    uint32_t offset = remainRow * chunkSize;
                    Add(tensorKKTFp32, tensorKKTFp32, tensorKKTFp32[offset], calcCnt);
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                     rowOffset += rowOwned * subBlockNum) {
                    uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                    Add(tensorDgAccFP32[rowOffset], tensorDgAccFP32[rowOffset], tensorKKTFp32[rowOffset], curRowNum);
                    PipeBarrier<PIPE_V>();
                }
            }

            for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                 rowOffset += rowOwned * subBlockNum) {
                uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                auto tensorOut = oneOutQue.AllocTensor<betaType>();
                if constexpr (std::is_same<betaType, float32_t>()) {
                    // float32 varlen tails can be non-32B aligned; Adds(x, 0)
                    // is used as a safe UB copy before DataCopyPad to GM.
                    Adds(tensorOut, tensorDbetaAccFP32[rowOffset], 0.0f, curRowNum);
                } else {
                    Cast(tensorOut, tensorDbetaAccFP32[rowOffset], RoundMode::CAST_RINT, curRowNum);
                }
                oneOutQue.EnQue(tensorOut);
                auto tensorDbetaOut = oneOutQue.DeQue<betaType>();
                DataCopyPad(dbetaTensor[betaOffset + rowOffset], tensorDbetaOut,
                            {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                oneOutQue.FreeTensor(tensorDbetaOut);
            }
            for (uint32_t rowOffset = subBlockIdx * rowOwned; rowOffset < curChunkSize;
                 rowOffset += rowOwned * subBlockNum) {
                uint32_t curRowNum = rowOffset + rowOwned > curChunkSize ? curChunkSize - rowOffset : rowOwned;
                auto tensorOut = oneOutQue.AllocTensor<betaType>();
                if constexpr (std::is_same<betaType, float32_t>()) {
                    // Same safe UB copy as dbeta for betaType=float32.
                    Adds(tensorOut, tensorDgAccFP32[rowOffset], 0.0f, curRowNum);
                } else {
                    Cast(tensorOut, tensorDgAccFP32[rowOffset], RoundMode::CAST_RINT, curRowNum);
                }
                oneOutQue.EnQue(tensorOut);
                auto tensorDgOut = oneOutQue.DeQue<betaType>();
                DataCopyPad(dgTensor[betaOffset + rowOffset], tensorDgOut,
                            {1, curRowNum * static_cast<uint32_t>(sizeof(betaType)), 0, 0, 0});
                oneOutQue.FreeTensor(tensorDgOut);
            }

        }
    }
    return;
}

#endif // PREPARE_WY_REPR_BWD_FULL_VECTOR_H
