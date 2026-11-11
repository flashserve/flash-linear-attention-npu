/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_VECTOR_H
#define CHUNK_KDA_BWD_RECOMPUTE_VECTOR_H

#include "chunk_kda_bwd_recompute_struct.h"
#include "chunk_kda_bwd_recompute_common.h"
#include "catlass/arch/cross_core_sync.hpp"

using namespace AscendC;

namespace KDA {

template <typename QkType, typename GateType, typename BetaType>
class ChunkKdaBwdRecomputeVectorProcess {
public:
    __aicore__ inline ChunkKdaBwdRecomputeVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR gk,
        GM_ADDR workspace, const ChunkKdaBwdRecomputeTilingData *tiling)
        : q_(q), k_(k), v_(v), g_(g), beta_(beta), aLog_(aLog), dtBias_(dtBias),
          cuSeqlens_(cuSeqlens), chunkIndices_(chunkIndices), w_(w), u_(u), qg_(qg), kg_(kg), gk_(gk),
          workspace_(workspace), tiling_(tiling)
    {
    }

    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe_ = pipe;
        qTensor_.SetGlobalBuffer((__gm__ QkType *)q_);
        kTensor_.SetGlobalBuffer((__gm__ QkType *)k_);
        vTensor_.SetGlobalBuffer((__gm__ QkType *)v_);
        gTensor_.SetGlobalBuffer((__gm__ GateType *)g_);
        betaTensor_.SetGlobalBuffer((__gm__ BetaType *)beta_);
        if (aLog_ != nullptr) {
            aLogTensor_.SetGlobalBuffer((__gm__ float *)aLog_);
        }
        if (dtBias_ != nullptr) {
            dtBiasTensor_.SetGlobalBuffer((__gm__ float *)dtBias_);
        }
        gkTensor_.SetGlobalBuffer((__gm__ float *)gk_);
        qgTensor_.SetGlobalBuffer((__gm__ QkType *)qg_);
        kgTensor_.SetGlobalBuffer((__gm__ QkType *)kg_);

        B_ = static_cast<uint64_t>(tiling_->B);
        Hk_ = static_cast<uint64_t>(tiling_->Hk);
        Hv_ = static_cast<uint64_t>(tiling_->Hv);
        hvPerHk_ = static_cast<uint64_t>(tiling_->hvPerHk);
        T_ = static_cast<uint64_t>(tiling_->T);
        K_ = static_cast<uint64_t>(tiling_->K);
        V_ = static_cast<uint64_t>(tiling_->V);
        chunkNum_ = static_cast<uint64_t>(tiling_->chunkNum);
        chunkSize_ = static_cast<uint64_t>(tiling_->chunkSize);
        isVariable_ = tiling_->isVariable;
        rowNum_ = static_cast<uint32_t>(tiling_->vecRow);
        if (rowNum_ < 8 || rowNum_ > 64) {
            rowNum_ = 8;
        }
        useGate_ = tiling_->useGateInKernel != 0;
        useExp2_ = tiling_->useExp2 != 0;
        hasALog_ = tiling_->hasALog != 0;
        hasDtBias_ = tiling_->hasDtBias != 0;
        lowerBound_ = KdaBwdRecomputeBitsToFloat(static_cast<uint32_t>(tiling_->lowerBoundBits));
        gateScale_ = lowerBound_ * (useExp2_ ? KDA_BWD_RECOMPUTE_RCP_LN2 : 1.0f);
        kbgBytes_ = B_ * Hv_ * T_ * K_ * sizeof(QkType);
        kbgTensor_.SetGlobalBuffer((__gm__ QkType *)workspace_);
        vbTensor_.SetGlobalBuffer((__gm__ QkType *)(workspace_ + kbgBytes_));
    }

    __aicore__ inline void Process()
    {
        ProcessVb();
        pipe_->Reset();
        AscendC::SyncAll<false>();
        ProcessGateAndK();
    }

private:
    __aicore__ inline void AllocAuxEvents()
    {
        auxMte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        auxMte3ToMte2Event_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void ReleaseAuxEvents()
    {
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(auxMte2ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(auxMte3ToMte2Event_);
    }

    __aicore__ inline void ProcessGateAndK()
    {
        uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        uint32_t coreNumAic = tiling_->usedAic > 0 ? static_cast<uint32_t>(tiling_->usedAic) : GetBlockNum();
        uint32_t vecTaskIdx = 0;
        uint32_t bos = 0;
        uint32_t eos = 0;

        pipe_->InitBuffer(gInQue_, 1, rowNum_ * K_ * sizeof(GateType));
        pipe_->InitBuffer(qInQue_, 1, rowNum_ * K_ * sizeof(QkType));
        pipe_->InitBuffer(kInQue_, 1, rowNum_ * K_ * sizeof(QkType));
        pipe_->InitBuffer(betaInQue_, 1, rowNum_ * sizeof(BetaType));
        pipe_->InitBuffer(gkOutQue_, 1, rowNum_ * K_ * sizeof(float));
        pipe_->InitBuffer(qgOutQue_, 1, rowNum_ * K_ * sizeof(QkType));
        pipe_->InitBuffer(kgOutQue_, 1, rowNum_ * K_ * sizeof(QkType));
        pipe_->InitBuffer(kbgOutQue_, 1, rowNum_ * K_ * sizeof(QkType));
        pipe_->InitBuffer(gFp32Buf_, rowNum_ * K_ * sizeof(float));
        pipe_->InitBuffer(tmpFp32Buf_, rowNum_ * K_ * sizeof(float));
        pipe_->InitBuffer(accFp32Buf_, K_ * sizeof(float));
        pipe_->InitBuffer(qFp32Buf_, rowNum_ * K_ * sizeof(float));
        pipe_->InitBuffer(kFp32Buf_, rowNum_ * K_ * sizeof(float));
        pipe_->InitBuffer(kRawFp32Buf_, rowNum_ * K_ * sizeof(float));
        pipe_->InitBuffer(betaFp32Buf_, rowNum_ * sizeof(float));
        pipe_->InitBuffer(betaBrcbBuf_, rowNum_ * KDA_BWD_RECOMPUTE_ONE_BLOCK_32);
        pipe_->InitBuffer(dtBiasBuf_, K_ * sizeof(float));
        pipe_->InitBuffer(scaleFp32Buf_, rowNum_ * K_ * sizeof(float));
        pipe_->InitBuffer(expABuf_, K_ * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, 32);
        AllocAuxEvents();

        auto gFp32 = gFp32Buf_.Get<float>();
        auto tmpFp32 = tmpFp32Buf_.Get<float>();
        auto accFp32 = accFp32Buf_.Get<float>();
        auto qFp32 = qFp32Buf_.Get<float>();
        auto kFp32 = kFp32Buf_.Get<float>();
        auto kRawFp32 = kRawFp32Buf_.Get<float>();
        auto betaFp32 = betaFp32Buf_.Get<float>();
        auto betaBrcb = betaBrcbBuf_.Get<float>();
        auto dtBias = dtBiasBuf_.Get<float>();
        auto expAVec = expABuf_.Get<float>();
        Duplicate(scaleFp32Buf_.Get<float>(), gateScale_, rowNum_ * K_);
        PipeBarrier<PIPE_V>();

        for (uint64_t taskIdx = coreIdx; taskIdx < chunkNum_ * Hv_; taskIdx += coreNumAic) {
            uint32_t loopIdx = static_cast<uint32_t>(taskIdx / Hv_);
            uint64_t h = taskIdx % Hv_;
            KdaBwdRecomputeGetChunkOffset(cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_, loopIdx, bos, eos,
                                          isVariable_);
            uint32_t curChunkSize = eos - bos;
            ++vecTaskIdx;
            uint64_t hk = h / hvPerHk_;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore_);
                continue;
            }
            uint64_t coreLoopsInB = (T_ + chunkSize_ - 1) / chunkSize_;
            uint64_t bIdx = isVariable_ ? 0 : (loopIdx / coreLoopsInB);
            uint64_t bosK = isVariable_ ? bos : (bos - bIdx * (Hv_ - Hk_) * T_);
            uint64_t gBase = (h * T_ + bos) * K_;
            uint64_t qkBase = (hk * T_ + bosK) * K_;
            uint64_t outBase = (h * T_ + bos) * K_;
            uint64_t betaBase = h * T_ + bos;

            if (hasDtBias_) {
                DataCopy(dtBias, dtBiasTensor_[h * K_], K_);
                SetFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
            }
            if (hasALog_) {
                Duplicate(expAVec, aLogTensor_.GetValue(h), K_);
                PipeBarrier<PIPE_V>();
                Exp(expAVec, expAVec, K_);
                PipeBarrier<PIPE_V>();
            }
            Duplicate(accFp32, 0.0f, K_);
            PipeBarrier<PIPE_V>();

            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum_) {
                uint32_t curRowNum = (rowOffset + rowNum_ > curChunkSize) ? (curChunkSize - rowOffset) : rowNum_;
                {
                    auto gIn = gInQue_.AllocTensor<GateType>();
                    auto betaIn = betaInQue_.AllocTensor<BetaType>();
                    DataCopy(gIn, gTensor_[gBase + rowOffset * K_], K_ * curRowNum);
                    DataCopyPad(betaIn, betaTensor_[betaBase + rowOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(BetaType)), 0, 0, 0}, {false, 0, 0, 0});
                    gInQue_.EnQue(gIn);
                    betaInQue_.EnQue(betaIn);
                }
                {
                    auto gIn = gInQue_.DeQue<GateType>();
                    auto betaIn = betaInQue_.DeQue<BetaType>();
                    if constexpr (std::is_same<GateType, float>::value) {
                        DataCopy(gFp32, gIn, K_ * curRowNum);
                    } else {
                        Cast(gFp32, gIn, RoundMode::CAST_NONE, K_ * curRowNum);
                    }
                    if constexpr (std::is_same<BetaType, float>::value) {
                        DataCopy(betaFp32, betaIn, curRowNum);
                    } else {
                        Cast(betaFp32, betaIn, RoundMode::CAST_NONE, curRowNum);
                    }
                    gInQue_.FreeTensor(gIn);
                    betaInQue_.FreeTensor(betaIn);
                }
                PipeBarrier<PIPE_V>();
                if (useGate_) {
                    ApplySafeGate(gFp32, curRowNum);
                } else if (useExp2_) {
                    Muls(gFp32, gFp32, KDA_BWD_RECOMPUTE_RCP_LN2, K_ * curRowNum);
                    PipeBarrier<PIPE_V>();
                }
                CumsumRows(gFp32, accFp32, curRowNum);
                {
                    auto gkOut = gkOutQue_.AllocTensor<float>();
                    DataCopy(gkOut, gFp32, K_ * curRowNum);
                    gkOutQue_.EnQue(gkOut);
                }
                ExpGate(gFp32, tmpFp32, curRowNum);
                PipeBarrier<PIPE_V>();
                {
                    auto qIn = qInQue_.AllocTensor<QkType>();
                    auto kIn = kInQue_.AllocTensor<QkType>();
                    DataCopy(qIn, qTensor_[qkBase + rowOffset * K_], K_ * curRowNum);
                    DataCopy(kIn, kTensor_[qkBase + rowOffset * K_], K_ * curRowNum);
                    qInQue_.EnQue(qIn);
                    kInQue_.EnQue(kIn);
                }
                {
                    auto qIn = qInQue_.DeQue<QkType>();
                    auto kIn = kInQue_.DeQue<QkType>();
                    Cast(qFp32, qIn, RoundMode::CAST_NONE, K_ * curRowNum);
                    Cast(kRawFp32, kIn, RoundMode::CAST_NONE, K_ * curRowNum);
                    qInQue_.FreeTensor(qIn);
                    kInQue_.FreeTensor(kIn);
                }
                PipeBarrier<PIPE_V>();
                Mul(qFp32, qFp32, tmpFp32, K_ * curRowNum);
                Brcb(betaBrcb, betaFp32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                PipeBarrier<PIPE_V>();
                MulKByBetaExp(kFp32, kRawFp32, betaBrcb, tmpFp32, curRowNum);
                {
                    auto gkOut = gkOutQue_.DeQue<float>();
                    DataCopy(gkTensor_[outBase + rowOffset * K_], gkOut, K_ * curRowNum);
                    gkOutQue_.FreeTensor(gkOut);
                    auto qgOut = qgOutQue_.AllocTensor<QkType>();
                    auto kbgOut = kbgOutQue_.AllocTensor<QkType>();
                    Cast(qgOut, qFp32, RoundMode::CAST_RINT, K_ * curRowNum);
                    Cast(kbgOut, kFp32, RoundMode::CAST_RINT, K_ * curRowNum);
                    qgOutQue_.EnQue(qgOut);
                    kbgOutQue_.EnQue(kbgOut);
                }
                {
                    auto qgOut = qgOutQue_.DeQue<QkType>();
                    auto kbgOut = kbgOutQue_.DeQue<QkType>();
                    DataCopy(qgTensor_[outBase + rowOffset * K_], qgOut, K_ * curRowNum);
                    DataCopy(kbgTensor_[outBase + rowOffset * K_], kbgOut, K_ * curRowNum);
                    qgOutQue_.FreeTensor(qgOut);
                    kbgOutQue_.FreeTensor(kbgOut);
                }
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore_);
            SetFlag<HardEvent::MTE3_MTE2>(auxMte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(auxMte3ToMte2Event_);
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum_) {
                uint32_t curRowNum = (rowOffset + rowNum_ > curChunkSize) ? (curChunkSize - rowOffset) : rowNum_;
                {
                    auto kIn = kInQue_.AllocTensor<QkType>();
                    DataCopy(gFp32, gkTensor_[outBase + rowOffset * K_], K_ * curRowNum);
                    DataCopy(kIn, kTensor_[qkBase + rowOffset * K_], K_ * curRowNum);
                    kInQue_.EnQue(kIn);
                }
                SetFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(auxMte2ToVEvent_);
                {
                    auto kIn = kInQue_.DeQue<QkType>();
                    Cast(kRawFp32, kIn, RoundMode::CAST_NONE, K_ * curRowNum);
                    kInQue_.FreeTensor(kIn);
                    PipeBarrier<PIPE_V>();
                    ComputeKg(kFp32, kRawFp32, gFp32, accFp32, curRowNum);
                }
                PipeBarrier<PIPE_V>();
                {
                    auto kgOut = kgOutQue_.AllocTensor<QkType>();
                    Cast(kgOut, kFp32, RoundMode::CAST_RINT, K_ * curRowNum);
                    kgOutQue_.EnQue(kgOut);
                }
                {
                    auto kgOut = kgOutQue_.DeQue<QkType>();
                    DataCopy(kgTensor_[outBase + rowOffset * K_], kgOut, K_ * curRowNum);
                    kgOutQue_.FreeTensor(kgOut);
                }
            }
        }
        ReleaseAuxEvents();
    }

    __aicore__ inline void ProcessVb()
    {
        uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        uint32_t coreNumAic = tiling_->usedAic > 0 ? static_cast<uint32_t>(tiling_->usedAic)
                                                   : GetBlockNum();
        uint32_t vecTaskIdx = 0;
        uint32_t bos = 0;
        uint32_t eos = 0;

        pipe_->InitBuffer(vInQue_, 1, rowNum_ * V_ * sizeof(QkType));
        pipe_->InitBuffer(betaInQue_, 1, rowNum_ * sizeof(BetaType));
        pipe_->InitBuffer(vbOutQue_, 1, rowNum_ * V_ * sizeof(QkType));
        pipe_->InitBuffer(vFp32Buf_, rowNum_ * V_ * sizeof(float));
        pipe_->InitBuffer(betaFp32Buf_, rowNum_ * sizeof(float));
        pipe_->InitBuffer(betaBrcbBuf_, rowNum_ * KDA_BWD_RECOMPUTE_ONE_BLOCK_32);

        auto vFp32 = vFp32Buf_.Get<float>();
        auto betaFp32 = betaFp32Buf_.Get<float>();
        auto betaBrcb = betaBrcbBuf_.Get<float>();

        for (uint64_t taskIdx = coreIdx; taskIdx < chunkNum_ * Hv_; taskIdx += coreNumAic) {
            uint32_t loopIdx = static_cast<uint32_t>(taskIdx / Hv_);
            uint64_t h = taskIdx % Hv_;
            KdaBwdRecomputeGetChunkOffset(cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_, loopIdx, bos, eos,
                                          isVariable_);
            uint32_t curChunkSize = eos - bos;
            ++vecTaskIdx;
            if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore_);
                continue;
            }
            for (uint32_t rowOffset = 0; rowOffset < curChunkSize; rowOffset += rowNum_) {
                uint32_t curRowNum = (rowOffset + rowNum_ > curChunkSize) ? (curChunkSize - rowOffset) : rowNum_;
                uint64_t vOffset = (h * T_ + bos + rowOffset) * V_;
                uint64_t betaOffset = h * T_ + bos + rowOffset;
                {
                    auto vIn = vInQue_.AllocTensor<QkType>();
                    auto betaIn = betaInQue_.AllocTensor<BetaType>();
                    DataCopy(vIn, vTensor_[vOffset], V_ * curRowNum);
                    DataCopyPad(betaIn, betaTensor_[betaOffset],
                                {1, curRowNum * static_cast<uint32_t>(sizeof(BetaType)), 0, 0, 0}, {false, 0, 0, 0});
                    vInQue_.EnQue(vIn);
                    betaInQue_.EnQue(betaIn);
                }
                {
                    auto vIn = vInQue_.DeQue<QkType>();
                    auto betaIn = betaInQue_.DeQue<BetaType>();
                    Cast(vFp32, vIn, RoundMode::CAST_NONE, V_ * curRowNum);
                    if constexpr (std::is_same<BetaType, float>::value) {
                        DataCopy(betaFp32, betaIn, curRowNum);
                    } else {
                        Cast(betaFp32, betaIn, RoundMode::CAST_NONE, curRowNum);
                    }
                    vInQue_.FreeTensor(vIn);
                    betaInQue_.FreeTensor(betaIn);
                }
                PipeBarrier<PIPE_V>();
                Brcb(betaBrcb, betaFp32, static_cast<uint8_t>(CeilDiv(curRowNum, 8)), {1, 8});
                PipeBarrier<PIPE_V>();
                uint64_t perchannelResOffset = 0;
                uint8_t repeatStride = V_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
                while (perchannelResOffset < V_) {
                    Mul(vFp32[perchannelResOffset], vFp32[perchannelResOffset], betaBrcb,
                        KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64, curRowNum, {1, 1, 0, repeatStride, repeatStride, 1});
                    perchannelResOffset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64;
                }
                PipeBarrier<PIPE_V>();
                {
                    auto vbOut = vbOutQue_.AllocTensor<QkType>();
                    Cast(vbOut, vFp32, RoundMode::CAST_RINT, V_ * curRowNum);
                    vbOutQue_.EnQue(vbOut);
                }
                {
                    auto vbOut = vbOutQue_.DeQue<QkType>();
                    DataCopy(vbTensor_[vOffset], vbOut, V_ * curRowNum);
                    vbOutQue_.FreeTensor(vbOut);
                }
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(flagAivFinishStore_);
        }
    }

    __aicore__ inline void ApplySafeGate(LocalTensor<float> &gate, uint32_t rows)
    {
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        const uint8_t rowStride = K_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
        if (hasDtBias_) {
            LocalTensor<float> bias = dtBiasBuf_.Get<float>();
            for (uint32_t offset = 0; offset < K_; offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64) {
                Add(gate[offset], gate[offset], bias[offset], KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64,
                    rows, {1, 1, 1, rowStride, rowStride, 0});
            }
            PipeBarrier<PIPE_V>();
        }
        if (hasALog_) {
            LocalTensor<float> expAVec = expABuf_.Get<float>();
            for (uint32_t offset = 0; offset < K_; offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64) {
                Mul(gate[offset], gate[offset], expAVec[offset], KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64,
                    rows, {1, 1, 1, rowStride, rowStride, 0});
            }
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> tmp = tmpFp32Buf_.Get<float>();
        LocalTensor<float> scale = scaleFp32Buf_.Get<float>();
        Muls(tmp, gate, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(tmp, tmp, n);
        PipeBarrier<PIPE_V>();
        Adds(tmp, tmp, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Div(gate, scale, tmp, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CumsumRows(LocalTensor<float> &gate, LocalTensor<float> &acc, uint32_t rows)
    {
        Add(gate, gate, acc, K_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> src = gate;
        LocalTensor<float> dst = tmpFp32Buf_.Get<float>();
        bool inScratch = false;
        for (uint32_t step = 1; step < rows; step <<= 1) {
            DataCopy(dst, src, step * K_);
            Add(dst[step * K_], src[step * K_], src, (rows - step) * K_);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> swap = src;
            src = dst;
            dst = swap;
            inScratch = !inScratch;
        }
        if (inScratch) {
            DataCopy(gate, src, rows * K_);
            PipeBarrier<PIPE_V>();
        }
        DataCopy(acc, gate[(rows - 1) * K_], K_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExpGate(LocalTensor<float> &gk, LocalTensor<float> &out, uint32_t rows)
    {
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        if (useExp2_) {
            Muls(out, gk, KDA_BWD_RECOMPUTE_LN2, n);
            PipeBarrier<PIPE_V>();
            Exp(out, out, n);
        } else {
            Exp(out, gk, n);
        }
    }

    __aicore__ inline void MulKByBetaExp(
        LocalTensor<float> &out, LocalTensor<float> &kVal, LocalTensor<float> &betaBrcb,
        LocalTensor<float> &expGk, uint32_t rows)
    {
        uint64_t offset = 0;
        uint8_t repeatStride = K_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
        while (offset < K_) {
            Mul(out[offset], kVal[offset], expGk[offset], KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64, rows,
                {1, 1, 1, repeatStride, repeatStride, repeatStride});
            offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64;
        }
        PipeBarrier<PIPE_V>();
        offset = 0;
        while (offset < K_) {
            Mul(out[offset], out[offset], betaBrcb, KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64, rows,
                {1, 1, 0, repeatStride, repeatStride, 1});
            offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64;
        }
    }

    __aicore__ inline void ComputeKg(
        LocalTensor<float> &kgOut, LocalTensor<float> &kRaw, LocalTensor<float> &gk,
        LocalTensor<float> &gkLast, uint32_t rows)
    {
        LocalTensor<float> delta = tmpFp32Buf_.Get<float>();
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        const uint8_t rowStride = K_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
        for (uint32_t offset = 0; offset < K_; offset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64) {
            Sub(delta[offset], gkLast[offset], gk[offset], KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64,
                rows, {1, 1, 1, rowStride, 0, rowStride});
        }
        PipeBarrier<PIPE_V>();
        if (useExp2_) {
            Muls(delta, delta, KDA_BWD_RECOMPUTE_LN2, n);
            PipeBarrier<PIPE_V>();
        }
        Exp(delta, delta, n);
        PipeBarrier<PIPE_V>();
        Mul(kgOut, kRaw, delta, n);
    }

    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR g_;
    GM_ADDR beta_;
    GM_ADDR aLog_;
    GM_ADDR dtBias_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkIndices_;
    GM_ADDR gk_;
    GM_ADDR w_;
    GM_ADDR u_;
    GM_ADDR qg_;
    GM_ADDR kg_;
    GM_ADDR workspace_;
    const ChunkKdaBwdRecomputeTilingData *tiling_;
    TPipe *pipe_ = nullptr;

    GlobalTensor<QkType> qTensor_;
    GlobalTensor<QkType> kTensor_;
    GlobalTensor<QkType> vTensor_;
    GlobalTensor<GateType> gTensor_;
    GlobalTensor<BetaType> betaTensor_;
    GlobalTensor<float> aLogTensor_;
    GlobalTensor<float> dtBiasTensor_;
    GlobalTensor<float> gkTensor_;
    GlobalTensor<QkType> qgTensor_;
    GlobalTensor<QkType> kgTensor_;
    GlobalTensor<QkType> kbgTensor_;
    GlobalTensor<QkType> vbTensor_;

    uint64_t B_ = 0;
    uint64_t Hk_ = 0;
    uint64_t Hv_ = 0;
    uint64_t hvPerHk_ = 1;
    uint64_t T_ = 0;
    uint64_t K_ = 128;
    uint64_t V_ = 128;
    uint64_t chunkNum_ = 0;
    uint64_t chunkSize_ = 64;
    int64_t isVariable_ = 0;
    uint64_t kbgBytes_ = 0;
    uint32_t rowNum_ = 64;
    bool useGate_ = true;
    bool useExp2_ = true;
    bool hasALog_ = false;
    bool hasDtBias_ = false;
    float lowerBound_ = -5.0f;
    float gateScale_ = 0.0f;

    Catlass::Arch::CrossCoreFlagWithReverse<> flagAivFinishStore_{
        KDA_BWD_RECOMPUTE_SYNC_AIC_AIV_FLAG, KDA_BWD_RECOMPUTE_SYNC_AIV_AIC_FLAG};
    TEventID auxMte2ToVEvent_;
    TEventID auxMte3ToMte2Event_;

    TQue<TPosition::VECIN, 1> gInQue_;
    TQue<TPosition::VECIN, 1> qInQue_;
    TQue<TPosition::VECIN, 1> kInQue_;
    TQue<TPosition::VECIN, 1> vInQue_;
    TQue<TPosition::VECIN, 1> betaInQue_;
    TQue<TPosition::VECOUT, 1> gkOutQue_;
    TQue<TPosition::VECOUT, 1> qgOutQue_;
    TQue<TPosition::VECOUT, 1> kgOutQue_;
    TQue<TPosition::VECOUT, 1> kbgOutQue_;
    TQue<TPosition::VECOUT, 1> vbOutQue_;
    TBuf<TPosition::VECCALC> gFp32Buf_;
    TBuf<TPosition::VECCALC> tmpFp32Buf_;
    TBuf<TPosition::VECCALC> accFp32Buf_;
    TBuf<TPosition::VECCALC> qFp32Buf_;
    TBuf<TPosition::VECCALC> kFp32Buf_;
    TBuf<TPosition::VECCALC> kRawFp32Buf_;
    TBuf<TPosition::VECCALC> betaFp32Buf_;
    TBuf<TPosition::VECCALC> betaBrcbBuf_;
    TBuf<TPosition::VECCALC> dtBiasBuf_;
    TBuf<TPosition::VECCALC> scaleFp32Buf_;
    TBuf<TPosition::VECCALC> expABuf_;
    TBuf<TPosition::VECCALC> vFp32Buf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_VECTOR_H
