/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H

#include "../chunk_kda_bwd_recompute_struct.h"
#include "../chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_common.h"

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
        (void)workspace_;
        (void)w_;
        (void)u_;
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
        if (gk_ != nullptr) {
            gkTensor_.SetGlobalBuffer((__gm__ float *)gk_);
        }
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
        useGate_ = tiling_->useGateInKernel != 0;
        useExp2_ = tiling_->useExp2 != 0;
        hasALog_ = tiling_->hasALog != 0;
        hasDtBias_ = tiling_->hasDtBias != 0;
        {
            union {
                uint32_t u;
                float f;
            } conv;
            conv.u = static_cast<uint32_t>(tiling_->lowerBoundBits);
            lowerBound_ = conv.f;
        }
        gateScale_ = lowerBound_ * (useExp2_ ? KDA_BWD_RECOMPUTE_RCP_LN2 : 1.0f);

        l1Buffer_ = LocalTensor<uint8_t>(TPosition::A1, 0, 512 * 1024);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t bt = KdaBwdRecomputeArch35::kBt;
        constexpr uint32_t kk = KdaBwdRecomputeArch35::kK;
        pipe_->InitBuffer(gFp32Buf_, bt * kk * sizeof(float));
        pipe_->InitBuffer(tmpFp32Buf_, bt * kk * sizeof(float));
        pipe_->InitBuffer(workFp32Buf_, bt * kk * sizeof(float));
        pipe_->InitBuffer(qBuf_, bt * kk * sizeof(QkType));
        pipe_->InitBuffer(kBuf_, bt * kk * sizeof(QkType));
        pipe_->InitBuffer(vBuf_, bt * kk * sizeof(QkType));
        pipe_->InitBuffer(outBuf_, bt * kk * sizeof(QkType));
        pipe_->InitBuffer(betaRawBuf_, bt * sizeof(BetaType));
        pipe_->InitBuffer(betaFp32Buf_, bt * sizeof(float));
        pipe_->InitBuffer(betaBrcbBuf_, bt * KDA_BWD_RECOMPUTE_ONE_BLOCK_32);
        pipe_->InitBuffer(accFp32Buf_, kk * sizeof(float));
        pipe_->InitBuffer(gkLastBuf_, kk * sizeof(float));
        pipe_->InitBuffer(dtBiasBuf_, kk * sizeof(float));
        pipe_->InitBuffer(expABuf_, kk * sizeof(float));
        pipe_->InitBuffer(oneFp32Buf_, kk * sizeof(float));

        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        gkMte3Event_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);

        const uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        const uint32_t coreNumAic = GetBlockNum();
        uint32_t vecTaskIdx = 0;

        for (uint32_t loopIdx = coreIdx; loopIdx < chunkNum_; loopIdx += coreNumAic) {
            uint32_t bos = 0;
            uint32_t eos = 0;
            KdaBwdRecomputeGetChunkOffset(
                cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_, loopIdx, bos, eos, isVariable_);
            const uint32_t curChunkSize = eos - bos;
            for (uint64_t h = 0; h < Hv_; ++h) {
                ++vecTaskIdx;
                KdaBwdRecomputeArch35::AivWaitChunkFree<PIPE_MTE3>();
                if (vecTaskIdx % GetSubBlockNum() != GetSubBlockIdx()) {
                    KdaBwdRecomputeArch35::AivSetChunkReady<PIPE_MTE3>();
                    continue;
                }
                ProcessHead(loopIdx, h, bos, curChunkSize);
                KdaBwdRecomputeArch35::AivSetChunkReady<PIPE_MTE3>();
            }
        }

        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(gkMte3Event_);
    }

private:
    __aicore__ inline void ProcessHead(
        uint32_t loopIdx, uint64_t h, uint32_t bos, uint32_t curChunkSize)
    {
        constexpr uint32_t kk = KdaBwdRecomputeArch35::kK;
        constexpr uint32_t vv = KdaBwdRecomputeArch35::kV;
        const uint32_t slot = (loopIdx * static_cast<uint32_t>(Hv_) + static_cast<uint32_t>(h)) & 1U;
        const uint64_t hk = h / hvPerHk_;
        const uint64_t coreLoopsInB = (T_ + chunkSize_ - 1) / chunkSize_;
        const uint64_t bIdx = isVariable_ ? 0 : (loopIdx / coreLoopsInB);
        const uint64_t bosK = isVariable_ ? bos : (bos - bIdx * (Hv_ - Hk_) * T_);
        const uint64_t gBase = (h * T_ + bos) * K_;
        const uint64_t qkBase = (hk * T_ + bosK) * K_;
        const uint64_t outBase = (h * T_ + bos) * K_;
        const uint64_t vBase = (h * T_ + bos) * V_;
        const uint64_t betaBase = h * T_ + bos;
        const uint32_t rowElems = curChunkSize * static_cast<uint32_t>(K_);
        const uint32_t vElems = curChunkSize * static_cast<uint32_t>(V_);

        auto gFp32 = gFp32Buf_.Get<float>();
        auto tmpFp32 = tmpFp32Buf_.Get<float>();
        auto workFp32 = workFp32Buf_.Get<float>();
        auto qLocal = qBuf_.Get<QkType>();
        auto kLocal = kBuf_.Get<QkType>();
        auto vLocal = vBuf_.Get<QkType>();
        auto outLocal = outBuf_.Get<QkType>();
        auto betaFp32 = betaFp32Buf_.Get<float>();
        auto betaBrcb = betaBrcbBuf_.Get<float>();
        auto accFp32 = accFp32Buf_.Get<float>();
        auto gkLast = gkLastBuf_.Get<float>();
        auto dtBias = dtBiasBuf_.Get<float>();
        auto expAVec = expABuf_.Get<float>();

        LocalTensor<QkType> kbgL1 =
            l1Buffer_[KdaBwdRecomputeArch35::KbgSlotOffset(slot)].template ReinterpretCast<QkType>();
        LocalTensor<QkType> vbL1 =
            l1Buffer_[KdaBwdRecomputeArch35::VbSlotOffset(slot)].template ReinterpretCast<QkType>();

        if (hasDtBias_) {
            DataCopy(dtBias, dtBiasTensor_[h * K_], K_);
        }
        if constexpr (std::is_same<GateType, float>::value) {
            DataCopy(gFp32, gTensor_[gBase], rowElems);
        } else {
            DataCopy(outLocal, gTensor_[gBase], rowElems);
        }
        DataCopy(qLocal, qTensor_[qkBase], rowElems);
        DataCopy(kLocal, kTensor_[qkBase], rowElems);
        DataCopy(vLocal, vTensor_[vBase], vElems);
        if constexpr (std::is_same<BetaType, float>::value) {
            DataCopyPad(betaFp32, betaTensor_[betaBase],
                        {1, static_cast<uint32_t>(curChunkSize * sizeof(float)), 0, 0, 0},
                        {false, 0, 0, 0});
        } else {
            auto betaRaw = betaRawBuf_.Get<BetaType>();
            DataCopyPad(betaRaw, betaTensor_[betaBase],
                        {1, static_cast<uint32_t>(curChunkSize * sizeof(BetaType)), 0, 0, 0},
                        {false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        if constexpr (!std::is_same<GateType, float>::value) {
            Cast(gFp32, outLocal, RoundMode::CAST_NONE, rowElems);
        }
        if constexpr (!std::is_same<BetaType, float>::value) {
            Cast(betaFp32, betaRawBuf_.Get<BetaType>(), RoundMode::CAST_NONE, curChunkSize);
        }
        if (hasALog_) {
            Duplicate(expAVec, aLogTensor_.GetValue(h), K_);
            PipeBarrier<PIPE_V>();
            Exp(expAVec, expAVec, K_);
        }
        Duplicate(accFp32, 0.0f, K_);
        PipeBarrier<PIPE_V>();

        if (useGate_) {
            ApplySafeGate(gFp32, curChunkSize);
        }
        CumsumRows(gFp32, accFp32, curChunkSize);
        PipeBarrier<PIPE_V>();
        DataCopy(gkLast, gFp32[(curChunkSize - 1) * K_], K_);
        PipeBarrier<PIPE_V>();

        if (gk_ != nullptr) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(gkTensor_[outBase], gFp32, rowElems);
            SetFlag<HardEvent::MTE3_MTE2>(gkMte3Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(gkMte3Event_);
        }

        ExpGate(gFp32, tmpFp32, curChunkSize);
        PipeBarrier<PIPE_V>();
        Cast(workFp32, qLocal, RoundMode::CAST_NONE, rowElems);
        PipeBarrier<PIPE_V>();
        Mul(workFp32, workFp32, tmpFp32, rowElems);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        Cast(outLocal, workFp32, RoundMode::CAST_RINT, rowElems);
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(qgTensor_[outBase], outLocal, rowElems);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);

        Cast(workFp32, kLocal, RoundMode::CAST_NONE, rowElems);
        Brcb(betaBrcb, betaFp32, static_cast<uint8_t>(CeilDiv(curChunkSize, 8)), {1, 8});
        PipeBarrier<PIPE_V>();
        MulKByBetaExp(workFp32, workFp32, betaBrcb, tmpFp32, curChunkSize);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        Cast(outLocal, workFp32, RoundMode::CAST_RINT, rowElems);
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        KdaBwdRecomputeArch35::CopyUbNdToL1Zn(
            kbgL1, outLocal, curChunkSize, kk, KdaBwdRecomputeArch35::kBt, 0);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);

        ComputeKg(workFp32, kLocal, gFp32, gkLast, curChunkSize);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        Cast(outLocal, workFp32, RoundMode::CAST_RINT, rowElems);
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(kgTensor_[outBase], outLocal, rowElems);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);

        Cast(workFp32, vLocal, RoundMode::CAST_NONE, vElems);
        PipeBarrier<PIPE_V>();
        uint64_t perchannelResOffset = 0;
        uint8_t repeatStride = V_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
        while (perchannelResOffset < V_) {
            Mul(workFp32[perchannelResOffset], workFp32[perchannelResOffset], betaBrcb,
                KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64, curChunkSize,
                {1, 1, 0, repeatStride, repeatStride, 1});
            perchannelResOffset += KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64;
        }
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        Cast(outLocal, workFp32, RoundMode::CAST_RINT, vElems);
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        KdaBwdRecomputeArch35::CopyUbNdToL1Zn(
            vbL1, outLocal, curChunkSize, vv, KdaBwdRecomputeArch35::kBt, 0);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void ApplySafeGate(LocalTensor<float> &gate, uint32_t rows)
    {
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        if (hasDtBias_) {
            LocalTensor<float> bias = dtBiasBuf_.Get<float>();
            for (uint32_t row = 0; row < rows; ++row) {
                Add(gate[row * K_], gate[row * K_], bias, K_);
            }
            PipeBarrier<PIPE_V>();
        }
        if (hasALog_) {
            LocalTensor<float> expAVec = expABuf_.Get<float>();
            for (uint32_t row = 0; row < rows; ++row) {
                Mul(gate[row * K_], gate[row * K_], expAVec, K_);
            }
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> tmp = tmpFp32Buf_.Get<float>();
        LocalTensor<float> one = oneFp32Buf_.Get<float>();
        Muls(tmp, gate, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(tmp, tmp, n);
        PipeBarrier<PIPE_V>();
        Adds(tmp, tmp, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Duplicate(one, 1.0f, K_);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; ++row) {
            Div(gate[row * K_], one, tmp[row * K_], K_);
        }
        PipeBarrier<PIPE_V>();
        Muls(gate, gate, gateScale_, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CumsumRows(LocalTensor<float> &gate, LocalTensor<float> &acc, uint32_t rows)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            Add(acc, acc, gate[row * K_], K_);
            PipeBarrier<PIPE_V>();
            DataCopy(gate[row * K_], acc, K_);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ExpGate(LocalTensor<float> &gk, LocalTensor<float> &out, uint32_t rows)
    {
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        if (useExp2_) {
            DataCopy(out, gk, n);
            PipeBarrier<PIPE_V>();
            Muls(out, out, KDA_BWD_RECOMPUTE_LN2, n);
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
        DataCopy(out, kVal, rows * K_);
        PipeBarrier<PIPE_V>();
        uint64_t offset = 0;
        uint8_t repeatStride = K_ * sizeof(float) / KDA_BWD_RECOMPUTE_ONE_BLOCK_32;
        while (offset < K_) {
            Mul(out[offset], out[offset], expGk[offset], KDA_BWD_RECOMPUTE_FP32_PER_REPEAT_64, rows,
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
        LocalTensor<float> &kgOut, LocalTensor<QkType> &kRaw, LocalTensor<float> &gk,
        LocalTensor<float> &gkLast, uint32_t rows)
    {
        LocalTensor<float> delta = tmpFp32Buf_.Get<float>();
        LocalTensor<float> kFp32 = kgOut;
        Cast(kFp32, kRaw, RoundMode::CAST_NONE, rows * K_);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows; ++row) {
            Sub(delta, gkLast, gk[row * K_], K_);
            PipeBarrier<PIPE_V>();
            if (useExp2_) {
                Muls(delta, delta, KDA_BWD_RECOMPUTE_LN2, K_);
            }
            Exp(delta, delta, K_);
            PipeBarrier<PIPE_V>();
            Mul(kgOut[row * K_], kFp32[row * K_], delta, K_);
        }
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
    GM_ADDR w_;
    GM_ADDR u_;
    GM_ADDR qg_;
    GM_ADDR kg_;
    GM_ADDR gk_;
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

    LocalTensor<uint8_t> l1Buffer_;

    TBuf<TPosition::VECCALC> gFp32Buf_;
    TBuf<TPosition::VECCALC> tmpFp32Buf_;
    TBuf<TPosition::VECCALC> workFp32Buf_;
    TBuf<TPosition::VECCALC> qBuf_;
    TBuf<TPosition::VECCALC> kBuf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> betaRawBuf_;
    TBuf<TPosition::VECCALC> betaFp32Buf_;
    TBuf<TPosition::VECCALC> betaBrcbBuf_;
    TBuf<TPosition::VECCALC> accFp32Buf_;
    TBuf<TPosition::VECCALC> gkLastBuf_;
    TBuf<TPosition::VECCALC> dtBiasBuf_;
    TBuf<TPosition::VECCALC> expABuf_;
    TBuf<TPosition::VECCALC> oneFp32Buf_;

    TEventID mte2ToVEvent_;
    TEventID vToMte3Event_;
    TEventID mte3ToVEvent_;
    TEventID gkMte3Event_;

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
    bool useGate_ = true;
    bool useExp2_ = true;
    bool hasALog_ = false;
    bool hasDtBias_ = false;
    float lowerBound_ = -5.0f;
    float gateScale_ = 0.0f;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H
