/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH35_VECTOR_H

#include "../chunk_kda_bwd_recompute_struct.h"
#include "../chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_regbase.h"

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
        KdaBwdRecomputeArch35::BypassL2(qTensor_);
        KdaBwdRecomputeArch35::BypassL2(kTensor_);
        KdaBwdRecomputeArch35::BypassL2(vTensor_);
        KdaBwdRecomputeArch35::BypassL2(gTensor_);
        KdaBwdRecomputeArch35::BypassL2(betaTensor_);
        if (aLog_ != nullptr) {
            aLogTensor_.SetGlobalBuffer((__gm__ float *)aLog_);
        }
        if (dtBias_ != nullptr) {
            dtBiasTensor_.SetGlobalBuffer((__gm__ float *)dtBias_);
        }
        if (gk_ != nullptr) {
            gkTensor_.SetGlobalBuffer((__gm__ float *)gk_);
            KdaBwdRecomputeArch35::BypassL2(gkTensor_);
        }
        qgTensor_.SetGlobalBuffer((__gm__ QkType *)qg_);
        kgTensor_.SetGlobalBuffer((__gm__ QkType *)kg_);
        KdaBwdRecomputeArch35::BypassL2(qgTensor_);
        KdaBwdRecomputeArch35::BypassL2(kgTensor_);

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
        lowerBound_ = KdaBwdRecomputeBitsToFloat(static_cast<uint32_t>(tiling_->lowerBoundBits));
        gateScale_ = lowerBound_ * (useExp2_ ? KDA_BWD_RECOMPUTE_RCP_LN2 : 1.0f);

        l1Buffer_ = LocalTensor<uint8_t>(TPosition::A1, 0, 512 * 1024);
    }

    __aicore__ inline void Process()
    {
        if (useGate_ && useExp2_) {
            ProcessFastPingPong();
            return;
        }
        ProcessFallback();
    }

private:
    __aicore__ inline void CopyALogOnce(TEventID mte2Event)
    {
        if (!hasALog_) {
            return;
        }
        auto aLogAll = aLogAllBuf_.Get<float>();
        uint32_t nAlign = (static_cast<uint32_t>(Hv_) + 7U) & ~7U;
        if (nAlign < 8U) {
            nAlign = 8U;
        }
        if (nAlign > 256U) {
            nAlign = 256U;
        }
        DataCopy(aLogAll, aLogTensor_, nAlign);
        SetFlag<HardEvent::MTE2_V>(mte2Event);
        WaitFlag<HardEvent::MTE2_V>(mte2Event);
    }

    __aicore__ inline void FastCopyHeadIn(
        uint32_t buf, uint32_t loopIdx, uint64_t h, uint32_t bos, uint32_t curChunkSize)
    {
        const uint64_t hk = h / hvPerHk_;
        const uint64_t coreLoopsInB = (T_ + chunkSize_ - 1) / chunkSize_;
        const uint64_t bIdx = isVariable_ ? 0 : (loopIdx / coreLoopsInB);
        const uint64_t bosK = isVariable_ ? bos : (bos - bIdx * (Hv_ - Hk_) * T_);
        const uint64_t gBase = (h * T_ + bos) * K_;
        const uint64_t qkBase = (hk * T_ + bosK) * K_;
        const uint64_t vBase = (h * T_ + bos) * V_;
        const uint64_t betaBase = h * T_ + bos;
        const uint32_t rowElems = curChunkSize * static_cast<uint32_t>(K_);
        const uint32_t vElems = curChunkSize * static_cast<uint32_t>(V_);

        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[buf]);
        auto gFp32 = gFp32Buf_[buf].Get<float>();
        auto qLocal = qBuf_[buf].Get<QkType>();
        auto kLocal = kBuf_[buf].Get<QkType>();
        auto vLocal = vBuf_[buf].Get<QkType>();
        auto outLocal = outBuf_[buf].Get<QkType>();
        auto betaFp32 = betaFp32Buf_[buf].Get<float>();
        auto dtBias = dtBiasBuf_[buf].Get<float>();
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
            auto betaRaw = betaRawBuf_[buf].Get<BetaType>();
            DataCopyPad(betaRaw, betaTensor_[betaBase],
                        {1, static_cast<uint32_t>(curChunkSize * sizeof(BetaType)), 0, 0, 0},
                        {false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_[buf]);
    }

    __aicore__ inline void FastRunVf(uint32_t buf, uint64_t h, uint32_t curChunkSize)
    {
        auto gFp32 = gFp32Buf_[buf].Get<float>();
        auto qLocal = qBuf_[buf].Get<QkType>();
        auto kLocal = kBuf_[buf].Get<QkType>();
        auto vLocal = vBuf_[buf].Get<QkType>();
        auto outLocal = outBuf_[buf].Get<QkType>();
        auto betaFp32 = betaFp32Buf_[buf].Get<float>();
        auto dtBias = dtBiasBuf_[buf].Get<float>();
        auto aLogAll = aLogAllBuf_.Get<float>();

        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_[buf]);
        __ubuf__ float *aLogPtr = hasALog_ ?
            ((__ubuf__ float *)aLogAll.GetPhyAddr() + static_cast<uint32_t>(h)) : nullptr;
        __ubuf__ float *gPtr = (__ubuf__ float *)gFp32.GetPhyAddr();
        __ubuf__ float *biasPtr = (__ubuf__ float *)dtBias.GetPhyAddr();
        __ubuf__ GateType *gInPtr;
        if constexpr (std::is_same<GateType, float>::value) {
            gInPtr = (__ubuf__ GateType *)gPtr;
        } else {
            gInPtr = (__ubuf__ GateType *)outLocal.GetPhyAddr();
        }
        __ubuf__ BetaType *betaInPtr;
        if constexpr (std::is_same<BetaType, float>::value) {
            betaInPtr = (__ubuf__ BetaType *)betaFp32.GetPhyAddr();
        } else {
            betaInPtr = (__ubuf__ BetaType *)betaRawBuf_[buf].Get<BetaType>().GetPhyAddr();
        }
        const uint16_t vfRows = static_cast<uint16_t>(curChunkSize);
        auto qPtr = (__ubuf__ QkType *)qLocal.GetPhyAddr();
        auto kPtr = (__ubuf__ QkType *)kLocal.GetPhyAddr();
        auto vPtr = (__ubuf__ QkType *)vLocal.GetPhyAddr();
        auto kgPtr = (__ubuf__ QkType *)outLocal.GetPhyAddr();
        if (curChunkSize == KdaBwdRecomputeArch35::kBt) {
            if (hasDtBias_ && hasALog_) {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, true, true, true>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            } else if (hasDtBias_) {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, true, false, true>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            } else if (hasALog_) {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, false, true, true>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            } else {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, false, false, true>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            }
        } else if (hasDtBias_ && hasALog_) {
            KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                QkType, QkType, GateType, BetaType, true, true, false>(
                gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                vfRows, lowerBound_);
        } else if (hasDtBias_) {
            KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                QkType, QkType, GateType, BetaType, true, false, false>(
                gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                vfRows, lowerBound_);
        } else if (hasALog_) {
            KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                QkType, QkType, GateType, BetaType, false, true, false>(
                gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                vfRows, lowerBound_);
        } else {
            KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                QkType, QkType, GateType, BetaType, false, false, false>(
                gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                vfRows, lowerBound_);
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
    }

    __aicore__ inline void FastPublishL1(
        uint32_t buf, uint32_t slot, uint32_t curChunkSize, bool waitL1Free)
    {
        constexpr uint32_t kk = KdaBwdRecomputeArch35::kK;
        constexpr uint32_t vv = KdaBwdRecomputeArch35::kV;
        auto kLocal = kBuf_[buf].Get<QkType>();
        auto vLocal = vBuf_[buf].Get<QkType>();
        LocalTensor<QkType> kbgL1 =
            l1Buffer_[KdaBwdRecomputeArch35::KbgSlotOffset(slot)].template ReinterpretCast<QkType>();
        LocalTensor<QkType> vbL1 =
            l1Buffer_[KdaBwdRecomputeArch35::VbSlotOffset(slot)].template ReinterpretCast<QkType>();

        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        if (waitL1Free) {
            KdaBwdRecomputeArch35::AivWaitChunkFree<PIPE_MTE3>();
        }
        KdaBwdRecomputeArch35::CopyUbNdToL1Zn(
            kbgL1, kLocal, curChunkSize, kk, KdaBwdRecomputeArch35::kBt, 0);
        KdaBwdRecomputeArch35::CopyUbNdToL1Zn(
            vbL1, vLocal, curChunkSize, vv, KdaBwdRecomputeArch35::kBt, 0);
        KdaBwdRecomputeArch35::AivSetChunkReady<PIPE_MTE3>();
    }

    __aicore__ inline void FastStoreGm(
        uint32_t buf, uint64_t h, uint32_t bos, uint32_t curChunkSize)
    {
        const uint64_t outBase = (h * T_ + bos) * K_;
        const uint32_t rowElems = curChunkSize * static_cast<uint32_t>(K_);
        auto gFp32 = gFp32Buf_[buf].Get<float>();
        auto qLocal = qBuf_[buf].Get<QkType>();
        auto outLocal = outBuf_[buf].Get<QkType>();
        if (gk_ != nullptr) {
            DataCopy(gkTensor_[outBase], gFp32, rowElems);
        }
        DataCopy(qgTensor_[outBase], qLocal, rowElems);
        DataCopy(kgTensor_[outBase], outLocal, rowElems);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[buf]);
    }

    // Depth-2 UB so tile N+1 MTE2+VF overlap Cube(N) and GM(N).
    // Each AIV still owns one L1 slot; WaitFree stays immediately before L1 refill.
    __aicore__ inline void ProcessFastPingPong()
    {
        constexpr uint32_t bt = KdaBwdRecomputeArch35::kBt;
        constexpr uint32_t kk = KdaBwdRecomputeArch35::kK;
        for (uint32_t i = 0; i < 2; ++i) {
            pipe_->InitBuffer(gFp32Buf_[i], bt * kk * sizeof(float));
            pipe_->InitBuffer(qBuf_[i], bt * kk * sizeof(QkType));
            pipe_->InitBuffer(kBuf_[i], bt * kk * sizeof(QkType));
            pipe_->InitBuffer(vBuf_[i], bt * kk * sizeof(QkType));
            pipe_->InitBuffer(outBuf_[i], bt * kk * sizeof(QkType));
            pipe_->InitBuffer(betaRawBuf_[i], bt * sizeof(BetaType));
            pipe_->InitBuffer(betaFp32Buf_[i], bt * sizeof(float));
            pipe_->InitBuffer(dtBiasBuf_[i], kk * sizeof(float));
            mte2ToVEvent_[i] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            mte3ToMte2Event_[i] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_[i]);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_[i]);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[i]);
        }
        pipe_->InitBuffer(aLogAllBuf_, 256 * sizeof(float));
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);

        CopyALogOnce(mte2ToVEvent_[0]);

        const uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        const uint32_t coreNumAic = tiling_->vecRow > 0 ? static_cast<uint32_t>(tiling_->vecRow)
                                                        : GetBlockNum();
        const uint32_t subIdx = GetSubBlockIdx();
        const uint64_t hv = (Hv_ == 0) ? 1 : Hv_;
        const uint64_t totalTasks = chunkNum_ * hv;
        uint64_t taskBegin = 0;
        uint64_t taskEnd = 0;
        KdaBwdRecomputeArch35::CoreTaskRange(
            coreIdx, coreNumAic, totalTasks, taskBegin, taskEnd);
        uint32_t pp = 0;
        bool hasLoaded = false;
        bool hasPublished = false;
        uint32_t loadBuf = 0;
        uint32_t loadBos = 0;
        uint32_t loadRows = 0;
        uint32_t loadSlot = 0;
        uint64_t loadH = 0;

        // Chunk-major contiguous tasks, 4-head rotation inner. This AIV takes
        // even or odd heads (slot = h&1); Cube sees the whole group.
        if (taskBegin < taskEnd) {
            const uint64_t firstLoop = taskBegin / hv;
            const uint64_t lastLoop = (taskEnd - 1U) / hv;
            for (uint64_t loopIdx = firstLoop; loopIdx <= lastLoop; ++loopIdx) {
                uint32_t bos = 0;
                uint32_t eos = 0;
                KdaBwdRecomputeGetChunkOffset(
                    cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_,
                    static_cast<uint32_t>(loopIdx), bos, eos, isVariable_);
                const uint32_t rows = eos - bos;
                uint64_t hStart = 0;
                uint64_t hEnd = 0;
                KdaBwdRecomputeArch35::HeadsOnChunk(
                    taskBegin, taskEnd, hv, loopIdx, firstLoop, lastLoop, hStart, hEnd);
                for (uint64_t hBase = hStart; hBase < hEnd; hBase += KdaBwdRecomputeArch35::kHeadRotate) {
                    const uint64_t groupEnd =
                        (hBase + KdaBwdRecomputeArch35::kHeadRotate < hEnd) ?
                            (hBase + KdaBwdRecomputeArch35::kHeadRotate) : hEnd;
                    for (uint64_t h = hBase; h < groupEnd; ++h) {
                        if ((h & 1U) != subIdx) {
                            continue;
                        }
                        const uint32_t slot = subIdx;

                        FastCopyHeadIn(pp, static_cast<uint32_t>(loopIdx), h, bos, rows);
                        if (hasLoaded) {
                            FastRunVf(loadBuf, loadH, loadRows);
                            FastPublishL1(loadBuf, loadSlot, loadRows, hasPublished);
                            FastStoreGm(loadBuf, loadH, loadBos, loadRows);
                            hasPublished = true;
                        }
                        hasLoaded = true;
                        loadBuf = pp;
                        loadBos = bos;
                        loadRows = rows;
                        loadSlot = slot;
                        loadH = h;
                        pp ^= 1U;
                    }
                }
            }
        }

        if (hasLoaded) {
            FastRunVf(loadBuf, loadH, loadRows);
            FastPublishL1(loadBuf, loadSlot, loadRows, hasPublished);
            FastStoreGm(loadBuf, loadH, loadBos, loadRows);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[1]);
        if (hasLoaded) {
            KdaBwdRecomputeArch35::AivWaitChunkFree<PIPE_MTE3>();
        }

        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_[0]);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_[1]);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[1]);
    }

    __aicore__ inline void ProcessFallback()
    {
        constexpr uint32_t bt = KdaBwdRecomputeArch35::kBt;
        constexpr uint32_t kk = KdaBwdRecomputeArch35::kK;
        pipe_->InitBuffer(gFp32Buf_[0], bt * kk * sizeof(float));
        pipe_->InitBuffer(tmpFp32Buf_, bt * kk * sizeof(float));
        pipe_->InitBuffer(workFp32Buf_, bt * kk * sizeof(float));
        pipe_->InitBuffer(qBuf_[0], bt * kk * sizeof(QkType));
        pipe_->InitBuffer(kBuf_[0], bt * kk * sizeof(QkType));
        pipe_->InitBuffer(vBuf_[0], bt * kk * sizeof(QkType));
        pipe_->InitBuffer(outBuf_[0], bt * kk * sizeof(QkType));
        pipe_->InitBuffer(betaRawBuf_[0], bt * sizeof(BetaType));
        pipe_->InitBuffer(betaFp32Buf_[0], bt * sizeof(float));
        pipe_->InitBuffer(betaBrcbBuf_, bt * KDA_BWD_RECOMPUTE_ONE_BLOCK_32);
        pipe_->InitBuffer(accFp32Buf_, kk * sizeof(float));
        pipe_->InitBuffer(gkLastBuf_, kk * sizeof(float));
        pipe_->InitBuffer(dtBiasBuf_[0], kk * sizeof(float));
        pipe_->InitBuffer(expABuf_, kk * sizeof(float));
        pipe_->InitBuffer(oneFp32Buf_, kk * sizeof(float));
        pipe_->InitBuffer(aLogAllBuf_, 256 * sizeof(float));

        mte2ToVEvent_[0] = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte3ToMte2Event_[0] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        vToSEvent_ = pipe_->AllocEventID<HardEvent::V_S>();
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_[0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_[0]);
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
        SetFlag<HardEvent::V_S>(vToSEvent_);
        WaitFlag<HardEvent::V_S>(vToSEvent_);

        CopyALogOnce(mte2ToVEvent_[0]);

        const uint32_t coreIdx = GetBlockIdx() / GetSubBlockNum();
        const uint32_t coreNumAic = tiling_->vecRow > 0 ? static_cast<uint32_t>(tiling_->vecRow)
                                                        : GetBlockNum();
        const uint32_t subIdx = GetSubBlockIdx();
        const uint64_t hv = (Hv_ == 0) ? 1 : Hv_;
        const uint64_t totalTasks = chunkNum_ * hv;
        uint64_t taskBegin = 0;
        uint64_t taskEnd = 0;
        KdaBwdRecomputeArch35::CoreTaskRange(
            coreIdx, coreNumAic, totalTasks, taskBegin, taskEnd);
        bool published = false;

        if (taskBegin < taskEnd) {
            const uint64_t firstLoop = taskBegin / hv;
            const uint64_t lastLoop = (taskEnd - 1U) / hv;
            for (uint64_t loopIdx = firstLoop; loopIdx <= lastLoop; ++loopIdx) {
                uint32_t bos = 0;
                uint32_t eos = 0;
                KdaBwdRecomputeGetChunkOffset(
                    cuSeqlens_, chunkIndices_, B_, Hv_, T_, chunkSize_,
                    static_cast<uint32_t>(loopIdx), bos, eos, isVariable_);
                const uint32_t curChunkSize = eos - bos;
                uint64_t hStart = 0;
                uint64_t hEnd = 0;
                KdaBwdRecomputeArch35::HeadsOnChunk(
                    taskBegin, taskEnd, hv, loopIdx, firstLoop, lastLoop, hStart, hEnd);
                for (uint64_t hBase = hStart; hBase < hEnd; hBase += KdaBwdRecomputeArch35::kHeadRotate) {
                    const uint64_t groupEnd =
                        (hBase + KdaBwdRecomputeArch35::kHeadRotate < hEnd) ?
                            (hBase + KdaBwdRecomputeArch35::kHeadRotate) : hEnd;
                    for (uint64_t h = hBase; h < groupEnd; ++h) {
                        if ((h & 1U) != subIdx) {
                            continue;
                        }
                        ProcessHead(
                            static_cast<uint32_t>(loopIdx), h, bos, curChunkSize, subIdx, published);
                        published = true;
                    }
                }
            }
        }
        if (published) {
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
            KdaBwdRecomputeArch35::AivWaitChunkFree<PIPE_MTE3>();
        }

        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_[0]);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
        pipe_->ReleaseEventID<HardEvent::V_S>(vToSEvent_);
    }

    __aicore__ inline void ProcessHead(
        uint32_t loopIdx, uint64_t h, uint32_t bos, uint32_t curChunkSize, uint32_t slot, bool waitL1Free)
    {
        constexpr uint32_t kk = KdaBwdRecomputeArch35::kK;
        constexpr uint32_t vv = KdaBwdRecomputeArch35::kV;
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

        auto gFp32 = gFp32Buf_[0].Get<float>();
        auto tmpFp32 = tmpFp32Buf_.Get<float>();
        auto workFp32 = workFp32Buf_.Get<float>();
        auto qLocal = qBuf_[0].Get<QkType>();
        auto kLocal = kBuf_[0].Get<QkType>();
        auto vLocal = vBuf_[0].Get<QkType>();
        auto outLocal = outBuf_[0].Get<QkType>();
        auto betaFp32 = betaFp32Buf_[0].Get<float>();
        auto betaBrcb = betaBrcbBuf_.Get<float>();
        auto accFp32 = accFp32Buf_.Get<float>();
        auto gkLast = gkLastBuf_.Get<float>();
        auto dtBias = dtBiasBuf_[0].Get<float>();
        auto expAVec = expABuf_.Get<float>();
        auto aLogAll = aLogAllBuf_.Get<float>();

        LocalTensor<QkType> kbgL1 =
            l1Buffer_[KdaBwdRecomputeArch35::KbgSlotOffset(slot)].template ReinterpretCast<QkType>();
        LocalTensor<QkType> vbL1 =
            l1Buffer_[KdaBwdRecomputeArch35::VbSlotOffset(slot)].template ReinterpretCast<QkType>();

        if (waitL1Free) {
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
        }
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
            auto betaRaw = betaRawBuf_[0].Get<BetaType>();
            DataCopyPad(betaRaw, betaTensor_[betaBase],
                        {1, static_cast<uint32_t>(curChunkSize * sizeof(BetaType)), 0, 0, 0},
                        {false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_[0]);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_[0]);

        if (useGate_ && useExp2_) {
            PipeBarrier<PIPE_V>();
            __ubuf__ float *aLogPtr = hasALog_ ?
                ((__ubuf__ float *)aLogAll.GetPhyAddr() + static_cast<uint32_t>(h)) : nullptr;
            __ubuf__ float *gPtr = (__ubuf__ float *)gFp32.GetPhyAddr();
            __ubuf__ float *biasPtr = (__ubuf__ float *)dtBias.GetPhyAddr();
            __ubuf__ GateType *gInPtr;
            if constexpr (std::is_same<GateType, float>::value) {
                gInPtr = (__ubuf__ GateType *)gPtr;
            } else {
                gInPtr = (__ubuf__ GateType *)outLocal.GetPhyAddr();
            }
            __ubuf__ BetaType *betaInPtr;
            if constexpr (std::is_same<BetaType, float>::value) {
                betaInPtr = (__ubuf__ BetaType *)betaFp32.GetPhyAddr();
            } else {
                betaInPtr = (__ubuf__ BetaType *)betaRawBuf_[0].Get<BetaType>().GetPhyAddr();
            }
            const uint16_t vfRows = static_cast<uint16_t>(curChunkSize);
            auto qPtr = (__ubuf__ QkType *)qLocal.GetPhyAddr();
            auto kPtr = (__ubuf__ QkType *)kLocal.GetPhyAddr();
            auto vPtr = (__ubuf__ QkType *)vLocal.GetPhyAddr();
            auto kgPtr = (__ubuf__ QkType *)outLocal.GetPhyAddr();
            if (hasDtBias_ && hasALog_) {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, true, true>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            } else if (hasDtBias_) {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, true, false>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            } else if (hasALog_) {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, false, true>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            } else {
                KdaBwdRecomputeArch35::FusedRecomputeChunk128Regbase<
                    QkType, QkType, GateType, BetaType, false, false>(
                    gPtr, gInPtr, biasPtr, aLogPtr, betaInPtr, qPtr, kPtr, vPtr, qPtr, kPtr, kgPtr, vPtr,
                    vfRows, lowerBound_);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (waitL1Free) {
                KdaBwdRecomputeArch35::AivWaitChunkFree<PIPE_MTE3>();
            }
            KdaBwdRecomputeArch35::CopyUbNdToL1Zn(
                kbgL1, kLocal, curChunkSize, kk, KdaBwdRecomputeArch35::kBt, 0);
            KdaBwdRecomputeArch35::CopyUbNdToL1Zn(
                vbL1, vLocal, curChunkSize, vv, KdaBwdRecomputeArch35::kBt, 0);
            KdaBwdRecomputeArch35::AivSetChunkReady<PIPE_MTE3>();
            if (gk_ != nullptr) {
                DataCopy(gkTensor_[outBase], gFp32, rowElems);
            }
            DataCopy(qgTensor_[outBase], qLocal, rowElems);
            DataCopy(kgTensor_[outBase], outLocal, rowElems);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
            return;
        }

        if constexpr (!std::is_same<GateType, float>::value) {
            Cast(gFp32, outLocal, RoundMode::CAST_NONE, rowElems);
        }
        if constexpr (!std::is_same<BetaType, float>::value) {
            Cast(betaFp32, betaRawBuf_[0].Get<BetaType>(), RoundMode::CAST_NONE, curChunkSize);
        }

        Duplicate(accFp32, 0.0f, K_);
        PipeBarrier<PIPE_V>();

        if (hasALog_) {
            Duplicate(expAVec, aLogAll.GetValue(h), K_);
            PipeBarrier<PIPE_V>();
            Exp(expAVec, expAVec, K_);
            PipeBarrier<PIPE_V>();
        }
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
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
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
        if (waitL1Free) {
            KdaBwdRecomputeArch35::AivWaitChunkFree<PIPE_MTE3>();
        }
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
        KdaBwdRecomputeArch35::AivSetChunkReady<PIPE_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_[0]);
    }

    __aicore__ inline void ApplySafeGate(LocalTensor<float> &gate, uint32_t rows)
    {
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        if (hasDtBias_) {
            LocalTensor<float> bias = dtBiasBuf_[0].Get<float>();
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
        const uint32_t n = rows * static_cast<uint32_t>(K_);
        Cast(kFp32, kRaw, RoundMode::CAST_NONE, n);
        for (uint32_t row = 0; row < rows; ++row) {
            DataCopy(delta[row * K_], gkLast, K_);
        }
        PipeBarrier<PIPE_V>();
        Sub(delta, delta, gk, n);
        PipeBarrier<PIPE_V>();
        if (useExp2_) {
            Muls(delta, delta, KDA_BWD_RECOMPUTE_LN2, n);
            PipeBarrier<PIPE_V>();
        }
        Exp(delta, delta, n);
        PipeBarrier<PIPE_V>();
        Mul(kgOut, kFp32, delta, n);
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

    TBuf<TPosition::VECCALC> gFp32Buf_[2];
    TBuf<TPosition::VECCALC> tmpFp32Buf_;
    TBuf<TPosition::VECCALC> workFp32Buf_;
    TBuf<TPosition::VECCALC> qBuf_[2];
    TBuf<TPosition::VECCALC> kBuf_[2];
    TBuf<TPosition::VECCALC> vBuf_[2];
    TBuf<TPosition::VECCALC> outBuf_[2];
    TBuf<TPosition::VECCALC> betaRawBuf_[2];
    TBuf<TPosition::VECCALC> betaFp32Buf_[2];
    TBuf<TPosition::VECCALC> betaBrcbBuf_;
    TBuf<TPosition::VECCALC> accFp32Buf_;
    TBuf<TPosition::VECCALC> gkLastBuf_;
    TBuf<TPosition::VECCALC> dtBiasBuf_[2];
    TBuf<TPosition::VECCALC> expABuf_;
    TBuf<TPosition::VECCALC> oneFp32Buf_;
    TBuf<TPosition::VECCALC> aLogAllBuf_;

    TEventID mte2ToVEvent_[2];
    TEventID vToMte3Event_;
    TEventID mte3ToVEvent_;
    TEventID mte3ToMte2Event_[2];
    TEventID vToSEvent_;

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
