/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH92_VECTOR_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH92_VECTOR_H

#include "chunk_kda_bwd_recompute_common.h"
#include "chunk_kda_bwd_recompute_regbase.h"

namespace KDA {
using namespace AscendC;

template <typename QkType, typename GateType, typename BetaType>
class ChunkKdaBwdRecomputeVectorProcess {
public:
    __aicore__ inline ChunkKdaBwdRecomputeVectorProcess(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR gk,
        GM_ADDR workspace, const ChunkKdaBwdRecomputeTilingData *tiling)
        : cu_(cuSeqlens), indices_(chunkIndices), td_(*tiling), hasGk_(gk != nullptr)
    {
        gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(q));
        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(k));
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(v));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ GateType *>(g));
        gmBeta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaType *>(beta));
        if (td_.hasALog) {
            gmALog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
        }
        if (td_.hasDtBias) {
            gmBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
        }
        if (hasGk_) {
            gmGk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        }
        gmQg_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(qg));
        gmKg_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(kg));
        (void)w;
        (void)u;
        (void)workspace;
    }

    __aicore__ inline void Init(TPipe *pipe)
    {
        using namespace KdaBwdRecomputeArch92;
        pipe_ = pipe;
        // One allocation makes the design's absolute byte intervals explicit.
        pipe_->InitBuffer(ubBuf_, kUbBytes);
        ub_ = ubBuf_.Get<uint8_t>();
        l1_ = LocalTensor<uint8_t>(TPosition::A1, 0, kL1Bytes);
        for (uint32_t s = 0; s < kSlots; ++s) {
            inputFree_[s] = pipe_->AllocEventID<HardEvent::V_MTE2>();
            inputReady_[s] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            outputReady_[s] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            outputFree_[s] = pipe_->AllocEventID<HardEvent::MTE3_V>();
            SetFlag<HardEvent::V_MTE2>(inputFree_[s]);
            SetFlag<HardEvent::MTE3_V>(outputFree_[s]);
        }
    }

    __aicore__ inline void Process()
    {
        if (td_.useGateInKernel) {
            if (td_.useExp2) {
                ProcessWorks<true, true>();
            } else {
                ProcessWorks<true, false>();
            }
        } else if (td_.useExp2) {
            ProcessWorks<false, true>();
        } else {
            ProcessWorks<false, false>();
        }
        Drain();
    }

private:
    template <typename T>
    __aicore__ inline LocalTensor<T> Input(uint32_t s, uint32_t offset)
    {
        using namespace KdaBwdRecomputeArch92;
        return ub_[kInputBase + s * kInputBytes + offset].template ReinterpretCast<T>();
    }
    template <typename T>
    __aicore__ inline LocalTensor<T> Output(uint32_t s, uint32_t offset)
    {
        using namespace KdaBwdRecomputeArch92;
        return ub_[kOutputBase + s * kOutputBytes + offset].template ReinterpretCast<T>();
    }
    template <typename T>
    __aicore__ inline LocalTensor<T> Meta(uint32_t s, uint32_t offset)
    {
        using namespace KdaBwdRecomputeArch92;
        return ub_[kMetaBase + s * kMetaBytes + offset].template ReinterpretCast<T>();
    }

    __aicore__ inline void CopyIn(const KdaBwdRecomputeArch92::WorkInfo &work)
    {
        using namespace KdaBwdRecomputeArch92;
        const uint32_t s = work.slot;
        WaitFlag<HardEvent::V_MTE2>(inputFree_[s]);
        CopyGmToUb(Input<GateType>(s, kG), gmG_[work.valueOffset], work.rows * kChannels);
        CopyGmToUb(Input<QkType>(s, kQ), gmQ_[work.qkOffset], work.rows * kChannels);
        CopyGmToUb(Input<QkType>(s, kK), gmK_[work.qkOffset], work.rows * kChannels);
        CopyGmToUb(Input<QkType>(s, kV), gmV_[work.valueOffset], work.rows * kChannels);
        constexpr uint32_t betaOffset = std::is_same<BetaType, float>::value ? kBetaFp32 : kBetaRaw;
        CopyGmToUb(Meta<BetaType>(s, betaOffset), gmBeta_[work.betaOffset], work.rows);
        if (td_.hasDtBias) {
            CopyGmToUb(Meta<float>(s, kDtBias), gmBias_[work.head * kChannels], kChannels);
        }
        if (td_.hasALog) {
            // Exact one-element copy, including HV=1 and the last head: no
            // aligned overread of the input A_log tensor and no HV<=256 cap.
            CopyGmToUb(Meta<float>(s, kALog), gmALog_[work.head], 1);
        }
        SetFlag<HardEvent::MTE2_V>(inputReady_[s]);
    }

    template <bool USE_GATE, bool USE_EXP2>
    __aicore__ inline void Compute(const KdaBwdRecomputeArch92::WorkInfo &work)
    {
        using namespace KdaBwdRecomputeArch92;
        const uint32_t s = work.slot;
        WaitFlag<HardEvent::MTE2_V>(inputReady_[s]);
        WaitFlag<HardEvent::MTE3_V>(outputFree_[s]);
        auto g = (__ubuf__ GateType *)Input<GateType>(s, kG).GetPhyAddr();
        auto q = (__ubuf__ QkType *)Input<QkType>(s, kQ).GetPhyAddr();
        auto k = (__ubuf__ QkType *)Input<QkType>(s, kK).GetPhyAddr();
        auto v = (__ubuf__ QkType *)Input<QkType>(s, kV).GetPhyAddr();
        auto gk = (__ubuf__ float *)Output<float>(s, kGk).GetPhyAddr();
        auto qg = (__ubuf__ QkType *)Output<QkType>(s, kQg).GetPhyAddr();
        auto kbg = (__ubuf__ QkType *)Output<QkType>(s, kKbg).GetPhyAddr();
        auto kg = (__ubuf__ QkType *)Output<QkType>(s, kKg).GetPhyAddr();
        auto vb = (__ubuf__ QkType *)Output<QkType>(s, kVb).GetPhyAddr();
        constexpr uint32_t betaOffset = std::is_same<BetaType, float>::value ? kBetaFp32 : kBetaRaw;
        auto beta = (__ubuf__ BetaType *)Meta<BetaType>(s, betaOffset).GetPhyAddr();
        auto bias = (__ubuf__ float *)Meta<float>(s, kDtBias).GetPhyAddr();
        auto aLog = (__ubuf__ float *)Meta<float>(s, kALog).GetPhyAddr();
        const float bound = KdaBwdRecomputeBitsToFloat(static_cast<uint32_t>(td_.lowerBoundBits));
        if (work.rows == kRows) {
            RecomputeFullVf<QkType, GateType, BetaType, USE_GATE, USE_EXP2>(
                gk, g, bias, aLog, beta, q, k, v, qg, kbg, kg, vb,
                bound, td_.hasDtBias != 0, td_.hasALog != 0);
        } else {
            GateTailVf<GateType, USE_GATE, USE_EXP2>(
                gk, g, bias, aLog, work.rows, bound, td_.hasDtBias != 0, td_.hasALog != 0);
            PipeBarrier<PIPE_V>();
            ResultsTailVf<QkType, BetaType, USE_GATE, USE_EXP2>(
                gk, beta, q, k, v, qg, kbg, kg, vb, work.rows);
        }
        // IN/META no longer have readers, even if OUT is still being copied.
        SetFlag<HardEvent::V_MTE2>(inputFree_[s]);
        SetFlag<HardEvent::V_MTE3>(outputReady_[s]);
    }

    __aicore__ inline void Publish(const KdaBwdRecomputeArch92::WorkInfo &work)
    {
        using namespace KdaBwdRecomputeArch92;
        const uint32_t s = work.slot;
        WaitFlag<HardEvent::V_MTE3>(outputReady_[s]);
        if (published_[s]) {
            WaitRhsFree<PIPE_MTE3>(s);
        }
        CopyRhsToL1(l1_[KbgOffset(s)].template ReinterpretCast<QkType>(), Output<QkType>(s, kKbg));
        CopyRhsToL1(l1_[VbOffset(s)].template ReinterpretCast<QkType>(), Output<QkType>(s, kVb));
        SetRhsReady<PIPE_MTE3>(s);
        published_[s] = true;
        const uint32_t elements = work.rows * kChannels;
        if (hasGk_) {
            DataCopy(gmGk_[work.valueOffset], Output<float>(s, kGk), elements);
        }
        DataCopy(gmQg_[work.valueOffset], Output<QkType>(s, kQg), elements);
        DataCopy(gmKg_[work.valueOffset], Output<QkType>(s, kKg), elements);
        SetFlag<HardEvent::MTE3_V>(outputFree_[s]);
    }

    template <bool USE_GATE, bool USE_EXP2>
    __aicore__ inline void ProcessWorks()
    {
        using namespace KdaBwdRecomputeArch92;
        const uint64_t total = td_.chunkNum * td_.Hv;
        const uint64_t stride = td_.usedAic;
        uint64_t id = GetBlockIdx(); // A6 has one AIV per block; no sub-block division.
        if (stride == 0 || id >= stride || id >= total) {
            return;
        }
        uint64_t ordinal = 0;
        WorkInfo current = GetWork(td_, cu_, indices_, id, ordinal);
        WorkInfo previous{};
        CopyIn(current);
        while (id < total) {
            const bool hasNext = id + stride < total;
            WorkInfo next{};
            if (hasNext) {
                next = GetWork(td_, cu_, indices_, id + stride, ordinal + 1);
                CopyIn(next);
            }
            Compute<USE_GATE, USE_EXP2>(current);
            // Queue the next VF before publishing the previous OUT. MTE3's
            // wait must not prevent VF(n+1) from being issued after VF(n).
            if (ordinal != 0) {
                Publish(previous);
            }
            previous = current;
            current = next;
            id += stride;
            ++ordinal;
        }
        Publish(previous);
    }

    __aicore__ inline void Drain()
    {
        using namespace KdaBwdRecomputeArch92;
        for (uint32_t s = 0; s < kSlots; ++s) {
            WaitFlag<HardEvent::V_MTE2>(inputFree_[s]);
            WaitFlag<HardEvent::MTE3_V>(outputFree_[s]);
            if (published_[s]) {
                WaitRhsFree<PIPE_MTE3>(s);
            }
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(inputFree_[s]);
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(inputReady_[s]);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(outputReady_[s]);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputFree_[s]);
        }
    }

    GM_ADDR cu_;
    GM_ADDR indices_;
    const ChunkKdaBwdRecomputeTilingData &td_;
    bool hasGk_;
    bool published_[2] = {false, false};
    GlobalTensor<QkType> gmQ_, gmK_, gmV_, gmQg_, gmKg_;
    GlobalTensor<GateType> gmG_;
    GlobalTensor<BetaType> gmBeta_;
    GlobalTensor<float> gmALog_, gmBias_, gmGk_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<uint8_t> ub_, l1_;
    TEventID inputFree_[2], inputReady_[2], outputReady_[2], outputFree_[2];
};

} // namespace KDA

#endif
