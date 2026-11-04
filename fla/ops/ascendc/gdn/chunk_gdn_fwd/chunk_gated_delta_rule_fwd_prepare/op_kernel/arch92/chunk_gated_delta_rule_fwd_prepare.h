/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A6 MIX 1:1, three consecutive HV per sequence-chunk pack.
 * S1 of the next pack overlaps S5/S7 of the previous pack. Khat and KKT
 * are shared only inside a pack; W/U have separate retained UB windows.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_ARCH92_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_ARCH92_H

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_fwd_prepare_common.h"
#include "chunk_gated_delta_rule_fwd_prepare_vf.h"
#include "chunk_gated_delta_rule_fwd_prepare_datacopy.h"
#include "chunk_gated_delta_rule_fwd_prepare_matmul.h"

namespace ChunkGatedDeltaRuleFwdPrepare {
using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename GateDtype, typename ALogDtype>
class ChunkGatedDeltaRuleFwdPrepareKernel : public PrepareState {
public:
    using InDtype = bfloat16_t;

    template <typename Td>
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog, GM_ADDR dtBias,
                                GM_ADDR cu, GM_ADDR idx, GM_ADDR gOut, GM_ADDR wOut, GM_ADDR uOut, GM_ADDR aOut,
                                GM_ADDR qHat, GM_ADDR kHat, GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR betaEff,
                                GM_ADDR workspace, const Td &td)
    {
        LoadTiling(td);
        InitOnChip();
        GM_ADDR userWs = GetUserWorkspace(workspace);
        const uint32_t wsBase = static_cast<uint32_t>(coreIdx) * kWsPerCoreBytes;
        if (userWs != nullptr) {
            gmWsY.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs + wsBase),
                                  kWsYSlots * kWsYElems);
        }
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(q));
        gmK.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(k));
        gmV.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(v));
        gmG.SetGlobalBuffer(reinterpret_cast<__gm__ GateDtype *>(g));
        gmBeta.SetGlobalBuffer(reinterpret_cast<__gm__ GateDtype *>(beta));
        if (aLog != nullptr) {
            gmALog.SetGlobalBuffer(reinterpret_cast<__gm__ ALogDtype *>(aLog));
        }
        if (dtBias != nullptr) {
            gmDt.SetGlobalBuffer(reinterpret_cast<__gm__ ALogDtype *>(dtBias));
        }
        if (cu != nullptr) {
            gmCu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu));
        }
        if (idx != nullptr) {
            gmIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(idx));
        }
        gmGOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gOut));
        gmW.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(wOut));
        gmU.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(uOut));
        if (outputA != 0) {
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aOut));
        }
        if (useQkL2norm != 0) {
            gmQHat.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(qHat));
            gmKHat.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(kHat));
            gmQRstd.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
            gmKRstd.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        }
        hasBetaOut = 0;
        if (betaEff != nullptr) {
            gmBetaEff.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(betaEff));
            hasBetaOut = 1;
        }
    }

    // UB: three 112 KiB compute slots + three 12 KiB state slots.
    // L1 preserves the A5 offsets with only three task slots active (496 KiB end).
    // L0A/B keep two 16 KiB operand banks; L0C has three 64 KiB task banks.
    __aicore__ inline void InitOnChip()
    {
        subBlock = 0;
        coreIdx = AscendC::GetBlockIdx();
        numCore = AscendC::GetBlockNum();
        auxReady = 0;

        OnChipBuffer buf;

        ubVcsIdx = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(kUbVcsIdx);

        ubIVcs = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbIVcs);
        ubS0Zero = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS0Zero);
        ubVcsIdx.SetSize(64);
        ubIVcs.SetSize(kVcsPackedElems32);
        ubS0Zero.SetSize(kChunk64 * kChunk64);

        for (int32_t db = 0; db < kTasksPerRound; ++db) {
            ubGPrime[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbGPrime[db]);
            ubBetaEff[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbBetaEff[db]);
            ubQ[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1Q[db]);
            ubK[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1K[db]);
            ubKkt[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS2Kkt[db]);
            ubS6K[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS6K[db]);
            const uint32_t vOff = kUbS6V[db];
            ubS6V[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(vOff);
            ubS6K[db].SetSize(static_cast<uint32_t>(kChunk64 * K));
            ubS6V[db].SetSize(static_cast<uint32_t>(kChunk64 * V));
            ubQHat[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1QHat[db]);
            ubKHat[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1KHat[db]);
            ubKRstd[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1KRstd[db]);
            ubQRstd[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1QRstd[db]);
            ubGPrime[db].SetSize(kChunk64);
            ubBetaEff[db].SetSize(kChunk64);
            ubKRstd[db].SetSize(kChunk64);
            ubQRstd[db].SetSize(kChunk64);
            ubQ[db].SetSize(kChunk64 * kGdnHeadDimK);
            ubK[db].SetSize(kChunk64 * kGdnHeadDimK);
            ubQHat[db].SetSize(kChunk64 * kGdnHeadDimK);
            ubKHat[db].SetSize(kChunk64 * kGdnHeadDimK);
            ubKkt[db].SetSize(kChunk64 * kChunk64);
        }

        for (int32_t db = 0; db < kTasksPerRound; ++db) {
            ubS7W[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS7W[db]);
            ubS7U[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS7U[db]);
            ubS7W[db].SetSize(static_cast<uint32_t>(kChunk64 * K));
            ubS7U[db].SetSize(kChunk64 * 128); // V256 is written directly to GM
        }

        for (int32_t db = 0; db < kTasksPerRound; ++db) {
            ubLPacked[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3LPacked[db]);
            ubResVcs[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3ResVcs[db]);
            ubLFull[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3LFull[db]);
            ubLPacked[db].SetSize(kVcsPackedElems32);
            ubResVcs[db].SetSize(kChunk64 * kChunk64);
            ubLFull[db].SetSize(kChunk64 * kChunk64);
        }

        for (uint32_t t = 0; t < static_cast<uint32_t>(kTasksPerRound); ++t) {
            l1KHat[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1KHat(t));
            l1NegL[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1NegL(t));
            l1LeafRight[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1LeafRight(t));
            l1LeafLeft[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1LeafLeft(t));
            l1A[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentA(t));
            l1Kbg[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentKbg(t));
            l1Vb[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentVb(t));
            l1Vb[t].SetSize(static_cast<uint32_t>(kChunk64 * V));
            l1Kbg[t].SetSize(static_cast<uint32_t>(kChunk64 * K));
            l1Vb1[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentVb(t) + kBytesK128);
            l1Vb1[t].SetSize(static_cast<uint32_t>(kChunk64 * kGdnHeadDimK));
            l1Y[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1Y(t));
        }

        l1I = buf.template GetBuffer<BufferType::ASCEND_CB, float>(kL1ResidentI);

        for (uint32_t b = 0; b < 2; ++b) {
            l0A[b] = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0PairOff[b]);
            l0B[b] = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0PairOff[b]);
            l0Af[b] = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(kL0PairOff[b]);
            l0Bf[b] = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(kL0PairOff[b]);
            l0AfP[b] = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(kL0S7Off[b]);
            l0BfP[b] = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(kL0S7Off[b]);
            l0AS7[b] = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0S7Off[b]);
            l0BS7[b] = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0S7Off[b]);
            l0AS7[b].SetSize(kL0S7Slot / sizeof(InDtype));
            l0BS7[b].SetSize(kL0S7Slot / sizeof(InDtype));
            l0CS7[b] = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(kL0CS7Off[b]);
            l0CS7[b].SetSize(static_cast<uint32_t>(kChunk64 * 128));
        }

        for (uint32_t t = 0; t < static_cast<uint32_t>(kTasksPerRound); ++t) {
            l0CTask[t] = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(t * kL0CTaskStride);
            l0CTask[t].SetSize(kL0CTaskStride / sizeof(float));
        }
    }

    // ========================= Stage 0 =========================
    __aicore__ inline void Stage0_GenerateResidentAux()
    {
        if (auxReady != 0) {
            return;
        }
        FillVcsIdentity(ubIVcs, ubVcsIdx);
        Duplicate(ubS0Zero, 0.0f, static_cast<int32_t>(kChunk64 * kChunk64));
        SetFlag<HardEvent::V_MTE3>(kHeUb0);
        WaitFlag<HardEvent::V_MTE3>(kHeUb0);
        for (uint32_t t = static_cast<uint32_t>(subBlock); t < static_cast<uint32_t>(kTasksPerRound); t += PREPARE_AIV_COUNT) {
            UbToL1Fp32(l1LeafLeft[t], ubS0Zero, kChunk64);
            UbToL1Fp32(l1LeafRight[t], ubS0Zero, kChunk64);
        }
        SetFlag<HardEvent::MTE3_V>(kHeUb0);
        WaitFlag<HardEvent::MTE3_V>(kHeUb0);
        if (subBlock == 0) {
            Identity64VF(ubS0Zero);
            SetFlag<HardEvent::V_MTE3>(kHeUb0);
            WaitFlag<HardEvent::V_MTE3>(kHeUb0);
            UbNd64ToL1Nz8(l1I, ubS0Zero);
            SetFlag<HardEvent::MTE3_V>(kHeUb0);
            WaitFlag<HardEvent::MTE3_V>(kHeUb0);
        }
        auxReady = 1;
    }

    // ========================= Stage 1 =========================
    __aicore__ inline void Stage1_PadTailIfNeeded(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t db = TaskSlot(taskIdx);
        const int32_t nValid = static_cast<int32_t>(chunk.M);
        const int32_t nPad = static_cast<int32_t>(chunkSize);
        if (nValid >= nPad) {
            return;
        }
        if (OwnsHk(hv, taskIdx)) {
            Duplicate(ubK[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            Duplicate(ubKHat[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            if (useQkL2norm != 0) {
                Duplicate(ubQ[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
                Duplicate(ubQHat[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            }
        }
        Duplicate(ubGPrime[db], 0.0f, nPad);
        Duplicate(ubBetaEff[db], 0.0f, nPad);
        SetFlag<HardEvent::V_MTE2>(kHeUb1);
        WaitFlag<HardEvent::V_MTE2>(kHeUb1);
    }

    // First local HV of each HK computes Khat, including split groups.
    // Only hv % G == 0 publishes global norm outputs; S6 never reads those outputs.
    __aicore__ inline void Stage1_UploadK(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        if (!OwnsHk(hv, taskIdx)) {
            WaitAicStage4Done(taskIdx);
            return;
        }
        const int64_t hk = hv / HRatio;
        const int32_t db = TaskSlot(taskIdx);
        const int32_t nElem = static_cast<int32_t>(chunk.M * K);
        const int32_t nValid = static_cast<int32_t>(chunk.M);
        const int64_t offQk = OffsetBHTD(chunk.batch, hk, chunk.tokenStart, HK, T, K);
        const int64_t offRstd = OffsetBHT(chunk.batch, hk, chunk.tokenStart, HK, T);
        LocalTensor<InDtype> l1K = l1KHat[static_cast<uint32_t>(taskIdx)];

        if (useQkL2norm != 0) {
            DataCopy(ubK[db], gmK[offQk], nElem);
        } else {
            DataCopy(ubKHat[db], gmK[offQk], nElem);
        }
        SetFlag<HardEvent::MTE2_V>(kHeUb0);
        WaitFlag<HardEvent::MTE2_V>(kHeUb0);
        if (useQkL2norm != 0) {
            L2NormK128VF<InDtype>(ubK[db], ubKHat[db], ubKRstd[db], static_cast<uint32_t>(nValid), kGdnL2NormEps);
            SetFlag<HardEvent::V_MTE3>(kHeUb0);
            WaitFlag<HardEvent::V_MTE3>(kHeUb0);
            WaitAicStage4Done(taskIdx);
            UploadBf16NdToL1(l1K, ubKHat[db], static_cast<uint32_t>(K));
            NotifyAicStage1Done(taskIdx);
            if (WritesHkOutput(hv)) {
                DataCopy(gmKHat[offQk], ubKHat[db], nElem);
                CopyUbToGmElems(gmKRstd[offRstd], ubKRstd[db], static_cast<uint32_t>(nValid));
                DataCopy(ubQ[db], gmQ[offQk], nElem);
                SetFlag<HardEvent::MTE2_V>(kHeUb0);
                WaitFlag<HardEvent::MTE2_V>(kHeUb0);
                L2NormK128VF<InDtype>(ubQ[db], ubQHat[db], ubQRstd[db],
                                      static_cast<uint32_t>(nValid), kGdnL2NormEps);
                SetFlag<HardEvent::V_MTE3>(kHeUb0);
                WaitFlag<HardEvent::V_MTE3>(kHeUb0);
                DataCopy(gmQHat[offQk], ubQHat[db], nElem);
                CopyUbToGmElems(gmQRstd[offRstd], ubQRstd[db], static_cast<uint32_t>(nValid));
            }
        } else {
            SetFlag<HardEvent::MTE2_MTE3>(kHeUb0);
            WaitFlag<HardEvent::MTE2_MTE3>(kHeUb0);
            WaitAicStage4Done(taskIdx);
            UploadBf16NdToL1(l1K, ubKHat[db], static_cast<uint32_t>(K));
            NotifyAicStage1Done(taskIdx);
        }
    }

    __aicore__ inline void Stage1_GateAndBeta(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t db = TaskSlot(taskIdx);
        const int32_t nValid = static_cast<int32_t>(chunk.M);
        const int64_t offG = OffsetBHT(chunk.batch, hv, chunk.tokenStart, HV, T);
        LocalTensor<float> ubGfp = ubGPrime[db];
        LocalTensor<float> ubBfp = ubBetaEff[db];
        LocalTensor<GateDtype> ubGRaw = ubGfp.template ReinterpretCast<GateDtype>();
        LocalTensor<GateDtype> ubBRaw = ubBfp.template ReinterpretCast<GateDtype>();

        CopyGmToUbElems(ubGRaw, gmG[offG], static_cast<uint32_t>(nValid));
        CopyGmToUbElems(ubBRaw, gmBeta[offG], static_cast<uint32_t>(nValid));
        SetFlag<HardEvent::MTE2_V>(kHeUb0);
        WaitFlag<HardEvent::MTE2_V>(kHeUb0);

        if (useGateInKernel != 0) {
            float aLog = ScalarToFp32(gmALog.GetValue(hv));
            float dt = (hasDtBias != 0) ? ScalarToFp32(gmDt.GetValue(hv)) : 0.0f;
            SetFlag<HardEvent::S_V>(kHeUb0);
            WaitFlag<HardEvent::S_V>(kHeUb0);
            GateSoftplusVF<GateDtype>(ubGRaw, ubGfp, aLog, dt, static_cast<uint32_t>(nValid));
        } else if constexpr (!IsSameType<GateDtype, float>::value) {
            CastToFp32VF<GateDtype>(ubGRaw, ubGfp, static_cast<uint32_t>(nValid));
        }
        float scale = (useExp2 != 0) ? kGdnRcpLn2 : 1.0f;
        CumsumScaleVF(ubGfp, scale, static_cast<uint32_t>(nValid));
        SetFlag<HardEvent::V_MTE3>(kHeUb0);
        WaitFlag<HardEvent::V_MTE3>(kHeUb0);
        CopyUbToGmElems(gmGOut[offG], ubGfp, static_cast<uint32_t>(nValid));

        if (useBetaSigmoid != 0) {
            float bscale = (allowNegEigval != 0) ? 2.0f : 1.0f;
            BetaSigmoidVF<GateDtype>(ubBRaw, ubBfp, bscale, static_cast<uint32_t>(nValid));
        } else if constexpr (!IsSameType<GateDtype, float>::value) {
            CastToFp32VF<GateDtype>(ubBRaw, ubBfp, static_cast<uint32_t>(nValid));
        }

        if (hasBetaOut != 0) {
            SetFlag<HardEvent::V_MTE3>(kHeUb0);
            WaitFlag<HardEvent::V_MTE3>(kHeUb0);
            CopyUbToGmElems(gmBetaEff[offG], ubBfp, static_cast<uint32_t>(nValid));
        }
    }

    __aicore__ inline void Stage1_OneTask(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t db = TaskSlot(taskIdx);
        if (db == 0) {
            SetFlag<HardEvent::MTE3_V>(kHeUb0);
            WaitFlag<HardEvent::MTE3_V>(kHeUb0);
            SetFlag<HardEvent::MTE3_MTE2>(kHeUb0);
            WaitFlag<HardEvent::MTE3_MTE2>(kHeUb0);
        }
        Stage1_PadTailIfNeeded(chunk, hv, taskIdx);
        Stage1_UploadK(chunk, hv, taskIdx);
        Stage1_GateAndBeta(chunk, hv, taskIdx);
    }

    __aicore__ inline void NotifyKktSiblings(int64_t base, int64_t nThis, int64_t ownerTask)
    {
        const int64_t ownerWork = PackWorkId(base, nThis, ownerTask);
        const int64_t ownerChunk = ownerWork / HV;
        const int64_t ownerHv = ownerWork % HV;
        const int64_t hvLo = ownerHv - ((HRatio <= 1) ? 0 : (ownerHv % HRatio));
        const int64_t hvHi = hvLo + ((HRatio <= 1) ? 1 : HRatio);
        for (int64_t t2 = 0; t2 < nThis; ++t2) {
            const int64_t work2 = PackWorkId(base, nThis, t2);
            if ((work2 / HV) != ownerChunk) {
                continue;
            }
            const int64_t hv2 = work2 % HV;
            if (hv2 >= hvLo && hv2 < hvHi) {
                NotifyAivKktDone(t2);
            }
        }
    }

    // ========================= Stage 2 =========================
    __aicore__ inline void Stage2_AicOne(int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const int32_t kk = static_cast<int32_t>(K);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        const int32_t db = TaskSlot(taskIdx);
        WaitFlag<HardEvent::M_MTE1>(bank);
        MatmulToL0C<InDtype>(l1KHat[taskIdx], l1KHat[taskIdx], l0A[bank], l0B[bank], l0CTask[bank], bt, bt, kk, true,
                             false, false, bank);
        SetFlag<HardEvent::M_FIX>(bank);
        SetFlag<HardEvent::M_MTE1>(bank);
        WaitFlag<HardEvent::M_FIX>(bank);
        FixpipeL0cToUbFp32Nd(ubKkt[db], l0CTask[bank], kChunk64);
        SetFlag<HardEvent::FIX_M>(bank);
        if (HRatio <= 1) {
            NotifyAivKktDone(taskIdx);
        }
    }

    // ========================= Stage 3 =========================
    __aicore__ inline void Stage3_PrepareGate(int64_t taskIdx)
    {
        const int32_t db = TaskSlot(taskIdx);
        if (useExp2 != 0) {
            GateLowerLVF<true>(ubGPrime[db], ubBetaEff[db], ubLFull[db]);
        } else {
            GateLowerLVF<false>(ubGPrime[db], ubBetaEff[db], ubLFull[db]);
        }
        DataCopy(ubResVcs[db], ubIVcs, static_cast<int32_t>(kVcsPackedElems32));
    }

    // Siblings use the owner's untouched KKT window, with their own gate/beta.
    __aicore__ inline void Stage3_AivOne(int64_t hv, int64_t taskIdx)
    {
        WaitCubeKktDone(taskIdx);
        const int32_t db = TaskSlot(taskIdx);
        const int32_t kktDb = TaskSlot(OwnerTaskIdx(hv, taskIdx));
        MulKktGateVF(ubKkt[kktDb], ubLFull[db]);
        SetFlag<HardEvent::V_MTE3>(kHeUb0);
        WaitFlag<HardEvent::V_MTE3>(kHeUb0);
        PackDiagLeavesFromUb(ubLPacked[db], ubLFull[db]);
        SetFlag<HardEvent::MTE3_V>(kHeS3Pack);
        WaitFlag<HardEvent::MTE3_V>(kHeS3Pack);
        UbNd64ToL1Nz8(l1NegL[taskIdx], ubLFull[db]);
        MulReduceScatterVF32(ubResVcs[db], ubLPacked[db], ubResVcs[db], ubVcsIdx);
        SetFlag<HardEvent::V_MTE3>(kHeUb0);
        WaitFlag<HardEvent::V_MTE3>(kHeUb0);
        UploadDiagLeavesToL1(l1LeafRight[taskIdx], l1LeafLeft[taskIdx], ubResVcs[db]);
        NotifyAicStage3Done(taskIdx);
    }

    // ========================= Stage 4 =========================
    // Y = I + D1 @ N, then S5 computes A = D1 + Y @ D0 in FP32.
    // Packed leaves are stored transposed, restored by the LoadData flags.
    __aicore__ inline void Stage4_PrefillI(int64_t nThis)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        if (nThis <= 0) {
            return;
        }
        WaitFlag<HardEvent::M_MTE1>(0);
        LoadL1NzToL0AB<float>(l1I, l1I, l0Af[0], l0Bf[0], bt, bt, bt, true, false, 0);
        for (int64_t t = 0; t < nThis; ++t) {
            const uint8_t bank = static_cast<uint8_t>(t & 1);
            MmadL0ABToL0C<float>(l0Af[0], l0Bf[0], l0CTask[t], bt, bt, bt, true, bank);
            SetFlag<HardEvent::M_FIX>(bank);
            WaitFlag<HardEvent::M_FIX>(bank);
            SetFlag<HardEvent::FIX_M>(bank);
        }
        SetFlag<HardEvent::M_MTE1>(0);
    }

    __aicore__ inline void Stage4_AccIssue(int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        WaitFlag<HardEvent::M_MTE1>(bank);
        MatmulToL0C<float>(l1LeafLeft[taskIdx], l1NegL[taskIdx], l0AfP[bank], l0BfP[bank], l0CTask[taskIdx], bt, bt, bt,
                           false, true, true, bank);
        SetFlag<HardEvent::M_FIX>(bank);
        SetFlag<HardEvent::M_MTE1>(bank);
    }

    __aicore__ inline void Stage4_Dump(int64_t taskIdx)
    {
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        WaitFlag<HardEvent::M_FIX>(bank);
        FixpipeL0cToGmNzCs(gmWsY[WsYOffset(taskIdx)], l0CTask[taskIdx], kChunk64);
        SetFlag<HardEvent::FIX_M>(bank);
        SetFlag<HardEvent::FIX_MTE2>(bank);
        WaitFlag<HardEvent::FIX_MTE2>(bank);
        CopyGmNzToL1Fp32(l1Y[taskIdx], gmWsY[WsYOffset(taskIdx)], kChunk64);
        NotifyAivStage4Done(taskIdx);
        SetFlag<HardEvent::MTE2_MTE1>(taskIdx);
    }

    // ========================= Stage 5 =========================
    __aicore__ inline void Stage5_Compute(int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        WaitFlag<HardEvent::M_MTE1>(bank);
        MatmulToL0C<float>(l1I, l1LeafLeft[taskIdx], l0Af[bank], l0Bf[bank], l0CTask[taskIdx], bt, bt, bt, true, false,
                           false, bank);
        SetFlag<HardEvent::FIX_M>(bank);
        MatmulToL0C<float>(l1Y[taskIdx], l1LeafRight[taskIdx], l0AfP[bank], l0BfP[bank], l0CTask[taskIdx], bt, bt, bt,
                           false, false, false, bank);
        SetFlag<HardEvent::M_FIX>(bank);
        SetFlag<HardEvent::M_MTE1>(bank);
    }

    __aicore__ inline void DumpAL0cToL1(int64_t taskIdx, uint32_t n)
    {
        FixpipeParamsArch3510<CO2Layout::NZ> aFixpipeParams;
        aFixpipeParams.nSize = n;
        aFixpipeParams.mSize = n;
        aFixpipeParams.srcStride = n;
        aFixpipeParams.dstStride = n * 16;
        aFixpipeParams.quantPre = QuantMode_t::F322BF16;
        aFixpipeParams.unitFlag = 0;
        aFixpipeParams.isChannelSplit = false;
        Fixpipe<InDtype, float, CFG_NZ_L1>(l1A[taskIdx], l0CTask[taskIdx], aFixpipeParams);
    }

    __aicore__ inline void Stage5_Dump(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        const uint32_t n = static_cast<uint32_t>(chunkSize);
        const uint32_t m = static_cast<uint32_t>(chunk.M);
        const int64_t offA = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, chunkSize);
        WaitFlag<HardEvent::M_FIX>(bank);
        if (outputA != 0) {
            FixpipeL0cToGmNd<InDtype>(gmA[offA], l0CTask[taskIdx], m, n, n);
            SetFlag<HardEvent::FIX_MTE2>(bank);
            WaitFlag<HardEvent::FIX_MTE2>(bank);
            if (m == n) {
                CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmA[offA], n, n);
                SetFlag<HardEvent::MTE2_MTE1>(taskIdx);
            } else {
                DumpAL0cToL1(taskIdx, n);
                SetFlag<HardEvent::FIX_MTE1>(taskIdx);
            }
        } else {
            DumpAL0cToL1(taskIdx, n);
            SetFlag<HardEvent::FIX_MTE1>(taskIdx);
        }
        SetFlag<HardEvent::FIX_M>(bank);
    }

    // ========================= Stage 6 =========================
    // Owner Khat stays at C+64 KiB through all sibling RHS computations.
    // V256 occupies [C+32,C+64); Kbg occupies [C+16,C+32).
    __aicore__ inline void Stage6_AivOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t offV = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
        const int32_t nValid = static_cast<int32_t>(chunk.M);
        const int32_t nPad = static_cast<int32_t>(chunkSize);
        const int32_t nV = nValid * static_cast<int32_t>(V);
        const int32_t db = TaskSlot(taskIdx);
        LocalTensor<InDtype> ubKnd = ubS6K[db];
        LocalTensor<InDtype> ubVnd = ubS6V[db];
        LocalTensor<float> beta = ubBetaEff[db];
        LocalTensor<float> g = ubGPrime[db];
        if (nValid < nPad) {
            SetFlag<HardEvent::MTE3_V>(kHeUb0);
            WaitFlag<HardEvent::MTE3_V>(kHeUb0);
            Duplicate(ubVnd, static_cast<InDtype>(0), nPad * static_cast<int32_t>(V));
            Duplicate(ubKnd, static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            SetFlag<HardEvent::V_MTE2>(kHeUb0);
            WaitFlag<HardEvent::V_MTE2>(kHeUb0);
        }
        DataCopy(ubVnd, gmV[offV], nV);
        SetFlag<HardEvent::MTE2_V>(kHeUb0);
        WaitFlag<HardEvent::MTE2_V>(kHeUb0);
        ScaleRowsVF<InDtype>(ubVnd, ubVnd, beta, static_cast<uint32_t>(chunkSize), static_cast<uint32_t>(V));
        SetFlag<HardEvent::V_MTE3>(kHeUb0);
        WaitFlag<HardEvent::V_MTE3>(kHeUb0);
        UploadBf16NdToL1(l1Vb[taskIdx], ubVnd, static_cast<uint32_t>(V));

        auto ownerK = ubKHat[TaskSlot(OwnerTaskIdx(hv, taskIdx))];
        if (useExp2 != 0) {
            ScaleRowsBetaExp2gVF<InDtype, true>(ownerK, ubKnd, beta, g, static_cast<uint32_t>(chunkSize));
        } else {
            ScaleRowsBetaExp2gVF<InDtype, false>(ownerK, ubKnd, beta, g, static_cast<uint32_t>(chunkSize));
        }
        SetFlag<HardEvent::V_MTE3>(kHeUb1);
        WaitFlag<HardEvent::V_MTE3>(kHeUb1);
        UploadBf16NdToL1(l1Kbg[taskIdx], ubKnd, static_cast<uint32_t>(K));
    }

    // ========================= Stage 7 =========================
    __aicore__ inline void Stage7_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const uint32_t rows = static_cast<uint32_t>(chunk.M);
        const uint32_t nK = static_cast<uint32_t>(K);
        const uint32_t nV = static_cast<uint32_t>(V);
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const int32_t db = TaskSlot(taskIdx);
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> wuFixpipeParams;
        wuFixpipeParams.nSize = nK;
        wuFixpipeParams.mSize = rows;
        wuFixpipeParams.srcStride = kChunk64;
        wuFixpipeParams.dstStride = nK;
        wuFixpipeParams.quantPre = QuantMode_t::F322BF16;
        wuFixpipeParams.dualDstCtl = 0;
        wuFixpipeParams.subBlockId = 0;
        wuFixpipeParams.isChannelSplit = false;
        wuFixpipeParams.params.ndNum = 1;
        wuFixpipeParams.params.srcNdStride = 0;
        wuFixpipeParams.params.dstNdStride = 0;
        WaitFlag<HardEvent::M_MTE1>(0);
        WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Kbg[taskIdx], l0AS7[0], l0BS7[0], l0CS7[0], bt, static_cast<int32_t>(nK),
                               bt, 0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        Fixpipe<InDtype, float, CFG_ROW_MAJOR_UB>(ubS7W[db], l0CS7[0], wuFixpipeParams);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(0);

        wuFixpipeParams.nSize = kGdnHeadDimK;
        wuFixpipeParams.dstStride = nV;
        WaitFlag<HardEvent::M_MTE1>(1);
        WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb[taskIdx], l0AS7[1], l0BS7[1], l0CS7[1], bt,
                               static_cast<int32_t>(kGdnHeadDimK), bt, 1);
        SetFlag<HardEvent::M_FIX>(1);
        SetFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::M_FIX>(1);
        // Each compute slot retains only 16 KiB for U. V256 uses two
        // 128-column Fixpipe-to-GM writes; W and U128 stay in UB.
        if (nV > kGdnHeadDimK) {
            const int64_t offU = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
            FixpipeL0cToGmNd<InDtype>(gmU[offU], l0CS7[1], rows, kGdnHeadDimK, nV);
            SetFlag<HardEvent::FIX_MTE2>(1);
            WaitFlag<HardEvent::FIX_MTE2>(1);
            SetFlag<HardEvent::FIX_M>(1);
            WaitFlag<HardEvent::M_MTE1>(1);
            WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb1[taskIdx], l0AS7[1], l0BS7[1], l0CS7[1], bt,
                                   static_cast<int32_t>(kGdnHeadDimK), bt, 1);
            SetFlag<HardEvent::M_FIX>(1);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToGmNd<InDtype>(gmU[offU + kGdnHeadDimK], l0CS7[1], rows, kGdnHeadDimK, nV);
            SetFlag<HardEvent::FIX_MTE2>(1);
            WaitFlag<HardEvent::FIX_MTE2>(1);
            SetFlag<HardEvent::FIX_M>(1);
        } else {
            Fixpipe<InDtype, float, CFG_ROW_MAJOR_UB>(ubS7U[db], l0CS7[1], wuFixpipeParams);
            SetFlag<HardEvent::FIX_M>(1);
        }
    }

    __aicore__ inline void DecodeTask(int64_t base, int64_t nThis, int64_t taskIdx, ChunkRange &chunk, int64_t &hv)
    {
        const int64_t workId = PackWorkId(base, nThis, taskIdx);
        hv = workId % HV;
        chunk = GetChunkRange(*this, gmCu, gmIdx, workId / HV);
    }

    __aicore__ inline void DrainPackWuToGm(int64_t base, int64_t nThis)
    {
        for (int64_t t = subBlock; t < nThis; t += PREPARE_AIV_COUNT) {
            ChunkRange chunk;
            int64_t hv;
            DecodeTask(base, nThis, t, chunk, hv);
            CopyUbToGmElems(gmW[OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, K)],
                            ubS7W[TaskSlot(t)], static_cast<uint32_t>(chunk.M * K));
            if (V <= kGdnHeadDimK) {
                CopyUbToGmElems(gmU[OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V)],
                                ubS7U[TaskSlot(t)], static_cast<uint32_t>(chunk.M * V));
            }
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        Stage0_GenerateResidentAux();

        const int64_t packSize = TasksPerPack();
        const int64_t nPacks = PackCount();
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const PackWalk cur = MakePackWalk(pack, packSize);
            PackWalk prev{};
            if (pack != coreIdx) {
                prev = MakePackWalk(pack - numCore, packSize);
            }
            for (int64_t t = subBlock; t < cur.nThis; t += PREPARE_AIV_COUNT) {
                ChunkRange chunk;
                int64_t hv;
                DecodeTask(cur.base, cur.nThis, t, chunk, hv);
                Stage1_OneTask(chunk, hv, t);
            }
            SetFlag<HardEvent::MTE3_V>(kHeUb0);
            WaitFlag<HardEvent::MTE3_V>(kHeUb0);
            for (int64_t t = subBlock; t < cur.nThis; t += PREPARE_AIV_COUNT) {
                Stage3_PrepareGate(t);
            }
            for (int64_t t = subBlock; t < cur.nThis; t += PREPARE_AIV_COUNT) {
                Stage3_AivOne(PackWorkId(cur.base, cur.nThis, t) % HV, t);
            }
            // S1/S3 of this pack can overlap previous C5/C7 because their
            // UB windows are disjoint from retained W/U. Drain before S6
            // publishes readiness for C7 to overwrite those output windows.
            if (pack != coreIdx) {
                WaitAicStage7Done();
                DrainPackWuToGm(prev.base, prev.nThis);
            }
            SetFlag<HardEvent::MTE3_MTE2>(kHeUb0);
            WaitFlag<HardEvent::MTE3_MTE2>(kHeUb0);
            SetFlag<HardEvent::MTE3_V>(kHeUb0);
            WaitFlag<HardEvent::MTE3_V>(kHeUb0);

            for (int64_t t = subBlock; t < cur.nThis; t += PREPARE_AIV_COUNT) {
                ChunkRange chunk;
                int64_t hv;
                DecodeTask(cur.base, cur.nThis, t, chunk, hv);
                Stage6_AivOne(chunk, hv, t);
                NotifyAicStage6Done(t);
            }
        }
        if (coreIdx < nPacks) {
            const int64_t pack = coreIdx + (nPacks - 1 - coreIdx) / numCore * numCore;
            const PackWalk last = MakePackWalk(pack, packSize);
            WaitAicStage7Done();
            DrainPackWuToGm(last.base, last.nThis);
        }
        // Every slot retains one free credit, including never-used tail slots.
        for (int64_t t = 0; t < kTasksPerRound; ++t) {
            WaitAicStage4Done(t);
        }
        SetFlag<HardEvent::MTE3_S>(kHeUb0);
        WaitFlag<HardEvent::MTE3_S>(kHeUb0);
    }

    __aicore__ inline void ProcessAic()
    {
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        SetFlag<HardEvent::FIX_MTE1>(0);
        const int64_t packSize = TasksPerPack();
        for (int64_t t = 0; t < packSize; ++t) {
            NotifyAivStage4Done(t);
        }
        const int64_t nPacks = PackCount();
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const PackWalk cur = MakePackWalk(pack, packSize);
            WaitFlag<HardEvent::FIX_MTE1>(0);
            for (int64_t t = 0; t < cur.nThis; ++t) {
                if (!OwnsHk(PackWorkId(cur.base, cur.nThis, t) % HV, t)) {
                    continue;
                }
                WaitAivStage1Done(t);
                Stage2_AicOne(t);
                if (HRatio > 1) {
                    NotifyKktSiblings(cur.base, cur.nThis, t);
                }
            }
            Stage4_PrefillI(cur.nThis);
            if (cur.nThis > 0) {
                WaitAivStage3Done(0);
                Stage4_AccIssue(0);
                for (int64_t t = 1; t < cur.nThis; ++t) {
                    WaitAivStage3Done(t);
                    Stage4_AccIssue(t);
                    Stage4_Dump(t - 1);
                }
                Stage4_Dump(cur.nThis - 1);
            }
            if (cur.nThis > 0) {
                for (int64_t t = 0; t < cur.nThis; ++t) {
                    WaitFlag<HardEvent::MTE2_MTE1>(t);
                    Stage5_Compute(t);
                    ChunkRange chunk;
                    int64_t hv;
                    DecodeTask(cur.base, cur.nThis, t, chunk, hv);
                    Stage5_Dump(chunk, hv, t);
                }
            }
            for (int64_t t = 0; t < cur.nThis; ++t) {
                WaitAivStage6Done(t);
                ChunkRange chunk;
                int64_t hv;
                DecodeTask(cur.base, cur.nThis, t, chunk, hv);
                if (outputA != 0 && chunk.M == chunkSize) {
                    WaitFlag<HardEvent::MTE2_MTE1>(t);
                } else {
                    WaitFlag<HardEvent::FIX_MTE1>(t);
                }
                Stage7_AicOne(chunk, hv, t);
            }
            NotifyAivStage7Done();
            SetFlag<HardEvent::FIX_MTE1>(0);
        }
        WaitFlag<HardEvent::FIX_MTE1>(0);
        for (uint16_t bank = 0; bank < 2; ++bank) {
            WaitFlag<HardEvent::M_MTE1>(bank);
            WaitFlag<HardEvent::FIX_M>(bank);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
        if ASCEND_IS_AIC {
            ProcessAic();
        }
    }

private:
    GlobalTensor<InDtype> gmQ;
    GlobalTensor<InDtype> gmK;
    GlobalTensor<InDtype> gmV;
    GlobalTensor<GateDtype> gmG;
    GlobalTensor<GateDtype> gmBeta;
    GlobalTensor<ALogDtype> gmALog;
    GlobalTensor<ALogDtype> gmDt;
    GlobalTensor<InDtype> gmQHat;
    GlobalTensor<InDtype> gmKHat;
    GlobalTensor<InDtype> gmW;
    GlobalTensor<InDtype> gmU;
    GlobalTensor<InDtype> gmA;
    GlobalTensor<float> gmQRstd;
    GlobalTensor<float> gmKRstd;
    GlobalTensor<float> gmGOut;
    GlobalTensor<float> gmBetaEff;
    GlobalTensor<float> gmWsY;

    LocalTensor<uint32_t> ubVcsIdx;
    LocalTensor<float> ubIVcs, ubS0Zero;

    LocalTensor<float> ubGPrime[kTasksPerRound], ubBetaEff[kTasksPerRound];
    LocalTensor<InDtype> ubQ[kTasksPerRound], ubK[kTasksPerRound], ubQHat[kTasksPerRound], ubKHat[kTasksPerRound];
    LocalTensor<float> ubKRstd[kTasksPerRound], ubQRstd[kTasksPerRound];

    LocalTensor<float> ubKkt[kTasksPerRound];

    LocalTensor<float> ubLPacked[kTasksPerRound], ubResVcs[kTasksPerRound], ubLFull[kTasksPerRound];

    LocalTensor<InDtype> ubS6K[kTasksPerRound], ubS6V[kTasksPerRound];
    LocalTensor<InDtype> ubS7W[kTasksPerRound], ubS7U[kTasksPerRound];

    LocalTensor<float> l1I;
    LocalTensor<InDtype> l1KHat[kTasksPerRound], l1A[kTasksPerRound], l1Kbg[kTasksPerRound], l1Vb[kTasksPerRound], l1Vb1[kTasksPerRound];
    LocalTensor<float> l1NegL[kTasksPerRound], l1LeafRight[kTasksPerRound], l1LeafLeft[kTasksPerRound], l1Y[kTasksPerRound];

    LocalTensor<InDtype> l0A[2], l0B[2], l0AS7[2], l0BS7[2];
    LocalTensor<float> l0Af[2], l0Bf[2], l0AfP[2], l0BfP[2], l0CS7[2];
    LocalTensor<float> l0CTask[kTasksPerRound];

    int64_t hasBetaOut;
};

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif
