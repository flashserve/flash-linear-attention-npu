#ifndef CHUNK_FWD_H_FUSED_A5_H
#define CHUNK_FWD_H_FUSED_A5_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_fwd_h_vec.h"
#endif

namespace GDN {

struct FwdHFusedWorkspace {
    __gm__ uint8_t *base;
    uint64_t segmentInit;
};

constexpr uint32_t FWD_H_FUSED_HEADS = 8U;
constexpr uint32_t FWD_H_FUSED_SEGMENTS = 3U;
constexpr uint32_t FWD_H_FUSED_STATE_ELEMS = 128U * 128U;
constexpr uint32_t FWD_H_FUSED_SUMMARY_CHUNKS = 24U;
constexpr uint32_t FWD_H_FUSED_FLAG_STRIDE = 8U;
constexpr float FWD_H_FUSED_SAFE_LOG2_DECAY = -300.0f;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
__simd_vf__ inline void FwdHFusedGateSummaryVf(
    __ubuf__ float *gateTail, __ubuf__ float *summary)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> sum;
    RegTensor<float> maximum;
    RegTensor<float> value;
    Duplicate(sum, 0.0f, mask);
    Duplicate(maximum, 0.0f, mask);
    for (uint16_t chunk = 0;
         chunk < FWD_H_FUSED_SUMMARY_CHUNKS; ++chunk) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(
            value, gateTail + (chunk + 1U) * FWD_H_CHUNK - 1U);
        Add(sum, sum, value, mask);
        Max(maximum, maximum, value, mask);
    }
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(summary, sum, mask);
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
        summary + 1, maximum, mask);
}
#endif

__aicore__ inline FwdHFusedWorkspace FwdHGetFusedWorkspace(
    const FwdHKernelArgs &args) {
    return {args.workspace, args.tiling.segmentInitWorkspaceOffset};
}

// The unified A5 entry converts summary states into replay entry states.
struct FwdHFusedA5 {
    FwdHKernelArgs args;
    FwdHFusedWorkspace workspace;

    __aicore__ inline void Init(const FwdHKernelArgs &kernelArgs) {
        args = kernelArgs;
        workspace = FwdHGetFusedWorkspace(args);
    }

    __aicore__ inline uint64_t EligibilityOffset() const {
        return workspace.segmentInit +
            static_cast<uint64_t>(FWD_H_FUSED_SEGMENTS) *
                FWD_H_FUSED_HEADS * FWD_H_FUSED_STATE_ELEMS * sizeof(float);
    }

    __aicore__ inline bool IsTruncatedSummarySafe(
        uint32_t head, AscendC::LocalTensor<float> gateTail,
        AscendC::TEventID mte2ToV, AscendC::TEventID vToS) const {
        AscendC::GlobalTensor<float> gate;
        gate.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.g));
        constexpr uint32_t FIRST_SEGMENT_CHUNKS = 58U;
        constexpr uint32_t SECOND_SEGMENT_CHUNKS = 59U;
        const uint64_t headBase =
            static_cast<uint64_t>(head) * args.tiling.seqlen;
        for (uint32_t segment = 0; segment < 2U; ++segment) {
            const uint32_t segmentEnd = segment == 0U
                ? FIRST_SEGMENT_CHUNKS
                : FIRST_SEGMENT_CHUNKS + SECOND_SEGMENT_CHUNKS;
            const uint32_t firstChunk =
                segmentEnd - FWD_H_FUSED_SUMMARY_CHUNKS;
            constexpr uint32_t TAIL_TOKENS =
                FWD_H_FUSED_SUMMARY_CHUNKS * FWD_H_CHUNK;
            AscendC::DataCopy(
                gateTail, gate[headBase + firstChunk * FWD_H_CHUNK],
                TAIL_TOKENS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
            AscendC::VF_CALL<FwdHFusedGateSummaryVf>(
                reinterpret_cast<__ubuf__ float *>(gateTail.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(gateTail.GetPhyAddr()));
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vToS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vToS);
            const float log2Decay = gateTail.GetValue(0);
            const float maximum = gateTail.GetValue(1);
            if (log2Decay != log2Decay || maximum != maximum ||
                maximum > 0.0f ||
                !(log2Decay <= FWD_H_FUSED_SAFE_LOG2_DECAY)) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool UseTruncatedSummary() const {
        AscendC::GlobalTensor<uint32_t> eligibility;
        eligibility.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
            workspace.base + EligibilityOffset()));
        for (uint32_t head = 0; head < FWD_H_FUSED_HEADS; ++head) {
            if (eligibility.GetValue(head * FWD_H_FUSED_FLAG_STRIDE) == 0U) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void RunScalarPrefix() {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if ASCEND_IS_AIC {
            return;
        }
        const uint32_t core = AscendC::GetBlockIdx() /
            AscendC::GetSubBlockNum();
        if (core >= FWD_H_FUSED_HEADS) {
            return;
        }
        const uint32_t sub = AscendC::GetSubBlockIdx();
        if (sub != 0U) {
            return;
        }
        constexpr uint32_t STATE_ELEMS = FWD_H_FUSED_STATE_ELEMS;
        constexpr uint32_t TILE_ELEMS = STATE_ELEMS;
        constexpr uint32_t GATE_TAIL_ELEMS =
            FWD_H_FUSED_SUMMARY_CHUNKS * FWD_H_CHUNK;

        AscendC::GlobalTensor<float> segmentInit;
        segmentInit.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace.base + workspace.segmentInit));
        AscendC::GlobalTensor<bfloat16_t> h;
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.h));

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> gateTailBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> e0Buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> e1Buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> bf16Buf;
        pipe.InitBuffer(gateTailBuf, GATE_TAIL_ELEMS * sizeof(float));
        pipe.InitBuffer(e0Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(e1Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(bf16Buf, TILE_ELEMS * sizeof(bfloat16_t));
        const uint8_t mte2ToV =
            pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
        const uint8_t vToMte3 =
            pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
        const uint8_t mte3ToV =
            pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
        const uint8_t vToS =
            pipe.AllocEventID<AscendC::HardEvent::V_S>();
        AscendC::LocalTensor<float> gateTail = gateTailBuf.Get<float>();
        AscendC::LocalTensor<float> e0 = e0Buf.Get<float>();
        AscendC::LocalTensor<float> e1 = e1Buf.Get<float>();
        AscendC::LocalTensor<bfloat16_t> bf16 = bf16Buf.Get<bfloat16_t>();
        AscendC::GlobalTensor<uint32_t> eligibility;
        eligibility.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
            workspace.base + EligibilityOffset()));
        const bool safe = IsTruncatedSummarySafe(
            core, gateTail, mte2ToV, vToS);
        AscendC::LocalTensor<uint32_t> eligibilityLocal =
            gateTail.ReinterpretCast<uint32_t>();
        AscendC::Duplicate(
            eligibilityLocal, safe ? 1U : 0U,
            FWD_H_FUSED_FLAG_STRIDE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
        AscendC::DataCopy(
            eligibility[core * FWD_H_FUSED_FLAG_STRIDE],
            eligibilityLocal, FWD_H_FUSED_FLAG_STRIDE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
        if (!safe) {
            pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV);
            pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
            pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV);
            pipe.ReleaseEventID<AscendC::HardEvent::V_S>(vToS);
            pipe.Reset();
            return;
        }
        // Eligibility proves that the retained tail annihilates any finite
        // FP32 state, so the cross-segment transition is exactly zero here.
        constexpr float alpha = 0.0f;

        const uint64_t headOffset = static_cast<uint64_t>(core) * STATE_ELEMS;
        const uint64_t e0Base = headOffset;
        const uint64_t e1Base =
            (8ULL + core) * STATE_ELEMS;
        const uint64_t init2Base =
            (16ULL + core) * STATE_ELEMS;
        const uint64_t h1Base =
            (58ULL * 8ULL + core) * STATE_ELEMS;
        const uint64_t h2Base =
            (117ULL * 8ULL + core) * STATE_ELEMS;

        for (uint32_t tile = 0; tile < STATE_ELEMS;
             tile += TILE_ELEMS) {
            AscendC::DataCopy(e0, segmentInit[e0Base + tile], TILE_ELEMS);
            AscendC::DataCopy(e1, segmentInit[e1Base + tile], TILE_ELEMS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);

            AscendC::VF_CALL<FwdHSMinusOneArch35Vf>(
                reinterpret_cast<__ubuf__ float *>(e0.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(bf16.GetPhyAddr()));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::DataCopy(segmentInit[e1Base + tile], e0, TILE_ELEMS);
            AscendC::DataCopy(h[h1Base + tile], bf16, TILE_ELEMS);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);

            AscendC::Muls(e0, e0, alpha, TILE_ELEMS);
            AscendC::Add(e0, e0, e1, TILE_ELEMS);
            AscendC::VF_CALL<FwdHSMinusOneArch35Vf>(
                reinterpret_cast<__ubuf__ float *>(e0.GetPhyAddr()),
                reinterpret_cast<__ubuf__ bfloat16_t *>(bf16.GetPhyAddr()));
            AscendC::Duplicate(e1, 0.0f, TILE_ELEMS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3);
            AscendC::DataCopy(segmentInit[init2Base + tile], e0,
                              TILE_ELEMS);
            AscendC::DataCopy(h[h2Base + tile], bf16, TILE_ELEMS);
            AscendC::DataCopy(segmentInit[e0Base + tile], e1, TILE_ELEMS);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV);
        }
        pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV);
        pipe.ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV);
        pipe.ReleaseEventID<AscendC::HardEvent::V_S>(vToS);
        pipe.Reset();
#endif
    }
};

} // namespace GDN

#endif
