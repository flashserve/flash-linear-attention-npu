#ifndef CHUNK_FWD_H_FUSED_A5_H
#define CHUNK_FWD_H_FUSED_A5_H

#include "chunk_fwd_h_fused_protocol.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_fwd_h_vec.h"
#include "chunk_fwd_h_k0/arch35/pre_process_fwd_kernel_merged_vector.h"
#include "chunk_fwd_h_k0/arch35/pre_process_fwd_kernel_merged_cube.h"
#include "chunk_fwd_h_k1/arch35/merge_fwd_bwd_kernel_vector.h"
#include "chunk_fwd_h_k1/arch35/merge_fwd_bwd_kernel_cube.h"
#endif

namespace GDN {

enum class FwdHFusedStage : uint8_t {
    K0AffineSummary = 0,
    K1Prefix = 1,
    K2SegmentReplay = 2,
};

struct FwdHFusedWorkspace {
    __gm__ uint8_t *base;
    uint64_t hm;
    uint64_t k0Scratch;
    uint64_t segmentInit;
    uint64_t segmentDesc;
    uint64_t splitSeqDesc;
};

__aicore__ inline FwdHFusedWorkspace FwdHGetFusedWorkspace(
    const FwdHKernelArgs &args) {
    return {args.workspace, args.tiling.hmWorkspaceOffset,
            args.tiling.k0ScratchWorkspaceOffset,
            args.tiling.segmentInitWorkspaceOffset,
            args.tiling.segmentDescWorkspaceOffset,
            args.tiling.splitSeqDescWorkspaceOffset};
}

// The unified A5 entry owns the stage boundaries and workspace protocol. The
// stage bodies are added one at a time; serial execution remains the fallback
// until K0/K1/K2 all pass their precision gates.
struct FwdHFusedA5 {
    FwdHKernelArgs args;
    FwdHFusedWorkspace workspace;

    __aicore__ inline void Init(const FwdHKernelArgs &kernelArgs) {
        args = kernelArgs;
        workspace = FwdHGetFusedWorkspace(args);
    }

    __aicore__ inline void RunScalarPrefix() {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if ASCEND_IS_AIC {
            return;
        }
        const uint32_t core = AscendC::GetBlockIdx() /
            AscendC::GetSubBlockNum();
        if (core >= 8U) {
            return;
        }
        const uint32_t sub = AscendC::GetSubBlockIdx();
        if (sub != 0U) {
            return;
        }
        constexpr uint32_t STATE_ELEMS = 128U * 128U;
        constexpr uint32_t TILE_ELEMS = STATE_ELEMS;

        AscendC::GlobalTensor<float> segmentInit;
        segmentInit.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace.base + workspace.segmentInit));
        AscendC::GlobalTensor<bfloat16_t> h;
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.h));

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> e0Buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> e1Buf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> bf16Buf;
        pipe.InitBuffer(e0Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(e1Buf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(bf16Buf, TILE_ELEMS * sizeof(bfloat16_t));
        const uint8_t mte2ToV =
            pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
        const uint8_t vToMte3 =
            pipe.AllocEventID<AscendC::HardEvent::V_MTE3>();
        const uint8_t mte3ToV =
            pipe.AllocEventID<AscendC::HardEvent::MTE3_V>();
        AscendC::LocalTensor<float> e0 = e0Buf.Get<float>();
        AscendC::LocalTensor<float> e1 = e1Buf.Get<float>();
        AscendC::LocalTensor<bfloat16_t> bf16 = bf16Buf.Get<bfloat16_t>();
        // The cumulative negative target gate makes the cross-segment scalar
        // transition underflow to zero in FP32.
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
        pipe.Reset();
#endif
    }

    __aicore__ inline void RunK0() {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310

        PpFwdCtx ctx{};
        ctx.k = args.k;
        ctx.w = args.w;
        ctx.u = args.u;
        ctx.g = args.g;
        ctx.gk = args.gk;
        ctx.v = nullptr;
        ctx.cu = nullptr;
        ctx.hm = workspace.base + workspace.hm;
        ctx.ws = workspace.base + workspace.k0Scratch;
        ctx.plan.B = 1;
        ctx.plan.Hk = 4;
        ctx.plan.Hv = 8;
        ctx.plan.hvPerHk = 2;
        ctx.plan.T = 11264;
        ctx.plan.K = 128;
        ctx.plan.V = 128;
        ctx.plan.chunkSize = 64;
        ctx.plan.nSeq = 3;
        ctx.plan.gateMode = PPFM_GATE_USE_G;
        ctx.plan.gateDtype = PPFM_DTYPE_FP32;
        ctx.plan.usedAicNum = 24;
        ctx.plan.colSplit = 1;
        ctx.plan.hybridS = 0;
        ctx.plan.hybridBase = 0;
        if ASCEND_IS_AIC {
            PpFwdCube cube(ctx);
            cube.Run();
        }
        if ASCEND_IS_AIV {
            PpFwdVector vec(ctx);
            vec.Run();
        }
#endif
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void FillK1Tiling(
        MergeFwBwd::MergeFwdBwdKernelTilingData &tiling) {
        tiling.S = 3;
        tiling.Hv = 8;
        tiling.K = 128;
        tiling.V = 128;
        tiling.rank = 2;
        tiling.N = 2;
        tiling.forward = 1;
        tiling.usedAic = 8;
        tiling.sysWorkspaceSize = 0;
        tiling.stateVFirst = 1;
    }

    __aicore__ inline bool K1CoreActive() {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            coreIdx /= AscendC::GetSubBlockNum();
        }
        return coreIdx < 8U;
    }
#endif

    __aicore__ inline void RunK1Pack() {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (!K1CoreActive()) {
            return;
        }
        MergeFwBwd::MergeFwdBwdKernelTilingData tiling{};
        FillK1Tiling(tiling);
        GM_ADDR hm = workspace.base + workspace.hm;
        GM_ADDR scratch = workspace.base + workspace.k0Scratch;
        GM_ADDR segmentInit = workspace.base + workspace.segmentInit;
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            MergeFwBwd::MergeFwdBwdKernelVectorProcess<float> vector(
                hm, scratch, segmentInit, scratch, &tiling);
            vector.Init(&pipe);
            vector.ProcessPackOnly();
            pipe.Reset();
        }
#endif
    }

    __aicore__ inline void RunK1Cube() {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (!K1CoreActive()) {
            return;
        }
        MergeFwBwd::MergeFwdBwdKernelTilingData tiling{};
        FillK1Tiling(tiling);
        GM_ADDR hm = workspace.base + workspace.hm;
        GM_ADDR scratch = workspace.base + workspace.k0Scratch;
        GM_ADDR segmentInit = workspace.base + workspace.segmentInit;
        if ASCEND_IS_AIC {
            MergeFwBwd::MergeFwdBwdKernelCubeProcess<float> cube(
                hm, scratch, segmentInit, scratch);
            cube.Init(&tiling);
            cube.ProcessGmOnly();
        }
#endif
    }

    __aicore__ inline void RunK1Add() {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (!K1CoreActive()) {
            return;
        }
        MergeFwBwd::MergeFwdBwdKernelTilingData tiling{};
        FillK1Tiling(tiling);
        GM_ADDR hm = workspace.base + workspace.hm;
        GM_ADDR scratch = workspace.base + workspace.k0Scratch;
        GM_ADDR segmentInit = workspace.base + workspace.segmentInit;
        if ASCEND_IS_AIV {
            AscendC::TPipe pipe;
            MergeFwBwd::MergeFwdBwdKernelVectorProcess<float> vector(
                hm, scratch, segmentInit, scratch, &tiling);
            vector.Init(&pipe);
            vector.ProcessAddOnly(args.h, 58U, 117U);
            pipe.Reset();
        }
#endif
    }

};

} // namespace GDN

#endif
