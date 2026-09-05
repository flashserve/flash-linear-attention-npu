/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Fused GDN prepare (arch35). Helpers: chunk_gated_delta_rule_fwd_prepare_common.h,
 * chunk_gated_delta_rule_fwd_prepare_vf.h, chunk_gated_delta_rule_fwd_prepare_datacopy.h,
 * chunk_gated_delta_rule_fwd_prepare_matmul.h.
 * Memory map: design/chunk_gated_delta_rule_fwd_prepare_ascendc-design.md.
 * Init binds every tile with OnChipBuffer::GetBuffer.
 *
 * kkt is Cube k' @ k'^T from L1 NZ k' (Stage1 (64,1,7,0) upload).
 * Cube MBH for strictly-lower L (solve_tri is_lower, LeafLeft driving):
 *   Y = I + LeafLeft @ (-L)
 *   A = LeafLeft + Y @ LeafRight
 *     tmp = I @ LeafLeft            (init_flag=True)
 *     A   = Y @ LeafRight + tmp     (init_flag=False)
 * LeafRight = blkdiag((I+L00)^{-1}, 0), LeafLeft = blkdiag(0, (I+L11)^{-1}).
 * Leaves / -L / I go L1 as 8-col ND->NZ (no UB 5HD). Leaf LoadData flips
 * ifTranspose vs I/-L/Y.
 * A = (I+L)^{-1} = [XR, 0; -XL L10 XR, XL].
 * Stage4 Fixpipe uses Arch3510 isChannelSplit (16x16 -> 16x8) into this
 * tile's u_out slot, then MTE2 back to L1[0, 64). Host audit compares
 * that NZ on u (no second ND Fixpipe). Stage5 Fixpipe L0C ->
 * gmA bf16 ND, then AIC MTE2 GM ND -> L1[256, 288) cube NZ. GET_TILING_DATA
 * / workspace GM are unbound (too many GM_ADDR).
 *
 * L1: Y [0, 64); k' aliases NegL [64, 128). Pack N+1 Stage1 waits pack N
 * Stage4 before the k' L1 write (NegL still live in Stage4). Stage5/6/7 of
 * pack N overlap that next Stage1. Do not gate Stage1 on Stage7: that
 * kills the overlap and regresses G=1 wall time.
 *
 * Scheduling: total work = Ceil(T/BT)*B*Hv. Pack size TG = 3 if G=3 else 4
 * (last pack may be shorter). AIV0: task 0,2; AIV1: task 1,3. L2Norm(q/k)
 * and Cube kkt run once per HK (OwnsHk: hv % G == 0). Sibling HV reuse
 * that kkt via Fixpipe to both AIVs when G>1. Per pack each AIV finishes
 * Stage1 ping then pong (Set taskIdx after k' L1, owners only). AIC Wait
 * that flag, kkt, then Set every sibling id (PIPE_FIX) so Stage3 can run.
 * After Stage3 L1 writes, AIV Set taskIdx+4; AIC Wait that (4, 21, 6, 23)
 * and Stage4, same 0,1,2,3 order and even/odd L0 banks as Stage2. Stage4
 * is pure AIC: the pack's Stage4 tasks finish before any Stage5. Stage6
 * is AIV after Stage3 (same 0,2 / 1,3 ping-pong). It reads k'/v from GM
 * and resident g'/β, so it does not wait Stage4/5. vb/kbg stay bf16 ND,
 * then DataCopy (64,1,srcGap,0) into L1 NZ with C0=16 (not fp32 C0=8).
 * Tail tiles (M<BT) zero-fill UB before copy-in and write M rows; aligned
 * tiles skip that. Per task like Stage5: W MMAD ping [32,48)+L0C[0,64),
 * U MMAD pong [48,64)+L0C[64,128), then M_FIX and both Fixpipes.

 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H
#define CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H

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
            gmWsY.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs + wsBase));
            gmWsA.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(userWs + wsBase + kWsYBytes),
                                  kWsASlots * kWsAElems);
        } else {
            gmWsY.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(uOut));
            gmWsA.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aOut));
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
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aOut));
        gmQHat.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(qHat));
        gmKHat.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(kHat));
        gmQRstd.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
        gmKRstd.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        hasBetaOut = 0;
        if (betaEff != nullptr) {
            gmBetaEff.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(betaEff));
            hasBetaOut = 1;
        }
    }

    // Explicit on-chip map. Offsets match design Stage 0-3.
    __aicore__ inline void InitOnChip()
    {
        OnChipBuffer buf;

        // =====================================================================
        // UB map (KiB). Offsets are from chunk_gated_delta_rule_fwd_prepare_common.h. Live ranges overlap
        // by design: S1 input [10, 75) is released after k' is on L1, then
        // S3 reuses [10, 82).
        //
        //   [0.00,  0.50)  unused      512 B         hole before ubVcsIdx
        //   [0.50,  1.00)  ubVcsIdx      512 B  u32    VCS scatter idx {0,32}
        //   [1.00,  9.00)  ubIVcs        8 KiB  fp32   I_vcs = [I32 | I32]
        //   [9.00,  9.25)  ubGPrime[0]  256 B  fp32   g' ping, [64]
        //   [9.25,  9.50)  ubBetaEff[0] 256 B  fp32   beta_eff ping, [64]
        //   [9.50,  9.75)  ubGPrime[1]  256 B  fp32   g' pong
        //   [9.75, 10.00)  ubBetaEff[1] 256 B  fp32   beta_eff pong
        //   [10.0, 26.00)  ubQ[0]      16 KiB  bf16   S1 q ping
        //   [26.0, 42.00)  ubK[0]      16 KiB  bf16   S1 k ping
        //   [42.0, 58.00)  ubQ[1]      16 KiB  bf16   S1 q pong
        //   [58.0, 74.00)  ubK[1]      16 KiB  bf16   S1 k pong
        //   [74.0, 74.50)  rstd pong    512 B  fp32   k/q rstd of 2nd task
        //   [75.0, 91.00)  ubQHat[0]   16 KiB  bf16   S1 q' ping
        //   [91.0, 107.0)  ubKHat[0]   16 KiB  bf16   S1 k' ping
        //   [107,  107.25) ubKRstd[0]  256 B  fp32   K rstd ping
        //   [107.25,107.50) ubQRstd[0] 256 B  fp32   Q rstd ping
        //   [108,  124.0)  ubQHat[1]   16 KiB  bf16   S1 q' pong
        //   [124,  140.0)  ubKHat[1]   16 KiB  bf16   S1 k' pong
        //   [140,  156.0)  ubKkt[0]    16 KiB  fp32   Cube kkt ping ND (Fixpipe NZ2ND)
        //   [156,  172.0)  ubKkt[1]    16 KiB  fp32   Cube kkt pong ND
        //   [172,  188.0)  ubMaskFp32  16 KiB  fp32   64x64 mask for VF
        // S0 only, before g' is stored:
        //   [9, 25) unused (was NZ I)   [25, 41) ubS0Zero
        // S3 (after S1 q/k released). Ping/pong for the two tasks on one AIV:
        //   ping db=0: [10, 18) LPacked  [18, 34) ResVcs  [34, 50) LFull
        //   pong db=1: [50, 58) LPacked  [58, 74) ResVcs  [74, 90) LFull
        // S6 (after S3; overlaps S1/S3, g'/β at [9,10) stay):
        //   [32, 48) ubS6K[0] 16 KiB k' ping   [48, 64) ubS6K[1] k' pong
        //   V=128: [64, 80) / [80, 96) 16 KiB. V=256: [64, 96) / [96, 128) 32 KiB.
        //   vb/kbg ND → L1 NZ C0=16 via DataCopy (64,1,srcGap,0)
        // GVA (G=2/3/4) reuses ubKkt ping/pong and the owner's L1 k' slot;
        // no extra UB/L1 buffers. Peak still UB 188 KiB (+ S6 V=256 to 128)
        // and L1 496 KiB of 512.
        // =====================================================================

        // UB[0.50, 1.00) KiB = 512 B, uint32. Only the first 8 values are used.
        // VCS scatter index for packing two 32x32 leaves into I_vcs 32x64:
        // values are {0, 32} (left leaf at col 0, right leaf at col 32).
        ubVcsIdx = buf.template GetBuffer<BufferType::ASCEND_UB, uint32_t>(kUbVcsIdx);

        // UB[1.00, 9.00) KiB = 8 KiB, fp32 ND [32, 64].
        // I_vcs = concat along K of two I_32 identity leaves. Stage3 VCS
        // copies this to ubResVcs and overwrites with (I+Lii)^{-1}.
        ubIVcs = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbIVcs);

        // UB[9.00, 25.00) KiB unused (old Cube NZ I). Overlaps g'/beta.

        // UB[25.00, 41.00) KiB = 16 KiB, fp32.
        // Stage0 ND I then zeros for leaf L1. Released before S1.
        ubS0Zero = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS0Zero);

        // UB[9.00, 10.00) resident g'/beta, then S1 q/k and S2 kkt ping/pong.
        // db=0: AIV0 task 0 / AIV1 task 1; db=1: AIV0 task 2 / AIV1 task 3.
        for (int32_t db = 0; db < 2; ++db) {
            ubGPrime[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbGPrime[db]);
            ubBetaEff[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbBetaEff[db]);
            ubQ[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1Q[db]);
            ubK[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1K[db]);
            ubKkt[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS2Kkt[db]);
            ubS6K[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS6K[db]);
            const uint32_t vOff = (V > 128 && db == 1) ? kUbS6VPong256 : kUbS6V[db];
            ubS6V[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(vOff);
            ubS6K[db].SetSize(static_cast<uint32_t>(kChunk64 * K));
            ubS6V[db].SetSize(static_cast<uint32_t>(kChunk64 * V));
            ubQHat[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1QHat[db]);
            ubKHat[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1KHat[db]);
            ubKRstd[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1KRstd[db]);
            ubQRstd[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1QRstd[db]);
        }

        // UB[172.0, 188.0) KiB = 16 KiB, fp32 ND [64, 64].
        // Strict-lower 0/1 mask. NegLowerLVF reads this to build
        // -L = -tril(exp2(g_i-g_j))*beta*kkt. Lives past S1/S2.
        ubMaskFp32 = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbMaskFp32);

        // UB[10, 90) Stage3 ping/pong. Overlaps S1 q/k; valid after k' is on L1.
        for (int32_t db = 0; db < 2; ++db) {
            ubLPacked[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3LPacked[db]);
            ubResVcs[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3ResVcs[db]);
            ubLFull[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS3LFull[db]);
        }

        // HardEvent ids (do not reuse a live id without Wait):
        //   0  S1 Q/K GM copy; S3 V_MTE3 before UB→L1 (after S1 Wait); S6 v GM
        //   1  S1 gate/beta GM copy; S6 k' GM copy
        //   2  S1 L2Norm store / gate scalar aLog
        //   3  S1 g' / beta_eff GM store
        //   4  unused
        //   5  S1 k' ND -> L1 NZ / S6 vb then kbg UB -> L1
        //   6  unused (was S3 PRINTF)
        //   7  unused
        // Stage4/5 reuse bank 0/1 for FIX_M / M_FIX / FIX_MTE2 / M_MTE1.

        // L1 512 KiB. Four task slots (taskIdx 0..3).
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
            // Y at [0,64). k' aliases NegL at [64,128); Stage3 overwrites
            // k' after Stage2.
            l1Y[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1Y(t));
        }

        // L1[480, 496) KiB = 16 KiB, fp32 cube-NZ I_64, C0=8.
        l1I = buf.template GetBuffer<BufferType::ASCEND_CB, float>(kL1ResidentI);

        // ---- L0A/L0B 64 KiB each, L0C 256 KiB. bf16 and fp32 views alias banks. ----

        // L0A/L0B [0, 16) KiB, bf16. Stage2 AIV0 tasks 0 and 2.
        l0A = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0B = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);

        // L0A/L0B [16, 32) KiB, bf16. Stage2 AIV1 tasks 1 and 3.
        l0A1 = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0Bf16Pair);
        l0B1 = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0Bf16Pair);

        // L0A/L0B [0, 16) KiB, fp32 view of the same banks as l0A/l0B.
        // Stage4/5 first MMAD (even).
        l0Af = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(0);
        l0Bf = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(0);

        // L0A/L0B [16, 32) KiB, fp32. Stage4/5 first MMAD (odd).
        l0Af1 = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(kL0Fp32Pair);
        l0Bf1 = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(kL0Fp32Pair);

        // L0A/L0B [32, 48) / [48, 64) KiB, fp32. Stage4/5 second MMAD.
        // Same physical slots as Stage7; Stage4/5 drain before Stage7.
        l0AfP = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(kL0S7Ping);
        l0BfP = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(kL0S7Ping);
        l0AfP1 = buf.template GetBuffer<BufferType::ASCEND_L0A, float>(kL0S7Pong);
        l0BfP1 = buf.template GetBuffer<BufferType::ASCEND_L0B, float>(kL0S7Pong);

        // L0C [0, 64) KiB, fp32. Stage2 kkt / Stage4 Y / Stage5 A (even).
        l0C = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);

        // L0C [64, 128) KiB, fp32. Stage2/4/5 odd tasks.
        l0C1 = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);

        // Stage7 L0A/L0B: [32, 48) even, [48, 64) odd. SetSize(16 KiB) so
        // ping does not keep the remaining 32 KiB and overlap pong.
        l0AS7 = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0S7Ping);
        l0BS7 = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0S7Ping);
        l0AS71 = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(kL0S7Pong);
        l0BS71 = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0S7Pong);
        l0AS7.SetSize(kL0S7Slot / sizeof(InDtype));
        l0BS7.SetSize(kL0S7Slot / sizeof(InDtype));
        l0AS71.SetSize(kL0S7Slot / sizeof(InDtype));
        l0BS71.SetSize(kL0S7Slot / sizeof(InDtype));
        // L0C: W 64x128 fp32 = 32 KiB at [0,64). U 64x256 = 64 KiB at [64,128).
        l0CS7 = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0CS71 = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(kL0C1);
        l0CS7.SetSize(static_cast<uint32_t>(kChunk64 * 128));
        l0CS71.SetSize(static_cast<uint32_t>(kChunk64 * 128));

        subBlock = AscendC::GetSubBlockIdx();
        if ASCEND_IS_AIV {
            // MIX 1:2: GetBlockIdx is 0,1 for cube 0's two Vectors. GetBlockNum
            // on AIV already equals the cube count (same as AIC, 28 here). The
            // runtime PRINTF "Block 0/56" is 2*cubes; do not divide again.
            coreIdx = AscendC::GetBlockIdx() / 2;
            numCore = AscendC::GetBlockNum();
        } else {
            coreIdx = AscendC::GetBlockIdx();
            numCore = AscendC::GetBlockNum();
        }
        auxReady = 0;
    }

    // ========================= Stage 0 =========================
    // Once per AIV: I_vcs + scatter idx, strict-lower 0/1 mask, zero L1
    // leaf slots for this AIV's tasks, AIV0 paints Cube I_64 ND→NZ C0=8.
    __aicore__ inline void Stage0_GenerateResidentAux()
    {
        if (auxReady != 0) {
            return;
        }
        FillVcsIdentity(ubIVcs, ubVcsIdx);
        LowerTriMaskVF(ubMaskFp32);
        // Leaf zeros first so both AIVs copy in parallel. ubS0Zero is then
        // free for AIV0 to paint I. AIV1 can leave for Stage1 while AIV0
        // uploads I. L1 leaf slots are disjoint by taskIdx (AIV0: 0,2;
        // AIV1: 1,3). Scatter only rewrites the 32x32 quadrant; TR/BL stay
        // zero if primed once.
        Duplicate(ubS0Zero, 0.0f, static_cast<int32_t>(kChunk64 * kChunk64));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (uint32_t t = static_cast<uint32_t>(subBlock); t < static_cast<uint32_t>(kTasksPerRound); t += 2) {
            UbToL1Fp32(l1LeafLeft[t], ubS0Zero, kChunk64);
            UbToL1Fp32(l1LeafRight[t], ubS0Zero, kChunk64);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        // Cube I_64 is shared L1: only AIV0 paints ND I and 8-col uploads.
        if (subBlock == 0) {
            Identity64VF(ubS0Zero);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            UbNd64ToL1Nz8(l1I, ubS0Zero);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
        auxReady = 1;
    }

    // ========================= Stage 1 =========================
    // Per-task g' (cumsum) and β_eff. Owner HK also L2Norm k then q, store
    // hat/rstd, k' ND→L1 NZ C0=16, NotifyAicStage1Done so Cube kkt can
    // overlap Q/gate. k' aliases NegL: Wait pack N Stage4 before the L1
    // write so Stage1 can overlap pack N Stage5/7 (V3 G=1 schedule).
    // Sibling HV skip K/Q and still wait Stage4 to consume the flag.
    // Tail (M<BT) zero-fills this tile before copy-in; aligned tiles skip
    // Duplicate.
    __aicore__ inline void Stage1_OneTask(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t hk = hv / HRatio;
        const int32_t db = PingPongSlot(taskIdx);
        const int32_t nElem = static_cast<int32_t>(chunk.M * K);
        const int32_t nValid = static_cast<int32_t>(chunk.M);
        const int32_t nPad = static_cast<int32_t>(chunkSize);
        const int64_t offQk = OffsetBHTD(chunk.batch, hk, chunk.tokenStart, HK, T, K);
        const int64_t offG = OffsetBHT(chunk.batch, hv, chunk.tokenStart, HV, T);
        const int64_t offRstd = OffsetBHT(chunk.batch, hk, chunk.tokenStart, HK, T);
        LocalTensor<InDtype> l1K = l1KHat[static_cast<uint32_t>(taskIdx)];
        LocalTensor<float> ubGfp = ubGPrime[db];
        LocalTensor<float> ubBfp = ubBetaEff[db];
        LocalTensor<GateDtype> ubGRaw = ubGfp.template ReinterpretCast<GateDtype>();
        LocalTensor<GateDtype> ubBRaw = ubBfp.template ReinterpretCast<GateDtype>();
        const bool ownsHk = OwnsHk(hv);

        if (db == 0) {
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }

        const bool isTail = (nValid < nPad);
        if (isTail) {
            if (ownsHk) {
                Duplicate(ubK[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
                Duplicate(ubKHat[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
                Duplicate(ubQ[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
                Duplicate(ubQHat[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            }
            Duplicate(ubGfp, 0.0f, nPad);
            Duplicate(ubBfp, 0.0f, nPad);
            SetFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::V_MTE2>(1);
        }

        if (ownsHk) {
            DataCopy(ubK[db], gmK[offQk], nElem);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            L2NormK128VF<InDtype>(ubK[db], ubKHat[db], ubKRstd[db], static_cast<uint32_t>(nValid), kGdnL2NormEps);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            WaitAicStage4Done(taskIdx);
            UploadBf16NdToL1(l1K, ubKHat[db], static_cast<uint32_t>(K));
            NotifyAicStage1Done(taskIdx);
            DataCopy(gmKHat[offQk], ubKHat[db], nElem);
            CopyUbToGmElems(gmKRstd[offRstd], ubKRstd[db], static_cast<uint32_t>(nValid));

            DataCopy(ubQ[db], gmQ[offQk], nElem);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            L2NormK128VF<InDtype>(ubQ[db], ubQHat[db], ubQRstd[db], static_cast<uint32_t>(nValid), kGdnL2NormEps);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(gmQHat[offQk], ubQHat[db], nElem);
            CopyUbToGmElems(gmQRstd[offRstd], ubQRstd[db], static_cast<uint32_t>(nValid));
        } else {
            WaitAicStage4Done(taskIdx);
        }

        // g / beta share the resident 256 B slots. GateDtype view of the same
        // address: fp32 copy is identity; b16/fp16 fills the first 128 B, then
        // one VL LoadCast stores 64 fp32. n<=64, load-then-store, no ubQHat.
        CopyGmToUbElems(ubGRaw, gmG[offG], static_cast<uint32_t>(nValid));
        CopyGmToUbElems(ubBRaw, gmBeta[offG], static_cast<uint32_t>(nValid));
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        if (useGateInKernel != 0) {
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(2);
            float aLog = ScalarToFp32(gmALog.GetValue(hv));
            float dt = (hasDtBias != 0) ? ScalarToFp32(gmDt.GetValue(hv)) : 0.0f;
            SetFlag<HardEvent::S_V>(2);
            WaitFlag<HardEvent::S_V>(2);
            GateSoftplusVF<GateDtype>(ubGRaw, ubGfp, aLog, dt, static_cast<uint32_t>(nValid));
        } else if constexpr (!IsSameType<GateDtype, float>::value) {
            CastToFp32VF<GateDtype>(ubGRaw, ubGfp, static_cast<uint32_t>(nValid));
        }
        float scale = (useExp2 != 0) ? kGdnRcpLn2 : 1.0f;
        CumsumScaleVF(ubGfp, scale, static_cast<uint32_t>(nValid));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyUbToGmElems(gmGOut[offG], ubGfp, static_cast<uint32_t>(nValid));

        if (useBetaSigmoid != 0) {
            float bscale = (allowNegEigval != 0) ? 2.0f : 1.0f;
            BetaSigmoidVF<GateDtype>(ubBRaw, ubBfp, bscale, static_cast<uint32_t>(nValid));
        } else if constexpr (!IsSameType<GateDtype, float>::value) {
            CastToFp32VF<GateDtype>(ubBRaw, ubBfp, static_cast<uint32_t>(nValid));
        }

        if (hasBetaOut != 0) {
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            CopyUbToGmElems(gmBetaEff[offG], ubBfp, static_cast<uint32_t>(nValid));
        }
    }

    // Stage2 helper: dualDstCtl splits M or N across the two AIVs; it does
    // not broadcast. G>1 needs the full 64x64 kkt on both Vectors, so two
    // Fixpipes.
    __aicore__ inline void DumpKktToUb(LocalTensor<float> l0c, int32_t db, uint8_t subBlk, uint8_t bank,
                                       bool shareBothAiv)
    {
        const uint32_t n = static_cast<uint32_t>(chunkSize);
        if (!shareBothAiv) {
            FixpipeL0cToUbFp32Nd(ubKkt[db], l0c, n, subBlk);
            return;
        }
        FixpipeL0cToUbFp32Nd(ubKkt[db], l0c, n, 0);
        SetFlag<HardEvent::FIX_MTE2>(bank);
        WaitFlag<HardEvent::FIX_MTE2>(bank);
        FixpipeL0cToUbFp32Nd(ubKkt[db], l0c, n, 1);
    }

    // After one Cube kkt, wake every HV in this pack that shares the same
    // (chunk, hk). G=1 notifies only the owner.
    __aicore__ inline void NotifyKktSiblings(int64_t base, int64_t nThis, int64_t ownerTask)
    {
        const int64_t ownerWork = base + ownerTask;
        const int64_t ownerChunk = ownerWork / HV;
        const int64_t ownerHv = ownerWork % HV;
        const int64_t hvLo = ownerHv - ((HRatio <= 1) ? 0 : (ownerHv % HRatio));
        const int64_t hvHi = hvLo + ((HRatio <= 1) ? 1 : HRatio);
        for (int64_t t2 = 0; t2 < nThis; ++t2) {
            const int64_t work2 = base + t2;
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
    // Cube kkt = k' @ k'^T, once per HK after WaitAivStage1Done. Even tasks
    // use L0[0,16)/L0C[0,64); odd use L0[16,32)/L0C[64,128). Wait M_MTE1
    // (primed in ProcessAic). Inner Matmul Wait FIX_M matches V3. G>1 dumps
    // the full 64x64 to both AIVs. G=1 NotifyAivKktDone here; G>1 siblings
    // from ProcessAic.
    __aicore__ inline void Stage2_AicOne(int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const int32_t kk = static_cast<int32_t>(K);
        const uint8_t subBlk = static_cast<uint8_t>(taskIdx & 1);
        const int32_t db = PingPongSlot(taskIdx);
        const bool shareBothAiv = (HRatio > 1);
        if ((taskIdx & 1) == 0) {
            WaitFlag<HardEvent::M_MTE1>(0);
            MatmulToL0C<InDtype>(l1KHat[taskIdx], l1KHat[taskIdx], l0A, l0B, l0C, bt, bt, kk, true, false, false, 0);
            SetFlag<HardEvent::M_FIX>(0);
            SetFlag<HardEvent::M_MTE1>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            DumpKktToUb(l0C, db, subBlk, 0, shareBothAiv);
            SetFlag<HardEvent::FIX_M>(0);
        } else {
            WaitFlag<HardEvent::M_MTE1>(1);
            MatmulToL0C<InDtype>(l1KHat[taskIdx], l1KHat[taskIdx], l0A1, l0B1, l0C1, bt, bt, kk, true, false, false, 1);
            SetFlag<HardEvent::M_FIX>(1);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            DumpKktToUb(l0C1, db, subBlk, 1, shareBothAiv);
            SetFlag<HardEvent::FIX_M>(1);
        }
        if (HRatio <= 1) {
            NotifyAivKktDone(taskIdx);
        }
    }

    // ========================= Stage 3 =========================
    // Wait Cube kkt, build -L = -tril(exp2(g_i-g_j))*β*kkt, pack diag
    // leaves, VCS (I+Lii)^{-1}, upload leaves/-L to L1 NZ C0=8, Notify AIC.
    __aicore__ inline void Stage3_AivOne(int64_t hv, int64_t taskIdx)
    {
        WaitCubeKktDone(taskIdx);
        const int32_t db = PingPongSlot(taskIdx);
        const int32_t kktDb = PingPongSlot(OwnerTaskIdx(hv, taskIdx));
        Stage3_ConstructLAndVcs(ubKkt[kktDb], ubGPrime[db], ubBetaEff[db], ubMaskFp32, ubLFull[db], ubLPacked[db],
                                ubIVcs, ubResVcs[db], ubVcsIdx);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadDiagLeavesAndFullAToL1(l1LeafRight[taskIdx], l1LeafLeft[taskIdx], l1NegL[taskIdx], ubResVcs[db],
                                     ubLFull[db]);
        NotifyAicStage3Done(taskIdx);
    }

    // ========================= Stage 4 =========================
    // Y = I + LeafLeft @ (-L). Two MMADs (init then accumulate), Fixpipe
    // NZ C0=8 into workspace, MTE2 to L1 Y. Set FIX_M between MMADs so the
    // inner Matmul Wait FIX_M can proceed.
    __aicore__ inline void Stage4_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        (void)chunk;
        (void)hv;
        if (bank == 0) {
            WaitFlag<HardEvent::M_MTE1>(0);
            MatmulToL0C<float>(l1I, l1I, l0Af, l0Bf, l0C, bt, bt, bt, true, true, false, 0);
            SetFlag<HardEvent::FIX_M>(0);
            MatmulToL0C<float>(l1LeafLeft[taskIdx], l1NegL[taskIdx], l0AfP, l0BfP, l0C, bt, bt, bt, false, true, true, 0);
            SetFlag<HardEvent::M_FIX>(0);
            SetFlag<HardEvent::M_MTE1>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeL0cToGmNzCs(gmWsY, l0C, kChunk64);
            SetFlag<HardEvent::FIX_M>(0);
            SetFlag<HardEvent::FIX_MTE2>(0);
            WaitFlag<HardEvent::FIX_MTE2>(0);
            CopyGmNzToL1Fp32(l1Y[taskIdx], gmWsY, kChunk64);
            NotifyAivStage4Done(taskIdx);
        } else {
            WaitFlag<HardEvent::M_MTE1>(1);
            MatmulToL0C<float>(l1I, l1I, l0Af1, l0Bf1, l0C1, bt, bt, bt, true, true, false, 1);
            SetFlag<HardEvent::FIX_M>(1);
            MatmulToL0C<float>(l1LeafLeft[taskIdx], l1NegL[taskIdx], l0AfP1, l0BfP1, l0C1, bt, bt, bt, false, true, true,
                               1);
            SetFlag<HardEvent::M_FIX>(1);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToGmNzCs(gmWsY, l0C1, kChunk64);
            SetFlag<HardEvent::FIX_M>(1);
            SetFlag<HardEvent::FIX_MTE2>(1);
            WaitFlag<HardEvent::FIX_MTE2>(1);
            CopyGmNzToL1Fp32(l1Y[taskIdx], gmWsY, kChunk64);
            NotifyAivStage4Done(taskIdx);
        }
    }

    // ========================= Stage 5 =========================
    // A = LeafLeft + Y @ LeafRight. Fixpipe ND to gmA; Nd2Nz into L1 A.
    // Aligned (M==BT) copies from gmA. Tail (M<BT): second Fixpipe to this
    // task's gmWsA slot (full 64x64) then Nd2Nz.
    __aicore__ inline void Stage5_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        const uint32_t n = static_cast<uint32_t>(chunkSize);
        const uint32_t m = static_cast<uint32_t>(chunk.M);
        const int64_t offA = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, chunkSize);
        if (bank == 0) {
            WaitFlag<HardEvent::M_MTE1>(0);
            MatmulToL0C<float>(l1I, l1LeafLeft[taskIdx], l0Af, l0Bf, l0C, bt, bt, bt, true, false, false, 0);
            SetFlag<HardEvent::FIX_M>(0);
            MatmulToL0C<float>(l1Y[taskIdx], l1LeafRight[taskIdx], l0AfP, l0BfP, l0C, bt, bt, bt, false, false, false, 0);
            SetFlag<HardEvent::M_FIX>(0);
            SetFlag<HardEvent::M_MTE1>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeL0cToGmNd<InDtype>(gmA[offA], l0C, m, n, n);
            SetFlag<HardEvent::FIX_M>(0);
            SetFlag<HardEvent::FIX_MTE2>(0);
            WaitFlag<HardEvent::FIX_MTE2>(0);
            if (m == n) {
                CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmA[offA], n, n);
            } else {
                FixpipeL0cToGmNd<InDtype>(gmWsA[WsAOffset(taskIdx)], l0C, n, n, n);
                SetFlag<HardEvent::FIX_MTE2>(0);
                WaitFlag<HardEvent::FIX_MTE2>(0);
                CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmWsA[WsAOffset(taskIdx)], n, n);
            }
        } else {
            WaitFlag<HardEvent::M_MTE1>(1);
            MatmulToL0C<float>(l1I, l1LeafLeft[taskIdx], l0Af1, l0Bf1, l0C1, bt, bt, bt, true, false, false, 1);
            SetFlag<HardEvent::FIX_M>(1);
            MatmulToL0C<float>(l1Y[taskIdx], l1LeafRight[taskIdx], l0AfP1, l0BfP1, l0C1, bt, bt, bt, false, false, false,
                               1);
            SetFlag<HardEvent::M_FIX>(1);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToGmNd<InDtype>(gmA[offA], l0C1, m, n, n);
            SetFlag<HardEvent::FIX_M>(1);
            SetFlag<HardEvent::FIX_MTE2>(1);
            WaitFlag<HardEvent::FIX_MTE2>(1);
            if (m == n) {
                CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmA[offA], n, n);
            } else {
                FixpipeL0cToGmNd<InDtype>(gmWsA[WsAOffset(taskIdx)], l0C1, n, n, n);
                SetFlag<HardEvent::FIX_MTE2>(1);
                WaitFlag<HardEvent::FIX_MTE2>(1);
                CopyGmNdToL1Nz<InDtype>(l1A[taskIdx], gmWsA[WsAOffset(taskIdx)], n, n);
            }
        }
    }

    // ========================= Stage 6 =========================
    // vb = v * β, kbg = k' * β * exp2(g'). ND→L1 NZ C0=16. V3 runs this
    // after Stage3 (no Stage5 wait) so it overlaps Cube Stage4/5.
    // Tail zero-fills UB then copies M rows.
    __aicore__ inline void Stage6_AivOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t hk = hv / HRatio;
        const int64_t offK = OffsetBHTD(chunk.batch, hk, chunk.tokenStart, HK, T, K);
        const int64_t offV = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
        const int32_t nValid = static_cast<int32_t>(chunk.M);
        const int32_t nPad = static_cast<int32_t>(chunkSize);
        const int32_t nK = nValid * static_cast<int32_t>(K);
        const int32_t nV = nValid * static_cast<int32_t>(V);
        const int32_t db = PingPongSlot(taskIdx);
        LocalTensor<InDtype> ubKnd = ubS6K[db];
        LocalTensor<InDtype> ubVnd = ubS6V[db];
        LocalTensor<float> beta = ubBetaEff[db];
        LocalTensor<float> g = ubGPrime[db];
        if (nValid < nPad) {
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            Duplicate(ubVnd, static_cast<InDtype>(0), nPad * static_cast<int32_t>(V));
            Duplicate(ubKnd, static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        DataCopy(ubVnd, gmV[offV], nV);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ScaleRowsVF<InDtype>(ubVnd, ubVnd, beta, static_cast<uint32_t>(chunkSize), static_cast<uint32_t>(V));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadBf16NdToL1(l1Vb[taskIdx], ubVnd, static_cast<uint32_t>(V));

        DataCopy(ubKnd, gmKHat[offK], nK);
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
        ScaleRowsBetaExp2gVF<InDtype>(ubKnd, ubKnd, beta, g, static_cast<uint32_t>(chunkSize));
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        UploadBf16NdToL1(l1Kbg[taskIdx], ubKnd, static_cast<uint32_t>(K));
    }

    // ========================= Stage 7 =========================
    // W = A @ kbg (L0 ping), U = A @ vb (L0 pong). Drain each Fixpipe
    // before the next MMAD so L0C/Fixpipe flags stay balanced. V=256 U is
    // two n=128 MMADs on the U L0C.
    __aicore__ inline void Stage7_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t offW = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, K);
        const int64_t offU = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
        const uint32_t rows = static_cast<uint32_t>(chunk.M);
        const uint32_t nK = static_cast<uint32_t>(K);
        const uint32_t nV = static_cast<uint32_t>(V);
        const int32_t bt = static_cast<int32_t>(chunkSize);
        WaitFlag<HardEvent::M_MTE1>(0);
        WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Kbg[taskIdx], l0AS7, l0BS7, l0CS7, bt, static_cast<int32_t>(nK), bt, 0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeL0cToGmNd<InDtype>(gmW[offW], l0CS7, rows, nK, nK);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(0);

        WaitFlag<HardEvent::M_MTE1>(1);
        WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb[taskIdx], l0AS71, l0BS71, l0CS71, bt,
                               static_cast<int32_t>(kGdnHeadDimK), bt, 1);
        SetFlag<HardEvent::M_FIX>(1);
        SetFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::M_FIX>(1);
        FixpipeL0cToGmNd<InDtype>(gmU[offU], l0CS71, rows, kGdnHeadDimK, nV);
        SetFlag<HardEvent::FIX_M>(1);

        if (nV > kGdnHeadDimK) {
            WaitFlag<HardEvent::M_MTE1>(1);
            WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb1[taskIdx], l0AS71, l0BS71, l0CS71, bt,
                                   static_cast<int32_t>(kGdnHeadDimK), bt, 1);
            SetFlag<HardEvent::M_FIX>(1);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToGmNd<InDtype>(gmU[offU + kGdnHeadDimK], l0CS71, rows, kGdnHeadDimK, nV);
            SetFlag<HardEvent::FIX_M>(1);
        }
    }

    __aicore__ inline bool PackHasTail(int64_t base, int64_t nThis)
    {
        for (int64_t t = 0; t < nThis; ++t) {
            if (GetChunkRange(*this, gmCu, gmIdx, (base + t) / HV).M < chunkSize) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void ProcessAiv()
    {
        Stage0_GenerateResidentAux();

        const int64_t packSize = TasksPerPack();
        const int64_t nPacks = CeilDiv(totalChunks, packSize);
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const int64_t base = pack * packSize;
            const int64_t nThis = (totalChunks - base) < packSize ? (totalChunks - base) : packSize;
            if (pack != coreIdx) {
                const int64_t prevPack = pack - numCore;
                const int64_t prevBase = prevPack * packSize;
                const int64_t prevN = (totalChunks - prevBase) < packSize ? (totalChunks - prevBase) : packSize;
                if (PackHasTail(base, nThis) || PackHasTail(prevBase, prevN)) {
                    WaitAicStage7Done();
                }
            }
            for (int64_t t = subBlock; t < nThis; t += 2) {
                const int64_t workId = base + t;
                Stage1_OneTask(GetChunkRange(*this, gmCu, gmIdx, workId / HV), workId % HV, t);
            }
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            for (int64_t t = subBlock; t < nThis; t += 2) {
                const int64_t workId = base + t;
                Stage3_AivOne(workId % HV, t);
            }

            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);

            for (int64_t t = subBlock; t < nThis; t += 2) {
                const int64_t workId = base + t;
                Stage6_AivOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV), workId % HV, t);
                NotifyAicStage6Done(t);
            }
        }
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
        const int64_t nPacks = CeilDiv(totalChunks, packSize);
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const int64_t base = pack * packSize;
            const int64_t nThis = (totalChunks - base) < packSize ? (totalChunks - base) : packSize;
            WaitFlag<HardEvent::FIX_MTE1>(0);
            for (int64_t t = 0; t < nThis; ++t) {
                const int64_t workId = base + t;
                if (!OwnsHk(workId % HV)) {
                    continue;
                }
                WaitAivStage1Done(t);
                Stage2_AicOne(t);
                if (HRatio > 1) {
                    NotifyKktSiblings(base, nThis, t);
                }
            }
            for (int64_t t = 0; t < nThis; ++t) {
                WaitAivStage3Done(t);
                const int64_t workId = base + t;
                Stage4_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV), workId % HV, t);
                SetFlag<HardEvent::MTE2_MTE1>(t);
            }
            for (int64_t t = 0; t < nThis; ++t) {
                const int64_t workId = base + t;
                WaitFlag<HardEvent::MTE2_MTE1>(t);
                Stage5_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV), workId % HV, t);
                SetFlag<HardEvent::MTE2_MTE1>(t);
            }
            for (int64_t t = 0; t < nThis; ++t) {
                const int64_t workId = base + t;
                WaitAivStage6Done(t);
                WaitFlag<HardEvent::MTE2_MTE1>(t);
                Stage7_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV), workId % HV, t);
            }
            const int64_t nextPack = pack + numCore;
            if (nextPack < nPacks) {
                const int64_t nextBase = nextPack * packSize;
                const int64_t nextN = (totalChunks - nextBase) < packSize ? (totalChunks - nextBase) : packSize;
                if (PackHasTail(base, nThis) || PackHasTail(nextBase, nextN)) {
                    NotifyAivStage7Done();
                }
            }
            SetFlag<HardEvent::FIX_MTE1>(0);
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
    // GM
    // Input
    GlobalTensor<InDtype> gmQ;
    GlobalTensor<InDtype> gmK;
    GlobalTensor<InDtype> gmV;
    GlobalTensor<GateDtype> gmG;
    GlobalTensor<GateDtype> gmBeta;
    GlobalTensor<ALogDtype> gmALog;
    GlobalTensor<ALogDtype> gmDt;
    // Output
    GlobalTensor<InDtype> gmQHat;
    GlobalTensor<InDtype> gmKHat;
    GlobalTensor<InDtype> gmW;
    GlobalTensor<InDtype> gmU;
    GlobalTensor<InDtype> gmA;
    GlobalTensor<float> gmQRstd;
    GlobalTensor<float> gmKRstd;
    GlobalTensor<float> gmGOut;
    GlobalTensor<float> gmBetaEff;
    // Workspace
    GlobalTensor<float> gmWsY;
    GlobalTensor<InDtype> gmWsA;

    // Local
    // UB
    // S0
    LocalTensor<uint32_t> ubVcsIdx;
    LocalTensor<float> ubIVcs, ubS0Zero, ubMaskFp32;

    // S1
    LocalTensor<float> ubGPrime[2], ubBetaEff[2];
    LocalTensor<InDtype> ubQ[2], ubK[2], ubQHat[2], ubKHat[2];
    LocalTensor<float> ubKRstd[2], ubQRstd[2];

    // S2
    LocalTensor<float> ubKkt[2];

    LocalTensor<float> ubLPacked[2], ubResVcs[2], ubLFull[2];

    LocalTensor<InDtype> ubS6K[2], ubS6V[2];

    // L1
    LocalTensor<float> l1I;
    LocalTensor<InDtype> l1KHat[4], l1A[4], l1Kbg[4], l1Vb[4], l1Vb1[4];
    LocalTensor<float> l1NegL[4], l1LeafRight[4], l1LeafLeft[4], l1Y[4];

    // L0
    LocalTensor<InDtype> l0A, l0B, l0A1, l0B1, l0AS7, l0BS7, l0AS71, l0BS71;
    LocalTensor<float> l0Af, l0Bf, l0Af1, l0Bf1, l0AfP, l0BfP, l0AfP1, l0BfP1;
    LocalTensor<float> l0C, l0C1, l0CS7, l0CS71;

    int64_t hasBetaOut;
};

} // namespace ChunkGatedDeltaRuleFwdPrepare

#endif
