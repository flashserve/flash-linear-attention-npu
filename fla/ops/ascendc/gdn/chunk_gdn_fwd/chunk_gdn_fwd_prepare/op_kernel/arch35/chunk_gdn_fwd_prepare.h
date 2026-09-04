/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Fused GDN prepare (arch35). Helpers: mem.h, common.h, offset.h,
 * vf.h, copy.h, matmul.h. Reference VF: l2norm_regbase_vf.h
 * and MulReduceScatterVF32 in vf.h (hoist + inner unroll 2).
 * Memory map: design/chunk_gdn_fwd_prepare_ascendc-design.md.
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
 * ifTranspose vs I/-L/Y (w_mmad_demo).
 * A = (I+L)^{-1} = [XR, 0; -XL L10 XR, XL].
 * Stage4 Fixpipe uses Arch3510 isChannelSplit (16x16 -> 16x8) into this
 * tile's u_out slot, then MTE2 back to L1[0, 64). Host audit compares
 * that NZ on u (no second ND Fixpipe). Stage5 Fixpipe L0C ->
 * gmA bf16 ND, then AIC MTE2 GM ND -> L1[256, 288) cube NZ. GET_TILING_DATA
 * / workspace GM are unbound (too many GM_ADDR).
 *
 * L1: Y [0, 64); k' aliases NegL [64, 128). Pack N+1 Stage1 waits Stage4.
 *
 * Scheduling: total work = Ceil(T/BT)*B*Hv. Each AIC + two AIV process up
 * to 4 tiles per pack (last pack may be shorter). AIV0: task 0,2; AIV1:
 * task 1,3. Per pack each AIV finishes Stage1 ping then pong (Set taskIdx
 * after k' L1). AIC Wait that flag, kkt, then Set the same id (PIPE_FIX)
 * so the matching AIV can Stage3. After Stage3 L1 writes, AIV Set
 * taskIdx+4; AIC Wait that (4, 21, 6, 23) and Stage4, same 0,1,2,3 order
 * and even/odd L0 banks as Stage2. Stage4 is pure AIC: the pack's 4 Stage4
 * tasks finish before any Stage5 task starts (same 0,1,2,3 / even-odd L0).
 * Stage6 is AIV after Stage3 (same 0,2 / 1,3 ping-pong). It reads k'/v from
 * GM and resident g'/β, so it does not wait Stage4/5. vb/kbg stay bf16 ND,
 * then DataCopy (64,1,srcGap,0) into L1 NZ with C0=16 (not fp32 C0=8).
 * After the pack's Stage3, AIV waits Stage5 (so the sibling AIV has
 * also left Stage3 UB) then Stage6. AIC PipeBarrier ALL then Stage7.
 * Per task like Stage5: W MMAD ping [32,48)+L0C[0,64), U MMAD pong
 * [48,64)+L0C[64,128), then M_FIX and both Fixpipes. FIX_MTE2 drains
 * FIX before the next task reuses ping.

 */

#ifndef CHUNK_GDN_FWD_PREPARE_H
#define CHUNK_GDN_FWD_PREPARE_H

#include "kernel_operator.h"
#include "mem.h"
#include "common.h"
#include "offset.h"
#include "vf.h"
#include "copy.h"
#include "matmul.h"
#include "l2norm_regbase_vf.h"

namespace GdnStage {
using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename GateDtype, typename ALogDtype>
class ChunkGdnFwdPrepareKernel : public PrepareState {
public:
    using InDtype = bfloat16_t;

    template <typename Td>
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
                                GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR cu, GM_ADDR idx,
                                GM_ADDR gOut, GM_ADDR wOut, GM_ADDR uOut, GM_ADDR aOut,
                                GM_ADDR qHat, GM_ADDR kHat, GM_ADDR qRstd, GM_ADDR kRstd,
                                GM_ADDR betaEff, GM_ADDR workspace, const Td &td)
    {
        LoadTiling(td);
        InitOnChip();
        GM_ADDR userWs = GetUserWorkspace(workspace);
        const uint32_t wsBase = static_cast<uint32_t>(coreIdx) * kWsPerCoreBytes;
        if (userWs != nullptr) {
            gmWsY.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(userWs + wsBase));
            gmWsA.SetGlobalBuffer(
                reinterpret_cast<__gm__ InDtype *>(userWs + wsBase + kWsYBytes),
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

    // Explicit on-chip map. Offsets match design Stage 0-3; GetBuffer style
    // matches solve_tri_ascend950_64.h Init.
    __aicore__ inline void InitOnChip()
    {
        OnChipBuffer buf;

        // =====================================================================
        // UB map (KiB). Offsets are from offset.h. Live ranges overlap
        // by design: S1 input [10, 75) is released after k' is on L1, then
        // S3 reuses [10, 82). Bit-mask at [0, 0.5) stays until kernel exit.
        //
        //   [0.00,  0.50)  ubMaskBits     512 B  uint8  64x64 strict-lower bits
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
        //   [107.50,107.75) ubGateTmp  256 B  fp32   fused-gate scratch [64]
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
        // =====================================================================

        // UB[0.00, 0.50) KiB = 512 B, uint8.
        // Strict-lower triangular bit-mask of the 64x64 chunk.
        // Packed as 64*64/8 bytes: row i, bit j is 1 iff token-row i > col j.
        // VF cannot consume bits; Stage0 expands this once into ubMaskFp32.
        ubMaskBits = buf.template GetBuffer<BufferType::ASCEND_UB, uint8_t>(kUbMaskBits);

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

        // UB[9.00, 10.00) resident g'/beta, then S1 q/k/g/beta and S2 kkt ping/pong.
        // db=0: AIV0 task 0 / AIV1 task 1; db=1: AIV0 task 2 / AIV1 task 3.
        for (int32_t db = 0; db < 2; ++db) {
            ubGPrime[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbGPrime[db]);
            ubBetaEff[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbBetaEff[db]);
            ubQ[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1Q[db]);
            ubK[db] = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(kUbS1K[db]);
            ubG[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1G[db]);
            ubBeta[db] = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1Beta[db]);
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
        // Expanded 0/1 mask from ubMaskBits. ConstructLowerLExp2VF reads this
        // to build L = tril(exp2(g_i-g_j))*beta*kkt. Lives past S1/S2.
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

        // UB[107.50, 107.75) KiB. Fused-gate tmp, [64]. Ping rstd is at 107.
        ubGateTmp = buf.template GetBuffer<BufferType::ASCEND_UB, float>(kUbS1GateTmp);

        // L1 512 KiB. Four task slots (taskIdx 0..3).
        for (uint32_t t = 0; t < static_cast<uint32_t>(kTasksPerRound); ++t) {
            l1KHat[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1KHat(t));
            l1NegL[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1NegL(t));
            l1LeafRight[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1LeafRight(t));
            l1LeafLeft[t] = buf.template GetBuffer<BufferType::ASCEND_CB, float>(L1LeafLeft(t));
            l1A[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentA(t));
            l1Kbg[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentKbg(t));
            l1Kbg1[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(
                L1ResidentKbg(t) + kSlotBf16_64);
            l1Vb[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(L1ResidentVb(t));
            l1Vb[t].SetSize(static_cast<uint32_t>(kChunk64 * V));
            l1Kbg[t].SetSize(static_cast<uint32_t>(kChunk64 * K));
            l1Vb1[t] = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(
                L1ResidentVb(t) + kBytesK128);
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
        // V=256 U-MMAD needs 64x256 bf16 = 32 KiB L0B. W then U are
        // PIPE_ALL-separated, so this can alias the ping slot.
        l0BS7Wide = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(kL0S7Ping);
        l0BS7Wide.SetSize((32 * kPrepareKb) / sizeof(InDtype));
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

    // AIV0: task 0 ping, 2 pong. AIV1: task 1 ping, 3 pong.
    __aicore__ inline int32_t PingPongSlot(int64_t taskIdx) const
    {
        return static_cast<int32_t>((taskIdx >> 1) & 1);
    }

    // ========================= Stage 0 =========================
    __aicore__ inline void Stage0_GenerateResidentAux()
    {
        if (auxReady != 0) {
            return;
        }
        Prepare::Stage0_GenIdentity(ubIVcs, ubVcsIdx);
        Prepare::Stage0_ExpandBitMaskToFp32(ubMaskFp32);
        // Leaf zeros first so both AIVs copy in parallel. ubS0Zero is then
        // free for AIV0 to paint I. AIV1 can leave for Stage1 while AIV0
        // uploads I. L1 leaf slots are disjoint by taskIdx (AIV0: 0,2;
        // AIV1: 1,3). Scatter only rewrites the 32x32 quadrant; TR/BL stay
        // zero if primed once.
        Duplicate(ubS0Zero, 0.0f, static_cast<int32_t>(kChunk64 * kChunk64));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (uint32_t t = static_cast<uint32_t>(subBlock);
             t < static_cast<uint32_t>(kTasksPerRound); t += 2) {
            UbToL1Fp32(l1LeafLeft[t], ubS0Zero, kChunk64);
            UbToL1Fp32(l1LeafRight[t], ubS0Zero, kChunk64);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        // Cube I_64 is shared L1: only AIV0 paints ND I and 8-col uploads.
        if (subBlock == 0) {
            Prepare::Stage0_PaintIdentity64(ubS0Zero);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            UbNd64ToL1Nz8(l1I, ubS0Zero);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
        auxReady = 1;
    }

    // ========================= Stage 1 =========================
    // K then Q: each is GM copy, L2Norm, hat/rstd store. K also ND→L1 NZ
    // (64,1,7,0). NotifyAic after k' is on L1 so Cube kkt overlaps Q/gate.
    // Hats/rstd are ping-pong (db), so Q L2Norm can overlap K rstd MTE3
    // and the next task can overlap this task's hat MTE3.
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


        if (db == 0) {
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }

        const bool isTail = (nValid < nPad);
        if (isTail) {
            Duplicate(ubK[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            Duplicate(ubKHat[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            Duplicate(ubQ[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            Duplicate(ubQHat[db], static_cast<InDtype>(0), nPad * static_cast<int32_t>(K));
            Duplicate(ubGfp, 0.0f, nPad);
            Duplicate(ubBfp, 0.0f, nPad);
            SetFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::V_MTE2>(1);
        }

        // Process K
        DataCopy(ubK[db], gmK[offQk], nElem);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        GdnL2Norm::L2NormFwdK128VF<InDtype>(ubK[db], ubKHat[db], ubKRstd[db], static_cast<uint32_t>(nValid), kGdnL2NormEps);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        // k' shares NegL slots. Wait previous pack Stage4 before the L1
        // write (not after SetFlag): writing first would stomp NegL.
        // Pack 0 uses AIC's primed NotifyAivStage4Done.
        WaitAicStage4Done(taskIdx);
        for (uint16_t frac = 0; frac < 8; ++frac) {
            DataCopy(l1K[static_cast<int32_t>(frac) * 16 * 64], ubKHat[db][static_cast<int32_t>(frac) * 16], DataCopyParams(64, 1, 7, 0));
        }
        CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx));
        DataCopy(gmKHat[offQk], ubKHat[db], nElem);
        CopyUbToGmElems(gmKRstd[offRstd], ubKRstd[db], static_cast<uint32_t>(nValid));

        // Process Q
        DataCopy(ubQ[db], gmQ[offQk], nElem);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        GdnL2Norm::L2NormFwdK128VF<InDtype>(ubQ[db], ubQHat[db], ubQRstd[db], static_cast<uint32_t>(nValid), kGdnL2NormEps);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(gmQHat[offQk], ubQHat[db], nElem);
        CopyUbToGmElems(gmQRstd[offRstd], ubQRstd[db], static_cast<uint32_t>(nValid));

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
            CopyToFp32VF<GateDtype>(ubGRaw, ubGfp, static_cast<uint32_t>(nValid));
        }
        float scale = (useExp2 != 0) ? kGdnRcpLn2 : 1.0f;
        ChunkCumsumScaleVF(ubGfp, scale, static_cast<uint32_t>(nValid));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyUbToGmElems(gmGOut[offG], ubGfp, static_cast<uint32_t>(nValid));

        if (useBetaSigmoid != 0) {
            float bscale = (allowNegEigval != 0) ? 2.0f : 1.0f;
            BetaSigmoidVF<GateDtype>(ubBRaw, ubBfp, bscale, static_cast<uint32_t>(nValid));
        } else if constexpr (!IsSameType<GateDtype, float>::value) {
            CopyToFp32VF<GateDtype>(ubBRaw, ubBfp, static_cast<uint32_t>(nValid));
        }

        if (hasBetaOut != 0) {
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            CopyUbToGmElems(gmBetaEff[offG], ubBfp, static_cast<uint32_t>(nValid));
        }
    }

    // Stage1 k' / Stage6 vb,kbg: ND row-major -> L1 cube NZ C0=16.
    // Dest N-frac fc at fc*16*64, src C0 at fc*16, src row gap = cols/16-1.
    // Caller owns V_MTE3 before and MTE3_V after.
    __aicore__ inline void UploadBf16NdToL1(LocalTensor<InDtype> l1,
                                            LocalTensor<InDtype> ubNd, uint32_t cols)
    {
        const uint32_t nFrac = cols / 16;
        const uint16_t srcGap = static_cast<uint16_t>(nFrac - 1);
        for (uint32_t fc = 0; fc < nFrac; ++fc) {
            DataCopy(l1[static_cast<int32_t>(fc) * 16 * static_cast<int32_t>(kChunk64)],
                     ubNd[static_cast<int32_t>(fc * 16)],
                     DataCopyParams(kChunk64, 1, srcGap, 0));
        }
    }


    __aicore__ inline void CopyNdToL1ViaUbNd2Nz(LocalTensor<InDtype> l1,
                                                LocalTensor<InDtype> ubNd,
                                                uint32_t cols)
    {
        Nd2NzParams nd;
        nd.ndNum = 1;
        nd.nValue = static_cast<uint32_t>(chunkSize);
        nd.dValue = cols;
        nd.srcDValue = cols;
        nd.srcNdMatrixStride = 0;
        nd.dstNzC0Stride = static_cast<uint32_t>(chunkSize);
        nd.dstNzNStride = 1;
        nd.dstNzMatrixStride = 0;
        LocalTensor<InDtype> ubNz = ubQ[0];
        SetFlag<HardEvent::V_MTE3>(5);
        WaitFlag<HardEvent::V_MTE3>(5);
        DataCopy(ubNz, ubNd, nd);
        UbToL1Elems<InDtype>(l1, ubNz, static_cast<uint32_t>(chunkSize * cols));
        SetFlag<HardEvent::MTE3_V>(5);
        WaitFlag<HardEvent::MTE3_V>(5);
    }

    // Cube wait id for Vector taskIdx. Odd tasks are AIV1 → +16.
    __aicore__ inline uint16_t CubeWaitFlagForTask(int64_t taskIdx) const
    {
        uint16_t flag = static_cast<uint16_t>(taskIdx);
        if ((taskIdx & 1) != 0) {
            flag += kAiv1IntraFlagOff;
        }
        return flag;
    }

    // Same ids as Stage1 (0,1,2,3). AIC already Waited this id for k' ready,
    // then Sets it again after kkt Fixpipe so that AIV can Stage3.
    // AIV waits the raw id; AIC Sets id+16 for AIV1.
    __aicore__ inline void NotifyAivKktDone(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlagForTask(taskIdx));
    }

    __aicore__ inline void WaitCubeKktDone(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4, PIPE_V>(static_cast<uint16_t>(taskIdx));
    }

    // Stage3 L1 ready → Stage4. AIV Sets raw taskIdx+4; AIC Waits +16 on AIV1.
    __aicore__ inline void NotifyAicStage3Done(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS3DoneBase));
    }

    __aicore__ inline void WaitAivStage3Done(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(taskIdx + kFlagS3DoneBase));
    }

    // Stage4 finished LoadData of NegL. AIV may overwrite those slots with
    // the next pack's k'. Y lives at [0, 64) so Stage5 can overlap Stage1.
    __aicore__ inline void NotifyAivStage4Done(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_FIX>(CubeWaitFlagForTask(taskIdx + kFlagS4DumpBase));
    }

    __aicore__ inline void WaitAicStage4Done(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4>(static_cast<uint16_t>(taskIdx + kFlagS4DumpBase));
    }

    // Stage6 L1 vb/kbg ready → Stage7. Reuse Stage3 ids 4..7 (consumed
    // after Stage4). AIV Sets raw taskIdx+4; AIC Waits +16 on AIV1.
    // Intra-block 12..15 is ignored on this SoC (Wait returns immediately).
    __aicore__ inline void NotifyAicStage6Done(int64_t taskIdx)
    {
        CrossCoreSetFlag<0x4, PIPE_MTE3>(static_cast<uint16_t>(taskIdx + kFlagS6DoneBase));
    }

    __aicore__ inline void WaitAivStage6Done(int64_t taskIdx)
    {
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(taskIdx + kFlagS6DoneBase));
    }

    // Pack-boundary gate: packN+1.Stage1 (AIV) must not start until
    // packN.Stage7 (AIC) finishes. Overlap of those two causes L0A ECC 507015.
    __aicore__ inline void NotifyAivStage7Done()
    {
        CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagS7Done);
    }

    __aicore__ inline void WaitAicStage7Done()
    {
        CrossCoreWaitFlag<0x2>(kFlagS7Done);
    }

    // ========================= Stage 3 =========================
    // Packed -L00 | -L11 from ubLFull (VF already wrote -L). VCS consumes this as-is.
    __aicore__ inline void PackDiagLeavesFromUb(LocalTensor<float> packed, LocalTensor<float> L)
    {
        const uint16_t burst = 4;
        const uint16_t gap = 4;
        DataCopy(packed, L, DataCopyParams(32, burst, gap, gap));
        DataCopy(packed[32], L[32 * 64 + 32], DataCopyParams(32, burst, gap, gap));
    }

    // Packed VCS ND -> L1 NZ quadrants (Right=TL L00, Left=BR L11), and
    // -L ND64 -> L1 NZ. 8-col DataCopy, no UB TransDataTo5HD.
    // Off-diagonal leaves stay Stage0 zero.
    __aicore__ inline void UploadDiagLeavesAndFullAToL1(LocalTensor<float> packedNd32x64,
                                                LocalTensor<float> negLNd64,
                                                int64_t taskIdx)
    {
        UbPackedLeafToL1(l1LeafRight[taskIdx], packedNd32x64, 0, 0, 0);
        UbPackedLeafToL1(l1LeafLeft[taskIdx], packedNd32x64, 1, 1,
                         static_cast<int32_t>(kVcs32));
        UbNd64ToL1Nz8(l1NegL[taskIdx], negLNd64);
    }

    __aicore__ inline void Stage3_ConstructLAndVcs(int32_t db, LocalTensor<float> kkt)
    {
        LocalTensor<float> ubL = ubLFull[db];
        LocalTensor<float> g = ubGPrime[db];
        LocalTensor<float> beta = ubBetaEff[db];
        ConstructLowerLExp2VF(kkt, g, beta, ubMaskFp32, ubL, static_cast<uint32_t>(chunkSize));
        PackDiagLeavesFromUb(ubLPacked[db], ubL);
        DataCopy(ubResVcs[db], ubIVcs, static_cast<int32_t>(kVcsPackedElems32));
        MulReduceScatterVF32(ubResVcs[db], ubLPacked[db], ubResVcs[db], ubVcsIdx);
    }

    __aicore__ inline void Stage3_AivOne(int64_t taskIdx)
    {
        WaitCubeKktDone(taskIdx);
        const int32_t db = PingPongSlot(taskIdx);
        Stage3_ConstructLAndVcs(db, ubKkt[db]);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadDiagLeavesAndFullAToL1(ubResVcs[db], ubLFull[db], taskIdx);
        NotifyAicStage3Done(taskIdx);
    }


    // ========================= Stage 2 =========================
    // kkt = k' @ k'^T. AIV0 tasks 0/2 use l0 (event 0); AIV1 tasks 1/3 use l01 (event 1).
    // Wait FIX_M(bank) before reuse of that L0 (0 vs 2, 1 vs 3, and across packs).
    // First Wait on each bank is primed by SetFlag FIX_M in ProcessAic, so
    // task 0 does not wait task 1 and vice versa. After Fixpipe only Set
    // FIX_M, no Wait: Notify is
    // PIPE_FIX and the other bank can Matmul while this Fixpipe runs.
    // ubKkt ping/pong is PingPongSlot. subBlockId: 0=AIV0, 1=AIV1.
    __aicore__ inline void Stage2_AicOne(int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const int32_t kk = static_cast<int32_t>(K);
        const uint32_t n = static_cast<uint32_t>(chunkSize);
        const uint8_t subBlk = static_cast<uint8_t>(taskIdx & 1);
        const int32_t db = PingPongSlot(taskIdx);
        if ((taskIdx & 1) == 0) {
            WaitFlag<HardEvent::M_MTE1>(0);
            MatmulToL0C<InDtype>(l1KHat[taskIdx], l1KHat[taskIdx], l0A, l0B, l0C,
                                 bt, bt, kk, true, false, false, 0);
            SetFlag<HardEvent::M_FIX>(0);
            SetFlag<HardEvent::M_MTE1>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeL0cToUbFp32Nd(ubKkt[db], l0C, n, subBlk);
            SetFlag<HardEvent::FIX_M>(0);
        } else {
            WaitFlag<HardEvent::M_MTE1>(1);
            MatmulToL0C<InDtype>(l1KHat[taskIdx], l1KHat[taskIdx], l0A1, l0B1, l0C1,
                                 bt, bt, kk, true, false, false, 1);
            SetFlag<HardEvent::M_FIX>(1);
            SetFlag<HardEvent::M_MTE1>(1);
            WaitFlag<HardEvent::M_FIX>(1);
            FixpipeL0cToUbFp32Nd(ubKkt[db], l0C1, n, subBlk);
            SetFlag<HardEvent::FIX_M>(1);
        }
        NotifyAivKktDone(taskIdx);
    }

    // ========================= Stage 4 =========================
    // Y = I + LeafLeft @ (-L)  (driving = LeafLeft / X_L).
    //   B: L0C = I @ I                init_flag=True
    //   C: L0C = LeafLeft @ (-L) + I  init_flag=False
    // Fixpipe Arch3510 isChannelSplit (16x16 -> 16x8) into this tile's u
    // GM (16 KiB), then MTE2 -> L1 Y. Host audit reads that NZ on u.
    // No MTE2_MTE1 after Copy: Stage5 LoadData Y is after PIPE_ALL.
    // Stage7 overwrites u.
    // L0A/B: even ping [0,16) pong [32,48); odd ping [16,32) pong [48,64).
    // Second MMAD uses the pong pair (no M_MTE1). Same L0C, initC=false.
    __aicore__ inline void Stage4_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int32_t bt = static_cast<int32_t>(chunkSize);
        const uint8_t bank = static_cast<uint8_t>(taskIdx & 1);
        if (bank == 0) {
            WaitFlag<HardEvent::M_MTE1>(0);
            MatmulToL0C<float>(l1I, l1I, l0Af, l0Bf, l0C, bt, bt, bt, true, true, false, 0);
            SetFlag<HardEvent::FIX_M>(0);
            MatmulToL0C<float>(l1LeafLeft[taskIdx], l1NegL[taskIdx], l0AfP, l0BfP, l0C,
                               bt, bt, bt, false, true, true, 0);
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
            MatmulToL0C<float>(l1LeafLeft[taskIdx], l1NegL[taskIdx], l0AfP1, l0BfP1, l0C1,
                               bt, bt, bt, false, true, true, 1);
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
    // A = LeafLeft + Y @ LeafRight  (other = LeafRight / X_R).
    //   tmp = I @ LeafLeft               init_flag=True
    //   A   = Y @ LeafRight + tmp        init_flag=False
    // Aligned (M==BT): one Fixpipe to gmA, then Nd2Nz gmA -> L1 A.
    // Tail (M<BT): gmA only has M valid rows; second Fixpipe to gmWsA
    // (full 64x64, pad in L0C) then Nd2Nz, so L1 A stays 64x64 for Stage7.
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
            MatmulToL0C<float>(l1Y[taskIdx], l1LeafRight[taskIdx], l0AfP, l0BfP, l0C,
                               bt, bt, bt, false, false, false, 0);
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
            MatmulToL0C<float>(l1Y[taskIdx], l1LeafRight[taskIdx], l0AfP1, l0BfP1, l0C1,
                               bt, bt, bt, false, false, false, 1);
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
    // vb = v * β, kbg = k' * β * exp2(g'). Tail tiles: zero-fill UB then
    // copy M rows; aligned tiles skip the Duplicate. NZ upload is always 64
    // rows (pad is 0). AIV0 tasks 0,2; AIV1 1,3.
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
        ScaleRowsAlignedVF<InDtype>(ubVnd, ubVnd, beta, static_cast<uint32_t>(chunkSize),
                                    static_cast<uint32_t>(V));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        UploadBf16NdToL1(l1Vb[taskIdx], ubVnd, static_cast<uint32_t>(V));

        DataCopy(ubKnd, gmKHat[offK], nK);
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
        ScaleRowsK128BetaExp2gVF<InDtype>(ubKnd, ubKnd, beta, g, static_cast<uint32_t>(chunkSize));
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        UploadBf16NdToL1(l1Kbg[taskIdx], ubKnd, static_cast<uint32_t>(K));
    }

    // ========================= Stage 7 =========================
    // Per task like Stage5: W ping then U pong, then both Fixpipes.
    // Stage5's two MMADs are fp32 onto the same L0C (accumulate), so they
    // need no PIPE_ALL between them. Stage7 W/U are independent bf16 MMADs
    // on both L0A halves that Stage4/5 just used as fp32; Cube back-to-back
    // on ping then pong is 507015. PIPE_ALL between the two MMADs.
    // V=256 U is two n=128 MMADs (same 16 KiB L0B path as V=128).
    __aicore__ inline void Stage7_AicOne(const ChunkRange &chunk, int64_t hv, int64_t taskIdx)
    {
        const int64_t offW = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, K);
        const int64_t offU = OffsetBHTD(chunk.batch, hv, chunk.tokenStart, HV, T, V);
        const uint32_t rows = static_cast<uint32_t>(chunk.M);
        const uint32_t nK = static_cast<uint32_t>(K);
        const uint32_t nV = static_cast<uint32_t>(V);
        const int32_t bt = static_cast<int32_t>(chunkSize);
        WaitFlag<HardEvent::M_MTE1>(0);
        WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Kbg[taskIdx], l0AS7, l0BS7, l0CS7,
                               bt, static_cast<int32_t>(nK), bt, 0);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        SetFlag<HardEvent::M_MTE1>(0);
        FixpipeL0cToGmNd<InDtype>(gmW[offW], l0CS7, rows, nK, nK);
        SetFlag<HardEvent::FIX_M>(0);
        
        WaitFlag<HardEvent::M_MTE1>(1);
        WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb[taskIdx], l0AS71, l0BS71, l0CS71,
                               bt, static_cast<int32_t>(kGdnHeadDimK), bt, 1);
        SetFlag<HardEvent::M_FIX>(1);
        SetFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::M_FIX>(1);
        FixpipeL0cToGmNd<InDtype>(gmU[offU], l0CS71, rows, kGdnHeadDimK, nV);
        SetFlag<HardEvent::FIX_M>(1);
        
        if (nV > kGdnHeadDimK) {
            SetFlag<HardEvent::FIX_M>(0);
            WaitFlag<HardEvent::FIX_M>(0);
            PipeBarrier<PIPE_ALL>();
            WuMatmulToL0C<InDtype>(l1A[taskIdx], l1Vb1[taskIdx], l0AS71, l0BS71, l0CS71,
                                   bt, static_cast<int32_t>(kGdnHeadDimK), bt, 2);
            SetFlag<HardEvent::M_FIX>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeL0cToGmNd<InDtype>(gmU[offU + kGdnHeadDimK], l0CS71, rows,
                                      kGdnHeadDimK, nV);
        }

    }

    __aicore__ inline void ProcessAiv()
    {
        Stage0_GenerateResidentAux();
        
        const int64_t nPacks = CeilDiv(totalChunks, kTasksPerRound);
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const int64_t base = pack * kTasksPerRound;
            const int64_t nThis = (totalChunks - base) < kTasksPerRound ? (totalChunks - base) : kTasksPerRound;
            
            for (int64_t t = subBlock; t < nThis; t += 2) {
                const int64_t workId = base + t;
                Stage1_OneTask(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                               workId % HV, t);
            }
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            for (int64_t t = subBlock; t < nThis; t += 2) {
                Stage3_AivOne(t);
            }

            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);

            for (int64_t t = subBlock; t < nThis; t += 2) {
                const int64_t workId = base + t;
                Stage6_AivOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                              workId % HV, t);
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
        // Prime Stage4 flags so pack 0 Stage1 WaitAicStage4Done returns.
        // Mode 4 is AIC→AIV: AIV cannot Set these ids for itself.
        for (int64_t t = 0; t < kTasksPerRound; ++t) {
            NotifyAivStage4Done(t);
        }
        const int64_t nPacks = CeilDiv(totalChunks, kTasksPerRound);
        for (int64_t pack = coreIdx; pack < nPacks; pack += numCore) {
            const int64_t base = pack * kTasksPerRound;
            const int64_t nThis = (totalChunks - base) < kTasksPerRound ? (totalChunks - base) : kTasksPerRound;
            WaitFlag<HardEvent::FIX_MTE1>(0);
            for (int64_t t = 0; t < nThis; ++t) {
                CrossCoreWaitFlag<0x4, PIPE_MTE1>(CubeWaitFlagForTask(t));
                Stage2_AicOne(t);
            }
            for (int64_t t = 0; t < nThis; ++t) {
                WaitAivStage3Done(t);
                const int64_t workId = base + t;
                Stage4_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                              workId % HV, t);
                SetFlag<HardEvent::MTE2_MTE1>(t);
            }
            for (int64_t t = 0; t < nThis; ++t) {
                const int64_t workId = base + t;
                WaitFlag<HardEvent::MTE2_MTE1>(t);
                Stage5_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                              workId % HV, t);
                SetFlag<HardEvent::MTE2_MTE1>(t);
            }
            for (int64_t t = 0; t < nThis; ++t) {
                const int64_t workId = base + t;
                WaitAivStage6Done(t);
                WaitFlag<HardEvent::MTE2_MTE1>(t);

                Stage7_AicOne(GetChunkRange(*this, gmCu, gmIdx, workId / HV),
                              workId % HV, t);
                
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
    LocalTensor<uint8_t> ubMaskBits;
    LocalTensor<uint32_t> ubVcsIdx;
    LocalTensor<float> ubIVcs, ubS0Zero, ubMaskFp32;

    // S1
    LocalTensor<float> ubGPrime[2],ubBetaEff[2];
    LocalTensor<InDtype> ubQ[2], ubK[2], ubQHat[2], ubKHat[2];
    LocalTensor<float> ubG[2], ubBeta[2], ubKRstd[2], ubQRstd[2];

    // S2
    LocalTensor<float> ubKkt[2];

    
    LocalTensor<float> ubLPacked[2], ubResVcs[2], ubLFull[2];

    LocalTensor<float> ubGateTmp;
    LocalTensor<InDtype> ubS6K[2], ubS6V[2];

    // L1
    LocalTensor<float> l1I;
    LocalTensor<InDtype> l1KHat[4], l1A[4], l1Kbg[4], l1Kbg1[4], l1Vb[4], l1Vb1[4];
    LocalTensor<float> l1NegL[4], l1LeafRight[4], l1LeafLeft[4], l1Y[4];

    // L0
    
    LocalTensor<InDtype> l0A, l0B, l0A1, l0B1, l0AS7, l0BS7, l0AS71, l0BS71, l0BS7Wide;
    LocalTensor<float> l0Af, l0Bf, l0Af1, l0Bf1, l0AfP, l0BfP, l0AfP1, l0BfP1;
    LocalTensor<float> l0C, l0C1, l0CS7, l0CS71;

    int64_t hasBetaOut;
};

} // namespace GdnStage

#endif
