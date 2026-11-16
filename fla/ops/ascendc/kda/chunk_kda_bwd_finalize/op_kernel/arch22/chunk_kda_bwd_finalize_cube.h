/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_CUBE_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_CUBE_H

#include "chunk_kda_bwd_finalize_common.h"
#include "kernel_operator.h"

namespace KDA::Arch22 {

constexpr AscendC::FixpipeConfig FIX_NZ = {AscendC::CO2Layout::NZ, false};

class ChunkKdaBwdFinalizeCube {
public:
    __aicore__ inline void Init(GM_ADDR v, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh, GM_ADDR dvScan,
                                GM_ADDR workspace, AscendC::TPipe *pipe)
    {
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        akk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(akk));
        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(vNew));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        dvScan_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dvScan));
        workspace_ = workspace;
        pipe_ = pipe;
        pipe_->InitBuffer(l1Buf_, L1_BYTES);
        pipe_->InitBuffer(l0ABuf_, 2 * L0_AB_SLOT_BYTES);
        pipe_->InitBuffer(l0BBuf_, 2 * L0_AB_SLOT_BYTES);
        pipe_->InitBuffer(l0CBuf_, 2 * L0_C_SLOT_BYTES);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            akkFree_[slot + 2] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[slot + 2]);
            akkFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[slot]);
            keReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
            keFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[slot]);
            inputReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE1>();
            l1Free_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Free_[slot]);
            operandReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_M>();
            operandFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::M_MTE1>();
            resultReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::M_FIX>();
            resultFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::FIX_M>();
            dwReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE1>();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(operandFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[slot]);
        }
    }

    // Only Stage 0 consumes these inputs; akk/dW have separate lifetimes.
    __aicore__ inline void PreloadState(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                        int64_t head, uint32_t owner)
    {
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t tokenOffset = ((batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart) * HEAD_DIM;
        const int64_t stateOffset = (chunkIndex * tiling.heads + head) * HEAD_DIM * HEAD_DIM;
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Free_[owner]);
        CopyToL1(l1[L1_H].ReinterpretCast<bfloat16_t>(), h_[stateOffset], HEAD_DIM, HEAD_DIM, HEAD_DIM, HEAD_DIM);
        CopyToL1(l1[L1_DH].ReinterpretCast<bfloat16_t>(), dh_[stateOffset], HEAD_DIM, HEAD_DIM, HEAD_DIM, HEAD_DIM);
        CopyToL1(l1[L1_V_NEW].ReinterpretCast<bfloat16_t>(), vNew_[tokenOffset], CHUNK_SIZE, HEAD_DIM, rows, HEAD_DIM);
        CopyToL1(l1[L1_DV_SCAN].ReinterpretCast<bfloat16_t>(), dvScan_[tokenOffset], CHUNK_SIZE, HEAD_DIM, rows,
                 HEAD_DIM);
        CopyToL1(l1[L1_V].ReinterpretCast<bfloat16_t>(), v_[tiling.vTokenMajor ? ((batch * tiling.tokens + chunk.tokenStart) * tiling.heads + head) * HEAD_DIM : tokenOffset],
                 CHUNK_SIZE, HEAD_DIM, rows, HEAD_DIM, tiling.vTokenMajor ? tiling.heads * HEAD_DIM : HEAD_DIM);
    }

    __aicore__ inline void LoadStage0Inputs(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                            int64_t head, uint32_t owner, bool loadState = true, uint32_t akkBank = 0)
    {
        if (loadState) {
            PreloadState(tiling, chunk, chunkIndex, head, owner);
        }
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t akkOffset = ((batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart) * CHUNK_SIZE;
        const uint32_t rows = static_cast<uint32_t>(chunk.validRows);
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[owner + 2 * akkBank]);
        CopyToL1(l1[L1_AKK + akkBank * L1_AKK_BANK_STRIDE].ReinterpretCast<bfloat16_t>(), akk_[akkOffset], CHUNK_SIZE,
                 CHUNK_SIZE, rows, rows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(inputReady_[owner]);
    }

    // Four GEMMs, two independent operand/result slots. Wait only at consumption.
    template <bool CAPTURE = false, bool CUBE_RH = false>
    __aicore__ inline void Stage0(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture = nullptr,
                                  uint32_t bank = 0, uint32_t akkBank = 0)
    {
        const uint64_t slotOffset =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        auto akkL1 = l1[L1_AKK + akkBank * L1_AKK_BANK_STRIDE].ReinterpretCast<bfloat16_t>();
        auto dwL1 = l1[L1_DW].ReinterpretCast<bfloat16_t>();
        auto hL1 = l1[L1_H].ReinterpretCast<bfloat16_t>();
        auto dhL1 = l1[L1_DH].ReinterpretCast<bfloat16_t>();
        auto vNewL1 = l1[L1_V_NEW].ReinterpretCast<bfloat16_t>();
        auto dvScanL1 = l1[L1_DV_SCAN].ReinterpretCast<bfloat16_t>();
        auto vL1 = l1[L1_V].ReinterpretCast<bfloat16_t>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(inputReady_[owner]);

        AscendC::GlobalTensor<float> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + slotOffset));
        AscendC::GlobalTensor<float> snapshot;
        AscendC::GlobalTensor<bfloat16_t> snapshotBf16;
        if constexpr (CAPTURE) {
            const uint64_t offset = static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES;
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(capture + offset));
            snapshotBf16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(capture + offset));
        }
        auto vectorFix = AscendC::FixpipeParamsV220(HEAD_DIM, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false);
        vectorFix.quantPre = QuantMode_t::NoQuant;

        // dk_state_raw = v_new @ dh^T; dv_b = akk^T @ dv_scan.
        LoadOperands<false, true>(vNewL1, dhL1, CHUNK_SIZE, HEAD_DIM, HEAD_DIM, 0);
        LoadOperands<true, false>(akkL1, dvScanL1, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, 1);
        auto dkState = Multiply(CHUNK_SIZE, HEAD_DIM, HEAD_DIM, 0);
        auto dvB = Multiply(CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, 1);
        // Preload dW in the released operand slot while dv_b is computing.
        LoadOperands<false, true>(dvScanL1, hL1, CHUNK_SIZE, HEAD_DIM, HEAD_DIM, 0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[0]);
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[WS_DK_STATE_RAW / sizeof(float)], dkState,
                                                               vectorFix);
        if constexpr (CAPTURE) {
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(snapshot[CAPTURE_DK_STATE_RAW / sizeof(float)],
                                                                   dkState, vectorFix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[0]);

        // dW = dv_scan @ h^T. Its L0C slot becomes free after dk_state_raw.
        auto dw = Multiply(CHUNK_SIZE, HEAD_DIM, HEAD_DIM, 0);
        LoadOperands<false, true>(dvScanL1, vL1, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, 1);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[1]);
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[WS_DV_B / sizeof(float)], dvB, vectorFix);
        if constexpr (CAPTURE) {
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(snapshot[CAPTURE_DV_B / sizeof(float)], dvB,
                                                                   vectorFix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[1]);

        // zV = dv_scan @ v^T; its MMAD overlaps the BF16 dW export below.
        auto zv = Multiply(CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, 1);
        if constexpr (!CUBE_RH) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Free_[owner]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[0]);
        // The only production dW destination is BF16 NZ L1, retained for Stage 2.
        auto dwFix = AscendC::FixpipeParamsV220(HEAD_DIM, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
        dwFix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, FIX_NZ>(dwL1, dw, dwFix);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(dwReady_[owner]);
        if constexpr (CAPTURE) {
            auto dumpFix = vectorFix;
            dumpFix.quantPre = QuantMode_t::F322BF16;
            AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(snapshotBf16[CAPTURE_DW / sizeof(bfloat16_t)],
                                                                        dw, dumpFix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[0]);

        // zV[64,64] = dv_scan[64,128] @ v[64,128]^T.
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[1]);
        auto matrixFix = AscendC::FixpipeParamsV220(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
        matrixFix.quantPre = QuantMode_t::NoQuant;
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[WS_ZV / sizeof(float)], zv, matrixFix);
        if constexpr (CAPTURE) {
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(snapshot[CAPTURE_ZV / sizeof(float)], zv, matrixFix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[1]);
        if constexpr (CUBE_RH) {
            // Four diagonal 32x32 blocks trade extra arithmetic for fewer small-GEMM launches.
            auto rhFix = AscendC::FixpipeParamsV220(32, 32, 32, 32, false);
            rhFix.quantPre = QuantMode_t::NoQuant;
            for (uint32_t block = 0; block < HEAD_DIM / 32; ++block) {
                const uint32_t slot = block & 1;
                LoadOperands<false, true>(hL1[block * 2 * FRACTAL_ELEMENTS], dhL1[block * 2 * FRACTAL_ELEMENTS], 32, 32,
                                          HEAD_DIM, slot, HEAD_DIM, HEAD_DIM);
                auto dots = Multiply(32, 32, HEAD_DIM, slot);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    output[WS_DQ_LOCAL / sizeof(float) + block * 4 * FRACTAL_ELEMENTS], dots, rhFix);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[slot]);
            }
            // Next-work h/dh preload may start only after the final operand read.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Free_[owner]);
        }
    }

    __aicore__ inline void LoadKe(uint32_t owner, uint32_t core, uint32_t bank = 0)
    {
        AscendC::GlobalTensor<bfloat16_t> ke;
        ke.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_ +
                                                                 static_cast<uint64_t>(core) * WS_CORE_BYTES +
                                                                 owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES + WS_KE));
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
        CopyToL1(l1[L1_KE].ReinterpretCast<bfloat16_t>(), ke, CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, HEAD_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
    }

    template <bool CAPTURE = false>
    __aicore__ inline void Stage2(const FinalizeTilingData &tiling, int64_t chunkIndex, int64_t head, uint32_t owner,
                                  uint32_t core, GM_ADDR capture = nullptr, uint32_t bank = 0, bool retainAkk = false)
    {
        const uint32_t akkBank = retainAkk ? bank : 0;
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        auto dw = l1[L1_DW].ReinterpretCast<bfloat16_t>();
        AscendC::GlobalTensor<float> output, snapshot;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            workspace_ + static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES));
        if constexpr (CAPTURE) {
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        // dKgb_raw = akk^T @ dW; it does not depend on the kE transfer.
        WaitDw(owner);
        LoadOperands<true, false>(l1[L1_AKK + akkBank * L1_AKK_BANK_STRIDE].ReinterpretCast<bfloat16_t>(), dw,
                                  CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, 0);
        if (!retainAkk) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[owner]);
        }
        auto dkgb = Multiply(CHUNK_SIZE, HEAD_DIM, CHUNK_SIZE, 0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
        LoadOperands<false, true>(dw, l1[L1_KE].ReinterpretCast<bfloat16_t>(), CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
        auto zw = Multiply(CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, 1);
        auto vectorFix = AscendC::FixpipeParamsV220(HEAD_DIM, CHUNK_SIZE, CHUNK_SIZE, HEAD_DIM, false);
        vectorFix.quantPre = QuantMode_t::NoQuant;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[0]);
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[WS_DKGB_RAW / sizeof(float)], dkgb, vectorFix);
        if constexpr (CAPTURE) {
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(snapshot[CAPTURE_DKGB_RAW / sizeof(float)], dkgb,
                                                                   vectorFix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[0]);
        auto matrixFix = AscendC::FixpipeParamsV220(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
        matrixFix.quantPre = QuantMode_t::NoQuant;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[1]);
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[WS_ZW / sizeof(float)], zw, matrixFix);
        if constexpr (CAPTURE) {
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(snapshot[CAPTURE_ZW / sizeof(float)], zw, matrixFix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[1]);
    }

    // Stage 2 has released kE's transfer events; reuse them for the Zb input.
    __aicore__ inline void LoadZb(uint32_t owner, uint32_t core, uint32_t bank)
    {
        AscendC::GlobalTensor<bfloat16_t> zb;
        zb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_ +
                                                                 static_cast<uint64_t>(core) * WS_CORE_BYTES +
                                                                 owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES + WS_ZB));
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
        CopyToL1(l1[L1_ZB].ReinterpretCast<bfloat16_t>(), zb, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
    }

    template <bool CAPTURE = false, bool RETAIN_AKK = false>
    __aicore__ inline void Stage4(const FinalizeTilingData &tiling, int64_t chunkIndex, int64_t head, uint32_t owner,
                                  uint32_t bank, GM_ADDR capture = nullptr)
    {
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
        // Tza = Zb @ akk^T. Two work generations retain separate akk and Tza.
        LoadOperands<false, true>(l1[L1_ZB].ReinterpretCast<bfloat16_t>(),
                                  l1[L1_AKK + bank * L1_AKK_BANK_STRIDE].ReinterpretCast<bfloat16_t>(), CHUNK_SIZE,
                                  CHUNK_SIZE, CHUNK_SIZE, owner);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
        auto tza = Multiply(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, owner);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[owner]);
        auto fix = AscendC::FixpipeParamsV220(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
        fix.quantPre = QuantMode_t::F322BF16;
        AscendC::Fixpipe<bfloat16_t, float, FIX_NZ>(
            l1[L1_TZA + bank * L1_TZA_BANK_STRIDE].ReinterpretCast<bfloat16_t>(), tza, fix);
        if constexpr (CAPTURE) {
            AscendC::GlobalTensor<bfloat16_t> snapshot;
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES + CAPTURE_TZA));
            AscendC::Fixpipe<bfloat16_t, float, AscendC::CFG_ROW_MAJOR>(snapshot, tza, fix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[owner]);
        if constexpr (CAPTURE) {
            // Test-only consumer proves the actual NZ L1 write, not just its GM
            // mirror. This extra GEMM and export are absent from performance builds.
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(dwReady_[owner]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(dwReady_[owner]);
            LoadOperands<false, true>(l1[L1_TZA + bank * L1_TZA_BANK_STRIDE].ReinterpretCast<bfloat16_t>(),
                                      l1[L1_AKK + bank * L1_AKK_BANK_STRIDE].ReinterpretCast<bfloat16_t>(), CHUNK_SIZE,
                                      CHUNK_SIZE, CHUNK_SIZE, owner);
            auto check = Multiply(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, owner);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[owner]);
            AscendC::GlobalTensor<float> snapshot;
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES +
                CAPTURE_TZA_CHECK));
            fix.quantPre = QuantMode_t::NoQuant;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(snapshot, check, fix);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[owner]);
        }
        if constexpr (RETAIN_AKK) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(dwReady_[owner]);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[owner + 2 * bank]);
        }
    }

    template <bool CAPTURE = false>
    __aicore__ inline void Stage6(const FinalizeTilingData &tiling, int64_t chunkIndex, int64_t head, uint32_t owner,
                                  uint32_t core, uint32_t bank, GM_ADDR capture = nullptr)
    {
        auto l1 = l1Buf_.Get<uint8_t>()[owner * L1_HEAD_BYTES];
        // dAkk_raw = akk^T @ Tza. Both BF16 operands remain in their NZ L1 slots.
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(dwReady_[owner]);
        LoadOperands<true, false>(l1[L1_AKK + bank * L1_AKK_BANK_STRIDE].ReinterpretCast<bfloat16_t>(),
                                  l1[L1_TZA + bank * L1_TZA_BANK_STRIDE].ReinterpretCast<bfloat16_t>(), CHUNK_SIZE,
                                  CHUNK_SIZE, CHUNK_SIZE, owner);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[owner + 2 * bank]);
        auto dAkk = Multiply(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, owner);
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DAKK_RAW));
        auto fix = AscendC::FixpipeParamsV220(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, false);
        fix.quantPre = QuantMode_t::NoQuant;
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[owner]);
        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output, dAkk, fix);
        if constexpr (CAPTURE) {
            AscendC::GlobalTensor<float> snapshot;
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES + CAPTURE_DAKK_RAW));
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(snapshot, dAkk, fix);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[owner]);
    }

    template <bool CAPTURE = false, bool FINALIZE_Q = false, bool COMPUTE_RIGHT = false, bool FINALIZE_K = false>
    __aicore__ inline void Stage8(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t firstHead, uint32_t core, uint32_t bank, GM_ADDR capture)
    {
        for (int32_t start = ((chunk.validRows - 1) / 32) * 32; start >= 0; start -= 32) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(BAND_READY);
            const bool compact = FINALIZE_K && !CAPTURE && chunk.validRows == CHUNK_SIZE && start == 0;
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner >= tiling.heads) {
                    continue;
                }
                auto l1 = l1Buf_.Get<uint8_t>()[L1_STAGE8 + owner * L1_STAGE8_HEAD_BYTES];
                AscendC::GlobalTensor<float> input;
                input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ +
                                                                       static_cast<uint64_t>(core) * WS_CORE_BYTES +
                                                                       owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
                // Both lower matrices stay in L1 through Stage 8/10 and both bands.
                if (start == ((chunk.validRows - 1) / 32) * 32) {
                    CopyToL1(l1.ReinterpretCast<float>(), input[WS_DAQK_FP32 / 4], 64, 64, 64, 64);
                    CopyToL1(l1[16384].ReinterpretCast<float>(), input[WS_DAKK_FP32 / 4], 64, 64, 64, 64);
                }
                const uint32_t kRows = compact ? 32 : 64;
                CopyToL1(l1[32768].ReinterpretCast<float>(), input[WS_KNEG_FP32 / 4], kRows, 128, kRows, 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
            }
            if constexpr (FINALIZE_Q) {
                // Stage 9 also waits for S8_READY, protecting the FP32 GM inputs.
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(BAND_LOADED);
            }
            if constexpr (FINALIZE_K && !CAPTURE) {
                // Publish both dq results before computing the two left results.
                // Each phase uses one L0 slot per head and keeps the two MMADs in flight.
                for (uint32_t matrix = 0; matrix < 2; ++matrix) {
                    for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                        if (firstHead + owner >= tiling.heads) {
                            continue;
                        }
                        auto l1 = l1Buf_.Get<uint8_t>()[L1_STAGE8 + owner * L1_STAGE8_HEAD_BYTES];
                        if (matrix == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
                        }
                        AscendC::SetHF32Mode(false);
                        auto a = l1[matrix * 16384].ReinterpretCast<float>();
                        auto b = l1[32768].ReinterpretCast<float>();
                        if (compact) {
                            LoadBandFp32<false, false, 32>(a[start * 8], b, owner);
                        } else {
                            LoadBandFp32<false>(a[start * 8], b, owner);
                        }
                        Multiply<float>(32, 128, compact ? 32 : 64, owner);
                        if (matrix == 1) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
                        }
                    }
                    auto fix = AscendC::FixpipeParamsV220(128, 32, 32, 128, false);
                    fix.quantPre = QuantMode_t::NoQuant;
                    for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                        if (firstHead + owner >= tiling.heads) {
                            continue;
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[owner]);
                        auto result = l0CBuf_.Get<float>()[owner * L0_C_SLOT_BYTES / 4];
                        AscendC::GlobalTensor<float> output;
                        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                            workspace_ + static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES +
                            bank * WS_BANK_BYTES + WS_DQ_LOCAL + matrix * 32768));
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[start * 128], result, fix);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[owner]);
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(matrix == 0 ? S8_READY : S8_SECOND_READY);
                }
            } else {
                for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                    if (firstHead + owner >= tiling.heads) {
                        if constexpr (FINALIZE_Q) {
                            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(S8_SECOND_READY);
                        }
                        continue;
                    }
                    auto l1 = l1Buf_.Get<uint8_t>()[L1_STAGE8 + owner * L1_STAGE8_HEAD_BYTES];
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
                    auto b = l1[32768].ReinterpretCast<float>();
                    AscendC::SetHF32Mode(false);
                    LoadBandFp32<false>(l1[16384].ReinterpretCast<float>()[start * 8], b, 1);
                    Multiply<float>(32, 128, 64, 1);
                    LoadBandFp32<false>(l1.ReinterpretCast<float>()[start * 8], b, 0);
                    Multiply<float>(32, 128, 64, 0);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
                    auto fix = AscendC::FixpipeParamsV220(128, 32, 32, 128, false);
                    fix.quantPre = QuantMode_t::NoQuant;
                    for (uint32_t matrix = 0; matrix < 2; ++matrix) {
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[matrix]);
                        auto result = l0CBuf_.Get<float>()[matrix * L0_C_SLOT_BYTES / 4];
                        AscendC::GlobalTensor<float> output;
                        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                            workspace_ + static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES +
                            bank * WS_BANK_BYTES + WS_DQ_LOCAL + matrix * 32768));
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[start * 128], result, fix);
                        if constexpr (CAPTURE) {
                            output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                                capture +
                                static_cast<uint64_t>((firstHead + owner) * tiling.chunks + chunkIndex) *
                                    CAPTURE_BYTES +
                                CAPTURE_DQ_LOCAL + matrix * 32768));
                            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[start * 128], result, fix);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[matrix]);
                    }
                    if constexpr (FINALIZE_Q) {
                        // Publish each head after its own Fixpipe, not after the pair.
                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(owner == 0 ? S8_READY : S8_SECOND_READY);
                    }
                }
            }
            if constexpr (!FINALIZE_Q) {
                // The Stage-8-only harness has no S8_READY wait before reuse.
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(BAND_LOADED);
            }
            if constexpr (COMPUTE_RIGHT) {
                Stage10<CAPTURE>(tiling, chunkIndex, firstHead, core, bank, capture, start,
                                 FINALIZE_K && !CAPTURE && chunk.validRows == CHUNK_SIZE && start == 32);
            }
        }
    }

    template <bool CAPTURE = false>
    __aicore__ inline void Stage10(const FinalizeTilingData &tiling, int64_t chunkIndex, int64_t firstHead,
                                   uint32_t core, uint32_t bank, GM_ADDR capture, uint32_t start,
                                   bool compact = false)
    {
        // Second band handshake: qPos/bkPos are ready; Stage 9 runs independently.
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(BAND_READY);
        for (uint32_t matrix = 0; matrix < 2; ++matrix) {
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner >= tiling.heads) {
                    continue;
                }
                auto l1 = l1Buf_.Get<uint8_t>()[L1_STAGE8 + owner * L1_STAGE8_HEAD_BYTES];
                AscendC::GlobalTensor<float> input;
                input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ +
                                                                       static_cast<uint64_t>(core) * WS_CORE_BYTES +
                                                                       owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES));
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
                const uint32_t kRows = compact ? 32 : 64;
                const uint32_t sourceRow = compact ? 32 : 0;
                CopyToL1(l1[32768].ReinterpretCast<float>(),
                         input[(matrix == 0 ? WS_QPOS : WS_BKPOS) / 4 + sourceRow * HEAD_DIM],
                         kRows, 128, kRows, 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner >= tiling.heads) {
                    continue;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(keReady_[owner]);
                auto l1 = l1Buf_.Get<uint8_t>()[L1_STAGE8 + owner * L1_STAGE8_HEAD_BYTES];
                auto a = l1[matrix * 16384].ReinterpretCast<float>();
                auto b = l1[32768].ReinterpretCast<float>();
                if (compact) {
                    // lower keeps its 64-row NZ stride; B is compact 32-row NZ.
                    LoadBandFp32<true, false, 32>(a[start * 64 + 32 * 8], b, owner);
                } else {
                    LoadBandFp32<true>(a[start * 64], b, owner);
                }
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner >= tiling.heads) {
                    continue;
                }
                AscendC::SetHF32Mode(false);
                Multiply<float, false, true>(32, 128, compact ? 32 : 64, owner, matrix == 0, matrix == 1);
                if (matrix == 1) {
                    auto fix = AscendC::FixpipeParamsV220(128, 32, 32, 128, false);
                    fix.quantPre = QuantMode_t::NoQuant;
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(resultReady_[owner]);
                    auto result = l0CBuf_.Get<float>()[owner * L0_C_SLOT_BYTES / 4];
                    AscendC::GlobalTensor<float> output;
                    output.SetGlobalBuffer(
                        reinterpret_cast<__gm__ float *>(workspace_ + static_cast<uint64_t>(core) * WS_CORE_BYTES +
                                                         owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES + WS_RIGHT));
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[start * 128], result, fix);
                    if constexpr (CAPTURE) {
                        output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                            capture +
                            static_cast<uint64_t>((firstHead + owner) * tiling.chunks + chunkIndex) * CAPTURE_BYTES +
                            CAPTURE_RIGHT));
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(output[start * 128], result, fix);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(resultFree_[owner]);

                    if (owner == 0) {
                        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(S10_FIRST_READY);
                    }
                }
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner < tiling.heads) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[owner]);
                }
            }
        }
        // Protect right_raw from the next band's kNeg stores (same GM slot).
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(BAND_LOADED);
    }

    __aicore__ inline void ReleaseAkk(uint32_t owner)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[owner]);
    }

    __aicore__ inline void WaitDw(uint32_t owner)
    {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(dwReady_[owner]);
    }

    __aicore__ inline void End()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Free_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(keFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(keReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(keFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[slot + 2]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(akkFree_[slot + 2]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(akkFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(akkFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE1>(inputReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(l1Free_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(operandFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(resultFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::M_MTE1>(operandFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::FIX_M>(resultFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::FIX_MTE1>(dwReady_[slot]);
        }
    }

private:
    // Shared mechanics only: the Stage method owns the formula and result path.
    template <typename T>
    __aicore__ inline void CopyToL1(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint32_t rows,
                                    uint32_t columns, uint32_t validRows, uint32_t validColumns, uint64_t rowStride = 0)
    {
        if (validRows != rows || validColumns != columns) {
            AscendC::InitConstValue(
                dst, AscendC::InitConstValueParams<T>(1, static_cast<uint16_t>(rows * columns * sizeof(T) / 32), 0,
                                                      static_cast<T>(0)));
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::Nd2NzParams copy{};
        copy.ndNum = 1;
        copy.nValue = validRows;
        copy.dValue = validColumns;
        const uint64_t stride = rowStride ? rowStride : columns;
        copy.srcDValue = stride <= 65535 ? static_cast<uint16_t>(stride) : 0;
        copy.dstNzC0Stride = rows;
        copy.dstNzNStride = 1;
        if (stride <= 65535) {
            AscendC::DataCopy(dst, src, copy);
        } else {
            copy.nValue = 1;
            for (uint32_t row = 0; row < validRows; ++row)
                AscendC::DataCopy(dst[row * 16], src[row * stride], copy);
        }
    }

    template <bool TRANSPOSE_A, bool TRANSPOSE_B, bool RESIDUAL = false>
    __aicore__ inline void LoadOperands(AscendC::LocalTensor<bfloat16_t> a, AscendC::LocalTensor<bfloat16_t> b,
                                        uint32_t m, uint32_t n, uint32_t k, uint32_t slot, uint32_t aRows = 0,
                                        uint32_t bRows = 0)
    {
        auto l0A = l0ABuf_.Get<bfloat16_t>()[slot * L0_AB_SLOT_BYTES / sizeof(bfloat16_t)];
        auto l0B = l0BBuf_.Get<bfloat16_t>()[slot * L0_AB_SLOT_BYTES / sizeof(bfloat16_t)];
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(operandFree_[slot]);
        // Residual high/low planes fit in one slot: A 8 KiB, B 32 KiB.
        for (uint32_t plane = 0; plane < (RESIDUAL ? 2 : 1); ++plane) {
            auto aSrc = a[plane * CHUNK_SIZE * CHUNK_SIZE];
            auto bSrc = b[plane * CHUNK_SIZE * HEAD_DIM];
            auto aDst = l0A[plane * m * k];
            auto bDst = l0B[plane * k * n];
            AscendC::LoadData2DParams load{};
            load.dstGap = 0;
            if constexpr (TRANSPOSE_A) {
                // Transposed ND -> NZ already has the required outer M1/K1 order.
                load.repeatTimes = (m / 16) * (k / 16);
                load.srcStride = 1;
                load.ifTranspose = true;
                AscendC::LoadData(aDst, aSrc, load);
            } else {
                load.repeatTimes = k / 16;
                load.srcStride = (aRows == 0 ? m : aRows) / 16;
                load.ifTranspose = false;
                for (uint32_t row = 0; row < m / 16; ++row) {
                    AscendC::LoadData(aDst[row * (k / 16) * FRACTAL_ELEMENTS], aSrc[row * FRACTAL_ELEMENTS], load);
                }
            }
            if constexpr (TRANSPOSE_B) {
                // ND [n,k] -> L1 NZ directly matches the transposed L0B order.
                load.ifTranspose = false;
                if (bRows == 0) {
                    load.repeatTimes = (k / 16) * (n / 16);
                    load.srcStride = 1;
                    AscendC::LoadData(bDst, bSrc, load);
                } else {
                    // Diagonal slices retain their parent's source stride; interleave L0B columns.
                    load.repeatTimes = k / 16;
                    load.srcStride = bRows / 16;
                    load.dstGap = n / 16 - 1;
                    for (uint32_t column = 0; column < n / 16; ++column) {
                        AscendC::LoadData(bDst[column * FRACTAL_ELEMENTS], bSrc[column * FRACTAL_ELEMENTS], load);
                    }
                }
            } else {
                load.repeatTimes = n / 16;
                load.srcStride = k / 16;
                load.ifTranspose = true;
                for (uint32_t row = 0; row < k / 16; ++row) {
                    AscendC::LoadData(bDst[row * (n / 16) * FRACTAL_ELEMENTS], bSrc[row * FRACTAL_ELEMENTS], load);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
    }

    // FP32 NZ uses 16x8 fractals; c220 L0B needs the 3Dv2 conversion.
    template <bool TRANSPOSE_A, bool REUSE_B = false, uint32_t K_ROWS = 64>
    __aicore__ inline void LoadBandFp32(AscendC::LocalTensor<float> a, AscendC::LocalTensor<float> b, uint32_t slot)
    {
        auto l0A = l0ABuf_.Get<float>()[slot * L0_AB_SLOT_BYTES / 4];
        auto l0B = l0BBuf_.Get<float>()[slot * L0_AB_SLOT_BYTES / 4];
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(operandFree_[slot]);
        if constexpr (TRANSPOSE_A) {
            // Same c220 FP32 transpose as CATLASS CopyL1ToL0A<nZ>.
            AscendC::LoadData3DParamsV2<float> loadA{};
            loadA.l1H = 1;
            loadA.l1W = 64;
            loadA.channelSize = 64;
            loadA.kExtension = 32;
            loadA.mExtension = K_ROWS;
            loadA.strideW = 1;
            loadA.strideH = 1;
            loadA.filterW = 1;
            loadA.filterH = 1;
            loadA.dilationFilterW = 1;
            loadA.dilationFilterH = 1;
            loadA.enTranspose = true;
            AscendC::LoadData(l0A, a, loadA);
        } else {
            AscendC::LoadData2DParams loadA{};
            loadA.repeatTimes = K_ROWS / 8;
            loadA.srcStride = 4;
            for (uint32_t row = 0; row < 2; ++row) {
                AscendC::LoadData(l0A[row * (K_ROWS / 8) * 128], a[row * 128], loadA);
            }
        }
        AscendC::LoadData3DParamsV2<float> loadB{};
        loadB.l1H = 1;
        loadB.l1W = K_ROWS;
        loadB.channelSize = 128;
        loadB.kExtension = 128;
        loadB.mExtension = K_ROWS;
        loadB.strideW = 1;
        loadB.strideH = 1;
        loadB.filterW = 1;
        loadB.filterH = 1;
        loadB.dilationFilterW = 1;
        loadB.dilationFilterH = 1;
        loadB.enTranspose = true;
        AscendC::LoadData(l0B, b, loadB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
    }

    template <typename T = bfloat16_t, bool RESIDUAL = false, bool TRANSPOSE_A = false>
    __aicore__ inline AscendC::LocalTensor<float> Multiply(uint32_t m, uint32_t n, uint32_t k, uint32_t slot,
                                                           bool initialize = true, bool finish = true,
                                                           uint32_t term = 0)
    {
        auto l0A = l0ABuf_.Get<T>()[slot * L0_AB_SLOT_BYTES / sizeof(T) + (RESIDUAL && term == 2 ? m * k : 0)];
        auto l0B = l0BBuf_.Get<T>()[slot * L0_AB_SLOT_BYTES / sizeof(T) + (RESIDUAL && term == 1 ? k * n : 0)];
        auto l0C = l0CBuf_.Get<float>()[slot * L0_C_SLOT_BYTES / sizeof(float)];
        if (!RESIDUAL || term == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(operandReady_[slot]);
        }
        if (initialize) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(resultFree_[slot]);
        }
        AscendC::MmadParams params{};
        params.m = m;
        params.n = n;
        params.k = k;
        params.cmatrixInitVal = initialize;
        params.cmatrixSource = false;
        params.unitFlag = 0;
        if constexpr (TRANSPOSE_A) {
            params.kDirectionAlign = true;
        }
        AscendC::Mmad(l0C, l0A, l0B, params);
        if (!RESIDUAL || term == 2) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(operandFree_[slot]);
        }
        if (finish) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(resultReady_[slot]);
        }
        return l0C;
    }

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::A1> l1Buf_;
    AscendC::TBuf<AscendC::TPosition::A2> l0ABuf_;
    AscendC::TBuf<AscendC::TPosition::B2> l0BBuf_;
    AscendC::TBuf<AscendC::TPosition::CO1> l0CBuf_;
    AscendC::GlobalTensor<bfloat16_t> v_, akk_, vNew_, h_, dh_, dvScan_;
    GM_ADDR workspace_ = nullptr;
    AscendC::TEventID inputReady_[2], l1Free_[2], akkFree_[4], keReady_[2], keFree_[2];
    AscendC::TEventID operandReady_[2], operandFree_[2], resultReady_[2], resultFree_[2], dwReady_[2];
};

} // namespace KDA::Arch22

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_CUBE_H
