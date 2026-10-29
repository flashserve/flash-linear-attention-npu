/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H

#include "chunk_kda_bwd_finalize_common.h"
#include "kernel_operator.h"

namespace KDA::Arch22 {

template <typename BetaT>
class ChunkKdaBwdFinalizeVector {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR gk, GM_ADDR h, GM_ADDR dh, GM_ADDR workspace,
                                AscendC::LocalTensor<uint8_t> ub, AscendC::TPipe *pipe)
    {
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gk));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(h));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dh));
        workspace_ = workspace;
        ub_ = ub;
        pipe_ = pipe;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            inputReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            inputFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            outputReady_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            outputFree_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

    template <bool CAPTURE = false, bool SAVE_RH = false, bool FULL_CHUNK = false, bool CUBE_RH = false>
    __aicore__ inline void Stage1(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture = nullptr,
                                  uint32_t bank = 0)
    {
        constexpr uint32_t TILE_ROWS = 32;
        constexpr uint32_t TILE_ELEMENTS = TILE_ROWS * HEAD_DIM;
        static_assert(2 * TILE_ELEMENTS * sizeof(bfloat16_t) == 2 * UB_WORK_SLOT_BYTES);
        static_assert(UB_WORK + 11 * UB_WORK_SLOT_BYTES <= UB_BYTES);
        constexpr float LN2 = 0.6931471805599453f;
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t tokenOffset = ((batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart) * HEAD_DIM;
        const int64_t stateOffset = (chunkIndex * tiling.heads + head) * HEAD_DIM * HEAD_DIM;
        const uint32_t rows = FULL_CHUNK ? CHUNK_SIZE : static_cast<uint32_t>(chunk.validRows);
        auto kSaved = ub_[UB_K].ReinterpretCast<bfloat16_t>();
        auto expSaved = ub_[UB_EXP2_GK].ReinterpretCast<float>();
        auto rH = ub_[UB_RH].ReinterpretCast<float>();
        auto gLast = ub_[UB_GK_LAST].ReinterpretCast<float>();

        // W0--W3 hold two BF16 input pairs; W4--W10 hold shared FP32 scratch.
        auto hFloat = ub_[UB_WORK + 4 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto dhFloat = ub_[UB_WORK + 6 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto product = ub_[UB_WORK + 8 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto partial = ub_[UB_WORK + 10 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        if constexpr (!CUBE_RH) {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                auto input = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                AscendC::DataCopy(input, h_[stateOffset + slot * TILE_ELEMENTS], TILE_ELEMENTS);
                AscendC::DataCopy(input[TILE_ELEMENTS], dh_[stateOffset + slot * TILE_ELEMENTS], TILE_ELEMENTS);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            }
            // Release each input after Cast, so the next DMA overlaps Mul/reduction.
            for (uint32_t tile = 0; tile < HEAD_DIM / TILE_ROWS; ++tile) {
                const uint32_t slot = tile & 1;
                auto input = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
                AscendC::Cast(hFloat, input, AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
                AscendC::Cast(dhFloat, input[TILE_ELEMENTS], AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                if (tile + 2 < HEAD_DIM / TILE_ROWS) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                    AscendC::DataCopy(input, h_[stateOffset + (tile + 2) * TILE_ELEMENTS], TILE_ELEMENTS);
                    AscendC::DataCopy(input[TILE_ELEMENTS], dh_[stateOffset + (tile + 2) * TILE_ELEMENTS],
                                      TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, hFloat, dhFloat, TILE_ELEMENTS);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(partial, product, 64, TILE_ROWS, 1, 1, 16);
                AscendC::WholeReduceSum(partial[64], product[64], 64, TILE_ROWS, 1, 1, 16);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(rH[tile * TILE_ROWS], partial, partial[64], TILE_ROWS);
            }
        }

        // Retain k, exp2_gk and gk_last in their final resident addresses.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::DataCopy(kSaved, k_[tokenOffset], rows * HEAD_DIM);
        AscendC::DataCopy(gLast, gk_[tokenOffset + (rows - 1) * HEAD_DIM], HEAD_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        // Valid rows are fully overwritten; clear only the padding after DMA.
        if (rows < CHUNK_SIZE) {
            AscendC::Duplicate(kSaved[rows * HEAD_DIM].ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                               (CHUNK_SIZE - rows) * HEAD_DIM);
            AscendC::Duplicate(expSaved[rows * HEAD_DIM], 0.0f, (CHUNK_SIZE - rows) * HEAD_DIM);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        // W0--W3 now hold gate ping/pong; W8/W9 hold BF16 kE outputs.
        auto kFloat = hFloat;
        auto kEFloat = dhFloat;
        AscendC::GlobalTensor<bfloat16_t> kEOut;
        const uint64_t slotOffset =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        kEOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_ + slotOffset + WS_KE));
        AscendC::GlobalTensor<uint8_t> snapshot;
        if constexpr (CAPTURE) {
            const uint64_t offset = static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES;
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(capture + offset));
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t row = slot * TILE_ROWS;
            const uint32_t activeRows = row < rows ? (rows - row < TILE_ROWS ? rows - row : TILE_ROWS) : 0;
            auto gate = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (activeRows != 0) {
                AscendC::DataCopy(gate, gk_[tokenOffset + row * HEAD_DIM], activeRows * HEAD_DIM);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
        }
        for (uint32_t tile = 0; tile < CHUNK_SIZE / TILE_ROWS; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t row = tile * TILE_ROWS;
            const uint32_t activeRows = row < rows ? (rows - row < TILE_ROWS ? rows - row : TILE_ROWS) : 0;
            const uint32_t count = activeRows * HEAD_DIM;
            auto gate = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            auto kEBf16 = ub_[UB_WORK + 8 * UB_WORK_SLOT_BYTES + slot * TILE_ELEMENTS * sizeof(bfloat16_t)]
                              .ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            if (count != 0) {
                AscendC::Cast(kFloat, kSaved[row * HEAD_DIM], AscendC::RoundMode::CAST_NONE, count);
                AscendC::Muls(expSaved[row * HEAD_DIM], gate, LN2, count);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < CHUNK_SIZE / TILE_ROWS) {
                const uint32_t nextRow = (tile + 2) * TILE_ROWS;
                const uint32_t nextRows =
                    nextRow < rows ? (rows - nextRow < TILE_ROWS ? rows - nextRow : TILE_ROWS) : 0;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                if (nextRows != 0) {
                    AscendC::DataCopy(gate, gk_[tokenOffset + nextRow * HEAD_DIM], nextRows * HEAD_DIM);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            }
            // Only the same output slot waits for its preceding MTE3 consumer.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            AscendC::Duplicate(kEBf16.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), TILE_ELEMENTS);
            if (count != 0) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(expSaved[row * HEAD_DIM], expSaved[row * HEAD_DIM], count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(kEFloat, kFloat, expSaved[row * HEAD_DIM], count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(kEBf16, kEFloat, AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::DataCopy(kEOut[row * HEAD_DIM], kEBf16, TILE_ELEMENTS);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(snapshot[CAPTURE_KE + row * HEAD_DIM * sizeof(bfloat16_t)],
                                  kEBf16.ReinterpretCast<uint8_t>(), TILE_ELEMENTS * sizeof(bfloat16_t));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // Acknowledge both consumed free flags before MTE3 primes them again.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        if constexpr (CAPTURE) {
            AscendC::DataCopy(snapshot[CAPTURE_EXP2_GK], expSaved.ReinterpretCast<uint8_t>(),
                              CHUNK_SIZE * HEAD_DIM * sizeof(float));
            AscendC::DataCopy(snapshot[CAPTURE_RH], rH.ReinterpretCast<uint8_t>(), HEAD_DIM * sizeof(float));
            AscendC::DataCopy(snapshot[CAPTURE_GK_LAST], gLast.ReinterpretCast<uint8_t>(), HEAD_DIM * sizeof(float));
        }
        if constexpr (SAVE_RH && !CUBE_RH) {
            AscendC::GlobalTensor<float> saved;
            saved.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + slotOffset + WS_RH));
            AscendC::DataCopy(saved, rH, HEAD_DIM);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // The caller publishes KE_READY on PIPE_MTE3; UB residents survive Stage 1.
    }

    template <bool COMPACT_MASK = false>
    __aicore__ inline void InitStage3(GM_ADDR v, GM_ADDR beta, GM_ADDR dv)
    {
        v_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(beta));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
        if constexpr (COMPACT_MASK) {
            auto bits = ub_[UB_LOWER_BITS].ReinterpretCast<uint64_t>();
            auto addresses = ub_[UB_MASK_ADDR].ReinterpretCast<uint64_t>();
            for (uint32_t row = 0; row < CHUNK_SIZE; ++row) {
                bits.SetValue(row, (1ULL << row) - 1);
            }
            for (uint32_t tile = 0; tile < 4; ++tile) {
                addresses.SetValue(tile * 4, reinterpret_cast<uint64_t>(bits[tile * 16].GetPhyAddr()));
                addresses.SetValue(tile * 4 + 1, 0);
            }
            AscendC::Duplicate(ub_[UB_MASK_ZERO].ReinterpretCast<float>(), 0.0f, 8);
            const auto maskReady = pipe_->AllocEventID<AscendC::HardEvent::S_V>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(maskReady);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(maskReady);
            pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(maskReady);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            auto lower = ub_[64 * 1024].ReinterpretCast<float>();
            AscendC::Duplicate(lower, 0.0f, CHUNK_SIZE * CHUNK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row = 1; row < CHUNK_SIZE; ++row) {
                AscendC::Duplicate(lower[row * CHUNK_SIZE], 1.0f, row);
            }
        }
    }

    template <bool CAPTURE = false, bool FULL_CHUNK = false, bool COMPACT_MASK = false>
    __aicore__ inline void Stage3(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture = nullptr,
                                  uint32_t bank = 0)
    {
        constexpr uint32_t ROWS = 32;
        constexpr uint32_t ELEMENTS = ROWS * HEAD_DIM;
        const uint32_t rows = FULL_CHUNK ? CHUNK_SIZE : static_cast<uint32_t>(chunk.validRows);
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> dvB, zV, zW;
        AscendC::GlobalTensor<bfloat16_t> zbOut;
        dvB.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DV_B));
        zV.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_ZV));
        zW.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_ZW));
        zbOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace_ + base + WS_ZB));
        AscendC::GlobalTensor<uint8_t> snapshot;
        if constexpr (CAPTURE) {
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        auto beta = ub_[UB_BETA].ReinterpretCast<float>();
        auto db = ub_[UB_DB_V].ReinterpretCast<float>();
        auto lower = ub_[64 * 1024].ReinterpretCast<float>();
        auto value = ub_[UB_WORK + 6 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto product = ub_[UB_WORK + 8 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto scaled = ub_[UB_WORK + 10 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto broadcast = ub_[UB_WORK + 12 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
        auto partial = broadcast[512];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        if (rows < CHUNK_SIZE) {
            AscendC::Duplicate(beta, 0.0f, CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        }
        // inputFree protects this slot; Stage 1 arithmetic scratch may still be live.
        auto betaInput = ub_[UB_WORK].template ReinterpretCast<BetaT>();
        if constexpr (sizeof(BetaT) == sizeof(float)) {
            betaInput = beta;
        }
        AscendC::DataCopyPad(betaInput, beta_[token],
                             {1, rows * static_cast<uint32_t>(sizeof(BetaT)), 0, 0, 0},
                             {true, 0, 0, static_cast<BetaT>(0)});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        if constexpr (sizeof(BetaT) != sizeof(float)) {
            AscendC::Cast(beta, betaInput, AscendC::RoundMode::CAST_NONE, rows);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(broadcast, beta, 8, {1, 8});

        // Two independent zV/zW input pairs and two BF16 output slots.
        for (uint32_t tile = 0; tile < 2; ++tile) {
            auto input = ub_[UB_WORK + tile * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[tile]);
            AscendC::DataCopy(input, zV[tile * ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::DataCopy(input[ROWS * CHUNK_SIZE], zW[tile * ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[tile]);
        }
        for (uint32_t tile = 0; tile < CHUNK_SIZE / ROWS; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t offset = tile * ROWS * CHUNK_SIZE;
            auto input = ub_[UB_WORK + slot * 2 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            auto output = ub_[slot * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(value, input, input[ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(value, value, beta, 64, ROWS, {1, 1, 1, 8, 8, 0});
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (COMPACT_MASK) {
                // The prebuilt pointer avoids Select's implicit scratch at UB[184 KiB].
                AscendC::SetCmpMask(ub_[UB_MASK_ADDR].ReinterpretCast<uint64_t>()[tile * ROWS / 16 * 4]);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<float>(64);
                AscendC::Select<float, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE>(
                    value, value, ub_[UB_MASK_ZERO].ReinterpretCast<float>(), ROWS, {1, 1, 0, 8, 8, 0});
                AscendC::ResetMask();
            } else {
                AscendC::Mul(value, value, lower[offset], ROWS * CHUNK_SIZE);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(output, value, AscendC::RoundMode::CAST_RINT, ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::DataCopy(zbOut[offset], output, ROWS * CHUNK_SIZE);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(snapshot[CAPTURE_ZB + offset * 2], output.ReinterpretCast<uint8_t>(),
                                  ROWS * CHUNK_SIZE * 2);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }

        // The new pairs overlap both old slots. Drain both before remapping;
        // the following loads consume these permissions directly, without
        // re-priming.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
        }
        // W0--W5 now hold two (FP32 dv_b, BF16 v) pairs. Keep beta/db_v resident.
        for (uint32_t tile = 0; tile < 2; ++tile) {
            auto input = ub_[UB_WORK + tile * 3 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            const uint32_t first = tile * ROWS;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            AscendC::DataCopy(input, dvB[first * HEAD_DIM], ELEMENTS);
            if (active != 0) {
                AscendC::DataCopy(input[ELEMENTS].ReinterpretCast<bfloat16_t>(), v_[(token + first) * HEAD_DIM],
                                  active * HEAD_DIM);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[tile]);
        }
        for (uint32_t tile = 0; tile < CHUNK_SIZE / ROWS; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t first = tile * ROWS;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            auto input = ub_[UB_WORK + slot * 3 * UB_WORK_SLOT_BYTES].ReinterpretCast<float>();
            auto vInput = input[ELEMENTS].ReinterpretCast<bfloat16_t>();
            auto output = ub_[slot * UB_WORK_SLOT_BYTES].ReinterpretCast<bfloat16_t>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            if (active < ROWS) {
                AscendC::Duplicate(vInput[active * HEAD_DIM].ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                                   (ROWS - active) * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(value, vInput, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            AscendC::Mul(scaled, input, broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(scaled[64], input[64], broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, value, input, ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(partial, product, 64, ROWS, 1, 1, 16);
            AscendC::WholeReduceSum(partial[64], product[64], 64, ROWS, 1, 1, 16);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(db[first], partial, partial[64], ROWS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            AscendC::Cast(output, scaled, AscendC::RoundMode::CAST_RINT, ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            if (active != 0) {
                if (tiling.outputTokenMajor) {
                    const uint64_t dst = (static_cast<uint64_t>(batch) * tiling.tokens + chunk.tokenStart + first) *
                                             tiling.heads * HEAD_DIM +
                                         static_cast<uint64_t>(head) * HEAD_DIM;
                    AscendC::DataCopyPad(dv_[dst], output,
                                         {static_cast<uint16_t>(active), HEAD_DIM * sizeof(bfloat16_t), 0,
                                          static_cast<uint32_t>((tiling.heads - 1) * HEAD_DIM * sizeof(bfloat16_t)),
                                          0});
                } else {
                    AscendC::DataCopy(dv_[(token + first) * HEAD_DIM], output, active * HEAD_DIM);
                }
            }
            if constexpr (CAPTURE) {
                if (tile + 1 == CHUNK_SIZE / ROWS) {
                    AscendC::DataCopy(snapshot[CAPTURE_DB_V], db.ReinterpretCast<uint8_t>(),
                                      CHUNK_SIZE * sizeof(float));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // The next Stage 1/3 loads slot 0 then slot 1, waiting each old reader.
        // Only shared Vector scratch needs a barrier here; output slots remain
        // independent.
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitStage5(GM_ADDR q, GM_ADDR dqRaw, float scale)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        scale_ = scale;
        auto rhIndices = ub_[85 * 1024 + 256].ReinterpretCast<uint32_t>();
        for (uint32_t row = 0; row < HEAD_DIM; ++row) {
            rhIndices.SetValue(row, ((row / 32) * 4 * FRACTAL_ELEMENTS + (row % 32) * 33) * sizeof(float));
        }
        const auto indicesReady = pipe_->AllocEventID<AscendC::HardEvent::S_V>();
        AscendC::SetFlag<AscendC::HardEvent::S_V>(indicesReady);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(indicesReady);
        pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(indicesReady);
    }

    template <bool CAPTURE = false, bool FULL_CHUNK = false, bool CUBE_RH = false, bool SAVE_DQ = false,
              bool KEEP_INPUTS = false>
    __aicore__ inline void Stage5(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture, uint32_t bank)
    {
        static_assert(!KEEP_INPUTS || (FULL_CHUNK && !CAPTURE));
        constexpr uint32_t ROWS = 16;
        constexpr uint32_t ELEMENTS = ROWS * HEAD_DIM;
        constexpr float LN2 = 0.6931471805599453f;
        const uint32_t rows = FULL_CHUNK ? CHUNK_SIZE : static_cast<uint32_t>(chunk.validRows);
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> state, gb, dq, dk, dg, dbOut, savedRh;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DK_STATE_RAW));
        gb.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DKGB_RAW));
        dq.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DQ_BASE));
        dk.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DK_BASE));
        dg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DG_BASE));
        dbOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DB_BASE));
        savedRh.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + (CUBE_RH ? WS_DQ_LOCAL : WS_RH)));
        AscendC::GlobalTensor<uint8_t> snapshot;
        if constexpr (CAPTURE) {
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        auto kFloat = ub_[(KEEP_INPUTS ? 136 : 48) * 1024].ReinterpretCast<float>();
        auto qFloat = ub_[(KEEP_INPUTS ? 144 : 56) * 1024].ReinterpretCast<float>();
        auto exponent = ub_[(KEEP_INPUTS ? 152 * 1024 : 168 * 1024 + 256)].ReinterpretCast<float>();
        auto product = ub_[(KEEP_INPUTS ? 160 * 1024 : 176 * 1024 + 512)].ReinterpretCast<float>();
        auto temp = ub_[(KEEP_INPUTS ? 144 : 56) * 1024].ReinterpretCast<float>();
        auto last = ub_[UB_GK_LAST].ReinterpretCast<float>();
        auto rh = ub_[UB_RH].ReinterpretCast<float>();
        auto gate = ub_[83 * 1024].ReinterpretCast<float>();
        auto broadcast = ub_[86 * 1024].ReinterpretCast<float>();
        auto db = ub_[UB_DB_V].ReinterpretCast<float>();
        // Both old Stage 3 input pairs overlap this layout. Its output DMA also
        // reads the region reused for FP32 outputs and the resident db_v snapshot.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::DataCopy(last, gk_[(token + rows - 1) * HEAD_DIM], HEAD_DIM);
        if constexpr (CUBE_RH) {
            AscendC::DataCopy(ub_[UB_WORK].ReinterpretCast<float>(), savedRh, HEAD_DIM * 32);
        } else {
            AscendC::DataCopy(rh, savedRh, HEAD_DIM);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        if constexpr (CUBE_RH) {
            AscendC::Gather(rh, ub_[UB_WORK].ReinterpretCast<float>(), ub_[85 * 1024 + 256].ReinterpretCast<uint32_t>(),
                            0, HEAD_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            // Gather must finish before the first Stage 5 input overwrites its source.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        }
        AscendC::TEventID slotFree[2];
        if constexpr (KEEP_INPUTS) {
            // Full q/k/gk stay at the addresses used by Stage 7. Stage 3
            // output readers and the rH gather have retired before this copy.
            if constexpr (!CUBE_RH) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
            }
            AscendC::DataCopy(ub_.ReinterpretCast<bfloat16_t>(), q_[token * HEAD_DIM], CHUNK_SIZE * HEAD_DIM);
            AscendC::DataCopy(ub_[UB_K].ReinterpretCast<bfloat16_t>(), k_[token * HEAD_DIM], CHUNK_SIZE * HEAD_DIM);
            AscendC::DataCopy(ub_[UB_EXP2_GK].ReinterpretCast<float>(), gk_[token * HEAD_DIM], CHUNK_SIZE * HEAD_DIM);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                slotFree[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
        AscendC::Muls(gate, last, LN2, HEAD_DIM);
        AscendC::Brcb(broadcast, ub_[UB_BETA].ReinterpretCast<float>(), 8, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gate, gate, HEAD_DIM);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(gate, gate, rh, HEAD_DIM);
        // Acknowledge the two old output permissions before re-priming.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            const uint32_t first = slot * ROWS;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            LoadStage5Tile<KEEP_INPUTS>(state, gb, token, first, active, slot);
        }
        for (uint32_t tile = 0; tile < 4; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t first = tile * ROWS;
            const uint32_t offset = first * HEAD_DIM;
            const uint32_t active = first < rows ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            auto input = ub_[UB_WORK + slot * (KEEP_INPUTS ? 24 : 40) * 1024].ReinterpretCast<float>();
            auto gIn = KEEP_INPUTS ? ub_[UB_EXP2_GK].ReinterpretCast<float>()[offset] : input;
            auto kIn = KEEP_INPUTS ? ub_[UB_K].ReinterpretCast<bfloat16_t>()[offset] :
                                    input[4 * ELEMENTS].ReinterpretCast<bfloat16_t>();
            auto qIn = KEEP_INPUTS ? ub_.ReinterpretCast<bfloat16_t>()[offset] : kIn[ELEMENTS];
            auto stateIn = input[KEEP_INPUTS ? 0 : ELEMENTS];
            auto gbIn = input[KEEP_INPUTS ? ELEMENTS : 2 * ELEMENTS];
            auto dqIn = input[KEEP_INPUTS ? 2 * ELEMENTS : 3 * ELEMENTS];
            auto dqOut = KEEP_INPUTS ? dqIn : ub_[slot * 24 * 1024].ReinterpretCast<float>();
            auto dkOut = KEEP_INPUTS ? stateIn : dqOut[ELEMENTS];
            auto dgOut = KEEP_INPUTS ? gbIn : dqOut[2 * ELEMENTS];
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            if (active < ROWS) {
                AscendC::Duplicate(input[active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(input[3 * ELEMENTS + active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(kIn[active * HEAD_DIM].ReinterpretCast<uint16_t>(), uint16_t(0),
                                   (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(qIn[active * HEAD_DIM].ReinterpretCast<uint16_t>(), uint16_t(0),
                                   (ROWS - active) * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(kFloat, kIn, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            // Form the exponent difference before Exp; never divide by exp(gk).
            AscendC::Sub(temp, last, gIn, 64, ROWS, {1, 1, 1, 16, 0, 16});
            AscendC::Sub(temp[64], last[64], gIn[64], 64, ROWS, {1, 1, 1, 16, 0, 16});
            AscendC::Muls(exponent, gIn, LN2, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(temp, temp, LN2, ELEMENTS);
            AscendC::Exp(exponent, exponent, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(temp, temp, ELEMENTS);
            AscendC::Mul(dqOut, dqIn, exponent, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(dkOut, stateIn, temp, ELEMENTS);
            AscendC::Muls(dqOut, dqOut, scale_, ELEMENTS);
            AscendC::Mul(exponent, exponent, gbIn, ELEMENTS);
            // Ratio is dead after dkOut; reuse its scratch for q before releasing
            // input.
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qFloat, qIn, AscendC::RoundMode::CAST_NONE, ELEMENTS);
            if constexpr (!KEEP_INPUTS) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                if (tile + 2 < 4) {
                    const uint32_t next = (tile + 2) * ROWS;
                    const uint32_t count = next < rows ? (rows - next < ROWS ? rows - next : ROWS) : 0;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                    LoadStage5Tile<KEEP_INPUTS>(state, gb, token, next, count, slot);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(product, kFloat, dkOut, ELEMENTS);
            AscendC::Mul(dgOut, qFloat, dqOut, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(dgOut, dgOut, product, ELEMENTS);
            // Tree reduction across tokens, retaining all 128 channels.
            for (uint32_t count = ELEMENTS / 2; count >= HEAD_DIM; count /= 2) {
                AscendC::Add(product, product, product[count], count);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(gate, gate, product, HEAD_DIM);
            AscendC::Mul(product, kFloat, exponent, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(temp, product, 64, ROWS, 1, 1, 16);
            AscendC::WholeReduceSum(temp[64], product[64], 64, ROWS, 1, 1, 16);
            AscendC::Mul(product, product, broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(product[64], product[64], broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(exponent, exponent, broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::Mul(exponent[64], exponent[64], broadcast[first * 8], 64, ROWS, {1, 1, 0, 16, 16, 1});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(temp, temp, temp[64], ROWS);
            AscendC::Sub(dkOut, dkOut, exponent, ELEMENTS);
            AscendC::Sub(dgOut, dgOut, product, ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(db[first], db[first], temp, ROWS);
            if (first < rows && rows <= first + ROWS) {
                AscendC::Add(dgOut[(rows - first - 1) * HEAD_DIM], dgOut[(rows - first - 1) * HEAD_DIM], gate,
                             HEAD_DIM);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            if constexpr (SAVE_DQ && FULL_CHUNK && !CAPTURE) {
                if (tile == 3) {
                    // Preserve the final eight rows until the first Stage 9 remap.
                    AscendC::Adds(ub_[188 * 1024].ReinterpretCast<float>(), dqOut[8 * HEAD_DIM], 0.0f,
                                  8 * HEAD_DIM);
                    AscendC::DataCopy(dq[offset], dqOut, 8 * HEAD_DIM);
                } else {
                    AscendC::DataCopy(dq[offset], dqOut, ELEMENTS);
                }
            } else {
                AscendC::DataCopy(dq[offset], dqOut, ELEMENTS);
            }
            AscendC::DataCopy(dk[offset], dkOut, ELEMENTS);
            AscendC::DataCopy(dg[offset], dgOut, ELEMENTS);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(snapshot[CAPTURE_DQ_BASE + offset * 4], dqOut.ReinterpretCast<uint8_t>(),
                                  ELEMENTS * 4);
                AscendC::DataCopy(snapshot[CAPTURE_DK_BASE + offset * 4], dkOut.ReinterpretCast<uint8_t>(),
                                  ELEMENTS * 4);
                AscendC::DataCopy(snapshot[CAPTURE_DG_BASE + offset * 4], dgOut.ReinterpretCast<uint8_t>(),
                                  ELEMENTS * 4);
            }
            if (tile == 3) {
                AscendC::DataCopy(dbOut, db, CHUNK_SIZE);
                if constexpr (CAPTURE) {
                    AscendC::DataCopy(snapshot[CAPTURE_DB_BASE], db.ReinterpretCast<uint8_t>(), CHUNK_SIZE * 4);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            if constexpr (KEEP_INPUTS) {
                if (tile + 2 < 4) {
                    // Input and output share the slot: refill only after all
                    // three stores retire. The other slot can compute meanwhile.
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
                    LoadStage5Tile<true>(state, gb, token, (tile + 2) * ROWS, ROWS, slot);
                } else {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                }
            }
        }
        if constexpr (KEEP_INPUTS) {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
            }
        }
        // Stage 1 remaps the scratch/output regions and writes rH/gk_last again.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

    __aicore__ inline void InitStage7(GM_ADDR dAqk)
    {
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        // CreateVecIndex starts with scalar stores on A2. Initialize once in a
        // persistent region, outside every Stage 1/3/5 scratch lifetime.
        auto indices = ub_[84 * 1024 + 768].ReinterpretCast<int32_t>();
        AscendC::CreateVecIndex(indices, 0, 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(indices, indices, static_cast<int32_t>((CHUNK_SIZE + 1) * 4), 32);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool CAPTURE = false, bool CONSUME_BANDS = false, bool FINALIZE_Q = false, bool Q_NORM = false,
              bool COMPUTE_RIGHT = false, bool FINALIZE_K = false>
    __aicore__ inline void Stage7(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture, uint32_t bank)
    {
        // Dispatch once per chunk; full chunks have fixed counts in Stages 7/9/11.
        if (chunk.validRows == CHUNK_SIZE) {
            Stage7Impl<CAPTURE, CONSUME_BANDS, FINALIZE_Q, Q_NORM, COMPUTE_RIGHT, FINALIZE_K, true>(
                tiling, chunk, chunkIndex, head, owner, core, capture, bank);
        } else {
            Stage7Impl<CAPTURE, CONSUME_BANDS, FINALIZE_Q, Q_NORM, COMPUTE_RIGHT, FINALIZE_K, false>(
                tiling, chunk, chunkIndex, head, owner, core, capture, bank);
        }
    }

    template <bool CAPTURE, bool CONSUME_BANDS, bool FINALIZE_Q, bool Q_NORM, bool COMPUTE_RIGHT, bool FINALIZE_K,
              bool FULL_CHUNK>
    __aicore__ inline void Stage7Impl(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                      int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture, uint32_t bank)
    {
        constexpr uint32_t ROWS = FULL_CHUNK && !CAPTURE ? 32 : 16;
        constexpr uint32_t ELEMENTS = ROWS * HEAD_DIM;
        constexpr float LN2 = 0.6931471805599453f;
        const uint32_t rows = FULL_CHUNK ? CHUNK_SIZE : chunk.validRows;
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> dAkk;
        AscendC::GlobalTensor<uint8_t> out, snapshot;
        dAkk.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DAKK_RAW));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(workspace_ + base));
        if constexpr (CAPTURE) {
            snapshot.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        // 0--64 KiB: q/k/original gk; 64--80: FP32 mask, or compact Stage 11 raw gates.
        // Full-chunk lower: inputs [88,120), value [120,128), FP32 outputs [144,160).
        // Tail/capture lower retains the 16-row input/value/hi-lo layout.
        // Bands: FP32 outputs [88,120), exponent/power scratch [120,160).
        // 160--176: band exponent; 176--178: beta broadcast; gather offsets stay
        // at 84.75 KiB.
        auto q = ub_.ReinterpretCast<bfloat16_t>();
        auto k = ub_[UB_K].ReinterpretCast<bfloat16_t>();
        auto g = ub_[UB_EXP2_GK].ReinterpretCast<float>();
        auto lower = ub_[64 * 1024].ReinterpretCast<float>();
        auto diag = ub_[80 * 1024 + 512].ReinterpretCast<float>();
        auto center = ub_[82 * 1024 + 768].ReinterpretCast<float>();
        auto value = ub_[(ROWS == 32 ? 120 : 104) * 1024].ReinterpretCast<float>();
        auto rebuilt = ub_[112 * 1024].ReinterpretCast<float>();
        auto bandExp = ub_[160 * 1024].ReinterpretCast<float>();
        auto broadcast = ub_[176 * 1024].ReinterpretCast<float>();
        auto indices = ub_[84 * 1024 + 768].ReinterpretCast<int32_t>();
        // Drain old layouts before remapping. Input and output permissions remain
        // independent.
        // Full lower stores use [144,160); 16-row lower stores use [178,186) KiB.
        // Bands then reuse retired inputs, independently of capture BF16 outputs.
        AscendC::TEventID fp32Free[2];
        for (uint32_t slot = 0; slot < 2; ++slot) {
            fp32Free[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // MTE3_V drains on V only; bridge to MTE2 before overwriting Stage 5
        // outputs.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        if constexpr (!FULL_CHUNK || CAPTURE || !FINALIZE_K) {
            AscendC::DataCopy(q, q_[token * HEAD_DIM], rows * HEAD_DIM);
            AscendC::DataCopy(k, k_[token * HEAD_DIM], rows * HEAD_DIM);
            AscendC::DataCopy(g, gk_[token * HEAD_DIM], rows * HEAD_DIM);
        } // Full production chunks retain q/k/gk from Stage 5.
        AscendC::Brcb(broadcast, ub_[UB_BETA].ReinterpretCast<float>(), 8, {1, 8});
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[UB_WORK + slot * ROWS * CHUNK_SIZE * 2 * sizeof(float)].ReinterpretCast<float>();
            const uint32_t active = rows > slot * ROWS ? (rows - slot * ROWS < ROWS ? rows - slot * ROWS : ROWS) : 0;
            if (active != 0) {
                AscendC::DataCopy(input, dAqk_[(token + slot * ROWS) * CHUNK_SIZE], active * CHUNK_SIZE);
            }
            AscendC::DataCopy(input[ROWS * CHUNK_SIZE], dAkk[slot * ROWS * CHUNK_SIZE], ROWS * CHUNK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // Chunk-wide strict lower matrices are generated once. Output planes are hi
        // then lo.
        for (uint32_t tile = 0; tile < CHUNK_SIZE / ROWS; ++tile) {
            const uint32_t first = tile * ROWS;
            const uint32_t slot = tile & 1;
            const uint32_t active = rows > first ? (rows - first < ROWS ? rows - first : ROWS) : 0;
            auto input = ub_[UB_WORK + slot * ROWS * CHUNK_SIZE * 2 * sizeof(float)].ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            if (active < ROWS) {
                AscendC::Duplicate(input[active * CHUNK_SIZE], 0.0f, (ROWS - active) * CHUNK_SIZE);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(diag[first], input, indices.ReinterpretCast<uint32_t>(), first * 4, ROWS);
            for (uint32_t matrix = 0; matrix < 2; ++matrix) {
                // Alternate by output matrix, independently of the input tile slot.
                const uint32_t outputSlot = matrix;
                auto output = ub_[120 * 1024 + outputSlot * 8 * 1024].ReinterpretCast<bfloat16_t>();
                if constexpr (FINALIZE_K && !Q_NORM && !CAPTURE) {
                    AscendC::SetCmpMask(ub_[UB_MASK_ADDR].ReinterpretCast<uint64_t>()[first / 16 * 4]);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetVectorMask<float>(64);
                    AscendC::Select<float, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE>(
                        value, input[matrix * ROWS * CHUNK_SIZE], ub_[UB_MASK_ZERO].ReinterpretCast<float>(),
                        ROWS, {1, 1, 0, 8, 8, 0});
                    AscendC::ResetMask();
                } else {
                    AscendC::Mul(value, input[matrix * ROWS * CHUNK_SIZE], lower[first * CHUNK_SIZE],
                                 ROWS * CHUNK_SIZE);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(value, value, matrix == 0 ? 65536.0f : -65536.0f, ROWS * CHUNK_SIZE);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[outputSlot]);
                AscendC::PipeBarrier<PIPE_V>();
                {
                    // Preserve both FP32 lower matrices for Stage 10.
                    const uint32_t fp32Slot = tile & 1;
                    auto fp32Output =
                        ub_[(ROWS == 32 ? 144 + fp32Slot * 8 : 178 + fp32Slot * 4) * 1024].ReinterpretCast<float>();
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(fp32Free[fp32Slot]);
                    AscendC::Adds(fp32Output, value, 0.0f, ROWS * CHUNK_SIZE);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[outputSlot]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[outputSlot]);
                    const uint32_t fp32Dst = matrix == 0 ? WS_DAQK_FP32 : WS_DAKK_FP32;
                    AscendC::DataCopy(out[fp32Dst + first * CHUNK_SIZE * 4], fp32Output.ReinterpretCast<uint8_t>(),
                                      ROWS * CHUNK_SIZE * 4);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(fp32Free[fp32Slot]);
                }
                if constexpr (!CAPTURE) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[outputSlot]);
                    continue;
                }
                AscendC::Cast(output, value, AscendC::RoundMode::CAST_RINT, ROWS * CHUNK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(rebuilt, output, AscendC::RoundMode::CAST_NONE, ROWS * CHUNK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(rebuilt, value, rebuilt, ROWS * CHUNK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(output[ROWS * CHUNK_SIZE], rebuilt, AscendC::RoundMode::CAST_RINT, ROWS * CHUNK_SIZE);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[outputSlot]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[outputSlot]);
                const uint32_t dst = matrix == 0 ? WS_DAQK_LOWER : WS_DAKK_LOWER;
                const uint32_t saved = matrix == 0 ? CAPTURE_DAQK_LOWER : CAPTURE_DAKK_LOWER;
                for (uint32_t part = 0; part < 2; ++part) {
                    AscendC::DataCopy(out[dst + part * 8192 + first * CHUNK_SIZE * 2],
                                      output[part * ROWS * CHUNK_SIZE].ReinterpretCast<uint8_t>(),
                                      ROWS * CHUNK_SIZE * 2);
                    if constexpr (CAPTURE) {
                        AscendC::DataCopy(snapshot[saved + part * 8192 + first * CHUNK_SIZE * 2],
                                          output[part * ROWS * CHUNK_SIZE].ReinterpretCast<uint8_t>(),
                                          ROWS * CHUNK_SIZE * 2);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[outputSlot]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < CHUNK_SIZE / ROWS) {
                const uint32_t next = first + 2 * ROWS;
                const uint32_t count = rows > next ? (rows - next < ROWS ? rows - next : ROWS) : 0;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                if (count != 0) {
                    AscendC::DataCopy(input, dAqk_[(token + next) * CHUNK_SIZE], count * CHUNK_SIZE);
                }
                AscendC::DataCopy(input[ROWS * CHUNK_SIZE], dAkk[next * CHUNK_SIZE], ROWS * CHUNK_SIZE);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            }
        }
        // The partial harness captures both bands; its reusable workspace retains
        // band s=0. Stage 8/10 must consume each band before the next iteration
        // overwrites these slots.
        if constexpr (FINALIZE_K) {
            BeginStage11<Q_NORM>(head, token, rows);
        }
        constexpr uint32_t BAND_ROWS = CAPTURE ? 16 : 32;
        constexpr uint32_t BAND_ELEMENTS = BAND_ROWS * HEAD_DIM;
        for (int32_t start = ((rows - 1) / 32) * 32; start >= 0; start -= 32) {
            const uint32_t end = (static_cast<uint32_t>(start + 32) < rows ? start + 32 : rows);
            const uint32_t saved = CAPTURE_BANDS + (start / 32) * CAPTURE_BAND_BYTES;
            AscendC::Add(center, g[start * HEAD_DIM], g[(end - 1) * HEAD_DIM], HEAD_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(center, center, 0.5f, HEAD_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            // The previous band has consumed bandExp; reuse it to round the anchor.
            AscendC::Cast(bandExp.ReinterpretCast<int32_t>(), center, AscendC::RoundMode::CAST_RINT, HEAD_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(center, bandExp.ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, HEAD_DIM);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(bandExp, 0.0f, 32 * HEAD_DIM);
            // qPos and bkPos use the same positive exponent in each tile.
            for (uint32_t phase = 0; phase < (CONSUME_BANDS ? 2 : 1); ++phase) {
                for (uint32_t tile = 0; tile < CHUNK_SIZE / BAND_ROWS; ++tile) {
                    const uint32_t first = tile * BAND_ROWS;
                    for (uint32_t matrix = CONSUME_BANDS && phase == 1 ? 1 : 0;
                         matrix < (CONSUME_BANDS && phase == 0 ? 1 : 3); ++matrix) {
                        const uint32_t dst = matrix == 0 ? WS_KNEG : (matrix == 1 ? WS_QPOS : WS_BKPOS);
                        // kNeg alternates across tiles; qPos/bkPos alternate within each tile.
                        const uint32_t slot = CONSUME_BANDS && phase == 1 ? matrix - 1 : (3 * tile + matrix) & 1;
                        const uint32_t limit = matrix == 0 ? end : rows;
                        const uint32_t active = first < limit && (matrix == 0 || first >= start) ?
                                                    (limit - first < BAND_ROWS ? limit - first : BAND_ROWS) :
                                                    0;
                        if constexpr (FULL_CHUNK && FINALIZE_K && !CAPTURE) {
                            // Cube reads only the valid 32-row half with K=32.
                            // An empty tile consumes no local ready/free event.
                            if (active == 0) {
                                continue;
                            }
                        }
                        auto output = ub_[120 * 1024 + slot * 8 * 1024].ReinterpretCast<bfloat16_t>();
                        // Production stores FP32 directly from the double-buffered
                        // output slot; it also holds the temporary exponent operand.
                        auto value = ub_[(CAPTURE ? 104 : 88 + slot * 16) * 1024].ReinterpretCast<float>();
                        auto operand = ub_[(CAPTURE ? 136 : 88 + slot * 16) * 1024].ReinterpretCast<float>();
                        auto rebuilt = ub_[(CAPTURE ? 112 : 136) * 1024].ReinterpretCast<float>();
                        auto exponent = ub_[(CAPTURE ? 144 : 120) * 1024].ReinterpretCast<float>();
                        if constexpr (!CAPTURE) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
                        }
                        if (active != 0) {
                            if (active < BAND_ROWS) {
                                AscendC::Duplicate(value, 0.0f, BAND_ELEMENTS);
                            }
                            // Never evaluate Exp on masked rows or reconstruct gk from Stage
                            // 1's exponent.
                            AscendC::PipeBarrier<PIPE_V>();
                            if (matrix != 2) {
                                const uint32_t count = active * HEAD_DIM;
                                auto power = rebuilt.ReinterpretCast<int32_t>();
                                if (matrix == 0) {
                                    AscendC::Sub(exponent, center, g[first * HEAD_DIM], 64, active, {1, 1, 1, 16, 0, 16});
                                    AscendC::Sub(exponent[64], center[64], g[first * HEAD_DIM + 64], 64, active,
                                                 {1, 1, 1, 16, 0, 16});
                                } else {
                                    AscendC::Sub(exponent, g[first * HEAD_DIM], center, 64, active, {1, 1, 1, 16, 16, 0});
                                    AscendC::Sub(exponent[64], g[first * HEAD_DIM + 64], center[64], 64, active,
                                                 {1, 1, 1, 16, 16, 0});
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Maxs(operand, exponent, -126.0f, count);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Mins(operand, operand, 127.0f, count);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Cast(power, operand, AscendC::RoundMode::CAST_RINT, count);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Cast(operand, power, AscendC::RoundMode::CAST_NONE, count);
                                AscendC::PipeBarrier<PIPE_V>();
                                // Recompute the residual from original gk, not the rounded
                                // gk-center difference. Integer center +/- power stays exact.
                                if (matrix == 0) {
                                    AscendC::Sub(operand, center, operand, 64, active, {1, 1, 1, 16, 0, 16});
                                    AscendC::Sub(operand[64], center[64], operand[64], 64, active,
                                                 {1, 1, 1, 16, 0, 16});
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::Sub(exponent, operand, g[first * HEAD_DIM], count);
                                } else {
                                    AscendC::Add(operand, operand, center, 64, active, {1, 1, 1, 16, 16, 0});
                                    AscendC::Add(operand[64], operand[64], center[64], 64, active,
                                                 {1, 1, 1, 16, 16, 0});
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::Sub(exponent, g[first * HEAD_DIM], operand, count);
                                }
                                AscendC::Adds(power, power, static_cast<int32_t>(127), count);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::ShiftLeft(power, power, static_cast<int32_t>(23), count);
                                AscendC::Muls(exponent, exponent, LN2, count);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Exp(exponent, exponent, count);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Mul(exponent, exponent, rebuilt, count);
                            }
                            AscendC::Cast(operand, matrix == 1 ? q[first * HEAD_DIM] : k[first * HEAD_DIM],
                                          AscendC::RoundMode::CAST_NONE, active * HEAD_DIM);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Mul(value, operand, exponent, active * HEAD_DIM);
                            if (matrix == 1 && first < end) {
                                AscendC::Adds(bandExp[(first - start) * HEAD_DIM], exponent, 0.0f, active * HEAD_DIM);
                            }
                            if (matrix == 2) {
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Mul(value, value, broadcast[first * 8], 64, active, {1, 1, 0, 16, 16, 1});
                                AscendC::Mul(value[64], value[64], broadcast[first * 8], 64, active,
                                             {1, 1, 0, 16, 16, 1});
                            }
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
                        AscendC::PipeBarrier<PIPE_V>();
                        {
                            auto fp32Output = ub_[(88 + slot * (CAPTURE ? 8 : 16)) * 1024].ReinterpretCast<float>();
                            if constexpr (CAPTURE) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
                            }
                            if (active == 0) {
                                AscendC::Duplicate(fp32Output, 0.0f, BAND_ELEMENTS);
                            } else if constexpr (CAPTURE) {
                                AscendC::Adds(fp32Output, value, 0.0f, BAND_ELEMENTS);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                            const uint32_t fp32Dst = matrix == 0 ? WS_KNEG_FP32 : dst;
                            AscendC::DataCopy(out[fp32Dst + first * HEAD_DIM * 4],
                                              fp32Output.ReinterpretCast<uint8_t>(), BAND_ELEMENTS * 4);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
                        }
                        if constexpr (!CAPTURE) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
                            continue;
                        }
                        if (active == 0) {
                            // Both BF16 planes are zero; no residual decomposition is needed.
                            AscendC::Duplicate(output.ReinterpretCast<float>(), 0.0f, BAND_ELEMENTS);
                        } else {
                            AscendC::Cast(output, value, AscendC::RoundMode::CAST_RINT, BAND_ELEMENTS);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Cast(rebuilt, output, AscendC::RoundMode::CAST_NONE, BAND_ELEMENTS);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Sub(rebuilt, value, rebuilt, BAND_ELEMENTS);
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Cast(output[BAND_ELEMENTS], rebuilt, AscendC::RoundMode::CAST_RINT, BAND_ELEMENTS);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
                        for (uint32_t part = 0; part < 2; ++part) {
                            if (matrix == 0) {
                                AscendC::DataCopy(out[dst + part * 16384 + first * HEAD_DIM * 2],
                                                  output[part * BAND_ELEMENTS].ReinterpretCast<uint8_t>(),
                                                  BAND_ELEMENTS * 2);
                            }
                            if constexpr (CAPTURE) {
                                AscendC::DataCopy(
                                    snapshot[saved + matrix * 32768 + part * 16384 + first * HEAD_DIM * 2],
                                    output[part * BAND_ELEMENTS].ReinterpretCast<uint8_t>(), BAND_ELEMENTS * 2);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
                    }
                }
                if constexpr (CONSUME_BANDS) {
                    if (phase == 0) {
                        // kNeg is complete; Cube need not wait for qPos/bkPos.
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(BAND_READY);
                    }
                }
            }
            // These Vector operands stay in UB for Stage 9/11; only the test build
            // exports them.
            if constexpr (CAPTURE) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
                AscendC::DataCopy(snapshot[CAPTURE_DAQK_DIAG], diag.ReinterpretCast<uint8_t>(), CHUNK_SIZE * 4);
                AscendC::DataCopy(snapshot[saved + 96 * 1024], bandExp.ReinterpretCast<uint8_t>(), 32 * HEAD_DIM * 4);
                AscendC::DataCopy(snapshot[saved + 112 * 1024], center.ReinterpretCast<uint8_t>(), HEAD_DIM * 4);
            }
            // Consume both output frees before the band/resident regions can change.
            for (uint32_t slot = 0; slot < 2; ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            }
            if constexpr (CAPTURE) {
                // The preceding frees were issued before the small capture stores.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[0]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            }
            if constexpr (CONSUME_BANDS) {
                // The next band may overwrite GM only after Cube's MTE2 read.
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
                if constexpr (COMPUTE_RIGHT) {
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(BAND_READY);
                }
                if constexpr (FINALIZE_Q) {
                    Stage9<CAPTURE, Q_NORM, FULL_CHUNK, (FINALIZE_K && !CAPTURE)>(
                        tiling, chunk, chunkIndex, head, owner, core, capture, bank, start, end);
                    if (owner == 0 || (FINALIZE_K && !CAPTURE)) {
                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S8_SECOND_READY);
                    }
                }
                if constexpr (COMPUTE_RIGHT) {
                    if constexpr (FINALIZE_K) {
                        if constexpr (FULL_CHUNK && !CAPTURE) {
                            if (start == 0) {
                                Stage11<CAPTURE, Q_NORM, FULL_CHUNK, 16>(tiling, chunk, chunkIndex, head, owner, core,
                                                                         capture, bank, start, end);
                            } else if constexpr (!Q_NORM) {
                                Stage11<CAPTURE, Q_NORM, FULL_CHUNK, 16, true>(
                                    tiling, chunk, chunkIndex, head, owner, core, capture, bank, start, end);
                            } else {
                                Stage11<CAPTURE, Q_NORM, FULL_CHUNK>(tiling, chunk, chunkIndex, head, owner, core,
                                                                     capture, bank, start, end);
                            }
                        } else {
                            Stage11<CAPTURE, Q_NORM, FULL_CHUNK>(tiling, chunk, chunkIndex, head, owner, core, capture,
                                                                 bank, start, end);
                        }
                    } else {
                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(S10_FIRST_READY);
                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
                    }
                }
            }
        }
        // Stage 1's next MTE2 writes overlap Stage 7 arithmetic scratch. Its old
        // input-free flags only covered the strict-lower reads, not the band loop.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(fp32Free[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
        }
    }

    __aicore__ inline void InitStage9(GM_ADDR dq, GM_ADDR qRstd)
    {
        dq_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dq));
        qRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
    }

    template <bool CAPTURE, bool Q_NORM, bool FULL_CHUNK = false, bool DEFER_LEFT = false>
    __aicore__ inline void Stage9(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                  int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture, uint32_t bank,
                                  uint32_t start, uint32_t end)
    {
        constexpr uint32_t ROWS = 16;
        constexpr uint32_t ELEMENTS = ROWS * HEAD_DIM;
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> raw, dqBase, dgBase, savedDq, savedDg;
        raw.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DQ_LOCAL));
        dqBase.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DQ_BASE));
        dgBase.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base + WS_DG_BASE));
        if constexpr (CAPTURE) {
            const uint64_t record = static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES;
            savedDq.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(capture + record + CAPTURE_DQ));
            savedDg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(capture + record + CAPTURE_DG));
        }
        // Keep Stage 7's q/k/gk, mask, diagonal and band exponent intact.
        // Two 24 KiB slots hold raw/dq/dg; bases are updated after their last read.
        // BF16 outputs: [136,144); q/k scratch: [144,160); product: [184,192) KiB.
        auto q = ub_.ReinterpretCast<bfloat16_t>();
        auto k = ub_[UB_K].ReinterpretCast<bfloat16_t>();
        auto exponent = ub_[160 * 1024].ReinterpretCast<float>();
        auto query = ub_[144 * 1024].ReinterpretCast<float>();
        auto key = ub_[152 * 1024].ReinterpretCast<float>();
        auto product = ub_[184 * 1024].ReinterpretCast<float>();
        auto partial = ub_[178 * 1024].ReinterpretCast<float>();
        auto rowScale = ub_[180 * 1024].ReinterpretCast<float>();
        auto rstd = ub_[81 * 1024].ReinterpretCast<float>();
        auto diag = ub_[80 * 1024 + 512].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        if constexpr (Q_NORM) {
            AscendC::DataCopyPad(rstd, qRstd_[token + start], {1, (end - start) * 4U, 0, 0, 0}, {true, 0, 0, 0.0f});
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[(88 + slot * 24) * 1024].ReinterpretCast<float>();
            const uint32_t first = start + slot * ROWS;
            // Both bases are from Stage 5; prefetch them before waiting for Cube.
            if constexpr (FULL_CHUNK && !CAPTURE && DEFER_LEFT && !Q_NORM) {
                if (first == 48) {
                    AscendC::DataCopy(input[ELEMENTS], dqBase[first * HEAD_DIM], 8 * HEAD_DIM);
                    // Consume the cache before product[184,192) reuses its storage.
                    AscendC::Adds(input[ELEMENTS + 8 * HEAD_DIM], ub_[188 * 1024].ReinterpretCast<float>(),
                                  0.0f, 8 * HEAD_DIM);
                } else {
                    AscendC::DataCopy(input[ELEMENTS], dqBase[first * HEAD_DIM], ELEMENTS);
                }
            } else {
                AscendC::DataCopy(input[ELEMENTS], dqBase[first * HEAD_DIM], ELEMENTS);
            }
            AscendC::DataCopy(input[2 * ELEMENTS], dgBase[first * HEAD_DIM], ELEMENTS);
        }
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S8_READY);
        if constexpr (!DEFER_LEFT) {
            if (owner != 0) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S8_SECOND_READY);
            }
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            auto input = ub_[(88 + slot * 24) * 1024].ReinterpretCast<float>();
            const uint32_t first = start + slot * ROWS;
            AscendC::DataCopy(input, raw[first * HEAD_DIM], ELEMENTS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        for (uint32_t tile = 0; tile < 2; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t first = start + tile * ROWS;
            const uint32_t active = FULL_CHUNK ? ROWS : (first < end ? (end - first < ROWS ? end - first : ROWS) : 0);
            auto input = ub_[(88 + slot * 24) * 1024].ReinterpretCast<float>();
            auto local = input;
            auto dq = input[ELEMENTS];
            auto dg = input[2 * ELEMENTS];
            auto packed = ub_[(136 + slot * 4) * 1024].ReinterpretCast<bfloat16_t>();
            // Resident q/k/diagonal are independent of the three GM inputs.
            if (active != 0) {
                AscendC::Cast(query, q[first * HEAD_DIM], AscendC::RoundMode::CAST_NONE, active * HEAD_DIM);
                AscendC::Cast(key, k[first * HEAD_DIM], AscendC::RoundMode::CAST_NONE, active * HEAD_DIM);
                AscendC::Brcb(rowScale, diag[first], 2, {1, 8});
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            if (active != 0) {
                AscendC::Mul(local, input, exponent[tile * ELEMENTS], active * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(local, local, 1.0f / 65536.0f, active * HEAD_DIM);
                AscendC::Mul(product, key, rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                AscendC::Mul(product[64], key[64], rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dq, input[ELEMENTS], local, active * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dq, dq, product, active * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, query, local, active * HEAD_DIM);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dg, dg, product, active * HEAD_DIM);
                if constexpr (Q_NORM) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(product, dq, query, active * HEAD_DIM);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceSum(partial, product, 64, active, 1, 1, 16);
                    AscendC::WholeReduceSum(partial[64], product[64], 64, active, 1, 1, 16);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(partial, partial, partial[64], ROWS);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(rowScale, partial, 2, {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(product, query, rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                    AscendC::Mul(product[64], query[64], rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(dq, dq, product, active * HEAD_DIM);
                    AscendC::Brcb(rowScale, rstd[tile * ROWS], 2, {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(dq, dq, rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                    AscendC::Mul(dq[64], dq[64], rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(packed, dq, AscendC::RoundMode::CAST_RINT, active * HEAD_DIM);
            }
            if (active < ROWS) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(dq[active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(dg[active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
            }
            // No third tile in this band. Next Stage 9 bridges output completion
            // to MTE2 before these in-place input/output slots are loaded again.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::DataCopy(dqBase[first * HEAD_DIM], dq, ELEMENTS);
            AscendC::DataCopy(dgBase[first * HEAD_DIM], dg, ELEMENTS);
            if (active != 0) {
                if (tiling.outputTokenMajor) {
                    const uint64_t dst = (static_cast<uint64_t>(batch) * tiling.tokens + chunk.tokenStart + first) *
                                             tiling.heads * HEAD_DIM +
                                         static_cast<uint64_t>(head) * HEAD_DIM;
                    AscendC::DataCopyPad(dq_[dst], packed,
                                         {static_cast<uint16_t>(active), HEAD_DIM * sizeof(bfloat16_t), 0,
                                          static_cast<uint32_t>((tiling.heads - 1) * HEAD_DIM * sizeof(bfloat16_t)),
                                          0});
                } else {
                    AscendC::DataCopy(dq_[(token + first) * HEAD_DIM], packed, active * HEAD_DIM);
                }
            }
            if constexpr (CAPTURE) {
                AscendC::DataCopy(savedDq[first * HEAD_DIM], dq, ELEMENTS);
                AscendC::DataCopy(savedDg[first * HEAD_DIM], dg, ELEMENTS);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        // Stage 7 remaps these regions on the next band; retire both writers first.
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

    __aicore__ inline void InitStage11(const FinalizeTailArgs &tail)
    {
        tail_ = tail;
        rawG_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_RAW_G *>(tail.rawG));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tail.aLog));
        dtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tail.dtBias));
        kRstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tail.kRstd));
        dk_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(tail.dk));
        dg_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_D_G *>(tail.dg));
        db_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(tail.db));
        partial_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tail.partial));
    }

    template <bool K_NORM>
    __aicore__ inline void BeginStage11(int64_t head, int64_t token, uint32_t rows)
    {
        auto scalar = ub_[87 * 1024 + 256].ReinterpretCast<float>();
        auto factor = ub_[UB11_BETA_FACTOR].ReinterpretCast<float>();
        auto bias = ub_[UB11_BIAS].ReinterpretCast<float>();
        AscendC::DataCopy(bias, dtBias_[head * HEAD_DIM], HEAD_DIM);
        AscendC::DataCopyPad(scalar, aLog_[head], {1, sizeof(float), 0, 0, 0},
                             {true, 0, 0, 0.0f});
        if constexpr (K_NORM) {
            AscendC::DataCopyPad(ub_[UB11_RSTD].ReinterpretCast<float>(), kRstd_[token], {1, rows * 4, 0, 0, 0},
                                 {true, 0, 0, 0.0f});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::Duplicate(ub_[UB11_SUFFIX].ReinterpretCast<float>(), 0.0f, HEAD_DIM + PARTIAL_FLOATS);
        if (tail_.betaMode == 0) {
            AscendC::Duplicate(factor, 1.0f, CHUNK_SIZE);
        } else {
            auto beta = ub_[UB_BETA].ReinterpretCast<float>();
            AscendC::Muls(factor, beta, tail_.betaMode == 1 ? -1.0f : -0.5f, CHUNK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(factor, factor, 1.0f, CHUNK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(factor, factor, beta, CHUNK_SIZE);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
        AscendC::Exp(scalar, scalar, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(ub_[UB11_EXP_A].ReinterpretCast<float>(), scalar, 1, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool CAPTURE, bool K_NORM, bool FULL_CHUNK = false, uint32_t ROWS = 8, bool COMPACT = false>
    __aicore__ inline void Stage11(const FinalizeTilingData &tiling, const ChunkInfo &chunk, int64_t chunkIndex,
                                   int64_t head, uint32_t owner, uint32_t core, GM_ADDR capture, uint32_t bank,
                                   uint32_t start, uint32_t end)
    {
        static_assert(!COMPACT || (FULL_CHUNK && !CAPTURE && !K_NORM && ROWS == 16));
        // Compact first band: two four-plane slots [88,120)/[120,152) KiB.
        // Raw gate has two slots at [64,80); right -> q -> derivative -> packed dk.
        // Key becomes scan scratch after dk formation; raw gate is already prefetched.
        // dk stays FP32 until the final Cast; dg becomes the gated output in place.
        constexpr uint32_t TILES = 32 / ROWS;
        constexpr uint32_t E = ROWS * HEAD_DIM;
        const int64_t batch = tiling.isVariable ? 0 : chunk.sequence;
        const int64_t token = (batch * tiling.heads + head) * tiling.tokens + chunk.tokenStart;
        const uint64_t base =
            static_cast<uint64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES + bank * WS_BANK_BYTES;
        AscendC::GlobalTensor<float> ws, saved;
        ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ + base));
        if constexpr (CAPTURE) {
            saved.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
                capture + static_cast<uint64_t>(head * tiling.chunks + chunkIndex) * CAPTURE_BYTES));
        }
        // Last-band 16-row path: input slots [24,64)/[88,128) KiB replace dead k/gk.
        // q[8,16) is dead too; gate outputs use [8,16)/[184,192). Lower mask stays live.
        // q/k/product use [128,152); gate temporaries reuse q/k after their last read.
        // The eight-row path keeps two 20 KiB inputs at [88,128).
        // 128--148: q/k and arithmetic; 148--160: independent dk/g output pairs.
        // Keep 160--178 band exponent/beta and 80--88 resident state intact.
        auto query = ub_[128 * 1024].ReinterpretCast<float>();
        auto key = ub_[(COMPACT ? 152 : (ROWS == 16 ? 136 : 132)) * 1024].ReinterpretCast<float>();
        auto product = ub_[(COMPACT ? 184 : (ROWS == 16 ? 144 : 136)) * 1024].ReinterpretCast<float>();
        auto gateTemp = ub_[(ROWS == 16 ? 128 : 140) * 1024].ReinterpretCast<float>();
        auto denom = ub_[(COMPACT ? 184 : (ROWS == 16 ? 136 : 144)) * 1024].ReinterpretCast<float>();
        auto sums = ub_[179 * 1024].ReinterpretCast<float>();
        auto rowScale = ub_[180 * 1024].ReinterpretCast<float>();
        auto suffix = ub_[UB11_SUFFIX].ReinterpretCast<float>();
        auto partial = ub_[UB11_PARTIAL].ReinterpretCast<float>();
        auto expA = ub_[UB11_EXP_A].ReinterpretCast<float>();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[0]);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            LoadStage11Tile<ROWS, COMPACT>(ws, token, start + (TILES - 1 - slot) * ROWS, end, slot);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        if constexpr (FULL_CHUNK && !CAPTURE && K_NORM && ROWS == 8) {
            // The first tile's gate coefficient does not depend on right or its scan.
            // These temporaries survive dk/db arithmetic until the gate multiply.
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[0]);
            auto raw = ub_[88 * 1024].ReinterpretCast<float>()[4 * E];
            constexpr uint32_t active = ROWS;
            constexpr uint32_t count = E;
            ExpandRawGate<COMPACT>(raw, count);
            AscendC::Add(raw, raw, ub_[UB11_BIAS].ReinterpretCast<float>(), 64, active, {1, 1, 1, 16, 16, 0});
            AscendC::Add(raw[64], raw[64], ub_[UB11_BIAS].ReinterpretCast<float>()[64], 64, active,
                         {1, 1, 1, 16, 16, 0});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(gateTemp, raw, expA, 64, active * 2, {1, 1, 0, 8, 8, 0});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(gateTemp, gateTemp, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(gateTemp, gateTemp, -1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(gateTemp, gateTemp, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(denom, gateTemp, 1.0f, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(denom, denom, denom, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(gateTemp, gateTemp, denom, count);
        }
        // Only right depends on Stage 10. Queue the other inputs before waiting.
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S10_FIRST_READY);
        if (owner != 0) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(BAND_LOADED);
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t first = start + (TILES - 1 - slot) * ROWS;
            auto input =
                ub_[(COMPACT ? 88 + slot * 32 : (ROWS == 16 ? (slot == 0 ? 24 : 88) : 88 + slot * 20)) * 1024]
                    .ReinterpretCast<float>();
            AscendC::DataCopy(input[E], ws[WS_RIGHT / 4 + first * HEAD_DIM], E);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
        }
        for (uint32_t tile = 0; tile < TILES; ++tile) {
            const uint32_t slot = tile & 1;
            const uint32_t first = start + (TILES - 1 - tile) * ROWS;
            const uint32_t active = FULL_CHUNK ? ROWS : (first < end ? (end - first < ROWS ? end - first : ROWS) : 0);
            auto input =
                ub_[(COMPACT ? 88 + slot * 32 : (ROWS == 16 ? (slot == 0 ? 24 : 88) : 88 + slot * 20)) * 1024]
                    .ReinterpretCast<float>();
            auto left = input;
            auto right = input[E];
            auto dk = input[2 * E];
            auto dg = input[3 * E];
            auto raw = COMPACT ? ub_[(64 + slot * 8) * 1024].ReinterpretCast<float>() : input[4 * E];
            auto queryTile = COMPACT ? right : query;
            auto gateTile = COMPACT ? right : gateTemp;
            auto db =
                ub_[(ROWS == 16 ? 181 * 1024 + slot * 64 : 178 * 1024 + 512 + slot * 32)].ReinterpretCast<float>();
            auto packedDb = ub_[178 * 1024 + slot * ROWS * 32].ReinterpretCast<float>();
            auto packedDk =
                COMPACT ? right.ReinterpretCast<bfloat16_t>() :
                          ub_[(ROWS == 16 ? 152 + slot * 4 : 148 + slot * 2) * 1024].ReinterpretCast<bfloat16_t>();
            auto gateOut =
                COMPACT ? dg :
                          ub_[(ROWS == 16 ? (slot == 0 ? 8 : 184) : 152 + slot * 4) * 1024].ReinterpretCast<float>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            if (active != 0) {
                const uint32_t count = active * HEAD_DIM;
                if constexpr (FULL_CHUNK && !CAPTURE && K_NORM && ROWS == 8) {
                    if (tile != 0) ExpandRawGate<COMPACT>(raw, count);
                } else {
                    ExpandRawGate<COMPACT>(raw, count);
                }
                auto exponent = ub_[160 * 1024].ReinterpretCast<float>()[(first - start) * HEAD_DIM];
                if constexpr (!COMPACT) {
                    AscendC::Cast(queryTile, ub_.ReinterpretCast<bfloat16_t>()[first * HEAD_DIM],
                                 AscendC::RoundMode::CAST_NONE, count);
                }
                AscendC::Cast(key, ub_[UB_K].ReinterpretCast<bfloat16_t>()[first * HEAD_DIM],
                              AscendC::RoundMode::CAST_NONE, count);
                AscendC::Mul(left, left, exponent, count);
                AscendC::Div(right, right, exponent, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(left, left, 1.0f / 65536.0f, count);
                AscendC::Muls(right, right, 1.0f / 65536.0f, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, left, key, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum(sums, product, 64, active, 1, 1, 16);
                AscendC::WholeReduceSum(sums[ROWS], product[64], 64, active, 1, 1, 16);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(sums, sums, sums[ROWS], ROWS);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(db, db, sums, active);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(db, db, ub_[UB11_BETA_FACTOR].ReinterpretCast<float>()[first], active);
                AscendC::Brcb(rowScale, ub_[UB_BETA].ReinterpretCast<float>()[first], ROWS / 8, {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(left, left, rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                AscendC::Mul(left[64], left[64], rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(product, left, right, count);
                AscendC::Add(dk, dk, left, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, product, key, count);
                AscendC::Add(dk, dk, right, count);
                AscendC::Brcb(rowScale, ub_[80 * 1024 + 512].ReinterpretCast<float>()[first], ROWS / 8, {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dg, dg, product, count);
                if constexpr (COMPACT) {
                    // right has been consumed by dk; reuse it only after that read.
                    AscendC::Cast(queryTile, ub_.ReinterpretCast<bfloat16_t>()[first * HEAD_DIM],
                                 AscendC::RoundMode::CAST_NONE, count);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Mul(product, queryTile, rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                AscendC::Mul(product[64], queryTile[64], rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(dk, dk, product, count);
                if constexpr (K_NORM) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(product, dk, key, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::WholeReduceSum(sums, product, 64, active, 1, 1, 16);
                    AscendC::WholeReduceSum(sums[ROWS], product[64], 64, active, 1, 1, 16);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(sums, sums, sums[ROWS], ROWS);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(rowScale, sums, ROWS / 8, {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(product, key, rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                    AscendC::Mul(product[64], key[64], rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Sub(dk, dk, product, count);
                    AscendC::Brcb(rowScale, ub_[UB11_RSTD].ReinterpretCast<float>()[first], ROWS / 8, {1, 8});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(dk, dk, rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                    AscendC::Mul(dk[64], dk[64], rowScale, 64, active, {1, 1, 0, 16, 16, 1});
                }
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (!COMPACT) {
                    AscendC::Cast(packedDk, dk, AscendC::RoundMode::CAST_RINT, count);
                }
                // Eight rows scan in three steps; left is dead after dk/db/dg formation.
                // Carry crosses tiles and 32-row bands, never chunks.
                if constexpr (ROWS == 16) {
                    // Keep the two original eight-row scans in reverse order.
                    for (int32_t group = 8; group >= 0; group -= 8) {
                        auto scanSrc = dg[group * HEAD_DIM];
                        // key is dead after dk formation on the compact path.
                        auto scanDst = COMPACT ? key[group * HEAD_DIM] : left[group * HEAD_DIM];
                        for (uint32_t step = 1; step < 8; step *= 2) {
                            const uint32_t prefix = (8 - step) * HEAD_DIM;
                            AscendC::Add(scanDst, scanSrc, scanSrc[step * HEAD_DIM], prefix);
                            AscendC::Adds(scanDst[prefix], scanSrc[prefix], 0.0f, step * HEAD_DIM);
                            AscendC::PipeBarrier<PIPE_V>();
                            auto temp = scanSrc;
                            scanSrc = scanDst;
                            scanDst = temp;
                        }
                        AscendC::Add(dg[group * HEAD_DIM], scanSrc, suffix, 64, 8, {1, 1, 1, 16, 16, 0});
                        AscendC::Add(dg[group * HEAD_DIM + 64], scanSrc[64], suffix[64], 64, 8, {1, 1, 1, 16, 16, 0});
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Adds(suffix, dg[group * HEAD_DIM], 0.0f, HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                } else if (active == ROWS) {
                    auto scanSrc = dg;
                    auto scanDst = left;
                    for (uint32_t step = 1; step < ROWS; step *= 2) {
                        const uint32_t prefix = (ROWS - step) * HEAD_DIM;
                        AscendC::Add(scanDst, scanSrc, scanSrc[step * HEAD_DIM], prefix);
                        AscendC::Adds(scanDst[prefix], scanSrc[prefix], 0.0f, step * HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        auto temp = scanSrc;
                        scanSrc = scanDst;
                        scanDst = temp;
                    }
                    AscendC::Add(dg, scanSrc, suffix, 64, ROWS, {1, 1, 1, 16, 16, 0});
                    AscendC::Add(dg[64], scanSrc[64], suffix[64], 64, ROWS, {1, 1, 1, 16, 16, 0});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(suffix, dg, 0.0f, HEAD_DIM);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    for (int32_t row = active - 1; row >= 0; --row) {
                        AscendC::Add(dg[row * HEAD_DIM], dg[row * HEAD_DIM], suffix, HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Adds(suffix, dg[row * HEAD_DIM], 0.0f, HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
                if constexpr (FULL_CHUNK && !CAPTURE && K_NORM && ROWS == 8) {
                    if (tile != 0) {
                        AscendC::Add(raw, raw, ub_[UB11_BIAS].ReinterpretCast<float>(), 64, active,
                                     {1, 1, 1, 16, 16, 0});
                        AscendC::Add(raw[64], raw[64], ub_[UB11_BIAS].ReinterpretCast<float>()[64], 64, active,
                                     {1, 1, 1, 16, 16, 0});
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(gateTile, raw, expA, 64, active * 2, {1, 1, 0, 8, 8, 0});
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Abs(gateTile, gateTile, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(gateTile, gateTile, -1.0f, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(gateTile, gateTile, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Adds(denom, gateTile, 1.0f, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(denom, denom, denom, count);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Div(gateTile, gateTile, denom, count);
                    }
                } else {
                    AscendC::Add(raw, raw, ub_[UB11_BIAS].ReinterpretCast<float>(), 64, active, {1, 1, 1, 16, 16, 0});
                    AscendC::Add(raw[64], raw[64], ub_[UB11_BIAS].ReinterpretCast<float>()[64], 64, active,
                                 {1, 1, 1, 16, 16, 0});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(gateTile, raw, expA, 64, active * 2, {1, 1, 0, 8, 8, 0});
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Abs(gateTile, gateTile, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(gateTile, gateTile, -1.0f, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Exp(gateTile, gateTile, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(denom, gateTile, 1.0f, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(denom, denom, denom, count);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Div(gateTile, gateTile, denom, count);
                }
                AscendC::Mul(gateOut, dg, expA, 64, active * 2, {1, 1, 0, 8, 8, 0});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(gateOut, gateOut, tail_.lowerBound, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(gateOut, gateOut, gateTile, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(product, gateOut, raw, count);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (!FULL_CHUNK) {
                    AscendC::Duplicate(sums, 0.0f, 16);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::WholeReduceSum(sums, product, 64, active, 1, 1, 16);
                AscendC::WholeReduceSum(sums[ROWS], product[64], 64, active, 1, 1, 16);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(sums, sums, sums[ROWS], ROWS);
                AscendC::PipeBarrier<PIPE_V>();
                if constexpr (ROWS == 16) {
                    // Preserve both eight-row parameter reductions and their accumulation order.
                    for (int32_t group = 8; group >= 0; group -= 8) {
                        AscendC::Add(partial[HEAD_DIM], partial[HEAD_DIM], sums[group], 8);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(product, gateOut[group * HEAD_DIM], gateOut[(group + 4) * HEAD_DIM], 4 * HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(product, product, product[2 * HEAD_DIM], 2 * HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(product, product, product[HEAD_DIM], HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(partial, partial, product, HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                } else {
                    AscendC::Add(partial[HEAD_DIM], partial[HEAD_DIM], sums, ROWS);
                    if (active == ROWS) {
                        // product is dead after the dA_log reduction; preserve gateOut for MTE3.
                        AscendC::Add(product, gateOut, gateOut[4 * HEAD_DIM], 4 * HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(product, product, product[2 * HEAD_DIM], 2 * HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(product, product, product[HEAD_DIM], HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(partial, partial, product, HEAD_DIM);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        for (uint32_t row = 0; row < active; ++row) {
                            AscendC::Add(partial, partial, gateOut[row * HEAD_DIM], HEAD_DIM);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                    }
                }
            }
            if (active < ROWS) {
                AscendC::Duplicate(dk[active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(dg[active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                AscendC::Duplicate(gateOut[active * HEAD_DIM], 0.0f, (ROWS - active) * HEAD_DIM);
                // db_base already has zero padding from Stage 5; valid-lane math preserves it.
            }
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (COMPACT) {
                // The derivative's final read is complete; right now holds packed dk.
                AscendC::Cast(packedDk, dk, AscendC::RoundMode::CAST_RINT, E);
            }
            if constexpr (sizeof(BetaT) == sizeof(float)) {
                AscendC::Brcb(packedDb, db, ROWS / 8, {1, 8});
            } else {
                // Each dBeta row starts at a 32-byte boundary for strided GM writes.
                AscendC::Brcb(rowScale, db, ROWS / 8, {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(packedDb.template ReinterpretCast<BetaT>(), rowScale,
                              AscendC::RoundMode::CAST_RINT, 8, ROWS, {1, 1, 1, 1});
            }
            // The raw-gate input slot is dead after the gate chain rule; reuse it
            // for BF16 writeback without rounding parameter-reduction partials.
            auto gatePacked = raw.template ReinterpretCast<DTYPE_D_G>();
            if constexpr (sizeof(DTYPE_D_G) == 2) {
                AscendC::Cast(gatePacked, gateOut, AscendC::RoundMode::CAST_RINT, E);
            } else {
                gatePacked = gateOut.template ReinterpretCast<DTYPE_D_G>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            if (active != 0) {
                const int64_t outputToken = tiling.outputTokenMajor ?
                    ((batch * tiling.tokens + chunk.tokenStart + first) * tiling.heads + head) : token + first;
                const uint32_t outputStride = tiling.outputTokenMajor ? tiling.heads - 1 : 0;
                if (tiling.outputTokenMajor) {
                    const uint64_t dst = (static_cast<uint64_t>(batch) * tiling.tokens + chunk.tokenStart + first) *
                                             tiling.heads * HEAD_DIM +
                                         static_cast<uint64_t>(head) * HEAD_DIM;
                    AscendC::DataCopyPad(dk_[dst], packedDk,
                                         {static_cast<uint16_t>(active), HEAD_DIM * sizeof(bfloat16_t), 0,
                                          static_cast<uint32_t>((tiling.heads - 1) * HEAD_DIM * sizeof(bfloat16_t)),
                                          0});
                } else {
                    AscendC::DataCopy(dk_[(token + first) * HEAD_DIM], packedDk, active * HEAD_DIM);
                }
                constexpr uint32_t gateBytes = HEAD_DIM * sizeof(DTYPE_D_G);
                AscendC::DataCopyPad(dg_[outputToken * HEAD_DIM], gatePacked,
                    {static_cast<uint16_t>(active), gateBytes, 0,
                     static_cast<uint32_t>(outputStride * gateBytes), 0});
                AscendC::DataCopyPad(
                    db_[outputToken], packedDb.template ReinterpretCast<BetaT>(),
                    {static_cast<uint16_t>(active), sizeof(BetaT), 0,
                     static_cast<uint32_t>(outputStride * sizeof(BetaT)), 0});
            }
            if constexpr (CAPTURE) {
                AscendC::DataCopy(saved[CAPTURE_DK / 4 + first * HEAD_DIM], dk, E);
                AscendC::DataCopy(saved[CAPTURE_SCAN / 4 + first * HEAD_DIM], dg, E);
                AscendC::DataCopy(saved[CAPTURE_GATE / 4 + first * HEAD_DIM], gateOut, E);
                AscendC::DataCopy(saved[CAPTURE_DB / 4 + first], db, ROWS);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            if constexpr (CAPTURE || sizeof(DTYPE_D_G) == 2) {
                // BF16 writeback and snapshots read input slots before MTE2 reuse.
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            if (tile + 2 < TILES) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
                LoadStage11Tile<ROWS, COMPACT>(ws, token, first - 2 * ROWS, end, slot);
                AscendC::DataCopy(input[E], ws[WS_RIGHT / 4 + (first - 2 * ROWS) * HEAD_DIM], E);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            }
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[0]);
        if (start == 0) {
            AscendC::DataCopy(partial_[(head * tiling.chunks + chunkIndex) * PARTIAL_FLOATS], partial, PARTIAL_FLOATS);
            if constexpr (CAPTURE) {
                AscendC::DataCopy(saved[CAPTURE_PARTIAL / 4], partial, PARTIAL_FLOATS);
            }
        }
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
        if (owner == 0) {
            // Both heads must retire right before the next band reuses its GM slot.
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
        }
    }

    __aicore__ inline void End()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(inputFree_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(outputReady_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(outputFree_[slot]);
        }
    }

private:
    template <bool COMPACT>
    __aicore__ inline void ExpandRawGate(AscendC::LocalTensor<float> raw, uint32_t count)
    {
        if constexpr (sizeof(DTYPE_RAW_G) == 2) {
            // Product (compact) or q scratch is not live yet; expanding Cast is disjoint.
            auto expanded = ub_[(COMPACT ? 184 : 128) * 1024].ReinterpretCast<float>();
            AscendC::Cast(expanded, raw.ReinterpretCast<DTYPE_RAW_G>(), AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Copy(raw, expanded, static_cast<uint64_t>(64), count / 64, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    template <uint32_t ROWS = 8, bool COMPACT = false>
    __aicore__ inline void LoadStage11Tile(AscendC::GlobalTensor<float> ws, int64_t token, uint32_t first, uint32_t end,
                                           uint32_t slot)
    {
        constexpr uint32_t E = ROWS * HEAD_DIM;
        auto input =
            ub_[(COMPACT ? 88 + slot * 32 : (ROWS == 16 ? (slot == 0 ? 24 : 88) : 88 + slot * 20)) * 1024]
                .ReinterpretCast<float>();
        AscendC::DataCopy(input, ws[WS_LEFT / 4 + first * HEAD_DIM], E);
        AscendC::DataCopy(input[2 * E], ws[WS_DK_BASE / 4 + first * HEAD_DIM], E);
        AscendC::DataCopy(input[3 * E], ws[WS_DG_BASE / 4 + first * HEAD_DIM], E);
        AscendC::DataCopy(
            ub_[(ROWS == 16 ? 181 * 1024 + slot * 64 : 178 * 1024 + 512 + slot * 32)].ReinterpretCast<float>(),
            ws[WS_DB_BASE / 4 + first], ROWS);
        const uint32_t active = first < end ? (end - first < ROWS ? end - first : ROWS) : 0;
        if constexpr (COMPACT) {
            // The bitmask frees the old [64,80) KiB triangle for two raw-gate tiles.
            AscendC::DataCopy(ub_[(64 + slot * 8) * 1024].ReinterpretCast<DTYPE_RAW_G>(),
                              rawG_[(token + first) * HEAD_DIM], E);
        } else if (active != 0) {
            AscendC::DataCopy(input[4 * E].ReinterpretCast<DTYPE_RAW_G>(), rawG_[(token + first) * HEAD_DIM], active * HEAD_DIM);
        }
    }
    template <bool KEEP_INPUTS = false>
    __aicore__ inline void LoadStage5Tile(AscendC::GlobalTensor<float> state, AscendC::GlobalTensor<float> gb,
                                          int64_t token, uint32_t first, uint32_t rows, uint32_t slot)
    {
        constexpr uint32_t ELEMENTS = 16 * HEAD_DIM;
        auto input = ub_[UB_WORK + slot * (KEEP_INPUTS ? 24 : 40) * 1024].ReinterpretCast<float>();
        if constexpr (KEEP_INPUTS) {
            AscendC::DataCopy(input, state[first * HEAD_DIM], ELEMENTS);
            AscendC::DataCopy(input[ELEMENTS], gb[first * HEAD_DIM], ELEMENTS);
            AscendC::DataCopy(input[2 * ELEMENTS], dqRaw_[(token + first) * HEAD_DIM], ELEMENTS);
        } else {
            AscendC::DataCopy(input[ELEMENTS], state[first * HEAD_DIM], ELEMENTS);
            AscendC::DataCopy(input[2 * ELEMENTS], gb[first * HEAD_DIM], ELEMENTS);
            if (rows != 0) {
                AscendC::DataCopy(input, gk_[(token + first) * HEAD_DIM], rows * HEAD_DIM);
                AscendC::DataCopy(input[3 * ELEMENTS], dqRaw_[(token + first) * HEAD_DIM], rows * HEAD_DIM);
                AscendC::DataCopy(input[4 * ELEMENTS].ReinterpretCast<bfloat16_t>(), k_[(token + first) * HEAD_DIM],
                                  rows * HEAD_DIM);
                AscendC::DataCopy(input[4 * ELEMENTS].ReinterpretCast<bfloat16_t>()[ELEMENTS],
                                  q_[(token + first) * HEAD_DIM], rows * HEAD_DIM);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady_[slot]);
    }

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::LocalTensor<uint8_t> ub_;
    AscendC::GlobalTensor<bfloat16_t> k_, h_, dh_, v_, dv_;
    AscendC::GlobalTensor<float> gk_, dqRaw_, dAqk_;
    AscendC::GlobalTensor<BetaT> beta_, db_;
    AscendC::GlobalTensor<bfloat16_t> q_, dq_;
    AscendC::GlobalTensor<float> qRstd_;
    AscendC::GlobalTensor<DTYPE_RAW_G> rawG_;
    AscendC::GlobalTensor<float> aLog_;
    AscendC::GlobalTensor<float> dtBias_;
    AscendC::GlobalTensor<float> kRstd_, partial_;
    AscendC::GlobalTensor<DTYPE_D_G> dg_;
    AscendC::GlobalTensor<bfloat16_t> dk_;
    FinalizeTailArgs tail_;
    float scale_ = 1.0f;
    GM_ADDR workspace_ = nullptr;
    AscendC::TEventID inputReady_[2], inputFree_[2], outputReady_[2], outputFree_[2];
};

} // namespace KDA::Arch22

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH22_VECTOR_H
