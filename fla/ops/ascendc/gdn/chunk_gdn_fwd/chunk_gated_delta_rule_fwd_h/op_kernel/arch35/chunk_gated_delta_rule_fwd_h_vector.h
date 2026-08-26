/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_ARCH35_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_ARCH35_VECTOR_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "../chunk_gated_delta_rule_fwd_h_policy.h"
#include "gemm/block/block_scheduler_gdn_fwd_h.hpp"

namespace GDN::FwdHStandalone {

namespace detail {

using namespace AscendC::MicroAPI;

constexpr CastTrait B16_TO_F32_ZERO = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr CastTrait F32_TO_B16_RINT = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__simd_callee__ inline void LoadAsFloat(
    RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        Cast<float, T, B16_TO_F32_ZERO>(dst, raw, mask);
    }
}

__simd_callee__ inline void LoadFloat(RegTensor<float> &dst, __ubuf__ float *src)
{
    DataCopy<float, LoadDist::DIST_NORM>(dst, src);
}

__simd_callee__ inline void StoreFloat(
    __ubuf__ float *dst, RegTensor<float> &src, MaskReg &mask)
{
    DataCopy<float, StoreDist::DIST_NORM_B32>(dst, src, mask);
}

template <typename T>
__simd_callee__ inline void StoreFromFloat(
    __ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        StoreFloat(dst, src, mask);
    } else {
        Cast<T, float, F32_TO_B16_RINT>((RegTensor<T> &)src, src, mask);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(dst, (RegTensor<T> &)src, mask);
    }
}

template <typename T>
static __simd_vf__ inline void ComputeVNew(
    __ubuf__ float *workspace, __ubuf__ T *uInput, uint32_t count)
{
    constexpr uint32_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> u0, u1, ws0, ws1;
    MaskReg mask0, mask1;
    uint32_t remaining = count;
    uint32_t offset = 0;
    while (remaining > ELEMS) {
        mask0 = UpdateMask<float>(remaining);
        mask1 = UpdateMask<float>(remaining);
        LoadAsFloat<T>(u0, uInput + offset, mask0);
        LoadAsFloat<T>(u1, uInput + offset + ELEMS, mask1);
        LoadFloat(ws0, workspace + offset);
        LoadFloat(ws1, workspace + offset + ELEMS);
        Sub(ws0, u0, ws0, mask0);
        Sub(ws1, u1, ws1, mask1);
        StoreFloat(workspace + offset, ws0, mask0);
        StoreFloat(workspace + offset + ELEMS, ws1, mask1);
        offset += 2 * ELEMS;
        remaining = remaining > 2 * ELEMS ? remaining - 2 * ELEMS : 0;
    }
    if (remaining > 0) {
        mask0 = UpdateMask<float>(remaining);
        LoadAsFloat<T>(u0, uInput + offset, mask0);
        LoadFloat(ws0, workspace + offset);
        Sub(ws0, u0, ws0, mask0);
        StoreFloat(workspace + offset, ws0, mask0);
    }
}

template <typename T>
static __simd_vf__ inline void ScaleAndPackVNew(
    __ubuf__ T *output, __ubuf__ float *vNew, __ubuf__ float *rowScale,
    uint32_t scaleOffset, uint16_t rows, uint16_t cols)
{
    constexpr uint16_t C0 = 16;
    RegTensor<float> value, scale;
    MaskReg mask;
    for (uint16_t row = 0; row < rows; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, rowScale + scaleOffset + row);
        for (uint16_t col = 0; col < cols; col += C0) {
            uint32_t count = cols - col > C0 ? C0 : cols - col;
            mask = UpdateMask<float>(count);
            LoadFloat(value, vNew + static_cast<uint32_t>(row) * cols + col);
            Mul(value, value, scale, mask);
            uint32_t outputOffset = ((col / C0) * rows + row) * C0;
            StoreFromFloat<T>(output + outputOffset, value, mask);
        }
    }
}

template <typename T, bool USE_EXP2>
static __simd_vf__ inline void PrepareKGate(
    __ubuf__ float *output, __ubuf__ T *input, uint16_t count)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr float LN2 = 0.6931471805599453f;
    RegTensor<float> gate;
    MaskReg mask;
    uint32_t remaining = count;
    uint32_t offset = 0;
    while (remaining > 0) {
        mask = UpdateMask<float>(remaining);
        LoadAsFloat<T>(gate, input + offset, mask);
        if constexpr (USE_EXP2) {
            Muls(gate, gate, LN2, mask);
        }
        Exp(gate, gate, mask);
        StoreFloat(output + offset, gate, mask);
        offset += ELEMS;
        remaining = remaining > ELEMS ? remaining - ELEMS : 0;
    }
}

template <typename T>
static __simd_vf__ inline void ApplyScalarGateUpdate(
    __ubuf__ float *update, __ubuf__ T *state, float scale, uint32_t count)
{
    constexpr uint32_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> stateReg, updateReg;
    MaskReg mask;
    uint32_t remaining = count;
    uint32_t offset = 0;
    while (remaining > 0) {
        mask = UpdateMask<float>(remaining);
        LoadAsFloat<T>(stateReg, state + offset, mask);
        LoadFloat(updateReg, update + offset);
        Muls(stateReg, stateReg, scale, mask);
        Add(updateReg, stateReg, updateReg, mask);
        StoreFloat(update + offset, updateReg, mask);
        offset += ELEMS;
        remaining = remaining > ELEMS ? remaining - ELEMS : 0;
    }
}

template <typename T>
static __simd_vf__ inline void ApplyKGateUpdate(
    __ubuf__ float *update, __ubuf__ T *state, __ubuf__ float *scales,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> stateReg, updateReg, scaleReg;
    MaskReg mask;
    for (uint16_t row = 0; row < rows; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, scales + row);
        uint32_t remaining = cols;
        for (uint16_t col = 0; col < cols; col += ELEMS) {
            mask = UpdateMask<float>(remaining);
            uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadAsFloat<T>(stateReg, state + offset, mask);
            LoadFloat(updateReg, update + offset);
            Mul(stateReg, stateReg, scaleReg, mask);
            Add(updateReg, stateReg, updateReg, mask);
            StoreFloat(update + offset, updateReg, mask);
            remaining = remaining > ELEMS ? remaining - ELEMS : 0;
        }
    }
}

static __simd_vf__ inline void ApplyRowScale(
    __ubuf__ float *matrix, __ubuf__ float *scales,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> matrixReg, scaleReg;
    MaskReg mask;
    for (uint16_t row = 0; row < rows; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, scales + row);
        uint32_t remaining = cols;
        for (uint16_t col = 0; col < cols; col += ELEMS) {
            mask = UpdateMask<float>(remaining);
            uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadFloat(matrixReg, matrix + offset);
            Mul(matrixReg, matrixReg, scaleReg, mask);
            StoreFloat(matrix + offset, matrixReg, mask);
            remaining = remaining > ELEMS ? remaining - ELEMS : 0;
        }
    }
}

template <typename InputT, bool SCALAR_GATE>
static __simd_vf__ inline void ComputeStage1Tile(
    __ubuf__ float *workspace, __ubuf__ InputT *uInput,
    __ubuf__ InputT *packedOutput, __ubuf__ InputT *rowOutput,
    __ubuf__ float *rowScale, uint32_t scaleOffset,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t C0 = 16;
    RegTensor<float> uReg, wsReg, scaleReg;
    RegTensor<InputT> outReg;
    MaskReg mask;
    for (uint16_t row = 0; row < rows; ++row) {
        if constexpr (SCALAR_GATE) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, rowScale + scaleOffset + row);
        }
        for (uint16_t col = 0; col < cols; col += C0) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > C0 ? C0 : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadAsFloat<InputT>(uReg, uInput + offset, mask);
            LoadFloat(wsReg, workspace + offset);
            Sub(wsReg, uReg, wsReg, mask);
            if constexpr (SCALAR_GATE) {
                Mul(wsReg, wsReg, scaleReg, mask);
            }
            StoreFloat(workspace + offset, wsReg, mask);
            if constexpr (std::is_same<InputT, float>::value) {
                StoreFloat(rowOutput + offset, wsReg, mask);
                const uint32_t packedOffset = ((col / C0) * rows + row) * C0;
                StoreFloat(packedOutput + packedOffset, wsReg, mask);
            } else {
                Cast<InputT, float, F32_TO_B16_RINT>(outReg, wsReg, mask);
                StoreAlign<InputT, StoreDist::DIST_PACK_B32>(rowOutput + offset, outReg, mask);
                const uint32_t packedOffset = ((col / C0) * rows + row) * C0;
                StoreAlign<InputT, StoreDist::DIST_PACK_B32>(packedOutput + packedOffset, outReg, mask);
            }
        }
    }
}

template <typename InputT, typename GateT, typename StateT, bool STATE_FLOAT, bool SEPARATE_STATE,
          bool K_GATE, bool USE_EXP2>
static __simd_vf__ inline void ComputeStage3Tile(
    __ubuf__ float *update, __ubuf__ float *stateFloat,
    __ubuf__ StateT *stateNative, __ubuf__ InputT *stateInput,
    __ubuf__ GateT *gateRaw, float scalarGate,
    __ubuf__ StateT *stateOutput, __ubuf__ InputT *hOutput,
    uint16_t rows, uint16_t cols, bool storeState, bool finalChunk)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> updateReg, stateReg, gateReg, gateScalarReg;
    RegTensor<StateT> stateOutReg;
    RegTensor<InputT> hOutReg;
    MaskReg mask;
    for (uint16_t row = 0; row < rows; ++row) {
        if constexpr (K_GATE) {
            MaskReg gateScalarMask = UpdateMask<float>(1);
            MaskReg gateMask = UpdateMask<float>(ELEMS);
            LoadAsFloat<GateT>(gateScalarReg, gateRaw + row, gateScalarMask);
            Duplicate(gateReg, gateScalarReg, gateMask);
            if constexpr (USE_EXP2) {
                Muls(gateReg, gateReg, 0.6931471805599453f, gateMask);
            }
            Exp(gateReg, gateReg, gateMask);
        }
        for (uint16_t col = 0; col < cols; col += ELEMS) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > ELEMS ? ELEMS : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadFloat(updateReg, update + offset);
            if constexpr (STATE_FLOAT) {
                LoadFloat(stateReg, stateFloat + offset);
            } else if constexpr (SEPARATE_STATE) {
                LoadAsFloat<StateT>(stateReg, stateNative + offset, mask);
            } else {
                LoadAsFloat<InputT>(stateReg, stateInput + offset, mask);
            }
            if constexpr (K_GATE) {
                Mul(stateReg, stateReg, gateReg, mask);
            } else {
                Muls(stateReg, stateReg, scalarGate, mask);
            }
            Add(updateReg, stateReg, updateReg, mask);
            if (storeState) {
                if constexpr (std::is_same<StateT, float>::value) {
                    StoreFloat(stateOutput + offset, updateReg, mask);
                } else {
                    Cast<StateT, float, F32_TO_B16_RINT>(stateOutReg, updateReg, mask);
                    StoreAlign<StateT, StoreDist::DIST_PACK_B32>(stateOutput + offset, stateOutReg, mask);
                }
            }
            if (!finalChunk || !storeState) {
                if constexpr (std::is_same<InputT, float>::value) {
                    StoreFloat(hOutput + offset, updateReg, mask);
                } else {
                    Cast<InputT, float, F32_TO_B16_RINT>(hOutReg, updateReg, mask);
                    StoreAlign<InputT, StoreDist::DIST_PACK_B32>(hOutput + offset, hOutReg, mask);
                }
            }
        }
    }
}

} // namespace detail

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
class ChunkGatedDeltaRuleFwdHVector {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using Scheduler = Catlass::Gemm::Block::BlockSchedulerGdnFwdHVec;
    using Offsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    static constexpr bool kScalarGate = GateMode == GDN_FWD_H_GATE_G;
    static constexpr bool kKGate = GateMode == GDN_FWD_H_GATE_GK;
    static constexpr bool kUseExp2 = ExpMode == GDN_FWD_H_EXP_2;
    static constexpr bool kSeparateBf16State =
        std::is_same<StateT, bfloat16_t>::value && !std::is_same<StateT, InputT>::value;
    static constexpr uint32_t PONG_EVENT_BASE = 4;
    static constexpr float LN2 = 0.6931471805599453f;

    static_assert(kScalarGate || kKGate, "unsupported FwdH gate mode");
    static_assert(ExpMode == GDN_FWD_H_EXP_E || ExpMode == GDN_FWD_H_EXP_2,
                  "unsupported FwdH exponent mode");
    static_assert(std::is_same<WorkspaceT, float>::value,
                  "A5 FwdH workspace must be FP32");

    __aicore__ inline ChunkGatedDeltaRuleFwdHVector() = default;

    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR vNew,
        GM_ADDR finalState, GM_ADDR user, GM_ADDR tiling)
    {
        auto data = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *>(tiling);
        batch_ = data->batch;
        seqlen_ = data->seqlen;
        kNumHead_ = data->kNumHead;
        vNumHead_ = data->vNumHead;
        kHeadDim_ = data->kHeadDim;
        vHeadDim_ = data->vHeadDim;
        chunkSize_ = data->chunkSize;
        useInitialState_ = data->useInitialState;
        storeFinalState_ = data->storeFinalState;
        isVariedLen_ = data->isVariedLen;
        shapeBatch_ = data->shapeBatch;
        tokenBatch_ = data->tokenBatch;
        vWorkspaceOffset_ = data->vWorkspaceOffset;
        vUpdateWorkspaceOffset_ = data->vUpdateWorkspaceOffset;
        hWorkspaceOffset_ = data->hWorkspaceOffset;
        numSeqWorkspaceOffset_ = data->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset_ = data->numChunksWorkspaceOffset;

        GM_ADDR effectiveFinalState = finalState;
        if constexpr (!std::is_same<StateT, InputT>::value) {
            if (!storeFinalState_) {
                constexpr uint64_t GM_ALIGN = 512;
                uint64_t numChunksBytes =
                    (static_cast<uint64_t>(tokenBatch_) + 1) * sizeof(int64_t);
                uint64_t hiddenStateOffset = numChunksWorkspaceOffset_ +
                    (numChunksBytes + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
                effectiveFinalState = user + hiddenStateOffset;
                storeFinalState_ = true;
            }
        }

        gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(k));
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(w));
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(u));
        gmG_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(g));
        gmGk_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(gk));
        gmInitialState_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(initialState));
        gmH_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(h));
        gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(vNew));
        gmFinalState_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT *>(effectiveFinalState));
        gmVWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ WorkspaceT *>(user + vWorkspaceOffset_));
        gmVUpdateWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ InputT *>(user + vUpdateWorkspaceOffset_));
        gmHWorkspace_.SetGlobalBuffer(
            reinterpret_cast<__gm__ WorkspaceT *>(user + hWorkspaceOffset_));

        scheduler_.Init(cuSeqlens, chunkIndices, tiling, user);
        BindLocalTensors();
    }

    __aicore__ inline void Process()
    {
        AscendC::SyncAll<false>();
        InitializeH();
        AscendC::SyncAll<false>();

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec2Done[0]);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec2Done[1]);
        PresetEvents();

        const bool useBoundedMmad = isVariedLen_ || (seqlen_ % chunkSize_ != 0);
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        while (scheduler_.isRunning) {
            scheduler_.InitTasks();
            if (!scheduler_.isRunning) {
                break;
            }
            const uint32_t windowId = scheduler_.GetWindowId();

            // Stage1: finish v_new and publish the zN operand consumed by Stage2.
            for (uint32_t i = 0; i < scheduler_.GetHeadsInRound(); ++i) {
                const auto &headTask = scheduler_.GetHeadTask(i);
                if (scheduler_.HeadTaskIsDone(headTask)) {
                    continue;
                }
                const Offsets &offsets = scheduler_.GetCurTaskOffsets(headTask);
                const bool ownsHead = i % subBlockNum == subBlockIdx;
                const uint32_t localSlot = i / subBlockNum;
                const bool isPing = localSlot == 0;
                if (!ownsHead) {
                    Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube1Done[windowId]);
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec1Done[windowId]);
                    continue;
                }
                const bool tailVectorPath = offsets.blockTokens < 16 && !useBoundedMmad;
                if (tailVectorPath) {
                    Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube1Done[windowId]);
                    ComputeTailVWorkspace(offsets, EVENT_ID3 + (isPing ? 0 : PONG_EVENT_BASE));
                }
                Stage1(offsets, windowId, isPing, tailVectorPath,
                              event0FromMte3_[localSlot]);
                event0FromMte3_[localSlot] = false;
            }

            // Stage3: consume the complete Stage2 round and publish the next state.
            for (uint32_t i = 0; i < scheduler_.GetHeadsInRound(); ++i) {
                const auto &headTask = scheduler_.GetHeadTask(i);
                if (scheduler_.HeadTaskIsDone(headTask)) {
                    continue;
                }
                const Offsets &offsets = scheduler_.GetCurTaskOffsets(headTask);
                const bool ownsHead = i % subBlockNum == subBlockIdx;
                const uint32_t localSlot = i / subBlockNum;
                const bool isPing = localSlot == 0;
                if (scheduler_.NeedProcessStage2(headTask)) {
                    if (!ownsHead) {
                        Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube2Done[windowId]);
                        continue;
                    }
                    const bool tailVectorPath = offsets.blockTokens < 16 && !useBoundedMmad;
                    if (tailVectorPath) {
                        Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube2Done[windowId]);
                        ComputeTailHWorkspace(offsets, EVENT_ID3 + (isPing ? 0 : PONG_EVENT_BASE));
                    }
                    if constexpr (std::is_same<StateT, float>::value) {
                        if (storeFinalState_) {
                            event0FromMte3_[localSlot] = true;
                            event2FromMte3_[localSlot] = !offsets.isFinalState;
                        }
                    }
                    Stage3(offsets, windowId, isPing, tailVectorPath);
                } else {
                    Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube2Done[windowId]);
                }
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec2Done[windowId]);
        }

        DrainEvents();
    }

private:
    __aicore__ inline void BindLocalTensors()
    {
        wsUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(32 * 1024);
        wsUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(96 * 1024);
        ioUb_[0] = resource_.ubBuf.template GetBufferByByte<InputT>(64 * 1024);
        ioUb_[1] = resource_.ubBuf.template GetBufferByByte<InputT>(128 * 1024);
        stage1OutputUb_[0] = resource_.ubBuf.template GetBufferByByte<InputT>(16 * 1024);
        stage1OutputUb_[1] = resource_.ubBuf.template GetBufferByByte<InputT>(24 * 1024);
        stage1PackedUb_[0] = resource_.ubBuf.template GetBufferByByte<InputT>(40 * 1024);
        stage1PackedUb_[1] = resource_.ubBuf.template GetBufferByByte<InputT>(48 * 1024);
        gUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(160 * 1024);
        gUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(161 * 1024);
        gLastUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(162 * 1024);
        gLastUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(163 * 1024);
        gInputUb_[0] = resource_.ubBuf.template GetBufferByByte<GateT>(164 * 1024);
        gInputUb_[1] = resource_.ubBuf.template GetBufferByByte<GateT>(165 * 1024);

        calcUb_ = resource_.ubBuf.template GetBufferByByte<float>(0);
        hUb_ = resource_.ubBuf.template GetBufferByByte<InputT>(160 * 1024);
        // Separate BF16 rolling state is only consumed in Stage3.  Reuse
        // the Stage1 packed banks after Stage2 has consumed them; keeping
        // these banks distinct from ioUb avoids update/state overwrite while
        // the two MTE2 transfers are in flight.
        stateUb_[0] = resource_.ubBuf.template GetBufferByByte<StateT>(
            kSeparateBf16State ? 40 * 1024 : 160 * 1024);
        stateUb_[1] = resource_.ubBuf.template GetBufferByByte<StateT>(
            kSeparateBf16State ? 48 * 1024 : 160 * 1024);
        gkLastUb_[0] = resource_.ubBuf.template GetBufferByByte<float>(170 * 1024);
        gkLastUb_[1] = resource_.ubBuf.template GetBufferByByte<float>(171 * 1024);
        gkInputUb_[0] = resource_.ubBuf.template GetBufferByByte<GateT>(172 * 1024);
        gkInputUb_[1] = resource_.ubBuf.template GetBufferByByte<GateT>(173 * 1024);
        stage3ScalarUb_ = resource_.ubBuf.template GetBufferByByte<float>(176 * 1024);
    }

    template <typename T>
    __aicore__ inline void CopyGmToUb(
        AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
        uint32_t rows, uint32_t cols, uint32_t srcStride)
    {
        if (cols == srcStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(T)),
            static_cast<uint32_t>((srcStride - cols) * sizeof(T)), 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, params, pad);
    }

    template <typename T>
    __aicore__ inline void CopyUbToGm(
        AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T> src,
        uint32_t rows, uint32_t cols, uint32_t dstStride)
    {
        if (cols == dstStride) {
            AscendC::DataCopy(dst, src, rows * cols);
            return;
        }
        AscendC::DataCopyExtParams params{
            static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(T)),
            0, static_cast<uint32_t>((dstStride - cols) * sizeof(T)), 0};
        AscendC::DataCopyPad(dst, src, params);
    }

    __aicore__ inline void InitializeH();
    __aicore__ inline void PresetEvents();
    __aicore__ inline void DrainEvents();
    __aicore__ inline void Stage1(
        const Offsets &offsets, uint32_t windowId, bool isPing,
        bool cubeAlreadyWaited, bool waitWsFromMte3);
    __aicore__ inline void LoadStage1Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex, bool waitWsFromMte3);
    __aicore__ inline void StoreStage1Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex);
    __aicore__ inline void Stage3(
        const Offsets &offsets, uint32_t windowId, bool isPing,
        bool cubeAlreadyWaited);
    __aicore__ inline void Stage3Legacy(
        const Offsets &offsets, uint32_t windowId, bool isPing,
        bool cubeAlreadyWaited);
    __aicore__ inline void ComputeTailVWorkspace(const Offsets &offsets, uint32_t eventId);
    __aicore__ inline void ComputeTailHWorkspace(const Offsets &offsets, uint32_t eventId);

    uint32_t batch_{0};
    uint32_t seqlen_{0};
    uint32_t kNumHead_{0};
    uint32_t vNumHead_{0};
    uint32_t kHeadDim_{0};
    uint32_t vHeadDim_{0};
    uint32_t chunkSize_{0};
    uint32_t isVariedLen_{0};
    uint32_t shapeBatch_{0};
    uint32_t tokenBatch_{0};
    uint32_t vWorkspaceOffset_{0};
    uint32_t vUpdateWorkspaceOffset_{0};
    uint32_t hWorkspaceOffset_{0};
    uint32_t numSeqWorkspaceOffset_{0};
    uint64_t numChunksWorkspaceOffset_{0};
    bool useInitialState_{false};
    bool storeFinalState_{false};
    bool event0FromMte3_[LOCAL_PING_PONG_STAGES] = {false, false};
    bool event2FromMte3_[LOCAL_PING_PONG_STAGES] = {false, false};

    AscendC::GlobalTensor<InputT> gmK_;
    AscendC::GlobalTensor<InputT> gmW_;
    AscendC::GlobalTensor<InputT> gmU_;
    AscendC::GlobalTensor<GateT> gmG_;
    AscendC::GlobalTensor<GateT> gmGk_;
    AscendC::GlobalTensor<StateT> gmInitialState_;
    AscendC::GlobalTensor<InputT> gmH_;
    AscendC::GlobalTensor<InputT> gmV_;
    AscendC::GlobalTensor<StateT> gmFinalState_;
    AscendC::GlobalTensor<WorkspaceT> gmVWorkspace_;
    AscendC::GlobalTensor<InputT> gmVUpdateWorkspace_;
    AscendC::GlobalTensor<WorkspaceT> gmHWorkspace_;

    AscendC::LocalTensor<float> wsUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<InputT> ioUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<InputT> stage1OutputUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<InputT> stage1PackedUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<float> gUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<float> gLastUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<GateT> gInputUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<float> calcUb_;
    AscendC::LocalTensor<InputT> hUb_;
    AscendC::LocalTensor<StateT> stateUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<float> gkLastUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<GateT> gkInputUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<float> stage3ScalarUb_;

    Scheduler scheduler_;
    Catlass::Arch::Resource<ArchTag> resource_;
};

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::InitializeH()
{
    const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    const uint32_t subBlockNum = AscendC::GetSubBlockNum();
    const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
    const uint32_t coreNum = AscendC::GetBlockNum();
    const uint32_t sequenceCount = isVariedLen_ ? scheduler_.tokenBatch : shapeBatch_;
    const uint32_t taskCount = sequenceCount * scheduler_.headWindowNum;
    const uint32_t hRowsPerTile = (32 * 1024) / (vHeadDim_ * sizeof(InputT));
    const uint32_t stateRowsPerTile = (64 * 1024) / (vHeadDim_ * sizeof(StateT));
    const uint32_t rowsPerTile = Min(hRowsPerTile, stateRowsPerTile);
    const uint32_t totalChunks = isVariedLen_ ? scheduler_.totalChunks :
        ((seqlen_ + chunkSize_ - 1) / chunkSize_);
    const uint32_t stateBlockSize = kHeadDim_ * vHeadDim_;
    AscendC::LocalTensor<StateT> stateInitUb[2] = {
        resource_.ubBuf.template GetBufferByByte<StateT>(0),
        resource_.ubBuf.template GetBufferByByte<StateT>(96 * 1024)};
    AscendC::LocalTensor<InputT> hInitUb[2] = {
        resource_.ubBuf.template GetBufferByByte<InputT>(64 * 1024),
        resource_.ubBuf.template GetBufferByByte<InputT>(160 * 1024)};

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t taskIdx = coreIdx; taskIdx < taskCount; taskIdx += coreNum) {
        const uint32_t batchIdx = taskIdx / scheduler_.headWindowNum;
        const uint32_t headBase = scheduler_.GetCoreHeadBase();
        for (uint32_t headOffset = 0; headOffset < scheduler_.GetHeadsPerCore(); ++headOffset) {
            const uint32_t vHeadIdx = headBase + headOffset;
            if (vHeadIdx >= vNumHead_ || headOffset % subBlockNum != subBlockIdx) {
                continue;
            }
            uint32_t buffer = ((headOffset / subBlockNum) & 1U) == 0 ? 1 : 0;
            const uint32_t chunkOffset =
                isVariedLen_ ? scheduler_.GetVarlenChunkOffset(batchIdx) : 0;
            const uint32_t shapeBatchIdx = isVariedLen_ ? 0 : batchIdx;
            const uint32_t hBaseOffset =
                (shapeBatchIdx * vNumHead_ * totalChunks + vHeadIdx * totalChunks + chunkOffset) *
                stateBlockSize;
            const uint32_t initialBaseOffset =
                (batchIdx * vNumHead_ + vHeadIdx) * stateBlockSize;
            for (uint32_t rowOffset = 0; rowOffset < kHeadDim_; rowOffset += rowsPerTile) {
                const uint32_t rowsThisTile = Min(rowsPerTile, kHeadDim_ - rowOffset);
                const uint32_t elements = rowsThisTile * vHeadDim_;
                const uint32_t hOffset = hBaseOffset + rowOffset * vHeadDim_;
                const uint32_t eventId = buffer ? EVENT_ID1 : EVENT_ID0;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                if (useInitialState_) {
                    const uint32_t initialOffset = initialBaseOffset + rowOffset * vHeadDim_;
                    AscendC::DataCopy(stateInitUb[buffer], gmInitialState_[initialOffset], elements);
                    if constexpr (std::is_same<StateT, InputT>::value) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
                        AscendC::DataCopy(gmH_[hOffset], stateInitUb[buffer], elements);
                    } else {
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                        AscendC::Cast(hInitUb[buffer], stateInitUb[buffer],
                                      AscendC::RoundMode::CAST_RINT, elements);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                        AscendC::DataCopy(gmH_[hOffset], hInitUb[buffer], elements);
                    }
                } else {
                    AscendC::Duplicate(hInitUb[buffer], static_cast<InputT>(0), elements);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                    AscendC::DataCopy(gmH_[hOffset], hInitUb[buffer], elements);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                buffer ^= 1U;
            }
        }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::PresetEvents()
{
    if constexpr (std::is_same<StateT, float>::value) {
        if (storeFinalState_) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + PONG_EVENT_BASE);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + PONG_EVENT_BASE);
            event2FromMte3_[0] = false;
            event2FromMte3_[1] = false;
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + PONG_EVENT_BASE);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + PONG_EVENT_BASE);
            event2FromMte3_[0] = true;
            event2FromMte3_[1] = true;
        }
    } else {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + PONG_EVENT_BASE);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + PONG_EVENT_BASE);
        event2FromMte3_[0] = true;
        event2FromMte3_[1] = true;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + PONG_EVENT_BASE);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + PONG_EVENT_BASE);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + PONG_EVENT_BASE);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + PONG_EVENT_BASE);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::DrainEvents()
{
    for (uint32_t slot = 0; slot < LOCAL_PING_PONG_STAGES; ++slot) {
        const uint32_t base = slot * PONG_EVENT_BASE;
        if (event0FromMte3_[slot]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + base);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + base);
        }
        if (event2FromMte3_[slot]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + base);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + base);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + base);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + base);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + base);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + base);
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::LoadStage1Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex, bool waitWsFromMte3)
{
    const uint32_t headSlot = tileIndex & 1U;
    const uint32_t eventBase = headSlot * PONG_EVENT_BASE;
    const uint32_t cols = offsets.vBlockDim;
    const uint32_t rowOffset = rowStart * vHeadDim_;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + eventBase);
    CopyGmToUb(ioUb_[headSlot], gmU_[offsets.uvOffset + rowOffset], rows, cols, vHeadDim_);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + eventBase);
    if (rowStart == 0 && waitWsFromMte3) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + eventBase);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
    }
    CopyGmToUb(wsUb_[headSlot], gmVWorkspace_[offsets.vWorkOffset + rowStart * cols],
               rows, cols, cols);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + eventBase);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::StoreStage1Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex)
{
    constexpr uint32_t C0 = 16;
    const uint32_t headSlot = tileIndex & 1U;
    const uint32_t eventBase = headSlot * PONG_EVENT_BASE;
    const uint32_t cols = offsets.vBlockDim;
    const uint32_t paddedRows = (offsets.blockTokens + NZ_BLOCK_SIZE - 1) /
        NZ_BLOCK_SIZE * NZ_BLOCK_SIZE;
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + eventBase);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + eventBase);
    AscendC::DataCopyParams znParams{
        static_cast<uint16_t>(cols / C0), static_cast<uint16_t>(rows), 0,
        static_cast<uint16_t>(paddedRows - rows)};
    AscendC::DataCopy(gmVUpdateWorkspace_[offsets.vWorkOffset + rowStart * C0],
                      stage1PackedUb_[headSlot], znParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + eventBase);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1 + eventBase);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + eventBase);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1 + eventBase);
    CopyUbToGm(gmV_[offsets.uvOffset + rowStart * vHeadDim_], stage1OutputUb_[headSlot],
               rows, cols, vHeadDim_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + eventBase);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::Stage1(
        const Offsets &offsets, uint32_t windowId, bool isPing,
        bool cubeAlreadyWaited, bool waitWsFromMte3)
{
    constexpr uint32_t ROW_TILE = 16;
    const uint32_t rows = offsets.blockTokens;
    const uint32_t cols = offsets.vBlockDim;
    AscendC::ResetMask();

    const uint32_t headSlot = isPing ? 0 : 1;
    const uint32_t headEventBase = headSlot * PONG_EVENT_BASE;
    const uint32_t slot = headSlot;
    const uint32_t eventBase = headEventBase;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + headEventBase);
    if constexpr (kScalarGate) {
        if (rows == 1) {
            AscendC::Duplicate(gUb_[slot], 1.0f, 1);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            if constexpr (std::is_same<GateT, float>::value) {
                AscendC::DataCopyParams params{
                    1, static_cast<uint16_t>(rows * sizeof(GateT)), 0, 0};
                AscendC::DataCopyPadParams pad{false, 0, 0, 0};
                AscendC::DataCopyPad(gUb_[slot], gmG_[offsets.gOffset], params, pad);
            } else {
                AscendC::DataCopyParams params{
                    1, static_cast<uint16_t>(rows * sizeof(GateT)), 0, 0};
                AscendC::DataCopyPadParams pad{false, 0, 0, 0};
                AscendC::DataCopyPad(gInputUb_[slot], gmG_[offsets.gOffset], params, pad);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + eventBase);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + eventBase);
            if constexpr (!std::is_same<GateT, float>::value) {
                AscendC::Cast(gUb_[slot], gInputUb_[slot],
                              AscendC::RoundMode::CAST_NONE, rows);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + eventBase);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + eventBase);
            const float last = gUb_[slot].GetValue(rows - 1);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
            AscendC::Duplicate(gLastUb_[slot], last, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(gUb_[slot], gLastUb_[slot], gUb_[slot], rows);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (kUseExp2) {
                AscendC::Muls(gUb_[slot], gUb_[slot], LN2, rows);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Exp(gUb_[slot], gUb_[slot], rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    if (!cubeAlreadyWaited) {
        Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube1Done[windowId]);
    }
    const uint32_t tileCount = (rows + ROW_TILE - 1) / ROW_TILE;
    LoadStage1Tile(offsets, 0, Min(ROW_TILE, rows), headSlot, waitWsFromMte3);
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t rowStart = tile * ROW_TILE;
        const uint32_t rowsThisTile = Min(ROW_TILE, rows - rowStart);
        const uint32_t cur = (headSlot ^ (tile & 1U)) & 1U;
        // Pipeline order is MTE2(next) -> VF(current) -> MTE3(current).
        if (tile + 1 < tileCount) {
            const uint32_t nextRow = (tile + 1) * ROW_TILE;
            LoadStage1Tile(offsets, nextRow, Min(ROW_TILE, rows - nextRow),
                            headSlot ^ ((tile + 1) & 1U), false);
        }
        const uint32_t eventBase = cur * PONG_EVENT_BASE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + eventBase);
        AscendC::VF_CALL<detail::ComputeStage1Tile<InputT, kScalarGate>>(
            reinterpret_cast<__ubuf__ float *>(wsUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ InputT *>(ioUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ InputT *>(stage1PackedUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ InputT *>(stage1OutputUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gUb_[headSlot].GetPhyAddr()),
            rowStart, static_cast<uint16_t>(rowsThisTile), static_cast<uint16_t>(cols));
        StoreStage1Tile(offsets, rowStart, rowsThisTile, cur);
        if (tile + 1 == tileCount) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec1Done[windowId]);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + headEventBase);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::Stage3(
        const Offsets &offsets, uint32_t windowId, bool isPing,
        bool cubeAlreadyWaited)
{
    constexpr uint32_t ROW_TILE = 16;
    const uint32_t headSlot = isPing ? 0 : 1;
    const uint32_t eventBase = headSlot * PONG_EVENT_BASE;
    const uint32_t rows = kHeadDim_;
    const uint32_t cols = offsets.vBlockDim;
    AscendC::ResetMask();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + eventBase);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
    if (!cubeAlreadyWaited) {
        Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube2Done[windowId]);
    }

    float scalarGate = 1.0f;
    if constexpr (kScalarGate) {
        GateT gateValue = gmG_[offsets.gOffset + offsets.blockTokens - 1].GetValue(0);
        if constexpr (std::is_same<GateT, float>::value) {
            scalarGate = gateValue;
        } else if constexpr (std::is_same<GateT, half>::value) {
            scalarGate = static_cast<float>(gateValue);
        } else {
            scalarGate = AscendC::ToFloat(gateValue);
        }
        stage3ScalarUb_.SetValue(0, scalarGate);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
        if constexpr (kUseExp2) {
            AscendC::Muls(stage3ScalarUb_, stage3ScalarUb_, LN2, 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Exp(stage3ScalarUb_, stage3ScalarUb_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + eventBase);
        scalarGate = stage3ScalarUb_.GetValue(0);
    }

    const uint32_t tileCount = (rows + ROW_TILE - 1) / ROW_TILE;
    // Stage1 has just published both update slots with V_MTE2.  State rows
    // come from InitializeH for the first chunk and from the previous
    // chunk's final-state stores afterwards.
    bool slotEvent0FromMte3[LOCAL_PING_PONG_STAGES] = {false, false};
    bool slotEvent2FromMte3[LOCAL_PING_PONG_STAGES] = {
        !offsets.isInitialState, !offsets.isInitialState};

    // Prime tile zero before entering the overlapped MTE2/V/MTE3 loop.
    const uint32_t firstBase = headSlot * PONG_EVENT_BASE;
    const uint32_t firstRows = Min(ROW_TILE, rows);
    if (slotEvent0FromMte3[headSlot]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + firstBase);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + firstBase);
    }
    CopyGmToUb(ioUb_[headSlot].template ReinterpretCast<float>(),
               gmHWorkspace_[offsets.hWorkOffset], firstRows, cols, cols);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + firstBase);

    if (slotEvent2FromMte3[headSlot] && !offsets.isInitialState) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + firstBase);
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + firstBase);
    }
    if constexpr (std::is_same<StateT, float>::value) {
        auto stateSource = offsets.isInitialState ?
            gmInitialState_[offsets.initialStateOffset] :
            gmFinalState_[offsets.finalStateOffset];
        CopyGmToUb(wsUb_[headSlot], stateSource, firstRows, cols, vHeadDim_);
    } else if constexpr (kSeparateBf16State) {
        auto stateSource = offsets.isInitialState ?
            gmInitialState_[offsets.initialStateOffset] :
            gmFinalState_[offsets.finalStateOffset];
        CopyGmToUb(stateUb_[headSlot], stateSource, firstRows, cols, vHeadDim_);
    } else {
        CopyGmToUb(stage1PackedUb_[headSlot], gmH_[offsets.hSrcOffset],
                   firstRows, cols, vHeadDim_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + firstBase);
    if constexpr (kKGate) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + firstBase);
        CopyGmToUb(gkInputUb_[headSlot],
                   gmGk_[offsets.gkOffset + (offsets.blockTokens - 1) * kHeadDim_],
                   firstRows, 1, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + firstBase);
    }

    constexpr bool kStateFloat = std::is_same<StateT, float>::value;
    constexpr bool kSeparateState = kSeparateBf16State;
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t rowStart = tile * ROW_TILE;
        const uint32_t rowsThisTile = Min(ROW_TILE, rows - rowStart);
        const uint32_t cur = (headSlot ^ (tile & 1U)) & 1U;
        // Pipeline order is MTE2(next) -> VF(current) -> MTE3(current).
        if (tile + 1 < tileCount) {
            const uint32_t nextRow = (tile + 1) * ROW_TILE;
            const uint32_t next = cur ^ 1U;
            const uint32_t nextBase = next * PONG_EVENT_BASE;
            const uint32_t nextRows = Min(ROW_TILE, rows - nextRow);
            if (slotEvent0FromMte3[next]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + nextBase);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + nextBase);
            }
            CopyGmToUb(ioUb_[next].template ReinterpretCast<float>(),
                       gmHWorkspace_[offsets.hWorkOffset + nextRow * cols],
                       nextRows, cols, cols);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + nextBase);
            if (slotEvent2FromMte3[next] && !offsets.isInitialState) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + nextBase);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + nextBase);
            }
            if constexpr (std::is_same<StateT, float>::value) {
                auto stateSource = offsets.isInitialState ?
                    gmInitialState_[offsets.initialStateOffset + nextRow * vHeadDim_] :
                    gmFinalState_[offsets.finalStateOffset + nextRow * vHeadDim_];
                CopyGmToUb(wsUb_[next],
                           stateSource, nextRows, cols, vHeadDim_);
            } else if constexpr (kSeparateBf16State) {
                auto stateSource = offsets.isInitialState ?
                    gmInitialState_[offsets.initialStateOffset + nextRow * vHeadDim_] :
                    gmFinalState_[offsets.finalStateOffset + nextRow * vHeadDim_];
                CopyGmToUb(stateUb_[next], stateSource, nextRows, cols, vHeadDim_);
            } else {
                CopyGmToUb(stage1PackedUb_[next],
                           gmH_[offsets.hSrcOffset + nextRow * vHeadDim_],
                           nextRows, cols, vHeadDim_);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + nextBase);
            if constexpr (kKGate) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + nextBase);
                CopyGmToUb(gkInputUb_[next],
                           gmGk_[offsets.gkOffset + (offsets.blockTokens - 1) * kHeadDim_ + nextRow],
                           nextRows, 1, 1);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + nextBase);
            }
        }

        const uint32_t curBase = cur * PONG_EVENT_BASE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + curBase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + curBase);
        if constexpr (kKGate) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1 + curBase);
        }
        const bool storeState = storeFinalState_;
        const bool finalChunk = offsets.isFinalState;
        AscendC::VF_CALL<detail::ComputeStage3Tile<
            InputT, GateT, StateT, kStateFloat, kSeparateState, kKGate, kUseExp2>>(
            reinterpret_cast<__ubuf__ float *>(ioUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(wsUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ StateT *>(stateUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ InputT *>(stage1PackedUb_[cur].GetPhyAddr()),
            reinterpret_cast<__ubuf__ GateT *>(gkInputUb_[cur].GetPhyAddr()),
            scalarGate,
            reinterpret_cast<__ubuf__ StateT *>(
                (kStateFloat ? wsUb_[cur].GetPhyAddr() :
                 (kSeparateState ? stateUb_[cur].GetPhyAddr() : stage1PackedUb_[cur].GetPhyAddr()))),
            reinterpret_cast<__ubuf__ InputT *>(stage1OutputUb_[cur].GetPhyAddr()),
            static_cast<uint16_t>(rowsThisTile), static_cast<uint16_t>(cols),
            storeState, finalChunk);
        AscendC::PipeBarrier<PIPE_V>();
        if (storeState) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + curBase);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + curBase);
            if constexpr (std::is_same<StateT, float>::value) {
                CopyUbToGm(gmFinalState_[offsets.finalStateOffset + rowStart * vHeadDim_],
                           wsUb_[cur],
                           rowsThisTile, cols, vHeadDim_);
            } else if constexpr (kSeparateBf16State) {
                CopyUbToGm(gmFinalState_[offsets.finalStateOffset + rowStart * vHeadDim_],
                           stateUb_[cur], rowsThisTile, cols, vHeadDim_);
            } else if (finalChunk) {
                CopyUbToGm(gmFinalState_[offsets.finalStateOffset + rowStart * vHeadDim_],
                           stage1PackedUb_[cur], rowsThisTile, cols, vHeadDim_);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + curBase);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + curBase);
        }
        if (!finalChunk || !storeState) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + curBase);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + curBase);
            CopyUbToGm(gmH_[offsets.hDstOffset + rowStart * vHeadDim_],
                       stage1OutputUb_[cur], rowsThisTile, cols, vHeadDim_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + curBase);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + curBase);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + curBase);
        slotEvent0FromMte3[cur] = storeState;
        slotEvent2FromMte3[cur] = !storeState || !finalChunk;
    }
    for (uint32_t slot = 0; slot < LOCAL_PING_PONG_STAGES; ++slot) {
        event0FromMte3_[slot] = slotEvent0FromMte3[slot];
        event2FromMte3_[slot] = slotEvent2FromMte3[slot];
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::Stage3Legacy(
        const Offsets &offsets, uint32_t windowId, bool isPing,
        bool cubeAlreadyWaited)
{
    constexpr uint32_t ROW_TILE = 16;
    const uint32_t slot = isPing ? 0 : 1;
    const uint32_t eventBase = slot * PONG_EVENT_BASE;
    const uint32_t rows = kHeadDim_;
    const uint32_t cols = offsets.vBlockDim;
    AscendC::ResetMask();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + eventBase);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);

    const bool useSeparateState = storeFinalState_ && kSeparateBf16State;
    const bool useFp32State = storeFinalState_ && std::is_same<StateT, float>::value &&
        (!offsets.isInitialState || useInitialState_);
    const bool useBf16State = useSeparateState &&
        (!offsets.isInitialState || useInitialState_);
    if (useSeparateState) {
        // BUF2 is shared with Stage1. EVENT1 is the ready/free hand-off.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + eventBase);
    }

    float scalarGate = 1.0f;
    if constexpr (kScalarGate) {
        const GateT gateValue = gmG_[offsets.gOffset + offsets.blockTokens - 1].GetValue(0);
        float gateFp32 = 0.0f;
        if constexpr (std::is_same<GateT, float>::value) {
            gateFp32 = gateValue;
        } else if constexpr (std::is_same<GateT, half>::value) {
            gateFp32 = static_cast<float>(gateValue);
        } else {
            gateFp32 = AscendC::ToFloat(gateValue);
        }
        stage3ScalarUb_.SetValue(0, gateFp32);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
        if constexpr (kUseExp2) {
            AscendC::Muls(stage3ScalarUb_, stage3ScalarUb_, LN2, 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Exp(stage3ScalarUb_, stage3ScalarUb_, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3 + eventBase);
        scalarGate = stage3ScalarUb_.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID3 + eventBase);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1 + eventBase);
    }

    if (!cubeAlreadyWaited) {
        Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube2Done[windowId]);
    }
    bool waitHFromV = storeFinalState_ && offsets.isInitialState &&
        std::is_same<StateT, float>::value;
    bool waitUpdateFromMte3 = false;
    const uint32_t updateReadyEvent = EVENT_ID3 + eventBase;
    for (uint32_t rowStart = 0; rowStart < rows; rowStart += ROW_TILE) {
        const uint32_t rowsThisTile = Min(ROW_TILE, rows - rowStart);
        auto hOutput = gmH_[offsets.hDstOffset + rowStart * vHeadDim_];
        auto hInput = gmH_[offsets.hSrcOffset + rowStart * vHeadDim_];
        auto hUpdate = gmHWorkspace_[offsets.hWorkOffset + rowStart * cols];
        auto finalState = gmFinalState_[offsets.finalStateOffset + rowStart * vHeadDim_];

        if (waitHFromV) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + eventBase);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
        }
        if constexpr (std::is_same<StateT, float>::value) {
            if (useFp32State) {
                if (offsets.isInitialState) {
                    CopyGmToUb(calcUb_, gmInitialState_[offsets.initialStateOffset +
                                   rowStart * vHeadDim_], rowsThisTile, cols, vHeadDim_);
                } else {
                    CopyGmToUb(calcUb_, finalState, rowsThisTile, cols, vHeadDim_);
                }
            } else {
                CopyGmToUb(hUb_, hInput, rowsThisTile, cols, vHeadDim_);
            }
        } else {
            if (useBf16State) {
                if (offsets.isInitialState) {
                    CopyGmToUb(stateUb_[slot], gmInitialState_[offsets.initialStateOffset +
                                   rowStart * vHeadDim_], rowsThisTile, cols, vHeadDim_);
                } else {
                    CopyGmToUb(stateUb_[slot], finalState, rowsThisTile, cols, vHeadDim_);
                }
            } else {
                CopyGmToUb(hUb_, hInput, rowsThisTile, cols, vHeadDim_);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + eventBase);

        if constexpr (kScalarGate) {
            if (useFp32State) {
                AscendC::Muls(calcUb_, calcUb_, scalarGate, rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        if (waitUpdateFromMte3) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
        }
        CopyGmToUb(wsUb_[slot], hUpdate, rowsThisTile, cols, cols);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + eventBase);

        if constexpr (kScalarGate) {
            if (useFp32State) {
                AscendC::Add(wsUb_[slot], calcUb_, wsUb_[slot], rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
            } else if (useBf16State) {
                AscendC::VF_CALL<detail::ApplyScalarGateUpdate<StateT>>(
                    reinterpret_cast<__ubuf__ float *>(wsUb_[slot].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ StateT *>(stateUb_[slot].GetPhyAddr()),
                    scalarGate, rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::VF_CALL<detail::ApplyScalarGateUpdate<InputT>>(
                    reinterpret_cast<__ubuf__ float *>(wsUb_[slot].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ InputT *>(hUb_.GetPhyAddr()),
                    scalarGate, rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            auto gateInput = gmGk_[offsets.gkOffset +
                (offsets.blockTokens - 1) * kHeadDim_ + rowStart];
            if (rowStart == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1 + eventBase);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + eventBase);
            }
            if constexpr (std::is_same<GateT, float>::value) {
                AscendC::DataCopy(gkLastUb_[slot], gateInput, rowsThisTile);
            } else {
                AscendC::DataCopy(gkInputUb_[slot], gateInput, rowsThisTile);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + eventBase);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2 + eventBase);
            if constexpr (std::is_same<GateT, float>::value) {
                AscendC::VF_CALL<detail::PrepareKGate<GateT, kUseExp2>>(
                    reinterpret_cast<__ubuf__ float *>(gkLastUb_[slot].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ GateT *>(gkLastUb_[slot].GetPhyAddr()),
                    static_cast<uint16_t>(rowsThisTile));
            } else {
                AscendC::VF_CALL<detail::PrepareKGate<GateT, kUseExp2>>(
                    reinterpret_cast<__ubuf__ float *>(gkLastUb_[slot].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ GateT *>(gkInputUb_[slot].GetPhyAddr()),
                    static_cast<uint16_t>(rowsThisTile));
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (useFp32State) {
                AscendC::VF_CALL<detail::ApplyRowScale>(
                    reinterpret_cast<__ubuf__ float *>(calcUb_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gkLastUb_[slot].GetPhyAddr()),
                    static_cast<uint16_t>(rowsThisTile), static_cast<uint16_t>(cols));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(wsUb_[slot], calcUb_, wsUb_[slot], rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
            } else if (useBf16State) {
                AscendC::VF_CALL<detail::ApplyKGateUpdate<StateT>>(
                    reinterpret_cast<__ubuf__ float *>(wsUb_[slot].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ StateT *>(stateUb_[slot].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gkLastUb_[slot].GetPhyAddr()),
                    static_cast<uint16_t>(rowsThisTile), static_cast<uint16_t>(cols));
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::VF_CALL<detail::ApplyKGateUpdate<InputT>>(
                    reinterpret_cast<__ubuf__ float *>(wsUb_[slot].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ InputT *>(hUb_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gkLastUb_[slot].GetPhyAddr()),
                    static_cast<uint16_t>(rowsThisTile), static_cast<uint16_t>(cols));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + eventBase);
        }

        if (storeFinalState_ && offsets.isFinalState &&
            std::is_same<StateT, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + eventBase);
            waitHFromV = true;
        } else {
            waitHFromV = false;
        }

        if constexpr (std::is_same<StateT, float>::value) {
            if (storeFinalState_) {
                if (!offsets.isFinalState) {
                    AscendC::Cast(hUb_, wsUb_[slot], AscendC::RoundMode::CAST_RINT,
                                  rowsThisTile * cols);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + eventBase);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0 + eventBase);
                CopyUbToGm(finalState, wsUb_[slot], rowsThisTile, cols, vHeadDim_);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(updateReadyEvent);
                waitUpdateFromMte3 = true;
                if (!offsets.isFinalState) {
                    CopyUbToGm(hOutput, hUb_, rowsThisTile, cols, vHeadDim_);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                        EVENT_ID2 + eventBase);
                }
            } else {
                AscendC::Cast(hUb_, wsUb_[slot], AscendC::RoundMode::CAST_RINT,
                              rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                CopyUbToGm(hOutput, hUb_, rowsThisTile, cols, vHeadDim_);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
                waitUpdateFromMte3 = false;
            }
        } else if constexpr (kSeparateBf16State) {
            AscendC::Cast(stateUb_[slot], wsUb_[slot], AscendC::RoundMode::CAST_RINT,
                          rowsThisTile * cols);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
            CopyUbToGm(finalState, stateUb_[slot], rowsThisTile, cols, vHeadDim_);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
            if (!offsets.isFinalState) {
                AscendC::Cast(hUb_, stateUb_[slot], AscendC::RoundMode::CAST_RINT,
                              rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                CopyUbToGm(hOutput, hUb_, rowsThisTile, cols, vHeadDim_);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
            waitUpdateFromMte3 = false;
        } else {
            if (storeFinalState_ && offsets.isFinalState) {
                AscendC::Cast(stateUb_[slot], wsUb_[slot], AscendC::RoundMode::CAST_RINT,
                              rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                CopyUbToGm(finalState, stateUb_[slot], rowsThisTile, cols, vHeadDim_);
            } else {
                AscendC::Cast(hUb_, wsUb_[slot], AscendC::RoundMode::CAST_RINT,
                              rowsThisTile * cols);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
                CopyUbToGm(hOutput, hUb_, rowsThisTile, cols, vHeadDim_);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
            waitUpdateFromMte3 = false;
        }
    }

    if constexpr (std::is_same<StateT, float>::value) {
        if (storeFinalState_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(updateReadyEvent);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + eventBase);
            if (!offsets.isFinalState) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
            }
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
        }
    } else {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
    }
    if (useSeparateState) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + eventBase);
    }
    if constexpr (kKGate) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + eventBase);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0 + eventBase);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::ComputeTailVWorkspace(
        const Offsets &offsets, uint32_t eventId)
{
    AscendC::ResetMask();
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
    auto input = resource_.ubBuf.template GetBufferByByte<InputT>(166 * 1024);
    auto inputFp32 = resource_.ubBuf.template GetBufferByByte<float>(167 * 1024);
    auto accum = resource_.ubBuf.template GetBufferByByte<float>(168 * 1024);
    auto weight = resource_.ubBuf.template GetBufferByByte<InputT>(169 * 1024);
    auto weightFp32 = resource_.ubBuf.template GetBufferByByte<float>(170 * 1024);
    for (uint32_t token = 0; token < offsets.blockTokens; ++token) {
        AscendC::DataCopy(weight, gmW_[offsets.wOffset + token * kHeadDim_], kHeadDim_);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
        AscendC::Cast(weightFp32, weight, AscendC::RoundMode::CAST_NONE, kHeadDim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
        AscendC::Duplicate(accum, 0.0f, offsets.vBlockDim);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < kHeadDim_; ++k) {
            AscendC::DataCopy(input, gmH_[offsets.hSrcOffset + k * vHeadDim_], offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::Cast(inputFp32, input, AscendC::RoundMode::CAST_NONE, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            const float factor = weightFp32.GetValue(k);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
            AscendC::Muls(inputFp32, inputFp32, factor, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accum, accum, inputFp32, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(eventId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::DataCopy(gmVWorkspace_[offsets.vWorkOffset + token * offsets.vBlockDim],
                          accum, offsets.vBlockDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::ComputeTailHWorkspace(
        const Offsets &offsets, uint32_t eventId)
{
    AscendC::ResetMask();
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
    auto input = resource_.ubBuf.template GetBufferByByte<InputT>(166 * 1024);
    auto inputFp32 = resource_.ubBuf.template GetBufferByByte<float>(167 * 1024);
    auto accum = resource_.ubBuf.template GetBufferByByte<float>(168 * 1024);
    auto weight = resource_.ubBuf.template GetBufferByByte<InputT>(169 * 1024);
    auto weightFp32 = resource_.ubBuf.template GetBufferByByte<float>(170 * 1024);
    for (uint32_t kRow = 0; kRow < kHeadDim_; ++kRow) {
        AscendC::Duplicate(accum, 0.0f, offsets.vBlockDim);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t token = 0; token < offsets.blockTokens; ++token) {
            AscendC::DataCopy(weight, gmK_[offsets.wkOffset + token * kHeadDim_], kHeadDim_);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::Cast(weightFp32, weight, AscendC::RoundMode::CAST_NONE, kHeadDim_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
            AscendC::DataCopy(input,
                              gmVUpdateWorkspace_[offsets.vWorkOffset + token * offsets.vBlockDim],
                              offsets.vBlockDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
            AscendC::Cast(inputFp32, input, AscendC::RoundMode::CAST_NONE, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            const float factor = weightFp32.GetValue(kRow);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
            AscendC::Muls(inputFp32, inputFp32, factor, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accum, accum, inputFp32, offsets.vBlockDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(eventId);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
        AscendC::DataCopy(gmHWorkspace_[offsets.hWorkOffset + kRow * offsets.vBlockDim],
                          accum, offsets.vBlockDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
}

} // namespace GDN::FwdHStandalone

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_ARCH35_VECTOR_H
