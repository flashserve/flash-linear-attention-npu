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
#include "../chunk_gated_delta_rule_fwd_h_scheduler.h"

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

struct Stage1WithP {};
struct Stage1WithoutP {};

struct FloatStateSource {};
struct NativeStateSource {};
struct InputStateSource {};

struct FloatStateAndHOutput {};
struct FloatStateOnlyOutput {};
struct SeparateB16StateAndHOutput {};
struct SharedB16StateAndHOutput {};
struct B16StateOnlyOutput {};
struct HOnlyOutput {};

template <typename InputT>
__simd_callee__ inline void LoadStage1VNew(
    RegTensor<float> &vNew, RegTensor<float> &uReg,
    __ubuf__ float *workspace, __ubuf__ InputT *uInput,
    uint32_t offset, MaskReg &mask, Stage1WithP)
{
    LoadAsFloat<InputT>(uReg, uInput + offset, mask);
    LoadFloat(vNew, workspace + offset);
    Sub(vNew, uReg, vNew, mask);
}

template <typename InputT>
__simd_callee__ inline void LoadStage1VNew(
    RegTensor<float> &vNew, RegTensor<float> &,
    __ubuf__ float *, __ubuf__ InputT *uInput,
    uint32_t offset, MaskReg &mask, Stage1WithoutP)
{
    LoadAsFloat<InputT>(vNew, uInput + offset, mask);
}

template <typename InputT>
__simd_callee__ inline void StoreStage1Outputs(
    __ubuf__ float *workspace, __ubuf__ InputT *packedOutput,
    __ubuf__ InputT *rowOutput, RegTensor<float> &vNew,
    RegTensor<float> &vUpdate, uint32_t offset, uint32_t packedOffset,
    MaskReg &mask)
{
    RegTensor<InputT> rowOutReg, packedOutReg;
    Cast<InputT, float, F32_TO_B16_RINT>(rowOutReg, vNew, mask);
    StoreAlign<InputT, StoreDist::DIST_PACK_B32>(rowOutput + offset, rowOutReg, mask);
    StoreFloat(workspace + offset, vUpdate, mask);
    Cast<InputT, float, F32_TO_B16_RINT>(packedOutReg, vUpdate, mask);
    StoreAlign<InputT, StoreDist::DIST_PACK_B32>(
        packedOutput + packedOffset, packedOutReg, mask);
}

template <typename InputT, typename GateT, typename Stage1Input>
static __simd_vf__ inline void ComputeStage1GExpTile(
    __ubuf__ float *workspace, __ubuf__ InputT *uInput,
    __ubuf__ InputT *packedOutput, __ubuf__ InputT *rowOutput,
    __ubuf__ GateT *gateRaw, uint32_t rowOffset, uint16_t totalRows,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t C0 = 16;
    RegTensor<float> uReg, vNewReg, vUpdateReg, scaleReg, gateReg, gateLastReg;
    MaskReg mask, scalarMask = UpdateMask<float>(1);
    LoadAsFloat<GateT>(gateLastReg, gateRaw + totalRows - 1, scalarMask);
    for (uint16_t row = 0; row < rows; ++row) {
        scalarMask = UpdateMask<float>(1);
        LoadAsFloat<GateT>(gateReg, gateRaw + rowOffset + row, scalarMask);
        Sub(gateReg, gateLastReg, gateReg, scalarMask);
        Exp(gateReg, gateReg, scalarMask);
        for (uint16_t col = 0; col < cols; col += C0) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > C0 ? C0 : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadStage1VNew<InputT>(vNewReg, uReg, workspace, uInput, offset, mask,
                                    Stage1Input{});
            Duplicate(scaleReg, gateReg, mask);
            Mul(vUpdateReg, vNewReg, scaleReg, mask);
            const uint32_t packedOffset = ((col / C0) * rows + row) * C0;
            StoreStage1Outputs<InputT>(workspace, packedOutput, rowOutput,
                                       vNewReg, vUpdateReg, offset, packedOffset, mask);
        }
    }
}

template <typename InputT, typename GateT, typename Stage1Input>
static __simd_vf__ inline void ComputeStage1GExp2Tile(
    __ubuf__ float *workspace, __ubuf__ InputT *uInput,
    __ubuf__ InputT *packedOutput, __ubuf__ InputT *rowOutput,
    __ubuf__ GateT *gateRaw, uint32_t rowOffset, uint16_t totalRows,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t C0 = 16;
    RegTensor<float> uReg, vNewReg, vUpdateReg, scaleReg, gateReg, gateLastReg;
    MaskReg mask, scalarMask = UpdateMask<float>(1);
    LoadAsFloat<GateT>(gateLastReg, gateRaw + totalRows - 1, scalarMask);
    for (uint16_t row = 0; row < rows; ++row) {
        scalarMask = UpdateMask<float>(1);
        LoadAsFloat<GateT>(gateReg, gateRaw + rowOffset + row, scalarMask);
        Sub(gateReg, gateLastReg, gateReg, scalarMask);
        Muls(gateReg, gateReg, 0.6931471805599453f, scalarMask);
        Exp(gateReg, gateReg, scalarMask);
        for (uint16_t col = 0; col < cols; col += C0) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > C0 ? C0 : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadStage1VNew<InputT>(vNewReg, uReg, workspace, uInput, offset, mask,
                                    Stage1Input{});
            Duplicate(scaleReg, gateReg, mask);
            Mul(vUpdateReg, vNewReg, scaleReg, mask);
            const uint32_t packedOffset = ((col / C0) * rows + row) * C0;
            StoreStage1Outputs<InputT>(workspace, packedOutput, rowOutput,
                                       vNewReg, vUpdateReg, offset, packedOffset, mask);
        }
    }
}

template <typename InputT, typename GateT, typename Stage1Input>
static __simd_vf__ inline void ComputeStage1GkTile(
    __ubuf__ float *workspace, __ubuf__ InputT *uInput,
    __ubuf__ InputT *packedOutput, __ubuf__ InputT *rowOutput,
    __ubuf__ GateT *, uint32_t, uint16_t, uint16_t rows, uint16_t cols)
{
    constexpr uint16_t C0 = 16;
    RegTensor<float> uReg, vNewReg;
    MaskReg mask;
    for (uint16_t row = 0; row < rows; ++row) {
        for (uint16_t col = 0; col < cols; col += C0) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > C0 ? C0 : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadStage1VNew<InputT>(vNewReg, uReg, workspace, uInput, offset, mask,
                                    Stage1Input{});
            const uint32_t packedOffset = ((col / C0) * rows + row) * C0;
            StoreStage1Outputs<InputT>(workspace, packedOutput, rowOutput,
                                       vNewReg, vNewReg, offset, packedOffset, mask);
        }
    }
}

template <typename InputT, typename StateT>
__simd_callee__ inline void LoadStage3State(
    RegTensor<float> &state, __ubuf__ float *stateFloat,
    __ubuf__ StateT *, __ubuf__ InputT *, uint32_t offset,
    MaskReg &, FloatStateSource)
{
    LoadFloat(state, stateFloat + offset);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void LoadStage3State(
    RegTensor<float> &state, __ubuf__ float *, __ubuf__ StateT *stateNative,
    __ubuf__ InputT *, uint32_t offset, MaskReg &mask, NativeStateSource)
{
    LoadAsFloat<StateT>(state, stateNative + offset, mask);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void LoadStage3State(
    RegTensor<float> &state, __ubuf__ float *, __ubuf__ StateT *,
    __ubuf__ InputT *stateInput, uint32_t offset, MaskReg &mask,
    InputStateSource)
{
    LoadAsFloat<InputT>(state, stateInput + offset, mask);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void StoreStage3Outputs(
    RegTensor<float> &update, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *hOutput, uint32_t offset, MaskReg &mask,
    FloatStateAndHOutput)
{
    RegTensor<InputT> hOutReg;
    StoreFloat(reinterpret_cast<__ubuf__ float *>(stateOutput) + offset, update, mask);
    Cast<InputT, float, F32_TO_B16_RINT>(hOutReg, update, mask);
    StoreAlign<InputT, StoreDist::DIST_PACK_B32>(hOutput + offset, hOutReg, mask);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void StoreStage3Outputs(
    RegTensor<float> &update, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *, uint32_t offset, MaskReg &mask, FloatStateOnlyOutput)
{
    StoreFloat(reinterpret_cast<__ubuf__ float *>(stateOutput) + offset, update, mask);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void StoreStage3Outputs(
    RegTensor<float> &update, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *hOutput, uint32_t offset, MaskReg &mask,
    SeparateB16StateAndHOutput)
{
    RegTensor<StateT> stateOutReg;
    RegTensor<float> quantizedStateReg;
    RegTensor<InputT> hOutReg;
    Cast<StateT, float, F32_TO_B16_RINT>(stateOutReg, update, mask);
    StoreAlign<StateT, StoreDist::DIST_PACK_B32>(stateOutput + offset, stateOutReg, mask);
    Cast<float, StateT, B16_TO_F32_ZERO>(quantizedStateReg, stateOutReg, mask);
    Cast<InputT, float, F32_TO_B16_RINT>(hOutReg, quantizedStateReg, mask);
    StoreAlign<InputT, StoreDist::DIST_PACK_B32>(hOutput + offset, hOutReg, mask);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void StoreStage3Outputs(
    RegTensor<float> &update, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *hOutput, uint32_t offset, MaskReg &mask,
    SharedB16StateAndHOutput)
{
    RegTensor<StateT> stateOutReg;
    Cast<StateT, float, F32_TO_B16_RINT>(stateOutReg, update, mask);
    StoreAlign<StateT, StoreDist::DIST_PACK_B32>(stateOutput + offset, stateOutReg, mask);
    StoreAlign<InputT, StoreDist::DIST_PACK_B32>(hOutput + offset, stateOutReg, mask);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void StoreStage3Outputs(
    RegTensor<float> &update, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *, uint32_t offset, MaskReg &mask, B16StateOnlyOutput)
{
    RegTensor<StateT> stateOutReg;
    Cast<StateT, float, F32_TO_B16_RINT>(stateOutReg, update, mask);
    StoreAlign<StateT, StoreDist::DIST_PACK_B32>(stateOutput + offset, stateOutReg, mask);
}

template <typename InputT, typename StateT>
__simd_callee__ inline void StoreStage3Outputs(
    RegTensor<float> &update, __ubuf__ StateT *, __ubuf__ InputT *hOutput,
    uint32_t offset, MaskReg &mask, HOnlyOutput)
{
    RegTensor<InputT> hOutReg;
    Cast<InputT, float, F32_TO_B16_RINT>(hOutReg, update, mask);
    StoreAlign<InputT, StoreDist::DIST_PACK_B32>(hOutput + offset, hOutReg, mask);
}

template <typename InputT, typename GateT, typename StateT,
          typename StateSource, typename OutputPolicy>
static __simd_vf__ inline void ComputeStage3GExpTile(
    __ubuf__ float *update, __ubuf__ float *stateFloat,
    __ubuf__ StateT *stateNative, __ubuf__ InputT *stateInput,
    __ubuf__ GateT *gateRaw,
    __ubuf__ StateT *stateOutput, __ubuf__ InputT *hOutput,
    uint16_t rows, uint16_t cols)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> updateReg, stateReg, gateReg, gateScalarReg;
    MaskReg mask, gateScalarMask = UpdateMask<float>(1);
    MaskReg gateMask = UpdateMask<float>(ELEMS);
    LoadAsFloat<GateT>(gateScalarReg, gateRaw, gateScalarMask);
    Exp(gateScalarReg, gateScalarReg, gateScalarMask);
    Duplicate(gateReg, gateScalarReg, gateMask);
    for (uint16_t row = 0; row < rows; ++row) {
        for (uint16_t col = 0; col < cols; col += ELEMS) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > ELEMS ? ELEMS : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadFloat(updateReg, update + offset);
            LoadStage3State<InputT, StateT>(stateReg, stateFloat, stateNative,
                                             stateInput, offset, mask, StateSource{});
            Mul(stateReg, stateReg, gateReg, mask);
            Add(updateReg, stateReg, updateReg, mask);
            StoreStage3Outputs<InputT, StateT>(updateReg, stateOutput, hOutput,
                                                offset, mask, OutputPolicy{});
        }
    }
}

template <typename InputT, typename GateT, typename StateT,
          typename StateSource, typename OutputPolicy>
static __simd_vf__ inline void ComputeStage3GExp2Tile(
    __ubuf__ float *update, __ubuf__ float *stateFloat,
    __ubuf__ StateT *stateNative, __ubuf__ InputT *stateInput,
    __ubuf__ GateT *gateRaw, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *hOutput, uint16_t rows, uint16_t cols)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> updateReg, stateReg, gateReg, gateScalarReg;
    MaskReg mask, gateScalarMask = UpdateMask<float>(1);
    MaskReg gateMask = UpdateMask<float>(ELEMS);
    LoadAsFloat<GateT>(gateScalarReg, gateRaw, gateScalarMask);
    Muls(gateScalarReg, gateScalarReg, 0.6931471805599453f, gateScalarMask);
    Exp(gateScalarReg, gateScalarReg, gateScalarMask);
    Duplicate(gateReg, gateScalarReg, gateMask);
    for (uint16_t row = 0; row < rows; ++row) {
        for (uint16_t col = 0; col < cols; col += ELEMS) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > ELEMS ? ELEMS : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadFloat(updateReg, update + offset);
            LoadStage3State<InputT, StateT>(stateReg, stateFloat, stateNative,
                                             stateInput, offset, mask, StateSource{});
            Mul(stateReg, stateReg, gateReg, mask);
            Add(updateReg, stateReg, updateReg, mask);
            StoreStage3Outputs<InputT, StateT>(updateReg, stateOutput, hOutput,
                                                offset, mask, OutputPolicy{});
        }
    }
}

template <typename InputT, typename GateT, typename StateT,
          typename StateSource, typename OutputPolicy>
static __simd_vf__ inline void ComputeStage3GkExpTile(
    __ubuf__ float *update, __ubuf__ float *stateFloat,
    __ubuf__ StateT *stateNative, __ubuf__ InputT *stateInput,
    __ubuf__ GateT *gateRaw, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *hOutput, uint16_t rows, uint16_t cols)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> updateReg, stateReg, gateReg, gateScalarReg;
    MaskReg mask, gateScalarMask, gateMask;
    for (uint16_t row = 0; row < rows; ++row) {
        gateScalarMask = UpdateMask<float>(1);
        gateMask = UpdateMask<float>(ELEMS);
        LoadAsFloat<GateT>(gateScalarReg, gateRaw + row, gateScalarMask);
        Duplicate(gateReg, gateScalarReg, gateMask);
        Exp(gateReg, gateReg, gateMask);
        for (uint16_t col = 0; col < cols; col += ELEMS) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > ELEMS ? ELEMS : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadFloat(updateReg, update + offset);
            LoadStage3State<InputT, StateT>(stateReg, stateFloat, stateNative,
                                             stateInput, offset, mask, StateSource{});
            Mul(stateReg, stateReg, gateReg, mask);
            Add(updateReg, stateReg, updateReg, mask);
            StoreStage3Outputs<InputT, StateT>(updateReg, stateOutput, hOutput,
                                                offset, mask, OutputPolicy{});
        }
    }
}

template <typename InputT, typename GateT, typename StateT,
          typename StateSource, typename OutputPolicy>
static __simd_vf__ inline void ComputeStage3GkExp2Tile(
    __ubuf__ float *update, __ubuf__ float *stateFloat,
    __ubuf__ StateT *stateNative, __ubuf__ InputT *stateInput,
    __ubuf__ GateT *gateRaw, __ubuf__ StateT *stateOutput,
    __ubuf__ InputT *hOutput, uint16_t rows, uint16_t cols)
{
    constexpr uint16_t ELEMS = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> updateReg, stateReg, gateReg, gateScalarReg;
    MaskReg mask, gateScalarMask, gateMask;
    for (uint16_t row = 0; row < rows; ++row) {
        gateScalarMask = UpdateMask<float>(1);
        gateMask = UpdateMask<float>(ELEMS);
        LoadAsFloat<GateT>(gateScalarReg, gateRaw + row, gateScalarMask);
        Duplicate(gateReg, gateScalarReg, gateMask);
        Muls(gateReg, gateReg, 0.6931471805599453f, gateMask);
        Exp(gateReg, gateReg, gateMask);
        for (uint16_t col = 0; col < cols; col += ELEMS) {
            uint32_t remaining = cols - col;
            uint16_t count = remaining > ELEMS ? ELEMS : static_cast<uint16_t>(remaining);
            mask = UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadFloat(updateReg, update + offset);
            LoadStage3State<InputT, StateT>(stateReg, stateFloat, stateNative,
                                             stateInput, offset, mask, StateSource{});
            Mul(stateReg, stateReg, gateReg, mask);
            Add(updateReg, stateReg, updateReg, mask);
            StoreStage3Outputs<InputT, StateT>(updateReg, stateOutput, hOutput,
                                                offset, mask, OutputPolicy{});
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

    __aicore__ inline ChunkGatedDeltaRuleFwdHVector() = default;

    __aicore__ inline void Init(
        GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR initialState,
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
        InitializeH();

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec2Done[0]);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec2Done[1]);
        PresetEvents();

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
                const bool hasStage0Output = scheduler_.NeedProcessStage0(headTask);
                const bool processStage3 = scheduler_.NeedProcessStage2(headTask);
                if (!ownsHead) {
                    if (hasStage0Output) {
                        Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube1Done[windowId]);
                    }
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec1Done[windowId]);
                    continue;
                }
                Stage1(offsets, windowId, isPing, hasStage0Output, processStage3);
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
                    Stage3(offsets, windowId, isPing);
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
        gInputUb_[0] = resource_.ubBuf.template GetBufferByByte<GateT>(164 * 1024);
        gInputUb_[1] = resource_.ubBuf.template GetBufferByByte<GateT>(165 * 1024);
        // Separate BF16 rolling state is only consumed in Stage3.  Reuse
        // the Stage1 packed banks after Stage2 has consumed them; keeping
        // these banks distinct from ioUb avoids update/state overwrite while
        // the two MTE2 transfers are in flight.
        stateUb_[0] = resource_.ubBuf.template GetBufferByByte<StateT>(
            kSeparateBf16State ? 40 * 1024 : 160 * 1024);
        stateUb_[1] = resource_.ubBuf.template GetBufferByByte<StateT>(
            kSeparateBf16State ? 48 * 1024 : 160 * 1024);
        gkInputUb_[0] = resource_.ubBuf.template GetBufferByByte<GateT>(172 * 1024);
        gkInputUb_[1] = resource_.ubBuf.template GetBufferByByte<GateT>(173 * 1024);
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
        bool hasStage0Output, bool processStage3);
    template <typename Stage1Input>
    __aicore__ inline void RunStage1Vf(
        uint32_t tileSlot, uint32_t headSlot, uint32_t rowStart,
        uint32_t totalRows, uint32_t rows, uint32_t cols);
    __aicore__ inline void LoadStage1Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex, bool hasStage0Output);
    __aicore__ inline void StoreStage1Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex);
    __aicore__ inline void Stage3(
        const Offsets &offsets, uint32_t windowId, bool isPing);
    __aicore__ inline void LoadStage3Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex);
    __aicore__ inline void StoreStage3Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex);
    template <typename StateSource, typename OutputPolicy>
    __aicore__ inline void RunStage3Vf(
        const Offsets &offsets, uint32_t tileSlot, uint32_t headSlot,
        uint32_t rows, uint32_t cols);
    template <typename StateSource>
    __aicore__ inline void RunStage3OutputPolicy(
        const Offsets &offsets, uint32_t tileSlot, uint32_t headSlot,
        uint32_t rows, uint32_t cols);
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
    AscendC::LocalTensor<GateT> gInputUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<StateT> stateUb_[LOCAL_PING_PONG_STAGES];
    AscendC::LocalTensor<GateT> gkInputUb_[LOCAL_PING_PONG_STAGES];

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
    for (uint32_t slot = 0; slot < LOCAL_PING_PONG_STAGES; ++slot) {
        const uint32_t base = slot * PONG_EVENT_BASE;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + base);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + base);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + base);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + base);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + base);
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::DrainEvents()
{
    for (uint32_t slot = 0; slot < LOCAL_PING_PONG_STAGES; ++slot) {
        const uint32_t base = slot * PONG_EVENT_BASE;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + base);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + base);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + base);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + base);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + base);
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::LoadStage1Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex, bool hasStage0Output)
{
    const uint32_t headSlot = tileIndex & 1U;
    const uint32_t eventBase = headSlot * PONG_EVENT_BASE;
    const uint32_t cols = offsets.vBlockDim;
    const uint32_t rowOffset = rowStart * vHeadDim_;

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
    CopyGmToUb(ioUb_[headSlot], gmU_[offsets.uvOffset + rowOffset],
               rows, cols, vHeadDim_);
    if (hasStage0Output) {
        CopyGmToUb(wsUb_[headSlot],
                   gmVWorkspace_[offsets.vWorkOffset + rowStart * cols],
                   rows, cols, cols);
    }
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
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
    AscendC::DataCopyParams znParams{
        static_cast<uint16_t>(cols / C0), static_cast<uint16_t>(rows), 0,
        static_cast<uint16_t>(paddedRows - rows)};
    AscendC::DataCopy(gmVUpdateWorkspace_[offsets.vWorkOffset + rowStart * C0],
                      stage1PackedUb_[headSlot], znParams);
    CopyUbToGm(gmV_[offsets.uvOffset + rowStart * vHeadDim_], stage1OutputUb_[headSlot],
               rows, cols, vHeadDim_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
template <typename Stage1Input>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::RunStage1Vf(
        uint32_t tileSlot, uint32_t headSlot, uint32_t rowStart,
        uint32_t totalRows, uint32_t rows, uint32_t cols)
{
    auto workspace = reinterpret_cast<__ubuf__ float *>(wsUb_[tileSlot].GetPhyAddr());
    auto uInput = reinterpret_cast<__ubuf__ InputT *>(ioUb_[tileSlot].GetPhyAddr());
    auto packedOutput = reinterpret_cast<__ubuf__ InputT *>(
        stage1PackedUb_[tileSlot].GetPhyAddr());
    auto rowOutput = reinterpret_cast<__ubuf__ InputT *>(
        stage1OutputUb_[tileSlot].GetPhyAddr());
    auto gateInput = reinterpret_cast<__ubuf__ GateT *>(gInputUb_[headSlot].GetPhyAddr());

    if constexpr (kScalarGate) {
        if constexpr (kUseExp2) {
            AscendC::VF_CALL<detail::ComputeStage1GExp2Tile<
                InputT, GateT, Stage1Input>>(
                workspace, uInput, packedOutput, rowOutput, gateInput,
                rowStart, static_cast<uint16_t>(totalRows),
                static_cast<uint16_t>(rows), static_cast<uint16_t>(cols));
        } else {
            AscendC::VF_CALL<detail::ComputeStage1GExpTile<
                InputT, GateT, Stage1Input>>(
                workspace, uInput, packedOutput, rowOutput, gateInput,
                rowStart, static_cast<uint16_t>(totalRows),
                static_cast<uint16_t>(rows), static_cast<uint16_t>(cols));
        }
    } else {
        AscendC::VF_CALL<detail::ComputeStage1GkTile<
            InputT, GateT, Stage1Input>>(
            workspace, uInput, packedOutput, rowOutput, gateInput,
            rowStart, static_cast<uint16_t>(totalRows),
            static_cast<uint16_t>(rows), static_cast<uint16_t>(cols));
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::Stage1(
        const Offsets &offsets, uint32_t windowId, bool isPing,
        bool hasStage0Output, bool processStage3)
{
    constexpr uint32_t ROW_TILE = 16;
    const uint32_t rows = offsets.blockTokens;
    const uint32_t cols = offsets.vBlockDim;
    const uint32_t headSlot = isPing ? 0 : 1;
    const uint32_t headEventBase = headSlot * PONG_EVENT_BASE;
    if constexpr (kScalarGate) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + headEventBase);
        CopyGmToUb(gInputUb_[headSlot], gmG_[offsets.gOffset], rows, 1, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + headEventBase);
    }

    if (hasStage0Output) {
        Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube1Done[windowId]);
    }
    const uint32_t tileCount = (rows + ROW_TILE - 1) / ROW_TILE;
    LoadStage1Tile(offsets, 0, Min(ROW_TILE, rows), headSlot,
                   hasStage0Output);
    if constexpr (kScalarGate) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3 + headEventBase);
    }
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t rowStart = tile * ROW_TILE;
        const uint32_t rowsThisTile = Min(ROW_TILE, rows - rowStart);
        const uint32_t cur = (headSlot ^ (tile & 1U)) & 1U;
        const uint32_t eventBase = cur * PONG_EVENT_BASE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + eventBase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
        if (hasStage0Output) {
            RunStage1Vf<detail::Stage1WithP>(
                cur, headSlot, rowStart, rows, rowsThisTile, cols);
        } else {
            RunStage1Vf<detail::Stage1WithoutP>(
                cur, headSlot, rowStart, rows, rowsThisTile, cols);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);

        // Enqueue MTE2(next) before MTE3(current).  Both run while the
        // current VF work drains, and neither helper waits for completion.
        if (tile + 1 < tileCount) {
            const uint32_t nextRow = (tile + 1) * ROW_TILE;
            LoadStage1Tile(offsets, nextRow, Min(ROW_TILE, rows - nextRow),
                           headSlot ^ ((tile + 1) & 1U), hasStage0Output);
        }
        StoreStage1Tile(offsets, rowStart, rowsThisTile, cur);
        if (tile + 1 == tileCount) {
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(scheduler_.vec1Done[windowId]);
        }
    }
    if constexpr (kScalarGate) {
        if (!processStage3) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + headEventBase);
        }
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::LoadStage3Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex)
{
    const uint32_t slot = tileIndex & 1U;
    const uint32_t eventBase = slot * PONG_EVENT_BASE;
    const uint32_t cols = offsets.vBlockDim;
    const uint32_t stateOffset = rowStart * vHeadDim_;

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + eventBase);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
    if constexpr (kKGate) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + eventBase);
    }

    CopyGmToUb(ioUb_[slot].template ReinterpretCast<float>(),
               gmHWorkspace_[offsets.hWorkOffset + rowStart * cols],
               rows, cols, cols);
    const bool useRollingState = !offsets.isInitialState || useInitialState_;
    if constexpr (std::is_same<StateT, float>::value) {
        if (useRollingState) {
            if (offsets.isInitialState) {
                CopyGmToUb(wsUb_[slot],
                           gmInitialState_[offsets.initialStateOffset + stateOffset],
                           rows, cols, vHeadDim_);
            } else {
                CopyGmToUb(wsUb_[slot],
                           gmFinalState_[offsets.finalStateOffset + stateOffset],
                           rows, cols, vHeadDim_);
            }
        } else {
            CopyGmToUb(stage1PackedUb_[slot],
                       gmH_[offsets.hSrcOffset + stateOffset],
                       rows, cols, vHeadDim_);
        }
    } else if constexpr (kSeparateBf16State) {
        if (useRollingState) {
            if (offsets.isInitialState) {
                CopyGmToUb(stateUb_[slot],
                           gmInitialState_[offsets.initialStateOffset + stateOffset],
                           rows, cols, vHeadDim_);
            } else {
                CopyGmToUb(stateUb_[slot],
                           gmFinalState_[offsets.finalStateOffset + stateOffset],
                           rows, cols, vHeadDim_);
            }
        } else {
            CopyGmToUb(stage1PackedUb_[slot],
                       gmH_[offsets.hSrcOffset + stateOffset],
                       rows, cols, vHeadDim_);
        }
    } else {
        CopyGmToUb(stage1PackedUb_[slot],
                   gmH_[offsets.hSrcOffset + stateOffset],
                   rows, cols, vHeadDim_);
    }
    if constexpr (kKGate) {
        CopyGmToUb(gkInputUb_[slot],
                   gmGk_[offsets.gkOffset +
                         (offsets.blockTokens - 1) * kHeadDim_ + rowStart],
                   rows, 1, 1);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + eventBase);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::StoreStage3Tile(
        const Offsets &offsets, uint32_t rowStart, uint32_t rows,
        uint32_t tileIndex)
{
    const uint32_t slot = tileIndex & 1U;
    const uint32_t eventBase = slot * PONG_EVENT_BASE;
    const uint32_t cols = offsets.vBlockDim;
    const uint32_t stateOffset = rowStart * vHeadDim_;
    const bool finalChunk = offsets.isFinalState;

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + eventBase);
    if (storeFinalState_) {
        if constexpr (std::is_same<StateT, float>::value) {
            CopyUbToGm(gmFinalState_[offsets.finalStateOffset + stateOffset],
                       wsUb_[slot], rows, cols, vHeadDim_);
        } else if constexpr (kSeparateBf16State) {
            CopyUbToGm(gmFinalState_[offsets.finalStateOffset + stateOffset],
                       stateUb_[slot], rows, cols, vHeadDim_);
        } else if (finalChunk) {
            CopyUbToGm(gmFinalState_[offsets.finalStateOffset + stateOffset],
                       stage1PackedUb_[slot], rows, cols, vHeadDim_);
        }
    }
    if (!finalChunk || !storeFinalState_) {
        CopyUbToGm(gmH_[offsets.hDstOffset + stateOffset],
                   stage1OutputUb_[slot], rows, cols, vHeadDim_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + eventBase);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + eventBase);
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
template <typename StateSource, typename OutputPolicy>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::RunStage3Vf(
        const Offsets &offsets, uint32_t tileSlot, uint32_t headSlot,
        uint32_t rows, uint32_t cols)
{
    auto update = reinterpret_cast<__ubuf__ float *>(ioUb_[tileSlot].GetPhyAddr());
    auto stateFloat = reinterpret_cast<__ubuf__ float *>(wsUb_[tileSlot].GetPhyAddr());
    auto stateNative = reinterpret_cast<__ubuf__ StateT *>(
        stateUb_[tileSlot].GetPhyAddr());
    auto stateInput = reinterpret_cast<__ubuf__ InputT *>(
        stage1PackedUb_[tileSlot].GetPhyAddr());
    auto hOutput = reinterpret_cast<__ubuf__ InputT *>(
        stage1OutputUb_[tileSlot].GetPhyAddr());

    __ubuf__ StateT *stateOutput;
    if constexpr (std::is_same<StateT, float>::value) {
        stateOutput = reinterpret_cast<__ubuf__ StateT *>(wsUb_[tileSlot].GetPhyAddr());
    } else if constexpr (kSeparateBf16State) {
        stateOutput = reinterpret_cast<__ubuf__ StateT *>(stateUb_[tileSlot].GetPhyAddr());
    } else {
        stateOutput = reinterpret_cast<__ubuf__ StateT *>(
            stage1PackedUb_[tileSlot].GetPhyAddr());
    }

    __ubuf__ GateT *gateInput;
    if constexpr (kScalarGate) {
        gateInput = reinterpret_cast<__ubuf__ GateT *>(gInputUb_[headSlot].GetPhyAddr()) +
            offsets.blockTokens - 1;
        if constexpr (kUseExp2) {
            AscendC::VF_CALL<detail::ComputeStage3GExp2Tile<
                InputT, GateT, StateT, StateSource, OutputPolicy>>(
                update, stateFloat, stateNative, stateInput, gateInput,
                stateOutput, hOutput, static_cast<uint16_t>(rows),
                static_cast<uint16_t>(cols));
        } else {
            AscendC::VF_CALL<detail::ComputeStage3GExpTile<
                InputT, GateT, StateT, StateSource, OutputPolicy>>(
                update, stateFloat, stateNative, stateInput, gateInput,
                stateOutput, hOutput, static_cast<uint16_t>(rows),
                static_cast<uint16_t>(cols));
        }
    } else {
        gateInput = reinterpret_cast<__ubuf__ GateT *>(
            gkInputUb_[tileSlot].GetPhyAddr());
        if constexpr (kUseExp2) {
            AscendC::VF_CALL<detail::ComputeStage3GkExp2Tile<
                InputT, GateT, StateT, StateSource, OutputPolicy>>(
                update, stateFloat, stateNative, stateInput, gateInput,
                stateOutput, hOutput, static_cast<uint16_t>(rows),
                static_cast<uint16_t>(cols));
        } else {
            AscendC::VF_CALL<detail::ComputeStage3GkExpTile<
                InputT, GateT, StateT, StateSource, OutputPolicy>>(
                update, stateFloat, stateNative, stateInput, gateInput,
                stateOutput, hOutput, static_cast<uint16_t>(rows),
                static_cast<uint16_t>(cols));
        }
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
template <typename StateSource>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::RunStage3OutputPolicy(
        const Offsets &offsets, uint32_t tileSlot, uint32_t headSlot,
        uint32_t rows, uint32_t cols)
{
    if (offsets.isFinalState) {
        if constexpr (std::is_same<StateT, float>::value) {
            RunStage3Vf<StateSource, detail::FloatStateOnlyOutput>(
                offsets, tileSlot, headSlot, rows, cols);
        } else {
            RunStage3Vf<StateSource, detail::B16StateOnlyOutput>(
                offsets, tileSlot, headSlot, rows, cols);
        }
        return;
    }

    if (storeFinalState_) {
        if constexpr (std::is_same<StateT, float>::value) {
            RunStage3Vf<StateSource, detail::FloatStateAndHOutput>(
                offsets, tileSlot, headSlot, rows, cols);
        } else if constexpr (kSeparateBf16State) {
            RunStage3Vf<StateSource, detail::SeparateB16StateAndHOutput>(
                offsets, tileSlot, headSlot, rows, cols);
        } else {
            RunStage3Vf<StateSource, detail::SharedB16StateAndHOutput>(
                offsets, tileSlot, headSlot, rows, cols);
        }
    } else {
        RunStage3Vf<StateSource, detail::HOnlyOutput>(
            offsets, tileSlot, headSlot, rows, cols);
    }
}

template <typename InputT, typename GateT, typename StateT, typename WorkspaceT,
          uint32_t GateMode, uint32_t ExpMode>
__aicore__ inline void ChunkGatedDeltaRuleFwdHVector<
    InputT, GateT, StateT, WorkspaceT, GateMode, ExpMode>::Stage3(
        const Offsets &offsets, uint32_t windowId, bool isPing)
{
    constexpr uint32_t ROW_TILE = 16;
    const uint32_t headSlot = isPing ? 0 : 1;
    const uint32_t rows = kHeadDim_;
    const uint32_t cols = offsets.vBlockDim;
    Catlass::Arch::CrossCoreWaitFlag(scheduler_.cube2Done[windowId]);

    const uint32_t tileCount = (rows + ROW_TILE - 1) / ROW_TILE;
    LoadStage3Tile(offsets, 0, Min(ROW_TILE, rows), headSlot);

    const bool useRollingState = !offsets.isInitialState || useInitialState_;
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t rowStart = tile * ROW_TILE;
        const uint32_t rowsThisTile = Min(ROW_TILE, rows - rowStart);
        const uint32_t cur = (headSlot ^ (tile & 1U)) & 1U;
        const uint32_t curBase = cur * PONG_EVENT_BASE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0 + curBase);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2 + curBase);
        if constexpr (std::is_same<StateT, float>::value) {
            if (useRollingState) {
                RunStage3OutputPolicy<detail::FloatStateSource>(
                    offsets, cur, headSlot, rowsThisTile, cols);
            } else {
                RunStage3OutputPolicy<detail::InputStateSource>(
                    offsets, cur, headSlot, rowsThisTile, cols);
            }
        } else if constexpr (kSeparateBf16State) {
            if (useRollingState) {
                RunStage3OutputPolicy<detail::NativeStateSource>(
                    offsets, cur, headSlot, rowsThisTile, cols);
            } else {
                RunStage3OutputPolicy<detail::InputStateSource>(
                    offsets, cur, headSlot, rowsThisTile, cols);
            }
        } else {
            RunStage3OutputPolicy<detail::InputStateSource>(
                offsets, cur, headSlot, rowsThisTile, cols);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + curBase);
        if constexpr (kKGate) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1 + curBase);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2 + curBase);

        if (tile + 1 < tileCount) {
            const uint32_t nextRow = (tile + 1) * ROW_TILE;
            LoadStage3Tile(offsets, nextRow, Min(ROW_TILE, rows - nextRow),
                           headSlot ^ ((tile + 1) & 1U));
        }
        StoreStage3Tile(offsets, rowStart, rowsThisTile, cur);
    }
    if constexpr (kScalarGate) {
        const uint32_t headEventBase = headSlot * PONG_EVENT_BASE;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + headEventBase);
    }
}

} // namespace GDN::FwdHStandalone

#endif // CHUNK_GATED_DELTA_RULE_FWD_H_ARCH35_VECTOR_H
