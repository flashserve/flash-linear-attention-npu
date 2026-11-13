/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

template <typename inputT, typename calT, bool IsVarlen>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT, IsVarlen>::CopyInA6Tile(
    uint64_t bos, uint32_t i_t, uint32_t i_d, uint32_t seqLen, uint32_t slot, event_t mte2ToVEvent)
{
    LocalTensor<inputT> xSlot = xBuf_.Get<inputT>()[slot * btBdCount_];
    LocalTensor<inputT> dySlot = dyBuf_.Get<inputT>()[slot * dyBdCount_];
    LocalTensor<inputT> ySlot = yBuf_.Get<inputT>()[slot * dyBdCount_];

    event_t vMte2Ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(vMte2Ev);
    WaitFlag<HardEvent::V_MTE2>(vMte2Ev);

    CopyInInputTileRawAsync(xGm_, xSlot, bos, i_t * BT_, i_d, BT_, seqLen, false);
    CopyInInputTileRawAsync(dyGm_, dySlot, bos, i_t * BT_, i_d, dyRows_, seqLen, true);
    if (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH) {
        CopyInInputTileRawAsync(yGm_, ySlot, bos, i_t * BT_, i_d, dyRows_, seqLen, true);
    }
    SetFlag<HardEvent::MTE2_V>(mte2ToVEvent);
}

template <typename inputT, typename calT, bool IsVarlen>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT, IsVarlen>::ComputeA6Tile(
    uint32_t i_t, uint32_t seqLen, uint32_t slot)
{
    if constexpr (!IsSameType<inputT, half>::value && !IsSameType<inputT, bfloat16_t>::value) {
        (void)i_t;
        (void)seqLen;
        (void)slot;
        return;
    }
    uint32_t startRow = i_t * BT_;
    uint32_t validRows = (startRow + BT_ <= seqLen) ? BT_ :
                         ((startRow < seqLen) ? (seqLen - startRow) : 0);
    uint32_t useInit = (useInitialState_ && startRow == 0 && validRows > 0) ? 1U : 0U;
    uint32_t computeDh0 = (hasDh0_ && useInit) ? 1U : 0U;
    LocalTensor<inputT> xRaw = xBuf_.Get<inputT>()[slot * btBdCount_];
    LocalTensor<inputT> dyRaw = dyBuf_.Get<inputT>()[slot * dyBdCount_];
    LocalTensor<inputT> yRaw = (activation_ == ACTIVATION_SILU || activation_ == ACTIVATION_SWISH)
        ? yBuf_.Get<inputT>()[slot * dyBdCount_] : dyRaw;
    LocalTensor<float> dh0Local = computeDh0 ? dh0Buf_.Get<float>() : tempBuf_.Get<float>();
    NsCausalConv1dBwd::ComputeTileDirectRegbase<inputT>(
        xRaw, dyRaw, yRaw, weightBuf_.Get<inputT>(),
        tempBuf_.Get<float>(), dxBuf_.Get<float>(), dwBuf_.Get<float>(), dbBuf_.Get<float>(), dh0Local,
        BT_, W_, dyRows_, activation_, validRows, useInit, computeDh0, hasBias_);
    PipeBarrier<PIPE_V>();
}

template <typename inputT, typename calT, bool IsVarlen>
__aicore__ inline bool CausalConv1dBwdKernel<inputT, calT, IsVarlen>::UseA6Direct() const
{
    if constexpr (!IsSameType<inputT, half>::value && !IsSameType<inputT, bfloat16_t>::value) {
        return false;
    } else {
        bool supportActivation = (activation_ == ACTIVATION_SILU &&
                                  !useInitialState_ && !useFinalState_);
        return hasWeight_ &&
               supportActivation &&
               BD_ == NsCausalConv1dBwd::BWD_DIRECT_BD &&
               W_ > 0 && W_ <= NsCausalConv1dBwd::BWD_DIRECT_W_MAX;
    }
}

template <typename inputT, typename calT, bool IsVarlen>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT, IsVarlen>::InitA6DirectBuffers()
{
    pipe_->InitBuffer(xBuf_, 2 * btBdCount_ * sizeof(inputT));
    pipe_->InitBuffer(dyBuf_, 2 * dyBdCount_ * sizeof(inputT));
    pipe_->InitBuffer(dxBuf_, btBdCount_ * FP32_DTYPE_SIZE);
    if constexpr (!IsSameType<inputT, float>::value) {
        pipe_->InitBuffer(castBuf_, btBdCount_ * sizeof(inputT));
    }
    pipe_->InitBuffer(tempBuf_, dyBdCount_ * FP32_DTYPE_SIZE);
    pipe_->InitBuffer(yBuf_, 2 * dyBdCount_ * sizeof(inputT));
    if (hasWeight_) {
        pipe_->InitBuffer(weightBuf_, wBdCount_ * sizeof(inputT));
        pipe_->InitBuffer(dwBuf_, wBdCount_ * FP32_DTYPE_SIZE);
    }
    if (hasBias_) {
        pipe_->InitBuffer(dbBuf_, BD_ * FP32_DTYPE_SIZE);
    }
}

template <typename inputT, typename calT, bool IsVarlen>
__aicore__ inline void CausalConv1dBwdKernel<inputT, calT, IsVarlen>::ProcessA6Direct()
{
    const uint32_t core = GetBlockIdx();
    const uint32_t count = chunkPerCore_ + (core < tailChunk_);
    const uint32_t first = core * chunkPerCore_ + (core < tailChunk_ ? core : tailChunk_);
    event_t ready[2] = {
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()),
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>())
    };
    for (uint32_t d = 0; d < numBlksD_; ++d) {
        CopyInWeightRaw(d);
        Duplicate(dwBuf_.Get<float>(), float(0), wBdCount_);
        if (hasBias_) {
            Duplicate(dbBuf_.Get<float>(), float(0), BD_);
        }
        PipeBarrier<PIPE_V>();
        bool prefetched = false;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t batch, t, length;
            uint64_t bos;
            if (!ResolveChunk(first + i, batch, t, bos, length)) {
                continue;
            }
            const uint32_t slot = i & 1U;
            if (!prefetched) {
                CopyInA6Tile(bos, t, d, length, slot, ready[slot]);
            }
            WaitFlag<HardEvent::MTE2_V>(ready[slot]);
            prefetched = false;
            if (i + 1 < count) {
                uint32_t nextBatch, nextT, nextLength;
                uint64_t nextBos;
                if (ResolveChunk(first + i + 1, nextBatch, nextT, nextBos, nextLength)) {
                    const uint32_t nextSlot = slot ^ 1U;
                    CopyInA6Tile(nextBos, nextT, d, nextLength, nextSlot, ready[nextSlot]);
                    prefetched = true;
                }
            }
            ComputeA6Tile(t, length, slot);
            CopyOutDx(bos, t, d, length);
        }
        CopyOutPartialDwDb(core, d);
    }
    SyncAll();
    for (uint32_t d = core; d < numBlksD_; d += GetBlockNum()) {
        ReducePartialDwDb(d);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(ready[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(ready[1]);
}
