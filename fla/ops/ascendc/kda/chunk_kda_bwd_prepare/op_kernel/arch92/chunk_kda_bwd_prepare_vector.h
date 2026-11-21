/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH92_VECTOR_H
#define CHUNK_KDA_BWD_PREPARE_ARCH92_VECTOR_H

#include "chunk_kda_bwd_prepare_common.h"

namespace KDA::Arch92 {
using namespace AscendC::MicroAPI;

__simd_vf__ inline void BuildTriScaleMask(__ubuf__ float *mask, float scale)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    RegTensor<float> scaled;
    Duplicate(zero, 0.0f, full);
    Duplicate(scaled, scale, full);
    for (uint16_t row = 0; row < kRows; ++row) {
        // A6 register width may differ from one 64-element row.
        for (uint16_t col = 0; col < kRows; col += ELEMS_PER_VF) {
            const uint32_t offset = static_cast<uint32_t>(row) * kRows + col;
            uint32_t count = kRows - col < ELEMS_PER_VF ? kRows - col : ELEMS_PER_VF;
            uint32_t lowerCount = row < col ? 0 : row - col + 1;
            if (lowerCount > count) lowerCount = count;
            MaskReg rowMask = UpdateMask<float>(count);
            MaskReg lower = UpdateMask<float>(lowerCount);
            StoreAlign(mask + offset, zero, rowMask);
            StoreAlign(mask + offset, scaled, lower);
        }
    }
}

__simd_vf__ inline void ApplyTriScaleMask(
    __ubuf__ float *output, __ubuf__ float *input, __ubuf__ float *mask, uint16_t validRows)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, full);
    for (uint16_t row = 0; row < validRows; ++row) {
        for (uint16_t col = 0; col < kRows; col += ELEMS_PER_VF) {
            const uint32_t offset = static_cast<uint32_t>(row) * kRows + col;
            RegTensor<float> rawInput;
            RegTensor<float> rawSafe;
            RegTensor<float> maskValue;
            RegTensor<float> result;
            uint32_t count = kRows - col < ELEMS_PER_VF ? kRows - col : ELEMS_PER_VF;
            uint32_t lowerCount = row < col ? 0 : row - col + 1;
            if (lowerCount > count) lowerCount = count;
            MaskReg rowMask = UpdateMask<float>(count);
            MaskReg lower = UpdateMask<float>(lowerCount);
            // FIX initializes align16(validRows) columns. Only load blocks
            // touched by the lower triangle, without crossing a row boundary.
            LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(rawInput, input + offset, 1, lower);
            Select(rawSafe, rawInput, zero, lower);
            LoadAlign<float, DataCopyMode::DATA_BLOCK_COPY>(maskValue, mask + offset, 1, rowMask);
            Mul(result, rawSafe, maskValue, rowMask);
            // Keep invalid elements strictly +0, including negative scale.
            Select(result, result, zero, lower);
            StoreAlign(output + offset, result, rowMask);
        }
    }
}

class ChunkKdaBwdPrepareVector {
public:
    __aicore__ inline void Init(
        GM_ADDR cu, GM_ADDR indices, GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
        const ChunkKdaBwdPrepareTilingData *tiling, AscendC::TPipe *pipe)
    {
        cu_ = cu;
        indices_ = indices;
        tiling_ = tiling;
        pipe_ = pipe;
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        dAqk_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dv_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        dqRaw_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        // First and only UB allocation: its base is zero, shared with the
        // AIC Fixpipe views. Subregions keep exactly the design's byte offsets.
        pipe_->InitBuffer(ub_, kUbBytes);
        AllocateEvents<AscendC::HardEvent::V_MTE3>(pipe_, outputReady_, kSlots, false);
        AllocateEvents<AscendC::HardEvent::MTE3_V>(pipe_, outFree_, kSlots, true);
    }

    __aicore__ inline void Process()
    {
        auto mask = ub_.Get<uint8_t>()[kMaskOffset].ReinterpretCast<float>();
        BuildTriScaleMask((__ubuf__ float *)mask.GetPhyAddr(), tiling_->scale);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint16_t s = 0; s < kSlots; ++s) {
            AscendC::CrossCoreSetFlag<kCrossMode, PIPE_MTE3>(kUbFree + s);
        }
        WorkIterator iterator(tiling_, cu_, indices_);
        WorkInfo work;
        while (iterator.Next(work)) {
            const uint32_t s = work.slot;
            auto group = ub_.Get<uint8_t>()[s * kGroupBytes];
            auto raw = group[kRawOffset].ReinterpretCast<float>();
            auto out = group[kOutOffset].ReinterpretCast<float>();
            auto q = group[kQOffset].ReinterpretCast<float>();
            auto d = group[kDOffset].ReinterpretCast<bfloat16_t>();
            AscendC::CrossCoreWaitFlag<kCrossMode, PIPE_V>(kAReady + s);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outFree_[s]);
            ApplyTriScaleMask((__ubuf__ float *)out.GetPhyAddr(),
                              (__ubuf__ float *)raw.GetPhyAddr(),
                              (__ubuf__ float *)mask.GetPhyAddr(), work.chunk.validRows);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReady_[s]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReady_[s]);
            const uint32_t rows = static_cast<uint32_t>(work.chunk.validRows);
            const int64_t aOffset = TokenOutputOffset(*tiling_, work.chunk, work.head, kRows);
            const int64_t tokenOffset = TokenOutputOffset(*tiling_, work.chunk, work.head, kDim);
            AscendC::DataCopy(dAqk_[aOffset], out, rows * kRows);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outFree_[s]);
            AscendC::CrossCoreWaitFlag<kCrossMode, PIPE_MTE3>(kQReady + s);
            AscendC::DataCopy(dqRaw_[tokenOffset], q, rows * kDim);
            AscendC::CrossCoreWaitFlag<kCrossMode, PIPE_MTE3>(kDReady + s);
            AscendC::DataCopy(dv_[tokenOffset], d, rows * kDim);
            // This MTE3 flag follows all three writes. Neither VF completion
            // nor A's earlier write is enough to release the whole group.
            AscendC::CrossCoreSetFlag<kCrossMode, PIPE_MTE3>(kUbFree + s);
        }
        ReleaseEvents<AscendC::HardEvent::V_MTE3>(pipe_, outputReady_, kSlots, false);
        ReleaseEvents<AscendC::HardEvent::MTE3_V>(pipe_, outFree_, kSlots, true);
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    const ChunkKdaBwdPrepareTilingData *tiling_ = nullptr;
    GM_ADDR cu_ = nullptr;
    GM_ADDR indices_ = nullptr;
    AscendC::GlobalTensor<float> dAqk_, dqRaw_;
    AscendC::GlobalTensor<bfloat16_t> dv_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_;
    AscendC::TEventID outputReady_[kSlots], outFree_[kSlots];
};

} // namespace KDA::Arch92
#endif
