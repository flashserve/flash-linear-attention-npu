/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_PREPARE_ARCH35_VECTOR_H
#define CHUNK_KDA_BWD_PREPARE_ARCH35_VECTOR_H

#include <type_traits>

#include "chunk_kda_bwd_prepare_common.h"
#include "kernel_utils/vector/regbase.hpp"

namespace KDA {

using namespace AscendC::MicroAPI;

__simd_vf__ inline void CastBf16ToFloatRegbase(
    __ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint16_t elements)
{
    constexpr uint32_t ELEMS_PER_VF = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    const uint16_t loopCount = static_cast<uint16_t>((elements + ELEMS_PER_VF - 1U) / ELEMS_PER_VF);
    MaskReg maskFp32 = CreateMask<float, MaskPattern::ALL>();
    MaskReg maskBf16 = CreateMask<half, MaskPattern::ALL>();
    RegTensor<bfloat16_t> input;
    RegTensor<float> even;
    RegTensor<float> odd;
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        const uint32_t offset = static_cast<uint32_t>(loop) * ELEMS_PER_VF;
        LoadIn<bfloat16_t, false>(input, src + offset);
        CastHalf2Float<bfloat16_t>(even, odd, input, maskBf16);
        StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst + offset, even, odd, maskFp32);
    }
}

__simd_vf__ inline void BuildTriScaleMaskRegbase(__ubuf__ float *mask, float scale)
{
    constexpr uint32_t ROW = KDA_PREPARE_CHUNK;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    RegTensor<float> scaled;
    Duplicate(zero, 0.0f, full);
    Duplicate(scaled, scale, full);
    for (uint16_t row = 0; row < ROW; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * ROW;
        StoreAlign(mask + offset, zero, full);
        uint32_t lowerCount = static_cast<uint32_t>(row) + 1U;
        MaskReg lower = UpdateMask<float>(lowerCount);
        StoreAlign(mask + offset, scaled, lower);
    }
}

__simd_vf__ inline void ApplyTriScaleMaskRegbase(
    __ubuf__ float *output, __ubuf__ float *mask, uint16_t validRows)
{
    constexpr uint32_t ROW = KDA_PREPARE_CHUNK;
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = static_cast<uint32_t>(validRows);
    MaskReg valid = UpdateMask<float>(validCount);
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, full);
    for (uint16_t row = 0; row < validRows; ++row) {
        const uint32_t offset = static_cast<uint32_t>(row) * ROW;
        RegTensor<float> rawInput;
        RegTensor<float> rawSafe;
        RegTensor<float> maskValue;
        RegTensor<float> result;
        LoadAlign(rawInput, output + offset);
        Select(rawSafe, rawInput, zero, valid);
        LoadAlign(maskValue, mask + offset);
        Mul(result, rawSafe, maskValue, full);
        StoreAlign(output + offset, result, full);
    }
}

class ChunkKdaBwdPrepareVector {
public:
    __aicore__ inline void Init(
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR dAqk, GM_ADDR dv, GM_ADDR dqRaw,
        const ChunkKdaBwdPrepareTilingData *tiling, AscendC::TPipe *pipe)
    {
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        dAqk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dAqk));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dv));
        dqRaw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dqRaw));
        subBlockNum_ = static_cast<uint32_t>(AscendC::GetSubBlockNum());
        if (subBlockNum_ == 0U) {
            subBlockNum_ = 1U;
        }
        subBlockIdx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
        if (subBlockIdx_ >= subBlockNum_) {
            subBlockIdx_ = 0U;
        }
        pipe_->InitBuffer(rawPing_, KDA_PREPARE_RAW_BF16_BYTES);
        pipe_->InitBuffer(rawPong_, KDA_PREPARE_RAW_BF16_BYTES);
        pipe_->InitBuffer(outputPing_, KDA_PREPARE_FP32_BYTES);
        pipe_->InitBuffer(outputPong_, KDA_PREPARE_FP32_BYTES);
        pipe_->InitBuffer(maskBuf_, KDA_PREPARE_FP32_BYTES);
        pipe_->InitBuffer(qRawBuf_, KDA_PREPARE_Q_FP32_BYTES);
        pipe_->InitBuffer(dRawBuf_, KDA_PREPARE_D_BF16_BYTES);
        raw_[0] = rawPing_.Get<bfloat16_t>();
        raw_[1] = rawPong_.Get<bfloat16_t>();
        output_[0] = outputPing_.Get<float>();
        output_[1] = outputPong_.Get<float>();
        mask_ = maskBuf_.Get<float>();
        qRaw_ = qRawBuf_.Get<float>();
        dRaw_ = dRawBuf_.Get<bfloat16_t>();
        for (uint32_t slot = 0; slot < KDA_PREPARE_RAW_SLOT_COUNT; ++slot) {
            vToMte3_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToV_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[slot]);
        }
    }

    __aicore__ inline void Process()
    {
        BuildTriScaleMaskRegbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(mask_.GetPhyAddr()), tiling_->scale);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t slot = 0; slot < KDA_PREPARE_RAW_SLOT_COUNT; ++slot) {
            AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_V>(
                KDA_PREPARE_FREE_FLAG_BASE + slot);
        }
        AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE3>(
            KDA_PREPARE_Q_FREE_FLAG);
        AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE3>(
            KDA_PREPARE_D_FREE_FLAG);

        // In a 1C2V launch GetBlockIdx() is the physical AIV index.  Both
        // vector sub-blocks belong to the same logical AIC task and therefore
        // must normalize it exactly as the mature DHU A5 implementation does.
        const int64_t blockIdx =
            static_cast<int64_t>(AscendC::GetBlockIdx() / subBlockNum_);
        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        uint64_t generation = 0;
        for (int64_t workTask = blockIdx; workTask < tiling_->workTaskNum;
             workTask += blockNum) {
            // Mirror the AIC head-major task order exactly so cross-core slots
            // and generation counters stay paired with the same work item.
            const int64_t headWindow = workTask / tiling_->chunkTaskNum;
            const int64_t chunkTask = workTask - headWindow * tiling_->chunkTaskNum;
            const int64_t headBegin = headWindow * HEADS_PER_WORK_TASK;
            const int64_t headEnd = KdaMin(headBegin + HEADS_PER_WORK_TASK, tiling_->NV);
            ChunkInfo chunk;
            ResolveChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            for (int64_t head = headBegin; head < headEnd; ++head, ++generation) {
                const uint32_t aivIdx = static_cast<uint32_t>(generation & 1U);
                if (aivIdx != subBlockIdx_) {
                    continue;
                }
                // Each AIV sees every other head and alternates its own two UB
                // slots: AIV0 h0/h2, AIV1 h1/h3, then wrap.
                const uint32_t slot = static_cast<uint32_t>((generation >> 1U) & 1U);
                AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_V>(
                    KDA_PREPARE_READY_FLAG_BASE + slot);
                const int64_t out = TokenOffset(*tiling_, chunk, head, tiling_->chunkSize);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[slot]);
                CastBf16ToFloatRegbase(
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(output_[slot].GetPhyAddr()),
                    (__ubuf__ bfloat16_t *)reinterpret_cast<uint64_t>(raw_[slot].GetPhyAddr()),
                    static_cast<uint16_t>(chunk.validRows * tiling_->chunkSize));
                ApplyTriScaleMaskRegbase(
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(output_[slot].GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(mask_.GetPhyAddr()),
                    static_cast<uint16_t>(chunk.validRows));
                // The raw BF16 UB slot is free after the Vector pipeline has
                // consumed it; MTE3 drains from the separate FP32 output slot.
                AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_V>(
                    KDA_PREPARE_FREE_FLAG_BASE + slot);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
                AscendC::DataCopy(dAqk_[out], output_[slot],
                                  static_cast<uint32_t>(chunk.validRows * tiling_->chunkSize));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[slot]);

                // Q/D are already in their public output dtypes.  Keep the
                // transfer entirely on MTE3 so Vector does not serialize the
                // A post-process, and publish FREE from PIPE_MTE3 only after
                // the corresponding GM write has completed.
                const int64_t qdOut = TokenOffset(*tiling_, chunk, head, tiling_->V);
                AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE3>(
                    KDA_PREPARE_Q_READY_FLAG);
                AscendC::DataCopy(
                    dqRaw_[qdOut], qRaw_,
                    static_cast<uint32_t>(chunk.validRows * tiling_->V));
                AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE3>(
                    KDA_PREPARE_Q_FREE_FLAG);

                AscendC::CrossCoreWaitFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE3>(
                    KDA_PREPARE_D_READY_FLAG);
                AscendC::DataCopy(
                    dv_[qdOut], dRaw_,
                    static_cast<uint32_t>(chunk.validRows * tiling_->V));
                AscendC::CrossCoreSetFlag<KDA_PREPARE_CROSS_CORE_MODE, PIPE_MTE3>(
                    KDA_PREPARE_D_FREE_FLAG);
            }
        }
        for (uint32_t slot = 0; slot < KDA_PREPARE_RAW_SLOT_COUNT; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_[slot]);
        }
    }

private:
    static constexpr int64_t HEADS_PER_WORK_TASK = 4;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkKdaBwdPrepareTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t subBlockIdx_ = 0;
    uint32_t subBlockNum_ = 1;
    AscendC::GlobalTensor<float> dAqk_;
    AscendC::GlobalTensor<bfloat16_t> dv_;
    AscendC::GlobalTensor<float> dqRaw_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPing_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputPong_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dRawBuf_;
    AscendC::LocalTensor<bfloat16_t> raw_[KDA_PREPARE_RAW_SLOT_COUNT];
    AscendC::LocalTensor<float> output_[KDA_PREPARE_RAW_SLOT_COUNT];
    AscendC::LocalTensor<float> mask_;
    AscendC::LocalTensor<float> qRaw_;
    AscendC::LocalTensor<bfloat16_t> dRaw_;
    AscendC::TEventID vToMte3_[KDA_PREPARE_RAW_SLOT_COUNT];
    AscendC::TEventID mte3ToV_[KDA_PREPARE_RAW_SLOT_COUNT];
};

} // namespace KDA

#endif
