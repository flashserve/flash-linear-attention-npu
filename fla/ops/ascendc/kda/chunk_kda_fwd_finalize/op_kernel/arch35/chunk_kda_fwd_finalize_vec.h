/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH35_CHUNK_KDA_FWD_FINALIZE_VEC_H
#define ARCH35_CHUNK_KDA_FWD_FINALIZE_VEC_H

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"
#include "../chunk_kda_fwd_finalize_struct.h"

namespace KdaFinalize::Arch35 {

namespace Detail {
using namespace AscendC::MicroAPI;

constexpr CastTrait kRoundLow = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT};
constexpr CastTrait kRoundHigh = {
    RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

__simd_vf__ inline void StageV1Vf(__ubuf__ float *p, __ubuf__ float *r,
                                   __ubuf__ bfloat16_t *out, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg bf16Mask = CreateMask<bfloat16_t, MaskPattern::ALL>();
    // 只有 VF 内的向量运算；循环中没有运行时条件分支。
    for (uint16_t row = 0; row < validRows; ++row) {
        RegTensor<float> pLow;
        RegTensor<float> pHigh;
        RegTensor<float> rLow;
        RegTensor<float> rHigh;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            pLow, pHigh, p + row * Shape::kHeadDim);
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            rLow, rHigh, r + row * Shape::kHeadDim);
        Add(pLow, pLow, rLow, floatMask);
        Add(pHigh, pHigh, rHigh, floatMask);
        RegTensor<bfloat16_t> packed;
        Cast<bfloat16_t, float, kRoundHigh>(packed, pHigh, floatMask);
        Cast<bfloat16_t, float, kRoundLow>(packed, pLow, floatMask);
        StoreAlign(out + row * Shape::kHeadDim, packed, bf16Mask);
    }
}
} // namespace Detail

template <bool OutputSequenceMajor>
class FinalizeVec {
    static constexpr uint32_t kSlotBytes = 80 * 1024;

public:
    __aicore__ inline void Init(const FinalizeArgs &args)
    {
        args_ = args;
        core_ = WorkgroupId();
        aiv_ = AscendC::GetSubBlockIdx();
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.attnOut));
    }

    __aicore__ inline void Process()
    {
        if (core_ >= args_.tiling.usedCoreNum || args_.tiling.usedCoreNum == 0) {
            return;
        }
        constexpr uint16_t kReady[2] = {0, 1};
        constexpr uint16_t kFree[2] = {4, 5};
        const uint32_t partitions = CeilDiv(args_.tiling.valueHeadNum,
                                            args_.tiling.headsPerPartition);
        const uint32_t total = TotalWorkItems(args_);
        const uint32_t begin = WorkBegin(total, core_, args_.tiling.usedCoreNum);
        const uint32_t end = WorkEnd(total, core_, args_.tiling.usedCoreNum);
        for (uint32_t work = begin; work < end; ++work) {
            FinalizeChunk chunk{};
            if (!ResolveChunk(args_, work / partitions, chunk)) {
                continue;
            }
            const uint32_t headBegin = (work % partitions) * args_.tiling.headsPerPartition;
            uint32_t headEnd = headBegin + args_.tiling.headsPerPartition;
            if (headEnd > args_.tiling.valueHeadNum) {
                headEnd = args_.tiling.valueHeadNum;
            }
            for (uint32_t group = headBegin; group < headEnd; group += Shape::kHeadsPerGroup) {
                uint32_t active = headEnd - group;
                if (active > Shape::kHeadsPerGroup) {
                    active = Shape::kHeadsPerGroup;
                }
                for (uint32_t slot = 0; slot < 2; ++slot) {
                    const uint32_t localHead = aiv_ * 2 + slot;
                    if (localHead >= active) {
                        continue;
                    }
                    AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(kReady[slot]);
                    StageV1(chunk, group + localHead, slot);
                    // 保护 AIC 下一轮 Fixpipe 不覆盖仍被 MTE3 消费的 UB。
                    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(kFree[slot]);
                }
            }
        }
    }

private:
    __aicore__ inline void StageV1(const FinalizeChunk &chunk,
                                   uint32_t head, uint32_t slot)
    {
        const uint32_t base = slot * kSlotBytes;
        auto p = resource_.ubBuf.template GetBufferByByte<float>(base);
        auto r = resource_.ubBuf.template GetBufferByByte<float>(base + Shape::kProductBytes);
        auto result = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(
            base + Shape::kRelayHeadBytes);
        AscendC::Mutex::Lock<PIPE_V>(static_cast<uint8_t>(slot));
        asc_vf_call<Detail::StageV1Vf>(
            reinterpret_cast<__ubuf__ float *>(p.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(r.GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(result.GetPhyAddr()),
            static_cast<uint16_t>(chunk.validRows));
        AscendC::Mutex::Unlock<PIPE_V>(static_cast<uint8_t>(slot));

        AscendC::Mutex::Lock<PIPE_MTE3>(static_cast<uint8_t>(slot));
        if constexpr (OutputSequenceMajor) {
            for (uint32_t row = 0; row < chunk.validRows; ++row) {
                AscendC::DataCopy(out_[OutputOffset(args_, chunk, head, row)],
                                  result[row * Shape::kHeadDim], Shape::kHeadDim);
            }
        } else {
            AscendC::DataCopy(out_[OutputOffset(args_, chunk, head, 0)],
                              result, chunk.validRows * Shape::kHeadDim);
        }
        AscendC::Mutex::Unlock<PIPE_MTE3>(static_cast<uint8_t>(slot));
    }

    FinalizeArgs args_{};
    uint32_t core_ = 0;
    uint32_t aiv_ = 0;
    AscendC::GlobalTensor<bfloat16_t> out_;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_;
};

} // namespace KdaFinalize::Arch35

#endif // ARCH35_CHUNK_KDA_FWD_FINALIZE_VEC_H
