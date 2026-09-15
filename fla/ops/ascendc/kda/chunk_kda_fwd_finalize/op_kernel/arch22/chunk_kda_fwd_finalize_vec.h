/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_CHUNK_KDA_FWD_FINALIZE_VEC_H
#define ARCH22_CHUNK_KDA_FWD_FINALIZE_VEC_H

#include "kernel_operator.h"
#include "../chunk_kda_fwd_finalize_struct.h"

namespace KdaFinalize::Arch22 {

template <bool OutputSequenceMajor>
class FinalizeVec {
    static constexpr uint32_t kSlotBytes = 80 * 1024;

public:
    __aicore__ inline void Init(const FinalizeArgs &args, AscendC::TPipe *pipe)
    {
        args_ = args;
        pipe_ = pipe;
        core_ = WorkgroupId();
        aiv_ = AscendC::GetSubBlockIdx();
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.attnOut));
        if (core_ >= args_.tiling.usedCoreNum || args_.tiling.usedCoreNum == 0) {
            return;
        }
        pipe_->InitBuffer(ub_, 2 * kSlotBytes);
        for (uint32_t pair = 0; pair < 2; ++pair) {
            mte2ToV_[pair] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[pair] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToMte2_[pair] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void Process()
    {
        if (core_ >= args_.tiling.usedCoreNum || args_.tiling.usedCoreNum == 0) {
            return;
        }
        constexpr uint16_t kReady[2] = {0, 1};
        constexpr uint16_t kFree[2] = {2, 3};
        bool outputPending[2] = {false, false};
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
                for (uint32_t pair = 0; pair < 2; ++pair) {
                    const uint32_t localHead = pair * 2 + aiv_;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kReady[pair]);
                    if (localHead < active) {
                        if (outputPending[pair]) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[pair]);
                        }
                        // MTE2 只读取本轮两条 FP32 plane；随后 Cube 可重用该 GM 槽。
                        LoadProducts(localHead, pair, chunk.validRows);
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(kFree[pair]);
                        StageV1(chunk, group + localHead, pair);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[pair]);
                        outputPending[pair] = true;
                    } else {
                        // A2/A3 的 mode 2 需要两个 AIV 都参与每个 pair。
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(kFree[pair]);
                    }
                }
            }
        }
        for (uint32_t pair = 0; pair < 2; ++pair) {
            if (outputPending[pair]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[pair]);
            }
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_[pair]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_[pair]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[pair]);
        }
    }

private:
    __aicore__ inline void LoadProducts(uint32_t localHead, uint32_t pair,
                                        uint32_t rows)
    {
        auto bytes = ub_.Get<uint8_t>();
        auto p = bytes[pair * kSlotBytes].template ReinterpretCast<float>();
        auto r = bytes[pair * kSlotBytes + Shape::kProductBytes]
                     .template ReinterpretCast<float>();
        const uint64_t relay = static_cast<uint64_t>(core_) * Shape::kRelayCoreBytes +
                               localHead * Shape::kRelayHeadBytes;
        AscendC::GlobalTensor<float> products;
        products.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + relay));
        AscendC::DataCopy(p, products, rows * Shape::kHeadDim);
        AscendC::DataCopy(r, products[Shape::kProductBytes / sizeof(float)],
                          rows * Shape::kHeadDim);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[pair]);
    }

    __aicore__ inline void StageV1(const FinalizeChunk &chunk,
                                   uint32_t head, uint32_t pair)
    {
        auto bytes = ub_.Get<uint8_t>();
        auto p = bytes[pair * kSlotBytes].template ReinterpretCast<float>();
        auto r = bytes[pair * kSlotBytes + Shape::kProductBytes]
                     .template ReinterpretCast<float>();
        auto result = bytes[pair * kSlotBytes + Shape::kRelayHeadBytes]
                          .template ReinterpretCast<bfloat16_t>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[pair]);
        const uint32_t elements = chunk.validRows * Shape::kHeadDim;
        AscendC::Add(p, p, r, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(result, p, AscendC::RoundMode::CAST_RINT, elements);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[pair]);
        if constexpr (OutputSequenceMajor) {
            for (uint32_t row = 0; row < chunk.validRows; ++row) {
                AscendC::DataCopy(out_[OutputOffset(args_, chunk, head, row)],
                                  result[row * Shape::kHeadDim], Shape::kHeadDim);
            }
        } else {
            AscendC::DataCopy(out_[OutputOffset(args_, chunk, head, 0)],
                              result, elements);
        }
    }

    FinalizeArgs args_{};
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t core_ = 0;
    uint32_t aiv_ = 0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> ub_;
    AscendC::GlobalTensor<bfloat16_t> out_;
    AscendC::TEventID mte2ToV_[2]{};
    AscendC::TEventID vToMte3_[2]{};
    AscendC::TEventID mte3ToMte2_[2]{};
};

} // namespace KdaFinalize::Arch22

#endif // ARCH22_CHUNK_KDA_FWD_FINALIZE_VEC_H
