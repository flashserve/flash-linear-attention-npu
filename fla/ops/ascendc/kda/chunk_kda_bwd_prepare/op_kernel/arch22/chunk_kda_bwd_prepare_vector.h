#ifndef CHUNK_KDA_BWD_PREPARE_ARCH22_VECTOR_H
#define CHUNK_KDA_BWD_PREPARE_ARCH22_VECTOR_H
#include "chunk_kda_bwd_prepare_common.h"
namespace KDA {
class ChunkKdaBwdPrepareVector {
public:
    __aicore__ inline void Init(GM_ADDR cu, GM_ADDR indices, GM_ADDR out, GM_ADDR scratch,
        const ChunkKdaBwdPrepareTilingData *tiling, AscendC::TPipe *pipe)
    {
        cu_ = cu; indices_ = indices; t_ = tiling; pipe_ = pipe;
        raw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        out_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe_->InitBuffer(ub_, UB_BYTES);
        AllocateEvents<AscendC::HardEvent::V_MTE2>(pipe_, rawFree_, 2, true);
        AllocateEvents<AscendC::HardEvent::MTE2_V>(pipe_, rawReady_, 2, false);
        AllocateEvents<AscendC::HardEvent::MTE3_V>(pipe_, outFree_, 2, true);
        AllocateEvents<AscendC::HardEvent::V_MTE3>(pipe_, outReady_, 2, false);
        auto mask = ub_.Get<uint64_t>()[4 * RAW_ELEMENTS * sizeof(float) / sizeof(uint64_t)];
        for (uint32_t row = 0; row < 64; ++row)
            mask.SetValue(row, row == 63 ? ~uint64_t{0} : (uint64_t{1} << (row + 1)) - 1);
        const auto maskReady = pipe_->AllocEventID<AscendC::HardEvent::S_V>();
        AscendC::SetFlag<AscendC::HardEvent::S_V>(maskReady);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(maskReady);
        pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(maskReady);
        for (uint16_t record = 0; record < 4; ++record)
            AscendC::CrossCoreSetFlag<CROSS_MODE, PIPE_MTE2>(FREE_BASE + record);
    }
    __aicore__ inline void Process()
    {
        const int64_t block = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const int64_t blocks = AscendC::GetBlockNum();
        const uint32_t aiv = AscendC::GetSubBlockIdx();
        uint64_t generation = 0;
        for (int64_t work = block; work < t_->workTaskNum; work += blocks) {
            const int64_t begin = (work / t_->chunkTaskNum) * 4;
            const int64_t end = KdaMin(begin + 4, t_->NV);
            ChunkInfo chunk;
            ResolveChunk(work % t_->chunkTaskNum, cu_, indices_, *t_, chunk);
            if (!chunk.valid) continue;
            for (int64_t head = begin; head < end; ++head, ++generation) {
                const uint32_t target = generation & 1, slot = (generation >> 1) & 1;
                const uint32_t record = 2 * target + slot;
                AscendC::CrossCoreWaitFlag<CROSS_MODE, PIPE_MTE2>(READY_BASE + record);
                if (target != aiv) {
                    AscendC::CrossCoreSetFlag<CROSS_MODE, PIPE_MTE2>(FREE_BASE + record);
                    continue;
                }
                const uint32_t rows = chunk.validRows, aligned = (rows + 7) / 8 * 8;
                auto raw = ub_.Get<float>()[slot * RAW_ELEMENTS];
                auto dst = ub_.Get<float>()[2 * RAW_ELEMENTS + slot * RAW_ELEMENTS];
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawFree_[slot]);
                AscendC::DataCopyExtParams p{static_cast<uint16_t>(rows), rows * 4,
                                            (64 - rows) * 4, (64 - aligned) / 8, 0};
                AscendC::DataCopyPadExtParams<float> pad{true, 0,
                    static_cast<uint8_t>(aligned - rows), 0.0f};
                AscendC::DataCopyPad(raw, raw_[(block * 4 + record) * RECORD_ELEMENTS], p, pad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawReady_[slot]);
                AscendC::CrossCoreSetFlag<CROSS_MODE, PIPE_MTE2>(FREE_BASE + record);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawReady_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outFree_[slot]);
                if (rows == 64) {
                    // The full raw tile is initialized by Fixpipe. Select
                    // once over the entire tile; masked NaN/Inf become exact
                    // zeros instead of relying on multiplication by zero.
                    auto mask = ub_.Get<uint8_t>()[4 * RAW_ELEMENTS * sizeof(float)];
                    AscendC::Select(dst, mask, raw, 0.0f,
                                   AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, RAW_ELEMENTS);
                } else {
                    // Tail workspace columns are unwritten. Read only each
                    // row's valid lower triangle and explicitly zero the rest.
                    AscendC::Duplicate(dst, 0.0f, rows * 64);
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t row = 0; row < rows; ++row)
                        AscendC::Adds(dst[row * 64], raw[row * 64], 0.0f, row + 1);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawFree_[slot]);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(dst, dst, t_->scale, rows * 64);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outReady_[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outReady_[slot]);
                AscendC::DataCopy(out_[TokenOutputOffset(*t_, chunk, head, 64)], dst, rows * 64);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outFree_[slot]);
            }
        }
        ReleaseEvents<AscendC::HardEvent::V_MTE2>(pipe_, rawFree_, 2, true);
        ReleaseEvents<AscendC::HardEvent::MTE2_V>(pipe_, rawReady_, 2, false);
        ReleaseEvents<AscendC::HardEvent::MTE3_V>(pipe_, outFree_, 2, true);
        ReleaseEvents<AscendC::HardEvent::V_MTE3>(pipe_, outReady_, 2, false);
    }
private:
    AscendC::TPipe *pipe_;
    const ChunkKdaBwdPrepareTilingData *t_;
    GM_ADDR cu_;
    GM_ADDR indices_;
    AscendC::GlobalTensor<float> raw_, out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ub_;
    AscendC::TEventID rawFree_[2], rawReady_[2], outFree_[2], outReady_[2];
};
}
#endif
