#pragma once
#include "int8_qkv.h"
#include "mega_recurrent_kda_v2_t02.h"

namespace MegaRecurrentKdaV2::LongSequence {
constexpr uint32_t kHistoryVectors = 9; // Three Q/K/V groups in three convolution history rows.
constexpr uint32_t kHistoryElements = kHistoryVectors * RDetail::kD;

// A value row depends on all 128 keys and only on that same value row of the preceding token.
// At low sequence/head occupancy, independent AIV groups can own disjoint value rows.
// The existing output allocation holds ungated FP32 values and the initial history until stage O.
CATLASS_DEVICE uint32_t GroupCount(const V2Params &p)
{
    const uint64_t units = uint64_t(p.batch) * p.n;
    uint32_t groups = 16;
    while (groups > 1 && units * groups > p.aivNum) {
        groups >>= 1;
    }
    return groups;
}
// Snapshot before any value owner writes the convolution history back in place.
CATLASS_DEVICE void CopyHistory(const V2Params &params, uint32_t id,
                                Catlass::Arch::Resource<Catlass::Arch::AtlasA2> &resource)
{
    uint32_t b = id / params.n, h = id % params.n;
    if (SequenceStart(params, b) == SequenceStart(params, b + 1)) {
        return;
    }
    auto slots = Int8Qkv::Global<int32_t>(params.stateIndices);
    auto accepted = Int8Qkv::Global<int32_t>(params.accepted);
    uint32_t count = accepted.GetValue(b), slot = slots.GetValue(b);
    auto cs = Int8Qkv::Global<bfloat16_t>(params.convState);
    auto hist = Int8Qkv::Global<bfloat16_t>(params.out + uint64_t(params.tokens) * params.p * sizeof(float));
    auto raw = resource.ubBuf.template GetBufferByByte<bfloat16_t>(0);
    const uint32_t first = count > 1 ? count - 1 : 0;
    const uint64_t offset = uint64_t(slot) * params.convStride + uint64_t(first) * 3 * params.p + h * RDetail::kD;
    AscendC::DataCopy(
        raw, cs[offset],
        AscendC::DataCopyParams{kHistoryVectors, RDetail::kD / 16, uint16_t((params.p - RDetail::kD) / 16), 0});
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(hist[uint64_t(id) * kHistoryElements], raw, kHistoryElements);
    AscendC::PipeBarrier<PIPE_ALL>();
}
// Normalize all 128 values together in their original FP32 reduction order.
CATLASS_DEVICE void FinishGatedRows(const V2Params &params, uint32_t id, AscendC::LocalTensor<float> f)
{
    constexpr uint32_t rowsPerGroup = 8, width = 128, elems = rowsPerGroup * width;
    auto y = f, square = f[elems], gate = f[2 * elems], ones = f[3 * elems];
    auto gamma = f[4 * elems], part = gamma[width], sums = part[16 * rowsPerGroup];
    auto bf = sums[16].template ReinterpretCast<bfloat16_t>();
    auto results = Int8Qkv::Global<float>(params.out);
    auto gated = Int8Qkv::Global<bfloat16_t>(params.wGated);
    auto go = Int8Qkv::Global<bfloat16_t>(params.wGateRaw);
    AscendC::DataCopy(gamma, Int8Qkv::Global<float>(params.gamma), width);
    AscendC::Duplicate(ones, 1.f, elems);
    const uint64_t tasks = uint64_t(CeilDiv(params.tokens, rowsPerGroup)) * params.n;
    for (uint64_t task = id; task < tasks; task += params.aivNum) {
        const uint32_t token = static_cast<uint32_t>(task / params.n) * rowsPerGroup;
        const uint32_t h = static_cast<uint32_t>(task % params.n);
        const uint32_t rows = params.tokens - token < rowsPerGroup ? params.tokens - token : rowsPerGroup;
        const uint64_t offset = uint64_t(token) * params.p + h * width;
        const uint32_t count = rows * width;
        AscendC::DataCopyParams floats{uint16_t(rows), width / 8, uint16_t((params.p - width) / 8), 0};
        AscendC::DataCopyParams halves{uint16_t(rows), width / 16, uint16_t((params.p - width) / 16), 0};
        AscendC::DataCopy(y, results[offset], floats);
        AscendC::DataCopy(bf, go[offset], halves);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(square, y, y, count);
        AscendC::Cast(gate, bf, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        RDetail::RowReduce128(sums, sums[8], square, part, rows);
        AscendC::Muls(gate, gate, -1.f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(sums, sums, 0.0078125f, 8);
        AscendC::Exp(gate, gate, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(sums, sums, params.rmsNormEps, 8);
        AscendC::Adds(gate, gate, 1.f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(sums, sums, 8);
        AscendC::Div(gate, ones, gate, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(sums, ones, sums, 8);
        const AscendC::BinaryRepeatParams broadcast(1, 1, 1, 16, 16, 0);
        AscendC::Mul(gate, gate, gamma, 64, rows, broadcast);
        AscendC::Mul(gate[64], gate[64], gamma[64], 64, rows, broadcast);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t r = 0; r < rows; ++r) {
            float inv = sums.GetValue(r);
            if (inv > 1e6f) {
                inv = 1e6f;
            }
            AscendC::Muls(y[r * width], y[r * width], inv, width);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(y, y, gate, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(bf, y, AscendC::RoundMode::CAST_RINT, count);
        AscendC::PipeBarrier<PIPE_ALL>();
        halves.dstStride = halves.srcStride;
        halves.srcStride = 0;
        AscendC::DataCopy(gated[offset], bf, halves);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
CATLASS_DEVICE void Run(V2Params &params, bool overlapAuxiliary, uint32_t team)
{
    const uint64_t units = uint64_t(params.batch) * params.n;
    if ASCEND_IS_AIC {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        if (!overlapAuxiliary) {
            for (uint32_t block = 0; block < params.pBlocks; ++block) {
                RunStageP(resource, params, block);
            }
        }
    }
    if ASCEND_IS_AIV {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        uint32_t id = AscendC::GetBlockIdx();
        if (id < units) {
            CopyHistory(params, id, resource);
        }
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        uint32_t id = AscendC::GetBlockIdx();
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        auto ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
        if (id < units * team) {
            uint32_t group = id / team, b = group / params.n, h = group % params.n;
            HeadConstants cached;
            const uint32_t rows = RDetail::kD / team;
            ProcessSequence<false, true>(params, ub, b, h, SequenceStart(params, b), SequenceStart(params, b + 1),
                                         cached, (id % team) * rows, rows);
        }
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        uint32_t id = AscendC::GetBlockIdx();
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        auto f = resource.ubBuf.template GetBufferByByte<float>(0);
        FinishGatedRows(params, id, f);
    }
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIC {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        params.skipSync |= 2u; // All gated rows became visible at the preceding grid barrier.
        for (uint32_t block = 0; block < params.oBlocks; ++block) {
            RunStageO(resource, params, block);
        }
    }
}
} // namespace MegaRecurrentKdaV2::LongSequence
