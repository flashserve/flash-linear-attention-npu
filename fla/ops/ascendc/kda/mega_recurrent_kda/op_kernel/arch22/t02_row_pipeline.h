#pragma once

// INT8 row panels use the common tiling parameters and vector recurrence.
namespace MegaRecurrentKdaV2 {
namespace T02Rows {
using Resource = Catlass::Arch::Resource<Int8Qkv::Arch>;
using namespace AscendC;
constexpr uint32_t kRows = 16;  // Vector quantization, recurrence and gated publication.
constexpr uint32_t kCubeRows = 2 * kRows;
constexpr bool kGroupOutputProjection = false;

CATLASS_DEVICE uint64_t Slab(uint64_t bytes) { return (bytes + 511) / 512 * 512; }

struct Buffers {
    GM_ADDR codes;
    GM_ADDR scales;
    GM_ADDR decayA;
    GM_ADDR dots;
    uint64_t bytes;

    CATLASS_DEVICE Buffers(V2Params &p, GM_ADDR ws) {
        const uint64_t rows = p.tokens, width = p.p;
        codes = ws;
        scales = codes + Slab(rows * p.hidden);
        decayA = scales + Slab(rows * 32);
        dots = decayA + Slab(rows * 256);
        p.wQkv = dots;
        p.wDecayRaw = dots + rows * 3 * width * 4;
        p.wGateRaw = p.wDecayRaw + rows * width * 2;
        p.wBetaRaw = p.wGateRaw + rows * width * 2;
        const uint64_t projBytes = rows * (16 * width + 32);
        p.wGated = dots + Slab(projBytes);
        bytes = uint64_t(p.wGated - ws) + Slab(rows * width * 2);
    }
};

enum PanelSignal { Quantized = 0, AuxReady = 1, Gated = 2, FirstHead = 3 };
enum HeadSignal { Kv = 0, Q = 1, Decay = 2, GateReady = 3 };
constexpr uint32_t kSignalsPerPanel = FirstHead + 4 * 8;
CATLASS_DEVICE uint32_t Event(uint32_t row, uint32_t kind) {
    return (row / kRows * kSignalsPerPanel + kind) * 64;
}
CATLASS_DEVICE uint32_t HeadEvent(uint32_t row, uint32_t head, HeadSignal signal) {
    return Event(row, FirstHead + 4 * head + signal);
}
CATLASS_DEVICE void CubeSignal(const V2Params &p, Resource &r, uint32_t offset) {
    PipeBarrier<PIPE_ALL>();
    Sync::AicAdd(r.l1Buf.GetBufferByByte<int32_t>(0), p.syncConst, p.counts, offset, 1);
}
CATLASS_DEVICE void VecSignal(const V2Params &p, Resource &r, uint32_t offset) {
    PipeBarrier<PIPE_ALL>();
    Sync::AivAdd(r.ubBuf.GetBufferByByte<int32_t>(0), p.counts, offset, 1);
}

CATLASS_DEVICE void Quant(const V2Params &p, const Buffers &ws, Resource &r, uint32_t row, uint32_t count) {
    V2Params tile = p;
    tile.tokens = count;
    tile.hiddenStates += uint64_t(row) * p.hidden * 2;
    tile.out = ws.codes + uint64_t(row) * p.hidden;
    Int8Qkv::QuantizeRows(tile, r.ubBuf.GetBufferByByte<uint8_t>(0), ws.scales + row * 32);
}

// Each Cube owns whole output columns. Retain the BF16 conversion between decay-A and decay-B.
CATLASS_DEVICE void Aux(const V2Params &p, const Buffers &w, Resource &r, uint32_t row, uint32_t rows) {
    const uint32_t core = GetBlockIdx();
    // Auxiliary GEMMs have at most 32 rows and 16 output columns per owner.
    using AuxMmad = Catlass::Gemm::Block::BlockMmadTla<typename KdaGemm::DispatchPolicy,
        tla::Shape<tla::_32, tla::_16, tla::Int<1024>>, tla::Shape<tla::_32, tla::_16, tla::_64>,
        bfloat16_t, bfloat16_t, bfloat16_t, void, typename KdaGemm::TileCopy>;
    AuxMmad mm(r);
    GM_ADDR x = p.hiddenStates + uint64_t(row) * p.hidden * 2;
    if (core == 0) {
        RunGemmReuse(mm, x, rows, p.hidden, p.betaWeight, p.n, p.hidden,
                     p.wBetaRaw + uint64_t(row) * p.n * 2, p.n, p.hidden);
    } else {
        for (uint32_t col = (core - 1) * 16; col < kHeadDim; col += (p.aicNum - 1) * 16) {
            RunGemmReuse(mm, x, rows, p.hidden, p.decayAWeight + uint64_t(col) * p.hidden * 2,
                         16, p.hidden, w.decayA + (uint64_t(row) * kHeadDim + col) * 2, kHeadDim, p.hidden);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

CATLASS_DEVICE void Dot(const V2Params &p, const Buffers &ws, Resource &r, uint32_t row, uint32_t rows,
                        uint32_t beginPlane, uint32_t endPlane, bool decay, bool &quantReady) {
    const uint32_t jobs = (endPlane - beginPlane) * p.n;
    // Q has its own output columns. Continue core numbering after K/V and decay-B jobs.
    const uint32_t firstOwner = beginPlane == 0 ? (3 * p.n) % p.aicNum : 0;
    const uint32_t core = GetBlockIdx();
    const uint32_t firstJob = core >= firstOwner ? core - firstOwner : core + p.aicNum - firstOwner;
    for (uint32_t job = firstJob; job < jobs + (decay ? p.n : 0); job += p.aicNum) {
        if (job >= jobs) {
            Sync::WaitEq(p.counts,Event(row,AuxReady),p.aicNum < 9 ? p.aicNum : 9);
            const uint32_t col = (job - jobs) * kHeadDim;
            KdaGemm::BlockMmad mm(r);
            RunGemmReuse(mm, ws.decayA + uint64_t(row) * kHeadDim * 2, rows, kHeadDim,
                         p.decayBWeight + uint64_t(col) * kHeadDim * 2, kHeadDim, kHeadDim,
                         p.wDecayRaw + (uint64_t(row) * p.p + col) * 2, p.p, kHeadDim);
            CubeSignal(p,r,HeadEvent(row,job-jobs,Decay));
        } else {
            if (!quantReady) {
                for (uint32_t panel = row; panel < row + rows; panel += kRows) {
                    const uint32_t pending = row + rows - panel;
                    const uint32_t count = pending < kRows ? pending : kRows;
                    Sync::WaitEq(p.counts, Event(panel, Quantized), count < p.aivNum ? count : p.aivNum);
                }
                quantReady = true;
            }
            const uint32_t col = beginPlane * p.p + job * kHeadDim;
            auto a = Int8Qkv::Global<int8_t>(ws.codes + uint64_t(row) * p.hidden);
            auto b = Int8Qkv::Global<int8_t>(p.qkvWeight + uint64_t(col) * p.hidden);
            auto c = Int8Qkv::Global<int32_t>(ws.dots + (uint64_t(row) * 3 * p.p + col) * 4);
            auto ta = tla::MakeTensor(a, tla::MakeLayout<int8_t, Int8Qkv::Row>(rows,p.hidden),Catlass::Arch::PositionGM{});
            auto tb = tla::MakeTensor(b, tla::MakeLayout<int8_t, Int8Qkv::Nz>(p.hidden,kHeadDim),Catlass::Arch::PositionGM{});
            auto tc = tla::MakeTensor(c, KdaGemm::MakeLayoutC(rows,kHeadDim,3*p.p),Catlass::Arch::PositionGM{});
            Int8Qkv::Mmad mm(r);
            mm(ta,tb,tc,Catlass::GemmCoord(rows,kHeadDim,p.hidden));
            CubeSignal(p,r,HeadEvent(row,job % p.n,beginPlane == 0 ? Q : Kv));
        }
    }
    PipeBarrier<PIPE_ALL>();
}

CATLASS_DEVICE void Gate(const V2Params &p, Resource &r, uint32_t row, uint32_t rows) {
    KdaGemm::BlockMmad mm(r);
    // Each gate head has two independent 64-column panels. Use spare owners for
    // the first pair so they can run while the original owners finish Q/K.
    uint32_t owner = GetBlockIdx();
    const uint32_t panels = 2 * p.n;
    if (p.aicNum >= panels + 2) {
        if (owner < 2) {
            return;
        }
        if (owner >= panels && owner < panels + 2) {
            owner -= panels;
        }
    }
    for (uint32_t col = owner * 64; col < p.p; col += p.aicNum*64) {
        RunGemmReuse(mm, p.hiddenStates + uint64_t(row)*p.hidden*2, rows, p.hidden,
                     p.gateWeight+uint64_t(col)*p.hidden*2,64,p.hidden,
                     p.wGateRaw+(uint64_t(row)*p.p+col)*2,p.p,p.hidden);
        CubeSignal(p,r,HeadEvent(row,col/kHeadDim,GateReady));
    }
    PipeBarrier<PIPE_ALL>();
}

CATLASS_DEVICE void Output(const V2Params &p, Resource &r, uint32_t row, uint32_t rows) {
    RunOutputPanels<KdaGemm::BlockMmad,kHeadDim>(r,p,p.wGated+uint64_t(row)*p.p*2,
        p.out+uint64_t(row)*p.hidden*2,rows,GetBlockIdx(),p.aicNum);
    PipeBarrier<PIPE_ALL>();
}

CATLASS_DEVICE void ReadDots(const V2Params &p, const Buffers &ws, Resource &r,
                             uint32_t first, uint32_t count, uint32_t head,
                             uint32_t beginPlane, uint32_t endPlane) {
    // Dot conversion scratch shares no live data with the two resident checkpoint halves.
    auto ub = r.ubBuf.GetBufferByByte<uint8_t>(RDetail::VectorLayout<false>::kUbScratch);
    constexpr uint32_t kBatchRows = 4;
    constexpr uint32_t kWidth = kHeadDim;
    constexpr uint32_t kFpTileBytes = kBatchRows * kWidth * sizeof(float);
    constexpr uint32_t kBfTileBytes = kBatchRows * kWidth * sizeof(bfloat16_t);
    // Convert consecutive tokens of this request together. The state update still advances one token at a time.
    auto dots = ub.ReinterpretCast<int32_t>();
    auto fp = ub[kFpTileBytes].ReinterpretCast<float>();
    auto scale = ub[2 * kFpTileBytes].ReinterpretCast<float>();
    auto bf = ub[3 * kFpTileBytes].ReinterpretCast<bfloat16_t>();
    auto channel = ub[3 * kFpTileBytes + kBfTileBytes].ReinterpretCast<float>();
    auto sx = Int8Qkv::Global<float>(ws.scales);
    for (uint32_t plane = beginPlane; plane < endPlane; ++plane) {
        const uint32_t col = plane * p.p + head * kWidth;
        DataCopy(bf, Int8Qkv::Global<bfloat16_t>(p.weightScale + col * 2), kWidth);
        PipeBarrier<PIPE_ALL>();
        Cast(channel, bf, RoundMode::CAST_NONE, kWidth);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < count; i += kBatchRows) {
            const uint32_t rows = count - i < kBatchRows ? count - i : kBatchRows;
            const uint32_t elements = rows * kWidth;
            const uint32_t rowStrideBlocks = 3 * p.p * sizeof(int32_t) / 32;
            GM_ADDR panel = ws.dots + (uint64_t(first + i) * 3 * p.p + col) * 4;
            DataCopyParams read{static_cast<uint16_t>(rows), static_cast<uint16_t>(kWidth / 8),
                                static_cast<uint16_t>(rowStrideBlocks - kWidth / 8), 0};
            DataCopy(dots, Int8Qkv::Global<int32_t>(panel), read);
            for (uint32_t j = 0; j < rows; ++j) {
                Muls(scale[j * kWidth], channel, sx.GetValue((first + i + j) * 8), kWidth);
            }
            PipeBarrier<PIPE_ALL>();
            Cast(fp, dots, RoundMode::CAST_RINT, elements);
            PipeBarrier<PIPE_V>();
            Mul(fp, fp, scale, elements);
            PipeBarrier<PIPE_V>();
            Cast(bf, fp, RoundMode::CAST_RINT, elements);
            PipeBarrier<PIPE_ALL>();
            DataCopyParams write{static_cast<uint16_t>(rows), static_cast<uint16_t>(kWidth / 16),
                                 0, static_cast<uint16_t>(rowStrideBlocks - kWidth / 16)};
            DataCopy(Int8Qkv::Global<bfloat16_t>(panel), bf, write);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

CATLASS_DEVICE void Produce(const V2Params &p, const Buffers &w, Resource &r, uint32_t row, uint32_t rows) {
    Aux(p,w,r,row,rows);
    if (GetBlockIdx() < 9) CubeSignal(p,r,Event(row,AuxReady));
    bool quantReady = false;
    Dot(p,w,r,row,rows,1,3,true,quantReady);
    Dot(p,w,r,row,rows,0,1,false,quantReady);
    Gate(p,r,row,rows);
}
CATLASS_DEVICE void Consume(V2Params p, const Buffers &w, Resource &r, uint32_t row, uint32_t rows, uint32_t seq) {
    // The same producer publishes each head once for both 16-row consumers.
    const uint32_t projectionRow = row / kCubeRows * kCubeRows;
    const uint32_t chains = rows / seq * p.n;
    for (uint32_t job = GetBlockIdx(); job < chains; job += p.aivNum) {
        const uint32_t head = job % p.n, first = row + job / p.n * seq;
        using Layout = RDetail::VectorLayout<false>;
        auto ub = r.ubBuf.GetBufferByByte<uint8_t>(0);
        const auto slot = Int8Qkv::Global<int32_t>(p.stateIndices).GetValue(first / seq);
        HeadConstants cache;
        LoadCheckpoint(p, ub[Layout::kUbh0].ReinterpretCast<float>(),
                       ub[Layout::kUbh1].ReinterpretCast<float>(), slot, head, 0, kHeadDim);
        PrepareHeadConstants<false>(p, ub, head, cache);
        Sync::WaitEq(p.counts,HeadEvent(projectionRow,head,Kv),2);
        ReadDots(p,w,r,first,seq,head,1,3);
        Sync::WaitEq(p.counts,HeadEvent(projectionRow,head,Q),1);
        ReadDots(p,w,r,first,seq,head,0,1);
        Sync::WaitEq(p.counts,HeadEvent(projectionRow,head,Decay),1);
        p.rowGateOffset = HeadEvent(projectionRow,head,GateReady);
        ProcessSequence<false,false,true,true>(p,r.ubBuf.GetBufferByByte<uint8_t>(0),first/seq,head,first,first+seq,cache);
        VecSignal(p,r,Event(row,Gated));
    }
}

// One window and one head group; each sequence must fit in its row panel.
CATLASS_DEVICE bool Try(const MegaRecurrentKdaArch22TilingData &plan, V2Params p, GM_ADDR metadata, GM_ADDR workspace) {
    if (!p.weightScale || p.n > 8 || p.tokens > kTileRows || p.aicNum < 2 || !p.batch) return false;
    uint32_t seq = p.sequence;
    if (!seq && plan.common.crosses == 2 && metadata) seq = reinterpret_cast<__gm__ const int32_t *>(metadata)[1];
    if (!seq || uint64_t(seq)*p.batch != p.tokens || kRows % seq != 0) return false;
    Buffers w(p,workspace);
    const uint32_t panels = (p.tokens+kRows-1)/kRows;
    const uint32_t controlBytes = panels * kSignalsPerPanel * 64;
    if (w.bytes+controlBytes+64 > plan.arch22.workspaceBytes) return false;
    p.counts = workspace + w.bytes;
    p.syncConst = p.counts + controlBytes;
    p.wideInt8Panels = true;
    p.skipSync |= 3u;
    Resource r;
    if ASCEND_IS_AIV {
        if (GetBlockIdx()==0) {
            auto z = r.ubBuf.GetBufferByByte<int32_t>(0);
            Duplicate(z,0,controlBytes/4+16);
            PipeBarrier<PIPE_ALL>();
            z.SetValue(controlBytes/4,1);
            PipeBarrier<PIPE_ALL>();
            DataCopy(Int8Qkv::Global<int32_t>(p.counts),z,controlBytes/4+16);
            PipeBarrier<PIPE_ALL>();
        }
    }
    SyncAll<false>();
    for (uint32_t start = 0; start<p.tokens; start+=kCubeRows) {
        const uint32_t end = p.tokens-start < kCubeRows ? p.tokens : start+kCubeRows;
        if ASCEND_IS_AIC {
            Produce(p, w, r, start, end - start);
            for (uint32_t row=start;row<end;row+=kRows) {
                const uint32_t rows = end-row < kRows ? end-row : kRows;
                Sync::WaitEq(p.counts,Event(row,Gated),rows/seq*p.n);
                if constexpr (!kGroupOutputProjection) {
                    Output(p, r, row, rows);
                }
            }
            if constexpr (kGroupOutputProjection) {
                Output(p, r, start, end - start);
            }
        }
        if ASCEND_IS_AIV {
            for (uint32_t row=start;row<end;row+=kRows) {
                const uint32_t rows = end-row < kRows ? end-row : kRows;
                Quant(p,w,r,row,rows);
                if (GetBlockIdx()<rows) VecSignal(p,r,Event(row,Quantized));
            }
            for (uint32_t row=start;row<end;row+=kRows) {
                const uint32_t rows = end-row < kRows ? end-row : kRows;
                Consume(p,w,r,row,rows,seq);
            }
        }
    }
    // No later operation in this kernel reads another core's outputs or reuses these slots.
    // Complete this core's writes before it exits; the stream observes completion of the whole kernel.
    PipeBarrier<PIPE_ALL>();
    return true;
}
} // namespace T02Rows
} // namespace MegaRecurrentKdaV2
