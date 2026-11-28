#pragma once
#include "mega_recurrent_kda_v2_cube.h"
#include "mega_recurrent_kda_v2_vector.h"
#include "tiling_data.h"
#include "int8_qkv.h"
#include "long_sequence.h"
#include "t02_row_pipeline.h"
namespace MegaRecurrentKdaV2 {
CATLASS_DEVICE void BindTilingAndWorkspace(const MegaRecurrentKdaArch22TilingData &plan, V2Params &params,
                                           GM_ADDR workspace)
{
    const auto &t = plan.arch22;
    const auto &data = plan.common;
    params.tokens = t.tokens;
    params.tiles = t.tiles;
    params.n = t.n;
    params.p = t.p;
    params.hidden = t.hidden;
    params.pBlocks = t.pBlocks;
    params.pSplit = t.pSplit;
    params.oBlocks = t.oBlocks;
    params.nQkv = t.nQkv;
    params.nGate = t.nGate;
    params.nBeta = t.nBeta;
    params.nDecayRaw = t.nDecayRaw;
    params.nProj = t.nProj;
    params.nOut = t.nOut;
    params.aicNum = t.aicNum;
    params.aivNum = t.aivNum;
    params.slotsPR = t.slotsPR;
    params.slotsGated = t.slotsGated;
    params.templateId = t.templateId;
    params.blockNum = plan.poolSlots;
    params.replayCapacity = data.capacity;
    params.batch = data.batch;
    params.sequence = data.sequence;
    params.convLength = plan.convLength;
    params.convLayout = data.convLayout;
    params.scale = data.scale;
    params.lowerBound = data.lowerBound;
    params.rmsNormEps = data.rmsNormEps;
    params.convStride = data.convStride;
    params.checkpointStride = data.checkpointStride;
    params.uStride = data.uStride;
    params.keyStride = data.keyStride;
    params.decayStride = data.decayStride;
    params.wQkv = workspace + t.offQkv;
    params.wDecayRaw = workspace + t.offDecayRaw;
    params.wGateRaw = workspace + t.offGateRaw;
    params.wBetaRaw = workspace + t.offBetaRaw;
    params.wGated = workspace + t.offGated;
    params.counts = workspace + t.offCounts;
    params.syncConst = workspace + t.offSyncAux;
    params.expect = workspace + t.offExpect;
    params.prefix64 = data.prefix64;
    params.stageMask = kStageP | kStageR | kStageO;
}

CATLASS_DEVICE void InitializeTileSchedule(const MegaRecurrentKdaArch22TilingData &plan, const V2Params &params,
                                           GM_ADDR sequencePlan)
{
    const auto &t = plan.arch22;
    const auto &data = plan.common;
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            auto owners = reinterpret_cast<__gm__ const int32_t *>(sequencePlan);
            auto expect = reinterpret_cast<__gm__ uint32_t *>(params.expect);
            auto counters = reinterpret_cast<__gm__ int32_t *>(params.counts);
            auto constants = reinterpret_cast<__gm__ int32_t *>(params.syncConst);
            uint32_t first = 0, end = 0;
            for (uint32_t tile = 0; tile < t.tiles; ++tile) {
                const uint32_t start = tile * kTileRows;
                const uint32_t stop = (tile + 1) * kTileRows < t.tokens ? (tile + 1) * kTileRows : t.tokens;
                if (data.crosses == 2) {
                    first = owners[2 + start / 16];
                    end = owners[2 + (stop - 1) / 16];
                } else {
                    while (first < data.batch && SequenceStart(params, first + 1) <= start) ++first;
                }
                while (end < data.batch && SequenceStart(params, end) < stop) ++end;
                expect[tile] = (end - first) * data.heads;
            }
            for (uint32_t i = 0; i < CounterWords(t.tiles); ++i) {
                counters[i] = 0;
            }
            for (uint32_t i = 0; i < 64; ++i) {
                constants[i] = i % 8 == 0 ? i / 8 + 1 : 0;
            }
            AscendC::GlobalTensor<int32_t> flag;
            flag.SetGlobalBuffer(counters);
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE>(flag);
        }
    }
}

// Publish Q/K/V and gate by head group so independent chains can start before all weights have arrived.
CATLASS_DEVICE void ProjectHeadGroup(const V2Params &p, uint32_t firstHead, uint32_t heads)
{
    constexpr uint32_t panel = 256;
    using Gemm = Catlass::Gemm::Block::BlockMmadTla<KdaGemm::DispatchPolicy,
        tla::Shape<tla::_128, tla::_256, tla::_256>, tla::Shape<tla::_128, tla::_256, tla::_64>,
        bfloat16_t, bfloat16_t, bfloat16_t, void, KdaGemm::TileCopy>;
    Catlass::Arch::Resource<KdaGemm::ArchTag> resource;
    Gemm multiply(resource);
    const uint32_t core = AscendC::GetBlockIdx(), cores = p.aicNum;
    const uint32_t width = heads * kHeadDim, panels = CeilDiv(width, panel);
    const uint32_t regular = 4 * panels;
    const uint32_t decayPanels = CeilDiv(p.p, panel);
    const uint32_t signal = CounterByteOffset(2u, 0, p.tiles);
    auto counter = resource.l1Buf.GetBufferByByte<int32_t>(0);
    if (!firstHead && core == regular % cores) {
        RunGemmReuse(multiply, p.hiddenStates, p.tokens, p.hidden, p.decayAWeight, kHeadDim, p.hidden,
                     p.wGated, kHeadDim, p.hidden);
        AscendC::PipeBarrier<PIPE_ALL>();
        Sync::AicAdd(counter, p.syncConst, p.counts, signal, 1);
    }
    const uint32_t jobs = regular + (firstHead ? 0 : p.nBeta + decayPanels);
    bool decayReady = false;
    for (uint32_t job = core; job < jobs; job += cores) {
        if (job < regular) {
            const uint32_t plane = job / panels;
            const uint32_t local = job % panels * panel;
            const uint32_t n = width - local < panel ? width - local : panel;
            const uint32_t col = (plane < 3 ? plane * p.p : 0) + firstHead * kHeadDim + local;
            const uint32_t stride = plane == 3 ? p.p : 3 * p.p;
            GM_ADDR w = plane == 3 ? p.gateWeight : p.qkvWeight;
            GM_ADDR out = plane == 3 ? p.wGateRaw : p.wQkv;
            RunGemmReuse(multiply, p.hiddenStates, p.tokens, p.hidden, w + uint64_t(col) * p.hidden * 2, n,
                         p.hidden, out + col * 2, stride, p.hidden, true);
        } else if (job < regular + p.nBeta) {
            const uint32_t col = (job - regular) * kNW;
            const uint32_t n = p.n - col < kNW ? p.n - col : kNW;
            RunGemmReuse(multiply, p.hiddenStates, p.tokens, p.hidden, p.betaWeight + uint64_t(col) * p.hidden * 2,
                         n, p.hidden, p.wBetaRaw + col * 2, p.n, p.hidden);
        } else {
            if (!decayReady) {
                Sync::WaitEq(p.counts, signal, 1);
                decayReady = true;
            }
            const uint32_t col = (job - regular - p.nBeta) * panel;
            const uint32_t n = p.p - col < panel ? p.p - col : panel;
            RunGemmReuse(multiply, p.wGated, p.tokens, kHeadDim, p.decayBWeight + uint64_t(col) * kHeadDim * 2,
                         n, kHeadDim, p.wDecayRaw + col * 2, p.p, kHeadDim);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool ReuseHeadConstants = false>
CATLASS_DEVICE void ConsumeHeadGroup(const V2Params &p, uint32_t firstHead, uint32_t heads, bool uniformRequests)
{
    Catlass::Arch::Resource<KdaGemm::ArchTag> resource;
    auto ub = resource.ubBuf.GetBufferByByte<uint8_t>(0);
    HeadConstants cached;
    if (ReuseHeadConstants && uniformRequests) {
        const uint64_t total = uint64_t(p.batch) * heads;
        const uint32_t worker = AscendC::GetBlockIdx();
        const uint64_t begin = total * worker / p.aivNum;
        const uint64_t end = total * (worker + 1) / p.aivNum;
        for (uint64_t chain = begin; chain < end; ++chain) {
            const uint32_t b = chain % p.batch, head = firstHead + chain / p.batch;
            ProcessSequence(p, ub, b, head, SequenceStart(p, b), SequenceStart(p, b + 1), cached);
        }
        return;
    }
    for (uint64_t chain = AscendC::GetBlockIdx(); chain < uint64_t(p.batch) * heads; chain += p.aivNum) {
        const uint32_t b = chain / heads, head = firstHead + chain % heads;
        ProcessSequence(p, ub, b, head, SequenceStart(p, b), SequenceStart(p, b + 1), cached);
    }
}

// Publish the second head group when the first output panel reaches it.
struct StreamingHeadOutputCopy : KdaGemm::TileCopy {
    template <class Source>
    struct CopyGmToL1A : KdaGemm::TileCopy::template CopyGmToL1A<Source> {
        template <class Destination, class Tile>
        CATLASS_DEVICE void operator()(Destination &dst, Tile &src) {
            const uint32_t width = tla::get<0>(src.layout().stride());
            const uint32_t second = (width / kHeadDim / 2) * kHeadDim;
            if (uint32_t(tla::get<1>(src.coord())) == second) {
                AscendC::SyncAll<false>();
            }
            KdaGemm::TileCopy::template CopyGmToL1A<Source>::operator()(dst, src);
        }
    };
};
CATLASS_DEVICE void StreamHeadOutput(const V2Params &p) {
    constexpr uint32_t panel = 128;
    using Gemm = Catlass::Gemm::Block::BlockMmadTla<KdaGemm::DispatchPolicy,
        tla::Shape<tla::_128, tla::_128, tla::_128>, tla::Shape<tla::_128, tla::_128, tla::_64>,
        bfloat16_t, bfloat16_t, bfloat16_t, void, StreamingHeadOutputCopy>;
    Catlass::Arch::Resource<KdaGemm::ArchTag> resource;
    const uint32_t core = AscendC::GetBlockIdx(), cores = p.aicNum;
    if (core * panel < p.hidden) {
        Gemm multiply(resource);
        const uint32_t col = core * panel, width = p.hidden-col < panel ? p.hidden-col : panel;
        RunGemmReuse(multiply, p.wGated, p.tokens, p.p, p.wOut+uint64_t(col)*p.p*2, width, p.p,
                     p.out+col*2, p.hidden, p.p);
    } else {
        AscendC::SyncAll<false>();
    }
    typename KdaGemm::BlockMmad rest(resource);
    for (uint32_t col=(core+cores)*panel; col<p.hidden; col+=cores*panel) {
        const uint32_t width = p.hidden-col < panel ? p.hidden-col : panel;
        RunGemmReuse(rest,p.wGated,p.tokens,p.p,p.wOut+uint64_t(col)*p.p*2,width,p.p,
                     p.out+col*2,p.hidden,p.p);
    }
}

template <bool splitSums = false, bool splitValues = false, bool bypassQkvL2 = false, bool ReuseHeadConstants = false, bool NarrowAux = false>
CATLASS_DEVICE void RunArch22(const MegaRecurrentKdaArch22TilingData &plan, V2Params &params, GM_ADDR sequencePlan,
                              GM_ADDR workspace)
{
    BindTilingAndWorkspace(plan, params, workspace);
    if (T02Rows::Try(plan, params, sequencePlan, workspace)) return;
    if (plan.common.crosses == 2 && reinterpret_cast<__gm__ const int32_t *>(sequencePlan)[0]) {
        params.templateId = kTemplateT02;
        params.pSplit = 0;
        params.pBlocks = CeilDiv(params.tiles, kBlockTiles);
    }
    // One projection block fits in the existing INT8 synchronization phases.
    // Its independent BF16 projections do not consume the tile counters.
    const bool overlapAuxiliary = params.weightScale && params.pBlocks == 1 && Int8SumColumns(params.hidden) != 0;
    // With all projections visible and just one output tile, one collective publishes every chain.
    const bool joinChains = overlapAuxiliary && !splitValues && params.tiles == 1;
    if (joinChains) {
        params.skipSync |= 2u;
    } else {
        InitializeTileSchedule(plan, params, sequencePlan);
    }
    if (!overlapAuxiliary) {
        AscendC::SyncAll<false>();
    }
    if constexpr (ReuseHeadConstants) {
        if (!params.weightScale && params.templateId == kTemplateT01 && params.tokens <= 128 &&
            params.n >= params.aicNum && uint64_t(params.batch) * params.n >= 2 * params.aivNum) {
            const uint32_t half = params.n / 2;
            const bool uniformRequests = params.sequence ||
                (plan.common.crosses == 2 && uint64_t(reinterpret_cast<__gm__ const int32_t *>(sequencePlan)[1]) *
                                                 params.batch == params.tokens);
            params.skipSync |= 3u;
            if ASCEND_IS_AIC { ProjectHeadGroup(params, 0, half); }
            AscendC::SyncAll<false>();
            if ASCEND_IS_AIC { ProjectHeadGroup(params, half, params.n - half); }
            if ASCEND_IS_AIV { ConsumeHeadGroup<ReuseHeadConstants>(params, 0, half, uniformRequests); }
            AscendC::SyncAll<false>();
            if (uniformRequests) {
                if ASCEND_IS_AIV {
                    ConsumeHeadGroup<ReuseHeadConstants>(params, half, params.n - half, true);
                    AscendC::SyncAll<false>();
                }
                if ASCEND_IS_AIC { StreamHeadOutput(params); }
            } else {
                if ASCEND_IS_AIV { ConsumeHeadGroup<false>(params, half, params.n - half, false); }
                AscendC::SyncAll<false>();
                if ASCEND_IS_AIC {
                    Catlass::Arch::Resource<KdaGemm::ArchTag> resource;
                    for (uint32_t block = 0; block < params.oBlocks; ++block) RunStageO(resource, params, block);
                }
            }
            return;
        }
    }
    if (params.weightScale) {
        if (overlapAuxiliary) {
            params.skipSync |= 1u; // The last INT8 barrier also publishes all BF16 projections to R.
        }
        params.nProj -= params.nQkv;
        params.nQkv = 0;
        GM_ADDR extraSums = nullptr;
        if constexpr (splitSums) {
            extraSums = workspace + SplitQkvSumOffset(plan.arch22, plan.common.crosses == 2);
        }
        ProjectInt8Qkv<splitSums, bypassQkvL2, NarrowAux>(params, overlapAuxiliary, extraSums);
    }
    if constexpr (splitValues) {
        // An explicit caller-provided output can share storage with hidden states.
        // That alias is only visible on device, after host tiling has chosen the geometry.
        if (params.out != params.hiddenStates) {
            LongSequence::Run(params, overlapAuxiliary, LongSequence::GroupCount(params));
            return;
        }
    }
    if ASCEND_IS_AIC {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        if ((params.stageMask & kStageP) != 0u && !overlapAuxiliary) {
            for (uint32_t block = 0; block < params.pBlocks; ++block) {
                RunStageP(resource, params, block);
            }
        }
        if ((params.stageMask & kStageO) != 0u) {
            if (joinChains) {
                AscendC::SyncAll<false>();
            }
            for (uint32_t block = 0; block < params.oBlocks; ++block) {
                RunStageO(resource, params, block);
            }
        }
    }
    if ASCEND_IS_AIV {
        if ((params.stageMask & kStageR) != 0u) {
            AscendC::TPipe pipe;
            AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
            pipe.InitBuffer(ubBuf, RDetail::kUbBytes);
            AscendC::LocalTensor<uint8_t> ub = ubBuf.Get<uint8_t>();
            RunStageR(params, ub);
            if (joinChains) {
                AscendC::SyncAll<false>();
            }
        }
    }
}
} // namespace MegaRecurrentKdaV2
