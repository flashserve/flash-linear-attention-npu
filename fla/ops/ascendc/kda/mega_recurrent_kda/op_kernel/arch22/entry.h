#pragma once
#include "mega_recurrent_kda_v2_cube.h"
#include "mega_recurrent_kda_v2_t02.h"
#include "tiling_data.h"
#include "int8_qkv.h"
#include "long_sequence.h"
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

template <bool splitSums = false, bool splitValues = false, bool bypassQkvL2 = false>
CATLASS_DEVICE void RunArch22(const MegaRecurrentKdaArch22TilingData &plan, V2Params &params, GM_ADDR sequencePlan,
                              GM_ADDR workspace)
{
    BindTilingAndWorkspace(plan, params, workspace);
    if (plan.common.crosses == 2 && reinterpret_cast<__gm__ const int32_t *>(sequencePlan)[0]) {
        params.templateId = kTemplateT02;
        params.pSplit = 0;
        params.pBlocks = CeilDiv(params.tiles, kBlockTiles);
    }
    // One projection block fits in the existing INT8 synchronization phases.
    // Its independent BF16 projections do not consume the tile counters.
    const bool overlapAuxiliary = params.weightScale && params.pBlocks == 1 && Int8SumColumns(params.hidden) != 0;
    InitializeTileSchedule(plan, params, sequencePlan);
    if (!overlapAuxiliary) {
        AscendC::SyncAll<false>();
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
        ProjectInt8Qkv<splitSums, bypassQkvL2>(params, overlapAuxiliary, extraSums);
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
            // T01 batches completion credits within one tile. T02 publishes a credit
            // at each tile boundary while the same core retains the sequence state.
            if (params.templateId == kTemplateT02) {
                RunStageR02(params, ub);
            } else {
                RunStageR(params, ub);
            }
        }
    }
}
} // namespace MegaRecurrentKdaV2
