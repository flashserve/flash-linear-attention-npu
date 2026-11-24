#pragma once
#include "mega_recurrent_kda_v2_cube.h"
#include "mega_recurrent_kda_v2_t02.h"
#include "tiling_data.h"
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
    params.prefix = workspace + t.workspaceBytes;
    params.stageMask = kStageP | kStageR | kStageO;
}

CATLASS_DEVICE void InitializeTileSchedule(const MegaRecurrentKdaArch22TilingData &plan, const V2Params &params,
                                           GM_ADDR inputPrefix)
{
    const auto &t = plan.arch22;
    const auto &data = plan.common;
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            auto seq = reinterpret_cast<__gm__ int32_t *>(params.prefix);
            auto expect = reinterpret_cast<__gm__ uint32_t *>(params.expect);
            auto counters = reinterpret_cast<__gm__ int32_t *>(params.counts);
            auto constants = reinterpret_cast<__gm__ int32_t *>(params.syncConst);
            for (uint32_t b = 0; b <= data.batch; ++b) {
                seq[b] = data.sequence ? b * data.sequence :
                                         static_cast<int32_t>(reinterpret_cast<__gm__ int64_t *>(inputPrefix)[b]);
            }
            uint32_t first = 0, end = 0;
            for (uint32_t tile = 0; tile < t.tiles; ++tile) {
                while (first < data.batch && uint32_t(seq[first + 1]) <= tile * 64) {
                    ++first;
                }
                while (end < data.batch && uint32_t(seq[end]) < (tile + 1) * 64) {
                    ++end;
                }
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
    AscendC::SyncAll<false>();
}

CATLASS_DEVICE void RunArch22(const MegaRecurrentKdaArch22TilingData &plan, V2Params &params, GM_ADDR inputPrefix,
                              GM_ADDR workspace)
{
    BindTilingAndWorkspace(plan, params, workspace);
    InitializeTileSchedule(plan, params, inputPrefix);
    if ASCEND_IS_AIC {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        if ((params.stageMask & kStageP) != 0u) {
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
