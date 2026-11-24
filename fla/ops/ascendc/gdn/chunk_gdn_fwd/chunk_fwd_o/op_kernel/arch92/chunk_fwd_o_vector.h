/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH92_VECTOR_H
#define CHUNK_FWD_O_ARCH92_VECTOR_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_utils/vector/regbase.hpp"
#include "../chunk_fwd_o_struct.h"
#include "chunk_fwd_o_common.h"

namespace GDN {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr float CHUNK_FWD_O_LN2 = 0.69314718055994530941723212145818f;
constexpr CastTrait CHUNK_FWD_O_FP32_TO_B16_PACK = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr CastTrait CHUNK_FWD_O_B16_TO_FP32 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE,
};

template <typename GT, bool UseExp2>
__simd_vf__ inline void Stage1Gate64VF(__ubuf__ float *gateAAddr, __ubuf__ float *gateOAddr,
                                      __ubuf__ GT *gAddr, uint16_t validRows)
{
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A6_BT);
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);
    RegTensor<float> gReg, zeroReg, gateOReg, gRowReg, rowGateReg;
    RegTensor<uint32_t> rowIndex;
    Duplicate(zeroReg, 0.0f, fullMask);
    if constexpr (std::is_same<GT, float>::value) {
        LoadAlign(gReg, gAddr);
    } else {
        RegTensor<GT> raw;
        LoadAlign<GT, LoadDist::DIST_UNPACK_B16>(raw, gAddr);
        Cast<float, GT, CHUNK_FWD_O_B16_TO_FP32>(gReg, raw, validMask);
    }
    Select(gReg, gReg, zeroReg, validMask);
    if constexpr (UseExp2) {
        Muls(gateOReg, gReg, CHUNK_FWD_O_LN2, fullMask);
    } else {
        Adds(gateOReg, gReg, 0.0f, fullMask);
    }
    Exp(gateOReg, gateOReg, fullMask);
    Select(gateOReg, gateOReg, zeroReg, validMask);
    StoreAlign(gateOAddr, gateOReg, fullMask);
    for (uint16_t row = 0; row < validRows; ++row) {
        Duplicate(rowIndex, static_cast<uint32_t>(row), fullMask);
        Gather(gRowReg, gReg, rowIndex);
        Sub(rowGateReg, gRowReg, gReg, fullMask);
        uint32_t causalCount = static_cast<uint32_t>(row) + 1U;
        MaskReg causalMask = UpdateMask<float>(causalCount);
        // Never exponentiate the invalid upper triangle or padding.
        Select(rowGateReg, rowGateReg, zeroReg, causalMask);
        if constexpr (UseExp2) {
            Muls(rowGateReg, rowGateReg, CHUNK_FWD_O_LN2, fullMask);
        }
        Exp(rowGateReg, rowGateReg, fullMask);
        Select(rowGateReg, rowGateReg, zeroReg, causalMask);
        StoreAlign(gateAAddr + static_cast<uint32_t>(row) * kBt, rowGateReg, fullMask);
    }
    for (uint16_t row = validRows; row < kBt; ++row) {
        StoreAlign(gateAAddr + static_cast<uint32_t>(row) * kBt, zeroReg, fullMask);
    }
}

template <bool FullChunk>
__simd_vf__ inline void Stage3APrime64VF(__ubuf__ bfloat16_t *aPrimeAddr,
                                        __ubuf__ float *aRawAddr, __ubuf__ float *gateAAddr,
                                        uint16_t validRows)
{
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A6_BT);
    const uint16_t activeRows = FullChunk ? kBt : validRows;
    RegTensor<float> zeroReg;
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zeroReg, 0.0f, floatMask);

    // Keep A5's four-row register schedule; QH gating runs in a separate VF.
    {
        MaskReg lowerMask0;
        MaskReg lowerMask1;
        MaskReg lowerMask2;
        MaskReg lowerMask3;
        RegTensor<float> aRawReg0;
        RegTensor<float> aRawReg1;
        RegTensor<float> aRawReg2;
        RegTensor<float> aRawReg3;
        RegTensor<float> gateAReg0;
        RegTensor<float> gateAReg1;
        RegTensor<float> gateAReg2;
        RegTensor<float> gateAReg3;
        RegTensor<float> aPrimeReg0;
        RegTensor<float> aPrimeReg1;
        RegTensor<float> aPrimeReg2;
        RegTensor<float> aPrimeReg3;
        RegTensor<bfloat16_t> aPrimeBf16Reg0;
        RegTensor<bfloat16_t> aPrimeBf16Reg1;
        RegTensor<bfloat16_t> aPrimeBf16Reg2;
        RegTensor<bfloat16_t> aPrimeBf16Reg3;
        uint16_t rowLoop = 0;
        const uint16_t rowLoopCount = activeRows / 4U;
        const uint16_t tailLoopCount = activeRows % 4U;
        const uint16_t tailRowBase = rowLoopCount * 4U;
        uint16_t rowBase = 0;
        uint32_t rowOffset0 = 0;
        uint32_t rowOffset1 = 0;
        uint32_t rowOffset2 = 0;
        uint32_t rowOffset3 = 0;
        uint32_t lowerCount0 = 0;
        uint32_t lowerCount1 = 0;
        uint32_t lowerCount2 = 0;
        uint32_t lowerCount3 = 0;

        // Four independent rows expose parallel load/compute/store chains.
        for (rowLoop = 0; rowLoop < rowLoopCount; ++rowLoop) {
            rowBase = static_cast<uint16_t>(rowLoop * 4U);
            rowOffset0 = static_cast<uint32_t>(rowBase) * kBt;
            rowOffset1 = static_cast<uint32_t>(rowBase + 1U) * kBt;
            rowOffset2 = static_cast<uint32_t>(rowBase + 2U) * kBt;
            rowOffset3 = static_cast<uint32_t>(rowBase + 3U) * kBt;
            lowerCount0 = static_cast<uint32_t>(rowBase + 1U);
            lowerCount1 = static_cast<uint32_t>(rowBase + 2U);
            lowerCount2 = static_cast<uint32_t>(rowBase + 3U);
            lowerCount3 = static_cast<uint32_t>(rowBase + 4U);
            lowerMask0 = UpdateMask<float>(lowerCount0);
            lowerMask1 = UpdateMask<float>(lowerCount1);
            lowerMask2 = UpdateMask<float>(lowerCount2);
            lowerMask3 = UpdateMask<float>(lowerCount3);

            LoadAlign(aRawReg0, aRawAddr + rowOffset0);
            LoadAlign(aRawReg1, aRawAddr + rowOffset1);
            LoadAlign(aRawReg2, aRawAddr + rowOffset2);
            LoadAlign(aRawReg3, aRawAddr + rowOffset3);
            LoadAlign(gateAReg0, gateAAddr + rowOffset0);
            LoadAlign(gateAReg1, gateAAddr + rowOffset1);
            LoadAlign(gateAReg2, gateAAddr + rowOffset2);
            LoadAlign(gateAReg3, gateAAddr + rowOffset3);
            Select(aRawReg0, aRawReg0, zeroReg, lowerMask0);
            Mul(aPrimeReg0, aRawReg0, gateAReg0, floatMask);
            Select(aRawReg1, aRawReg1, zeroReg, lowerMask1);
            Mul(aPrimeReg1, aRawReg1, gateAReg1, floatMask);
            Select(aRawReg2, aRawReg2, zeroReg, lowerMask2);
            Mul(aPrimeReg2, aRawReg2, gateAReg2, floatMask);
            Select(aRawReg3, aRawReg3, zeroReg, lowerMask3);
            Mul(aPrimeReg3, aRawReg3, gateAReg3, floatMask);
            Select(aPrimeReg0, aPrimeReg0, zeroReg, lowerMask0);
            Select(aPrimeReg1, aPrimeReg1, zeroReg, lowerMask1);
            Select(aPrimeReg2, aPrimeReg2, zeroReg, lowerMask2);
            Select(aPrimeReg3, aPrimeReg3, zeroReg, lowerMask3);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg0, aPrimeReg0, floatMask);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg1, aPrimeReg1, floatMask);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg2, aPrimeReg2, floatMask);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg3, aPrimeReg3, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset0, aPrimeBf16Reg0, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset1, aPrimeBf16Reg1, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset2, aPrimeBf16Reg2, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset3, aPrimeBf16Reg3, floatMask);
        }

        for (rowLoop = 0; rowLoop < tailLoopCount; ++rowLoop) {
            rowBase = static_cast<uint16_t>(tailRowBase + rowLoop);
            rowOffset0 = static_cast<uint32_t>(rowBase) * kBt;
            lowerCount0 = static_cast<uint32_t>(rowBase + 1U);
            lowerMask0 = UpdateMask<float>(lowerCount0);
            LoadAlign(aRawReg0, aRawAddr + rowOffset0);
            LoadAlign(gateAReg0, gateAAddr + rowOffset0);
            Select(aRawReg0, aRawReg0, zeroReg, lowerMask0);
            Mul(aPrimeReg0, aRawReg0, gateAReg0, floatMask);
            Select(aPrimeReg0, aPrimeReg0, zeroReg, lowerMask0);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg0, aPrimeReg0, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset0, aPrimeBf16Reg0, floatMask);
        }

        Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg0, zeroReg, floatMask);
        const uint16_t invalidRowCount = static_cast<uint16_t>(kBt - activeRows);
        for (rowLoop = 0; rowLoop < invalidRowCount; ++rowLoop) {
            rowBase = static_cast<uint16_t>(activeRows + rowLoop);
            rowOffset0 = static_cast<uint32_t>(rowBase) * kBt;
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset0, aPrimeBf16Reg0, floatMask);
        }
    }

}

template <bool FullChunk>
__simd_vf__ inline void Stage3OSPrime64VF(__ubuf__ float *oSAddr, __ubuf__ float *gateOAddr,
                                         uint16_t validRows)
{
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A6_BT);
    const uint16_t activeRows = FullChunk ? kBt : validRows;
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zeroReg;
    Duplicate(zeroReg, 0.0f, floatMask);
    // Phase 2 keeps one row and its two fixed V tiles in nested hardware loops.
    RegTensor<float> oSReg;
    RegTensor<float> gateOVectorReg;
    RegTensor<float> gateORowReg0;
    RegTensor<uint32_t> rowIndexReg0;
    LoadAlign(gateOVectorReg, gateOAddr);

    constexpr uint16_t kTilesPerRow = static_cast<uint16_t>(CHUNK_FWD_O_A6_V / CHUNK_FWD_O_A6_BT);
    for (uint16_t row = 0; row < activeRows; ++row) {
        Duplicate(rowIndexReg0, static_cast<uint32_t>(row), floatMask);
        Gather(gateORowReg0, gateOVectorReg, rowIndexReg0);
        for (uint16_t tile = 0; tile < kTilesPerRow; ++tile) {
            const uint32_t offset = static_cast<uint32_t>(row) * CHUNK_FWD_O_A6_V +
                                    static_cast<uint32_t>(tile) * kBt;
            LoadAlign(oSReg, oSAddr + offset);
            Mul(oSReg, oSReg, gateORowReg0, floatMask);
            StoreAlign(oSAddr + offset, oSReg, floatMask);
        }
    }

    for (uint16_t row = activeRows; row < kBt; ++row) {
        const uint32_t offset0 = static_cast<uint32_t>(row) * CHUNK_FWD_O_A6_V;
        StoreAlign(oSAddr + offset0, zeroReg, floatMask);
        StoreAlign(oSAddr + offset0 + kBt, zeroReg, floatMask);
    }
}

__simd_vf__ inline void Stage5Fuse64VF(__ubuf__ bfloat16_t *oOutAddr, __ubuf__ float *oSPrimeAddr,
                                      __ubuf__ float *oLAddr, float scale, uint16_t validRows)
{
    constexpr uint16_t kV = static_cast<uint16_t>(CHUNK_FWD_O_A6_V);
    constexpr uint16_t kTilesPerRow = static_cast<uint16_t>(CHUNK_FWD_O_A6_V / CHUNK_FWD_O_A6_BT);
    RegTensor<float> oSPrimeReg;
    RegTensor<float> oLReg;
    RegTensor<float> oOutReg;
    RegTensor<bfloat16_t> oOutBf16Reg;
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t row = 0; row < validRows; ++row) {
        for (uint16_t tile = 0; tile < kTilesPerRow; ++tile) {
            const uint32_t offset = static_cast<uint32_t>(row) * kV +
                                    static_cast<uint32_t>(tile) * CHUNK_FWD_O_A6_BT;
            LoadAlign(oSPrimeReg, oSPrimeAddr + offset);
            LoadAlign(oLReg, oLAddr + offset);
            Add(oOutReg, oSPrimeReg, oLReg, floatMask);
            Muls(oOutReg, oOutReg, scale, floatMask);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(oOutBf16Reg, oOutReg, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(oOutAddr + offset, oOutBf16Reg, floatMask);
        }
    }
}

template <typename GT, bool UseExp2>
class ChunkFwdOA6VectorProcess {
public:
    static constexpr uint32_t kHeads = CHUNK_FWD_O_A6_HEAD_COUNT;
    static constexpr uint32_t bt = CHUNK_FWD_O_A6_BT;
    static constexpr uint32_t vDim = CHUNK_FWD_O_A6_V;

    __aicore__ inline ChunkFwdOA6VectorProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                               GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o, GM_ADDR workspace)
        : g_(g), cuSeqlens_(cuSeqlens), chunkOffsets_(chunkOffsets), o_(o)
    {
        (void)q;
        (void)k;
        (void)v;
        (void)h;
        (void)workspace;
    }

    __aicore__ inline void Init(const ChunkFwdOTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        oGm_.SetGlobalBuffer((__gm__ bfloat16_t *)o_);
        pipe->InitBuffer(ubBuf_, CHUNK_FWD_O_UB_TOTAL_BYTES);
        // Use the same absolute L1 offsets as the AIC for A-prime.
        pipe->InitBuffer(l1Buf_, CHUNK_FWD_O_L1_TOTAL_BYTES);
        for (uint32_t slot = 0; slot < kHeads; ++slot) {
            gReady_[slot] = pipe->AllocEventID<HardEvent::MTE2_V>();
            gFree_[slot] = pipe->AllocEventID<HardEvent::V_MTE2>();
            aPrimeReady_[slot] = pipe->AllocEventID<HardEvent::V_MTE3>();
            oReady_[slot] = pipe->AllocEventID<HardEvent::V_MTE3>();
            SetFlag<HardEvent::V_MTE2>(gFree_[slot]);
        }
    }

    __aicore__ inline void Process(uint32_t coreIdx, uint32_t coreNum)
    {
        const uint32_t headGroupNum = ChunkFwdOHeadGroupNum(tiling_);
        const uint32_t groupTaskNum = static_cast<uint32_t>(tiling_.chunkNum) * headGroupNum;
        for (uint32_t task = coreIdx; task < groupTaskNum; task += coreNum) {
            ChunkFwdOChunkLoc loc;
            ChunkFwdOResolveChunkLoc(cuSeqlens_, chunkOffsets_, tiling_, task / headGroupNum, loc);
            if (loc.chunkLen == 0U) {
                continue;
            }
            const int64_t hvBase = static_cast<int64_t>(task % headGroupNum) * tiling_.taskGroupSize;
            const int64_t remaining = tiling_.vNumHead - hvBase;
            const uint32_t count = static_cast<uint32_t>(remaining < tiling_.taskGroupSize ?
                                                         remaining : tiling_.taskGroupSize);
            for (uint32_t slot = 0; slot < count; ++slot) {
                WaitFlag<HardEvent::V_MTE2>(gFree_[slot]);
                LoadStage1G(loc, hvBase + slot, slot);
                SetFlag<HardEvent::MTE2_V>(gReady_[slot]);
                WaitFlag<HardEvent::MTE2_V>(gReady_[slot]);
                auto gateO = ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(slot));
                auto gateA = ubBuf_.GetWithOffset<float>(bt * bt, ChunkFwdOGateAOffset(slot));
                auto g = ubBuf_.GetWithOffset<GT>(bt, ChunkFwdOGScratchOffset(slot));
                AscendC::VF_CALL<Stage1Gate64VF<GT, UseExp2>>(
                    reinterpret_cast<__ubuf__ float *>(gateA.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gateO.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ GT *>(g.GetPhyAddr()), static_cast<uint16_t>(loc.chunkLen));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(gFree_[slot]);
            }
            for (uint32_t slot = 0; slot < count; ++slot) {
                if (loc.chunkLen == bt) {
                    ProcessStage3<true>(loc, slot);
                } else {
                    ProcessStage3<false>(loc, slot);
                }
            }
            for (uint32_t slot = 0; slot < count; ++slot) {
                auto os = ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOsPrimeOffset(slot));
                auto ol = ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOlOffset(slot));
                auto out = ubBuf_.GetWithOffset<bfloat16_t>(bt * vDim, ChunkFwdOOlOffset(slot));
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(CHUNK_FWD_O_OL_READY + slot);
                PipeBarrier<PIPE_V>();
                AscendC::VF_CALL<Stage5Fuse64VF>(
                    reinterpret_cast<__ubuf__ bfloat16_t *>(out.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(os.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(ol.GetPhyAddr()), tiling_.scale,
                    static_cast<uint16_t>(loc.chunkLen));
                SetFlag<HardEvent::V_MTE3>(oReady_[slot]);
                WaitFlag<HardEvent::V_MTE3>(oReady_[slot]);
                DataCopyExtParams params{static_cast<uint16_t>(loc.chunkLen), vDim * sizeof(bfloat16_t), 0,
                    static_cast<uint32_t>((tiling_.vNumHead - 1) * vDim * sizeof(bfloat16_t)), 0};
                DataCopyPad(oGm_[ChunkFwdOOOffset(tiling_, loc, hvBase + slot)], out, params);
            }
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(CHUNK_FWD_O_ROUND_DONE);
        }
        for (uint32_t slot = 0; slot < kHeads; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(gFree_[slot]);
        }
    }

private:
    template <bool FullChunk>
    __aicore__ inline void ProcessStage3(const ChunkFwdOChunkLoc &loc, uint32_t slot)
    {
        auto gateO = ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(slot));
        auto gateA = ubBuf_.GetWithOffset<float>(bt * bt, ChunkFwdOGateAOffset(slot));
        auto raw = ubBuf_.GetWithOffset<float>(bt * bt, ChunkFwdOARawOffset(slot));
        auto os = ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOsPrimeOffset(slot));
        auto aPrime = ubBuf_.GetWithOffset<bfloat16_t>(bt * bt, ChunkFwdOAPrimeBf16Offset(slot));
        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(CHUNK_FWD_O_ARAW_READY + slot);
        AscendC::VF_CALL<Stage3APrime64VF<FullChunk>>(
            reinterpret_cast<__ubuf__ bfloat16_t *>(aPrime.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateA.GetPhyAddr()), static_cast<uint16_t>(loc.chunkLen));
        SetFlag<HardEvent::V_MTE3>(aPrimeReady_[slot]);
        WaitFlag<HardEvent::V_MTE3>(aPrimeReady_[slot]);
        auto l1 = l1Buf_.GetWithOffset<bfloat16_t>(bt * bt, ChunkFwdOL1APrimeOffset(slot));
        DataCopyParams params{bt, 1, bt / 16U - 1U, 0};
        for (uint32_t col = 0; col < bt / 16U; ++col) {
            DataCopy(l1[col * bt * 16U], aPrime[col * 16U], params);
        }
        AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(CHUNK_FWD_O_APRIME_READY + slot);
        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(CHUNK_FWD_O_OSRAW_READY + slot);
        AscendC::VF_CALL<Stage3OSPrime64VF<FullChunk>>(
            reinterpret_cast<__ubuf__ float *>(os.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gateO.GetPhyAddr()), static_cast<uint16_t>(loc.chunkLen));
    }

    __aicore__ inline void LoadStage1G(const ChunkFwdOChunkLoc &loc, int64_t hv, uint32_t streamSlot)
    {
        GlobalTensor<GT> gGm;
        gGm.SetGlobalBuffer((__gm__ GT *)g_);
        gGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        const int64_t gOffset = ChunkFwdOGOffset(tiling_, loc, hv);
        const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A6_BT);
        const uint32_t gScratchOff = ChunkFwdOGScratchOffset(streamSlot);

        if constexpr (std::is_same<GT, float>::value) {
            LocalTensor<float> gFp32 = ubBuf_.GetWithOffset<float>(bt, gScratchOff);
            DataCopyPad(gFp32, gGm[gOffset],
                        {1, static_cast<uint32_t>(loc.chunkLen * sizeof(GT)), 0, 0, 0},
                        {false, 0, 0, 0});
        } else {
            LocalTensor<GT> gLocal = ubBuf_.GetWithOffset<GT>(bt, gScratchOff);
            DataCopyPad(gLocal, gGm[gOffset],
                        {1, static_cast<uint32_t>(loc.chunkLen * sizeof(GT)), 0, 0, 0},
                        {false, 0, 0, 0});
        }
    }


    GM_ADDR g_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkOffsets_;
    GM_ADDR o_;
    ChunkFwdOTilingData tiling_{};
    TBuf<TPosition::VECCALC> ubBuf_;
    TBuf<TPosition::A1> l1Buf_;
    GlobalTensor<bfloat16_t> oGm_;
    TEventID gReady_[kHeads];
    TEventID gFree_[kHeads];
    TEventID aPrimeReady_[kHeads];
    TEventID oReady_[kHeads];
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH92_VECTOR_H
