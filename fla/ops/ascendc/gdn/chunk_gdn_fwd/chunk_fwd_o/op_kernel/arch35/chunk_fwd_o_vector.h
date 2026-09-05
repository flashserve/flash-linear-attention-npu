/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef CHUNK_FWD_O_ARCH35_VECTOR_H
#define CHUNK_FWD_O_ARCH35_VECTOR_H

#include "kernel_operator.h"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
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
    AscendC::RoundMode::CAST_ROUND,
};

__simd_callee__ inline void LoadGateFloatPair(RegTensor<float> &zero, RegTensor<float> &one, __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void StoreGateFloatPair(__ubuf__ float *dst, RegTensor<float> &zero, RegTensor<float> &one,
                                              MaskReg &maskF32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, maskF32);
}

__simd_vf__ inline void PadGateInput64VF(__ubuf__ float *gAddr, uint16_t validRows)
{
    RegTensor<float> gReg;
    RegTensor<float> zeroReg;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = static_cast<uint32_t>(validRows);
    MaskReg validMask = UpdateMask<float>(validCount);
    LoadAlign(gReg, gAddr);
    Duplicate(zeroReg, 0.0f, fullMask);
    Select(gReg, gReg, zeroReg, validMask);
    StoreAlign(gAddr, gReg, fullMask);
}

template <bool UseExp2>
__simd_vf__ inline void Stage1Gate64VF(__ubuf__ float *gateAAddr, __ubuf__ float *gAddr, uint16_t chunkLen)
{
    (void)chunkLen;
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A5_BT);
    MaskReg maskFull32 = CreateMask<float, MaskPattern::ALL>();

    RegTensor<float> gZeroReg;
    RegTensor<float> gOneReg;
    RegTensor<float> gRowReg;
    RegTensor<float> rowGateZeroReg;
    RegTensor<float> rowGateOneReg;

    LoadGateFloatPair(gZeroReg, gOneReg, gAddr);

    for (uint16_t row = 0; row < kBt; ++row) {
        LoadIn<float, true>(gRowReg, gAddr + row);
        SubFloatTwoReg(rowGateZeroReg, rowGateOneReg, gRowReg, gRowReg, gZeroReg, gOneReg, maskFull32);
        if constexpr (UseExp2) {
            Muls(rowGateZeroReg, rowGateZeroReg, CHUNK_FWD_O_LN2, maskFull32);
            Muls(rowGateOneReg, rowGateOneReg, CHUNK_FWD_O_LN2, maskFull32);
        }
        ExpFloatTwoReg(rowGateZeroReg, rowGateOneReg, rowGateZeroReg, rowGateOneReg, maskFull32);
        StoreGateFloatPair(gateAAddr + static_cast<uint32_t>(row) * kBt, rowGateZeroReg, rowGateOneReg, maskFull32);
    }
}

__simd_vf__ inline void Stage3Gate64VF(__ubuf__ bfloat16_t *aPrimeAddr, __ubuf__ float *oSAddr,
                                      __ubuf__ float *aRawAddr, __ubuf__ float *gateAAddr,
                                      __ubuf__ float *gateOAddr, uint16_t validRows)
{
    constexpr uint16_t kBt = static_cast<uint16_t>(CHUNK_FWD_O_A5_BT);
    RegTensor<float> zeroReg;
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zeroReg, 0.0f, floatMask);

    // Phase 1 computes A'. Its registers are dead before the O_s' phase starts,
    // allowing the compiler to reuse the same physical register bank.
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
        const uint16_t rowLoopCount = validRows / 4U;
        const uint16_t tailLoopCount = validRows % 4U;
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
            Mul(aPrimeReg0, aRawReg0, gateAReg0, floatMask);
            Mul(aPrimeReg1, aRawReg1, gateAReg1, floatMask);
            Mul(aPrimeReg2, aRawReg2, gateAReg2, floatMask);
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
            Mul(aPrimeReg0, aRawReg0, gateAReg0, floatMask);
            Select(aPrimeReg0, aPrimeReg0, zeroReg, lowerMask0);
            Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg0, aPrimeReg0, floatMask);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset0, aPrimeBf16Reg0, floatMask);
        }

        Cast<bfloat16_t, float, CHUNK_FWD_O_FP32_TO_B16_PACK>(aPrimeBf16Reg0, zeroReg, floatMask);
        const uint16_t invalidRowCount = static_cast<uint16_t>(kBt - validRows);
        for (rowLoop = 0; rowLoop < invalidRowCount; ++rowLoop) {
            rowBase = static_cast<uint16_t>(validRows + rowLoop);
            rowOffset0 = static_cast<uint32_t>(rowBase) * kBt;
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(aPrimeAddr + rowOffset0, aPrimeBf16Reg0, floatMask);
        }
    }

    // Phase 2 handles two rows per loop while explicitly reusing one row's registers.
    RegTensor<float> oSReg00;
    RegTensor<float> oSReg01;
    RegTensor<float> gateOVectorReg;
    RegTensor<float> gateORowReg0;
    RegTensor<uint32_t> rowIndexReg0;
    LoadAlign(gateOVectorReg, gateOAddr);

    const uint16_t validRowPairCount = validRows / 2U;
    for (uint16_t rowPair = 0; rowPair < validRowPairCount; ++rowPair) {
        const uint16_t row0 = static_cast<uint16_t>(rowPair * 2U);
        const uint16_t row1 = static_cast<uint16_t>(row0 + 1U);
        const uint32_t offset00 = static_cast<uint32_t>(row0) * CHUNK_FWD_O_A5_V;
        const uint32_t offset01 = offset00 + kBt;
        const uint32_t offset10 = static_cast<uint32_t>(row1) * CHUNK_FWD_O_A5_V;
        const uint32_t offset11 = offset10 + kBt;

        Duplicate(rowIndexReg0, static_cast<uint32_t>(row0), floatMask);
        Gather(gateORowReg0, gateOVectorReg, rowIndexReg0);
        LoadAlign(oSReg00, oSAddr + offset00);
        LoadAlign(oSReg01, oSAddr + offset01);
        Mul(oSReg00, oSReg00, gateORowReg0, floatMask);
        Mul(oSReg01, oSReg01, gateORowReg0, floatMask);
        StoreAlign(oSAddr + offset00, oSReg00, floatMask);
        StoreAlign(oSAddr + offset01, oSReg01, floatMask);

        Duplicate(rowIndexReg0, static_cast<uint32_t>(row1), floatMask);
        Gather(gateORowReg0, gateOVectorReg, rowIndexReg0);
        LoadAlign(oSReg00, oSAddr + offset10);
        LoadAlign(oSReg01, oSAddr + offset11);
        Mul(oSReg00, oSReg00, gateORowReg0, floatMask);
        Mul(oSReg01, oSReg01, gateORowReg0, floatMask);
        StoreAlign(oSAddr + offset10, oSReg00, floatMask);
        StoreAlign(oSAddr + offset11, oSReg01, floatMask);
    }

    if ((validRows & 1U) != 0U) {
        const uint32_t offset0 = static_cast<uint32_t>(validRows - 1U) * CHUNK_FWD_O_A5_V;
        const uint32_t offset1 = offset0 + kBt;
        Duplicate(rowIndexReg0, static_cast<uint32_t>(validRows - 1U), floatMask);
        Gather(gateORowReg0, gateOVectorReg, rowIndexReg0);
        LoadAlign(oSReg00, oSAddr + offset0);
        LoadAlign(oSReg01, oSAddr + offset1);
        Mul(oSReg00, oSReg00, gateORowReg0, floatMask);
        Mul(oSReg01, oSReg01, gateORowReg0, floatMask);
        StoreAlign(oSAddr + offset0, oSReg00, floatMask);
        StoreAlign(oSAddr + offset1, oSReg01, floatMask);
    }

    for (uint16_t row = validRows; row < kBt; ++row) {
        const uint32_t offset0 = static_cast<uint32_t>(row) * CHUNK_FWD_O_A5_V;
        StoreAlign(oSAddr + offset0, zeroReg, floatMask);
        StoreAlign(oSAddr + offset0 + kBt, zeroReg, floatMask);
    }
}

__simd_vf__ inline void Stage5Fuse64VF(__ubuf__ bfloat16_t *oOutAddr, __ubuf__ float *oSPrimeAddr,
                                      __ubuf__ float *oLAddr, float scale, uint16_t validRows)
{
    constexpr uint16_t kV = static_cast<uint16_t>(CHUNK_FWD_O_A5_V);
    constexpr uint16_t kTilesPerRow = static_cast<uint16_t>(CHUNK_FWD_O_A5_V / CHUNK_FWD_O_A5_BT);
    RegTensor<float> oSPrimeReg;
    RegTensor<float> oLReg;
    RegTensor<float> oOutReg;
    RegTensor<bfloat16_t> oOutBf16Reg;
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t row = 0; row < validRows; ++row) {
        for (uint16_t tile = 0; tile < kTilesPerRow; ++tile) {
            const uint32_t offset = static_cast<uint32_t>(row) * kV +
                                    static_cast<uint32_t>(tile) * CHUNK_FWD_O_A5_BT;
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
class ChunkFwdOA5VectorProcess {
public:
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr uint32_t BANK_COUNT_2 = CHUNK_FWD_O_STREAM_BANK_COUNT;

    __aicore__ inline ChunkFwdOA5VectorProcess(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g,
                                               GM_ADDR cuSeqlens, GM_ADDR chunkOffsets, GM_ADDR o, GM_ADDR workspace)
        : q_(q), k_(k), v_(v), h_(h), g_(g), cuSeqlens_(cuSeqlens), chunkOffsets_(chunkOffsets), o_(o),
          workspace_(workspace)
    {
    }

    __aicore__ inline void Init(const ChunkFwdOTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        oGm_.SetGlobalBuffer((__gm__ bfloat16_t *)o_);

        pipe_->InitBuffer(ubBuf_, CHUNK_FWD_O_UB_TOTAL_BYTES);
        for (uint32_t bankIdx = 0; bankIdx < BANK_COUNT_2; ++bankIdx) {
            mte2ToV_[bankIdx] = pipe_->AllocEventID<HardEvent::MTE2_V>();
            vToMte3Stream_[bankIdx] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            mte3ToVStream_[bankIdx] = pipe_->AllocEventID<HardEvent::MTE3_V>();
            // No previous MTE3 owns either Stage3 output slot on the first group.
            vToMte3Event_[bankIdx] = pipe_->AllocEventID<HardEvent::V_MTE3>();
            SetFlag<HardEvent::MTE3_V>(mte3ToVStream_[bankIdx]);
        }
    }

    // Keep the AIV stage order in one place. Cross-core and local pipeline
    // events are intentionally adjacent to the head that consumes or produces
    // the corresponding data.
    __aicore__ inline void Process(uint32_t coreIdx, uint32_t coreNum)
    {
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        constexpr uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        constexpr uint32_t vDim = static_cast<uint32_t>(CHUNK_FWD_O_A5_V);
        const uint32_t matrixElems = bt * bt;
        ChunkFwdOChunkLoc loc;

        const uint32_t headGroupNum = ChunkFwdOHeadGroupNum(tiling_);
        const uint32_t groupTaskNum = static_cast<uint32_t>(tiling_.chunkNum) * headGroupNum;
        for (uint32_t groupTaskIdx = coreIdx; groupTaskIdx < groupTaskNum; groupTaskIdx += coreNum) {
            const uint32_t loopIdx = groupTaskIdx / headGroupNum;
            const uint32_t headGroupIdx = groupTaskIdx % headGroupNum;
            const int64_t hvBase = static_cast<int64_t>(headGroupIdx) * tiling_.taskGroupSize;
            ChunkFwdOResolveChunkLoc(cuSeqlens_, chunkOffsets_, tiling_, loopIdx, loc);
            const int64_t remaining = tiling_.vNumHead - hvBase;
            const int64_t taskCount = remaining < tiling_.taskGroupSize ? remaining : tiling_.taskGroupSize;
            const int64_t groupRound = ChunkFwdOGroupRound(groupTaskIdx, coreNum);

            // Stage 1: prepare gate_o/gate_A for every owner HEAD. The
            // producer/consumer events stay at the exact transfer site.
            streamSlot_ = 0U;
            for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                if (ownerSubBlock != subBlockIdx) {
                    continue;
                }
                LoadStage1G(loc, hvBase + headOffset, streamSlot_);
                SetFlag<HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                WaitFlag<HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                ComputeStage1Gate(loc, streamSlot_, streamSlot_);
                streamSlot_ ^= 1U;
            }

            // Stage 3: consume one Stage 2 ready per HEAD. Both subblocks
            // participate in the handshake; only the owner executes the
            // VF and writes A-prime to GM.
            // Select the first Stage3 ping-pong UB slot for this task group.
            streamSlot_ = 0U;
            for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                if (ownerSubBlock == subBlockIdx) {
                    LocalTensor<float> gateO =
                        ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(streamSlot_));
                    LocalTensor<float> gateA =
                        ubBuf_.GetWithOffset<float>(matrixElems, ChunkFwdOGateAOffset(streamSlot_));
                    LocalTensor<float> aRaw =
                        ubBuf_.GetWithOffset<float>(matrixElems, ChunkFwdOARawOffset(streamSlot_));
                    LocalTensor<float> oSPrime =
                        ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOsPrimeOffset(streamSlot_));
                    LocalTensor<bfloat16_t> aPrimeBf16 =
                        ubBuf_.GetWithOffset<bfloat16_t>(matrixElems,
                                                         ChunkFwdOAPrimeBf16Offset(streamSlot_));
                    WaitFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot_]);
                    PipeBarrier<PIPE_V>();
                    AscendC::VF_CALL<Stage3Gate64VF>(
                        reinterpret_cast<__ubuf__ bfloat16_t *>(aPrimeBf16.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(oSPrime.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(aRaw.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gateA.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gateO.GetPhyAddr()),
                        static_cast<uint16_t>(loc.chunkLen));
                    SetFlag<HardEvent::V_MTE3>(vToMte3Stream_[streamSlot_]);
                    WaitFlag<HardEvent::V_MTE3>(vToMte3Stream_[streamSlot_]);

                    const uint32_t aivCoreIdx = AscendC::GetBlockIdx() / 2U;
                    GM_ADDR aPrimeAddr =
                        ChunkFwdOAPrimeGmOffset(workspace_, tiling_, aivCoreIdx, groupRound, headOffset);
                    GlobalTensor<bfloat16_t> aPrimeGm;
                    aPrimeGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(aPrimeAddr));
                    DataCopyExtParams aPrimeCopyParams{1, CHUNK_FWD_O_APRIME_SLOT_BYTES, 0, 0, 0};
                    DataCopyPad(aPrimeGm, aPrimeBf16, aPrimeCopyParams);

                    SetFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot_]);
                    streamSlot_ ^= 1U;
                }
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
            }

            // Stage 5: consume one Stage 4 ready per HEAD, then publish
            // the final output only from the owning subblock.
            streamSlot_ = 0U;
            for (int64_t headOffset = 0; headOffset < taskCount; ++headOffset) {
                const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % 2);
                const int64_t hv = hvBase + headOffset;
                Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                if (ownerSubBlock != subBlockIdx) {
                    continue;
                }

                LocalTensor<float> oSPrime =
                    ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOsPrimeOffset(streamSlot_));
                LocalTensor<float> oL =
                    ubBuf_.GetWithOffset<float>(bt * vDim, ChunkFwdOOlOffset(streamSlot_));
                LocalTensor<bfloat16_t> oOut =
                    ubBuf_.GetWithOffset<bfloat16_t>(bt * vDim, ChunkFwdOOlOffset(streamSlot_));
                WaitFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot_]);
                PipeBarrier<PIPE_V>();
                AscendC::VF_CALL<Stage5Fuse64VF>(
                    reinterpret_cast<__ubuf__ bfloat16_t *>(oOut.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(oSPrime.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(oL.GetPhyAddr()), tiling_.scale,
                    static_cast<uint16_t>(loc.chunkLen));
                SetFlag<HardEvent::V_MTE3>(vToMte3Event_[streamSlot_]);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Event_[streamSlot_]);
                const int64_t oOffset = ChunkFwdOOOffset(tiling_, loc, hv);
                DataCopyExtParams outputCopyParams{
                    static_cast<uint16_t>(loc.chunkLen),
                    static_cast<uint32_t>(vDim * sizeof(bfloat16_t)),
                    0,
                    static_cast<uint32_t>((tiling_.vNumHead - 1) * vDim * sizeof(bfloat16_t)),
                    0};
                DataCopyPad(oGm_[oOffset], oOut, outputCopyParams);
                SetFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot_]);
                streamSlot_ ^= 1U;
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
        }

        // Reuse waits consume the preceding MTE3 completion directly. Only
        // the last completion of each slot needs to be drained at kernel end.
        for (uint32_t streamSlot = 0; streamSlot < BANK_COUNT_2; ++streamSlot) {
            WaitFlag<HardEvent::MTE3_V>(mte3ToVStream_[streamSlot]);
        }
    }

    __aicore__ inline void LoadStage1G(const ChunkFwdOChunkLoc &loc, int64_t hv, uint32_t streamSlot)
    {
        GlobalTensor<GT> gGm;
        gGm.SetGlobalBuffer((__gm__ GT *)g_);
        const int64_t gOffset = ChunkFwdOGOffset(tiling_, loc, hv);
        const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
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

    __aicore__ inline void ComputeStage1Gate(const ChunkFwdOChunkLoc &loc, uint32_t localSlot, uint32_t streamSlot)
    {
        const uint32_t bt = static_cast<uint32_t>(CHUNK_FWD_O_A5_BT);
        const uint32_t gScratchOff = ChunkFwdOGScratchOffset(streamSlot);

        LocalTensor<float> gFp32 = ubBuf_.GetWithOffset<float>(bt, gScratchOff);
        if constexpr (!std::is_same<GT, float>::value) {
            LocalTensor<GT> gLocal = ubBuf_.GetWithOffset<GT>(bt, gScratchOff);
            Cast(gFp32, gLocal, RoundMode::CAST_NONE, loc.chunkLen);
        }
        if (loc.chunkLen < static_cast<uint32_t>(CHUNK_FWD_O_A5_BT)) {
            AscendC::VF_CALL<PadGateInput64VF>(
                reinterpret_cast<__ubuf__ float *>(gFp32.GetPhyAddr()),
                static_cast<uint16_t>(loc.chunkLen));
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> gateO = ubBuf_.GetWithOffset<float>(bt, ChunkFwdOGateOOffset(localSlot));
        LocalTensor<float> gateA =
            ubBuf_.GetWithOffset<float>(bt * bt, ChunkFwdOGateAOffset(localSlot));
        if constexpr (UseExp2) {
            Muls(gateO, gFp32, CHUNK_FWD_O_LN2, bt);
            PipeBarrier<PIPE_V>();
            Exp(gateO, gateO, bt);
        } else {
            Exp(gateO, gFp32, bt);
        }
        PipeBarrier<PIPE_V>();
        AscendC::VF_CALL<Stage1Gate64VF<UseExp2>>(
            (__ubuf__ float *)gateA.GetPhyAddr(), (__ubuf__ float *)gFp32.GetPhyAddr(),
            static_cast<uint16_t>(loc.chunkLen));
        PipeBarrier<PIPE_V>();
    }

private:
    GM_ADDR q_;
    GM_ADDR k_;
    GM_ADDR v_;
    GM_ADDR h_;
    GM_ADDR g_;
    GM_ADDR cuSeqlens_;
    GM_ADDR chunkOffsets_;
    GM_ADDR o_;
    GM_ADDR workspace_;
    ChunkFwdOTilingData tiling_{};
    TPipe *pipe_ = nullptr;

    TBuf<TPosition::VECCALC> ubBuf_;
    GlobalTensor<bfloat16_t> oGm_;
    uint32_t streamSlot_ = 0;
    TEventID mte2ToV_[BANK_COUNT_2];
    TEventID vToMte3Stream_[BANK_COUNT_2];
    TEventID mte3ToVStream_[BANK_COUNT_2];
    TEventID vToMte3Event_[BANK_COUNT_2];
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{CHUNK_FWD_O_VEC_TO_CUBE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CHUNK_FWD_O_CUBE_TO_VEC_READY_FLAG};
};

} // namespace GDN

#endif // CHUNK_FWD_O_ARCH35_VECTOR_H
