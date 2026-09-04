/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * Reference copy of solve_tri VCS helpers. Not included by the fused
 * prepare kernel (MulReduceScatterVF32 lives in arch35/vf.h).
 * Do not edit VF bodies.
 */

#ifndef SOLVE_TRI_ASCEND950_COMMON_H
#define SOLVE_TRI_ASCEND950_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;

// 8x16 ND 块对角 mask
// [0] = ODD  (奇数条带, 对角在 col 8..15)
// [1] = EVEN (偶数条带, 对角在 col 0..7)
constexpr uint64_t DIAG_MASK_8X16[2][2] = {
    { 0x0800040002000100ULL, 0x8000400020001000ULL },  // ODD
    { 0x0008000400020001ULL, 0x0080004000200010ULL }   // EVEN
};

constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
constexpr int64_t DATA_BLOCK_COUNT = 16;
constexpr int64_t DATA_BLOCK_COUNT_HALF = 8;
constexpr int32_t kFracLen = 16 * 16;
constexpr int32_t kFracLen8 = 16 * 8; // FP32 cube 分型
constexpr int32_t CONST_TWO = 2;

// chunk64：每 Vector 2 个 32×32 叶子，ND 打包 32×64
constexpr uint32_t kVcs32 = 32;
constexpr uint32_t kVcsPack32 = 64;
constexpr uint32_t kVcsPackedElems32 = kVcs32 * kVcsPack32; // 2048
constexpr uint32_t kLeavesPerVec32 = 2;
constexpr uint32_t kVcs32NzElems = kVcs32 * kVcs32; // 1024，单叶 32×32 NZ

constexpr uint64_t SOLVE_TRI_TILING_KEY_64 = 64;

template <typename T>
__aicore__ inline void TransposeB32(LocalTensor<T> dst, LocalTensor<T> src, uint32_t curTileALen)
{
    int32_t rRepeartTimes = 16 / 8;
    for (int32_t i = 0; i < rRepeartTimes; i++) {
        TransDataTo5HDParams params;
        LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
        LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];

        uint32_t aRepeartTimes = (curTileALen + static_cast<uint32_t>(DATA_BLOCK_COUNT) - 1) /
                                 static_cast<uint32_t>(DATA_BLOCK_COUNT);
        params.repeatTimes = aRepeartTimes;
        params.srcRepStride = aRepeartTimes == 1 ? 0 : CONST_TWO;
        params.dstRepStride = aRepeartTimes == 1 ? 0 : DATA_BLOCK_COUNT * rRepeartTimes;
        for (int32_t j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
            uint32_t offset = DATA_BLOCK_COUNT_HALF * curTileALen * i + curTileALen * j;
            srcLocalList[j] = src[offset];
            srcLocalList[j + DATA_BLOCK_COUNT_HALF] = src[offset + DATA_BLOCK_COUNT_HALF];
        }
        for (int32_t j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
            uint32_t offset = DATA_BLOCK_COUNT_HALF * i + DATA_BLOCK_COUNT_HALF * rRepeartTimes * j;
            dstLocalList[j * CONST_TWO] = dst[offset];
            dstLocalList[j * CONST_TWO + 1] = dst[offset + DATA_BLOCK_COUNT_HALF * DATA_BLOCK_COUNT_HALF * rRepeartTimes];
        }

        AscendC::TransDataTo5HD(dstLocalList, srcLocalList, params);
    }
}

// UB 上 FP32 NZ 16×16 分型重排为 cube 所需的 16×8 分型。
// 每个 16-wide 列拆成 2 个 8-wide 列：blockLen=1（8 个 float），srcStride=1 跳过另一半。
template <typename T>
__aicore__ inline void NzFp32Blk16ToBlk8(LocalTensor<T> dst, LocalTensor<T> src, uint32_t n)
{
    const uint32_t numFracs = n / 16;
    for (uint32_t i = 0; i < numFracs; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            AscendC::DataCopy(dst[(i * 2 + j) * 8 * n], src[i * 16 * n + j * 8],
                              AscendC::DataCopyParams(static_cast<uint16_t>(n), 1, 1, 0));
        }
    }
}

// VCS NZ 叶子写 GM。每个叶子 16×16 连续占 kFracLen。
template <typename T>
__aicore__ inline void WriteVcsNzLeafMte3(GlobalTensor<T> gmOut, LocalTensor<T> ubNz,
                                          uint32_t leafIdx, uint32_t actualRows, uint32_t rowStride,
                                          int64_t gmOffset)
{
    const uint16_t dstBlkStride = static_cast<uint16_t>(rowStride / 16 - 1);
    AscendC::DataCopy(gmOut[gmOffset], ubNz[leafIdx * kFracLen],
                      AscendC::DataCopyParams(static_cast<uint16_t>(actualRows), 1,
                                              0, dstBlkStride));
}

// 32×32 叶子 VF。两叶打包 32×64：oneRepeatSize=64，scatterCount=2。
template <typename T, typename U>
__simd_vf__ inline void MulReduceScatterVF32(__ubuf__ T *dstAddr, __ubuf__ T *src0Addr, __ubuf__ T *src1Addr,
                                             __ubuf__ U *idxAddr, uint32_t scatterCount, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> srcReg0;
    AscendC::Reg::RegTensor<T> srcReg1;
    AscendC::Reg::RegTensor<T> dstMulReg;
    AscendC::Reg::RegTensor<T> dstReduceBlkReg;
    AscendC::Reg::RegTensor<T> dstReducePairReg;
    AscendC::Reg::RegTensor<T> dstReducePairReg2;
    AscendC::Reg::RegTensor<U> scatterIdxReg;

    uint32_t maskCount = oneRepeatSize;
    uint32_t pairMaskCount = scatterCount * 2;
    AscendC::Reg::MaskReg inputMask;
    AscendC::Reg::MaskReg pairMask;
    AscendC::Reg::MaskReg scatterMask;
    inputMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(maskCount);
    pairMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(pairMaskCount);
    scatterMask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(scatterCount);
    for (uint16_t iterIdx = 1; iterIdx < 32; iterIdx++) {
        AscendC::Reg::LoadAlign(scatterIdxReg, idxAddr);
        AscendC::Reg::Adds(scatterIdxReg, scatterIdxReg, (uint32_t)iterIdx, scatterMask);
        for (uint16_t i = 0; i < iterIdx; i++) {
            AscendC::Reg::LoadAlign(srcReg0, src0Addr + iterIdx * oneRepeatSize);
            AscendC::Reg::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
            AscendC::Reg::Mul(dstMulReg, srcReg0, srcReg1, inputMask);
            AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(dstReduceBlkReg, dstMulReg, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(dstReducePairReg, dstReduceBlkReg, inputMask);
            AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(dstReducePairReg2, dstReducePairReg, pairMask);
            AscendC::Reg::Scatter(dstAddr + i * oneRepeatSize, dstReducePairReg2, scatterIdxReg, scatterMask);
        }
        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    }
}

// 打包 32×64 ND（VF 双叶结果）→ 两个连续 32×32 NZ。
// TransposeB32 一次处理 16 行×64 列，故 32 行调两次；tmp 需 64×32。
template <typename T>
__aicore__ inline void TransposeB32Vcs32(LocalTensor<T> dstTwoNz32, LocalTensor<T> srcNd32x64,
                                         LocalTensor<T> tmp64x32)
{
    constexpr uint32_t kBlk = static_cast<uint32_t>(kFracLen);
    TransposeB32(tmp64x32, srcNd32x64, kVcsPack32);
    TransposeB32(tmp64x32[16 * kVcsPack32], srcNd32x64[16 * kVcsPack32], kVcsPack32);
    SetFlag<AscendC::HardEvent::V_MTE3>(2);
    WaitFlag<AscendC::HardEvent::V_MTE3>(2);
    AscendC::DataCopy(dstTwoNz32[0], tmp64x32[0], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[kBlk], tmp64x32[kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[2 * kBlk], tmp64x32[4 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[3 * kBlk], tmp64x32[5 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[4 * kBlk], tmp64x32[2 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[5 * kBlk], tmp64x32[3 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[6 * kBlk], tmp64x32[6 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    AscendC::DataCopy(dstTwoNz32[7 * kBlk], tmp64x32[7 * kBlk], AscendC::DataCopyParams(1, kBlk / 8, 0, 0));
    SetFlag<AscendC::HardEvent::MTE3_V>(2);
    WaitFlag<AscendC::HardEvent::MTE3_V>(2);
}

#endif  // SOLVE_TRI_ASCEND950_COMMON_H
