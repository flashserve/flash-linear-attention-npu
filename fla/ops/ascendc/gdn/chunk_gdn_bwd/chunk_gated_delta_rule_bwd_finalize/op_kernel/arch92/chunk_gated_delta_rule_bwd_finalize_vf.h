#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_VF_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ARCH92_VF_H

#include <type_traits>

#include "chunk_gated_delta_rule_bwd_finalize_common.h"

namespace GDN::FinalizeArch92 {

using namespace AscendC::MicroAPI;

constexpr float LN2 = 0.6931471805599453f;
constexpr CastTrait B16_TO_FP32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr CastTrait FP32_TO_B16_PACK = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_ROUND,
};

// Stage 0 每次 VF 只处理当前槽位对应的一个完整 HV。
template <typename DT, typename GT, typename BT, bool USE_EXP2>
__simd_vf__ inline void Stage0VF(
    __ubuf__ DT *gateDA, __ubuf__ DT *kbg, __ubuf__ DT *vb,
    __ubuf__ float *gExp, __ubuf__ float *gLast, __ubuf__ float *decay,
    __ubuf__ float *bg, __ubuf__ float *betaSaved, __ubuf__ DT *k, __ubuf__ DT *v,
    __ubuf__ GT *g, __ubuf__ BT *beta, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);

    RegTensor<GT> gInputReg;
    RegTensor<BT> betaInputReg;
    RegTensor<float> gReg;
    RegTensor<float> betaReg;
    RegTensor<float> gExpReg;
    RegTensor<float> gLastReg;
    RegTensor<float> decayReg;
    RegTensor<float> bgReg;
    RegTensor<uint32_t> indexReg0;
    RegTensor<uint32_t> indexReg1;
    RegTensor<uint32_t> indexReg2;
    RegTensor<uint32_t> indexReg3;
    RegTensor<float> rowReg0;
    RegTensor<float> rowReg1;
    RegTensor<float> rowReg2;
    RegTensor<float> rowReg3;
    RegTensor<float> resultReg0;
    RegTensor<float> resultReg1;
    RegTensor<float> resultReg2;
    RegTensor<float> resultReg3;
    RegTensor<float> resultReg4;
    RegTensor<float> resultReg5;
    RegTensor<float> resultReg6;
    RegTensor<float> resultReg7;
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<DT> dataReg4;
    RegTensor<DT> dataReg5;
    RegTensor<DT> dataReg6;
    RegTensor<DT> dataReg7;

    uint16_t vectorRowLoop = 0;
    uint16_t vectorRowLoopCount = validRows / 2;
    uint16_t vectorTailLoop = 0;
    uint16_t vectorTailLoopCount = validRows % 2;
    uint16_t vectorTailRowBase = vectorRowLoopCount * 2;
    uint16_t gateRowLoop = 0;
    uint16_t gateRowLoopCount = validRows / 4;
    uint16_t gateTailLoop = 0;
    uint16_t gateTailLoopCount = validRows % 4;
    uint16_t gateTailRowBase = gateRowLoopCount * 4;
    uint16_t rowBase = 0;
    uint32_t vectorOffset0 = 0;
    uint32_t vectorOffset1 = 0;
    uint32_t vectorOffset2 = 0;
    uint32_t vectorOffset3 = 0;

    // 原始输入保持各自 dtype 驻留，首次读取时才在寄存器内转换 FP32。
    if constexpr (std::is_same<GT, float>::value) {
        LoadAlign(gReg, g);
    } else {
        DataCopy<GT, LoadDist::DIST_UNPACK_B16>(gInputReg, g);
        Cast<float, GT, B16_TO_FP32>(gReg, gInputReg, validMask);
    }
    if constexpr (std::is_same<BT, float>::value) {
        LoadAlign(betaReg, beta);
    } else {
        DataCopy<BT, LoadDist::DIST_UNPACK_B16>(betaInputReg, beta);
        Cast<float, BT, B16_TO_FP32>(betaReg, betaInputReg, validMask);
        // BF16 beta 只扩展一次，当前 VF 后直接以 FP32
        // 保留在同一 HV 的固定 resident，供后续 Stage 复用。
        StoreAlign(betaSaved, betaReg, validMask);
    }

    // gLastReg 直接从已经转换好的 gReg Gather 最后一个有效元素，
    // 先供 decay 使用，再按 use_exp2 转换为对应指数值。gLast 按完整
    // FP32[BT] 广播保存，避免后续 VF 特判 scalar。
    if constexpr (USE_EXP2) {
        Muls(gExpReg, gReg, LN2, validMask);
    } else {
        Adds(gExpReg, gReg, 0.0f, validMask);
    }
    Duplicate(indexReg0, static_cast<uint32_t>(validRows - 1), floatMask);
    Gather(gLastReg, gReg, indexReg0);
    Sub(decayReg, gLastReg, gReg, validMask);
    if constexpr (USE_EXP2) {
        Muls(decayReg, decayReg, LN2, validMask);
        Muls(gLastReg, gLastReg, LN2, floatMask);
    }
    Exp(gExpReg, gExpReg, validMask);
    Exp(decayReg, decayReg, validMask);
    Exp(gLastReg, gLastReg, floatMask);
    Mul(bgReg, betaReg, gExpReg, validMask);
    StoreAlign(gExp, gExpReg, validMask);
    StoreAlign(gLast, gLastReg, floatMask);
    StoreAlign(decay, decayReg, validMask);
    StoreAlign(bg, bgReg, validMask);

    // kbg/vb 按两行展开。同一组行索引同时从 bgReg/betaReg
    // Gather 广播，不将刚写入 resident 的 bg 再从 UB 读回，因此不需要
    // VEC_STORE -> VEC_LOAD 局部内存屏障。
    for (vectorRowLoop = 0; vectorRowLoop < vectorRowLoopCount; ++vectorRowLoop) {
        rowBase = vectorRowLoop * 2;
        Duplicate(indexReg0, static_cast<uint32_t>(rowBase), floatMask);
        Duplicate(indexReg1, static_cast<uint32_t>(rowBase + 1), floatMask);
        Gather(rowReg0, bgReg, indexReg0);
        Gather(rowReg1, betaReg, indexReg0);
        Gather(rowReg2, bgReg, indexReg1);
        Gather(rowReg3, betaReg, indexReg1);
        vectorOffset0 = static_cast<uint32_t>(rowBase) * K_SIZE_128;
        vectorOffset1 = static_cast<uint32_t>(rowBase + 1) * K_SIZE_128;
        vectorOffset2 = vectorOffset0 + CHUNK_SIZE_64;
        vectorOffset3 = vectorOffset1 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, k + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, v + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, k + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, v + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg4, k + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg5, v + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg6, k + vectorOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg7, v + vectorOffset3);
        Cast<float, DT, B16_TO_FP32>(resultReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg1, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg2, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg3, dataReg3, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg4, dataReg4, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg5, dataReg5, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg6, dataReg6, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg7, dataReg7, floatMask);
        Mul(resultReg0, resultReg0, rowReg0, floatMask);
        Mul(resultReg1, resultReg1, rowReg1, floatMask);
        Mul(resultReg2, resultReg2, rowReg2, floatMask);
        Mul(resultReg3, resultReg3, rowReg3, floatMask);
        Mul(resultReg4, resultReg4, rowReg0, floatMask);
        Mul(resultReg5, resultReg5, rowReg1, floatMask);
        Mul(resultReg6, resultReg6, rowReg2, floatMask);
        Mul(resultReg7, resultReg7, rowReg3, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, resultReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, resultReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg3, resultReg3, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg4, resultReg4, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg5, resultReg5, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg6, resultReg6, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg7, resultReg7, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kbg + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vb + vectorOffset0, dataReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kbg + vectorOffset1, dataReg2, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vb + vectorOffset1, dataReg3, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kbg + vectorOffset2, dataReg4, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vb + vectorOffset2, dataReg5, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kbg + vectorOffset3, dataReg6, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vb + vectorOffset3, dataReg7, floatMask);
    }
    for (vectorTailLoop = 0; vectorTailLoop < vectorTailLoopCount; ++vectorTailLoop) {
        rowBase = vectorTailRowBase + vectorTailLoop;
        Duplicate(indexReg0, static_cast<uint32_t>(rowBase), floatMask);
        Gather(rowReg0, bgReg, indexReg0);
        Gather(rowReg1, betaReg, indexReg0);
        vectorOffset0 = static_cast<uint32_t>(rowBase) * K_SIZE_128;
        vectorOffset1 = vectorOffset0 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, k + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, v + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, k + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, v + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(resultReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg1, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg2, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg3, dataReg3, floatMask);
        Mul(resultReg0, resultReg0, rowReg0, floatMask);
        Mul(resultReg1, resultReg1, rowReg1, floatMask);
        Mul(resultReg2, resultReg2, rowReg0, floatMask);
        Mul(resultReg3, resultReg3, rowReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, resultReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, resultReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg3, resultReg3, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kbg + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vb + vectorOffset0, dataReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kbg + vectorOffset1, dataReg2, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vb + vectorOffset1, dataReg3, floatMask);
    }

    // gate_dA 按四行展开。gReg 已经是完整 FP32[BT]，同样通过
    // RegTensor Gather 生成四个行广播，不重新读取原始 g 或重复 Cast。
    // 每行只计算前 M 列；尾部物理空间没有消费者，不做无效清零。
    for (gateRowLoop = 0; gateRowLoop < gateRowLoopCount; ++gateRowLoop) {
        rowBase = gateRowLoop * 4;
        Duplicate(indexReg0, static_cast<uint32_t>(rowBase), floatMask);
        Duplicate(indexReg1, static_cast<uint32_t>(rowBase + 1), floatMask);
        Duplicate(indexReg2, static_cast<uint32_t>(rowBase + 2), floatMask);
        Duplicate(indexReg3, static_cast<uint32_t>(rowBase + 3), floatMask);
        Gather(rowReg0, gReg, indexReg0);
        Gather(rowReg1, gReg, indexReg1);
        Gather(rowReg2, gReg, indexReg2);
        Gather(rowReg3, gReg, indexReg3);
        Sub(resultReg0, rowReg0, gReg, validMask);
        Sub(resultReg1, rowReg1, gReg, validMask);
        Sub(resultReg2, rowReg2, gReg, validMask);
        Sub(resultReg3, rowReg3, gReg, validMask);
        if constexpr (USE_EXP2) {
            Muls(resultReg0, resultReg0, LN2, validMask);
            Muls(resultReg1, resultReg1, LN2, validMask);
            Muls(resultReg2, resultReg2, LN2, validMask);
            Muls(resultReg3, resultReg3, LN2, validMask);
        }
        Exp(resultReg0, resultReg0, validMask);
        Exp(resultReg1, resultReg1, validMask);
        Exp(resultReg2, resultReg2, validMask);
        Exp(resultReg3, resultReg3, validMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, resultReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, resultReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg3, resultReg3, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(
            gateDA + static_cast<uint32_t>(rowBase) * CHUNK_SIZE_64, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(
            gateDA + static_cast<uint32_t>(rowBase + 1) * CHUNK_SIZE_64, dataReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(
            gateDA + static_cast<uint32_t>(rowBase + 2) * CHUNK_SIZE_64, dataReg2, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(
            gateDA + static_cast<uint32_t>(rowBase + 3) * CHUNK_SIZE_64, dataReg3, floatMask);
    }
    for (gateTailLoop = 0; gateTailLoop < gateTailLoopCount; ++gateTailLoop) {
        rowBase = gateTailRowBase + gateTailLoop;
        Duplicate(indexReg0, static_cast<uint32_t>(rowBase), floatMask);
        Gather(rowReg0, gReg, indexReg0);
        Sub(resultReg0, rowReg0, gReg, validMask);
        if constexpr (USE_EXP2) {
            Muls(resultReg0, resultReg0, LN2, validMask);
        }
        Exp(resultReg0, resultReg0, validMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(
            gateDA + static_cast<uint32_t>(rowBase) * CHUNK_SIZE_64, dataReg0, floatMask);
    }
}

// Stage 2 每次 VF 只处理一个 HV。dA_u 在原物理地址生成 strict-lower
// dA_u_lower；kb 使用独立输出槽，随后由 MTE3 搬入 L1 resident。
template <typename DT>
__simd_vf__ inline void Stage2VF(
    __ubuf__ DT *dAu, __ubuf__ DT *kb, __ubuf__ DT *k,
    __ubuf__ float *beta, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg lowerMask0;
    MaskReg lowerMask1;
    MaskReg lowerMask2;
    MaskReg lowerMask3;

    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<DT> dataReg4;
    RegTensor<DT> dataReg5;
    RegTensor<DT> dataReg6;
    RegTensor<DT> dataReg7;
    RegTensor<float> inputReg0;
    RegTensor<float> inputReg1;
    RegTensor<float> inputReg2;
    RegTensor<float> inputReg3;
    RegTensor<float> resultReg0;
    RegTensor<float> resultReg1;
    RegTensor<float> resultReg2;
    RegTensor<float> resultReg3;
    RegTensor<float> resultReg4;
    RegTensor<float> resultReg5;
    RegTensor<float> resultReg6;
    RegTensor<float> resultReg7;
    RegTensor<float> zeroReg;
    RegTensor<float> betaReg;
    RegTensor<float> betaRowReg0;
    RegTensor<float> betaRowReg1;
    RegTensor<float> betaRowReg2;
    RegTensor<float> betaRowReg3;
    RegTensor<uint32_t> indexReg0;
    RegTensor<uint32_t> indexReg1;
    RegTensor<uint32_t> indexReg2;
    RegTensor<uint32_t> indexReg3;

    uint16_t matrixRowLoop = 0;
    uint16_t matrixRowLoopCount = validRows / 4;
    uint16_t matrixTailLoop = 0;
    uint16_t matrixTailLoopCount = validRows % 4;
    uint16_t matrixTailRowBase = matrixRowLoopCount * 4;
    uint16_t vectorRowLoop = 0;
    uint16_t vectorRowLoopCount = validRows / 4;
    uint16_t vectorTailLoop = 0;
    uint16_t vectorTailLoopCount = validRows % 4;
    uint16_t vectorTailRowBase = vectorRowLoopCount * 4;
    uint16_t rowBase = 0;
    uint32_t rowOffset0 = 0;
    uint32_t rowOffset1 = 0;
    uint32_t rowOffset2 = 0;
    uint32_t rowOffset3 = 0;
    uint32_t rowOffset4 = 0;
    uint32_t rowOffset5 = 0;
    uint32_t rowOffset6 = 0;
    uint32_t rowOffset7 = 0;
    uint32_t lowerCount0 = 0;
    uint32_t lowerCount1 = 0;
    uint32_t lowerCount2 = 0;
    uint32_t lowerCount3 = 0;

    Duplicate(zeroReg, 0.0f, floatMask);
    // 四行展开原位生成 strict-lower。矩阵有效区中的对角线和上三角
    // 后续会被 Stage 4 读取，因此必须写成 0；物理尾列没有消费者，
    // Store 只覆盖 validRows 列，不清零无语义 padding。
    for (matrixRowLoop = 0; matrixRowLoop < matrixRowLoopCount; ++matrixRowLoop) {
        rowBase = matrixRowLoop * 4;
        rowOffset0 = static_cast<uint32_t>(rowBase) * CHUNK_SIZE_64;
        rowOffset1 = static_cast<uint32_t>(rowBase + 1) * CHUNK_SIZE_64;
        rowOffset2 = static_cast<uint32_t>(rowBase + 2) * CHUNK_SIZE_64;
        rowOffset3 = static_cast<uint32_t>(rowBase + 3) * CHUNK_SIZE_64;
        lowerCount0 = static_cast<uint32_t>(rowBase);
        lowerCount1 = static_cast<uint32_t>(rowBase + 1);
        lowerCount2 = static_cast<uint32_t>(rowBase + 2);
        lowerCount3 = static_cast<uint32_t>(rowBase + 3);
        lowerMask0 = UpdateMask<float>(lowerCount0);
        lowerMask1 = UpdateMask<float>(lowerCount1);
        lowerMask2 = UpdateMask<float>(lowerCount2);
        lowerMask3 = UpdateMask<float>(lowerCount3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dAu + rowOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dAu + rowOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dAu + rowOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, dAu + rowOffset3);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg1, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg2, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg3, dataReg3, floatMask);
        Select(resultReg0, inputReg0, zeroReg, lowerMask0);
        Select(resultReg1, inputReg1, zeroReg, lowerMask1);
        Select(resultReg2, inputReg2, zeroReg, lowerMask2);
        Select(resultReg3, inputReg3, zeroReg, lowerMask3);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, resultReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, resultReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg3, resultReg3, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dAu + rowOffset0, dataReg0, validMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dAu + rowOffset1, dataReg1, validMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dAu + rowOffset2, dataReg2, validMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dAu + rowOffset3, dataReg3, validMask);
    }
    for (matrixTailLoop = 0; matrixTailLoop < matrixTailLoopCount; ++matrixTailLoop) {
        rowBase = matrixTailRowBase + matrixTailLoop;
        rowOffset0 = static_cast<uint32_t>(rowBase) * CHUNK_SIZE_64;
        lowerCount0 = static_cast<uint32_t>(rowBase);
        lowerMask0 = UpdateMask<float>(lowerCount0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dAu + rowOffset0);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg0, floatMask);
        Select(resultReg0, inputReg0, zeroReg, lowerMask0);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dAu + rowOffset0, dataReg0, validMask);
    }

    // beta 已由 Stage 0 以 FP32 常驻。四行展开并按两个 64 元素列块
    // 完成 kb=k*beta；原始 k 保持不变，供后续 Stage 继续使用。
    LoadAlign(betaReg, beta);
    for (vectorRowLoop = 0; vectorRowLoop < vectorRowLoopCount; ++vectorRowLoop) {
        rowBase = vectorRowLoop * 4;
        Duplicate(indexReg0, static_cast<uint32_t>(rowBase), floatMask);
        Duplicate(indexReg1, static_cast<uint32_t>(rowBase + 1), floatMask);
        Duplicate(indexReg2, static_cast<uint32_t>(rowBase + 2), floatMask);
        Duplicate(indexReg3, static_cast<uint32_t>(rowBase + 3), floatMask);
        Gather(betaRowReg0, betaReg, indexReg0);
        Gather(betaRowReg1, betaReg, indexReg1);
        Gather(betaRowReg2, betaReg, indexReg2);
        Gather(betaRowReg3, betaReg, indexReg3);
        rowOffset0 = static_cast<uint32_t>(rowBase) * K_SIZE_128;
        rowOffset1 = static_cast<uint32_t>(rowBase + 1) * K_SIZE_128;
        rowOffset2 = static_cast<uint32_t>(rowBase + 2) * K_SIZE_128;
        rowOffset3 = static_cast<uint32_t>(rowBase + 3) * K_SIZE_128;
        rowOffset4 = rowOffset0 + CHUNK_SIZE_64;
        rowOffset5 = rowOffset1 + CHUNK_SIZE_64;
        rowOffset6 = rowOffset2 + CHUNK_SIZE_64;
        rowOffset7 = rowOffset3 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, k + rowOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, k + rowOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, k + rowOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, k + rowOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg4, k + rowOffset4);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg5, k + rowOffset5);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg6, k + rowOffset6);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg7, k + rowOffset7);
        Cast<float, DT, B16_TO_FP32>(resultReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg1, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg2, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg3, dataReg3, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg4, dataReg4, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg5, dataReg5, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg6, dataReg6, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg7, dataReg7, floatMask);
        Mul(resultReg0, resultReg0, betaRowReg0, floatMask);
        Mul(resultReg1, resultReg1, betaRowReg1, floatMask);
        Mul(resultReg2, resultReg2, betaRowReg2, floatMask);
        Mul(resultReg3, resultReg3, betaRowReg3, floatMask);
        Mul(resultReg4, resultReg4, betaRowReg0, floatMask);
        Mul(resultReg5, resultReg5, betaRowReg1, floatMask);
        Mul(resultReg6, resultReg6, betaRowReg2, floatMask);
        Mul(resultReg7, resultReg7, betaRowReg3, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, resultReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, resultReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg3, resultReg3, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg4, resultReg4, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg5, resultReg5, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg6, resultReg6, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg7, resultReg7, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset1, dataReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset2, dataReg2, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset3, dataReg3, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset4, dataReg4, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset5, dataReg5, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset6, dataReg6, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset7, dataReg7, floatMask);
    }
    for (vectorTailLoop = 0; vectorTailLoop < vectorTailLoopCount; ++vectorTailLoop) {
        rowBase = vectorTailRowBase + vectorTailLoop;
        Duplicate(indexReg0, static_cast<uint32_t>(rowBase), floatMask);
        Gather(betaRowReg0, betaReg, indexReg0);
        rowOffset0 = static_cast<uint32_t>(rowBase) * K_SIZE_128;
        rowOffset1 = rowOffset0 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, k + rowOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, k + rowOffset1);
        Cast<float, DT, B16_TO_FP32>(resultReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg1, dataReg1, floatMask);
        Mul(resultReg0, resultReg0, betaRowReg0, floatMask);
        Mul(resultReg1, resultReg1, betaRowReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, resultReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(kb + rowOffset1, dataReg1, floatMask);
    }
}

// Stage 4 每次 VF 只处理一个 HV，并在一次调用中完成完整 MxM 矩阵。
// dA_u 已由 Stage 2 变为 strict-lower；dA_w0 仅在 strict-lower 区参与
// 相减。结果原位覆盖 dA_w0，矩阵有效区的对角线和上三角明确写为 0。
template <typename DT>
__simd_vf__ inline void Stage4VF(
    __ubuf__ DT *dA0, __ubuf__ DT *dAuLower, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg lowerMask0;
    MaskReg lowerMask1;
    MaskReg lowerMask2;
    MaskReg lowerMask3;
    RegTensor<DT> dAuDataReg0;
    RegTensor<DT> dAuDataReg1;
    RegTensor<DT> dAuDataReg2;
    RegTensor<DT> dAuDataReg3;
    RegTensor<DT> dAwDataReg0;
    RegTensor<DT> dAwDataReg1;
    RegTensor<DT> dAwDataReg2;
    RegTensor<DT> dAwDataReg3;
    RegTensor<float> dAuReg0;
    RegTensor<float> dAuReg1;
    RegTensor<float> dAuReg2;
    RegTensor<float> dAuReg3;
    RegTensor<float> dAwReg0;
    RegTensor<float> dAwReg1;
    RegTensor<float> dAwReg2;
    RegTensor<float> dAwReg3;
    RegTensor<float> resultReg0;
    RegTensor<float> resultReg1;
    RegTensor<float> resultReg2;
    RegTensor<float> resultReg3;
    RegTensor<float> zeroReg;
    uint16_t rowLoop = 0;
    uint16_t rowLoopCount = validRows / 4;
    uint16_t tailLoop = 0;
    uint16_t tailLoopCount = validRows % 4;
    uint16_t tailRowBase = rowLoopCount * 4;
    uint16_t rowBase = 0;
    uint32_t rowOffset0 = 0;
    uint32_t rowOffset1 = 0;
    uint32_t rowOffset2 = 0;
    uint32_t rowOffset3 = 0;
    uint32_t lowerCount0 = 0;
    uint32_t lowerCount1 = 0;
    uint32_t lowerCount2 = 0;
    uint32_t lowerCount3 = 0;

    Duplicate(zeroReg, 0.0f, floatMask);
    for (rowLoop = 0; rowLoop < rowLoopCount; ++rowLoop) {
        rowBase = rowLoop * 4;
        rowOffset0 = static_cast<uint32_t>(rowBase) * CHUNK_SIZE_64;
        rowOffset1 = static_cast<uint32_t>(rowBase + 1) * CHUNK_SIZE_64;
        rowOffset2 = static_cast<uint32_t>(rowBase + 2) * CHUNK_SIZE_64;
        rowOffset3 = static_cast<uint32_t>(rowBase + 3) * CHUNK_SIZE_64;
        lowerCount0 = static_cast<uint32_t>(rowBase);
        lowerCount1 = static_cast<uint32_t>(rowBase + 1);
        lowerCount2 = static_cast<uint32_t>(rowBase + 2);
        lowerCount3 = static_cast<uint32_t>(rowBase + 3);
        lowerMask0 = UpdateMask<float>(lowerCount0);
        lowerMask1 = UpdateMask<float>(lowerCount1);
        lowerMask2 = UpdateMask<float>(lowerCount2);
        lowerMask3 = UpdateMask<float>(lowerCount3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAuDataReg0, dAuLower + rowOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAuDataReg1, dAuLower + rowOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAuDataReg2, dAuLower + rowOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAuDataReg3, dAuLower + rowOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAwDataReg0, dA0 + rowOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAwDataReg1, dA0 + rowOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAwDataReg2, dA0 + rowOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAwDataReg3, dA0 + rowOffset3);
        Cast<float, DT, B16_TO_FP32>(dAuReg0, dAuDataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAuReg1, dAuDataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAuReg2, dAuDataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAuReg3, dAuDataReg3, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAwReg0, dAwDataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAwReg1, dAwDataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAwReg2, dAwDataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAwReg3, dAwDataReg3, floatMask);
        Sub(resultReg0, dAuReg0, dAwReg0, lowerMask0);
        Sub(resultReg1, dAuReg1, dAwReg1, lowerMask1);
        Sub(resultReg2, dAuReg2, dAwReg2, lowerMask2);
        Sub(resultReg3, dAuReg3, dAwReg3, lowerMask3);
        Select(resultReg0, resultReg0, zeroReg, lowerMask0);
        Select(resultReg1, resultReg1, zeroReg, lowerMask1);
        Select(resultReg2, resultReg2, zeroReg, lowerMask2);
        Select(resultReg3, resultReg3, zeroReg, lowerMask3);
        Cast<DT, float, FP32_TO_B16_PACK>(dAwDataReg0, resultReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dAwDataReg1, resultReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dAwDataReg2, resultReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dAwDataReg3, resultReg3, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dA0 + rowOffset0, dAwDataReg0, validMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dA0 + rowOffset1, dAwDataReg1, validMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dA0 + rowOffset2, dAwDataReg2, validMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dA0 + rowOffset3, dAwDataReg3, validMask);
    }
    for (tailLoop = 0; tailLoop < tailLoopCount; ++tailLoop) {
        rowBase = tailRowBase + tailLoop;
        rowOffset0 = static_cast<uint32_t>(rowBase) * CHUNK_SIZE_64;
        lowerCount0 = static_cast<uint32_t>(rowBase);
        lowerMask0 = UpdateMask<float>(lowerCount0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAuDataReg0, dAuLower + rowOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dAwDataReg0, dA0 + rowOffset0);
        Cast<float, DT, B16_TO_FP32>(dAuReg0, dAuDataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(dAwReg0, dAwDataReg0, floatMask);
        Sub(resultReg0, dAuReg0, dAwReg0, lowerMask0);
        Select(resultReg0, resultReg0, zeroReg, lowerMask0);
        Cast<DT, float, FP32_TO_B16_PACK>(dAwDataReg0, resultReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dA0 + rowOffset0, dAwDataReg0, validMask);
    }
}

template <typename DT>
__simd_vf__ inline void Stage6VF(
    __ubuf__ DT *dvb, __ubuf__ DT *v, __ubuf__ float *beta,
    __ubuf__ float *dbVPartial, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t scalarCount = 1;
    MaskReg scalarMask = UpdateMask<float>(scalarCount);
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<float> dvbReg0;
    RegTensor<float> dvbReg1;
    RegTensor<float> vReg0;
    RegTensor<float> vReg1;
    RegTensor<float> resultReg0;
    RegTensor<float> resultReg1;
    RegTensor<float> sumReg0;
    RegTensor<float> sumReg1;
    RegTensor<float> betaReg;
    RegTensor<float> betaRowReg;
    RegTensor<uint32_t> indexReg;
    uint16_t row = 0;
    uint32_t vectorOffset0 = 0;
    uint32_t vectorOffset1 = 0;

    LoadAlign(betaReg, beta);
    for (row = 0; row < validRows; ++row) {
        Duplicate(indexReg, static_cast<uint32_t>(row), floatMask);
        Gather(betaRowReg, betaReg, indexReg);
        vectorOffset0 = static_cast<uint32_t>(row) * K_SIZE_128;
        vectorOffset1 = vectorOffset0 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dvb + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, v + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dvb + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, v + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(dvbReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(vReg0, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(dvbReg1, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(vReg1, dataReg3, floatMask);
        Mul(resultReg0, dvbReg0, vReg0, floatMask);
        Mul(resultReg1, dvbReg1, vReg1, floatMask);
        ReduceSum(sumReg0, resultReg0, floatMask);
        ReduceSum(sumReg1, resultReg1, floatMask);
        Mul(dvbReg0, dvbReg0, betaRowReg, floatMask);
        Mul(dvbReg1, dvbReg1, betaRowReg, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, dvbReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, dvbReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dvb + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dvb + vectorOffset1, dataReg2, floatMask);
        // ReduceSum 只保证首 lane 有效，后续只能用单 lane mask 消费。
        Add(sumReg1, sumReg0, sumReg1, scalarMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dbVPartial + row, sumReg1, scalarMask);
    }
}

// Stage 8 每次 VF 处理一个 HV。矩阵和标量公式在一次调用内完成；dA 原位
// 覆盖 dA2，dbV 原位覆盖 dbVPartial，dgPrepare 使用独立 FP32 resident。
template <typename DT>
__simd_vf__ inline void Stage8VF(
    __ubuf__ DT *dA2, __ubuf__ DT *gateDA, __ubuf__ DT *a2,
    __ubuf__ DT *dkbg0, __ubuf__ DT *k, __ubuf__ float *beta,
    __ubuf__ float *gExp, __ubuf__ float *bg, __ubuf__ float *dbV,
    __ubuf__ float *dgPrepare, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg lowerMask;
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<float> dAReg;
    RegTensor<float> gateReg;
    RegTensor<float> a2Reg;
    RegTensor<float> dkbgReg;
    RegTensor<float> dkbgReg1;
    RegTensor<float> kReg;
    RegTensor<float> kReg1;
    RegTensor<float> resultReg;
    RegTensor<float> resultReg1;
    RegTensor<float> sumReg0;
    RegTensor<float> sumReg1;
    RegTensor<float> columnSumReg;
    RegTensor<float> betaReg;
    RegTensor<float> gExpReg;
    RegTensor<float> bgReg;
    RegTensor<float> scalarReg;
    RegTensor<float> scalarReg1;
    RegTensor<float> rowScalarReg;
    RegTensor<float> zeroReg;
    RegTensor<uint32_t> indexReg;
    uint16_t row = 0;
    uint32_t rowOffset = 0;
    uint32_t vectorOffset0 = 0;
    uint32_t vectorOffset1 = 0;
    uint32_t lowerCount = 0;

    LoadAlign(betaReg, beta);
    LoadAlign(gExpReg, gExp);
    LoadAlign(bgReg, bg);
    Duplicate(columnSumReg, 0.0f, floatMask);
    Duplicate(zeroReg, 0.0f, floatMask);
    for (row = 0; row < validRows; ++row) {
        rowOffset = static_cast<uint32_t>(row) * CHUNK_SIZE_64;
        lowerCount = static_cast<uint32_t>(row);
        lowerMask = UpdateMask<float>(lowerCount);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dA2 + rowOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, gateDA + rowOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, a2 + rowOffset);
        Cast<float, DT, B16_TO_FP32>(dAReg, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(gateReg, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(a2Reg, dataReg2, floatMask);
        Mul(dAReg, dAReg, gateReg, lowerMask);
        Muls(dAReg, dAReg, -1.0f, lowerMask);
        Select(dAReg, dAReg, zeroReg, lowerMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, dAReg, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dA2 + rowOffset, dataReg0, validMask);

        Duplicate(indexReg, static_cast<uint32_t>(row), floatMask);
        Gather(rowScalarReg, betaReg, indexReg);
        Mul(resultReg, a2Reg, rowScalarReg, validMask);
        Mul(resultReg, resultReg, dAReg, validMask);
        ReduceSum(sumReg0, resultReg, validMask);
        Add(columnSumReg, columnSumReg, resultReg, validMask);

        vectorOffset0 = static_cast<uint32_t>(row) * K_SIZE_128;
        vectorOffset1 = vectorOffset0 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dkbg0 + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, k + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dkbg0 + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, k + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(dkbgReg, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(kReg, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(dkbgReg1, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(kReg1, dataReg3, floatMask);
        Mul(resultReg, dkbgReg, kReg, floatMask);
        Mul(resultReg1, dkbgReg1, kReg1, floatMask);
        ReduceSum(scalarReg, resultReg, floatMask);
        ReduceSum(scalarReg1, resultReg1, floatMask);
        Add(sumReg1, scalarReg, scalarReg1, floatMask);
        Gather(rowScalarReg, bgReg, indexReg);
        Mul(rowScalarReg, rowScalarReg, sumReg1, floatMask);
        Muls(rowScalarReg, rowScalarReg, -1.0f, floatMask);
        Add(sumReg0, sumReg0, rowScalarReg, floatMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dgPrepare + row, sumReg0, floatMask);

        Gather(rowScalarReg, gExpReg, indexReg);
        Mul(rowScalarReg, rowScalarReg, sumReg1, floatMask);
        Muls(rowScalarReg, rowScalarReg, -1.0f, floatMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, dbV + row);
        Add(rowScalarReg, rowScalarReg, scalarReg, floatMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dbV + row, rowScalarReg, floatMask);
    }
    // dgPrepare 的逐行基础值已写入 UB；本屏障只保护本 VF 内的 store->load。
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    LoadAlign(resultReg, dgPrepare);
    Sub(resultReg, resultReg, columnSumReg, validMask);
    StoreAlign(dgPrepare, resultReg, validMask);
}

// Stage 10 每次 VF 处理一个 HV。h/dh 已完整搬入 UB，VF 在一次调用内
// 归约 128x128 个乘积，并用 g_last 生成该 chunk 的 state_term。
template <typename DT>
__simd_vf__ inline void Stage10VF(
    __ubuf__ DT *h, __ubuf__ DT *dh, __ubuf__ float *stateTerm)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    uint32_t scalarCount = 1;
    MaskReg scalarMask = UpdateMask<float>(scalarCount);
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<DT> dataReg4;
    RegTensor<DT> dataReg5;
    RegTensor<DT> dataReg6;
    RegTensor<DT> dataReg7;
    RegTensor<float> hReg;
    RegTensor<float> hReg1;
    RegTensor<float> hReg2;
    RegTensor<float> hReg3;
    RegTensor<float> dhReg;
    RegTensor<float> dhReg1;
    RegTensor<float> dhReg2;
    RegTensor<float> dhReg3;
    RegTensor<float> productReg;
    RegTensor<float> productReg1;
    RegTensor<float> productReg2;
    RegTensor<float> productReg3;
    RegTensor<float> totalSumReg;
    RegTensor<float> reducedSumReg;
    RegTensor<float> gLastReg;
    uint16_t tile = 0;
    constexpr uint16_t tileLoopCount = K_SIZE_128 * V_SIZE_128 / CHUNK_SIZE_64 / 4;
    uint32_t vectorOffset0 = 0;
    uint32_t vectorOffset1 = 0;
    uint32_t vectorOffset2 = 0;
    uint32_t vectorOffset3 = 0;

    Duplicate(totalSumReg, 0.0f, floatMask);
    for (tile = 0; tile < tileLoopCount; ++tile) {
        vectorOffset0 = static_cast<uint32_t>(tile) * CHUNK_SIZE_64 * 4;
        vectorOffset1 = vectorOffset0 + CHUNK_SIZE_64;
        vectorOffset2 = vectorOffset1 + CHUNK_SIZE_64;
        vectorOffset3 = vectorOffset2 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, h + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dh + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, h + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, dh + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg4, h + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg5, dh + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg6, h + vectorOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg7, dh + vectorOffset3);
        Cast<float, DT, B16_TO_FP32>(hReg, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(dhReg, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(hReg1, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(dhReg1, dataReg3, floatMask);
        Cast<float, DT, B16_TO_FP32>(hReg2, dataReg4, floatMask);
        Cast<float, DT, B16_TO_FP32>(dhReg2, dataReg5, floatMask);
        Cast<float, DT, B16_TO_FP32>(hReg3, dataReg6, floatMask);
        Cast<float, DT, B16_TO_FP32>(dhReg3, dataReg7, floatMask);
        Mul(productReg, hReg, dhReg, floatMask);
        Mul(productReg1, hReg1, dhReg1, floatMask);
        Mul(productReg2, hReg2, dhReg2, floatMask);
        Mul(productReg3, hReg3, dhReg3, floatMask);
        // 四个 tile 的加载与乘法链独立，累加仍保持原 tile 顺序。
        Add(totalSumReg, totalSumReg, productReg, floatMask);
        Add(totalSumReg, totalSumReg, productReg1, floatMask);
        Add(totalSumReg, totalSumReg, productReg2, floatMask);
        Add(totalSumReg, totalSumReg, productReg3, floatMask);
    }
    ReduceSum(reducedSumReg, totalSumReg, floatMask);
    LoadAlign(gLastReg, stateTerm);
    // ReduceSum 只保证首 lane 有效，禁止用全 mask 读取其余未定义 lane。
    Mul(reducedSumReg, reducedSumReg, gLastReg, scalarMask);
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(stateTerm, reducedSumReg, scalarMask);
}

// Stage 12 每次 VF 处理一个 HV。所有 GM 输入在调用前完整进入 UB；本 VF
// 同时生成 dk_prepare、dbeta、dq_base 和 ds，不拆分 pass。
template <typename DT, typename BT, bool USE_BETA_SIGMOID>
__simd_vf__ inline void Stage12VF(
    __ubuf__ DT *dkb, __ubuf__ DT *dkbT, __ubuf__ DT *ds,
    __ubuf__ DT *doG, __ubuf__ DT *vDecay, __ubuf__ DT *dkPrepare, __ubuf__ DT *k,
    __ubuf__ DT *gateDA, __ubuf__ float *beta, __ubuf__ float *gExp,
    __ubuf__ float *decay, __ubuf__ float *bg, __ubuf__ float *dbV,
    __ubuf__ BT *betaRaw,
    __ubuf__ float *betaRawFp32, __ubuf__ BT *dbetaOutput,
    float scale, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    uint32_t validCount = validRows;
    MaskReg validMask = UpdateMask<float>(validCount);
    MaskReg lowerMask;
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<DT> dataReg4;
    RegTensor<DT> dataReg5;
    RegTensor<DT> dataReg6;
    RegTensor<DT> dataReg7;
    RegTensor<BT> betaRawDataReg;
    RegTensor<float> dkbReg;
    RegTensor<float> dkbTReg;
    RegTensor<float> dkbgReg;
    RegTensor<float> kReg;
    RegTensor<float> resultReg;
    RegTensor<float> scalarReg;
    RegTensor<float> rowScalarReg;
    RegTensor<float> bgRowReg;
    RegTensor<float> betaRowReg;
    RegTensor<float> betaRawReg;
    RegTensor<float> gExpFullReg;
    RegTensor<float> decayFullReg;
    RegTensor<float> oneReg;
    RegTensor<uint32_t> indexReg;
    RegTensor<float> gateReg;
    RegTensor<float> dsReg;
    RegTensor<float> dqReg;
    RegTensor<float> dqReg1;
    RegTensor<float> resultReg1;
    RegTensor<float> zeroReg;
    uint16_t row = 0;
    uint32_t vectorOffset = 0;
    uint32_t vectorOffset1 = 0;
    uint32_t matrixOffset = 0;

    if constexpr (USE_BETA_SIGMOID) {
        if constexpr (std::is_same<BT, float>::value) {
            // FP32 输入已经位于最终驻留地址，无需转换。
        } else {
            DataCopy<BT, LoadDist::DIST_UNPACK_B16>(betaRawDataReg, betaRaw);
            Cast<float, BT, B16_TO_FP32>(betaRawReg, betaRawDataReg, floatMask);
            // 每个 slot 为 beta_raw 预留 256 B，原位扩展为 FP32[64]；
            // 后续逐行广播，避免 Gather 在最后一个 lane 上的边界行为。
            StoreAlign(betaRawFp32, betaRawReg, floatMask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        Duplicate(oneReg, 1.0f, floatMask);
    }
    for (row = 0; row < validRows; ++row) {
        Duplicate(indexReg, static_cast<uint32_t>(row), floatMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(bgRowReg, bg + row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(betaRowReg, beta + row);
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, dbV + row);
        vectorOffset = static_cast<uint32_t>(row) * K_SIZE_128;
        vectorOffset1 = vectorOffset + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dkb + vectorOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dkbT + vectorOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dkPrepare + vectorOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, k + vectorOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg4, dkb + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg5, dkbT + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg6, dkPrepare + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg7, k + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(dkbReg, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(dkbTReg, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(dkbgReg, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(kReg, dataReg3, floatMask);
        Cast<float, DT, B16_TO_FP32>(dqReg, dataReg4, floatMask);
        Cast<float, DT, B16_TO_FP32>(dqReg1, dataReg5, floatMask);
        Cast<float, DT, B16_TO_FP32>(gateReg, dataReg6, floatMask);
        Cast<float, DT, B16_TO_FP32>(dsReg, dataReg7, floatMask);
        Mul(resultReg, dkbReg, kReg, floatMask);
        Mul(resultReg1, dqReg, dsReg, floatMask);
        ReduceSum(rowScalarReg, resultReg, floatMask);
        ReduceSum(zeroReg, resultReg1, floatMask);
        // 两列归约按原 columnTile=0、1 的顺序累加，保持 db_prepare 位级一致。
        Add(scalarReg, scalarReg, rowScalarReg, scalarMask);
        Add(scalarReg, scalarReg, zeroReg, scalarMask);
        Mul(resultReg, dkbgReg, bgRowReg, floatMask);
        Mul(resultReg1, gateReg, bgRowReg, floatMask);
        Muls(resultReg, resultReg, -1.0f, floatMask);
        Muls(resultReg1, resultReg1, -1.0f, floatMask);
        Mul(dkbReg, dkbReg, betaRowReg, floatMask);
        Mul(dqReg, dqReg, betaRowReg, floatMask);
        Add(resultReg, resultReg, dkbReg, floatMask);
        Add(resultReg1, resultReg1, dqReg, floatMask);
        Add(resultReg, resultReg, dkbTReg, floatMask);
        Add(resultReg1, resultReg1, dqReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, resultReg, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg4, resultReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkPrepare + vectorOffset, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkPrepare + vectorOffset1, dataReg4, floatMask);
        // dbV 的跨 Stage 生命周期到此结束，原位覆盖为 db_prepare。
        // Reduce/Gather 结果只有首 lane 具备当前行语义，因此仅写一个 FP32。
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dbV + row, scalarReg, scalarMask);
    }
    // db_prepare 与 beta_raw 均已形成连续 FP32[64]，一次向量计算完成
    // sigmoid backward，避免逐行广播，也完整覆盖第 64 个 lane。
    if constexpr (USE_BETA_SIGMOID) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(scalarReg, dbV);
        LoadAlign(betaRawReg, betaRawFp32);
        Muls(betaRawReg, betaRawReg, -1.0f, floatMask);
        Exp(betaRawReg, betaRawReg, floatMask);
        Add(betaRawReg, betaRawReg, oneReg, floatMask);
        Div(betaRawReg, oneReg, betaRawReg, floatMask);
        Sub(resultReg, oneReg, betaRawReg, floatMask);
        Mul(betaRawReg, betaRawReg, resultReg, floatMask);
        Mul(scalarReg, scalarReg, betaRawReg, floatMask);
        StoreAlign(dbV, scalarReg, floatMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // dbeta 始终以连续 FP32[64] 驻留在 dbV。BF16 ABI 仅在最终写回前
    // 生成一份临时打包结果；FP32 ABI 由 MTE3 直接消费 dbV。
    if constexpr (!std::is_same<BT, float>::value) {
        LoadAlign(scalarReg, dbV);
        Cast<BT, float, FP32_TO_B16_PACK>(betaRawDataReg, scalarReg, floatMask);
        StoreAlign<BT, StoreDist::DIST_PACK_B32>(dbetaOutput, betaRawDataReg, floatMask);
    }

    // 仍在同一次 VF 调用内，但等 K/dbeta 公式全部结束后再执行
    // do_g/v_decay/DS，缩短两组 RegTensor 的重叠生命周期。
    // 缩短两组 RegTensor 的重叠生命周期，避免寄存器 spill。
    // DQ 完成后再处理 DS，使两组大型向量寄存器不同时存活。
    LoadAlign(gExpFullReg, gExp);
    LoadAlign(decayFullReg, decay);
    for (row = 0; row < validRows; ++row) {
        Duplicate(indexReg, static_cast<uint32_t>(row), floatMask);
        Gather(rowScalarReg, gExpFullReg, indexReg);
        Gather(scalarReg, decayFullReg, indexReg);
        vectorOffset = static_cast<uint32_t>(row) * K_SIZE_128;
        vectorOffset1 = vectorOffset + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, doG + vectorOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, vDecay + vectorOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg4, doG + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg5, vDecay + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(dqReg, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(dqReg1, dataReg4, floatMask);
        Cast<float, DT, B16_TO_FP32>(resultReg1, dataReg5, floatMask);
        Mul(dqReg, dqReg, rowScalarReg, floatMask);
        Mul(dqReg1, dqReg1, rowScalarReg, floatMask);
        Muls(dqReg, dqReg, scale, floatMask);
        Muls(dqReg1, dqReg1, scale, floatMask);
        Mul(resultReg, resultReg, scalarReg, floatMask);
        Mul(resultReg1, resultReg1, scalarReg, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, dqReg, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, resultReg, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg4, dqReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg5, resultReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(doG + vectorOffset, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vDecay + vectorOffset, dataReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(doG + vectorOffset1, dataReg4, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(vDecay + vectorOffset1, dataReg5, floatMask);
    }
    // zeroReg 只供 DS 下三角掩码使用，在消费前初始化，避免让该寄存器
    // 跨越前面整段 dk/dbeta/do_g/v_decay 计算长期存活。
    Duplicate(zeroReg, 0.0f, floatMask);
    for (row = 0; row < validRows; ++row) {
        matrixOffset = static_cast<uint32_t>(row) * CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, ds + matrixOffset);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, gateDA + matrixOffset);
        Cast<float, DT, B16_TO_FP32>(dsReg, dataReg0, validMask);
        Cast<float, DT, B16_TO_FP32>(gateReg, dataReg1, validMask);
        Mul(dsReg, dsReg, gateReg, validMask);
        Muls(dsReg, dsReg, scale, validMask);
        validCount = static_cast<uint32_t>(row + 1);
        lowerMask = UpdateMask<float>(validCount);
        Select(dsReg, dsReg, zeroReg, lowerMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, dsReg, floatMask);
        validCount = validRows;
        validMask = UpdateMask<float>(validCount);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(ds + matrixOffset, dataReg0, validMask);
    }
}

// Stage 14 每次 VF 处理一个完整 HV。dq_hv 保持不变并由 MTE3 写 W14；
// dk_base 原位生成 dk_raw_hv；dgScratch 始终保存 FP32 dg，BF16 ABI
// 只在最终写回前原位生成临时打包结果。
template <typename DT, typename GT>
__simd_vf__ inline void Stage14VF(
    __ubuf__ DT *dkBase, __ubuf__ DT *dqHv, __ubuf__ DT *dkIntra,
    __ubuf__ DT *dkPrepare, __ubuf__ DT *q, __ubuf__ DT *k,
    __ubuf__ float *dgPrepare, __ubuf__ float *stateTerm,
    __ubuf__ float *dgScratch, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<float> dqReg;
    RegTensor<float> qReg;
    RegTensor<float> dkBaseReg;
    RegTensor<float> dkIntraReg;
    RegTensor<float> dkPrepareReg;
    RegTensor<float> kReg;
    RegTensor<float> productReg;
    RegTensor<float> rowDqSumReg;
    RegTensor<float> rowDkBaseSumReg;
    RegTensor<float> rowDkIntraSumReg;
    RegTensor<float> reducedReg;
    RegTensor<float> totalDkBaseReg;
    RegTensor<float> scalarReg;
    RegTensor<float> suffixReg;
    uint16_t row = 0;
    uint16_t columnTile = 0;
    uint32_t vectorOffset = 0;

    Duplicate(totalDkBaseReg, 0.0f, floatMask);
    for (row = 0; row < validRows; ++row) {
        Duplicate(rowDqSumReg, 0.0f, floatMask);
        Duplicate(rowDkBaseSumReg, 0.0f, floatMask);
        Duplicate(rowDkIntraSumReg, 0.0f, floatMask);
        for (columnTile = 0; columnTile < 2; ++columnTile) {
            vectorOffset = static_cast<uint32_t>(row) * K_SIZE_128 +
                static_cast<uint32_t>(columnTile) * CHUNK_SIZE_64;
            DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dqHv + vectorOffset);
            DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, q + vectorOffset);
            DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dkBase + vectorOffset);
            DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, k + vectorOffset);
            Cast<float, DT, B16_TO_FP32>(dqReg, dataReg0, floatMask);
            Cast<float, DT, B16_TO_FP32>(qReg, dataReg1, floatMask);
            Cast<float, DT, B16_TO_FP32>(dkBaseReg, dataReg2, floatMask);
            Cast<float, DT, B16_TO_FP32>(kReg, dataReg3, floatMask);
            Mul(productReg, dqReg, qReg, floatMask);
            ReduceSum(reducedReg, productReg, floatMask);
            Add(rowDqSumReg, rowDqSumReg, reducedReg, scalarMask);
            Mul(productReg, dkBaseReg, kReg, floatMask);
            ReduceSum(reducedReg, productReg, floatMask);
            Add(rowDkBaseSumReg, rowDkBaseSumReg, reducedReg, scalarMask);

            DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dkIntra + vectorOffset);
            DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dkPrepare + vectorOffset);
            Cast<float, DT, B16_TO_FP32>(dkIntraReg, dataReg0, floatMask);
            Cast<float, DT, B16_TO_FP32>(dkPrepareReg, dataReg1, floatMask);
            Mul(productReg, dkIntraReg, kReg, floatMask);
            ReduceSum(reducedReg, productReg, floatMask);
            Add(rowDkIntraSumReg, rowDkIntraSumReg, reducedReg, scalarMask);
            Add(dkBaseReg, dkBaseReg, dkIntraReg, floatMask);
            Add(dkBaseReg, dkBaseReg, dkPrepareReg, floatMask);
            Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, dkBaseReg, floatMask);
            StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkBase + vectorOffset, dataReg2, floatMask);
        }
        Add(totalDkBaseReg, totalDkBaseReg, rowDkBaseSumReg, scalarMask);
        Sub(rowDqSumReg, rowDqSumReg, rowDkBaseSumReg, scalarMask);
        Sub(rowDqSumReg, rowDqSumReg, rowDkIntraSumReg, scalarMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dgScratch + row, rowDqSumReg, scalarMask);
    }

    // 前面逐行结果已经完整写入，尾元素补偿与 reverse suffix sum 才能读取。
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(
        scalarReg, dgScratch + static_cast<uint32_t>(validRows - 1));
    DataCopy<float, LoadDist::DIST_BRC_B32>(suffixReg, stateTerm);
    Add(scalarReg, scalarReg, totalDkBaseReg, scalarMask);
    Add(scalarReg, scalarReg, suffixReg, scalarMask);
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
        dgScratch + static_cast<uint32_t>(validRows - 1), scalarReg, scalarMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    Duplicate(suffixReg, 0.0f, floatMask);
    for (row = validRows; row > 0; --row) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(scalarReg, dgScratch + row - 1);
        DataCopy<float, LoadDist::DIST_BRC_B32>(reducedReg, dgPrepare + row - 1);
        Add(scalarReg, scalarReg, reducedReg, scalarMask);
        Add(suffixReg, suffixReg, scalarReg, scalarMask);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
            dgScratch + row - 1, suffixReg, scalarMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    if constexpr (!std::is_same<GT, float>::value) {
        LoadAlign(suffixReg, dgScratch);
        Cast<GT, float, FP32_TO_B16_PACK>(dataReg0, suffixReg, floatMask);
        StoreAlign<GT, StoreDist::DIST_PACK_B32>(
            reinterpret_cast<__ubuf__ GT *>(dgScratch), dataReg0, floatMask);
    }
}

// Stage 15 将一个新增 HV partial 原位累加到当前 HK accumulator。
template <typename DT>
__simd_vf__ inline void Stage15AccumulateVF(
    __ubuf__ DT *dqAcc, __ubuf__ DT *dkAcc,
    __ubuf__ DT *dqInput, __ubuf__ DT *dkInput, uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<DT> dataReg2;
    RegTensor<DT> dataReg3;
    RegTensor<DT> dataReg4;
    RegTensor<DT> dataReg5;
    RegTensor<DT> dataReg6;
    RegTensor<DT> dataReg7;
    RegTensor<DT> dataReg8;
    RegTensor<DT> dataReg9;
    RegTensor<DT> dataReg10;
    RegTensor<DT> dataReg11;
    RegTensor<DT> dataReg12;
    RegTensor<DT> dataReg13;
    RegTensor<DT> dataReg14;
    RegTensor<DT> dataReg15;
    RegTensor<float> accReg0;
    RegTensor<float> accReg1;
    RegTensor<float> accReg2;
    RegTensor<float> accReg3;
    RegTensor<float> accReg4;
    RegTensor<float> accReg5;
    RegTensor<float> accReg6;
    RegTensor<float> accReg7;
    RegTensor<float> inputReg0;
    RegTensor<float> inputReg1;
    RegTensor<float> inputReg2;
    RegTensor<float> inputReg3;
    RegTensor<float> inputReg4;
    RegTensor<float> inputReg5;
    RegTensor<float> inputReg6;
    RegTensor<float> inputReg7;
    uint16_t rowLoop = 0;
    uint16_t rowLoopCount = validRows / 4;
    uint16_t tailLoop = 0;
    uint16_t tailLoopCount = validRows % 4;
    uint16_t tailRowBase = rowLoopCount * 4;
    uint16_t rowBase = 0;
    uint32_t vectorOffset0 = 0;
    uint32_t vectorOffset1 = 0;
    uint32_t vectorOffset2 = 0;
    uint32_t vectorOffset3 = 0;
    uint32_t vectorOffset4 = 0;
    uint32_t vectorOffset5 = 0;
    uint32_t vectorOffset6 = 0;
    uint32_t vectorOffset7 = 0;

    // 四行、两列全部展开，恰好使用 32 个 RegTensor。
    for (rowLoop = 0; rowLoop < rowLoopCount; ++rowLoop) {
        rowBase = rowLoop * 4;
        vectorOffset0 = static_cast<uint32_t>(rowBase) * K_SIZE_128;
        vectorOffset1 = vectorOffset0 + CHUNK_SIZE_64;
        vectorOffset2 = static_cast<uint32_t>(rowBase + 1) * K_SIZE_128;
        vectorOffset3 = vectorOffset2 + CHUNK_SIZE_64;
        vectorOffset4 = static_cast<uint32_t>(rowBase + 2) * K_SIZE_128;
        vectorOffset5 = vectorOffset4 + CHUNK_SIZE_64;
        vectorOffset6 = static_cast<uint32_t>(rowBase + 3) * K_SIZE_128;
        vectorOffset7 = vectorOffset6 + CHUNK_SIZE_64;

        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dqAcc + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dqInput + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dqAcc + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, dqInput + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg4, dqAcc + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg5, dqInput + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg6, dqAcc + vectorOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg7, dqInput + vectorOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg8, dqAcc + vectorOffset4);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg9, dqInput + vectorOffset4);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg10, dqAcc + vectorOffset5);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg11, dqInput + vectorOffset5);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg12, dqAcc + vectorOffset6);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg13, dqInput + vectorOffset6);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg14, dqAcc + vectorOffset7);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg15, dqInput + vectorOffset7);
        Cast<float, DT, B16_TO_FP32>(accReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg1, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg1, dataReg3, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg2, dataReg4, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg2, dataReg5, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg3, dataReg6, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg3, dataReg7, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg4, dataReg8, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg4, dataReg9, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg5, dataReg10, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg5, dataReg11, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg6, dataReg12, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg6, dataReg13, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg7, dataReg14, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg7, dataReg15, floatMask);
        Add(accReg0, accReg0, inputReg0, floatMask);
        Add(accReg1, accReg1, inputReg1, floatMask);
        Add(accReg2, accReg2, inputReg2, floatMask);
        Add(accReg3, accReg3, inputReg3, floatMask);
        Add(accReg4, accReg4, inputReg4, floatMask);
        Add(accReg5, accReg5, inputReg5, floatMask);
        Add(accReg6, accReg6, inputReg6, floatMask);
        Add(accReg7, accReg7, inputReg7, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, accReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, accReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg4, accReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg6, accReg3, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg8, accReg4, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg10, accReg5, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg12, accReg6, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg14, accReg7, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset1, dataReg2, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset2, dataReg4, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset3, dataReg6, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset4, dataReg8, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset5, dataReg10, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset6, dataReg12, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset7, dataReg14, floatMask);

        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dkAcc + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dkInput + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dkAcc + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, dkInput + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg4, dkAcc + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg5, dkInput + vectorOffset2);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg6, dkAcc + vectorOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg7, dkInput + vectorOffset3);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg8, dkAcc + vectorOffset4);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg9, dkInput + vectorOffset4);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg10, dkAcc + vectorOffset5);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg11, dkInput + vectorOffset5);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg12, dkAcc + vectorOffset6);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg13, dkInput + vectorOffset6);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg14, dkAcc + vectorOffset7);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg15, dkInput + vectorOffset7);
        Cast<float, DT, B16_TO_FP32>(accReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg1, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg1, dataReg3, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg2, dataReg4, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg2, dataReg5, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg3, dataReg6, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg3, dataReg7, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg4, dataReg8, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg4, dataReg9, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg5, dataReg10, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg5, dataReg11, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg6, dataReg12, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg6, dataReg13, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg7, dataReg14, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg7, dataReg15, floatMask);
        Add(accReg0, accReg0, inputReg0, floatMask);
        Add(accReg1, accReg1, inputReg1, floatMask);
        Add(accReg2, accReg2, inputReg2, floatMask);
        Add(accReg3, accReg3, inputReg3, floatMask);
        Add(accReg4, accReg4, inputReg4, floatMask);
        Add(accReg5, accReg5, inputReg5, floatMask);
        Add(accReg6, accReg6, inputReg6, floatMask);
        Add(accReg7, accReg7, inputReg7, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, accReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, accReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg4, accReg2, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg6, accReg3, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg8, accReg4, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg10, accReg5, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg12, accReg6, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg14, accReg7, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset1, dataReg2, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset2, dataReg4, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset3, dataReg6, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset4, dataReg8, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset5, dataReg10, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset6, dataReg12, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset7, dataReg14, floatMask);
    }

    for (tailLoop = 0; tailLoop < tailLoopCount; ++tailLoop) {
        rowBase = tailRowBase + tailLoop;
        vectorOffset0 = static_cast<uint32_t>(rowBase) * K_SIZE_128;
        vectorOffset1 = vectorOffset0 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dqAcc + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dqInput + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dqAcc + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, dqInput + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(accReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg1, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg1, dataReg3, floatMask);
        Add(accReg0, accReg0, inputReg0, floatMask);
        Add(accReg1, accReg1, inputReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, accReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, accReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dqAcc + vectorOffset1, dataReg2, floatMask);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dkAcc + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dkInput + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg2, dkAcc + vectorOffset1);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg3, dkInput + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(accReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg1, floatMask);
        Cast<float, DT, B16_TO_FP32>(accReg1, dataReg2, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg1, dataReg3, floatMask);
        Add(accReg0, accReg0, inputReg0, floatMask);
        Add(accReg1, accReg1, inputReg1, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, accReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg2, accReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dkAcc + vectorOffset1, dataReg2, floatMask);
    }
}

// G>4 uses these temporary views only after all S12 MTE3 sources are released.
__simd_vf__ inline void ZeroGvaScratchVF(__ubuf__ float *scratch)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    Duplicate(zero, 0.0f, mask);
    for (uint16_t i = 0; i < 128; ++i) {
        StoreAlign(scratch + static_cast<uint32_t>(i) * 64, zero, mask);
    }
}

template <typename DT>
__simd_vf__ inline void CastBf16ToFp32VF(
    __ubuf__ float *dst, __ubuf__ DT *src, uint16_t validRows)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<DT> input;
    RegTensor<float> value;
    for (uint16_t i = 0; i < validRows * 2; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * 64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(input, src + offset);
        Cast<float, DT, B16_TO_FP32>(value, input, mask);
        StoreAlign(dst + offset, value, mask);
    }
}

template <typename DT>
__simd_vf__ inline void MergeFp32ToBf16VF(
    __ubuf__ DT *dst, __ubuf__ float *a, __ubuf__ float *b, uint16_t validRows)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> lhs, rhs;
    RegTensor<DT> result;
    for (uint16_t i = 0; i < validRows * 2; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * 64;
        DataCopy(lhs, a + offset);
        DataCopy(rhs, b + offset);
        Add(lhs, lhs, rhs, mask);
        Cast<DT, float, FP32_TO_B16_PACK>(result, lhs, mask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dst + offset, result, mask);
    }
}

// Stage 15 对一个完整 HK accumulator 执行 Q/K norm backward。
template <typename DT>
__simd_vf__ inline void Stage15NormVF(
    __ubuf__ DT *dq, __ubuf__ DT *dk, __ubuf__ DT *q, __ubuf__ DT *k,
    __ubuf__ float *qRstd, __ubuf__ float *kRstd,
    uint16_t validRows)
{
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg scalarMask = CreateMask<float, MaskPattern::VL1>();
    RegTensor<DT> dataReg0;
    RegTensor<DT> dataReg1;
    RegTensor<float> gradReg0;
    RegTensor<float> gradReg1;
    RegTensor<float> inputReg0;
    RegTensor<float> inputReg1;
    RegTensor<float> productReg;
    RegTensor<float> dotReg;
    RegTensor<float> reducedReg;
    RegTensor<float> rstdReg;
    uint16_t row = 0;
    uint32_t vectorOffset0 = 0;
    uint32_t vectorOffset1 = 0;

    for (row = 0; row < validRows; ++row) {
        vectorOffset0 = static_cast<uint32_t>(row) * K_SIZE_128;
        vectorOffset1 = vectorOffset0 + CHUNK_SIZE_64;
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dq + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dq + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(gradReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(gradReg1, dataReg1, floatMask);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, q + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, q + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg1, dataReg1, floatMask);
        Mul(productReg, gradReg0, inputReg0, floatMask);
        Mul(reducedReg, gradReg1, inputReg1, floatMask);
        Add(productReg, productReg, reducedReg, floatMask);
        ReduceSum(dotReg, productReg, floatMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, qRstd + row);
        // ReduceSum 仅保证标量 lane 有效，不能直接把 dotReg 当成广播
        // 向量使用。当前行的 rstd 已进入寄存器，因此复用其 UB 标量槽
        // 暂存 dot，再显式广播到全部 lane，不额外占用 UB。
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(qRstd + row, dotReg, scalarMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(dotReg, qRstd + row);
        Mul(productReg, inputReg0, dotReg, floatMask);
        Sub(gradReg0, gradReg0, productReg, floatMask);
        Mul(productReg, inputReg1, dotReg, floatMask);
        Sub(gradReg1, gradReg1, productReg, floatMask);
        Mul(gradReg0, gradReg0, rstdReg, floatMask);
        Mul(gradReg1, gradReg1, rstdReg, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, gradReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, gradReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dq + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dq + vectorOffset1, dataReg1, floatMask);

        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, dk + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, dk + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(gradReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(gradReg1, dataReg1, floatMask);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg0, k + vectorOffset0);
        DataCopy<DT, LoadDist::DIST_UNPACK_B16>(dataReg1, k + vectorOffset1);
        Cast<float, DT, B16_TO_FP32>(inputReg0, dataReg0, floatMask);
        Cast<float, DT, B16_TO_FP32>(inputReg1, dataReg1, floatMask);
        Mul(productReg, gradReg0, inputReg0, floatMask);
        Mul(reducedReg, gradReg1, inputReg1, floatMask);
        Add(productReg, productReg, reducedReg, floatMask);
        ReduceSum(dotReg, productReg, floatMask);
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, kRstd + row);
        // 与 Q 分支相同，先把归约标量显式广播，再计算投影项。
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(kRstd + row, dotReg, scalarMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        DataCopy<float, LoadDist::DIST_BRC_B32>(dotReg, kRstd + row);
        Mul(productReg, inputReg0, dotReg, floatMask);
        Sub(gradReg0, gradReg0, productReg, floatMask);
        Mul(productReg, inputReg1, dotReg, floatMask);
        Sub(gradReg1, gradReg1, productReg, floatMask);
        Mul(gradReg0, gradReg0, rstdReg, floatMask);
        Mul(gradReg1, gradReg1, rstdReg, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg0, gradReg0, floatMask);
        Cast<DT, float, FP32_TO_B16_PACK>(dataReg1, gradReg1, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dk + vectorOffset0, dataReg0, floatMask);
        StoreAlign<DT, StoreDist::DIST_PACK_B32>(dk + vectorOffset1, dataReg1, floatMask);
    }
}

} // namespace GDN::FinalizeArch92

#endif
