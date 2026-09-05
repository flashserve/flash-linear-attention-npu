#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VECTOR_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VECTOR_H

#include <type_traits>

#include "kernel_operator.h"
#include "chunk_gated_delta_rule_bwd_finalize_common.h"

namespace GDN {

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

// Stage 0 每次 VF 只处理一个 HV。当前 VF 计算时，MTE2 使用另一物理
// slot 搬入本 AIV 承包的下一个 HV，形成双缓冲流水。
template <typename DT, typename GT, typename BT>
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
    // 先供 decay 使用，再原位转换为 exp2(g[M-1])。gLast 按完整
    // FP32[BT] 广播保存，避免后续 VF 特判 scalar。
    Muls(gExpReg, gReg, LN2, validMask);
    Duplicate(indexReg0, static_cast<uint32_t>(validRows - 1), floatMask);
    Gather(gLastReg, gReg, indexReg0);
    Sub(decayReg, gLastReg, gReg, validMask);
    Muls(decayReg, decayReg, LN2, validMask);
    Muls(gLastReg, gLastReg, LN2, floatMask);
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
        Muls(resultReg0, resultReg0, LN2, validMask);
        Muls(resultReg1, resultReg1, LN2, validMask);
        Muls(resultReg2, resultReg2, LN2, validMask);
        Muls(resultReg3, resultReg3, LN2, validMask);
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
        Muls(resultReg0, resultReg0, LN2, validMask);
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
template <typename DT, typename BT, bool USE_BETA_SIGMOID, bool ALLOW_NEG_EIGVAL>
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
        if constexpr (ALLOW_NEG_EIGVAL) {
            Muls(scalarReg, scalarReg, 2.0f, floatMask);
        }
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

template <typename DT, typename GT, typename BT, bool USE_QK_L2NORM, bool USE_BETA_SIGMOID,
          bool ALLOW_NEG_EIGVAL>
class ChunkGatedDeltaRuleBwdFinalizeVector {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR vNew, GM_ADDR dO,
        GM_ADDR g, GM_ADDR beta, GM_ADDR h, GM_ADDR dh,
        GM_ADDR qRstd, GM_ADDR kRstd, GM_ADDR dqOut, GM_ADDR dkOut, GM_ADDR dgOut,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR kbgOut, GM_ADDR vbOut,
        GM_ADDR dvbOut, GM_ADDR dA0Out,
        GM_ADDR dkbIn, GM_ADDR dkbTIn, GM_ADDR doGOut,
        GM_ADDR betaRawIn, GM_ADDR dbetaOut, GM_ADDR dsOut,
        GM_ADDR dqStage12Out, GM_ADDR dkStage12Out, GM_ADDR dk0Stage13Out,
        GM_ADDR workspace,
        const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling, AscendC::TPipe *pipe)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(v));
        vNewGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vNew));
        doGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dO));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(beta));
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(h));
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh));
        if constexpr (USE_QK_L2NORM) {
            qRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qRstd));
            kRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kRstd));
        }
        dqOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dqOut));
        dkOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkOut));
        dgOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(dgOut));
        kbgOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(kbgOut));
        vbOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vbOut));
        a2OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dvbOut));
        dA0OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dA0Out));
        dkbInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbIn));
        dkbTInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbTIn));
        doGOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(doGOut));
        if constexpr (USE_BETA_SIGMOID) {
            betaRawGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(betaRawIn));
        }
        dbetaOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(dbetaOut));
        dsOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dsOut));
        dqStage12OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dqStage12Out));
        dkStage12OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkStage12Out));
        dk0Stage13OutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dk0Stage13Out));
        kbWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace));
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        pipe_ = pipe;
        stateChunkNum_ = tiling_->isVariable != 0 ? tiling_->totalChunkNum : tiling_->chunkNumForT;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx()) / AIV_COUNT_2;
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        subBlockIdx_ = static_cast<int64_t>(AscendC::GetSubBlockIdx());
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
        if (subBlockIdx_ < 0 || subBlockIdx_ >= AIV_COUNT_2) {
            subBlockIdx_ = 0;
        }

        // UB 按最新设计使用完整 248 KiB。所有跨 Stage 数据在
        // 真实末次消费前保持固定地址；Stage 0 的两份 slot 轮转使用，
        // 每轮只搬入和计算当前 owner HV，不提前搬运下一 HV。
        pipe_->InitBuffer(ubBuf_, UB_TOTAL_BYTES);
        k_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, K_OFFSET);
        k_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, K_OFFSET + VECTOR_BYTES);
        v_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, V_OFFSET);
        v_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, V_OFFSET + VECTOR_BYTES);
        gateDA_[0] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, GATE_DA_OFFSET);
        gateDA_[1] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, GATE_DA_OFFSET + MATRIX_BYTES);
        vb_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, VB_OFFSET);
        vb_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, VB_OFFSET + VECTOR_BYTES);
        dvb_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, VB_OFFSET);
        dvb_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, VB_OFFSET + VECTOR_BYTES);
        kbg_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, KBG_OFFSET);
        kbg_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, KBG_OFFSET + VECTOR_BYTES);
        dAu_[0] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, DAU_OFFSET);
        dAu_[1] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, DAU_OFFSET + MATRIX_BYTES);
        // Stage 3 复用原 kbg 的两个 16 KiB bank，每个 bank 的前 8 KiB
        // 保存一份 dA_w0。跨核 Fixpipe 和 Stage 4 必须使用相同 bank 起点。
        dAw0_[0] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, KBG_OFFSET);
        dAw0_[1] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, KBG_OFFSET + VECTOR_BYTES);
        // Stage 5 的 a2 两份 resident 连续占用 UB[128,144) KiB，不能沿用
        // dA_w0 的 16 KiB bank 间距解释第二份数据。
        a2_[0] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, KBG_OFFSET);
        a2_[1] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, KBG_OFFSET + MATRIX_BYTES);
        // Stage 6 开始前，Stage 4 已完成对 dA_u resident 的最终消费。
        // 在原 dA_u 区域起点重新解释两份连续 FP32 标量结果，避开聚合 UB
        // 尾部的小输入区；Stage 7 使用 UB[192,240) KiB，不与此处重叠。
        dbVPartial_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, DB_V_PARTIAL_OFFSET);
        dbVPartial_[1] =
            ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, DB_V_PARTIAL_OFFSET + SMALL_PAIR_SLOT_BYTES);
        dgPrepare_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, DG_PREPARE_OFFSET);
        dgPrepare_[1] =
            ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, DG_PREPARE_OFFSET + SMALL_PAIR_SLOT_BYTES);
        hState_[0] = ubBuf_.GetWithOffset<DT>(STATE_ELEMS, H_STATE_0_OFFSET);
        hState_[1] = ubBuf_.GetWithOffset<DT>(STATE_ELEMS, H_STATE_1_OFFSET);
        dhState_[0] = ubBuf_.GetWithOffset<DT>(STATE_ELEMS, DH_STATE_0_OFFSET);
        dhState_[1] = ubBuf_.GetWithOffset<DT>(STATE_ELEMS, DH_STATE_1_OFFSET);
        dA2_[0] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, DA2_OFFSET);
        dA2_[1] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, DA2_OFFSET + MATRIX_BYTES);
        dkbg0_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, DKBG0_OFFSET);
        dkbg0_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, DKBG0_OFFSET + VECTOR_BYTES);
        kb_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, KB_OFFSET);
        kb_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, KB_OFFSET + VECTOR_BYTES);
        g_[0] = ubBuf_.GetWithOffset<GT>(CHUNK_SIZE_64, G_OFFSET);
        g_[1] = ubBuf_.GetWithOffset<GT>(CHUNK_SIZE_64, G_OFFSET + SMALL_PAIR_SLOT_BYTES);
        beta_[0] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, BETA_OFFSET);
        beta_[1] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, BETA_OFFSET + SMALL_PAIR_SLOT_BYTES);
        betaSaved_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, BETA_OFFSET);
        betaSaved_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, BETA_OFFSET + SMALL_PAIR_SLOT_BYTES);
        gExp_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, G_EXP_OFFSET);
        gExp_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, G_EXP_OFFSET + SMALL_PAIR_SLOT_BYTES);
        gLast_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, G_LAST_OFFSET);
        gLast_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, G_LAST_OFFSET + SMALL_PAIR_SLOT_BYTES);
        decay_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, DECAY_OFFSET);
        decay_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, DECAY_OFFSET + SMALL_PAIR_SLOT_BYTES);
        bg_[0] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, BG_OFFSET);
        bg_[1] = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, BG_OFFSET + SMALL_PAIR_SLOT_BYTES);
        dkb_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_OFFSET);
        dkb_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_OFFSET + VECTOR_BYTES);
        dkbT_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_T_OFFSET);
        dkbT_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_T_OFFSET + VECTOR_BYTES);
        ds_[0] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, STAGE12_DS_OFFSET);
        ds_[1] = ubBuf_.GetWithOffset<DT>(MATRIX_ELEMS, STAGE12_DS_OFFSET + MATRIX_BYTES);
        doG_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DO_G_OFFSET);
        doG_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DO_G_OFFSET + VECTOR_BYTES);
        vDecay_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_V_DECAY_OFFSET);
        vDecay_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_V_DECAY_OFFSET + VECTOR_BYTES);
        betaRaw_[0] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, BETA_RAW_OFFSET);
        betaRaw_[1] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, BETA_RAW_OFFSET + SMALL_PAIR_SLOT_BYTES);
        dbetaOutput_[0] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, BETA_RAW_OFFSET);
        dbetaOutput_[1] = ubBuf_.GetWithOffset<BT>(CHUNK_SIZE_64, BETA_RAW_OFFSET + SMALL_PAIR_SLOT_BYTES);
        dk0Stage13_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_OFFSET);
        dk0Stage13_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_OFFSET + VECTOR_BYTES);
        dqIntraStage13_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_T_OFFSET);
        dqIntraStage13_[1] =
            ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE12_DKB_T_OFFSET + VECTOR_BYTES);
        dkIntraStage13_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE13_DK_INTRA_OFFSET);
        dkIntraStage13_[1] =
            ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE13_DK_INTRA_OFFSET + VECTOR_BYTES);
        // Stage 14 的 q 双槽位固定使用 UB[112,144)。Stage 14 完成后该区
        // 释放，Stage 15 可按新语义解释同一物理地址。
        qStage14_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, 112 * 1024);
        qStage14_[1] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, 128 * 1024);
        // Stage 15 的八个 16 KiB 大槽与 Stage 13 三份 Fixpipe 目标
        // 交错排布，二者完全不重叠。S14 组聚合后即可释放 AIC，
        // S15 在补集空间内运行，不再延长 Stage 13 Fixpipe 的复用等待。
        dqAcc_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_DQ_ACC_OFFSET);
        dqAcc_[1] =
            ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_DQ_ACC_OFFSET + VECTOR_BYTES);
        dqPartialInput_ =
            ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_DQ_PARTIAL_OFFSET);
        dkAcc_[0] = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_DK_ACC_OFFSET);
        dkAcc_[1] =
            ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_DK_ACC_OFFSET + VECTOR_BYTES);
        dkPartialInput_ =
            ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_DK_PARTIAL_OFFSET);
        qStage15_ = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_Q_OFFSET);
        kStage15_ = ubBuf_.GetWithOffset<DT>(VECTOR_ELEMS, STAGE15_K_OFFSET);
        qRstdStage15_ = ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, STAGE15_RSTD_OFFSET);
        kRstdStage15_ =
            ubBuf_.GetWithOffset<float>(CHUNK_SIZE_64, STAGE15_RSTD_OFFSET + 256);

        uint32_t eventIndex = 0;
        for (eventIndex = 0; eventIndex < BANK_COUNT_2; ++eventIndex) {
            mte2ToV_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            vToMte3_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>();
            mte3ToV_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>();
            mte3ToMte2_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            stage12Mte3ToMte2_[eventIndex] =
                pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            stateVToMte2_[eventIndex] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
            // 首轮不存在旧的 MTE3 消费者，先允许 Stage 0 的 MTE2 写入该 slot。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[eventIndex]);
            // 首轮 Stage 0 之前还没有 Stage 12 MTE3，先开放对应 slot。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                stage12Mte3ToMte2_[eventIndex]);
            // 首轮没有旧的 Stage 10 V 消费者，允许 Stage 0 复用对应物理区。
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[eventIndex]);
        }
        stage15VToMte2_ = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        // Stage 15 首次 MTE2 写入前不存在旧的 V 消费者。
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
    }

    __aicore__ inline void Process()
    {
        int64_t taskIdx = 0;
        int64_t chunkTaskIdx = 0;
        int64_t headGroupIdx = 0;
        int64_t taskRound = 0;
        int64_t hvBase = 0;
        int64_t taskCount = 0;
        int64_t headOffset = 0;
        int64_t hv = 0;
        int64_t hk = 0;
        int64_t gOffset = 0;
        int64_t kOffset = 0;
        int64_t vOffset = 0;
        int64_t stateOffset = 0;
        int64_t vectorBase = 0;
        int64_t qOffset = 0;
        int64_t workspaceVectorOffset = 0;
        int64_t workspaceGroupRound = 0;
        int64_t localHv = 0;
        int64_t stage15Begin = 0;
        int64_t stage15End = 0;
        int64_t stage15Head = 0;
        int64_t stage15Hk = 0;
        int64_t stage15AccId = 0;
        int64_t stage15FirstHead = 0;
        int64_t stage15OutputCount = 0;
        int64_t stage15Output = 0;
        uint32_t qSlot = 0;
        uint16_t stage14To15Flag = 0;
        uint32_t matrixBlockBytes = 0;
        uint32_t matrixAlignedBlockBytes = 0;
        AscendC::DataCopyExtParams vectorCopyParams;
        AscendC::DataCopyExtParams matrixCopyParams;
        AscendC::DataCopyExtParams matrixGmToUbCopyParams;
        AscendC::DataCopyExtParams a2CopyParams;
        AscendC::DataCopyExtParams scalarCopyParams;
        ChunkInfo chunk;

        for (taskIdx = coreIdx_; taskIdx < tiling_->taskNum; taskIdx += coreNum_) {
            chunkTaskIdx = taskIdx / tiling_->headGroupNum;
            headGroupIdx = taskIdx % tiling_->headGroupNum;
            GetChunkInfo(chunkTaskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            matrixBlockBytes = static_cast<uint32_t>(chunk.chunkLen * sizeof(DT));
            matrixAlignedBlockBytes = (matrixBlockBytes + UB_ALIGN_BYTES - 1U) /
                UB_ALIGN_BYTES * UB_ALIGN_BYTES;
            matrixCopyParams = {
                static_cast<uint16_t>(chunk.chunkLen),
                matrixBlockBytes,
                static_cast<uint32_t>((CHUNK_SIZE_64 * sizeof(DT) - matrixAlignedBlockBytes) /
                    UB_ALIGN_BYTES),
                static_cast<uint32_t>((CHUNK_SIZE_64 - chunk.chunkLen) * sizeof(DT)), 0};
            // GM 与 UB 的 stride 单位不同，反向搬运不能复用 UB->GM 参数。
            // GM 源矩阵按 chunkLen 紧密存放，UB 目的矩阵固定保持 64 列行距。
            matrixGmToUbCopyParams = {
                static_cast<uint16_t>(chunk.chunkLen),
                matrixBlockBytes,
                static_cast<uint32_t>((CHUNK_SIZE_64 - chunk.chunkLen) * sizeof(DT)),
                static_cast<uint32_t>((CHUNK_SIZE_64 * sizeof(DT) - matrixAlignedBlockBytes) /
                    UB_ALIGN_BYTES), 0};
            a2CopyParams = {
                static_cast<uint16_t>(chunk.chunkLen), matrixBlockBytes,
                static_cast<uint32_t>((CHUNK_SIZE_64 * sizeof(DT) - matrixAlignedBlockBytes) /
                    UB_ALIGN_BYTES),
                static_cast<uint32_t>((K_SIZE_128 - chunk.chunkLen) * sizeof(DT)), 0};
            scalarCopyParams = {
                1, static_cast<uint32_t>(chunk.chunkLen * sizeof(float)), 0, 0, 0};
            hvBase = headGroupIdx * tiling_->taskGroupSize;
            taskCount = Min(tiling_->taskGroupSize, tiling_->HV - hvBase);
            taskRound = (taskIdx - coreIdx_) / coreNum_;
            workspaceGroupRound = taskRound;

                // 唯一 streamSlot 在本 AIV 承包的 HV 之间轮转。
                // 每轮只搬入并计算当前 owner HV。
                streamSlot_ = 0;
                // Stage 0 的 v 搬入及后续 V 写入会复用 Stage 10 的 h/dh
                // 物理区。进入新任务组前等待上一轮 Stage 10 的 V 消费完成。
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[1]);

                // 两个 AIV 按相同的 HEAD 顺序推进。当前 owner 每次
                // 只调用一次 VF 处理一个 HV；non-owner 只参与该 HEAD
                // 的 0x2 聚合。AIC 可以逐 HEAD 消费，不增加任务组 barrier。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        // mode=0x2 要聚合两个 AIV 的 ready。当前 non-owner 虽然
                        // 不生产 vb，也必须发送本 HEAD 的参与信号；否则 AIC 的
                        // 聚合 wait 永远无法满足，owner 已写入的 L1 数据也不能被消费。
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        continue;
                    }
                    hv = hvBase + headOffset;
                    hk = hv / tiling_->headRatio;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    kOffset =
                        ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * tiling_->K;
                    vOffset = ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * tiling_->V;

                    // g 的物理区在 Stage 8 被重新解释为 dgPrepare。必须等上一轮
                    // Stage 8 的最终 MTE3 完成，才能让本轮 MTE2 覆盖整个 slot。
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[streamSlot_]);
                    // 同一物理 slot 在上一任务组末尾还承担 Stage 12 的
                    // dbeta/ds/dq/dk MTE3 源。该专用事件确认全部读取完成。
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                        stage12Mte3ToMte2_[streamSlot_]);
                    AscendC::DataCopyPad(g_[streamSlot_], gGm_[gOffset],
                        {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(GT)), 0, 0, 0},
                        {false, 0, 0, 0});
                    AscendC::DataCopyPad(beta_[streamSlot_], betaGm_[gOffset],
                        {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0},
                        {false, 0, 0, 0});
                    AscendC::DataCopy(k_[streamSlot_], kGm_[kOffset], chunk.chunkLen * K_SIZE_128);
                    AscendC::DataCopy(v_[streamSlot_], vGm_[vOffset], chunk.chunkLen * V_SIZE_128);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);

                    Stage0VF<DT, GT, BT>(
                        reinterpret_cast<__ubuf__ DT *>(gateDA_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(kbg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(vb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gExp_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gLast_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(decay_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(bg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(v_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ GT *>(g_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ BT *>(beta_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);

                    // gExp/gLast/decay/bg/gateDA 只供后续 VEC 使用，保持在
                    // owner AIV 的固定 UB slot。只有 Cube 需要读取的 kbg/vb
                    // 才落到 GM；其有效 [M,128] 使用单条 MTE3 搬运。
                    vectorCopyParams = {
                        1, static_cast<uint32_t>(chunk.chunkLen * K_SIZE_128 * sizeof(DT)), 0, 0, 0};

                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        kbgOutGm_[workspaceVectorOffset], kbg_[streamSlot_], vectorCopyParams);
                    AscendC::DataCopyPad(
                        vbOutGm_[workspaceVectorOffset], vb_[streamSlot_], vectorCopyParams);
                    // owner 的 MTE3 已把本 HEAD 的 kbg/vb 写入当前 workspace slot。
                    // 这里向 AIC 发送 ready，并由 mode=0x2 与 non-owner 的参与
                    // 信号聚合；AIC 收到后才从 GM 搬入自己的 L1 双缓冲。
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                    // Stage 0 与 Stage 1 没有反向依赖。当前 owner 完成本 HEAD
                    // 后立即切换到另一物理 slot，继续处理自己承包的下一个 HEAD；
                    // dA_u 的等待统一放到下面独立的 Stage 2 循环中。
                    streamSlot_ ^= 1U;
                }

                // Stage 0 已独立完成全部 owner HEAD，不被 Stage 1 反向阻塞。
                // 之后按 HEAD 顺序消费 AIC 的逐 HEAD dA_u ready：owner 收到后
                // 立即执行对应 Stage 2，non-owner 只完成 mode=0x2 握手。
                // 每个 HEAD 的 Fixpipe->UB 可见性完全由 Cube->Vector 有序链保证，
                // 不再等待 Stage 1 整组完成信号。
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                        continue;
                    }
                    // 当前 HEAD 的 ready 已保证 AIC Fixpipe 写入 owner UB 完成。
                    // Stage 0 与 Stage 2 使用相同的 owner 顺序和 slot 轮转顺序。
                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    Stage2VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dAu_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(kb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    // kb 在当前 UB slot 生成，随后写 workspace 供 Stage 9 使用。
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    // kb 与后续 v_decay 共用 Region5 的当前 HEAD slot。
                    // Stage9 完成末次读取后，Stage12 才原址覆盖。
                    vectorCopyParams = {
                        1, static_cast<uint32_t>(chunk.chunkLen * K_SIZE_128 * sizeof(DT)), 0, 0, 0};
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        kbWorkspaceGm_[workspaceVectorOffset],
                        kb_[streamSlot_], vectorCopyParams);
                    streamSlot_ ^= 1U;
                }

                // Stage 3 每完成一个 HEAD 的 dA_w0/dvb Fixpipe 就广播一次 ready。
                // 两个 AIV 都按 HEAD 顺序消费；owner 立即执行对应 Stage 4，
                // non-owner 只闭合固定 EventID。Stage 4 不再等待整组 Stage 3。
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        continue;
                    }
                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    // Stage3 的 dA_w0/dvb 已由 Fixpipe 落 UB，不再额外写 GM。
                    // Stage0 在同一 PIPE_MTE3 上完成 kbg/vb 搬出后才发送
                    // Vector->Cube 有序链，Cube 的 Stage3 因此天然晚于该搬出。
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    Stage4VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dAw0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dAu_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    // dA0 由后续 Cube Stage5 使用，按规则写入 GM，不走 UB->L1。
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        dA0OutGm_[workspaceVectorOffset], dAw0_[streamSlot_], matrixCopyParams);
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                    streamSlot_ ^= 1U;
                }

                // Stage 6 只依赖 Stage 3 的 dvb 以及 Stage 0 常驻的 v/beta。
                // AIV 不等待 Stage 5/7 的任何信号，独立连续完成整组 S6；此时
                // AIC 已可根据上一个循环发布的 ready 并行执行 Stage 5/7。
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        continue;
                    }
                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    vOffset = gOffset * tiling_->V;
                    Stage6VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dvb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(v_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dbVPartial_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    // Stage 6 已得到最终 dv。后续 Stage 只消费 db_v_partial，
                    // 不再需要 dv 的中间副本，因此直接写算子输出，避免借用
                    // workspace 后又被 Stage 12 的调试数据覆盖。
                    AscendC::DataCopyPad(
                        dkStage12OutGm_[vOffset], dvb_[streamSlot_], vectorCopyParams);
                    // db_v_partial 保留在 UB，Stage 8 将在原地累加 db0。
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);
                    streamSlot_ ^= 1U;
                }

                // Stage 7 独立完成后发布 ready。AIV 只在 Stage 8 入口
                // 等待其结果；S6 和 S7 之间没有核间同步或执行顺序约束。
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        // S8 完成信号使用 mode=0x2 聚合两个 AIV。
                        // non-owner 不计算，但必须参与本 HEAD 的握手。
                        Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        continue;
                    }
                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    vOffset = gOffset * tiling_->V;
                    // Stage 8 一次 VF 处理本 HV 的全部数据：dA 原地覆盖
                    // dA2，db_v 原地覆盖 db_v_partial，dg_prepare 写入
                    // 独立 FP32 resident。
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    Stage8VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(dA2_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(gateDA_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(a2_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbg0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gExp_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(bg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dbVPartial_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dgPrepare_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    // S8 仅将后续 Cube 需要的 dA 写入 GM。db_v 和
                    // dg_prepare 的后续消费者均为 Vector，继续驻留 UB。
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        dA0OutGm_[workspaceVectorOffset], dA2_[streamSlot_], matrixCopyParams);
                    // PIPE_MTE3 保证 Cube 收到信号前 dA 已落盘，且可以
                    // 复用 S7 的跨核 UB slot。
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                    // 本 slot 在 S8 的最后一次 MTE3 读取已经发射，允许下一
                    // 任务组 Stage 0 在等待该事件后覆盖对应大 buffer。
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[streamSlot_]);
                    streamSlot_ ^= 1U;
                }

                // Stage 8 结束后，新布局中的 h/dh 区域已完成旧语义的最终
                // V 消费。向 Stage 10 的 MTE2 发布整组两份物理 slot。
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[0]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[1]);

                // Stage 10 在 Stage 8 释放旧语义后复用 UB[32,64) 和
                // UB[80,176)。每个 owner
                // HV 完整搬入一份 h/dh，并由一次 VF 归约；non-owner 不参与计算。
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        continue;
                    }
                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    stateOffset =
                        ((chunk.bIdx * tiling_->HV + hv) * stateChunkNum_ + chunk.stateChunkIdx) *
                        STATE_ELEMS;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[streamSlot_]);
                    AscendC::DataCopy(hState_[streamSlot_], hGm_[stateOffset], STATE_ELEMS);
                    AscendC::DataCopy(dhState_[streamSlot_], dhGm_[stateOffset], STATE_ELEMS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    Stage10VF<DT>(
                        reinterpret_cast<__ubuf__ DT *>(hState_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dhState_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gLast_[streamSlot_].GetPhyAddr()));
                    // state_term 的后续消费者是 Vector，结果保持在 UB。
                    // Stage 0 复用该大 slot 前只需等待 S10 的 V 计算完成。
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[streamSlot_]);
                    streamSlot_ ^= 1U;
                }

                // Stage 12 复用 Stage 10 的 h/dh 物理区之前，先分别等待两份
                // slot 的末次 V 消费完成，避免 MTE2 覆盖仍被 V 读取的数据。
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[1]);
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    // Stage 11 已将 ds0 直接 Fixpipe 到 owner 的固定 UB slot；
                    // 两个 AIV 都消费广播，只有 owner 搬入其余输入并执行一次 VF。
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        // mode=0x2 由两个 AIV 共同聚合当前 HEAD 的 ready。
                        // non-owner 不会被本 HEAD 的 Fixpipe 写回，可直接到达。
                        Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                        continue;
                    }
                    hv = hvBase + headOffset;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    vectorBase = gOffset * K_SIZE_128;
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopy(
                        doG_[streamSlot_], doGm_[vectorBase], chunk.chunkLen * V_SIZE_128);
                    AscendC::DataCopy(
                        vDecay_[streamSlot_], vNewGm_[vectorBase], chunk.chunkLen * V_SIZE_128);
                    if constexpr (USE_BETA_SIGMOID) {
                        AscendC::DataCopyPad(betaRaw_[streamSlot_], betaRawGm_[gOffset],
                            {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0},
                            {false, 0, 0, 0});
                    }
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    AscendC::DataCopy(
                        dkb_[streamSlot_], dkbInGm_[workspaceVectorOffset], chunk.chunkLen * K_SIZE_128);
                    AscendC::DataCopy(
                        dkbT_[streamSlot_], dkbTInGm_[workspaceVectorOffset], chunk.chunkLen * K_SIZE_128);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[streamSlot_]);
                    Stage12VF<DT, BT, USE_BETA_SIGMOID, ALLOW_NEG_EIGVAL>(
                        reinterpret_cast<__ubuf__ DT *>(dkb_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbT_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(ds_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(doG_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(vDecay_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbg0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(gateDA_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaSaved_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gExp_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(decay_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(bg_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dbVPartial_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ BT *>(betaRaw_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(betaRaw_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ BT *>(dbetaOutput_[streamSlot_].GetPhyAddr()),
                        tiling_->scale, static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    if constexpr (std::is_same<BT, float>::value) {
                        AscendC::DataCopyPad(dbetaOutGm_[gOffset], dbVPartial_[streamSlot_],
                            {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0});
                    } else {
                        AscendC::DataCopyPad(dbetaOutGm_[gOffset], dbetaOutput_[streamSlot_],
                            {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(BT)), 0, 0, 0});
                    }
                    AscendC::DataCopyPad(
                        dsOutGm_[workspaceVectorOffset], ds_[streamSlot_], matrixCopyParams);
                    AscendC::DataCopyPad(
                        doGOutGm_[workspaceVectorOffset], doG_[streamSlot_], vectorCopyParams);
                    AscendC::DataCopyPad(
                        dk0Stage13OutGm_[workspaceVectorOffset], vDecay_[streamSlot_], vectorCopyParams);
                    // dk_prepare 的下一消费者是 Stage 14 Vector，继续驻留 UB。
                    // 不写最终 dv 输出，也不为调试额外落 GM。
                    // Stage 13 的 dk_base/dq_hv/dk_intra Fixpipe 会逐 slot 覆盖
                    // UB[32,64)、UB[80,112) 和 UB[144,176)，其中包含本轮 Stage 12 的
                    // MTE3 源。必须先确认这些 MTE3 已读完，再允许 AIC 进入
                    // Stage 13；跨核 ready 本身不替代本核 MTE3 的完成事件。
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToV_[streamSlot_]);
                    // Stage 13 按 owner/local-HV 写回同一个物理 slot。当前
                    // HEAD 的 MTE3 已经读完 dkb/dkbT/vDecay 重叠区，可立即
                    // 发布单 HEAD ready；后续同 owner HEAD 使用另一个 slot。
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                    // 当前调试链还会由 Stage 13 复用并从该大 slot 搬出结果；
                    // 因此这里不提前发布反向事件，统一在最终 MTE3 后发布一次。
                    streamSlot_ ^= 1U;
                }
                // Stage 14 逐 HEAD 等待 Stage 13 Fixpipe ready，直接消费三份
                // 固定 UB resident。中间结果不再为了调试重复写 GM。
                streamSlot_ = 0;
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    Catlass::Arch::CrossCoreWaitFlag(cubeToVecFlag_);
                    if (headOffset % AIV_COUNT_2 != subBlockIdx_) {
                        continue;
                    }
                    hv = hvBase + headOffset;
                    hk = hv / tiling_->headRatio;
                    localHv = hv - hk * tiling_->headRatio;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart;
                    qOffset = ((chunk.bIdx * tiling_->HK + hk) * tiling_->T +
                        chunk.tokenStart) * tiling_->K;
                    // G=3/4 时同一 AIV 会处理同一 HK 的两个 HV，第二个 HV
                    // 直接复用本 AIV 在 slot0 中已经驻留的 q，不重复读 GM。
                    qSlot = tiling_->headRatio >= 3 ? 0U : streamSlot_;
                    if (tiling_->headRatio < 3 || localHv < AIV_COUNT_2) {
                        AscendC::DataCopy(
                            qStage14_[qSlot], qGm_[qOffset], chunk.chunkLen * K_SIZE_128);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[qSlot]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[qSlot]);
                    }
                    Stage14VF<DT, GT>(
                        reinterpret_cast<__ubuf__ DT *>(dk0Stage13_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dqIntraStage13_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkIntraStage13_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(dkbg0_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(qStage14_[qSlot].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ DT *>(k_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(dgPrepare_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(gLast_[streamSlot_].GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float *>(decay_[streamSlot_].GetPhyAddr()),
                        static_cast<uint16_t>(chunk.chunkLen));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[streamSlot_]);
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, headOffset);
                    AscendC::DataCopyPad(
                        dqStage12OutGm_[workspaceVectorOffset],
                        dqIntraStage13_[streamSlot_], vectorCopyParams);
                    AscendC::DataCopyPad(
                        a2OutGm_[workspaceVectorOffset],
                        dk0Stage13_[streamSlot_], vectorCopyParams);
                    AscendC::DataCopyPad(
                        dgOutGm_[gOffset],
                        decay_[streamSlot_].template ReinterpretCast<GT>(),
                        {1, static_cast<uint32_t>(chunk.chunkLen * sizeof(GT)), 0, 0, 0});
                    streamSlot_ ^= 1U;
                }

                // Stage 12 已消费 Stage 10 发布的 V->MTE2 状态。此处在
                // Stage 14 全部 VF 之后重新发布，保护 Stage 15 首次 MTE2
                // 对 Stage 14 V 输入/输出物理区的重新解释。
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[0]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[1]);

                // Stage 14 -> Stage 15 每个 task group 只同步一次。mode=1
                // 等待同一 AI Core 内两个 AIV 的全部 W14 MTE3 完成，Stage 15
                // 才能读取由另一 AIV 生产的 HV partial。三个普通 EventID 按组
                // 轮转，避免大 HV case 在同一 ID 上超过普通计数上限。
                stage14To15Flag = static_cast<uint16_t>(8 + taskRound % 3);
                AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(stage14To15Flag);
                AscendC::CrossCoreWaitFlag(stage14To15Flag);
                // 两个 AIV 的 S14 已完成对 S13 Fixpipe 目标的末次消费。
                // S15 使用与这三份目标不重叠的 UB，在此立即释放 AIC。
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCubeFlag_);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[0]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[1]);

                // G=1/2 时两个 AIV 各承包 task group 中连续两个 HV；G=3/4
                // 仅 AIV0 承包全部 HV。这里的分工只用于 Stage 15，不改变
                // Stage 0--14 的交错 owner 规则。
                if (tiling_->headRatio <= 2) {
                    stage15Begin = subBlockIdx_ * AIV_COUNT_2;
                    stage15End = Min(stage15Begin + AIV_COUNT_2, taskCount);
                } else {
                    stage15Begin = 0;
                    stage15End = subBlockIdx_ == 0 ? taskCount : 0;
                }
                stage15OutputCount = 0;
                stage15AccId = 0;
                stage15FirstHead = stage15Begin;
                for (stage15Head = stage15Begin; stage15Head < stage15End; ++stage15Head) {
                    stage15AccId = (stage15Head - stage15Begin) / tiling_->headRatio;
                    hv = hvBase + stage15Head;
                    gOffset = (chunk.bIdx * tiling_->HV + hv) * tiling_->T +
                        chunk.tokenStart;
                    workspaceVectorOffset = GetWorkspaceHeadOffset(
                        coreIdx_, workspaceGroupRound, stage15Head);
                    if ((stage15Head - stage15Begin) % tiling_->headRatio == 0) {
                        AscendC::DataCopy(
                            dqAcc_[stage15AccId], dqStage12OutGm_[workspaceVectorOffset],
                            chunk.chunkLen * K_SIZE_128);
                        AscendC::DataCopy(
                            dkAcc_[stage15AccId], a2OutGm_[workspaceVectorOffset],
                            chunk.chunkLen * K_SIZE_128);
                        stage15OutputCount = stage15AccId + 1;
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                        AscendC::DataCopy(
                            dqPartialInput_, dqStage12OutGm_[workspaceVectorOffset],
                            chunk.chunkLen * K_SIZE_128);
                        AscendC::DataCopy(
                            dkPartialInput_, a2OutGm_[workspaceVectorOffset],
                            chunk.chunkLen * K_SIZE_128);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                        Stage15AccumulateVF<DT>(
                            reinterpret_cast<__ubuf__ DT *>(dqAcc_[stage15AccId].GetPhyAddr()),
                            reinterpret_cast<__ubuf__ DT *>(dkAcc_[stage15AccId].GetPhyAddr()),
                            reinterpret_cast<__ubuf__ DT *>(dqPartialInput_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ DT *>(dkPartialInput_.GetPhyAddr()),
                            static_cast<uint16_t>(chunk.chunkLen));
                        // 当前 VF 已完成对共享 partial 输入槽的读取，下一份
                        // MTE2 覆盖该槽之前必须等待这条反向事件。
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                    }
                }

                // 每个活跃 AIV 按 accumulator 顺序写自己承包的最终 HK。
                for (stage15Output = 0; stage15Output < stage15OutputCount; ++stage15Output) {
                    stage15FirstHead = stage15Begin + stage15Output * tiling_->headRatio;
                    stage15Hk = (hvBase + stage15FirstHead) / tiling_->headRatio;
                    qOffset = ((chunk.bIdx * tiling_->HK + stage15Hk) * tiling_->T +
                        chunk.tokenStart) * tiling_->K;
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                    if constexpr (USE_QK_L2NORM) {
                        AscendC::DataCopy(
                            qStage15_, qGm_[qOffset], chunk.chunkLen * K_SIZE_128);
                        AscendC::DataCopy(
                            kStage15_, kGm_[qOffset], chunk.chunkLen * K_SIZE_128);
                        AscendC::DataCopyPad(qRstdStage15_,
                            qRstdGm_[qOffset / K_SIZE_128], scalarCopyParams,
                            {false, 0, 0, 0});
                        AscendC::DataCopyPad(kRstdStage15_,
                            kRstdGm_[qOffset / K_SIZE_128], scalarCopyParams,
                            {false, 0, 0, 0});
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV_[0]);
                    if constexpr (USE_QK_L2NORM) {
                        Stage15NormVF<DT>(
                            reinterpret_cast<__ubuf__ DT *>(dqAcc_[stage15Output].GetPhyAddr()),
                            reinterpret_cast<__ubuf__ DT *>(dkAcc_[stage15Output].GetPhyAddr()),
                            reinterpret_cast<__ubuf__ DT *>(qStage15_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ DT *>(kStage15_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float *>(qRstdStage15_.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float *>(kRstdStage15_.GetPhyAddr()),
                            static_cast<uint16_t>(chunk.chunkLen));
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vToMte3_[0]);
                    AscendC::DataCopyPad(dqOutGm_[qOffset], dqAcc_[stage15Output], vectorCopyParams);
                    AscendC::DataCopyPad(dkOutGm_[qOffset], dkAcc_[stage15Output], vectorCopyParams);
                }
                // 所有 Stage 14/15 MTE3 已发射，下一任务组 Stage 0 才能
                // 覆盖 Stage 12/13/14 共用的两份大 UB slot。
                if (taskCount > subBlockIdx_) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                        stage12Mte3ToMte2_[0]);
                }
                if (taskCount > subBlockIdx_ + AIV_COUNT_2) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                        stage12Mte3ToMte2_[1]);
                }
                // Stage 15 已完成最终 V 与 MTE3 消费，重新发布给下一任务组
                // Stage 0，保持两套事件在任务组边界完整闭环。
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[0]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[1]);
                // 本 AIV 完成本组全部 Stage 4 后自然进入下一组 Stage 0。
                // 下一组 AIC Stage 1 仍逐 HEAD等待 Vector->Cube 有序链，
                // 因此无需额外发送任务组边界 flag。
        }
        // 事件在 kernel 退出前按 slot 闭环并释放，不留单独的小函数。
        uint32_t eventIndex = 0;
        for (eventIndex = 0; eventIndex < BANK_COUNT_2; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[eventIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                stage12Mte3ToMte2_[eventIndex]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stateVToMte2_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2ToV_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3ToV_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(
                stage12Mte3ToMte2_[eventIndex]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(stateVToMte2_[eventIndex]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(stage15VToMte2_);
    }

private:
    static constexpr uint32_t VECTOR_ELEMS = CHUNK_SIZE_64 * K_SIZE_128;
    static constexpr uint32_t MATRIX_ELEMS = CHUNK_SIZE_64 * CHUNK_SIZE_64;
    static constexpr uint32_t STATE_ELEMS = K_SIZE_128 * V_SIZE_128;
    static constexpr uint32_t VECTOR_BYTES = VECTOR_ELEMS * sizeof(DT);
    static constexpr uint32_t MATRIX_BYTES = MATRIX_ELEMS * sizeof(DT);
    static constexpr uint32_t STATE_BYTES = STATE_ELEMS * sizeof(DT);
    static constexpr uint32_t SMALL_PAIR_SLOT_BYTES = CHUNK_SIZE_64 * sizeof(float);

    static constexpr uint32_t K_OFFSET = 0 * 1024;
    static constexpr uint32_t V_OFFSET = 32 * 1024;
    static constexpr uint32_t GATE_DA_OFFSET = 64 * 1024;
    static constexpr uint32_t VB_OFFSET = 80 * 1024;
    static constexpr uint32_t H_STATE_0_OFFSET = 32 * 1024;
    static constexpr uint32_t H_STATE_1_OFFSET = 80 * 1024;
    static constexpr uint32_t DH_STATE_0_OFFSET = 112 * 1024;
    static constexpr uint32_t DH_STATE_1_OFFSET = 144 * 1024;
    static constexpr uint32_t DAU_OFFSET = 112 * 1024;
    static constexpr uint32_t DB_V_PARTIAL_OFFSET = 244 * 1024 + 512;
    static constexpr uint32_t DG_PREPARE_OFFSET = 245 * 1024;
    static constexpr uint32_t KBG_OFFSET = 128 * 1024;
    static constexpr uint32_t KB_OFFSET = 160 * 1024;
    static constexpr uint32_t DA2_OFFSET = 192 * 1024;
    static constexpr uint32_t DKBG0_OFFSET = 208 * 1024;
    static constexpr uint32_t STAGE12_DKB_OFFSET = 32 * 1024;
    static constexpr uint32_t STAGE12_DKB_T_OFFSET = 80 * 1024;
    static constexpr uint32_t STAGE12_DO_G_OFFSET = 112 * 1024;
    static constexpr uint32_t STAGE12_V_DECAY_OFFSET = 144 * 1024;
    static constexpr uint32_t STAGE12_DS_OFFSET = 176 * 1024;
    static constexpr uint32_t STAGE13_DK_INTRA_OFFSET = 144 * 1024;
    static constexpr uint32_t STAGE15_DQ_ACC_OFFSET = 0 * 1024;
    static constexpr uint32_t STAGE15_DQ_PARTIAL_OFFSET = 64 * 1024;
    static constexpr uint32_t STAGE15_DK_ACC_OFFSET = 112 * 1024;
    static constexpr uint32_t STAGE15_DK_PARTIAL_OFFSET = 176 * 1024;
    static constexpr uint32_t STAGE15_Q_OFFSET = 192 * 1024;
    static constexpr uint32_t STAGE15_K_OFFSET = 208 * 1024;
    static constexpr uint32_t STAGE15_RSTD_OFFSET = 244 * 1024;
    static constexpr uint32_t BETA_RAW_OFFSET = 244 * 1024;
    static constexpr uint32_t G_OFFSET = 245 * 1024;
    static constexpr uint32_t BETA_OFFSET = 245 * 1024 + 512;
    static constexpr uint32_t G_EXP_OFFSET = 246 * 1024;
    static constexpr uint32_t G_LAST_OFFSET = 246 * 1024 + 512;
    static constexpr uint32_t DECAY_OFFSET = 247 * 1024;
    static constexpr uint32_t BG_OFFSET = 247 * 1024 + 512;
    static constexpr uint32_t UB_TOTAL_BYTES = 248 * 1024;

    static_assert(VECTOR_BYTES == 16 * 1024, "Stage 0 vector slot must be 16 KiB.");
    static_assert(MATRIX_BYTES == 8 * 1024, "Stage 0 matrix slot must be 8 KiB.");
    static_assert(VB_OFFSET + 2 * VECTOR_BYTES == DAU_OFFSET,
                  "Stage 0 vb ping/pong must end at the Stage 1 dA_u slots.");
    static_assert(DAU_OFFSET + 2 * MATRIX_BYTES == KBG_OFFSET,
                  "Stage 1 dA_u slots must occupy UB[112,128) KiB.");
    static_assert(DB_V_PARTIAL_OFFSET + 2 * SMALL_PAIR_SLOT_BYTES == G_OFFSET,
                  "Stage 6 db_v slots must occupy UB[244.5,245) KiB.");
    static_assert(KBG_OFFSET + 2 * VECTOR_BYTES == 160 * 1024,
                   "Stage 0 kbg ping/pong must end at UB 160 KiB.");
    static_assert(KB_OFFSET + 2 * VECTOR_BYTES == 192 * 1024,
                  "Stage 2 kb slots must occupy UB[160,192) KiB.");
    static_assert(DA2_OFFSET + 2 * MATRIX_BYTES == DKBG0_OFFSET,
                  "Stage 7 dA2 slots must occupy UB[192,208) KiB.");
    static_assert(DKBG0_OFFSET + 2 * VECTOR_BYTES == 240 * 1024,
                  "Stage 7 dkbg0 slots must occupy UB[208,240) KiB.");
    static_assert(H_STATE_0_OFFSET + STATE_BYTES == 64 * 1024,
                  "Stage 10 h slot 0 must occupy UB[32,64) KiB.");
    static_assert(H_STATE_1_OFFSET + STATE_BYTES == DH_STATE_0_OFFSET,
                  "Stage 10 h slot 1 must occupy UB[80,112) KiB.");
    static_assert(DH_STATE_0_OFFSET + STATE_BYTES == DH_STATE_1_OFFSET,
                  "Stage 10 dh slot 0 must occupy UB[112,144) KiB.");
    static_assert(DH_STATE_1_OFFSET + STATE_BYTES == 176 * 1024,
                  "Stage 10 dh slot 1 must occupy UB[144,176) KiB.");
    static_assert(STAGE12_DKB_T_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == 112 * 1024,
                  "Stage 12 dkbT and Stage 13 dq_hv slots must occupy UB[80,112) KiB.");
    static_assert(STAGE15_DQ_ACC_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == STAGE12_DKB_OFFSET,
                  "Stage 15 dq accumulators must end before Stage 13 dk_base slots.");
    static_assert(STAGE12_DKB_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == STAGE15_DQ_PARTIAL_OFFSET,
                  "Stage 15 dq partial must follow Stage 13 dk_base slots.");
    static_assert(STAGE15_DQ_PARTIAL_OFFSET + VECTOR_BYTES == STAGE12_DKB_T_OFFSET,
                  "Stage 15 dq partial must end before Stage 13 dq_hv slots.");
    static_assert(STAGE12_DKB_T_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == STAGE15_DK_ACC_OFFSET,
                  "Stage 15 dk accumulators must follow Stage 13 dq_hv slots.");
    static_assert(STAGE15_DK_ACC_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == STAGE13_DK_INTRA_OFFSET,
                  "Stage 15 dk accumulators must end before Stage 13 dk_intra slots.");
    static_assert(STAGE13_DK_INTRA_OFFSET + BANK_COUNT_2 * VECTOR_BYTES ==
                      STAGE15_DK_PARTIAL_OFFSET,
                  "Stage 15 dk partial must follow Stage 13 dk_intra slots.");
    static_assert(STAGE15_DK_PARTIAL_OFFSET + VECTOR_BYTES == STAGE15_Q_OFFSET &&
                      STAGE15_Q_OFFSET + VECTOR_BYTES == STAGE15_K_OFFSET &&
                      STAGE15_K_OFFSET + VECTOR_BYTES <= STAGE15_RSTD_OFFSET,
                  "Stage 15 tail tensors must be contiguous and remain below the scalar area.");
    static_assert(BG_OFFSET + 2 * SMALL_PAIR_SLOT_BYTES == UB_TOTAL_BYTES,
                  "Stage 0 small resident must end at UB 248 KiB.");
    static_assert(UB_TOTAL_BYTES == 248 * 1024, "Stage 0 UB layout must use exactly 248 KiB.");

    const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    int64_t subBlockIdx_ = 0;
    uint32_t streamSlot_ = 0;
    // 两条核间同步链均按 HEAD 顺序推进。每条链只使用一个业务 ready id，
    // 两个 AIV 与 AIC 共用两条普通事件链；相反方向的逐 HEAD 依赖负责限制未消费计数。
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{VEC_TO_CUBE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CUBE_TO_VEC_READY_FLAG};
    AscendC::GlobalTensor<DT> kGm_;
    AscendC::GlobalTensor<DT> qGm_;
    AscendC::GlobalTensor<DT> vGm_;
    AscendC::GlobalTensor<DT> vNewGm_;
    AscendC::GlobalTensor<DT> doGm_;
    AscendC::GlobalTensor<GT> gGm_;
    AscendC::GlobalTensor<BT> betaGm_;
    AscendC::GlobalTensor<DT> hGm_;
    AscendC::GlobalTensor<DT> dhGm_;
    AscendC::GlobalTensor<float> qRstdGm_;
    AscendC::GlobalTensor<float> kRstdGm_;
    AscendC::GlobalTensor<DT> dqOutGm_;
    AscendC::GlobalTensor<DT> dkOutGm_;
    AscendC::GlobalTensor<GT> dgOutGm_;
    AscendC::GlobalTensor<DT> kbgOutGm_;
    AscendC::GlobalTensor<DT> vbOutGm_;
    AscendC::GlobalTensor<DT> a2OutGm_;
    AscendC::GlobalTensor<DT> dA0OutGm_;
    AscendC::GlobalTensor<DT> kbWorkspaceGm_;
    AscendC::GlobalTensor<DT> dkbInGm_;
    AscendC::GlobalTensor<DT> dkbTInGm_;
    AscendC::GlobalTensor<DT> doGOutGm_;
    AscendC::GlobalTensor<BT> betaRawGm_;
    AscendC::GlobalTensor<BT> dbetaOutGm_;
    AscendC::GlobalTensor<DT> dsOutGm_;
    AscendC::GlobalTensor<DT> dqStage12OutGm_;
    AscendC::GlobalTensor<DT> dkStage12OutGm_;
    AscendC::GlobalTensor<DT> dk0Stage13OutGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_;
    AscendC::LocalTensor<DT> k_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> v_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> gateDA_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> vb_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dvb_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> kbg_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dAu_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dAw0_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> a2_[BANK_COUNT_2];
    AscendC::LocalTensor<float> dbVPartial_[BANK_COUNT_2];
    AscendC::LocalTensor<float> dgPrepare_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> hState_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dhState_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dA2_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dkbg0_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dkb_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dkbT_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> ds_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> doG_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> vDecay_[BANK_COUNT_2];
    AscendC::LocalTensor<BT> betaRaw_[BANK_COUNT_2];
    AscendC::LocalTensor<BT> dbetaOutput_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dk0Stage13_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dqIntraStage13_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dkIntraStage13_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> qStage14_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dqAcc_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dkAcc_[BANK_COUNT_2];
    AscendC::LocalTensor<DT> dqPartialInput_;
    AscendC::LocalTensor<DT> dkPartialInput_;
    AscendC::LocalTensor<DT> qStage15_;
    AscendC::LocalTensor<DT> kStage15_;
    AscendC::LocalTensor<float> qRstdStage15_;
    AscendC::LocalTensor<float> kRstdStage15_;
    AscendC::LocalTensor<DT> kb_[BANK_COUNT_2];
    AscendC::LocalTensor<GT> g_[BANK_COUNT_2];
    AscendC::LocalTensor<BT> beta_[BANK_COUNT_2];
    AscendC::LocalTensor<float> betaSaved_[BANK_COUNT_2];
    AscendC::LocalTensor<float> gExp_[BANK_COUNT_2];
    AscendC::LocalTensor<float> gLast_[BANK_COUNT_2];
    AscendC::LocalTensor<float> decay_[BANK_COUNT_2];
    AscendC::LocalTensor<float> bg_[BANK_COUNT_2];
    AscendC::TEventID mte2ToV_[BANK_COUNT_2];
    AscendC::TEventID vToMte3_[BANK_COUNT_2];
    AscendC::TEventID mte3ToV_[BANK_COUNT_2];
    AscendC::TEventID mte3ToMte2_[BANK_COUNT_2];
    AscendC::TEventID stage12Mte3ToMte2_[BANK_COUNT_2];
    AscendC::TEventID stateVToMte2_[BANK_COUNT_2];
    AscendC::TEventID stage15VToMte2_;
    int64_t stateChunkNum_ = 0;
};

} // namespace GDN

#endif
