/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_TARGET_CHUNK_KDA_FWD_PREPARE_VEC_H
#define ARCH22_TARGET_CHUNK_KDA_FWD_PREPARE_VEC_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_utils.h"
#include "chunk_kda_fwd_prepare_target_common.h"

namespace KdaPrepare::Arch22Target {

// softplus 的 log1p 补偿：u=exp(-|x|)，log1p(u)=Ln(1+u)+r/(1+u)，
// r=u-(fl(1+u)-1)。两个 FP32 临时区固定复用共享 scratch 的前 7 KiB（与 V1Vf
// 同一约定）：scratch 之后紧邻第二份 private 槽，0x1F00 起是 sequence-major 的
// beta gather 偏移表，因此临时区不得越过 0x1C00。


// Computation helpers consume local tensors; GM ownership stays in the stage functions.
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void SubTileMinusRow(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> tile,
                                       AscendC::LocalTensor<float> row, uint32_t rows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void SubRowMinusTile(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> row,
                                       AscendC::LocalTensor<float> tile, uint32_t rows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void AddRowToTile(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> tile,
                                    AscendC::LocalTensor<float> row, uint32_t rows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ReplicateRowScalarsToRepeat(AscendC::LocalTensor<float> dstTile,
                                                   AscendC::LocalTensor<float> rowScalars, uint32_t rows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void MulTileByRowScalars(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> tile,
                                           AscendC::LocalTensor<float> rowScalarRepeat, uint32_t rows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void SolveLeafRightLooking(AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> lkk,
                                             AscendC::LocalTensor<uint32_t> coeffOffset,
                                             AscendC::LocalTensor<float> coeffTile, AscendC::LocalTensor<float> prod,
                                             uint32_t leafRows, uint32_t lkkBase, uint32_t rowStride);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void
SolveTwoLeavesRightLooking(AscendC::LocalTensor<float> x0, AscendC::LocalTensor<float> x1,
                           AscendC::LocalTensor<float> lkk, AscendC::LocalTensor<uint32_t> coeffOffset,
                           AscendC::LocalTensor<float> coeffA, AscendC::LocalTensor<float> prodA,
                           AscendC::LocalTensor<float> coeffB, AscendC::LocalTensor<float> prodB, uint32_t topRows,
                           uint32_t bottomRows, uint32_t baseA, uint32_t baseB, uint32_t rowStride);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void V1Vf(AscendC::LocalTensor<bfloat16_t> qHat, AscendC::LocalTensor<bfloat16_t> kHat,
                            AscendC::LocalTensor<bfloat16_t> qPlus, AscendC::LocalTensor<bfloat16_t> kPlus,
                            AscendC::LocalTensor<bfloat16_t> kMinus, AscendC::LocalTensor<float> g,
                            AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> stageBuf,
                            AscendC::LocalTensor<float> bandDeltaBuf, uint32_t validRows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void V6Vf(AscendC::LocalTensor<bfloat16_t> qg, AscendC::LocalTensor<bfloat16_t> qgScaled,
                            AscendC::LocalTensor<bfloat16_t> kg, AscendC::LocalTensor<bfloat16_t> vBeta,
                            AscendC::LocalTensor<bfloat16_t> kBetaG, AscendC::LocalTensor<float> g,
                            AscendC::LocalTensor<float> betaEff, AscendC::LocalTensor<float> scratch,
                            AscendC::LocalTensor<float> stageBuf, uint32_t validRows, float scale,
                            AscendC::TEventID restReady);

// 行广播二元运算：把 128 lane 的行张量广播到 rows 行。
// fp32 一次 repeat 只有 64 lane，因此按低/高半行各下发一条指令，
// 广播侧用 repeatStride=0，一次 replace 内 8 个 block 连续读取。
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void SubTileMinusRow(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> tile,
                                       AscendC::LocalTensor<float> row, uint32_t rows)
{
    const uint8_t rowStride = static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
    AscendC::Sub(dst, tile, row, 64, static_cast<uint8_t>(rows), {1, 1, 1, rowStride, rowStride, 0});
    AscendC::Sub(dst[64], tile[64], row[64], 64, static_cast<uint8_t>(rows), {1, 1, 1, rowStride, rowStride, 0});
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void SubRowMinusTile(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> row,
                                       AscendC::LocalTensor<float> tile, uint32_t rows)
{
    const uint8_t rowStride = static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
    AscendC::Sub(dst, row, tile, 64, static_cast<uint8_t>(rows), {1, 1, 1, rowStride, 0, rowStride});
    AscendC::Sub(dst[64], row[64], tile[64], 64, static_cast<uint8_t>(rows), {1, 1, 1, rowStride, 0, rowStride});
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void AddRowToTile(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> tile,
                                    AscendC::LocalTensor<float> row, uint32_t rows)
{
    const uint8_t rowStride = static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
    AscendC::Add(dst, tile, row, 64, static_cast<uint8_t>(rows), {1, 1, 1, rowStride, rowStride, 0});
    AscendC::Add(dst[64], tile[64], row[64], 64, static_cast<uint8_t>(rows), {1, 1, 1, rowStride, rowStride, 0});
}

// 逐行标量缩放（fp32，行宽 128 lane = 两个 repeat）：
// dataBlockStride 不支持 0，所以先把“每行 1 个 block”的 Brcb 展开
// 复制成“每行 8 个相同 block”（正好填满一个 64 lane repeat），
// 再用 repStride=8 让每个 repeat 取到自己那一行。
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ReplicateRowScalarsToRepeat(AscendC::LocalTensor<float> dstTile,
                                                   AscendC::LocalTensor<float> rowScalars, uint32_t rows)
{
    for (uint8_t block = 0; block < 8; ++block) {
        AscendC::DataCopy(dstTile[8 * block], rowScalars,
                          AscendC::DataCopyParams(static_cast<uint16_t>(rows), 1, 0, 7));
    }
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void MulTileByRowScalars(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> tile,
                                           AscendC::LocalTensor<float> rowScalarRepeat, uint32_t rows)
{
    const uint8_t rowStride = static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
    AscendC::Mul(dst, tile, rowScalarRepeat, 64, static_cast<uint8_t>(rows), {1, 1, 1, rowStride, rowStride, 8});
    AscendC::Mul(dst[64], tile[64], rowScalarRepeat, 64, static_cast<uint8_t>(rows),
                 {1, 1, 1, rowStride, rowStride, 8});
}

// 单位下三角逆的右看消元：对 leafRows 阶单位下三角块求 X=(I+L)^-1。
// 每轮需要“L 的第 k 列按行复制成 4 个 block”的系数张量。这里用带元素
// 偏移的 Gather 直接从 lkk 取：dst 的第 j 行 32 个 lane 全取
// lkk[lkkBase + (k+1+j)*rowStride + k]，等价于该列复制 4 个 block，
// 一条指令替代原来的 1 次 DataCopy + 1 次 Brcb + 4 次 DataCopy。
// 偏移张量与 k 无关（只随行变化），k 由 srcBaseAddr 吸收。
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void SolveLeafRightLooking(AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> lkk,
                                             AscendC::LocalTensor<uint32_t> coeffOffset,
                                             AscendC::LocalTensor<float> coeffTile, AscendC::LocalTensor<float> prod,
                                             uint32_t leafRows, uint32_t lkkBase, uint32_t rowStride)
{
    constexpr uint32_t kLeaf = 32;
    constexpr uint8_t kLeafRepStride = static_cast<uint8_t>(kLeaf * sizeof(float) / 32);
    for (uint32_t k = 0; k + 1 < leafRows; ++k) {
        const uint32_t rows = leafRows - k - 1;
        // (k+1+j)*rowStride + k = (lkkBase + (rowStride+1)*k) + rowStride*(j+1)
        const uint32_t baseElems = lkkBase + (rowStride + 1) * k;
        AscendC::Gather(coeffTile, lkk, coeffOffset, static_cast<uint32_t>(baseElems * sizeof(float)), rows * kLeaf);
        AscendC::PipeBarrier<PIPE_V>();
        // src0 用 repStride=0 反复读取 X[k,:]，src1 是逐行不同的系数，
        // dst 是 k 以下各行。
        AscendC::Mul(prod, x[k * kLeaf], coeffTile, kLeaf, static_cast<uint8_t>(rows),
                     {1, 1, 1, kLeafRepStride, 0, kLeafRepStride});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(x[(k + 1) * kLeaf], x[(k + 1) * kLeaf], prod, static_cast<int32_t>(rows * kLeaf));
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// 两个叶子的 right-looking 三角逆合并进同一个循环：X0 与 X1 互不依赖，
// 原来两次调用各 31 步 × 3 屏障 = 186 个屏障，合并后 31 步 × 3 屏障 = 93，
// 而且两条独立链可以在 vector pipe 里互相掩盖。每步的算子与顺序（Gather→
// Mul→Sub）与单叶版本完全一致，因此逐位舍入顺序不变。
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void
SolveTwoLeavesRightLooking(AscendC::LocalTensor<float> x0, AscendC::LocalTensor<float> x1,
                           AscendC::LocalTensor<float> lkk, AscendC::LocalTensor<uint32_t> coeffOffset,
                           AscendC::LocalTensor<float> coeffA, AscendC::LocalTensor<float> prodA,
                           AscendC::LocalTensor<float> coeffB, AscendC::LocalTensor<float> prodB, uint32_t topRows,
                           uint32_t bottomRows, uint32_t baseA, uint32_t baseB, uint32_t rowStride)
{
    constexpr uint32_t kLeaf = 32;
    constexpr uint8_t kLeafRepStride = static_cast<uint8_t>(kLeaf * sizeof(float) / 32);
    const uint32_t maxRows = topRows > bottomRows ? topRows : bottomRows;
    // k=0 的系数先取好；之后每轮把「下一轮 Gather」与「本轮 Sub」放进
    // 同一屏障区间（两者互不依赖），屏障数从 3/步降到 2/步。
    if (topRows > 1) {
        AscendC::Gather(coeffA, lkk, coeffOffset, static_cast<uint32_t>(baseA * sizeof(float)), (topRows - 1) * kLeaf);
    }
    if (bottomRows > 1) {
        AscendC::Gather(coeffB, lkk, coeffOffset, static_cast<uint32_t>(baseB * sizeof(float)),
                        (bottomRows - 1) * kLeaf);
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t k = 0; k + 1 < maxRows; ++k) {
        const uint32_t rowsA = k + 1 < topRows ? topRows - k - 1 : 0;
        const uint32_t rowsB = k + 1 < bottomRows ? bottomRows - k - 1 : 0;
        // src0 用 repStride=0 反复读本叶的 X[k,:]，src1 是逐行系数。
        if (rowsA > 0) {
            AscendC::Mul(prodA, x0[k * kLeaf], coeffA, kLeaf, static_cast<uint8_t>(rowsA),
                         {1, 1, 1, kLeafRepStride, 0, kLeafRepStride});
        }
        if (rowsB > 0) {
            AscendC::Mul(prodB, x1[k * kLeaf], coeffB, kLeaf, static_cast<uint8_t>(rowsB),
                         {1, 1, 1, kLeafRepStride, 0, kLeafRepStride});
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (rowsA > 0) {
            AscendC::Sub(x0[(k + 1) * kLeaf], x0[(k + 1) * kLeaf], prodA, static_cast<int32_t>(rowsA * kLeaf));
        }
        if (rowsB > 0) {
            AscendC::Sub(x1[(k + 1) * kLeaf], x1[(k + 1) * kLeaf], prodB, static_cast<int32_t>(rowsB * kLeaf));
        }
        // 下一轮系数只依赖 lkk 与偏移表，与本次 Sub 无依赖。
        const uint32_t nextA = k + 2 < topRows ? topRows - k - 2 : 0;
        const uint32_t nextB = k + 2 < bottomRows ? bottomRows - k - 2 : 0;
        if (nextA > 0) {
            AscendC::Gather(coeffA, lkk, coeffOffset,
                            static_cast<uint32_t>((baseA + (rowStride + 1) * (k + 1)) * sizeof(float)), nextA * kLeaf);
        }
        if (nextB > 0) {
            AscendC::Gather(coeffB, lkk, coeffOffset,
                            static_cast<uint32_t>((baseB + (rowStride + 1) * (k + 1)) * sizeof(float)), nextB * kLeaf);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void V1Vf(AscendC::LocalTensor<bfloat16_t> qHat, AscendC::LocalTensor<bfloat16_t> kHat,
                            AscendC::LocalTensor<bfloat16_t> qPlus, AscendC::LocalTensor<bfloat16_t> kPlus,
                            AscendC::LocalTensor<bfloat16_t> kMinus, AscendC::LocalTensor<float> g,
                            AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> stageBuf,
                            AscendC::LocalTensor<float> bandDeltaBuf, uint32_t validRows)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    constexpr float base2Min = ExpDomain::kV1Bf16LowerBase2;
    constexpr float base2Max = ExpDomain::kV1Bf16UpperBase2;
    constexpr float clampMin = Domain::StoredBound(base2Min);
    constexpr float clampMax = Domain::StoredBound(base2Max);
    // 共享 scratch 8 KiB 切成两块 8 行 FP32：factor 存参考行差值取指后的
    // 指数因子，stage 存 BF16 输入展开的 FP32 落位。
    // beta gather 偏移表已移到 private 槽尾部，scratch 可整块使用；
    // 每块 8 行 FP32（factor + stage 各 4 KiB）。
    constexpr uint32_t kTileRows = 16;
    constexpr uint32_t kTileElems = kTileRows * Shape::kHeadDim;
    auto factor = scratch;
    auto stage = stageBuf;
    const uint32_t active = CeilDiv(validRows, Shape::kSubChunkRows);
    const uint32_t blockEnds[Shape::kSubChunkCount] = {
        validRows < 16 ? validRows : 16,
        validRows < 32 ? validRows : 32,
        validRows < 48 ? validRows : 48,
        validRows,
    };

    // Kplus 与 Khat 原位复用，所以必须先生成完四个 Kminus 前缀。
    // 否则后一个参考块会错误读取已经舍入成 Kplus 的数据。
    // 整块 chunk 时每个 band 的 [0,prefix) 行都会被下面的 tile 循环写满，
    // 只有尾 chunk 才需要预清零。
    if (validRows < Shape::kChunkRows) {
        for (uint32_t s = 0; s < Shape::kSubChunkCount; ++s) {
            auto kMinusBlock = kMinus[(Arch22Ub::kKMinus[s] - Arch22Ub::kGateOrKMinus) / sizeof(bfloat16_t)];
            AscendC::Duplicate(kMinusBlock, static_cast<bfloat16_t>(0), Shape::kPrefixRows[s] * Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    // Kminus 前缀：band s 的列范围 [0,prefix(s)) 与 band s-1 高度重叠
    // （160 行 vs 64 行唯一数据）。把「列 tile」放外层、「band」放内层：
    // 同一列 tile 只算一次 exp 链，后续 band 用 fp32 因子乘
    // exp(g[m_s]-g[m_{s-1}]) 递推。因子在 fp32 里递乘，最后仍只做一次
    // bf16 舍入，数值与直接计算只差 fp32 级误差。
    auto bandDelta = bandDeltaBuf;
    for (uint32_t s = 1; s < active; ++s) {
        const uint32_t prevBegin = (s - 1) * Shape::kSubChunkRows;
        const uint32_t prevMid = (prevBegin + blockEnds[s - 1]) / 2;
        const uint32_t mid = (s * Shape::kSubChunkRows + blockEnds[s]) / 2;
        auto delta = bandDelta[(s - 1) * Shape::kHeadDim];
        SubRowMinusTile<GateT, BetaT, CompilePolicy>(delta, g[mid * Shape::kHeadDim], g[prevMid * Shape::kHeadDim], 1);
        AscendC::PipeBarrier<PIPE_V>();
        // g 已是自然单位，直接 exp。
        AscendC::Exp(delta, delta, Shape::kHeadDim);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const uint8_t kFp32RowStride = static_cast<uint8_t>(Shape::kHeadDim * sizeof(float) / 32);
    const uint32_t tileCount = CeilDiv(validRows, kTileRows);
    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t rowBegin = tile * kTileRows;
        const uint32_t rows = validRows - rowBegin < kTileRows ? validRows - rowBegin : kTileRows;
        const uint32_t elems = rows * Shape::kHeadDim;
        for (uint32_t s = tile; s < active; ++s) {
            const uint32_t blockBegin = s * Shape::kSubChunkRows;
            const uint32_t blockEnd = blockEnds[s];
            const uint32_t midpoint = (blockBegin + blockEnd) / 2;
            auto kMinusBlock = kMinus[(Arch22Ub::kKMinus[s] - Arch22Ub::kGateOrKMinus) / sizeof(bfloat16_t)];
            if (s == tile) {
                SubRowMinusTile<GateT, BetaT, CompilePolicy>(factor, g[midpoint * Shape::kHeadDim],
                                                             g[rowBegin * Shape::kHeadDim], rows);
                AscendC::PipeBarrier<PIPE_V>();
                // g 已是自然单位。
                AscendC::Exp(factor, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                // factor *= exp(g[m_s]-g[m_{s-1}])，逐行广播同一个
                // 128-lane 向量。
                auto delta = bandDelta[(s - 1) * Shape::kHeadDim];
                AscendC::Mul(factor, delta, factor, 64, static_cast<uint8_t>(rows),
                             {1, 1, 1, kFp32RowStride, 0, kFp32RowStride});
                AscendC::Mul(factor[64], delta[64], factor[64], 64, static_cast<uint8_t>(rows),
                             {1, 1, 1, kFp32RowStride, 0, kFp32RowStride});
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Cast(stage, kHat[rowBegin * Shape::kHeadDim], AscendC::RoundMode::CAST_NONE, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kMinusBlock[rowBegin * Shape::kHeadDim], stage, AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // 四个 Kminus 都已完成，此时可以把 Qhat/Khat 原位改写为
    // Qplus/Kplus；无效行沿用 V0 写入的零。同样按 8 行一块下发。
    for (uint32_t s = 0; s < active; ++s) {
        const uint32_t blockBegin = s * Shape::kSubChunkRows;
        const uint32_t blockEnd = blockEnds[s];
        const uint32_t midpoint = (blockBegin + blockEnd) / 2;
        for (uint32_t rowBegin = blockBegin; rowBegin < blockEnd; rowBegin += kTileRows) {
            const uint32_t rows = blockEnd - rowBegin < kTileRows ? blockEnd - rowBegin : kTileRows;
            const uint32_t elems = rows * Shape::kHeadDim;
            SubTileMinusRow<GateT, BetaT, CompilePolicy>(factor, g[rowBegin * Shape::kHeadDim],
                                                         g[midpoint * Shape::kHeadDim], rows);
            AscendC::PipeBarrier<PIPE_V>();
            // g 已是自然单位，直接 Exp。
            AscendC::Exp(factor, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(stage, qHat[rowBegin * Shape::kHeadDim], AscendC::RoundMode::CAST_NONE, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(qPlus[rowBegin * Shape::kHeadDim], stage, AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(stage, kHat[rowBegin * Shape::kHeadDim], AscendC::RoundMode::CAST_NONE, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(stage, stage, factor, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(kPlus[rowBegin * Shape::kHeadDim], stage, AscendC::RoundMode::CAST_RINT, elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void V6Vf(AscendC::LocalTensor<bfloat16_t> qg, AscendC::LocalTensor<bfloat16_t> qgScaled,
                            AscendC::LocalTensor<bfloat16_t> kg, AscendC::LocalTensor<bfloat16_t> vBeta,
                            AscendC::LocalTensor<bfloat16_t> kBetaG, AscendC::LocalTensor<float> g,
                            AscendC::LocalTensor<float> betaEff, AscendC::LocalTensor<float> scratch,
                            AscendC::LocalTensor<float> stageBuf, uint32_t validRows, float scale,
                            AscendC::TEventID restReady)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    const uint32_t rhsRows = validRows > 32 ? 64 : 32;
    if (validRows == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(restReady);
        AscendC::Duplicate(qg, static_cast<bfloat16_t>(0), rhsRows * Shape::kHeadDim);
        AscendC::Duplicate(kg, static_cast<bfloat16_t>(0), rhsRows * Shape::kHeadDim);
        AscendC::Duplicate(kBetaG, static_cast<bfloat16_t>(0), rhsRows * Shape::kHeadDim);
        AscendC::Duplicate(vBeta, static_cast<bfloat16_t>(0), rhsRows * Shape::kValueDim);
        return;
    }
    const uint32_t last = (validRows - 1) * Shape::kHeadDim;
    constexpr float directMin = Domain::StoredBound(ExpDomain::kV6LowerBase2);
    constexpr float directMax = Domain::StoredBound(ExpDomain::kV6UpperBase2);
    // 同样的公式按整块（8 行）下发，逐行只影响行列步长参数，
    // 数学与舍入顺序与逐行版本完全一致。
    // 与 V1 相同：beta gather 偏移表已移出 scratch，每块 8 行 FP32。
    constexpr uint32_t kTileRows = 16;
    constexpr uint32_t kTileElems = kTileRows * Shape::kHeadDim;
    auto factor = scratch;
    auto stage = stageBuf;
    bool restWaited = false;
    for (uint32_t rowBegin = 0; rowBegin < validRows; rowBegin += kTileRows) {
        // S1：后 32 行的搬运与前 32 行的计算重叠，到这里才需要等。
        if (!restWaited && rowBegin >= 32) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(restReady);
            restWaited = true;
        }
        const uint32_t rows = validRows - rowBegin < kTileRows ? validRows - rowBegin : kTileRows;
        const uint32_t elems = rows * Shape::kHeadDim;
        const uint32_t offset = rowBegin * Shape::kHeadDim;
        // E(G)=exp(clamp(g[row]))：这里刻意不减 g[last]，与 arch35 的
        // V6 语义（qg/qBeta 用行内累计量，kg 才用 g[last]-g[row]）一致。
        AscendC::Exp(factor, g[offset], elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(stage, qg[offset], AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(stage, stage, factor, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(qg[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        // 第一次舍入：Khat*E(G) 先写入 kBetaG 的 BF16 存储。
        AscendC::Cast(stage, kg[offset], AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(stage, stage, factor, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(kBetaG[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        // E'(G)=exp(clamp(g[last]-g[row]))
        SubRowMinusTile<GateT, BetaT, CompilePolicy>(factor, g[last], g[offset], rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(factor, factor, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(stage, kg[offset], AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(stage, stage, factor, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(kg[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        // qgScaled 从已舍入的 BF16 qg 中间量回读。正序处理时，BF16 输出
        // 只覆盖已经消费完的 FP32 G 低地址，不会覆盖后续 G 行。
        AscendC::Cast(stage, qg[offset], AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(stage, stage, scale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(qgScaled[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (!restWaited) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(restReady);
    }
    // 逐行 beta 缩放：betaEff 先 Brcb 展开成“每行 1 个 32B block”，
    // 再用两条整块 Mul（低/高半行）完成 betaEff*round(Khat*E(G)) 与
    // betaEff*V。这两步必须在 G 全部消费完之后进行，才能借用 G 尾部
    // （BF16 qgScaled 只覆盖 G 低 validRows*256 字节）放展开结果。
    AscendC::Brcb(g[validRows * Shape::kHeadDim / 2], betaEff, static_cast<uint8_t>(Shape::kChunkRows / 8), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    auto betaBrcb = g[validRows * Shape::kHeadDim / 2];
    // 每块再把“每行 1 个 block”复制成“每行 8 个 block”，供 fp32 的
    // 64 lane repeat 使用（dataBlockStride 不支持 0）。
    auto betaRepeat = g[validRows * Shape::kHeadDim / 2 + 512];
    for (uint32_t rowBegin = 0; rowBegin < validRows; rowBegin += kTileRows) {
        const uint32_t rows = validRows - rowBegin < kTileRows ? validRows - rowBegin : kTileRows;
        const uint32_t elems = rows * Shape::kHeadDim;
        const uint32_t offset = rowBegin * Shape::kHeadDim;
        ReplicateRowScalarsToRepeat<GateT, BetaT, CompilePolicy>(betaRepeat, betaBrcb[8 * rowBegin], rows);
        AscendC::PipeBarrier<PIPE_V>();
        // 第二次舍入：betaEff*(round(Khat*E(G)))。
        AscendC::Cast(stage, kBetaG[offset], AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        MulTileByRowScalars<GateT, BetaT, CompilePolicy>(stage, stage, betaRepeat, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(kBetaG[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(stage, vBeta[offset], AscendC::RoundMode::CAST_NONE, elems);
        AscendC::PipeBarrier<PIPE_V>();
        MulTileByRowScalars<GateT, BetaT, CompilePolicy>(stage, stage, betaRepeat, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(vBeta[offset], stage, AscendC::RoundMode::CAST_RINT, elems);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // 有效行与补零行分开，VF 循环体内不做 runtime 分支。
    for (uint32_t row = validRows; row < rhsRows; ++row) {
        const uint32_t offset = row * Shape::kHeadDim;
        AscendC::Duplicate(qg[offset], static_cast<bfloat16_t>(0), Shape::kHeadDim);
        AscendC::Duplicate(kg[offset], static_cast<bfloat16_t>(0), Shape::kHeadDim);
        AscendC::Duplicate(kBetaG[offset], static_cast<bfloat16_t>(0), Shape::kHeadDim);
        AscendC::Duplicate(vBeta[row * Shape::kValueDim], static_cast<bfloat16_t>(0), Shape::kValueDim);
    }
}

template <typename GateT, typename BetaT, typename CompilePolicy>
struct TargetVectorContext {
    static_assert(CompilePolicy::arch22CubeLeaf, "The target pipeline requires the dedicated leaf policy");
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    static constexpr uint32_t kSlotStride =
        CompilePolicy::arch22CubeLeaf ? Workspace::kArch22TargetSlotStride : Workspace::kArch22BaseSlotStride;
    static constexpr uint32_t kWorkgroupStride = Workspace::kArch22SlotCount * kSlotStride;

    PrepareKernelArgs args_{};
    AscendC::TPipe *pipe_ = nullptr;
    uint32_t workgroup_ = 0;
    uint32_t aiv_ = 0;
    uint32_t coreCount_ = 0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_{};
    AscendC::TEventID ioFree_[2]{};
    AscendC::TEventID inputReady_[2]{};
    AscendC::TEventID restReady_[2]{};
    AscendC::TEventID outputReady_[2]{};
    AscendC::TEventID mte3ToV_[2]{};
    AscendC::TEventID scalarRead_{};
    AscendC::TEventID scalarWrite_{};
    AscendC::TEventID sharedFree_{};
    AscendC::GlobalTensor<bfloat16_t> qGm_{};
    AscendC::GlobalTensor<bfloat16_t> kGm_{};
    AscendC::GlobalTensor<bfloat16_t> vGm_{};
    AscendC::GlobalTensor<GateT> gateGm_{};
    AscendC::GlobalTensor<BetaT> betaGm_{};
    AscendC::GlobalTensor<float> dtBiasGm_{};
    AscendC::GlobalTensor<float> aLogGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgGm_{};
    AscendC::GlobalTensor<bfloat16_t> qgScaledGm_{};
    AscendC::GlobalTensor<bfloat16_t> kgGm_{};
    AscendC::GlobalTensor<float> gkGm_{};
    AscendC::GlobalTensor<bfloat16_t> aqkGm_{};
    AscendC::GlobalTensor<bfloat16_t> akkGm_{};
    AscendC::GlobalTensor<bfloat16_t> qHatGm_{};
    AscendC::GlobalTensor<bfloat16_t> kHatGm_{};
    AscendC::GlobalTensor<float> qRstdGm_{};
    AscendC::GlobalTensor<float> kRstdGm_{};
    AscendC::GlobalTensor<float> betaEffGm_{};
};

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void InitTargetVec(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                     const PrepareKernelArgs &args, AscendC::TPipe *pipe);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ProcessTargetVec(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline float ReadScalar(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                   AscendC::LocalTensor<float> tensor, uint32_t index);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ReleasePairEvents(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, uint32_t pair);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV0(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t valueHead, uint32_t localHead, uint32_t pair);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV1(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t localHead, uint32_t pair);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV3(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t valueHead, uint32_t localHead, uint32_t pair);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV6(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t valueHead, uint32_t localHead, uint32_t pair);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void L2NormRowsBlock(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                       AscendC::LocalTensor<bfloat16_t> src, AscendC::LocalTensor<float> rstdOut,
                                       AscendC::LocalTensor<float> staging, AscendC::LocalTensor<float> workspace,
                                       AscendC::LocalTensor<float> scaleBuf, uint32_t validRows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ScaleRowsByRstd(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                       AscendC::LocalTensor<bfloat16_t> src, AscendC::LocalTensor<float> rstd,
                                       AscendC::LocalTensor<float> staging, AscendC::LocalTensor<float> brcbScalars,
                                       AscendC::LocalTensor<float> repeatScalars, uint32_t validRows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void FillRstdFromSumBlocks(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                             AscendC::LocalTensor<float> rstdOut, AscendC::LocalTensor<float> sumBlocks,
                                             AscendC::LocalTensor<float> diagMask, AscendC::LocalTensor<float> prod,
                                             AscendC::LocalTensor<float> oneVec, uint32_t validRows);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void
V0Vf(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, AscendC::LocalTensor<bfloat16_t> q,
     AscendC::LocalTensor<bfloat16_t> k, AscendC::LocalTensor<float> qRstd, AscendC::LocalTensor<float> kRstd,
     AscendC::LocalTensor<GateT> gate, AscendC::LocalTensor<BetaT> beta, AscendC::LocalTensor<float> betaEff,
     AscendC::LocalTensor<float> dtBias, AscendC::LocalTensor<float> aLog, AscendC::LocalTensor<float> g,
     AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> scaleBuf, AscendC::LocalTensor<float> tileBuf,
     uint32_t validRows, AscendC::LocalTensor<float> sumQ, AscendC::LocalTensor<float> sumK, bool cubeSums);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void V3Vf(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, AscendC::LocalTensor<float> raw,
                            AscendC::LocalTensor<float> betaEff, AscendC::LocalTensor<bfloat16_t> aqk,
                            AscendC::LocalTensor<float> lkk, AscendC::LocalTensor<float> b,
                            AscendC::LocalTensor<float> x0, AscendC::LocalTensor<float> x1,
                            AscendC::LocalTensor<float> negX1, AscendC::LocalTensor<bfloat16_t> akkPack,
                            AscendC::LocalTensor<int32_t> coeffOffsetInt, uint32_t validRows, float scale);

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void InitTargetVec(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                     const PrepareKernelArgs &args, AscendC::TPipe *pipe)
{
    ctx.args_ = args;
    ctx.pipe_ = pipe;
    ctx.workgroup_ = WorkgroupId();
    ctx.aiv_ = AscendC::GetSubBlockIdx();
    ctx.coreCount_ = ctx.args_.tiling.usedCoreNum;
    ctx.qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.q));
    ctx.kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.k));
    ctx.vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.v));
    ctx.gateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(ctx.args_.rawGate));
    ctx.betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ BetaT *>(ctx.args_.beta));
    if (ctx.args_.dtBias != nullptr) {
        ctx.dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.dtBias));
    }
    if (ctx.args_.aLog != nullptr) {
        ctx.aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.aLog));
    }
    if constexpr (CompilePolicy::outputMode == OutputMode::Save) {
        ctx.qgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.qg));
    }
    ctx.qgScaledGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.qgScaled));
    ctx.kgGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.kg));
    ctx.gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.gk));
    ctx.aqkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.aqk));
    if constexpr (CompilePolicy::outputAkk) {
        ctx.akkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.akk));
    }
    if constexpr (CompilePolicy::outputRecomputeAux) {
        ctx.qHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.qHat));
    }
    if constexpr (CompilePolicy::outputRecomputeAux) {
        ctx.kHatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.kHat));
    }
    if constexpr (CompilePolicy::outputRecomputeAux) {
        ctx.qRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.qRstd));
    }
    if constexpr (CompilePolicy::outputRecomputeAux) {
        ctx.kRstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.kRstd));
    }
    if constexpr (CompilePolicy::outputRecomputeAux) {
        ctx.betaEffGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.betaEff));
    }
    if (ctx.coreCount_ == 0) {
        return;
    }
    ctx.pipe_->InitBuffer(ctx.ubBuf_, Arch22Ub::kUsableBytes);
    // 每种 HardEvent 有独立 ID 池。这里保留 Alloc/Release，让后续
    // 基础 API 能看到占用状态；注释给出 CANN 9.1 分配器的预期返回值。
    ctx.scalarRead_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::V_S>();    // ID 0
    ctx.scalarWrite_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::S_V>();   // ID 0
    ctx.sharedFree_ = ctx.pipe_->template AllocEventID<AscendC::HardEvent::V_MTE2>(); // ID 0
    // 两个 pair 分时复用共享 G/scratch；初始许可只发布一次。
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    ctx.ioFree_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE3_MTE2>();   // ID 0
    ctx.ioFree_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE3_MTE2>();   // ID 1
    ctx.inputReady_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_V>();  // ID 0
    ctx.inputReady_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_V>();  // ID 1
    ctx.restReady_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_V>();   // ID 2
    ctx.restReady_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE2_V>();   // ID 3
    ctx.outputReady_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::V_MTE3>(); // ID 0
    ctx.outputReady_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::V_MTE3>(); // ID 1
    ctx.mte3ToV_[0] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE3_V>();     // ID 0
    ctx.mte3ToV_[1] = ctx.pipe_->template AllocEventID<AscendC::HardEvent::MTE3_V>();     // ID 1
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[0]);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[1]);
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ProcessTargetVec(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx)
{
    if (ctx.coreCount_ == 0) {
        return;
    }
    // mode 0x2 的核间 flag 是固定物理编号，不经过 EventID 分配器。
    // pair0/pair1: ready=0/1，free=2/3。
    constexpr uint16_t kReadyFlagId[2] = {0, 1};
    constexpr uint16_t kFreeFlagId[2] = {2, 3};
    const uint32_t total = TargetTotalWorkItems(ctx.args_.tiling);
    const uint32_t workBegin = WorkBegin(total, ctx.workgroup_, ctx.coreCount_);
    const uint32_t workEnd = WorkEnd(total, ctx.workgroup_, ctx.coreCount_);
    if (ctx.workgroup_ >= ctx.coreCount_ || workBegin >= workEnd) {
        for (uint32_t pair = 0; pair < 2; ++pair) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
            ReleasePairEvents<GateT, BetaT, CompilePolicy>(ctx, pair);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::V_S>(ctx.scalarRead_);
        ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
        return;
    }
    for (uint32_t work = workBegin; work < workEnd; ++work) {
        uint32_t globalChunk = 0;
        uint32_t headPartition = 0;
        TargetDecodeWorkItem(ctx.args_.tiling, work, globalChunk, headPartition);
        ChunkRange chunk{};
        if (!ResolveChunk(ctx.args_, globalChunk, chunk)) {
            continue;
        }
        uint32_t headBegin = 0;
        uint32_t headEnd = 0;
        TargetHeadRange(ctx.args_.tiling, headPartition, headBegin, headEnd);
        for (uint32_t groupBegin = headBegin; groupBegin < headEnd;) {
            uint32_t activeHeads = headEnd - groupBegin;
            if (activeHeads > Shape::kHeadsPerGroup) {
                activeHeads = Shape::kHeadsPerGroup;
            }
            // AIV0 处理 0/2，AIV1 处理 1/3；两个 pair 分时复用共享 G 区。
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t localHead = pair * 2 + ctx.aiv_;
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kFreeFlagId[pair]);
                if (localHead < activeHeads) {
                    const uint32_t valueHead = groupBegin + localHead;
                    StageV0<GateT, BetaT, CompilePolicy>(ctx, chunk, valueHead, localHead, pair);
                    StageV1<GateT, BetaT, CompilePolicy>(ctx, chunk, localHead, pair);
                }
                // 尾部无任务的 AIV 仍参加集合，但不计算地址或访问 GM。
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kReadyFlagId[pair]);
            }
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t localHead = pair * 2 + ctx.aiv_;
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kFreeFlagId[pair]);
                if (localHead < activeHeads) {
                    const uint32_t valueHead = groupBegin + localHead;
                    StageV3<GateT, BetaT, CompilePolicy>(ctx, chunk, valueHead, localHead, pair);
                }
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kReadyFlagId[pair]);
            }
            for (uint32_t pair = 0; pair < 2; ++pair) {
                const uint32_t localHead = pair * 2 + ctx.aiv_;
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kFreeFlagId[pair]);
                if (localHead < activeHeads) {
                    const uint32_t valueHead = groupBegin + localHead;
                    StageV6<GateT, BetaT, CompilePolicy>(ctx, chunk, valueHead, localHead, pair);
                }
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kReadyFlagId[pair]);
            }
            groupBegin += activeHeads;
        }
    }
    // 消费最后一次 C7 发布，保证每次 set 都有对应 wait。
    for (uint32_t pair = 0; pair < 2; ++pair) {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(kFreeFlagId[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
        ReleasePairEvents<GateT, BetaT, CompilePolicy>(ctx, pair);
    }
    // 消费最后一轮 V6（或从未使用时的初始）共享区许可后再释放事件。
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::V_S>(ctx.scalarRead_);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline float ReadScalar(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                   AscendC::LocalTensor<float> tensor, uint32_t index)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(ctx.scalarRead_);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(ctx.scalarRead_);
    const float value = tensor.GetValue(index);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
    return value;
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ReleasePairEvents(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, uint32_t pair)
{
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_V>(ctx.inputReady_[pair]);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE2_V>(ctx.restReady_[pair]);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    ctx.pipe_->template ReleaseEventID<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV_[pair]);
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV0(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t valueHead, uint32_t localHead, uint32_t pair)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
    const uint32_t base = Arch22Ub::kPrivateBase[pair];
    auto ub = ctx.ubBuf_.template Get<uint8_t>();
    auto q = ub[base + Arch22Ub::kQ].template ReinterpretCast<bfloat16_t>();
    auto k = ub[base + Arch22Ub::kK].template ReinterpretCast<bfloat16_t>();
    const uint32_t gateBase = sizeof(GateT) == 4 ? Arch22Ub::kSharedG : base + Arch22Ub::kGateOrKMinus;
    auto gate = ub[gateBase].template ReinterpretCast<GateT>();
    auto betaStrided = ub[base + Arch22Ub::kBetaRawStrided].template ReinterpretCast<BetaT>();
    auto beta = ub[base + Arch22Ub::kBetaRaw].template ReinterpretCast<BetaT>();
    auto betaEff = ub[base + Arch22Ub::kBetaEff].template ReinterpretCast<float>();
    auto dtBias = ub[base + Arch22Ub::kDtBias].template ReinterpretCast<float>();
    auto aLog = ub[base + Arch22Ub::kALog].template ReinterpretCast<float>();
    auto qRstd = ub[base + Arch22Ub::kQRstd].template ReinterpretCast<float>();
    auto kRstd = ub[base + Arch22Ub::kKRstd].template ReinterpretCast<float>();
    auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
    auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
    const uint32_t qkHead = QkHeadForValueHead(ctx.args_.tiling, valueHead);
    const uint64_t qkOffset = QkInputOffset(ctx.args_.tiling, chunk, qkHead);
    const uint64_t gateOffset = RawGateInputOffset(ctx.args_.tiling, chunk, valueHead);
    const uint64_t betaOffset = BetaInputOffset(ctx.args_.tiling, chunk, valueHead);
    const uint64_t headOutputOffset = HeadTensorOffset(ctx.args_.tiling, chunk, valueHead, Shape::kHeadDim);
    const uint32_t qkStride = ctx.args_.tiling.inputSequenceMajor ?
                                  static_cast<uint32_t>(static_cast<uint64_t>(ctx.args_.tiling.qkHeadNum - 1) *
                                                        Shape::kHeadDim * sizeof(bfloat16_t)) :
                                  0;
    const uint32_t gateStride = ctx.args_.tiling.inputSequenceMajor ?
                                    static_cast<uint32_t>(static_cast<uint64_t>(ctx.args_.tiling.valueHeadNum - 1) *
                                                          Shape::kHeadDim * sizeof(GateT)) :
                                    0;
    // 获得共享 G/scratch 的独占许可；V1 完成最后一次 V 读取后归还。
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    AscendC::DataCopyExtParams qkCopy{static_cast<uint16_t>(chunk.validRows),
                                      static_cast<uint32_t>(Shape::kHeadDim * sizeof(bfloat16_t)), qkStride, 0, 0};
    AscendC::DataCopyPadExtParams<bfloat16_t> qkPad{false, 0, 0, 0};
    AscendC::DataCopyPad(q, ctx.qGm_[qkOffset], qkCopy, qkPad);
    AscendC::DataCopyPad(k, ctx.kGm_[qkOffset], qkCopy, qkPad);
    AscendC::DataCopyExtParams gateCopy{static_cast<uint16_t>(chunk.validRows),
                                        static_cast<uint32_t>(Shape::kHeadDim * sizeof(GateT)), gateStride, 0, 0};
    AscendC::DataCopyPadExtParams<GateT> gatePad{false, 0, 0, 0};
    AscendC::DataCopyPad(gate, ctx.gateGm_[gateOffset], gateCopy, gatePad);
    AscendC::DataCopyPadExtParams<BetaT> betaPad{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> fp32Pad{false, 0, 0, 0};
    if (ctx.args_.tiling.inputSequenceMajor) {
        const uint32_t betaStride =
            static_cast<uint32_t>(static_cast<uint64_t>(ctx.args_.tiling.valueHeadNum - 1) * sizeof(BetaT));
        AscendC::DataCopyPad(betaStrided, ctx.betaGm_[betaOffset],
                             AscendC::DataCopyExtParams{static_cast<uint16_t>(chunk.validRows),
                                                        static_cast<uint32_t>(sizeof(BetaT)), betaStride, 0, 0},
                             betaPad);
    } else {
        AscendC::DataCopyPad(
            beta, ctx.betaGm_[betaOffset],
            AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(BetaT)), 0, 0, 0}, betaPad);
    }
    if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
        if (ctx.args_.tiling.hasDtBias) {
            // 公开接口将 dt_bias 固定为展平的 [HV,K]。
            AscendC::DataCopyPad(
                dtBias, ctx.dtBiasGm_[valueHead * Shape::kHeadDim],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(Shape::kHeadDim * sizeof(float)), 0, 0, 0},
                fp32Pad);
        }
        if (ctx.args_.aLog != nullptr) {
            // 公开接口将 a_log 固定为 [HV]。
            AscendC::DataCopyPad(aLog, ctx.aLogGm_[valueHead],
                                 AscendC::DataCopyExtParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0}, fp32Pad);
        }
    }
    // Gram relay 占 16 KiB，必须避开尚未消费的 gate。FP32 gate 在共享 G，
    // BF16 gate 在私有 kGateOrKMinus，因此按 dtype 选择另一侧的 norm staging。
    // V0 先提取 q/k 两组 rstd，再复用该区进行行缩放；无需增加 UB。
    // 条件必须与 AIC StageNormSums 一致，否则读到的是上一组残留。
    const uint32_t normSumBase = sizeof(GateT) == sizeof(float) ?
                                     base + Arch22Ub::kGateOrKMinus : Arch22Ub::kSharedG;
    auto sumBufQ = ub[normSumBase].template ReinterpretCast<float>();
    auto sumBufK = sumBufQ[2048];
    bool cubeSums = false;
    if constexpr (CompilePolicy::normMode == QkNormMode::L2) {
        cubeSums = chunk.validRows == Shape::kChunkRows && !ctx.args_.tiling.inputSequenceMajor;
        if (cubeSums) {
            const uint64_t v0Slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
            AscendC::GlobalTensor<float> normRelayQ;
            normRelayQ.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(ctx.args_.workspace + v0Slot + Workspace::kArch22NormSum), 2048);
            AscendC::GlobalTensor<float> normRelayK;
            normRelayK.SetGlobalBuffer(
                reinterpret_cast<__gm__ float *>(ctx.args_.workspace + v0Slot + Workspace::kArch22NormSum + 0x2000),
                2048);
            AscendC::DataCopy(sumBufQ, normRelayQ, 2048);
            AscendC::DataCopy(sumBufK, normRelayK, 2048);
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.inputReady_[pair]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.inputReady_[pair]);
    if (ctx.args_.tiling.inputSequenceMajor) {
        // 偏移表放在本 pair 的 private 尾部，V1 的 Kminus 会覆盖它，
        // 因此每个 work 的 StageV0 都重写一次；只影响 sequence-major。
        auto offsets = ub[base + Arch22Ub::kBetaGatherOffsetsPrivate].template ReinterpretCast<uint32_t>();
        for (uint32_t row = 0; row < Shape::kChunkRows; ++row) {
            offsets.SetValue(row, row * 32U);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
        AscendC::Gather(beta, betaStrided, offsets, static_cast<uint32_t>(0), chunk.validRows);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // 本 Stage 的一次向量计算完成 Q/K 可选 L2 norm、beta 变换、
    // gate 变换和逐 token cumsum，并生成后续 Stage 所需中间量。
    // tileBuf 复用 kMinus 所在的私有区（V1 之前未被使用；BF16 gate
    // 场景下 gate 自身也在此区，但已整体 Cast 到 g，可以安全复用）。
    V0Vf<GateT, BetaT, CompilePolicy>(ctx, q, k, qRstd, kRstd, gate, beta, betaEff, dtBias, aLog, g, scratch,
                                      ub[Arch22Ub::kScratchTail].template ReinterpretCast<float>(),
                                      ub[base + Arch22Ub::kGateOrKMinus].template ReinterpretCast<float>(),
                                      chunk.validRows, sumBufQ, sumBufK, cubeSums);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
    AscendC::GlobalTensor<bfloat16_t> qhatContext;
    AscendC::GlobalTensor<bfloat16_t> khatContext;
    AscendC::GlobalTensor<float> betaContext;
    qhatContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kQHat));
    khatContext.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kKHat));
    betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kBetaEff));
    AscendC::DataCopy(qhatContext, q, chunk.validRows * Shape::kHeadDim);
    AscendC::DataCopy(khatContext, k, chunk.validRows * Shape::kHeadDim);
    AscendC::DataCopyPad(
        betaContext, betaEff,
        AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0});
    // Q/K 保存量按 HK 写回。GVA 中只有 QK 头组的第一个 HV 是 owner，
    // 其余 HV 仍保留各自 workspace context，供本 kernel 的 V1/V6 使用。
    if (IsQkOutputOwner(ctx.args_.tiling, valueHead)) {
        const uint64_t qkOutputOffset = QkHeadTensorOffset(ctx.args_.tiling, chunk, qkHead, Shape::kHeadDim);
        const uint64_t rstdOutputOffset = QkHeadScalarOffset(ctx.args_.tiling, chunk, qkHead);
        if constexpr (CompilePolicy::outputRecomputeAux) {
            AscendC::DataCopy(ctx.qHatGm_[qkOutputOffset], q, chunk.validRows * Shape::kHeadDim);
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            AscendC::DataCopy(ctx.kHatGm_[qkOutputOffset], k, chunk.validRows * Shape::kHeadDim);
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            AscendC::DataCopyPad(
                ctx.qRstdGm_[rstdOutputOffset], qRstd,
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0});
        }
        if constexpr (CompilePolicy::outputRecomputeAux) {
            AscendC::DataCopyPad(
                ctx.kRstdGm_[rstdOutputOffset], kRstd,
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0});
        }
    }
    if constexpr (CompilePolicy::outputRecomputeAux) {
        AscendC::DataCopyPad(
            ctx.betaEffGm_[HeadScalarOffset(ctx.args_.tiling, chunk, valueHead)], betaEff,
            AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0});
    }
    // g 内部按自然对数单位存放；gk 输出契约仍是 log2 域，这里换算一次。
    if constexpr (Domain::useExp2) {
        auto gkStage = ub[base + Arch22Ub::kGateOrKMinus].template ReinterpretCast<float>();
        const uint32_t gkCount = chunk.validRows * Shape::kHeadDim;
        AscendC::Muls(gkStage, g, Domain::stepScale, gkCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
        AscendC::DataCopy(ctx.gkGm_[headOutputOffset], gkStage, gkCount);
    } else {
        AscendC::DataCopy(ctx.gkGm_[headOutputOffset], g, chunk.validRows * Shape::kHeadDim);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV_[pair]);
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV1(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t localHead, uint32_t pair)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV_[pair]);
    const uint32_t base = Arch22Ub::kPrivateBase[pair];
    auto ub = ctx.ubBuf_.template Get<uint8_t>();
    auto qHat = ub[base + Arch22Ub::kQ].template ReinterpretCast<bfloat16_t>();
    auto kHat = ub[base + Arch22Ub::kK].template ReinterpretCast<bfloat16_t>();
    auto qPlus = ub[base + Arch22Ub::kQ].template ReinterpretCast<bfloat16_t>();
    auto kPlus = ub[base + Arch22Ub::kK].template ReinterpretCast<bfloat16_t>();
    auto kMinus = ub[base + Arch22Ub::kGateOrKMinus].template ReinterpretCast<bfloat16_t>();
    auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
    auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
    auto scratchTail = ub[Arch22Ub::kScratchTail].template ReinterpretCast<float>();
    auto bandDelta = ub[base + Arch22Ub::kBandDelta].template ReinterpretCast<float>();
    // 唯一一次 VF：按四个 16 行中点广播 Gref，生成 Qplus/Kplus 和
    // 16/32/48/64 行 Kminus；Gref 不物化为矩阵。
    // SIMD 路径按 USE_EXP2 选择等价截断域；仅 log2 分支乘 ln(2)，
    // 两个分支最终都调用自然底 Exp。
    V1Vf<GateT, BetaT, CompilePolicy>(qHat, kHat, qPlus, kPlus, kMinus, g, scratch, scratchTail, bandDelta,
                                      chunk.validRows);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    AscendC::GlobalTensor<bfloat16_t> payload;
    const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
    payload.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kPayload));
    AscendC::DataCopy(payload, qPlus, Shape::kScorePayloadBytes / sizeof(bfloat16_t));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV3(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t valueHead, uint32_t localHead, uint32_t pair)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
    const uint32_t base = Arch22Ub::kPrivateBase[pair];
    auto ub = ctx.ubBuf_.template Get<uint8_t>();
    auto raw = ub[base + Arch22Ub::kV3CompactRaw].template ReinterpretCast<float>();
    auto aqk = ub[base + Arch22Ub::kV3Aqk].template ReinterpretCast<bfloat16_t>();
    auto lkk = ub[base + Arch22Ub::kV3Lkk].template ReinterpretCast<float>();
    auto b = ub[base + Arch22Ub::kV3B].template ReinterpretCast<float>();
    auto x0 = ub[base + Arch22Ub::kV3X0].template ReinterpretCast<float>();
    auto x1 = ub[base + Arch22Ub::kV3X1].template ReinterpretCast<float>();
    auto negX1 = ub[base + Arch22Ub::kV3NegX1].template ReinterpretCast<float>();
    auto betaEff = ub[Arch22Ub::kV3BetaEff].template ReinterpretCast<float>();
    auto akkPack = ub[base + Arch22Ub::kV3AkkPack].template ReinterpretCast<bfloat16_t>();
    const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
    AscendC::GlobalTensor<float> betaContext;
    betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kBetaEff));
    AscendC::GlobalTensor<float> invAux;
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        invAux.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kArch22InvAux),
                               Workspace::kArch22InvAuxBytes / sizeof(float));
    }
    const uint32_t active = CeilDiv(chunk.validRows, Shape::kSubChunkRows);
    uint32_t compactElements = 0;
    for (uint32_t s = 0; s < active; ++s) {
        const uint32_t remaining = chunk.validRows - s * Shape::kSubChunkRows;
        const uint32_t rows = remaining < Shape::kSubChunkRows ? remaining : Shape::kSubChunkRows;
        const uint32_t bandElements = 2 * rows * Shape::kPrefixRows[s];
        compactElements += bandElements;
    }
    AscendC::GlobalTensor<float> rawScoreRelay;
    rawScoreRelay.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kArch22CubeRelay),
        Workspace::kArch22RawScoreBytes / sizeof(float));
    // C2 按四段连续写入独占 relay；尾 band 只包含有效行。
    AscendC::DataCopy(raw, rawScoreRelay, compactElements);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(betaEff, betaContext,
                         AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0},
                         pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.inputReady_[pair]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.inputReady_[pair]);
    // 本 Stage 一次完成因果 mask、scale、beta 和两个 32x32 叶逆，
    // 生成 Aqk/B/X0/X1/negX1/稳定 Akk；negX1 供 C5 做普通 Mmad。
    // Leaf0 区在 V3 内不被其它 buffer 使用，借给 Gather 的偏移表。
    auto coeffOffsetInt = ub[base + Arch22Ub::kV3Leaf0].template ReinterpretCast<int32_t>();
    V3Vf<GateT, BetaT, CompilePolicy>(ctx, raw, betaEff, aqk, lkk, b, x0, x1, negX1, akkPack, coeffOffsetInt,
                                      chunk.validRows, ctx.args_.tiling.scale);
    auto zeroAkk = akkPack;
    auto packedDiag = coeffOffsetInt.template ReinterpretCast<float>();
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        AscendC::Duplicate(zeroAkk, static_cast<bfloat16_t>(0), 4096);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t topRows = chunk.validRows < 32 ? chunk.validRows : 32;
        const uint32_t bottomRows = chunk.validRows > 32 ? chunk.validRows - 32 : 0;
        // Preserve one identity tile for the AIC recurrence, then form
        // Y00=I-L00 and Y11=I-L11 in-place.
        // x0 is already a complete 32x32 identity here. Copy it as the
        // recurrence's constant factor instead of forming I+I = 2I.
        AscendC::Adds(negX1, x0, 0.0F, 1024);
        // The identity copy still reads X0; finish it before forming I-L00.
        AscendC::PipeBarrier<PIPE_V>();
        constexpr uint8_t kFp32TileStride = 4; // 32 FP32 = 4 blocks
        constexpr uint8_t kFp32LkkStride = 8;  // 64 FP32 = 8 blocks
        if (topRows != 0) {
            AscendC::Sub(x0, x0, lkk, 32, static_cast<uint8_t>(topRows),
                         {1, 1, 1, kFp32TileStride, kFp32TileStride, kFp32LkkStride});
        }
        if (bottomRows != 0) {
            AscendC::Sub(x1, x1, lkk[32 * Shape::kChunkRows + 32], 32, static_cast<uint8_t>(bottomRows),
                         {1, 1, 1, kFp32TileStride, kFp32TileStride, kFp32LkkStride});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    AscendC::DataCopy(ctx.aqkGm_[AOutputOffset(ctx.args_.tiling, chunk, valueHead)], aqk,
                      chunk.validRows * Shape::kChunkRows);
    // C4 固定读取完整 64x64 矩阵，因此补零后的中转矩阵始终写入
    // 工作空间；公开 Akk 只有 T 行，尾 chunk 只能写有效行。
    AscendC::GlobalTensor<bfloat16_t> akkRelay;
    akkRelay.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kPayload + Workspace::kAkk));
    constexpr bool readFullAkkFromOutput = CompilePolicy::arch22CubeLeaf && CompilePolicy::outputAkk;
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        if (readFullAkkFromOutput && chunk.validRows == Shape::kChunkRows) {
            // C4 reads the final public Akk for full model chunks.
        } else {
            AscendC::DataCopy(akkRelay, zeroAkk, Shape::kChunkRows * Shape::kChunkRows);
        }
    } else {
        AscendC::DataCopy(akkRelay, akkPack, Shape::kChunkRows * Shape::kChunkRows);
    }
    if constexpr (CompilePolicy::outputAkk) {
        const uint64_t outputOffset = AOutputOffset(ctx.args_.tiling, chunk, valueHead);
        const uint32_t topRows = chunk.validRows < 32 ? chunk.validRows : 32;
        if constexpr (CompilePolicy::arch22CubeLeaf) {
            AscendC::DataCopyPad(
                ctx.akkGm_[outputOffset], zeroAkk,
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(topRows * Shape::kChunkRows * sizeof(bfloat16_t)),
                                           0, 0, 0});
        } else {
            AscendC::DataCopyPad(
                ctx.akkGm_[outputOffset], akkPack,
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(topRows * Shape::kChunkRows * sizeof(bfloat16_t)),
                                           0, 0, 0});
        }
        if (chunk.validRows > 32) {
            const uint32_t bottomRows = chunk.validRows - 32;
            constexpr uint32_t kBottomRightOffset = 32 * Shape::kChunkRows + 32;
            if constexpr (CompilePolicy::arch22CubeLeaf) {
                AscendC::DataCopyPad(
                    ctx.akkGm_[outputOffset + kBottomRightOffset], zeroAkk[kBottomRightOffset],
                    AscendC::DataCopyExtParams{static_cast<uint16_t>(bottomRows), 32 * sizeof(bfloat16_t), 2, 64, 0});
            } else {
                AscendC::DataCopyPad(
                    ctx.akkGm_[outputOffset + kBottomRightOffset], akkPack[kBottomRightOffset],
                    AscendC::DataCopyExtParams{static_cast<uint16_t>(bottomRows), 32 * sizeof(bfloat16_t), 2, 64, 0});
            }
        }
    }
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        const uint32_t topRows = chunk.validRows < 32 ? chunk.validRows : 32;
        const uint32_t bottomRows = chunk.validRows > 32 ? chunk.validRows - 32 : 0;
        AscendC::DataCopy(invAux[Workspace::kArch22InvL00 / sizeof(float)], lkk, 2048);
        AscendC::DataCopy(invAux[Workspace::kArch22InvY00 / sizeof(float)], x0, static_cast<uint32_t>(topRows * 32));
        if (topRows < 32) {
            const uint32_t tail = (32 - topRows) * 32;
            AscendC::DataCopy(invAux[Workspace::kArch22InvY00 / sizeof(float) + topRows * 32], x0[topRows * 32], tail);
        }
        AscendC::DataCopy(invAux[Workspace::kArch22InvL11 / sizeof(float)], lkk[32 * Shape::kChunkRows], 2048);
        AscendC::DataCopy(invAux[Workspace::kArch22InvY11 / sizeof(float)], x1, static_cast<uint32_t>(bottomRows * 32));
        if (bottomRows < 32) {
            const uint32_t tail = (32 - bottomRows) * 32;
            AscendC::DataCopy(invAux[Workspace::kArch22InvY11 / sizeof(float) + bottomRows * 32], x1[bottomRows * 32],
                              tail);
        }
        AscendC::DataCopy(invAux[Workspace::kArch22InvIdentity / sizeof(float)], negX1, 1024);
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(invAux);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }
    if (chunk.validRows > 32) {
        AscendC::GlobalTensor<float> payload;
        payload.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kPayload));
        if constexpr (!CompilePolicy::arch22CubeLeaf) {
            AscendC::DataCopy(payload[Workspace::kX0 / sizeof(float)], x0, 1024);
            AscendC::DataCopy(payload[Workspace::kNegX1 / sizeof(float)], negX1, 1024);
        }
        if constexpr (CompilePolicy::arch22CubeLeaf) {
            // C5 consumes negX1 @ T. Publish X1 positively for Akk, and
            // negate B before C4 so its T becomes -T for that final MMAD.
            AscendC::Muls(b, b, -1.0F, 1024);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
        }
        AscendC::DataCopy(payload[Workspace::kB / sizeof(float)], b, 1024);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void StageV6(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, const ChunkRange &chunk,
                               uint32_t valueHead, uint32_t localHead, uint32_t pair)
{
    using Domain = ExpDomainTraits<CompilePolicy::useExp2>;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
    const uint32_t base = Arch22Ub::kPrivateBase[pair];
    auto ub = ctx.ubBuf_.template Get<uint8_t>();
    auto qg = ub[base + Arch22Ub::kV6Qg].template ReinterpretCast<bfloat16_t>();
    auto qgScaled = ub[Arch22Ub::kV6QgScaled].template ReinterpretCast<bfloat16_t>();
    auto kg = ub[base + Arch22Ub::kV6Kg].template ReinterpretCast<bfloat16_t>();
    auto vBeta = ub[base + Arch22Ub::kV6VBeta].template ReinterpretCast<bfloat16_t>();
    auto kBetaG = ub[base + Arch22Ub::kV6KBetaG].template ReinterpretCast<bfloat16_t>();
    auto g = ub[Arch22Ub::kSharedG].template ReinterpretCast<float>();
    auto scratch = ub[Arch22Ub::kSharedScratch].template ReinterpretCast<float>();
    auto scratchTail = ub[Arch22Ub::kScratchTail].template ReinterpretCast<float>();
    auto betaEff = ub[base + Arch22Ub::kBetaEff].template ReinterpretCast<float>();
    const uint64_t slot = WorkspaceSlotBase(ctx.workgroup_, localHead, ctx.kWorkgroupStride, ctx.kSlotStride);
    AscendC::GlobalTensor<bfloat16_t> qhat;
    AscendC::GlobalTensor<bfloat16_t> khat;
    AscendC::GlobalTensor<float> betaContext;
    qhat.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kQHat));
    khat.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kKHat));
    betaContext.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx.args_.workspace + slot + Workspace::kBetaEff));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
    // S1：tile 级 ping-pong。先搬前 32 行（外加 g 的最后一行，供
    // exp(g_last-g[row]) 使用），向量侧先开跑，后半搬运躲在计算里。
    const uint64_t gBase = HeadTensorOffset(ctx.args_.tiling, chunk, valueHead, Shape::kHeadDim);
    const uint32_t firstRows = chunk.validRows > 32 ? 32 : chunk.validRows;
    AscendC::DataCopy(qg, qhat, firstRows * Shape::kHeadDim);
    AscendC::DataCopy(kg, khat, firstRows * Shape::kHeadDim);
    // g 整块留在第一批：它只有 32 KiB，而且是所有 tile 共用的参考量
    // （exp(g[last]-g[row])），拆开反而要在 tile 里补缩放。
    AscendC::DataCopy(g, ctx.gkGm_[gBase], chunk.validRows * Shape::kHeadDim);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.inputReady_[pair]);
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(betaEff, betaContext,
                         AscendC::DataCopyExtParams{1, static_cast<uint32_t>(chunk.validRows * sizeof(float)), 0, 0, 0},
                         pad);
    if (chunk.validRows > 32) {
        const uint32_t restRows = chunk.validRows - 32;
        AscendC::DataCopy(qg[32 * Shape::kHeadDim], qhat[32 * Shape::kHeadDim], restRows * Shape::kHeadDim);
        AscendC::DataCopy(kg[32 * Shape::kHeadDim], khat[32 * Shape::kHeadDim], restRows * Shape::kHeadDim);
    }
    const uint32_t vStride = ctx.args_.tiling.inputSequenceMajor ?
                                 static_cast<uint32_t>(static_cast<uint64_t>(ctx.args_.tiling.valueHeadNum - 1) *
                                                       Shape::kValueDim * sizeof(bfloat16_t)) :
                                 0;
    AscendC::DataCopyPad(vBeta, ctx.vGm_[ValueInputOffset(ctx.args_.tiling, chunk, valueHead)],
                         AscendC::DataCopyExtParams{static_cast<uint16_t>(chunk.validRows),
                                                    static_cast<uint32_t>(Shape::kValueDim * sizeof(bfloat16_t)),
                                                    vStride, 0, 0},
                         AscendC::DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ctx.restReady_[pair]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ctx.inputReady_[pair]);
    if constexpr (Domain::useExp2) {
        // gk 从 GM 取回的是 log2 域（公开契约），内部统一转成自然对数，
        // 这样 V6 的 8 个 tile 都省掉一次 ×ln2。
        AscendC::Muls(g, g, Domain::expInputScale, chunk.validRows * Shape::kHeadDim);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // 本 Stage 一次生成 qg、qgScaled、kg、两次舍入的 K_beta_g 和
    // V_beta；两条编译路径先在对应域截断，再统一调用自然底 Exp。
    V6Vf<GateT, BetaT, CompilePolicy>(qg, qgScaled, kg, vBeta, kBetaG, g, betaEff, scratch, scratchTail,
                                      chunk.validRows, ctx.args_.tiling.scale, ctx.restReady_[pair]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ctx.outputReady_[pair]);
    const uint64_t out = HeadTensorOffset(ctx.args_.tiling, chunk, valueHead, Shape::kHeadDim);
    // 共享区只有 qgScaled 仍被 MTE3 读取，优先搬出并单独发布完成事件；
    // 后续私有区输出可继续在 MTE3 流水中执行。
    AscendC::DataCopy(ctx.qgScaledGm_[out], qgScaled, chunk.validRows * Shape::kHeadDim);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV_[pair]);
    if constexpr (CompilePolicy::outputMode == OutputMode::Save) {
        AscendC::DataCopy(ctx.qgGm_[out], qg, chunk.validRows * Shape::kHeadDim);
    }
    AscendC::DataCopy(ctx.kgGm_[out], kg, chunk.validRows * Shape::kHeadDim);
    const uint32_t rhsRows = chunk.validRows > 32 ? 64 : 32;
    AscendC::GlobalTensor<bfloat16_t> kBetaRelay;
    AscendC::GlobalTensor<bfloat16_t> vBetaRelay;
    kBetaRelay.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kArch22RhsKBetaG));
    vBetaRelay.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(ctx.args_.workspace + slot + Workspace::kArch22RhsVBeta));
    AscendC::DataCopy(kBetaRelay, kBetaG, rhsRows * Shape::kHeadDim);
    AscendC::DataCopy(vBetaRelay, vBeta, rhsRows * Shape::kValueDim);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ctx.ioFree_[pair]);
    // qgScaled 与共享 G 复用地址；必须等 MTE3 读完，才能把共享区
    // 通过 V_MTE2 许可交给下一个 pair 的 MTE2。
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ctx.mte3ToV_[pair]);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ctx.sharedFree_);
}

// 块状行 L2 归一化：src（BF16 [validRows,128]）原地改写为 BF16(x*rstd)，
// 同时把 rstd 写进 rstdOut（FP32 [validRows]）。
//
// 逐行版本每个 (chunk,head) 要下发 64x2 行 x ~20 条指令、~512 个
// PipeBarrier 和 128 次 V_S/S_V 标量往返；234 实测这一项占整算子 45.5%
// （case 63：8,620 / 18,949.7 us）。块状版本把同样的数学压成：
//   Cast 整块 -> 平方 -> 折半 Add + WholeReduceSum 出行和 ->
//   标量压紧 -> 块状 eps/Sqrt/Div -> 标量读一次 -> 逐行 Muls/Cast
// 折半 + WholeReduceSum 的写法与本仓 arch22 已上量的
// recurrent_gated_delta_rule.h::ReduceSumAddFoldK128 保持一致。
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void L2NormRowsBlock(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                       AscendC::LocalTensor<bfloat16_t> src, AscendC::LocalTensor<float> rstdOut,
                                       AscendC::LocalTensor<float> staging, AscendC::LocalTensor<float> workspace,
                                       AscendC::LocalTensor<float> scaleBuf, uint32_t validRows)
{
    constexpr uint32_t kRepeatLanes = 64;                // 一个 vector repeat
    constexpr uint32_t kRowBlocks = Shape::kHeadDim / 8; // 16 个 32B block/行
    // 块状链的准入行数：<32 行的尾块退回逐行实现（见下方注释）。
    constexpr uint32_t kBlockNormMinRows = 32;
    auto sumBlocks = workspace; // 每行 1 个 block
    auto rowSum = workspace[Shape::kChunkRows * 8];
    auto oneVec = workspace[Shape::kChunkRows * 9];
    const uint32_t count = validRows * Shape::kHeadDim;
    const int32_t rows = static_cast<int32_t>(validRows);
    const int32_t rowStrideBlocks = static_cast<int32_t>(kRowBlocks);

    // 尾块很小（实测 validRows=1 时块状链拿到的 scale 恒为 1.0，rstd 本身
    // 却是对的）⇒ 这种尾块退回改动前的逐行实现，逐位与基线一致。
    if (validRows < kBlockNormMinRows) {
        uint32_t reduceShape[2] = {1, Shape::kHeadDim};
        auto rowStage = workspace;
        auto rowSq = workspace[Shape::kHeadDim];
        auto rowRstd = workspace[2 * Shape::kHeadDim];
        auto rowOne = workspace[3 * Shape::kHeadDim];
        // level-2 ReduceSum 的临时区参数要求 uint8_t 视图（与改动前一致）。
        auto rowWork = workspace[4 * Shape::kHeadDim].template ReinterpretCast<uint8_t>();
        for (uint32_t row = 0; row < validRows; ++row) {
            AscendC::Cast(rowStage, src[row * Shape::kHeadDim], AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(rowSq, rowStage, rowStage, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(rowRstd, rowSq, rowWork, reduceShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(rowRstd, rowRstd, ctx.args_.tiling.epsilon, 1);
            AscendC::Sqrt(rowRstd, rowRstd, 1);
            AscendC::Duplicate(rowOne, 1.0F, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(rowRstd, rowOne, rowRstd, 1);
            // 标量取一次 scale：dst 必须是 32B 对齐的临时区，rstdOut[row]
            // 只有 4B 对齐，直接当向量 dst 会触发 aicore exception。
            AscendC::SetFlag<AscendC::HardEvent::V_S>(ctx.scalarRead_);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(ctx.scalarRead_);
            const float rowScale = rowRstd.GetValue(0);
            rstdOut.SetValue(row, rowScale);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
            AscendC::Cast(rowStage, src[row * Shape::kHeadDim], AscendC::RoundMode::CAST_NONE, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(rowStage, rowStage, rowScale, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(src[row * Shape::kHeadDim], rowStage, AscendC::RoundMode::CAST_RINT, Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        return;
    }

    AscendC::Cast(staging, src, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(staging, staging, staging, count);
    AscendC::PipeBarrier<PIPE_V>();
    // 列方向折半：把每行 [64,128) 的和写回该行的 [64,128)。
    AscendC::Add(staging[kRepeatLanes], staging, staging[kRepeatLanes], static_cast<uint64_t>(kRepeatLanes), rows,
                 {1, 1, 1, static_cast<uint8_t>(rowStrideBlocks), static_cast<uint8_t>(rowStrideBlocks),
                  static_cast<uint8_t>(rowStrideBlocks)});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(sumBlocks, staging[kRepeatLanes], static_cast<int32_t>(kRepeatLanes), rows, 1, 1,
                            rowStrideBlocks);
    AscendC::PipeBarrier<PIPE_V>();
    // WholeReduceSum 的结果在 dst 上**连续**落位（dst[repeat]，实测确认：
    // 若按 1 block/repeat 去读，第 i 行会读到第 8i 行的和），这里直接取
    // 第 row 个元素；随后用块指令完成 eps -> Sqrt -> 倒数。
    // P3-A：WholeReduceSum 的结果在 dst 上连续落位 ⇒ 原地做
    // eps→Sqrt→Div，省掉 64 次 GetValue/SetValue 与一次 V_S/S_V 往返。
    AscendC::Adds(sumBlocks, sumBlocks, ctx.args_.tiling.epsilon, validRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sqrt(sumBlocks, sumBlocks, validRows);
    AscendC::Duplicate(oneVec, 1.0F, validRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(rstdOut, oneVec, sumBlocks, validRows);
    // 折半 Add 覆盖了 staging 的后半列，ScaleRowsByRstd 会重新下发原始值。
    ScaleRowsByRstd<GateT, BetaT, CompilePolicy>(ctx, src, rstdOut, staging, workspace[640], scaleBuf, validRows);
}

// 只有 rstd 已知时的逐行缩放：staging 必须是空闲的 64x128 FP32 区。
// P4：逐行缩放用 "Brcb + 行列重复展开 + 整块 Mul"（与 V6 beta 路径同款），
// 避免 64 次标量 GetValue 与 64 条逐行 Muls/Cast。
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void ScaleRowsByRstd(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                       AscendC::LocalTensor<bfloat16_t> src, AscendC::LocalTensor<float> rstd,
                                       AscendC::LocalTensor<float> staging, AscendC::LocalTensor<float> brcbScalars,
                                       AscendC::LocalTensor<float> repeatScalars, uint32_t validRows)
{
    AscendC::Cast(staging, src, AscendC::RoundMode::CAST_NONE, validRows * Shape::kHeadDim);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(brcbScalars, rstd, static_cast<uint8_t>(Shape::kChunkRows / 8), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    // 单/双 chunk 是纯延迟场景，16 行 tile 在这里反而更慢（实测 T64 +33%）；
    // 大规模吞吐场景才用 16 行。
    const uint32_t kScaleRows = ctx.args_.tiling.totalChunks <= 2 ? 7U : 16U;
    for (uint32_t rowBegin = 0; rowBegin < validRows; rowBegin += kScaleRows) {
        const uint32_t rows = validRows - rowBegin < kScaleRows ? validRows - rowBegin : kScaleRows;
        ReplicateRowScalarsToRepeat<GateT, BetaT, CompilePolicy>(repeatScalars, brcbScalars[8 * rowBegin], rows);
        AscendC::PipeBarrier<PIPE_V>();
        MulTileByRowScalars<GateT, BetaT, CompilePolicy>(staging[rowBegin * Shape::kHeadDim],
                                                         staging[rowBegin * Shape::kHeadDim], repeatScalars, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(src[rowBegin * Shape::kHeadDim], staging[rowBegin * Shape::kHeadDim],
                      AscendC::RoundMode::CAST_RINT, rows * Shape::kHeadDim);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// AIC 预算的 2 个 32x32 Gram 块：用对角掩码 + 行归约取出该 32 行的
// |x|^2（不使用 Gather：对角线步长 33 个 FP32 不是 32B 对齐），
// 再做 eps -> Sqrt -> 倒数得到 rstd。
template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void FillRstdFromSumBlocks(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx,
                                             AscendC::LocalTensor<float> rstdOut, AscendC::LocalTensor<float> sumBlocks,
                                             AscendC::LocalTensor<float> diagMask, AscendC::LocalTensor<float> prod,
                                             AscendC::LocalTensor<float> oneVec, uint32_t validRows)
{
    constexpr uint32_t kBlockRows = 32;
    constexpr uint32_t kBlockElems = kBlockRows * kBlockRows;
    constexpr uint8_t kRowStrideBlocks = static_cast<uint8_t>(kBlockRows * sizeof(float) / 32);
    for (uint32_t block = 0; block < Shape::kChunkRows / kBlockRows; ++block) {
        AscendC::Mul(prod, sumBlocks[block * kBlockElems], diagMask, kBlockElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum(rstdOut[block * kBlockRows], prod, kBlockRows, kBlockRows, 1, 1, kRowStrideBlocks);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Adds(rstdOut, rstdOut, ctx.args_.tiling.epsilon, validRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sqrt(rstdOut, rstdOut, validRows);
    AscendC::Duplicate(oneVec, 1.0F, validRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(rstdOut, oneVec, rstdOut, validRows);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void
V0Vf(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, AscendC::LocalTensor<bfloat16_t> q,
     AscendC::LocalTensor<bfloat16_t> k, AscendC::LocalTensor<float> qRstd, AscendC::LocalTensor<float> kRstd,
     AscendC::LocalTensor<GateT> gate, AscendC::LocalTensor<BetaT> beta, AscendC::LocalTensor<float> betaEff,
     AscendC::LocalTensor<float> dtBias, AscendC::LocalTensor<float> aLog, AscendC::LocalTensor<float> g,
     AscendC::LocalTensor<float> scratch, AscendC::LocalTensor<float> scaleBuf, AscendC::LocalTensor<float> tileBuf,
     uint32_t validRows, AscendC::LocalTensor<float> sumQ, AscendC::LocalTensor<float> sumK, bool cubeSums)
{
    const uint32_t count = validRows * Shape::kHeadDim;
    if constexpr (CompilePolicy::normMode == QkNormMode::L2) {
        // staging 需要 64x128 FP32 = 32 KiB：KDA（FP32 gate，gate 在共享 G）
        // 用 V1 之前空闲的 kGateOrKMinus 私有区；GDN（BF16 gate）的 gate
        // 本身就落在该私有区，改用尚未写入的共享 G 区。
        auto normStaging = (sizeof(GateT) == 4) ? tileBuf : g;
        if (cubeSums) {
            // AIC 已在 C2 的等待窗口里算好 q/k 的 16x16 Gram 块，
            // 这里只取对角线做 eps -> Sqrt -> 倒数，再走同一个缩放。
            // scratch[0,1024) 与 [1024,2048) 分别是 q/k 的 Gram 块；
            // 缩放用的展开表放在 UB 尾部 8 KiB（V1/V6 之后才用）。
            auto repeatBuf = scaleBuf;     // 1024 fp32
            auto brcbBuf = scaleBuf[1024]; // 512 fp32
            auto oneVec = scaleBuf[1600];
            // 对角掩码（32x32，1 KiB）与乘积临时区都放在共享 scratch：
            // 这条路径不再需要 L2NormRowsBlock 的归约缓冲。
            auto diagMask = scratch;
            auto prod = scratch[1024];
            AscendC::Duplicate(diagMask, 0.0F, 1024);
            AscendC::PipeBarrier<PIPE_V>();
            // 对角线用纯向量写法（与 V3Vf 的对角初始化同款）：同一 lane 的
            // 相邻对角元素跨 32+1 个 FP32，repeat stride 取 33。省掉 32 次
            // 标量 SetValue 与两次 V/S 同步。
            constexpr uint8_t kDiagonalRepeatStride = 33;
            for (uint32_t lane = 0; lane < 8; ++lane) {
                uint64_t diagonalOne[2] = {1ULL << lane, 0};
                AscendC::Duplicate(diagMask[lane * 32], 1.0F, diagonalOne, static_cast<uint8_t>((32 - lane + 7) / 8), 1,
                                   kDiagonalRepeatStride);
            }
            AscendC::PipeBarrier<PIPE_V>();
            FillRstdFromSumBlocks<GateT, BetaT, CompilePolicy>(ctx, qRstd, sumQ, diagMask, prod, oneVec, validRows);
            FillRstdFromSumBlocks<GateT, BetaT, CompilePolicy>(ctx, kRstd, sumK, diagMask, prod, oneVec, validRows);
            ScaleRowsByRstd<GateT, BetaT, CompilePolicy>(ctx, q, qRstd, normStaging, brcbBuf, repeatBuf, validRows);
            ScaleRowsByRstd<GateT, BetaT, CompilePolicy>(ctx, k, kRstd, normStaging, brcbBuf, repeatBuf, validRows);
        } else {
            L2NormRowsBlock<GateT, BetaT, CompilePolicy>(ctx, q, qRstd, normStaging, scratch, scaleBuf, validRows);
            L2NormRowsBlock<GateT, BetaT, CompilePolicy>(ctx, k, kRstd, normStaging, scratch, scaleBuf, validRows);
        }
    } else {
        // 固定输出合同下 Identity 仍写确定值；反向关闭 L2Norm 时不会
        // 使用 rstd，但不能留下未初始化输出。
        AscendC::Duplicate(qRstd, 1.0F, validRows);
        AscendC::Duplicate(kRstd, 1.0F, validRows);
        AscendC::PipeBarrier<PIPE_V>();
    }


    if constexpr (CompilePolicy::betaMode == BetaMode::Raw) {
        // Raw 模式只统一转成 FP32，公开 betaEff 不做 sigmoid 变换。
        if constexpr (std::is_same_v<BetaT, float>) {
            AscendC::Adds(betaEff, beta, 0.0F, validRows);
        } else {
            AscendC::Cast(betaEff, beta, AscendC::RoundMode::CAST_NONE, validRows);
        }
    } else {
        if constexpr (std::is_same_v<BetaT, float>) {
            AscendC::Muls(scratch, beta, -1.0F, validRows);
        } else {
            AscendC::Cast(scratch, beta, AscendC::RoundMode::CAST_NONE, validRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(scratch, scratch, -1.0F, validRows);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scratch, scratch, validRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(scratch, scratch, 1.0F, validRows);
        AscendC::Duplicate(betaEff, 1.0F, validRows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(betaEff, betaEff, scratch, validRows);
        if constexpr (CompilePolicy::betaMode == BetaMode::TwoSigmoid) {
            // betaEff 原址读写，必须等待前一条 Div 完成。
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(betaEff, betaEff, 2.0F, validRows);
        }
    }

    if constexpr (std::is_same_v<GateT, float>) {
        AscendC::Adds(g, gate, 0.0F, count);
    } else {
        AscendC::Cast(g, gate, AscendC::RoundMode::CAST_NONE, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (CompilePolicy::gateMode != GateMode::PrecomputedStep) {
        float gateA = 1.0F;
        if (ctx.args_.aLog != nullptr) {
            // a_h = exp(A_log[h])，不能直接把 A_log 当成乘数。
            AscendC::Exp(aLog, aLog, 1);
            AscendC::PipeBarrier<PIPE_V>();
            gateA = ReadScalar<GateT, BetaT, CompilePolicy>(ctx, aLog, 0);
        }
        if (ctx.args_.tiling.hasDtBias) {
            // dtBias 是 [K] 行向量，整块广播下发，避免逐行一次 Add。
            AddRowToTile<GateT, BetaT, CompilePolicy>(g, g, dtBias, validRows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (!CompilePolicy::safeGate && CompilePolicy::gateMode == GateMode::Softplus) {
            // deltaG=-a*(max(x,0)+log(1+exp(-abs(x))))，其中
            // log1p(u)=Ln(1+u)+r/(1+u)，r=u-(fl(1+u)-1)。直接取 Ln(1+u)
            // 会把 1+u 的舍入按 1/u 放大（u=exp(-|x|) 越小越明显），双标杆
            // L1 下会超限，故用补偿项抵消。
            //
            // 两个 FP32 临时区只能落在共享 scratch 的前 7 KiB：scratch 之后
            // 紧邻第二份 private 槽（q/k/gate/kMinus/score），0x1F00 起又是
            // sequence-major 的 beta gather 偏移表，因此按 7 行一个 tile
            // 下发，与 V1Vf 使用同一段 scratch 的约定保持一致。
            const float factor = -gateA;
            constexpr uint32_t kLog1pRows = 7;
            constexpr uint32_t kLog1pElems = kLog1pRows * Shape::kHeadDim;
            auto gateSum = scratch;
            auto gateResidual = scratch[kLog1pElems];
            for (uint32_t rowBegin = 0; rowBegin < validRows; rowBegin += kLog1pRows) {
                const uint32_t rows = validRows - rowBegin < kLog1pRows ? validRows - rowBegin : kLog1pRows;
                const uint32_t elems = rows * Shape::kHeadDim;
                const uint32_t offset = rowBegin * Shape::kHeadDim;
                AscendC::Abs(tileBuf, g[offset], elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(tileBuf, tileBuf, -1.0F, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(tileBuf, tileBuf, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(gateSum, tileBuf, 1.0F, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(gateResidual, gateSum, -1.0F, elems);
                AscendC::PipeBarrier<PIPE_V>();
                // r = u - (fl(1+u) - 1)：补偿 1+u 的舍入残差。
                AscendC::Sub(gateResidual, tileBuf, gateResidual, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(tileBuf, gateSum, 0.0F, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Ln(tileBuf, tileBuf, elems);
                AscendC::PipeBarrier<PIPE_V>();
                // log1p(u) = Ln(1+u) + r/(1+u)
                AscendC::Div(gateResidual, gateResidual, gateSum, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(tileBuf, tileBuf, gateResidual, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(g[offset], g[offset], 0.0F, elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(tileBuf, tileBuf, g[offset], elems);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(g[offset], tileBuf, factor, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
        } else {
            // deltaG=lower_bound/(1+exp(-a*x))。
            // 整块一次下发，替代原先每行 5 条指令。
            const float negativeA = -gateA;
            AscendC::Muls(tileBuf, g, negativeA, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tileBuf, tileBuf, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tileBuf, tileBuf, 1.0F, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(g, ctx.args_.tiling.lowerBound, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(g, g, tileBuf, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    // gate 统一按自然对数单位存放：exp2 域只在写出 gk 时做一次换算，
    // 省掉 V1/V6 里每个 tile 各一次的 ×ln2。
    // P3-B：token 维前缀和改成 Hillis-Steele 对数扫描。
    // 原来 63 步"逐行 Add + 1 个屏障"= 63 次流水排空（屏蔽实测 333 µs）；
    // 现在 log2(validRows) 轮、每轮 2 条块指令 + 1 个屏障，源/目的在
    // g（共享区）与 tileBuf（私有区 0x8000，本 Stage 的 norm/gate 已用完）
    // 之间乒乓。⚠ gk 的求和顺序变成树形，末位与改动前不同（相对 ~1e-7）。
    {
        auto scanSrc = g;
        auto scanDst = tileBuf;
        bool inG = true;
        for (uint32_t d = 1; d < validRows; d <<= 1) {
            AscendC::Adds(scanDst, scanSrc, 0.0F, d * Shape::kHeadDim);
            AscendC::Add(scanDst[d * Shape::kHeadDim], scanSrc[d * Shape::kHeadDim], scanSrc,
                         (validRows - d) * Shape::kHeadDim);
            AscendC::PipeBarrier<PIPE_V>();
            auto swap = scanSrc;
            scanSrc = scanDst;
            scanDst = swap;
            inG = !inG;
        }
        if (!inG) {
            AscendC::Adds(g, tileBuf, 0.0F, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    if (validRows < Shape::kChunkRows) {
        const uint32_t tail = (Shape::kChunkRows - validRows) * Shape::kHeadDim;
        AscendC::Duplicate(q[validRows * Shape::kHeadDim], static_cast<bfloat16_t>(0), tail);
        AscendC::Duplicate(k[validRows * Shape::kHeadDim], static_cast<bfloat16_t>(0), tail);
        AscendC::Duplicate(g[validRows * Shape::kHeadDim], 0.0F, tail);
    }
}

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void V3Vf(TargetVectorContext<GateT, BetaT, CompilePolicy> &ctx, AscendC::LocalTensor<float> raw,
                            AscendC::LocalTensor<float> betaEff, AscendC::LocalTensor<bfloat16_t> aqk,
                            AscendC::LocalTensor<float> lkk, AscendC::LocalTensor<float> b,
                            AscendC::LocalTensor<float> x0, AscendC::LocalTensor<float> x1,
                            AscendC::LocalTensor<float> negX1, AscendC::LocalTensor<bfloat16_t> akkPack,
                            AscendC::LocalTensor<int32_t> coeffOffsetInt, uint32_t validRows, float scale)
{
    AscendC::Duplicate(aqk, static_cast<bfloat16_t>(0), Shape::kChunkRows * Shape::kChunkRows);
    AscendC::Duplicate(b, 0.0F, 1024);
    if constexpr (CompilePolicy::arch22CubeLeaf) {
        // Tail chunks do not touch every element of L. AIC consumes a
        // full 32x32 tile, so make its padding explicit. Do not clear
        // akkPack here: it aliases the tail of the compact raw buffer and
        // must remain untouched until unpacking has consumed rawAkk.
        AscendC::Duplicate(lkk, 0.0F, 4096);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // C2 按 s 堆叠 [rawAqk, rawAkk]，s 之前的 FP32 元素数为
    // subChunkRows^2*s*(s+1)。按全局行解包后，
    // 每个循环不再需要根据行号分支。
    // rawAqk 段的 scale 整体下发一次，随后逐行 Cast 进 Aqk；
    // Akk 段不乘 scale，只能按 s 分别处理。
    const uint32_t activeBands = CeilDiv(validRows, Shape::kSubChunkRows);
    for (uint32_t s = 0; s < activeBands; ++s) {
        const uint32_t remaining = validRows - s * Shape::kSubChunkRows;
        const uint32_t bandRows = remaining < Shape::kSubChunkRows ? remaining : Shape::kSubChunkRows;
        const uint32_t band = Shape::kSubChunkRows * Shape::kSubChunkRows * s * (s + 1);
        AscendC::Muls(raw[band], raw[band], scale, bandRows * Shape::kPrefixRows[s]);
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t globalRow = 0; globalRow < validRows; ++globalRow) {
        const uint32_t s = globalRow / Shape::kSubChunkRows;
        const uint32_t row = globalRow % Shape::kSubChunkRows;
        const uint32_t n = Shape::kPrefixRows[s];
        const uint32_t stackedBand = Shape::kSubChunkRows * Shape::kSubChunkRows * s * (s + 1);
        auto rawAqk = raw[stackedBand + row * n];
        const uint32_t aqkCount = globalRow + 1;
        AscendC::Cast(aqk[globalRow * Shape::kChunkRows], rawAqk, AscendC::RoundMode::CAST_RINT, aqkCount);
    }

    const uint32_t topRows = validRows < 32 ? validRows : 32;
    // beta 的逐行标量只读一次（一次 V->S 同步）；下面的逐行 Muls 读写区间
    // 互不重叠，因此不再需要 per-row 屏障。
    float betaValues[Shape::kChunkRows];
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(ctx.scalarRead_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(ctx.scalarRead_);
        for (uint32_t row = 0; row < validRows; ++row) {
            betaValues[row] = betaEff.GetValue(row);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(ctx.scalarWrite_);
    }
    // A00 的第 0 行没有严格下三角元素，从第 1 行开始写。
    for (uint32_t globalRow = 1; globalRow < topRows; ++globalRow) {
        const uint32_t s = globalRow / Shape::kSubChunkRows;
        const uint32_t row = globalRow % Shape::kSubChunkRows;
        const uint32_t n = Shape::kPrefixRows[s];
        const uint32_t stackedBand = Shape::kSubChunkRows * Shape::kSubChunkRows * s * (s + 1);
        const uint32_t bandRemaining = validRows - s * Shape::kSubChunkRows;
        const uint32_t bandRows = bandRemaining < Shape::kSubChunkRows ? bandRemaining : Shape::kSubChunkRows;
        auto rawAkk = raw[stackedBand + bandRows * n + row * n];
        AscendC::Muls(lkk[globalRow * Shape::kChunkRows], rawAkk, betaValues[globalRow], globalRow);
    }

    if (validRows > 32) {
        constexpr uint32_t firstBottomRow = 32;
        constexpr uint32_t firstBottomSubChunk = firstBottomRow / Shape::kSubChunkRows;
        constexpr uint32_t firstBottomN = Shape::kPrefixRows[firstBottomSubChunk];
        constexpr uint32_t firstBottomBand =
            Shape::kSubChunkRows * Shape::kSubChunkRows * firstBottomSubChunk * (firstBottomSubChunk + 1);
        const uint32_t firstBottomRemaining = validRows - firstBottomSubChunk * Shape::kSubChunkRows;
        const uint32_t firstBottomBandRows =
            firstBottomRemaining < Shape::kSubChunkRows ? firstBottomRemaining : Shape::kSubChunkRows;
        auto firstBottomRawAkk = raw[firstBottomBand + firstBottomBandRows * firstBottomN];
        AscendC::Muls(b, firstBottomRawAkk, betaValues[firstBottomRow], 32);

        // 第 32 行的 L11 长度为 0，单独处理后，余下行同时写 B 和 L11。
        for (uint32_t globalRow = firstBottomRow + 1; globalRow < validRows; ++globalRow) {
            const uint32_t s = globalRow / Shape::kSubChunkRows;
            const uint32_t row = globalRow % Shape::kSubChunkRows;
            const uint32_t n = Shape::kPrefixRows[s];
            const uint32_t stackedBand = Shape::kSubChunkRows * Shape::kSubChunkRows * s * (s + 1);
            const uint32_t bandRemaining = validRows - s * Shape::kSubChunkRows;
            const uint32_t bandRows = bandRemaining < Shape::kSubChunkRows ? bandRemaining : Shape::kSubChunkRows;
            auto rawAkk = raw[stackedBand + bandRows * n + row * n];
            AscendC::Muls(b[(globalRow - 32) * 32], rawAkk, betaValues[globalRow], 32);
            AscendC::Muls(lkk[globalRow * Shape::kChunkRows + 32], rawAkk[32], betaValues[globalRow], globalRow - 32);
        }
    }

    AscendC::PipeBarrier<PIPE_V>();
    // I+Lkk=[[A00,0],[B,A11]]；在同一次 VF 中求两个 32 阶单位下三角逆。
    AscendC::Duplicate(x0, 0.0F, 1024);
    AscendC::Duplicate(x1, 0.0F, 1024);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t bottomRows = validRows > 32 ? validRows - 32 : 0;
    // FP32 向量指令要求 UB 首地址按 32 Byte 对齐，不能直接从
    // x[row, row] 发起长度为 1 的 Duplicate。按对角元素在 32 Byte
    // block 内的 lane 分组：同一 lane 相邻两个对角元素跨 8 行，
    // repeat stride 为 8 * 4 + 1 = 33 个 block。
    constexpr uint32_t kFp32PerBlock = 8;
    constexpr uint16_t kDiagonalBlockStride = 1;
    constexpr uint8_t kDiagonalRepeatStride = 33;
    const uint32_t topDiagonalLanes = topRows < kFp32PerBlock ? topRows : kFp32PerBlock;
    for (uint32_t lane = 0; lane < topDiagonalLanes; ++lane) {
        uint64_t diagonalMask[2] = {1ULL << lane, 0};
        const uint8_t repeat = static_cast<uint8_t>(CeilDiv(topRows - lane, kFp32PerBlock));
        AscendC::Duplicate(x0[lane * 32], 1.0F, diagonalMask, repeat, kDiagonalBlockStride, kDiagonalRepeatStride);
    }
    const uint32_t bottomDiagonalLanes = bottomRows < kFp32PerBlock ? bottomRows : kFp32PerBlock;
    for (uint32_t lane = 0; lane < bottomDiagonalLanes; ++lane) {
        uint64_t diagonalMask[2] = {1ULL << lane, 0};
        const uint8_t repeat = static_cast<uint8_t>(CeilDiv(bottomRows - lane, kFp32PerBlock));
        AscendC::Duplicate(x1[lane * 32], 1.0F, diagonalMask, repeat, kDiagonalBlockStride, kDiagonalRepeatStride);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (!CompilePolicy::arch22CubeLeaf) {
        // 单位下三角逆：X[i,:]=-sum(k<i,L[i,k]*X[k,:])，X[i,i]=1。
        // 系数靠 Gather 每次只取一列，偏移张量与 k 无关，只随行号变化：
        // offsets[i] = 256 * (i / 32 + 1) 字节，用 4 条矢量指令生成本次调用的
        // 偏移表（Leaf0 区在 V3 内空闲）；系数区与乘积区放在已消费的
        // compact raw 低地址。
        // 每个 dst 行 32 个 lane 共用一个偏移，行数取两个叶里更大的那个。
        constexpr uint32_t kLeafLanes = 32;
        const uint32_t coeffCount = (topRows > bottomRows ? topRows : bottomRows) * kLeafLanes;
        AscendC::CreateVecIndex(coeffOffsetInt, 0, coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(coeffOffsetInt, coeffOffsetInt, 5, coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(coeffOffsetInt, coeffOffsetInt, 1, coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(coeffOffsetInt, coeffOffsetInt, static_cast<int32_t>(Shape::kChunkRows * sizeof(float)),
                      coeffCount);
        AscendC::PipeBarrier<PIPE_V>();
        auto coeffOffset = coeffOffsetInt.template ReinterpretCast<uint32_t>();
        auto coeffTile = raw;
        auto prodTile = raw[2048];
        // 合并两个叶子的逆：4 块 scratch 各 1024 FP32，仍在 compact raw
        // 区间内（raw 可用到 5120 FP32，解包已经消费完）。
        auto coeffTileB = raw[3072];
        auto prodTileB = raw[4096];
        SolveTwoLeavesRightLooking<GateT, BetaT, CompilePolicy>(x0, x1, lkk, coeffOffset, coeffTile, prodTile,
                                                                coeffTileB, prodTileB, topRows, bottomRows, 0,
                                                                32 * Shape::kChunkRows + 32, Shape::kChunkRows);
        // B 已在解包 rawAkk 时直接写入；其余行保持零。
        AscendC::PipeBarrier<PIPE_V>();
        // C5 直接执行普通 MMAD：negX1@T，不依赖不存在的 negate 参数。
        AscendC::Muls(negX1, x1, -1.0F, 1024);
        AscendC::PipeBarrier<PIPE_V>();

        // q00=X0、q01=0、q11=X1；q10 留给 C5 的 negX1@T。
        AscendC::Duplicate(akkPack, static_cast<bfloat16_t>(0), 4096);
        AscendC::PipeBarrier<PIPE_V>();
        // 两段对角块的目标互不重叠，逐行屏障没有保护任何依赖。
        for (uint32_t row = 0; row < topRows; ++row) {
            AscendC::Cast(akkPack[row * Shape::kChunkRows], x0[row * 32], AscendC::RoundMode::CAST_RINT, 32);
        }
        for (uint32_t row = 0; row < bottomRows; ++row) {
            AscendC::Cast(akkPack[(row + 32) * Shape::kChunkRows + 32], x1[row * 32], AscendC::RoundMode::CAST_RINT,
                          32);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}


} // namespace KdaPrepare::Arch22Target

#endif // ARCH22_TARGET_CHUNK_KDA_FWD_PREPARE_VEC_H
