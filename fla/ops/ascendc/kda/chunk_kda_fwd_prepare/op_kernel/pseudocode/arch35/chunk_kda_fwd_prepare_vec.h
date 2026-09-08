/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_VEC_H
#define FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_VEC_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../chunk_kda_fwd_prepare_policy.h"
#include "../chunk_kda_fwd_prepare_struct.h"
#include "../chunk_kda_fwd_prepare_tiling_key.h"
#include "../chunk_kda_fwd_prepare_utils.h"

namespace kda_prepare_pseudocode {
namespace arch35 {
namespace detail {

inline BufferSpan Subspan(const BufferSpan &parent, const char *name,
                          Offset relativeOffset, Offset bytes)
{
    BufferSpan span = parent;
    span.name = name;
    span.byteOffset += relativeOffset;
    span.byteSize = bytes;
    span.rows = 0U;
    span.columns = 0U;
    span.leadingDimension = 0U;
    span.elementBytes = 0U;
    return span;
}

inline BufferSpan UbMainSpan(const HeadTask &head, const char *name,
                             const Region &region, std::uint64_t generation)
{
    return {name, MemorySpace::Ub,
            static_cast<std::uint64_t>(UbPolicy::kMainBase[head.aivLocalSlot]) +
                region.offset,
            region.size, head.localBankId, generation, CoreRole::Aiv,
            head.aivId};
}

inline BufferSpan UbVectorStateSpan(const HeadTask &head, const char *name,
                                    const Region &region,
                                    std::uint64_t generation)
{
    return {name, MemorySpace::Ub,
            static_cast<std::uint64_t>(
                UbPolicy::kVectorStateBase[head.aivLocalSlot]) +
                region.offset,
            region.size, head.localBankId, generation, CoreRole::Aiv,
            head.aivId};
}

inline BufferSpan SymbolicGmSpan(const HeadTask &head, const char *name,
                                 Offset bytes, std::uint64_t generation,
                                 std::uint32_t logicalHeadId =
                                     kAllGroupLocalHeads)
{
    // 公开跨步和规范 GM 偏移仍需由 ABI/切分参数确认。这里的零值是刻意
    // 保留的符号占位，不能照搬到真实核函数中。
    BufferSpan span{name, MemorySpace::Gm, 0U, bytes, head.workspaceSlot,
                    generation, CoreRole::Shared, 0U};
    span.logicalHeadId = logicalHeadId == kAllGroupLocalHeads
                             ? head.headId
                             : logicalHeadId;
    return span;
}

constexpr Offset MatrixFootprintBytes(Offset rows, Offset columns,
                                      Offset leadingDimension,
                                      Offset elementBytes)
{
    return rows == 0U || columns == 0U
               ? 0U
               : ((rows - 1U) * leadingDimension + columns) * elementBytes;
}

inline BufferSpan UbMainMatrixSpan(
    const HeadTask &head, const char *name, Offset base, Offset row,
    Offset column, Offset rows, Offset columns, Offset leadingDimension,
    Offset elementBytes, std::uint64_t generation)
{
    const Offset relativeOffset =
        base + (row * leadingDimension + column) * elementBytes;
    BufferSpan span = UbMainSpan(
        head, name,
        {relativeOffset, MatrixFootprintBytes(
                             rows, columns, leadingDimension, elementBytes)},
        generation);
    span.rows = rows;
    span.columns = columns;
    span.leadingDimension = leadingDimension;
    span.elementBytes = elementBytes;
    return span;
}

inline BufferSpan SymbolicGmMatrixSpan(
    const HeadTask &head, const char *name, Offset row, Offset column,
    Offset rows, Offset columns, Offset leadingDimension,
    Offset elementBytes, std::uint64_t generation)
{
    // 该伪接口刻意保留 row/column/rows/columns/leadingDimension。
    // BufferSpan 保存跨步视图精确的物理覆盖范围；真实 ABI 必须冻结并编码对应的
    // 二维 DataCopy。
    const Offset relativeOffset =
        (row * leadingDimension + column) * elementBytes;
    const Offset footprint = MatrixFootprintBytes(
        rows, columns, leadingDimension, elementBytes);
    BufferSpan span{name,
                    MemorySpace::Gm,
                    relativeOffset,
                    footprint,
                    head.workspaceSlot,
                    generation,
                    CoreRole::Shared,
                    0U,
                    rows,
                    columns,
                    leadingDimension,
                    elementBytes};
    span.logicalHeadId = head.headId;
    return span;
}

inline bool IsOwnedActiveHead(const HeadTask &head,
                              const VectorStageArgs &args)
{
    return head.active && head.aivId == args.aivId;
}

inline std::uint32_t ActiveScoreBlocks(std::uint32_t validRows)
{
    return std::min<std::uint32_t>(ShapePolicy::kScoreBlockCount,
                                   (validRows + ShapePolicy::kScoreBlockRows - 1U) /
                                       ShapePolicy::kScoreBlockRows);
}

inline bool IsSupportedKey(const ProposedTilingKey &key)
{
    // FP32Internal 仍是容量已证明可行的布局候选，但目标平台混合使用
    // FP32 Akk 与双字节右操作数的 Cube 操作数合同尚未验证。
    return IsSupportedTilingKey(key);
}

// VF 伪接口不会在仅主机的设计构建中实例化；它只冻结一次 VF 内的数学顺序。
template <typename Vf>
inline void V0OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    const ProposedTilingKey &key, float epsilon,
                    float lowerBound)
{
    const Offset gOffset = IsTwoByteGateStorage(key.gateStorage)
                               ? V0Gate2BLayout::kG.offset
                               : V0GateFp32Layout::kG.offset;
    const bool usesSelectiveGate =
        key.gateMode != GateMode::PrecomputedStep;
    auto gateCoefficient = vf.OneFp32();
    auto dtBias = vf.ZeroFp32Row(ShapePolicy::kK);
    if (usesSelectiveGate) {
        // 每个头的系数和 K 向量偏置只在词元扫描外搬运或
        // 计算一次。它们在每头 UB 向量状态区中的地址与 GRef[0:2] 复用；
        // 本循环消费完两个值后，才物化 GRef[0:2]。
        gateCoefficient = vf.Exp(vf.LoadALogScalarOnce(
            UbPolicy::kVectorStateBase[head.aivLocalSlot] +
            VectorStateLayout::kALogOrGateAttrs.offset));
        dtBias = vf.LoadDtBiasRow(
            UbPolicy::kVectorStateBase[head.aivLocalSlot] +
            VectorStateLayout::kDtBias.offset);
    }
    auto carry = vf.ZeroFp32Row(ShapePolicy::kK);
    auto zeroInputStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), key.inputStorage),
        key.inputStorage);
    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        if (row >= validRows) {
            vf.StoreZeroQHatKHatAndGPadding(
                row, zeroInputStorage, key.inputStorage);
            vf.StoreBetaEffScalar(row, 0.0F);
            continue;
        }
        auto gateRaw = vf.LoadGateRow(row, key.gateStorage);
        auto betaRaw = vf.LoadBetaFp32Scalar(
            UbPolicy::kVectorStateBase[head.aivLocalSlot] +
                VectorStateLayout::kBetaRaw.offset,
            row);
        auto betaEff = betaRaw;
        if (key.betaMode == BetaMode::Sigmoid) {
            betaEff = vf.Sigmoid(betaRaw);
        } else if (key.betaMode == BetaMode::TwoSigmoid) {
            betaEff = vf.Mul(2.0F, vf.Sigmoid(betaRaw));
        }

        if (head.qkOwner && key.qkNormMode == QkNormMode::L2) {
            // 与仓库内 FLA L2 核函数及精度标杆保持一致，归一化公式为：
            // x_hat = x * rsqrt(sum_d(x_d^2) + epsilon)。
            const auto q =
                vf.ToFp32(vf.LoadQStorageRow(row, key.inputStorage));
            const auto k =
                vf.ToFp32(vf.LoadKStorageRow(row, key.inputStorage));
            const auto qHat =
                vf.L2NormalizeRsqrtSumPlusEpsilon(q, epsilon);
            const auto kHat =
                vf.L2NormalizeRsqrtSumPlusEpsilon(k, epsilon);
            vf.StoreStorageRow(
                V0Gate2BLayout::kQHat.offset, row,
                vf.RoundToInputStorage(
                    vf.ClampForInputStorage(qHat, key.inputStorage),
                    key.inputStorage),
                key.inputStorage);
            vf.StoreStorageRow(
                V0Gate2BLayout::kKHat.offset, row,
                vf.RoundToInputStorage(
                    vf.ClampForInputStorage(kHat, key.inputStorage),
                    key.inputStorage),
                key.inputStorage);
        }

        auto gateStep = gateRaw;
        if (key.gateMode == GateMode::PrecomputedStep) {
            gateStep = vf.Div(gateRaw, vf.Ln2());
        } else {
            auto x = vf.Add(gateRaw, dtBias);
            if (key.gateMode == GateMode::Softplus) {
                auto stableSoftplus = vf.Add(
                    vf.Max(x, vf.ZeroFp32()),
                    vf.Log1p(vf.Exp(vf.Neg(vf.Abs(x)))));
                gateStep = vf.Div(
                    vf.Neg(vf.Mul(gateCoefficient, stableSoftplus)),
                    vf.Ln2());
            } else {
                gateStep = vf.Div(
                    vf.Mul(lowerBound,
                           vf.Sigmoid(vf.Mul(gateCoefficient, x))),
                    vf.Ln2());
            }
        }
        carry = vf.Add(carry, gateStep); // 词元顺序构成真实的扫描依赖。
        // 恒等路径所有者和映射到它的非所有者保留已常驻 Qhat/Khat 的 2 字节
        // MTE2 结果。再次转换和舍入只会造成重复向量计算，无法提高精度。
        vf.StoreFp32Row(gOffset, row, carry);
        vf.StoreBetaEffScalar(row, betaEff);
    }

    for (std::uint32_t s = 0; s < ShapePolicy::kScoreBlockCount; ++s) {
        const std::uint32_t begin = s * ShapePolicy::kScoreBlockRows;
        const std::uint32_t end = std::min(begin + ShapePolicy::kScoreBlockRows,
                                           validRows);
        if (begin >= end) {
            vf.ZeroFp32Row(UbPolicy::kVectorStateBase[head.aivLocalSlot] +
                           VectorStateLayout::kGRef[s].offset);
            continue;
        }
        const std::uint32_t referenceRow = begin + (end - begin) / 2U;
        auto reference = vf.LoadFp32Row(gOffset, referenceRow);
        vf.StoreFp32Row(UbPolicy::kVectorStateBase[head.aivLocalSlot] +
                            VectorStateLayout::kGRef[s].offset,
                        reference);
    }
    vf.StoreGLast(carry);
}

template <bool UseExp2, typename Vf>
inline void V1OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    GateStorage gateStorage, InputStorage inputStorage,
                    ScoreStorage scoreStorage)
{
    const Offset gOffset = IsTwoByteGateStorage(gateStorage)
                               ? V1Gate2BLayout::kLiveG.offset
                               : V1GateFp32Layout::kLiveG.offset;
    const auto &kMinus = IsTwoByteGateStorage(gateStorage)
                             ? V1Gate2BLayout::kKMinus
                             : V1GateFp32Layout::kKMinus;
    const std::uint32_t activeBlocks = ActiveScoreBlocks(validRows);
    const float exp2InputMin = ScoreExp2InputMin(scoreStorage);
    const float exp2InputMax = ScoreExp2InputMax(scoreStorage);
    auto zeroScoreStorage = vf.RoundToScoreStorage(
        vf.ClampForScoreStorage(vf.ZeroFp32(), scoreStorage), scoreStorage);

    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        if (row < validRows) {
            // 在任何存在别名的 Q/K 写入前，先将三个源行读入临时寄存器。
            auto qHat = vf.ToFp32(vf.LoadStorageRow(
                V1Gate2BLayout::kQPlus.offset, row, inputStorage));
            auto kHat = vf.ToFp32(vf.LoadStorageRow(
                V1Gate2BLayout::kKPlus.offset, row, inputStorage));
            auto g = vf.LoadFp32Row(gOffset, row);
            const std::uint32_t owner = row / ShapePolicy::kScoreBlockRows;
            auto ownerRef = vf.LoadFp32Row(
                UbPolicy::kVectorStateBase[head.aivLocalSlot] +
                VectorStateLayout::kGRef[owner].offset);
            auto plusFactor = EvaluatePow2<UseExp2>(
                vf, vf.Sub(g, ownerRef), exp2InputMin, exp2InputMax);
            auto qPlus = vf.Mul(qHat, plusFactor);
            auto kPlus = vf.Mul(kHat, plusFactor);

            // 在所有所需前缀生成完成前，Khat 一直保留在寄存器中；所有
            // Kminus 目标均不与仍在生命周期内的 G 重叠。
            for (std::uint32_t s = 0; s < ShapePolicy::kScoreBlockCount; ++s) {
                const std::uint32_t physicalEnd =
                    ShapePolicy::kPrefixRows[s];
                const std::uint32_t logicalEnd =
                    ShapePolicy::LogicalPrefixRows(s, validRows);
                if (row >= physicalEnd) {
                    continue;
                }
                if (s < activeBlocks && row < logicalEnd) {
                    auto reference = vf.LoadFp32Row(
                        UbPolicy::kVectorStateBase[head.aivLocalSlot] +
                        VectorStateLayout::kGRef[s].offset);
                    auto minusFactor = EvaluatePow2<UseExp2>(
                        vf, vf.Sub(reference, g), exp2InputMin,
                        exp2InputMax);
                    auto kMinusStorage = vf.RoundToScoreStorage(
                        vf.ClampForScoreStorage(
                            vf.Mul(kHat, minusFactor), scoreStorage),
                        scoreStorage);
                    vf.StoreStorageRow(kMinus[s].offset, row, kMinusStorage,
                                       scoreStorage);
                } else {
                    vf.StoreStorageRow(kMinus[s].offset, row,
                                       zeroScoreStorage, scoreStorage);
                }
            }
            auto qPlusStorage = vf.RoundToScoreStorage(
                vf.ClampForScoreStorage(qPlus, scoreStorage), scoreStorage);
            auto kPlusStorage = vf.RoundToScoreStorage(
                vf.ClampForScoreStorage(kPlus, scoreStorage), scoreStorage);
            vf.StoreStorageRow(V1Gate2BLayout::kQPlus.offset, row,
                               qPlusStorage, scoreStorage);
            vf.StoreStorageRow(V1Gate2BLayout::kKPlus.offset, row,
                               kPlusStorage, scoreStorage);
        } else {
            vf.StoreStorageRow(V1Gate2BLayout::kQPlus.offset, row,
                               zeroScoreStorage, scoreStorage);
            vf.StoreStorageRow(V1Gate2BLayout::kKPlus.offset, row,
                               zeroScoreStorage, scoreStorage);
            for (std::uint32_t s = 0; s < ShapePolicy::kScoreBlockCount; ++s) {
                // 尾块保留固定的 B_s 字节区段，并显式清零
                // [b_s=min(B_s,M), B_s)；不会缩小物理槽位。
                if (row < ShapePolicy::kPrefixRows[s]) {
                    vf.StoreStorageRow(kMinus[s].offset, row,
                                       zeroScoreStorage, scoreStorage);
                }
            }
        }
    }
}

template <typename Vf>
inline void V3ReadAndTransformRaw(Vf &vf, std::uint32_t validRows,
                                  float scale)
{
    for (std::uint32_t row = 0; row < ShapePolicy::kBt; ++row) {
        for (std::uint32_t col = 0; col < ShapePolicy::kBt; ++col) {
            const bool readAqk =
                V3AqkRawReadRequired(validRows, row, col);
            const bool readAkk =
                V3AkkRawReadRequired(validRows, row, col);
            auto aqk = readAqk
                           ? vf.Mul(scale, vf.LoadRawAqk(row, col))
                           : vf.ZeroFp32();
            auto lkk = readAkk
                           ? vf.Mul(vf.LoadBetaEff(row),
                                    vf.LoadRawAkk(row, col))
                           : vf.ZeroFp32();
            vf.StoreAqk(row, col, aqk);
            vf.StoreLkkOrIdentityPadding(row, col, lkk, validRows);
        }
    }
}

template <typename Vf>
inline void V3OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    PrepareAbi abi, AkkStorage akkStorage,
                    InputStorage inputStorage, float scale)
{
    (void)head;
    // 此处仍只调用一次 VF。每次原始数据搬运都由谓词限定在共享物理写入域和
    // 有效因果域内；无效输出直接生成，不读取残留的原始数据存储。
    V3ReadAndTransformRaw(vf, validRows, scale);
    vf.InvertTwo32By32LeavesWithFixedColumnScan();
    vf.MaterializeX0X1AndBAtFinalOffsets();
    if (akkStorage == AkkStorage::TwoByteAbi) {
        // q00/q01/q11 直接写入各自最终的紧凑象限优先布局地址。
        // ClampForInputStorage 只对 FP16 执行有限值 +/-65504 饱和；随后 FP16
        // 和 BF16 分别使用各自的舍入方式。
        auto zeroStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.ZeroFp32(), inputStorage),
            inputStorage);
        for (std::uint32_t row = 0U;
             row < Akk2BPackPolicy::kQuadrantRows; ++row) {
            for (std::uint32_t col = 0U;
                 col < Akk2BPackPolicy::kQuadrantColumns; ++col) {
                // q00/q11 暂存在其最终 UB 偏移。Current 还需要 q01 作为公开
                // Akk 输出；Fused 不物化这个已知零值，由 C4 在最终 L1 地址
                // 填充 q01。
                if (V3StableAkkWriteRequired(
                        Architecture::Arch35, abi, validRows, row, col)) {
                    auto q00 = vf.LoadStableAkkQ00OrZeroColumnPadding(
                        row, col, validRows);
                    auto q00Storage = vf.RoundToInputStorage(
                        vf.ClampForInputStorage(q00, inputStorage),
                        inputStorage);
                    vf.StoreInputStorageMatrix(
                        V3Layout::kX0Tau.offset, row, col, q00Storage,
                        inputStorage);
                }
                if (V3StableAkkWriteRequired(Architecture::Arch35, abi,
                                             validRows, row, col + 32U)) {
                    vf.StoreInputStorageMatrix(
                        V3Layout::kQ01Zero.offset, row, col, zeroStorage,
                        inputStorage);
                }
                if (V3StableAkkWriteRequired(Architecture::Arch35, abi,
                                             validRows, row + 32U,
                                             col + 32U)) {
                    auto q11 = vf.LoadStableAkkQ11OrZeroColumnPadding(
                        row, col, validRows);
                    auto q11Storage = vf.RoundToInputStorage(
                        vf.ClampForInputStorage(q11, inputStorage),
                        inputStorage);
                    vf.StoreInputStorageMatrix(
                        V3Layout::kX1Tau.offset, row, col, q11Storage,
                        inputStorage);
                }
            }
        }
    }
    // Aqk 此后不再有 FP32 读取方。以相同基址在原地向低地址执行类型转换；
    // 按行、列正序处理时只会覆盖已经读取的值。这不属于 UB 数据移动，
    // 也不是第二次 VF 调用。
    for (std::uint32_t row = 0U; row < ShapePolicy::kBt; ++row) {
        for (std::uint32_t col = 0U; col < ShapePolicy::kBt; ++col) {
            auto aqkStorage = vf.RoundToInputStorage(
                vf.ClampForInputStorage(vf.LoadAqk(row, col), inputStorage),
                inputStorage);
            vf.StoreAqkStorageInPlace(row, col, aqkStorage, inputStorage);
        }
    }
}

template <bool UseExp2, typename Vf>
inline void V6OneVf(Vf &vf, const HeadTask &head, std::uint32_t validRows,
                    PrepareAbi abi, InputStorage inputStorage,
                    InputStorage valueStorage, float scale)
{
    (void)head;
    auto zeroQkStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), inputStorage), inputStorage);
    auto zeroValueStorage = vf.RoundToInputStorage(
        vf.ClampForInputStorage(vf.ZeroFp32(), valueStorage), valueStorage);
    const std::uint32_t rhsRows =
        validRows > Akk2BPackPolicy::kQuadrantRows
            ? ShapePolicy::kBt
            : Akk2BPackPolicy::kQuadrantRows;
    if (validRows == 0U) {
        for (std::uint32_t row = 0; row < rhsRows; ++row) {
            vf.StoreZeroV6OutputRows(row, abi, zeroQkStorage,
                                     zeroValueStorage, inputStorage,
                                     valueStorage);
        }
        return;
    }
    auto gLast = vf.LoadFp32Row(V6Layout::kGInput.offset, validRows - 1U);
    for (std::uint32_t row = 0; row < rhsRows; ++row) {
        if (row >= validRows) {
            vf.StoreZeroV6OutputRows(row, abi, zeroQkStorage,
                                     zeroValueStorage, inputStorage,
                                     valueStorage);
            continue;
        }
        // 在 Q/K/V 原地写入以及可选的 2 字节 QgScaled 压缩到 FP32 G 低半区前，
        // 先将本行所有源操作数读入寄存器。
        auto qHat = vf.ToFp32(vf.LoadStorageRow(
            V6Layout::kQHatToQg.offset, row, inputStorage));
        auto kHat = vf.ToFp32(vf.LoadStorageRow(
            V6Layout::kKHatToKg.offset, row, inputStorage));
        auto v = vf.ToFp32(vf.LoadStorageRow(
            V6Layout::kVToVBeta.offset, row, valueStorage));
        auto g = vf.LoadFp32Row(V6Layout::kGInput.offset, row);
        auto beta = vf.LoadBetaEff(row);
        auto expG = EvaluatePow2<UseExp2>(
            vf, g, kDirectExp2InputMin, kDirectExp2InputMax);
        auto qgFp32 = vf.Mul(qHat, expG);
        auto kgFp32 = vf.Mul(
            kHat,
            EvaluatePow2<UseExp2>(vf, vf.Sub(gLast, g),
                                  kDirectExp2InputMin,
                                  kDirectExp2InputMax));
        auto qgStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(qgFp32, inputStorage), inputStorage);
        auto kgStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(kgFp32, inputStorage), inputStorage);

        // 保持拆分 Prepare 的双重舍入点。kHat*exp2(G) 先物化到 InputStorage，
        // 再提升回 FP32 参与 beta 计算，最后为 K_beta_g 再次执行饱和与舍入。
        auto kGateStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(kHat, expG), inputStorage),
            inputStorage);
        auto kBetaGFp32 = vf.Mul(beta, vf.ToFp32(kGateStorage));
        auto kBetaGStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(kBetaGFp32, inputStorage), inputStorage);
        auto vBetaStorage = vf.RoundToInputStorage(
            vf.ClampForInputStorage(vf.Mul(beta, v), valueStorage),
            valueStorage);

        vf.StoreStorageRow(V6Layout::kQHatToQg.offset, row, qgStorage,
                           inputStorage);
        vf.StoreStorageRow(V6Layout::kKHatToKg.offset, row, kgStorage,
                           inputStorage);
        vf.StoreStorageRow(V6Layout::kKBetaG.offset, row, kBetaGStorage,
                           inputStorage);
        vf.StoreStorageRow(V6Layout::kVToVBeta.offset, row, vBetaStorage,
                           valueStorage);
        if (abi == PrepareAbi::Fused) {
            // 2 字节目标与 FP32 G 的 floor(row/2) 行复用，该行不会晚于当前 row。
            // 当前 G 行已在上方完整读取。缩放先消费第一次舍入后的 qg，再执行
            // FUSED ABI 要求的第二次 InputStorage 饱和与舍入。
            auto qgScaledStorage = vf.RoundToInputStorage(
                vf.ClampForInputStorage(
                    vf.Mul(scale, vf.ToFp32(qgStorage)), inputStorage),
                inputStorage);
            vf.StoreStorageRow(V6Layout::kQgScaled.offset, row,
                               qgScaledStorage, inputStorage);
        }
    }
}

inline bool ValidArgs(const VectorStageArgs &args)
{
    return args.work != nullptr && args.workspace != nullptr &&
           args.sync != nullptr && args.ops != nullptr &&
           IsSupportedKey(args.key);
}

} // namespace detail

inline void RunV0(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        args.sync->Wait(SyncPoint::SlotFree, head.workspaceSlot,
                        workspaceGeneration,
                        Stage::V0, Pipe::Control);
        if (head.qkOwner) {
            args.sync->Wait(SyncPoint::QkCacheFree, head.qkCacheSlot,
                            head.qkCacheGeneration, Stage::V0,
                            Pipe::Mte2);
        } else {
            // QkCacheReady 是工作区控制页中的电平式（非消费式）代际状态。
            // 多个映射到该缓存的 HV 头获取同一次发布，但不会消费它。
            args.sync->Wait(SyncPoint::QkCacheReady, head.qkCacheSlot,
                            head.qkCacheGeneration, Stage::V0,
                            Pipe::Mte2);
        }

        const SymbolicMutexId ubMutex =
            Arch35VectorMutexIds::UbBank(head.aivLocalSlot);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V0, Pipe::Mte2);

        const bool gate2B = IsTwoByteGateStorage(args.key.gateStorage);
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset qkInputBytes =
            validRows * ShapePolicy::kK * ShapePolicy::kStorageBytes;
        const Offset gateInputBytes =
            validRows * ShapePolicy::kK *
            (gate2B ? ShapePolicy::kStorageBytes : ShapePolicy::kFp32Bytes);
        const Offset gOutputBytes =
            validRows * ShapePolicy::kK * ShapePolicy::kFp32Bytes;
        const Offset betaInputBytes =
            validRows * ShapePolicy::kFp32Bytes;
        const Region q = gate2B ? V0Gate2BLayout::kQHat : V0GateFp32Layout::kQHat;
        const Region k = gate2B ? V0Gate2BLayout::kKHat : V0GateFp32Layout::kKHat;
        const Region g = gate2B ? V0Gate2BLayout::kGateRaw : V0GateFp32Layout::kG;
        const BufferSpan qkCache = args.workspace->Span(
            WorkspaceRegion::Context, head.qkCacheSlot,
            head.qkCacheGeneration);
        const BufferSpan qSource =
            head.qkOwner
                ? detail::SymbolicGmSpan(
                      head, "q", qkInputBytes, workspaceGeneration,
                      head.qkHeadId)
                : detail::Subspan(qkCache, "qhat-HK-cache", 0x0000U,
                                  qkInputBytes);
        const BufferSpan kSource =
            head.qkOwner
                ? detail::SymbolicGmSpan(
                      head, "k", qkInputBytes, workspaceGeneration,
                      head.qkHeadId)
                : detail::Subspan(qkCache, "khat-HK-cache", 0x4000U,
                                  qkInputBytes);
        args.ops->Load(Stage::V0,
                       qSource,
                       detail::UbMainSpan(
                           head, "q-to-qhat", {q.offset, qkInputBytes},
                           localGeneration));
        args.ops->Load(Stage::V0,
                       kSource,
                       detail::UbMainSpan(
                           head, "k-to-khat", {k.offset, qkInputBytes},
                           localGeneration));
        args.ops->Load(Stage::V0,
                       detail::SymbolicGmSpan(head, "gate", gateInputBytes,
                                              workspaceGeneration),
                       detail::UbMainSpan(
                           head, "gate-to-G", {g.offset, gateInputBytes},
                           localGeneration));
        // 拆分前向核函数边界上的 beta 为 FP32。公开的 2 字节 beta 在本次
        // 搬运前由 op_api/L2 完成类型转换，因此 V0 恰好读取 M 个标量。
        args.ops->Load(Stage::V0,
                       detail::SymbolicGmSpan(head, "beta", betaInputBytes,
                                              workspaceGeneration),
                       detail::UbVectorStateSpan(
                           head, "beta-raw",
                           {VectorStateLayout::kBetaRaw.offset,
                            betaInputBytes},
                           localGeneration));
        if (args.key.gateMode != GateMode::PrecomputedStep) {
            // 每个选择性门控输入只搬运一次。可选 dt_bias 缺失时，必须在
            // 该 dt_bias UB 落点清零，不能越过空指针或过短的 GM 张量读取。
            const BufferSpan dtBias = detail::UbVectorStateSpan(
                head, "dt-bias", VectorStateLayout::kDtBias,
                localGeneration);
            if (args.hasDtBias) {
                args.ops->Load(
                    Stage::V0,
                    detail::SymbolicGmSpan(
                        head, "dt-bias",
                        ShapePolicy::kK * ShapePolicy::kFp32Bytes,
                        workspaceGeneration),
                    dtBias);
            } else {
                args.ops->Zero(Stage::V0, dtBias);
            }
            args.ops->Load(
                Stage::V0,
                detail::SymbolicGmSpan(head, "A-log", ShapePolicy::kFp32Bytes,
                                       workspaceGeneration),
                detail::UbVectorStateSpan(
                    head, "A-log",
                    {VectorStateLayout::kALogOrGateAttrs.offset,
                     ShapePolicy::kFp32Bytes},
                    localGeneration));
        }

        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V0, Pipe::Mte2);
        args.sync->Local(LocalDependency::Mte2ToVectorInputs, Stage::V0);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V0, Pipe::Vector);
        // 仅调用一次；所需函数体为 detail::V0OneVf，并传入 args.key 以及已冻结
        // 的 epsilon/lowerBound 标量属性。
        args.ops->RunVf(Stage::V0, head);
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V0, Pipe::Vector);

        const BufferSpan context = args.workspace->Span(
            WorkspaceRegion::Context, head.workspaceSlot,
            workspaceGeneration);
        args.sync->Local(LocalDependency::VectorToMte3Outputs, Stage::V0);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V0, Pipe::Mte3);
        const Region gResult = gate2B ? V0Gate2BLayout::kG : V0GateFp32Layout::kG;
        if (head.qkOwner) {
            args.ops->Store(
                Stage::V0,
                detail::UbMainSpan(head, "qhat", {q.offset, qkInputBytes},
                                   localGeneration),
                detail::Subspan(
                    qkCache, "qhat-HK-cache", 0x0000U, qkInputBytes));
            args.ops->Store(
                Stage::V0,
                detail::UbMainSpan(head, "khat", {k.offset, qkInputBytes},
                                   localGeneration),
                detail::Subspan(
                    qkCache, "khat-HK-cache", 0x4000U, qkInputBytes));
        }
        if (args.key.abi == PrepareAbi::Current) {
            // Current 已将 G 作为 gk 公开。V6 复用这一份 GM 数据，不再在
            // 上下文中物化相同的 Vector 数据。
            args.ops->Store(
                Stage::V0,
                detail::UbMainSpan(
                    head, "G-output", {gResult.offset, gOutputBytes},
                    localGeneration),
                detail::SymbolicGmSpan(head, "gk-output", gOutputBytes,
                                       workspaceGeneration));
        } else {
            args.ops->Store(
                Stage::V0,
                detail::UbMainSpan(head, "G-context-source",
                                   {gResult.offset, gOutputBytes},
                                   localGeneration),
                detail::Subspan(context, "G-context", 0x8200U,
                                gOutputBytes));
        }
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V0, Pipe::Mte3);

        // 仅在最后一个已启用的 MTE3/输出搬运完成后，跨核状态才可见。
        if (head.qkOwner) {
            args.sync->Set(SyncPoint::QkCacheReady, head.qkCacheSlot,
                           head.qkCacheGeneration, Stage::V0, Pipe::Mte3);
        }
    }
}

inline void RunV1(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        const SymbolicMutexId ubMutex =
            Arch35VectorMutexIds::UbBank(head.aivLocalSlot);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V1, Pipe::Vector);
        // 仅调用一次。主机分发根据 args.key 中的输入/门控/分数存储类型
        // 特化 detail::V1OneVf<useExp2>；两个 2^x 计算点以及所有饱和与舍入点
        // 都保留在同一次 VF 中。
        args.ops->RunVf(Stage::V1, head,
                        ResolvePow2Primitive(args.key.useExp2));
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V1, Pipe::Vector);
        args.sync->Local(LocalDependency::VectorToMte3Outputs, Stage::V1);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V1, Pipe::Mte3);

        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        if (IsTwoByteGateStorage(args.key.gateStorage)) {
            for (const CopyRegion &copy : V1Gate2BLayout::kScoreWriteback) {
                const Region source{copy.source, copy.size};
                args.ops->Store(
                    Stage::V1,
                    detail::UbMainSpan(head, "score-source-2b", source,
                                       localGeneration),
                    detail::Subspan(payload, "packed-score", copy.destination,
                                    copy.size));
            }
        } else {
            for (const CopyRegion &copy : V1GateFp32Layout::kScoreWriteback) {
                const Region source{copy.source, copy.size};
                args.ops->Store(
                    Stage::V1,
                    detail::UbMainSpan(head, "score-source-fp32", source,
                                       localGeneration),
                    detail::Subspan(payload, "packed-score", copy.destination,
                                    copy.size));
            }
        }
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V1, Pipe::Mte3);
        // 一个回写阶段恰好包含两个固定的 MTE3 区域，并非一次连续 DataCopy。
        // 随后发布的三个同步状态均位于最后一个源读取方之后，且各自由指定所有者消费一次。
        args.sync->Set(SyncPoint::V1MainSourceFree, head.localBankId,
                       localGeneration, Stage::V1, Pipe::Mte3);
        args.sync->Set(SyncPoint::C2RawDstFree, head.localBankId,
                       localGeneration, Stage::V1, Pipe::Mte3);
        args.sync->Set(SyncPoint::V1ScoreReady, head.workspaceSlot,
                       workspaceGeneration, Stage::V1, Pipe::Mte3);
    }
}

inline void RunV3(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        const Offset validRows = args.work->group.chunk.validRows;
        args.sync->Wait(SyncPoint::C2ScorePayloadFree, head.workspaceSlot,
                        workspaceGeneration, Stage::V3, Pipe::Mte3);
        args.sync->Wait(SyncPoint::C2RawReady, head.localBankId,
                        localGeneration,
                        Stage::V3, Pipe::Vector);
        const SymbolicMutexId ubMutex =
            Arch35VectorMutexIds::UbBank(head.aivLocalSlot);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V3, Pipe::Vector);
        // 仅调用一次；detail::V3OneVf 还接收 ABI、AkkStorage、InputStorage 和
        // 显式运行时缩放。缩放仅对 Aqk 应用一次。
        // 原始数据读取方仅访问 C2 定义的有效因果域；其余输出通道均不读取
        // 原始 UB，直接生成结果。
        args.ops->RunVf(Stage::V3, head, args.scale, RuntimeScaleUse::Aqk,
                        1U);
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V3, Pipe::Vector);
        args.sync->Local(LocalDependency::VectorToMte3Outputs, Stage::V3);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V3, Pipe::Mte3);
        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        const bool hasQ10 = validRows > Akk2BPackPolicy::kQuadrantRows;
        if (hasQ10) {
            args.ops->Store(
                Stage::V3,
                detail::UbMainSpan(head, "X0", V3Layout::kX0,
                                   localGeneration),
                detail::Subspan(payload, "X0-fp32", 0x0000U, 0x1000U));
            args.ops->Store(
                Stage::V3,
                detail::UbMainSpan(head, "X1", V3Layout::kX1,
                                   localGeneration),
                detail::Subspan(payload, "X1-fp32", 0x1000U, 0x1000U));
            args.ops->Store(
                Stage::V3,
                detail::UbMainSpan(head, "B", V3Layout::kB,
                                   localGeneration),
                detail::Subspan(payload, "B-fp32", 0x2000U, 0x1000U));
        }
        if (args.key.abi == PrepareAbi::Fused &&
            args.key.akkStorage == AkkStorage::TwoByteAbi) {
            args.ops->Store(Stage::V3,
                            detail::UbMainSpan(head, "X0-tau", V3Layout::kX0Tau,
                                               localGeneration),
                            detail::Subspan(payload, "X0-tau", 0x3000U, 0x0800U));
            if (hasQ10) {
                args.ops->Store(
                    Stage::V3,
                    detail::UbMainSpan(head, "X1-tau", V3Layout::kX1Tau,
                                       localGeneration),
                    detail::Subspan(payload, "X1-tau", 0x3800U, 0x0800U));
            }
        }
        // Aqk 与可选 AkkOut 使用相互独立的公开 GM 地址。其 ABI 偏移和类型转换
        // 刻意保持符号化，但归还本地主计算区和向量状态与临时区所有权前
        // 必须包含对应的 MTE3 搬运。
        if (validRows != 0U) {
            args.ops->Store(
                Stage::V3,
                detail::UbMainMatrixSpan(
                    head, "Aqk-valid-ld64", V3Layout::kRawAqk.offset, 0U,
                    0U, validRows, ShapePolicy::kBt, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes, localGeneration),
                detail::SymbolicGmMatrixSpan(
                    head, "Aqk-output-valid-ld64", 0U, 0U, validRows,
                    ShapePolicy::kBt, ShapePolicy::kBt,
                    ShapePolicy::kStorageBytes, workspaceGeneration));
        }
        if (args.key.abi == PrepareAbi::Current &&
            args.key.akkStorage == AkkStorage::TwoByteAbi) {
            constexpr Offset kQuadrant = 32U;
            const Offset top = std::min(validRows, kQuadrant);
            const Offset bottom = validRows > kQuadrant
                                      ? validRows - kQuadrant
                                      : 0U;
            if (top != 0U) {
                args.ops->Store(
                    Stage::V3,
                    detail::UbMainMatrixSpan(
                        head, "Akk-q00-valid-ld32",
                        V3Layout::kX0Tau.offset, 0U, 0U, top, kQuadrant,
                        kQuadrant, ShapePolicy::kStorageBytes,
                        localGeneration),
                    detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q00-valid-ld64", 0U, 0U, top,
                        kQuadrant, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration));
                args.ops->Store(
                    Stage::V3,
                    detail::UbMainMatrixSpan(
                        head, "Akk-q01-valid-ld32",
                        V3Layout::kQ01Zero.offset, 0U, 0U, top, kQuadrant,
                        kQuadrant, ShapePolicy::kStorageBytes,
                        localGeneration),
                    detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q01-valid-ld64", 0U, kQuadrant, top,
                        kQuadrant, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration));
            }
            if (bottom != 0U) {
                args.ops->Store(
                    Stage::V3,
                    detail::UbMainMatrixSpan(
                        head, "Akk-q11-valid-ld32",
                        V3Layout::kX1Tau.offset, 0U, 0U, bottom, kQuadrant,
                        kQuadrant, ShapePolicy::kStorageBytes,
                        localGeneration),
                    detail::SymbolicGmMatrixSpan(
                        head, "AkkOut-q11-valid-ld64", kQuadrant,
                        kQuadrant, bottom, kQuadrant, ShapePolicy::kBt,
                        ShapePolicy::kStorageBytes, workspaceGeneration));
            }
        }
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V3, Pipe::Mte3);
        // Fused 发布紧凑稳定象限；Current 发布唯一一次公开 AkkOut 搬运。
        // 两条路径都必须在最后一个 MTE3 源读取方完成后再通知 C4。
        if (args.key.abi == PrepareAbi::Fused ||
            args.key.akkStorage == AkkStorage::TwoByteAbi) {
            args.sync->Set(SyncPoint::V3VcsReady, head.workspaceSlot,
                           workspaceGeneration, Stage::V3, Pipe::Mte3);
        }
    }
}

inline void RunV6(const VectorStageArgs &args)
{
    if (!detail::ValidArgs(args)) {
        return;
    }
    for (const HeadTask &head : args.work->group.heads) {
        if (!detail::IsOwnedActiveHead(head, args)) {
            continue;
        }
        const std::uint64_t workspaceGeneration = head.workspaceGeneration;
        const std::uint64_t localGeneration = head.localGeneration;
        const Offset validRows = args.work->group.chunk.validRows;
        const Offset tokenStorageBytes =
            validRows * ShapePolicy::kV * ShapePolicy::kStorageBytes;
        const Offset gBytes =
            validRows * ShapePolicy::kK * ShapePolicy::kFp32Bytes;
        // C4PayloadFree 是跨核前置；V0/V3 对同一 UB 槽的本核释放由 ubMutex
        // 在 MTE3 -> MTE2 之间直接串接，不再占用额外 flag。
        args.sync->Wait(SyncPoint::C4PayloadFree, head.workspaceSlot,
                        workspaceGeneration, Stage::V6, Pipe::Mte2);
        const SymbolicMutexId ubMutex =
            Arch35VectorMutexIds::UbBank(head.aivLocalSlot);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V6, Pipe::Mte2);

        const BufferSpan context = args.workspace->Span(
            WorkspaceRegion::Context, head.workspaceSlot,
            workspaceGeneration);
        // 每个映射到该缓存的 HV 都直接从所有者副本重新装载 Qhat/Khat。C7 在
        // 释放缓存前汇聚所有映射头发布的 V6RhsReady；非所有者上下文
        // 的 Q/K 区间从不物化。
        const BufferSpan qkContext = args.workspace->Span(
            WorkspaceRegion::Context, head.qkCacheSlot,
            head.qkCacheGeneration);
        args.ops->Load(
            Stage::V6,
            detail::Subspan(qkContext, "qhat-HK-cache", 0x0000U,
                            tokenStorageBytes),
            detail::UbMainSpan(
                head, "qhat-to-qg",
                {V6Layout::kQHatToQg.offset, tokenStorageBytes},
                localGeneration));
        args.ops->Load(
            Stage::V6,
            detail::Subspan(qkContext, "khat-HK-cache", 0x4000U,
                            tokenStorageBytes),
            detail::UbMainSpan(
                head, "khat-to-kg",
                {V6Layout::kKHatToKg.offset, tokenStorageBytes},
                localGeneration));
        const BufferSpan gSource =
            args.key.abi == PrepareAbi::Current
                ? detail::SymbolicGmSpan(
                      head, "gk-output-reuse", gBytes,
                      workspaceGeneration)
                : detail::Subspan(context, "G-context", 0x8200U, gBytes);
        args.ops->Load(Stage::V6, gSource,
                       detail::UbMainSpan(
                           head, "G",
                           {V6Layout::kGInput.offset, gBytes},
                           localGeneration));
        args.ops->Load(Stage::V6,
                       detail::SymbolicGmSpan(head, "V", tokenStorageBytes,
                                              workspaceGeneration),
                       detail::UbMainSpan(
                           head, "V-to-Vbeta",
                           {V6Layout::kVToVBeta.offset, tokenStorageBytes},
                           localGeneration));
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V6, Pipe::Mte2);
        args.sync->Local(LocalDependency::Mte2ToVectorInputs, Stage::V6);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V6, Pipe::Vector);
        // 仅调用一次；主机分发特化 detail::V6OneVf<useExp2>，并传入相互
        // 独立的 Q/K 存储类型、V 存储类型及运行时缩放。Current 在 V6
        // 不执行缩放乘法；Fused 在 qg 第一次舍入后恰好应用一次。两个直接
        // 2^x 计算点和所有存储舍入点都在此执行。
        args.ops->RunVf(Stage::V6, head,
                        ResolvePow2Primitive(args.key.useExp2), args.scale,
                        args.key.abi == PrepareAbi::Fused
                            ? RuntimeScaleUse::FusedQg
                            : RuntimeScaleUse::None,
                        args.key.abi == PrepareAbi::Fused ? 1U : 0U);
        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V6, Pipe::Vector);
        args.sync->Local(LocalDependency::VectorToMte3Outputs, Stage::V6);
        args.sync->MutexLock(MutexResource::AivUbBank, ubMutex,
                             head.localBankId, Stage::V6, Pipe::Mte3);

        const BufferSpan payload = args.workspace->Span(
            WorkspaceRegion::SharedPayload, head.workspaceSlot,
            workspaceGeneration);
        const Offset rhsRows =
            validRows > Akk2BPackPolicy::kQuadrantRows
                ? ShapePolicy::kBt
                : Akk2BPackPolicy::kQuadrantRows;
        const Offset rhsPlaneBytes =
            rhsRows * ShapePolicy::kK * ShapePolicy::kStorageBytes;
        // 各平面基址保持固定。仅上半区有效的尾块在每个平面只搬运 32 个
        // 物理行；完整尾块搬运全部 64 行。
        args.ops->Store(
            Stage::V6,
            detail::UbMainSpan(
                head, "K-beta-g",
                {V6Layout::kKBetaG.offset, rhsPlaneBytes}, localGeneration),
            detail::Subspan(payload, "K-beta-g", 0x0000U,
                            rhsPlaneBytes));
        args.ops->Store(
            Stage::V6,
            detail::UbMainSpan(
                head, "V-beta",
                {V6Layout::kVToVBeta.offset, rhsPlaneBytes},
                localGeneration),
            detail::Subspan(payload, "V-beta", 0x4000U, rhsPlaneBytes));

        if (args.key.abi == PrepareAbi::Current) {
            args.ops->Store(
                Stage::V6,
                detail::UbMainSpan(
                    head, "qg",
                    {V6Layout::kQHatToQg.offset, tokenStorageBytes},
                    localGeneration),
                detail::SymbolicGmSpan(head, "qg-output", tokenStorageBytes,
                                       workspaceGeneration));
        } else {
            args.ops->Store(
                Stage::V6,
                detail::UbMainSpan(
                    head, "Qg-scaled",
                    {V6Layout::kQgScaled.offset, tokenStorageBytes},
                    localGeneration),
                detail::SymbolicGmSpan(head, "Qg-scaled-output",
                                       tokenStorageBytes,
                                       workspaceGeneration));
        }
        args.ops->Store(
            Stage::V6,
            detail::UbMainSpan(
                head, "kg",
                {V6Layout::kKHatToKg.offset, tokenStorageBytes},
                localGeneration),
            detail::SymbolicGmSpan(head, "kg-output-or-handoff",
                                   tokenStorageBytes,
                                   workspaceGeneration));

        args.sync->MutexUnlock(MutexResource::AivUbBank, ubMutex,
                               head.localBankId, Stage::V6, Pipe::Mte3);

        args.sync->Set(SyncPoint::V6RhsReady, head.workspaceSlot,
                       workspaceGeneration, Stage::V6, Pipe::Mte3);
    }
}

} // namespace arch35
} // namespace kda_prepare_pseudocode

#endif // FLA_OPS_ASCENDC_KDA_CHUNK_KDA_FWD_PREPARE_PSEUDOCODE_ARCH35_VEC_H
