/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "chunk_kda_fwd_prepare_policy.h"
#include "chunk_kda_fwd_prepare_tiling_key.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

template <typename Policy>
float AccumulateGateSteps(const float *naturalSteps, uint32_t stepCount)
{
    using Domain = KdaPrepare::ExpDomainTraits<Policy::useExp2>;
    float storedExponent = 0.0F;
    for (uint32_t i = 0; i < stepCount; ++i) {
        storedExponent += naturalSteps[i] * Domain::stepScale;
    }
    return storedExponent;
}

bool NearlyEqual(float lhs, float rhs)
{
    const float scale = std::fabs(lhs) > std::fabs(rhs) ? std::fabs(lhs)
                                                        : std::fabs(rhs);
    return std::fabs(lhs - rhs) <= scale * 8.0e-6F + 1.0e-7F;
}

template <typename Exp2Policy, typename ExpPolicy>
bool CheckExpDomainPair()
{
    using Exp2Domain = KdaPrepare::ExpDomainTraits<Exp2Policy::useExp2>;
    using ExpDomain = KdaPrepare::ExpDomainTraits<ExpPolicy::useExp2>;

    // PrecomputedStep 是尚未累计的自然对数 step；这些前缀同时覆盖负值和正值，
    // 且不触发截断，能够识别漏掉 stepScale 或 expInputScale 的分支。
    constexpr float kNaturalSteps[] = {-0.75F, 1.25F, -0.25F, -0.5F};
    float expectedNatural = 0.0F;
    for (uint32_t i = 0;
         i < sizeof(kNaturalSteps) / sizeof(kNaturalSteps[0]); ++i) {
        expectedNatural += kNaturalSteps[i];
        const float exp2Stored =
            AccumulateGateSteps<Exp2Policy>(kNaturalSteps, i + 1);
        const float expStored =
            AccumulateGateSteps<ExpPolicy>(kNaturalSteps, i + 1);
        if (!NearlyEqual(Exp2Domain::ToExpInput(exp2Stored), expectedNatural) ||
            !NearlyEqual(ExpDomain::ToExpInput(expStored), expectedNatural)) {
            return false;
        }
    }

    struct Base2Bounds {
        float lower;
        float upper;
    };
    constexpr Base2Bounds kBounds[] = {
        {KdaPrepare::ExpDomain::kV1Bf16LowerBase2,
         KdaPrepare::ExpDomain::kV1Bf16UpperBase2},
        {KdaPrepare::ExpDomain::kV6LowerBase2,
         KdaPrepare::ExpDomain::kV6UpperBase2},
    };
    for (const auto &bounds : kBounds) {
        const float base2Inputs[] = {
            bounds.lower - 1.0F, bounds.lower, -1.25F,
            0.75F, bounds.upper, bounds.upper + 1.0F};
        for (float base2Input : base2Inputs) {
            const float expectedBase2 =
                base2Input < bounds.lower
                    ? bounds.lower
                    : (base2Input > bounds.upper ? bounds.upper : base2Input);
            const float exp2Stored =
                Exp2Domain::ClampStored(base2Input, bounds.lower, bounds.upper);
            const float expStored = ExpDomain::ClampStored(
                base2Input * KdaPrepare::ExpDomain::kLn2,
                bounds.lower, bounds.upper);
            const float expectedExpInput =
                expectedBase2 * KdaPrepare::ExpDomain::kLn2;
            if (!NearlyEqual(exp2Stored, expectedBase2) ||
                !NearlyEqual(expStored, expectedExpInput) ||
                !NearlyEqual(Exp2Domain::ToExpInput(exp2Stored),
                             expectedExpInput) ||
                !NearlyEqual(ExpDomain::ToExpInput(expStored),
                             expectedExpInput)) {
                return false;
            }
        }
    }
    return true;
}

float L2NormalizationScale(const float *values, uint32_t count, float epsilon)
{
    float squareSum = 0.0F;
    for (uint32_t i = 0; i < count; ++i) {
        squareSum += values[i] * values[i];
    }
    return 1.0F / std::sqrt(squareSum + epsilon);
}

float RoundToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool CheckArch22QgScaledOverlay()
{
    constexpr uint32_t kGRowBytes =
        KdaPrepare::Shape::kHeadDim * sizeof(float);
    constexpr uint32_t kQgScaledRowBytes =
        KdaPrepare::Shape::kHeadDim * 2U;
    for (uint32_t validRows = 1; validRows <= KdaPrepare::Shape::kChunkRows;
         ++validRows) {
        const uint32_t gLastBegin = (validRows - 1U) * kGRowBytes;
        for (uint32_t row = 0; row < validRows; ++row) {
            const uint32_t qgScaledEnd = (row + 1U) * kQgScaledRowBytes;
            const uint32_t nextGRowBegin = (row + 1U) * kGRowBytes;
            if (qgScaledEnd > nextGRowBegin ||
                (row + 1U < validRows && qgScaledEnd > gLastBegin)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main()
{
    using namespace KdaPrepare;

    using Exp2Policy = PrepareCompilePolicy<
        QkNormMode::L2, BetaMode::Sigmoid, GateMode::PrecomputedStep,
        true, false>;
    using ExpPolicy = PrepareCompilePolicy<
        QkNormMode::L2, BetaMode::Sigmoid, GateMode::PrecomputedStep,
        false, false>;
    static_assert(Exp2Policy::gateMode == GateMode::PrecomputedStep);
    static_assert(ExpPolicy::gateMode == GateMode::PrecomputedStep);
    if (!Exp2Policy::useExp2 || ExpPolicy::useExp2) {
        return 1;
    }

    if (!CheckExpDomainPair<Exp2Policy, ExpPolicy>()) {
        return 2;
    }

    const uint32_t kMinusBytes =
        Shape::kKMinusBytes[0] + Shape::kKMinusBytes[1] +
        Shape::kKMinusBytes[2] + Shape::kKMinusBytes[3];
    if (kMinusBytes != 40 * 1024 ||
        Shape::kScorePayloadBytes != 72 * 1024 ||
        ScorePayload::kQPlus != 0 ||
        ScorePayload::kKPlus != Shape::kBf16MatrixBytes ||
        ScorePayload::kKMinus[0] != 2 * Shape::kBf16MatrixBytes ||
        ScorePayload::kKMinus[0] + Shape::kKMinusBytes[0] !=
            ScorePayload::kKMinus[1] ||
        ScorePayload::kKMinus[1] + Shape::kKMinusBytes[1] !=
            ScorePayload::kKMinus[2] ||
        ScorePayload::kKMinus[2] + Shape::kKMinusBytes[2] !=
            ScorePayload::kKMinus[3] ||
        ScorePayload::kKMinus[3] + Shape::kKMinusBytes[3] !=
            Shape::kScorePayloadBytes) {
        return 3;
    }

    if (Arch35Ub::kComputeSlotBase[1] + Arch35Ub::kComputeSlotBytes !=
            Arch35Ub::kStateBase[0] ||
        Arch35Ub::kStateBase[1] + Arch35Ub::kStateBytes !=
            Arch35Ub::kCapacity ||
        Shape::kRstdBytes != Shape::kChunkRows * sizeof(float) ||
        Arch35Ub::kQRstd + Shape::kRstdBytes != Arch35Ub::kKRstd ||
        Arch35Ub::kKRstd + Shape::kRstdBytes != Arch35Ub::kGLast ||
        Arch35Ub::kRawScore + 20 * 1024 != Arch35Ub::kAqk ||
        Arch35Ub::kAqk + Shape::kBf16MatrixBytes != Arch35Ub::kLkk ||
        Arch35Ub::kLkk + Shape::kScoreMatrixBytes != Arch35Ub::kB ||
        Arch35Ub::kB + Shape::kQuadrantFp32Bytes != Arch35Ub::kX0 ||
        Arch35Ub::kX0 + Shape::kQuadrantFp32Bytes != Arch35Ub::kX1 ||
        Arch35Ub::kX1 + Shape::kQuadrantFp32Bytes != Arch35Ub::kNegX1 ||
        Arch35Ub::kNegX1 + Shape::kQuadrantFp32Bytes !=
            Arch35Ub::kAkkPack ||
        Arch35Ub::kAkkPack + 4 * Shape::kQuadrantBf16Bytes >
            Arch35Ub::kComputeSlotBytes) {
        return 4;
    }

    if (Arch35Ub::kQg != 0 ||
        Arch35Ub::kQg + Shape::kBf16MatrixBytes != Arch35Ub::kKg ||
        Arch35Ub::kKg + Shape::kBf16MatrixBytes != Arch35Ub::kVBeta ||
        Arch35Ub::kVBeta + Shape::kBf16MatrixBytes !=
            Arch35Ub::kGForPost ||
        Arch35Ub::kGForPost + Shape::kGateMatrixBytes !=
            Arch35Ub::kKBetaG ||
        Arch35Ub::kKBetaG + Shape::kBf16MatrixBytes !=
            Arch35Ub::kQgScaled ||
        Arch35Ub::kQgScaled + Shape::kBf16MatrixBytes !=
            Arch35Ub::kComputeSlotBytes) {
        return 5;
    }

    if (2 * Arch22Ub::kPrivateBytes + 40 * 1024 !=
            Arch22Ub::kUsableBytes ||
        Arch22Ub::kV6QgScaled != Arch22Ub::kSharedG ||
        Arch22Ub::kV6QgScaled + Shape::kBf16MatrixBytes >
            Arch22Ub::kSharedScratch ||
        Arch22Ub::kQRstd + Shape::kRstdBytes != Arch22Ub::kKRstd ||
        Arch22Ub::kKRstd + Shape::kRstdBytes > Arch22Ub::kGLast ||
        !CheckArch22QgScaledOverlay() ||
        Arch22Ub::kKMinus[3] + Shape::kKMinusBytes[3] !=
            Arch22Ub::kPrivateBytes ||
        L1::kHeadLane[3] + L1::kHeadLaneBytes != L1::kX0 ||
        L1::kAkk + 4 * L1::kAkkStride != L1::kPeak ||
        L1::kAkkQ10Elements != 32 * 16 ||
        L1::kPeak > L1::kCapacity) {
        return 6;
    }

    if (Workspace::kPayload + Shape::kScorePayloadBytes !=
            Workspace::kSlotStride ||
        Workspace::kRawScore + 20 * 1024 > Shape::kScorePayloadBytes ||
        Workspace::kB + Shape::kQuadrantFp32Bytes > Workspace::kTArch22 ||
        Workspace::kTArch22 + Shape::kQuadrantFp32Bytes > Workspace::kAkk ||
        Workspace::kAkk + 4 * Shape::kQuadrantBf16Bytes >
            Shape::kScorePayloadBytes ||
        Workspace::kKBetaG + Shape::kBf16MatrixBytes !=
            Workspace::kVBeta ||
        Workspace::kVBeta + Shape::kBf16MatrixBytes >
            Shape::kScorePayloadBytes) {
        return 7;
    }

    constexpr float kEpsilon = 1.0e-6F;
    constexpr float kZero[] = {0.0F, 0.0F, 0.0F, 0.0F};
    const float zeroScale = L2NormalizationScale(
        kZero, sizeof(kZero) / sizeof(kZero[0]), kEpsilon);
    if (!std::isfinite(zeroScale) || !NearlyEqual(zeroScale, 1000.0F) ||
        !NearlyEqual(kZero[0] * zeroScale, 0.0F)) {
        return 8;
    }

    // 小范数必须在平方和上加 epsilon，不能退化为 max(norm, epsilon)。
    constexpr float kSmallNorm[] = {1.0e-7F, 0.0F, 0.0F, 0.0F};
    const float smallScale = L2NormalizationScale(
        kSmallNorm, sizeof(kSmallNorm) / sizeof(kSmallNorm[0]), kEpsilon);
    const float smallNormalized = kSmallNorm[0] * smallScale;
    const float expectedSmall =
        kSmallNorm[0] / std::sqrt(kSmallNorm[0] * kSmallNorm[0] + kEpsilon);
    const float legacySmall = kSmallNorm[0] / kEpsilon;
    if (!NearlyEqual(smallNormalized, expectedSmall) ||
        NearlyEqual(smallNormalized, legacySmall)) {
        return 9;
    }

    // qgScaled 必须消费已经舍入的 qg，不能把 scale 合并到第一次写回前。
    constexpr float kQgFp32 = 1.003F;
    constexpr float kScale = 0.7F;
    const float qgBf16 = RoundToBf16(kQgFp32);
    const float qgScaledBf16 = RoundToBf16(qgBf16 * kScale);
    const float mergedRoundBf16 = RoundToBf16(kQgFp32 * kScale);
    if (!NearlyEqual(qgBf16, 1.0F) ||
        !NearlyEqual(qgScaledBf16, 0.69921875F) ||
        NearlyEqual(qgScaledBf16, mergedRoundBf16)) {
        return 10;
    }

    return 0;
}
