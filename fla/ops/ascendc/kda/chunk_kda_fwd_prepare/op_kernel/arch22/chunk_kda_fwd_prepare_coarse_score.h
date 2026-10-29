/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef ARCH22_CHUNK_KDA_FWD_PREPARE_COARSE_SCORE_H
#define ARCH22_CHUNK_KDA_FWD_PREPARE_COARSE_SCORE_H

#include "../chunk_kda_fwd_prepare_runtime_policy.h"

namespace KdaPrepare::Arch22::CoarseScore {
// Two gate references serve four fine16 score bands. Unused payload space
// preserves both references before V1 reuses its current FP32 G tile.
constexpr uint32_t kAnchorRows = 32;
constexpr uint32_t kAnchors = 2;
constexpr uint32_t kMinusOffset[2] = {0x8000, 0xA000};
constexpr uint32_t kPrefixRows[2] = {32, 64};
constexpr uint32_t kPayloadBytes = 0xE000;
constexpr uint32_t kReferences = kPayloadBytes;
constexpr uint32_t kReferenceBytes = kAnchors * Shape::kHeadDim * sizeof(float);
static_assert(kMinusOffset[0] + 32 * Shape::kHeadDim * sizeof(uint16_t) == kMinusOffset[1]);
static_assert(kMinusOffset[1] + 64 * Shape::kHeadDim * sizeof(uint16_t) == kPayloadBytes);
static_assert(kReferences + kReferenceBytes <= Shape::kScorePayloadBytes);

template <typename Policy, typename Args>
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline bool Eligible(const Args &args)
{
    return !args.tiling.inputSequenceMajor &&
        args.tiling.qkHeadNum != 0 &&
        args.tiling.valueHeadNum >= args.tiling.qkHeadNum &&
        args.tiling.valueHeadNum % args.tiling.qkHeadNum == 0 &&
        RuntimeMath<Policy>::Norm(args.tiling) == QkNormMode::L2 &&
        args.tiling.gateMode == static_cast<uint32_t>(GateMode::SafeSigmoid) &&
        (RuntimeMath<Policy>::Output(args.tiling) == OutputMode::Recompute ||
         RuntimeMath<Policy>::Output(args.tiling) == OutputMode::Save) &&
        args.tiling.lowerBound >= -5.0F &&
        args.tiling.lowerBound < 0.0F;
}
} // namespace KdaPrepare::Arch22::CoarseScore
#endif
