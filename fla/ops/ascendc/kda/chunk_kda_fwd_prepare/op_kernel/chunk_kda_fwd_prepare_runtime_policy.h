/** Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause */
#ifndef CHUNK_KDA_FWD_PREPARE_RUNTIME_POLICY_H
#define CHUNK_KDA_FWD_PREPARE_RUNTIME_POLICY_H

#include "chunk_kda_fwd_prepare_policy.h"

namespace KdaPrepare {
enum class PreparePipeline : uint32_t { Generic = 0, Target = 1 };

// Only physical pipeline selection is static. Math and output choices are
// immutable tiling fields, shared by both engines for the entire invocation.
struct PrepareRuntimePolicy {
    static constexpr bool commonMath = false;
    static constexpr bool arch22FastLeaf = false;
    static constexpr bool arch22CubeLeaf = false;
};

// One bounded common math contract, independent of head/token counts and gate
// transform. This avoids rebuilding the former Cartesian product internally.
struct PrepareCommonMathPolicy : PrepareRuntimePolicy {
    static constexpr bool commonMath = true;
};

template <typename Policy>
struct RuntimeMath {
    template <typename T>
#if defined(__CCE_AICORE__)
    __aicore__
#endif
    static inline QkNormMode Norm(const T &t)
    { return Policy::commonMath ? QkNormMode::L2 : static_cast<QkNormMode>(t.normMode); }
    template <typename T>
#if defined(__CCE_AICORE__)
    __aicore__
#endif
    static inline BetaMode Beta(const T &t)
    { return Policy::commonMath ? BetaMode::Sigmoid : static_cast<BetaMode>(t.betaMode); }
    template <typename T>
#if defined(__CCE_AICORE__)
    __aicore__
#endif
    static inline OutputMode Output(const T &t)
    { return Policy::commonMath ? OutputMode::Save : static_cast<OutputMode>(t.outputMode); }
};

// The specialized pipeline has an explicit identity; SAFE_GATE is no longer
// used as a hidden algorithm selector. Its supported math contract is fixed.
struct PrepareTargetPolicy : PrepareCompilePolicy<
    QkNormMode::L2, BetaMode::Sigmoid, GateMode::PrecomputedStep,
    true, false, OutputMode::Save> {
    static constexpr bool arch22FastLeaf = true;
    static constexpr bool arch22CubeLeaf = true;
};

struct RuntimeExpDomain {
    bool useExp2 = false;
    float stepScale = 1.0F;
    float expInputScale = 1.0F;

#if defined(__CCE_AICORE__)
    __aicore__
#endif
    inline void Init(bool exp2)
    {
        useExp2 = exp2;
        stepScale = exp2 ? ExpDomain::kRcpLn2 : 1.0F;
        expInputScale = exp2 ? ExpDomain::kLn2 : 1.0F;
    }

#if defined(__CCE_AICORE__)
    __aicore__
#endif
    inline float StoredBound(float base2Bound) const
    {
        return useExp2 ? base2Bound : base2Bound * ExpDomain::kLn2;
    }
};
} // namespace KdaPrepare
#endif
