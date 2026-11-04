/** Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause */
#ifndef CHUNK_KDA_FWD_PREPARE_RUNTIME_POLICY_H
#define CHUNK_KDA_FWD_PREPARE_RUNTIME_POLICY_H

#include "chunk_kda_fwd_prepare_policy.h"

namespace KdaPrepare {
enum class PreparePipeline : uint32_t { Generic = 0 };

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
