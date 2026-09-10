/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
#define PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H

#include <cstdint>

#ifndef TORCH_MODE
#include "ascendc/host_api/tiling/template_argument.h"
#endif

#define CHUNK_KDA_FWD_PREPARE_TPL_BF16 10
#define CHUNK_KDA_FWD_PREPARE_TPL_FP32 30

#define CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY 0
#define CHUNK_KDA_FWD_PREPARE_NORM_L2 1

#define CHUNK_KDA_FWD_PREPARE_BETA_RAW 0
#define CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID 1
#define CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID 2

#define CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP 0
#define CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS 1
#define CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID 2

namespace KdaPrepare {

enum class QkNormMode : uint8_t {
    Identity = CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
    L2 = CHUNK_KDA_FWD_PREPARE_NORM_L2,
};

enum class BetaMode : uint8_t {
    Raw = CHUNK_KDA_FWD_PREPARE_BETA_RAW,
    Sigmoid = CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID,
    TwoSigmoid = CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID,
};

enum class GateMode : uint8_t {
    PrecomputedStep = CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
    Softplus = CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS,
    SafeSigmoid = CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID,
};

namespace ExpDomain {
constexpr float kLn2 = 0.69314718055994530942F;
constexpr float kRcpLn2 = 1.44269504088896340736F;
constexpr float kV1Bf16LowerBase2 = -126.0F;
constexpr float kV1Bf16UpperBase2 = 120.0F;
constexpr float kV6LowerBase2 = -80.0F;
constexpr float kV6UpperBase2 = 80.0F;
} // namespace ExpDomain

// 两套架构共同消费这一份域合同，host 测试只验证这里的纯数值选择。
template <bool USE_EXP2>
struct ExpDomainTraits {
    static constexpr bool useExp2 = USE_EXP2;
    static constexpr float stepScale = USE_EXP2 ? ExpDomain::kRcpLn2 : 1.0F;
    static constexpr float expInputScale = USE_EXP2 ? ExpDomain::kLn2 : 1.0F;

    static constexpr float StoredBound(float base2Bound)
    {
        return USE_EXP2 ? base2Bound : base2Bound * ExpDomain::kLn2;
    }

    static constexpr float ClampStored(float value, float base2Lower,
                                       float base2Upper)
    {
        const float lower = StoredBound(base2Lower);
        const float upper = StoredBound(base2Upper);
        return value < lower ? lower : (value > upper ? upper : value);
    }

    static constexpr float ToExpInput(float storedExponent)
    {
        return storedExponent * expInputScale;
    }
};

template <QkNormMode NORM_MODE, BetaMode BETA_MODE, GateMode GATE_MODE,
          bool USE_EXP2, bool SAFE_GATE>
struct PrepareCompilePolicy {
    static constexpr QkNormMode normMode = NORM_MODE;
    static constexpr BetaMode betaMode = BETA_MODE;
    static constexpr GateMode gateMode = GATE_MODE;
    static constexpr bool useExp2 = USE_EXP2;
    static constexpr bool safeGate = SAFE_GATE;
};

#ifndef TORCH_MODE
ASCENDC_TPL_ARGS_DECL(
    ChunkKdaFwdPrepare,
    ASCENDC_TPL_DTYPE_DECL(D_T_GATE, CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                           CHUNK_KDA_FWD_PREPARE_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(D_T_BETA, CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                           CHUNK_KDA_FWD_PREPARE_TPL_FP32),
    ASCENDC_TPL_UINT_DECL(NORM_MODE, 1, ASCENDC_TPL_UI_LIST,
                          CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
                          CHUNK_KDA_FWD_PREPARE_NORM_L2),
    ASCENDC_TPL_UINT_DECL(BETA_MODE, 2, ASCENDC_TPL_UI_LIST,
                          CHUNK_KDA_FWD_PREPARE_BETA_RAW,
                          CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID,
                          CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID),
    ASCENDC_TPL_UINT_DECL(GATE_MODE, 2, ASCENDC_TPL_UI_LIST,
                          CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
                          CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS,
                          CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID),
    ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1),
    ASCENDC_TPL_BOOL_DECL(SAFE_GATE, 0, 1));

// SAFE_GATE 只为 SafeSigmoid 置位，避免生成数学等价的重复实例。
#define CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                      \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, EXP_VALUE,    \
    SAFE_VALUE)                                                              \
    ASCENDC_TPL_ARGS_SEL(                                                     \
        ASCENDC_TPL_DTYPE_SEL(D_T_GATE, GATE_TYPE),                          \
        ASCENDC_TPL_DTYPE_SEL(D_T_BETA, BETA_TYPE),                          \
        ASCENDC_TPL_UINT_SEL(NORM_MODE, ASCENDC_TPL_UI_LIST, NORM_VALUE),    \
        ASCENDC_TPL_UINT_SEL(BETA_MODE, ASCENDC_TPL_UI_LIST, BETA_VALUE),    \
        ASCENDC_TPL_UINT_SEL(GATE_MODE, ASCENDC_TPL_UI_LIST, GATE_VALUE),    \
        ASCENDC_TPL_BOOL_SEL(USE_EXP2, EXP_VALUE),                           \
        ASCENDC_TPL_BOOL_SEL(SAFE_GATE, SAFE_VALUE))

#define CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                      \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, SAFE_VALUE)   \
    CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, 0,        \
        SAFE_VALUE),                                                         \
    CHUNK_KDA_FWD_PREPARE_SEL_ONE(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE, GATE_VALUE, 1,        \
        SAFE_VALUE)

#define CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                     \
    GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE)                            \
    CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE,                        \
        CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP, 0),                     \
    CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE,                        \
        CHUNK_KDA_FWD_PREPARE_GATE_SOFTPLUS, 0),                             \
    CHUNK_KDA_FWD_PREPARE_SEL_EXP(                                           \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, BETA_VALUE,                        \
        CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID, 1)

#define CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                                \
    GATE_TYPE, BETA_TYPE, NORM_VALUE)                                        \
    CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                          \
        GATE_TYPE, BETA_TYPE, NORM_VALUE, CHUNK_KDA_FWD_PREPARE_BETA_RAW),  \
    CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                          \
        GATE_TYPE, BETA_TYPE, NORM_VALUE,                                    \
        CHUNK_KDA_FWD_PREPARE_BETA_SIGMOID),                                 \
    CHUNK_KDA_FWD_PREPARE_SEL_GATE(                                          \
        GATE_TYPE, BETA_TYPE, NORM_VALUE,                                    \
        CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID)

#define CHUNK_KDA_FWD_PREPARE_SEL_NORM(GATE_TYPE, BETA_TYPE)                \
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                                     \
        GATE_TYPE, BETA_TYPE, CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY),         \
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE(                                     \
        GATE_TYPE, BETA_TYPE, CHUNK_KDA_FWD_PREPARE_NORM_L2)

#define CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(GATE_TYPE)                      \
    CHUNK_KDA_FWD_PREPARE_SEL_NORM(                                          \
        GATE_TYPE, CHUNK_KDA_FWD_PREPARE_TPL_BF16),                          \
    CHUNK_KDA_FWD_PREPARE_SEL_NORM(                                          \
        GATE_TYPE, CHUNK_KDA_FWD_PREPARE_TPL_FP32)

ASCENDC_TPL_SEL(
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(CHUNK_KDA_FWD_PREPARE_TPL_BF16),
    CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE(CHUNK_KDA_FWD_PREPARE_TPL_FP32));

#undef CHUNK_KDA_FWD_PREPARE_SEL_BETA_TYPE
#undef CHUNK_KDA_FWD_PREPARE_SEL_NORM
#undef CHUNK_KDA_FWD_PREPARE_SEL_BETA_MODE
#undef CHUNK_KDA_FWD_PREPARE_SEL_GATE
#undef CHUNK_KDA_FWD_PREPARE_SEL_EXP
#undef CHUNK_KDA_FWD_PREPARE_SEL_ONE
#endif

} // namespace KdaPrepare

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
