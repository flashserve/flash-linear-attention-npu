/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
#define PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H

#include <cstdint>

namespace KdaPrepare {

// 这些枚举表示正式实现应放入编译期 TilingKey 的语义轴，本文不冻结数值编码。
enum class QkNormMode : uint8_t {
    Identity,
    L2,
};

enum class BetaMode : uint8_t {
    Raw,
    Sigmoid,
    TwoSigmoid,
};

enum class GateMode : uint8_t {
    PrecomputedStep,
    Softplus,
    SafeSigmoid,
};

enum class PrepareAbi : uint8_t {
    Current,
    Fused,
};

namespace ExpDomain {
constexpr float kLn2 = 0.69314718055994530942F;
constexpr float kRcpLn2 = 1.44269504088896340736F;
constexpr float kV1Fp16LowerBase2 = -80.0F;
constexpr float kV1Fp16UpperBase2 = 80.0F;
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

// USE_EXP2 编译轴的 0/1 编码已经冻结；正式 Host Tiling 必须为两个值都生成实例。
// 其余轴尚未冻结，所以这里不伪造参数不完整的 ASCENDC_TPL_ARGS_DECL；
// 正式声明必须包含 ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1)。
template <QkNormMode NORM_MODE, BetaMode BETA_MODE, GateMode GATE_MODE,
          PrepareAbi ABI, bool USE_EXP2, bool SAFE_GATE>
struct PrepareCompilePolicy {
    static constexpr QkNormMode normMode = NORM_MODE;
    static constexpr BetaMode betaMode = BETA_MODE;
    static constexpr GateMode gateMode = GATE_MODE;
    static constexpr PrepareAbi abi = ABI;
    static constexpr bool useExp2 = USE_EXP2;
    static constexpr bool safeGate = SAFE_GATE;
};

// 正式实现中由 op_host 生成；这里只保存 shape、调度和标量参数。
// Arch22/Arch35 由 __CCE_AICORE__ 在编译期选择。
struct ChunkKdaFwdPrepareTilingData {
    uint32_t batch = 0;
    uint32_t seqNum = 0;
    uint32_t seqLen = 0;
    uint32_t qkHeadNum = 0;
    uint32_t valueHeadNum = 0;
    uint32_t totalChunks = 0;
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    float epsilon = 1.0e-6F;
    float lowerBound = -5.0F;
    float scale = 1.0F;
    bool isVarLen = false;
    bool inputSequenceMajor = false;
    bool hasDtBias = false;
};

} // namespace KdaPrepare

#endif // PSEUDOCODE_CHUNK_KDA_FWD_PREPARE_TILING_KEY_H
