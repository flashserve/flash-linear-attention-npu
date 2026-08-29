/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_FWD_H_TILING_KEY_H
#define CHUNK_FWD_H_TILING_KEY_H

#include <cstdint>

namespace GDN {

constexpr uint32_t FWD_H_TILING_KEY = 1;
constexpr int64_t FWD_H_DTYPE_BF16 = 1;
constexpr int64_t FWD_H_DTYPE_FP32 = 2;

enum class FwdHGateMode : uint8_t {
    SCALAR_G = 0,
    KEY_GK = 1,
};

template <FwdHGateMode GATE_MODE_, bool USE_EXP2_, bool STATE_FP32_>
struct FwdHCompilePolicy {
    static constexpr FwdHGateMode GATE_MODE = GATE_MODE_;
    static constexpr bool USE_EXP2 = USE_EXP2_;
    static constexpr bool STATE_FP32 = STATE_FP32_;
};

} // namespace GDN

#endif // CHUNK_FWD_H_TILING_KEY_H
