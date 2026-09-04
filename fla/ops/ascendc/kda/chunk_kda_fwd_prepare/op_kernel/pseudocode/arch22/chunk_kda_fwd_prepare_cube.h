/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H
#define PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H

#include "../chunk_kda_fwd_prepare_struct.h"

namespace kda_prepare_pseudocode::arch22 {

inline constexpr bool kCubeSupported = false;

// Arch22 is intentionally unavailable. Deleted declarations preserve the
// mirrored file shape without providing a fallback implementation.
void RunC2(const CubeStageArgs &) = delete;
void RunC4(const CubeStageArgs &) = delete;
void RunC5(const CubeStageArgs &) = delete;
void RunC7(const CubeStageArgs &) = delete;

} // namespace kda_prepare_pseudocode::arch22

#endif // PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H
