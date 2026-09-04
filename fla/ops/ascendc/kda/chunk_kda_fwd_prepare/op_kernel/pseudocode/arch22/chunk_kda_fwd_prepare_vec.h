/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
#define PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H

#include "../chunk_kda_fwd_prepare_struct.h"

namespace kda_prepare_pseudocode::arch22 {

inline constexpr bool kVectorSupported = false;

// Arch22 is intentionally unavailable. Deleted declarations preserve the
// mirrored file shape without providing a fallback implementation.
void RunV0(const VectorStageArgs &) = delete;
void RunV1(const VectorStageArgs &) = delete;
void RunV3(const VectorStageArgs &) = delete;
void RunV6(const VectorStageArgs &) = delete;

} // namespace kda_prepare_pseudocode::arch22

#endif // PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H
