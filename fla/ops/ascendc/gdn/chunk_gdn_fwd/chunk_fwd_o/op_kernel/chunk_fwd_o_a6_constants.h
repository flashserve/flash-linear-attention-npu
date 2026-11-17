/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A6 tile constants shared by host tiling and device kernel.
 */

#ifndef CHUNK_FWD_O_A6_CONSTANTS_H
#define CHUNK_FWD_O_A6_CONSTANTS_H

#include <cstdint>

namespace GDN {

constexpr int64_t CHUNK_FWD_O_A6_BT = 64;
constexpr int64_t CHUNK_FWD_O_A6_K = 128;
constexpr int64_t CHUNK_FWD_O_A6_V = 128;

constexpr uint32_t CHUNK_FWD_O_A6_HEAD_COUNT = 3U;

} // namespace GDN

#endif // CHUNK_FWD_O_A6_CONSTANTS_H
