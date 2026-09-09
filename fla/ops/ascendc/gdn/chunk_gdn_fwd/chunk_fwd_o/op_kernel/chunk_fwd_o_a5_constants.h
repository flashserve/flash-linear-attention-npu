/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 *
 * A5 tile constants shared by host tiling and device kernel.
 */

#ifndef CHUNK_FWD_O_A5_CONSTANTS_H
#define CHUNK_FWD_O_A5_CONSTANTS_H

#include <cstdint>

namespace GDN {

constexpr int64_t CHUNK_FWD_O_A5_BT = 64;
constexpr int64_t CHUNK_FWD_O_A5_K = 128;
constexpr int64_t CHUNK_FWD_O_A5_V = 128;

constexpr uint32_t CHUNK_FWD_O_APRIME_SLOT_BYTES =
    static_cast<uint32_t>(CHUNK_FWD_O_A5_BT * CHUNK_FWD_O_A5_BT * sizeof(uint16_t));
constexpr uint32_t CHUNK_FWD_O_APRIME_WINDOW_COUNT = 2U;
constexpr uint32_t CHUNK_FWD_O_APRIME_HEADS_PER_WINDOW = 4U;
constexpr uint32_t CHUNK_FWD_O_APRIME_SLOT_COUNT =
    CHUNK_FWD_O_APRIME_WINDOW_COUNT * CHUNK_FWD_O_APRIME_HEADS_PER_WINDOW;
constexpr uint32_t CHUNK_FWD_O_APRIME_WORKSPACE_BYTES =
    CHUNK_FWD_O_APRIME_SLOT_BYTES * CHUNK_FWD_O_APRIME_SLOT_COUNT;

} // namespace GDN

#endif // CHUNK_FWD_O_A5_CONSTANTS_H
