#ifndef CHUNK_SCALED_DOT_KKT_COMMON_H
#define CHUNK_SCALED_DOT_KKT_COMMON_H

#include <cstdint>

namespace NsChunkScaledDotKkt {
constexpr int32_t SCORE_WORKSPACE_BUFFER_NUM = 3;
constexpr int32_t SCORE_WORKSPACE_HEAD_BATCH = 8;
constexpr int32_t SCORE_ROW_BLOCK_A2 = 16;
constexpr int32_t SCORE_ROW_BLOCK_A5_BT64 = 64;
constexpr int32_t SCORE_ROW_BLOCK_A5_BT128 = 128;
constexpr int32_t CATLASS_SCORE_MIN_BT = 16;
}  // namespace NsChunkScaledDotKkt

#endif  // CHUNK_SCALED_DOT_KKT_COMMON_H
