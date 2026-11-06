#ifndef FUSED_RECURRENT_RWKV8_LIMITS_H
#define FUSED_RECURRENT_RWKV8_LIMITS_H

#include <cstdint>
#include <limits>

namespace FusedRecurrentRwkv8 {

// Public aclnn attrs are int64; the kernel tiling field is uint32.
constexpr int64_t MAX_CHUNK_LEN = std::numeric_limits<uint32_t>::max();
constexpr bool IsValidChunkLen(int64_t value)
{
    return value >= 1 && value <= MAX_CHUNK_LEN;
}

} // namespace FusedRecurrentRwkv8

#endif // FUSED_RECURRENT_RWKV8_LIMITS_H
