// Standalone CPU check: no CANN headers or NPU execution required.
#include "../op_host/fused_recurrent_rwkv8_limits.h"

using FusedRecurrentRwkv8::IsValidChunkLen;
static_assert(IsValidChunkLen(1), "minimum chunk length");
static_assert(IsValidChunkLen(16), "default chunk length");
static_assert(IsValidChunkLen(4294967295LL), "maximum uint32 chunk length");
static_assert(!IsValidChunkLen(0), "zero must not reach kernel division");
static_assert(!IsValidChunkLen(-1), "negative chunk length");
static_assert(!IsValidChunkLen(4294967296LL), "uint32 wraparound to zero");
static_assert(!IsValidChunkLen(4294967297LL), "uint32 wraparound to one");
static_assert(!IsValidChunkLen(9223372036854775807LL), "int64 maximum");

int main() { return 0; }
