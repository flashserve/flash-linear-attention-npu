// Canonical aclnn route contract for ChunkScaledDotKkt.
// Runtime inputs are defined by tests/op_cases/chunk_scaled_dot_kkt.json.
#include "aclnn_chunk_scaled_dot_kkt.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkScaledDotKktGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkScaledDotKkt;
} // namespace
