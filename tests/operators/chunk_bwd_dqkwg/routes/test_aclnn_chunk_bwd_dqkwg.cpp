// Canonical aclnn route contract for ChunkBwdDqkwg.
// Runtime inputs are defined by tests/op_cases/chunk_bwd_dqkwg.json.
#include "aclnn_chunk_bwd_dqkwg.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkBwdDqkwgGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkBwdDqkwg;
} // namespace
