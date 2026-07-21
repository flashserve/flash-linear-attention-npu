// Canonical aclnn route contract for ChunkBwdDvLocal.
// Runtime inputs are defined by tests/op_cases/chunk_bwd_dv_local.json.
#include "aclnn_chunk_bwd_dv_local.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkBwdDvLocalGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkBwdDvLocal;
} // namespace
