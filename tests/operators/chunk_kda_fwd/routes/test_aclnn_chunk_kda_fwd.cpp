// Canonical aclnn route contract for ChunkKdaFwd.
// Runtime inputs are defined by tests/op_cases/chunk_kda_fwd.json.
#include "aclnn_chunk_kda_fwd.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkKdaFwdGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkKdaFwd;
} // namespace
