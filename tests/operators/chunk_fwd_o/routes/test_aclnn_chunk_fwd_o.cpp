// Canonical aclnn route contract for ChunkFwdO.
// Runtime inputs are defined by tests/op_cases/chunk_fwd_o.json.
#include "aclnn_chunk_fwd_o.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkFwdOGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkFwdO;
} // namespace
