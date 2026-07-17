// Canonical aclnn route contract for ChunkLocalCumsum.
// Runtime inputs are defined by tests/op_cases/chunk_local_cumsum.json.
#include "aclnn_chunk_local_cumsum.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkLocalCumsumGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkLocalCumsum;
} // namespace
