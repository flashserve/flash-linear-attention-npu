// Canonical aclnn route contract for ChunkGatedDeltaRuleBwdDhu.
// Runtime inputs are defined by tests/op_cases/chunk_gated_delta_rule_bwd_dhu.json.
#include "aclnn_chunk_gated_delta_rule_bwd_dhu.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkGatedDeltaRuleBwdDhu;
} // namespace
