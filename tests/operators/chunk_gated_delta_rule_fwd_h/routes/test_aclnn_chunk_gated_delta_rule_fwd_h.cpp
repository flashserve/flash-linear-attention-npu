// Canonical aclnn route contract for ChunkGatedDeltaRuleFwdH.
// Runtime inputs are defined by tests/op_cases/chunk_gated_delta_rule_fwd_h.json.
#include "aclnn_chunk_gated_delta_rule_fwd_h.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkGatedDeltaRuleFwdH;
} // namespace
