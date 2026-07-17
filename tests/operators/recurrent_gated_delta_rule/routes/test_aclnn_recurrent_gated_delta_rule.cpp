// Canonical aclnn route contract for RecurrentGatedDeltaRule.
// Runtime inputs are defined by tests/op_cases/recurrent_gated_delta_rule.json.
#include "aclnn_recurrent_gated_delta_rule.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnRecurrentGatedDeltaRuleGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnRecurrentGatedDeltaRule;
} // namespace
