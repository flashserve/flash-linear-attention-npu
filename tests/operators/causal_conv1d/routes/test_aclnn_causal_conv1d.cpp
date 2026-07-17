// Canonical aclnn route contract for CausalConv1d.
// Runtime inputs are defined by tests/op_cases/causal_conv1d.json.
#include "aclnn_causal_conv1d.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnCausalConv1dGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnCausalConv1d;
} // namespace
