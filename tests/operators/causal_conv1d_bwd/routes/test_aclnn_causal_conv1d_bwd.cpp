// Canonical aclnn route contract for CausalConv1dBwd.
// Runtime inputs are defined by tests/op_cases/causal_conv1d_bwd.json.
#include "aclnn_causal_conv1d_bwd.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnCausalConv1dBwdGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnCausalConv1dBwd;
} // namespace
