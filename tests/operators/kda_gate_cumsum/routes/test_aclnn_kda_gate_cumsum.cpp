// Canonical aclnn route contract for KdaGateCumsum.
// Runtime inputs are defined by tests/op_cases/kda_gate_cumsum.json.
#include "aclnn_kda_gate_cumsum.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnKdaGateCumsumGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnKdaGateCumsum;
} // namespace
