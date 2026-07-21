// Canonical aclnn route contract for KdaLayoutSwap12.
// Runtime inputs are defined by tests/op_cases/kda_layout_swap12.json.
#include "aclnn_kda_layout_swap12.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnKdaLayoutSwap12GetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnKdaLayoutSwap12;
} // namespace
