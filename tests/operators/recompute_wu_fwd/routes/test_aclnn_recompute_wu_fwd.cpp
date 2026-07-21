// Canonical aclnn route contract for RecomputeWUFwd.
// Runtime inputs are defined by tests/op_cases/recompute_wu_fwd.json.
#include "aclnn_recompute_wu_fwd.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnRecomputeWUFwdGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnRecomputeWUFwd;
} // namespace
