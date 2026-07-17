// Canonical aclnn route contract for PrepareWyReprBwdFull.
// Runtime inputs are defined by tests/op_cases/prepare_wy_repr_bwd_full.json.
#include "aclnn_prepare_wy_repr_bwd_full.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnPrepareWyReprBwdFullGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnPrepareWyReprBwdFull;
} // namespace
