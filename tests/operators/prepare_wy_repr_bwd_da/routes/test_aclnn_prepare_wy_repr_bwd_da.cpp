// Canonical aclnn route contract for PrepareWyReprBwdDa.
// Runtime inputs are defined by tests/op_cases/prepare_wy_repr_bwd_da.json.
#include "aclnn_prepare_wy_repr_bwd_da.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnPrepareWyReprBwdDaGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnPrepareWyReprBwdDa;
} // namespace
