// Canonical aclnn route contract for SolveTri.
// Runtime inputs are defined by tests/op_cases/solve_tri.json.
#include "aclnn_solve_tri.h"

namespace {
[[maybe_unused]] auto *const kGetWorkspace = &aclnnSolveTriGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnSolveTri;
} // namespace
