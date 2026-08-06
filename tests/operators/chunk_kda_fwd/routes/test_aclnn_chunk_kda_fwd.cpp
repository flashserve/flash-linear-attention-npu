// Canonical aclnn route contract for ChunkKdaFwd.
// Runtime inputs are defined by tests/op_cases/chunk_kda_fwd.json.
#include "aclnn_chunk_kda_fwd.h"

#include <type_traits>

namespace {
using ExpectedGetWorkspace = aclnnStatus (*)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclIntArray *, const aclIntArray *, const char *, double, int64_t,
    bool, double, bool, bool, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, uint64_t *, aclOpExecutor **);

static_assert(
    std::is_same<decltype(&aclnnChunkKdaFwdGetWorkspaceSize), ExpectedGetWorkspace>::value,
    "aclnnChunkKdaFwd must keep output-policy booleans out of the L2 ABI");

[[maybe_unused]] auto *const kGetWorkspace = &aclnnChunkKdaFwdGetWorkspaceSize;
[[maybe_unused]] auto *const kRun = &aclnnChunkKdaFwd;
} // namespace
