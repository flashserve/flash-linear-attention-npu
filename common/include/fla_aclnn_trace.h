#pragma once

// Compile-time diagnostics only. The default expansion has no branch, string,
// environment lookup or function call, including in optimized production builds.
#if defined(FLA_NPU_OP_TRACE) && FLA_NPU_OP_TRACE
#include <cstdio>
#define FLA_ACLNN_TRACE() \
    do { std::fprintf(stderr, "[FLA][ACLNN] %s file=%s\n", __func__, __FILE__); } while (0)
#else
#define FLA_ACLNN_TRACE() do {} while (0)
#endif
