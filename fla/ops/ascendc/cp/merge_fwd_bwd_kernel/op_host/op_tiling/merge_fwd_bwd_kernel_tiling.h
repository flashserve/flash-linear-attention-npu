/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef MERGE_FWD_BWD_KERNEL_TILING_H
#define MERGE_FWD_BWD_KERNEL_TILING_H

#include "register/op_impl_registry.h"

namespace optiling {
struct MergeFwdBwdKernelCompileInfo {
    uint32_t aicNum = 28;
    uint32_t aivNum = 56;
};
ge::graphStatus Tiling4MergeFwdBwdKernel(gert::TilingContext *context);
ge::graphStatus TilingParse4MergeFwdBwdKernel(gert::TilingParseContext *context);
} // namespace optiling

#endif
