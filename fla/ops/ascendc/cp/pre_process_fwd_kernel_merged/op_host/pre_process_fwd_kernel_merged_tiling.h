/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_tiling.h
 * \brief Tiling data 复用 kernel 侧的纯结构体定义（保证 host / device 字段一致）。
 */

#pragma once

#include "../op_kernel/pre_process_fwd_kernel_merged_struct.h"

namespace optiling {

using GDN::PreProcessFwdKernelMergedTilingData;

struct PreProcessFwdKernelMergedCompileInfo {};

} // namespace optiling
