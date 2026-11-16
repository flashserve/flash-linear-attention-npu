/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_kernel.h
 * \brief arch 分派：__CCE_AICORE__ == 310（A5/950）→ arch35/；否则（A2/A3）→ arch22/。
 */

#ifndef PREF_PROCESS_FWD_KERNEL_MERGED_KERNEL_H
#define PREF_PROCESS_FWD_KERNEL_MERGED_KERNEL_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/pre_process_fwd_kernel_merged_vector.h"
#include "arch35/pre_process_fwd_kernel_merged_cube.h"
#else
#include "arch22/pre_process_fwd_kernel_merged_vector.h"
#include "arch22/pre_process_fwd_kernel_merged_cube.h"
#endif

#endif  // PREF_PROCESS_FWD_KERNEL_MERGED_KERNEL_H
