/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_DETAIL_MACROS_HPP
#define CATLASS_DETAIL_MACROS_HPP

#if defined(__CCE__)
#include <kernel_operator.h>
#endif

#define CATLASS_DEVICE __forceinline__ __aicore__
#ifdef __CCE__
#define CATLASS_HOST_DEVICE __forceinline__ [host, aicore]
#else
#define CATLASS_HOST_DEVICE
#endif
#define CATLASS_GLOBAL __global__ __aicore__

#endif  // CATLASS_DETAIL_MACROS_HPP