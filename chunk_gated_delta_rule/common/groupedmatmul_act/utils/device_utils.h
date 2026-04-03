/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file device_utils.h
 * \brief
 */

#ifndef UTILS_DEVICE_UTILS_H
#define UTILS_DEVICE_UTILS_H
#include "kernel_operator.h"
namespace Act {
namespace Gemm {

template <AscendC::HardEvent event>
__aicore__ inline void TPipeSetWaitFlag()
{
    auto eventID = GetTPipePtr()->FetchEventID(event);
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// blockAlign 32B
template <typename DataType, int64_t blockSize = 32>
__aicore__ inline int64_t AlignBlock(const int64_t& t)
{
    return AscendC::Align(t, static_cast<int64_t>(blockSize / sizeof(DataType)));
}

} // namespace Gemm
} // namespace Act
#endif