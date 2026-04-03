/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP

#include "catlass/catlass.hpp"
namespace Catlass::Gemv::Block {

template <
    class DispatchPolicy,
    class... Args
>
struct BlockGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockGemv is not implemented for this DispatchPolicy");
};
}  // namespace Catlass::Gemv::Block

#include "catlass/gemv/block/block_gemv_aiv.hpp"
#include "catlass/gemv/block/block_gemv_aic.hpp"

#endif
