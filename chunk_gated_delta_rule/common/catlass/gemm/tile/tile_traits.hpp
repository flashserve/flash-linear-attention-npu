/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CATLASS_GEMM_PROLOGUE_TRATIS_HPP
#define CATLASS_GEMM_PROLOGUE_TRATIS_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Gemm::Tile {

struct EmptyType {};

template <class Prologue>
struct PrologueTraits : public Prologue {
    using Prologue::Prologue;

    using TensorSrc = AscendC::GlobalTensor<typename Prologue::ElementSrc>;
    using TensorDst = AscendC::GlobalTensor<typename Prologue::ElementDst>;

};

template <>
struct PrologueTraits<void> {
    using ElementSrc = EmptyType;
    using LayoutSrc = EmptyType;
    using ElementDst = EmptyType;
    using LayoutDst = EmptyType;

    using TensorSrc = EmptyType;
    using TensorDst = EmptyType;

    using Params = EmptyType;

    template <class... Args>
    CATLASS_DEVICE
    PrologueTraits(Args...) {}
};

}  //namespace Catlass::Gemm::Tile

#endif