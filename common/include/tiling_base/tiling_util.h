/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file tiling_util.h
 * \brief
 */

#pragma once

#include "register/op_impl_registry.h"

namespace Ops {
namespace Transformer {
namespace OpTiling {
bool IsRegbaseSocVersion(const gert::TilingParseContext* context);

bool IsRegbaseSocVersion(const gert::TilingContext* context);

const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);
} // namespace OpTiling
} // namespace Transformer
} // namespace Ops