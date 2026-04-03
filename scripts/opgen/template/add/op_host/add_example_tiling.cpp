/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file add_example_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "add_example/op_kernel/add_example_tiling_data.h"
#include "add_example/op_kernel/add_example_tiling_key.h"

namespace optiling {

struct AddExampleCompileInfo {};

// tiling 分发入口
static ge::graphStatus AddExampleTilingFunc(gert::TilingContext* context)
{
}

static ge::graphStatus TilingParseForAddExample([[maybe_unused]] gert::TilingParseContext* context)
{   
}

// tiling注册入口.
IMPL_OP_OPTILING(AddExample).Tiling(AddExampleTilingFunc).TilingParse<AddExampleCompileInfo>(TilingParseForAddExample);
} // namespace optiling
