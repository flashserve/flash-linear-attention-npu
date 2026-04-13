/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file tiling_templates_registry.cpp
 * \brief
 */

#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;

TilingRegistry &TilingRegistry::GetInstance()
{
    static TilingRegistry registry_impl_;
    return registry_impl_;
}

TilingRegistryNew &TilingRegistryNew::GetInstance()
{
    static TilingRegistryNew registry_impl_;
    return registry_impl_;
}
