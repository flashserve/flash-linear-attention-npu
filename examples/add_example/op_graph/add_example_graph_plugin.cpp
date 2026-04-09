/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file histogram_v2_graph_plugin.cpp
 * \brief histogram_v2 operater graph plugin resource
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAddExample");

    // 设置输出的dtype
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, sizeDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAddExample");
    return GRAPH_SUCCESS;
}

IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);

}; // namespace ops