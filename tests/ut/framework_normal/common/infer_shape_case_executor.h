/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPE_CASE_EXECUTOR_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPE_CASE_EXECUTOR_H

#include "infer_shape_context_faker.h"

void ExecuteTestCase(gert::InfershapeContextPara&             infershapeContextPara, 
                     ge::graphStatus                          expectResult = ge::GRAPH_FAILED,
                     const std::vector<std::vector<int64_t>>& expectOutputShape = {});

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPE_CASE_EXECUTOR_H