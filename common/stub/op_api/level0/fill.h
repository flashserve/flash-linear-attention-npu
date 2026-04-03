/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file fill.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_OP_FILL_OP_H_
#define OP_API_INC_LEVEL0_OP_FILL_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *Fill(const aclTensor *dims, const aclTensor *value, const aclIntArray *outShape,
    aclOpExecutor *executor);
}

#endif
