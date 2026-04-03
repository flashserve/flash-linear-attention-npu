/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_NNOPBASE_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_NNOPBASE_H

#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
aclnnStatus NnopbaseSetHcomGroup(void *const executor, char *const group);
aclnnStatus NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);
}
#endif // OPS_TRANSFORMER_DEV_TESTS_UT_OP_API_STUB_OPDEV_NNOPBASE_H