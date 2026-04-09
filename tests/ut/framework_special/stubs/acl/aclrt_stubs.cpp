/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclrt_stubs.cpp
 * \brief
 */

#include <cstdlib>
#include "securec.h"
#include "acl/acl.h"
#include "tests/utils/log.h"

extern "C" {
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    LOG_IF_EXPR(devPtr == nullptr, LOG_ERR("Invalid parameter, devPtr is nullptr"), return ACL_ERROR_INVALID_PARAM);
    LOG_IF_EXPR(size <= 0, LOG_ERR("Invalid parameter, size = %zu", size), return ACL_ERROR_INVALID_PARAM);
    *devPtr = malloc(size);
    LOG_IF_EXPR(*devPtr == nullptr, LOG_ERR("malloc failed."), return ACL_ERROR_BAD_ALLOC);
    return ACL_SUCCESS;
}

aclError aclrtFree(void *devPtr)
{
    LOG_IF_EXPR(devPtr == nullptr, LOG_ERR("Invalid parameter, devPtr is nullptr"), return ACL_ERROR_INVALID_PARAM);
    free(devPtr);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    LOG_IF_EXPR(devPtr == nullptr, LOG_ERR("Invalid parameter, devPtr is nullptr"), return ACL_ERROR_INVALID_PARAM);
    auto ret = memset_s(devPtr, maxCount, value, count);
    LOG_IF_EXPR(ret != EOK, LOG_ERR("memset_s failed, ERROR: %d", ret), return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    LOG_IF_EXPR(dst == nullptr, LOG_ERR("Invalid parameter, dst is nullptr"), return ACL_ERROR_INVALID_PARAM);
    LOG_IF_EXPR(dst == nullptr, LOG_ERR("Invalid parameter, src is nullptr"), return ACL_ERROR_INVALID_PARAM);
    auto ret = memcpy_s(dst, destMax, src, count);
    LOG_IF_EXPR(ret != EOK, LOG_ERR("memcpy_s failed, ERROR: %d", ret), return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

} // extern "C"