/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file tensor.cpp
 * \brief 封装 CPU模式, 简化 Tiling 及 Kernel 阶段对 Tensor 操作.
 */

#include "tests/utils/tensor.h"
#include <tikicpulib.h>
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

Tensor::Tensor(const char *name, const std::initializer_list<int64_t> &shape, const char *shapeType, ge::DataType dType,
               ge::Format format, TensorType type)
    : TensorIntf(name, shape, shapeType, dType, format, type)
{
}

Tensor::Tensor(const char *name, const std::vector<int64_t> &shape, const char *shapeType, ge::DataType dType,
               ge::Format format, TensorType type)
    : TensorIntf(name, shape, shapeType, dType, format, type)
{
}

Tensor::~Tensor()
{
    this->FreeDevData();
}

uint8_t *Tensor::AllocDevDataImpl(int64_t size)
{
    auto *ptr = (uint8_t *)AscendC::GmAlloc(size);
    LOG_IF(ptr == nullptr, LOG_ERR("AscendC::GmAlloc failed, Size(%ld)", size));
    return ptr;
}

void Tensor::FreeDevDataImpl(uint8_t *devPtr)
{
    AscendC::GmFree(devPtr);
}

bool Tensor::MemSetDevDataImpl(uint8_t *devPtr, int64_t devMax, int32_t val, int64_t cnt)
{
    auto ret = memset_s(devPtr, devMax, val, cnt);
    LOG_IF(ret != EOK,
           LOG_ERR("memset_s failed, ERROR: %d, Dev(%p, %ld), Param(%d, %ld)", ret, devPtr, devMax, val, cnt));
    return ret == EOK;
}

bool Tensor::MemCpyHostToDevDataImpl(uint8_t *devPtr, int64_t devMax, const void *hostPtr, int64_t cnt)
{
    auto ret = memcpy_s(devPtr, devMax, hostPtr, cnt);
    LOG_IF(ret != EOK,
           LOG_ERR("memcpy_s failed, ERROR: %d, Dev(%p, %ld), Param(%p, %ld)", ret, devPtr, devMax, hostPtr, cnt));
    return ret == EOK;
}

bool Tensor::MemCpyDevDataToHostImpl(void *hostPtr, int64_t hostMax, const uint8_t *devPtr, int64_t cnt)
{
    auto ret = memcpy_s(hostPtr, hostMax, devPtr, cnt);
    LOG_IF(ret != EOK,
           LOG_ERR("memcpy_s failed, ERROR: %d, Host(%p, %ld), Param(%p, %ld)", ret, hostPtr, hostMax, devPtr, cnt));
    return ret == EOK;
}
