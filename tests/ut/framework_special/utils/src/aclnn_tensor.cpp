/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclnn_tensor.cpp
 * \brief 封装 ACLNN Tensor, 简化 Tiling 及 Kernel 阶段对 Tensor 操作.
 */

#include "tests/utils/aclnn_tensor.h"
#include <map>
#include <graph/utils/type_utils.h>
#include <acl/acl.h>
#include "tests/utils/log.h"
#include "tests/utils/io.h"

namespace {
std::map<ge::DataType, aclDataType> geDtype2AclDtypeMap = {
    {ge::DataType::DT_FLOAT16, ACL_FLOAT16}, {ge::DataType::DT_BF16, ACL_BF16},   {ge::DataType::DT_FLOAT, ACL_FLOAT},
    {ge::DataType::DT_BOOL, ACL_BOOL},       {ge::DataType::DT_UINT8, ACL_UINT8}, {ge::DataType::DT_INT4, ACL_INT4},
    {ge::DataType::DT_INT8, ACL_INT8},       {ge::DataType::DT_INT32, ACL_INT32}, {ge::DataType::DT_INT64, ACL_INT64},
    {ge::DataType::DT_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN}, {ge::DataType::DT_FLOAT8_E8M0, ACL_FLOAT8_E8M0},
    {ge::DataType::DT_FLOAT8_E5M2, ACL_FLOAT8_E5M2}, {ge::DataType::DT_HIFLOAT8, ACL_HIFLOAT8}};
}

using namespace ops::adv::tests::utils;

AclnnTensor::AclnnTensor(const char *name, const std::initializer_list<int64_t> &shape, const char *shapeType,
                         ge::DataType dType, ge::Format format, TensorType type)
    : AclnnTensor(name, std::vector<int64_t>(shape), shapeType, dType, format, type)
{
}

AclnnTensor::AclnnTensor(const char *name, const std::vector<int64_t> &shape, const char *shapeType, ge::DataType dType,
                         ge::Format format, TensorType type)
    : TensorIntf(name, shape, shapeType, dType, format, type), aclDataType_(ACL_DT_UNDEFINED),
      aclTensorDataStrides_({}), aclTensor_(nullptr)
{
    /* 计算连续 tensor 的 strides */
    aclTensorDataStrides_.resize(shape_.GetDimNum(), 1);
        for (auto i = static_cast<int64_t>(shape_.GetDimNum()) - 2; i >= 0; i--) {
            aclTensorDataStrides_[i] = shape_[i + 1] * aclTensorDataStrides_[i + 1];
        }
        /* 获取 aclDataType */
        auto iter = geDtype2AclDtypeMap.find(dType);
        if (iter == geDtype2AclDtypeMap.end()) {
            LOG_ERR("Tensor(%s), Unknown dtype(%s)", name_.c_str(), ge::TypeUtils::DataTypeToSerialString(dType).c_str());
        } else {
            aclDataType_ = iter->second;
        }
}

AclnnTensor::AclnnTensor(const Tensor &t)
    : AclnnTensor(t.Name().c_str(), t.ShapeView(), t.ShapeType().c_str(), t.GetDataType(), t.GetFormat(),
                  t.GetTensorType())
{
}

AclnnTensor::~AclnnTensor()
{
    this->Destroy();
}

aclDataType AclnnTensor::GetAclDataType() const
{
    return aclDataType_;
}

aclTensor *AclnnTensor::GetAclTensor() const
{
    return aclTensor_;
}

uint8_t *AclnnTensor::AllocDevData(int32_t initVal, int64_t minSize)
{
    if (TensorIntf::AllocDevData(initVal, minSize) == nullptr) {
        return nullptr;
    }
    /* 调用 aclCreateTensor 创建 aclTensor  */
    aclTensor_ = aclCreateTensor(shapeView_.data(), shapeView_.size(), aclDataType_, aclTensorDataStrides_.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shapeView_.data(), shapeView_.size(), devData_);
    if (aclTensor_ == nullptr) {
        LOG_ERR("aclCreateTensor failed, Tensor(%s))", name_.c_str());
        this->FreeDevData();
    }
    return devData_;
}

void AclnnTensor::FreeDevData()
{
    this->Destroy();
}

uint8_t *AclnnTensor::AllocDevDataImpl(int64_t size)
{
    /* 调用 aclrtMalloc 申请 device 侧内存 */
    void *devPtr = nullptr;
    auto ret = aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERR("aclrtMalloc failed, ERROR: %d, Tensor(%s), Size(%ld)", ret, name_.c_str(), size);
    }
    //  LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMalloc failed, ERROR: %d, Tensor(%s), Size(%ld)", ret, name_.c_str(),
    //  size));
    return (uint8_t *)devPtr;
}

void AclnnTensor::FreeDevDataImpl(uint8_t *devPtr)
{
    auto ret = aclrtFree(devPtr);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtFree failed, ERROR: %d, Tensor(%s))", ret, name_.c_str()));
}

bool AclnnTensor::MemSetDevDataImpl(uint8_t *devPtr, int64_t devMax, int32_t val, int64_t cnt)
{
    /* 调用 aclrtMemset 设置值 */
    auto ret = aclrtMemset(devPtr, devMax, val, cnt);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMemset failed, ERROR: %d, Tensor(%s)", ret, name_.c_str()));
    return ret == ACL_SUCCESS;
}

bool AclnnTensor::MemCpyHostToDevDataImpl(uint8_t *devPtr, int64_t devMax, const void *hostPtr, int64_t cnt)
{
    /* 调用 aclrtMemcpy  */
    auto ret = aclrtMemcpy(devPtr, devMax, hostPtr, cnt, ACL_MEMCPY_HOST_TO_DEVICE);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMemcpy failed, ERROR: %d, Tensor(%s)", ret, name_.c_str()));
    return ret == ACL_SUCCESS;
}

bool AclnnTensor::MemCpyDevDataToHostImpl(void *hostPtr, int64_t hostMax, const uint8_t *devPtr, int64_t cnt)
{
    /* 调用 aclrtMemcpy  */
    auto ret = aclrtMemcpy(hostPtr, hostMax, devPtr, cnt, ACL_MEMCPY_DEVICE_TO_HOST);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMemcpy failed, ERROR: %d, Tensor(%s)", ret, name_.c_str()));
    return ret == ACL_SUCCESS;
}

void AclnnTensor::Destroy()
{
    if (aclTensor_ != nullptr) {
        auto ret = aclDestroyTensor(aclTensor_);
        LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclDestroyTensor failed, ERROR: %d, Tensor(%s))", ret, name_.c_str()));
        aclTensor_ = nullptr;
    }
    TensorIntf::FreeDevData();
}
