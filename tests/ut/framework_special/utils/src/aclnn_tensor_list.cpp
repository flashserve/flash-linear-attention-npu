/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclnn_tensor_list.cpp
 * \brief 封装 ACLNN TensorList, 简化 Tiling 及 Kernel 阶段对 TensorList 操作.
 */

#include "tests/utils/aclnn_tensor_list.h"
#include <map>
#include <graph/utils/type_utils.h>
#include <acl/acl.h>
#include "tests/utils/log.h"
#include "tests/utils/io.h"

namespace {
std::map<ge::DataType, aclDataType> geDtype2AclDtypeMap = {{ge::DataType::DT_FLOAT16, ACL_FLOAT16},
                                                           {ge::DataType::DT_BF16, ACL_BF16},
                                                           {ge::DataType::DT_FLOAT, ACL_FLOAT},
                                                           {ge::DataType::DT_BOOL, ACL_BOOL},
                                                           {ge::DataType::DT_UINT8, ACL_UINT8},
                                                           {ge::DataType::DT_INT4, ACL_INT4},
                                                           {ge::DataType::DT_INT8, ACL_INT8},
                                                           {ge::DataType::DT_INT32, ACL_INT32},
                                                           {ge::DataType::DT_INT64, ACL_INT64},
                                                           {ge::DataType::DT_UINT64, ACL_UINT64},
                                                           {ge::DataType::DT_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN},
                                                           {ge::DataType::DT_FLOAT8_E8M0, ACL_FLOAT8_E8M0},
                                                           {ge::DataType::DT_FLOAT8_E5M2, ACL_FLOAT8_E5M2},
                                                           {ge::DataType::DT_HIFLOAT8, ACL_HIFLOAT8},
                                                           {ge::DataType::DT_FLOAT4_E2M1, ACL_FLOAT4_E2M1},
                                                           {ge::DataType::DT_FLOAT4_E1M2, ACL_FLOAT4_E1M2}
                                                           };
}

using namespace ops::adv::tests::utils;

AclnnTensorList::AclnnTensorList(const char *name, const std::vector<std::vector<int64_t>> &shapes,
    const char *shapeType, ge::DataType dType, ge::Format format, TensorType type, bool isTrans)
    : TensorIntf(name, {}, shapeType, dType, format, type), aclDataType_(ACL_DT_UNDEFINED),
      aclTensorListDataStrides_({}), aclTensorList_(nullptr)
{
    for (auto shape : shapes) {
        std::vector<int64_t> shapeView{};
        gert::Shape myShape;
        for (auto dim : shape) {
            shapeView.push_back(dim);
            myShape.AppendDim(dim);
        }
        std::vector<int64_t> aclTensorDataStrides{};
        aclTensorDataStrides.resize(myShape.GetDimNum(), 1);
        auto dim1 = static_cast<int64_t>(myShape.GetDimNum() - 1);
        auto dim2 = static_cast<int64_t>(myShape.GetDimNum() - 2);
        for (auto i = dim2; i >= 0; i--) {
            aclTensorDataStrides[i] = myShape[i + 1] * aclTensorDataStrides[i + 1];
        }
        // grouped matmul type mx quant mode need speical stride assignment
        if (std::strcmp("pertoken_scale", name) == 0 && isTrans && dType == ge::DataType::DT_FLOAT8_E8M0 &&
            shape.size() == 3 && shape[2] == 2) { // in gmm mx typek quant mode, scale dim num is 3 and last dim value is 2
            aclTensorDataStrides[0] = 2; // first dim is m, stride is 2, given by the last axis
            aclTensorDataStrides[1] = myShape[0] * 2; // k is outer axis, stride should be m dim value times 2
            aclTensorDataStrides[2] = 1; // 2 is for last dim index, which stride is 1
        } else if (isTrans && dim2 >= 0) {
            aclTensorDataStrides[dim2] = 1;
            aclTensorDataStrides[dim1] = myShape[dim2];
        }
        this->shapes_.push_back(myShape);
        this->shapesView_.push_back(shapeView);
        aclTensorListDataStrides_.push_back(aclTensorDataStrides);
    }
    this->isArray_ = true;
    /* 获取 aclDataType */
    auto iter = geDtype2AclDtypeMap.find(dType);
    if (iter == geDtype2AclDtypeMap.end()) {
        LOG_ERR("TensorList(%s), Unknown dtype(%s)", name_.c_str(), ge::TypeUtils::DataTypeToSerialString(dType).c_str());
    } else {
        aclDataType_ = iter->second;
    }
}

AclnnTensorList::AclnnTensorList(const TensorList &t, bool isTrans)
    : AclnnTensorList(t.Name().c_str(), t.ShapesView(), t.ShapeType().c_str(), t.GetDataType(), t.GetFormat(),
                      t.GetTensorType(), isTrans)
{
}

AclnnTensorList::~AclnnTensorList()
{
    this->Destroy();
}

aclDataType AclnnTensorList::GetAclDataType() const
{
    return aclDataType_;
}

aclTensorList *AclnnTensorList::GetAclTensorList() const
{
    return aclTensorList_;
}

uint8_t *AclnnTensorList::AllocDevDataNz(int32_t initVal, int64_t minSize)
{
    int size = this->shapesView_.size();
    if (size <= 0) {
        return nullptr;
    }
    aclTensor** tensors = reinterpret_cast<aclTensor**>(malloc(size * sizeof(aclTensor*)));
    if (tensors == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < size; i++) {
        aclTensor** tmpTensor = tensors + i;
        this->shape_ = this->shapes_[i];
        if (TensorIntf::AllocDevData(initVal, minSize) == nullptr) {
            return nullptr;
        }
        /* 调用 aclCreateTensor 创建 aclTensor  */
        *tmpTensor = aclCreateTensor(shapesView_[i].data(), shapesView_[i].size(), aclDataType_, aclTensorListDataStrides_[i].data(), 0,
                                     aclFormat::ACL_FORMAT_FRACTAL_NZ, shapesView_[i].data(), shapesView_[i].size(), devData_);
        if (tmpTensor == nullptr) {
            LOG_ERR("aclCreateTensor failed, Tensor(%s))", name_.c_str());
            this->FreeDevData();
        }
    }
    aclTensorList_ = aclCreateTensorList(tensors, size);
    free(tensors);
    return devData_;
}

uint8_t *AclnnTensorList::AllocDevData(int32_t initVal, int64_t minSize)
{
    int size = this->shapesView_.size();
    if (size <= 0) {
        return nullptr;
    }
    aclTensor** tensors = reinterpret_cast<aclTensor**>(malloc(size * sizeof(aclTensor*)));
    if (tensors == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < size; i++) {
        aclTensor** tmpTensor = tensors + i;
        this->shape_ = this->shapes_[i];
        if (TensorIntf::AllocDevData(initVal, minSize) == nullptr) {
            return nullptr;
        }
        /* 调用 aclCreateTensor 创建 aclTensor  */
        *tmpTensor = aclCreateTensor(shapesView_[i].data(), shapesView_[i].size(), aclDataType_, aclTensorListDataStrides_[i].data(), 0,
                                     aclFormat::ACL_FORMAT_ND, shapesView_[i].data(), shapesView_[i].size(), devData_);
        if (tmpTensor == nullptr) {
            LOG_ERR("aclCreateTensor failed, Tensor(%s))", name_.c_str());
            this->FreeDevData();
        }
    }
    aclTensorList_ = aclCreateTensorList(tensors, size);
    free(tensors);
    return devData_;
}

void AclnnTensorList::FreeDevData()
{
    this->Destroy();
}

uint8_t *AclnnTensorList::AllocDevDataImpl(int64_t size)
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

void AclnnTensorList::FreeDevDataImpl(uint8_t *devPtr)
{
    auto ret = aclrtFree(devPtr);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtFree failed, ERROR: %d, Tensor(%s))", ret, name_.c_str()));
}

bool AclnnTensorList::MemSetDevDataImpl(uint8_t *devPtr, int64_t devMax, int32_t val, int64_t cnt)
{
    /* 调用 aclrtMemset 设置值 */
    auto ret = aclrtMemset(devPtr, devMax, val, cnt);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMemset failed, ERROR: %d, Tensor(%s)", ret, name_.c_str()));
    return ret == ACL_SUCCESS;
}

bool AclnnTensorList::MemCpyHostToDevDataImpl(uint8_t *devPtr, int64_t devMax, const void *hostPtr, int64_t cnt)
{
    /* 调用 aclrtMemcpy  */
    auto ret = aclrtMemcpy(devPtr, devMax, hostPtr, cnt, ACL_MEMCPY_HOST_TO_DEVICE);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMemcpy failed, ERROR: %d, Tensor(%s)", ret, name_.c_str()));
    return ret == ACL_SUCCESS;
}

bool AclnnTensorList::MemCpyDevDataToHostImpl(void *hostPtr, int64_t hostMax, const uint8_t *devPtr, int64_t cnt)
{
    /* 调用 aclrtMemcpy  */
    auto ret = aclrtMemcpy(hostPtr, hostMax, devPtr, cnt, ACL_MEMCPY_DEVICE_TO_HOST);
    LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclrtMemcpy failed, ERROR: %d, Tensor(%s)", ret, name_.c_str()));
    return ret == ACL_SUCCESS;
}

void AclnnTensorList::Destroy()
{
    if (aclTensorList_ != nullptr) {
        auto ret = aclDestroyTensorList(aclTensorList_);
        LOG_IF(ret != ACL_SUCCESS, LOG_ERR("aclDestroyTensorList failed, ERROR: %d, TensorList(%s))", ret, name_.c_str()));
        aclTensorList_ = nullptr;
    }
    TensorIntf::FreeDevData();
}
