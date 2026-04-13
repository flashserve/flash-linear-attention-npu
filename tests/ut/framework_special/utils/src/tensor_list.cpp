/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file tensor_list.cpp
 * \brief 封装 CPU模式, 简化 Tiling 及 Kernel 阶段对 TensorList操作.
 */

#include "tests/utils/tensor_list.h"
#include <tikicpulib.h>
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

TensorList::TensorList(const char *name, const std::vector<std::vector<int64_t>> &shapes, const char *shapeType,
                       ge::DataType dType, ge::Format format, TensorType type)
    : TensorIntf(name, {}, shapeType, dType, format, type)
{
    for (auto shape : shapes) {
        std::vector<int64_t> shapeView{};
        gert::Shape myShape;
        for (auto dim : shape) {
            shapeView.push_back(dim);
            myShape.AppendDim(dim);
        }
        this->shapes_.push_back(myShape);
        this->shapesView_.push_back(shapeView);
    }
    this->isArray_ = true;
}

TensorList::TensorList(const char *name, const std::initializer_list<int64_t> &shapes, const char *shapeType,
                       ge::DataType dType, ge::Format format, TensorType type)
    : TensorIntf(name, shapes, shapeType, dType, format, type)
{
}


TensorList::~TensorList()
{
    this->FreeDevData();
}

uint8_t *TensorList::AllocDevDataImpl(int64_t size)
{
    uint64_t *ptr = (uint64_t *)AscendC::GmAlloc(size);
    LOG_IF(ptr == nullptr, LOG_ERR("AscendC::GmAlloc failed, Size(%ld)", size));
    return (uint8_t *)ptr;
}

void TensorList::FreeDevDataImpl(uint8_t *devPtr)
{
    uint64_t *ptr = reinterpret_cast<uint64_t *>(devPtr);
    uint64_t ptrOffset = (*ptr) >> 3;

    for (uint64_t i = 0; i < this->shapes_.size(); i++) {
        AscendC::GmFree(reinterpret_cast<uint64_t *>(*(ptr + ptrOffset + i)));
    }
    AscendC::GmFree(devPtr);
}

bool TensorList::MemSetDevDataImpl(uint8_t *devPtr, int64_t devMax, int32_t val, int64_t cnt)
{
    const int64_t shiftBits = 32;
    const int64_t byteAlignMent = 8;
    auto ret = EOK;
    uint64_t *ptr = reinterpret_cast<uint64_t *>(devPtr);
    int64_t ptrOffset = 1;
    int64_t indexCur = 1;
    std::vector<int64_t> needSize{};
    for (uint64_t i = 0; i < this->shapes_.size(); i++) {
        ptrOffset += (shapes_[i].GetDimNum() + 1);
        *(ptr + indexCur) = (i << shiftBits) + shapes_[i].GetDimNum();
        indexCur++;
        needSize.push_back(shapes_[i].GetShapeSize() * sizeof(this->dType_));
        for (uint64_t j = 0; j < this->shapes_[i].GetDimNum() ; j++) {
            *(ptr + indexCur) = this->shapes_[i].GetDim(j);
            indexCur++;
        }
    }
    *ptr = ptrOffset * byteAlignMent;
    for (uint64_t i = 0; i < needSize.size(); i++) {
        uint64_t *dataPtr = (uint64_t *)AscendC::GmAlloc(needSize[i]);
        *(ptr + indexCur + i) = reinterpret_cast<uint64_t>(dataPtr);
        ret = memset_s(dataPtr, needSize[i], val, needSize[i]);
        LOG_IF(ret != EOK,
               LOG_ERR("memset_s failed, ERROR: %d, Dev(%p, %ld), Param(%d, %ld)", ret, devPtr, devMax, val, cnt));
    }

    return ret == EOK;
}

bool TensorList::MemCpyHostToDevDataImpl(uint8_t *devPtr, int64_t devMax, const void *hostPtr, int64_t cnt)
{
    auto ret = memcpy_s(devPtr, devMax, hostPtr, cnt);
    LOG_IF(ret != EOK,
           LOG_ERR("memcpy_s failed, ERROR: %d, Dev(%p, %ld), Param(%p, %ld)", ret, devPtr, devMax, hostPtr, cnt));
    return ret == EOK;
}

bool TensorList::MemCpyDevDataToHostImpl(void *hostPtr, int64_t hostMax, const uint8_t *devPtr, int64_t cnt)
{
    auto ret = memcpy_s(hostPtr, hostMax, devPtr, cnt);
    LOG_IF(ret != EOK,
           LOG_ERR("memcpy_s failed, ERROR: %d, Host(%p, %ld), Param(%p, %ld)", ret, hostPtr, hostMax, devPtr, cnt));
    return ret == EOK;
}
