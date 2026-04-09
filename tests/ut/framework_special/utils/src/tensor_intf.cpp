/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file tensor_intf.cpp
 * \brief 封装 Tensor 基类, 简化 Tiling 及 Kernel 阶段对 Tensor 操作.
 */

#include "tests/utils/tensor_intf.h"
#include <sstream>
#include "tests/utils/io.h"
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

TensorIntf::TensorIntf(const char *name, const std::initializer_list<int64_t> &shape, const char *shapeType,
                       ge::DataType dType, ge::Format format, TensorType type)
    : name_(name), shape_(shape), shapeView_(shape), shapeType_(shapeType), dType_(dType), format_(format), type_(type),
      devData_(nullptr), devDataSize_(0)
{
}

TensorIntf::TensorIntf(const char *name, const std::vector<int64_t> &shape, const char *shapeType, ge::DataType dType,
                       ge::Format format, TensorType type)
    : TensorIntf(name, {}, shapeType, dType, format, type)
{
    for (auto d : shape) {
        this->shapeView_.push_back(d);
        this->shape_.AppendDim(d);
    }
}

bool TensorIntf::IsInput() const
{
    return type_ == TensorType::REQUIRED_INPUT || type_ == TensorType::OPTIONAL_INPUT;
}

bool TensorIntf::IsOutput() const
{
    return type_ == TensorType::REQUIRED_OUTPUT;
}

bool TensorIntf::IsRequired() const
{
    return type_ == TensorType::REQUIRED_INPUT || type_ == TensorType::REQUIRED_OUTPUT;
}

bool TensorIntf::IsOptional() const
{
    return type_ == TensorType::OPTIONAL_INPUT;
}

std::string TensorIntf::GetTilingStr() const
{
    std::string str;
    std::string finalStr = "[";
    if (this->isArray_ == false) {
        if (shape_.GetDimNum() <= 0) {
            str = "null";
        } else {
            str = R"({ )"
                  R"("name": ")" +
                  name_ + R"(", )" + R"("dtype": ")" + TensorIntf::ToString(dType_) + R"(", )" + R"("format": ")" +
                  TensorIntf::ToString(format_) + R"(", )" + R"("ori_format": ")" + TensorIntf::ToString(format_) +
                  R"(", )" + R"("shape": )" + TensorIntf::ToString(shape_) + R"(, )" + R"("ori_shape": )" +
                  TensorIntf::ToString(shape_) + R"( })";
        }
        return str;
    } else {
        uint64_t i = 0;
        for (; i < this->shapes_.size() - 1; i++) {
            str = R"({ )"
                  R"("name": ")" +
                  name_ + R"(", )" + R"("dtype": ")" + TensorIntf::ToString(dType_) + R"(", )" + R"("format": ")" +
                  TensorIntf::ToString(format_) + R"(", )" + R"("ori_format": ")" + TensorIntf::ToString(format_) +
                  R"(", )" + R"("shape": )" + TensorIntf::ToString(this->shapes_[i]) + R"(, )" + R"("ori_shape": )" +
                  TensorIntf::ToString(this->shapes_[i]) + R"( })";
            finalStr += str;
            finalStr += ",";
        }
        str = R"({ )"
              R"("name": ")" +
              name_ + R"(", )" + R"("dtype": ")" + TensorIntf::ToString(dType_) + R"(", )" + R"("format": ")" +
              TensorIntf::ToString(format_) + R"(", )" + R"("ori_format": ")" + TensorIntf::ToString(format_) +
              R"(", )" + R"("shape": )" + TensorIntf::ToString(this->shapes_[i]) + R"(, )" + R"("ori_shape": )" +
              TensorIntf::ToString(this->shapes_[i]) + R"( })";
        finalStr += str;
        finalStr += "]";
        return finalStr;
    }
}

const std::string &TensorIntf::Name() const
{
    return name_;
}

const gert::Shape &TensorIntf::Shape() const
{
    return shape_;
}

const std::vector<int64_t> &TensorIntf::ShapeView() const
{
    return shapeView_;
}

const std::vector<std::vector<int64_t>> &TensorIntf::ShapesView() const
{
    return shapesView_;
}

const std::string &TensorIntf::ShapeType() const
{
    return shapeType_;
}

ge::DataType TensorIntf::GetDataType() const
{
    return dType_;
}

ge::Format TensorIntf::GetFormat() const
{
    return format_;
}

TensorIntf::TensorType TensorIntf::GetTensorType() const
{
    return type_;
}

size_t TensorIntf::GetDimNum() const
{
    return shape_.GetDimNum();
}

int64_t TensorIntf::GetExpDataSize() const
{
    constexpr size_t kShapeEntryPadding = 2;
    constexpr size_t kStorageUnitSize = 8;
    if (this->isArray_ == false) {
        if (shape_.GetDimNum() <= 0) {
            return 0;
        }
        return shape_.GetShapeSize() * ge::GetSizeByDataType(dType_);
    } else {
        int64_t needSize = 1;
        for (uint64_t i = 0; i < this->shapes_.size(); i++) {
            needSize += kShapeEntryPadding + this->shapes_[i].GetDimNum();
        }
        return needSize * kStorageUnitSize;
    }
}

uint8_t *TensorIntf::GetDevData() const
{
    return devData_;
}

int64_t TensorIntf::GetDevDataSize() const
{
    return devDataSize_;
}

uint8_t *TensorIntf::AllocDevDataNz(int32_t initVal, int64_t minSize)
{
    if (devData_ != nullptr) {
        return devData_;
    }
    devDataSize_ = std::max(this->GetExpDataSize(), minSize);
    devData_ = this->AllocDevDataImpl(devDataSize_);
    if (devData_ == nullptr) {
        goto ErrRet;
    }
    if (!this->MemSetDevDataImpl(devData_, devDataSize_, initVal, devDataSize_)) {
        goto ErrRet;
    }
    return devData_;
ErrRet:
    this->FreeDevData();
    return nullptr;
}

uint8_t *TensorIntf::AllocDevData(int32_t initVal, int64_t minSize)
{
    if (devData_ != nullptr) {
        return devData_;
    }
    devDataSize_ = std::max(this->GetExpDataSize(), minSize);
    devData_ = this->AllocDevDataImpl(devDataSize_);
    if (devData_ == nullptr) {
        goto ErrRet;
    }
    if (!this->MemSetDevDataImpl(devData_, devDataSize_, initVal, devDataSize_)) {
        goto ErrRet;
    }
    return devData_;
ErrRet:
    this->FreeDevData();
    return nullptr;
}

void TensorIntf::FreeDevData()
{
    if (this->devData_ != nullptr) {
        this->FreeDevDataImpl(devData_);
        devData_ = nullptr;
    }
    devDataSize_ = 0;
}

bool TensorIntf::CopyHostToDevData(uint8_t *hostBuff, int64_t hostBuffSize)
{
    if (devData_ == nullptr || hostBuff == nullptr || devDataSize_ < hostBuffSize) {
        LOG_ERR("Tensor(%s), Invalid param, Host[%p, %ld], Device[%p, %ld]", name_.c_str(), hostBuff, hostBuffSize,
                devData_, devDataSize_);
        return false;
    }
    return this->MemCpyHostToDevDataImpl(devData_, devDataSize_, hostBuff, hostBuffSize);
}

bool TensorIntf::LoadFileToDevData(std::string &filePath)
{
    // 将文件内容读取到host侧
    size_t readSize = 0;
    int64_t hostBuffSize = this->GetExpDataSize();
    std::vector<uint8_t> hostBuff;
    hostBuff.resize(hostBuffSize, 0);
    if (!ReadFile(filePath, readSize, hostBuff.data(), hostBuffSize)) {
        return false;
    }
    // 将host内容copy到dev侧
    return this->CopyHostToDevData(hostBuff);
}

bool TensorIntf::CopyDevDataToHost(uint8_t *hostBuff, int64_t hostBuffSize)
{
    if (devData_ == nullptr || hostBuff == nullptr || hostBuffSize < devDataSize_) {
        LOG_ERR("Tensor(%s), Invalid param, Host[%p, %ld], Device[%p, %ld]", name_.c_str(), hostBuff, hostBuffSize,
                devData_, devDataSize_);
        return false;
    }
    return this->MemCpyDevDataToHostImpl(hostBuff, hostBuffSize, devData_, devDataSize_);
}

bool TensorIntf::SaveDevDataToFile(std::string &filePath)
{
    // 将dev内容copy到host侧
    std::vector<uint8_t> hostData;
    hostData.resize(devDataSize_, 0);
    if (!this->CopyDevDataToHost(hostData)) {
        return false;
    }
    // 将host内容save到file
    return WriteFile(filePath, hostData.data(), hostData.size());
}

std::string TensorIntf::ToString(const ge::DataType &dType)
{
    switch (dType) {
        case ge::DT_FLOAT:
            return "float";
        case ge::DT_FLOAT16:
            return "float16";
        case ge::DT_UINT8:
            return "uint8";
        case ge::DT_INT8:
            return "int8";
        case ge::DT_BOOL:
            return "bool";
        case ge::DT_BF16:
            return "bfloat16";
        case ge::DT_INT64:
            return "int64";
        case ge::DT_UINT64:
            return "uint64";
        case ge::DT_INT32:
            return "int32";
        case ge::DT_INT4:
            return "int4";
        case ge::DT_HIFLOAT8:
            return "hifloat8";
        case ge::DT_FLOAT8_E5M2:
            return "float8_e5m2";
        case ge::DT_FLOAT8_E4M3FN:
            return "float8_e4m3fn";
        case ge::DT_FLOAT8_E8M0:
            return "float8_e8m0";
        case ge::DT_FLOAT6_E3M2:
            return "float6_e3m2";
        case ge::DT_FLOAT6_E2M3:
            return "float6_e2m3";
        case ge::DT_FLOAT4_E2M1:
            return "float4_e2m1";
        case ge::DT_FLOAT4_E1M2:
            return "float4_e1m2";
        default:
            return "undefined";
    }
}

std::string TensorIntf::ToString(const ge::Format &format)
{
    switch (format) {
        case ge::FORMAT_ND:
            return "ND";
        case ge::FORMAT_FRACTAL_NZ:
            return "FRACTAL_NZ";
        default:
            return "undefined";
    }
}

std::string TensorIntf::ToString(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);
    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    std::ostringstream oss;
    oss << "[";
    if (!shapeVec.empty()) {
        for (size_t i = 0; i < shapeVec.size() - 1; ++i) {
            oss << shapeVec[i] << ", ";
        }
        oss << shapeVec[shapeVec.size() - 1];
    }
    oss << "]";
    return oss.str();
}
