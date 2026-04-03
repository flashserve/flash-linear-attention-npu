/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file tensor_intf.h
 * \brief 封装 Tensor 基类, 简化 Tiling 及 Kernel 阶段对 Tensor 操作.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <initializer_list>
#include <graph/types.h>
#include <exe_graph/runtime/shape.h>

namespace ops::adv::tests::utils {


class TensorIntf {
public:
    enum class TensorType {
        REQUIRED_INPUT,
        OPTIONAL_INPUT,
        REQUIRED_OUTPUT,
    };

public:
    TensorIntf() = default;

    TensorIntf(const char *name, const std::initializer_list<int64_t> &shape, const char *shapeType, ge::DataType dType,
               ge::Format format, TensorType type = TensorType::REQUIRED_INPUT);

    TensorIntf(const char *name, const std::vector<int64_t> &shape, const char *shapeType, ge::DataType dType,
               ge::Format format, TensorType type = TensorType::REQUIRED_INPUT);

    TensorIntf(const char *name, const std::vector<std::vector<int64_t>> &shape, const char *shapeType,
               ge::DataType dType, ge::Format format, TensorType type = TensorType::REQUIRED_INPUT);
    TensorIntf(const TensorIntf &o) = default;

    TensorIntf &operator=(const TensorIntf &o) = default;

    virtual ~TensorIntf() = default;

    [[maybe_unused]] bool IsInput() const;
    [[maybe_unused]] bool IsOutput() const;
    [[maybe_unused]] bool IsRequired() const;
    [[maybe_unused]] bool IsOptional() const;

    [[nodiscard]] std::string GetTilingStr() const;
    int64_t GetExpDataSize() const;

    const std::string &Name() const;
    const gert::Shape &Shape() const;
    const std::vector<int64_t> &ShapeView() const;
    const std::vector<std::vector<int64_t>> &ShapesView() const;

    const std::string &ShapeType() const;
    ge::DataType GetDataType() const;
    ge::Format GetFormat() const;
    TensorType GetTensorType() const;

    size_t GetDimNum() const;
    uint8_t *GetDevData() const;
    int64_t GetDevDataSize() const;

    virtual uint8_t *AllocDevDataNz(int32_t initVal, int64_t minSize);
    virtual uint8_t *AllocDevData(int32_t initVal, int64_t minSize);
    virtual void FreeDevData();

    template <class T> [[maybe_unused]] bool CopyHostToDevData(std::vector<T> &hostData)
    {
        int64_t hostDataSize = hostData.size() * sizeof(T);
        return this->CopyHostToDevData((uint8_t *)hostData.data(), hostDataSize);
    }
    [[maybe_unused]] virtual bool CopyHostToDevData(uint8_t *hostBuff, int64_t hostBuffSize);
    [[maybe_unused]] virtual bool LoadFileToDevData(std::string &filePath);

    template <class T> [[maybe_unused]] bool CopyDevDataToHost(std::vector<T> &hostData)
    {
        auto hostEleNum = static_cast<int64_t>(std::ceil(devDataSize_ / sizeof(T)));
        int64_t hostDataSize = hostEleNum * sizeof(T);
        hostData.resize(hostEleNum, 0);
        return this->CopyDevDataToHost((uint8_t *)hostData.data(), hostDataSize);
    }
    [[maybe_unused]] virtual bool CopyDevDataToHost(uint8_t *hostBuff, int64_t hostBuffSize);
    [[maybe_unused]] virtual bool SaveDevDataToFile(std::string &filePath);

protected:
    std::string name_;
    gert::Shape shape_;
    std::vector<gert::Shape> shapes_;
    std::vector<int64_t> shapeView_;
    std::vector<std::vector<int64_t>> shapesView_;
    bool isArray_ = false;
    std::string shapeType_;
    ge::DataType dType_ = ge::DT_MAX;
    ge::Format format_ = ge::FORMAT_MAX;
    TensorType type_ = TensorType::REQUIRED_INPUT;
    uint8_t *devData_ = nullptr;
    int64_t devDataSize_ = 0;

protected:
    virtual uint8_t *AllocDevDataImpl(int64_t size) = 0;
    virtual void FreeDevDataImpl(uint8_t *devPtr) = 0;
    virtual bool MemSetDevDataImpl(uint8_t *devPtr, int64_t devMax, int32_t val, int64_t cnt) = 0;
    virtual bool MemCpyHostToDevDataImpl(uint8_t *devPtr, int64_t devMax, const void *hostPtr, int64_t cnt) = 0;
    virtual bool MemCpyDevDataToHostImpl(void *hostPtr, int64_t hostMax, const uint8_t *devPtr, int64_t cnt) = 0;

private:
    static std::string ToString(const ge::DataType &dType);
    static std::string ToString(const ge::Format &format);
    static std::string ToString(const gert::Shape &shape);
};

} // namespace ops::adv::tests::utils
