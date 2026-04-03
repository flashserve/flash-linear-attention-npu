/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file tensor_list.h
 * \brief 封装 CPU模式 TensorList, 简化 Tiling 及 Kernel 阶段对 TensorList 操作.
 */

#pragma once

#include "tests/utils/tensor_intf.h"

namespace ops::adv::tests::utils {

class TensorList : public ops::adv::tests::utils::TensorIntf {
public:
    TensorList() = default;

    TensorList(const char *name, const std::vector<std::vector<int64_t>> &shapes, const char *shapeType,
               ge::DataType dType, ge::Format format, TensorType type = TensorType::REQUIRED_INPUT);

    TensorList(const char *name, const std::initializer_list<int64_t> &shapes, const char *shapeType,
               ge::DataType dType, ge::Format format, TensorType type = TensorType::REQUIRED_INPUT);

    TensorList(const TensorList &o) = default;

    TensorList &operator=(const TensorList &o) = default;

    ~TensorList() override;
protected:
    uint8_t *AllocDevDataImpl(int64_t size) override;
    void FreeDevDataImpl(uint8_t *devPtr) override;
    bool MemSetDevDataImpl(uint8_t *devPtr, int64_t devMax, int32_t val, int64_t cnt) override;
    bool MemCpyHostToDevDataImpl(uint8_t *devPtr, int64_t devMax, const void *hostPtr, int64_t cnt) override;
    bool MemCpyDevDataToHostImpl(void *hostPtr, int64_t hostMax, const uint8_t *devPtr, int64_t cnt) override;
};

} // namespace ops::adv::tests::utils
