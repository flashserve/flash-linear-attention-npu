/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_torch.h
 * \brief
 */
#ifndef ASCEND_OPS_MOE_DISTRIBUTE_DISPATCH_TORCH_H
#define ASCEND_OPS_MOE_DISTRIBUTE_DISPATCH_TORCH_H

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>

template <typename TensorType, typename ElementType>
ElementType *get_first_tensor_address_by_type(const TensorType &input, bool allow_empty)
{
    const auto &get_tensor = [&]() -> const torch::Tensor * {
        // 处理 optional<torch::Tensor>
        if constexpr (std::is_same_v<TensorType, c10::optional<torch::Tensor>>) {
            if (!input.has_value()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "optional<Tensor> has no value");
                return nullptr;
            }
            if (!input->defined()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "optional<Tensor> is undefined");
                return nullptr;
            }
            return &input.value();
        }
        // 处理 optional<TensorList>
        else if constexpr (std::is_same_v<TensorType, c10::optional<at::TensorList>>) {
            if (!input.has_value()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "optional<TensorList> has no value");
                return nullptr;
            }
            if (input->empty()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "optional<TensorList> is empty");
                return nullptr;
            }
            const auto &tensor = input.value()[0];
            if (!tensor.defined()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "First tensor in optional<TensorList> is undefined");
                return nullptr;
            }
            return &tensor;
        }
        // 处理 TensorList
        else if constexpr (std::is_same_v<TensorType, at::TensorList>) {
            if (input.empty()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "TensorList is empty");
                return nullptr;
            }
            const auto &tensor = input[0];
            if (!tensor.defined()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "First tensor in TensorList is undefined");
                return nullptr;
            }
            return &tensor;
        }
        // 处理 torch::Tensor
        else if constexpr (std::is_same_v<TensorType, torch::Tensor>) {
            if (!input.defined()) {
                if (!allow_empty)
                    TORCH_CHECK(false, "Tensor is undefined");
                return nullptr;
            }
            return &input;
        }
        // 不支持的类型
        else {
            static_assert(std::is_same_v<TensorType, void>, "Unsupported tensor type");
            return nullptr;
        }
    };

    const torch::Tensor *tensor_ptr = get_tensor();
    if (!tensor_ptr) {
        return nullptr;
    }

    // 修复：使用非模板版本的data_ptr()，然后进行类型转换
    void *raw_ptr = tensor_ptr->data_ptr();
    return reinterpret_cast<ElementType *>(raw_ptr);
}

template <typename TensorType>
void *get_first_tensor_address(c10::ScalarType dataType, const TensorType &input, bool allow_empty = false)
{
    return get_first_tensor_address_by_type<TensorType, void>(input, allow_empty);
}


#endif
