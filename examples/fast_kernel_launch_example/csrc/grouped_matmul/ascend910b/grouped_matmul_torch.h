/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_torch.h
 * \brief
 */
#ifndef ASCEND_OPS_GROUPED_MATMUL_TORCH_H
#define ASCEND_OPS_GROUPED_MATMUL_TORCH_H

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

inline std::string build_error_msg(const std::string &func_name, const std::string &name, const std::string &message)
{
    std::ostringstream oss;
    oss << func_name << name << message;
    return oss.str();
}

// 类型组合结构
struct TypeCombo {
    c10::ScalarType x;
    c10::ScalarType bias;
    c10::ScalarType scale;
    c10::ScalarType offset;
    c10::ScalarType antiquantScale;
    c10::ScalarType antiquantOffset;
    c10::ScalarType groupList;
    c10::ScalarType perTokenScale;
    c10::ScalarType weight;
    c10::ScalarType output;
};

// 类型组合管理器
class TypeComboManager {
public:
    static std::vector<TypeCombo> createCombosFromLists(
        const std::vector<c10::ScalarType> &x_list, const std::vector<c10::ScalarType> &bias_list,
        const std::vector<c10::ScalarType> &scale_list, const std::vector<c10::ScalarType> &offset_list,
        const std::vector<c10::ScalarType> &antiquantScale_list,
        const std::vector<c10::ScalarType> &antiquantOffset_list, const std::vector<c10::ScalarType> &groupList_list,
        const std::vector<c10::ScalarType> &perTokenScale_list, const std::vector<c10::ScalarType> &weight_list,
        const std::vector<c10::ScalarType> &output_list)
    {
        std::vector<TypeCombo> combos;
        combos.reserve(x_list.size());

        for (size_t i = 0; i < x_list.size(); ++i) {
            combos.push_back({x_list[i], bias_list[i], scale_list[i], offset_list[i], antiquantScale_list[i],
                              antiquantOffset_list[i], groupList_list[i], perTokenScale_list[i], weight_list[i],
                              output_list[i]});
        }

        return combos;
    }


    template <typename TensorContainer>
    static bool checkTensorType(const TensorContainer &container, c10::ScalarType expected_type,
                                const std::string &name = "curTensor", bool allow_empty = true)
    {
        const std::string &func_name = "[checkTensorType]";

        // 处理 optional 类型
        if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>> ||
                      std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
            if (!container.has_value()) {
                TORCH_CHECK(allow_empty, build_error_msg(func_name, name,
                                                         " is not provided (null optional), but empty is not allowed"));
                return allow_empty;
            }
        }
        // 获取实际的值或引用
        auto get_value = [&]() -> auto &
        {
            if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>>) {
                return container.value();
            } else if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
                return container.value();
            } else if constexpr (std::is_same_v<TensorContainer, torch::TensorList>) {
                return container;
            } else if constexpr (std::is_same_v<TensorContainer, torch::Tensor>) {
                return container;
            }
        };

        const auto &value = get_value();

        // 处理 Tensor 和 TensorList 的不同检查逻辑
        if constexpr (std::is_same_v<TensorContainer, torch::Tensor> ||
                      std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
            // 单个 Tensor 的处理
            if (!value.defined()) {
                TORCH_CHECK(allow_empty, build_error_msg(func_name, name, " tensor is not defined"));
                return allow_empty;
            }
            if (value.scalar_type() != expected_type) {
                return false;
            }
            return true;
        } else {
            // TensorList 的处理
            if (value.empty()) {
                TORCH_CHECK(allow_empty,
                            build_error_msg(func_name, name, " tensor list is empty, but empty is not allowed"));
                return allow_empty;
            }

            for (size_t i = 0; i < value.size(); ++i) {
                const torch::Tensor &tensor = value[i];
                if (!tensor.defined()) {
                    TORCH_CHECK(allow_empty,
                                build_error_msg(func_name, name, " tensor list contains undefined tensor"));
                    return allow_empty;
                }
                if (tensor.scalar_type() != expected_type) {
                    return false;
                }
            }
            return true;
        }
    }


    static int findMatchingCombo(const std::vector<TypeCombo> &combos, const torch::TensorList &x,
                                 const torch::TensorList &weight, const c10::optional<torch::TensorList> &bias,
                                 const c10::optional<torch::TensorList> &scale,
                                 const c10::optional<torch::TensorList> &offset,
                                 const c10::optional<torch::TensorList> &antiquantScale,
                                 const c10::optional<torch::TensorList> &antiquantOffset,
                                 const c10::optional<torch::Tensor> &groupList,
                                 const c10::optional<torch::TensorList> &perTokenScale)
    {
        for (size_t i = 0; i < combos.size(); ++i) {
            const auto &combo = combos[i];
            if (checkTensorType(x, combo.x, "x", false) && checkTensorType(weight, combo.weight, "weight", false) &&
                checkTensorType(bias, combo.bias, "bias", true) && checkTensorType(scale, combo.scale, "scale", true) &&
                checkTensorType(offset, combo.offset, "offset", true) &&
                checkTensorType(antiquantScale, combo.antiquantScale, "antiquantScale", true) &&
                checkTensorType(antiquantOffset, combo.antiquantOffset, "antiquantOffset", true) &&
                checkTensorType(perTokenScale, combo.perTokenScale, "perTokenScale", true) &&
                checkTensorType(groupList, combo.groupList, "groupList", true)) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }
};

template <typename TensorContainer>
void checkTensorOnNPU(const TensorContainer &container, const std::string &name = "curTensor", bool allow_empty = false)
{
    const std::string &func_name = "[checkTensorOnNPU]";

    // 处理 optional 类型
    if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>> ||
                  std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
        if (!container.has_value()) {
            TORCH_CHECK(allow_empty,
                        build_error_msg(func_name, name, " is not provided (null optional), but empty is not allowed"));
            return;
        }
    }
    // 获取实际的值或引用
    auto get_value = [&]() -> auto &
    {
        if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::TensorList>>) {
            return container.value();
        } else if constexpr (std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
            return container.value();
        } else if constexpr (std::is_same_v<TensorContainer, torch::TensorList>) {
            return container;
        } else if constexpr (std::is_same_v<TensorContainer, torch::Tensor>) {
            return container;
        }
    };
    const auto &value = get_value();
    // 处理 Tensor 和 TensorList 的不同检查逻辑
    if constexpr (std::is_same_v<TensorContainer, torch::Tensor> ||
                  std::is_same_v<TensorContainer, c10::optional<torch::Tensor>>) {
        // 单个 Tensor 的处理
        if (!value.defined()) {
            TORCH_CHECK(allow_empty,
                        build_error_msg(func_name, name, " tensor is undefined, but empty is not allowed"));
            return;
        }

        if (!torch_npu::utils::is_npu(value)) {
            TORCH_CHECK(false, build_error_msg(func_name, name, " tensor must be on NPU device"));
        }
    } else {
        // TensorList 的处理
        if (value.empty()) {
            TORCH_CHECK(allow_empty,
                        build_error_msg(func_name, name, " tensor list is empty, but empty is not allowed"));
            return;
        }
        for (size_t i = 0; i < value.size(); ++i) {
            const torch::Tensor &tensor = value[i];
            if (tensor.defined() && !torch_npu::utils::is_npu(tensor)) {
                std::string msg = " tensor at index " + std::to_string(i) + " must be on NPU device";
                TORCH_CHECK(false, build_error_msg(func_name, name, msg));
            }
        }
    }
}

template <typename TensorType, typename ElementType>
ElementType *get_first_tensor_address_by_type(const TensorType &input, bool allow_empty = false)
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


static std::vector<int64_t> getGroupListShape(const c10::optional<torch::Tensor> &groupList)
{
    std::vector<int64_t> shape;
    if (groupList.has_value()) {
        try {
            const torch::Tensor &tensor = groupList.value();
            shape = std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
        } catch (const std::exception &e) {
            std::cerr << "Error getting groupList shape: " << e.what() << std::endl;
            shape = {0};
        }
    } else {
        std::cout << "groupList is empty (no Tensor available)" << std::endl;
        shape = {0};
    }
    return shape;
}

static std::vector<int64_t> getTensorListFirstShape(const at::TensorList &tensorList)
{
    std::vector<int64_t> shape;
    if (!tensorList.empty()) {
        try {
            const at::Tensor &firstTensor = tensorList[0];
            shape = std::vector<int64_t>(firstTensor.sizes().begin(), firstTensor.sizes().end());
        } catch (const std::exception &e) {
            std::cerr << "Error getting first tensor shape from TensorList: " << e.what() << std::endl;
            shape = {0};
        }
    } else {
        std::cout << "TensorList is empty, no available tensor" << std::endl;
        shape = {0};
    }
    return shape;
}

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15U; // 15: 16 bytes up-align
    }
    return a & ~15U; // ~15: 16 bytes down-align
}

#endif