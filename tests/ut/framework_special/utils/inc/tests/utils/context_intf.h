/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file context_intf.h
 * \brief 提供 Tiling / Kernel 阶段上下文功能基类, 辅助 Tiling / Kernel 运行.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "tests/utils/tensor_intf.h"
#include "tests/utils/platform.h"

namespace ops::adv::tests::utils {

class ContextIntf {
public:
    static constexpr uint64_t kWorkspaceMaxSize = std::uint64_t(2) * std::uint64_t(1024 * 1024 * 1024);  // 2GB

    /**
     * Kernel 运行前设置 TensorData 回调函数
     */
    typedef bool (*KernelRunPrepareTensorDataCbf)(void *curCase);

public:
    ContextIntf() = default;
    virtual ~ContextIntf() = default;

    /* 属性设置 */
    [[maybe_unused]] [[nodiscard]] bool SetOpName(const char *name);
    [[maybe_unused]] [[nodiscard]] bool SetInputs(const std::vector<TensorIntf *> &inputs);
    [[maybe_unused]] [[nodiscard]] bool SetOutputs(const std::vector<TensorIntf *> &outputs);
    [[maybe_unused]] [[nodiscard]] bool SetKernelRunPrepareTensorDataCbf(KernelRunPrepareTensorDataCbf cbf);

    /**
     * 属性获取
     * 获取 Tiling 结果中 BlockDim 数量(分核数)
     */
    [[maybe_unused]] [[nodiscard]] int64_t GetTilingBlockDim() const;

    /**
     * 属性获取
     * 获取 Tiling 结果中 TilingKey 值
     */
    [[maybe_unused]] [[nodiscard]] uint64_t GetTilingKey() const;

    /**
     * 属性获取
     * 获取 Workspace 指针
     *
     * \attention Workspace 在 RunKernelPrepare 回调函数内完成申请
     */
    [[maybe_unused]] [[nodiscard]] uint8_t *GetWorkspacePtr() const;

    /**
     * 属性获取
     * 获取 Tiling 结果中 Workspace 大小
     */
    [[maybe_unused]] [[nodiscard]] uint64_t GetWorkspaceSize() const;

    /* Tiling */
    virtual bool RunTiling(std::string &caseName, bool withSocversion) = 0;

    /* Kernel */
    virtual bool RunKernel(std::string &caseName);

    /* Others */
    bool SaveOutputsToDir(const std::string &dir, const std::string &filePrefix);

protected:
    Platform *platform_ = nullptr;
    /* 属性 */
    std::string opName_;
    std::vector<TensorIntf *> inputs_;
    std::vector<TensorIntf *> outputs_;
    KernelRunPrepareTensorDataCbf kernelRunPrepareTensorDataCbf_ = nullptr;

    /* Tiling & Kernel */
    int64_t tilingBlockDim_ = 0;
    uint64_t tilingKey_ = std::numeric_limits<uint64_t>::max();
    uint64_t workspaceSize_ = 0;
    uint8_t *workspacePtr_ = nullptr;

protected:
    virtual bool RunKernelPrepare(std::string &caseName);
    virtual bool RunKernelProcess(std::string &caseName) = 0;

    virtual uint8_t *AllocWorkspaceImpl(uint64_t size) = 0;
    virtual void FreeWorkspaceImpl(uint8_t *ptr) = 0;
};

} // namespace ops::adv::tests::utils
