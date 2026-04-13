/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file context_intf.cpp
 * \brief 提供 Tiling / Kernel 阶段上下文功能基类, 辅助 Tiling / Kernel 运行.
 */

#include "tests/utils/context_intf.h"
#include <algorithm>
#include "tests/utils/case.h"
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

[[maybe_unused]] [[nodiscard]] bool ContextIntf::SetOpName(const char *name)
{
    opName_ = name;
    return true;
}

bool ContextIntf::SetInputs(const std::vector<TensorIntf *> &inputs)
{
    if (std::any_of(inputs.begin(), inputs.end(), [](TensorIntf *t) { return t == nullptr; })) {
        return false;
    }
    inputs_.insert(inputs_.end(), inputs.begin(), inputs.end());
    return true;
}

bool ContextIntf::SetOutputs(const std::vector<TensorIntf *> &outputs)
{
    if (std::any_of(outputs.begin(), outputs.end(), [](TensorIntf *t) { return t == nullptr; })) {
        return false;
    }
    outputs_.insert(outputs_.end(), outputs.begin(), outputs.end());
    return true;
}

bool ContextIntf::SetKernelRunPrepareTensorDataCbf(KernelRunPrepareTensorDataCbf cbf)
{
    if (cbf == nullptr) {
        return false;
    }
    kernelRunPrepareTensorDataCbf_ = cbf;
    return true;
}

int64_t ContextIntf::GetTilingBlockDim() const
{
    return tilingBlockDim_;
}

uint64_t ContextIntf::GetTilingKey() const
{
    return tilingKey_;
}

uint8_t *ContextIntf::GetWorkspacePtr() const
{
    return workspacePtr_;
}

size_t ContextIntf::GetWorkspaceSize() const
{
    return workspaceSize_;
}

bool ContextIntf::RunKernel(std::string &caseName)
{
    platform_ = platform_ == nullptr ? Platform::GetGlobalPlatform() : platform_;
    if (platform_ == nullptr) {
        LOG_ERR("[%s:%s] Can't get global platform.", opName_.c_str(), caseName.c_str());
        return false;
    }
    auto ret = this->RunKernelPrepare(caseName);
    ret = ret && this->RunKernelProcess(caseName);
    return ret;
}

bool ContextIntf::SaveOutputsToDir(const std::string &dir, const std::string &filePrefix)
{
    for (auto o : outputs_) {
        if (o->GetExpDataSize() <= 0) {
            continue;
        }
        std::string filePath = dir + filePrefix + o->Name() + ".bin";
        if (!o->SaveDevDataToFile(filePath)) {
            return false;
        }
    }
    return true;
}

bool ContextIntf::RunKernelPrepare(std::string &caseName)
{
    if (workspaceSize_ > kWorkspaceMaxSize) {
        LOG_ERR("[%s:%s] workspace(%lu) > kWorkspaceMaxSize(%lu)", opName_.c_str(), caseName.c_str(), workspaceSize_,
                kWorkspaceMaxSize);
        return false;
    }
    if (workspaceSize_ > 0) {
        workspacePtr_ = this->AllocWorkspaceImpl(workspaceSize_);
        if (workspacePtr_ == nullptr) {
            return false;
        }
    }
    int initV = 0;
    for (auto i : inputs_) {
        if (i->GetExpDataSize() > 0) {
            if (i->AllocDevData(++initV, 0) == nullptr) {
                return false;
            }
        }
    }
    for (auto o : outputs_) {
        if (o->GetExpDataSize() > 0) {
            if (o->AllocDevData(0, 0) == nullptr) {
                return false;
            }
        }
    }
    if (kernelRunPrepareTensorDataCbf_ != nullptr) {
        auto *cs = static_cast<ops::adv::tests::utils::Case *>(ops::adv::tests::utils::Case::GetCurrentCase());
        LOG_IF_EXPR(cs == nullptr, LOG_ERR("[%s:%s], Can't get current case", opName_.c_str(), caseName.c_str()),
                    return false);
        LOG_IF_EXPR(!KernelRunPrepareTensorDataCbf(cs),
                    LOG_ERR("[%s:%s] KernelRunPrepareTensorDataCbf failed", opName_.c_str(), caseName.c_str()),
                    return false);
    }
    return true;
}
