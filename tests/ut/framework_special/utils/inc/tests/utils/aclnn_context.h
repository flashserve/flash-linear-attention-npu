/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclnn_context.h
 * \brief  提供 Aclnn Tiling / Kernel 阶段上下文功能, 辅助 Tiling / Kernel 运行.
 */

#pragma once

#include "tests/utils/context_intf.h"
#include <aclnn/aclnn_base.h>

namespace ops::adv::tests::utils {

class AclnnContext : public ops::adv::tests::utils::ContextIntf {
public:
    /**
     * Tiling 运行回调函数
     */
    typedef bool (*TilingRunCbf)(void *curCase, uint64_t *workSpaceSize, aclOpExecutor **opExecutor);
    typedef bool (*KernelRunCbf)(void *curCase);

public:
    AclnnContext() = default;
    ~AclnnContext() override;

    /* 属性设置 */
    [[maybe_unused]] [[nodiscard]] bool SetTilingRunCbf(TilingRunCbf cbf);
    [[maybe_unused]] [[nodiscard]] bool SetKernelRunCbf(KernelRunCbf cbf);

    aclrtStream GetAclRtStream() const;
    aclOpExecutor *GetAclOpExecutor() const;

    /* Tiling */
    bool RunTiling(std::string &caseName, bool withSocversion) override;

protected:
    TilingRunCbf tilingRunCbf_ = nullptr;
    KernelRunCbf kernelRunCbf_ = nullptr;
    aclrtStream aclrtStream_ = nullptr;
    aclOpExecutor *aclOpExecutor_ = nullptr;
    bool needDestroyExecutor_ = true;

protected:
    bool RunKernelProcess(std::string &caseName) override;
    uint8_t *AllocWorkspaceImpl(uint64_t size) override;
    void FreeWorkspaceImpl(uint8_t *ptr) override;

private:
    void Destroy();
};

} // namespace ops::adv::tests::utils
