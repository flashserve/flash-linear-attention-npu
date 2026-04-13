/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file control_info.h
 * \brief 测试用例控制信息.
 */

#pragma once

namespace ops::adv::tests::utils {

class ControlInfo {
public:
    /**
     * 是否执行 Tiling
     *
     * \attention
     * 默认值 true, 谨慎修改默认值, 默认 false 会导致未设置该值的用例不执行
     */
    bool mRunTiling = true;

    /**
     * 是否执行 Kernel
     *
     * \attention
     * 默认值 false, Kernel 执行比较耗时, 如需执行, 需指定开启
     */
    bool mRunKernel = false;

    /**
     * 确定性计算值
     */
    bool mDeterministic = false;

public:
    ControlInfo() = default;
    ControlInfo(bool runTiling, bool runKernel, bool deterministic = false);
};

} // namespace ops::adv::tests::utils
