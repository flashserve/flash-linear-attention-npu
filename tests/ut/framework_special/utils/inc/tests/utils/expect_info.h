/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file expect_info.h
 * \brief 测试用例执行结果期望信息.
 */

#pragma once

#include <cstdint>
#include <limits>

namespace ops::adv::tests::utils {

class ExpectInfo {
public:
    static constexpr uint64_t kInvalidTilingKey = std::numeric_limits<uint64_t>::max();
    static constexpr int64_t kInvalidTilingBlockDim = 0;
    static constexpr int64_t kFullTilingBlockDim = -1;

public:
    /**
     * 期望该用例成功
     */
    bool mSuccess = true;

    /**
     * 期望 TilingKey 取值
     */
    uint64_t mTilingKey = kInvalidTilingKey;

    /**
     * 期望 TilingBlockDim 取值
     */
    int64_t mTilingBlockDim = kInvalidTilingBlockDim;

public:
    ExpectInfo() = default;
    ExpectInfo(bool success, uint64_t tilingKey, int64_t tilingBlockDim);
};

} // namespace ops::adv::tests::utils
