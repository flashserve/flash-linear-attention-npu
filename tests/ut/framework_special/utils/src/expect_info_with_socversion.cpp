/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file expect_info.cpp
 * \brief 测试用例执行结果期望信息.
 */

#include "tests/utils/expect_info_with_socversion.h"

using namespace ops::adv::tests::utils;

ExpectInfoWithSocversion::ExpectInfoWithSocversion() noexcept {
    for (auto &ele : mTilingKeys) {
        ele = kInvalidTilingKey;
    }
}

ExpectInfoWithSocversion::ExpectInfoWithSocversion(bool success, uint64_t tilingKey, int64_t tilingBlockDim)
    : mSuccess(success)
{
    for (auto &ele : mTilingKeys) {
        ele = tilingKey;
    }

    for (auto &ele : mTilingBlockDims) {
        ele = tilingBlockDim;
    }
}

ExpectInfoWithSocversion::ExpectInfoWithSocversion(bool success, const std::map<int32_t, uint64_t> &expTilingKeys, int64_t tilingBlockDim)
    : mSuccess(success)
{
    for (auto &ele : expTilingKeys) {
        mTilingKeys[ele.first] = ele.second;
        if (ele.first == (uint64_t)Platform::SocVersion::Ascend910B2) { // Ascend910B2
            mTilingKeys[(uint64_t)Platform::SocVersion::Ascend910B1] = ele.second;  // Ascend910B1
            mTilingKeys[(uint64_t)Platform::SocVersion::Ascend910B3] = ele.second;  // Ascend910B3
        }
    }

    for (auto &ele : mTilingBlockDims) {
        ele = tilingBlockDim;
    }
}

ExpectInfoWithSocversion::ExpectInfoWithSocversion(bool success, const std::map<int32_t, uint64_t> &expTilingKeys, 
        const std::map<int32_t, int64_t> &expTilingBlockDims) : mSuccess(success)
{
    for (auto &ele : expTilingKeys) {
        mTilingKeys[ele.first] = ele.second;
        if (ele.first == (uint64_t)Platform::SocVersion::Ascend910B2) { // Ascend910B2
            mTilingKeys[(uint64_t)Platform::SocVersion::Ascend910B1] = ele.second;  // Ascend910B1
            mTilingKeys[(uint64_t)Platform::SocVersion::Ascend910B3] = ele.second;  // Ascend910B3
        }
    }

    for (auto &ele : expTilingBlockDims) {
        mTilingBlockDims[ele.first] = ele.second;
        if (ele.first == (uint64_t)Platform::SocVersion::Ascend910B2) { // Ascend910B2
            mTilingBlockDims[(uint64_t)Platform::SocVersion::Ascend910B1] = ele.second;  // Ascend910B1
            mTilingBlockDims[(uint64_t)Platform::SocVersion::Ascend910B3] = ele.second;  // Ascend910B3
        }
    }
}
