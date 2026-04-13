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

#include "tests/utils/expect_info.h"

using namespace ops::adv::tests::utils;

ExpectInfo::ExpectInfo(bool success, uint64_t tilingKey, int64_t tilingBlockDim)
    : mSuccess(success), mTilingKey(tilingKey), mTilingBlockDim(tilingBlockDim)
{
}
