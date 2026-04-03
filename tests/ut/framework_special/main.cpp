/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file main.cpp
 * \brief 本文件作为 GTest 用例 main 桩文件, 一般不需修改任何内容.
 */

#include <gtest/gtest.h>
#include "tests/utils/platform.h"

using Platform = ops::adv::tests::utils::Platform;

int main(int argc, char **argv)
{
    /**
     * 加载 OpTiling.so
     * 注意必须加载 OpTiling.so, 因为加载时才会触发框架(register) 执行算子处理函数注册.
     */
    Platform platform;
    Platform::SetGlobalPlatform(&platform);
    if (!platform.InitArgsInfo(argc, argv)) {
        return 1;
    }
    if (!platform.LoadOpTilingSo()) {
        return 1;
    }

    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();

    if (!platform.UnLoadOpTilingSo()) {
        return 1;
    }

    return ret;
}
