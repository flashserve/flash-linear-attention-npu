/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file log.cpp
 * \brief
 */

#include "tests/utils/log.h"

namespace {
uint32_t g_LogErrCnt = 0;
}

void ops::adv::tests::utils::AddLogErrCnt()
{
    g_LogErrCnt++;
}

bool ops::adv::tests::utils::ChkLogErrCnt()
{
    auto rst = g_LogErrCnt == 0;
    if (!rst) {
        fprintf(stdout, "%s:%d [ERROR] There have %u error log in current case.\n", __FILE__, __LINE__, g_LogErrCnt);
        g_LogErrCnt = 0;
    }
    return rst;
}
