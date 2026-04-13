/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file log.h
 * \brief
 */

#pragma once

#include <cstdio>
#include <type_traits>
#include <alog_pub.h>

namespace ops::adv::tests::utils {

void AddLogErrCnt();

bool ChkLogErrCnt();

#define LOG_ERR(fmt, args...)                                                                                          \
    do {                                                                                                               \
        fprintf(stdout, "%s:%d [ERROR] " fmt "\n", __FILE__, __LINE__, ##args);                                        \
        ops::adv::tests::utils::AddLogErrCnt();                                                                        \
    } while (0)

#define LOG_DBG(fmt, args...)                                                                                          \
    do {                                                                                                               \
        if (AlogCheckDebugLevel(OP, DLOG_DEBUG) == 1) {                                                                      \
            fprintf(stdout, "%s:%d [DEBUG] " fmt "\n", __FILE__, __LINE__, ##args);                                    \
        }                                                                                                              \
    } while (0)

#define LOG_IF(COND, LOG_FUNC)                                                                                         \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool");            \
    do {                                                                                                               \
        if (__builtin_expect((COND), 0)) {                                                                             \
            LOG_FUNC;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define LOG_IF_EXPR(COND, LOG_FUNC, EXPR)                                                                              \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool");            \
    do {                                                                                                               \
        if (__builtin_expect((COND), 0)) {                                                                             \
            LOG_FUNC;                                                                                                  \
            EXPR;                                                                                                      \
        }                                                                                                              \
    } while (0)

#define IF_EXPR(COND, EXPR)                                                                                            \
    static_assert(std::is_same<bool, std::decay<decltype(COND)>::type>::value, "condition should be bool");            \
    do {                                                                                                               \
        if (__builtin_expect((COND), 0)) {                                                                             \
            EXPR;                                                                                                      \
        }                                                                                                              \
    } while (0)

} // namespace ops::adv::tests::utils
