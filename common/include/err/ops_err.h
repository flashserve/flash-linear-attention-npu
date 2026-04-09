/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file ops_err.h
 * \brief
 */

#ifndef Transformer_COMMON_OPS_ERR_H
#define Transformer_COMMON_OPS_ERR_H

#include "log/log.h"

#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)                                                           \
    do {                                                                                                               \
        OpLogSub(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__);                                                    \
        REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);                                                        \
    } while (0)


/* 基础报错 */
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E89999", OPS_DESC, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E69999", OPS_DESC, __VA_ARGS__)

#endif // Transformer_COMMON_OPS_ERR_H