/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_example_tiling.h
 * \brief
 */

#ifndef _I_ADD_EXAMPLE_TILING_H_
#define _I_ADD_EXAMPLE_TILING_H_

#include <cstdint>

#include "../../../op_kernel/add_example_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__
#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, AddExampleTilingData *constData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)constData;
    for (size_t i = 0; i < sizeof(AddExampleTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t *tiling, AddExampleTilingData *constData)
{
    memcpy(constData, tiling, sizeof(AddExampleTilingData));
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    AddExampleTilingData tilingData;    \
    InitTilingData(tilingArg, &tilingData)

#endif // _I_ADD_EXAMPLE_TILING_H_