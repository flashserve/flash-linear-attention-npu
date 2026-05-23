/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file prepare_wy_repr_bwd_full_tiling_a5.h
 * \brief
 */
#ifndef PREPARE_WY_REPR_BWD_FULL_TILING_A5_H
#define PREPARE_WY_REPR_BWD_FULL_TILING_A5_H

// #include <exe_graph/runtime/tiling_context.h>
// #include <graph/utils/type_utils.h>

// #include "register/tilingdata_base.h"
// #include "tiling/tiling_api.h"

#include "../prepare_wy_repr_bwd_full_tiling.h"
#include "../../../op_kernel/arch35/prepare_wy_repr_bwd_full_tiling_data_apt.h"
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_impl_registry.h"

namespace optiling {

class PrepareWyReprBwdFullTilingA5 {
private:
    gert::TilingContext *context_;
    PrepareWyReprBwdFullTilingDataA5 tiling_;
    int64_t B = 0;
    int64_t HK = 0;
    int64_t HV = 0;
    int64_t K = 0;
    int64_t V = 0;
    int64_t T = 0;
    int64_t chunkSize = 0;

public:
    bool SetTiling(gert::TilingContext *context);
    ge::graphStatus CommonTiling();
    ge::graphStatus FixLenTiling();
    ge::graphStatus VariableLenTiling();
    ge::graphStatus RequiredInputDimNumCheck(const gert::StorageShape *curShape, size_t validDimNum,
                                            const char *inputName);
    ge::graphStatus CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, const char *inputName1,
                                const char *inputName2, size_t compareDimNum);
    ge::graphStatus SetKBeteVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType);
    ge::graphStatus SetDkbVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType);
    ge::graphStatus SetDkbgVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType);
    ge::graphStatus SetDvbVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType);
    ge::graphStatus SetKKTVecRowRegbase(uint64_t ubSize, ge::DataType kType, ge::DataType betaType);
    int64_t CeilDiv(int64_t a, int64_t b);
    void PrepareWyReprBwdFullTilingDataPrint();
};

} // namespace optiling

#endif // PREPARE_WY_REPR_BWD_FULL_TILING_H