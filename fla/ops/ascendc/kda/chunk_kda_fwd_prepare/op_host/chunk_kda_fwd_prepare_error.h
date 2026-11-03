/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_KDA_FWD_PREPARE_ERROR_H
#define CHUNK_KDA_FWD_PREPARE_ERROR_H

#include <string>
#include <vector>
#include "op_common/log/log.h"
#include "chunk_kda_fwd_prepare_error_catalog.h"

namespace KdaPrepareDfx {
// A macro is usable only when its message is registered by this opbase source.
// Retain named values and constraints with official legacy reporters otherwise.
inline void Shape(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_SHAPE
    if constexpr (kHasEz0008) {
        OP_LOGE_FOR_INVALID_SHAPE(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    const std::string value = "shape=" + actual;
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), value.c_str(), expected.c_str());
}

inline void ShapeReason(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
    if constexpr (kHasEz0009) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    const std::string value = "shape=" + actual;
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), value.c_str(), expected.c_str());
}

inline void ShapesReason(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
    if constexpr (kHasEz0010) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    const std::string value = "shapes=" + actual;
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), value.c_str(), expected.c_str());
}

inline void RanksReason(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON
    if constexpr (kHasEz0013) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    const std::string value = "ranks=" + actual;
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), value.c_str(), expected.c_str());
}

inline void Format(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_FORMAT
    if constexpr (kHasEz0017) {
        OP_LOGE_FOR_INVALID_FORMAT(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_INPUT_FORMAT(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void FormatsReason(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON
    if constexpr (kHasEz0018) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_INPUT_FORMAT(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void Dtype(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_DTYPE
    if constexpr (kHasEz0019) {
        OP_LOGE_FOR_INVALID_DTYPE(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_INPUT_DTYPE(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void DtypesReason(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
    if constexpr (kHasEz0021) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_INPUT_DTYPE(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void Value(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_VALUE
    if constexpr (kHasEz0024) {
        OP_LOGE_FOR_INVALID_VALUE(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void ListSize(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_LISTSIZE
    if constexpr (kHasEz0025) {
        OP_LOGE_FOR_INVALID_LISTSIZE(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_ATTR_SIZE(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void ValueReason(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
    if constexpr (kHasEz0026) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void ValuesReason(const std::string &entity, const std::string &parameter,
    const std::string &actual, const std::string &expected)
{
#ifdef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
    if constexpr (kHasEz0027) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entity.c_str(), parameter.c_str(), actual, expected);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), actual.c_str(), expected.c_str());
}

inline void ArgumentReason(const std::string &entity, const std::string &parameter,
    const std::string &reason)
{
#ifdef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
    if constexpr (kHasEz0037) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity.c_str(), parameter.c_str(), reason);
        return;
    }
#endif
    OP_LOGE_WITH_INVALID_ATTR(entity.c_str(), parameter.c_str(), "invalid", reason.c_str());
}

inline void Missing(const std::string &entity, const std::string &parameter,
    const std::string &reason)
{
#ifdef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
    if constexpr (kHasEz0037) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(entity.c_str(), parameter.c_str(), reason);
        return;
    }
#endif
    // Legacy OP_LOGE_WITH_INVALID_INPUT contains a param_name spelling error.
    OP_LOGE_WITHOUT_REPORT(entity.c_str(), "Required parameter %s is null: %s",
        parameter.c_str(), reason.c_str());
    const std::vector<const char *> keys = {"op_name", "param_name"};
    const std::vector<const char *> values = {entity.c_str(), parameter.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0004", keys, values);
}
} // namespace KdaPrepareDfx
#endif
