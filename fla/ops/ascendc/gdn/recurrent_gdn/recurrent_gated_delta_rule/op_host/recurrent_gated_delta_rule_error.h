/** Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef RECURRENT_GATED_DELTA_RULE_ERROR_H
#define RECURRENT_GATED_DELTA_RULE_ERROR_H

// Load the structured reporters before opdev's legacy logging definitions.
#include "op_common/log/log.h"
#include "graph/utils/type_utils.h"
#include <sstream>
#include <string>
#include <cstdio>
#include <vector>

namespace rgdn_error {
template <typename... Args>
inline std::string Format(const char *format, Args... args)
{
    const int size = std::snprintf(nullptr, 0, format, args...);
    if (size < 0) return format;
    std::vector<char> text(static_cast<size_t>(size) + 1);
    std::snprintf(text.data(), text.size(), format, args...);
    return std::string(text.data(), static_cast<size_t>(size));
}

template <typename Shape>
inline std::string ShapeText(const Shape &shape)
{
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) text << ',';
        text << shape.GetDim(i);
    }
    text << ']';
    return text.str();
}

template <typename Strides>
inline std::string StrideText(const Strides &strides)
{
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i != 0) text << ',';
        text << strides[i];
    }
    text << ']';
    return text.str();
}
} // namespace rgdn_error
#endif
