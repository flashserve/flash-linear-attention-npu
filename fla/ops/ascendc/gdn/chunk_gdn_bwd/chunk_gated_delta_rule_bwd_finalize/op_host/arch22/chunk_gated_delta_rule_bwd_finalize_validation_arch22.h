#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VALIDATION_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_VALIDATION_ARCH22_H

#include "../chunk_gated_delta_rule_bwd_finalize_validation.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <sstream>
#include <string>

namespace GDN {
namespace FinalizeHostArch22 {

using FinalizeHost::ChunkEndFitsUint32;

template <typename Shape>
std::string ShapeString(const Shape &shape)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
        if (index != 0) {
            stream << ", ";
        }
        stream << shape.GetDim(index);
    }
    stream << "]";
    return stream.str();
}

inline std::string DimensionsString(std::initializer_list<int64_t> dimensions)
{
    std::ostringstream stream;
    stream << "[";
    size_t index = 0;
    for (const int64_t dimension : dimensions) {
        stream << (index++ == 0 ? "" : ", ") << dimension;
    }
    stream << "]";
    return stream.str();
}

template <typename Shape>
bool HasShape(const Shape &shape, std::initializer_list<int64_t> expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    size_t index = 0;
    for (const int64_t dimension : expected) {
        if (shape.GetDim(index++) != dimension) {
            return false;
        }
    }
    return true;
}

} // namespace FinalizeHostArch22
} // namespace GDN

#endif
