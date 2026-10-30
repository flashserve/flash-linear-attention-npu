/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#include "chunk_gated_delta_rule_fwd_error.h"

#include <vector>
#include "op_common/log/log.h"

namespace gdn_error {
namespace {
constexpr const char *ENTITY = "aclnnChunkGatedDeltaRuleFwd";
}

void Required(const char *name)
{
    OP_LOGE_WITH_INVALID_INPUT(ENTITY, name);
}

void Rank(const char *name, size_t actual, size_t expected)
{
    OP_LOGE_FOR_INVALID_SHAPEDIM(ENTITY, name, std::to_string(actual), std::to_string(expected));
}

void Shape(const char *name, const std::string &actual, const std::string &expected)
{
    OP_LOGE_FOR_INVALID_SHAPE(ENTITY, name, actual, expected);
}

void Dtype(const char *name, const std::string &actual, const std::string &expected, bool output)
{
    if (output) {
        OP_LOGE_FOR_INVALID_DTYPE(ENTITY, name, actual, expected);
    } else {
        OP_LOGE_WITH_INVALID_INPUT_DTYPE(ENTITY, name, actual, expected);
    }
}

void Attr(const char *name, const std::string &actual, const std::string &expected)
{
    OP_LOGE_WITH_INVALID_ATTR(ENTITY, name, actual, expected);
}

void Argument(const char *name, const std::string &reason)
{
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ENTITY, name, reason);
}

void ListSize(const char *name, size_t actual, const std::string &expected)
{
    OP_LOGE_FOR_INVALID_LISTSIZE(ENTITY, name, std::to_string(actual), expected);
}
} // namespace gdn_error
