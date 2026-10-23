/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef CHUNK_GATED_DELTA_RULE_FWD_ERROR_H
#define CHUNK_GATED_DELTA_RULE_FWD_ERROR_H

#include <string>

// Keep op_common and opdev logging macros in separate translation units.
namespace gdn_error {
void Required(const char *name);
void Rank(const char *name, size_t actual, size_t expected);
void Shape(const char *name, const std::string &actual, const std::string &expected);
void Dtype(const char *name, const std::string &actual, const std::string &expected, bool output);
void Attr(const char *name, const std::string &actual, const std::string &expected);
void Argument(const char *name, const std::string &reason);
void ListSize(const char *name, size_t actual, const std::string &expected);
} // namespace gdn_error

#endif
