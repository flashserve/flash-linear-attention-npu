/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef PRE_PROCESS_BWD_KERNEL_MERGED_DFX_REPORT_H
#define PRE_PROCESS_BWD_KERNEL_MERGED_DFX_REPORT_H

#include <string>
#include <vector>
#include "base/err_msg.h"

// L2 uses opdev logging, which cannot be replaced by op_common's same-name
// macros. Report registered fields directly there; prefer the native semantic
// macros when the host translation unit has already imported them.
#define CP_DFX_NULL(op, param) \
    do { \
        const std::string o(op), p(param); \
        REPORT_PREDEFINED_ERR_MSG("EZ0004", std::vector<const char *>({"param_name", "op_name"}), \
            std::vector<const char *>({p.c_str(), o.c_str()})); \
    } while (false)

#define CP_DFX_ARGUMENT(op, param, reason) \
    do { \
        const std::string o(op), p(param), r(reason); \
        REPORT_PREDEFINED_ERR_MSG("EZ0037", std::vector<const char *>({"param_name", "op_name", "reason"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), r.c_str()})); \
    } while (false)

#define CP_DFX_SHAPE(op, param, actual, expected) \
    do { \
        const std::string o(op), p(param), a(actual), e(expected); \
        REPORT_PREDEFINED_ERR_MSG("EZ0008", std::vector<const char *>({"param_name", "op_name", "incorrect_shape", "correct_shape"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), e.c_str()})); \
    } while (false)

#define CP_DFX_SHAPE_REASON(op, param, actual, reason) \
    do { \
        const std::string o(op), p(param), a(actual), r(reason); \
        REPORT_PREDEFINED_ERR_MSG("EZ0009", std::vector<const char *>({"param_name", "op_name", "incorrect_shape", "reason"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), r.c_str()})); \
    } while (false)

#define CP_DFX_RANK(op, param, actual, expected) \
    do { \
        const std::string o(op), p(param), a(actual), e(expected); \
        REPORT_PREDEFINED_ERR_MSG("EZ0011", std::vector<const char *>({"param_name", "op_name", "incorrect_dim", "correct_dim"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), e.c_str()})); \
    } while (false)

#define CP_DFX_DTYPE(op, param, actual, reason) \
    do { \
        const std::string o(op), p(param), a(actual), r(reason); \
        REPORT_PREDEFINED_ERR_MSG("EZ0020", std::vector<const char *>({"param_name", "op_name", "incorrect_dtype", "reason"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), r.c_str()})); \
    } while (false)

#define CP_DFX_VALUE(op, param, actual, expected) \
    do { \
        const std::string o(op), p(param), a(actual), e(expected); \
        REPORT_PREDEFINED_ERR_MSG("EZ0024", std::vector<const char *>({"param_name", "op_name", "incorrect_value", "correct_value"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), e.c_str()})); \
    } while (false)

#define CP_DFX_LIST(op, param, actual, expected) \
    do { \
        const std::string o(op), p(param), a(actual), e(expected); \
        REPORT_PREDEFINED_ERR_MSG("EZ0025", std::vector<const char *>({"param_name", "op_name", "incorrect_size", "correct_size"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), e.c_str()})); \
    } while (false)

#define CP_DFX_VALUE_REASON(op, param, actual, reason) \
    do { \
        const std::string o(op), p(param), a(actual), r(reason); \
        REPORT_PREDEFINED_ERR_MSG("EZ0026", std::vector<const char *>({"param_name", "op_name", "incorrect_value", "reason"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), r.c_str()})); \
    } while (false)

#define CP_DFX_VALUES(op, params, actual, reason) \
    do { \
        const std::string o(op), p(params), a(actual), r(reason); \
        REPORT_PREDEFINED_ERR_MSG("EZ0027", std::vector<const char *>({"param_names", "op_name", "incorrect_values", "reason"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), r.c_str()})); \
    } while (false)

#define CP_DFX_TILING(op, reason) \
    do { \
        const std::string o(op), r(reason); \
        REPORT_PREDEFINED_ERR_MSG("EZ1008", std::vector<const char *>({"opName", "reason"}), \
            std::vector<const char *>({o.c_str(), r.c_str()})); \
    } while (false)

#define CP_DFX_STRIDE(op, param, actual, expected) \
    do { \
        const std::string o(op), p(param), a(actual), e(expected); \
        REPORT_PREDEFINED_ERR_MSG("EZ0028", std::vector<const char *>({"param_name", "op_name", "incorrect_stride", "correct_stride"}), \
            std::vector<const char *>({p.c_str(), o.c_str(), a.c_str(), e.c_str()})); \
    } while (false)

#ifdef OP_LOGE_FOR_INVALID_SHAPE
#undef CP_DFX_NULL
#define CP_DFX_NULL OP_LOGE_WITH_INVALID_INPUT
#undef CP_DFX_ARGUMENT
#define CP_DFX_ARGUMENT OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#undef CP_DFX_SHAPE
#define CP_DFX_SHAPE OP_LOGE_FOR_INVALID_SHAPE
#undef CP_DFX_SHAPE_REASON
#define CP_DFX_SHAPE_REASON OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#undef CP_DFX_RANK
#define CP_DFX_RANK OP_LOGE_FOR_INVALID_SHAPEDIM
#undef CP_DFX_DTYPE
#define CP_DFX_DTYPE OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#undef CP_DFX_VALUE
#define CP_DFX_VALUE OP_LOGE_FOR_INVALID_VALUE
#undef CP_DFX_LIST
#define CP_DFX_LIST OP_LOGE_FOR_INVALID_LISTSIZE
#undef CP_DFX_VALUE_REASON
#define CP_DFX_VALUE_REASON OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#undef CP_DFX_VALUES
#define CP_DFX_VALUES OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#undef CP_DFX_STRIDE
#define CP_DFX_STRIDE OP_LOGE_FOR_INVALID_STRIDE
#endif

#endif
