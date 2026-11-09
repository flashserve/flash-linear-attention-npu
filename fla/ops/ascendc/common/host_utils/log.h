/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef FLA_NPU_ASCENDC_COMMON_HOST_UTILS_LOG_H
#define FLA_NPU_ASCENDC_COMMON_HOST_UTILS_LOG_H

#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "graph/types.h"
#include "base/err_msg.h"

// 上下文分流：tiling TU 已引入 op_common/log/log.h 时统一定义 FLA_NPU_HOST_TILING。
#if defined(OP_COMMON_LOG_LOG_H)
#ifndef FLA_NPU_HOST_TILING
#define FLA_NPU_HOST_TILING
#endif
#endif

// 兼容性桥接层：为 aclnn 层与 tiling 层提供统一的结构化 DFX 上报接口。
//
// 上下文自动分流（issue #991 / PR review：宏检查统一放本文件）：
//   - tiling 层：本 TU 已引入 op_common/log/log.h（其保护宏 OP_COMMON_LOG_LOG_H
//                 已定义）时，在此统一定义 FLA_NPU_HOST_TILING，
//                 OP_LOGE 为 (opName, ...) 签名；
//   - aclnn 层： 默认不定义 FLA_NPU_HOST_TILING，OP_LOGE 为 (errno, ...) 签名；
//                 aclnn TU 不引入 op_common，故不会误判为 tiling。
//  算子文件不再维护 __has_include / FLA_NPU_HOST_TILING 探测。
//
// 新版本（CANN 9.2+ / 提供 OP_LOGE_FOR_* 结构化宏的环境）走结构化宏，
// 产生 EZ 语义码（EZ0004/EZ0024/...）；旧版本（CANN 8.5/9.0/9.1 或 opbase
// 无 FOR_* 宏的环境）自动降级为可读文本，保证算子的旧版本兼容性承诺。

namespace fla_npu {
namespace host_utils {

inline std::string DfxMakeShapeString(std::initializer_list<int64_t> dims)
{
    std::ostringstream stream;
    stream << "[";
    bool first = true;
    for (int64_t dim : dims) {
        if (!first) {
            stream << ",";
        }
        stream << dim;
        first = false;
    }
    stream << "]";
    return stream.str();
}

inline std::string DfxShapeString(const gert::Shape &shape)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t idx = 0; idx < static_cast<size_t>(shape.GetDimNum()); ++idx) {
        if (idx != 0) {
            stream << ",";
        }
        stream << shape.GetDim(idx);
    }
    stream << "]";
    return stream.str();
}

inline std::string DfxDtypeString(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BF16";
        case ge::DT_FLOAT:
            return "FLOAT";
        default:
            return std::to_string(static_cast<int32_t>(dtype));
    }
}

// 降级上报日志：tiling 上下文的 OP_LOGE(opName, ...) 无 errno；aclnn 上下文的 OP_LOGE(errno, ...)。
#if defined(FLA_NPU_HOST_TILING)
#define FLA_NPU_DFX_LOGE(entity, errno, ...) OP_LOGE(entity, __VA_ARGS__)
#else
#define FLA_NPU_DFX_LOGE(entity, errno, ...) OP_LOGE(errno, __VA_ARGS__)
#endif

inline void DfxReportInvalidInput(const char *entity, const char *paramName)
{
#if defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name"};
    const std::vector<const char *> msgvalue = {paramName, entity};
    REPORT_PREDEFINED_ERR_MSG("EZ0004", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_NULLPTR, "Parameter %s of %s is required, but it is empty.",
                     paramName, entity);
#endif
}

inline void DfxReportInvalidShape(const char *entity, const char *paramName, const std::string &actual,
                                  const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_SHAPE)
    OP_LOGE_FOR_INVALID_SHAPE(entity, paramName, actual.c_str(), expected.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_shape", "correct_shape"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0008", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect shape %s. It should be %s.",
                     paramName, entity, actual.c_str(), expected.c_str());
#endif
}

inline void DfxReportInvalidShapeWithReason(const char *entity, const char *paramName,
                                            const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON)
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entity, paramName, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_shape", "reason"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0009", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect shape %s. Reason: %s.",
                     paramName, entity, actual.c_str(), reason.c_str());
#endif
}

inline void DfxReportInvalidShapes(const char *entity, const char *paramNames, const std::string &actual,
                                   const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON)
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(entity, paramNames, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_shapes", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameters %s of %s have incorrect shapes %s. Reason: %s.",
                     paramNames, entity, actual.c_str(), reason.c_str());
#endif
}

inline void DfxReportInvalidShapesWithReason(const char *entity, const char *paramNames,
                                             const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON)
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(entity, paramNames, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_shapes", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameters %s of %s have incorrect shapes %s. Reason: %s.",
                     paramNames, entity, actual.c_str(), reason.c_str());
#endif
}

inline void DfxReportInvalidDtype(const char *entity, const char *paramName, const std::string &actual,
                                  const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_DTYPE)
    OP_LOGE_FOR_INVALID_DTYPE(entity, paramName, actual.c_str(), expected.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_dtype", "correct_dtype"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0019", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect dtype %s. It should be %s.",
                     paramName, entity, actual.c_str(), expected.c_str());
#endif
}

inline void DfxReportInvalidDtypes(const char *entity, const char *paramNames, const std::string &actual,
                                   const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON)
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(entity, paramNames, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_dtypes", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0021", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameters %s of %s have incorrect dtypes %s. Reason: %s.",
                     paramNames, entity, actual.c_str(), reason.c_str());
#endif
}

inline void DfxReportInvalidValue(const char *entity, const char *paramName, const std::string &actual,
                                  const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_VALUE)
    OP_LOGE_FOR_INVALID_VALUE(entity, paramName, actual.c_str(), expected.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_value", "correct_value"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0024", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect value %s. It should be %s.",
                     paramName, entity, actual.c_str(), expected.c_str());
#endif
}

inline void DfxReportInvalidValueWithReason(const char *entity, const char *paramName,
                                            const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_VALUE_WITH_REASON)
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entity, paramName, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_value", "reason"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0026", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect value %s. Reason: %s.",
                     paramName, entity, actual.c_str(), reason.c_str());
#endif
}

inline void DfxReportInvalidValues(const char *entity, const char *paramNames, const std::string &actual,
                                   const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_VALUES_WITH_REASON)
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entity, paramNames, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_values", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0027", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameters %s of %s have incorrect values %s. Reason: %s.",
                     paramNames, entity, actual.c_str(), reason.c_str());
#endif
}

inline void DfxReportInvalidValuesWithReason(const char *entity, const char *paramNames,
                                             const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_VALUES_WITH_REASON)
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entity, paramNames, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_values", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0027", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameters %s of %s have incorrect values %s. Reason: %s.",
                     paramNames, entity, actual.c_str(), reason.c_str());
#endif
}

inline void DfxReportInvalidListSize(const char *entity, const char *paramName, const std::string &actual,
                                     const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_LISTSIZE)
    OP_LOGE_FOR_INVALID_LISTSIZE(entity, paramName, actual.c_str(), expected.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_size", "correct_size"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0025", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect element nums %s. It should be %s.",
                     paramName, entity, actual.c_str(), expected.c_str());
#endif
}

inline void DfxReportInvalidFormat(const char *entity, const char *paramName, const std::string &actual,
                                     const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_FORMAT)
    OP_LOGE_FOR_INVALID_FORMAT(entity, paramName, actual.c_str(), expected.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_format", "correct_format"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0017", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect format %s. It should be %s.",
                     paramName, entity, actual.c_str(), expected.c_str());
#endif
}

inline void DfxReportInvalidFormatWithReason(const char *entity, const char *paramName,
                                             const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON)
    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(entity, paramName, actual.c_str(), reason.c_str());
#elif defined(REPORT_PREDEFINED_ERR_MSG)
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_format", "reason"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0035", msgKey, msgvalue);
#else
    FLA_NPU_DFX_LOGE(entity, ACLNN_ERR_PARAM_INVALID,
                     "Parameter %s of %s has incorrect format %s. Reason: %s.",
                     paramName, entity, actual.c_str(), reason.c_str());
#endif
}

#undef FLA_NPU_DFX_LOGE

} // namespace host_utils
} // namespace fla_npu

// ---------------------------------------------------------------------------
// 宏封装层（_COMPAT 兼容接口）：把上层调用从 "fla_npu::host_utils::DfxReport*"
// 的多层调用封装成单层宏，调用点只写一个宏名 + 参数，不再逐层展开。
// _COMPAT 后缀表明这是兼容层（compatible）宏，同时保留结构化 EZ 码上报。
// ---------------------------------------------------------------------------

#define OP_LOGE_LIBOPAPI_REPORT_COMPAT(entity, paramName) \
    fla_npu::host_utils::DfxReportInvalidInput((entity), (paramName))

#define OP_LOGE_LIBOPAPI_REPORT_SHAPE_COMPAT(entity, paramName, actual, expected) \
    fla_npu::host_utils::DfxReportInvalidShape((entity), (paramName), (actual), (expected))

#define OP_LOGE_LIBOPAPI_REPORT_SHAPE_WITH_REASON_COMPAT(entity, paramName, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidShapeWithReason((entity), (paramName), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_SHAPES_COMPAT(entity, paramNames, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidShapes((entity), (paramNames), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_SHAPES_WITH_REASON_COMPAT(entity, paramNames, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidShapesWithReason((entity), (paramNames), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_DTYPE_COMPAT(entity, paramName, actual, expected) \
    fla_npu::host_utils::DfxReportInvalidDtype((entity), (paramName), (actual), (expected))

#define OP_LOGE_LIBOPAPI_REPORT_DTYPES_COMPAT(entity, paramNames, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidDtypes((entity), (paramNames), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_VALUE_COMPAT(entity, paramName, actual, expected) \
    fla_npu::host_utils::DfxReportInvalidValue((entity), (paramName), (actual), (expected))

#define OP_LOGE_LIBOPAPI_REPORT_VALUE_WITH_REASON_COMPAT(entity, paramName, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidValueWithReason((entity), (paramName), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_VALUES_COMPAT(entity, paramNames, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidValues((entity), (paramNames), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_VALUES_WITH_REASON_COMPAT(entity, paramNames, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidValuesWithReason((entity), (paramNames), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(entity, paramName, actual, expected) \
    fla_npu::host_utils::DfxReportInvalidListSize((entity), (paramName), (actual), (expected))

#define OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(entity, paramName, actual, expected) \
    fla_npu::host_utils::DfxReportInvalidFormat((entity), (paramName), (actual), (expected))

#define OP_LOGE_LIBOPAPI_REPORT_FORMAT_WITH_REASON_COMPAT(entity, paramName, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidFormatWithReason((entity), (paramName), (actual), (reason))

#endif // FLA_NPU_ASCENDC_COMMON_HOST_UTILS_LOG_H
