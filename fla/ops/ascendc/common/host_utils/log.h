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
#include <utility>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "graph/types.h"
#include "base/err_msg.h"

// 兼容性桥接层：为 aclnn 层与 tiling 层提供统一的结构化 DFX 上报接口。
//
// 只有两档，判据是“typed 宏是否可用”，不再按编译上下文分流：
//   1) CANN 9.1+（op_common/log/log.h 已提供 OP_LOGE_FOR_*）且本 TU 已包含该头
//      （tiling 侧一律包含；aclnn 侧通常不包含）→ 调 typed 宏，人类可读日志与 EZ 码由
//      op_common 一次产出；
//   2) 其余情况（CANN 9.0 无 typed 宏，或 aclnn TU 未包含 op_common）→ 直接
//      REPORT_PREDEFINED_ERR_MSG 上报同族 EZ 码。
//
// 为什么删掉原来的“纯文本降级档”和 FLA_NPU_HOST_TILING 上下文分流：
//   * base/err_msg.h 在 CANN 9.0.0 / 9.1.0 / 9.2.0 都存在并定义 REPORT_PREDEFINED_ERR_MSG，
//     而本文件无条件包含它，所以 `#else` 纯文本档在任何能编过的环境里都不会被编译；
//   * 该纯文本档是 FLA_NPU_HOST_TILING 的唯一使用者，删除后“tiling 用 OP_LOGE(opName,…)、
//     aclnn 用 OP_LOGE(errno,…)”的分流随之消失，也消除了“先包含 op_common 还是先包含本头”
//     的顺序约束（此前有两个 aclnn TU 因此被误判成 tiling）。
//
// 维护约定（新增一类上报时）：
//   1. 先加 typed 分支，再加 EZ 分支，两者 key 名/顺序/值必须完全一致；
//   2. key 名与顺序照抄 op_common/log/log.h 的 OP_LOGE_FOR_*（9.2.0 的
//      OP_LOGE_FOR_INVALID_SHAPE 为 {"param_name","op_name","incorrect_shape","correct_shape"}），
//      不要自行调整；
//   3. 不允许再新增纯文本分支；
//   4. 送进 *_shape 槽位的文本先去外层方括号：EZ0008/EZ0009 模板自带一层方括号，
//      实测报文形如 "incorrect shape [[1,2,256]]"，去括号后为 "[1,2,256]"。

#if !defined(REPORT_PREDEFINED_ERR_MSG)
#error "FLA NPU DFX: REPORT_PREDEFINED_ERR_MSG is required (base/err_msg.h, CANN >= 8.0)."
#endif

namespace fla_npu {
namespace host_utils {

// EZ0008/EZ0009 的 shape 槽位自带方括号，这里去掉调用方传来的外层方括号。
inline std::string DfxStripBrackets(const std::string &text)
{
    if (text.size() >= 2 && text.front() == '[' && text.back() == ']') {
        return text.substr(1, text.size() - 2);
    }
    return text;
}

inline void DfxReportInvalidInput(const char *entity, const char *paramName)
{
    const std::vector<const char *> msgKey = {"param_name", "op_name"};
    const std::vector<const char *> msgvalue = {paramName, entity};
    REPORT_PREDEFINED_ERR_MSG("EZ0004", msgKey, msgvalue);
}

inline void DfxReportInvalidShape(const char *entity, const char *paramName, const std::string &actual,
                                  const std::string &expected)
{
    const std::string strippedActual = DfxStripBrackets(actual);
    const std::string strippedExpected = DfxStripBrackets(expected);
#if defined(OP_LOGE_FOR_INVALID_SHAPE)
    OP_LOGE_FOR_INVALID_SHAPE(entity, paramName, strippedActual.c_str(), strippedExpected.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_shape", "correct_shape"};
    const std::vector<const char *> msgvalue = {paramName, entity, strippedActual.c_str(), strippedExpected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0008", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidShapeWithReason(const char *entity, const char *paramName,
                                            const std::string &actual, const std::string &reason)
{
    const std::string strippedActual = DfxStripBrackets(actual);
#if defined(OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON)
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(entity, paramName, strippedActual.c_str(), reason.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_shape", "reason"};
    const std::vector<const char *> msgvalue = {paramName, entity, strippedActual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0009", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidShapesWithReason(const char *entity, const char *paramNames,
                                             const std::string &actual, const std::string &reason)
{
    const std::string strippedActual = DfxStripBrackets(actual);
#if defined(OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON)
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(entity, paramNames, strippedActual.c_str(), reason.c_str());
#else
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_shapes", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, strippedActual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0010", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidDtype(const char *entity, const char *paramName, const std::string &actual,
                                  const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_DTYPE)
    OP_LOGE_FOR_INVALID_DTYPE(entity, paramName, actual.c_str(), expected.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_dtype", "correct_dtype"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0019", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidDtypes(const char *entity, const char *paramNames, const std::string &actual,
                                   const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON)
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(entity, paramNames, actual.c_str(), reason.c_str());
#else
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_dtypes", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0021", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidValue(const char *entity, const char *paramName, const std::string &actual,
                                  const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_VALUE)
    OP_LOGE_FOR_INVALID_VALUE(entity, paramName, actual.c_str(), expected.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_value", "correct_value"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0024", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidValueWithReason(const char *entity, const char *paramName,
                                            const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_VALUE_WITH_REASON)
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(entity, paramName, actual.c_str(), reason.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_value", "reason"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0026", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidValuesWithReason(const char *entity, const char *paramNames,
                                             const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_VALUES_WITH_REASON)
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(entity, paramNames, actual.c_str(), reason.c_str());
#else
    const std::vector<const char *> msgKey = {"param_names", "op_name", "incorrect_values", "reason"};
    const std::vector<const char *> msgvalue = {paramNames, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0027", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidListSize(const char *entity, const char *paramName, const std::string &actual,
                                     const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_LISTSIZE)
    OP_LOGE_FOR_INVALID_LISTSIZE(entity, paramName, actual.c_str(), expected.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_size", "correct_size"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0025", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidFormat(const char *entity, const char *paramName, const std::string &actual,
                                   const std::string &expected)
{
#if defined(OP_LOGE_FOR_INVALID_FORMAT)
    OP_LOGE_FOR_INVALID_FORMAT(entity, paramName, actual.c_str(), expected.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_format", "correct_format"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), expected.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0017", msgKey, msgvalue);
#endif
}

inline void DfxReportInvalidFormatWithReason(const char *entity, const char *paramName,
                                             const std::string &actual, const std::string &reason)
{
#if defined(OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON)
    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(entity, paramName, actual.c_str(), reason.c_str());
#else
    const std::vector<const char *> msgKey = {"param_name", "op_name", "incorrect_format", "reason"};
    const std::vector<const char *> msgvalue = {paramName, entity, actual.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ0035", msgKey, msgvalue);
#endif
}

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
    fla_npu::host_utils::DfxReportInvalidShapesWithReason((entity), (paramNames), (actual), (reason))

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
    fla_npu::host_utils::DfxReportInvalidValuesWithReason((entity), (paramNames), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_VALUES_WITH_REASON_COMPAT(entity, paramNames, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidValuesWithReason((entity), (paramNames), (actual), (reason))

#define OP_LOGE_LIBOPAPI_REPORT_LISTSIZE_COMPAT(entity, paramName, actual, expected) \
    fla_npu::host_utils::DfxReportInvalidListSize((entity), (paramName), (actual), (expected))

#define OP_LOGE_LIBOPAPI_REPORT_FORMAT_COMPAT(entity, paramName, actual, expected) \
    fla_npu::host_utils::DfxReportInvalidFormat((entity), (paramName), (actual), (expected))

#define OP_LOGE_LIBOPAPI_REPORT_FORMAT_WITH_REASON_COMPAT(entity, paramName, actual, reason) \
    fla_npu::host_utils::DfxReportInvalidFormatWithReason((entity), (paramName), (actual), (reason))

// ---------------------------------------------------------------------------
// 纯宏封装层（无 host_utils 命名空间函数依赖，自包含可复用工具）
// ---------------------------------------------------------------------------

#define FLA_NPU_DFX_TENSOR_SHAPE_STRING(t) \
    FLA_NPU_DFX_SHAPE_STRING((t)->GetViewShape())

#define FLA_NPU_DFX_DTYPE_STRING(dtype) \
    ((dtype) == ge::DT_FLOAT16 ? "FLOAT16" : \
     (dtype) == ge::DT_BF16 ? "BF16" : \
     (dtype) == ge::DT_FLOAT ? "FLOAT" : \
     std::to_string(static_cast<int32_t>(dtype)))

#define FLA_NPU_DFX_SHAPE_STRING(shape) \
    ([&]() { std::ostringstream __s; __s << "["; \
            size_t __n = static_cast<size_t>((shape).GetDimNum()); \
            for (size_t __i = 0; __i < __n; ++__i) { if (__i) __s << ","; __s << (shape).GetDim(__i); } \
            __s << "]"; return __s.str(); }())

#define FLA_NPU_DFX_MAKE_SHAPE_STRING(...) \
    ([&]() { std::ostringstream __s; __s << "["; bool __first = true; \
            for (int64_t __d : std::initializer_list<int64_t>{__VA_ARGS__}) { \
                if (!__first) __s << ","; __s << __d; __first = false; } \
            __s << "]"; return __s.str(); }())

#define FLA_NPU_DFX_MAKE_SHAPE_STRING_IL(dims) \
    ([&]() { std::ostringstream __s; __s << "["; bool __first = true; \
            for (int64_t __d : (dims)) { if (!__first) __s << ","; __s << __d; __first = false; } \
            __s << "]"; return __s.str(); }())

#define FLA_NPU_CHECK_NOT_NULL(entity, ...) \
    ([&]() -> bool { const std::pair<const void *, const char *> __c[] = {__VA_ARGS__}; \
                     for (const auto &__i : __c) \
                         if (__i.first == nullptr) { \
                             OP_LOGE_LIBOPAPI_REPORT_COMPAT((entity), __i.second); \
                             return false; \
                         } \
                     return true; }())


#endif // FLA_NPU_ASCENDC_COMMON_HOST_UTILS_LOG_H
