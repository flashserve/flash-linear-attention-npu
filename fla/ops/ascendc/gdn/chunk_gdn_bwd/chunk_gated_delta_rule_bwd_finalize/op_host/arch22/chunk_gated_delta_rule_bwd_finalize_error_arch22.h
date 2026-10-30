#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ERROR_ARCH22_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_ERROR_ARCH22_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "base/err_msg.h"

namespace GDN {
namespace FinalizeHostArch22 {

enum class ParameterKind {
    VALUE,
    SHAPE,
    DTYPE,
    FORMAT,
};

// 复用目标 opbase/log.h 已有的 EZ0001/2/6/7 模板和字段，不依赖新版 FOR_INVALID_* 宏。
// 不包含 op_common/log/log.h，避免它与 L2 的 opdev/op_log.h 宏签名冲突。
inline bool ReportParameterError(const std::string &parameter, const std::string &actual,
                                 const std::string &expected, ParameterKind kind, size_t inputIndex)
{
    constexpr const char *OP_NAME = "ChunkGatedDeltaRuleBwdFinalize";
    using Reporter = int32_t (*)(const char *, const std::vector<const char *> &, const std::vector<const char *> &);
    const auto report = static_cast<Reporter>(error_message::ReportPredefinedErrMsg);
    if (report == nullptr) {
        return false;
    }
    const std::vector<const char *> values = {OP_NAME, parameter.c_str(), actual.c_str(), expected.c_str()};
    if (kind == ParameterKind::SHAPE) {
        const std::string index = std::to_string(inputIndex);
        const std::string actualShape = parameter + "=" + actual;
        const std::vector<const char *> keys = {"op_name", "index", "incorrect_shape", "correct_shape"};
        const std::vector<const char *> shapeValues = {OP_NAME, index.c_str(), actualShape.c_str(), expected.c_str()};
        return report("EZ0001", keys, shapeValues) == 0;
    }
    if (kind == ParameterKind::DTYPE) {
        const std::vector<const char *> keys = {"op_name", "param_name", "data_dtype", "expected_dtype_list"};
        return report("EZ0007", keys, values) == 0;
    } else if (kind == ParameterKind::FORMAT) {
        const std::vector<const char *> keys = {"op_name", "param_name", "data_format", "expected_format_list"};
        return report("EZ0006", keys, values) == 0;
    } else {
        // attr_name 表示本次命名值/关联约束，包括 host metadata 的元素和长度约束。
        const std::vector<const char *> keys = {"op_name", "attr_name", "incorrect_val", "correct_val"};
        return report("EZ0002", keys, values) == 0;
    }
}

} // namespace FinalizeHostArch22
} // namespace GDN

#endif
