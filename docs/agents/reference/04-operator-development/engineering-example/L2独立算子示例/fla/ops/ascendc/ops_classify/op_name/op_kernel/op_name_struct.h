/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_kernel/op_name_struct.h
 *
 * 结构参考：finalize/op_kernel/chunk_gated_delta_rule_bwd_finalize_struct.h（根目录兼容头）
 *
 * 注意事项：
 *   1. 根目录的 `_struct.h` 只做兼容转发：按编译架构 include 对应 archXX 的实际结构定义，
 *      避免"根目录一份、arch 目录又一份"导致字段漂移。
 *   2. 真正的内容（TilingData、模板参数、平台常量）放 `archXX/<算子>_struct.h`。
 *   3. 需要包含根头文件的代码（例如 host 侧或 UT）不用关心架构分支。
 */

#ifndef OP_NAME_STRUCT_COMPAT_H
#define OP_NAME_STRUCT_COMPAT_H

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/op_name_struct.h"
#else
#include "arch22/op_name_struct.h"
#endif

#endif // OP_NAME_STRUCT_COMPAT_H
