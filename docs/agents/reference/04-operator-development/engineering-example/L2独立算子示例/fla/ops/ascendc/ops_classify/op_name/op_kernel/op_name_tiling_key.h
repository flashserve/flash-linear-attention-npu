/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_kernel/op_name_tiling_key.h
 *
 * 注意事项：
 *   1. 本文件是**必需件**，不是可选件：每个算子的模板参数与实例都写在这里，host 侧用
 *      GET_TPL_TILING_KEY 生成 key，kernel 入口按模板参数实例化；不允许用"单实例 + 运行期分支"代替。
 *   2. template_argument.h 与 `ASCENDC_TPL_*` 只在"算子正常编译环境"存在，因此用
 *      `FLA_TORCH_EXTENSION_INLINE_BUILD` 这一**按场景命名**的编译路径开关来跳过它们：
 *        该开关表示"算子源码被 torch 扩展以源码形式内联编译"（如 fast kernel launch 例子），
 *        此时 CANN host 侧 tiling 框架头与 torch_npu 的 ge_error_codes.h 冲突、且入口由外部提供。
 *      不要用 `TORCH_MODE` 这类泛化负向宏：名字会让人误以为"torch_custom 构建会定义它"
 *      （实际仓内只有 `examples/fast_kernel_launch_example/CMakeLists.txt` 定义），而且
 *      `#ifndef` 的默认分支会依赖一个本仓常规构建里根本不出现的符号。
 *      若将来"跳过框架头"与"入口由外部提供"需要独立控制，再按用途各拆一个宏，不要继续复用泛化名。
 *   3. 模板参数顺序 = ASCENDC_TPL_ARGS_DECL 声明顺序 = GET_TPL_TILING_KEY 实参顺序，
 *      任何一处不一致都会选到错误实例（编译通过、结果错）。
 *   4. 实例枚举用宏逐层展开（参照本文件的 _SEL_* 分层），不要手抄笛卡尔积：漏实例会落到
 *      "没有对应模板"的编译错误，抄错组合则会静默选错。
 *   5. 只给"必须在该实例中消除的分支"开口（dtype、模式、档位）；shape 相关的量走 TilingData，
 *      不进模板参数，否则实例数爆炸。
 *   6. 这里不写平台：A2/A3/A5 用同一份实例集合，平台在 kernel 入口按 __CCE_AICORE__ 选择实现。
 *   7. TilingData 结构属于 op_kernel/<算子>_struct.h，不要与本文件混放。
 *   8. 归属：本示例按本仓新规范把 TPL 单独放 `_tiling_key.h`；结构样板
 *      `chunk_gated_delta_rule_bwd_finalize` 把它放在 `archXX/<算子>_struct.h`。两者等价，
 *      但同一算子只能有一份：`archXX/<算子>_struct.h` 通过 `../op_name_tiling_key.h` 引用本文件。
 */

#ifndef OP_NAME_TILING_KEY_H
#define OP_NAME_TILING_KEY_H

#if !defined(FLA_TORCH_EXTENSION_INLINE_BUILD)
#include "ascendc/host_api/tiling/template_argument.h"
#endif // FLA_TORCH_EXTENSION_INLINE_BUILD

// 命名空间规则：kernel 侧统一用**算子类别目录名**的 PascalCase（与 archXX 实现、common.h 一致）。
//   真实例：fla/ops/ascendc/gdn/... → namespace GDN
//   本示例：类别占位符是 ops_classify（实际替换为 gdn / kda 一类类别名）→ namespace OpsClassify
// host 侧按框架保留命名空间（`ops` 给 *_def.cpp、`optiling` 给 tiling），不重复声明类别命名空间：
// host tiling 通过 `#include "../op_kernel/<算子>_tiling_key.h"` 复用同一份模板参数，靠这里的
// **宏 token 取值**与 TilingData 字段两边对齐（host 用 static_assert 钉住取值）。
// 不要在 kernel 侧另起第二个"算子私有"命名空间（如 OpNameNs）。
// 注意：下面的档位 token 是宏，host 与 kernel 都直接写宏名（`OP_NAME_TPL_BF16`），
// 不写命名空间限定；命名空间用于结构体、文件作用域函数与 `ASCENDC_TPL_*` 声明块。
namespace OpsClassify {

// dtype 档位 token（主机侧 DtypeToTemplateToken 必须返回同一组取值）。
#define OP_NAME_TPL_BF16 10
#define OP_NAME_TPL_FP16 20
#define OP_NAME_TPL_FP32 30

// 模式档位 token：数值必须与 op_host/<算子>_output_mask.h 的档位枚举一致，
// 命名带 TPL_ 前缀，避免与 host 侧的同名枚举/宏在同一编译单元里冲突
// （host 的 tiling.cpp 同时 include 两侧头文件，用 static_assert 钉住相等）。
#define OP_NAME_TPL_NORM_L2 1
#define OP_NAME_TPL_OUTPUT_NONE 0
#define OP_NAME_TPL_OUTPUT_SAVE 1

#if !defined(FLA_TORCH_EXTENSION_INLINE_BUILD)
ASCENDC_TPL_ARGS_DECL(OpName,
    ASCENDC_TPL_DTYPE_DECL(D_T_X, OP_NAME_TPL_BF16, OP_NAME_TPL_FP16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, OP_NAME_TPL_BF16, OP_NAME_TPL_FP32),
    ASCENDC_TPL_UINT_DECL(NORM_MODE, 1, ASCENDC_TPL_UI_LIST, OP_NAME_TPL_NORM_L2),
    ASCENDC_TPL_BOOL_DECL(USE_STATE, 0, 1),
    ASCENDC_TPL_UINT_DECL(OUTPUT_MODE, 1, ASCENDC_TPL_UI_LIST,
                          OP_NAME_TPL_OUTPUT_NONE, OP_NAME_TPL_OUTPUT_SAVE),
);

#define OP_NAME_SEL_ONE(D_T_X_VALUE, D_T_G_VALUE, NORM_VALUE, STATE_VALUE, OUT_VALUE) \
    ASCENDC_TPL_ARGS_SEL(                                                                  \
        ASCENDC_TPL_DTYPE_SEL(D_T_X, D_T_X_VALUE),                                         \
        ASCENDC_TPL_DTYPE_SEL(D_T_G, D_T_G_VALUE),                                         \
        ASCENDC_TPL_UINT_SEL(NORM_MODE, ASCENDC_TPL_UI_LIST, NORM_VALUE),                  \
        ASCENDC_TPL_BOOL_SEL(USE_STATE, STATE_VALUE),                                      \
        ASCENDC_TPL_UINT_SEL(OUTPUT_MODE, ASCENDC_TPL_UI_LIST, OUT_VALUE))

#define OP_NAME_SEL_MODE(D_T_X_VALUE, D_T_G_VALUE, NORM_VALUE, STATE_VALUE) \
    OP_NAME_SEL_ONE(D_T_X_VALUE, D_T_G_VALUE, NORM_VALUE, STATE_VALUE,      \
                         OP_NAME_TPL_OUTPUT_NONE),                          \
    OP_NAME_SEL_ONE(D_T_X_VALUE, D_T_G_VALUE, NORM_VALUE, STATE_VALUE,      \
                         OP_NAME_TPL_OUTPUT_SAVE)

#define OP_NAME_SEL_STATE(D_T_X_VALUE, D_T_G_VALUE, NORM_VALUE) \
    OP_NAME_SEL_MODE(D_T_X_VALUE, D_T_G_VALUE, NORM_VALUE, 0),  \
    OP_NAME_SEL_MODE(D_T_X_VALUE, D_T_G_VALUE, NORM_VALUE, 1)

#define OP_NAME_SEL_G_DTYPE(D_T_X_VALUE)                             \
    OP_NAME_SEL_STATE(D_T_X_VALUE, OP_NAME_TPL_BF16,            \
                           OP_NAME_TPL_NORM_L2),                     \
    OP_NAME_SEL_STATE(D_T_X_VALUE, OP_NAME_TPL_FP32,            \
                           OP_NAME_TPL_NORM_L2)

ASCENDC_TPL_SEL(
    OP_NAME_SEL_G_DTYPE(OP_NAME_TPL_BF16),
    OP_NAME_SEL_G_DTYPE(OP_NAME_TPL_FP16));

#undef OP_NAME_SEL_G_DTYPE
#undef OP_NAME_SEL_STATE
#undef OP_NAME_SEL_MODE
#undef OP_NAME_SEL_ONE
#endif // FLA_TORCH_EXTENSION_INLINE_BUILD

} // namespace OpsClassify

#endif // OP_NAME_TILING_KEY_H
