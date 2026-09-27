/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_kernel/op_name.cpp
 *
 * 结构参考：fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_finalize/op_kernel/
 *          chunk_gated_delta_rule_bwd_finalize.cpp
 * 目标：入口足够薄，只做"接线"，不出现计算、循环与同步。
 *
 * 注意事项（可读性结构）：
 *   1. 入口只做四件事：dtype traits、tiling 注册与解析、workspace 区域命名、AIC/AIV 分派；
 *      任何 Stage 计算都写进 archXX/<算子>_cube.h 与 archXX/<算子>_vec.h 的类方法里。
 *   2. workspace 区按"语义 + 生命周期"命名，并在同一行注释生产/消费/复用关系
 *      （参考 finalize 的 `kbgDoGWorkspace` 注释 `// S0 kbg -> S12 doG`），不要用 ws0/ws1 这类无名偏移。
 *   3. 平台差异只用编译期宏选择 archXX 目录，入口不写运行期平台判断。
 *   4. 编译期开关（NORM_MODE/USE_STATE/OUTPUT_MODE）作为模板参数传给 Cube/Vector 类，
 *      与参考算子把 USE_QK_L2NORM/USE_BETA_SIGMOID/USE_EXP2 传给 Vector 的写法一致；
 *      类里不要再用运行期 if 判断这些档位。
 *   5. buffer 偏移与同步协议分别属于 archXX/struct.h 与 archXX/<算子>_{vec,cube}.h，入口不聚合它们。
 *   6. **函数不放进类里**：数据放 `archXX/<算子>_{vec,cube}.h` 的 Context 结构体，行为写成
 *      文件作用域 inline 函数（`InitOpNameVector`/`ProcessOpNameVector`/`StageN...`）；
 *      入口只构造结构体并调用两个函数，不出现 Stage 细节。
 *   7. arch35 的向量计算必须是 VF 融合函数（`__simd_vf__ inline` + MicroAPI），arch22 用
 *      同名函数 + 普通向量指令；两平台函数名、事件数组名与 Stage 划分保持一致，便于逐行对照。
 *   8. kernel 侧命名空间取**算子类别目录名**的 PascalCase，archXX 实现、common.h、tiling_key 与入口
 *      共用一个：真实例 `fla/ops/ascendc/gdn/...` → `namespace GDN`；本示例类别占位符是 `ops_classify`
 *      （实际替换为 gdn / kda 一类类别名）→ `namespace OpsClassify`。不要另起 OpNameNs 这类第二命名空间；
 *      host 侧仍按框架用 `ops`/`optiling`，两边靠 TPL 宏 token 取值与 TilingData 字段对齐。
 */

#include "kernel_operator.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/op_name_struct.h"
#include "arch35/op_name_cube.h"
#include "arch35/op_name_vec.h"
#else
#include "arch22/op_name_struct.h"
#include "arch22/op_name_cube.h"
#include "arch22/op_name_vec.h"
#endif
#include "op_name_common.h"

namespace OpsClassify {

// dtype 档位 token -> 实际类型。token 取值来自 op_name_tiling_key.h 的 TPL 常量。
template <int D_T>
struct DTypeTraits;

template <>
struct DTypeTraits<OP_NAME_TPL_BF16> { using type = bfloat16_t; };
template <>
struct DTypeTraits<OP_NAME_TPL_FP16> { using type = half; };
template <>
struct DTypeTraits<OP_NAME_TPL_FP32> { using type = float; };

} // namespace OpsClassify

#ifndef TORCH_MODE
template <int D_T_X, int D_T_G, uint32_t NORM_MODE, bool USE_STATE, uint32_t OUTPUT_MODE>
__global__ __aicore__ void op_name(
    GM_ADDR x, GM_ADDR g, GM_ADDR a_log, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR y, GM_ADDR state, GM_ADDR x_norm,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // 与参考算子一致：入口先打开溢出饱和，再取用户 workspace。
    AscendC::AscendCUtils::SetOverflow(1);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

    REGISTER_TILING_DEFAULT(OpsClassify::OpNameTilingData);
    GET_TILING_DATA_WITH_STRUCT(OpsClassify::OpNameTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    using XType = typename OpsClassify::DTypeTraits<D_T_X>::type;
    const int64_t workspaceRegionBytes = static_cast<int64_t>(AscendC::GetBlockNum()) *
        OpsClassify::WORKSPACE_CHUNK_COUNT * OpsClassify::WORKSPACE_VECTOR_ELEMS * sizeof(XType);

    // 区域语义与生命周期：写清"哪个 Stage 生产、哪个 Stage 消费、何时可复用"。
    // 后续 Stage 复用同一块 workspace 时，顺序必须能从这些注释直读出来，不靠推断。
    GM_ADDR normWorkspace = userWorkspace;                             // S0 norm -> S2 写回
    GM_ADDR stateWorkspace = userWorkspace + workspaceRegionBytes;     // S0/S2 state -> save 档公开导出

    // 两个角色各拿一个 TPipe 指针：buffer 划分留在各自的 Init 函数里，入口不替它们管 L1/L0/UB。
    AscendC::TPipe pipe;
    if ASCEND_IS_AIC {
        // AIC：数据放 Context，行为是文件作用域函数；入口不做任何矩阵/搬运细节。
        OpsClassify::OpNameCubeContext<XType, NORM_MODE, OUTPUT_MODE> cubeCtx;
        OpsClassify::InitOpNameCube(cubeCtx, x, g, normWorkspace, stateWorkspace,
                                    cu_seqlens, chunk_indices, &tilingData, &pipe);
        OpsClassify::ProcessOpNameCube(cubeCtx);
    } else {
        using GType = typename OpsClassify::DTypeTraits<D_T_G>::type;
        OpsClassify::OpNameVectorContext<XType, GType, NORM_MODE, USE_STATE, OUTPUT_MODE> vecCtx;
        OpsClassify::InitOpNameVector(vecCtx, x, g, a_log, initial_state, cu_seqlens,
                                      chunk_indices, y, state, x_norm,
                                      normWorkspace, stateWorkspace, &tilingData, &pipe);
        OpsClassify::ProcessOpNameVector(vecCtx);
    }
}
#endif
