/**
 * 示例文件：fla/ops/ascendc/ops_classify/op_name/op_kernel/op_name.cpp
 *
 * 注意事项：
 *   1. 一个算子只有这一个 kernel 入口，函数名与算子名一致；多 Stage 用私有属性/tiling 字段分派，
 *      不要新增第二个入口或第二个 L0 原型。
 *   2. 入口必须是 `template <...> __global__ __aicore__ void <算子>(GM_ADDR ..., GM_ADDR workspace, GM_ADDR tiling)`，
 *      模板参数顺序与 <算子>_tiling_key.h 的声明一致，形参顺序与 def 的 Input/Output 顺序一致。
 *   3. 架构选择只看编译期宏 `__CCE_AICORE__ == 310`（A5），写在 include 层；不要用运行期判断。
 *   4. workspace 必须通过 AscendC::GetUserWorkspace(workspace) 取，并检查非空后再使用。
 *   5. TilingData 用 REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT 解析，字段逐个赋给 args，
 *      不要直接按裸偏移读 tiling。
 *   6. 入口只做：解析 tiling、组装 args、构造 Policy、调用实现；不放具体计算与同步。
 *   7. `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 表示 AIC:AIV=1:2，必须与 host 的
 *      SetScheduleMode / SetBlockDim 以及设计与资源规划一致。
 */

#include "kernel_operator.h"
#include "op_name_tiling_key.h"
#include "op_name_struct.h"
#include "op_name_compute.h"

template <int D_T_X, int D_T_G, uint32_t NORM_MODE, bool USE_STATE, uint32_t OUTPUT_MODE>
__global__ __aicore__ void op_name(
    GM_ADDR x, GM_ADDR g, GM_ADDR a_log, GM_ADDR initial_state, GM_ADDR cu_seqlens,
    GM_ADDR chunk_indices, GM_ADDR y, GM_ADDR state, GM_ADDR x_norm, GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(OpNameTilingData);
    GET_TILING_DATA_WITH_STRUCT(OpNameTilingData, tilingData, tiling);

    OpNameArgs args{};
    args.x = x;
    args.g = g;
    args.aLog = a_log;
    args.initialState = initial_state;
    args.cuSeqlens = cu_seqlens;
    args.chunkIndices = chunk_indices;
    args.y = y;
    args.state = state;
    args.xNorm = x_norm;
    args.userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (args.userWorkspace == nullptr) {
        return;
    }
    args.tiling.batch = tilingData.batch;
    args.tiling.seqLen = tilingData.seqLen;
    args.tiling.headNum = tilingData.headNum;
    args.tiling.dim = tilingData.dim;
    args.tiling.chunkSize = tilingData.chunkSize;
    args.tiling.chunksPerSequence = tilingData.chunksPerSequence;
    args.tiling.usedCoreNum = tilingData.usedCoreNum;
    args.tiling.headsPerCore = tilingData.headsPerCore;
    args.tiling.outputMode = tilingData.outputMode;
    args.tiling.scale = tilingData.scale;
    args.tiling.epsilon = tilingData.epsilon;
    args.tiling.isVarLen = tilingData.isVarLen;
    args.tiling.hasInitialState = tilingData.hasInitialState;

    using Policy = OpNamePolicy<
        static_cast<OpNameNormMode>(NORM_MODE), USE_STATE,
        static_cast<OpNameOutputMode>(OUTPUT_MODE)>;
    using XT = typename OpNameStorageType<D_T_X>::type;
    using GT = typename OpNameStorageType<D_T_G>::type;
    OpNameNs::RunOpName<XT, GT, Policy>(args);
}
