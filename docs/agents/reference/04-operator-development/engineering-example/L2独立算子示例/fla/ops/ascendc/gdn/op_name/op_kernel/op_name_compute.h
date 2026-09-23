/**
 * 示例文件：fla/ops/ascendc/gdn/op_name/op_kernel/op_name_compute.h
 *
 * 注意事项：
 *   1. 本文件是"Stage 实现 + AIC/AIV 分工"的落点：AIC 走 Cube 实现类，AIV 走 Vector 实现类，
 *      两者用同一套任务描述（args.tiling）保证顺序一致，固定 EventID 才能按序复用。
 *   2. 架构差异在这里用编译期宏 include 对应实现头：`__CCE_AICORE__ == 310` 走 arch35，
 *      否则走 arch22；不要在实现里写运行期平台判断。
 *   3. 每个 Stage 的生产/消费对都要有对应的 set/wait；`PipeBarrier` 只处理同 pipe 依赖，
 *      跨 pipe 用事件同步（前置知识见仓库 AGENTS.md）。
 *   4. 编译期档位（OUTPUT_MODE）用 `if constexpr` 消除不做的搬出分支；档位与 host 的
 *      output_mask 必须一致，否则会出现"kernel 没写但 L2 以为写了"。
 *   5. 空任务/尾块/varlen 无效区仍要走完整的 ready/free 计数，不能提前 return 破坏计数配平。
 *   6. 不在本文件做参数校验：非法组合在 host 拦截，设备侧不承担兜底。
 */

#ifndef OP_NAME_COMPUTE_H
#define OP_NAME_COMPUTE_H

#include "op_name_struct.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/op_name_cube.h"
#include "arch35/op_name_vec.h"
#else
#include "arch22/op_name_cube.h"
#include "arch22/op_name_vec.h"
#endif

enum class OpNameNormMode : uint8_t { L2 = 1 };
enum class OpNameOutputMode : uint8_t { None = 0, Save = 1 };

template <int D_T>
struct OpNameStorageType;

template <>
struct OpNameStorageType<OP_NAME_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct OpNameStorageType<OP_NAME_TPL_FP16> {
    using type = half;
};

template <>
struct OpNameStorageType<OP_NAME_TPL_FP32> {
    using type = float;
};

// 编译期策略：只承载"必须在该实例中消除的分支"，不放运行期数据。
template <OpNameNormMode NORM, bool USE_STATE, OpNameOutputMode OUT>
struct OpNamePolicy {
    static constexpr OpNameNormMode kNorm = NORM;
    static constexpr bool kUseState = USE_STATE;
    static constexpr OpNameOutputMode kOutput = OUT;
};

namespace OpNameNs {

template <typename XT, typename GT, typename Policy>
__aicore__ inline void RunOpName(const OpNameArgs &args)
{
    if ASCEND_IS_AIC {
        OpNameCube<XT, GT, Policy> cube;
        cube.Init(args);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        OpNameVector<XT, GT, Policy> vec;
        vec.Init(args, &pipe);
        vec.Process();
    }
}

} // namespace OpNameNs

#endif // OP_NAME_COMPUTE_H
