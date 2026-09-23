/**
 * 示例文件：.../op_kernel/arch22/op_name_cube.h
 *
 * 注意事项：
 *   1. arch22 = A2/A3（ascend910b / ascend910_93）。文件名与根目录同名，便于对照检视。
 *   2. 只实现"平台真正不同"的部分：tile 形状、L1/L0 驻留份数、搬运粒度、同步流水；
 *      与 arch35 相同的数学与分核顺序不要各写一份。
 *   3. 每个 Stage 的 SetFlag/WaitFlag 必须成对且 pipe 正确；buffer 复用前要有反向同步或
 *      free 计数（禁止无消费地连续 set 同一 EventID）。
 *   4. L1/L0 的申请份数必须与 host 侧 op_tiling/arch22/op_name_tiling_impl.h 的常量一致；
 *      改一处必须同步另一处。
 *   5. 这里不做运行期平台判断：本文件只在非 A5 编译时被 include。
 */

#ifndef OP_NAME_CUBE_ARCH22_H
#define OP_NAME_CUBE_ARCH22_H

template <typename XT, typename GT, typename Policy>
class OpNameCube {
public:
    __aicore__ inline void Init(const OpNameArgs &args) { args_ = &args; }

    __aicore__ inline void Process()
    {
        // 按 outputMode / USE_STATE 的编译期策略消除不需要的分支：
        // if constexpr (Policy::kOutput == OpNameOutputMode::Save) { ... }
        // MTE2 搬入 -> MTE1 搬 L1 -> Cube 计算 -> Fixpipe 写 L0C；
        // 每次跨 pipe 依赖用成对事件同步，具体参考 chunk_bwd_dv_local 的 arch22 实现。
    }

private:
    const OpNameArgs *args_ = nullptr;
};

#endif // OP_NAME_CUBE_ARCH22_H
