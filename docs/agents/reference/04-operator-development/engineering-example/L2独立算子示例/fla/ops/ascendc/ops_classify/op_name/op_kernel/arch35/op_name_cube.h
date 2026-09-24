/**
 * 示例文件：.../op_kernel/arch35/op_name_cube.h
 *
 * 注意事项：
 *   1. arch35 = A5（ascend950），只在 `__CCE_AICORE__ == 310` 时被 include。
 *   2. 与 arch22 的差异要能一句话说清（tile 更大 / 份数不同 / 同步流水不同 / 新指令）；
 *      说不清差异说明这段代码本应放根目录共用。
 *   3. 更大的 tile 意味着更大的 UB/L1/L0 占用：申请份数与 host 侧 arch35 tiling 常量必须一致，
 *      并显式核对对齐 padding 之后的容量，不能只按元素数乘 dtype 大小。
 *   4. A5 特有的执行形态变化（例如 RegBase/VF 融合、Cube/Vector 合并）写在实现里，
 *      不要通过运行期分支与 arch22 混编。
 *   5. 公开原型、TilingKey、输出契约与 arch22 完全一致，不因为平台不同改变对外行为。
 */

#ifndef OP_NAME_CUBE_ARCH35_H
#define OP_NAME_CUBE_ARCH35_H

template <typename XT, typename GT, typename Policy>
class OpNameCube {
public:
    __aicore__ inline void Init(const OpNameArgs &args) { args_ = &args; }

    __aicore__ inline void Process()
    {
        // A5：tile 取 OP_NAME_ARCH35_TILE_T/D，L1 驻留按 arch35 常量申请。
        // Stage 顺序与 arch22 相同：MTE2 -> Cube -> Fixpipe，事件同步成对。
    }

private:
    const OpNameArgs *args_ = nullptr;
};

#endif // OP_NAME_CUBE_ARCH35_H
