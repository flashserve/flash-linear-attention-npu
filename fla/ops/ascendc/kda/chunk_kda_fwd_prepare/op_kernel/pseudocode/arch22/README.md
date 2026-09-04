# Arch22 状态

本伪代码包没有 `chunk_kda_fwd_prepare` 的 Arch22 实现。当前资源账本和 MIX AIC/AIV 执行合同
只面向 A5/Arch35，尚未在 A2/A3 上验证。

本目录的 `_vec.h` 与 `_cube.h` 只提供与 Arch35 文件名对称的**删除声明**，用于让代码检视者
明确看到该架构不可实例化；它们不包含算法、不会被入口 include，也不进入 CMake 或运行时分派。
因此不能据此声称 Arch22 编译或功能支持。

后续 Arch22 方案必须独立冻结并验证 UB/L1 容量、Cube operand/accumulate 能力、VF/Matmul API、
GM relay、核内 event、核间 ready/free、TilingKey、数值 cast 点和测试矩阵。在这些条件完成前，
Host 必须在 launch 前拒绝该平台，不能静默切换算法或复用 Arch35 实现。
