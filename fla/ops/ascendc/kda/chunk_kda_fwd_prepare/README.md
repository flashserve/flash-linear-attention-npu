# Chunk KDA Forward Prepare

本目录当前只包含 `op_kernel/pseudocode/` 下的 A2/A3 Arch22 与 A5 Arch35 设计伪代码，
不是已注册算子。
仓库构建系统、Host Tiling、算子定义、aclnn/Python API 和设备 launch ABI 均未接入，
因此不能从本目录导入、构建或运行 `chunk_kda_fwd_prepare`。
`chunk_kda_fwd_finalize` 的设计与实现也不在本目录范围内。

伪代码用于冻结共享的八 Stage 数据流、S=4 causal-prefix 72 KiB score packing、chunk-first 分核，
`HK` Q/K head 到 `HV` value/gate head 的 cohort 映射与一次归一化 cache，独立的 q/k、v/u 与
raw-g dtype 语义，以及分架构的 AIV owner、UB/L1/workspace 生命周期和 ready/free 合同。Arch22 设计分支及其
host 合同已经实现完整；L0A/L0B operand 使用独立 bank/generation 账本，不能借用 L1 或 L0C
generation 表示复用安全。GM relay、mode-0x2 collective 的具体设备 API 仍是 **PROPOSED**，
不能据此声称 A2/A3 已具备可构建、可调用的生产支持。详细说明见
[`op_kernel/pseudocode/README.md`](op_kernel/pseudocode/README.md)。

所有标为 **PROPOSED** 的设备 API、同步原语、内存 offset、TilingKey 与 ABI 必须在正式实现前
依据目标 CANN 版本重新确认；host C++17 语法检查不等价于 NPU 编译或测试。
