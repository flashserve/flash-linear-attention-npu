# Chunk KDA Forward Prepare

本目录当前只包含 `op_kernel/pseudocode/` 下的 A5/Arch35 设计伪代码，不是已注册算子。
仓库构建系统、Host Tiling、算子定义、aclnn/Python API 和设备 launch ABI 均未接入，
因此不能从本目录导入、构建或运行 `chunk_kda_fwd_prepare`。

伪代码用于冻结八 Stage 数据流、S=4 causal-prefix 72 KiB score packing、chunk-first 分核、
AIV owner、UB/L1/workspace 生命周期及 ready/free 合同。详细说明见
[`op_kernel/pseudocode/README.md`](op_kernel/pseudocode/README.md)。

所有标为 **PROPOSED** 的设备 API、同步原语、内存 offset、TilingKey 与 ABI 必须在正式实现前
依据目标 CANN 版本重新确认；host C++17 语法检查不等价于 NPU 编译或测试。
