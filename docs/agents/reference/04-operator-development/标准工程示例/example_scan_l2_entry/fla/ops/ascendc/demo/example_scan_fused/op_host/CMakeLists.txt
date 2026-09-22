# ============================================================================
# 示例文件：fla/ops/ascendc/demo/example_scan_fused/op_host/CMakeLists.txt
#
# 注意事项：
#   1. 形态 C（只有 L2）的 CMakeLists 只有三件事：登记算子、声明依赖、登记源文件来源。
#      没有 def.cpp，所以不需要 target_sources(op_host_aclnnExc ...)。
#   2. add_op_to_compiled_list() 必须调用：算子发现靠 "fla/ops/ascendc/**/op_host/CMakeLists.txt"
#      的 GLOB + 本调用把算子名加入编译列表；漏掉等于整个算子目录不进构建。
#   3. depends 必须列全被组合算子（含 kernel 源）：只编本算子时构建系统据此把依赖算子一起编；
#      漏一个会在 tiling/launch 阶段以"找不到实现"失败。
#   4. ACLNNTYPE aclnn_exclude 表示本算子没有自己的 aclnn 原型生成（无 def），组合入口手写在 op_api。
#   5. 本算子不参与 host UT：ENABLE_TEST 打开时构建系统要求 ${OP_DIR}/tests/CMakeLists.txt 存在，
#      若确实不做 UT，要在 README 里写明，避免 CI 静默跳过本算子。
# ============================================================================

add_op_to_compiled_list()

set(example_scan_fused_depends
    "fla/ops/ascendc/demo/example_scan;fla/ops/ascendc/demo/example_scan_tail"
    CACHE STRING "Kernel source dependencies for example_scan_fused" FORCE)

add_modules_sources(OPTYPE example_scan_fused ACLNNTYPE aclnn_exclude)
