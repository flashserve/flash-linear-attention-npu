# ============================================================================
# 示例文件：fla/ops/ascendc/ops_classify/op_name/op_host/tests/CMakeLists.txt
#
# 注意事项：
#   1. 只在本目录编译 host/tiling 单测，目标名与被测文件的 stem 一致。
#   2. 单测必须能在无 NPU 的机器上跑（只验证 tiling 计算与档位推导），需要 NPU 的验证放 tests/atk。
#   3. 被 include 的实现头（如 *_tiling_processor.h）保持 header-only，避免 UT 与算子目标重复定义。
#   4. 新增用例文件时在这里加 add_executable/add_test，漏加等于没写。
# ============================================================================

add_executable(
    op_name_tiling_processor_test
    op_name_tiling_processor_test.cpp
)
target_compile_features(op_name_tiling_processor_test PRIVATE cxx_std_17)
target_include_directories(
    op_name_tiling_processor_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_test(
    NAME op_name_tiling_processor_test
    COMMAND op_name_tiling_processor_test
)
