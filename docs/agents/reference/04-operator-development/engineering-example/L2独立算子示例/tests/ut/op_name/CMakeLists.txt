# ============================================================================
# 示例文件：tests/ut/op_name/CMakeLists.txt（仓库根目录的 tests/ 下）
#
# 注意事项：
#   1. 测试放在根 `tests/` 下，算子目录里不放 `op_host/tests/`、`op_kernel/tests/`；
#      本文件由仓库根 tests/ 的构建入口 add_subdirectory 进来（按仓库当前 UT 构建方式接入）。
#   2. 单测必须能在无 NPU 的机器上跑：只验证 tiling 计算与档位推导；需要 NPU 的验证放 tests/atk。
#   3. 被测实现头（`op_host/op_name_tiling_processor.h`）保持 header-only，测试直接 include，
#      避免与算子编译目标重复定义，也避免为测试新增链接目标。
#   4. 用例文件与目标名一致；新增用例在这里加 add_executable/add_test，漏加等于没写。
#   5. 不要 include 算子目录里的 .cpp，也不要链接算子目标：单测只依赖头文件。
# ============================================================================

set(OP_NAME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../fla/ops/ascendc/ops_classify/op_name)

add_executable(
    op_name_tiling_processor_test
    op_name_tiling_processor_test.cpp
)
target_compile_features(op_name_tiling_processor_test PRIVATE cxx_std_17)
target_include_directories(
    op_name_tiling_processor_test
    PRIVATE ${OP_NAME_DIR}/op_host
)

add_test(
    NAME op_name_tiling_processor_test
    COMMAND op_name_tiling_processor_test
)
