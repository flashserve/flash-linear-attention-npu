# ============================================================================
# 示例文件：fla/ops/ascendc/ops_classify/op_name/CMakeLists.txt
#
# 注意事项：
#   1. 本文件只做子目录遍历，不写任何与算子相关的编译逻辑；算子来源在
#      op_host/CMakeLists.txt 里用 add_modules_sources 声明。
#   2. 算子目录下不放测试：所有测试用例集中在仓库根 tests/ 下（tests/atk/<算子>/、tests/ut/<算子>/），
#      因此这里既没有 tests 子目录，也不需要按 ENABLE_TEST 过滤子目录。
#   3. 新增算子必须在上一级模块（fla/ops/ascendc/ops_classify/CMakeLists.txt）被
#      add_subdirectory 到，否则整个算子目录不进构建，且不会报错。
#   4. 不要在这里加 -D 宏来控制平台分支；平台差异走 op_host/arch22|arch35。
# ============================================================================

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
