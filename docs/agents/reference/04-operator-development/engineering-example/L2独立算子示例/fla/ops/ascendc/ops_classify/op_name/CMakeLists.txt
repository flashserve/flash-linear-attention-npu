# ============================================================================
# 示例文件：fla/ops/ascendc/ops_classify/op_name/CMakeLists.txt
#
# 注意事项：
#   1. 本文件只做子目录遍历，不写任何与算子相关的编译逻辑；算子来源在
#      op_host/CMakeLists.txt 里用 add_modules_sources 声明。
#   2. tests 只在 ENABLE_TEST / BENCHMARK 打开时进入，默认构建不带出测试目标。
#   3. 新增算子必须在上一级模块（fla/ops/ascendc/gdn/CMakeLists.txt）被
#      add_subdirectory 到，否则整个算子目录不进构建，且不会报错。
#   4. 不要在这里加 -D 宏来控制平台分支；平台差异走 op_host/arch22|arch35。
# ============================================================================

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
