# ============================================================================
# 示例文件：fla/ops/ascendc/demo/example_scan/op_host/CMakeLists.txt
#
# 注意事项：
#   1. 四段结构固定：add_op_to_compiled_list -> 算子来源 -> 编译选项 -> (可选) tests。
#      ABI 敏感文件 *_def.cpp 单独加进 op_host_aclnnExc，不要塞进 add_modules_sources。
#   2. --cce-auto-sync=off 必须保持：同步由代码显式表达，不允许打开编译器自动同步。
#   3. 平台相关编译选项只写在平台分支里（示例：ascend950）。
#   4. 组合入口（形态 B/C）需要的依赖算子在这里用 set(<算子>_depends ...) 声明，
#      否则只编本算子时组合入口会在 tiling 阶段失败。形态 A 可整段删除。
#   5. host 单测目录放在 tests/ 下，用 ENABLE_TEST 控制。
# ============================================================================

add_op_to_compiled_list()

# ---- 形态 B/C 才需要：组合入口依赖的其它算子（形态 A 删除本段） --------------
set(example_scan_depends
    "fla/ops/ascendc/demo/example_scan_base"
    CACHE STRING "Kernel source dependencies for example_scan" FORCE)
foreach(_dep ${example_scan_depends})
    set(_dep_dir "${CMAKE_SOURCE_DIR}/${_dep}")
    if (EXISTS "${_dep_dir}/op_host/CMakeLists.txt" AND NOT "${_dep}" IN_LIST ASCEND_OP_NAME)
        message(STATUS "[example-scan-deps] add dependency op ${_dep}")
        add_subdirectory(${_dep_dir}/op_host ${CMAKE_BINARY_DIR}/example_scan_deps/${_dep})
    endif()
endforeach()
# ---------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        example_scan_def.cpp
    )
endif()

add_modules_sources(OPTYPE example_scan ACLNNTYPE aclnn_exclude)

add_ops_compile_options(
    OP_NAME ExampleScan
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ExampleScan
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()

if(ENABLE_TEST)
    add_subdirectory(tests)
endif()
