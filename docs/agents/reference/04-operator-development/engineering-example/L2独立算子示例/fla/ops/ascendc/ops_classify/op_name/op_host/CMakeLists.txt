# ============================================================================
# 示例文件：fla/ops/ascendc/ops_classify/op_name/op_host/CMakeLists.txt
#
# 注意事项：
#   1. 四段结构固定：add_op_to_compiled_list -> 算子来源 -> 编译选项 -> (可选) tests。
#      ABI 敏感文件 *_def.cpp 单独加进 op_host_aclnnExc，不要塞进 add_modules_sources。
#   2. --cce-auto-sync=off 必须保持：同步由代码显式表达，不允许打开编译器自动同步。
#   3. 平台相关编译选项只写在平台分支里（示例：ascend950）。
#   4. 组合入口（形态 B/C）需要的依赖算子用 set(<算子>_depends ...) 声明，并按下面的 foreach
#      展开成依赖闭包：只编本算子（FLA_NPU_OPS / --ops 过滤构建）时，依赖算子的 op_host 必须一起
#      进本次构建；打包侧由 cmake/custom_build.cmake 按同一份 ${<算子>_depends} 安装依赖算子产物。
#      漏掉任一侧的症状是"构建安装都成功、跑到组合入口才报 aclnnStatus=561103 +
#      Config_Error(EZ1013): ... the JSON configuration file of operator ... cannot be found"，
#      属于依赖配置缺失，不要误判为算子数值或 kernel 问题。形态 A（无组合入口）可整段删除。
#   5. 全量构建通过不等于过滤构建自洽：改动组合入口后要用"只列主算子"的过滤构建 + 一条组合路径
#      端到端用例验证（全量构建不会暴露该缺陷）。
#   6. 测试不放算子目录：host / tiling 单测在仓库根 tests/ut/<算子>/（见工程结构规范 §3.6），
#      ATK 用例在 tests/atk/<算子>/；本文件不需要 add_subdirectory(tests)。
# ============================================================================

add_op_to_compiled_list()

# ---- 形态 B/C 才需要：组合入口依赖的其它算子（形态 A 删除本段，含闭包展开） ------
set(op_name_depends
    "fla/ops/ascendc/ops_classify/op_name_base"
    CACHE STRING "Kernel source dependencies for op_name" FORCE)

set(_op_name_build_all_ops FALSE)
if (NOT DEFINED ASCEND_OP_NAME OR "${ASCEND_OP_NAME}" STREQUAL ""
        OR "${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
    set(_op_name_build_all_ops TRUE)
endif()

foreach(_op_name_dep ${op_name_depends})
    if (ENABLE_EXPERIMENTAL)
        set(_dep_path "experimental/${_op_name_dep}")
    else()
        set(_dep_path "${_op_name_dep}")
    endif()
    set(_dep_dir "${CMAKE_SOURCE_DIR}/${_dep_path}")
    if (NOT EXISTS "${_dep_dir}/op_host/CMakeLists.txt")
        continue()
    endif()
    get_filename_component(_dep_name "${_op_name_dep}" NAME)
    if (NOT BUILD_OPEN_PROJECT)
        if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_dep_name})
            continue()
        endif()
    endif()
    if (_op_name_build_all_ops)
        continue()
    endif()
    if (${_dep_name} IN_LIST ASCEND_OP_NAME)
        continue()
    endif()
    message(STATUS "[op-name-deps] add dependency op ${_dep_name} to this build")
    add_subdirectory(${_dep_dir}/op_host ${CMAKE_BINARY_DIR}/op_name_deps/${_dep_name})
endforeach()
# ---------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        op_name_def.cpp
    )
endif()

add_modules_sources(OPTYPE op_name ACLNNTYPE aclnn_exclude)

add_ops_compile_options(
    OP_NAME OpName
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME OpName
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()

# 注意：测试不放算子目录。host / tiling 单测在仓库根 tests/ut/<算子>/，ATK 用例在 tests/atk/<算子>/；
# 算子目录下不建 tests/，也不要在这里 add_subdirectory(tests)。
