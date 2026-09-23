# ============================================================================
# 示例文件：fla/ops/ascendc/demo/example_scan_fused/op_host/CMakeLists.txt
#
# 注意事项：
#   1. 形态 C（只有 L2）的 CMakeLists 只有三件事：登记算子、声明依赖、登记源文件来源。
#      没有 def.cpp，所以不需要 target_sources(op_host_aclnnExc ...)。
#   2. add_op_to_compiled_list() 必须调用：算子发现靠 "fla/ops/ascendc/**/op_host/CMakeLists.txt"
#      的 GLOB + 本调用把算子名加入编译列表；漏掉等于整个算子目录不进构建。
#   3. <算子>_depends 必须列全被组合算子，并且**必须展开成依赖闭包**（见下面 foreach）：
#      - 构建侧：过滤构建（FLA_NPU_OPS / --ops 只列主算子）时，靠本循环把依赖算子的 op_host
#        （def/config/tiling）加进本次构建；
#      - 打包侧：cmake/custom_build.cmake 会按 ${<算子>_depends} 把依赖算子的 op_kernel 产物
#        安装到 impl 目录，所以 depends 里的路径必须与实际目录一致。
#      漏任一侧的表现都是"编译、安装都成功，但跑到组合入口才失败"：aclnnStatus=561103 且
#      Config_Error(EZ1013): ... the JSON configuration file of operator ... cannot be found /
#      AclOpKernelInit failed。**这是依赖配置缺失，不是算子数值或 kernel 问题**，
#      排查时不要先去查精度（真实案例：chunk_kda_fwd 的 V2 组合入口，缺 chunk_fwd_h 的配置）。
#   4. 全量构建通过不能证明过滤构建自洽：新增或修改组合入口后，至少用一次"只列主算子"的
#      过滤构建 + 一条走组合路径的端到端用例验证（全量构建不受该缺陷影响）。
#   5. ACLNNTYPE aclnn_exclude 表示本算子没有自己的 aclnn 原型生成（无 def），组合入口手写在 op_api。
#   6. 本算子不参与 host UT：ENABLE_TEST 打开时构建系统要求 ${OP_DIR}/tests/CMakeLists.txt 存在，
#      若确实不做 UT，要在 README 里写明，避免 CI 静默跳过本算子。
# ============================================================================

add_op_to_compiled_list()

set(example_scan_fused_depends
    "fla/ops/ascendc/demo/example_scan;fla/ops/ascendc/demo/example_scan_tail"
    CACHE STRING "Kernel source dependencies for example_scan_fused" FORCE)

# 依赖闭包展开：只编主算子时，把不在本次 ASCEND_OP_NAME 里的依赖算子的 op_host 一起加进来。
# 全量构建（ASCEND_OP_NAME 为空/all）时本段自动跳过，不做重复 add_subdirectory。
set(_example_scan_fused_build_all_ops FALSE)
if (NOT DEFINED ASCEND_OP_NAME OR "${ASCEND_OP_NAME}" STREQUAL ""
        OR "${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
    set(_example_scan_fused_build_all_ops TRUE)
endif()

foreach(_example_scan_fused_dep ${example_scan_fused_depends})
    if (ENABLE_EXPERIMENTAL)
        set(_dep_path "experimental/${_example_scan_fused_dep}")
    else()
        set(_dep_path "${_example_scan_fused_dep}")
    endif()
    set(_dep_dir "${CMAKE_SOURCE_DIR}/${_dep_path}")
    if (NOT EXISTS "${_dep_dir}/op_host/CMakeLists.txt")
        continue()
    endif()
    get_filename_component(_dep_name "${_example_scan_fused_dep}" NAME)
    if (NOT BUILD_OPEN_PROJECT)
        if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_dep_name})
            continue()
        endif()
    endif()
    if (_example_scan_fused_build_all_ops)
        continue()
    endif()
    if (${_dep_name} IN_LIST ASCEND_OP_NAME)
        continue()
    endif()
    message(STATUS "[example-scan-fused-deps] add dependency op ${_dep_name} to this build")
    add_subdirectory(
        ${_dep_dir}/op_host
        ${CMAKE_BINARY_DIR}/example_scan_fused_deps/${_dep_name}
    )
endforeach()

add_modules_sources(OPTYPE example_scan_fused ACLNNTYPE aclnn_exclude)
