add_op_to_compiled_list()
set(chunk_gated_delta_rule_fwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o"
    CACHE STRING "Kernel source dependencies for chunk_gated_delta_rule_fwd" FORCE)

# 上面只解决了"依赖算子的 kernel 源码参与编译"（libcust_opapi.so 的符号）。
# 算子的编译产物与算子配置来自本次构建的 ops-info（opbuild 的
# aic-<soc>-ops-info.ini），即只覆盖"参与构建的算子"。因此
# `--ops=chunk_gated_delta_rule_fwd` 的过滤构建产出的包里没有
# ChunkGatedDeltaRuleFwdPrepare/ChunkFwdH/ChunkFwdO 等依赖算子的 kernel 二进制与
# kernel/config/<soc>/<op>.json，运行期报 JSON configuration file of operator
# aclnnChunkGatedDeltaRuleFwd_0_ChunkGatedDeltaRuleFwdPrepare cannot be found。
#
# 这里在过滤构建下把依赖算子的 op_host 一并加入本次构建（等价于它们被 --ops 选中），
# 使它们照常参与 opbuild 与 kernel 编译打包；全量构建或依赖已被 --ops 选中时跳过，
# 避免重复 add_subdirectory。改动只在算子目录内，不修改公共构建脚本。
set(_gdn_fwd_build_all_ops FALSE)
if (NOT DEFINED ASCEND_OP_NAME OR "${ASCEND_OP_NAME}" STREQUAL ""
        OR "${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
    set(_gdn_fwd_build_all_ops TRUE)
endif()

foreach(_gdn_fwd_dep ${chunk_gated_delta_rule_fwd_depends})
    if (ENABLE_EXPERIMENTAL)
        set(_gdn_fwd_dep_path "experimental/${_gdn_fwd_dep}")
    else()
        set(_gdn_fwd_dep_path "${_gdn_fwd_dep}")
    endif()
    set(_gdn_fwd_dep_dir "${CMAKE_SOURCE_DIR}/${_gdn_fwd_dep_path}")
    if (NOT EXISTS "${_gdn_fwd_dep_dir}/op_host/CMakeLists.txt")
        continue()
    endif()
    get_filename_component(_gdn_fwd_dep_name "${_gdn_fwd_dep}" NAME)
    if (NOT BUILD_OPEN_PROJECT)
        if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_gdn_fwd_dep_name})
            continue()
        endif()
    endif()
    if (_gdn_fwd_build_all_ops)
        continue()
    endif()
    if (${_gdn_fwd_dep_name} IN_LIST ASCEND_OP_NAME)
        continue()
    endif()
    message(STATUS "[gdn-fwd-deps] add dependency op ${_gdn_fwd_dep_name} to this build")
    add_subdirectory(
        ${_gdn_fwd_dep_dir}/op_host
        ${CMAKE_BINARY_DIR}/chunk_gated_delta_rule_fwd_deps/${_gdn_fwd_dep_name}
    )
endforeach()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/coefficient_generation/gated_delta_rule_solve_tri
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/operators/recompute_w_u_fwd/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_kkt_solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_scaled_dot_kkt/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../solve_tri/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/chunk_fwd_o/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/internal/arch22/operators/recompute_w_u_fwd/op_kernel
)
