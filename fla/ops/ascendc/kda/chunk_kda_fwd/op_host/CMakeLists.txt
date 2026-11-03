# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
# 融合实现（aclnnChunkKdaFwd）需要 chunk_gated_delta_rule_fwd_h / kda_gate_cumsum
# 的 kernel 源码；V2 组合入口（aclnnChunkKdaFwdV2 -> KdaFwdV2）在
# chunk_kda_fwd_v2.cpp 里直接 include 并调用这些独立算子的 op_api，
# 因此它们也必须随 chunk_kda_fwd 一起编译，否则单独
# `FLA_NPU_OPS=chunk_kda_fwd` 会产出 libcust_opapi.so 缺符号的坏包。
set(chunk_kda_fwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h;fla/ops/ascendc/kda/kda_gate_cumsum;fla/ops/ascendc/kda/chunk_kda_fwd_prepare;fla/ops/ascendc/kda/chunk_kda_fwd_finalize;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h"
    CACHE STRING "Kernel source dependencies for chunk_kda_fwd" FORCE)

# 上面只解决了"依赖算子的源码参与编译"（libcust_opapi.so 的符号）。算子的编译产物
# 与算子配置来自本次构建的 ops-info（opbuild 的 aic-<soc>-ops-info.ini），即只覆盖
# "参与构建的算子"。因此 `--ops=chunk_kda_fwd` 的过滤构建产出的包里没有
# ChunkKdaFwdPrepare/ChunkFwdH/ChunkKdaFwdFinalize 等依赖算子的配置，V2 组合入口
# 运行期报 Config_Error(EZ1013) / aclnnStatus=561103。
#
# 这里在过滤构建下把依赖算子的 op_host 一并加入本次构建（等价于它们被 --ops 选中），
# 使它们照常参与 opbuild 与 kernel 编译打包；全量构建或依赖已被 --ops 选中时跳过，
# 避免重复 add_subdirectory。
set(_kda_fwd_build_all_ops FALSE)
if (NOT DEFINED ASCEND_OP_NAME OR "${ASCEND_OP_NAME}" STREQUAL ""
        OR "${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
    set(_kda_fwd_build_all_ops TRUE)
endif()

foreach(_kda_dep ${chunk_kda_fwd_depends})
    if (ENABLE_EXPERIMENTAL)
        set(_kda_dep_path "experimental/${_kda_dep}")
    else()
        set(_kda_dep_path "${_kda_dep}")
    endif()
    set(_kda_dep_dir "${CMAKE_SOURCE_DIR}/${_kda_dep_path}")
    if (NOT EXISTS "${_kda_dep_dir}/op_host/CMakeLists.txt")
        continue()
    endif()
    get_filename_component(_kda_dep_name "${_kda_dep}" NAME)
    if (NOT BUILD_OPEN_PROJECT)
        if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_kda_dep_name})
            continue()
        endif()
    endif()
    if (_kda_fwd_build_all_ops)
        continue()
    endif()
    if (${_kda_dep_name} IN_LIST ASCEND_OP_NAME)
        continue()
    endif()
    message(STATUS "[kda-deps] add dependency op ${_kda_dep_name} to this build")
    add_subdirectory(
        ${_kda_dep_dir}/op_host
        ${CMAKE_BINARY_DIR}/chunk_kda_fwd_deps/${_kda_dep_name}
    )
endforeach()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaFwd
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()
