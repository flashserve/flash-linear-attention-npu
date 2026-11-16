add_op_to_compiled_list()
set(_gdn_bwd_enable_arch22 0)
foreach(_gdn_bwd_soc IN LISTS ASCEND_COMPUTE_UNIT)
    string(TOLOWER "${_gdn_bwd_soc}" _gdn_bwd_soc_lower)
    if (_gdn_bwd_soc_lower STREQUAL "ascend910b" OR _gdn_bwd_soc_lower STREQUAL "ascend910_93")
        set(_gdn_bwd_enable_arch22 1)
    endif()
endforeach()

# 先保留 A5 基线依赖，再按构建的 SoC 加入 A2/A3 私有前序算子。
# 混合 SoC 构建共用 Host 对象，包含 A2/A3 时保留其依赖和运行期分派。
set(_gdn_bwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gdn_bwd_intra;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h;fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu;fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_finalize")
if (_gdn_bwd_enable_arch22)
    list(APPEND _gdn_bwd_depends
        "fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd"
        "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local")
endif()
set(chunk_gated_delta_rule_bwd_depends
    "${_gdn_bwd_depends}"
    CACHE STRING "Kernel source dependencies for chunk_gated_delta_rule_bwd" FORCE)

# 上面只解决了"依赖算子的 kernel 源码参与编译"（libcust_opapi.so 的符号）。
# 算子的编译产物与算子配置来自本次构建的 ops-info（opbuild 的
# aic-<soc>-ops-info.ini），即只覆盖"参与构建的算子"。因此
# `--ops=chunk_gated_delta_rule_bwd` 的过滤构建产出的包里没有
# ChunkGatedDeltaRuleBwdIntra/ChunkFwdH/ChunkGatedDeltaRuleBwdDhu 等依赖算子的
# kernel 二进制与 kernel/config/<soc>/<op>.json，运行期报 JSON configuration
# file of operator aclnnChunkGatedDeltaRuleBwd_0_* cannot be found。
#
# 这里在过滤构建下把依赖算子的 op_host 一并加入本次构建（等价于它们被 --ops 选中），
# 使它们照常参与 opbuild 与 kernel 编译打包；全量构建或依赖已被 --ops 选中时跳过，
# 避免重复 add_subdirectory。改动只在算子目录内，不修改公共构建脚本。
set(_gdn_bwd_build_all_ops FALSE)
if (NOT DEFINED ASCEND_OP_NAME OR "${ASCEND_OP_NAME}" STREQUAL ""
        OR "${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
    set(_gdn_bwd_build_all_ops TRUE)
endif()

foreach(_gdn_bwd_dep ${chunk_gated_delta_rule_bwd_depends})
    if (ENABLE_EXPERIMENTAL)
        set(_gdn_bwd_dep_path "experimental/${_gdn_bwd_dep}")
    else()
        set(_gdn_bwd_dep_path "${_gdn_bwd_dep}")
    endif()
    set(_gdn_bwd_dep_dir "${CMAKE_SOURCE_DIR}/${_gdn_bwd_dep_path}")
    if (NOT EXISTS "${_gdn_bwd_dep_dir}/op_host/CMakeLists.txt")
        continue()
    endif()
    get_filename_component(_gdn_bwd_dep_name "${_gdn_bwd_dep}" NAME)
    if (NOT BUILD_OPEN_PROJECT)
        if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_gdn_bwd_dep_name})
            continue()
        endif()
    endif()
    if (_gdn_bwd_build_all_ops)
        continue()
    endif()
    if (${_gdn_bwd_dep_name} IN_LIST ASCEND_OP_NAME)
        continue()
    endif()
    message(STATUS "[gdn-bwd-deps] add dependency op ${_gdn_bwd_dep_name} to this build")
    add_subdirectory(
        ${_gdn_bwd_dep_dir}/op_host
        ${CMAKE_BINARY_DIR}/chunk_gated_delta_rule_bwd_deps/${_gdn_bwd_dep_name}
    )
endforeach()

add_modules_sources(OPTYPE chunk_gated_delta_rule_bwd ACLNNTYPE aclnn_exclude)
target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE
    GDN_BWD_ENABLE_ARCH22=${_gdn_bwd_enable_arch22})
