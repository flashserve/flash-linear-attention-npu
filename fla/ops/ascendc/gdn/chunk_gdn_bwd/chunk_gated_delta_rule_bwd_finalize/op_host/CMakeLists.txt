add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE chunk_gated_delta_rule_bwd_finalize_def.cpp)
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_bwd_finalize ACLNNTYPE aclnn_exclude)

# Host object 按本次 SoC 列表共用。纯 A5 构建不编入 arch22 helper 或平台分派；
# 含 A2/A3 的构建保留运行期分派，混合 SoC 包仍共用同一份 Host object。
set(_gdn_finalize_enable_arch22 0)
foreach(_gdn_finalize_compute_unit IN LISTS ASCEND_COMPUTE_UNIT)
    string(TOLOWER "${_gdn_finalize_compute_unit}" _gdn_finalize_compute_unit_lower)
    if(_gdn_finalize_compute_unit_lower STREQUAL "ascend910b" OR
       _gdn_finalize_compute_unit_lower STREQUAL "ascend910_93")
        set(_gdn_finalize_enable_arch22 1)
    endif()
endforeach()
target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE
    GDN_FINALIZE_ENABLE_ARCH22=${_gdn_finalize_enable_arch22})
target_compile_definitions(${OPHOST_NAME}_tiling_obj PRIVATE
    GDN_FINALIZE_ENABLE_ARCH22=${_gdn_finalize_enable_arch22})
unset(_gdn_finalize_enable_arch22)
unset(_gdn_finalize_compute_unit)
unset(_gdn_finalize_compute_unit_lower)

add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleBwdFinalize
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleBwdFinalize
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
