add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE chunk_gated_delta_rule_bwd_finalize_def.cpp)
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_bwd_finalize ACLNNTYPE aclnn_exclude)
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
