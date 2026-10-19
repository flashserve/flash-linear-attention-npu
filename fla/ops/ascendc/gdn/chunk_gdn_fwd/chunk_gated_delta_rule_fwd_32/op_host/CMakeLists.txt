add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd32_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd_32 ACLNNTYPE aclnn_exclude)

add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwd32
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CMAKE_CURRENT_SOURCE_DIR}
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleFwd32
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
