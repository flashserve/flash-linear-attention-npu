add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd_ho_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd_ho ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwdHO
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel
)
