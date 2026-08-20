add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gdn_core_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gdn_core_fwd ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGdnCoreFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_kkt_solve_tri/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/op_kernel
)
