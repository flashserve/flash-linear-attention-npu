add_op_to_compiled_list()

set(chunk_kda_bwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu"
    "fla/ops/ascendc/kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post"
    "fla/ops/ascendc/kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post_varlen"
    CACHE STRING "Kernel source dependencies for chunk_kda_bwd" FORCE)

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_bwd_def.cpp
    )
endif()

add_modules_sources(
    OPTYPE chunk_kda_bwd
    ACLNNTYPE aclnn_exclude
)

add_ops_compile_options(
    OP_NAME ChunkKdaBwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaBwd
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()
