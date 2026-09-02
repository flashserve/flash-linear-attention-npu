add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        kda_gate_bwd_post_def.cpp
    )
endif()

add_modules_sources(
    OPTYPE kda_gate_bwd_post
    ACLNNTYPE aclnn_exclude
)

add_ops_compile_options(
    OP_NAME KdaGateBwdPost
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
