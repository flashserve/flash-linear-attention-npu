add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_finalize_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd_finalize ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaFwdFinalize
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaFwdFinalize
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()
