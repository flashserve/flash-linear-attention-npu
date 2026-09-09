if("chunk_kda_bwd" IN_LIST OP_LIST)
    foreach(INTERNAL_OP kda_gate_bwd_post kda_gate_bwd_post_varlen)
        if(NOT "${INTERNAL_OP}" IN_LIST OP_LIST)
            add_subdirectory(
                "${CMAKE_CURRENT_LIST_DIR}/../internal_ops/${INTERNAL_OP}/op_host"
                "${CMAKE_CURRENT_BINARY_DIR}/internal_ops/${INTERNAL_OP}"
            )
        endif()
    endforeach()
endif()

add_op_to_compiled_list()

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
