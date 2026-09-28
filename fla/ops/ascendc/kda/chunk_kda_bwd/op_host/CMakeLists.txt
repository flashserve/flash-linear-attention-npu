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

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    set(chunk_kda_bwd_depends
        "fla/ops/ascendc/kda/chunk_kda_bwd_prepare;fla/ops/ascendc/kda/chunk_kda_bwd_finalize;fla/ops/ascendc/kda/chunk_kda_bwd_recompute;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h;fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu"
        CACHE STRING "Optimized KDA backward L0 dependencies" FORCE)
    foreach(DEP_PATH IN LISTS chunk_kda_bwd_depends)
        get_filename_component(DEP_NAME "${DEP_PATH}" NAME)
        if(NOT "${DEP_NAME}" IN_LIST OP_LIST)
            add_subdirectory("${PROJECT_SOURCE_DIR}/${DEP_PATH}/op_host"
                "${CMAKE_CURRENT_BINARY_DIR}/optimized/${DEP_NAME}")
        endif()
    endforeach()
else()
    unset(chunk_kda_bwd_depends CACHE)
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

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_chunk_kda_bwd_v2.cpp
        TARGET_DIRECTORY ${OPHOST_NAME}_opapi_obj PROPERTIES
        COMPILE_DEFINITIONS KDA_BWD_OPTIMIZED_A5)
endif()

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
