unset(chunk_kda_bwd_finalize_depends CACHE)
if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_compile_definitions(KDA_BWD_STAGES_ARCH22)
endif()
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE chunk_kda_bwd_finalize_def.cpp)
    if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        target_compile_definitions(op_host_aclnnExc PRIVATE KDA_BWD_STAGES_ARCH22)
    endif()
endif()

add_modules_sources(OPTYPE chunk_kda_bwd_finalize ACLNNTYPE aclnn_exclude)
if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/chunk_kda_bwd_finalize.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_chunk_kda_bwd_finalize.cpp
        TARGET_DIRECTORY ${OPHOST_NAME}_opapi_obj PROPERTIES
        COMPILE_DEFINITIONS KDA_BWD_STAGES_ARCH22)
endif()
add_ops_compile_options(
    OP_NAME ChunkKdaBwdFinalize
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaBwdFinalize
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
