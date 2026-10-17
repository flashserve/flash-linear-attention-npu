if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_compile_definitions(KDA_BWD_STAGES_ARCH22)
    set(chunk_kda_bwd_finalize_depends
    "fla/ops/ascendc/kda/chunk_kda_bwd"
    CACHE STRING "Shared KDA backward stage sources" FORCE)
else()
    unset(chunk_kda_bwd_finalize_depends CACHE)
endif()
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE chunk_kda_bwd_finalize_def.cpp)
    if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        target_compile_definitions(op_host_aclnnExc PRIVATE KDA_BWD_STAGES_ARCH22)
    endif()
endif()

add_modules_sources(OPTYPE chunk_kda_bwd_finalize ACLNNTYPE aclnn_exclude)
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
