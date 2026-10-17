# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# Licensed under the BSD 3-Clause License.
# -----------------------------------------------------------------------------------------------------------
if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_compile_definitions(KDA_BWD_STAGES_ARCH22)
    set(chunk_kda_bwd_prepare_depends
    "fla/ops/ascendc/kda/chunk_kda_bwd"
    CACHE STRING "Shared KDA backward stage sources" FORCE)
else()
    unset(chunk_kda_bwd_prepare_depends CACHE)
endif()
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_bwd_prepare_def.cpp
    )
    if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        target_compile_definitions(op_host_aclnnExc PRIVATE KDA_BWD_STAGES_ARCH22)
    endif()
endif()

add_modules_sources(OPTYPE chunk_kda_bwd_prepare ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaBwdPrepare
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
