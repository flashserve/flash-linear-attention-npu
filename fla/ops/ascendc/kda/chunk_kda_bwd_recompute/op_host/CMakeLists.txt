# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# Licensed under the BSD 3-Clause License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_bwd_recompute_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_bwd_recompute ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaBwdRecompute
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaBwdRecompute
        --cce-auto-sync=off
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
