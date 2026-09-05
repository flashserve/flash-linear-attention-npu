# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# Licensed under the BSD 3-Clause License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_bwd_prepare_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_bwd_prepare ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaBwdPrepare
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
