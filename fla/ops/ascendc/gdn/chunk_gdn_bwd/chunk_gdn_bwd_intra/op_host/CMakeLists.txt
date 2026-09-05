# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# Licensed under the BSD 3-Clause License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gdn_bwd_intra_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gdn_bwd_intra ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGdnBwdIntra
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
