# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_bwd_intra_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_bwd_intra ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaBwdIntra
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
