add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_intra_sub_chunk_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd_intra_sub_chunk ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaFwdIntraSubChunk
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -g
)
