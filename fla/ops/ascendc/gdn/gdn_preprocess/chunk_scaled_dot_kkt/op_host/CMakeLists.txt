add_op_to_compiled_list()

get_filename_component(_chunk_scaled_dot_kkt_op_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(chunk_scaled_dot_kkt_dir "${_chunk_scaled_dot_kkt_op_dir}" CACHE INTERNAL
    "Source directory alias for ChunkScaledDotKkt opFile")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        chunk_scaled_dot_kkt_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_scaled_dot_kkt ACLNNTYPE aclnn)

add_ops_compile_options(
    OP_NAME ChunkScaledDotKkt
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
