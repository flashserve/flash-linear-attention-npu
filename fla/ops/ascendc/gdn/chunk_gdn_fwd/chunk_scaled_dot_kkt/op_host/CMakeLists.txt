add_op_to_compiled_list()

get_filename_component(_chunk_scaled_dot_kkt_op_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(chunk_scaled_dot_kkt_dir "${_chunk_scaled_dot_kkt_op_dir}" CACHE INTERNAL
    "Source directory alias for ChunkScaledDotKkt opFile")
set(chunk_cumsum_kkt_dir "${_chunk_scaled_dot_kkt_op_dir}" CACHE INTERNAL
    "Source directory alias for ChunkCumsumKkt opFile")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_scaled_dot_kkt_def.cpp
        chunk_cumsum_kkt_def.cpp
    )
endif()

add_modules_sources(
    OPTYPE chunk_scaled_dot_kkt chunk_cumsum_kkt
    ACLNNTYPE aclnn_exclude aclnn_exclude
)

add_ops_compile_options(
    OP_NAME ChunkScaledDotKkt
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

add_ops_compile_options(
    OP_NAME ChunkCumsumKkt
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
