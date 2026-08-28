add_op_to_compiled_list()

get_filename_component(_chunk_kkt_solve_tri_op_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(chunk_kkt_solve_tri_dir "${_chunk_kkt_solve_tri_op_dir}" CACHE INTERNAL
    "Source directory alias for ChunkKktSolveTri opFile")
set(chunk_cumsum_kkt_solve_tri_dir "${_chunk_kkt_solve_tri_op_dir}" CACHE INTERNAL
    "Source directory alias for ChunkCumsumKktSolveTri opFile")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kkt_solve_tri_def.cpp
        chunk_cumsum_kkt_solve_tri_def.cpp
    )
endif()

add_modules_sources(
    OPTYPE chunk_kkt_solve_tri chunk_cumsum_kkt_solve_tri
    ACLNNTYPE aclnn_exclude aclnn_exclude
)
add_ops_compile_options(
    OP_NAME ChunkKktSolveTri
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel
)

add_ops_compile_options(
    OP_NAME ChunkCumsumKktSolveTri
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel
)
