add_executable(
    chunk_kda_fwd_finalize_tiling_processor_test
    chunk_kda_fwd_finalize_tiling_processor_test.cpp
)
target_compile_features(
    chunk_kda_fwd_finalize_tiling_processor_test
    PRIVATE cxx_std_17
)
add_test(
    NAME chunk_kda_fwd_finalize_tiling_processor_test
    COMMAND chunk_kda_fwd_finalize_tiling_processor_test
)
