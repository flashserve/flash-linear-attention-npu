cmake_minimum_required(VERSION 3.16)
project(chunk_kda_bwd_finalize_tiling_test LANGUAGES CXX)

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/latest" CACHE PATH "CANN package root")
set(UT_COMMON_DIR "" CACHE PATH "Directory containing tiling_context_faker.cpp")
if(NOT UT_COMMON_DIR)
    message(FATAL_ERROR "Set UT_COMMON_DIR to the existing host UT common directory")
endif()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../.." REALPATH)
set(CANN_ARCH_DIR "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")

add_executable(chunk_kda_bwd_finalize_tiling_test
    chunk_kda_bwd_finalize_tiling_test.cpp
    ../op_tiling/chunk_kda_bwd_finalize_tiling.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)
target_compile_features(chunk_kda_bwd_finalize_tiling_test PRIVATE cxx_std_17)
target_compile_definitions(chunk_kda_bwd_finalize_tiling_test PRIVATE
    LOG_CPP
    _GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(chunk_kda_bwd_finalize_tiling_test PRIVATE
    ${REPO_ROOT}/common/include
    ${UT_COMMON_DIR}
    ${CANN_ARCH_DIR}/include
    ${CANN_ARCH_DIR}/include/op_common
    ${CANN_ARCH_DIR}/include/base/context_builder
    ${CANN_ARCH_DIR}/pkg_inc
    ${CANN_ARCH_DIR}/asc/include
    ${CANN_ARCH_DIR}/asc/include/utils/tiling
)
target_link_directories(chunk_kda_bwd_finalize_tiling_test PRIVATE
    ${CANN_ARCH_DIR}/lib64
)
target_link_libraries(chunk_kda_bwd_finalize_tiling_test PRIVATE
    graph
    graph_base
    exe_graph
    platform
    register
    opp_registry
    ascendalog
    unified_dlog
    tiling_api
    c_sec
    error_manager
    metadef
    dl
    pthread
)
set_target_properties(chunk_kda_bwd_finalize_tiling_test PROPERTIES
    BUILD_RPATH "${CANN_ARCH_DIR}/lib64"
)

enable_testing()
add_test(NAME chunk_kda_bwd_finalize_tiling_test
    COMMAND chunk_kda_bwd_finalize_tiling_test
)
