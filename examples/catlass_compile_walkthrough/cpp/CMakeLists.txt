cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
project(catlass_compile_walkthrough LANGUAGES ASC CXX)

set(CATLASS_ROOT "" CACHE PATH "CATLASS checkout (FLA pinned revision recommended)")
if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/gemm/kernel/basic_matmul.hpp")
    message(FATAL_ERROR "Set -DCATLASS_ROOT=/path/to/catlass")
endif()
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Source the CANN set_env.sh first")
endif()
set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE ASC)
add_executable(matmul_cpp matmul.cpp)
target_compile_features(matmul_cpp PRIVATE cxx_std_17)
target_include_directories(matmul_cpp PRIVATE "${CATLASS_ROOT}/include")
# Match the host C++ compiler chosen by CMake. Bisheng may otherwise discover
# a newer, incomplete GCC installation (for example GCC 12 without its headers).
set(HOST_STDLIB_INCLUDES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
list(FILTER HOST_STDLIB_INCLUDES INCLUDE REGEX "/c[+][+]/")
# Do not add GCC's builtin include directory: arm_neon.h must come from Bisheng.
target_include_directories(matmul_cpp SYSTEM PRIVATE ${HOST_STDLIB_INCLUDES})
target_compile_definitions(matmul_cpp PRIVATE CATLASS_ARCH=2201 _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(matmul_cpp PRIVATE
    --npu-arch=dav-2201 -O2 -save-temps=obj
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-unused-value)
target_link_directories(matmul_cpp PRIVATE
    "${CANN_ROOT}/lib64" "${CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib"
    ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(matmul_cpp PRIVATE
    ascendc_runtime runtime profapi mmpa ascend_dump c_sec error_manager
    ascendcl tiling_api nnopbase platform dl ascend_hal)
foreach(log_library ascendalog unified_dlog)
    if(EXISTS "${CANN_ROOT}/lib64/lib${log_library}.so")
        target_link_libraries(matmul_cpp PRIVATE ${log_library})
    endif()
endforeach()
