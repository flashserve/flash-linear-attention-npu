# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

project(cann_ops-transformer)

option(BUILD_OPEN_PROJECT         "Build open ascend ops project."  ON)
option(BUILD_OPS_RTY_KERNEL       "Build return yellow kernel."     OFF)
option(ENABLE_CCACHE              "Enable ccache capability"        ON)
option(ENABLE_BUILT_IN            "Enable built-in package"         OFF)
option(ENABLE_STATIC              "Enable Static"                   OFF)
option(ENABLE_EXPERIMENTAL        "Enable experimental module"      OFF)
option(ENABLE_TEST                "Enable test"                     OFF)
option(ENABLE_UT_EXEC             "Enable exec ut"                  OFF)
option(ENABLE_ASAN                "Enable asan"                     OFF)
option(ENABLE_VALGRIND            "Enable valgrind"                 OFF)
option(OP_HOST_UT                 "Enable ophost ut"                OFF)
option(OP_API_UT                  "Enable opapi ut"                 OFF)
option(OP_GRAPH_UT                "Enable graph ut"                 OFF)
option(OP_KERNEL_UT               "Enable kernel ut"                OFF)
option(OP_KERNEL_AICPU_UT         "Enable aicpu kernel ut"          OFF)
option(UT_TEST_ALL                "Enable all ut"                   OFF)
option(ENABLE_OOM                 "Enable kernel oom"               OFF)

set(ASCEND_COMPUTE_UNIT           "ascend910b"           CACHE   STRING   "soc that need to be compiled")
set(ASCEND_OP_NAME                "ALL"                           CACHE   STRING   "operators that need to be compiled")
set(ARCH_DIRECTORY                ""                           CACHE   STRING   "arch directory that need to be compiled")
set(VENDOR_NAME                   "custom"                     CACHE   STRING   "vendor name")
set(ASCEND_ALL_COMPUTE_UNIT       "ascend910b;ascend910_93;ascend910_95;kirinx90" CACHE STRING "all soc list")

set(SOC_VERSION_LIST ascend910b ascend910_93 ascend910_95 kirinx90)
set(ARCH_DIRECTORY_LIST arch32 arch32 arch35 arch32)

foreach(SOC_VERSION ${ASCEND_COMPUTE_UNIT})
  list(FIND SOC_VERSION_LIST ${SOC_VERSION} INDEX)
  if(NOT INDEX EQUAL -1)
    list(GET ARCH_DIRECTORY_LIST ${INDEX} VAL)
    list(APPEND ARCH_DIRECTORY ${VAL})
  else()
    message(STATUS "unsupport chip type")
    if ((NOT BUILD_OPS_RTY_KERNEL) AND (BUILD_OPEN_PROJECT))
      include(cmake/build_empty_package.cmake)
      cpack_empty_package()
      return()
    endif()
  endif()
endforeach()

list(FIND ARCH_DIRECTORY "arch32" INDXE)
if(NOT INDEX EQUAL -1)
    list(APPEND ARCH_DIRECTORY "arch22")
endif()

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "compile project with library")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-transformer with cann pkg" ON)
else()
  message(STATUS "compile project with src")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-transformer with cann source" OFF)
endif()

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()

#外部传参
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
  if(ENABLE_DEBUG)
    set(CMAKE_BUILD_MODE "${CMAKE_BUILD_MODE} -g")
  endif()
  set(COMPILE_OP_MODE ${CMAKE_BUILD_MODE})
else()
  if(ENABLE_TEST)
    set(COMPILE_OP_MODE "-O0 -g")
  endif()
  if(ENABLE_DEBUG)
    set(CMAKE_BUILD_MODE "-g")
  endif()
endif()

set(PKG_NAME transformer)
set(OPS_TRANSFORMER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17 CACHE STRING "c++17 is needed for this project")
set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_BINARY_DIR}/_CPack_Packages"
)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/third_party/json.cmake)
if (ENABLE_TEST)
    include(${PROJECT_SOURCE_DIR}/cmake/third_party/gtest.cmake)
endif()
include(${OPS_ADV_CMAKE_DIR}/ut.cmake)

if (BUILD_OPEN_PROJECT)
    include(cmake/intf.cmake)
    add_definitions(-DBUILD_OPEN_PROJECT)
    message(STATUS "Start building custom package.")
    include(ExternalProject)
    include(cmake/dependencies.cmake)
    include(cmake/variables.cmake)
    include(cmake/obj_func.cmake)
    include(cmake/third_party/abseil-cpp.cmake)
    include(cmake/third_party/ascend_protobuf.cmake)

    include(cmake/third_party/makeself-fetch.cmake)
    include(cmake/custom_build.cmake)
    message(STATUS "End building custom package.")
    include(cmake/opbuild.cmake)
    merge_graph_headers(
        TARGET merge_ops_proto ALL
        OUT_DIR ${ASCEND_GRAPH_CONF_DST}
    )
    if (ENABLE_OPS_HOST)
        gen_aclnn_with_opdef()
    endif()
    if (ENABLE_STATIC)
        include(cmake/static.cmake)
    endif()
    if (ENABLE_BUILT_IN)
        message(STATUS "Start building built-in package.")
        include(cmake/symbol.cmake)
        gen_norm_symbol()
        include(cmake/package.cmake)
        pack_built_in()
    else()
        include(cmake/package.cmake)
        pack_tiling_sink()
    endif()
    return()
endif()