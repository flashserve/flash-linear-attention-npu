# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
# Catlass 头文件（include/catlass、include/tla）：与 opbase 相同，在配置阶段用 FetchContent_Populate 拉取到 CANN_3RD_LIB_PATH/catlass。
# 默认 CANN_3RD_LIB_PATH 与 variables.cmake 一致为 ${PROJECT_SOURCE_DIR}/third_party，故路径通常即仓库内 third_party/catlass。

include_guard(GLOBAL)

if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH "${PROJECT_SOURCE_DIR}/third_party")
endif()

set(CATLASS_VENDOR_GIT_REPOSITORY
    "https://gitcode.com/cann/catlass.git")
set(CATLASS_VENDOR_GIT_TAG
    "9034426342c8f53c02fd469b4cfe880d49e4b60c")
set(_CATLASS_SOURCE_DIR "${CANN_3RD_LIB_PATH}/catlass")
set(_CATLASS_MARK "${_CATLASS_SOURCE_DIR}/include/catlass/catlass.hpp")

if(EXISTS "${_CATLASS_MARK}")
  if(EXISTS "${_CATLASS_SOURCE_DIR}/.git")
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${_CATLASS_SOURCE_DIR}"
      OUTPUT_VARIABLE _CATLASS_HEAD OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _CATLASS_RESULT)
    if(NOT _CATLASS_RESULT EQUAL 0 OR NOT _CATLASS_HEAD STREQUAL CATLASS_VENDOR_GIT_TAG)
      message(FATAL_ERROR
        "Cached Catlass revision '${_CATLASS_HEAD}' does not match ${CATLASS_VENDOR_GIT_TAG}. "
        "Preserve local changes, then update ${_CATLASS_SOURCE_DIR} to the required revision and rebuild.")
    endif()
  endif()
  get_filename_component(OPS_ADV_CATLASS_INC "${_CATLASS_SOURCE_DIR}/include" REALPATH)
  message(STATUS "[ThirdPartyLib][catlass] Found headers at ${OPS_ADV_CATLASS_INC}")
else()
  message(STATUS
          "[ThirdPartyLib][catlass] Missing ${_CATLASS_MARK}, fetching via FetchContent "
          "(GIT_REPOSITORY ${CATLASS_VENDOR_GIT_REPOSITORY}, "
          "GIT_TAG ${CATLASS_VENDOR_GIT_TAG}) …")
  if(EXISTS "${CMAKE_BINARY_DIR}/_deps/catlass_vendor-subbuild")
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/_deps/catlass_vendor-subbuild")
  endif()
  include(FetchContent)
  FetchContent_Declare(
    catlass_vendor
    GIT_REPOSITORY ${CATLASS_VENDOR_GIT_REPOSITORY}
    GIT_TAG ${CATLASS_VENDOR_GIT_TAG}
    GIT_PROGRESS TRUE
    SOURCE_DIR "${_CATLASS_SOURCE_DIR}")
  FetchContent_Populate(catlass_vendor)
  if(NOT EXISTS "${_CATLASS_MARK}")
    message(FATAL_ERROR
            "[ThirdPartyLib][catlass] Still missing ${_CATLASS_MARK} after FetchContent. "
            "Check network and gitcode.com access. If ${_CATLASS_SOURCE_DIR} is incomplete, remove it and re-run CMake configure.")
  endif()
  get_filename_component(OPS_ADV_CATLASS_INC "${_CATLASS_SOURCE_DIR}/include" REALPATH)
  message(STATUS "[ThirdPartyLib][catlass] Populated at ${OPS_ADV_CATLASS_INC}")
endif()
