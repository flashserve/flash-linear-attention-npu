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
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _CATLASS_SOCS)
string(REPLACE "," ";" _CATLASS_SOCS "${_CATLASS_SOCS}")
list(REMOVE_DUPLICATES _CATLASS_SOCS)
if("ascend960dt" IN_LIST _CATLASS_SOCS)
  list(LENGTH _CATLASS_SOCS _CATLASS_SOC_COUNT)
  if(_CATLASS_SOC_COUNT GREATER 1)
    message(FATAL_ERROR "Build ascend960dt separately from other SoCs: they require different Catlass revisions.")
  endif()
  set(CATLASS_VENDOR_GIT_TAG "86acf1db511ee727d6b2995dd1b41f7f4f9221a8")
  set(_CATLASS_ARCH_SHA256 "ae634315dbedfa1f0b66c808caf2763573b801dd0ddba52f2611e1e5a237bca8")
else()
  set(CATLASS_VENDOR_GIT_TAG "41bf90da655bba3c66d0acd7e00abe33960ecfd6")
  set(_CATLASS_ARCH_SHA256 "f9f730ef9dee785d90fb4a824421a3d514eb44d333abc7ec29efc44e40aa44a1")
endif()
message(STATUS "[ThirdPartyLib][catlass] SoC=${_CATLASS_SOCS}, revision=${CATLASS_VENDOR_GIT_TAG}")
set(_CATLASS_SOURCE_DIR "${CANN_3RD_LIB_PATH}/catlass")
set(_CATLASS_MARK "${_CATLASS_SOURCE_DIR}/include/catlass/catlass.hpp")

if(EXISTS "${_CATLASS_MARK}")
  if(EXISTS "${_CATLASS_SOURCE_DIR}/.git")
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${_CATLASS_SOURCE_DIR}"
      OUTPUT_VARIABLE _CATLASS_HEAD OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _CATLASS_RESULT)
    if(NOT _CATLASS_RESULT EQUAL 0)
      message(FATAL_ERROR "Cannot read Catlass revision at ${_CATLASS_SOURCE_DIR}")
    endif()
    if(NOT _CATLASS_HEAD STREQUAL CATLASS_VENDOR_GIT_TAG)
      execute_process(COMMAND "${GIT_EXECUTABLE}" diff --quiet HEAD
        WORKING_DIRECTORY "${_CATLASS_SOURCE_DIR}" RESULT_VARIABLE _CATLASS_RESULT)
      if(NOT _CATLASS_RESULT EQUAL 0)
        message(FATAL_ERROR "Catlass has local changes; preserve them before switching to ${CATLASS_VENDOR_GIT_TAG}.")
      endif()
      execute_process(COMMAND "${GIT_EXECUTABLE}" cat-file -e "${CATLASS_VENDOR_GIT_TAG}^{commit}"
        WORKING_DIRECTORY "${_CATLASS_SOURCE_DIR}" RESULT_VARIABLE _CATLASS_RESULT ERROR_QUIET)
      if(NOT _CATLASS_RESULT EQUAL 0)
        if(FETCHCONTENT_FULLY_DISCONNECTED OR FETCHCONTENT_UPDATES_DISCONNECTED)
          message(FATAL_ERROR "Required Catlass revision ${CATLASS_VENDOR_GIT_TAG} is not cached for offline use.")
        endif()
        execute_process(COMMAND "${GIT_EXECUTABLE}" fetch "${CATLASS_VENDOR_GIT_REPOSITORY}" "${CATLASS_VENDOR_GIT_TAG}"
          WORKING_DIRECTORY "${_CATLASS_SOURCE_DIR}" RESULT_VARIABLE _CATLASS_RESULT)
        if(NOT _CATLASS_RESULT EQUAL 0)
          message(FATAL_ERROR "Failed to fetch Catlass revision ${CATLASS_VENDOR_GIT_TAG}")
        endif()
      endif()
      execute_process(COMMAND "${GIT_EXECUTABLE}" checkout --detach "${CATLASS_VENDOR_GIT_TAG}"
        WORKING_DIRECTORY "${_CATLASS_SOURCE_DIR}" RESULT_VARIABLE _CATLASS_RESULT)
      if(NOT _CATLASS_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to select Catlass revision ${CATLASS_VENDOR_GIT_TAG}")
      endif()
    endif()
  else()
    # Offline bundles omit .git. Identify the two supported header layouts by
    # arch.hpp, whose contents differ between the pinned revisions.
    set(_CATLASS_ARCH_HEADER "${_CATLASS_SOURCE_DIR}/include/catlass/arch/arch.hpp")
    if(NOT EXISTS "${_CATLASS_ARCH_HEADER}")
      message(FATAL_ERROR "Missing Catlass arch.hpp in offline sources: ${_CATLASS_SOURCE_DIR}")
    endif()
    file(SHA256 "${_CATLASS_ARCH_HEADER}" _CATLASS_BUNDLED_ARCH_SHA256)
    if(NOT _CATLASS_BUNDLED_ARCH_SHA256 STREQUAL _CATLASS_ARCH_SHA256)
      message(FATAL_ERROR "Offline Catlass headers do not match ${CATLASS_VENDOR_GIT_TAG}. Prepare a matching bundle for ${_CATLASS_SOCS}.")
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
