# ----------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPBASE_TAG_ID c8d83f3e57a63a7375e89a2d6937452c0ae2e522)
set(OPBASE_REVISION_FILE ".fla_npu_revision")

if(FLA_NPU_OFFLINE_BUILD)
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/opbase REALPATH)
  message(STATUS "Find offline opbase source dir: ${OPBASE_SOURCE_PATH}")
elseif(EXISTS "${PROJECT_SOURCE_DIR}/../../ops-base")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../../ops-base REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/opbase")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/opbase REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
  if(EXISTS "${OPBASE_SOURCE_PATH}/.git")
    execute_process(
      COMMAND git checkout ${OPBASE_TAG_ID}
      WORKING_DIRECTORY ${OPBASE_SOURCE_PATH}
      RESULT_VARIABLE OPBASE_CHECKOUT_RESULT
    )
    if(NOT OPBASE_CHECKOUT_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to checkout opbase revision ${OPBASE_TAG_ID}")
    endif()
    file(WRITE "${OPBASE_SOURCE_PATH}/${OPBASE_REVISION_FILE}" "${OPBASE_TAG_ID}\n")
  elseif(EXISTS "${OPBASE_SOURCE_PATH}/${OPBASE_REVISION_FILE}")
    file(READ "${OPBASE_SOURCE_PATH}/${OPBASE_REVISION_FILE}" OPBASE_CACHED_REVISION)
    string(STRIP "${OPBASE_CACHED_REVISION}" OPBASE_CACHED_REVISION)
    if(NOT OPBASE_CACHED_REVISION STREQUAL OPBASE_TAG_ID)
      message(FATAL_ERROR
              "Cached opbase revision mismatch: expected ${OPBASE_TAG_ID}, "
              "got ${OPBASE_CACHED_REVISION}")
    endif()
  else()
    message(FATAL_ERROR
            "The cached opbase directory has no git metadata or revision marker. "
            "Prepare it with scripts/tools/third_lib_download.py.")
  endif()
else()
  if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild")
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/build/_deps/opbase-subbuild)
  endif()
  include(FetchContent)

  FetchContent_Declare(
    opbase
    GIT_REPOSITORY https://gitcode.com/cann/opbase.git
    GIT_TAG ${OPBASE_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/opbase)

  FetchContent_Populate(opbase)

  set(OPBASE_SOURCE_PATH ${CANN_3RD_LIB_PATH}/opbase)
  file(WRITE "${OPBASE_SOURCE_PATH}/${OPBASE_REVISION_FILE}" "${OPBASE_TAG_ID}\n")

  if(EXISTS ${OPBASE_SOURCE_PATH}/include)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/include)
  endif()
  if(EXISTS ${OPBASE_SOURCE_PATH}/aicpu_common)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/aicpu_common)
  endif()
endif()
