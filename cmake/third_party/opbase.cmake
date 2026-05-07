# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPBASE_TAG_ID c8d83f3e57a63a7375e89a2d6937452c0ae2e522)

if(EXISTS "${PROJECT_SOURCE_DIR}/../../ops-base")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../../ops-base REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/opbase")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/opbase REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
  execute_process(
    COMMAND git checkout ${OPBASE_TAG_ID}
    WORKING_DIRECTORY ${OPBASE_SOURCE_PATH}
  )
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

  if(EXISTS ${OPBASE_SOURCE_PATH}/include)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/include)
  endif()
  if(EXISTS ${OPBASE_SOURCE_PATH}/aicpu_common)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/aicpu_common)
  endif()
endif()
