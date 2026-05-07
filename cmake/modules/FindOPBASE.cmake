# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(OPBASE_FOUND)
  message(STATUS "OpBase has been found")
  return()
endif()

include(FindPackageHandleStandardArgs)

set(OPBASE_HEAD_SEARCH_PATHS
  ${OPBASE_SOURCE_PATH}/pkg_inc
  ${TOP_DIR}/ops-base/pkg_inc             # compile with ci
)

find_path(OPBASE_INC_DIR
  NAMES op_common/op_host/util/opbase_export.h
  PATHS ${OPBASE_HEAD_SEARCH_PATHS}
  NO_CMAKE_SYSTEM_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

find_package_handle_standard_args(OPBASE
            REQUIRED_VARS OPBASE_INC_DIR)

get_filename_component(OPBASE_INC_DIR ${OPBASE_INC_DIR} REALPATH)

if(OPBASE_FOUND)
  if(NOT OPBASE_FIND_QUIETLY)
    message(STATUS "Found OPABSE include:${OPBASE_INC_DIR}")
    message(STATUS "Found OPABSE lib:${OPBASE_LIB_DIR}")
  endif()
  set(OPBASE_INC_DIRS
    ${OPBASE_INC_DIR}
    ${OPBASE_INC_DIR}/op_common
    ${OPBASE_INC_DIR}/op_common/op_host
    ${OPBASE_INC_DIR}/op_common/atvoss
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc
  )
endif()