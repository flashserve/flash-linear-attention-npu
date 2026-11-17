# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
# Adapted for flash-linear-attention-npu by Tianjin University.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
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
  ${OPBASE_SOURCE_PATH}/include
  ${OPBASE_SOURCE_PATH}/pkg_inc
  ${TOP_DIR}/ops-base/include
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
  # AICPU public headers are separate from opbase_export.h in source checkouts.
  foreach(OPBASE_HEAD_PATH IN LISTS OPBASE_HEAD_SEARCH_PATHS)
    set(OPBASE_AICPU_CONTEXT_DIR "${OPBASE_HEAD_PATH}/op_common/aicpu_common/context")
    if(EXISTS "${OPBASE_AICPU_CONTEXT_DIR}/common/cpu_kernel.h")
      list(APPEND OPBASE_INC_DIRS
        ${OPBASE_AICPU_CONTEXT_DIR}
        ${OPBASE_AICPU_CONTEXT_DIR}/common
        ${OPBASE_AICPU_CONTEXT_DIR}/cpu_proto
        ${OPBASE_AICPU_CONTEXT_DIR}/utils
        ${OPBASE_AICPU_CONTEXT_DIR}/cust_op
      )
    endif()
  endforeach()
  list(REMOVE_DUPLICATES OPBASE_INC_DIRS)
endif()
