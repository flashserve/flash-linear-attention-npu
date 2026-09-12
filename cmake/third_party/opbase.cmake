# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------
# Adapted for flash-linear-attention-npu by Tianjin University.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPBASE_LEGACY_TAG_ID c8d83f3e57a63a7375e89a2d6937452c0ae2e522)
set(OPBASE_CONST_API_TAG_ID c8cfc45e350d4e07cd8cab8448d3b40f9727ea4c)
set(OPBASE_TAG_ID ${OPBASE_LEGACY_TAG_ID})

set(OPBASE_ELEWISE_TILING_HEADER
    "${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h")
if(EXISTS "${OPBASE_ELEWISE_TILING_HEADER}")
  file(READ "${OPBASE_ELEWISE_TILING_HEADER}" OPBASE_ELEWISE_TILING_CONTENT)
  string(FIND "${OPBASE_ELEWISE_TILING_CONTENT}" "int64_t GetBlockDim() const;" OPBASE_CONST_API_POSITION)
  if(NOT OPBASE_CONST_API_POSITION EQUAL -1)
    set(OPBASE_TAG_ID ${OPBASE_CONST_API_TAG_ID})
  endif()
endif()
message(STATUS "Select opbase revision: ${OPBASE_TAG_ID}")
unset(OPBASE_ELEWISE_TILING_CONTENT)
unset(OPBASE_CONST_API_POSITION)
unset(OPBASE_ELEWISE_TILING_HEADER)

if(EXISTS "${PROJECT_SOURCE_DIR}/../../ops-base")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../../ops-base REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/opbase")
  get_filename_component(OPBASE_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/opbase REALPATH)
  message(STATUS "Find opbase source dir: ${OPBASE_SOURCE_PATH}")
  # 优先使用 git 检出目录并切到目标 tag（在线/预置 git 检出场景）。
  # 离线 bundle 提供的是不带 .git 的解压源码目录，此时不执行 git checkout，
  # 直接使用 bundle 制作时已固化的源码。
  if(EXISTS "${OPBASE_SOURCE_PATH}/.git")
    execute_process(
      COMMAND git checkout ${OPBASE_TAG_ID}
      WORKING_DIRECTORY ${OPBASE_SOURCE_PATH}
    )
  else()
    message(STATUS "opbase source dir has no .git; using bundled tarball sources as-is")
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

  if(EXISTS ${OPBASE_SOURCE_PATH}/include)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/include)
  endif()
  if(EXISTS ${OPBASE_SOURCE_PATH}/aicpu_common)
    file(REMOVE_RECURSE ${OPBASE_SOURCE_PATH}/aicpu_common)
  endif()
endif()
