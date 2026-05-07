# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
set(EIGEN_NAME "eigen")
set(EIGEN_DST_DIR "${CANN_3RD_LIB_PATH}/eigen")
# 检查目录是否为空
file(GLOB EIGEN_CONTENT "${CANN_3RD_LIB_PATH}/eigen/*")

if (IS_DIRECTORY "${CANN_3RD_LIB_PATH}/eigen-5.0.0")
    message(STATUS "Eigen path found in cache: ${CANN_3RD_LIB_PATH}/eigen-5.0.0")
    set(REQ_URL "${CANN_3RD_LIB_PATH}/eigen-5.0.0")
elseif (EXISTS "${EIGEN_DST_DIR}" AND IS_DIRECTORY "${CANN_3RD_LIB_PATH}/eigen" AND EIGEN_CONTENT)
    message(STATUS "Eigen path found in cache: ${CANN_3RD_LIB_PATH}/eigen")
    message(STATUS "Eigen already exists at ${EIGEN_DST_DIR}")
    # 当Eigen目录已存在时，创建一个空的自定义目标
    add_custom_target(external_eigen_transformer)
else()
    message("The eigen package needs to be downloaded.")
    set(REQ_URL "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz")
    set(EIGEN_ARCHIVE ${CANN_3RD_LIB_PATH}/pkg/eigen-5.0.0.tar.gz)
    file(MAKE_DIRECTORY ${CANN_3RD_LIB_PATH}/pkg)

    # Search in CANN_3RD_LIB_PATH and move to pkg if found
    if(EXISTS ${CANN_3RD_LIB_PATH}/eigen-5.0.0.tar.gz AND NOT EXISTS ${EIGEN_ARCHIVE})
        message(STATUS "Found eigen archive in ${CANN_3RD_LIB_PATH}, moving to pkg")
        file(RENAME ${CANN_3RD_LIB_PATH}/eigen-5.0.0.tar.gz ${EIGEN_ARCHIVE})
    endif()

    if(EXISTS ${EIGEN_ARCHIVE})
        message(STATUS "Found eigen archive at ${EIGEN_ARCHIVE}")
        set(EIGEN_URL "file://${EIGEN_ARCHIVE}")
    else()
        message(STATUS "Downloading ${EIGEN_NAME} from ${REQ_URL}")
        set(EIGEN_URL ${REQ_URL})
    endif()

    include(ExternalProject)
    ExternalProject_Add(external_eigen_transformer
            URL               ${EIGEN_URL}
            DOWNLOAD_DIR      "${CANN_3RD_LIB_PATH}/download/eigen"
            PREFIX            "${CANN_3RD_LIB_PATH}/third_party/eigen"
            SOURCE_DIR        "${EIGEN_DST_DIR}"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND     ""
            INSTALL_COMMAND   ""
            UPDATE_COMMAND    ""
    )
endif()

add_library(EigenTransformer INTERFACE)
target_compile_options(EigenTransformer INTERFACE -w)

target_include_directories(EigenTransformer INTERFACE "${EIGEN_DST_DIR}")

# 添加依赖关系（如果目标存在的话）
if(TARGET external_eigen_transformer)
    add_dependencies(EigenTransformer external_eigen_transformer)
endif()

add_library(Eigen3::EigenTransformer ALIAS EigenTransformer)