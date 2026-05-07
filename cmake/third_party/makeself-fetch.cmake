# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(MAKESELF_NAME "makeself")
set(MAKESELF_PATH ${CANN_3RD_LIB_PATH}/makeself)
# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    set(MAKESELF_DOWNLOAD_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")
    set(MAKESELF_ARCHIVE ${CANN_3RD_LIB_PATH}/pkg/makeself-release-2.5.0-patch1.tar.gz)
 	file(MAKE_DIRECTORY ${CANN_3RD_LIB_PATH}/pkg)

    # Search in CANN_3RD_LIB_PATH and move to pkg if found
    if(EXISTS ${CANN_3RD_LIB_PATH}/makeself-release-2.5.0-patch1.tar.gz AND NOT EXISTS ${MAKESELF_ARCHIVE})
        message(STATUS "Found makeself archive in ${CANN_3RD_LIB_PATH}, moving to pkg")
        file(RENAME ${CANN_3RD_LIB_PATH}/makeself-release-2.5.0-patch1.tar.gz ${MAKESELF_ARCHIVE})
    endif()
    if(EXISTS ${MAKESELF_ARCHIVE})
        message(STATUS "Found makeself archive at ${MAKESELF_ARCHIVE}")
        set(MAKESELF_URL "file://${MAKESELF_ARCHIVE}")
    else()
        message(STATUS "Downloading ${MAKESELF_NAME} from ${MAKESELF_DOWNLOAD_URL}")
        set(MAKESELF_URL ${MAKESELF_DOWNLOAD_URL})
    endif()
    include(FetchContent)
    FetchContent_Declare(
        ${MAKESELF_NAME}
        URL ${MAKESELF_URL}
        URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
        SOURCE_DIR "${MAKESELF_PATH}"  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${MAKESELF_NAME})
    execute_process(
        COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself.sh"
        COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself-header.sh"
        RESULT_VARIABLE CHMOD_RESULT
        ERROR_VARIABLE CHMOD_ERROR
    )
else()
    execute_process(
        COMMAND cp -fr ${MAKESELF_PATH} ${CMAKE_BINARY_DIR}
        COMMAND cp -fr ${MAKESELF_PATH} ${CMAKE_SOURCE_DIR}/third_party
        COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself.sh"
        COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself-header.sh"
        RESULT_VARIABLE CHMOD_RESULT
        ERROR_VARIABLE CHMOD_ERROR
        )
endif()