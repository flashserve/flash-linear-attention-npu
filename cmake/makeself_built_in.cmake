# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# makeself.cmake - 自定义 makeself 打包脚本

message(STATUS "CPACK_CMAKE_SOURCE_DIR = ${CPACK_CMAKE_SOURCE_DIR}")
message(STATUS "CPACK_CMAKE_CURRENT_SOURCE_DIR = ${CPACK_CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CPACK_CMAKE_BINARY_DIR = ${CPACK_CMAKE_BINARY_DIR}")
message(STATUS "CPACK_PACKAGE_FILE_NAME = ${CPACK_PACKAGE_FILE_NAME}")
message(STATUS "CPACK_CMAKE_INSTALL_PREFIX = ${CPACK_CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_COMMAND = ${CMAKE_COMMAND}")
message(STATUS "CPACK_TEMPORARY_DIRECTORY = ${CPACK_TEMPORARY_DIRECTORY}")
message(STATUS "CPACK_VERSION_DEST = ${CPACK_VERSION_DEST}")
message(STATUS "CPACK_VERSION_SRC = ${CPACK_VERSION_SRC}")
message(STATUS "CPACK_EXTRA_VERSION_FILES = ${CPACK_EXTRA_VERSION_FILES}")

# 设置 makeself 路径
set(MAKESELF_EXE ${CPACK_MAKESELF_PATH}/makeself.sh)
set(MAKESELF_HEADER_EXE ${CPACK_MAKESELF_PATH}/makeself-header.sh)
if(NOT MAKESELF_EXE)
    message(FATAL_ERROR "makeself not found!")
endif()

# 创建临时安装目录
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")
file(MAKE_DIRECTORY "${STAGING_DIR}")

# 执行安装到临时目录
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
    RESULT_VARIABLE INSTALL_RESULT
)

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation to staging directory failed: ${INSTALL_RESULT}")
endif()

if(CPACK_REMOVE_LIB_FILES)
    message("Remove files from ${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging/lib")
    file(GLOB ALL_FILES "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging/lib/*")

    file(REMOVE_RECURSE "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging/lib")
endif()

# 生成安装配置文件
set(CSV_OUTPUT ${CPACK_CMAKE_BINARY_DIR}/filelist.csv)
execute_process(
    COMMAND python3 ${CPACK_CMAKE_SOURCE_DIR}/scripts/package/package.py --pkg_name ${CPACK_PACKAGE_PARAM_NAME} --chip_name ${CPACK_SOC} --os_arch linux-${CPACK_ARCH} --version_dir ${CPACK_VERSION} --delivery_dir ${CPACK_CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CPACK_CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE result
    ERROR_VARIABLE error
    RESULT_VARIABLE code
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "package.py result: ${code}")
if (NOT code EQUAL 0)
    message(FATAL_ERROR "Filelist generation failed: ${result}")
else ()
    message(STATUS "Filelist generated successfully: ${result}")

    if (NOT EXISTS ${CSV_OUTPUT})
        message(FATAL_ERROR "Output file not created: ${CSV_OUTPUT}")
    endif ()
endif ()

string(STRIP "${CPACK_PACKAGE_PARAM_NAME}" PKG_NAME)

# --- 1. 默认值处理 ---

# 默认 Scene 目录：share/info/<包名>
if(NOT DEFINED CPACK_SCENE_DEST)
    set(FINAL_SCENE_DEST "${STAGING_DIR}/share/info/${PKG_NAME}")
else()
    set(FINAL_SCENE_DEST "${CPACK_SCENE_DEST}")
endif()

# 默认 Script 目录：Scene目录下的 script
if(NOT DEFINED CPACK_SCRIPT_DEST)
    set(FINAL_SCRIPT_DEST "${FINAL_SCENE_DEST}/script")
else()
    set(FINAL_SCRIPT_DEST "${CPACK_SCRIPT_DEST}")
endif()

# 默认 Version 源文件：<包名>_version.h
if(NOT DEFINED CPACK_VERSION_SRC)
    set(FINAL_VERSION_SRC "${CPACK_CMAKE_BINARY_DIR}/${PKG_NAME}_version.h")
else()
    set(FINAL_VERSION_SRC "${CPACK_VERSION_SRC}")
endif()

# 默认 Version 目标是不是具体文件名：FALSE
if(NOT DEFINED CPACK_VERSION_DEST)
    set(FINAL_VERSION_DEST "${FINAL_SCENE_DEST}")
else()
    set(FINAL_VERSION_DEST "${CPACK_VERSION_DEST}")
endif()

# 默认 Version 目标是不是具体文件名：FALSE
if(NOT DEFINED CPACK_VERSION_IS_FILE)
    set(FINAL_VERSION_IS_FILE FALSE)
else()
    set(FINAL_VERSION_IS_FILE "${CPACK_VERSION_IS_FILE}")
endif()

# --- 2. 执行拷贝动作（只做事，不判断包名）---

# A. 拷贝 Scene info
if(EXISTS "${CPACK_CMAKE_BINARY_DIR}/scene.info")
    configure_file("${CPACK_CMAKE_BINARY_DIR}/scene.info" "${FINAL_SCENE_DEST}/" COPYONLY)
endif()

# B. 拷贝 Script
if(CSV_OUTPUT AND EXISTS "${CSV_OUTPUT}")
    configure_file("${CSV_OUTPUT}" "${FINAL_SCRIPT_DEST}/" COPYONLY)
endif()

# C. 拷贝 Version 文件
if(EXISTS "${FINAL_VERSION_SRC}")
    if(FINAL_VERSION_IS_FILE)
        configure_file("${FINAL_VERSION_SRC}" "${FINAL_VERSION_DEST}" COPYONLY)
    else()
        configure_file("${FINAL_VERSION_SRC}" "${FINAL_VERSION_DEST}/" COPYONLY)
    endif()
endif()

# D. 拷贝额外版本文件
if(DEFINED CPACK_EXTRA_VERSION_FILES)
    foreach(EXTRA_FILE ${CPACK_EXTRA_VERSION_FILES})
        if(EXISTS "${EXTRA_FILE}")
            configure_file("${EXTRA_FILE}" "${FINAL_VERSION_DEST}/" COPYONLY)
        endif()
    endforeach()
endif()

# makeself打包
file(STRINGS ${CPACK_CMAKE_BINARY_DIR}/makeself.txt script_output)
string(REPLACE " " ";" makeself_param_string "${script_output}")
string(REGEX MATCH "cann.*\\.run" package_name "${makeself_param_string}")

list(LENGTH makeself_param_string LIST_LENGTH)
math(EXPR INSERT_INDEX "${LIST_LENGTH} - 2")
list(INSERT makeself_param_string ${INSERT_INDEX} "${STAGING_DIR}")

message(STATUS "script output: ${script_output}")
message(STATUS "makeself: ${makeself_param_string}")
message(STATUS "package: ${package_name}")

if(NOT DEFINED CPACK_HELP_HEADER_PATH)
    set(CPACK_HELP_HEADER_PATH "share/info/${CPACK_PACKAGE_PARAM_NAME}/script/help.info")
endif()

if(NOT DEFINED CPACK_INSTALL_PATH)
    set(CPACK_INSTALL_PATH "share/info/${CPACK_PACKAGE_PARAM_NAME}/script/install.sh")
endif()


execute_process(COMMAND bash ${MAKESELF_EXE}
        --header ${MAKESELF_HEADER_EXE}
        --help-header ${CPACK_HELP_HEADER_PATH}
        ${makeself_param_string} ${CPACK_INSTALL_PATH}
        WORKING_DIRECTORY ${STAGING_DIR}
        RESULT_VARIABLE EXEC_RESULT
        ERROR_VARIABLE  EXEC_ERROR
)

if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself packaging failed: ${EXEC_ERROR}")
endif()

if(NOT DEFINED CPACK_BUILD_MODE)
    set(CPACK_BUILD_MODE "DIR_MOVE")
endif()

if(CPACK_BUILD_MODE STREQUAL "RUN_COPY")
    execute_process(
        COMMAND find ${STAGING_DIR} -name "cann-*.run"
        COMMAND xargs cp --target-directory=${CPACK_CMAKE_INSTALL_PREFIX}
        WORKING_DIRECTORY ${STAGING_DIR}
        RESULT_VARIABLE EXEC_RESULT
        ERROR_VARIABLE  EXEC_ERROR
    )
    if(NOT "${EXEC_RESULT}" STREQUAL "0")
        message(FATAL_ERROR "Failed to copy run files: ${EXEC_ERROR}")
    endif()
elseif(CPACK_BUILD_MODE STREQUAL "DIR_MOVE")
    execute_process(
        COMMAND mkdir -p ${CPACK_PACKAGE_DIRECTORY}
        COMMAND mv ${STAGING_DIR}/${package_name} ${CPACK_PACKAGE_DIRECTORY}/
        COMMAND echo "build pkg success: ${CPACK_PACKAGE_DIRECTORY}/${package_name}"
        WORKING_DIRECTORY ${STAGING_DIR}
    )
endif()
