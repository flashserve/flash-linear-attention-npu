# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_Target ${UTest_NamePrefix}_Stubs)

add_library(${_Target}_Wno INTERFACE)
target_compile_options(${_Target}_Wno
        INTERFACE
            -Wno-unused-parameter
            -Werror
)

set(_Sources
        acl/aclrt_stubs.cpp
        platform/platform_stubs.cpp
        runtime/runtime_stubs.cpp
)
add_library(${_Target} SHARED)
target_sources(${_Target} PRIVATE ${_Sources})

target_include_directories(${_Target} PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
)

target_link_libraries(${_Target}
        PRIVATE
            $<BUILD_INTERFACE:intf_pub_utest>
            $<BUILD_INTERFACE:${_Target}_Wno>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
            ${UTest_NamePrefix}_Utils
)
