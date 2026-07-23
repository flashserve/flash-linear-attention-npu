# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

if(NOT FLA_NPU_OFFLINE_BUILD)
    return()
endif()

if(NOT IS_DIRECTORY "${CANN_3RD_LIB_PATH}")
    message(FATAL_ERROR
            "[OfflineBuild] Third-party cache directory does not exist: ${CANN_3RD_LIB_PATH}")
endif()

function(_fla_npu_verify_archive filename expected_sha256)
    set(archive_path "${CANN_3RD_LIB_PATH}/pkg/${filename}")
    if(NOT EXISTS "${archive_path}")
        message(FATAL_ERROR "[OfflineBuild] Missing archive: ${archive_path}")
    endif()
    file(SHA256 "${archive_path}" actual_sha256)
    if(NOT actual_sha256 STREQUAL expected_sha256)
        message(FATAL_ERROR
                "[OfflineBuild] Checksum mismatch for ${archive_path}: "
                "expected ${expected_sha256}, got ${actual_sha256}")
    endif()
endfunction()

function(_fla_npu_verify_repository name expected_revision required_file)
    set(repository_path "${CANN_3RD_LIB_PATH}/${name}")
    set(revision_path "${repository_path}/.fla_npu_revision")
    if(NOT EXISTS "${revision_path}")
        message(FATAL_ERROR "[OfflineBuild] Missing revision marker: ${revision_path}")
    endif()
    file(READ "${revision_path}" actual_revision)
    string(STRIP "${actual_revision}" actual_revision)
    if(NOT actual_revision STREQUAL expected_revision)
        message(FATAL_ERROR
                "[OfflineBuild] ${name} revision mismatch: expected ${expected_revision}, "
                "got ${actual_revision}")
    endif()
    if(NOT EXISTS "${repository_path}/${required_file}")
        message(FATAL_ERROR
                "[OfflineBuild] Missing required file: ${repository_path}/${required_file}")
    endif()
endfunction()

_fla_npu_verify_archive(
    "include.zip"
    "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d")
_fla_npu_verify_archive(
    "makeself-release-2.5.0-patch1.tar.gz"
    "bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a")
_fla_npu_verify_archive(
    "eigen-5.0.0.tar.gz"
    "93f7f0462988b934e632a9fba58af55192ffceae38e8f46233f4f62cb1e79370")
_fla_npu_verify_archive(
    "protobuf-25.1.tar.gz"
    "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42")
_fla_npu_verify_archive(
    "abseil-cpp-20230802.1.tar.gz"
    "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed")

_fla_npu_verify_repository(
    "opbase"
    "c8d83f3e57a63a7375e89a2d6937452c0ae2e522"
    "CMakeLists.txt")
_fla_npu_verify_repository(
    "catlass"
    "41bf90da655bba3c66d0acd7e00abe33960ecfd6"
    "include/catlass/catlass.hpp")

# Local archives still need to be extracted, so FULLY_DISCONNECTED cannot be
# enabled. Every network fallback is made unreachable by the checks above.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
message(STATUS "[OfflineBuild] Third-party cache verified: ${CANN_3RD_LIB_PATH}")
