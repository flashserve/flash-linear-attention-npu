# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set_package(ops-transformer VERSION "9.0.0")

set_build_dependencies(runtime ">=8.5")
set_build_dependencies(opbase ">=8.5")
set_build_dependencies(hcomm ">=8.5")
set_build_dependencies(ge-executor ">=8.5")
set_build_dependencies(metadef ">=8.5")
set_build_dependencies(ge-compiler ">=8.5")
set_build_dependencies(asc-devkit ">=8.5")
set_build_dependencies(bisheng-compiler ">=8.5")
set_build_dependencies(asc-tools ">=8.5")

set_run_dependencies(runtime ">=8.5")
set_run_dependencies(opbase ">=8.5")
set_run_dependencies(hcomm ">=8.5")
set_run_dependencies(ge-executor ">=8.5")
set_run_dependencies(metadef ">=8.5")
set_run_dependencies(ge-compiler ">=8.5")
set_run_dependencies(bisheng-compiler ">=8.5")
set_run_dependencies(asc-tools ">=8.5")
set_run_dependencies(bisheng-compiler ">=8.5")
set_run_dependencies(ops-nn ">=8.5")
set_run_dependencies(hccl ">=8.5")