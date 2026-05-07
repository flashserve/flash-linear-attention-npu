#!/bin/bash\n"
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


ascend_install_dir=$1
gen_file_dir=$2

# create version.info
compiler_version=$(grep "Version" -w ${ascend_install_dir}/compiler/version.info | awk -F = '{print $2}')
echo "custom_opp_compiler_version=${compiler_version}" > ${gen_file_dir}/version.info