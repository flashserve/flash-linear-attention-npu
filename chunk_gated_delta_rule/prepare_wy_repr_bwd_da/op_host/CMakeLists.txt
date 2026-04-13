# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        prepare_wy_repr_bwd_da_def.cpp
    )
endif()


add_modules_sources(OPTYPE prepare_wy_repr_bwd_da ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME PrepareWyReprBwdDa
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)