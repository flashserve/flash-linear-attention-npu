# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        recompute_wu_fwd_def.cpp
    )
endif()

# CANN host/proto generation collapses the consecutive uppercase letters in
# RecomputeWUFwd. Keep this internal def/proto basename while the operator
# op_file, runtime config and public APIs use recompute_w_u_fwd.
add_modules_sources(OPTYPE recompute_wu_fwd ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME RecomputeWUFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME RecomputeWUFwd
        --cce-auto-sync=off
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()

# arch22（910b/910_93）保持运行时 key 入口；3510/9201 走 TPL 模板化 key
foreach(RECOMPUTE_RUNTIME_SOC ascend910b3 ascend910_9391)
    add_ops_compile_options(
        OP_NAME RecomputeWUFwd
        COMPUTE_UNIT ${RECOMPUTE_RUNTIME_SOC}
        OPTIONS -DRECOMPUTE_W_U_FWD_RUNTIME_KEYS=1
    )
endforeach()

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend960dt")
    add_ops_compile_options(
        OP_NAME RecomputeWUFwd
        COMPUTE_UNIT Ascend960DT_968B7
        OPTIONS --cce-no-dcache-flush
    )
endif()
