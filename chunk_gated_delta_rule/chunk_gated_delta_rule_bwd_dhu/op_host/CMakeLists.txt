# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        chunk_gated_delta_rule_bwd_dhu_def.cpp
    )
    add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleBwdDhu
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910_95")
        add_ops_compile_options(
            OP_NAME ChunkGatedDeltaRuleBwdDhu
            COMPUTE_UNIT ascend910_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    endif()
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend910_95")
    set(gdr_depends chunk_gated_delta_rule/common CACHE INTERNAL "Dependencies for gdr")
    add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleBwdDhu
        COMPUTE_UNIT ascend910_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_bwd_dhu ACLNNTYPE aclnn)

