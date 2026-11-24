# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_fwd_h_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_fwd_h ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRuleFwdH
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

# arch22（910b/910_93）保持运行时 key 入口；3510/9201 走 TPL 模板化 key
foreach(FWD_H_RUNTIME_SOC ascend910b3 ascend910_9391)
    add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleFwdH
        COMPUTE_UNIT ${FWD_H_RUNTIME_SOC}
        OPTIONS -DCHUNK_GDR_FWD_H_RUNTIME_KEYS=1
    )
endforeach()

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend960dt")
    add_ops_compile_options(
        OP_NAME ChunkGatedDeltaRuleFwdH
        COMPUTE_UNIT Ascend960DT_968B7
        OPTIONS --cce-no-dcache-flush
    )
endif()
