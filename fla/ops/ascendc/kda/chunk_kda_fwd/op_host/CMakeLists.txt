# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
# 融合实现（aclnnChunkKdaFwd）需要 chunk_gated_delta_rule_fwd_h / kda_gate_cumsum
# 的 kernel 源码；V2 组合入口（aclnnChunkKdaFwdV2 -> KdaFwdThreeStage）在
# chunk_kda_fwd_three_stage.cpp 里直接 include 并调用这三个独立算子的 op_api，
# 因此它们也必须随 chunk_kda_fwd 一起编译，否则单独
# `FLA_NPU_OPS=chunk_kda_fwd` 会产出 libcust_opapi.so 缺符号的坏包。
set(chunk_kda_fwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h;fla/ops/ascendc/kda/kda_gate_cumsum;fla/ops/ascendc/kda/chunk_kda_fwd_prepare;fla/ops/ascendc/kda/chunk_kda_fwd_finalize;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h"
    CACHE STRING "Kernel source dependencies for chunk_kda_fwd" FORCE)
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKdaFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaFwd
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()
