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
        pre_process_fwd_kernel_merged_def.cpp
    )
endif()

add_modules_sources(OPTYPE pre_process_fwd_kernel_merged ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME PreProcessFwdKernelMerged
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME PreProcessFwdKernelMerged
        --cce-auto-sync=off
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
# 诊断开关（默认关闭）：需要源码行号 / code hotspot 时
#   PPFM_KERNEL_G=1 bash build.sh --pkg --soc=ascend950 ...
# 只影响 kernel 的调试信息，不改变算子语义。
if(DEFINED ENV{PPFM_KERNEL_G})
    add_ops_compile_options(
        OP_NAME PreProcessFwdKernelMerged
        OPTIONS -g
    )
endif()

