# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# Licensed under the BSD 3-Clause License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        merge_fwd_bwd_kernel_def.cpp
    )
endif()

add_modules_sources(OPTYPE merge_fwd_bwd_kernel ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME MergeFwdBwdKernel
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME MergeFwdBwdKernel
        --cce-auto-sync=off
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
