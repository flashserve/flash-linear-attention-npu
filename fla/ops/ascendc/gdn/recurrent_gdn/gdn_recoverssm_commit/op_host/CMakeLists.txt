# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        gdn_recoverssm_commit_def.cpp
    )
endif()

add_modules_sources(OPTYPE gdn_recoverssm_commit ACLNNTYPE aclnn_inner)
add_ops_compile_options(
    OP_NAME GdnRecoverssmCommit
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME GdnRecoverssmCommit
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
