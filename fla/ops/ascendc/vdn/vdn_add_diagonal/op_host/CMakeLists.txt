# SPDX-License-Identifier: BSD-3-Clause
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE vdn_add_diagonal_def.cpp)
endif()
add_modules_sources(OPTYPE vdn_add_diagonal ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME VdnAddDiagonal
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)
