# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/solve_tril_def.cpp
    )
endif()

add_modules_sources(OPTYPE solve_tril ACLNNTYPE aclnn_exclude)

# MBH 调试模式：屏蔽 MCH，MCH 输出/全 0 矩阵/单位矩阵 作为接口入参，单独调试 MBH 模块。
# 正式构建时移除 -DSOLVE_TRIL_MBH_DEBUG_ONLY=1。
#
# 【arch3510 UB 优化（AIC+AIV 协作，单个共享 Arch::Resource）】MBH 递归用 UB 中转：
#   - AIC：Mmad + Fixpipe(L0C->UB) 暂存递归结果；
#   - AIV：MCH 输出 GM->UB(nd2nz) 暂存、每层 raw UB->L1 提取（UB 在 AIV，搬运在 AIV）。
#   - L1/UB 跨核共享：两核用同一 Catlass::Arch::Resource<Ascend950>（solve_tril.cpp 构造、
#     GetBufferByByte 相同偏移），其 ctor 经 pipe.Destroy() 去除 TPipe 隐式同步，避免 setup 死锁。
#   开关 SOLVE_TRIL_MBH_UB_OPT 由 __NPU_ARCH__==3510 自动决定：ascend950 build 启用 UB 通路，
#   ascend910b build(==2201) 为 0 走已验证 11/11 的 GM 通路（私有 TPipe）。
#   ENABLE_CV_COMM_VIA_SSBUF=1：UB<->L1 DataCopy 走 SSBuffer 硬件通道（950）；910b 上无副作用。
add_ops_compile_options(OP_NAME SolveTril OPTIONS
    --cce-auto-sync=off
    -DSOLVE_TRIL_MBH_DEBUG_ONLY=1
    -DENABLE_CV_COMM_VIA_SSBUF=1
)
