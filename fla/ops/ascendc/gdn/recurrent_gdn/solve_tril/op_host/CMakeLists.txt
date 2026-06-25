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
# 【临时诊断】-DSOLVE_TRIL_MBH_PASSTHROUGH=N（0=关闭，跑完整 MBH）：
#   =1：BT>16 跳过 RecursiveMerge，仅 X×I 写回（已验证：对角块==mch_out，正确）。
#   =2：提取对角驱动块后 SLOT_Y×I（已验证：输出全 0 → ExtractBlocksToSlot 的
#       L1->GM 直拷在本 arch 失效）。
# 已据诊断改用 GM-only 数据流（ExtractBlocksFromGM + SpillL0CToXGM），故关闭诊断。
#
# 【arch3510 UB 优化（AIC+AIV 协作）】MBH 递归改用 UB 中转：
#   - AIC：Mmad + Fixpipe(L0C->UB) 暂存递归结果；
#   - AIV：MCH 输出 GM->UB(nd2nz) 暂存、每层 raw UB->L1 提取（UB 在 AIV，搬运在 AIV）。
#   开关 SOLVE_TRIL_MBH_UB_OPT 由 __NPU_ARCH__==3510 自动决定：ascend950 build 启用 UB 通路，
#   ascend910b build(==2201) 为 0 走已验证 11/11 的 GM 通路（Fixpipe L0C->UB 是 dav_3510 专有，
#   c220 上不存在，故不能在 910b build 上强开）。
#   ENABLE_CV_COMM_VIA_SSBUF=1：UB<->L1 DataCopy 走 SSBuffer 硬件通道（950）；910b 上无副作用。
add_ops_compile_options(OP_NAME SolveTril OPTIONS
    --cce-auto-sync=off
    -DSOLVE_TRIL_MBH_DEBUG_ONLY=1
    -DSOLVE_TRIL_MBH_PASSTHROUGH=0
    -DENABLE_CV_COMM_VIA_SSBUF=1
)
