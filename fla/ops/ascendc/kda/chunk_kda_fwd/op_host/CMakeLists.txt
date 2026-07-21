# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
set(chunk_kda_fwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h;fla/ops/ascendc/kda/kda_layout_swap12;fla/ops/ascendc/kda/chunk_kda_fwd_prepare;fla/ops/ascendc/kda/chunk_kda_fwd_post_wu;fla/ops/ascendc/kda/chunk_kda_fwd_finalize"
    CACHE STRING "Kernel source dependencies for chunk_kda_fwd" FORCE)
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd ACLNNTYPE aclnn_exclude)
