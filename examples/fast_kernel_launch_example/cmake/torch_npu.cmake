# ----------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# find torch_npu
set(TORCH_NPU_INCLUDE_PATH "${TORCH_NPU_PATH}/include")
set(TORCH_NPU_LIB_PATH "${TORCH_NPU_PATH}/lib")
message(STATUS "Using Torch NPU path: ${TORCH_NPU_PATH}")
message(STATUS "Torch NPU include path: ${TORCH_NPU_INCLUDE_PATH}")
message(STATUS "Torch NPU lib path: ${TORCH_NPU_LIB_PATH}")
