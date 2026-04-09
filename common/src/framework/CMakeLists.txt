# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Tianjin University, Ltd.
# This file is a part of the CANN Open Software.
# Licensed under the BSD 3-Clause License (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT AND NOT BUILD_OPS_RTY_KERNEL)
    add_onnx_plugin_sources()
endif()