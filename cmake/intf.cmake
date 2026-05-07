# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    include(${OPS_ADV_CMAKE_DIR}/intf_pub.cmake)

    if (TESTS_UT_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/intf_pub_utest.cmake)
    endif ()

    if (TESTS_EXAMPLE_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/intf_pub_examples.cmake)
    endif ()
endif ()
