# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------
file(GLOB SUB_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${SUB_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt)
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
