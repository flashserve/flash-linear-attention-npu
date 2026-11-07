# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_prepare_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd_prepare ACLNNTYPE aclnn_exclude)
# SDK headers can expose newer macros than the selected opbase catalog.
# Derive per-code capabilities from the exact source registered by this operator.
set(PREPARE_OPBASE_LOG_CPP "${OPBASE_SOURCE_PATH}/src/op_common/log/log.cpp")
file(READ "${PREPARE_OPBASE_LOG_CPP}" PREPARE_DFX_CATALOG_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PREPARE_OPBASE_LOG_CPP}")
foreach(PREPARE_DFX_CODE EZ0008 EZ0009 EZ0010 EZ0013 EZ0017 EZ0018 EZ0019
        EZ0021 EZ0024 EZ0025 EZ0026 EZ0027 EZ0037)
    string(REGEX MATCH "\"ErrCode\"[ \t\r\n]*:[ \t\r\n]*\"${PREPARE_DFX_CODE}\""
        PREPARE_DFX_MATCH "${PREPARE_DFX_CATALOG_SOURCE}")
    if(PREPARE_DFX_MATCH)
        set(PREPARE_DFX_HAS_${PREPARE_DFX_CODE} true)
    else()
        set(PREPARE_DFX_HAS_${PREPARE_DFX_CODE} false)
    endif()
endforeach()
set(PREPARE_DFX_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}/prepare_error_catalog")
configure_file(chunk_kda_fwd_prepare_error_catalog.h.in
    "${PREPARE_DFX_CONFIG_DIR}/chunk_kda_fwd_prepare_error_catalog.h" @ONLY)
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_chunk_kda_fwd_prepare.cpp"
    TARGET_DIRECTORY ${OPHOST_NAME}_opapi_obj
    PROPERTIES INCLUDE_DIRECTORIES "${PREPARE_DFX_CONFIG_DIR}")
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/chunk_kda_fwd_prepare_infershape.cpp"
    TARGET_DIRECTORY ${OPHOST_NAME}_infer_obj
    PROPERTIES INCLUDE_DIRECTORIES "${PREPARE_DFX_CONFIG_DIR}")
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/chunk_kda_fwd_prepare_tiling.cpp"
    TARGET_DIRECTORY ${OPHOST_NAME}_tiling_obj
    PROPERTIES INCLUDE_DIRECTORIES "${PREPARE_DFX_CONFIG_DIR}")
# Tiling is loaded later; register the catalog before the first L2 validation.
target_sources(${OPHOST_NAME}_opapi_obj PRIVATE "${PREPARE_OPBASE_LOG_CPP}")
unset(PREPARE_DFX_CATALOG_SOURCE)
unset(PREPARE_DFX_MATCH)
add_ops_compile_options(
    OP_NAME ChunkKdaFwdPrepare
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

# The template extractor receives this option as well as the device compiler.
# Keep A5 on its existing declaration and implementation.
# This repository's options parser expects concrete SoC names and normalizes
# them to the respective ascend910b / ascend910_93 family.
foreach(PREPARE_RUNTIME_SOC ascend910b3 ascend910_9391)
    add_ops_compile_options(
        OP_NAME ChunkKdaFwdPrepare
        COMPUTE_UNIT ${PREPARE_RUNTIME_SOC}
        OPTIONS -DCHUNK_KDA_PREPARE_RUNTIME_KEYS=1
    )
endforeach()
