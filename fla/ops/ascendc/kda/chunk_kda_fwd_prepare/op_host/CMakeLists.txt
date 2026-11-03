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
# Register opbase's predefined messages before L2 validation can reject a call.
# The tiling DSO is loaded later, so its registration alone cannot serve L2.
target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
    ${OPBASE_SOURCE_PATH}/src/op_common/log/log.cpp
)
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
