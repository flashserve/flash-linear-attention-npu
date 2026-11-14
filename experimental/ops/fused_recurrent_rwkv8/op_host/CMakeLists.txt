add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        fused_recurrent_rwkv8_def.cpp
    )
endif()

add_modules_sources(OPTYPE fused_recurrent_rwkv8 ACLNNTYPE aclnn_exclude)
