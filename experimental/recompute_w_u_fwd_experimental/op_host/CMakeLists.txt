add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE recompute_wu_fwd_experimental_def.cpp)
endif()

# CANN collapses consecutive uppercase letters in the generated proto basename.
add_modules_sources(OPTYPE recompute_wu_fwd_experimental ACLNNTYPE aclnn_exclude)
