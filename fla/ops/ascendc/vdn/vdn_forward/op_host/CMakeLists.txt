# The Python composite has no device kernel of its own. In a filtered build
# include the definitions, tiling libraries, kernels and ACLNN of its stages.
add_op_to_compiled_list()
if(ASCEND_OP_NAME AND NOT "all" IN_LIST ASCEND_OP_NAME AND NOT "ALL" IN_LIST ASCEND_OP_NAME)
    foreach(stage vdn_scan vdn_bridge vdn_add_diagonal)
        if(NOT "${stage}" IN_LIST ASCEND_OP_NAME)
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../${stage}/op_host"
                             "${CMAKE_BINARY_DIR}/vdn_forward_deps/${stage}")
        endif()
    endforeach()
endif()
