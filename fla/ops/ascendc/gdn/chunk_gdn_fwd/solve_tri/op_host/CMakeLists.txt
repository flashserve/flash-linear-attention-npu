add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/solve_tri_def.cpp
    )
    target_compile_definitions(op_host_aclnnExc PRIVATE
        ASCEND_SOC_VERSION="${ASCEND_COMPUTE_UNIT}"
    )
endif()

add_modules_sources(OPTYPE solve_tri ACLNNTYPE aclnn_exclude)

add_ops_compile_options(OP_NAME SolveTri OPTIONS
    --cce-auto-sync=off
)

# arch22（910b/910_93）保持运行时 key 入口；3510/9201 走 TPL 模板化 key
foreach(SOLVE_TRI_RUNTIME_SOC ascend910b3 ascend910_9391)
    add_ops_compile_options(
        OP_NAME SolveTri
        COMPUTE_UNIT ${SOLVE_TRI_RUNTIME_SOC}
        OPTIONS -DSOLVE_TRI_RUNTIME_KEYS=1
    )
endforeach()

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend960dt")
    add_ops_compile_options(
        OP_NAME SolveTri
        COMPUTE_UNIT Ascend960DT_968B7
        OPTIONS --cce-no-dcache-flush
    )
endif()
