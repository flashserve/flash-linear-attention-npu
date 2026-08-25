add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        fused_recurrent_rwkv8_def.cpp
    )
endif()

add_modules_sources(OPTYPE fused_recurrent_rwkv8 ACLNNTYPE aclnn_exclude)

# 框架的 op_add_ut_subdirectory 用 `**/**/tests/CMakeLists.txt`（仅两级深度）glob，
# 够不到 fla/ops/ascendc/ 下的深层算子目录；这里在 UT 构建时自行挂载 tests。
# 时点早于 tests/ut/framework_normal 的 empty.cpp 兜底，保证 *_cases_obj 编入真实用例。
if (ENABLE_TEST AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests ${CMAKE_CURRENT_BINARY_DIR}/tests)
endif()
