# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(filter_copy_files SELECTED_FILES SELECTED_DIRS)
    set(_selected_files "")
    set(_selected_dirs "")
    foreach(item ${KERNEL_SUB_DIRS})
        set(path "${CURRENT_KERNEL_DIR}/${item}")
        if(IS_DIRECTORY "${path}")
            if(item MATCHES "^arch")
                list(FIND ARCH_DIRECTORY "${item}" idx)
                if(idx EQUAL -1)
                    continue()
                endif()
            endif()
            list(APPEND _selected_dirs "${path}")
        else()
            list(APPEND _selected_files "${path}")
        endif()
    endforeach()

    set(${SELECTED_FILES} "${_selected_files}" PARENT_SCOPE)
    set(${SELECTED_DIRS} "${_selected_dirs}" PARENT_SCOPE)
endfunction()

function(add_target_source)
    cmake_parse_arguments(ADD "" "BASE_TARGET;SRC_DIR" "TARGET_NAME" ${ARGN})

    get_target_property(all_srcs ${ADD_BASE_TARGET} SOURCES)
    set(add_srcs)
    foreach(_src ${all_srcs})
        string(REGEX MATCH "^${ADD_SRC_DIR}" is_match "${_src}")
        if (is_match)
            list(APPEND add_srcs ${_src})
        endif ()
    endforeach()

    get_target_property(all_includes ${ADD_BASE_TARGET} INCLUDE_DIRECTORIES)
    set(add_includes)
    foreach(_include ${all_includes})
        string(REGEX MATCH "^${ADD_SRC_DIR}" is_match "${_include}")
        if (is_match)
            list(APPEND add_includes ${_include})
        endif ()
    endforeach()

    foreach(_target_name ${ADD_TARGET_NAME})
        target_sources(${_target_name} PRIVATE
                ${add_srcs}
        )

        target_include_directories(${_target_name} PRIVATE
                ${add_includes}
        )
    endforeach()
endfunction()

function(op_add_subdirectory OP_LIST OP_DIR_LIST)
    set(_OP_LIST)
    set(_OP_DIR_LIST)

    if(ENABLE_EXPERIMENTAL)
        message(STATUS "Build experimental module")
        file(GLOB OP_HOST_CMAKE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/experimental/ffn/**/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/experimental/chunk_gated_delta_rule/**/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/experimental/gmm/**/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/experimental/moe/**/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/experimental/posembedding/**/op_host/CMakeLists.txt"
        )
    else()
        file(GLOB OP_HOST_CMAKE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/chunk_gated_delta_rule/**/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/chunk_gated_delta_rule/**/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/gmm/**/op_host/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/gmm/**/CMakeLists.txt"
        )
    endif()

    foreach(OP_CMAKE_FILE ${OP_HOST_CMAKE_FILES})
        if ("${OP_CMAKE_FILE}" MATCHES "op_host")
            get_filename_component(OP_HOST_DIR "${OP_CMAKE_FILE}" DIRECTORY)
            get_filename_component(OP_DIR "${OP_HOST_DIR}" DIRECTORY)
        else()
            get_filename_component(OP_DIR "${OP_CMAKE_FILE}" DIRECTORY)
        endif()
        get_filename_component(OP_NAME "${OP_DIR}" NAME)

        if (NOT BUILD_OPEN_PROJECT)
            if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${OP_NAME})
                continue()
            endif ()
        endif ()

        if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
            if (NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
                if (NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
                    continue()
                endif ()
            endif ()
        endif ()

        if (ENABLE_TEST)
            if (NOT EXISTS "${OP_DIR}/tests/CMakeLists.txt")
                continue()
            endif()
            
            file(READ "${OP_DIR}/tests/CMakeLists.txt" CML_CONTENT)
            if (CML_CONTENT MATCHES "OpsTest_Level2_AddOp")
                set(UTEST_FRAMEWORK_OLD TRUE CACHE BOOL "UTEST_FRAMEWORK_OLD" FORCE)
            else()
                set(UTEST_FRAMEWORK_NEW TRUE CACHE BOOL "UTEST_FRAMEWORK_NEW" FORCE)
            endif()
        endif()

        if (NOT ENABLE_AICPU)
            if(EXISTS "${OP_DIR}/op_kernel_aicpu" AND IS_DIRECTORY "${OP_DIR}/op_kernel_aicpu")
                MESSAGE(STATUS "disable aicpu kernel ${OP_NAME}, skip it.")
                continue()
            endif()
        endif()

        list(APPEND _OP_LIST ${OP_NAME})
        list(APPEND _OP_DIR_LIST ${OP_DIR})
    endforeach()

    list(REMOVE_DUPLICATES _OP_LIST)
    list(REMOVE_DUPLICATES _OP_DIR_LIST)
    list(SORT _OP_LIST)
    list(SORT _OP_DIR_LIST)
    set(${OP_LIST} ${_OP_LIST} PARENT_SCOPE)
    set(${OP_DIR_LIST} ${_OP_DIR_LIST} PARENT_SCOPE)
endfunction()

macro(add_op_to_compiled_list)
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${PARENT_DIR} NAME)
    # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如moe/moe_token_permute_with_routing_map_grad
    set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
    set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)
endmacro()


function(op_add_depend_directory)
    cmake_parse_arguments(DEP "" "OP_DIR_LIST" "OP_LIST" ${ARGN})
    set(_OP_DEPEND_DIR_LIST)
    foreach(op_name ${DEP_OP_LIST})
        if (DEFINED ${op_name}_depends)
            foreach(depend_info ${${op_name}_depends})
                if (ENABLE_EXPERIMENTAL)
 	                set(depend_info_update "experimental/${depend_info}")
 	            else()
 	                set(depend_info_update ${depend_info})
 	            endif()
 	            if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${depend_info_update}/op_host/CMakeLists.txt AND NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${depend_info_update}/CMakeLists.txt)
                    continue()
                endif ()

                get_filename_component(_depend_op_name "${depend_info}" NAME)
                if (NOT BUILD_OPEN_PROJECT)
                    if (EXISTS ${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_depend_op_name})
                        continue()
                    endif ()
                endif ()

                if (NOT ${_depend_op_name} IN_LIST DEP_OP_LIST)
                    list(APPEND _OP_DEPEND_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/${depend_info_update})
                endif ()
            endforeach()
        endif()
    endforeach()

    list(SORT _OP_DEPEND_DIR_LIST)
    list(REMOVE_DUPLICATES _OP_DEPEND_DIR_LIST)
    set(${DEP_OP_DIR_LIST} ${_OP_DEPEND_DIR_LIST} PARENT_SCOPE)
endfunction()

function(add_compile_cmd_target)
    cmake_parse_arguments(CMD "" "COMPUTE_UNIT" "" ${ARGN})

    if(ADD_OPS_COMPILE_OPTION_V2)
        set(OP_DEBUG_CONFIG_OPTION --opc-config-file ${ASCEND_CUSTOM_OPC_OPTIONS})
        set(OP_TILING_KEY_OPTION --kernel_template_input ${KERNEL_TEMPLATE_INPUT})
    else()
        if(OP_DEBUG_CONFIG)
            set(OP_DEBUG_CONFIG_OPTION --op-debug-config ${OP_DEBUG_CONFIG})
        endif()
        set(OP_TILING_KEY_OPTION --tiling-keys ${ASCEND_CUSTOM_TILING_KEYS})
    endif()

    set(_OUT_DIR           ${ASCEND_BINARY_OUT_DIR}/${CMD_COMPUTE_UNIT})
    set(GEN_OUT_DIR        ${_OUT_DIR}/gen)
    set(COMPILE_CMD_TARGET generate_compile_cmd_${CMD_COMPUTE_UNIT})

    set(SED_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/fix_format.sh)

    add_custom_target(${COMPILE_CMD_TARGET} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_OUT_DIR}
        COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_bin_param_build.py
            ${base_aclnn_binary_dir}/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR}
            ${CMD_COMPUTE_UNIT}
            ${BISHENG_FLAGS}
            ${OP_TILING_KEY_OPTION}
            ${OP_DEBUG_CONFIG_OPTION}
        COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_bin_param_build.py
            ${base_aclnn_binary_dir}/inner/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR}
            ${CMD_COMPUTE_UNIT}
            ${BISHENG_FLAGS}
            ${OP_TILING_KEY_OPTION}
            ${OP_DEBUG_CONFIG_OPTION}
        COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_bin_param_build.py
            ${base_aclnn_binary_dir}/exc/aic-${CMD_COMPUTE_UNIT}-ops-info.ini
            ${GEN_OUT_DIR}
            ${CMD_COMPUTE_UNIT}
            ${BISHENG_FLAGS}
            ${OP_TILING_KEY_OPTION}
            ${OP_DEBUG_CONFIG_OPTION}
        COMMAND bash ${SED_SCRIPT} ${GEN_OUT_DIR}
    )

    add_dependencies(${COMPILE_CMD_TARGET} opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)
    add_dependencies(generate_compile_cmd ${COMPILE_CMD_TARGET})
endfunction()

function(add_ops_info_target)
    cmake_parse_arguments(OPINFO "" "COMPUTE_UNIT" "" ${ARGN})

    set(OPS_INFO_TARGET generate_ops_info_${OPINFO_COMPUTE_UNIT})
    if (ENABLE_BUILT_IN)
        set(OPS_INFO_JSON ${ASCEND_AUTOGEN_DIR}/aic-${OPINFO_COMPUTE_UNIT}-ops-info-transformer.json)
    else()
        set(OPS_INFO_JSON ${ASCEND_AUTOGEN_DIR}/aic-${OPINFO_COMPUTE_UNIT}-ops-info.json)
    endif()
    set(CUSTOM_OPS_INFO_DIR ${CUSTOM_DIR}/op_impl/ai_core/tbe/config/${OPINFO_COMPUTE_UNIT})

    set(OPS_INFO_INI          ${base_aclnn_binary_dir}/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini)
    set(OPS_INFO_INNER_INI    ${base_aclnn_binary_dir}/inner/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini)
    set(OPS_INFO_EXCLUDE_INI  ${base_aclnn_binary_dir}/exc/aic-${OPINFO_COMPUTE_UNIT}-ops-info.ini)

    add_custom_command(OUTPUT ${OPS_INFO_JSON}
            COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/parse_ini_to_json.py
            ${OPS_INFO_INI}
            ${OPS_INFO_INNER_INI}
            ${OPS_INFO_EXCLUDE_INI}
            ${OPS_INFO_JSON}
            COMMAND mkdir -p ${CUSTOM_OPS_INFO_DIR}
            COMMAND cp -f ${OPS_INFO_JSON} ${CUSTOM_OPS_INFO_DIR}
    )

    add_custom_target(${OPS_INFO_TARGET} ALL
            DEPENDS ${OPS_INFO_JSON}
    )

    add_dependencies(${OPS_INFO_TARGET} opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)
    add_dependencies(generate_ops_info ${OPS_INFO_TARGET})

    if (ENABLE_BUILT_IN)
        install(FILES ${OPS_INFO_JSON}
                DESTINATION ops_transformer/built-in/op_impl/ai_core/tbe/config/${OPINFO_COMPUTE_UNIT} OPTIONAL
        )
    else()
        install(FILES ${OPS_INFO_JSON}
                DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/config/${OPINFO_COMPUTE_UNIT} OPTIONAL
        )
    endif()
endfunction()

function(add_ops_compile_options)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT;OPTIONS" ${ARGN})

    if(NOT OP_COMPILE_OPTIONS)
        return()
    endif()

    if(ADD_OPS_COMPILE_OPTION_V2)
        execute_process(COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_gen_options.py
                ${ASCEND_CUSTOM_OPTIONS} ${OP_COMPILE_OP_NAME}
                ${OP_COMPILE_COMPUTE_UNIT} ${OP_COMPILE_OPTIONS}
                RESULT_VARIABLE EXEC_RESULT
                OUTPUT_VARIABLE EXEC_INFO
                ERROR_VARIABLE EXEC_ERROR)
        if (EXEC_RESULT)
            message("add ops compile options info: ${EXEC_INFO}")
            message("add ops compile options error: ${EXEC_ERROR}")
            message(FATAL_ERROR "Error: add ops compile options failed!")
        endif ()
    else()
        file(APPEND ${ASCEND_CUSTOM_OPTIONS}
                "${OP_COMPILE_OP_NAME},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n"
        )
    endif()
endfunction()

function(add_ops_tiling_keys)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT;TILING_KEYS" ${ARGN})

    if(NOT OP_COMPILE_TILING_KEYS)
        return()
    endif()

    if(ADD_OPS_COMPILE_OPTION_V2)
        list(JOIN OP_COMPILE_TILING_KEYS "," STRING_TILING_KEYS)
        add_ops_compile_options(
                OP_NAME ${OP_COMPILE_OP_NAME}
                OPTIONS --tiling_key=${STRING_TILING_KEYS}
        )
    else()
        file(APPEND ${ASCEND_CUSTOM_TILING_KEYS}
                "${OP_COMPILE_OP_NAME},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_TILING_KEYS}\n"
        )
    endif()
endfunction()

function(add_opc_config)
    cmake_parse_arguments(OP_COMPILE "" "OP_NAME" "COMPUTE_UNIT;CONFIG" ${ARGN})

    if(NOT ADD_OPS_COMPILE_OPTION_V2)
        return()
    endif()

    set(_OPC_CONFIG)

    if(NOT OP_COMPILE_CONFIG)
        list(APPEND _OPC_CONFIG "-DNOT_DYNAMIC_COMPILE")
    else()
        string(REPLACE "," ";" OP_COMPILE_CONFIG_LIST "${OP_COMPILE_CONFIG}")
        list(APPEND _OPC_CONFIG "-DNOT_DYNAMIC_COMPILE")

        foreach(_option ${OP_COMPILE_CONFIG_LIST})
            if("${_option}" STREQUAL "ccec_g")
                list(APPEND _OPC_CONFIG "-g")
            elseif("${_option}" STREQUAL "ccec_O0")
                list(APPEND _OPC_CONFIG "-O0")
            elseif("${_option}" STREQUAL "sanitizer")
                list(APPEND _OPC_CONFIG "-sanitizer")
            elseif("${_option}" STREQUAL "dump_cce")
                list(APPEND _OPC_CONFIG "--save-temp-files")
            endif()
        endforeach()
    endif()

    if(ENABLE_OOM)
        list(APPEND _OPC_CONFIG "--oom")
        list(APPEND _OPC_CONFIG "-ffunction-sections -fdata-sections")
    endif()

    if(_OPC_CONFIG)
        add_ops_compile_options(
                OP_NAME ${OP_COMPILE_OP_NAME}
                OPTIONS ${_OPC_CONFIG}
        )
    endif()
endfunction()

function(add_ops_src_copy)
    cmake_parse_arguments(SRC_COPY "" "TARGET_NAME;SRC;DST;BE_RELIED;COMPUTE_UNIT" "" ${ARGN})

    set(OPS_UTILS_INC_KERNEL_TARGET ops_utils_inc_kernel_${SRC_COPY_COMPUTE_UNIT})
    if (EXISTS ${OPS_ADV_UTILS_KERNEL_INC})
        if (NOT TARGET ${OPS_UTILS_INC_KERNEL_TARGET})
            get_filename_component(_ROOT_OPS_SRC_DIR    "${SRC_COPY_DST}" DIRECTORY)
            set(OPS_UTILS_INC_KERNEL_DIR ${_ROOT_OPS_SRC_DIR}/ascendc/common)
            add_custom_command(OUTPUT ${OPS_UTILS_INC_KERNEL_DIR}
                    COMMAND mkdir -p ${OPS_UTILS_INC_KERNEL_DIR}/regbase
                    COMMAND cp -rf ${OPS_ADV_UTILS_KERNEL_INC}/* ${OPS_UTILS_INC_KERNEL_DIR}
            )

            add_custom_target(${OPS_UTILS_INC_KERNEL_TARGET}
                    DEPENDS ${OPS_UTILS_INC_KERNEL_DIR}
            )
        endif ()
    endif ()

    set(MC2_OPS_LIST "matmul_reduce_scatter;"
        "matmul_reduce_scatter_v2;"
        "grouped_mat_mul_allto_allv;"
        "quant_grouped_mat_mul_allto_allv;"
        "grouped_mat_mul_all_reduce;"
        "batch_mat_mul_reduce_scatter_allto_all;"
        "allto_allv_grouped_mat_mul;"
        "allto_allv_quant_grouped_mat_mul;"
        "allto_all_all_gather_batch_mat_mul;"
        "distribute_barrier;"
        "moe_distribute_combine_add_rms_norm;"
        "moe_distribute_dispatch;"
        "moe_distribute_combine;"
        "moe_distribute_dispatch_v2;"
        "moe_distribute_combine_v2;"
        "moe_distribute_dispatch_v3;"
        "moe_distribute_combine_v3;"
        "moe_update_expert;"
        "all_gather_matmul;"
        "all_gather_matmul_v2;"
        "matmul_all_reduce;"
        "matmul_all_reduce_apt;"
        "matmul_all_reduce_add_rms_norm;"
        "inplace_matmul_all_reduce_add_rms_norm;"
        "quant_all_reduce;"
        "quant_reduce_scatter;"
        "allto_all_matmul;"
        "matmul_allto_all;"
        "attention_to_ffn;"
        "ffn_to_attention;"
    ) # mc2算子列表

    get_filename_component(FOLDER_NAME "${SRC_COPY_DST}" NAME_WE)
    list(FIND MC2_OPS_LIST "${FOLDER_NAME}" INDEX)
    if(NOT INDEX EQUAL -1)
        set(BELONG_MC2_OPS TRUE)
    endif()

    if(NOT BUILD_OPS_RTY_KERNEL AND BELONG_MC2_OPS)
        file(GLOB SRC_FILES ${SRC_COPY_SRC}/* ${SRC_COPY_SRC}/op_kernel/*)
    else()
        file(GLOB SRC_FILES ${SRC_COPY_SRC}/*)
    endif()
    list(FILTER SRC_FILES EXCLUDE REGEX "op_host")

    get_filename_component(PARENT_PTH "${SRC_COPY_SRC}" DIRECTORY)
    get_filename_component(CUR_NAME "${SRC_COPY_SRC}" NAME)
    get_filename_component(PARENT_NAME "${PARENT_PTH}" NAME)

    set(DOING_TARGET_NAME ${SRC_COPY_TARGET_NAME})
    if(${CUR_NAME} STREQUAL "common")
        set(DOING_TARGET_NAME ${PARENT_NAME}_${DOING_TARGET_NAME})
    endif()

    if (NOT TARGET ${DOING_TARGET_NAME})
        set(_BUILD_FLAG ${SRC_COPY_DST}/${DOING_TARGET_NAME}.done)
        if (NOT BUILD_OPS_RTY_KERNEL AND BELONG_MC2_OPS)
            add_custom_command(OUTPUT ${_BUILD_FLAG}
                    COMMAND mkdir -p ${SRC_COPY_DST}
                    COMMAND cp -rf ${SRC_FILES} ${SRC_COPY_DST}
                    COMMAND rm -rf ${SRC_COPY_DST}/op_kernel/
                    COMMAND touch ${_BUILD_FLAG}
            )
        else()
            add_custom_command(OUTPUT ${_BUILD_FLAG}
                    COMMAND mkdir -p ${SRC_COPY_DST}
                    COMMAND cp -rf ${SRC_FILES} ${SRC_COPY_DST}
                    COMMAND touch ${_BUILD_FLAG}
            )
        endif()

        add_custom_target(${DOING_TARGET_NAME}
                DEPENDS ${_BUILD_FLAG}
        )
    endif ()

    if (TARGET ${OPS_UTILS_INC_KERNEL_TARGET})
        add_dependencies(${DOING_TARGET_NAME} ${OPS_UTILS_INC_KERNEL_TARGET})
    endif ()

    if (DEFINED SRC_COPY_BE_RELIED)
        add_dependencies(${SRC_COPY_BE_RELIED} ${DOING_TARGET_NAME})
    endif ()

endfunction()

function(add_bin_compile_target)
    cmake_parse_arguments(BINARY "" "COMPUTE_UNIT" "OP_INFO" ${ARGN})

    if (ENABLE_BUILT_IN)
        set(_INSTALL_DIR ops_transformer/built-in/op_impl/ai_core/tbe/kernel)
    else()
        set(_INSTALL_DIR packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/kernel)
    endif()
    set(_OUT_DIR ${ASCEND_BINARY_OUT_DIR}/${BINARY_COMPUTE_UNIT})

    set(BIN_OUT_DIR      ${_OUT_DIR}/bin)
    set(GEN_OUT_DIR      ${_OUT_DIR}/gen)
    set(SRC_OUT_DIR      ${_OUT_DIR}/src)
    file(MAKE_DIRECTORY  ${BIN_OUT_DIR})

    foreach(_op_info ${BINARY_OP_INFO})
        get_filename_component(_op_name "${_op_info}" NAME)
        set(${_op_name}_dir ${_op_info})
        set(${_op_name}_apt_dir ${_op_info})
    endforeach()

    set(_ops_target_list)
    set(compile_scripts)
    file(GLOB scripts_list ${GEN_OUT_DIR}/*.sh)
    list(APPEND compile_scripts ${scripts_list})

    foreach(bin_script ${compile_scripts})
        get_filename_component(bin_file ${bin_script} NAME_WE)
        string(REPLACE "-" ";" bin_sep ${bin_file})
        list(GET bin_sep 0 op_type)
        list(GET bin_sep 1 op_file)
        list(GET bin_sep 2 op_index)

        if (NOT DEFINED ${op_file}_dir)
            continue()
        endif ()

        if (NOT TARGET ${op_file})
            add_custom_target(${op_file})
            add_dependencies(ops_transformer_kernel ${op_file})
        endif ()

        set(OP_TARGET_NAME ${op_file}_${BINARY_COMPUTE_UNIT})

        if (NOT TARGET ${OP_TARGET_NAME})
            add_custom_target(${OP_TARGET_NAME})
            add_dependencies(${op_file} ${OP_TARGET_NAME})
            list(APPEND _ops_target_list ${OP_TARGET_NAME})

            set(OP_SRC_OUT_DIR  ${SRC_OUT_DIR}/${op_file})
            set(OP_BIN_OUT_DIR  ${BIN_OUT_DIR}/${op_file})
            file(MAKE_DIRECTORY ${OP_SRC_OUT_DIR})

            add_ops_src_copy(
                    TARGET_NAME
                    ${OP_TARGET_NAME}_src_copy
                    SRC
                    ${${op_file}_dir}
                    DST
                    ${OP_SRC_OUT_DIR}
                    COMPUTE_UNIT
                    ${BINARY_COMPUTE_UNIT}
            )

            if (DEFINED ${op_file}_depends)
                foreach(depend_info ${${op_file}_depends})
                    if (ENABLE_EXPERIMENTAL)
 	                    set(depend_info_update "experimental/${depend_info}")
 	                else()
 	                    set(depend_info_update ${depend_info})
 	                endif()
                    get_filename_component(_depend_op_name "${depend_info}" NAME)
                    set(_depend_op_target ${_depend_op_name}_${BINARY_COMPUTE_UNIT}_src_copy)
                    add_ops_src_copy(
                            TARGET_NAME
                            ${_depend_op_target}
                            SRC
                            ${CMAKE_SOURCE_DIR}/${depend_info_update}
                            DST
                            ${SRC_OUT_DIR}/${_depend_op_name}
                            COMPUTE_UNIT
                            ${BINARY_COMPUTE_UNIT}
                            BE_RELIED
                            ${OP_TARGET_NAME}_src_copy
                    )
                endforeach()
            endif ()

            set(DYNAMIC_PY_FILE ${OP_SRC_OUT_DIR}/${op_type}.py)
            add_custom_command(OUTPUT ${DYNAMIC_PY_FILE}
                    COMMAND cp -rf ${ASCEND_IMPL_OUT_DIR}/dynamic/${op_file}.py ${DYNAMIC_PY_FILE}
                    # COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/update_get_kernel_source.sh ${DYNAMIC_PY_FILE}
            )

            add_custom_target(${OP_TARGET_NAME}_py_copy
                    DEPENDS ${DYNAMIC_PY_FILE}
            )

            add_custom_command(OUTPUT ${OP_BIN_OUT_DIR}
                    COMMAND mkdir -p ${OP_BIN_OUT_DIR}
            )

            add_custom_target(${OP_TARGET_NAME}_mkdir
                    DEPENDS ${OP_BIN_OUT_DIR}
            )

            if (ENABLE_BUILT_IN)
                install(DIRECTORY ${OP_BIN_OUT_DIR}
                        DESTINATION ${_INSTALL_DIR}/${BINARY_COMPUTE_UNIT}/ops_transformer OPTIONAL
                )
                install(FILES ${BIN_OUT_DIR}/${op_file}.json
                        DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT}/ops_transformer OPTIONAL
                )
            else()
                install(DIRECTORY ${OP_BIN_OUT_DIR}
                        DESTINATION ${_INSTALL_DIR}/${BINARY_COMPUTE_UNIT} OPTIONAL
                )
                install(FILES ${BIN_OUT_DIR}/${op_file}.json
                        DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT} OPTIONAL
                )
            endif()
        endif ()

        set(_group "1-0")
        if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
            if (NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
                string(REGEX MATCH "^(.*_apt)$" _match_apt ${op_file})
                if(_match_apt)
                    #如果以_apt结尾，使用去掉后缀的文件名进行查找
                    string(REGEX REPLACE "_apt$" "" _op_file_strip_apt ${op_file})
                    list(FIND ASCEND_OP_NAME ${_op_file_strip_apt} _index)
                else()
                    list(FIND ASCEND_OP_NAME ${op_file} _index)
                    set(_op_file_strip_apt ${op_file})
                endif()
                if (${op_file} IN_LIST ASCEND_OP_NAME OR ${_op_file_strip_apt} IN_LIST ASCEND_OP_NAME)
                    list(LENGTH ASCEND_OP_NAME _len)
                    math(EXPR _next_index "${_index} + 1")
                    if (${_next_index} LESS ${_len})
                        list(GET ASCEND_OP_NAME ${_next_index} _group_str)
                        set(_regex "^[0-9]+-[0-9]+$")
                        string(REGEX MATCH "${_regex}" match "${_group_str}")
                        if (match)
                            set(_group   ${_group_str})
                        endif ()
                    endif ()
                endif ()
            endif ()
        endif ()

        string(REPLACE "-" ";" _group_sep ${_group})

        list(GET _group_sep 1 start_index)
        set(end_index         ${op_index})
        list(GET _group_sep 0 step)

        set(_compile_flag false)
        if (${start_index} LESS ${end_index})
            foreach(i RANGE ${start_index} ${end_index} ${step})
                if (${i} EQUAL ${end_index})
                    set(_compile_flag true)
                    break()
                endif ()
            endforeach()
        elseif (${start_index} EQUAL ${end_index})
            set(_compile_flag true)
        else()
            set(_compile_flag false)
        endif ()

        if (_compile_flag)
            set(_BUILD_COMMAND)
            set(_BUILD_FLAG ${GEN_OUT_DIR}/${OP_TARGET_NAME}_${op_index}.done)
            if (ENABLE_OPS_HOST OR ENABLE_HOST_TILING)
                list(APPEND _BUILD_COMMAND export ASCEND_CUSTOM_OPP_PATH=${CUSTOM_DIR} &&)
            endif ()
            list(APPEND _BUILD_COMMAND export HI_PYTHON="python3" &&)
            list(APPEND _BUILD_COMMAND export TILINGKEY_PAR_COMPILE=1 &&)
            list(APPEND _BUILD_COMMAND export BIN_FILENAME_HASHED=1 &&)
            list(APPEND _BUILD_COMMAND bash ${bin_script} ${OP_SRC_OUT_DIR}/${op_type}.py ${OP_BIN_OUT_DIR})
            if(CMAKE_GENERATOR MATCHES "Unix Makefiles")
                list(APPEND _BUILD_COMMAND && echo $(MAKE))
            endif()

            add_custom_command(OUTPUT ${_BUILD_FLAG}
                    COMMAND ${_BUILD_COMMAND}
                    COMMAND touch ${_BUILD_FLAG}
                    WORKING_DIRECTORY ${GEN_OUT_DIR}
            )

            add_custom_target(${OP_TARGET_NAME}_${op_index}
                DEPENDS ${_BUILD_FLAG}
            )

            if (ENABLE_OPS_HOST OR ENABLE_HOST_TILING)
                add_dependencies(${OP_TARGET_NAME}_${op_index} optiling_compat generate_ops_info)
            endif ()
            add_dependencies(${OP_TARGET_NAME}_${op_index} ${OP_TARGET_NAME}_src_copy ${OP_TARGET_NAME}_py_copy ${OP_TARGET_NAME}_mkdir)
            add_dependencies(${OP_TARGET_NAME} ${OP_TARGET_NAME}_${op_index})
        endif ()
    endforeach()

    if (_ops_target_list)
        set(OPS_CONFIG_TARGET ops_config_${BINARY_COMPUTE_UNIT})
        set(BINARY_INFO_CONFIG_FILE ${BIN_OUT_DIR}/binary_info_config.json)
        set(RELOCATABLE_KERNEL_INFO_CONFIG_FILE ${BIN_OUT_DIR}/relocatable_kernel_info_config.json)

        add_custom_command(OUTPUT ${BINARY_INFO_CONFIG_FILE}
                COMMAND ${HI_PYTHON} ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_ops_config.py -p ${BIN_OUT_DIR} -s ${BINARY_COMPUTE_UNIT}
        )

        add_custom_target(${OPS_CONFIG_TARGET}
                DEPENDS ${BINARY_INFO_CONFIG_FILE}
        )

        add_dependencies(ops_transformer_config ${OPS_CONFIG_TARGET})

        foreach(_op_target ${_ops_target_list})
            add_dependencies(${OPS_CONFIG_TARGET} ${_op_target})
        endforeach()

        if (ENABLE_BUILT_IN)
            install(FILES ${BINARY_INFO_CONFIG_FILE}
                    DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT}/ops_transformer OPTIONAL
            )
            install(FILES ${RELOCATABLE_KERNEL_INFO_CONFIG_FILE}
                    DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT}/ops_transformer OPTIONAL
            )
        else()
            install(FILES ${RELOCATABLE_KERNEL_INFO_CONFIG_FILE}
                    DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT} OPTIONAL
            )
            install(FILES ${BINARY_INFO_CONFIG_FILE}
                    DESTINATION ${_INSTALL_DIR}/config/${BINARY_COMPUTE_UNIT} OPTIONAL
            )
        endif()
    endif ()
endfunction()

function(redefine_file_macro)
    cmake_parse_arguments(_FILE "" "" "TARGET_NAME" ${ARGN})

    foreach(_target_name ${_FILE_TARGET_NAME})
        target_compile_options(${_target_name} PRIVATE
                -Wno-builtin-macro-redefined
        )

        get_target_property(_srcs ${_target_name} SOURCES)

        foreach(_src ${_srcs})
            get_filename_component(_src_name "${_src}" NAME)
            set_source_files_properties(${_src}
                    PROPERTIES COMPILE_DEFINITIONS __FILE__="${_src_name}"
            )
        endforeach()
    endforeach()
endfunction()

function(add_static_ops)
    cmake_parse_arguments(STATIC "" "SRC_DIR" "ACLNN_SRC;ACLNN_INNER_SRC" ${ARGN})
    set(prepare_ops_adv_static_target prepare_ops_transformer_static)
    set(static_src_temp_dir ${CMAKE_CURRENT_BINARY_DIR}/static_src_temp_dir)
    set(modified_files)
    foreach(ops_type ${OPS_STATIC_TYPES})
        get_target_property(all_srcs aclnn_ops_${ops_type} SOURCES)
        set(add_srcs)
        set(generate_aclnn_srcs)
        foreach(_src ${all_srcs})
            string(REGEX MATCH "^${STATIC_SRC_DIR}" is_match "${_src}")
            if (is_match)
                list(APPEND add_srcs ${_src})
            endif ()
        endforeach()

        foreach(_src ${add_srcs})
            get_filename_component(name_without_ext ${_src} NAME_WE)
            string(REGEX REPLACE "^aclnn_" "" _op_name ${name_without_ext})

            foreach(_aclnn_src ${STATIC_ACLNN_SRC})
                get_filename_component(aclnn_name ${_aclnn_src} NAME_WE)
                if("aclnn_${_op_name}" STREQUAL "${aclnn_name}")
                    list(APPEND generate_aclnn_srcs ${_aclnn_src})
                    break()
                endif()
            endforeach()

            foreach(_aclnn_inner_src ${STATIC_ACLNN_INNER_SRC})
                get_filename_component(aclnn_inner_name ${_aclnn_inner_src} NAME_WE)
                if("aclnnInner_${_op_name}" STREQUAL "${aclnn_inner_name}")
                    list(APPEND generate_aclnn_srcs ${_aclnn_inner_src})
                    break()
                endif()
            endforeach()
        endforeach()

        if(add_srcs)
            list(TRANSFORM add_srcs REPLACE "${STATIC_SRC_DIR}" "${static_src_temp_dir}" OUTPUT_VARIABLE add_static_srcs)
            list(APPEND modified_files ${add_static_srcs})
            set(aclnn_ops_static_target aclnn_ops_${ops_type}_static)
            set_source_files_properties(${add_static_srcs}
                    TARGET_DIRECTORY ${aclnn_ops_static_target}
                    PROPERTIES GENERATED TRUE
            )

            target_sources(${aclnn_ops_static_target} PRIVATE
                    ${add_static_srcs}
            )
            add_dependencies(${aclnn_ops_static_target} ${prepare_ops_adv_static_target})
        endif()

        if(generate_aclnn_srcs)
            list(REMOVE_DUPLICATES generate_aclnn_srcs)
            set(aclnn_op_target acl_op_${ops_type}_builtin)
            set_source_files_properties(${generate_aclnn_srcs}
                    TARGET_DIRECTORY ${aclnn_op_target}
                    PROPERTIES GENERATED TRUE
            )

            target_sources(${aclnn_op_target} PRIVATE
                    ${generate_aclnn_srcs}
            )
        endif()
    endforeach()

    if(NOT TARGET ${prepare_ops_adv_static_target})
        list(REMOVE_DUPLICATES modified_files)
        add_custom_command(OUTPUT ${static_src_temp_dir}
            COMMAND mkdir -p ${static_src_temp_dir}
            COMMAND cp -rf ${STATIC_SRC_DIR}/chunk_gated_delta_rule ${STATIC_SRC_DIR}/gmm ${STATIC_SRC_DIR}/mc2 ${STATIC_SRC_DIR}/attention ${static_src_temp_dir} || true
            COMMAND ${HI_PYTHON} -B ${OPS_STATIC_SCRIPT} InsertIni -p ${static_src_temp_dir} -f ${modified_files}
        )

        add_custom_target(${prepare_ops_adv_static_target}
                DEPENDS ${static_src_temp_dir}
        )
    endif()
endfunction()

function(pack_tiling_sink)
  ExternalProject_Get_Property(tiling_sink_task BINARY_DIR)

  if(ENABLE_BUILT_IN)
    set(TRANSFORMER_OPMASTER_SO ${BINARY_DIR}/libtiling_device_transformer.so)
    set(INSTALL_DIR "ops_transformer/built-in/op_impl/ai_core/tbe/op_tiling_device/lib")
  else()
    set(TRANSFORMER_OPMASTER_SO ${BINARY_DIR}/libcust_opmaster.so)
    set(INSTALL_DIR "packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/op_master_device/lib")
  endif()
  install(CODE "
    if(EXISTS \"${TRANSFORMER_OPMASTER_SO}\")
      file(
        INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${INSTALL_DIR}\"
        TYPE FILE FILES \"${TRANSFORMER_OPMASTER_SO}\")
    endif()
  ")
endfunction()

if (BUILD_OPEN_PROJECT)
    if (TESTS_UT_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/func_utest.cmake)
    endif ()
    if (TESTS_EXAMPLE_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/func_examples.cmake)
    endif ()
endif ()

function(concat_op_names)
    set(multiValueArgs OPTYPE ACLNNTYPE ACLNN_EXTRA_VERSION)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(${ARG_ACLNNTYPE} STREQUAL "aclnn")
        set(ACLNN_PREFIX aclnn_${ARG_OPTYPE})
        set(ACLNN_EXTRA_HEADER "")
        set(ACLNN_EXTRA_SRC "")

        list(LENGTH ARG_ACLNN_EXTRA_VERSION AclnnExtraVersionLen)
        math(EXPR index "${AclnnExtraVersionLen} - 1")
        if (index GREATER_EQUAL 0)
            foreach(i RANGE ${index})
                list(GET ARG_ACLNN_EXTRA_VERSION ${i} version)
                list(APPEND ACLNN_EXTRA_HEADER ${ACLNN_PREFIX}_${version}.h)
                list(APPEND ACLNN_EXTRA_SRC ${ACLNN_PREFIX}_${version}.cpp)
            endforeach()
        endif()

        list(APPEND ACLNN_EXTRA_HEADERS ${ACLNN_EXTRA_HEADER})
        list(REMOVE_DUPLICATES ACLNN_EXTRA_HEADERS)
        list(APPEND ACLNN_EXTRA_SRCS ${ACLNN_EXTRA_SRC})
        list(REMOVE_DUPLICATES ACLNN_EXTRA_SRCS)

        set(ACLNN_EXTRA_HEADERS
            ${ACLNN_EXTRA_HEADERS}
            CACHE STRING "Aclnn Extra Headers" FORCE
        )
        set(ACLNN_EXTRA_SRCS
            ${ACLNN_EXTRA_SRCS}
            CACHE STRING "Aclnn Extra Sources" FORCE
        )

    elseif(${ARG_ACLNNTYPE} STREQUAL "aclnn_inner")
        set(ACLNNINNER_PREFIX aclnnInner_${ARG_OPTYPE})
        set(ACLNNINNER_EXTRA_HEADER "")
        set(ACLNNINNER_EXTRA_SRC "")

        list(LENGTH ARG_ACLNN_EXTRA_VERSION AclnnExtraVersionLen)
        math(EXPR index "${AclnnExtraVersionLen} - 1")
        if (index GREATER_EQUAL 0)
            foreach(i RANGE ${index})
                list(GET ARG_ACLNN_EXTRA_VERSION ${i} version)
                list(APPEND ACLNNINNER_EXTRA_HEADER ${ACLNNINNER_PREFIX}_${version}.h)
                list(APPEND ACLNNINNER_EXTRA_SRC ${ACLNNINNER_PREFIX}_${version}.cpp)
            endforeach()
        endif()

        list(APPEND ACLNNINNER_EXTRA_HEADERS ${ACLNNINNER_EXTRA_HEADER})
        list(REMOVE_DUPLICATES ACLNNINNER_EXTRA_HEADERS)
        list(APPEND ACLNNINNER_EXTRA_SRCS ${ACLNNINNER_EXTRA_SRC})
        list(REMOVE_DUPLICATES ACLNNINNER_EXTRA_SRCS)

        set(ACLNNINNER_EXTRA_HEADERS
            ${ACLNNINNER_EXTRA_HEADERS}
            CACHE STRING "AclnnInner Extra Headers" FORCE
        )
        set(ACLNNINNER_EXTRA_SRCS
            ${ACLNNINNER_EXTRA_SRCS}
            CACHE STRING "AclnnInner Extra Sources" FORCE
        )
    endif()
endfunction()

macro(replace_cur_major_minor_ver)
    string(REPLACE CUR_MAJOR_MINOR_VER "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_VERSION_MAJOR_MINOR}" depend "${depend}")
endmacro()
 	 
# 设置包和版本号
function(set_package name)
    cmake_parse_arguments(VERSION "" "VERSION" "" ${ARGN})
    set(VERSION "${VERSION_VERSION}")
    if(NOT name)
        message(FATAL_ERROR "The name parameter is not set in set_package.")
    endif()
    if(NOT VERSION)
        message(FATAL_ERROR "The VERSION parameter is not set in set_package(${name}).")
    endif()
    string(REGEX MATCH "^([0-9]+\\.[0-9]+)" VERSION_MAJOR_MINOR "${VERSION}")
    list(APPEND CANN_VERSION_PACKAGES "${name}")
    set(CANN_VERSION_PACKAGES "${CANN_VERSION_PACKAGES}" PARENT_SCOPE)
    set(CANN_VERSION_CURRENT_PACKAGE "${name}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_VERSION "${VERSION}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_VERSION_MAJOR_MINOR "${VERSION_MAJOR_MINOR}" PARENT_SCOPE)
    set(CANN_VERSION_${name}_BUILD_DEPS PARENT_SCOPE)
    set(CANN_VERSION_${name}_RUN_DEPS PARENT_SCOPE)
endfunction()
 	 
# 设置构建依赖
function(set_build_dependencies pkg_name depend)
    if(NOT CANN_VERSION_CURRENT_PACKAGE)
        message(FATAL_ERROR "The set_package must be invoked first.")
    endif()
    if(NOT pkg_name)
        message(FATAL_ERROR "The pkg_name parameter is not set in set_build_dependencies.")
    endif()
    if(NOT depend)
        message(FATAL_ERROR "The depend parameter is not set in set_build_dependencies.")
    endif()
    replace_cur_major_minor_ver()
    list(APPEND CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS "${pkg_name}" "${depend}")
    set(CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_BUILD_DEPS}" PARENT_SCOPE)
endfunction()
 	 
# 设置运行依赖
function(set_run_dependencies pkg_name depend)
    if(NOT CANN_VERSION_CURRENT_PACKAGE)
        message(FATAL_ERROR "The set_package must be invoked first.")
    endif()
    if(NOT pkg_name)
        message(FATAL_ERROR "The pkg_name parameter is not set in set_run_dependencies.")
    endif()
    if(NOT depend)
        message(FATAL_ERROR "The depend parameter is not set in set_run_dependencies.")
    endif()
    replace_cur_major_minor_ver()
    list(APPEND CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS "${pkg_name}" "${depend}")
    set(CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS "${CANN_VERSION_${CANN_VERSION_CURRENT_PACKAGE}_RUN_DEPS}" PARENT_SCOPE)
endfunction()
 	 
# 检查构建依赖
function(check_pkg_build_deps pkg_name)
    execute_process(
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_build_dependencies.py "${ASCEND_CANN_PACKAGE_PATH}" ${CANN_VERSION_${pkg_name}_BUILD_DEPS}
        RESULT_VARIABLE result
    )
    if(result)
        message(FATAL_ERROR "Check ${pkg_name} build dependencies failed!")
    endif()
endfunction()
 	 
# 添加生成version.info的目标
# 目标名格式为：version_${包名}_info
function(add_version_info_targets)
    foreach(pkg_name ${CANN_VERSION_PACKAGES})
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
            COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py --output ${CMAKE_BINARY_DIR}/version.${pkg_name}.info
                    "${CANN_VERSION_${pkg_name}_VERSION}" ${CANN_VERSION_${pkg_name}_RUN_DEPS}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/version.cmake ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_version_info.py
            VERBATIM
        )
        add_custom_target(version_${pkg_name}_info ALL DEPENDS ${CMAKE_BINARY_DIR}/version.${pkg_name}.info)
    endforeach()
endfunction()