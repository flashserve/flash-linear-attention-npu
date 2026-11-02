# CPack runs this after staging and before either makeself implementation.
file(GLOB FLA_VENDOR_LIBRARIES
    "${CPACK_TEMPORARY_DIRECTORY}/packages/vendors/*/op_api/lib/libcust_opapi.so")
if(NOT FLA_VENDOR_LIBRARIES)
    message(FATAL_ERROR "No staged custom op_api library for ABI manifest generation")
endif()
foreach(FLA_LIBRARY IN LISTS FLA_VENDOR_LIBRARIES)
    get_filename_component(FLA_VENDOR "${FLA_LIBRARY}/../../.." ABSOLUTE)
    execute_process(
        COMMAND "${CPACK_FLA_PYTHON}" "${CPACK_FLA_ABI_HELPER}" --write-manifest "${FLA_VENDOR}"
        RESULT_VARIABLE FLA_ABI_RESULT)
    if(NOT FLA_ABI_RESULT EQUAL 0)
        message(FATAL_ERROR "ABI manifest generation failed for ${FLA_VENDOR}")
    endif()
endforeach()
