# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------
# Pin the metadef source tree so metadef host headers (exe_graph/runtime/...)
# come from the selected source revision instead of the CANN toolkit install.
# Mirrors cmake/third_party/opbase.cmake (PR #1299).
# ----------------------------------------------------------------------------
set(METADEF_TAG_ID 6814f2d661b56e07ee35c277812a3566caca9419)   # branch 9.2.0
message(STATUS "Select metadef revision: ${METADEF_TAG_ID}")

if(EXISTS "${PROJECT_SOURCE_DIR}/../../metadef")
  get_filename_component(METADEF_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/../../metadef REALPATH)
  message(STATUS "Find metadef source dir: ${METADEF_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/metadef")
  get_filename_component(METADEF_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/metadef REALPATH)
  message(STATUS "Find metadef source dir: ${METADEF_SOURCE_PATH}")
  if(EXISTS "${METADEF_SOURCE_PATH}/.git")
    execute_process(
      COMMAND git checkout ${METADEF_TAG_ID}
      WORKING_DIRECTORY ${METADEF_SOURCE_PATH}
    )
  else()
    message(STATUS "metadef source dir has no .git; using bundled tarball sources as-is")
  endif()
else()
  include(FetchContent)

  FetchContent_Declare(
    metadef
    GIT_REPOSITORY https://gitcode.com/cann/metadef.git
    GIT_TAG ${METADEF_TAG_ID}
    GIT_PROGRESS TRUE
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/metadef)

  FetchContent_Populate(metadef)

  set(METADEF_SOURCE_PATH ${CANN_3RD_LIB_PATH}/metadef)
endif()

# The source tree keeps external headers under inc/external/ and expects both
# "exe_graph/..." and "external/ge_common/..." style includes.  Build a staged
# include root that mirrors the installed CANN layout (include/exe_graph,
# include/external/ge_common, ...) so the pinned source can be used verbatim.
set(METADEF_SOURCE_INCLUDE ${CMAKE_BINARY_DIR}/metadef_include)
file(MAKE_DIRECTORY ${METADEF_SOURCE_INCLUDE})
file(GLOB _metadef_inc_entries LIST_DIRECTORIES true ${METADEF_SOURCE_PATH}/inc/external/*)
foreach(_metadef_entry ${_metadef_inc_entries})
  get_filename_component(_metadef_entry_name ${_metadef_entry} NAME)
  if(NOT EXISTS ${METADEF_SOURCE_INCLUDE}/${_metadef_entry_name})
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ${_metadef_entry} ${METADEF_SOURCE_INCLUDE}/${_metadef_entry_name})
  endif()
endforeach()
if(NOT EXISTS ${METADEF_SOURCE_INCLUDE}/external)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ${METADEF_SOURCE_PATH}/inc/external ${METADEF_SOURCE_INCLUDE}/external)
endif()
unset(_metadef_inc_entries)
unset(_metadef_entry)
unset(_metadef_entry_name)
message(STATUS "Staged metadef include dir: ${METADEF_SOURCE_INCLUDE}")
