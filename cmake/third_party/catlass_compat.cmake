# Keep the downloaded dependency intact; package and compile the same patched
# header view. Catlass 490ae058 HSTU headers use unqualified factories, which can
# bind to AscendC::MakeShape/MakeStride after a caller imports AscendC globally.
function(fla_prepare_catlass_headers source_include output_var)
  set(_include "${CMAKE_BINARY_DIR}/_deps/catlass_fla/include")
  file(REMOVE_RECURSE "${_include}")
  file(COPY "${source_include}/" DESTINATION "${_include}")
  foreach(_header IN ITEMS
      block_mmad_hstu_qk_split_row_tla.hpp
      block_mmad_hstu_pv_split_row_tla.hpp)
    set(_path "${_include}/catlass/gemm/block/${_header}")
    if(NOT EXISTS "${_path}")
      message(FATAL_ERROR
        "[ThirdPartyLib][catlass] Missing ${_header}; clear stale Catlass/build caches and reconfigure.")
    endif()
    file(READ "${_path}" _contents)
    # Only these two assignments need qualification. Already-qualified upstream
    # headers remain unchanged; no implementation or umbrella include is removed.
    string(REPLACE "auto shape = MakeShape(" "auto shape = tla::MakeShape("
      _contents "${_contents}")
    string(REPLACE "auto strideInPage = MakeStride(" "auto strideInPage = tla::MakeStride("
      _contents "${_contents}")
    file(WRITE "${_path}" "${_contents}")
  endforeach()
  set(${output_var} "${_include}" PARENT_SCOPE)
endfunction()
