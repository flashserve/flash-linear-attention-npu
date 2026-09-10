"""ChunkKdaFwdPrepare 调度测试的 CMake 接入合同。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OP_HOST = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_host"


def test_tiling_processor_test_is_built_when_tests_are_enabled():
    host_cmake = (OP_HOST / "CMakeLists.txt").read_text(encoding="utf-8")
    test_cmake = (OP_HOST / "tests/CMakeLists.txt").read_text(encoding="utf-8")
    test_source = OP_HOST / "tests/chunk_kda_fwd_prepare_tiling_processor_test.cpp"
    assert "if(ENABLE_TEST)" in host_cmake
    assert "add_subdirectory(tests)" in host_cmake
    assert "add_executable(" in test_cmake
    assert "chunk_kda_fwd_prepare_tiling_processor_test" in test_cmake
    assert test_source.is_file()


def test_output_mode_is_an_internal_tiling_key_axis():
    tiling_source = (OP_HOST / "chunk_kda_fwd_prepare_tiling.cpp").read_text(
        encoding="utf-8"
    )
    tiling_header = (OP_HOST / "chunk_kda_fwd_prepare_tiling.h").read_text(
        encoding="utf-8"
    )
    assert "PREPARE_ATTR_OUTPUT_MODE" in tiling_header
    assert "GetAttrPointer<int64_t>(PREPARE_ATTR_OUTPUT_MODE)" in tiling_source
    assert "GET_TPL_TILING_KEY(" in tiling_source
    assert "*safeGatePtr), outputMode)" in tiling_source
    assert "TILING_DATA_FIELD_DEF(uint32_t, outputMask)" not in tiling_header


def test_all_ir_outputs_remain_required():
    op_def = (OP_HOST / "chunk_kda_fwd_prepare_def.cpp").read_text(
        encoding="utf-8"
    )
    output_names = (
        "gk", "aqk", "akk", "w", "u", "qg", "kg", "qg_scaled",
        "q_hat", "k_hat", "q_rstd", "k_rstd", "beta_eff",
    )
    assert op_def.count('this->Output("') == len(output_names)
    for name in output_names:
        assert f'this->Output("{name}").ParamType(REQUIRED)' in op_def


def test_l0_uses_placeholders_for_null_l2_outputs():
    l2_source = (OP_HOST / "op_api/aclnn_chunk_kda_fwd_prepare.cpp").read_text(
        encoding="utf-8"
    )
    l0_source = (OP_HOST / "op_api/chunk_kda_fwd_prepare.cpp").read_text(
        encoding="utf-8"
    )
    assert "GetOutputMode(params)" in l2_source
    assert "只支持 none/recompute/save 三档" in l2_source
    for name in (
        "akkForKernel", "qgForKernel", "qHatForKernel", "kHatForKernel",
        "qRstdForKernel", "kRstdForKernel", "betaEffForKernel",
    ):
        assert name in l0_source
    assert "OutputOrEmptyDescriptor" in l0_source
    assert "AllocTensor(MakeShape({0})" in l0_source
    assert "useExp2, outputMode" in l0_source


def test_l2_helpers_keep_cpp_internal_linkage():
    l2_source = (OP_HOST / "op_api/aclnn_chunk_kda_fwd_prepare.cpp").read_text(
        encoding="utf-8"
    )
    namespace_begin = l2_source.index("namespace {")
    namespace_end = l2_source.index("} // namespace")
    c_linkage_begin = l2_source.index('extern "C" {')
    assert namespace_begin < namespace_end < c_linkage_begin
