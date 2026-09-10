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
