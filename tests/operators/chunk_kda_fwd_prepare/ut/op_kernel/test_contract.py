"""ChunkKdaFwdPrepare 的 direct launch 源码合同。"""

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "tests/op_cases/chunk_kda_fwd_prepare.json"
DIRECT = (
    ROOT
    / "tests/operators/chunk_kda_fwd_prepare/routes"
    / "test_direct_chunk_kda_fwd_prepare.cpp"
)
KERNEL = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel"
FAST_EXAMPLE = ROOT / "examples/fast_kernel_launch_example"
DIRECT_KERNEL = (
    FAST_EXAMPLE
    / "csrc/chunk_kda_fwd_prepare/chunk_kda_fwd_prepare_direct_kernel.h"
)
DIRECT_WRAPPER = (
    FAST_EXAMPLE
    / "csrc/chunk_kda_fwd_prepare/chunk_kda_fwd_prepare_direct.cpp"
)
DIRECT_DEVICE_TEST = (
    FAST_EXAMPLE
    / "tests/chunk_kda_fwd_prepare/test_chunk_kda_fwd_prepare_direct.py"
)
WHEEL_CHECKER = ROOT / "scripts/check_packaged_wheel_api.py"


def _read_kernel(relative_path):
    return (KERNEL / relative_path).read_text(encoding="utf-8")


def _function_body(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    raise AssertionError(f"函数没有闭合: {marker}")


_CPP_TOKEN = re.compile(
    r'''(?:"(?:\\.|[^"\\])*")|(?:'(?:\\.|[^'\\])*')'''
    r"|(?://[^\n]*)|(?:/\*.*?\*/)"
    r"|(?:[A-Za-z_]\w*)"
    r"|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d*)?(?:[eEpP][+-]?\d+)?[A-Za-z_]*)"
    r"|(?:::|->|<<=|>>=|==|!=|<=|>=|&&|\|\||\+\+|--|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=)"
    r"|(?:\S)",
    re.DOTALL,
)


def _normalized_cpp_tokens(source):
    """忽略注释和排版，只保留会影响 C++ 编译结果的 token。"""
    return tuple(
        token
        for token in _CPP_TOKEN.findall(source)
        if not token.startswith("//") and not token.startswith("/*")
    )


def _literal_assignment(source, name):
    module = ast.parse(source)
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"没有找到常量赋值: {name}")


def test_prepare_does_not_use_test_only_build_switch():
    forbidden_macro = "TORCH" + "_MODE"
    sources = [
        DIRECT.read_text(encoding="utf-8"),
        DIRECT_KERNEL.read_text(encoding="utf-8"),
    ]
    sources.extend(
        path.read_text(encoding="utf-8")
        for path in KERNEL.rglob("*")
        if path.is_file() and path.suffix in {".h", ".cpp"}
    )
    assert all(forbidden_macro not in source for source in sources)


def test_direct_launch_source_declares_representative_instantiations():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    direct_cases = {
        case["id"]: case
        for case in manifest["cases"]
        if "direct_launch" in case["run_on"]
    }
    assert manifest["route_validation"]["direct_launch"] == "device_compare"
    source = DIRECT.read_text(encoding="utf-8")
    kernel_source = DIRECT_KERNEL.read_text(encoding="utf-8")
    assert set(direct_cases) == {
        "prepare_dense_bnsd_raw",
        "prepare_dense_bsnd_fused",
        "prepare_dense_bsnd_fused_save",
    }
    assert {
        case["attrs"]["backward_mode"] for case in direct_cases.values()
    } == {"none", "recompute", "save"}
    assert direct_cases["prepare_dense_bsnd_fused_save"]["expect"][
        "present_outputs"
    ] == manifest["output_policy"]["fixed_slots"]
    assert all(case_id in source for case_id in direct_cases)
    assert "<<<blockDim, nullptr, stream>>>" in kernel_source
    assert "ChunkKdaFwdPrepareTilingData" in kernel_source
    assert "SetSysWorkspaceForce" in kernel_source
    assert "RunPrepare" in kernel_source
    assert "op_kernel/chunk_kda_fwd_prepare_kernel.h" in kernel_source
    assert "op_kernel/chunk_kda_fwd_prepare.cpp" not in kernel_source
    assert "chunk_kda_fwd_prepare_tiling_key.h" not in _read_kernel(
        Path("chunk_kda_fwd_prepare_struct.h")
    )
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE" in source
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE" in source
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE" in source
    assert "outputMask" not in source + kernel_source


def test_fast_kernel_launch_declares_all_soc_targets_and_device_comparison():
    wrapper = DIRECT_WRAPPER.read_text(encoding="utf-8")
    device_test = DIRECT_DEVICE_TEST.read_text(encoding="utf-8")
    expected_arches = {
        "ascend910b": "dav-2201",
        "ascend910_93": "dav-2201",
        "ascend950": "dav-3510",
    }
    for soc, npu_arch in expected_arches.items():
        source = (
            FAST_EXAMPLE
            / f"csrc/chunk_kda_fwd_prepare/{soc}/chunk_kda_fwd_prepare.cpp"
        ).read_text(encoding="utf-8")
        cmake = (
            FAST_EXAMPLE
            / f"csrc/chunk_kda_fwd_prepare/{soc}/CMakeLists.txt"
        ).read_text(encoding="utf-8")
        assert '#include "../chunk_kda_fwd_prepare_direct.cpp"' in source
        assert f"--npu-arch={npu_arch}" in cmake
        assert "--cce-auto-sync=off" in cmake

    assert "constexpr size_t kOutputCount = 13" in wrapper
    assert "std::array<at::Tensor, kOutputCount>" in wrapper
    assert "KdaPrepareDirect::Launch<" in wrapper
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE" in wrapper
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE" in wrapper
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE" in wrapper
    assert "TensorAddress(outputs[kQg])" in wrapper
    assert _literal_assignment(device_test, "CASE_IDS") == (
        "prepare_dense_bnsd_raw",
        "prepare_dense_bsnd_fused",
        "prepare_dense_bsnd_fused_save",
    )
    assert "from fla_npu.ops.ascendc import chunk_kda_fwd_prepare" in device_test
    assert "len(direct) == len(OUTPUT_NAMES) == 13" in device_test
    assert "torch.testing.assert_close" in device_test


def test_direct_launch_rejects_private_formats_and_invalid_fp32_attributes():
    wrapper = DIRECT_WRAPPER.read_text(encoding="utf-8")
    device_test = DIRECT_DEVICE_TEST.read_text(encoding="utf-8")
    compact = " ".join(wrapper.split())

    assert 'torch_npu/csrc/core/npu/NPUFormat.h' in wrapper
    assert "at_npu::native::get_npu_format(tensor)" in wrapper
    for name in ("q", "k", "v", "g", "beta"):
        assert f'CheckStandardNpuFormat({name}, "{name}")' in compact
    for name in ("aLog", "dtBias"):
        assert f"CheckStandardNpuFormat(*{name}" in compact
    for name in ("scale", "epsilon", "lowerBound"):
        assert f"static_cast<float>({name})" in compact
    assert "std::isfinite(scaleFp32)" in compact
    assert "std::isfinite(epsilonFp32) && epsilonFp32 > 0.0F" in compact
    assert "std::isfinite(lowerBoundFp32)" in compact
    assert "lowerBoundFp32 >= -5.0F && lowerBoundFp32 < 0.0F" in compact

    assert "torch_npu.npu_format_cast" in device_test
    assert "private NPU format 29" in device_test
    assert "rejects_private_required_input_format" in device_test
    assert "rejects_private_optional_input_format" in device_test
    assert "rejects_invalid_fp32_attributes" in device_test


def test_packaged_wheel_checker_requires_prepare_opp_config():
    source = WHEEL_CHECKER.read_text(encoding="utf-8")
    required_configs = source.split("REQUIRED_ASCENDC_CONFIGS = (", 1)[1]
    required_configs = required_configs.split(")", 1)[0]
    assert '"chunk_kda_fwd_prepare.json"' in required_configs


def test_output_modes_are_compile_time_template_values():
    for prefix in (Path(), Path("pseudocode")):
        policy = _read_kernel(prefix / "chunk_kda_fwd_prepare_policy.h")
        key = _read_kernel(prefix / "chunk_kda_fwd_prepare_tiling_key.h")
        compact = " ".join(key.split())
        assert "#define CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE 0" in policy
        assert "#define CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE 1" in policy
        assert "#define CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE 2" in policy
        assert "ASCENDC_TPL_UINT_DECL(OUTPUT_MODE, 2, ASCENDC_TPL_UI_LIST" in compact
        assert "static constexpr OutputMode outputMode = OUTPUT_MODE;" in policy


def test_output_mode_is_not_forwarded_as_runtime_tiling_data():
    for prefix in (Path(), Path("pseudocode")):
        struct_source = _read_kernel(prefix / "chunk_kda_fwd_prepare_struct.h")
        assert "outputMask" not in struct_source
        assert "outputMode" not in struct_source

        launch_source = _read_kernel(prefix / "chunk_kda_fwd_prepare.cpp")
        assert "uint32_t OUTPUT_MODE" in launch_source
        assert "static_cast<KdaPrepare::OutputMode>(OUTPUT_MODE)>" in launch_source
        assert "outputMask" not in launch_source


def test_optional_output_guards_stay_outside_vf_functions():
    vf_functions = {
        "arch22/chunk_kda_fwd_prepare_vec.h": (
            "__aicore__ inline void V0Vf(",
            "__aicore__ inline void V1Vf(",
            "__aicore__ inline void V3Vf(",
            "__aicore__ inline void V6Vf(",
        ),
        "arch35/chunk_kda_fwd_prepare_vec.h": (
            "__simd_vf__ inline void StageV0Vf(",
            "__simd_vf__ inline void StageV1Vf(",
            "__simd_vf__ inline void StageV3Vf(",
            "__simd_vf__ inline void StageV6Vf(",
        ),
    }
    for prefix in (Path(), Path("pseudocode")):
        for relative_path, markers in vf_functions.items():
            source = _read_kernel(prefix / relative_path)
            for marker in markers:
                body = _function_body(source, marker)
                assert "outputMask" not in body
                assert "CompilePolicy::outputMode" not in body


def test_arch35_vector_pseudocode_tracks_runtime_semantics():
    implementation = _read_kernel("arch35/chunk_kda_fwd_prepare_vec.h")
    pseudocode = _read_kernel(
        "pseudocode/arch35/chunk_kda_fwd_prepare_vec.h"
    ).replace(
        "PSEUDOCODE_ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H",
        "ARCH35_CHUNK_KDA_FWD_PREPARE_VEC_H",
    )

    assert _normalized_cpp_tokens(pseudocode) == _normalized_cpp_tokens(
        implementation
    )


def test_arch35_load2d_sources_use_direct_byte_addresses():
    for prefix in (Path(), Path("pseudocode")):
        source = _read_kernel(prefix / "arch35/chunk_kda_fwd_prepare_cube.h")
        compact = " ".join(source.split())
        assert "lane + ScorePayload::kQPlus" in compact
        assert "lane + ScorePayload::kKPlus" in compact
        assert "lane + ScorePayload::kKMinus[s]" in compact
        assert "scoreL1[" not in source
        assert "stackedQkL0[" not in source
        assert "rawScoreUb[" not in source
        assert "akkL1[L1::kAkkQ10Elements]" not in source


def test_only_non_forward_outputs_use_compile_time_store_guards():
    implementation_paths = (
        "arch22/chunk_kda_fwd_prepare_vec.h",
        "arch22/chunk_kda_fwd_prepare_cube.h",
        "arch35/chunk_kda_fwd_prepare_vec.h",
        "arch35/chunk_kda_fwd_prepare_cube.h",
    )
    for prefix in (Path(), Path("pseudocode")):
        sources = [_read_kernel(prefix / path) for path in implementation_paths]
        combined = "\n".join(sources)
        compact = " ".join(combined.split())
        assert "outputMask" not in combined
        assert "OutputMask::" not in combined
        assert "if constexpr (CompilePolicy::outputMode == OutputMode::Save)" in compact
        assert "if constexpr (CompilePolicy::outputMode != OutputMode::None)" in compact
        assert "qgGm_.SetGlobalBuffer" in compact
        for name in (
            "akkGm_", "qHatGm_", "kHatGm_", "qRstdGm_", "kRstdGm_",
            "betaEffGm_",
        ):
            assert name in compact


def test_masking_does_not_remove_internal_relays_or_required_stores():
    for prefix in (Path(), Path("pseudocode")):
        arch22_vec = _read_kernel(
            prefix / "arch22/chunk_kda_fwd_prepare_vec.h"
        )
        arch35_vec = _read_kernel(
            prefix / "arch35/chunk_kda_fwd_prepare_vec.h"
        )
        arch22_cube = _read_kernel(
            prefix / "arch22/chunk_kda_fwd_prepare_cube.h"
        )
        arch35_cube = _read_kernel(
            prefix / "arch35/chunk_kda_fwd_prepare_cube.h"
        )
        for source in (arch22_vec, arch35_vec):
            assert "Workspace::kQHat" in source
            assert "Workspace::kKHat" in source
            assert "Workspace::kAkk" in source
            assert "DataCopy(akkRelay" in source
            assert "DataCopy(qgScaledGm_" in source
            assert "DataCopy(kgGm_" in source
            assert "DataCopy(gkGm_" in source
            assert "DataCopy(aqkGm_" in source
        for source in (arch22_cube, arch35_cube):
            assert "L1::kAkkQ10Elements" in source
