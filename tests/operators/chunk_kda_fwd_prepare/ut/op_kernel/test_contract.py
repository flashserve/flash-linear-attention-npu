"""ChunkKdaFwdPrepare 的 direct launch 源码合同。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "tests/op_cases/chunk_kda_fwd_prepare.json"
DIRECT = (
    ROOT
    / "tests/operators/chunk_kda_fwd_prepare/routes"
    / "test_direct_chunk_kda_fwd_prepare.cpp"
)
KERNEL = ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel"


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


def test_prepare_does_not_use_test_only_build_switch():
    forbidden_macro = "TORCH" + "_MODE"
    sources = [DIRECT.read_text(encoding="utf-8")]
    sources.extend(
        path.read_text(encoding="utf-8")
        for path in KERNEL.rglob("*")
        if path.is_file() and path.suffix in {".h", ".cpp"}
    )
    assert all(forbidden_macro not in source for source in sources)


def test_direct_launch_source_declares_representative_instantiations():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case_ids = {
        case["id"]
        for case in manifest["cases"]
        if "direct_launch" in case["run_on"]
    }
    assert manifest["route_validation"]["direct_launch"] == "source_contract"
    source = DIRECT.read_text(encoding="utf-8")
    assert case_ids == {
        "prepare_dense_bnsd_raw",
        "prepare_dense_bsnd_fused",
    }
    assert all(case_id in source for case_id in case_ids)
    assert "<<<blockDim, nullptr, stream>>>" in source
    assert "ChunkKdaFwdPrepareTilingData" in source
    assert "SetSysWorkspaceForce" in source
    assert "RunPrepare" in source
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE" in source
    assert "CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE" in source
    assert "outputMask" not in source


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
