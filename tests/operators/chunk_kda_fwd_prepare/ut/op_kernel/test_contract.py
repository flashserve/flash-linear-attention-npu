"""ChunkKdaFwdPrepare 的 kernel 源码合同。"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
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


def test_prepare_does_not_use_test_only_build_switch():
    forbidden_macro = "TORCH" + "_MODE"
    sources = [
        path.read_text(encoding="utf-8")
        for path in KERNEL.rglob("*")
        if path.is_file() and path.suffix in {".h", ".cpp"}
    ]
    assert all(forbidden_macro not in source for source in sources)


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


def test_arch22_pseudocode_tracks_runtime_semantics():
    for filename, pseudocode_guard, implementation_guard in (
        (
            "chunk_kda_fwd_prepare_vec.h",
            "PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H",
            "ARCH22_CHUNK_KDA_FWD_PREPARE_VEC_H",
        ),
        (
            "chunk_kda_fwd_prepare_cube.h",
            "PSEUDOCODE_ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H",
            "ARCH22_CHUNK_KDA_FWD_PREPARE_CUBE_H",
        ),
    ):
        implementation = _read_kernel(Path("arch22") / filename)
        pseudocode = _read_kernel(
            Path("pseudocode/arch22") / filename
        ).replace(pseudocode_guard, implementation_guard)
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


def test_arch22_cross_core_gm_relays_have_disjoint_writers():
    for prefix in (Path(), Path("pseudocode")):
        policy = _read_kernel(prefix / "chunk_kda_fwd_prepare_policy.h")
        arch22_vec = _read_kernel(
            prefix / "arch22/chunk_kda_fwd_prepare_vec.h"
        )
        arch22_cube = _read_kernel(
            prefix / "arch22/chunk_kda_fwd_prepare_cube.h"
        )
        compact_vec = " ".join(arch22_vec.split())

        assert "kArch22RawScoreBytes = 0x5000" in policy
        assert "kArch22SlotStride" in policy
        assert "Workspace::kArch22CubeRelay" in arch22_vec
        assert "Workspace::kArch22CubeRelay" in arch22_cube
        assert "Workspace::kArch22TRelay" in arch22_cube
        assert "rawScoreRelayGm_" not in arch22_vec + arch22_cube
        assert "reinterpret_cast<__gm__ float *>(args_.w)" not in arch22_cube
        assert "reinterpret_cast<__gm__ float *>(args_.u)" not in arch22_cube
        assert "reinterpret_cast<__gm__ float *>(args_.kg)" not in arch22_cube
        assert (
            "DataCopyExtParams{ static_cast<uint16_t>(bottomRows), "
            "32 * sizeof(bfloat16_t), 2, 64, 0}"
        ) in compact_vec


def test_arch22_relay_and_akk_writer_ranges_cover_every_tail_length():
    max_rows = 64
    sub_chunk_rows = 16
    raw_score_relay_bytes = 0x5000

    for valid_rows in range(1, max_rows + 1):
        active_sub_chunks = (valid_rows + sub_chunk_rows - 1) // sub_chunk_rows
        compact_bytes = 0
        raw_score_ranges = []
        for sub_chunk in range(active_sub_chunks):
            rows = min(
                sub_chunk_rows,
                valid_rows - sub_chunk * sub_chunk_rows,
            )
            prefix_rows = sub_chunk_rows * (sub_chunk + 1)
            begin = (
                sub_chunk_rows
                * sub_chunk_rows
                * sub_chunk
                * (sub_chunk + 1)
                * 4
            )
            # C2 的固定 band 起点必须与 V3 的紧凑读取游标一致。
            assert begin == compact_bytes
            end = begin + 2 * rows * prefix_rows * 4
            raw_score_ranges.append(range(begin, end))
            compact_bytes = end

        assert compact_bytes <= raw_score_relay_bytes
        assert sum(len(span) for span in raw_score_ranges) == compact_bytes

        # V3 写上半行和右下象限，C5 只写左下象限。
        top = {
            (row, column)
            for row in range(min(valid_rows, 32))
            for column in range(max_rows)
        }
        bottom_right = {
            (row, column)
            for row in range(32, valid_rows)
            for column in range(32, max_rows)
        }
        bottom_left = {
            (row, column)
            for row in range(32, valid_rows)
            for column in range(32)
        }
        expected = {
            (row, column)
            for row in range(valid_rows)
            for column in range(max_rows)
        }
        assert top.isdisjoint(bottom_right)
        assert top.isdisjoint(bottom_left)
        assert bottom_right.isdisjoint(bottom_left)
        assert top | bottom_right | bottom_left == expected
